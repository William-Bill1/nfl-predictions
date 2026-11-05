"""
Player Props release storage (offline slice): archive a verified S1 release
package with its prediction schedule, describe the stored bytes, and restore
them into an isolated directory with full verification. Nothing here uploads,
downloads, approves, activates or publishes anything, and nothing is wired
into a workflow or production path.

Three separate operations, never implied by each other:

* ``pack`` verifies the package (``release_manifest.verify_package``) and that
  the prediction schedule is the one the package's rehearsal recorded, then
  writes, into a new directory:
  - ``player-props-<release_id>.tar``: a deterministic, uncompressed ustar
    archive of exactly those verified bytes (``package/<asset>`` for every
    package file, ``schedule/nfl_schedule_<season>.csv``). Entries are sorted
    regular files with fixed metadata, so the same inputs always give the
    same archive bytes;
  - ``player-props-<release_id>.storage.json``: the storage descriptor. It
    names the release, the manifest hash, the archive's hash and size, and
    every entry's hash and size. It never contains its own hash (any record
    of the descriptor's hash lives outside it).
* ``restore`` checks the descriptor strictly, hashes the archive, inspects
  every entry before anything is written, extracts into a private staging
  directory, re-reads every written file, re-runs S1 verification on the
  restored package and re-checks the schedule relationship. Only then is the
  staging directory renamed into place, with ``restore_report.json`` (the
  verification report) inside it.
* S2 publication (``player_props/publication.py``) stays a separate, explicit
  operation on the restored directory. Restoring never publishes.

A storage descriptor is consistency metadata for candidate storage. It is not
an approval record: creating or validating one never approves or activates a
release, and it can't record an approval (its fields are fixed; an
``approved`` or similar field is refused). Hashes in it detect changed bytes;
they don't make any copy immutable and don't establish who produced them.

Archive inspection refuses, before any write: anything other than regular
files; absolute, traversing, backslash, drive-letter or otherwise unsafe
names; names that alias on Windows or macOS (case or Unicode normalisation
collisions, reserved device names, trailing dots or spaces); a file that is
also a directory of another entry; duplicate, unexpected or missing entries;
and entries, entry counts or archive sizes over the limits. Archive contents
are only ever read as data: nothing from an archive is imported or executed
(the bundle inside is checked by the existing safe archive reader).

Destinations: resolved fully (links and junctions included), and refused when
they overlap the repository, a production path, a protected path, the archive
or the descriptor. An existing destination is replaced only if it is a
previous restore (it holds a readable ``restore_report.json``).

Failure handling (catchable errors):
* before the commit boundary (the staged directory renamed into place),
  the previous destination is untouched and the staging directory is removed:
  RestoreError, state ``not_restored``. If the staging directory can't be
  removed it is listed in ``retained_files``.
* if the swap fails after the previous destination was moved aside, it is
  moved back: state ``rolled_back``; if that fails too, ``rollback_failed``
  with the moved-aside copy listed in ``recovery_files``.
* after the commit boundary, deleting the moved-aside previous copy is cleanup;
  a failure is reported in ``cleanup_warnings`` / ``retained_files`` (state
  ``restored_with_cleanup_warnings``), never as a failed restore.
* NOT crash-atomic: nothing is fsynced, and a crash can leave the staging or
  moved-aside directory behind (named ``.<dest>.restore-*`` /
  ``.<dest>.previous-*`` beside the destination).
* Concurrent restores to the same destination: an advisory lock beside it
  (``ontario_spreads.SlotLock``) makes a second restore refuse. Other writers
  are not excluded.

CLI (offline; ASCII-only JSON; exit 0 ok, 1 refused, 2 could not run, 3
restored with cleanup warnings):

    python player_props/release_storage.py pack --package PKG --prediction-schedule CSV --out NEW_DIR
    python player_props/release_storage.py inspect --archive TAR --descriptor JSON
    python player_props/release_storage.py restore --archive TAR --descriptor JSON --dest DIR

Policy: docs/architecture.md ("Release storage: archive and restore").
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import stat
import sys
import tarfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from player_props import release_manifest as rm  # noqa: E402
from player_props.rehearsal import PRODUCTION_PATHS, _overlaps, _real  # noqa: E402

DESCRIPTOR_KIND = 'player_props_release_storage_descriptor'
DESCRIPTOR_VERSION = 1
ARCHIVE_FORMAT = 'ustar-uncompressed-v1'
STATUS = 'candidate_storage_only'
STATEMENT = ("describes stored bytes for consistency checks only; it is not an approval record and does not "
             "approve, activate or publish a release")
REPORT_KIND = 'player_props_release_restore_report'
REPORT_VERSION = 1
REPORT_NAME = 'restore_report.json'
PACKAGE_DIR = 'package'
SCHEDULE_DIR = 'schedule'
LOCK_PREFIX = 'release-restore-'

# Limits. GitHub release assets must be under 2 GiB each; package assets are
# already capped at 1 GiB by S1.
MAX_ARCHIVE_BYTES = 2 * 1024 ** 3 - 1
MAX_ENTRY_BYTES = rm.MAX_ASSET_BYTES
MAX_ENTRIES = 64
MAX_DESCRIPTOR_BYTES = 1024 * 1024
MAX_NAME_LENGTH = 200

_NAME_CHARS = re.compile(r'[A-Za-z0-9._-]+')
_RESERVED = {'con', 'prn', 'aux', 'nul', *(f'com{i}' for i in range(10)), *(f'lpt{i}' for i in range(10))}
_SCHEDULE = re.compile(r'nfl_schedule_(\d{4})\.csv')
_CHUNK = 1024 * 1024
DESCRIPTOR_KEYS = {'kind', 'descriptor_version', 'status', 'statement', 'release_id', 'manifest_sha256',
                   'inference_target', 'archive', 'entries', 'prediction_schedule'}
NOT_ESTABLISHED = [
    "approval or activation: a storage descriptor and a restore never approve or activate a release",
    "authenticity: hashes show the restored bytes match the descriptor and the package's own manifest, not who "
    "produced them; the descriptor came with the archive and is not a trusted record",
    "S2 publication: not run (a separate, explicit operation)",
    "the runtime environment: Python and the installed dependencies are whatever ran this restore",
]


class RestoreError(RuntimeError):
    """Pack, inspection or restore refused or failed. ``state`` says what is
    on disk: 'not_restored' (the previous destination untouched),
    'rolled_back' (the previous destination moved back) or 'rollback_failed'
    (``recovery_files`` names the moved-aside copy). ``retained_files`` lists
    other leftovers."""

    def __init__(self, problems, state: str = 'not_restored', recovery_files: Optional[List[str]] = None,
                 retained_files: Optional[List[str]] = None):
        self.problems = [str(p) for p in problems]
        self.state = state
        self.recovery_files = list(recovery_files or [])
        self.retained_files = list(retained_files or [])
        super().__init__("; ".join(self.problems))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def archive_name(release_id: str) -> str:
    return f'player-props-{release_id}.tar'


def descriptor_name(release_id: str) -> str:
    return f'player-props-{release_id}.storage.json'


def schedule_entry(season: int) -> str:
    return f'{SCHEDULE_DIR}/nfl_schedule_{season}.csv'


def summary_schedule_key(season: int) -> str:
    return f'runtime/data/nfl_schedule_{season}.csv'


def package_entries() -> set:
    """Exactly the archive names an S1 package occupies: every fixed role path
    (``release_manifest.ROLES``) plus the manifest, under ``package/``."""
    return {f'{PACKAGE_DIR}/{p}' for p in rm.ROLES.values()} | {f'{PACKAGE_DIR}/{rm.MANIFEST_NAME}'}


# ---------------------------------------------------------------------------
# Entry names
# ---------------------------------------------------------------------------

def name_problem(name) -> Optional[str]:
    """Why ``name`` is not a safe, portable archive entry name, or None."""
    if not isinstance(name, str) or not name:
        return "entry name must be a non-empty string"
    if len(name) > MAX_NAME_LENGTH:
        return f"entry name {name[:40]!r}... is longer than {MAX_NAME_LENGTH} characters"
    if unicodedata.normalize('NFC', name) != name or not name.isascii():
        return f"entry name {name!r} is not plain ASCII"
    if name.startswith('/') or '\\' in name or ':' in name:
        return f"entry name {name!r} is absolute or uses a backslash or drive/stream separator"
    parts = name.split('/')
    for part in parts:
        if part in ('', '.', '..'):
            return f"entry name {name!r} has an empty, '.' or '..' component"
        if not _NAME_CHARS.fullmatch(part):
            return f"entry name {name!r} has characters outside A-Z a-z 0-9 . _ -"
        if part.endswith('.') or part.startswith('-'):
            return f"entry name {name!r} has a component Windows or a shell would alias"
        if part.split('.')[0].lower() in _RESERVED:
            return f"entry name {name!r} uses a reserved Windows device name"
    if parts[0] not in (PACKAGE_DIR, SCHEDULE_DIR) or len(parts) < 2:
        return f"entry name {name!r} is outside {PACKAGE_DIR}/ and {SCHEDULE_DIR}/"
    return None


def collision_problems(names: List[str]) -> List[str]:
    """Duplicates, case/normalisation aliases and file-vs-directory clashes."""
    problems, seen, folded = [], set(), {}
    for n in names:
        if n in seen:
            problems.append(f"duplicate entry {n!r}")
            continue
        seen.add(n)
        key = unicodedata.normalize('NFC', n).casefold()
        if key in folded:
            problems.append(f"entries {folded[key]!r} and {n!r} collide on a case-insensitive filesystem")
        folded.setdefault(key, n)
    dirs = {'/'.join(k.split('/')[:i]) for k in folded for i in range(1, len(k.split('/')))}
    for key, n in folded.items():
        if key in dirs:
            problems.append(f"entry {n!r} is a file and also a directory of another entry")
    return problems


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------

def _type_name(v) -> str:
    return 'null' if v is None else type(v).__name__


def _not_str(v, where: str) -> List[str]:
    """[problem] unless ``v`` is a string. Checked before ``v`` is used as a
    key, compared with a path or formatted into one; never coerced."""
    return [] if isinstance(v, str) else [f"{where} must be a string, not {_type_name(v)}"]


def _file_entry(entry, where: str, keys: set) -> List[str]:
    if not isinstance(entry, dict) or set(entry) != keys:
        return [f"{where} must have exactly the fields {sorted(keys)}"]
    problems = []
    for k in sorted(keys - {'sha256', 'bytes'}):           # the remaining fields are all strings
        problems += _not_str(entry[k], f"{where}.{k}")
    if 'sha256' in keys and not rm._is_hash(entry['sha256']):
        problems.append(f"{where}.sha256 must be a 64-character lowercase hex SHA-256")
    if 'bytes' in keys and (not rm._is_int(entry['bytes']) or entry['bytes'] < 0):
        problems.append(f"{where}.bytes must be a non-negative integer")
    return problems


def descriptor_problems(doc) -> List[str]:
    """Every problem with a storage descriptor (strict: exact fields, types,
    names, limits and internal agreement). An empty list means it is
    well-formed and self-consistent, nothing more: not authentic, not
    approved."""
    if not isinstance(doc, dict):
        return ["the descriptor must be a JSON object"]
    if set(doc) != DESCRIPTOR_KEYS:
        extra, missing = sorted(set(doc) - DESCRIPTOR_KEYS), sorted(DESCRIPTOR_KEYS - set(doc))
        return [f"the descriptor must have exactly the fields {sorted(DESCRIPTOR_KEYS)}"
                + (f"; unexpected {extra}" if extra else "") + (f"; missing {missing}" if missing else "")]
    problems = _not_str(doc['kind'], 'kind')
    if type(doc['descriptor_version']) is not int:
        problems.append(f"descriptor_version must be an integer, not {_type_name(doc['descriptor_version'])}")
    if problems:
        return problems
    if doc['kind'] != DESCRIPTOR_KIND or doc['descriptor_version'] != DESCRIPTOR_VERSION:
        return [f"not a version-{DESCRIPTOR_VERSION} {DESCRIPTOR_KIND}"]
    problems += _not_str(doc['status'], 'status') + _not_str(doc['statement'], 'statement')
    if not problems and (doc['status'] != STATUS or doc['statement'] != STATEMENT):
        problems.append(f"status must be {STATUS!r} with the fixed statement: a storage descriptor never records "
                        f"an approval or activation")
    rid = doc['release_id']
    if not isinstance(rid, str):
        problems += _not_str(rid, 'release_id')
        rid = None
    elif not rm._RELEASE_ID.fullmatch(rid):
        problems.append("release_id must look like YYYYMMDDTHHMMSSZ-xxxxxxxx")
        rid = None
    if not rm._is_hash(doc['manifest_sha256']):
        problems.append("manifest_sha256 must be a 64-character lowercase hex SHA-256")
    t = doc['inference_target']
    season = None
    if not (isinstance(t, dict) and set(t) == {'season', 'week'} and rm._is_int(t['season'])
            and rm._is_int(t['week']) and 2000 <= t['season'] <= 2100 and 1 <= t['week'] <= 22):
        problems.append("inference_target must be {season, week} integers in range")
    else:
        season = t['season']
    a = doc['archive']
    problems += _file_entry(a, 'archive', {'file', 'sha256', 'bytes', 'format'})
    if isinstance(a, dict) and set(a) == {'file', 'sha256', 'bytes', 'format'}:
        if rid and isinstance(a['file'], str) and a['file'] != archive_name(rid):
            problems.append(f"archive.file must be {archive_name(rid)}")
        if isinstance(a['format'], str) and a['format'] != ARCHIVE_FORMAT:
            problems.append(f"archive.format must be {ARCHIVE_FORMAT!r}")
        if rm._is_int(a['bytes']) and a['bytes'] > MAX_ARCHIVE_BYTES:
            problems.append(f"archive.bytes exceeds the {MAX_ARCHIVE_BYTES}-byte limit")
    entries = doc['entries']
    by_name: Dict[str, dict] = {}
    if not isinstance(entries, list) or not entries:
        problems.append("entries must be a non-empty list")
        entries = []
    elif len(entries) > MAX_ENTRIES:
        problems.append(f"entries has {len(entries)} items; at most {MAX_ENTRIES} are allowed")
    names = []
    for i, e in enumerate(entries):
        found = _file_entry(e, f"entries[{i}]", {'name', 'sha256', 'bytes'})
        if found:
            problems += found
            continue
        why = name_problem(e['name'])
        if why:
            problems.append(f"entries[{i}]: {why}")
            continue
        if e['bytes'] > MAX_ENTRY_BYTES:
            problems.append(f"entries[{i}] ({e['name']}) exceeds the {MAX_ENTRY_BYTES}-byte entry limit")
        names.append(e['name'])
        by_name.setdefault(e['name'], e)
    problems += collision_problems(names)
    if names != sorted(names):
        problems.append("entries must be sorted by name")
    if sum(e['bytes'] for e in by_name.values()) > MAX_ARCHIVE_BYTES:
        problems.append("the entries' total size exceeds the archive limit")
    man = by_name.get(f'{PACKAGE_DIR}/{rm.MANIFEST_NAME}')
    if man is None:
        problems.append(f"entries must include {PACKAGE_DIR}/{rm.MANIFEST_NAME}")
    elif man['sha256'] != doc['manifest_sha256']:
        problems.append(f"manifest_sha256 is not the hash of the {PACKAGE_DIR}/{rm.MANIFEST_NAME} entry")
    pkg = {n for n in names if n.startswith(f'{PACKAGE_DIR}/')}
    required = package_entries()
    if pkg - required:
        problems.append(f"entries under {PACKAGE_DIR}/ that are not S1 package paths: {sorted(pkg - required)}")
    if required - pkg:
        problems.append(f"S1 package paths missing from entries: {sorted(required - pkg)}")
    sched = doc['prediction_schedule']
    found = _file_entry(sched, 'prediction_schedule', {'entry', 'sha256', 'bytes', 'summary_artifact'})
    problems += found
    if not found and season:                 # every field has its type now; safe to compare and look up
        if sched['entry'] != schedule_entry(season):
            problems.append(f"prediction_schedule.entry must be {schedule_entry(season)}")
        if sched['summary_artifact'] != summary_schedule_key(season):
            problems.append(f"prediction_schedule.summary_artifact must be {summary_schedule_key(season)}")
        listed = by_name.get(sched['entry'])
        if listed is None:
            problems.append(f"entries must include the prediction schedule {sched['entry']}")
        elif (listed['sha256'], listed['bytes']) != (sched['sha256'], sched['bytes']):
            problems.append("prediction_schedule does not match its entry's hash and size")
    others = [n for n in names if n.startswith(f'{SCHEDULE_DIR}/')]
    if season and others != [schedule_entry(season)]:
        problems.append(f"{SCHEDULE_DIR}/ must hold exactly {schedule_entry(season)}, not {others}")
    return problems


def validate_descriptor(doc) -> dict:
    """{'ok', 'problems', 'approval'}. 'approval' is always 'not_established':
    a valid descriptor says the stored bytes are self-consistent, never that
    the release is approved or active."""
    problems = descriptor_problems(doc)
    return {'ok': not problems, 'problems': problems, 'approval': 'not_established',
            'note': "descriptor validation checks consistency only; it does not approve or activate a release"}


def load_descriptor(path: Path) -> tuple[dict, bytes]:
    data = _read_input(Path(path), "descriptor", MAX_DESCRIPTOR_BYTES)

    def unique(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate field(s) {sorted({k for k in keys if keys.count(k) > 1})}")
        return dict(pairs)
    try:
        doc = json.loads(data.decode('utf-8'), object_pairs_hook=unique)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RestoreError([f"the descriptor {path} is not valid UTF-8 JSON ({type(exc).__name__}: {exc})"]) from exc
    problems = descriptor_problems(doc)
    if problems:
        raise RestoreError([f"the descriptor is invalid: {p}" for p in problems])
    return doc, data


def _read_input(path: Path, what: str, limit: int) -> bytes:
    try:
        st = os.stat(path)
    except OSError as exc:
        raise RestoreError([f"{what} {path} can't be read ({type(exc).__name__}: {exc})"]) from exc
    if not stat.S_ISREG(st.st_mode):
        raise RestoreError([f"{what} {path} is not a regular file"])
    if st.st_size > limit:
        raise RestoreError([f"{what} {path} is larger than {limit} bytes"])
    with open(path, 'rb') as f:
        data = f.read(limit + 1)
    if len(data) > limit:
        raise RestoreError([f"{what} {path} is larger than {limit} bytes"])
    return data


# ---------------------------------------------------------------------------
# Archive writing and inspection
# ---------------------------------------------------------------------------

def _tar_bytes(files: Dict[str, bytes]) -> bytes:
    """Deterministic ustar archive: sorted regular files, fixed metadata."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w', format=tarfile.USTAR_FORMAT) as tar:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime, info.type = len(files[name]), 0o644, 0, tarfile.REGTYPE
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            tar.addfile(info, io.BytesIO(files[name]))
    return buf.getvalue()


def _hash_file(path: Path, limit: int) -> tuple[str, int]:
    h, n = hashlib.sha256(), 0
    with open(path, 'rb') as f:
        while True:
            block = f.read(_CHUNK)
            if not block:
                break
            n += len(block)
            if n > limit:
                raise RestoreError([f"{path} is larger than {limit} bytes"])
            h.update(block)
    return h.hexdigest(), n


def inspect_archive(archive: Path | str, descriptor: dict) -> dict:
    """Check the archive against a (valid) descriptor without writing
    anything: the archive's hash and size, then every member's type, name,
    size and hash. Returns {'entries': {name: (sha256, bytes)}}; raises
    RestoreError listing every problem."""
    archive = Path(archive)
    problems = descriptor_problems(descriptor)
    if problems:
        raise RestoreError([f"the descriptor is invalid: {p}" for p in problems])
    try:
        st = os.stat(archive)
    except OSError as exc:
        raise RestoreError([f"archive {archive} can't be read ({type(exc).__name__}: {exc})"]) from exc
    if not stat.S_ISREG(st.st_mode):
        raise RestoreError([f"archive {archive} is not a regular file"])
    want = descriptor['archive']
    if st.st_size > MAX_ARCHIVE_BYTES:
        raise RestoreError([f"archive {archive} is larger than {MAX_ARCHIVE_BYTES} bytes"])
    digest, size = _hash_file(archive, MAX_ARCHIVE_BYTES)
    if size != want['bytes'] or digest != want['sha256']:
        raise RestoreError([f"archive {archive.name} doesn't match the descriptor (size {size} vs {want['bytes']}, "
                            f"sha256 {digest[:12]} vs {want['sha256'][:12]})"])
    expected = {e['name']: e for e in descriptor['entries']}
    seen: Dict[str, tuple] = {}
    names: List[str] = []
    total = 0
    try:
        with tarfile.open(archive, mode='r:') as tar:          # uncompressed only
            for i, m in enumerate(tar):
                if i >= MAX_ENTRIES:
                    problems.append(f"the archive has more than {MAX_ENTRIES} entries")
                    break
                names.append(m.name)
                why = name_problem(m.name)
                if why:
                    problems.append(why)
                    continue
                if not m.isreg() or m.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
                    problems.append(f"entry {m.name!r} is not a regular file (type {m.type!r})")
                    continue
                if m.size > MAX_ENTRY_BYTES:
                    problems.append(f"entry {m.name!r} is larger than {MAX_ENTRY_BYTES} bytes")
                    continue
                total += m.size
                if total > MAX_ARCHIVE_BYTES:
                    problems.append("the archive's entries exceed the total size limit")
                    break
                h = hashlib.sha256()
                src = tar.extractfile(m)
                read = 0
                while True:
                    block = src.read(_CHUNK)
                    if not block:
                        break
                    read += len(block)
                    h.update(block)
                if read != m.size:
                    problems.append(f"entry {m.name!r} is truncated ({read} of {m.size} bytes)")
                    continue
                seen.setdefault(m.name, (h.hexdigest(), m.size))
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise RestoreError([f"archive {archive.name} can't be read as an uncompressed tar "
                            f"({type(exc).__name__}: {exc})"] + problems) from exc
    problems += collision_problems(names)
    for n in sorted(set(seen) - set(expected)):
        problems.append(f"unexpected entry {n!r} (not in the descriptor)")
    for n in sorted(set(expected) - set(names)):
        problems.append(f"entry {n!r} is in the descriptor but missing from the archive")
    for n, (digest, size) in sorted(seen.items()):
        e = expected.get(n)
        if e and (digest, size) != (e['sha256'], e['bytes']):
            problems.append(f"entry {n!r} doesn't match the descriptor (size {size} vs {e['bytes']}, "
                            f"sha256 {digest[:12]} vs {e['sha256'][:12]})")
    if problems:
        raise RestoreError(problems)
    return {'entries': seen}


# ---------------------------------------------------------------------------
# Pack
# ---------------------------------------------------------------------------

def _verified_package(package: Path, schedule: bytes) -> tuple[dict, bytes, Dict[str, bytes], dict]:
    """(manifest, manifest bytes, {package-relative name: bytes}, summary)
    after S1 verification and the schedule check. Raises RestoreError."""
    try:
        report = rm.verify_package(package)
    except rm.ReleaseError as exc:
        raise RestoreError(["the release package does not verify: " + p for p in exc.problems]) from exc
    if not report['ok']:
        raise RestoreError(["the release package does not verify: " + p for p in report['problems']])
    try:
        manifest, mbytes, contents = rm.read_package(package)
    except rm.ReleaseError as exc:
        raise RestoreError(["the release package changed after it verified: " + p for p in exc.problems]) from exc
    if _sha256(mbytes) != report['manifest_sha256']:
        raise RestoreError(["the release package's manifest changed after it verified"])
    summary = json.loads(contents['rehearsal_summary'].decode('utf-8'))
    problems = schedule_problems(manifest, summary, schedule)
    if problems:
        raise RestoreError(problems)
    files = {rm.MANIFEST_NAME: mbytes}
    for a in manifest['assets']:
        files[a['name']] = contents[a['role']]
    return manifest, mbytes, files, summary


def schedule_problems(manifest: dict, summary: dict, schedule: bytes) -> List[str]:
    t = manifest['derived']['inference_target']
    key = summary_schedule_key(t['season'])
    want = (summary.get('artifacts') or {}).get(key)
    if not rm._is_hash(want):
        return [f"the package's rehearsal summary records no prediction schedule ({key})"]
    if _sha256(schedule) != want:
        return [f"the prediction schedule ({_sha256(schedule)[:12]}) is not the one the package's verified "
                f"rehearsal used for {t['season']} week {t['week']} ({want[:12]}, recorded as {key})"]
    return []


def pack(package: Path | str, prediction_schedule: Path | str, out_dir: Path | str,
         protected: Optional[List[Path]] = None) -> dict:
    """Archive a verified package and its prediction schedule into a new
    directory ``out_dir``. Returns the descriptor and the written paths. The
    package and schedule are only read."""
    package, schedule_path, out_dir = Path(package), Path(prediction_schedule), Path(out_dir)
    try:
        out_real = rm.check_output_location(out_dir, {'package': package, 'prediction schedule': schedule_path},
                                            protected=protected)
    except rm.ReleaseError as exc:
        raise RestoreError(exc.problems) from exc
    if os.path.lexists(out_dir):
        raise RestoreError([f"output {out_dir} already exists; pack writes only into a new directory"])
    schedule = _read_input(schedule_path, "prediction schedule", rm.MAX_ASSET_BYTES)
    manifest, mbytes, pkg_files, _ = _verified_package(package, schedule)
    t = manifest['derived']['inference_target']
    season, week, rid = t['season'], t['week'], manifest['release_id']
    files = {f'{PACKAGE_DIR}/{n}': b for n, b in pkg_files.items()}
    files[schedule_entry(season)] = schedule
    tar = _tar_bytes(files)
    entries = [{'name': n, 'sha256': _sha256(files[n]), 'bytes': len(files[n])} for n in sorted(files)]
    descriptor = {
        'kind': DESCRIPTOR_KIND, 'descriptor_version': DESCRIPTOR_VERSION, 'status': STATUS, 'statement': STATEMENT,
        'release_id': rid, 'manifest_sha256': _sha256(mbytes), 'inference_target': {'season': season, 'week': week},
        'archive': {'file': archive_name(rid), 'sha256': _sha256(tar), 'bytes': len(tar), 'format': ARCHIVE_FORMAT},
        'entries': entries,
        'prediction_schedule': {'entry': schedule_entry(season), 'sha256': _sha256(schedule), 'bytes': len(schedule),
                                'summary_artifact': summary_schedule_key(season)},
    }
    problems = descriptor_problems(descriptor)
    if problems:                                              # e.g. limits; nothing written yet
        raise RestoreError(["the package can't be described for storage: " + p for p in problems])
    desc_bytes = (json.dumps(descriptor, indent=2, sort_keys=True) + "\n").encode('utf-8')
    out_real.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_real.parent / f".{out_real.name}.pack-{secrets.token_hex(4)}"
    try:
        tmp.mkdir()
        (tmp / archive_name(rid)).write_bytes(tar)
        (tmp / descriptor_name(rid)).write_bytes(desc_bytes)
        inspect_archive(tmp / archive_name(rid), json.loads(desc_bytes))      # read back what was written
        os.replace(tmp, out_real)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        if os.path.lexists(tmp):
            raise RestoreError([f"packing failed and its temporary directory {tmp} was left behind"],
                               retained_files=[str(tmp)])
        raise
    return {'packed': True, 'release_id': rid, 'manifest_sha256': descriptor['manifest_sha256'],
            'archive': {'path': str(out_real / archive_name(rid)), **descriptor['archive']},
            'descriptor': {'path': str(out_real / descriptor_name(rid)), 'sha256': _sha256(desc_bytes),
                           'bytes': len(desc_bytes)},
            'entries': len(entries), 'approval': 'not_established'}


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------

def _hook(stage: str) -> None:
    """Test hook: called at named points of a restore. Does nothing."""


def check_destination(dest: Path, inputs: Dict[str, Path], protected: Optional[List[Path]] = None) -> Path:
    """The resolved destination, or RestoreError if it overlaps the
    repository, a production path, a protected path or an input (equal,
    ancestor or descendant, after resolving links and junctions)."""
    real = _real(dest)
    for p in [ROOT] + list(PRODUCTION_PATHS) + list(protected or []):
        if _overlaps(real, _real(p)):
            raise RestoreError([f"destination {dest} (resolves to {real}) overlaps protected location {p}; "
                                f"restore into an isolated directory"])
    for what, p in inputs.items():
        if _overlaps(real, _real(p)):
            raise RestoreError([f"destination {dest} (resolves to {real}) overlaps the {what} ({p})"])
    return real


def _previous_restore(dest: Path) -> bool:
    """True if ``dest`` is a directory left by an earlier restore."""
    try:
        if rm._is_link(dest) or not dest.is_dir():
            return False
        report = json.loads((dest / REPORT_NAME).read_text(encoding='utf-8'))
        return isinstance(report, dict) and report.get('kind') == REPORT_KIND
    except (OSError, ValueError):
        return False


def _remove(path: Path) -> Optional[str]:
    try:
        if os.path.lexists(path):
            shutil.rmtree(path)
        return None
    except OSError as exc:
        return f"{path} could not be removed ({type(exc).__name__}: {exc})"


def _extract(archive: Path, staging: Path, expected: Dict[str, dict]) -> None:
    """Write the (already inspected) entries into ``staging`` and confirm the
    written bytes. Only names from the descriptor are ever written."""
    with tarfile.open(archive, mode='r:') as tar:
        for m in tar:
            # Inspection already checked these; re-checked in case the file changed since.
            if m.name not in expected or name_problem(m.name) or m.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
                raise RestoreError([f"archive entry {m.name!r} was not inspected; the archive changed during the "
                                    f"restore"])
            e = expected[m.name]
            target = staging.joinpath(*PurePosixPath(m.name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            h, n = hashlib.sha256(), 0
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0), 0o644)
            with os.fdopen(fd, 'wb') as out:
                while True:
                    block = src.read(_CHUNK)
                    if not block:
                        break
                    n += len(block)
                    h.update(block)
                    out.write(block)
            if (h.hexdigest(), n) != (e['sha256'], e['bytes']):
                raise RestoreError([f"entry {m.name!r} changed while it was extracted"])
    present = sorted(p.relative_to(staging).as_posix() for p in staging.rglob('*') if p.is_file())
    if present != sorted(expected):
        raise RestoreError([f"the restored files are not exactly the descriptor's entries "
                            f"(extra {sorted(set(present) - set(expected))[:5]}, "
                            f"missing {sorted(set(expected) - set(present))[:5]})"])
    for name, e in expected.items():                    # re-read what was written
        digest, size = _hash_file(staging.joinpath(*PurePosixPath(name).parts), MAX_ENTRY_BYTES)
        if (digest, size) != (e['sha256'], e['bytes']):
            raise RestoreError([f"restored file {name} doesn't match the descriptor on re-read"])


def _verify_restored(staging: Path, descriptor: dict) -> dict:
    """S1 verification of the restored package plus the identity and schedule
    checks. Returns the S1 report; raises RestoreError."""
    pkg = staging / PACKAGE_DIR
    try:
        report = rm.verify_package(pkg)
    except rm.ReleaseError as exc:
        raise RestoreError(["the restored package does not verify: " + p for p in exc.problems]) from exc
    if not report['ok']:
        raise RestoreError(["the restored package does not verify: " + p for p in report['problems']])
    manifest, mbytes, contents = rm.read_package(pkg)
    problems = []
    if _sha256(mbytes) != descriptor['manifest_sha256'] or report['manifest_sha256'] != descriptor['manifest_sha256']:
        problems.append("the restored manifest is not the one the descriptor names")
    if manifest['release_id'] != descriptor['release_id']:
        problems.append(f"the restored release is {manifest['release_id']}, not {descriptor['release_id']}")
    if manifest['derived']['inference_target'] != descriptor['inference_target']:
        problems.append("the restored package's inference target differs from the descriptor's")
    schedule = (staging / descriptor['prediction_schedule']['entry']).read_bytes()
    summary = json.loads(contents['rehearsal_summary'].decode('utf-8'))
    problems += schedule_problems(manifest, summary, schedule)
    if problems:
        raise RestoreError(problems)
    return report


def restore(archive: Path | str, descriptor: Path | str, dest: Path | str,
            protected: Optional[List[Path]] = None) -> dict:
    """Restore a stored release into ``dest`` after full verification.
    Returns the report (state 'restored' or 'restored_with_cleanup_warnings');
    raises RestoreError otherwise. Never publishes or activates anything."""
    from ontario_spreads import CaptureError, SlotLock, new_id
    archive, desc_path, dest = Path(archive), Path(descriptor), Path(dest)
    dest_real = check_destination(dest, {'archive': archive, 'descriptor': desc_path}, protected)
    parent = dest_real.parent
    if not parent.is_dir():
        raise RestoreError([f"the destination's parent {parent} does not exist"])
    doc, desc_bytes = load_descriptor(desc_path)
    if os.path.lexists(dest_real) and not _previous_restore(dest_real):
        raise RestoreError([f"destination {dest} exists and is not a previous restore (no readable {REPORT_NAME}); "
                            f"it was not changed"])
    lock = SlotLock(parent, LOCK_PREFIX + dest_real.name, new_id(datetime.now(timezone.utc)), what='release restore')
    try:
        lock.__enter__()
    except CaptureError as exc:
        raise RestoreError([str(exc)]) from exc
    token = secrets.token_hex(4)
    staging = parent / f".{dest_real.name}.restore-{token}"
    aside = parent / f".{dest_real.name}.previous-{token}"
    failure: Optional[RestoreError] = None
    outcome = {}
    try:
        outcome = _restore_locked(archive, doc, desc_bytes, dest_real, staging, aside)
    except RestoreError as exc:
        failure = exc
    except Exception as exc:  # noqa: BLE001 - reported with the on-disk state
        failure = RestoreError([f"restore failed ({type(exc).__name__}: {exc})"])
        why = _remove(staging)
        if why:
            failure.retained_files.append(str(staging))
    finally:
        from player_props.publication import _release_lock
        released = _release_lock(lock)
        if released is not None:
            note, retained = released
            target = failure if failure is not None else None
            if target is not None:
                target.problems.append(note)
                if retained:
                    target.retained_files.append(retained)
            else:
                outcome['cleanup_warnings'].append(note)
                if retained:
                    outcome['retained_files'].append(retained)
    if failure is not None:
        raise failure
    outcome['state'] = 'restored_with_cleanup_warnings' if outcome['cleanup_warnings'] else 'restored'
    return outcome


def _restore_locked(archive: Path, doc: dict, desc_bytes: bytes, dest: Path, staging: Path, aside: Path) -> dict:
    try:
        inspect_archive(archive, doc)
        _hook('inspected')
        staging.mkdir()
        _extract(archive, staging, {e['name']: e for e in doc['entries']})
        _hook('extracted')
        s1 = _verify_restored(staging, doc)
        report = {
            'kind': REPORT_KIND, 'report_version': REPORT_VERSION,
            'release_id': doc['release_id'], 'manifest_sha256': doc['manifest_sha256'],
            'inference_target': doc['inference_target'],
            'descriptor_sha256': _sha256(desc_bytes), 'archive': {k: doc['archive'][k] for k in ('file', 'sha256', 'bytes')},
            'prediction_schedule': {k: doc['prediction_schedule'][k] for k in ('entry', 'sha256', 'bytes')},
            'entries': len(doc['entries']),
            'verified': [
                "the descriptor is well-formed and self-consistent (strict schema)",
                "the archive's size and SHA-256 equal the descriptor's",
                "every archive entry was inspected before writing: regular files only, safe portable names, "
                "no collisions, exactly the descriptor's entries, sizes and hashes within limits",
                "every restored file was re-read and matches the descriptor",
                "S1 release-manifest verification passed on the restored package",
                "the restored manifest, release ID and inference target are the descriptor's",
                "the restored prediction schedule is the one the package's rehearsal summary records",
            ],
            's1_verified': s1['verified'], 's1_declared': s1['declared'],
            'not_established': NOT_ESTABLISHED,
        }
        (staging / REPORT_NAME).write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode('utf-8'))
        _hook('verified')
    except Exception as exc:  # noqa: BLE001 - nothing installed yet; the previous destination is untouched
        err = exc if isinstance(exc, RestoreError) else RestoreError(
            [f"restore failed ({type(exc).__name__}: {exc}); the previous destination was not changed"])
        why = _remove(staging)
        if why:
            err.retained_files.append(str(staging))
            err.problems.append(f"also left behind: {why}")
        if err is exc:
            raise
        raise err from exc

    had_previous = os.path.lexists(dest)
    moved_aside = False
    try:
        if had_previous:
            os.replace(dest, aside)
            moved_aside = True
        _hook('moved_aside')
        os.replace(staging, dest)
    except Exception as exc:  # noqa: BLE001 - put the previous destination back
        problems = [f"restore failed while installing ({type(exc).__name__}: {exc})"]
        retained, recovery, state = [], [], 'not_restored'
        if moved_aside:
            try:
                os.replace(aside, dest)
                state = 'rolled_back'
                problems.append("the previous destination was moved back")
            except OSError as back:
                state = 'rollback_failed'
                recovery.append(str(aside))
                problems.append(f"moving the previous destination back failed ({back}); it is kept at {aside}")
        why = _remove(staging)
        if why:
            retained.append(str(staging))
            problems.append(f"also left behind: {why}")
        raise RestoreError(problems, state=state, recovery_files=recovery, retained_files=retained) from exc
    # ---- commit boundary: dest now holds the verified restore ----
    warnings, retained = [], []
    if moved_aside:
        why = _remove(aside)
        if why:
            warnings.append(f"the previous restore was not deleted: {why}")
            retained.append(str(aside))
    return {'restored': True, 'destination': str(dest), 'release_id': doc['release_id'],
            'manifest_sha256': doc['manifest_sha256'], 'previous': 'replaced' if had_previous else 'none',
            'report': str(dest / REPORT_NAME), 'cleanup_warnings': warnings, 'retained_files': retained,
            'approval': 'not_established', 's2_publication': 'not run'}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Offline archive and restore of a Player Props release package")
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('pack')
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--prediction-schedule', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True, help="new directory, outside the repository")
    i = sub.add_parser('inspect')
    i.add_argument('--archive', type=Path, required=True)
    i.add_argument('--descriptor', type=Path, required=True)
    r = sub.add_parser('restore')
    r.add_argument('--archive', type=Path, required=True)
    r.add_argument('--descriptor', type=Path, required=True)
    r.add_argument('--dest', type=Path, required=True, help="isolated directory, outside the repository")
    args = ap.parse_args(argv)
    try:
        if args.cmd == 'pack':
            report = pack(args.package, args.prediction_schedule, args.out)
        elif args.cmd == 'inspect':
            doc, _ = load_descriptor(args.descriptor)
            found = inspect_archive(args.archive, doc)
            report = {'ok': True, 'entries': len(found['entries']), 'approval': 'not_established',
                      'note': "inspection checks consistency only; nothing was written, approved or activated"}
        else:
            report = restore(args.archive, args.descriptor, args.dest)
    except RestoreError as exc:
        print(json.dumps({'ok': False, 'state': exc.state, 'problems': exc.problems,
                          'recovery_files': exc.recovery_files, 'retained_files': exc.retained_files}, indent=2))
        return 1
    except Exception as exc:  # noqa: BLE001 - could not run: nothing is claimed
        print(json.dumps({'ok': False, 'error': f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2
    print(json.dumps(report, indent=2))
    return 3 if report.get('cleanup_warnings') else 0


if __name__ == '__main__':
    sys.exit(main())
