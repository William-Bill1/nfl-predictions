"""
Player Props release manifest (S1): build and verify one candidate release
package, offline. Nothing here uploads, tags, activates or publishes anything.

A package is a directory holding ``release_manifest.json`` and the assets of
one coherent set: the training and inference play-by-play sources and their
manifests, the training and inference stats and their manifests, the two
schedules, the candidate and runtime players references, the exported model
bundle, the rehearsal summary that verified the set, and the predictions
snapshot that rehearsal verified. Each role has one fixed path (ROLES).

Training and inference inputs are different files and are never required to
share hashes. They are linked explicitly instead:

* the bundle names its training stats (stats manifest hash and every stats
  file hash) and their source; those must be this package's training assets;
* each stats manifest names its source; the inference stats must name the
  inference source, the training stats the training source;
* both use the bundle's feature definition and rolling windows, the training
  data ends before the inference feature cutoff, and the inference source
  covers every season the training source does.

The candidate reference (the builder's output) and the runtime reference (the
bytes prediction read) are separate assets with separate hashes. Their
relationship is declared (``byte_copy`` or ``reserialized``) and checked:
a byte copy must be identical; a reserialization must differ in bytes but
keep every identity column text-identical, row by row. Bytes are never
normalised. Eligibility evidence must name the runtime reference's exact hash.

Verification claims are evidence to cross-check, not to accept: the rehearsal
summary must list every run artifact in this package with the same hash, name
this bundle, these sources and this reference, and agree with the bundle's
windows. ``verify_package`` re-reads every asset once, checks it against the
manifest, re-runs the existing validators (source manifest, Stage 2 stats
provenance and freshness, the safe bundle-archive reader and bundle
verification) on a private copy of exactly those bytes, recomputes the
manifest's derived sections and compares them.

What stays a declared claim (recorded, not provable from the bytes): that the
downloads came from nflverse at the recorded times, that the rehearsal ran the
stages it reports, that the runtime reference was produced from the candidate,
git commits and library versions, the parent release and external tag or
pointer. Hashes detect a changed asset; they do not make a release immutable.

CLI (offline; exit 0 = valid, 1 = refused, 2 = could not run):
    python player_props/release_manifest.py build --run-dir RUN --candidate-reference CSV \\
        --reference-derivation reserialized --out PACKAGE_DIR [--parent PACKAGE_DIR] \\
        [--tag TAG] [--pointer TEXT]
    python player_props/release_manifest.py verify --package PACKAGE_DIR [--parent PACKAGE_DIR]

Policy: docs/architecture.md ("Release manifest (S1)").
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from player_props.aggregators import SCHEDULE_FILE, STATS_FILES  # noqa: E402
from player_props.provenance import MANIFEST_FILE as STATS_MANIFEST, TORONTO  # noqa: E402
from player_props.rehearsal import PRODUCTION_PATHS, STAGES, _overlaps, _real  # noqa: E402

MANIFEST_NAME = 'release_manifest.json'
MANIFEST_KIND = 'player_props_release'
MANIFEST_VERSION = 1
MAX_ASSET_BYTES = 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
CLOCK_SKEW = timedelta(minutes=5)
SOURCE_NAME = 'nfl_play_by_play_props.csv.gz'
SOURCE_MANIFEST = 'player_props_source_manifest.json'
SNAPSHOT = re.compile(r'runtime/data/player_props_predictions_week(\d{1,2})_(\d{4})\.csv')
# Reference columns prediction reads (identity and eligibility).
IDENTITY_COLUMNS = ('gsis_id', 'short_name', 'display_name', 'last_name', 'position', 'position_group')
DERIVATIONS = ('byte_copy', 'reserialized')
_RELEASE_ID = re.compile(r'\d{8}T\d{6}Z-[0-9a-f]{8}')
_HEX64 = re.compile(r'[0-9a-f]{64}')
CODE_FILES = ('player_props/release_manifest.py', 'player_props/rehearsal.py', 'player_props/provenance.py',
              'player_props/source_refresh.py', 'player_props/model_bundle.py',
              'player_props/bundle_store.py', 'player_props/predict.py')


def _stats_roles(prefix: str, folder: str) -> Dict[str, str]:
    return {f"{prefix}_stats_{k}": f"{folder}/{name}" for k, name in STATS_FILES.items()}


# role -> fixed path inside the package
ROLES: Dict[str, str] = {
    'training_source': f'source/training/{SOURCE_NAME}',
    'training_source_manifest': f'source/training/{SOURCE_MANIFEST}',
    'training_schedule': f'stats/training/{SCHEDULE_FILE}',
    **_stats_roles('training', 'stats/training'),
    'training_stats_manifest': f'stats/training/{STATS_MANIFEST}',
    'inference_source': f'source/inference/{SOURCE_NAME}',
    'inference_source_manifest': f'source/inference/{SOURCE_MANIFEST}',
    'inference_schedule': f'stats/inference/{SCHEDULE_FILE}',
    **_stats_roles('inference', 'stats/inference'),
    'inference_stats_manifest': f'stats/inference/{STATS_MANIFEST}',
    'reference_candidate': 'reference/candidate/players.csv',
    'reference_runtime': 'reference/runtime/players.csv',
    'bundle_archive': 'bundle/bundle.tar',
    'rehearsal_summary': 'evidence/summary.json',
    'verified_predictions': 'evidence/predictions_snapshot.csv',
}
# role -> where the rehearsal run directory holds it (rehearsal.py layout)
RUN_PATHS: Dict[str, str] = {
    'training_source': f'train/source/{SOURCE_NAME}',
    'training_source_manifest': f'train/source/{SOURCE_MANIFEST}',
    'training_schedule': f'train/stats/{SCHEDULE_FILE}',
    **{f"training_stats_{k}": f'train/stats/{n}' for k, n in STATS_FILES.items()},
    'training_stats_manifest': f'train/stats/{STATS_MANIFEST}',
    'inference_source': f'inference/source/{SOURCE_NAME}',
    'inference_source_manifest': f'inference/source/{SOURCE_MANIFEST}',
    'inference_schedule': f'runtime/data/{SCHEDULE_FILE}',
    **{f"inference_stats_{k}": f'runtime/data/{n}' for k, n in STATS_FILES.items()},
    'inference_stats_manifest': f'runtime/data/{STATS_MANIFEST}',
    'reference_runtime': 'runtime/data/players.csv',
    'bundle_archive': 'export/bundle.tar',
}
TOP_LEVEL = {'kind', 'manifest_version', 'release_id', 'created_at', 'parent_release', 'external', 'assets',
             'reference', 'code', 'runtime', 'derived', 'claims'}
DERIVED_SECTIONS = ('identities', 'bundle', 'eligibility', 'windows', 'inference_target', 'verification')
CLAIMS = [
    "the play-by-play downloads came from nflverse at the times their source manifests record",
    "the rehearsal executed the stages its summary reports, offline, with the assets named here",
    "the runtime reference was produced from the candidate reference as declared "
    "(only identity-column equality is checked)",
    "git commits, dirty flags and library versions recorded at build time",
    "the parent release and any external tag or pointer (descriptive only)",
]
IMMUTABILITY = ("hashes detect a changed asset; they do not make this package or any copy of it immutable, "
                "and do not prevent replacement or deletion")


class ReleaseError(RuntimeError):
    """A package can't be built or doesn't verify. ``problems`` lists every reason."""

    def __init__(self, problems):
        self.problems = [str(p) for p in problems]
        super().__init__("; ".join(self.problems))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _is_hash(v) -> bool:
    return isinstance(v, str) and _HEX64.fullmatch(v) is not None


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _timestamp(v) -> Optional[datetime]:
    if not isinstance(v, str):
        return None
    try:
        ts = datetime.fromisoformat(v)
    except ValueError:
        return None
    return ts if ts.tzinfo is not None else None


def _json(data: bytes, what: str):
    """Parsed JSON, or ReleaseError naming the line and column."""
    try:
        return json.loads(data.decode('utf-8'))
    except UnicodeDecodeError as exc:
        raise ReleaseError([f"{what} is not UTF-8 text ({exc})"]) from exc
    except json.JSONDecodeError as exc:
        raise ReleaseError([f"{what} is not valid JSON: line {exc.lineno}, column {exc.colno}: {exc.msg}"]) from exc


def _get(doc, *keys):
    for k in keys:
        if not isinstance(doc, dict):
            return None
        doc = doc.get(k)
    return doc


# ---------------------------------------------------------------------------
# Manifest structure
# ---------------------------------------------------------------------------

def _safe_name(name) -> bool:
    if not isinstance(name, str) or not name or '\\' in name or ':' in name or name.startswith('/'):
        return False
    # Split by hand: PurePosixPath would silently drop '.' and repeated slashes.
    return all(p not in ('', '.', '..') for p in name.split('/'))


def structure_problems(manifest) -> List[str]:
    """Every structural problem in a parsed release manifest (empty = ok)."""
    if not isinstance(manifest, dict):
        return [f"{MANIFEST_NAME} must be a JSON object"]
    if manifest.get('kind') != MANIFEST_KIND or manifest.get('manifest_version') != MANIFEST_VERSION:
        return [f"{MANIFEST_NAME} is not a version-{MANIFEST_VERSION} {MANIFEST_KIND} manifest "
                f"(kind {manifest.get('kind')!r}, manifest_version {manifest.get('manifest_version')!r})"]
    problems = []
    unknown, missing = set(manifest) - TOP_LEVEL, TOP_LEVEL - set(manifest)
    if unknown:
        problems.append(f"unexpected top-level key(s) {sorted(unknown)}")
    if missing:
        problems.append(f"missing top-level key(s) {sorted(missing)}")
    if not (isinstance(manifest.get('release_id'), str) and _RELEASE_ID.fullmatch(manifest['release_id'])):
        problems.append("release_id must look like 20261030T120000Z-1a2b3c4d")
    created = _timestamp(manifest.get('created_at'))
    if created is None:
        problems.append("created_at must be an ISO timestamp with a time zone")
    elif created > datetime.now(timezone.utc) + CLOCK_SKEW:
        problems.append("created_at is in the future")
    parent = manifest.get('parent_release')
    if parent is not None and not (isinstance(parent, dict) and set(parent) == {'release_id', 'manifest_sha256'}
                                   and isinstance(parent['release_id'], str)
                                   and _RELEASE_ID.fullmatch(parent['release_id'])
                                   and _is_hash(parent['manifest_sha256'])):
        problems.append("parent_release must be null or {release_id, manifest_sha256}")
    ext = manifest.get('external')
    if not (isinstance(ext, dict) and set(ext) == {'tag', 'pointer', 'note'}
            and all(v is None or (isinstance(v, str) and len(v) <= 500) for v in ext.values())):
        problems.append("external must be {tag, pointer, note}, each a string (500 characters at most) or null")
    ref = manifest.get('reference')
    if not (isinstance(ref, dict) and set(ref) == {'derivation', 'identity_columns'}
            and ref.get('derivation') in DERIVATIONS and ref.get('identity_columns') == list(IDENTITY_COLUMNS)):
        problems.append(f"reference must be {{derivation: one of {list(DERIVATIONS)}, "
                        f"identity_columns: {list(IDENTITY_COLUMNS)}}}")
    for key in ('code', 'runtime'):
        if not isinstance(manifest.get(key), dict):
            problems.append(f"{key} must be an object")
    derived = manifest.get('derived')
    if not (isinstance(derived, dict) and set(derived) == set(DERIVED_SECTIONS)):
        problems.append(f"derived must be an object with exactly {list(DERIVED_SECTIONS)}")
    if not (isinstance(manifest.get('claims'), list) and all(isinstance(c, str) for c in manifest['claims'])):
        problems.append("claims must be a list of strings")
    problems += asset_entry_problems(manifest.get('assets'))
    return problems


def asset_entry_problems(assets) -> List[str]:
    """Missing, unexpected, duplicate, conflicting, unsafe or malformed asset entries."""
    if not isinstance(assets, list):
        return ["assets must be a list of {role, name, bytes, sha256}"]
    problems, roles, names = [], {}, {}
    for i, a in enumerate(assets):
        where = f"assets[{i}]"
        if not isinstance(a, dict):
            problems.append(f"{where} must be an object")
            continue
        if set(a) != {'role', 'name', 'bytes', 'sha256'}:
            problems.append(f"{where} must have exactly role, name, bytes and sha256 (has {sorted(a)})")
            continue
        role, name = a['role'], a['name']
        if not isinstance(role, str):        # checked before any lookup: lists and objects aren't hashable
            problems.append(f"{where}.role must be a string naming one of the {len(ROLES)} asset roles, "
                            f"not {type(role).__name__} {repr(role)[:80]}")
            continue
        if not isinstance(name, str):
            problems.append(f"{where}.name must be a string path, not {type(name).__name__} "
                            f"{repr(name)[:80]}")
            continue
        if not _safe_name(name):
            problems.append(f"{where}: name {name!r} is not a safe relative path")
            continue
        if role not in ROLES:
            problems.append(f"{where}: unexpected role {role!r}")
            continue
        if name != ROLES[role]:
            problems.append(f"{where}: {role} must be stored at {ROLES[role]}, not {name!r}")
        if role in roles:
            problems.append(f"{where}: role {role} is listed more than once")
        if name in names:
            problems.append(f"{where}: name {name} is listed more than once")
        roles.setdefault(role, i)
        names.setdefault(name, i)
        if not _is_int(a['bytes']) or not 0 <= a['bytes'] <= MAX_ASSET_BYTES:
            problems.append(f"{where}: bytes must be an integer from 0 to {MAX_ASSET_BYTES}")
        if not _is_hash(a['sha256']):
            problems.append(f"{where}: sha256 must be a 64-character lowercase hex SHA-256")
    absent = [r for r in ROLES if r not in roles]
    if absent:
        problems.append(f"missing asset role(s) {absent}")
    return problems


# ---------------------------------------------------------------------------
# Reading a package
# ---------------------------------------------------------------------------

def _is_link(path: Path) -> bool:
    """A symlink, or (Windows) any reparse point such as a junction."""
    st = os.lstat(path)
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, 'st_file_attributes', 0) & 0x400)


def _read_regular(path: Path, root_real: Path, limit: int) -> bytes:
    """Bytes of a regular file that really lies inside root (no link, junction
    or reparse point anywhere below root), at most ``limit`` bytes."""
    rel = path.relative_to(root_real)
    cur = root_real
    for part in rel.parts:
        cur = cur / part
        if _is_link(cur):
            raise ReleaseError([f"{rel.as_posix()} is or passes through a link; refused"])
    if not stat.S_ISREG(os.lstat(path).st_mode):
        raise ReleaseError([f"{rel.as_posix()} is not a regular file"])
    if not _real(path).is_relative_to(root_real):
        raise ReleaseError([f"{rel.as_posix()} resolves outside the package"])
    size = os.path.getsize(path)
    if size > limit:
        raise ReleaseError([f"{rel.as_posix()} is larger than {limit} bytes"])
    with open(path, 'rb') as f:
        data = f.read(limit + 1)
    if len(data) > limit:
        raise ReleaseError([f"{rel.as_posix()} is larger than {limit} bytes"])
    return data


def read_package(package: Path | str) -> Tuple[dict, bytes, Dict[str, bytes]]:
    """(manifest, manifest bytes, {role: bytes}) after the structural, file and
    hash checks. Every asset is read once; later checks use these bytes only."""
    root = Path(package)
    if not root.is_dir():
        raise ReleaseError([f"package {root} is not a directory"])
    root_real = _real(root)
    manifest_bytes = _read_regular(root_real / MANIFEST_NAME, root_real, MAX_MANIFEST_BYTES) \
        if (root_real / MANIFEST_NAME).exists() else None
    if manifest_bytes is None:
        raise ReleaseError([f"package has no {MANIFEST_NAME}"])
    manifest = _json(manifest_bytes, MANIFEST_NAME)
    problems = structure_problems(manifest)
    if problems:
        raise ReleaseError([f"{MANIFEST_NAME}: {p}" for p in problems])
    listed = {a['name']: a for a in manifest['assets']}
    present = set()
    for dirpath, dirnames, filenames in os.walk(root_real):
        links = [d for d in dirnames if _is_link(Path(dirpath) / d)]
        problems += [f"{(Path(dirpath) / d).relative_to(root_real).as_posix()} is a link or junction; refused"
                     for d in links]
        dirnames[:] = [d for d in dirnames if d not in links]       # never followed
        for f in filenames:
            present.add((Path(dirpath) / f).relative_to(root_real).as_posix())
    unexpected = sorted(present - set(listed) - {MANIFEST_NAME})
    if unexpected:
        problems.append(f"unlisted file(s) in the package: {unexpected[:10]}")
    contents: Dict[str, bytes] = {}
    for name, entry in sorted(listed.items()):
        if name not in present:
            problems.append(f"{name} ({entry['role']}) is listed but missing")
            continue
        try:
            data = _read_regular(root_real / name, root_real, MAX_ASSET_BYTES)
        except ReleaseError as exc:
            problems += exc.problems
            continue
        except OSError as exc:
            problems.append(f"{name} can't be read ({exc})")
            continue
        if len(data) != entry['bytes']:
            problems.append(f"{name}: size {len(data)} != manifest {entry['bytes']}")
        elif _sha256(data) != entry['sha256']:
            problems.append(f"{name}: SHA-256 differs from the manifest")
        else:
            contents[entry['role']] = data
    if problems:
        raise ReleaseError(problems)
    return manifest, manifest_bytes, contents


# ---------------------------------------------------------------------------
# Content checks on a private copy of the verified bytes
# ---------------------------------------------------------------------------

def _text_csv(data: bytes, what: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, na_values=[''],
                           low_memory=False)
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise ReleaseError([f"{what} is not readable CSV ({exc})"]) from exc


def reference_problems(candidate: bytes, runtime: bytes, derivation: str) -> List[str]:
    """The declared candidate -> runtime relationship, checked without
    normalising either file."""
    problems = []
    same = _sha256(candidate) == _sha256(runtime)
    if derivation == 'byte_copy' and not same:
        return ["the runtime reference is declared a byte copy of the candidate, but their bytes differ"]
    if derivation == 'reserialized' and same:
        return ["the runtime reference is declared reserialized, but it is byte-identical to the "
                "candidate; declare byte_copy"]
    cand, run = _text_csv(candidate, "the candidate reference"), _text_csv(runtime, "the runtime reference")
    for label, df in (("candidate", cand), ("runtime", run)):
        missing = [c for c in IDENTITY_COLUMNS if c not in df.columns]
        if missing:
            problems.append(f"the {label} reference lacks identity column(s) {missing}")
        elif df['gsis_id'].isna().any() or df['gsis_id'].duplicated().any():
            problems.append(f"the {label} reference has blank or repeated gsis_id values")
    if problems:
        return problems
    if len(cand) != len(run):
        return [f"the runtime reference has {len(run)} rows, the candidate {len(cand)}"]
    for col in IDENTITY_COLUMNS:
        a, b = cand[col].fillna('\0missing'), run[col].fillna('\0missing')
        diff = (a.to_numpy() != b.to_numpy())
        if diff.any():
            i = int(diff.argmax())
            problems.append(f"identity column {col} differs in {int(diff.sum())} row(s) between the candidate "
                            f"and runtime references (first: row {i + 2}, {cand[col].iloc[i]!r} vs "
                            f"{run[col].iloc[i]!r})")
    return problems


def _write_snapshot(contents: Dict[str, bytes], base: Path) -> None:
    for role, data in contents.items():
        path = base / ROLES[role]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _stats_files(stats_manifest: dict) -> dict:
    return {e['file']: e['sha256'] for e in stats_manifest['stats'].values()}


def derive(contents: Dict[str, bytes], bundle_manifest: dict, bundle_manifest_sha: str,
           export_index: dict) -> dict:
    """The manifest's derived sections, extracted from the asset bytes alone
    (no relationship is judged here). Malformed evidence raises ReleaseError."""
    j = {r: _json(contents[r], ROLES[r]) for r in ('training_source_manifest', 'inference_source_manifest',
                                                   'training_stats_manifest', 'inference_stats_manifest',
                                                   'rehearsal_summary')}
    bad = [ROLES[r] for r, doc in j.items() if not isinstance(doc, dict)]
    if bad:
        raise ReleaseError([f"{name} must be a JSON object" for name in bad])
    try:
        return _derive(contents, j, bundle_manifest, bundle_manifest_sha, export_index)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ReleaseError([f"the rehearsal summary or a source/stats manifest has an unexpected shape "
                            f"({type(exc).__name__}: {exc})"]) from exc


def _derive(contents: Dict[str, bytes], j: dict, bundle_manifest: dict, bundle_manifest_sha: str,
            export_index: dict) -> dict:
    summary = j['rehearsal_summary']

    def source(prefix):
        m = j[f'{prefix}_source_manifest']
        cov = _get(m, 'source', 'coverage') or {}
        return {'sha256': _sha256(contents[f'{prefix}_source']), 'bytes': len(contents[f'{prefix}_source']),
                'manifest_sha256': _sha256(contents[f'{prefix}_source_manifest']),
                'seasons': sorted((cov.get('seasons') or {}).keys()),
                'first_date': cov.get('first_date'), 'last_date': cov.get('last_date'),
                'schedule_sha256': _get(m, 'schedule', 'sha256')}

    def stats(prefix):
        m = j[f'{prefix}_stats_manifest']
        return {'manifest_sha256': _sha256(contents[f'{prefix}_stats_manifest']),
                'files': {ROLES[f'{prefix}_stats_{k}'].rsplit('/', 1)[1]: _sha256(contents[f'{prefix}_stats_{k}'])
                          for k in STATS_FILES},
                'source_sha256': _get(m, 'source', 'sha256'),
                'feature_definition': _get(m, 'aggregation', 'feature_definition'),
                'rolling_windows': _get(m, 'aggregation', 'rolling_windows'),
                'written_at': m.get('written_at')}

    snap_sha = _sha256(contents['verified_predictions'])
    targets = sorted({(int(mt.group(2)), int(mt.group(1))) for k, v in (summary.get('artifacts') or {}).items()
                      if (mt := SNAPSHOT.fullmatch(k)) and v == snap_sha})
    models = bundle_manifest['models']
    windows = {n: {'line': e['line_value'],
                   'training': {k: e['training'][k] for k in ('seasons', 'rows', 'first_date', 'last_date')},
                   'evaluation': {k: e['evaluation'][k] for k in ('seasons', 'rows', 'first_date', 'last_date')}}
               for n, e in sorted(models.items())}
    elig = summary.get('eligibility') or {}
    verify = _get(summary, 'outputs', 'verify_outputs') or {}
    return {
        'identities': {
            'training_source': source('training'), 'inference_source': source('inference'),
            'training_stats': stats('training'), 'inference_stats': stats('inference'),
            'training_schedule_sha256': _sha256(contents['training_schedule']),
            'inference_schedule_sha256': _sha256(contents['inference_schedule']),
            'reference_candidate': {'sha256': _sha256(contents['reference_candidate']),
                                    'bytes': len(contents['reference_candidate'])},
            'reference_runtime': {'sha256': _sha256(contents['reference_runtime']),
                                  'bytes': len(contents['reference_runtime'])},
        },
        'bundle': {'bundle_id': bundle_manifest['bundle_id'], 'manifest_sha256': bundle_manifest_sha,
                   'archive_sha256': _sha256(contents['bundle_archive']),
                   'export_files': len(export_index['files']),
                   'feature_contract': bundle_manifest['feature_contract'], 'scorer': bundle_manifest['scorer'],
                   'training_source': bundle_manifest['training_source'],
                   'code': bundle_manifest['code'], 'libraries': bundle_manifest['libraries'],
                   'models': sorted(models)},
        'eligibility': {k: elig.get(k) for k in ('policy', 'restated_policy', 'field', 'players_reference_sha256',
                                                 'players_reference_source', 'verified')},
        'windows': {'models': windows,
                    'training_seasons': sorted({s for w in windows.values() for s in w['training']['seasons']}),
                    'evaluation_seasons': sorted({s for w in windows.values() for s in w['evaluation']['seasons']}),
                    'trained_through': bundle_manifest['training_source']['last_game_date'],
                    'feature_cutoff': _get(summary, 'coverage', 'inputs', 'feature_cutoff')},
        'inference_target': ({'season': targets[0][0], 'week': targets[0][1]} if len(targets) == 1
                             else {'season': None, 'week': None, 'candidates': [list(t) for t in targets]}),
        'verification': {'kind': summary.get('kind'), 'summary_version': summary.get('summary_version'),
                         'run_id': summary.get('run_id'), 'status': summary.get('status'),
                         'finished_at': summary.get('finished_at'),
                         'summary_sha256': _sha256(contents['rehearsal_summary']),
                         # Every record, as written (a malformed one appears as [None, None]).
                         'stages': ([[s.get('stage'), s.get('status')] if isinstance(s, dict) else [None, None]
                                     for s in summary['stages']] if isinstance(summary.get('stages'), list)
                                    else None),
                         'completeness_status': _get(summary, 'completeness', 'status'),
                         'expected_rows': _get(summary, 'completeness', 'expected'),
                         'predictions_verified': verify.get('predictions_verified'),
                         'snapshot_sha256': snap_sha,
                         'models_metric_parity': verify.get('models_metric_parity')},
    }


def stage_problems(stages) -> List[str]:
    """Why a rehearsal summary's stage records are not exactly the supported
    stages (rehearsal.STAGES), in order, each once, each with status 'ok'."""
    if not isinstance(stages, list):
        return [f"the rehearsal summary's stages must be a list of {{stage, status}} records, "
                f"not {type(stages).__name__}"]
    problems, names = [], []
    for i, s in enumerate(stages):
        where = f"rehearsal summary stages[{i}]"
        if not isinstance(s, dict):
            problems.append(f"{where} must be an object, not {type(s).__name__}")
            continue
        if not isinstance(s.get('stage'), str) or not isinstance(s.get('status'), str):
            problems.append(f"{where} must have a string stage and status (has {repr(s)[:120]})")
            continue
        unexpected = set(s) - {'stage', 'status', 'error'}
        if unexpected:
            problems.append(f"{where} ({s['stage']}) has unexpected field(s) {sorted(unexpected)}")
        if s['stage'] not in STAGES:
            problems.append(f"{where}: unknown stage {s['stage']!r}")
        elif s['stage'] in names:
            problems.append(f"{where}: stage {s['stage']} is recorded more than once")
        if s['status'] != 'ok':
            problems.append(f"{where}: stage {s['stage']} has status {s['status']!r}, not 'ok'")
        elif 'error' in s:
            problems.append(f"{where}: stage {s['stage']} is 'ok' but records an error")
        names.append(s['stage'])
    missing = [n for n in STAGES if n not in names]
    if missing:
        problems.append(f"the rehearsal summary has no record of stage(s) {missing}; all {len(STAGES)} "
                        f"supported stages must be recorded as 'ok'")
    elif not problems and names != STAGES:
        problems.append(f"the rehearsal summary records the stages out of order: {names}")
    return problems


def relationship_problems(contents: Dict[str, bytes], d: dict, manifest: dict, snapshot: Path) -> List[str]:
    """Do the assets describe one coherent set? Uses the existing validators
    on the private copy ``snapshot`` of exactly the verified bytes."""
    from player_props.features import FEATURE_CONTRACT
    from player_props.predict import ELIGIBILITY_FIELD, ELIGIBILITY_POLICY
    from player_props.provenance import _read_stats, check_inputs, check_provenance
    from player_props.source_refresh import SourceRefreshError, verify_installed
    problems: List[str] = []
    ids, b, w = d['identities'], d['bundle'], d['windows']
    summary = _json(contents['rehearsal_summary'], ROLES['rehearsal_summary'])

    # Each source against its own manifest (source_refresh.verify_installed).
    for prefix in ('training', 'inference'):
        try:
            verify_installed(snapshot / f'source/{prefix}')
        except SourceRefreshError as exc:
            problems += [f"{prefix} source: {p}" for p in exc.problems]
        if ids[f'{prefix}_source']['schedule_sha256'] != ids[f'{prefix}_schedule_sha256']:
            problems.append(f"the {prefix} source manifest names schedule "
                            f"{str(ids[f'{prefix}_source']['schedule_sha256'])[:12]}, not this package's "
                            f"{prefix} schedule {ids[f'{prefix}_schedule_sha256'][:12]}")

    # Each stats set against its Stage 2 manifest, and that manifest against its source.
    for prefix in ('training', 'inference'):
        data_dir = snapshot / f'stats/{prefix}'
        raw, tables, read = _read_stats(data_dir)
        found, _, _ = check_provenance(data_dir, raw, tables)
        problems += [f"{prefix} stats: {p}" for p in read + found]
        st, src = ids[f'{prefix}_stats'], ids[f'{prefix}_source']
        if st['source_sha256'] != src['sha256']:
            problems.append(f"the {prefix} stats were built from source {str(st['source_sha256'])[:12]}, "
                            f"not this package's {prefix} source {src['sha256'][:12]}")

    # The bundle names its actual training stats and their source.
    ts = b['training_source']
    tstats, tsrc = ids['training_stats'], ids['training_source']
    if ts.get('stats_manifest_sha256') != tstats['manifest_sha256']:
        problems.append("the bundle was trained on stats manifest "
                        f"{str(ts.get('stats_manifest_sha256'))[:12]}, not this package's training stats manifest "
                        f"{tstats['manifest_sha256'][:12]}")
    if ts.get('stats_files') != tstats['files']:
        problems.append("the bundle's training stats file hashes differ from this package's training stats")
    if _get(ts, 'play_by_play', 'sha256') != tsrc['sha256']:
        problems.append("the bundle's training play-by-play hash differs from this package's training source")
    if ts.get('feature_definition') != tstats['feature_definition']:
        problems.append("the bundle's feature definition differs from its training stats'")

    # Training and inference: distinct inputs, linked explicitly.
    istats, isrc = ids['inference_stats'], ids['inference_source']
    if istats['feature_definition'] != b['feature_contract'].get('feature_definition') \
            or b['feature_contract'] != FEATURE_CONTRACT:
        problems.append(f"feature contract mismatch: bundle {b['feature_contract']}, inference stats "
                        f"{istats['feature_definition']!r}, this code {FEATURE_CONTRACT}")
    if istats['rolling_windows'] != tstats['rolling_windows']:
        problems.append(f"inference stats use rolling windows {istats['rolling_windows']}, "
                        f"training stats {tstats['rolling_windows']}")
    missing = sorted(set(tsrc['seasons']) - set(isrc['seasons']))
    if missing:
        problems.append(f"the inference source lacks training season(s) {missing}")
    cutoff = w['feature_cutoff']
    if not isinstance(cutoff, str) or not (isinstance(w['trained_through'], str) and w['trained_through'] < cutoff):
        problems.append(f"training data (through {w['trained_through']}) must end before the inference "
                        f"feature cutoff ({cutoff})")

    # Inference target and inputs: re-run the Stage 2 gate on the inference stats.
    target = d['inference_target']
    if target.get('season') is None:
        problems.append("the rehearsal summary lists the verified predictions snapshot under "
                        f"{len(target.get('candidates', []))} target weeks; exactly one is required")
    else:
        as_of = _timestamp(summary.get('finished_at'))
        check = check_inputs(target['season'], target['week'], snapshot / 'stats/inference',
                             today=as_of.astimezone(TORONTO).date() if as_of else None)
        problems += [f"inference inputs for {target['season']} week {target['week']}: {p}"
                     for p in check.provenance_problems + check.freshness_problems]
        if check.summary.get('feature_cutoff') != cutoff:
            problems.append(f"the Stage 2 feature cutoff {check.summary.get('feature_cutoff')} differs from the "
                            f"rehearsal's {cutoff}")

    # Eligibility evidence names the exact runtime reference bytes.
    elig, ref = d['eligibility'], manifest['reference']
    if elig['policy'] != ELIGIBILITY_POLICY or elig['restated_policy'] != ELIGIBILITY_POLICY \
            or elig['field'] != ELIGIBILITY_FIELD:
        problems.append(f"eligibility policy {elig['policy']!r} (restated {elig['restated_policy']!r}, field "
                        f"{elig['field']!r}) is not this code's {ELIGIBILITY_POLICY!r} on {ELIGIBILITY_FIELD!r}")
    if elig['players_reference_source'] != 'verification' or elig['verified'] is not True:
        problems.append("the eligibility evidence was not recorded by a passing verification "
                        f"(source {elig['players_reference_source']!r}, verified {elig['verified']!r})")
    run_ref, cand_ref = ids['reference_runtime']['sha256'], ids['reference_candidate']['sha256']
    if elig['players_reference_sha256'] != run_ref:
        which = "the candidate reference" if elig['players_reference_sha256'] == cand_ref else "another file"
        problems.append(f"the eligibility evidence names players reference "
                        f"{str(elig['players_reference_sha256'])[:12]} ({which}), not the runtime reference "
                        f"{run_ref[:12]} prediction read")
    if _get(summary, 'outputs', 'verify_outputs', 'players_reference_sha256') != run_ref:
        problems.append("the rehearsal's output verification names another players reference")
    problems += reference_problems(contents['reference_candidate'], contents['reference_runtime'],
                                   ref['derivation'])

    # The verification evidence describes these exact assets.
    v = d['verification']
    if v['kind'] != 'player_props_rehearsal' or v['summary_version'] != 1:
        problems.append("the evidence is not a version-1 player_props_rehearsal summary")
    if v['status'] != 'succeeded':
        problems.append(f"the rehearsal's status is {v['status']!r}, not 'succeeded'")
    problems += stage_problems(summary.get('stages'))
    if v['completeness_status'] != 'complete' or v['predictions_verified'] != v['expected_rows']:
        problems.append(f"the rehearsal's completeness is {v['completeness_status']!r} "
                        f"({v['predictions_verified']} verified of {v['expected_rows']} expected)")
    artifacts = summary.get('artifacts') if isinstance(summary.get('artifacts'), dict) else {}
    for role, run_path in RUN_PATHS.items():
        if artifacts.get(run_path) != _sha256(contents[role]):
            problems.append(f"the rehearsal summary records {run_path} as "
                            f"{str(artifacts.get(run_path))[:12]}, not this package's {role} "
                            f"{_sha256(contents[role])[:12]}")
    for where in ('train', 'runtime'):
        key = f"{where}/models/bundles/{b['bundle_id']}/manifest.json"
        if artifacts.get(key) != b['manifest_sha256']:
            problems.append(f"the rehearsal summary has no {key} matching this bundle's manifest")
    for k, want in (('verify_outputs', 'snapshot_sha256'), ('publish', 'snapshot_sha256')):
        if _get(summary, 'outputs', k, want) != v['snapshot_sha256']:
            problems.append(f"the rehearsal's {k} names another predictions snapshot")
    if _get(summary, 'bundle', 'bundle_id') != b['bundle_id'] \
            or _get(summary, 'bundle', 'feature_contract') != b['feature_contract'] \
            or _get(summary, 'bundle', 'trained_through') != w['trained_through']:
        problems.append("the rehearsal summary names another bundle, contract or training end")
    for prefix, key in (('training', 'training_source'), ('inference', 'inference_source')):
        if _get(summary, 'coverage', key, 'sha256') != ids[f'{prefix}_source']['sha256']:
            problems.append(f"the rehearsal summary's {key} differs from this package's {prefix} source")
    if summary.get('windows') != w['models']:
        problems.append("the rehearsal summary's training/evaluation windows differ from the bundle's")
    return problems


def _bundle(contents: Dict[str, bytes], snapshot: Path) -> Tuple[dict, str, dict, List[str]]:
    """(bundle manifest, its sha, export index, problems) via the safe archive
    reader and full bundle verification, on a private extraction."""
    from player_props.bundle_store import read_archive
    from player_props.features import PROP_LINES
    from player_props.model_bundle import BUNDLES_DIR, MANIFEST_FILE, BundleError, verify_bundle_dir
    try:
        index, files = read_archive(snapshot / ROLES['bundle_archive'])
    except BundleError as exc:
        return None, None, None, [f"bundle archive: {p}" for p in exc.problems]
    bundle_id = index['bundle_id']
    target = snapshot / '_bundle' / BUNDLES_DIR / bundle_id
    target.mkdir(parents=True)
    for name, data in files.items():
        (target / PurePosixPath(name).name).write_bytes(data)
    try:
        verify_bundle_dir(target, PROP_LINES, index['manifest_sha256'], bundle_id)
    except BundleError as exc:
        return None, None, None, [f"bundle {bundle_id}: {p}" for p in exc.problems]
    manifest_bytes = files[f"{bundle_id}/{MANIFEST_FILE}"]
    return json.loads(manifest_bytes.decode('utf-8')), _sha256(manifest_bytes), index, []


def _differences(a, b, path='') -> List[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}.{k}" if path else str(k))
            else:
                out += _differences(a[k], b[k], f"{path}.{k}" if path else str(k))
        return out
    return [] if a == b else [path or '(root)']


def verify_package(package: Path | str, parent: Optional[Path | str] = None) -> dict:
    """Verify a package completely. Returns the report
    {'ok', 'release_id', 'manifest_sha256', 'problems', 'verified', 'declared'}.
    Raises ReleaseError when the manifest or the assets themselves fail (structure,
    missing/unlisted/unsafe files, size or hash), since nothing else can be trusted then."""
    manifest, manifest_bytes, contents = read_package(package)
    problems: List[str] = []
    with tempfile.TemporaryDirectory(prefix='release-verify-') as tmp:
        snapshot = Path(tmp)
        _write_snapshot(contents, snapshot)
        bundle_manifest, bundle_sha, index, found = _bundle(contents, snapshot)
        problems += found
        if bundle_manifest is not None:
            try:
                d = derive(contents, bundle_manifest, bundle_sha, index)
            except ReleaseError as exc:
                d, problems = None, problems + exc.problems
            if d is not None:
                for section in DERIVED_SECTIONS:
                    diff = _differences(manifest['derived'].get(section), d[section])
                    if diff:
                        problems.append(f"derived.{section} doesn't match the assets at "
                                        f"{', '.join(diff[:5])}{' ...' if len(diff) > 5 else ''}")
                problems += relationship_problems(contents, d, manifest, snapshot)
    declared = list(CLAIMS)
    parent_doc = manifest['parent_release']
    if parent_doc is not None:
        if parent is None:
            declared.append(f"parent release {parent_doc['release_id']} (not checked: no parent package given)")
        else:
            try:
                p_manifest, p_bytes, _ = read_package(parent)
            except ReleaseError as exc:
                problems += [f"parent package: {p}" for p in exc.problems]
            else:
                if _sha256(p_bytes) != parent_doc['manifest_sha256'] or p_manifest['release_id'] != parent_doc['release_id']:
                    problems.append("the parent package is not the parent this manifest names")
                elif _timestamp(p_manifest['created_at']) >= _timestamp(manifest['created_at']):
                    problems.append("the parent release was not created before this one")
    elif parent is not None:
        problems.append("a parent package was given, but this manifest names no parent release")
    return {'ok': not problems, 'release_id': manifest['release_id'],
            'manifest_sha256': _sha256(manifest_bytes), 'problems': problems,
            'verified': [] if problems else [
                "every asset matches its listed size and SHA-256; no asset is missing, unlisted, duplicated, "
                "linked or outside the package",
                "each source matches its own manifest and names this package's schedule",
                "each stats set matches its Stage 2 manifest, which names this package's matching source",
                "the bundle archive passes the safe reader and full bundle verification, and names this "
                "package's training stats manifest, stats files and source",
                "the inference stats pass the Stage 2 gate for the target week, use the bundle's feature "
                "definition and windows, and start after the training data ends",
                "the eligibility evidence names this runtime reference's exact bytes; the candidate and "
                f"runtime references keep separate hashes and match the declared {manifest['reference']['derivation']} "
                "relationship on every identity column",
                "the rehearsal summary records each of the supported stages once, in order, as 'ok', lists "
                "every run asset here with the same hash, and agrees with the bundle, sources, windows and "
                "verified predictions",
                "every derived section of the manifest equals what the assets show",
            ] + (["the parent package matches parent_release"] if parent is not None and parent_doc else []),
            'declared': declared, 'immutability': IMMUTABILITY}


# ---------------------------------------------------------------------------
# Building a package
# ---------------------------------------------------------------------------

def _code_and_runtime() -> Tuple[dict, dict]:
    from player_props.model_bundle import library_versions
    code = {'files': {f: _sha256((ROOT / f).read_bytes()) for f in CODE_FILES if (ROOT / f).exists()},
            'git_commit': None, 'git_dirty': None}
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, timeout=10)
        status = subprocess.run(['git', 'status', '--porcelain', '--', 'player_props'], cwd=ROOT,
                                capture_output=True, text=True, timeout=10)
        if commit.returncode == 0 and status.returncode == 0:
            code['git_commit'], code['git_dirty'] = commit.stdout.strip(), bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return code, {**library_versions(), 'platform': platform.platform()}


def inputs_from_run(run_dir: Path | str, candidate_reference: Path | str) -> Dict[str, Path]:
    """role -> file for a finished rehearsal run directory plus the candidate
    reference the runtime reference was produced from."""
    run_dir = Path(run_dir)
    paths = {role: run_dir / rel for role, rel in RUN_PATHS.items()}
    paths['rehearsal_summary'] = run_dir / 'summary.json'
    snaps = sorted(p for p in (run_dir / 'runtime' / 'data').glob('player_props_predictions_week*_*.csv')
                   if SNAPSHOT.fullmatch(p.relative_to(run_dir).as_posix()))
    if len(snaps) != 1:
        raise ReleaseError([f"expected exactly one predictions snapshot in {run_dir / 'runtime' / 'data'}, "
                            f"found {len(snaps)}"])
    paths['verified_predictions'] = snaps[0]
    paths['reference_candidate'] = Path(candidate_reference)
    return paths


def check_output_location(out_dir: Path, inputs: Dict[str, Path], protected: Optional[List[Path]] = None,
                          extra: Optional[List[Path]] = None, parent: Optional[Path] = None) -> Path:
    """The resolved output directory, or ReleaseError if it overlaps the
    repository, a production path, any input file, an input directory or the
    parent package (equal, ancestor or descendant; paths compared fully
    resolved, links and junctions included). Nothing is created or read."""
    real = _real(out_dir)
    if parent is not None and _overlaps(real, _real(parent)):
        rel = ("is" if real == _real(parent) else
               "is inside" if real.is_relative_to(_real(parent)) else "contains")
        raise ReleaseError([f"output {out_dir} (resolves to {real}) {rel} the parent package {parent} "
                            f"(resolves to {_real(parent)}); the parent is an input and must not be replaced "
                            f"or written into"])
    for p in [ROOT] + list(PRODUCTION_PATHS) + list(protected or []):
        if _overlaps(real, _real(p)):
            raise ReleaseError([f"output {out_dir} (resolves to {real}) overlaps protected location {p}"])
    for role, p in inputs.items():
        if _overlaps(real, _real(p)):
            raise ReleaseError([f"output {out_dir} overlaps input {role} ({p}); choose a separate directory"])
    for p in extra or []:
        if _overlaps(real, _real(p)):
            raise ReleaseError([f"output {out_dir} overlaps input directory {p}"])
    return real


def build_package(inputs: Dict[str, Path], out_dir: Path | str, derivation: str,
                  parent: Optional[Path | str] = None, tag: Optional[str] = None,
                  pointer: Optional[str] = None, protected: Optional[List[Path]] = None,
                  input_dirs: Optional[List[Path]] = None) -> dict:
    """Assemble, verify and install a package at ``out_dir``. Returns the
    verification report. The package is written to a sibling staging
    directory and verified there; only a package that verifies replaces
    ``out_dir``. On any catchable failure the previous ``out_dir`` is left as
    it was (or restored) and the staging directory removed."""
    out_dir = Path(out_dir)
    if derivation not in DERIVATIONS:
        raise ReleaseError([f"reference derivation must be one of {list(DERIVATIONS)}"])
    if set(inputs) != set(ROLES):
        raise ReleaseError([f"inputs must cover exactly the roles {sorted(ROLES)} "
                            f"(missing {sorted(set(ROLES) - set(inputs))}, unexpected {sorted(set(inputs) - set(ROLES))})"])
    # Before anything is read, created or moved: the parent is an input too.
    out_real = check_output_location(out_dir, inputs, protected, input_dirs,
                                     None if parent is None else Path(parent))
    parent_doc = None
    if parent is not None:
        p_manifest, p_bytes, _ = read_package(parent)
        parent_doc = {'release_id': p_manifest['release_id'], 'manifest_sha256': _sha256(p_bytes)}
    out_real.parent.mkdir(parents=True, exist_ok=True)
    staging = out_real.parent / f".{out_real.name}.staging-{secrets.token_hex(4)}"
    os.mkdir(staging)
    try:
        contents = {}
        for role, src in sorted(inputs.items()):
            st = os.stat(src)
            if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_ASSET_BYTES:
                raise ReleaseError([f"input {role} ({src}) is not a regular file of at most {MAX_ASSET_BYTES} bytes"])
            data = Path(src).read_bytes()          # read once; everything below uses these bytes
            dest = staging / ROLES[role]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            contents[role] = data
        with tempfile.TemporaryDirectory(prefix='release-build-') as tmp:
            snapshot = Path(tmp)
            _write_snapshot(contents, snapshot)
            bundle_manifest, bundle_sha, index, problems = _bundle(contents, snapshot)
            if problems:
                raise ReleaseError(["not packaged: " + p for p in problems])
            derived = derive(contents, bundle_manifest, bundle_sha, index)
        code, runtime = _code_and_runtime()
        manifest = {
            'kind': MANIFEST_KIND, 'manifest_version': MANIFEST_VERSION,
            'release_id': f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}",
            # When this package was assembled - not when its data or models were produced.
            'created_at': _now(), 'parent_release': parent_doc,
            'external': {'tag': tag, 'pointer': pointer,
                         'note': "descriptive only: nothing was uploaded, tagged, released or activated"},
            'assets': [{'role': r, 'name': ROLES[r], 'bytes': len(contents[r]), 'sha256': _sha256(contents[r])}
                       for r in ROLES],
            'reference': {'derivation': derivation, 'identity_columns': list(IDENTITY_COLUMNS)},
            'code': code, 'runtime': runtime, 'derived': derived, 'claims': list(CLAIMS),
        }
        (staging / MANIFEST_NAME).write_bytes(json.dumps(manifest, indent=2, sort_keys=True).encode('utf-8'))
        report = verify_package(staging, parent)
        if not report['ok']:
            raise ReleaseError(["not packaged: " + p for p in report['problems']])
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    previous = None
    if out_real.exists():
        previous = out_real.parent / f".{out_real.name}.previous-{secrets.token_hex(4)}"
        os.rename(out_real, previous)
    try:
        os.rename(staging, out_real)
    except BaseException:
        if previous is not None:
            os.rename(previous, out_real)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if previous is not None:
        shutil.rmtree(previous, ignore_errors=True)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build or verify a Player Props release package (offline)")
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--run-dir', type=Path, required=True, help="a finished rehearsal run directory")
    b.add_argument('--candidate-reference', type=Path, required=True)
    b.add_argument('--reference-derivation', choices=DERIVATIONS, required=True)
    b.add_argument('--out', type=Path, required=True, help="package directory (outside the repository)")
    b.add_argument('--parent', type=Path)
    b.add_argument('--tag')
    b.add_argument('--pointer')
    v = sub.add_parser('verify')
    v.add_argument('--package', type=Path, required=True)
    v.add_argument('--parent', type=Path)
    args = ap.parse_args(argv)
    try:
        if args.cmd == 'build':
            report = build_package(inputs_from_run(args.run_dir, args.candidate_reference), args.out,
                                   args.reference_derivation, args.parent, args.tag, args.pointer,
                                   input_dirs=[args.run_dir])
        else:
            report = verify_package(args.package, args.parent)
    except ReleaseError as exc:
        print(json.dumps({'ok': False, 'problems': exc.problems}, indent=2))
        return 1
    except Exception as exc:  # noqa: BLE001 - could not run: not verified
        print(json.dumps({'ok': False, 'error': f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
