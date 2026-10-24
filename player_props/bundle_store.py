"""
Durable storage for Stage 4 model bundles (Stage 5A): export, validated import,
and an inventory. Nothing here is wired into a workflow yet.

* ``export_bundle`` writes one verified, immutable bundle to a tar archive:
  ``export.json`` (bundle id, manifest hash, every file's SHA-256 and size)
  plus ``<bundle_id>/<file>.json`` entries. Entries are regular files with
  fixed metadata, so the same bundle exports byte-identically.
* ``import_bundle`` refuses anything it can't fully check before writing:
  unreadable or truncated archives, non-regular entries (links, devices,
  directories), absolute or traversing paths, unexpected or duplicate names,
  oversized entries, hash or size mismatches, and a bundle id that is already
  installed with different content. It then, under publisher_lock, stages the
  files beside the bundles, verifies the staged bundle exactly as prediction
  loads it (model_bundle.verify_bundle_dir), renames it into place, and only
  then switches the pointer (model_bundle.switch_pointer). Any failure removes
  the staged copy and leaves the previous current bundle in use.
* ``inventory`` lists installed bundles, which one is current, whether each
  verifies, and any leftover staging directories. It never deletes anything.

Retention: bundles are immutable and are never deleted automatically. The
current bundle, and the only verified copy of any bundle, must not be removed;
pruning old bundles is a manual decision after checking ``inventory``.

CLI:
    python player_props/bundle_store.py export --models-dir DIR --out FILE.tar [--bundle-id ID]
    python player_props/bundle_store.py import --models-dir DIR --archive FILE.tar [--no-activate]
    python player_props/bundle_store.py inventory --models-dir DIR
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
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from player_props.features import PROP_LINES  # noqa: E402
from player_props.model_bundle import (  # noqa: E402
    BUNDLES_DIR, MANIFEST_FILE, POINTER_FILE, BundleError, _BUNDLE_ID, load_bundle, publisher_lock,
    switch_pointer, verify_bundle_dir,
)

EXPORT_FILE = 'export.json'
EXPORT_KIND = 'player_props_bundle_export'
EXPORT_VERSION = 1
MAX_ENTRY_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
_FILE_NAME = re.compile(r'[a-z_]+\.json')


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_bundle(models_dir: Path | str, out_path: Path | str, bundle_id: Optional[str] = None,
                  prop_lines: dict = PROP_LINES) -> dict:
    """Export a verified bundle (default: the current one) to ``out_path``.
    Returns the export index. The archive is written to a temporary file and
    renamed, so a failed export never leaves a partial archive at out_path."""
    models_dir, out_path = Path(models_dir), Path(out_path)
    if bundle_id is None:
        bundle_id = load_bundle(models_dir, prop_lines)['manifest']['bundle_id']
    if not isinstance(bundle_id, str) or not _BUNDLE_ID.fullmatch(bundle_id):
        raise BundleError([f"bundle id {bundle_id!r} is malformed"])
    bundle_dir = models_dir / BUNDLES_DIR / bundle_id
    manifest_bytes = (bundle_dir / MANIFEST_FILE).read_bytes()
    verify_bundle_dir(bundle_dir, prop_lines, _sha256(manifest_bytes), bundle_id)
    files = {}
    for path in sorted(bundle_dir.iterdir()):
        if not path.is_file() or not _FILE_NAME.fullmatch(path.name):
            raise BundleError([f"bundle {bundle_id} contains an unexpected entry {path.name}"])
        data = path.read_bytes()
        files[f"{bundle_id}/{path.name}"] = {'sha256': _sha256(data), 'bytes': len(data), 'data': data}
    index = {'kind': EXPORT_KIND, 'export_version': EXPORT_VERSION, 'bundle_id': bundle_id,
             'manifest_sha256': _sha256(manifest_bytes),
             'files': {k: {'sha256': v['sha256'], 'bytes': v['bytes']} for k, v in files.items()}}
    tmp = out_path.with_name(f".{out_path.name}.{secrets.token_hex(4)}.tmp")
    try:
        with tarfile.open(tmp, 'w') as tar:
            for name, data in [(EXPORT_FILE, json.dumps(index, indent=2, sort_keys=True).encode())] + \
                    [(k, v['data']) for k, v in files.items()]:
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime, info.uid, info.gid = len(data), 0o644, 0, 0, 0
                info.uname = info.gname = ''
                tar.addfile(info, io.BytesIO(data))
        os.replace(tmp, out_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return index


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def _entry_problem(member: tarfile.TarInfo) -> Optional[str]:
    name = member.name
    if not member.isreg():
        return f"{name!r} is not a regular file (links, devices and directories are refused)"
    if '\\' in name or ':' in name or name.startswith('/'):
        return f"{name!r} is not a safe relative path"
    parts = PurePosixPath(name).parts
    if any(p in ('', '.', '..') for p in parts) or len(parts) not in (1, 2):
        return f"{name!r} is not a safe relative path"
    if len(parts) == 1 and name != EXPORT_FILE:
        return f"unexpected entry {name!r}"
    if len(parts) == 2 and not (_BUNDLE_ID.fullmatch(parts[0]) and _FILE_NAME.fullmatch(parts[1])):
        return f"unexpected entry {name!r}"
    if member.size > MAX_ENTRY_BYTES:
        return f"{name!r} is larger than {MAX_ENTRY_BYTES} bytes"
    return None


def read_archive(archive: Path | str) -> tuple[dict, Dict[str, bytes]]:
    """(export index, {entry name: bytes}) after every structural and hash
    check. Raises BundleError; nothing is written."""
    try:
        tar = tarfile.open(archive, 'r:*')
    except (OSError, tarfile.TarError, EOFError) as exc:
        raise BundleError([f"archive {archive} can't be read ({exc})"]) from exc
    contents: Dict[str, bytes] = {}
    problems = []
    try:
        members = tar.getmembers()
        total = 0
        for m in members:
            bad = _entry_problem(m)
            if bad:
                problems.append(bad)
                continue
            if m.name in contents:
                problems.append(f"duplicate entry {m.name!r}")
                continue
            total += m.size
            if total > MAX_TOTAL_BYTES:
                problems.append(f"archive contents exceed {MAX_TOTAL_BYTES} bytes")
                break
            f = tar.extractfile(m)
            data = f.read() if f is not None else b''
            if len(data) != m.size:
                problems.append(f"{m.name!r} is truncated")
                continue
            contents[m.name] = data
    except (OSError, tarfile.TarError, EOFError) as exc:
        problems.append(f"archive is truncated or corrupt ({exc})")
    finally:
        tar.close()
    if problems:
        raise BundleError(problems)
    if EXPORT_FILE not in contents:
        raise BundleError([f"archive has no {EXPORT_FILE}"])
    try:
        index = json.loads(contents.pop(EXPORT_FILE).decode('utf-8'))
    except ValueError as exc:
        raise BundleError([f"{EXPORT_FILE} is not valid JSON ({exc})"]) from exc
    if not isinstance(index, dict) or index.get('kind') != EXPORT_KIND \
            or index.get('export_version') != EXPORT_VERSION:
        raise BundleError([f"{EXPORT_FILE} is not a version-{EXPORT_VERSION} bundle export"])
    bundle_id = index.get('bundle_id')
    if not isinstance(bundle_id, str) or not _BUNDLE_ID.fullmatch(bundle_id):
        raise BundleError(["export bundle_id is malformed"])
    files = index.get('files')
    if not isinstance(files, dict) or not files:
        raise BundleError(["export lists no files"])
    listed, present = set(files), set(contents)
    if listed != present:
        raise BundleError([f"archive entries differ from the export index (missing "
                           f"{sorted(listed - present)}, unlisted {sorted(present - listed)})"])
    for name, entry in files.items():
        if not name.startswith(bundle_id + '/'):
            problems.append(f"{name!r} is outside bundle {bundle_id}")
        elif not isinstance(entry, dict) or entry.get('sha256') != _sha256(contents[name]) \
                or entry.get('bytes') != len(contents[name]):
            problems.append(f"{name!r} doesn't match its export hash")
    manifest_name = f"{bundle_id}/{MANIFEST_FILE}"
    if manifest_name not in contents:
        problems.append(f"archive has no {manifest_name}")
    elif index.get('manifest_sha256') != _sha256(contents[manifest_name]):
        problems.append("the bundle manifest doesn't match the export's manifest hash")
    if problems:
        raise BundleError(problems)
    return index, contents


def import_bundle(models_dir: Path | str, archive: Path | str, activate: bool = True,
                  prop_lines: dict = PROP_LINES) -> dict:
    """Install a bundle from an export archive, and make it current unless
    ``activate`` is False. Returns {'bundle_id', 'installed': bool (False if
    an identical copy was already installed), 'activated': bool}."""
    models_dir = Path(models_dir)
    index, contents = read_archive(archive)          # all checks before any write
    bundle_id = index['bundle_id']
    bundles = models_dir / BUNDLES_DIR
    final = bundles / bundle_id
    with publisher_lock(models_dir, f"import-{bundle_id}"):
        installed = False
        if final.exists():
            existing = {f"{bundle_id}/{p.name}": _sha256(p.read_bytes())
                        for p in final.iterdir() if p.is_file()}
            wanted = {k: v['sha256'] for k, v in index['files'].items()}
            if existing != wanted:
                raise BundleError([f"bundle {bundle_id} is already installed with different "
                                   f"content; refusing to overwrite it"])
        else:
            bundles.mkdir(parents=True, exist_ok=True)
            staging = bundles / f".{bundle_id}.importing-{secrets.token_hex(4)}"
            try:
                staging.mkdir()
                for name, data in contents.items():
                    (staging / PurePosixPath(name).name).write_bytes(data)
                verify_bundle_dir(staging, prop_lines, index['manifest_sha256'], bundle_id)
                os.replace(staging, final)
                installed = True
            finally:
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)
        if activate:
            switch_pointer(models_dir, bundle_id, prop_lines)
    return {'bundle_id': bundle_id, 'installed': installed, 'activated': activate}


# ---------------------------------------------------------------------------
# Inventory (read-only)
# ---------------------------------------------------------------------------

def inventory(models_dir: Path | str, prop_lines: dict = PROP_LINES) -> dict:
    """Installed bundles, the current one, whether each verifies, and any
    leftover staging directories. Deletes nothing."""
    models_dir = Path(models_dir)
    current = None
    try:
        current = json.loads((models_dir / POINTER_FILE).read_text(encoding='utf-8')).get('bundle_id')
    except (OSError, ValueError, AttributeError):
        pass
    bundles, leftovers = [], []
    root = models_dir / BUNDLES_DIR
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        if not _BUNDLE_ID.fullmatch(path.name):
            leftovers.append(path.name)
            continue
        try:
            verify_bundle_dir(path, prop_lines, _sha256((path / MANIFEST_FILE).read_bytes()), path.name)
            problems = []
        except (BundleError, OSError) as exc:
            problems = getattr(exc, 'problems', [str(exc)])
        bundles.append({'bundle_id': path.name, 'current': path.name == current,
                        'verified': not problems, 'problems': problems})
    return {'current': current, 'bundles': bundles, 'leftovers': leftovers}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Export, import or list player-prop model bundles")
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('export')
    e.add_argument('--models-dir', type=Path, required=True)
    e.add_argument('--out', type=Path, required=True)
    e.add_argument('--bundle-id')
    i = sub.add_parser('import')
    i.add_argument('--models-dir', type=Path, required=True)
    i.add_argument('--archive', type=Path, required=True)
    i.add_argument('--no-activate', action='store_true')
    v = sub.add_parser('inventory')
    v.add_argument('--models-dir', type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == 'export':
            index = export_bundle(args.models_dir, args.out, args.bundle_id)
            print(f"Exported bundle {index['bundle_id']} ({len(index['files'])} files) to {args.out}")
        elif args.cmd == 'import':
            r = import_bundle(args.models_dir, args.archive, activate=not args.no_activate)
            print(f"Bundle {r['bundle_id']}: {'installed' if r['installed'] else 'already installed'}"
                  f"{', now current' if r['activated'] else ''}")
        else:
            print(json.dumps(inventory(args.models_dir), indent=2))
    except BundleError as exc:
        print("Refused:")
        for p in exc.problems:
            print(f"  - {p}")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
