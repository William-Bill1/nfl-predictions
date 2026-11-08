"""Read-only page inputs: legacy CSVs, or an explicitly pinned managed release.

No configuration is created here. A present but invalid configuration never falls
back to legacy data. Managed inputs are captured, checked and parsed from those
same bytes; no training, publication, pointer switch or live fetch occurs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import tarfile
import warnings

import numpy as np
import pandas as pd

from player_props.reference_join import build_reference, load_reference

CONFIG_FILE = 'player_props_release.json'
LATEST_FILE = 'player_props_predictions.csv'
STATS = {'passing': 'player_passing_stats.csv', 'rushing': 'player_rushing_stats.csv',
         'receiving': 'player_receiving_stats.csv'}
REQUIRED = {'player_name', 'team', 'opponent', 'prop_type', 'line_value', 'prob_over',
            'recommendation', 'confidence', 'avg_L3', 'avg_L5', 'avg_L10', 'trend'}


class PageDataError(ValueError):
    """A readable diagnostic safe to display instead of partial prop outputs."""


@dataclass
class PageInputs:
    predictions: pd.DataFrame | None
    reference: object = None
    managed: bool = False
    notice: str = 'Legacy stored outputs: no managed release is configured; release integrity is not verified.'
    payloads: dict[str, bytes] = field(default_factory=dict)


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def _read(path: Path) -> bytes:
    if 'bet_journal' in {p.lower() for p in path.resolve().parts}:
        raise PageDataError('the Bet Journal is not a player-prop input')
    return path.read_bytes()


def _json(data: bytes, label: str) -> dict:
    try:
        obj = json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise PageDataError(f'{label} is not readable JSON: {exc}') from exc
    if not isinstance(obj, dict):
        raise PageDataError(f'{label} must be an object')
    return obj


def _frame(raw: bytes, exact: bool = False) -> pd.DataFrame:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', pd.errors.ParserWarning)
            df = pd.read_csv(BytesIO(raw), dtype={'player_id': str},
                             float_precision='round_trip' if exact else None)
    except (ValueError, UnicodeError, pd.errors.ParserError, pd.errors.ParserWarning) as exc:
        raise PageDataError(f'the predictions CSV could not be read ({type(exc).__name__}: {exc})') from exc
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise PageDataError(f'the predictions CSV lacks required columns: {missing}')
    for col in ('line_value', 'prob_over', 'confidence', 'prob_under', 'raw_model_prob_over'):
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors='coerce')
        if not np.isfinite(values).all():
            raise PageDataError(f'the predictions CSV has a missing or invalid {col} value')
        if col != 'line_value' and not values.between(0, 1).all():
            raise PageDataError(f'the predictions CSV has a {col} value outside [0, 1]')
    for col in ('avg_L3', 'avg_L5', 'avg_L10'):
        values = pd.to_numeric(df[col], errors='coerce')
        if (df[col].notna() & values.isna()).any() or np.isinf(values).any():
            raise PageDataError(f'the predictions CSV has an invalid {col} value')
    if not df['recommendation'].isin(['OVER', 'UNDER']).all():
        raise PageDataError('the predictions CSV has an invalid recommendation')
    return df


@lru_cache(maxsize=4)
def _verified_package(package: str, fingerprint: tuple, policy: str) -> None:
    # Only the expensive S1 derivation/model checks are cached. read_package
    # re-hashes ALL assets on every call before this cache can be consulted.
    from player_props import release_manifest as rm
    report = rm.verify_package(Path(package))
    if report.get('manifest_sha256') != fingerprint[0][0]:
        raise PageDataError('the S1 verification result names different package bytes')
    if not report['ok']:
        raise PageDataError('the managed package failed S1 verification: ' + '; '.join(report['problems'][:5]))
    _, manifest_bytes, captured = rm.read_package(Path(package))
    actual = ((_sha(manifest_bytes),), tuple(sorted((k, _sha(v)) for k, v in captured.items())))
    if actual != fingerprint:
        raise PageDataError('the managed package changed during S1 verification')


def _managed(data: Path, config_bytes: bytes) -> PageInputs:
    from player_props import release_manifest as rm, publication as pub, model_bundle as mb
    from player_props.features import FEATURE_CONTRACT
    from player_props.rehearsal_expected import RESTATED_POLICY

    cfg = _json(config_bytes, CONFIG_FILE)
    keys = {'version', 'package', 'release_id', 'manifest_sha256', 'publication_sha256', 'season', 'week'}
    if set(cfg) != keys or type(cfg.get('version')) is not int or cfg['version'] != 1:
        raise PageDataError(f'{CONFIG_FILE} must have exactly {sorted(keys)} and version 1')
    if (not isinstance(cfg['package'], str) or not cfg['package'].strip()
            or not isinstance(cfg['release_id'], str) or not re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{8}', cfg['release_id'])
            or not isinstance(cfg['manifest_sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', cfg['manifest_sha256'])
            or not isinstance(cfg['publication_sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', cfg['publication_sha256'])
            or type(cfg['season']) is not int or not 2000 <= cfg['season'] <= 2100
            or type(cfg['week']) is not int or not 1 <= cfg['week'] <= 22):
        raise PageDataError(f'{CONFIG_FILE} has invalid package, identity or target fields')
    package = Path(cfg['package'])
    if not package.is_absolute():
        package = data.parent / package
    package = package.resolve()
    if 'bet_journal' in {p.lower() for p in package.parts}:
        raise PageDataError('the Bet Journal is not a release package')
    manifest, mbytes, contents = rm.read_package(package)
    if _sha(mbytes) != cfg['manifest_sha256'] or manifest['release_id'] != cfg['release_id']:
        raise PageDataError('the managed package does not match the configured release identity')
    target = {'season': cfg['season'], 'week': cfg['week']}
    if manifest['derived']['inference_target'] != target:
        raise PageDataError('the managed package target differs from the configured target')
    fingerprint = ((_sha(mbytes),), tuple(sorted((k, _sha(v)) for k, v in contents.items())))
    _verified_package(str(package), fingerprint, json.dumps([FEATURE_CONTRACT, RESTATED_POLICY], sort_keys=True))

    # Capture the exact files used by the UI, including all stats and reference.
    names = [LATEST_FILE, pub.snapshot_name(**target), pub.sidecar_name(**target),
             'players.csv', *STATS.values(), f'nfl_schedule_{cfg["season"]}.csv']
    captured = {name: _read(data / name) for name in names}
    if _sha(captured[pub.sidecar_name(**target)]) != cfg['publication_sha256']:
        raise PageDataError('the publication sidecar does not match the configured publication hash')
    side = _json(captured[pub.sidecar_name(**target)], 'publication sidecar')
    problems = pub.sidecar_problems(side, cfg['season'], cfg['week'], current=True)
    if problems:
        raise PageDataError('the managed publication is invalid: ' + '; '.join(problems[:5]))
    bundle = manifest['derived']['bundle']
    if (side['release'] != {'release_id': cfg['release_id'], 'manifest_sha256': cfg['manifest_sha256']}
            or side.get('bundle') != {k: bundle[k] for k in ('bundle_id', 'manifest_sha256', 'feature_contract')}
            or bundle['feature_contract'] != FEATURE_CONTRACT):
        raise PageDataError('the publication release, bundle or feature contract does not match the package')
    for role, filename in [('latest', LATEST_FILE), ('snapshot', pub.snapshot_name(**target))]:
        entry = side['outputs'][role]
        if len(captured[filename]) != entry['bytes'] or _sha(captured[filename]) != entry['sha256']:
            raise PageDataError(f'the managed {role} does not match its publication sidecar')
    if captured[LATEST_FILE] != captured[pub.snapshot_name(**target)]:
        raise PageDataError('the managed latest feed differs from the frozen snapshot')
    if captured['players.csv'] != contents['reference_runtime']:
        raise PageDataError('the runtime players reference differs from the managed package')
    for kind, filename in STATS.items():
        if captured[filename] != contents[f'inference_stats_{kind}']:
            raise PageDataError(f'{filename} differs from the managed package')
    schedule = f'nfl_schedule_{cfg["season"]}.csv'
    summary = _json(contents['rehearsal_summary'], 'rehearsal summary')
    evidence = summary['artifacts'][f'runtime/data/{schedule}']
    expected_schedule = evidence['sha256'] if isinstance(evidence, dict) else evidence
    if _sha(captured[schedule]) != expected_schedule:
        raise PageDataError('the prediction schedule differs from the package evidence')
    derived = manifest['derived']
    wanted_reference = {
        'runtime_sha256': derived['identities']['reference_runtime']['sha256'],
        'candidate_sha256': derived['identities']['reference_candidate']['sha256'],
        'derivation': manifest['reference']['derivation'],
    }
    wanted_inputs = {
        'inference_source_sha256': derived['identities']['inference_source']['sha256'],
        'inference_stats_manifest_sha256': derived['identities']['inference_stats']['manifest_sha256'],
        'prediction_schedule': {'file': schedule, 'sha256': expected_schedule},
    }
    wanted_eligibility = {k: derived['eligibility'][k] for k in
                          ('policy', 'restated_policy', 'field', 'players_reference_sha256')}
    if (side.get('reference') != wanted_reference or side.get('inputs') != wanted_inputs
            or side.get('eligibility') != wanted_eligibility
            or wanted_eligibility['policy'] != RESTATED_POLICY
            or wanted_eligibility['restated_policy'] != RESTATED_POLICY):
        raise PageDataError('the sidecar input, reference or eligibility identities differ from the package')

    # S1 verified the stored bundle. The installed bundle must be byte-for-byte
    # that bundle, including its current pointer, rather than a cross-model fallback.
    models = data.parent / 'player_props' / 'models'
    pointer_bytes = _read(models / mb.POINTER_FILE)
    pointer = _json(pointer_bytes, 'current bundle pointer')
    if pointer != {'kind': mb.POINTER_KIND, 'bundle_id': bundle['bundle_id'],
                   'manifest_sha256': bundle['manifest_sha256']}:
        raise PageDataError('the installed bundle pointer differs from the managed package')
    model_reads = {models / mb.POINTER_FILE: pointer_bytes}
    active_dir = models / mb.BUNDLES_DIR / bundle['bundle_id']
    expected_model_names = set()
    with tarfile.open(fileobj=BytesIO(contents['bundle_archive']), mode='r:*') as archive:
        for member in archive.getmembers():
            if member.name == 'export.json':
                continue
            parts = Path(member.name).parts
            if not member.isfile() or len(parts) != 2 or parts[0] != bundle['bundle_id']:
                raise PageDataError('the packaged bundle has an invalid entry')
            expected_model_names.add(parts[1])
            expected = archive.extractfile(member).read()
            installed = models / mb.BUNDLES_DIR / parts[0] / parts[1]
            actual = _read(installed)
            if actual != expected:
                raise PageDataError(f'the installed bundle file {parts[1]} differs from the managed package')
            model_reads[installed] = actual
    if {p.name for p in active_dir.iterdir()} != expected_model_names:
        raise PageDataError('the active model bundle contains missing or unlisted files')
    df = _frame(captured[LATEST_FILE], exact=True)
    v = side['verification']
    if (type(v.get('rows')) is not int or type(v.get('expected_rows')) is not int
            or v['rows'] != len(df) or v['expected_rows'] != len(df) or v.get('completeness') != 'complete'):
        raise PageDataError('the sidecar does not record complete verification of these rows')
    if (not {'player_id', 'model_bundle', 'season', 'week'} <= set(df.columns)
            or df.empty or df['player_id'].isna().any()
            or not df['model_bundle'].eq(bundle['bundle_id']).all()
            or not df['season'].eq(cfg['season']).all() or not df['week'].eq(cfg['week']).all()):
        raise PageDataError('the managed predictions have missing identities or a different bundle/target')
    ref = build_reference(pd.read_csv(BytesIO(captured['players.csv']), dtype=str, keep_default_na=False))
    if ref is None or not ref.usable or ref.problems or not set(df['player_id']) <= set(ref.by_id.index):
        raise PageDataError('the managed predictions do not have unique known player IDs')
    # Checkpoints, not a lock. The UI only receives the captured bytes even if a
    # non-cooperating writer changes the on-disk set after this final checkpoint.
    for path, expected in {**{data / k: v for k, v in captured.items()},
                           **model_reads, data / CONFIG_FILE: config_bytes}.items():
        if _read(path) != expected:
            raise PageDataError(f'{path.name} changed while the managed release was being checked')
    _, after_manifest, after = rm.read_package(package)
    if after_manifest != mbytes or any(after[k] != v for k, v in contents.items()):
        raise PageDataError('the managed package changed while the page inputs were being checked')
    return PageInputs(df, ref, True,
                      f'Managed release {cfg["release_id"]}: file integrity and release consistency checked. '
                      'This is not release approval or validated betting confidence.', captured)


def load_page_inputs(data_dir: Path | str = 'data_files') -> PageInputs:
    """No configuration means legacy; invalid managed configuration fails closed."""
    data = Path(data_dir)
    try:
        config = data / CONFIG_FILE
        # A broken symlink still counts as configured: never silently downgrade.
        try:
            config_bytes = _read(config)
        except FileNotFoundError:
            if os.path.lexists(config):
                raise PageDataError('the managed release configuration is a broken link')
        else:
            return _managed(data, config_bytes)
        latest = data / LATEST_FILE
        try:
            latest_bytes = _read(latest)
        except FileNotFoundError:
            df = None
        else:
            df = _frame(latest_bytes)
        return PageInputs(df, load_reference(data / 'players.csv'))
    except PageDataError:
        raise
    except Exception as exc:
        raise PageDataError(f'player-prop inputs could not be checked ({type(exc).__name__}: {str(exc)[:400]})') from exc
