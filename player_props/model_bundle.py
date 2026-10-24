"""
Player-prop model bundles (Stage 4): one scorer, versioned provenance, and
verified loading.

* ``score_probabilities`` is the only way a prop model is scored, in training
  evaluation and in prediction alike: exact feature order, class 1 = the stat
  goes OVER the model's line, and every probability validated.
* A bundle is an immutable directory ``<models_dir>/bundles/<bundle_id>/``
  holding the XGBoost model files, ``model_metrics.json`` and
  ``manifest.json``. ``<models_dir>/current_bundle.json`` names the bundle in
  use and the SHA-256 of its manifest.
* ``publish_bundle`` writes a new bundle beside the old ones, verifies it from
  disk, and only then switches the pointer with one ``os.replace``. Until that
  replace the previous bundle stays in use; a failure removes the unfinished
  directory or reports where it was left.
* ``load_bundle`` re-verifies everything: pointer, manifest structure and
  version, scorer, feature contract, every file's hash, the metrics, and each
  model's own identity. Any problem raises BundleError and nothing is used.

Model files outside a bundle (everything committed before Stage 4) have no
manifest and are never loaded.

Policy: docs/architecture.md ("Stage 4 model provenance").
"""
import hashlib
import json
import math
import os
import platform
import re
import secrets
import shutil
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from player_props.features import FEATURE_CONTRACT, PROP_STAT, model_features
except ImportError:  # run as a script from inside player_props/
    from features import FEATURE_CONTRACT, PROP_STAT, model_features

ROOT = Path(__file__).parent.parent
MANIFEST_KIND = 'player_props_model_bundle'
MANIFEST_VERSION = 1
POINTER_KIND = 'player_props_current_bundle'
POINTER_FILE = 'current_bundle.json'
BUNDLES_DIR = 'bundles'
MANIFEST_FILE = 'manifest.json'
METRICS_FILE = 'model_metrics.json'
# The one supported scorer: XGBoost's binary classifier, probability of class 1.
SCORER = {'name': 'xgboost.XGBClassifier.predict_proba[:, 1]', 'version': 1,
          'objective': 'binary:logistic', 'positive_class': 'stat > line (OVER)'}
METRIC_KEYS = ('accuracy', 'roc_auc', 'base_rate', 'precision', 'recall', 'f1', 'brier', 'log_loss')
CODE_FILES = ('player_props/models.py', 'player_props/features.py', 'player_props/aggregators.py',
              'player_props/model_bundle.py')
CLOCK_SKEW = timedelta(minutes=5)
_BUNDLE_ID = re.compile(r'\d{8}T\d{6}Z-[0-9a-f]{8}')
_HEX64 = re.compile(r'[0-9a-f]{64}')
_MODEL_NAME = re.compile(r'[a-z_]+')


class ScoringError(ValueError):
    """A model or its input can't be scored under the supported scorer."""


class BundleError(ValueError):
    """A bundle can't be published or used. ``problems`` lists every reason."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


# ---------------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------------

def model_problems(model, prop_type: str) -> List[str]:
    """Why ``model`` isn't a supported scorer for ``prop_type`` (empty = ok)."""
    problems = []
    try:
        booster = model.get_booster()
        config = json.loads(booster.save_config())
    except Exception as exc:  # noqa: BLE001 - anything unreadable is unsupported
        return [f"not a readable XGBoost model ({type(exc).__name__}: {exc})"]
    objective = config.get('learner', {}).get('objective', {}).get('name')
    if objective != SCORER['objective']:
        problems.append(f"objective {objective!r} is not {SCORER['objective']!r}")
    if booster.feature_names != model_features(prop_type):
        problems.append(f"model feature names {booster.feature_names} != {model_features(prop_type)}")
    classes = getattr(model, 'classes_', None)
    if classes is not None and [int(c) for c in classes] != [0, 1]:
        problems.append(f"classes {list(classes)} are not [0, 1]")
    return problems


def score_probabilities(model, X: pd.DataFrame, prop_type: str) -> np.ndarray:
    """P(OVER) for each row of ``X``: the probability of class 1, where the
    training target is 1 when the stat is above the model's line.

    ``X`` must have exactly ``model_features(prop_type)`` as columns, in order
    (build it with features.model_matrix). The model must be a binary:logistic
    XGBoost classifier trained on those names. Every output must be a finite
    probability in [0, 1] with the two classes summing to 1; anything else
    raises ScoringError - no default, clip or other model is substituted."""
    names = model_features(prop_type)
    if not isinstance(X, pd.DataFrame) or list(X.columns) != names:
        got = list(X.columns) if isinstance(X, pd.DataFrame) else type(X).__name__
        raise ScoringError(f"{prop_type}: input columns {got} != {names}")
    problems = model_problems(model, prop_type)
    if problems:
        raise ScoringError(f"{prop_type}: " + "; ".join(problems))
    if len(X) == 0:
        return np.empty(0)
    proba = np.asarray(model.predict_proba(X), dtype=float)
    if proba.shape != (len(X), 2):
        raise ScoringError(f"{prop_type}: predict_proba returned shape {proba.shape}, "
                           f"expected {(len(X), 2)}")
    if not np.isfinite(proba).all() or (proba < 0).any() or (proba > 1).any():
        raise ScoringError(f"{prop_type}: predict_proba returned values outside [0, 1]")
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
        raise ScoringError(f"{prop_type}: class probabilities don't sum to 1")
    return proba[:, 1]


# ---------------------------------------------------------------------------
# Manifest pieces
# ---------------------------------------------------------------------------

def library_versions() -> dict:
    import sklearn
    import xgboost
    return {'python': platform.python_version(), 'xgboost': xgboost.__version__,
            'numpy': np.__version__, 'pandas': pd.__version__, 'scikit-learn': sklearn.__version__}


def code_revision(root: Path = ROOT) -> dict:
    """Hashes of the code that trains and scores, plus the git commit and
    whether player_props/ had uncommitted changes."""
    rev = {'files': {f: _sha256((root / f).read_bytes()) for f in CODE_FILES if (root / f).exists()},
           'git_commit': None, 'git_dirty': None}
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True,
                                text=True, timeout=10)
        status = subprocess.run(['git', 'status', '--porcelain', '--', 'player_props'], cwd=root,
                                capture_output=True, text=True, timeout=10)
        if commit.returncode == 0 and status.returncode == 0:
            rev['git_commit'] = commit.stdout.strip()
            rev['git_dirty'] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return rev


def window(df: pd.DataFrame, target: Optional[pd.Series] = None) -> dict:
    """Seasons, rows, positives (target = 1) and dates a set of training or
    evaluation rows covers."""
    dates = pd.to_datetime(df['game_date'], format='%Y-%m-%d', errors='coerce')
    return {'seasons': sorted(int(s) for s in df['season'].unique()), 'rows': int(len(df)),
            'positives': None if target is None else int(target.sum()),
            'first_date': dates.min().strftime('%Y-%m-%d') if dates.notna().any() else None,
            'last_date': dates.max().strftime('%Y-%m-%d') if dates.notna().any() else None}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_hash(v) -> bool:
    return isinstance(v, str) and _HEX64.fullmatch(v) is not None


def _timestamp(v) -> Optional[datetime]:
    if not isinstance(v, str):
        return None
    try:
        ts = datetime.fromisoformat(v)
    except ValueError:
        return None
    return ts if ts.tzinfo is not None else None


def _is_date(v) -> bool:
    if not isinstance(v, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', v):
        return False
    try:
        date.fromisoformat(v)
        return True
    except ValueError:
        return False


def _window_problems(w, where: str) -> List[str]:
    if not isinstance(w, dict):
        return [f"{where} must be an object"]
    problems = []
    if not (isinstance(w.get('seasons'), list) and w['seasons'] and all(_is_int(s) for s in w['seasons'])):
        problems.append(f"{where}.seasons must be a non-empty list of seasons")
    if not _is_int(w.get('rows')) or w['rows'] <= 0:
        problems.append(f"{where}.rows must be a positive integer")
    elif not _is_int(w.get('positives')) or not 0 < w['positives'] < w['rows']:
        problems.append(f"{where}.positives must count both classes' split (0 < positives < rows)")
    for k in ('first_date', 'last_date'):
        if not _is_date(w.get(k)):
            problems.append(f"{where}.{k} must be a YYYY-MM-DD date")
    return problems


def model_entry_problems(name: str, entry, prop_lines: dict) -> List[str]:
    """Structural and identity problems in one manifest model entry."""
    if not isinstance(entry, dict):
        return [f"models.{name} must be an object"]
    problems = []
    prop, line = entry.get('prop_type'), entry.get('line_type')
    if prop not in PROP_STAT:
        return [f"models.{name}: unknown prop_type {prop!r}"]
    if line not in prop_lines.get(prop, {}):
        problems.append(f"models.{name}: unknown line_type {line!r} for {prop}")
    elif entry.get('line_value') != prop_lines[prop][line]:
        problems.append(f"models.{name}: line_value {entry.get('line_value')!r} != "
                        f"{prop_lines[prop][line]} for {prop}/{line}")
    if name != f"{prop}_{line}":
        problems.append(f"models.{name}: name doesn't match its prop and line ({prop}_{line})")
    if entry.get('file') != f"{name}.json":
        problems.append(f"models.{name}: file must be {name}.json")
    if not _is_hash(entry.get('sha256')) or not _is_int(entry.get('bytes')):
        problems.append(f"models.{name}: sha256 and bytes are required")
    if entry.get('features') != model_features(prop):
        problems.append(f"models.{name}: features differ from the contract for {prop}")
    if entry.get('scorer') != SCORER:
        problems.append(f"models.{name}: unsupported scorer {entry.get('scorer')!r}")
    if entry.get('target') != {'stat': PROP_STAT[prop][1], 'rule': 'stat > line'}:
        problems.append(f"models.{name}: target must be {PROP_STAT[prop][1]} > line")
    problems += _window_problems(entry.get('training'), f"models.{name}.training")
    evaluation = entry.get('evaluation')
    problems += _window_problems(evaluation, f"models.{name}.evaluation")
    if isinstance(evaluation, dict) and not isinstance(evaluation.get('protocol'), str):
        problems.append(f"models.{name}.evaluation.protocol must be described")
    metrics = entry.get('metrics')
    if not isinstance(metrics, dict):
        problems.append(f"models.{name}.metrics must be an object")
    else:
        bad = [k for k in METRIC_KEYS
               if not (isinstance(metrics.get(k), (int, float)) and not isinstance(metrics.get(k), bool)
                       and math.isfinite(metrics[k]))]
        if bad:
            problems.append(f"models.{name}.metrics missing or non-finite: {bad}")
        if not isinstance(metrics.get('reliable'), bool):
            problems.append(f"models.{name}.metrics.reliable must be true or false")
        if metrics.get('scorer') != SCORER['name']:
            problems.append(f"models.{name}.metrics were not computed with the bundle scorer")
    if (isinstance(entry.get('training'), dict) and isinstance(evaluation, dict)
            and _is_date(entry['training'].get('last_date')) and _is_date(evaluation.get('first_date'))
            and entry['training']['last_date'] >= evaluation['first_date']):
        problems.append(f"models.{name}: training rows reach into the evaluation window")
    return problems


def manifest_problems(manifest, prop_lines: dict, now: Optional[datetime] = None) -> List[str]:
    """Every structural problem in a parsed bundle manifest (empty = ok)."""
    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]
    if manifest.get('kind') != MANIFEST_KIND or manifest.get('manifest_version') != MANIFEST_VERSION:
        return [f"not a version-{MANIFEST_VERSION} {MANIFEST_KIND} manifest"]
    now = now or datetime.now(timezone.utc)
    problems = []
    if not isinstance(manifest.get('bundle_id'), str) or not _BUNDLE_ID.fullmatch(manifest['bundle_id']):
        problems.append("bundle_id is malformed")
    built = _timestamp(manifest.get('built_at'))
    if built is None:
        problems.append("built_at must be an ISO timestamp with a time zone")
    elif built > now + CLOCK_SKEW:
        problems.append("built_at is in the future")
    if manifest.get('scorer') != SCORER:
        problems.append(f"unsupported scorer {manifest.get('scorer')!r}")
    if manifest.get('feature_contract') != FEATURE_CONTRACT:
        problems.append(f"feature contract {manifest.get('feature_contract')!r} != current {FEATURE_CONTRACT!r}")
    src = manifest.get('training_source')
    if not isinstance(src, dict):
        problems.append("training_source must be an object")
    else:
        if not _is_hash(src.get('stats_manifest_sha256')):
            problems.append("training_source.stats_manifest_sha256 is required")
        if src.get('feature_definition') != FEATURE_CONTRACT['feature_definition']:
            problems.append("training_source.feature_definition differs from the contract")
        files = src.get('stats_files')
        if not (isinstance(files, dict) and files and all(_is_hash(v) for v in files.values())):
            problems.append("training_source.stats_files must map each stats file to its SHA-256")
        if not _is_date(src.get('last_game_date')):
            problems.append("training_source.last_game_date must be a YYYY-MM-DD date")
    for key in ('code', 'libraries'):
        if not isinstance(manifest.get(key), dict):
            problems.append(f"{key} must be an object")
    libs = manifest.get('libraries')
    if isinstance(libs, dict) and not isinstance(libs.get('xgboost'), str):
        problems.append("libraries.xgboost is required")
    metrics_file = manifest.get('metrics_file')
    if not (isinstance(metrics_file, dict) and metrics_file.get('file') == METRICS_FILE
            and _is_hash(metrics_file.get('sha256'))):
        problems.append(f"metrics_file must name {METRICS_FILE} and its SHA-256")
    models = manifest.get('models')
    if not isinstance(models, dict) or not models:
        problems.append("models must be a non-empty object")
    else:
        for name, entry in models.items():
            if not isinstance(name, str) or not _MODEL_NAME.fullmatch(name):
                problems.append(f"model name {name!r} is malformed")
                continue
            found = model_entry_problems(name, entry, prop_lines)
            problems += found
            # A model can't have been evaluated on games its training source lacks.
            if (not found and isinstance(src, dict) and _is_date(src.get('last_game_date'))
                    and entry['evaluation']['last_date'] > src['last_game_date']):
                problems.append(f"models.{name}: evaluated on games after the training source ends")
    return problems


def _metrics_document(manifest: dict) -> dict:
    """The bundle's model_metrics.json content, derived from the manifest."""
    return {'bundle_id': manifest['bundle_id'], 'scorer': SCORER['name'],
            'models': [{'model_name': n, **e['metrics'],
                        'holdout_season': e['evaluation']['seasons'][-1],
                        'train_samples': e['training']['rows'], 'test_samples': e['evaluation']['rows']}
                       for n, e in sorted(manifest['models'].items())]}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_json(path: Path, problems: List[str], what: str):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        problems.append(f"{what} is missing ({path.name})")
    except (OSError, ValueError) as exc:
        problems.append(f"{what} could not be read: {exc}")
    return None


def verify_bundle_dir(bundle_dir: Path, prop_lines: dict, expected_manifest_sha: Optional[str] = None,
                      expected_id: Optional[str] = None) -> dict:
    """Verify one bundle directory completely; return {name: loaded model info}.
    Raises BundleError listing every problem. Used before publication and at load."""
    import xgboost as xgb
    problems: List[str] = []
    manifest_path = bundle_dir / MANIFEST_FILE
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise BundleError([f"bundle manifest can't be read ({exc})"]) from exc
    if expected_manifest_sha is not None and _sha256(manifest_bytes) != expected_manifest_sha:
        raise BundleError(["bundle manifest doesn't match the hash recorded in the pointer"])
    try:
        manifest = json.loads(manifest_bytes.decode('utf-8'))
    except ValueError as exc:
        raise BundleError([f"bundle manifest is not valid JSON ({exc})"]) from exc
    problems += manifest_problems(manifest, prop_lines)
    if problems:
        raise BundleError(problems)
    if expected_id is not None and manifest['bundle_id'] != expected_id:
        raise BundleError([f"manifest bundle_id {manifest['bundle_id']} != pointer {expected_id}"])
    current_xgb = library_versions()['xgboost']
    if manifest['libraries']['xgboost'] != current_xgb:
        problems.append(f"bundle was scored with xgboost {manifest['libraries']['xgboost']}, "
                        f"this environment has {current_xgb}")

    metrics_bytes = None
    try:
        metrics_bytes = (bundle_dir / METRICS_FILE).read_bytes()
    except OSError as exc:
        problems.append(f"{METRICS_FILE} can't be read ({exc})")
    if metrics_bytes is not None:
        if _sha256(metrics_bytes) != manifest['metrics_file']['sha256']:
            problems.append(f"{METRICS_FILE} doesn't match its manifest hash")
        else:
            try:
                if json.loads(metrics_bytes.decode('utf-8')) != _metrics_document(manifest):
                    problems.append(f"{METRICS_FILE} doesn't match the manifest's metrics")
            except ValueError:
                problems.append(f"{METRICS_FILE} is not valid JSON")

    loaded = {}
    for name, entry in sorted(manifest['models'].items()):
        path = bundle_dir / entry['file']
        try:
            data = path.read_bytes()
        except OSError as exc:
            problems.append(f"{entry['file']} can't be read ({exc})")
            continue
        if _sha256(data) != entry['sha256'] or len(data) != entry['bytes']:
            problems.append(f"{entry['file']} doesn't match its manifest hash")
            continue
        model = xgb.XGBClassifier()
        try:
            model.load_model(bytearray(data))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{entry['file']} isn't a loadable XGBoost model ({exc})")
            continue
        found = model_problems(model, entry['prop_type'])
        if found:
            problems += [f"{name}: {p}" for p in found]
            continue
        stat_type, stat_col = PROP_STAT[entry['prop_type']]
        loaded[name] = {'model': model, 'prop_type': entry['prop_type'], 'stat_type': stat_type,
                        'stat_col': stat_col, 'line_type': entry['line_type'],
                        'line_value': entry['line_value'], 'reliable': entry['metrics']['reliable'],
                        'metrics': entry['metrics']}
    if problems:
        raise BundleError(problems)
    return {'manifest': manifest, 'models': loaded}


def load_bundle(models_dir: Path, prop_lines: dict) -> dict:
    """The current, fully verified bundle: {'manifest': ..., 'models': {name: info}}.
    Raises BundleError when there is no pointer, or anything fails to verify."""
    models_dir = Path(models_dir)
    problems: List[str] = []
    pointer = _read_json(models_dir / POINTER_FILE, problems,
                         "the model bundle pointer (no verified model bundle has been published)")
    if pointer is None:
        raise BundleError(problems + ["model files without a bundle manifest (including every "
                                      "model committed before Stage 4) are unverified and not used"])
    if not (isinstance(pointer, dict) and pointer.get('kind') == POINTER_KIND
            and isinstance(pointer.get('bundle_id'), str) and _BUNDLE_ID.fullmatch(pointer['bundle_id'])
            and _is_hash(pointer.get('manifest_sha256'))):
        raise BundleError([f"{POINTER_FILE} is malformed"])
    bundle_dir = models_dir / BUNDLES_DIR / pointer['bundle_id']
    if not bundle_dir.is_dir():
        raise BundleError([f"bundle {pointer['bundle_id']} named by {POINTER_FILE} doesn't exist"])
    return verify_bundle_dir(bundle_dir, prop_lines, pointer['manifest_sha256'], pointer['bundle_id'])


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------

def new_bundle_id(now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"{now:%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"


def build_manifest(bundle_id: str, models: Dict[str, dict], training_source: dict) -> dict:
    """The manifest for already-serialised models.

    ``models``: name -> {'bytes': serialised model, plus every manifest entry
    field except file/sha256/bytes}. ``training_source`` describes the stats
    the models were trained on (models.training_inputs)."""
    entries = {}
    for name, m in models.items():
        entry = {k: v for k, v in m.items() if k != 'data'}
        entry.update(file=f"{name}.json", sha256=_sha256(m['data']), bytes=len(m['data']))
        entries[name] = entry
    manifest = {'kind': MANIFEST_KIND, 'manifest_version': MANIFEST_VERSION, 'bundle_id': bundle_id,
                # When the bundle was built - not the period the training data covers
                # (see each model's training window and training_source).
                'built_at': _now(), 'scorer': SCORER, 'feature_contract': dict(FEATURE_CONTRACT),
                'training_source': training_source, 'code': code_revision(),
                'libraries': library_versions(), 'models': entries}
    manifest['metrics_file'] = {'file': METRICS_FILE, 'sha256': _sha256(
        json.dumps(_metrics_document(manifest), indent=2, sort_keys=True).encode('utf-8'))}
    return manifest


LOCK_SLOT = 'model-bundle'


class publisher_lock:
    """Exclusive lock on a models directory for anything that installs a
    bundle or switches the pointer (ontario_spreads.SlotLock: owner recorded,
    never removed on age alone, removed on exit only if still ours). A
    conflict raises BundleError with the recorded owner and recovery advice."""

    def __init__(self, models_dir: Path, run_id: str):
        from ontario_spreads import SlotLock
        self._lock = SlotLock(Path(models_dir), LOCK_SLOT, run_id, what='model bundle publication')

    def __enter__(self):
        from ontario_spreads import CaptureError
        Path(self._lock.path.parent).mkdir(parents=True, exist_ok=True)
        try:
            self._lock.__enter__()
        except CaptureError as exc:
            raise BundleError([str(exc)]) from exc
        return self

    def __exit__(self, *exc):
        return self._lock.__exit__(*exc)


def switch_pointer(models_dir: Path, bundle_id: str, prop_lines: dict) -> None:
    """Make an installed bundle current. The bundle is verified from disk
    first; the pointer is replaced with one os.replace. Call under
    publisher_lock. On any failure the previous pointer is untouched."""
    models_dir = Path(models_dir)
    if not isinstance(bundle_id, str) or not _BUNDLE_ID.fullmatch(bundle_id):
        raise BundleError([f"bundle id {bundle_id!r} is malformed"])
    bundle_dir = models_dir / BUNDLES_DIR / bundle_id
    if not bundle_dir.is_dir():
        raise BundleError([f"bundle {bundle_id} is not installed"])
    manifest_sha = _sha256((bundle_dir / MANIFEST_FILE).read_bytes())
    verify_bundle_dir(bundle_dir, prop_lines, manifest_sha, bundle_id)
    pointer_tmp = models_dir / f".{POINTER_FILE}.{bundle_id}.{secrets.token_hex(4)}.tmp"
    try:
        pointer_tmp.write_text(json.dumps({'kind': POINTER_KIND, 'bundle_id': bundle_id,
                                           'manifest_sha256': manifest_sha}, indent=2),
                               encoding='utf-8')
        os.replace(pointer_tmp, models_dir / POINTER_FILE)
    finally:
        if pointer_tmp.exists():
            pointer_tmp.unlink()


def publish_bundle(models_dir: Path, models: Dict[str, dict], training_source: dict,
                   prop_lines: dict) -> Path:
    """Publish a new bundle under publisher_lock (see _publish_unlocked). An
    invalid manifest is refused before anything (even the lock) is written."""
    models_dir = Path(models_dir)
    problems = manifest_problems(build_manifest(new_bundle_id(), models, training_source), prop_lines)
    if problems:
        raise BundleError(["bundle not published: " + p for p in problems])
    with publisher_lock(models_dir, new_bundle_id()):
        return _publish_unlocked(models_dir, models, training_source, prop_lines)


def _publish_unlocked(models_dir: Path, models: Dict[str, dict], training_source: dict,
                      prop_lines: dict) -> Path:
    """Write, verify and switch to a new bundle. Returns its directory.

    1. Everything is written to ``bundles/.<id>.partial`` and verified from
       disk (verify_bundle_dir) exactly as prediction will load it.
    2. The directory is renamed to ``bundles/<id>``.
    3. ``current_bundle.json`` is replaced with one ``os.replace``.

    Until step 3 the previous pointer and bundle are untouched and stay in
    use. A failure before step 3 removes the unfinished directory; if even
    that fails, BundleError names the path left behind. Not handled: a crash
    or power loss mid-write (the pointer still names the old bundle, and a
    leftover directory is ignored) and durability of the rename (no fsync).
    Concurrent publishers are excluded by publisher_lock."""
    models_dir = Path(models_dir)
    bundle_id = new_bundle_id()
    manifest = build_manifest(bundle_id, models, training_source)
    problems = manifest_problems(manifest, prop_lines)
    if problems:
        raise BundleError(["bundle not published: " + p for p in problems])
    bundles = models_dir / BUNDLES_DIR
    bundles.mkdir(parents=True, exist_ok=True)
    partial = bundles / f".{bundle_id}.partial"
    final = bundles / bundle_id
    pointer_tmp = models_dir / f".{POINTER_FILE}.{bundle_id}.tmp"
    published = False
    created: List[Path] = []
    try:
        partial.mkdir()
        created.append(partial)
        for name, m in models.items():
            (partial / f"{name}.json").write_bytes(m['data'])
        (partial / METRICS_FILE).write_bytes(
            json.dumps(_metrics_document(manifest), indent=2, sort_keys=True).encode('utf-8'))
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode('utf-8')
        (partial / MANIFEST_FILE).write_bytes(manifest_bytes)
        verify_bundle_dir(partial, prop_lines, _sha256(manifest_bytes), bundle_id)
        os.replace(partial, final)
        created.append(final)
        pointer = {'kind': POINTER_KIND, 'bundle_id': bundle_id,
                   'manifest_sha256': _sha256(manifest_bytes)}
        created.append(pointer_tmp)
        pointer_tmp.write_text(json.dumps(pointer, indent=2), encoding='utf-8')
        os.replace(pointer_tmp, models_dir / POINTER_FILE)
        published = True
        return final
    finally:
        if not published:
            leftovers = []
            for path in created:          # only what this call created
                if path.exists():
                    try:
                        shutil.rmtree(path) if path.is_dir() else path.unlink()
                    except OSError:
                        leftovers.append(str(path))
            if leftovers:
                raise BundleError([f"bundle {bundle_id} was not published and could not be cleaned "
                                   f"up; the previous bundle is still current. Left behind: "
                                   f"{', '.join(leftovers)}"])
