"""
Isolated end-to-end Player Props rehearsal (Stage 5B preparation).

One command runs the real production functions, in order, inside a workspace:

    fixture downloads -> source refresh (training, inference) -> aggregation with
    stats manifest -> training -> verified bundle -> export -> restore into an
    empty runtime -> input validation -> prediction -> independent verification
    -> publish (snapshot-immutable)

* Every path is explicit and lives under ``<workspace>/runs/<run_id>/`` (plus
  ``<workspace>/published/``). The workspace, and every directory the run
  creates, is resolved (symlinks, junctions) and refused if it is, contains or
  lies inside the repository or a protected production location.
* Downloads are synthetic (player_props/rehearsal_fixtures.py) and the network
  is blocked for the whole run.
* Training uses a source refreshed only through ``train_seasons``; inference
  uses its own, newer source. The bundle is restored into a fresh, empty
  runtime models directory, and prediction is pointed only there.
* External inputs are deterministic: injuries, weather and market odds are
  skipped, so those adjustments are not exercised (only the model clamps can
  appear in ``adjustments``).
* A stage failure stops every later stage. The run directory keeps its
  partial outputs, ``summary.json`` (status ``failed``) and
  ``diagnostics/traceback.txt``; nothing is published.
* Publishing copies the run's predictions to ``<workspace>/published/``. A
  week's frozen snapshot is never overwritten: an identical one is accepted, a
  different one makes the run ``conflict`` and nothing is published.

No accuracy or betting edge is measured or claimed.

CLI (synthetic fixtures only; there is no live-data option):
    python player_props/rehearsal.py --workspace DIR [--seed 7]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import socket
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from player_props import rehearsal_fixtures as fixtures  # noqa: E402
from player_props.aggregators import (  # noqa: E402
    SCHEDULE_FILE, STATS_FILES, build_player_stats, load_matchup_schedule, write_player_stats,
)
from player_props.features import (  # noqa: E402
    PROP_LINES, PROP_STAT, TARGET_COLUMNS, inference_rows, model_matrix, resolve_game, training_rows,
)
from player_props.model_bundle import load_bundle, score_probabilities  # noqa: E402
from player_props.provenance import (  # noqa: E402
    MANIFEST_FILE as STATS_MANIFEST, check_inputs, load_games, read_source, stats_manifest,
)

SUMMARY_KIND = 'player_props_rehearsal'
SUMMARY_VERSION = 1
PRODUCTION_PATHS = [ROOT / 'data_files', ROOT / 'player_props' / 'models']
STAGES = ['workspace', 'source_training', 'source_inference', 'aggregate_training', 'train',
          'export', 'restore', 'aggregate_inference', 'validate_inputs', 'predict',
          'verify_outputs', 'publish']
LIMITATIONS = [
    "synthetic league and downloads only; no real nflverse data, runner, artifact or cache storage",
    "injury, weather and market-odds inputs skipped: those adjustments are not exercised",
    "output verification checks every row present, not which players/props should have been included",
    "at most one clamp applies under these rules, so adjustment-label order is only checked as an exact string",
    "Stage 2 completion checks at prediction use the real current date",
    "no accuracy, calibration or betting edge is measured or claimed",
]


class RehearsalError(RuntimeError):
    """The rehearsal can't run as configured (paths, configuration, results)."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _real(path: Path) -> Path:
    """The fully resolved path (symlinks and Windows junctions), even if it
    doesn't exist yet."""
    return Path(os.path.realpath(path))


def _overlaps(a: Path, b: Path) -> bool:
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def check_workspace(workspace: Path, protected: Optional[List[Path]] = None) -> Path:
    """The resolved workspace, or RehearsalError if it is, contains or lies in
    the repository or a protected production location."""
    real = _real(workspace)
    for p in [ROOT] + list(PRODUCTION_PATHS) + list(protected or []):
        if _overlaps(real, _real(p)):
            raise RehearsalError(f"workspace {workspace} (resolves to {real}) overlaps protected "
                                 f"location {p}; use a directory outside the repository")
    return real


def _check_location(path: Path, real: Path, workspace_real: Path,
                    protected: Optional[List[Path]]) -> None:
    if not real.is_relative_to(workspace_real):
        raise RehearsalError(f"{path} resolves to {real}, outside the workspace")
    for p in [ROOT] + list(PRODUCTION_PATHS) + list(protected or []):
        if _overlaps(real, _real(p)):
            raise RehearsalError(f"{path} resolves to {real}, overlapping protected location {p}")


def safe_dir(path: Path, workspace_real: Path, protected: Optional[List[Path]] = None) -> Path:
    """Create ``path`` only if it really points inside the workspace and
    nowhere near a protected location. Checked before anything is created
    (realpath resolves links and junctions in the part that exists, including
    ones placed inside the workspace) and again afterwards."""
    _check_location(path, _real(path), workspace_real, protected)
    path.mkdir(parents=True, exist_ok=True)
    _check_location(path, _real(path), workspace_real, protected)
    return path


@contextmanager
def no_network():
    """Refuse every socket connection and name lookup for the duration."""
    def refuse(*a, **k):
        raise RehearsalError("network access attempted during the rehearsal")
    saved = socket.socket.connect, socket.create_connection, socket.getaddrinfo
    socket.socket.connect = refuse
    socket.create_connection = refuse
    socket.getaddrinfo = refuse
    try:
        yield
    finally:
        socket.socket.connect, socket.create_connection, socket.getaddrinfo = saved


# ---------------------------------------------------------------------------
# Configuration and run state
# ---------------------------------------------------------------------------

@dataclass
class RehearsalConfig:
    workspace: Path
    train_seasons: List[int] = field(default_factory=lambda: [2022, 2023, 2024, 2025])
    target_season: int = 2026
    target_week: int = 3
    seed: int = 7
    # Weeks already played in the target season (the rest are scheduled, unscored).
    played_target_weeks: int = 2
    protected: List[Path] = field(default_factory=list)
    drop_seasons: tuple = ()
    drop_games: frozenset = frozenset()

    @property
    def seasons(self) -> List[int]:
        return list(range(min(self.train_seasons), self.target_season + 1))


@dataclass
class Run:
    config: RehearsalConfig
    workspace: Path
    run_id: str
    dir: Path
    paths: Dict[str, Path] = field(default_factory=dict)
    stages: List[dict] = field(default_factory=list)
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stages (each uses the production function; returns details for the summary)
# ---------------------------------------------------------------------------

def _stage_workspace(run: Run) -> dict:
    cfg = run.config
    if cfg.train_seasons != list(range(cfg.train_seasons[0], cfg.train_seasons[-1] + 1)):
        raise RehearsalError(f"train_seasons {cfg.train_seasons} must be consecutive")
    if len(cfg.train_seasons) < 2:
        raise RehearsalError("training needs at least two seasons (train + out-of-time evaluation)")
    names = {'fixtures': 'fixtures', 'train_source': 'train/source', 'train_stats': 'train/stats',
             'train_models': 'train/models', 'infer_source': 'inference/source', 'export': 'export',
             'runtime_data': 'runtime/data', 'runtime_models': 'runtime/models',
             'diagnostics': 'diagnostics'}
    for key, rel in names.items():
        run.paths[key] = safe_dir(run.dir / rel, run.workspace, cfg.protected)
    if any(run.paths['runtime_models'].iterdir()):
        raise RehearsalError("the runtime models directory is not empty")
    played = {s: fixtures.WEEKS for s in cfg.train_seasons}
    played[cfg.target_season] = cfg.played_target_weeks
    plays, schedule = fixtures.league(cfg.seasons, played, cfg.seed)
    run.info['plays'] = plays
    sched_path = run.paths['fixtures'] / SCHEDULE_FILE
    schedule.to_csv(sched_path, sep='\t', index=False)
    fixtures.players().to_csv(run.paths['fixtures'] / 'players.csv', index=False)
    target = schedule[(schedule['season'] == cfg.target_season) & (schedule['week'] == cfg.target_week)]
    if target.empty:
        raise RehearsalError(f"no scheduled games for {cfg.target_season} week {cfg.target_week}")
    run.info['target_first_game'] = str(target['gameday'].min())
    return {'run_dir': str(run.dir), 'seasons': cfg.seasons,
            'target': {'season': cfg.target_season, 'week': cfg.target_week,
                       'first_game': run.info['target_first_game']}}


def _refresh(run: Run, key: str, seasons: List[int], as_of: date) -> dict:
    from player_props.source_refresh import refresh_source, verify_installed
    fetch = fixtures.fetcher(run.info['plays'], run.config.drop_seasons, set(run.config.drop_games))
    manifest = refresh_source(run.paths[key], run.paths['fixtures'] / SCHEDULE_FILE, fetch=fetch,
                              seasons=seasons, today=as_of)
    verify_installed(run.paths[key])
    cov = manifest['source']['coverage']
    return {'sha256': manifest['source']['sha256'], 'seasons': sorted(cov['seasons']),
            'first_date': cov['first_date'], 'last_date': cov['last_date'], 'games': cov['games']}


def _stage_source_training(run: Run) -> dict:
    cfg = run.config
    detail = _refresh(run, 'train_source', cfg.train_seasons, date(cfg.train_seasons[-1] + 1, 3, 1))
    if detail['last_date'] >= run.info['target_first_game']:
        raise RehearsalError(f"training source reaches {detail['last_date']}, not strictly before the "
                             f"target week's first game {run.info['target_first_game']}")
    return detail


def _stage_source_inference(run: Run) -> dict:
    return _refresh(run, 'infer_source', run.config.seasons,
                    date.fromisoformat(run.info['target_first_game']))


def _aggregate(run: Run, source_key: str, stats_key: str) -> dict:
    from player_props.source_refresh import SOURCE_NAME
    out = run.paths[stats_key]
    shutil.copy2(run.paths['fixtures'] / SCHEDULE_FILE, out / SCHEDULE_FILE)
    pbp, observation = read_source(run.paths[source_key] / SOURCE_NAME)
    tables, reports = build_player_stats(pbp, load_matchup_schedule(out))
    write_player_stats(tables, out, manifest_for=stats_manifest(observation))
    return {'stats_manifest_sha256': _sha256_file(out / STATS_MANIFEST),
            'rows': {k: int(len(v)) for k, v in tables.items()},
            'excluded_plays': {k: r['excluded_plays'] for k, r in reports.items()}}


def _stage_aggregate_training(run: Run) -> dict:
    return _aggregate(run, 'train_source', 'train_stats')


def _stage_train(run: Run) -> dict:
    from player_props.models import train_all_models
    result = train_all_models(data_dir=run.paths['train_stats'], models_dir=run.paths['train_models'])
    if not result['published']:
        raise RehearsalError("training published no bundle: " + "; ".join(result['problems']))
    run.info['bundle_dir'] = Path(result['bundle'])
    return {'bundle_id': Path(result['bundle']).name, 'models': result['models'],
            'skipped': result['skipped']}


def _stage_export(run: Run) -> dict:
    from player_props.bundle_store import export_bundle
    archive = run.paths['export'] / 'bundle.tar'
    index = export_bundle(run.paths['train_models'], archive)
    run.info['archive'] = archive
    return {'archive_sha256': _sha256_file(archive), 'bundle_id': index['bundle_id'],
            'files': len(index['files'])}


def _stage_restore(run: Run) -> dict:
    from player_props.bundle_store import import_bundle
    if any(run.paths['runtime_models'].iterdir()):
        raise RehearsalError("the runtime models directory is not empty before restore")
    result = import_bundle(run.paths['runtime_models'], run.info['archive'])
    bundle = load_bundle(run.paths['runtime_models'], PROP_LINES)
    if bundle['manifest']['bundle_id'] != run.info['bundle_dir'].name:
        raise RehearsalError("restored bundle id differs from the trained bundle")
    run.info['bundle'] = bundle
    return {'bundle_id': result['bundle_id'], 'installed': result['installed']}


def _stage_aggregate_inference(run: Run) -> dict:
    detail = _aggregate(run, 'infer_source', 'runtime_data')
    data = run.paths['runtime_data']
    shutil.copy2(run.paths['fixtures'] / 'players.csv', data / 'players.csv')
    schedule = pd.read_csv(run.paths['fixtures'] / SCHEDULE_FILE, sep='\t')
    fixtures.prediction_schedule(schedule, run.config.target_season).to_csv(
        data / f'nfl_schedule_{run.config.target_season}.csv', index=False)
    return detail


def _stage_validate_inputs(run: Run) -> dict:
    cfg = run.config
    check = check_inputs(cfg.target_season, cfg.target_week, run.paths['runtime_data'])
    if not check.ok:
        raise RehearsalError("inference inputs not established: "
                             + "; ".join(check.provenance_problems + check.freshness_problems))
    run.info['check'] = check
    bundle = run.info['bundle']
    trained_through = bundle['manifest']['training_source']['last_game_date']
    return {'feature_cutoff': check.summary['feature_cutoff'], 'trained_through': trained_through,
            'notes': check.notes}


def _stage_predict(run: Run) -> dict:
    from player_props import predict
    cfg = run.config
    code = predict.generate_predictions(season=cfg.target_season, week=cfg.target_week, freeze=True,
                                        skip_injuries=True, skip_weather=True, skip_market_odds=True,
                                        data_dir=run.paths['runtime_data'],
                                        models_dir=run.paths['runtime_models'])
    if code == 1:
        raise RehearsalError("prediction refused its inputs or models (see the log above)")
    latest = run.paths['runtime_data'] / 'player_props_predictions.csv'
    snap = run.paths['runtime_data'] / f'player_props_predictions_week{cfg.target_week}_{cfg.target_season}.csv'
    if not latest.exists() or not snap.exists():
        raise RehearsalError("prediction wrote no predictions file")
    run.info['latest'], run.info['snapshot'] = latest, snap
    return {'rows': int(len(pd.read_csv(latest))), 'sha256': _sha256_file(latest)}


# predict.predict_props_for_game's post-processing, restated here rather than
# imported so that a change there fails the rehearsal instead of silently moving
# the check with it. With injury, weather and market inputs disabled, the only
# permitted change to the raw score is one model clamp.
STAR_TIERS = ('elite_rb', 'star_rb', 'elite_wr', 'star_wr')
STAR_FLOOR, STAR_RATIO, STAR_LABEL = 0.75, 1.3, 'clamp_star_floor_0.75'
LOW_CAP, LOW_RATIO, LOW_LABEL = 0.25, 0.6, 'clamp_low_average_cap_0.25'
RECOMMEND_OVER_AT = 0.55
# What the skipped inputs must leave behind on every row.
SKIPPED_INPUT_FIELDS = {'weather_adjusted': False, 'weather_conditions': 'Skipped',
                        'market_line_available': False}
EMPTY_MARKET_FIELDS = ['market_line', 'market_book', 'market_implied_prob', 'market_edge']


def expected_post_processing(raw: float, tier: str, recent_avg: float, line: float) -> dict:
    """The final fields production should derive from the raw score here."""
    p = raw
    if tier in STAR_TIERS and recent_avg > line * STAR_RATIO:
        p = max(p, STAR_FLOOR)
    elif recent_avg < line * LOW_RATIO:
        p = min(p, LOW_CAP)
    labels = [] if p == raw else [STAR_LABEL if p > raw else LOW_LABEL]
    return {'prob_over': p, 'prob_under': 1 - p, 'confidence': max(p, 1 - p),
            'recommendation': 'OVER' if p >= RECOMMEND_OVER_AT else 'UNDER',
            'adjustments': ';'.join(labels) or 'none',
            'included': p >= RECOMMEND_OVER_AT or p <= 1 - RECOMMEND_OVER_AT}


def read_predictions(path: Path) -> pd.DataFrame:
    # Only empty cells are missing: 'none' and 'None' stay text.
    return pd.read_csv(path, float_precision='round_trip', keep_default_na=False, na_values=[''])


def _same(a, b) -> bool:
    return (pd.isna(a) and pd.isna(b)) if pd.isna(a) or pd.isna(b) else a == b


class _Expectations:
    """Recomputes, per prediction, what production should have written: the
    tier and model it would pick, the raw score, and the post-processed fields."""

    def __init__(self, run: Run):
        from player_props.predict import get_player_performance_tier
        self.tier_of = get_player_performance_tier
        self.cfg = run.config
        self.models = run.info['bundle']['models']
        players = pd.read_csv(run.paths['runtime_data'] / 'players.csv')
        self.ids = dict(zip(players['display_name'], players['gsis_id']))
        self.games = load_games(run.paths['runtime_data'])
        self.feature = run.info['check'].feature_tables
        self.cache: Dict[tuple, dict] = {}

    def for_row(self, r: dict) -> dict:
        key = (r['display_name'], r['player_name'], r['team'], r['opponent'], r['is_home'], r['prop_type'])
        if key not in self.cache:
            self.cache[key] = self._compute(r)
        return self.cache[key]

    def _compute(self, r: dict) -> dict:
        cfg, prop_type = self.cfg, r['prop_type']
        player_id = self.ids.get(r['display_name'])
        if player_id is None:
            return {'problem': f"unknown player {r['display_name']!r}"}
        tier = self.tier_of(player_id, prop_type, self.feature)
        # predict_props_for_game takes the first model with this prop type and tier.
        name = next((n for n, i in self.models.items()
                     if i['prop_type'] == prop_type and i['line_type'] == tier), None)
        if name is None:
            return {'problem': f"no restored model for {prop_type} / {tier}"}
        info = self.models[name]
        home, away = (r['team'], r['opponent']) if r['is_home'] else (r['opponent'], r['team'])
        game = resolve_game(self.games, cfg.target_season, cfg.target_week, home, away)
        target = pd.DataFrame([dict(player_id=player_id, player_name=r['player_name'],
                                    game_id=game['game_id'], season=cfg.target_season,
                                    week=cfg.target_week, team=r['team'], opponent=r['opponent'],
                                    game_date=game['gameday'])], columns=TARGET_COLUMNS)
        rows = inference_rows(self.feature[info['stat_type']], target, info['stat_type'], self.games)
        X = model_matrix(rows, prop_type)
        raw = float(score_probabilities(info['model'], X, prop_type)[0])
        features = X.iloc[0].to_dict()
        recent_avg = features.get(f"{info['stat_col']}_L5", 0)
        exp = {'model_name': name, 'line_type': tier, 'line_value': info['line_value'],
               'model_reliable': info.get('reliable', False), 'raw_model_prob_over': raw,
               **{f'avg_{w}': features.get(f"{info['stat_col']}_{w}") for w in ('L3', 'L5', 'L10')},
               **expected_post_processing(raw, tier, recent_avg, info['line_value'])}
        return exp


EXPECTED_FIELDS = ['model_name', 'line_type', 'line_value', 'model_reliable', 'raw_model_prob_over',
                   'prob_over', 'prob_under', 'confidence', 'recommendation', 'adjustments',
                   'avg_L3', 'avg_L5', 'avg_L10']


def prediction_problems(frame: pd.DataFrame, label: str, run: Run, expect: _Expectations) -> List[str]:
    """Every row of one predictions file checked against the independent
    recomputation, the run's bundle identity and the skipped-input rules."""
    cfg = run.config
    bundle_id = run.info['bundle']['manifest']['bundle_id']
    trained_id = run.info['bundle_dir'].name
    problems = []
    if frame.empty:
        return [f"{label}: no predictions"]
    keys = frame[['player_name', 'team', 'prop_type']].apply(tuple, axis=1)
    if keys.duplicated().any():
        problems.append(f"{label}: duplicate predictions {sorted(set(keys[keys.duplicated()]))[:3]}")
    for r in frame.to_dict('records'):
        tag = f"{label}: {r['player_name']} {r['prop_type']}"
        if r['season'] != cfg.target_season or r['week'] != cfg.target_week:
            problems.append(f"{tag}: season/week {r['season']}/{r['week']} is not the target")
        if r['model_bundle'] != bundle_id or r['model_bundle'] != trained_id:
            problems.append(f"{tag}: bundle {r['model_bundle']} is not this run's bundle {trained_id}")
        exp = expect.for_row(r)
        if 'problem' in exp:
            problems.append(f"{tag}: {exp['problem']}")
            continue
        if not exp['included']:
            problems.append(f"{tag}: below the confidence threshold, production would not include it")
        for f in EXPECTED_FIELDS:
            if not _same(r[f], exp[f]):
                problems.append(f"{tag}: {f} is {r[f]!r}, expected {exp[f]!r}")
        for f, v in SKIPPED_INPUT_FIELDS.items():
            if not _same(r[f], v):
                problems.append(f"{tag}: {f} is {r[f]!r}, but that input is skipped ({v!r} expected)")
        for f in EMPTY_MARKET_FIELDS:
            if not pd.isna(r[f]):
                problems.append(f"{tag}: {f} is {r[f]!r}, but market odds are skipped")
    return problems


def _stage_verify_outputs(run: Run) -> dict:
    """Independent checks before anything is published:

    * every row of the latest feed AND of the frozen snapshot is recomputed: the
      tier and model production would pick, the restored model's raw score, the
      permitted clamp and its exact label, prob_over / prob_under / confidence /
      recommendation, the skipped inputs, the line and this run's bundle ID;
    * the snapshot holds exactly the latest feed's rows and values;
    * each model's stored metrics equal a recomputation with the restored model
      on its evaluation rows.

    The verified files' hashes are recorded; publish refuses files that changed."""
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
    models, manifest = run.info['bundle']['models'], run.info['bundle']['manifest']
    latest, snapshot = read_predictions(run.info['latest']), read_predictions(run.info['snapshot'])
    expect = _Expectations(run)
    problems = (prediction_problems(latest, 'latest feed', run, expect)
                + prediction_problems(snapshot, 'frozen snapshot', run, expect))
    if list(latest.columns) != list(snapshot.columns) or not latest.equals(snapshot):
        problems.append("the frozen snapshot does not hold exactly the latest feed's predictions")
    train_tables = {s: pd.read_csv(run.paths['train_stats'] / f, float_precision='round_trip')
                    for s, f in STATS_FILES.items()}
    parity = 0
    for name, info in models.items():
        entry = manifest['models'][name]
        rows, _ = training_rows(train_tables[info['stat_type']], info['prop_type'])
        rows = rows[rows['season'].isin(entry['evaluation']['seasons'])]
        y = (rows[PROP_STAT[info['prop_type']][1]] > info['line_value']).astype(int)
        p = score_probabilities(info['model'], model_matrix(rows, info['prop_type']), info['prop_type'])
        got = {'accuracy': float(accuracy_score(y, (p >= 0.5).astype(int))),
               'roc_auc': float(roc_auc_score(y, p)), 'brier': float(brier_score_loss(y, p)),
               'log_loss': float(log_loss(y, p, labels=[0, 1]))}
        for k, v in got.items():
            if v != entry['metrics'][k]:
                problems.append(f"{name}: {k} recomputed {v} != stored {entry['metrics'][k]}")
        parity += 1
    if problems:
        more = f" (+{len(problems) - 10} more)" if len(problems) > 10 else ""
        raise RehearsalError("output verification failed: " + "; ".join(problems[:10]) + more)
    run.info['verified'] = {k: _sha256_file(run.info[k]) for k in ('latest', 'snapshot')}
    return {'predictions_verified': int(len(latest)), 'snapshot_rows_verified': int(len(snapshot)),
            'models_metric_parity': parity,
            'adjustments': {k: int(v) for k, v in latest['adjustments'].value_counts().sort_index().items()},
            'latest_sha256': run.info['verified']['latest'],
            'snapshot_sha256': run.info['verified']['snapshot']}


def _stage_publish(run: Run) -> dict:
    """Copy the run's predictions to <workspace>/published, never overwriting a
    different frozen snapshot."""
    from player_props.source_refresh import install_files
    for k in ('latest', 'snapshot'):
        if _sha256_file(run.info[k]) != run.info['verified'][k]:
            raise RehearsalError(f"{run.info[k].name} changed after verification; nothing was published")
    published = safe_dir(run.workspace / 'published', run.workspace, run.config.protected)
    snap_name = run.info['snapshot'].name
    record_path = published / 'snapshots.json'
    record = json.loads(record_path.read_text(encoding='utf-8')) if record_path.exists() else {}
    existing = published / snap_name
    new_sha = _sha256_file(run.info['snapshot'])
    if existing.exists():
        old_sha = _sha256_file(existing)
        if old_sha != new_sha:
            run.info['conflict'] = True
            owner = record.get(snap_name, {}).get('run_id', 'unknown run')
            raise RehearsalError(f"snapshot {snap_name} is already frozen by {owner} with different "
                                 f"content; it was not overwritten and nothing was published")
        snapshot_status = 'identical_existing'
    else:
        snapshot_status = 'frozen'
    staging = safe_dir(published / f".publish-{run.run_id}", run.workspace, run.config.protected)
    try:
        staged = {}
        shutil.copy2(run.info['latest'], staging / run.info['latest'].name)
        staged[published / run.info['latest'].name] = staging / run.info['latest'].name
        if snapshot_status == 'frozen':
            shutil.copy2(run.info['snapshot'], staging / snap_name)
            staged[existing] = staging / snap_name
            record[snap_name] = {'run_id': run.run_id, 'sha256': new_sha,
                                 'bundle_id': run.info['bundle']['manifest']['bundle_id']}
        (staging / 'snapshots.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        staged[record_path] = staging / 'snapshots.json'
        install_files(staged)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {'published_dir': str(published), 'snapshot': snap_name, 'snapshot_status': snapshot_status,
            'snapshot_sha256': new_sha}


STAGE_FUNCS: Dict[str, Callable[[Run], dict]] = {
    'workspace': _stage_workspace, 'source_training': _stage_source_training,
    'source_inference': _stage_source_inference, 'aggregate_training': _stage_aggregate_training,
    'train': _stage_train, 'export': _stage_export, 'restore': _stage_restore,
    'aggregate_inference': _stage_aggregate_inference, 'validate_inputs': _stage_validate_inputs,
    'predict': _stage_predict, 'verify_outputs': _stage_verify_outputs, 'publish': _stage_publish,
}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _artifacts(run: Run) -> dict:
    out = {}
    for path in sorted(run.dir.rglob('*')):
        if path.is_file() and path.suffix in ('.gz', '.json', '.csv', '.tar') \
                and 'diagnostics' not in path.parts and path.name != 'summary.json':
            out[str(path.relative_to(run.dir)).replace('\\', '/')] = _sha256_file(path)
    return out


def _summary(run: Run, status: str) -> dict:
    bundle = run.info.get('bundle')
    windows = {}
    if bundle:
        for name, e in bundle['manifest']['models'].items():
            windows[name] = {'line': e['line_value'], 'training': {k: e['training'][k] for k in
                                                                   ('seasons', 'rows', 'first_date', 'last_date')},
                             'evaluation': {k: e['evaluation'][k] for k in
                                            ('seasons', 'rows', 'first_date', 'last_date')}}
    stage = {s['stage']: s for s in run.stages}
    return {
        'kind': SUMMARY_KIND, 'summary_version': SUMMARY_VERSION, 'run_id': run.run_id,
        'status': status, 'finished_at': _now(),
        'stages': [{'stage': s, 'status': stage.get(s, {}).get('status', 'not_run'),
                    **({'error': stage[s]['error']} if 'error' in stage.get(s, {}) else {})}
                   for s in STAGES],
        'coverage': {'training_source': stage.get('source_training', {}).get('detail'),
                     'inference_source': stage.get('source_inference', {}).get('detail'),
                     'inputs': stage.get('validate_inputs', {}).get('detail')},
        'bundle': ({'bundle_id': bundle['manifest']['bundle_id'],
                    'feature_contract': bundle['manifest']['feature_contract'],
                    'scorer': bundle['manifest']['scorer'],
                    'trained_through': bundle['manifest']['training_source']['last_game_date']}
                   if bundle else None),
        'windows': windows,
        'exclusions': {'skipped_models': stage.get('train', {}).get('detail', {}).get('skipped', {}),
                       'external_inputs_skipped': ['injuries', 'weather', 'market_odds']},
        'outputs': {k: stage.get(k, {}).get('detail') for k in ('predict', 'verify_outputs', 'publish')},
        'artifacts': _artifacts(run),
        'limitations': LIMITATIONS,
    }


def _write_json(path: Path, doc: dict) -> None:
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_rehearsal(config: RehearsalConfig,
                  after_stage: Optional[Callable[[str, Run], None]] = None) -> dict:
    """Run every stage in order; stop at the first failure. Returns the summary
    (also written to <run dir>/summary.json). ``after_stage(name, run)`` is
    called after each successful stage (tests use it to inject failures)."""
    workspace = check_workspace(Path(config.workspace), config.protected)
    workspace.mkdir(parents=True, exist_ok=True)
    workspace = check_workspace(workspace, config.protected)
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"
    run_dir = safe_dir(workspace / 'runs' / run_id, workspace, config.protected)
    run = Run(config, workspace, run_id, run_dir)
    status = 'succeeded'
    with no_network():
        for name in STAGES:
            try:
                detail = STAGE_FUNCS[name](run)
                run.stages.append({'stage': name, 'status': 'ok', 'detail': detail})
                if after_stage:
                    after_stage(name, run)
            except Exception as exc:  # noqa: BLE001 - any failure stops the run and is recorded
                problems = getattr(exc, 'problems', None)
                error = {'type': type(exc).__name__, 'message': str(exc)[:2000]}
                if problems:
                    error['problems'] = [str(p)[:500] for p in problems[:20]]
                run.stages.append({'stage': name, 'status': 'failed', 'error': error})
                diag = safe_dir(run_dir / 'diagnostics', workspace, config.protected)
                (diag / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
                status = 'conflict' if run.info.get('conflict') else 'failed'
                break
    summary = _summary(run, status)
    _write_json(run_dir / 'summary.json', summary)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Isolated end-to-end Player Props rehearsal "
                                             "(synthetic data, no network)")
    ap.add_argument('--workspace', type=Path, required=True,
                    help="directory outside the repository for this and later rehearsal runs")
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args(argv)
    try:
        summary = run_rehearsal(RehearsalConfig(workspace=args.workspace, seed=args.seed))
    except RehearsalError as exc:
        print(json.dumps({'status': 'refused', 'error': str(exc)}))
        return 2
    print(json.dumps({k: summary[k] for k in ('run_id', 'status', 'stages', 'bundle')}, indent=2,
                     default=str))
    return {'succeeded': 0, 'conflict': 3}.get(summary['status'], 1)


if __name__ == '__main__':
    sys.exit(main())
