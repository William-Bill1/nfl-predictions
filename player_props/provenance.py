"""
Player-prop source freshness and provenance checks (Stage 2).

The chain is play-by-play -> aggregated player stats -> models -> predictions.
This module covers the first link and the gate before predictions:

* ``read_source`` reads the play-by-play once and describes what is
  actually in it (seasons, games, dates, per-game END GAME evidence, content
  hash) at the time it was read. Nothing comes from pbp_metadata.json or the
  requested seasons.
* ``stats_manifest`` is passed to aggregators.write_player_stats, which writes
  the manifest in the same set as the three stats files. It links each file to
  its own hash, the source hash and coverage, the aggregation code revision and
  the feature definition.
* ``check_inputs`` is the read-only gate for a target week: the stats must match
  a well-formed manifest, and every earlier completed game must be in them. It
  returns the feature source: completed earlier games only.

Stats without a manifest are unverified; no provenance is created for them.

CLI (read-only; exit 0 = valid, 1 = not established, 2 = could not run):
    python player_props/provenance.py [--season 2026 --week 8] [--data-dir DIR]

Policy: docs/architecture.md ("Stage 2 freshness and provenance").
"""
import argparse
import hashlib
import io
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

try:
    from player_props.aggregators import (
        DATA_DIR, FEATURE_DEFINITION, ROLLING_WINDOWS, SCHEDULE_FILE, STATS_FILES, output_columns,
        read_stats_csv,
    )
except ImportError:  # run as a script from inside player_props/
    from aggregators import (
        DATA_DIR, FEATURE_DEFINITION, ROLLING_WINDOWS, SCHEDULE_FILE, STATS_FILES, output_columns,
        read_stats_csv,
    )

ROOT = Path(__file__).parent.parent
SOURCE_FILE = 'nfl_play_by_play_historical.csv.gz'
MANIFEST_FILE = 'player_props_stats_manifest.json'
MANIFEST_KIND = 'player_props_stats'
MANIFEST_VERSION = 1
AGGREGATION_CODE = 'player_props/aggregators.py'
# Read with any usecols, so the END GAME evidence is always observed.
COMPLETION_COLUMNS = ['game_id', 'desc', 'total_home_score', 'total_away_score']
FRESHNESS_COLUMNS = ['game_id', 'season', 'week', 'team', 'game_date']
GAME_COLUMNS = ['game_id', 'season', 'week', 'game_type', 'gameday', 'home_team', 'away_team',
                'home_score', 'away_score', 'result', 'total']
TORONTO = ZoneInfo('America/Toronto')
CLOCK_SKEW = timedelta(minutes=5)
EXAMPLES = 5
_HEX64 = re.compile(r'[0-9a-f]{64}')
_DATE = re.compile(r'\d{4}-\d{2}-\d{2}')


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _examples(items) -> str:
    items = sorted(items)
    more = f" (+{len(items) - EXAMPLES} more)" if len(items) > EXAMPLES else ""
    return ", ".join(str(i) for i in items[:EXAMPLES]) + more


def _dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format='%Y-%m-%d', errors='coerce')


def toronto_today() -> date:
    return datetime.now(TORONTO).date()


# ---------------------------------------------------------------------------
# Coverage from content
# ---------------------------------------------------------------------------

def content_coverage(df: pd.DataFrame) -> dict:
    """Seasons, games and dates actually present in a play-by-play or stats table."""
    dates = _dates(df['game_date'])
    out = {'rows': int(len(df)), 'games': int(df['game_id'].nunique()),
           'first_date': None, 'last_date': None, 'invalid_dates': int(dates.isna().sum()),
           'seasons': {}}
    if dates.notna().any():
        out['first_date'] = dates.min().strftime('%Y-%m-%d')
        out['last_date'] = dates.max().strftime('%Y-%m-%d')
    for season, part in df.assign(_d=dates).groupby('season'):
        d = part['_d'].dropna()
        out['seasons'][str(int(season))] = {
            'games': int(part['game_id'].nunique()), 'rows': int(len(part)),
            'weeks': [int(part['week'].min()), int(part['week'].max())],
            'first_date': d.min().strftime('%Y-%m-%d') if len(d) else None,
            'last_date': d.max().strftime('%Y-%m-%d') if len(d) else None,
        }
    if 'player_id' in df.columns:
        out['players'] = int(df['player_id'].nunique())
    return out


def _source_completion(df: pd.DataFrame) -> Dict[str, dict]:
    """Per game: whether its END GAME play is present and that play's score
    (backtest.pbp_game_completion, the rule the weekly backtest uses)."""
    try:
        from player_props.backtest import pbp_game_completion
    except ImportError:
        from backtest import pbp_game_completion
    return pbp_game_completion(df)


def _observation(path: Path, data: bytes, df: pd.DataFrame) -> dict:
    return {'file': path.name, 'sha256': _sha256(data), 'bytes': len(data),
            # When this content was read - not when the file was generated,
            # which nothing records.
            'observed_at': _now(), 'coverage': content_coverage(df),
            'completion': _source_completion(df)}


def read_source(path: Path | str, usecols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, dict]:
    """Read the tab-separated, gzipped play-by-play once; return it and an
    observation whose hash, coverage and completion evidence describe exactly
    the bytes parsed. The END GAME columns are always read when present."""
    path = Path(path)
    data = path.read_bytes()
    compression = 'gzip' if path.suffix == '.gz' else None
    wanted = None if usecols is None else set(usecols) | set(COMPLETION_COLUMNS)
    df = pd.read_csv(io.BytesIO(data), compression=compression, sep='\t', low_memory=False,
                     usecols=None if wanted is None else (lambda c: c in wanted))
    return df, _observation(path, data, df)


def code_revision(root: Path = ROOT) -> dict:
    """The aggregation code that built the stats: its file hash (authoritative)
    plus the git commit and whether that file had uncommitted changes."""
    code = root / AGGREGATION_CODE
    rev = {'file': AGGREGATION_CODE, 'sha256': _sha256(code.read_bytes()),
           'git_commit': None, 'git_dirty': None}
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True,
                                text=True, timeout=10)
        status = subprocess.run(['git', 'status', '--porcelain', '--', AGGREGATION_CODE], cwd=root,
                                capture_output=True, text=True, timeout=10)
        if commit.returncode == 0 and status.returncode == 0:
            rev['git_commit'] = commit.stdout.strip()
            rev['git_dirty'] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return rev


def stats_manifest(source: dict, windows: List[int] = ROLLING_WINDOWS,
                   root: Path = ROOT) -> Callable[[Dict[str, Path]], dict]:
    """Manifest builder for write_player_stats(manifest_for=...). It is called
    with the temporary stats files about to be moved into place, so each hash
    and coverage describes the exact bytes that are installed."""
    problems = _source_problems(source) if isinstance(source, dict) else ["not an object"]
    if problems:
        raise ValueError(f"a source observation from read_source() is required ({problems[0]})")
    code = code_revision(root)

    def build(paths: Dict[str, Path]) -> dict:
        stats = {}
        for stat_type, name in STATS_FILES.items():
            data = Path(paths[stat_type]).read_bytes()
            df = read_stats_csv(io.BytesIO(data))
            stats[stat_type] = {'file': name, 'sha256': _sha256(data), 'bytes': len(data),
                                'columns': list(df.columns), 'coverage': content_coverage(df)}
        return {'kind': MANIFEST_KIND, 'manifest_version': MANIFEST_VERSION,
                # The manifest is installed with the stats, so this is when they were written.
                'written_at': _now(),
                'source': source,
                'aggregation': {'code': code, 'feature_definition': FEATURE_DEFINITION,
                                'rolling_windows': list(windows)},
                'stats': stats}
    return build


# ---------------------------------------------------------------------------
# Manifest structure
# ---------------------------------------------------------------------------

def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_hash(v) -> bool:
    return isinstance(v, str) and _HEX64.fullmatch(v) is not None


def _is_date(v) -> bool:
    if not isinstance(v, str) or not _DATE.fullmatch(v):
        return False
    try:
        date.fromisoformat(v)
        return True
    except ValueError:
        return False


def _timestamp(v) -> Optional[datetime]:
    """A time-zone-aware ISO timestamp, or None."""
    if not isinstance(v, str):
        return None
    try:
        ts = datetime.fromisoformat(v)
    except ValueError:
        return None
    return ts if ts.tzinfo is not None else None


def _coverage_problems(cov, where: str) -> List[str]:
    if not isinstance(cov, dict):
        return [f"{where} must be an object"]
    problems = [f"{where}.{k} must be a non-negative integer"
                for k in ('rows', 'games', 'invalid_dates')
                if not _is_int(cov.get(k)) or cov[k] < 0]
    problems += [f"{where}.{k} must be a YYYY-MM-DD date or null"
                 for k in ('first_date', 'last_date')
                 if cov.get(k) is not None and not _is_date(cov.get(k))]
    seasons = cov.get('seasons')
    if not isinstance(seasons, dict):
        problems.append(f"{where}.seasons must be an object")
    else:
        bad = [s for s, e in seasons.items()
               if not (isinstance(e, dict) and _is_int(e.get('games')) and _is_int(e.get('rows')))]
        if bad:
            problems.append(f"{where}.seasons has malformed entries: {_examples(bad)}")
    return problems


def _source_problems(source: dict) -> List[str]:
    problems = []
    if not isinstance(source.get('file'), str) or not source.get('file'):
        problems.append("source.file must be a file name")
    if not _is_hash(source.get('sha256')):
        problems.append("source.sha256 must be a 64-character hex SHA-256")
    if not _is_int(source.get('bytes')) or source['bytes'] <= 0:
        problems.append("source.bytes must be a positive integer")
    if _timestamp(source.get('observed_at')) is None:
        problems.append("source.observed_at must be an ISO timestamp with a time zone")
    problems += _coverage_problems(source.get('coverage'), "source.coverage")
    completion = source.get('completion')
    if not isinstance(completion, dict):
        problems.append("source.completion must be an object")
    else:
        bad = [g for g, e in completion.items()
               if not (isinstance(e, dict) and isinstance(e.get('end_game'), bool)
                       and all(e.get(k) is None or _is_int(e.get(k)) for k in ('home_total', 'away_total')))]
        if bad:
            problems.append(f"source.completion has malformed entries: {_examples(bad)}")
    return problems


def manifest_structure_problems(manifest, windows: List[int] = ROLLING_WINDOWS,
                                now: Optional[datetime] = None) -> List[str]:
    """Every structural problem in a parsed manifest (empty when well-formed).
    Nested values are type-checked before any is used."""
    if not isinstance(manifest, dict):
        return ["the manifest must be a JSON object"]
    if manifest.get('kind') != MANIFEST_KIND or manifest.get('manifest_version') != MANIFEST_VERSION:
        return [f"this is not a version-{MANIFEST_VERSION} {MANIFEST_KIND} manifest"]
    now = now or datetime.now(timezone.utc)
    problems = []
    written = _timestamp(manifest.get('written_at'))
    if written is None:
        problems.append("written_at must be an ISO timestamp with a time zone")
    elif written > now + CLOCK_SKEW:
        problems.append("written_at is in the future")

    source = manifest.get('source')
    if not isinstance(source, dict):
        problems.append("source must be an object")
    else:
        problems += _source_problems(source)
        observed = _timestamp(source.get('observed_at'))
        if observed and observed > now + CLOCK_SKEW:
            problems.append("source.observed_at is in the future")
        if observed and written and observed > written:
            problems.append("source.observed_at is after written_at (stats written before their source was read)")

    agg = manifest.get('aggregation')
    if not isinstance(agg, dict):
        problems.append("aggregation must be an object")
    else:
        code = agg.get('code')
        if not isinstance(code, dict):
            problems.append("aggregation.code must be an object")
        else:
            if code.get('file') != AGGREGATION_CODE:
                problems.append(f"aggregation.code.file must be {AGGREGATION_CODE}")
            if not _is_hash(code.get('sha256')):
                problems.append("aggregation.code.sha256 must be a 64-character hex SHA-256")
            if code.get('git_commit') is not None and not isinstance(code.get('git_commit'), str):
                problems.append("aggregation.code.git_commit must be a string or null")
            if code.get('git_dirty') is not None and not isinstance(code.get('git_dirty'), bool):
                problems.append("aggregation.code.git_dirty must be true, false or null")
        if not isinstance(agg.get('feature_definition'), str):
            problems.append("aggregation.feature_definition must be a string")
        windows_seen = agg.get('rolling_windows')
        if not (isinstance(windows_seen, list) and all(_is_int(w) for w in windows_seen)):
            problems.append("aggregation.rolling_windows must be a list of integers")

    stats = manifest.get('stats')
    if not isinstance(stats, dict):
        problems.append("stats must be an object")
    else:
        for stat_type, name in STATS_FILES.items():
            entry = stats.get(stat_type)
            where = f"stats.{stat_type}"
            if not isinstance(entry, dict):
                problems.append(f"{where} must be an object (no entry for {name})")
                continue
            if entry.get('file') != name:
                problems.append(f"{where}.file must be {name}")
            if not _is_hash(entry.get('sha256')):
                problems.append(f"{where}.sha256 must be a 64-character hex SHA-256")
            if not _is_int(entry.get('bytes')) or entry['bytes'] < 0:
                problems.append(f"{where}.bytes must be a non-negative integer")
            cols = entry.get('columns')
            if not (isinstance(cols, list) and all(isinstance(c, str) for c in cols)):
                problems.append(f"{where}.columns must be a list of column names")
            problems += _coverage_problems(entry.get('coverage'), f"{where}.coverage")
    return problems


# ---------------------------------------------------------------------------
# Provenance check
# ---------------------------------------------------------------------------

def _read_stats(data_dir: Path) -> Tuple[Dict[str, Optional[bytes]], Dict[str, pd.DataFrame], List[str]]:
    raw, tables, problems = {}, {}, []
    for stat_type, name in STATS_FILES.items():
        path = data_dir / name
        try:
            raw[stat_type] = path.read_bytes()
            tables[stat_type] = read_stats_csv(io.BytesIO(raw[stat_type]))
        except FileNotFoundError:
            raw[stat_type] = None
            problems.append(f"{name} is missing")
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raw[stat_type] = None
            problems.append(f"{name} could not be read: {exc}")
    return raw, tables, problems


def check_provenance(data_dir: Path, raw: Dict[str, Optional[bytes]],
                     tables: Dict[str, pd.DataFrame], windows: List[int] = ROLLING_WINDOWS
                     ) -> Tuple[List[str], List[str], Optional[dict]]:
    """(problems, notes, manifest) for the stats bytes already read. The
    manifest is returned only when it is well-formed."""
    problems, notes = [], []
    path = data_dir / MANIFEST_FILE
    if not path.exists():
        return ([f"no provenance manifest ({MANIFEST_FILE}): the stats files are unverified - "
                 f"their source, build code and generation time are unknown"], notes, None)
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return [f"{MANIFEST_FILE} could not be read: {exc}"], notes, None
    structure = manifest_structure_problems(manifest, windows)
    if structure:
        return [f"{MANIFEST_FILE} is malformed: {p}" for p in structure], notes, None

    source, agg, code = manifest['source'], manifest['aggregation'], manifest['aggregation']['code']
    if agg['feature_definition'] != FEATURE_DEFINITION:
        problems.append(f"stats were built under feature definition {agg['feature_definition']!r}, "
                        f"current is {FEATURE_DEFINITION!r}")
    if agg['rolling_windows'] != list(windows):
        problems.append(f"stats were built with rolling windows {agg['rolling_windows']}, "
                        f"expected {list(windows)}")
    current = (ROOT / AGGREGATION_CODE)
    if not current.exists() or _sha256(current.read_bytes()) != code['sha256']:
        notes.append(f"{AGGREGATION_CODE} has changed since the stats were built "
                     f"(built at commit {code.get('git_commit') or 'unknown'}); "
                     f"the feature definition is unchanged")
    if code.get('git_dirty'):
        notes.append(f"stats were built from uncommitted changes to {AGGREGATION_CODE}; "
                     f"only its file hash identifies that code")

    for stat_type, name in STATS_FILES.items():
        entry = manifest['stats'][stat_type]
        expected = output_columns(stat_type, windows)
        if raw.get(stat_type) is None:
            continue  # already reported as missing or unreadable
        if entry['sha256'] != _sha256(raw[stat_type]) or entry['bytes'] != len(raw[stat_type]):
            problems.append(f"{name} does not match its manifest hash (changed after it was built, "
                            f"or the manifest belongs to other files)")
        elif entry['coverage'] != content_coverage(tables[stat_type]):
            problems.append(f"{name}: manifest coverage does not match the file's content")
        if entry['columns'] != expected:
            problems.append(f"{name}: manifest columns differ from the current contract")
        if list(tables[stat_type].columns) != expected:
            problems.append(f"{name}: columns differ from the current contract")

    src = data_dir / SOURCE_FILE
    if src.exists():
        data = src.read_bytes()
        if data.startswith(b'version https://git-lfs'):
            notes.append(f"the current {SOURCE_FILE} is a Git LFS pointer, not play-by-play data")
        elif _sha256(data) != source['sha256']:
            notes.append(f"the current {SOURCE_FILE} is not the source these stats were built from; "
                         f"only the stats' own coverage is used for freshness")
    return problems, notes, manifest


# ---------------------------------------------------------------------------
# Completion evidence
# ---------------------------------------------------------------------------

# Why an earlier game does (or doesn't) count as completed, in the order
# checked. Only 'completed' counts; schedule scores are recorded results, not
# an authoritative final status, so they must agree with each other and with
# the END GAME play of the play-by-play the stats were built from.
COMPLETION_REASONS = {
    'no_score': "have no recorded score, so completion is unknown (postponed, in progress or not yet recorded)",
    'one_score': "have only one team's score",
    'invalid_score': "have invalid scores (not whole, non-negative numbers, or an impossible 1)",
    'placeholder': "show 0-0, which is treated as a placeholder, not a result",
    'no_result_total': "have scores but no recorded result/total to corroborate them",
    'inconsistent': "have scores that disagree with their recorded result/total",
    'not_past': "are dated today or later (Toronto), so they are not treated as complete yet",
    'no_source': "can't be corroborated: there is no valid manifest recording the stats' source END GAME evidence",
    'not_in_source': "are not in the play-by-play the stats were built from",
    'source_unended': "have no single END GAME score in the stats' source (possibly partial)",
    'source_mismatch': "have an END GAME score in the stats' source that differs from the recorded score",
}
SCHEDULE_LEVEL_OK = {'no_source', 'not_in_source', 'source_unended', 'source_mismatch'}


def _whole(value) -> Optional[int]:
    """A whole, finite, non-negative number as int, else None."""
    if isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f < 0 or f != int(f):
        return None
    return int(f)


def _number(value) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def completion_status(row, source_completion: Optional[Dict[str, dict]], today: date) -> str:
    """'completed', or the COMPLETION_REASONS key that stops a game counting.

    A game is completed only when: both scores are present, whole,
    non-negative and possible (no 1); they are not 0-0 (unplayed rows are
    filled with 0 elsewhere - the betting log and weekly backtest use the same
    'total > 0' convention); the recorded result and total equal home - away
    and home + away; the game is dated before today in Toronto (the betting
    log's 'gameday < today'); and the play-by-play the stats were built from
    has that game's END GAME play showing the same score (the weekly
    backtest's rule, backtest.pbp_game_completion)."""
    h_missing, a_missing = pd.isna(row.home_score), pd.isna(row.away_score)
    if h_missing and a_missing:
        return 'no_score'
    if h_missing or a_missing:
        return 'one_score'
    h, a = _whole(row.home_score), _whole(row.away_score)
    if h is None or a is None or 1 in (h, a):
        return 'invalid_score'
    if h == 0 and a == 0:
        return 'placeholder'
    if pd.isna(row.result) or pd.isna(row.total):
        return 'no_result_total'
    if _number(row.result) != h - a or _number(row.total) != h + a:
        return 'inconsistent'
    if row.parsed_day.date() >= today:
        return 'not_past'
    if source_completion is None:
        return 'no_source'
    evidence = source_completion.get(str(row.game_id))
    if evidence is None:
        return 'not_in_source'
    if not evidence.get('end_game') or evidence.get('home_total') is None \
            or evidence.get('away_total') is None:
        return 'source_unended'
    if (evidence['home_total'], evidence['away_total']) != (h, a):
        return 'source_mismatch'
    return 'completed'


# ---------------------------------------------------------------------------
# Freshness check
# ---------------------------------------------------------------------------

def load_games(data_dir: Path) -> pd.DataFrame:
    """The game schedule with recorded results (nfl_games_historical.csv,
    tab-separated), read-only."""
    return pd.read_csv(data_dir / SCHEDULE_FILE, sep='\t', low_memory=False, usecols=GAME_COLUMNS)


def _before(season: pd.Series, week: pd.Series, target: Tuple[int, int]) -> pd.Series:
    s, w = target
    return (season < s) | ((season == s) & (week < w))


def check_freshness(tables: Dict[str, pd.DataFrame], games: pd.DataFrame, season: int, week: int,
                    prediction_matchups: Optional[Set[Tuple[str, str]]] = None,
                    source_completion: Optional[Dict[str, dict]] = None,
                    today: Optional[date] = None,
                    ) -> Tuple[List[str], List[str], Dict[str, pd.DataFrame], dict]:
    """(problems, notes, feature_tables, summary) for target (season, week).

    feature_tables keep only rows from completed games (completion_status)
    before the target week and dated before its first game."""
    problems, notes = [], []
    target = (int(season), int(week))
    today = today or toronto_today()
    summary: dict = {'target_season': target[0], 'target_week': target[1]}

    g = games.copy()
    g['parsed_day'] = _dates(g['gameday'])
    dup = g['game_id'][g['game_id'].duplicated(keep=False)].unique()
    if len(dup):
        problems.append(f"schedule lists {len(dup)} game_id(s) more than once: {_examples(dup)}")
    sides = pd.concat([g[['season', 'week', 'home_team']].rename(columns={'home_team': 'team'}),
                       g[['season', 'week', 'away_team']].rename(columns={'away_team': 'team'})])
    twice = sides[sides.duplicated(keep=False)].drop_duplicates()
    if len(twice):
        problems.append(f"schedule puts a team in two games in one week: "
                        f"{_examples(f'{r.season} wk{r.week} {r.team}' for r in twice.itertuples())}")

    tw = g[(g['season'] == target[0]) & (g['week'] == target[1])]
    if tw.empty:
        problems.append(f"schedule has no games for {target[0]} week {target[1]}")
        return problems, notes, {}, summary
    if tw['parsed_day'].isna().any():
        problems.append(f"schedule has invalid dates for target-week game(s): "
                        f"{_examples(tw.loc[tw['parsed_day'].isna(), 'game_id'])}")
        return problems, notes, {}, summary
    cutoff = tw['parsed_day'].min()
    summary['feature_cutoff'] = cutoff.strftime('%Y-%m-%d')
    if prediction_matchups is not None:
        sched = set(zip(tw['home_team'], tw['away_team']))
        if set(prediction_matchups) != sched:
            problems.append(f"the prediction schedule and the game schedule disagree on "
                            f"{target[0]} week {target[1]}: only in prediction schedule "
                            f"{_examples(f'{a}@{h}' for h, a in set(prediction_matchups) - sched) or 'none'}; "
                            f"only in game schedule {_examples(f'{a}@{h}' for h, a in sched - set(prediction_matchups)) or 'none'}")
    if tw['home_score'].notna().any():
        notes.append(f"{int(tw['home_score'].notna().sum())} target-week game(s) already have recorded "
                     f"scores; they are excluded from the feature source")

    # Seasons to check: the target season, the one before it (its history feeds
    # early-season features) and every earlier season the stats contain.
    stats_seasons = [int(s) for t in tables.values() if 'season' in t.columns
                     for s in pd.to_numeric(t['season'], errors='coerce').dropna().unique()]
    first_season = min([target[0] - 1] + stats_seasons)
    prior = g[(g['season'] >= first_season) & _before(g['season'], g['week'], target)]
    bad_day = prior['parsed_day'].isna()
    later = prior['parsed_day'] >= cutoff
    if bad_day.any():
        problems.append(f"{int(bad_day.sum())} earlier game(s) have no valid date: "
                        f"{_examples(prior.loc[bad_day, 'game_id'])}")
    if later.any():
        notes.append(f"{int(later.sum())} game(s) from earlier weeks are dated on or after the target week "
                     f"starts and are not expected: {_examples(prior.loc[later, 'game_id'])}")
    due = prior[~bad_day & ~later]
    status = pd.Series([completion_status(r, source_completion, today) for r in due.itertuples()],
                       index=due.index, dtype=object)
    for reason, text in COMPLETION_REASONS.items():
        ids = due.loc[status == reason, 'game_id']
        if len(ids):
            problems.append(f"{len(ids)} earlier game(s) {text}: {_examples(ids)}")
    # Expected in the stats: schedule evidence is complete (source corroboration
    # failures are reported above). Feature source: fully completed games only.
    expected = due[(status == 'completed') | status.isin(SCHEDULE_LEVEL_OK)]
    completed = due[status == 'completed']
    summary['expected_games'] = {str(s): int(n) for s, n in expected.groupby('season').size().items()}
    summary['first_checked_season'] = first_season

    known = set(g['game_id'])
    allowed = set(completed['game_id'])
    teams_seen: Dict[str, Set[str]] = {}
    feature_tables, excluded = {}, {}
    for stat_type, df in tables.items():
        name = STATS_FILES[stat_type]
        missing_cols = [c for c in FRESHNESS_COLUMNS if c not in df.columns]
        if missing_cols:
            problems.append(f"{name}: missing column(s) {missing_cols}")
            continue
        day = _dates(df['game_date'])
        s = pd.to_numeric(df['season'], errors='coerce')
        w = pd.to_numeric(df['week'], errors='coerce')
        if day.isna().any() or s.isna().any() or w.isna().any():
            problems.append(f"{name}: rows with invalid game_date, season or week")
            continue
        unknown = set(df['game_id']) - known
        if unknown:
            problems.append(f"{name}: {len(unknown)} game(s) not in the schedule: {_examples(unknown)}")
        sched_day = df['game_id'].map(dict(zip(g['game_id'], g['parsed_day'])))
        off_date = df['game_id'].isin(known) & (day != sched_day)
        if off_date.any():
            problems.append(f"{name}: {int(off_date.sum())} row(s) dated differently from their "
                            f"schedule game: {_examples(set(df.loc[off_date, 'game_id']))}")
        earlier = _before(s, w, target) & (day < cutoff) & df['game_id'].isin(known)
        unfinished = (set(df.loc[earlier, 'game_id']) - set(expected['game_id'])
                      - set(prior.loc[later, 'game_id']))
        if unfinished:
            problems.append(f"{name}: {len(unfinished)} game(s) without completion evidence "
                            f"(possibly partial): {_examples(unfinished)}")
        # Completed games before the target week, dated before its first game -
        # nothing from the target week or later can be in this set.
        keep = df['game_id'].isin(allowed)
        excluded[stat_type] = int((~keep & df['game_id'].isin(known)).sum())
        feature = df[keep].reset_index(drop=True)
        feature_tables[stat_type] = feature
        present = set(df.loc[earlier, 'game_id'])
        for gid, team in zip(df.loc[earlier, 'game_id'], df.loc[earlier, 'team']):
            teams_seen.setdefault(gid, set()).add(team)
        absent = expected[~expected['game_id'].isin(present)]
        if len(absent):
            by_season = ", ".join(f"{int(k)}: {n}" for k, n in absent.groupby('season').size().items())
            problems.append(f"{name}: {len(absent)} completed earlier game(s) missing ({by_season}): "
                            f"{_examples(absent['game_id'])}")
        if len(feature):
            summary.setdefault('feature_last_date', {})[stat_type] = _dates(feature['game_date']).max().strftime('%Y-%m-%d')
    if sum(excluded.values()):
        notes.append("rows from games on or after the target week start, or without completion "
                     "evidence, excluded from the feature source: "
                     + ", ".join(f"{k} {v}" for k, v in excluded.items()))
    one_side = [r.game_id for r in expected.itertuples()
                if r.game_id in teams_seen and not {r.home_team, r.away_team} <= teams_seen[r.game_id]]
    if one_side:
        problems.append(f"{len(one_side)} completed game(s) have stats for only one team: {_examples(one_side)}")
    summary['excluded_rows'] = excluded
    return problems, notes, feature_tables, summary


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

@dataclass
class InputCheck:
    season: int
    week: int
    provenance_problems: List[str] = field(default_factory=list)
    freshness_problems: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    feature_tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    manifest: Optional[dict] = None

    @property
    def ok(self) -> bool:
        return not self.provenance_problems and not self.freshness_problems


def check_inputs(season: int, week: int, data_dir: Path | str = DATA_DIR,
                 prediction_matchups: Optional[Set[Tuple[str, str]]] = None,
                 windows: List[int] = ROLLING_WINDOWS, today: Optional[date] = None) -> InputCheck:
    """Read-only check that the stats files can feed predictions for
    (season, week). Never writes. ``ok`` is False unless both provenance and
    freshness are established."""
    data_dir = Path(data_dir)
    result = InputCheck(int(season), int(week))
    raw, tables, read_problems = _read_stats(data_dir)
    result.provenance_problems += read_problems
    problems, notes, manifest = check_provenance(data_dir, raw, tables, windows)
    result.provenance_problems += problems
    result.notes += notes
    result.manifest = manifest
    if (data_dir / 'pbp_metadata.json').exists():
        result.notes.append("pbp_metadata.json is not used as evidence: it records the seasons requested "
                            "and a run time, not what the files contain")
    try:
        games = load_games(data_dir)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        result.freshness_problems.append(f"schedule {SCHEDULE_FILE} could not be read: {exc}")
        return result
    problems, notes, feature_tables, summary = check_freshness(
        tables, games, season, week, prediction_matchups,
        source_completion=manifest['source']['completion'] if manifest else None, today=today)
    result.freshness_problems += problems
    result.notes += notes
    result.summary = summary
    result.feature_tables = feature_tables if result.ok else {}
    return result


def format_report(result: InputCheck) -> str:
    lines = [f"Player-prop input check for {result.season} week {result.week}"]
    if result.summary.get('feature_cutoff'):
        lines.append(f"  Feature source: completed games dated before {result.summary['feature_cutoff']} "
                     f"(the target week's first game)")
    if result.summary.get('expected_games'):
        lines.append("  Earlier games with recorded results expected in the stats: " + ", ".join(
            f"{s}: {n}" for s, n in result.summary['expected_games'].items()))
    m = result.manifest
    if m:
        src, cov = m['source'], m['source']['coverage']
        lines.append(f"  Stats written {m['written_at']} from {src['file']} read at {src['observed_at']} "
                     f"(sha256 {src['sha256'][:12]}, seasons {', '.join(cov['seasons']) or 'none'}, "
                     f"last game {cov['last_date']})")
    for title, items in (("PROVENANCE", result.provenance_problems),
                         ("FRESHNESS", result.freshness_problems)):
        lines.append(f"{title}: {'NOT ESTABLISHED' if items else 'ok'}")
        lines += [f"  - {p}" for p in items]
    if result.notes:
        lines.append("NOTES:")
        lines += [f"  - {n}" for n in result.notes]
    lines.append("RESULT: " + ("PASS" if result.ok else
                 "FAIL - player-prop predictions must not be generated from these inputs"))
    return "\n".join(lines)


def _default_target() -> Tuple[int, int]:
    """The week predict.py would target now (same schedule and rule)."""
    try:
        from player_props.predict import load_schedule
    except ImportError:
        from predict import load_schedule
    schedule = load_schedule()
    if schedule is None or schedule.empty:
        raise ValueError("no target week could be resolved from the prediction schedule")
    return int(schedule['season'].iloc[0]), int(schedule['week'].iloc[0])


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only player-prop freshness and provenance check")
    ap.add_argument('--season', type=int)
    ap.add_argument('--week', type=int)
    ap.add_argument('--data-dir', type=Path, default=None,
                    help="directory holding the stats, manifest and nfl_games_historical.csv "
                         "(default: data_files/)")
    args = ap.parse_args(argv)
    if (args.season is None) != (args.week is None):
        ap.error("give both --season and --week, or neither")
    if args.season is None and args.data_dir is not None:
        ap.error("--data-dir needs an explicit --season and --week")
    try:
        season, week = (args.season, args.week) if args.season is not None else _default_target()
    except (OSError, ValueError) as exc:
        print(f"Could not resolve the target week: {exc}")
        return 2
    try:
        result = check_inputs(season, week, args.data_dir or DATA_DIR)
    except Exception as exc:  # noqa: BLE001 - any failure means "not established"
        print(f"The input check could not complete ({type(exc).__name__}: {exc}); "
              f"freshness and provenance are not established.")
        return 2
    print(format_report(result))
    return 0 if result.ok else 1


if __name__ == '__main__':
    sys.exit(main())
