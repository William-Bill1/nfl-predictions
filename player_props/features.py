"""
Shared player-prop model features (Stage 3): one definition for training and
prediction.

* MODEL_FEATURES names the exact, ordered model inputs for each prop type.
* Training rows are the Stage 1 stats rows themselves (aggregators): each row's
  rolling stats and days_rest describe that game from the player's strictly
  earlier appearances, and is_home comes from its unique schedule game.
* ``inference_rows`` builds the same row for a game that hasn't been played:
  it appends the target game to the player's eligible history and runs the
  Stage 1 functions (calculate_rolling_averages, derive_days_rest,
  derive_is_home) on it, so a historical game reconstructed this way equals its
  stored training row.
* ``model_matrix`` turns rows into the exact model input and rejects missing,
  extra-named, duplicated or non-finite values.
* FEATURE_CONTRACT is the compatibility boundary: models record the contract
  they were trained under, and prediction refuses any model whose contract
  differs (see predict.load_models).

Policy: docs/architecture.md ("Stage 3 train/serve parity").
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from player_props.aggregators import (
        DAYS_REST_CAP, FEATURE_DEFINITION, IDENTITY_COLUMNS, RAW_COLUMNS, ROLLING_STATS,
        ROLLING_WINDOWS, FeatureValidationError, calculate_rolling_averages, derive_days_rest,
        derive_is_home,
    )
except ImportError:  # run as a script from inside player_props/
    from aggregators import (
        DAYS_REST_CAP, FEATURE_DEFINITION, IDENTITY_COLUMNS, RAW_COLUMNS, ROLLING_STATS,
        ROLLING_WINDOWS, FeatureValidationError, calculate_rolling_averages, derive_days_rest,
        derive_is_home,
    )

# Bump whenever MODEL_FEATURES, the training-row policy or inference_rows
# changes meaning; models trained under another version are refused.
FEATURE_SCHEMA_VERSION = 'props-features-1'
FEATURE_CONTRACT = {'schema_version': FEATURE_SCHEMA_VERSION,
                    'feature_definition': FEATURE_DEFINITION}

# prop_type -> (stat table, the stat the prop is on)
PROP_STAT = {
    'passing_yards': ('passing', 'passing_yards'),
    'passing_tds': ('passing', 'pass_tds'),
    'rushing_yards': ('rushing', 'rushing_yards'),
    'rushing_tds': ('rushing', 'rush_tds'),
    'receiving_yards': ('receiving', 'receiving_yards'),
    'receiving_tds': ('receiving', 'rec_tds'),
    'receptions': ('receiving', 'receptions'),
}
MATCHUP = ['is_home', 'days_rest']
# Exact, ordered model inputs: the names the committed models were trained on.
MODEL_FEATURES: Dict[str, List[str]] = {
    'passing_yards': ['passing_yards_L3', 'passing_yards_L5', 'passing_yards_L10',
                      'pass_tds_L3', 'pass_tds_L5', 'completions_L3', 'completions_L5',
                      'attempts_L3', 'attempts_L5'] + MATCHUP,
    'passing_tds': ['pass_tds_L3', 'pass_tds_L5', 'pass_tds_L10',
                    'completions_L3', 'completions_L5', 'completions_L10',
                    'attempts_L3', 'attempts_L5', 'attempts_L10'] + MATCHUP,
    'rushing_yards': ['rushing_yards_L3', 'rushing_yards_L5', 'rushing_yards_L10',
                      'rush_tds_L3', 'rush_tds_L5', 'rush_attempts_L3', 'rush_attempts_L5'] + MATCHUP,
    'rushing_tds': ['rush_tds_L3', 'rush_tds_L5', 'rush_tds_L10',
                    'rush_attempts_L3', 'rush_attempts_L5', 'rush_attempts_L10'] + MATCHUP,
    'receiving_yards': ['receiving_yards_L3', 'receiving_yards_L5', 'receiving_yards_L10',
                        'rec_tds_L3', 'rec_tds_L5', 'receptions_L3', 'receptions_L5',
                        'targets_L3', 'targets_L5'] + MATCHUP,
    'receiving_tds': ['rec_tds_L3', 'rec_tds_L5', 'rec_tds_L10',
                      'receptions_L3', 'receptions_L5', 'receptions_L10',
                      'targets_L3', 'targets_L5', 'targets_L10'] + MATCHUP,
    'receptions': ['receptions_L3', 'receptions_L5', 'receptions_L10',
                   'rec_tds_L3', 'rec_tds_L5', 'targets_L3', 'targets_L5'] + MATCHUP,
}
TARGET_COLUMNS = ['player_id', 'player_name', 'game_id', 'season', 'week', 'team', 'opponent',
                  'game_date']


class FeatureContractError(ValueError):
    """Model inputs can't be built or don't match the contract."""


def model_features(prop_type: str) -> List[str]:
    if prop_type not in MODEL_FEATURES:
        raise FeatureContractError(f"unknown prop type {prop_type!r}")
    return list(MODEL_FEATURES[prop_type])


def rolling_columns(prop_type: str) -> List[str]:
    return [c for c in model_features(prop_type) if c not in MATCHUP]


def contract_problems(recorded, features, prop_type: str) -> List[str]:
    """Why a model's recorded contract and feature list can't be used now.
    Both are required: the feature list must be an explicit list of unique
    names, in exactly MODEL_FEATURES order."""
    problems = []
    if not isinstance(recorded, dict):
        problems.append("no feature contract recorded (trained before Stage 3, or on stats "
                        "without a valid provenance manifest)")
    else:
        for key, want in FEATURE_CONTRACT.items():
            if recorded.get(key) != want:
                problems.append(f"{key} {recorded.get(key)!r} != current {want!r}")
    if features is None:
        problems.append("no feature list recorded")
    elif not isinstance(features, list) or not all(isinstance(f, str) for f in features):
        problems.append(f"recorded features must be a list of names, not {features!r}")
    elif len(set(features)) != len(features):
        problems.append(f"recorded features repeat names: {features}")
    elif features != model_features(prop_type):
        problems.append(f"features {features} != {model_features(prop_type)}")
    return problems


# ---------------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------------

def has_history(rows: pd.DataFrame, prop_type: str) -> pd.Series:
    """True where every rolling input exists - i.e. the player has an earlier
    appearance in this stat table. Rows without one are excluded from both
    training and prediction (there is nothing to average)."""
    cols = rolling_columns(prop_type)
    missing = [c for c in cols if c not in rows.columns]
    if missing:
        raise FeatureContractError(f"{prop_type}: missing feature column(s) {missing}")
    return rows[cols].notna().all(axis=1)


def model_matrix(rows: pd.DataFrame, prop_type: str) -> pd.DataFrame:
    """The exact model input: MODEL_FEATURES[prop_type] in order, all finite.

    Raises FeatureContractError for a missing or duplicated column, a value that
    isn't a finite number, is_home not 0/1, or days_rest outside 1-21. Callers
    exclude rows without history first (has_history); nothing is dropped here."""
    names = model_features(prop_type)
    if rows.columns.duplicated().any():
        dup = sorted(set(rows.columns[rows.columns.duplicated()]))
        raise FeatureContractError(f"{prop_type}: duplicated column(s) {dup}")
    missing = [c for c in names if c not in rows.columns]
    if missing:
        raise FeatureContractError(f"{prop_type}: missing feature column(s) {missing}")
    X = rows[names]
    try:
        values = X.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise FeatureContractError(f"{prop_type}: non-numeric feature values ({exc})") from exc
    if X.select_dtypes(include=['bool']).shape[1]:
        raise FeatureContractError(f"{prop_type}: boolean feature columns are not accepted")
    if not np.isfinite(values).all():
        bad = [c for c, ok in zip(names, np.isfinite(values).all(axis=0)) if not ok]
        raise FeatureContractError(f"{prop_type}: missing or non-finite values in {bad}")
    if not X['is_home'].isin([0, 1]).all():
        raise FeatureContractError(f"{prop_type}: is_home must be 0 or 1")
    rest = X['days_rest']
    if not (rest.between(1, DAYS_REST_CAP).all() and (rest % 1 == 0).all()):
        raise FeatureContractError(f"{prop_type}: days_rest must be a whole number from 1 to {DAYS_REST_CAP}")
    return X.astype(float)


def training_rows(table: pd.DataFrame, prop_type: str) -> Tuple[pd.DataFrame, dict]:
    """Stage 1 stats rows usable for training this prop type, and a report.

    Rows without an earlier appearance (no rolling history) are excluded and
    counted. A rolling value missing anywhere else means the table breaks the
    Stage 1 contract and raises."""
    keep = has_history(table, prop_type)
    first = table[~keep]
    partial = first[first[rolling_columns(prop_type)].notna().any(axis=1)]
    if len(partial):
        raise FeatureContractError(
            f"{prop_type}: {len(partial)} row(s) have some but not all rolling values; "
            f"the stats don't follow the Stage 1 feature definition")
    rows = table[keep]
    model_matrix(rows, prop_type)            # validates; raises on any bad value
    return rows, {'rows': int(len(table)), 'used': int(len(rows)),
                  'excluded_no_history': int((~keep).sum())}


# ---------------------------------------------------------------------------
# Inference rows
# ---------------------------------------------------------------------------

def _day(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format='%Y-%m-%d', errors='coerce')


def history_problems(history: pd.DataFrame, stat_type: str) -> List[str]:
    """Why the history can't be used as one row per player-game (empty =
    usable). Nothing is deduplicated or summed: a repeated row would be
    averaged twice, so it is a failure, not something to repair here.

    * the identity, date and raw stat columns must be present, identities and
      dates filled, and dates valid YYYY-MM-DD;
    * one row per (player_id, game_id);
    * every row of a game agrees on its season, week and date, the game has at
      most two teams, and each row's opponent is the other one;
    * a player has at most one game per date.
    Appearances on different teams and seasons are expected (trades, new
    seasons) and are kept."""
    raw = [c for c in RAW_COLUMNS[stat_type] if c != 'game_date']
    needed = IDENTITY_COLUMNS + ['game_date'] + raw
    missing = [c for c in needed if c not in history.columns]
    if missing:
        return [f"{stat_type}: history missing column(s) {missing}"]
    if history.empty:
        return []
    problems = []
    if history[IDENTITY_COLUMNS + ['game_date']].isna().any(axis=None):
        problems.append(f"{stat_type}: history has missing identity or date values")
        return problems
    day = _day(history['game_date'])
    if day.isna().any():
        problems.append(f"{stat_type}: history has invalid game dates")
        return problems
    dup = history.duplicated(['player_id', 'game_id'], keep=False)
    if dup.any():
        examples = sorted({f"{p}/{g}" for p, g in history.loc[dup, ['player_id', 'game_id']].itertuples(index=False)})
        problems.append(f"{stat_type}: history has {int(dup.sum())} rows repeating a player-game "
                        f"(e.g. {examples[:5]}); each player-game must appear once")
    labels = history.assign(_d=day).groupby('game_id')[['season', 'week', '_d']].nunique()
    conflicting = labels[(labels > 1).any(axis=1)].index
    if len(conflicting):
        problems.append(f"{stat_type}: history rows of a game disagree on season, week or date "
                        f"(e.g. {sorted(conflicting)[:5]})")
    named = pd.concat([history[['game_id', 'team']],
                       history[['game_id', 'opponent']].rename(columns={'opponent': 'team'})])
    sides = named.groupby('game_id')['team'].nunique()
    too_many = sides[sides != 2].index
    self_play = history[history['team'] == history['opponent']]['game_id'].unique()
    bad_sides = sorted(set(too_many) | set(self_play))
    if bad_sides:
        problems.append(f"{stat_type}: history rows of a game don't name the same two teams as "
                        f"team and opponent (e.g. {bad_sides[:5]})")
    two = history.assign(_d=day).groupby(['player_id', '_d'])['game_id'].nunique()
    if (two > 1).any():
        examples = [f"{p} on {d:%Y-%m-%d}" for p, d in two[two > 1].index[:5]]
        problems.append(f"{stat_type}: history puts a player in two games on one date (e.g. {examples})")
    return problems


def inference_rows(history: pd.DataFrame, targets: pd.DataFrame, stat_type: str,
                   schedule: pd.DataFrame, windows: List[int] = ROLLING_WINDOWS) -> pd.DataFrame:
    """Feature rows for upcoming games, built exactly as the stored training rows.

    history:  the stat table's eligible rows (for prediction, Stage 2's feature
              source). Only each player's rows dated strictly before that
              target's game date are used, from every team and season.
    targets:  one row per (player_id, game_id) with TARGET_COLUMNS; game_date
              is the target game's date (YYYY-MM-DD).
    schedule: game schedule (game_id, season, week, home_team, away_team);
              is_home needs the target's unique, consistent game.

    Returns the targets with every rolling column, is_home and days_rest, in
    the targets' order. Raises FeatureValidationError / FeatureContractError on
    bad input instead of guessing."""
    missing = [c for c in TARGET_COLUMNS if c not in targets.columns]
    if missing:
        raise FeatureContractError(f"targets missing column(s) {missing}")
    if targets.empty:
        return targets.copy()
    if targets.duplicated(['player_id', 'game_id']).any():
        raise FeatureContractError("targets list a player more than once for one game")
    if targets[TARGET_COLUMNS].isna().any(axis=None):
        raise FeatureContractError("targets have missing identity values")
    target_day = _day(targets['game_date'])
    if target_day.isna().any():
        raise FeatureContractError("targets have invalid game dates")
    if targets.assign(_d=target_day).duplicated(['player_id', '_d']).any():
        raise FeatureContractError("targets put a player in two games on one date")
    problems = history_problems(history, stat_type)
    if problems:
        raise FeatureValidationError(problems)
    hist_day = _day(history['game_date'])
    if history['game_id'].isin(set(targets['game_id'])).any():
        raise FeatureContractError("history already contains a target game's result")
    same_day = history[['player_id']].assign(_d=hist_day).merge(
        targets[['player_id']].assign(_d=target_day), on=['player_id', '_d'])
    if len(same_day):
        raise FeatureContractError("history has another game for a target player on the target date")

    raw = [c for c in RAW_COLUMNS[stat_type] if c != 'game_date']
    stats = ROLLING_STATS[stat_type]
    keep_cols = IDENTITY_COLUMNS + ['game_date'] + raw
    out = []
    # Each target date sees only history strictly before it.
    for day, group in targets.assign(_d=target_day, _order=range(len(targets))).groupby('_d'):
        players = set(group['player_id'])
        prior = history[(hist_day < day) & history['player_id'].isin(players)][keep_cols]
        frame = pd.concat([prior.assign(_target=False),
                           group[TARGET_COLUMNS + ['_order']].assign(_target=True)],
                          ignore_index=True)
        for c in raw:
            frame[c] = pd.to_numeric(frame[c], errors='coerce')   # targets have no result
        frame = calculate_rolling_averages(frame, stats, windows)
        frame['days_rest'] = derive_days_rest(frame)
        rows = frame[frame['_target']].copy()
        rows['is_home'] = derive_is_home(rows, schedule, stat_type)
        out.append(rows)
    result = pd.concat(out).sort_values('_order')
    rolling = [f'{s}_L{w}' for s in stats for w in windows]
    return result[TARGET_COLUMNS + rolling + MATCHUP].reset_index(drop=True)


def resolve_game(schedule: pd.DataFrame, season: int, week: int, home: str, away: str) -> pd.Series:
    """The unique schedule game for a matchup; anything else raises."""
    hit = schedule[(schedule['season'] == season) & (schedule['week'] == week)
                   & (schedule['home_team'] == home) & (schedule['away_team'] == away)]
    if len(hit) != 1:
        raise FeatureValidationError([
            f"{away}@{home} {season} week {week}: {len(hit)} matching schedule game(s); "
            f"is_home and the target date need exactly one"])
    row = hit.iloc[0]
    if not isinstance(row.get('gameday'), str) or _day(pd.Series([row['gameday']])).isna().any():
        raise FeatureValidationError([f"{row['game_id']}: schedule game has no valid date"])
    return row

