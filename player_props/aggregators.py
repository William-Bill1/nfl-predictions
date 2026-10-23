"""
Player stat aggregation functions.
Converts play-by-play data to game-level player statistics.

Usage:
    from player_props.aggregators import aggregate_passing_stats
    
    pbp = pd.read_csv('data_files/nfl_play_by_play_historical.csv.gz', compression='gzip', sep='\t')
    passing_stats = aggregate_passing_stats(pbp)

Persisted stats (build_player_stats -> write_player_stats) carry the matchup
features, are validated before writing, and are written atomically; see
docs/architecture.md ("Stage 1 feature generation"). With a provenance
manifest (player_props/provenance.py) the manifest is written in the same set.
"""
import json
import os
import shutil
import tempfile

import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data_files'
SCHEDULE_FILE = 'nfl_games_historical.csv'
STATS_FILES = {
    'passing': 'player_passing_stats.csv',
    'rushing': 'player_rushing_stats.csv',
    'receiving': 'player_receiving_stats.csv',
}
ROLLING_WINDOWS = [3, 5, 10]
DAYS_REST_FIRST = 7     # first-ever appearance in the stat table
DAYS_REST_CAP = 21

# Persisted column contract, matching the stats CSVs downstream code reads
# (minus the removed opponent_def_rank). Columns computed but not listed here
# (receiving target_share) are not written, so a later retrain cannot pick
# them up silently; changing the schema is the train/serve parity stage.
IDENTITY_COLUMNS = ['player_id', 'player_name', 'game_id', 'season', 'week', 'team', 'opponent']
RAW_COLUMNS = {
    'passing': ['passing_yards', 'pass_tds', 'completions', 'attempts', 'interceptions',
                'game_date', 'completion_pct'],
    'rushing': ['rushing_yards', 'rush_tds', 'rush_attempts', 'game_date', 'ypc'],
    'receiving': ['receiving_yards', 'receptions', 'rec_tds', 'targets', 'game_date', 'ypr',
                  'catch_rate'],
}
ROLLING_STATS = {
    'passing': ['passing_yards', 'pass_tds', 'completions', 'attempts', 'interceptions'],
    'rushing': ['rushing_yards', 'rush_tds', 'rush_attempts', 'ypc'],
    'receiving': ['receiving_yards', 'receptions', 'rec_tds', 'targets'],
}
MATCHUP_COLUMNS = ['is_home', 'days_rest']
MANIFEST_FILE = 'player_props_stats_manifest.json'
# Names how the persisted features are computed (the rolling, days_rest and
# is_home definitions below). It is recorded in the provenance manifest; change
# it whenever a feature's meaning changes, so stats built under the old
# definition stop validating.
FEATURE_DEFINITION = 'stage1-v1'


class FeatureValidationError(ValueError):
    """Stats cannot be built or written safely; nothing is written. `problems`
    lists each failed check with examples."""

    def __init__(self, problems: List[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def read_stats_csv(source) -> pd.DataFrame:
    """Read a stats CSV with exact float round-tripping. pandas' default parser
    can be off by one unit in the last place, which would make a stored
    feature differ from the same value recomputed at prediction time."""
    return pd.read_csv(source, float_precision='round_trip')


def output_columns(stat_type: str, windows: List[int] = ROLLING_WINDOWS) -> List[str]:
    """The exact columns written for a stat table, in order."""
    rolling = [f'{s}_L{w}' for s in ROLLING_STATS[stat_type] for w in windows]
    return IDENTITY_COLUMNS + RAW_COLUMNS[stat_type] + rolling + MATCHUP_COLUMNS


def _ratio(numerator: pd.Series, denominator: pd.Series, scale: float = 1.0) -> pd.Series:
    """numerator / denominator, NaN (never inf) when the denominator is 0."""
    return (numerator / denominator.replace(0, np.nan) * scale).round(1)


def _canonical_play_labels(plays: pd.DataFrame, id_col: str, name_col: str,
                           stat_type: str) -> pd.DataFrame:
    """Make every play of one player in one game carry the same labels, so the
    game aggregates to exactly one row and no play is counted twice.

    * Plays missing a grouping label (game, season, week, team, opponent or
      date) are excluded and counted in ``attrs['excluded_plays']``.
    * A player listed for two teams, two opponents or two dates in one game is
      contradictory and raises FeatureValidationError.
    * Name spellings that differ within one game are unified to the most
      frequent spelling in that game (ties: alphabetical); every play is still
      summed. Each case is recorded in ``attrs['name_merges']``.
    """
    labels = ['game_id', 'season', 'week', 'posteam', 'defteam', 'game_date']
    missing = plays[labels].isna().any(axis=1)
    excluded = int(missing.sum())
    plays = plays[~missing].copy()

    keys = [id_col, 'game_id']
    spread = plays.groupby(keys, observed=True)[['posteam', 'defteam', 'game_date', 'season', 'week']].nunique()
    conflicts = spread[(spread > 1).any(axis=1)]
    if not conflicts.empty:
        examples = ", ".join(f"{pid}/{gid}" for pid, gid in conflicts.index[:5])
        raise FeatureValidationError([
            f"{stat_type}: {len(conflicts)} player-game(s) list more than one team, opponent, "
            f"date, season or week (e.g. {examples})"])

    names = plays[name_col].astype('string')
    counts = (plays.assign(_name=names).dropna(subset=['_name'])
              .groupby(keys + ['_name'], observed=True).size().reset_index(name='n')
              .sort_values(keys + ['n', '_name'], ascending=[True, True, False, True]))
    canonical = counts.drop_duplicates(keys).set_index(keys)['_name']
    # A game with no name at all takes the player's most frequent name elsewhere.
    overall = (counts.groupby([id_col, '_name'], observed=True)['n'].sum().reset_index()
               .sort_values([id_col, 'n', '_name'], ascending=[True, False, True])
               .drop_duplicates(id_col).set_index(id_col)['_name'])
    key_index = pd.MultiIndex.from_frame(plays[keys])
    resolved = pd.Series(canonical.reindex(key_index).to_numpy(), index=plays.index)
    resolved = resolved.fillna(plays[id_col].map(overall))
    if resolved.isna().any():
        bad = plays.loc[resolved.isna(), id_col].unique()[:5]
        raise FeatureValidationError([f"{stat_type}: player(s) with no name in any play: {list(bad)}"])

    multi = counts.groupby(keys, observed=True)['_name'].nunique()
    merges = multi[multi > 1]
    plays[name_col] = resolved.astype(object)
    plays.attrs = {'excluded_plays': excluded,
                   'name_merges': [{'player_id': pid, 'game_id': gid} for pid, gid in merges.index]}
    return plays


def aggregate_passing_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate passing stats by player and game.
    
    Args:
        pbp: Play-by-play DataFrame
    
    Returns:
        DataFrame with columns: 
            [player_id, player_name, game_id, season, week, game_date,
             team, opponent, passing_yards, pass_tds, completions, 
             attempts, interceptions, completion_pct]
    """
    # Filter to passing plays only
    passing_plays = pbp[pbp['passer_player_id'].notna()].copy()
    
    if passing_plays.empty:
        print("⚠️  No passing plays found in dataset")
        return pd.DataFrame()
    passing_plays = _canonical_play_labels(passing_plays, 'passer_player_id',
                                           'passer_player_name', 'passing')
    notes = dict(passing_plays.attrs)

    # Aggregate by player and game
    agg_stats = passing_plays.groupby([
        'passer_player_id', 
        'passer_player_name', 
        'game_id', 
        'season', 
        'week',
        'posteam',  # Team the player is on
        'defteam'   # Opponent
    ], observed=True).agg({
        'passing_yards': 'sum',
        'pass_touchdown': 'sum',
        'complete_pass': 'sum',
        'pass_attempt': 'sum',
        'interception': 'sum',
        'game_date': 'first'  # Get game date
    }).reset_index()
    
    # Rename columns
    agg_stats.columns = [
        'player_id', 'player_name', 'game_id', 'season', 'week',
        'team', 'opponent', 'passing_yards', 'pass_tds', 
        'completions', 'attempts', 'interceptions', 'game_date'
    ]
    
    # Calculate completion percentage (NaN, not inf, with no attempts)
    agg_stats['completion_pct'] = _ratio(agg_stats['completions'], agg_stats['attempts'], 100)

    # Sort by date
    agg_stats = agg_stats.sort_values(['player_id', 'season', 'week'])
    agg_stats.attrs = notes

    print(f"✅ Aggregated {len(agg_stats):,} player-game passing records")
    print(f"   Players: {agg_stats['player_id'].nunique():,}")
    print(f"   Games: {agg_stats['game_id'].nunique():,}")
    
    return agg_stats


def aggregate_rushing_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate rushing stats by player and game.
    
    Returns:
        DataFrame with columns:
            [player_id, player_name, game_id, season, week, game_date,
             team, opponent, rushing_yards, rush_tds, rush_attempts, ypc]
    """
    rushing_plays = pbp[pbp['rusher_player_id'].notna()].copy()
    
    if rushing_plays.empty:
        print("⚠️  No rushing plays found in dataset")
        return pd.DataFrame()
    rushing_plays = _canonical_play_labels(rushing_plays, 'rusher_player_id',
                                           'rusher_player_name', 'rushing')
    notes = dict(rushing_plays.attrs)

    agg_stats = rushing_plays.groupby([
        'rusher_player_id',
        'rusher_player_name',
        'game_id',
        'season',
        'week',
        'posteam',
        'defteam'
    ], observed=True).agg({
        'rushing_yards': 'sum',
        'rush_touchdown': 'sum',
        'rush_attempt': 'sum',
        'game_date': 'first'
    }).reset_index()
    
    agg_stats.columns = [
        'player_id', 'player_name', 'game_id', 'season', 'week',
        'team', 'opponent', 'rushing_yards', 'rush_tds', 
        'rush_attempts', 'game_date'
    ]
    
    # Calculate yards per carry (NaN, not inf, with no attempts)
    agg_stats['ypc'] = _ratio(agg_stats['rushing_yards'], agg_stats['rush_attempts'])

    agg_stats = agg_stats.sort_values(['player_id', 'season', 'week'])
    agg_stats.attrs = notes

    print(f"✅ Aggregated {len(agg_stats):,} player-game rushing records")
    print(f"   Players: {agg_stats['player_id'].nunique():,}")
    
    return agg_stats


def aggregate_receiving_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate receiving stats by player and game.
    
    Returns:
        DataFrame with columns:
            [player_id, player_name, game_id, season, week, game_date,
             team, opponent, receiving_yards, receptions, rec_tds, 
             targets, ypr, catch_rate]
    """
    receiving_plays = pbp[pbp['receiver_player_id'].notna()].copy()
    
    if receiving_plays.empty:
        print("⚠️  No receiving plays found in dataset")
        return pd.DataFrame()
    receiving_plays = _canonical_play_labels(receiving_plays, 'receiver_player_id',
                                             'receiver_player_name', 'receiving')
    notes = dict(receiving_plays.attrs)

    # Count targets (all passes to this receiver, caught or not)
    receiving_plays['target'] = 1

    # Compute team-level total targets per game for target_share calculation
    team_game_targets = (
        receiving_plays.groupby(['game_id', 'posteam'], observed=True)['pass_attempt']
        .transform('sum')
    )
    receiving_plays['target_share'] = (
        receiving_plays['pass_attempt'] / team_game_targets.replace(0, np.nan)
    )

    agg_stats = receiving_plays.groupby([
        'receiver_player_id',
        'receiver_player_name',
        'game_id',
        'season',
        'week',
        'posteam',
        'defteam'
    ], observed=True).agg({
        'receiving_yards': 'sum',
        'complete_pass': 'sum',  # Receptions = completed passes
        'pass_touchdown': 'sum',  # Receiving TDs
        'pass_attempt': 'sum',    # Targets
        'target_share': 'mean',   # Player's share of team targets in this game
        'game_date': 'first'
    }).reset_index()
    
    agg_stats.columns = [
        'player_id', 'player_name', 'game_id', 'season', 'week',
        'team', 'opponent', 'receiving_yards', 'receptions',
        'rec_tds', 'targets', 'target_share', 'game_date'
    ]
    
    # Calculate yards per reception and catch rate (NaN, not inf, on a 0 denominator)
    agg_stats['ypr'] = _ratio(agg_stats['receiving_yards'], agg_stats['receptions'])
    agg_stats['catch_rate'] = _ratio(agg_stats['receptions'], agg_stats['targets'], 100)

    agg_stats = agg_stats.sort_values(['player_id', 'season', 'week'])
    agg_stats.attrs = notes

    print(f"✅ Aggregated {len(agg_stats):,} player-game receiving records")
    print(f"   Players: {agg_stats['player_id'].nunique():,}")
    
    return agg_stats


def _chronological(df: pd.DataFrame, player_id_col: str = 'player_id') -> pd.DataFrame:
    """Rows ordered per player by calendar game date, then game_id."""
    order = pd.to_datetime(df['game_date'], errors='coerce')
    return (df.assign(_order_date=order)
              .sort_values([player_id_col, '_order_date', 'game_id'], kind='mergesort')
              .drop(columns='_order_date'))


def calculate_rolling_averages(
    player_stats: pd.DataFrame,
    stat_cols: List[str],
    windows: List[int] = [3, 5, 10],
    player_id_col: str = 'player_id'
) -> pd.DataFrame:
    """
    Calculate exponentially-weighted rolling averages for stat columns.
    More recent games get higher weight for better prediction accuracy.

    Each value uses only the player's appearances strictly before that game
    (shift(1) in calendar-date order), so a game's own result never enters its
    features and later games never change earlier rows. Ordering is by
    game_date, not season/week, so history carries across seasons and teams.

    Args:
        player_stats: Game-level player stats (must have player_id, game_date, game_id)
        stat_cols: List of stat columns to calculate rolling averages for
        windows: List of window sizes (e.g., [3, 5, 10] for L3, L5, L10)
        player_id_col: Column name for player ID

    Returns:
        DataFrame with additional columns for exponentially-weighted rolling averages
        (e.g., passing_yards_L3, passing_yards_L5)
    """
    player_stats = _chronological(player_stats.copy(), player_id_col)

    for stat_col in stat_cols:
        if stat_col not in player_stats.columns:
            print(f"⚠️  Column '{stat_col}' not found, skipping")
            continue

        for window in windows:
            col_name = f'{stat_col}_L{window}'
            # Exponentially-weighted average of earlier games only (shift(1)).
            player_stats[col_name] = player_stats.groupby(player_id_col, observed=False)[stat_col].transform(
                lambda x: x.shift(1).ewm(span=window, adjust=False).mean()
            )

    print(f"✅ Calculated exponentially-weighted rolling averages: windows={windows}")

    return player_stats


def load_matchup_schedule(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """The schedule used for is_home (game_id, season, week, home_team, away_team).
    A missing or unreadable file is a validation failure, never a default."""
    path = Path(data_dir) / SCHEDULE_FILE
    try:
        return pd.read_csv(path, sep='\t', low_memory=False,
                           usecols=['game_id', 'season', 'week', 'home_team', 'away_team'])
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise FeatureValidationError([f"schedule {path.name} could not be read: {exc}"]) from exc


def derive_is_home(df: pd.DataFrame, schedule: pd.DataFrame, stat_type: str) -> pd.Series:
    """1 if the row's team is the home team of its schedule game, else 0.

    Each row must match exactly one schedule game_id, its season and week must
    agree, its team must be that game's home or away team, and its opponent the
    other one. Anything else raises FeatureValidationError; no value is guessed.
    """
    problems = []
    sched = schedule[['game_id', 'season', 'week', 'home_team', 'away_team']].copy()
    dup = sched['game_id'][sched['game_id'].duplicated(keep=False)].unique()
    if len(dup):
        problems.append(f"schedule has {len(dup)} game_id(s) on more than one row (e.g. {list(dup[:5])})")
    if sched[['game_id', 'home_team', 'away_team']].isna().any(axis=None):
        problems.append("schedule has rows missing game_id, home_team or away_team")
    if problems:
        raise FeatureValidationError([f"{stat_type}: {p}" for p in problems])

    m = df[['game_id', 'season', 'week', 'team', 'opponent']].merge(
        sched, on='game_id', how='left', suffixes=('', '_sched'), validate='many_to_one')
    missing = m['home_team'].isna()
    if missing.any():
        problems.append(f"{int(missing.sum())} row(s) have no schedule game "
                        f"(e.g. {list(m.loc[missing, 'game_id'].unique()[:5])})")
    present = ~missing
    wrong_when = present & ((m['season'] != m['season_sched']) | (m['week'] != m['week_sched']))
    if wrong_when.any():
        problems.append(f"{int(wrong_when.sum())} row(s) disagree with the schedule's season/week "
                        f"(e.g. {list(m.loc[wrong_when, 'game_id'].unique()[:5])})")
    home = m['team'] == m['home_team']
    away = m['team'] == m['away_team']
    wrong_team = present & ~(home ^ away)
    wrong_opp = present & ((home & (m['opponent'] != m['away_team'])) |
                           (away & (m['opponent'] != m['home_team'])))
    if wrong_team.any():
        problems.append(f"{int(wrong_team.sum())} row(s) list a team that is not exactly one side of "
                        f"the schedule game (e.g. {list(m.loc[wrong_team, 'game_id'].unique()[:5])})")
    if wrong_opp.any():
        problems.append(f"{int(wrong_opp.sum())} row(s) list an opponent that is not the other side "
                        f"of the schedule game (e.g. {list(m.loc[wrong_opp, 'game_id'].unique()[:5])})")
    if problems:
        raise FeatureValidationError([f"{stat_type}: {p}" for p in problems])
    return pd.Series(home.astype('int8').to_numpy(), index=df.index, name='is_home')


def derive_days_rest(df: pd.DataFrame) -> pd.Series:
    """Calendar days since the player's previous appearance in this stat table,
    capped at DAYS_REST_CAP; a first-ever appearance is DAYS_REST_FIRST.

    History carries across seasons and teams (grouped by player_id only). This
    measures gaps between recorded appearances, not physiological rest: a game
    without a row in this table (no targets, no carries) lengthens the gap.
    """
    dates = pd.to_datetime(df['game_date'], errors='coerce').dt.normalize()
    ordered = _chronological(df.assign(_d=dates))
    gap = (ordered['_d'] - ordered.groupby('player_id')['_d'].shift(1)).dt.days
    rest = gap.clip(upper=DAYS_REST_CAP).fillna(DAYS_REST_FIRST).astype('int16')
    return rest.reindex(df.index).rename('days_rest')


def add_matchup_features(df: pd.DataFrame, stat_type: str,
                         all_stats: Optional[Dict[str, pd.DataFrame]] = None,
                         schedule: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Add matchup and situational features to player stats DataFrame.

    - is_home: from the row's unique schedule game (derive_is_home); missing or
      conflicting schedule evidence raises FeatureValidationError.
    - days_rest: days since the player's previous appearance in this table
      (derive_days_rest): capped at 21, 7 for a first-ever appearance.

    (opponent_def_rank was removed - its implementation averaged a stat over the
    whole dataset, which both leaked future games and clipped to a constant.
    all_stats is accepted for backward compatibility and unused.)

    Args:
        df: Player stats DataFrame (must include game_id, team, opponent, season, week, game_date)
        stat_type: 'passing', 'rushing', or 'receiving'
        schedule: game schedule; loaded from data_files/nfl_games_historical.csv if omitted

    Returns:
        DataFrame with added matchup features
    """
    df = df.copy()
    if schedule is None:
        schedule = load_matchup_schedule()
    df['is_home'] = derive_is_home(df, schedule, stat_type)
    df['days_rest'] = derive_days_rest(df)
    return df


def build_player_stats(
    pbp: pd.DataFrame,
    schedule: pd.DataFrame,
    windows: List[int] = ROLLING_WINDOWS,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, dict]]:
    """Aggregate, roll and add matchup features for all three stat tables.

    Returns ({stat_type: table with exactly output_columns()}, {stat_type: report}).
    Raises FeatureValidationError instead of guessing on bad input."""
    builders = {'passing': aggregate_passing_stats, 'rushing': aggregate_rushing_stats,
                'receiving': aggregate_receiving_stats}
    tables, reports = {}, {}
    for stat_type, aggregate in builders.items():
        df = aggregate(pbp)
        if df.empty:
            raise FeatureValidationError([f"{stat_type}: no plays to aggregate"])
        notes = dict(df.attrs)
        day = pd.to_datetime(df['game_date'], errors='coerce').dt.normalize()
        same_day = df.assign(_d=day).duplicated(['player_id', '_d'], keep=False)
        if same_day.any():
            raise FeatureValidationError([
                f"{stat_type}: {int(same_day.sum())} row(s) put a player in two games on one date, "
                f"so 'strictly before' is undefined"])
        df = calculate_rolling_averages(df, ROLLING_STATS[stat_type], windows)
        df = add_matchup_features(df, stat_type, schedule=schedule)
        df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        tables[stat_type] = df[output_columns(stat_type, windows)].reset_index(drop=True)
        reports[stat_type] = {'excluded_plays': notes.get('excluded_plays', 0),
                              'name_merges': len(notes.get('name_merges', []))}
    return tables, reports


def validate_player_stats(df: pd.DataFrame, stat_type: str,
                          windows: List[int] = ROLLING_WINDOWS) -> List[str]:
    """Problems that must block writing (an empty list means valid)."""
    problems = []
    expected = output_columns(stat_type, windows)
    if list(df.columns) != expected:
        missing = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected]
        return [f"{stat_type}: columns differ from the contract (missing {missing}, extra {extra})"]
    if df.empty:
        return [f"{stat_type}: no rows"]
    if df[IDENTITY_COLUMNS].isna().any(axis=None):
        problems.append(f"{stat_type}: identity columns have missing values")
    dup = df.duplicated(['player_id', 'game_id']).sum()
    if dup:
        problems.append(f"{stat_type}: {int(dup)} duplicate player/game row(s)")
    dates = pd.to_datetime(df['game_date'], format='%Y-%m-%d', errors='coerce')
    if dates.isna().any():
        problems.append(f"{stat_type}: {int(dates.isna().sum())} invalid game_date value(s)")
    numeric = [c for c in expected if c not in IDENTITY_COLUMNS and c != 'game_date']
    for c in numeric:
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            problems.append(f"{stat_type}: {c} is not numeric")
    if problems:
        return problems
    if np.isinf(df[numeric].to_numpy(dtype=float)).any():
        problems.append(f"{stat_type}: infinite values present")
    counts = [c for c in ROLLING_STATS[stat_type] if c != 'ypc']
    if df[counts].isna().any(axis=None):
        problems.append(f"{stat_type}: counting stats have missing values")
    if not df['is_home'].isin([0, 1]).all():
        problems.append(f"{stat_type}: is_home must be 0 or 1")
    rest = df['days_rest']
    if rest.isna().any() or not rest.between(1, DAYS_REST_CAP).all() or not (rest % 1 == 0).all():
        problems.append(f"{stat_type}: days_rest must be a whole number from 1 to {DAYS_REST_CAP}")
    # A rolling value may be missing only when the player has no earlier value of that stat.
    ordered = _chronological(df)
    for stat in ROLLING_STATS[stat_type]:
        seen_before = ordered.groupby('player_id')[stat].transform(
            lambda s: s.notna().shift(1, fill_value=False).cumsum()) > 0
        for w in windows:
            col = f'{stat}_L{w}'
            bad = ordered[col].isna() & seen_before
            if bad.any():
                problems.append(f"{stat_type}: {col} missing for {int(bad.sum())} row(s) with earlier history")
            unexpected = ordered[col].notna() & ~seen_before
            if unexpected.any():
                problems.append(f"{stat_type}: {col} present for {int(unexpected.sum())} row(s) "
                                f"with no earlier appearance")
    return problems


class PersistenceRollbackError(RuntimeError):
    """A replacement failed and restoring the previous set of stats files also
    failed. The backups in ``recovery_files`` (destination -> backup) are the
    only copies of the previous files and are left on disk."""

    def __init__(self, message: str, recovery_files: Dict[Path, Path], unrestored: List[Path]):
        self.recovery_files = dict(recovery_files)
        self.unrestored = list(unrestored)
        super().__init__(message)


def write_player_stats(tables: Dict[str, pd.DataFrame], out_dir: Path | str,
                       windows: List[int] = ROLLING_WINDOWS,
                       manifest_for: Optional[Callable[[Dict[str, Path]], dict]] = None
                       ) -> Dict[str, Path]:
    """Validate every table, then replace the three stats files as a set.

    ``manifest_for`` (see provenance.stats_manifest) is called with the
    temporary stats files once they are written; the manifest it returns is
    written as MANIFEST_FILE and replaced in the same set, so it is restored
    or removed with them. Without it no manifest is written, and an existing
    one no longer matches the new files.

    1. Every table is validated first; any problem raises
       FeatureValidationError and nothing is written.
    2. Each table is written to a temporary file in ``out_dir``, and each
       existing destination is copied to a ``.bak`` file beside it.
    3. Each destination is replaced atomically on its own (os.replace).
    4. If a replacement raises, the previous set is restored: replaced files
       get their backup back, and destinations that did not exist before are
       removed. The original error is then re-raised.
    5. If that restore itself fails, PersistenceRollbackError is raised and
       every backup is kept, since it may be the only copy of the old data.

    This covers errors Python can catch. It is not a transaction across the
    three files: a crash or power loss between two replacements can leave a
    mixed set, with the ``.bak`` files left behind for manual recovery."""
    missing = [s for s in STATS_FILES if s not in tables]
    problems = [f"missing table(s): {missing}"] if missing else []
    for stat_type in STATS_FILES:
        if stat_type in tables:
            problems += validate_player_stats(tables[stat_type], stat_type, windows)
    if problems:
        raise FeatureValidationError(problems)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = dict(STATS_FILES)
    if manifest_for is not None:
        names['manifest'] = MANIFEST_FILE
    dests = {s: out_dir / name for s, name in names.items()}
    temps: Dict[str, Path] = {}
    backups: Dict[Path, Path] = {}       # destination -> backup of its previous content
    keep_backups = False
    try:
        for stat_type, name in STATS_FILES.items():
            fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=f'.{name}.', suffix='.tmp')
            os.close(fd)
            temps[stat_type] = Path(tmp)
            tables[stat_type].to_csv(tmp, index=False)
        if manifest_for is not None:
            manifest = manifest_for({s: temps[s] for s in STATS_FILES})
            fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=f'.{MANIFEST_FILE}.', suffix='.tmp')
            os.close(fd)
            temps['manifest'] = Path(tmp)
            Path(tmp).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        for stat_type, dest in dests.items():
            if dest.exists():
                fd, bak = tempfile.mkstemp(dir=out_dir, prefix=f'.{dest.name}.', suffix='.bak')
                os.close(fd)
                backups[dest] = Path(bak)      # registered first so a failed copy is cleaned up
                shutil.copy2(dest, bak)

        replaced: List[Path] = []
        try:
            for stat_type, dest in dests.items():
                os.replace(temps[stat_type], dest)
                replaced.append(dest)
        except Exception as exc:
            unrestored = []
            for dest in reversed(replaced):
                try:
                    if dest in backups:
                        os.replace(backups[dest], dest)
                        del backups[dest]
                    else:
                        dest.unlink()
                except Exception:
                    unrestored.append(dest)
            if unrestored:
                keep_backups = True
                kept = {d: b for d, b in backups.items() if b.exists()}
                raise PersistenceRollbackError(
                    f"replacing the stats files failed ({exc!r}) and restoring the previous set "
                    f"also failed for {', '.join(d.name for d in unrestored)}. Previous copies "
                    f"are kept at: {', '.join(str(b) for b in kept.values()) or 'none'}",
                    kept, unrestored) from exc
            raise
        return dict(dests)
    finally:
        for tmp in temps.values():
            if tmp.exists():
                tmp.unlink()
        if not keep_backups:
            for bak in backups.values():
                if bak.exists():
                    bak.unlink()


def aggregate_all_stats(
    pbp: pd.DataFrame,
    rolling_windows: List[int] = [3, 5, 10],
    save_to_dir: Optional[Path] = None,
    schedule: Optional[pd.DataFrame] = None,
    source: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aggregate all player stats, calculate rolling averages and matchup features.

    Args:
        pbp: Play-by-play DataFrame
        rolling_windows: Window sizes for rolling averages
        save_to_dir: If provided, validate and atomically save the stats there
        schedule: game schedule for is_home (default: data_files/nfl_games_historical.csv)
        source: observation of the play-by-play (provenance.read_source); when
            saving, a provenance manifest is written with the stats

    Returns:
        Tuple of (passing_stats, rushing_stats, receiving_stats)
    """
    print("\n🏈 Aggregating Player Stats...")
    print("=" * 60)
    if schedule is None:
        schedule = load_matchup_schedule()
    tables, reports = build_player_stats(pbp, schedule, rolling_windows)
    for stat_type, report in reports.items():
        print(f"   {stat_type}: {len(tables[stat_type]):,} rows, "
              f"{report['excluded_plays']} plays excluded, {report['name_merges']} name merges")

    if save_to_dir:
        manifest_for = None
        if source is not None:
            from player_props.provenance import stats_manifest
            manifest_for = stats_manifest(source, rolling_windows)
        written = write_player_stats(tables, save_to_dir, rolling_windows, manifest_for)
        for stat_type in STATS_FILES:
            print(f"💾 Saved {written[stat_type].name} ({len(tables[stat_type]):,} rows)")
        if 'manifest' in written:
            print(f"💾 Saved {written['manifest'].name}")
        else:
            print("⚠️  No provenance manifest written (no source observation); the stats are unverified")

    print("\n✅ All stats aggregated successfully!")

    return tables['passing'], tables['rushing'], tables['receiving']


# CLI script for running aggregation
if __name__ == '__main__':
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from player_props.provenance import read_source

    print("🏈 NFL Player Stats Aggregator")
    print("=" * 60)
    
    # Load play-by-play data
    DATA_DIR = Path('data_files')
    pbp_file = DATA_DIR / 'nfl_play_by_play_historical.csv.gz'
    
    if not pbp_file.exists():
        print(f"\n❌ File not found: {pbp_file}")
        print("Please ensure nfl_play_by_play_historical.csv.gz exists")
        sys.exit(1)
    
    print(f"\n📂 Loading {pbp_file.name}...")
    pbp, source = read_source(pbp_file)
    print(f"✅ Loaded {len(pbp):,} plays from {pbp['season'].min()}-{pbp['season'].max()}")

    # Aggregate all stats
    passing, rushing, receiving = aggregate_all_stats(
        pbp,
        rolling_windows=[3, 5, 10],
        save_to_dir=DATA_DIR,
        source=source,
    )
    
    print("\n" + "=" * 60)
    print("✅ Aggregation complete! Ready for model training.")
    print("\nNext steps:")
    print("  1. Review generated CSV files in data_files/")
    print("  2. Run player_props/models.py to train baseline models")
