"""
Player Props Prediction Pipeline
Generate prop predictions for upcoming NFL games using trained XGBoost models.
"""
import argparse
import os
import sys
from functools import lru_cache
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings('ignore')

try:
    from season_utils import upcoming_or_current_season
except ImportError:  # run from inside player_props/
    import sys as _sys
    _sys.path.append(str(Path(__file__).parent.parent))
    from season_utils import upcoming_or_current_season

# Import injury functions
try:
    from .injuries import get_injury_report, find_player_injury, adjust_prediction_for_injury
    from .weather import get_weather_for_game, adjust_for_weather
    from .market_odds import fetch_market_odds, find_market_line, attach_market_odds
    from .provenance import check_inputs, format_report, load_games
    from .aggregators import FeatureValidationError
    from .features import (
        PROP_STAT, TARGET_COLUMNS, FeatureContractError, contract_problems, has_history,
        inference_rows, model_features, model_matrix, resolve_game,
    )
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from injuries import get_injury_report, find_player_injury, adjust_prediction_for_injury
    from weather import get_weather_for_game, adjust_for_weather
    from market_odds import fetch_market_odds, find_market_line, attach_market_odds
    from provenance import check_inputs, format_report, load_games
    from aggregators import FeatureValidationError
    from features import (
        PROP_STAT, TARGET_COLUMNS, FeatureContractError, contract_problems, has_history,
        inference_rows, model_features, model_matrix, resolve_game,
    )

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(__file__).parent.parent / 'data_files'
MODELS_DIR = Path(__file__).parent / 'models'


class _SkipWeather(Exception):
    """Internal sentinel to bypass the weather-adjustment block."""

# Prop lines matching training configuration - UPDATED TO REALISTIC VALUES
PROP_LINES = {
    'passing_yards': {
        'elite_qb': 275.0,   # For QBs averaging 280+ ypg (Stafford, Jones, Goff)
        'star_qb': 250.0,    # For QBs averaging 250-280 ypg (Prescott, Mahomes)
        'good_qb': 225.0,    # For QBs averaging 220-250 ypg (Lawrence, Herbert)
        'starter': 200.0     # For QBs averaging <220 ypg
    },
    'rushing_yards': {
        'elite_rb': 75.0,    # For RBs averaging 80+ ypg
        'star_rb': 55.0,     # For RBs averaging 60-80 ypg
        'good_rb': 40.0,     # For RBs averaging 45-60 ypg
        'starter': 24.0      # For RBs averaging <45 ypg
    },
    'receiving_yards': {
        'elite_wr': 70.0,    # For WRs averaging 75+ ypg
        'star_wr': 50.0,     # For WRs averaging 55-75 ypg
        'good_wr': 35.0,     # For WRs averaging 40-55 ypg
        'starter': 27.0      # For WRs averaging <40 ypg
    },
    'passing_tds': {
        'elite_qb': 2.5,     # Elite QBs: Over 2.5 TDs (will they throw 3+?)
        'star_qb': 1.5,      # Star QBs: Over 1.5 TDs (will they throw 2+?)
        'good_qb': 1.5,      # Good QBs: Over 1.5 TDs (will they throw 2+?)
        'starter': 0.5       # Starter QBs: Over 0.5 TDs (will they throw at least 1?)
    },
    'rushing_tds': {
        'elite_rb': 1.5,     # Elite RBs: Over 1.5 TDs (will they score 2+?)
        'star_rb': 0.5,      # Star RBs: Over 0.5 TDs (will they score at least 1?)
        'good_rb': 0.5,      # Good RBs: Over 0.5 TDs (will they score at least 1?)
        'anytime': 0.5       # Backup RBs: Over 0.5 TDs (will they score at least 1?)
    },
    'receiving_tds': {
        'elite_wr': 1.5,     # Elite WRs: Over 1.5 TDs (will they score 2+?)
        'star_wr': 0.5,      # Star WRs: Over 0.5 TDs (will they score at least 1?)
        'good_wr': 0.5,      # Good WRs: Over 0.5 TDs (will they score at least 1?)
        'anytime': 0.5       # Role WRs: Over 0.5 TDs (will they score at least 1?)
    },
    'receptions': {
        'elite_wr': 7.5,     # Elite WRs: Over 7.5 receptions
        'star_wr': 5.5,      # Star WRs: Over 5.5 receptions
        'good_wr': 4.5,      # Good WRs: Over 4.5 receptions
        'starter': 3.5       # Role players: Over 3.5 receptions
    }
}

# Minimum confidence threshold for recommendations
MIN_CONFIDENCE = 0.55  # 55% probability for a recommendation

# Position mapping for prop types
POSITION_MAP = {
    'QB': ['passing_yards', 'passing_tds'],
    'RB': ['rushing_yards', 'rushing_tds', 'receptions'],
    'WR': ['receiving_yards', 'receiving_tds', 'receptions'],
    'TE': ['receiving_yards', 'receiving_tds', 'receptions']
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_players():
    """Load player metadata with full names."""
    players_path = DATA_DIR / 'players.csv'
    if not players_path.exists():
        print(f"❌ Players data not found: {players_path}")
        return None
    return pd.read_csv(players_path)[['gsis_id', 'short_name', 'display_name', 'last_name', 'position']]


def load_schedule(season=None, week=None):
    """Load the games to predict.

    season: NFL season year (default: the current/upcoming season).
    week:   if given, return exactly that week. Otherwise return the next
            games by date, falling back to the earliest week that has no
            'game_date' in the past.
    """
    season = season or upcoming_or_current_season()
    schedule_path = DATA_DIR / f'nfl_schedule_{season}.csv'
    if not schedule_path.exists():
        print(f"❌ Schedule not found: {schedule_path}")
        return None

    df = pd.read_csv(schedule_path)
    df['game_date'] = pd.to_datetime(df['date'])
    if df['game_date'].dt.tz is None:
        df['game_date'] = df['game_date'].dt.tz_localize('UTC')
    else:
        df['game_date'] = df['game_date'].dt.tz_convert('UTC')
    df['season'] = season

    if week is not None:
        wk = df[df['week'] == int(week)].copy()
        print(f"✅ Week {week}, {season}: {len(wk)} games")
        return wk

    now_utc = pd.Timestamp.now(tz='UTC')
    cutoff = now_utc - timedelta(hours=12)  # stdlib: avoids pandas' numpy-generic-unit DeprecationWarning
    upcoming = df[df['game_date'] >= cutoff].copy()
    if not upcoming.empty:
        target_week = int(upcoming['week'].min())
        upcoming = upcoming[upcoming['week'] == target_week].copy()
        print(f"✅ Next up: Week {target_week}, {season} ({len(upcoming)} games)")
        return upcoming

    # Whole season already in the past -> earliest week (useful off-season / testing).
    first_week = int(df['week'].min())
    wk = df[df['week'] == first_week].copy()
    print(f"⚠️  No upcoming games for {season}; using Week {first_week} ({len(wk)} games)")
    return wk


def load_player_stats():
    """Load all player stats files."""
    stats = {}
    
    for stat_type in ['passing', 'rushing', 'receiving']:
        file_map = {
            'passing': 'player_passing_stats.csv',
            'rushing': 'player_rushing_stats.csv',
            'receiving': 'player_receiving_stats.csv'
        }
        
        file_path = DATA_DIR / file_map[stat_type]
        if file_path.exists():
            stats[stat_type] = pd.read_csv(file_path)
            print(f"✅ Loaded {len(stats[stat_type]):,} {stat_type} records")
        else:
            print(f"⚠️  {stat_type} stats not found")
            stats[stat_type] = None
    
    return stats


def _load_model_records():
    """model_name -> its training metrics entry (model_metrics.json), or {}."""
    import json
    path = MODELS_DIR / 'model_metrics.json'
    try:
        with open(path) as f:
            data = json.load(f)
        return {m['model_name']: m for m in data.get('models', [])
                if isinstance(m, dict) and 'model_name' in m}
    except Exception:
        return {}


def _load_reliability():
    """model_name -> bool, from the training metrics (temporal hold-out).

    TD props and any model with no out-of-time edge come back False. Missing
    file / key defaults to False so nothing is over-sold.
    """
    return {name: bool(m.get('reliable', False)) for name, m in _load_model_records().items()}


def model_incompatibility(model_name, prop_type, record, model):
    """Why a model can't be used with the current feature contract (empty =
    usable). It must record the contract it was trained under and the exact
    feature list, and the booster's own feature names must agree."""
    problems = contract_problems((record or {}).get('feature_contract'),
                                 (record or {}).get('features'), prop_type)
    names = model.get_booster().feature_names
    if names != model_features(prop_type):
        problems.append(f"model feature names {names} != {model_features(prop_type)}")
    return problems


def load_models():
    """Load trained XGBoost models that meet the current feature contract.

    A model without a recorded contract (every model trained before Stage 3, or
    on stats without a valid provenance manifest) or with a different feature
    list is refused and reported, never fed the current features."""
    models = {}
    records = _load_model_records()
    reliability = _load_reliability()
    refused = {}
    
    # Map model names to their configurations
    model_configs = [
        # Passing Yards Models
        ('passing_yards_elite_qb', 'passing_yards', 'elite_qb'),
        ('passing_yards_star_qb', 'passing_yards', 'star_qb'),
        ('passing_yards_good_qb', 'passing_yards', 'good_qb'),
        ('passing_yards_starter', 'passing_yards', 'starter'),
        
        # Passing TDs Models
        ('passing_tds_elite_qb', 'passing_tds', 'elite_qb'),
        ('passing_tds_star_qb', 'passing_tds', 'star_qb'),
        ('passing_tds_good_qb', 'passing_tds', 'good_qb'),
        ('passing_tds_starter', 'passing_tds', 'starter'),
        
        # Rushing Yards Models
        ('rushing_yards_elite_rb', 'rushing_yards', 'elite_rb'),
        ('rushing_yards_star_rb', 'rushing_yards', 'star_rb'),
        ('rushing_yards_good_rb', 'rushing_yards', 'good_rb'),
        ('rushing_yards_starter', 'rushing_yards', 'starter'),
        
        # Rushing TDs Models
        ('rushing_tds_elite_rb', 'rushing_tds', 'elite_rb'),
        ('rushing_tds_star_rb', 'rushing_tds', 'star_rb'),
        ('rushing_tds_good_rb', 'rushing_tds', 'good_rb'),
        ('rushing_tds_anytime', 'rushing_tds', 'anytime'),
        
        # Receiving Yards Models
        ('receiving_yards_elite_wr', 'receiving_yards', 'elite_wr'),
        ('receiving_yards_star_wr', 'receiving_yards', 'star_wr'),
        ('receiving_yards_good_wr', 'receiving_yards', 'good_wr'),
        ('receiving_yards_starter', 'receiving_yards', 'starter'),
        
        # Receiving TDs Models
        ('receiving_tds_elite_wr', 'receiving_tds', 'elite_wr'),
        ('receiving_tds_star_wr', 'receiving_tds', 'star_wr'),
        ('receiving_tds_good_wr', 'receiving_tds', 'good_wr'),
        ('receiving_tds_anytime', 'receiving_tds', 'anytime'),
        
        # Receptions Models
        ('receptions_elite_wr', 'receptions', 'elite_wr'),
        ('receptions_star_wr', 'receptions', 'star_wr'),
        ('receptions_good_wr', 'receptions', 'good_wr'),
        ('receptions_starter', 'receptions', 'starter')
    ]
    
    for model_name, prop_type, line_type in model_configs:
        model_path = MODELS_DIR / f'{model_name}.json'
        
        if model_path.exists():
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            problems = model_incompatibility(model_name, prop_type, records.get(model_name), model)
            if problems:
                refused[model_name] = problems
                continue
            
            # Derive stat type and stat column from prop_type
            if prop_type in ['passing_yards', 'passing_tds']:
                stat_type = 'passing'
                stat_col = 'passing_yards' if prop_type == 'passing_yards' else 'pass_tds'
            elif prop_type in ['rushing_yards', 'rushing_tds']:
                stat_type = 'rushing'
                stat_col = 'rushing_yards' if prop_type == 'rushing_yards' else 'rush_tds'
            elif prop_type in ['receiving_yards', 'receiving_tds', 'receptions']:
                stat_type = 'receiving'
                if prop_type == 'receiving_yards':
                    stat_col = 'receiving_yards'
                elif prop_type == 'receiving_tds':
                    stat_col = 'rec_tds'
                elif prop_type == 'receptions':
                    stat_col = 'receptions'
                else:
                    continue
            else:
                continue
            
            models[model_name] = {
                'model': model,
                'prop_type': prop_type,
                'stat_type': stat_type,
                'stat_col': stat_col,
                'line_type': line_type,
                'line_value': PROP_LINES[prop_type][line_type],
                'reliable': reliability.get(model_name, False),
            }
        else:
            print(f"⚠️  Model {model_name} not found, skipping")

    if refused:
        first = next(iter(refused.values()))
        print(f"⛔ {len(refused)} model(s) refused - not trained under the current feature "
              f"contract ({'; '.join(first)})")

    # Handle missing models with fallbacks
    if 'receiving_yards_elite_wr' not in models and 'receiving_yards_star_wr' in models:
        print("🔄 Using star_wr model as fallback for elite_wr")
        star_model = models['receiving_yards_star_wr']
        models['receiving_yards_elite_wr'] = {
            'model': star_model['model'],
            'prop_type': star_model['prop_type'],
            'stat_type': star_model['stat_type'],
            'stat_col': star_model['stat_col'],
            'line_type': 'elite_wr',
            'line_value': PROP_LINES['receiving_yards']['elite_wr']
        }
    
    # Add fallback for receptions elite_wr (use star_wr model)
    if 'receptions_elite_wr' not in models and 'receptions_star_wr' in models:
        print("🔄 Using star_wr model as fallback for receptions elite_wr")
        star_model = models['receptions_star_wr']
        models['receptions_elite_wr'] = {
            'model': star_model['model'],
            'prop_type': star_model['prop_type'],
            'stat_type': star_model['stat_type'],
            'stat_col': star_model['stat_col'],
            'line_type': 'elite_wr',
            'line_value': PROP_LINES['receptions']['elite_wr']
        }
    
    return models


# ============================================================================
# PLAYER IDENTIFICATION
# ============================================================================

SKILL_POSITIONS = ('QB', 'RB', 'FB', 'WR', 'TE')
# Opt-in: set PROP_ROSTER_FILTER=1 to drop players who are no longer on an NFL
# roster. Off by default because the pre-season nflverse roster feed is
# unreliable early (shuffled teams, missing veterans) and would cut real
# starters. The plumbing below is ready for when the real rosters publish.
ROSTER_FILTER_ON = os.getenv('PROP_ROSTER_FILTER', '').lower() in ('1', 'true', 'yes')
_MIN_SANE_SKILL_IDS = 400  # a real league-wide skill roster is well over this


@lru_cache(maxsize=4)
def load_active_roster(season):
    """frozenset(player_id) of skill players on the `season` roster, or None.

    Team-agnostic on purpose: early-season roster data has players on the wrong
    team, so this only answers "is this person still in the league". Returns
    None (caller then does no filtering) when the flag is off, the fetch fails,
    or the result looks too small to trust.
    """
    if not ROSTER_FILTER_ON:
        return None
    try:
        import nfl_data_py as nfl
        r = nfl.import_seasonal_rosters([int(season)])
    except Exception as e:  # offline, season not published yet, etc.
        print(f"[predict] roster fetch for {season} failed ({e}); no roster filter")
        return None
    if r is None or r.empty or 'player_id' not in r.columns:
        return None
    ids = frozenset(
        r.loc[r['position'].isin(SKILL_POSITIONS), 'player_id'].dropna().astype(str)
    )
    if len(ids) < _MIN_SANE_SKILL_IDS:
        print(f"[predict] {season} roster only has {len(ids)} skill players; "
              "looks incomplete, skipping roster filter")
        return None
    return ids


def get_recent_starters(stats_df, team, n_games=3, roster_ids=None):
    """
    Get players who have recently played for a team.
    Uses last N games to identify likely starters.

    If `roster_ids` (a set of player_id strings for the current-season roster)
    is given, players no longer on it are dropped.
    """
    if stats_df is None or stats_df.empty:
        return []

    # Filter to this team's recent games
    team_stats = stats_df[stats_df['team'] == team].copy()

    if team_stats.empty:
        return []

    # Get most recent games
    team_stats = team_stats.sort_values('game_date', ascending=False)
    recent_games = team_stats['game_id'].unique()[:n_games]
    recent_stats = team_stats[team_stats['game_id'].isin(recent_games)]

    if roster_ids is not None and 'player_id' in recent_stats.columns:
        recent_stats = recent_stats[recent_stats['player_id'].astype(str).isin(roster_ids)]

    # Find players with most activity
    player_activity = recent_stats.groupby('player_name', observed=True).agg({
        'game_id': 'nunique',  # Games played
        'game_date': 'max'      # Last appearance
    }).reset_index()
    
    # Filter to players who played in at least 1 of last 3 games
    active_players = player_activity[player_activity['game_id'] >= 1]['player_name'].tolist()
    
    return active_players


def calculate_trend(player_stats, stat_col):
    """
    Calculate if player is trending up, down, or stable.
    Returns: '🔥' (hot), '➡️' (stable), '❄️' (cold)
    """
    if len(player_stats) < 5:
        return '➡️'  # Not enough data
    
    recent = player_stats.head(3)[stat_col].mean()  # Last 3 games
    older = player_stats.iloc[3:6][stat_col].mean()  # Games 4-6
    
    if older == 0:
        return '➡️'  # Avoid division by zero
    
    pct_change = (recent - older) / older
    
    if pct_change > 0.20:
        return '🔥'  # Hot streak (20%+ increase)
    elif pct_change < -0.20:
        return '❄️'  # Cold streak (20%+ decrease)
    else:
        return '➡️'  # Stable


# ============================================================================
# PREDICTION GENERATION
# ============================================================================

def get_player_position(player_id):
    """Position and display name by stable ID (players.csv gsis_id); short
    names are not unique (e.g. several 'J.Smith')."""
    players_df = load_players()
    if players_df is not None:
        player_record = players_df[players_df['gsis_id'] == player_id]
        if len(player_record) == 1:
            record = player_record.iloc[0]
            position = record['position']
            display_name = record['display_name']
            # Ensure we return strings or None, not NaN
            position = position if isinstance(position, str) else None
            display_name = display_name if isinstance(display_name, str) else None
            return position, display_name
    return None, None


# Tier thresholds on the player's last 8 appearances (raw stat mean): the
# tiers chosen before Stage 3, keyed by prop type, highest first.
TIER_RULES = {
    'passing_yards': ('passing_yards', [(280, 'elite_qb'), (250, 'star_qb'), (220, 'good_qb')], 'starter'),
    'passing_tds': ('pass_tds', [(2.0, 'elite_qb'), (1.5, 'star_qb'), (1.0, 'good_qb')], 'starter'),
    'rushing_yards': ('rushing_yards', [(80, 'elite_rb'), (60, 'star_rb'), (45, 'good_rb')], 'starter'),
    'rushing_tds': ('rush_tds', [(1.0, 'elite_rb'), (0.7, 'star_rb'), (0.4, 'good_rb')], 'anytime'),
    'receiving_yards': ('receiving_yards', [(75, 'elite_wr'), (55, 'star_wr'), (40, 'good_wr')], 'starter'),
    'receiving_tds': ('rec_tds', [(1.0, 'elite_wr'), (0.7, 'star_wr'), (0.4, 'good_wr')], 'anytime'),
    'receptions': ('receptions', [(7.5, 'elite_wr'), (5.5, 'star_wr'), (4.5, 'good_wr')], 'starter'),
}


def player_history(stats_df, player_id):
    """A player's appearances in one stat table by stable ID, every team and
    season, newest first."""
    if stats_df is None or stats_df.empty:
        return stats_df
    rows = stats_df[stats_df['player_id'] == player_id]
    return rows.assign(_d=pd.to_datetime(rows['game_date'])).sort_values(
        ['_d', 'game_id'], ascending=False).drop(columns='_d')


def get_player_performance_tier(player_id, prop_type, all_stats):
    """Line tier from the player's last 8 appearances (by player_id, all teams).
    Fewer than 3 appearances -> the lowest tier."""
    stat_col, cuts, lowest = TIER_RULES[prop_type]
    stat_type = PROP_STAT[prop_type][0]
    history = player_history(all_stats.get(stat_type), player_id)
    if history is None or len(history) < 3:
        return lowest
    avg = history.head(8)[stat_col].mean()
    for cut, tier in cuts:
        if avg >= cut:
            return tier
    return lowest


def candidate_players(all_stats, team, n_games=3, roster_ids=None):
    """Players to predict for `team`, by player_id, with a report.

    A player is a candidate when he appears in one of the team's last
    `n_games` games in any stat table AND his latest appearance in any table
    was for this team. A player whose latest appearance was for another team
    is excluded and counted: the stats can't show which team he is on now (a
    trade after his last game isn't visible without rosters)."""
    seen = []
    for stats_df in all_stats.values():
        if stats_df is None or stats_df.empty:
            continue
        team_rows = stats_df[stats_df['team'] == team]
        recent = (team_rows.assign(_d=pd.to_datetime(team_rows['game_date']))
                  .sort_values('_d', ascending=False)['game_id'].unique()[:n_games])
        seen += list(team_rows.loc[team_rows['game_id'].isin(recent), 'player_id'].unique())
    ids = sorted(set(seen))
    if roster_ids is not None:
        ids = [i for i in ids if str(i) in roster_ids]
    every = pd.concat([t[['player_id', 'player_name', 'team', 'game_id', 'game_date']]
                       for t in all_stats.values() if t is not None and not t.empty])
    every = every[every['player_id'].isin(ids)]
    latest = (every.assign(_d=pd.to_datetime(every['game_date']))
              .sort_values(['_d', 'game_id']).groupby('player_id').tail(1).set_index('player_id'))
    keep = [i for i in ids if latest.loc[i, 'team'] == team]
    moved = [i for i in ids if latest.loc[i, 'team'] != team]
    names = {i: latest.loc[i, 'player_name'] for i in keep}
    return names, {'team': team, 'candidates': len(keep), 'latest_elsewhere': moved}


def predict_props_for_game(game_row, all_stats, models, skip_injuries=False, skip_weather=False,
                            market_odds_df=None, games=None):
    """
    Generate prop predictions for all players in a game.

    Features come from features.inference_rows over `all_stats` (the
    Stage 2 feature source) for the game's unique schedule entry in `games`
    (nfl_games_historical.csv): the same definition as the training rows.
    Players are identified by player_id across teams and seasons. A feature
    or model-input error raises (nothing is silently skipped).

    `market_odds_df`: optional DataFrame from market_odds.fetch_market_odds()
    for the whole week (see market_odds.ODDS_COLUMNS). None/empty is fine -
    every prediction just gets market_line_available=False.
    """
    predictions = []
    if games is None:
        games = load_games(DATA_DIR)

    teams = [game_row['home_team'], game_row['away_team']]
    game_date = game_row['game_date']
    week = game_row['week']
    game_season = game_row.get('season', None)
    target = resolve_game(games, int(game_season), int(week), game_row['home_team'], game_row['away_team'])

    # Injury adjustment is an optional nudge; the ESPN scrape can hang / fail
    # on restricted networks, so it is skippable.
    injuries_df = pd.DataFrame() if skip_injuries else get_injury_report()

    try:
        _season = int(game_season) if game_season else upcoming_or_current_season()
    except (TypeError, ValueError):
        _season = upcoming_or_current_season()
    roster_ids = load_active_roster(_season)  # None -> no filter

    # Candidates for both sides, then one feature row per player per stat table.
    rosters = {}
    for team in teams:
        names, report = candidate_players(all_stats, team, roster_ids=roster_ids)
        if report['latest_elsewhere']:
            print(f"   {team}: {len(report['latest_elsewhere'])} recent player(s) excluded - their "
                  f"latest appearance was for another team, so current team is unknown")
        rosters[team] = names
    target_rows = pd.DataFrame([
        {'player_id': pid, 'player_name': name, 'game_id': target['game_id'],
         'season': int(target['season']), 'week': int(target['week']), 'team': team,
         'opponent': teams[1] if team == teams[0] else teams[0], 'game_date': target['gameday']}
        for team, names in rosters.items() for pid, name in names.items()],
        columns=TARGET_COLUMNS)
    feature_rows = {
        stat_type: inference_rows(stats_df, target_rows, stat_type, games).set_index('player_id')
        for stat_type, stats_df in all_stats.items() if stats_df is not None
    }

    for team in teams:
        opponent = teams[1] if team == teams[0] else teams[0]
        is_home = team == game_row['home_team']

        for player_id, player_name in rosters[team].items():
            # Position and display name by stable ID (players.csv gsis_id).
            player_position, player_display_name = get_player_position(player_id)

            eligible_props = []
            if player_position and player_position in POSITION_MAP:
                eligible_props = POSITION_MAP[player_position]

            for prop_type in eligible_props:
                player_tier = get_player_performance_tier(player_id, prop_type, all_stats)

                matching_model = None
                for model_name, model_info in models.items():
                    if (model_info['prop_type'] == prop_type and
                        model_info['line_type'] == player_tier):
                        matching_model = (model_name, model_info)
                        break
                if matching_model is None:
                    continue

                model_name, model_info = matching_model
                stat_type = model_info['stat_type']
                rows = feature_rows.get(stat_type)
                if rows is None or player_id not in rows.index:
                    continue
                row = rows.loc[[player_id]]
                # First appearance in this table: no history to average, the
                # same exclusion as training (features.has_history).
                if not has_history(row, prop_type).iloc[0]:
                    continue
                # Exact model input; raises on anything missing or non-finite.
                X = model_matrix(row, prop_type)
                features = X.iloc[0].to_dict()
                player_stats_history = player_history(all_stats.get(stat_type), player_id)

                line_value = PROP_LINES[prop_type].get(player_tier, 0.5)  # Default to 0.5 if tier not found
                prob_over = float(model_info['model'].predict_proba(X)[0, 1])

                try:
                    # For star players with tier-appropriate lines, adjust recommendations
                    stat_col = model_info['stat_col']
                    recent_avg = features.get(f"{stat_col}_L5", 0)  # Last 5 games average

                    # If player is a star (averages much more than the line), force OVER
                    if player_tier in ['elite_rb', 'star_rb', 'elite_wr', 'star_wr'] and recent_avg > line_value * 1.3:
                        prob_over = max(prob_over, 0.75)  # Ensure high confidence OVER for stars
                    # If player averages much less than the line, force UNDER
                    elif recent_avg < line_value * 0.6:
                        prob_over = min(prob_over, 0.25)  # Ensure high confidence UNDER
                    
                    # Only include if meets minimum confidence
                    if prob_over >= MIN_CONFIDENCE or prob_over <= (1 - MIN_CONFIDENCE):
                        prediction = {
                            'week': week,
                            'season': game_season,
                            'game_date': game_date,
                            'player_name': player_name,
                            'display_name': player_display_name or player_name,  # Use display_name if available, fallback to short_name
                            'position': player_position or 'Unknown',
                            'team': team,
                            'opponent': opponent,
                            'is_home': is_home,
                            'prop_type': prop_type,
                            'line_type': player_tier,  # Use performance-based tier
                            'line_value': line_value,  # Use tier-appropriate line
                            'prob_over': prob_over,
                            'prob_under': 1 - prob_over,
                            'recommendation': 'OVER' if prob_over >= MIN_CONFIDENCE else 'UNDER',
                            'confidence': max(prob_over, 1 - prob_over),
                            'model_reliable': model_info.get('reliable', False),
                            'avg_L3': features.get(f"{model_info['stat_col']}_L3"),
                            'avg_L5': features.get(f"{model_info['stat_col']}_L5"),
                            'avg_L10': features.get(f"{model_info['stat_col']}_L10")
                        }
                        
                        # Calculate performance trend
                        trend_indicator = calculate_trend(player_stats_history, model_info['stat_col'])
                        prediction['trend'] = trend_indicator
                        
                        # Check for injuries and adjust prediction
                        # Try matching with display_name first, then player_name
                        display_name = player_display_name if player_display_name and isinstance(player_display_name, str) else player_name
                        if isinstance(display_name, str):
                            injury_info = find_player_injury(display_name, injuries_df)
                            if not injury_info and display_name != player_name:
                                injury_info = find_player_injury(player_name, injuries_df)
                        else:
                            injury_info = None
                        
                        if injury_info:
                            adjusted_prediction = adjust_prediction_for_injury(prediction, injury_info)
                            if adjusted_prediction is None:
                                # Player is out - skip this prediction
                                continue
                            prediction = adjusted_prediction
                        
                        # Apply weather adjustments for outdoor games. Skippable -
                        # the Open-Meteo lookup is per player/prop and slow/flaky.
                        if skip_weather:
                            prediction['weather_adjusted'] = False
                            prediction['weather_conditions'] = "Skipped"
                        try:
                            if skip_weather:
                                raise _SkipWeather()
                            # Ensure game_date is a string
                            game_date_str = str(game_date) if hasattr(game_date, 'strftime') else str(game_date)
                            if len(game_date_str) > 10:  # If it has time component, extract date only
                                game_date_str = game_date_str[:10]
                            weather_data = get_weather_for_game(team, game_row['venue'], game_date_str)
                            if not weather_data.get('is_dome', False):
                                original_prob = prediction['prob_over']
                                adjusted_prob = adjust_for_weather(original_prob, prop_type, weather_data)
                                prediction['prob_over'] = adjusted_prob
                                prediction['prob_under'] = 1 - adjusted_prob
                                prediction['confidence'] = max(adjusted_prob, 1 - adjusted_prob)
                                prediction['recommendation'] = 'OVER' if adjusted_prob >= MIN_CONFIDENCE else 'UNDER'
                                prediction['weather_adjusted'] = True
                                prediction['weather_conditions'] = f"{weather_data['temp_f']:.0f}°F, {weather_data['wind_mph']:.1f} mph wind"
                            else:
                                prediction['weather_adjusted'] = False
                                prediction['weather_conditions'] = "Dome"
                        except _SkipWeather:
                            pass
                        except Exception as e:
                            print(f"⚠️ Weather adjustment failed for {player_name}: {e}")
                            prediction['weather_adjusted'] = False
                            prediction['weather_conditions'] = "Unknown"

                        # Attach real DK/FanDuel market odds when available
                        # (opt-in, see market_odds.py). Computed against the
                        # final prob_over, after injury/weather adjustments.
                        market_info = find_market_line(
                            prediction['display_name'], team, prop_type, market_odds_df
                        )
                        prediction = attach_market_odds(prediction, market_info)

                        predictions.append(prediction)
                    
                except Exception as e:
                    # Injury / weather / market post-processing only; model
                    # inputs and the model call above fail loudly.
                    print(f"⚠️  Post-processing failed for {player_name} {prop_type}: {e}")
                    continue
    
    return predictions


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def generate_predictions(season=None, week=None, freeze=True,
                         skip_injuries=False, skip_weather=False,
                         skip_market_odds=False):
    """Main pipeline to generate all prop predictions.

    Writes data_files/player_props_predictions.csv (always, the 'latest' feed)
    and, when `freeze` is set, a per-week snapshot
    data_files/player_props_predictions_week{W}_{season}.csv that is written
    ONCE and never overwritten - that frozen file is what backtest.py scores,
    so the weekly accuracy check is a genuine prospective test.

    Same shape as skip_injuries/skip_weather: `skip_market_odds=False` is the
    default, but market_odds.fetch_market_odds() is itself a no-op unless
    ODDS_API_KEY is set, so this stays a true no-cost no-op everywhere the key
    isn't configured. See docs/ODDS_API_INTEGRATION_PLAN.md.

    Before anything is fetched or written, provenance.check_inputs must pass
    for the target week: the stats must match their provenance manifest and
    cover every earlier completed game. Otherwise this returns 1 and leaves
    every existing predictions file untouched. Features come only from games
    before the target week. See docs/architecture.md ("Stage 2 freshness and
    provenance").
    """
    print("=" * 70)
    print("🎯 NFL Player Props Prediction Pipeline")
    print("=" * 70)
    print()

    # Load data
    print("📂 Loading data...")
    schedule = load_schedule(season=season, week=week)
    if schedule is None or schedule.empty:
        print("❌ No games to predict")
        return

    resolved_season = int(schedule['season'].dropna().iloc[0]) if 'season' in schedule.columns and schedule['season'].notna().any() else (season or upcoming_or_current_season())
    resolved_week = int(schedule['week'].dropna().iloc[0])

    # Freshness + provenance gate (read-only). Nothing below runs, and no
    # predictions file is touched, unless it passes.
    try:
        check = check_inputs(resolved_season, resolved_week, DATA_DIR,
                             prediction_matchups=set(zip(schedule['home_team'], schedule['away_team'])))
    except Exception as exc:  # noqa: BLE001 - an unfinished check is a failed check
        print(f"❌ The player-prop input check could not complete ({type(exc).__name__}: {exc}) - "
              f"no predictions written; existing files are unchanged.")
        return 1
    print(format_report(check))
    if not check.ok:
        print("❌ Player-prop inputs are not verified fresh for this week - no predictions "
              "written; existing files are unchanged.")
        return 1
    # Only rows from completed games before the target week.
    all_stats = check.feature_tables
    models = load_models()

    if not models:
        print("❌ No model meets the current feature contract (features.FEATURE_CONTRACT) - "
              "no predictions written; existing files are unchanged. Retrain on stats with a "
              "valid provenance manifest.")
        return 1
    games = load_games(DATA_DIR)

    # Real DK/FanDuel lines for the reliable prop types, fetched once for the
    # whole week (not per game). No-op (empty DataFrame) unless ODDS_API_KEY
    # is set - see market_odds.py.
    market_odds_df = (
        pd.DataFrame() if skip_market_odds
        else fetch_market_odds(resolved_season, resolved_week, schedule)
    )

    print()
    print("🔮 Generating predictions...")
    print("-" * 70)

    # Generate predictions for each game
    all_predictions = []

    for idx, game_row in schedule.iterrows():
        print(f"\n📅 {game_row['away_team']} @ {game_row['home_team']} (Week {game_row['week']})")

        try:
            game_predictions = predict_props_for_game(
                game_row, all_stats, models,
                skip_injuries=skip_injuries, skip_weather=skip_weather,
                market_odds_df=market_odds_df, games=games,
            )
        except (FeatureContractError, FeatureValidationError) as exc:
            # Nothing has been written yet; a feature that can't be built
            # stops the whole run rather than dropping players silently.
            print(f"❌ Features could not be built for {game_row['away_team']} @ "
                  f"{game_row['home_team']}: {exc} - no predictions written; existing files "
                  f"are unchanged.")
            return 1
        all_predictions.extend(game_predictions)
        
        print(f"   Generated {len(game_predictions)} prop predictions")
    
    # Save predictions
    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)
        
        # Sort by confidence descending
        pred_df = pred_df.sort_values('confidence', ascending=False)
        
        # Save 'latest' feed
        output_path = DATA_DIR / 'player_props_predictions.csv'
        pred_df.to_csv(output_path, index=False)

        print()
        print("=" * 70)
        print(f"✅ Generated {len(pred_df)} total prop predictions")
        print(f"💾 Saved to: {output_path}")

        # Frozen per-week snapshot (write once, never overwrite)
        if freeze and 'week' in pred_df.columns and pred_df['week'].notna().any():
            wk = int(pred_df['week'].dropna().iloc[0])
            szn = int(pred_df['season'].dropna().iloc[0]) if 'season' in pred_df.columns and pred_df['season'].notna().any() else (season or upcoming_or_current_season())
            snap_path = DATA_DIR / f'player_props_predictions_week{wk}_{szn}.csv'
            if snap_path.exists():
                print(f"🔒 Snapshot already frozen: {snap_path.name} (not overwriting)")
            else:
                pred_df.to_csv(snap_path, index=False)
                print(f"🔒 Froze Week {wk} {szn} snapshot -> {snap_path.name}")
        print()
        
        # Summary statistics
        print("📊 Prediction Summary:")
        print(f"   Total props: {len(pred_df)}")
        print(f"   High confidence (≥65%): {len(pred_df[pred_df['confidence'] >= 0.65])}")
        print(f"   Medium confidence (60-65%): {len(pred_df[(pred_df['confidence'] >= 0.60) & (pred_df['confidence'] < 0.65)])}")
        print(f"   Standard confidence (55-60%): {len(pred_df[(pred_df['confidence'] >= 0.55) & (pred_df['confidence'] < 0.60)])}")
        print()
        
        # Top recommendations
        print("🔥 Top 10 Recommendations:")
        print("-" * 70)
        top_props = pred_df.head(10)
        for idx, row in top_props.iterrows():
            print(f"{row['player_name']:20s} {row['team']:3s} | "
                  f"{row['prop_type']:15s} {row['recommendation']:5s} {row['line_value']:6.1f} | "
                  f"Conf: {row['confidence']:.1%} | L3: {row['avg_L3']:.1f}")
        
        print()
        print("=" * 70)
        print("✅ Prediction pipeline complete!")
        print("\nNext step: View predictions in Player Props UI page")
        print("=" * 70)
    else:
        print("\n⚠️  No predictions generated. Check if player stats are available.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Generate player-prop predictions")
    ap.add_argument('--season', type=int, default=None,
                    help="NFL season year (default: current/upcoming)")
    ap.add_argument('--week', type=int, default=None,
                    help="specific week to predict (default: next upcoming week)")
    ap.add_argument('--no-freeze', action='store_true',
                    help="do not write the per-week frozen snapshot")
    ap.add_argument('--no-injuries', action='store_true',
                    help="skip the ESPN injury scrape (optional nudge; can hang on restricted networks)")
    ap.add_argument('--no-weather', action='store_true',
                    help="skip per-player Open-Meteo weather lookups (slow/flaky)")
    ap.add_argument('--no-market-odds', action='store_true',
                    help="skip fetching real DK/FanDuel prop lines (also a no-op if ODDS_API_KEY is unset)")
    args = ap.parse_args()
    sys.exit(generate_predictions(
        season=args.season, week=args.week, freeze=not args.no_freeze,
        skip_injuries=args.no_injuries, skip_weather=args.no_weather,
        skip_market_odds=args.no_market_odds,
    ) or 0)
