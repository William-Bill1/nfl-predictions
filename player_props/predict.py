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
    from .model_bundle import BundleError, load_bundle, score_probabilities
    from .features import (
        PROP_LINES, PROP_STAT, TARGET_COLUMNS, FeatureContractError, has_history,
        inference_rows, model_matrix, resolve_game,
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
    from model_bundle import BundleError, load_bundle, score_probabilities
    from features import (
        PROP_LINES, PROP_STAT, TARGET_COLUMNS, FeatureContractError, has_history,
        inference_rows, model_matrix, resolve_game,
    )

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(__file__).parent.parent / 'data_files'
MODELS_DIR = Path(__file__).parent / 'models'


class _SkipWeather(Exception):
    """Internal sentinel to bypass the weather-adjustment block."""


# Minimum confidence threshold for recommendations
MIN_CONFIDENCE = 0.55  # 55% probability for a recommendation

# Prop eligibility (policy 'position-group-v1'): a player's prop families come
# from the players reference's position_group, looked up by gsis_id. The
# position field stays the displayed label (a fullback is shown as FB), but it
# no longer selects props: nflverse/NFL label some fullbacks FB and others RB,
# while position_group puts every one of them in RB. The models are trained on
# every row of their stat table regardless of position, so this changes which
# (player, prop) pairs are predicted, not the models.
ELIGIBILITY_POLICY = 'position-group-v1'
ELIGIBILITY_FIELD = 'position_group'
POSITION_MAP = {            # keys are position_group values
    'QB': ['passing_yards', 'passing_tds'],
    'RB': ['rushing_yards', 'rushing_tds', 'receptions'],
    'WR': ['receiving_yards', 'receiving_tds', 'receptions'],
    'TE': ['receiving_yards', 'receiving_tds', 'receptions']
}
# The group each skill label must carry; any other label must not carry an
# eligible group. A record breaking this is inconsistent and gets no props.
LABEL_GROUP = {'QB': 'QB', 'RB': 'RB', 'FB': 'RB', 'WR': 'WR', 'TE': 'TE'}

# ============================================================================
# DATA LOADING
# ============================================================================

PLAYER_COLUMNS = ['gsis_id', 'short_name', 'display_name', 'last_name', 'position', 'position_group']


def load_players():
    """Load player metadata with full names. A reference without the
    eligibility field is refused (never a silent fall back to position)."""
    players_path = DATA_DIR / 'players.csv'
    if not players_path.exists():
        print(f"❌ Players data not found: {players_path}")
        return None
    df = pd.read_csv(players_path, dtype={'gsis_id': str})
    missing = [c for c in PLAYER_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{players_path.name} lacks column(s) {missing}; prop eligibility "
                         f"({ELIGIBILITY_POLICY}) needs {ELIGIBILITY_FIELD}, so no props are generated")
    return df[PLAYER_COLUMNS]


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


def load_model_bundle():
    """The current verified model bundle (model_bundle.load_bundle): every
    model file, the metrics and the manifest checked against each other and
    against the feature contract and scorer. Raises BundleError otherwise -
    including for the model files committed before Stage 4, which have no
    manifest. No model is ever substituted for another."""
    return load_bundle(MODELS_DIR, PROP_LINES)


def load_models():
    """{model_name: info} from the current verified bundle, or {} (with the
    reason printed) when there is none."""
    try:
        return load_model_bundle()['models']
    except BundleError as exc:
        print("⛔ No verified model bundle - " + "; ".join(exc.problems[:5]))
        return {}


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
    identity = get_player_identity(player_id)
    return identity['position'], identity['display_name']


def eligibility_group(position, group):
    """(group, None) for a usable record, else (None, problem). The group is
    used exactly as written: a missing, blank or padded value, or one that
    contradicts the position label (LABEL_GROUP), is a problem - never
    repaired and never replaced by the position."""
    def text(v):
        return v if isinstance(v, str) else None
    position, group = text(position), text(group)
    if group is None or not group.strip():
        return None, f"missing {ELIGIBILITY_FIELD}"
    if group != group.strip():
        return None, f"{ELIGIBILITY_FIELD} {group!r} has surrounding whitespace"
    if position is None or not position.strip():
        return None, "missing position"
    if position != position.strip():
        return None, f"position {position!r} has surrounding whitespace"
    if position in LABEL_GROUP and group != LABEL_GROUP[position]:
        return None, f"position {position!r} with {ELIGIBILITY_FIELD} {group!r} (expected {LABEL_GROUP[position]!r})"
    if position not in LABEL_GROUP and group in POSITION_MAP:
        return None, f"position {position!r} with eligible {ELIGIBILITY_FIELD} {group!r}"
    return group, None


def get_player_identity(player_id):
    """{'position', 'display_name', 'group', 'problem'} for one stable ID.
    position is the displayed label; group (position_group) selects props.
    A missing, repeated or unusable record has group None and says why."""
    out = {'position': None, 'display_name': None, 'group': None, 'problem': None}
    players_df = load_players()
    if players_df is None:
        out['problem'] = 'no players reference'
        return out
    player_record = players_df[players_df['gsis_id'] == str(player_id)]
    if len(player_record) != 1:
        out['problem'] = ('not in the players reference' if player_record.empty
                          else 'several players-reference records')
        return out
    record = player_record.iloc[0]
    # Ensure we return strings or None, not NaN
    out['position'] = record['position'] if isinstance(record['position'], str) else None
    out['display_name'] = record['display_name'] if isinstance(record['display_name'], str) else None
    out['group'], out['problem'] = eligibility_group(record['position'], record['position_group'])
    return out


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


def injury_adjustment_label(injury_info):
    """Which injuries.adjust_prediction_for_injury rule scaled the probability
    (questionable 0.8, doubtful 0.7, limited practice 0.9), by the same
    precedence. Only called when the probability actually changed."""
    status = str(injury_info.get('status') or '').lower()
    if status in ('questionable', 'doubtful'):
        return f'injury_{status}'
    if str(injury_info.get('practice_participation') or '').lower() == 'limited':
        return 'injury_limited_practice'
    return 'injury'


def predict_props_for_game(game_row, all_stats, models, skip_injuries=False, skip_weather=False,
                            market_odds_df=None, games=None, bundle_id=None):
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

    ineligible = []          # reference records that can't select props, with why
    for team in teams:
        opponent = teams[1] if team == teams[0] else teams[0]
        is_home = team == game_row['home_team']

        for player_id, player_name in rosters[team].items():
            # Position label, display name and eligibility group by stable ID
            # (players.csv gsis_id). Props come from the group, never the label.
            identity = get_player_identity(player_id)
            player_position, player_display_name = identity['position'], identity['display_name']
            player_group = identity['group']
            if identity['problem'] and identity['problem'] != 'not in the players reference':
                ineligible.append(f"{player_id} ({identity['problem']})")

            eligible_props = POSITION_MAP.get(player_group, []) if player_group else []

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

                # The model's own line: its probability is P(stat > this line).
                line_value = model_info['line_value']
                # Raw model output from the shared scorer (model_bundle.score_probabilities),
                # the exact quantity the bundle's metrics evaluate. Everything after
                # this adjusts it and is recorded in 'adjustments'.
                raw_prob_over = float(score_probabilities(model_info['model'], X, prop_type)[0])
                prob_over = raw_prob_over
                adjustments = []

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
                    if prob_over != raw_prob_over:
                        adjustments.append('clamp_star_floor_0.75' if prob_over > raw_prob_over
                                           else 'clamp_low_average_cap_0.25')
                    
                    # Only include if meets minimum confidence
                    if prob_over >= MIN_CONFIDENCE or prob_over <= (1 - MIN_CONFIDENCE):
                        prediction = {
                            'week': week,
                            'season': game_season,
                            'game_date': game_date,
                            'player_name': player_name,
                            # The stable ID that selected this feature row and
                            # scored the model (never derived from a name).
                            'player_id': str(player_id),
                            'display_name': player_display_name or player_name,  # Use display_name if available, fallback to short_name
                            'position': player_position or 'Unknown',
                            # The position_group that selected this prop (policy
                            # ELIGIBILITY_POLICY); position above is the label.
                            'eligibility_group': player_group,
                            'team': team,
                            'opponent': opponent,
                            'is_home': is_home,
                            'prop_type': prop_type,
                            'line_type': player_tier,  # Use performance-based tier
                            'line_value': line_value,  # Use tier-appropriate line
                            'raw_model_prob_over': raw_prob_over,
                            'prob_over': prob_over,
                            'prob_under': 1 - prob_over,
                            'recommendation': 'OVER' if prob_over >= MIN_CONFIDENCE else 'UNDER',
                            'confidence': max(prob_over, 1 - prob_over),
                            # The raw model's out-of-time check, not a check of the
                            # adjusted probability shown.
                            'model_reliable': model_info.get('reliable', False),
                            'model_name': model_name,
                            'model_bundle': bundle_id,
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
                            # The helper scales prediction in place and returns the same
                            # dict, so the value it starts from is captured first.
                            prob_before_injury = prediction['prob_over']
                            adjusted_prediction = adjust_prediction_for_injury(prediction, injury_info)
                            if adjusted_prediction is None:
                                # Player is out - skip this prediction
                                continue
                            if adjusted_prediction['prob_over'] != prob_before_injury:
                                adjustments.append(injury_adjustment_label(injury_info))
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
                                if adjusted_prob != original_prob:
                                    adjustments.append('weather')
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
                        prediction['adjustments'] = ';'.join(adjustments) or 'none'

                        predictions.append(prediction)
                    
                except Exception as e:
                    # Injury / weather / market post-processing only; model
                    # inputs and the model call above fail loudly.
                    print(f"⚠️  Post-processing failed for {player_name} {prop_type}: {e}")
                    continue

    if ineligible:
        print(f"   {len(ineligible)} candidate(s) get no props: their players-reference record can't "
              f"select an eligibility group ({ELIGIBILITY_POLICY}): " + "; ".join(ineligible[:10]))
    return predictions


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def generate_predictions(season=None, week=None, freeze=True,
                         skip_injuries=False, skip_weather=False,
                         skip_market_odds=False, data_dir=None, models_dir=None):
    """generate_predictions, optionally against explicit directories.

    ``data_dir`` (stats, manifest, schedules, players, predictions output) and
    ``models_dir`` (the verified model bundle) replace DATA_DIR / MODELS_DIR
    for this call only and are restored afterwards; every helper here reads
    them. Used by the Stage 5B rehearsal; omitted, nothing changes."""
    global DATA_DIR, MODELS_DIR
    saved = DATA_DIR, MODELS_DIR
    try:
        if data_dir is not None:
            DATA_DIR = Path(data_dir)
        if models_dir is not None:
            MODELS_DIR = Path(models_dir)
        return _generate_predictions(season, week, freeze, skip_injuries, skip_weather,
                                     skip_market_odds)
    finally:
        DATA_DIR, MODELS_DIR = saved


def _generate_predictions(season=None, week=None, freeze=True,
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
    # Verified model bundle (Stage 4). Its training source is recorded
    # separately from the feature source checked above; both meet the same
    # feature contract, but a model trained on games from this week or later
    # must not predict it.
    try:
        bundle = load_model_bundle()
    except BundleError as exc:
        print("❌ No verified model bundle (" + "; ".join(exc.problems[:5]) + ") - no "
              "predictions written; existing files are unchanged.")
        return 1
    trained_through = bundle['manifest']['training_source']['last_game_date']
    if trained_through >= check.summary['feature_cutoff']:
        print(f"❌ Model bundle {bundle['manifest']['bundle_id']} was trained on games through "
              f"{trained_through}, on or after this week's first game "
              f"({check.summary['feature_cutoff']}) - no predictions written; existing files "
              f"are unchanged.")
        return 1
    models = bundle['models']
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
                bundle_id=bundle['manifest']['bundle_id'],
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
