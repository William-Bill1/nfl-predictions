"""
Prediction Accuracy Tracking and Backtesting Module

This module provides tools to track prediction accuracy by comparing model predictions
against actual game results. It calculates hit rates, ROI analysis, and performance metrics.
"""

import sys

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import nfl_data_py as nfl
from typing import Dict, List, Tuple, Optional

SCHEDULE_PATH = "data_files/nfl_games_historical.csv"
# collect_actual_results() records, in DataFrame.attrs, per-game completion
# evidence from the play-by-play it aggregated: {game_id: {"end_game": bool,
# "home_total": int, "away_total": int}}. Without it (e.g. pre-aggregated
# weekly stats, which carry no game IDs) completeness can't be established.
COMPLETION_ATTR = "game_completion"
END_GAME_DESC = "END GAME"


def _say(message: str) -> None:
    """Write one console diagnostic for this module.

    The messages are plain ASCII, but they can carry text from elsewhere (an
    exception, a file path, a player's name). With stdout redirected on
    Windows, Python writes in the locale code page (e.g. cp1252), which can't
    encode every character. Two output failures are handled so they don't
    reach the caller:

    * UnicodeEncodeError: the message is written again with the characters
      the stream can't encode escaped (``\\u2603``); if that retry fails too,
      the message is dropped.
    * OSError / ValueError from an unavailable stream (e.g. closed): the
      message is dropped.

    (An emoji here once turned a successful collection into its failure path,
    and then crashed the error report itself.)
    """
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        try:
            print(message.encode(encoding, "backslashreplace").decode(encoding))
        except Exception:  # noqa: BLE001 - only a log line
            pass
    except (OSError, ValueError):
        pass


def pbp_game_completion(pbp: pd.DataFrame) -> Dict[str, Dict]:
    """Per-game completion evidence from nflverse play-by-play.

    For each ``game_id``: whether the game's ``END GAME`` play is present, and
    that play's score as one state (``total_home_score``, ``total_away_score``
    read from the same row). The totals are ``None`` when the game has no
    ``END GAME`` play, or when its ``END GAME`` rows give no single complete
    state. Running totals are NOT monotonic (a reversed score dips and
    recovers; ~44% of 2020-2025 games do this), so per-column maxima could pair
    scores that never stood together and are not used. (PBP's ``home_score`` /
    ``away_score`` columns are copied from the schedule and say nothing about
    whether the plays are complete, so they aren't used either.)
    """
    needed = {"game_id", "desc", "total_home_score", "total_away_score"}
    if pbp.empty or not needed <= set(pbp.columns):
        return {}
    is_end = pbp["desc"].astype(str).str.strip().str.upper() == END_GAME_DESC
    end_rows = pbp[is_end]
    result = {str(gid): {"end_game": False, "home_total": None, "away_total": None}
              for gid in pbp["game_id"].dropna().unique()}
    for gid, rows in end_rows.groupby("game_id"):
        states = rows[["total_home_score", "total_away_score"]].drop_duplicates()
        entry = result[str(gid)]
        entry["end_game"] = True
        if len(states) == 1 and not states.isna().any(axis=None):
            entry["home_total"] = int(states.iloc[0]["total_home_score"])
            entry["away_total"] = int(states.iloc[0]["total_away_score"])
    return result


def week_results_status(actuals_df: pd.DataFrame, week: int, season: int,
                        schedule: Optional[pd.DataFrame] = None) -> Tuple[bool, str]:
    """Whether ``actuals_df`` holds final results for every game of the week.

    Final means, for **each** regular-season game of (season, week) by
    ``game_id``:

    * the schedule has both final scores;
    * the play-by-play includes that game and its ``END GAME`` play;
    * the play-by-play's final running score equals the schedule's final score.

    Seeing every team in the stats isn't enough: a game whose play-by-play
    stops early still lists both teams. If the evidence is missing (for
    example pre-aggregated stats, which have no game IDs), the result can't be
    shown to be complete and stays provisional. Only final results may be
    cached.

    Limitation: non-scoring plays missing from the middle of a game can't be
    detected when the game's ``END GAME`` play and final score are present.

    Returns ``(final, reason)``; ``reason`` explains why results aren't final.
    """
    if schedule is None:
        try:
            schedule = pd.read_csv(SCHEDULE_PATH, sep="\t", usecols=[
                "game_id", "season", "game_type", "week", "home_score", "away_score"])
        except (OSError, ValueError) as e:
            return False, f"the schedule couldn't be read to confirm the week is complete ({e})"
    games = schedule[(schedule["season"] == season) & (schedule["week"] == week)
                     & (schedule["game_type"] == "REG")]
    if games.empty:
        return False, f"no Week {week} {season} regular-season games are in the schedule"
    unfinished = int((games["home_score"].isna() | games["away_score"].isna()).sum())
    if unfinished:
        return False, f"{unfinished} of {len(games)} Week {week} games aren't final yet"
    if actuals_df is None or actuals_df.empty:
        return False, "no actual results are available yet"
    completion = actuals_df.attrs.get(COMPLETION_ATTR)
    if not completion:
        return False, ("completeness can't be verified game by game from this data "
                       "(no play-by-play game IDs)")

    missing, unended, mismatched = [], [], []
    for game in games.itertuples(index=False):
        evidence = completion.get(str(game.game_id))
        if evidence is None:
            missing.append(game.game_id)
        elif not evidence.get("end_game"):
            unended.append(game.game_id)
        elif evidence.get("home_total") is None or evidence.get("away_total") is None:
            mismatched.append(f"{game.game_id} (no valid final score state in the play-by-play)")
        elif (evidence["home_total"], evidence["away_total"]) != (
                int(game.home_score), int(game.away_score)):
            mismatched.append(f"{game.game_id} ({evidence['away_total']}-"
                              f"{evidence['home_total']} vs final "
                              f"{int(game.away_score)}-{int(game.home_score)})")
    problems = []
    if missing:
        problems.append("no play-by-play for " + ", ".join(missing))
    if unended:
        problems.append("play-by-play ends before the final play for " + ", ".join(unended))
    if mismatched:
        problems.append("play-by-play score doesn't match the final score for "
                        + ", ".join(mismatched))
    if problems:
        return False, "; ".join(problems) + " (data may be delayed or incomplete)"
    return True, ""


def collect_actual_results(week: int, season: int = 2025) -> tuple[pd.DataFrame, str]:
    """
    Fetch actual player stats for completed games in a given week.
    
    Tries two methods in order:
    1. Pre-aggregated weekly stats (fast, preferred)
    2. PBP data aggregation (slower, fallback)

    Args:
        week: NFL week number
        season: NFL season year

    Returns:
        Tuple of (DataFrame with actual player statistics, error message if any)
    """
    # Method 1: Try pre-aggregated weekly stats first (much faster)
    try:
        _say(f"Loading pre-aggregated stats for {season} season, Week {week}...")
        actual_stats = nfl.import_weekly_data([season], columns=[
            'player_name', 'week', 'season', 'passing_yards', 'passing_tds',
            'rushing_yards', 'rushing_tds', 'receiving_yards', 'receiving_tds',
            'receptions', 'completions', 'attempts'
        ])
        
        # Filter to specific week
        week_stats = actual_stats[actual_stats['week'] == week].copy()
        
        if week_stats.empty:
            _say(f"WARNING: pre-aggregated stats are empty for Week {week}; trying play-by-play aggregation...")
            raise ValueError("Empty weekly stats")
        
        # Clean up player names
        week_stats['player_name'] = week_stats['player_name'].str.strip()
        
        _say(f"OK: collected pre-aggregated stats for {len(week_stats)} players in Week {week}")
        return week_stats, ""
        
    except Exception as e:
        # Method 2: Fallback to PBP aggregation
        _say(f"   Pre-aggregated stats unavailable ({str(e)[:50]})")
        _say("   Falling back to play-by-play data aggregation...")
        
        try:
            # Load play-by-play data directly from nflverse parquet files
            pbp_url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
            
            pbp_data = pd.read_parquet(pbp_url)
            
            # Filter to specific week
            week_pbp = pbp_data[pbp_data['week'] == week].copy()
            
            if week_pbp.empty:
                return pd.DataFrame(), f"No play-by-play data found for Week {week}, Season {season}"
            
            _say(f"   Loaded {len(week_pbp):,} plays for Week {week}")
            
            # Aggregate passing stats
            passing_plays = week_pbp[week_pbp['pass'] == 1].copy()
            passing_stats = passing_plays.groupby('passer_player_name', observed=True).agg({
                'passing_yards': 'sum',
                'pass_touchdown': 'sum'
            }).reset_index()
            passing_stats.columns = ['player_name', 'passing_yards', 'passing_tds']
            
            # Aggregate rushing stats
            rushing_plays = week_pbp[week_pbp['rush'] == 1].copy()
            rushing_stats = rushing_plays.groupby('rusher_player_name', observed=True).agg({
                'rushing_yards': 'sum',
                'rush_touchdown': 'sum'
            }).reset_index()
            rushing_stats.columns = ['player_name', 'rushing_yards', 'rushing_tds']
            
            # Aggregate receiving stats
            receiving_plays = week_pbp[(week_pbp['pass'] == 1) & 
                                        (week_pbp['receiver_player_name'].notna())].copy()
            receiving_stats = receiving_plays.groupby('receiver_player_name', observed=True).agg({
                'receiving_yards': 'sum',
                'complete_pass': 'sum',  # Receptions
                'pass_touchdown': 'sum'   # Receiving TDs
            }).reset_index()
            receiving_stats.columns = ['player_name', 'receiving_yards', 'receptions', 'receiving_tds']
            
            # Merge all stats together
            combined_stats = pd.DataFrame()
            
            if not passing_stats.empty:
                combined_stats = passing_stats
            
            if not rushing_stats.empty:
                if combined_stats.empty:
                    combined_stats = rushing_stats
                else:
                    combined_stats = combined_stats.merge(
                        rushing_stats, on='player_name', how='outer'
                    )
            
            if not receiving_stats.empty:
                if combined_stats.empty:
                    combined_stats = receiving_stats
                else:
                    combined_stats = combined_stats.merge(
                        receiving_stats, on='player_name', how='outer'
                    )
            
            # Fill NaN with 0 and add metadata
            combined_stats = combined_stats.fillna(0)
            combined_stats['week'] = week
            combined_stats['season'] = season
            
            # Clean up player names
            combined_stats['player_name'] = combined_stats['player_name'].str.strip()
            
            _say(f"OK: collected play-by-play stats for {len(combined_stats)} players in Week {week}")
            combined_stats.attrs[COMPLETION_ATTR] = pbp_game_completion(week_pbp)

            return combined_stats, ""
            
        except Exception as pbp_error:
            error_msg = f"Both methods failed - Pre-aggregated: {str(e)[:50]}, PBP: {str(pbp_error)[:50]}"
            _say(f"ERROR: could not collect actual results: {error_msg}")
            return pd.DataFrame(), error_msg


def calculate_hit_rate(predictions_df: pd.DataFrame, actuals_df: pd.DataFrame) -> Dict:
    """
    Calculate prediction accuracy (hit rate) by comparing predictions vs actual results.

    Args:
        predictions_df: DataFrame with model predictions
        actuals_df: DataFrame with actual game results

    Returns:
        Dictionary with accuracy metrics
    """
    if predictions_df.empty or actuals_df.empty:
        return {
            'overall_accuracy': 0.0,
            'by_confidence_tier': pd.Series(),
            'by_prop_type': pd.Series(),
            'total_predictions': 0,
            'detailed_results': pd.DataFrame()
        }

    # Merge predictions with actual results
    merged = predictions_df.merge(
        actuals_df,
        on=['player_name'],
        suffixes=('_pred', '_actual'),
        how='left'
    )

    # Filter out players with no actual stats (DNP, etc.)
    # Check if any of the key stat columns have data
    stat_columns = ['passing_yards', 'rushing_yards', 'receiving_yards']
    merged = merged.dropna(subset=stat_columns, how='all')

    if merged.empty:
        _say("WARNING: no predictions matched any actual results")
        return {
            'overall_accuracy': 0.0,
            'by_confidence_tier': pd.Series(),
            'by_prop_type': pd.Series(),
            'total_predictions': 0,
            'detailed_results': pd.DataFrame()
        }

    results = []

    for _, row in merged.iterrows():
        prop_type = row['prop_type']
        line_value = row['line_value']
        predicted_rec = row['recommendation']
        confidence = row['confidence']

        # Get actual stat value based on prop type
        stat_mapping = {
            'passing_yards': 'passing_yards',
            'passing_tds': 'passing_tds',
            'rushing_yards': 'rushing_yards',
            'rushing_tds': 'rushing_tds',
            'receiving_yards': 'receiving_yards',
            'receiving_tds': 'receiving_tds',
            'receptions': 'receptions'
        }

        stat_col = stat_mapping.get(prop_type)
        if not stat_col or pd.isna(row.get(stat_col, np.nan)):
            continue

        actual_value = row[stat_col]

        # Determine actual outcome
        actual_outcome = 'OVER' if actual_value > line_value else 'UNDER'

        # Check if prediction was correct
        hit = (predicted_rec == actual_outcome)

        results.append({
            'player_name': row['player_name'],
            'display_name': row.get('display_name', row['player_name']),
            'prop_type': prop_type,
            'line_value': line_value,
            'predicted': predicted_rec,
            'actual': actual_outcome,
            'actual_value': actual_value,
            'hit': hit,
            'confidence': confidence,
            'model_reliable': bool(row.get('model_reliable', True)),
            'team': row.get('team', ''),
            'week': row.get('week_pred', 0)
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return {
            'overall_accuracy': 0.0,
            'by_confidence_tier': pd.Series(),
            'by_prop_type': pd.Series(),
            'total_predictions': 0,
            'detailed_results': pd.DataFrame()
        }

    # Calculate overall accuracy
    overall_hit_rate = results_df['hit'].mean()

    # Calculate accuracy by confidence tier
    confidence_bins = [0.50, 0.60, 0.65, 0.70, 0.75, 1.0]
    confidence_labels = ['50-60%', '60-65%', '65-70%', '70-75%', '75%+']

    results_df['confidence_tier'] = pd.cut(
        results_df['confidence'],
        bins=confidence_bins,
        labels=confidence_labels,
        include_lowest=True
    )

    by_confidence = results_df.groupby('confidence_tier', observed=True)['hit'].mean()

    # Calculate accuracy by prop type
    by_prop_type = results_df.groupby('prop_type', observed=True)['hit'].mean()

    # Accuracy split by whether the model cleared the out-of-time reliability
    # bar (models.py). The "reliable-only" number is the one worth quoting -
    # the rest are display-only tiers / TD props.
    _rel = results_df.groupby('model_reliable', observed=True)['hit'].agg(['mean', 'count'])
    by_reliable = {
        str(bool(k)): {'hit_rate': float(v['mean']), 'n': int(v['count'])}
        for k, v in _rel.iterrows()
    }
    reliable_hit_rate = by_reliable.get('True', {}).get('hit_rate')
    reliable_n = by_reliable.get('True', {}).get('n', 0)

    _say("Accuracy analysis complete:")
    _say(f"   Total predictions evaluated: {len(results_df)}")
    _say(f"   Overall hit rate: {overall_hit_rate:.1%}")
    if reliable_n:
        _say(f"   Reliable-model props only ({reliable_n}): {reliable_hit_rate:.1%}")

    return {
        'overall_accuracy': overall_hit_rate,
        'reliable_accuracy': reliable_hit_rate,
        'reliable_predictions': reliable_n,
        'by_confidence_tier': by_confidence,
        'by_prop_type': by_prop_type,
        'by_reliable': by_reliable,
        'total_predictions': len(results_df),
        'detailed_results': results_df
    }


def calculate_roi(results_df: pd.DataFrame, odds: float = -110) -> Dict:
    """
    Calculate hypothetical ROI assuming standard betting odds.

    Args:
        results_df: DataFrame with prediction results (must have 'hit' column)
        odds: American odds format (default -110)

    Returns:
        Dictionary with ROI metrics
    """
    if results_df.empty or 'hit' not in results_df.columns:
        return {
            'total_bets': 0,
            'wins': 0,
            'losses': 0,
            'hit_rate': 0.0,
            'total_wagered': 0,
            'net_profit': 0,
            'roi': 0.0,
            'breakeven_rate': 52.4
        }

    total_bets = len(results_df)
    total_hits = results_df['hit'].sum()

    # Convert American odds to decimal for calculations
    if odds < 0:  # Negative odds (e.g., -110)
        decimal_odds = (100 / abs(odds)) + 1
    else:  # Positive odds (e.g., +150)
        decimal_odds = (odds / 100) + 1

    # Calculate P&L
    # Each bet: risk $110 to win $100 (or lose $110)
    risk_amount = abs(odds) if odds < 0 else 100
    win_amount = abs(odds) if odds > 0 else 100

    total_wagered = total_bets * risk_amount
    total_won = total_hits * win_amount
    total_lost = (total_bets - total_hits) * risk_amount

    net_profit = total_won - total_lost
    roi = (net_profit / total_wagered) * 100 if total_wagered > 0 else 0

    # Calculate breakeven rate
    breakeven_rate = (risk_amount / (risk_amount + win_amount)) * 100

    return {
        'total_bets': total_bets,
        'wins': total_hits,
        'losses': total_bets - total_hits,
        'hit_rate': total_hits / total_bets if total_bets > 0 else 0,
        'total_wagered': total_wagered,
        'net_profit': net_profit,
        'roi': roi,
        'breakeven_rate': breakeven_rate
    }


def profitable_subset(results_df: pd.DataFrame, min_confidence: float = 0.65) -> pd.DataFrame:
    """
    Calculate ROI for different confidence thresholds.

    Args:
        results_df: DataFrame with prediction results
        min_confidence: Minimum confidence threshold to test

    Returns:
        DataFrame with ROI analysis by confidence threshold
    """
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]
    roi_by_threshold = {}

    for threshold in thresholds:
        subset = results_df[results_df['confidence'] >= threshold].copy()
        if len(subset) > 0:
            roi_metrics = calculate_roi(subset)
            roi_by_threshold[threshold] = roi_metrics

    return pd.DataFrame(roi_by_threshold).T


def save_accuracy_results(accuracy_metrics: Dict, week: int, filepath: Optional[str] = None,
                          season: Optional[int] = None) -> None:
    """
    Save accuracy results to a timestamped file for historical tracking.

    Only final results should be saved (see ``week_results_status``):
    ``load_accuracy_results_for_week`` serves a saved file as the week's
    result. ``season`` is recorded in the file so it's only served for that
    season.

    Args:
        accuracy_metrics: Dictionary returned by calculate_hit_rate
        week: NFL week number
        filepath: Optional custom filepath
        season: NFL season year
    """
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"data_files/accuracy_results_week{week}_{timestamp}.json"

    # Convert Series to dict for JSON serialization
    results_to_save = accuracy_metrics.copy()
    results_to_save['by_confidence_tier'] = accuracy_metrics['by_confidence_tier'].to_dict()
    results_to_save['by_prop_type'] = accuracy_metrics['by_prop_type'].to_dict()

    # Convert detailed results to dict (without the DataFrame)
    results_to_save['detailed_results'] = accuracy_metrics['detailed_results'].to_dict('records')
    if season is not None:
        results_to_save['season'] = int(season)
        results_to_save['week'] = int(week)

    import json
    with open(filepath, 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)

    _say(f"Accuracy results saved to: {filepath}")


def load_accuracy_history() -> pd.DataFrame:
    """
    Load historical accuracy results from saved files.

    Returns:
        DataFrame with historical accuracy data
    """
    import glob
    import json

    accuracy_files = glob.glob("data_files/accuracy_results_week*.json")

    if not accuracy_files:
        return pd.DataFrame()

    history_data = []

    for filepath in accuracy_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Extract week and timestamp from filename
            # Filename format: accuracy_results_week{week}_{date}_{time}.json
            # Example: accuracy_results_week18_20260110_141431.json
            import os
            filename = os.path.basename(filepath)  # Remove directory path
            parts = filename.split('_')
            
            # Extract week number from 'week18' -> 18
            week_str = parts[2]  # 'week18'
            week = int(week_str.replace('week', ''))
            
            # Parse timestamp from date and time parts
            if len(parts) >= 5:
                date_str = parts[3]  # e.g., '20260110'
                time_str = parts[4].replace('.json', '')  # e.g., '141431'
                
                # Format as readable datetime: 20260110 -> 2026-01-10, 141431 -> 14:14:31
                try:
                    formatted_timestamp = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
                except Exception:
                    formatted_timestamp = f"{date_str} {time_str}"
            else:
                formatted_timestamp = data.get('timestamp', 'Unknown')

            history_data.append({
                'week': week,
                'overall_accuracy': data.get('overall_accuracy', 0),
                'total_predictions': data.get('total_predictions', 0),
                'timestamp': formatted_timestamp
            })

        except Exception as e:
            _say(f"WARNING: could not load {filepath}: {e}")
            continue

    if not history_data:
        return pd.DataFrame()

    # Create DataFrame and deduplicate by week (keep most recent timestamp)
    df = pd.DataFrame(history_data)
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Group by week and keep the row with the most recent timestamp
    df_deduped = df.loc[df.groupby('week')['timestamp_dt'].idxmax()].copy()
    df_deduped = df_deduped.drop('timestamp_dt', axis=1)
    
    return df_deduped.sort_values('week')


def load_accuracy_results_for_week(week: int, season: int = 2025) -> Optional[Dict]:
    """
    Load the most recent saved accuracy results for a specific season and week.

    Only files that record this ``season`` are used. Older files that don't
    record a season are never served, because they can't be attributed to a
    season; the week is recalculated instead.

    Args:
        week: NFL week number
        season: NFL season year

    Returns:
        Dictionary with accuracy results, or None if not found
    """
    import glob
    import json
    import os

    # Newest first, by the YYYYMMDD_HHMMSS timestamp in the filename.
    def _timestamp(path):
        parts = os.path.basename(path)[:-len('.json')].split('_')
        return parts[-2:]

    accuracy_files = sorted(glob.glob(f"data_files/accuracy_results_week{week}_*.json"),
                            key=_timestamp, reverse=True)

    for path in accuracy_files:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            if data.get('season') != season or data.get('week', week) != week:
                continue

            # Convert back to proper format
            data['by_confidence_tier'] = pd.Series(data['by_confidence_tier'])
            data['by_prop_type'] = pd.Series(data['by_prop_type'])
            data['detailed_results'] = pd.DataFrame(data['detailed_results'])

            return data

        except Exception as e:
            _say(f"WARNING: could not load {path}: {e}")
    return None


# Example usage and testing functions
def run_weekly_accuracy_check(week: int, season: int = 2025) -> Dict:
    """
    Complete workflow to check accuracy for a given week.

    Args:
        week: NFL week number
        season: NFL season year

    Returns:
        Dictionary with complete accuracy analysis
    """
    _say(f"Running accuracy check for Week {week}, Season {season}")
    _say("=" * 60)

    # Load predictions for the week. Prefer the frozen per-week snapshot (a true
    # prospective test); fall back to the season-less name, then the latest feed.
    candidates = [
        f"data_files/player_props_predictions_week{week}_{season}.csv",
        f"data_files/player_props_predictions_week{week}.csv",
        "data_files/player_props_predictions.csv",
    ]
    predictions_file = next((c for c in candidates if Path(c).exists()), None)
    if predictions_file is None:
        _say(f"ERROR: no predictions file found for Week {week}")
        return {}
    if not predictions_file.endswith(f"week{week}_{season}.csv"):
        _say(f"WARNING: using {predictions_file} (no frozen Week {week} {season} snapshot) - "
             f"results are not a clean prospective test")

    predictions_df = pd.read_csv(predictions_file)
    _say(f"Loaded {len(predictions_df)} predictions")

    # Collect actual results
    actuals_df, error_msg = collect_actual_results(week, season)

    if actuals_df.empty:
        _say(f"ERROR: no actual results available for Week {week}: {error_msg}")
        return {}

    # Calculate accuracy
    accuracy_metrics = calculate_hit_rate(predictions_df, actuals_df)

    # Calculate ROI analysis
    if accuracy_metrics['total_predictions'] > 0:
        detailed = accuracy_metrics['detailed_results']
        roi_metrics = calculate_roi(detailed)
        accuracy_metrics['roi_analysis'] = roi_metrics

        _say("ROI analysis (at -110 odds):")
        _say(f"   Hit Rate: {roi_metrics['hit_rate']:.1%}")
        _say(f"   ROI: {roi_metrics['roi']:.1f}%")
        _say(f"   Breakeven Rate: {roi_metrics['breakeven_rate']:.1f}%")

        # Same, restricted to props from models that cleared the reliability bar.
        rel = detailed[detailed.get('model_reliable', True)]
        if len(rel):
            roi_rel = calculate_roi(rel)
            accuracy_metrics['roi_analysis_reliable'] = roi_rel
            _say(f"   [reliable models only, n={len(rel)}] "
                 f"Hit Rate: {roi_rel['hit_rate']:.1%}, ROI: {roi_rel['roi']:.1f}%")

    # Save results - only when they're final. Provisional results (a game not
    # final yet, or play-by-play still missing a game) are returned but never
    # cached, so they can't later be served as the week's result.
    final, reason = week_results_status(actuals_df, week, season)
    accuracy_metrics['final'] = final
    accuracy_metrics['provisional_reason'] = reason
    if final:
        save_accuracy_results(accuracy_metrics, week, season=season)
        _say("=" * 60)
        _say("OK: weekly accuracy check complete")
    else:
        _say("=" * 60)
        _say(f"WARNING: provisional results, not saved: {reason}")

    return accuracy_metrics


if __name__ == '__main__':
    # Example: Check accuracy for last week
    current_week = 19  # Adjust based on current NFL week
    results = run_weekly_accuracy_check(current_week)

    if results:
        _say("\nKey metrics:")
        _say(f"Overall Accuracy: {results['overall_accuracy']:.1%}")
        _say(f"Total Predictions: {results['total_predictions']}")

        if 'by_confidence_tier' in results and not results['by_confidence_tier'].empty:
            _say("\nBy confidence tier:")
            for tier, accuracy in results['by_confidence_tier'].items():
                _say(f"  {tier}: {accuracy:.1%}")
