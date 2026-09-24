"""Run the weekly player-prop accuracy backtest for the most recently
completed NFL week, and save the results.

Standalone script rather than an inline CI snippet, because
run_weekly_accuracy_check(week, season) requires an explicit week - there is
no "just figure out last week" convenience path in player_props/backtest.py.
The previous inline `python -c "..."` in weekly-model-performance.yml called
it with no arguments at all (`TypeError: missing 1 required positional
argument: 'week'`), and separately called save_accuracy_results(results)
with no week either (same error class) - both silently swallowed by that
step's `continue-on-error: true`, so the workflow reported "Success" on
every Monday run while never actually committing anything. Caught
2026-09-24 when data_files/spread_performance.json on main turned out to be
stuck at 9/14 numbers through two more weeks of real results.

CLI: python scripts/run_weekly_backtest.py
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from season_utils import upcoming_or_current_season  # noqa: E402
from player_props.backtest import run_weekly_accuracy_check, save_accuracy_results  # noqa: E402

DATA_DIR = "data_files"
PREDICTIONS_PATH = os.path.join(DATA_DIR, "nfl_games_historical_with_predictions.csv")


def latest_completed_week(predictions_df: pd.DataFrame, season: int, today: date | None = None) -> int | None:
    """The most recent NFL week in `season` where every game has a real
    final score.

    A game counts as played when `gameday < today` AND
    `home_score + away_score > 0` - mirrors betting_log.grade_pending's
    exact convention, since nfl-gather-data.py fillna(0)s unplayed rows so
    "scores present" alone isn't enough (a NaN-vs-0 check would miss it).
    Returns None if no week in the season is fully complete yet (e.g.
    pre-season, or the first week is still in progress).
    """
    df = predictions_df[predictions_df["season"] == season]
    if df.empty:
        return None
    gameday = pd.to_datetime(df["gameday"], errors="coerce")
    cutoff = pd.to_datetime(today or date.today())
    total_score = df["home_score"].fillna(0) + df["away_score"].fillna(0)
    played = (gameday < cutoff) & (total_score > 0)
    week_fully_played = played.groupby(df["week"]).all()
    completed_weeks = week_fully_played[week_fully_played].index
    return int(completed_weeks.max()) if len(completed_weeks) else None


def _load_predictions() -> pd.DataFrame:
    # Despite the .csv extension this file is TAB-separated - see
    # betting_log.py's own PREDICTIONS_PATH read.
    return pd.read_csv(PREDICTIONS_PATH, sep="\t")


def main() -> None:
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"[weekly_backtest] {PREDICTIONS_PATH} not found")
        raise SystemExit(1)

    predictions_df = _load_predictions()
    season = upcoming_or_current_season()
    week = latest_completed_week(predictions_df, season)

    if week is None:
        print(f"[weekly_backtest] no fully completed week yet for season {season} "
              "(off-season, or this week's games are still in progress) - nothing to backtest")
        return

    print(f"[weekly_backtest] running accuracy check for season {season}, week {week}")
    results = run_weekly_accuracy_check(week, season)
    if results:
        save_accuracy_results(results, week)
        print("[weekly_backtest] backtest complete:", results)
    else:
        print(f"[weekly_backtest] no backtest results returned for week {week}")


if __name__ == "__main__":
    main()
