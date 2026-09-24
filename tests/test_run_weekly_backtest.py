"""scripts/run_weekly_backtest.latest_completed_week - finds the most
recently completed NFL week so the weekly CI backtest can call
run_weekly_accuracy_check(week, season) with a real value instead of the
missing-argument TypeError the old inline CI snippet silently ate.
"""

import importlib.util
from datetime import date
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "run_weekly_backtest",
    Path(__file__).resolve().parent.parent / "scripts" / "run_weekly_backtest.py",
)
rwb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rwb)

TODAY = date(2026, 9, 24)  # a Thursday, week 3 not yet played, week 2 fully final


def _row(week, game_id, gameday, home_score, away_score, season=2026):
    return dict(season=season, week=week, game_id=game_id, gameday=gameday,
                home_score=home_score, away_score=away_score)


class TestLatestCompletedWeek:
    def test_returns_max_fully_completed_week(self):
        df = pd.DataFrame([
            _row(1, "g1", "2026-09-10", 24, 17),
            _row(1, "g2", "2026-09-11", 20, 20),
            _row(2, "g3", "2026-09-17", 31, 10),
            _row(2, "g4", "2026-09-20", 14, 21),
            _row(3, "g5", "2026-09-27", 0, 0),   # not played yet
        ])
        assert rwb.latest_completed_week(df, season=2026, today=TODAY) == 2

    def test_partial_week_not_counted_complete(self):
        # Week 2 has one game not yet final (0-0) - the whole week doesn't count.
        df = pd.DataFrame([
            _row(1, "g1", "2026-09-10", 24, 17),
            _row(2, "g2", "2026-09-20", 31, 10),
            _row(2, "g3", "2026-09-21", 0, 0),
        ])
        assert rwb.latest_completed_week(df, season=2026, today=TODAY) == 1

    def test_zero_zero_score_not_played(self):
        # nfl-gather-data.py fillna(0)s unplayed rows - a 0-0 "final" isn't real.
        df = pd.DataFrame([_row(1, "g1", "2026-09-10", 0, 0)])
        assert rwb.latest_completed_week(df, season=2026, today=TODAY) is None

    def test_game_today_not_counted_as_played(self):
        # gameday == today -> not `< today`, treated as still upcoming even
        # though it carries a nonzero score in this synthetic case.
        df = pd.DataFrame([_row(1, "g1", TODAY.isoformat(), 24, 17)])
        assert rwb.latest_completed_week(df, season=2026, today=TODAY) is None

    def test_ignores_other_seasons(self):
        df = pd.DataFrame([
            _row(18, "g1", "2025-12-01", 24, 17, season=2025),
            _row(1, "g2", "2026-09-10", 20, 13, season=2026),
        ])
        assert rwb.latest_completed_week(df, season=2026, today=TODAY) == 1

    def test_empty_season_returns_none(self):
        df = pd.DataFrame([_row(1, "g1", "2026-09-10", 24, 17)])
        assert rwb.latest_completed_week(df, season=2099, today=TODAY) is None

    def test_empty_dataframe_returns_none(self):
        cols = ["season", "week", "game_id", "gameday", "home_score", "away_score"]
        assert rwb.latest_completed_week(pd.DataFrame(columns=cols), season=2026, today=TODAY) is None
