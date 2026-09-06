"""Tests for betting_log - headless management of betting_recommendations_log.csv."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

import betting_log as bl


@pytest.mark.parametrize("prob,tier", [
    (0.49, "Lean"), (0.50, "Lean"), (0.549, "Lean"),
    (0.55, "Good"), (0.589, "Good"),
    (0.59, "Strong"), (0.649, "Strong"),
    (0.65, "Elite"), (0.90, "Elite"),
])
def test_spread_tier_cutoffs(prob, tier):
    assert bl._spread_tier(prob) == tier


def _preds(rows):
    """Minimal predictions frame with the columns grade_pending / append need."""
    cols = {
        "game_id": [], "season": [], "week": [], "gameday": [],
        "home_team": [], "away_team": [], "spread_line": [], "total_line": [],
        "home_score": [], "away_score": [], "underdogCovered": [], "spreadPush": [],
        "pred_spreadCovered_optimal": [], "prob_underdogCovered": [],
        "edge_underdog_spread": [],
    }
    for r in rows:
        for k in cols:
            cols[k].append(r.get(k, 0))
    return pd.DataFrame(cols)


class TestGradePending:
    def _log(self, tmp_path, game_id, bet_type="spread"):
        p = tmp_path / "log.csv"
        row = {c: "" for c in bl.LOG_COLUMNS}
        row.update(game_id=game_id, bet_type=bet_type, spread_line=3.0,
                   bet_result="pending", week=1)
        pd.DataFrame([row], columns=bl.LOG_COLUMNS).to_csv(p, index=False)
        return str(p)

    def test_underdog_cover_is_a_win(self, tmp_path):
        log = self._log(tmp_path, "2025_18_A_B")
        preds = _preds([dict(game_id="2025_18_A_B", gameday="2025-01-01",
                             home_score=16, away_score=14, underdogCovered=1,
                             spreadPush=0)])
        assert bl.grade_pending(preds, log) == 1
        out = pd.read_csv(log).iloc[0]
        assert out.bet_result == "win"
        assert out.bet_profit == pytest.approx(bl.WIN_PROFIT)
        assert out.actual_home_score == 16 and out.actual_away_score == 14

    def test_favorite_cover_is_a_loss(self, tmp_path):
        log = self._log(tmp_path, "2025_18_A_B")
        preds = _preds([dict(game_id="2025_18_A_B", gameday="2025-01-01",
                             home_score=30, away_score=3, underdogCovered=0,
                             spreadPush=0)])
        assert bl.grade_pending(preds, log) == 1
        out = pd.read_csv(log).iloc[0]
        assert out.bet_result == "loss"
        assert out.bet_profit == pytest.approx(bl.LOSS_PROFIT)

    def test_push_pays_zero(self, tmp_path):
        log = self._log(tmp_path, "2025_18_A_B")
        preds = _preds([dict(game_id="2025_18_A_B", gameday="2025-01-01",
                             home_score=23, away_score=20, underdogCovered=0,
                             spreadPush=1)])
        assert bl.grade_pending(preds, log) == 1
        out = pd.read_csv(log).iloc[0]
        assert out.bet_result == "push"
        assert out.bet_profit == 0.0

    def test_unplayed_zero_zero_game_is_not_graded(self, tmp_path):
        # nfl-gather-data.py fillna(0)s unplayed rows - a 0-0 "final" in the
        # future must stay pending, not be scored as a loss.
        log = self._log(tmp_path, "2026_01_A_B")
        future = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        preds = _preds([dict(game_id="2026_01_A_B", gameday=future,
                             home_score=0, away_score=0, underdogCovered=0,
                             spreadPush=0)])
        assert bl.grade_pending(preds, log) == 0
        assert pd.read_csv(log).iloc[0].bet_result == "pending"


class TestAppendRecommendations:
    def test_logs_only_near_term_signals_and_dedupes(self, tmp_path):
        log = str(tmp_path / "log.csv")
        soon = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
        far = (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d")
        preds = _preds([
            dict(game_id="2026_01_A_B", gameday=soon, home_team="B", away_team="A",
                 spread_line=3.0, pred_spreadCovered_optimal=1,
                 prob_underdogCovered=0.61),
            dict(game_id="2026_08_C_D", gameday=far, home_team="D", away_team="C",
                 spread_line=3.0, pred_spreadCovered_optimal=1,
                 prob_underdogCovered=0.61),
        ])
        assert bl.append_recommendations(preds, log) == 1
        assert bl.append_recommendations(preds, log) == 0  # idempotent
        out = pd.read_csv(log)
        assert list(out.game_id) == ["2026_01_A_B"]
        assert out.iloc[0].confidence_tier == "Strong"  # 0.59-0.65
