"""scripts/model_line_shop.compute_model_edges_for_game - extends the spread
model's own nflverse-line probability to every tracked book's actual line.

Priority: extrapolate_prob's identity/monotonicity properties, and that
output labels are read straight from the tracker log's bettor-facing
home_point/away_point columns (same labeling-bug guard as
test_spread_value_finder.py's TestComputeBookEdges).
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "model_line_shop",
    Path(__file__).resolve().parent.parent / "scripts" / "model_line_shop.py",
)
mls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mls)

LOG_COLS = [
    "season", "week", "game_id", "gameday", "home_team", "away_team",
    "book_key", "book_title", "region",
    "home_point", "home_price", "away_point", "away_price",
    "home_spread_normalized",
    "home_implied_prob_devigged", "away_implied_prob_devigged",
    "nflverse_spread_line", "deviation_pts",
    "fetched_at", "source_week_snapshot",
]

PRED_COLS = ["season", "week", "game_id", "gameday", "home_team", "away_team",
             "spread_line", "prob_underdogCovered", "pred_spreadCovered_optimal"]


def _log_row(week, game_id, home_team, away_team, book_key, book_title, home_point, home_price, home_prob):
    away_point = -home_point
    return dict(
        season=2026, week=week, game_id=game_id, gameday="2026-09-20",
        home_team=home_team, away_team=away_team,
        book_key=book_key, book_title=book_title, region="us",
        home_point=home_point, home_price=home_price,
        away_point=away_point, away_price=-home_price,
        home_spread_normalized=-home_point,
        home_implied_prob_devigged=home_prob, away_implied_prob_devigged=1 - home_prob,
        nflverse_spread_line=None, deviation_pts=None,
        fetched_at="2026-09-17T00:00:00", source_week_snapshot="market_spreads_week2_2026.csv",
    )


def _pred_row(week, game_id, home_team, away_team, spread_line, prob_underdog, optimal=1,
              gameday="2099-01-01"):
    return dict(season=2026, week=week, game_id=game_id, gameday=gameday,
                home_team=home_team, away_team=away_team,
                spread_line=spread_line, prob_underdogCovered=prob_underdog,
                pred_spreadCovered_optimal=optimal)


# ---------------------------------------------------------------------------
# extrapolate_prob
# ---------------------------------------------------------------------------

class TestExtrapolateProb:
    def test_identity_same_line_returns_same_prob(self):
        for p in (0.55, 0.591285, 0.62):
            assert mls.extrapolate_prob(p, 3.5, 3.5) == pytest.approx(p, abs=1e-6)

    def test_more_points_increases_prob(self):
        base = mls.extrapolate_prob(0.591285, 3.5, 3.5)
        more = mls.extrapolate_prob(0.591285, 3.5, 4.5)
        assert more > base

    def test_fewer_points_decreases_prob(self):
        base = mls.extrapolate_prob(0.591285, 3.5, 3.5)
        fewer = mls.extrapolate_prob(0.591285, 3.5, 3.0)
        assert fewer < base

    def test_matches_hand_computed_ari_fanduel_example(self):
        # From the live analysis: model 59.1% at ARI +3.5 (nflverse) ->
        # extrapolated ~62.0% at FanDuel's actual +4.5.
        p = mls.extrapolate_prob(0.591285, 3.5, 4.5, sigma=13.5)
        assert p == pytest.approx(0.620, abs=0.001)


# ---------------------------------------------------------------------------
# compute_model_edges_for_game
# ---------------------------------------------------------------------------

def _ari_sea_fixture():
    preds = pd.DataFrame([
        _pred_row(2, "2026_02_SEA_ARI", "ARI", "SEA", spread_line=-3.5, prob_underdog=0.591285),
    ], columns=PRED_COLS)
    log = pd.DataFrame([
        _log_row(2, "2026_02_SEA_ARI", "ARI", "SEA", "draftkings", "DraftKings", 3.5, 100, 0.478),
        _log_row(2, "2026_02_SEA_ARI", "ARI", "SEA", "fanduel", "FanDuel", 4.5, -120, 0.519),
    ], columns=LOG_COLS)
    return preds, log


class TestComputeModelEdgesForGame:
    def test_home_underdog_edges(self):
        preds, log = _ari_sea_fixture()
        edges = mls.compute_model_edges_for_game("2026_02_SEA_ARI", preds, log)
        assert len(edges) == 2
        assert set(edges["underdog_team"]) == {"ARI"}
        assert set(edges["favorite_team"]) == {"SEA"}

        dk = edges[edges["book_key"] == "draftkings"].iloc[0]
        assert dk["book_line"] == pytest.approx(3.5)
        assert dk["extrapolated_prob"] == pytest.approx(0.591285, abs=1e-4)
        assert dk["edge_pts"] == pytest.approx((0.591285 - 0.478) * 100, abs=0.1)

        fd = edges[edges["book_key"] == "fanduel"].iloc[0]
        assert fd["book_line"] == pytest.approx(4.5)
        assert fd["extrapolated_prob"] == pytest.approx(0.620, abs=0.001)
        assert fd["edge_pts"] == pytest.approx((0.620 - 0.519) * 100, abs=0.15)

    def test_away_underdog_case(self):
        # spread_line positive -> home favored -> AWAY team is the underdog.
        preds = pd.DataFrame([
            _pred_row(2, "2026_02_NO_BAL", "BAL", "NO", spread_line=8.5, prob_underdog=0.569713),
        ], columns=PRED_COLS)
        log = pd.DataFrame([
            _log_row(2, "2026_02_NO_BAL", "BAL", "NO", "draftkings", "DraftKings", -8.5, -105, 0.489),
        ], columns=LOG_COLS)
        edges = mls.compute_model_edges_for_game("2026_02_NO_BAL", preds, log)
        assert len(edges) == 1
        row = edges.iloc[0]
        assert row["underdog_team"] == "NO"
        assert row["favorite_team"] == "BAL"
        # NO is away -> its bettor-facing line is the log's away_point (+8.5), not home_point.
        assert row["book_line"] == pytest.approx(8.5)

    def test_labels_match_raw_bettor_facing_columns(self):
        # Regression: output must read home_point/away_point directly, never
        # re-derive from home_spread_normalized (nflverse's internal
        # favorite-positive convention) - the exact bug class this whole
        # tool chain exists to prevent.
        preds, log = _ari_sea_fixture()
        edges = mls.compute_model_edges_for_game("2026_02_SEA_ARI", preds, log)
        dk_log_row = log[log["book_key"] == "draftkings"].iloc[0]
        dk_edge_row = edges[edges["book_key"] == "draftkings"].iloc[0]
        assert dk_edge_row["book_line"] == pytest.approx(dk_log_row["home_point"])
        assert dk_edge_row["book_line"] == pytest.approx(-dk_log_row["home_spread_normalized"])

    def test_pick_em_returns_empty(self):
        preds = pd.DataFrame([
            _pred_row(2, "2026_02_X_Y", "Y", "X", spread_line=0.0, prob_underdog=0.5),
        ], columns=PRED_COLS)
        log = pd.DataFrame([
            _log_row(2, "2026_02_X_Y", "Y", "X", "draftkings", "DraftKings", 0.0, -110, 0.5),
        ], columns=LOG_COLS)
        edges = mls.compute_model_edges_for_game("2026_02_X_Y", preds, log)
        assert edges.empty

    def test_missing_game_in_predictions_returns_empty(self):
        _, log = _ari_sea_fixture()
        edges = mls.compute_model_edges_for_game("nonexistent", pd.DataFrame(columns=PRED_COLS), log)
        assert edges.empty

    def test_missing_game_in_log_returns_empty(self):
        preds, _ = _ari_sea_fixture()
        edges = mls.compute_model_edges_for_game(
            "2026_02_SEA_ARI", preds, pd.DataFrame(columns=LOG_COLS)
        )
        assert edges.empty

    def test_missing_required_prob_skipped(self):
        preds, log = _ari_sea_fixture()
        log.loc[log["book_key"] == "fanduel", "home_implied_prob_devigged"] = None
        edges = mls.compute_model_edges_for_game("2026_02_SEA_ARI", preds, log)
        assert set(edges["book_key"]) == {"draftkings"}

    def test_missing_model_prob_returns_empty(self):
        preds, log = _ari_sea_fixture()
        preds.loc[0, "prob_underdogCovered"] = None
        edges = mls.compute_model_edges_for_game("2026_02_SEA_ARI", preds, log)
        assert edges.empty


# ---------------------------------------------------------------------------
# select_candidate_games
# ---------------------------------------------------------------------------

from datetime import date  # noqa: E402


def _mixed_gameday_fixture():
    # "today" = 2026-09-18 in every test below.
    return pd.DataFrame([
        _pred_row(2, "2026_02_PLAYED", "A", "B", spread_line=3.5, prob_underdog=0.6,
                  optimal=1, gameday="2026-09-17"),   # already played (Thursday)
        _pred_row(2, "2026_02_UPCOMING_PICK", "C", "D", spread_line=3.5, prob_underdog=0.6,
                  optimal=1, gameday="2026-09-20"),   # upcoming, qualifies as a pick
        _pred_row(2, "2026_02_UPCOMING_NO_SIGNAL", "E", "F", spread_line=3.5, prob_underdog=0.5,
                  optimal=0, gameday="2026-09-20"),   # upcoming, no signal
        _pred_row(3, "2026_03_OTHER_WEEK", "G", "H", spread_line=3.5, prob_underdog=0.6,
                  optimal=1, gameday="2026-09-27"),   # different week
    ], columns=PRED_COLS)


class TestSelectCandidateGames:
    TODAY = date(2026, 9, 18)

    def test_excludes_already_played_games_by_default(self):
        games = mls.select_candidate_games(_mixed_gameday_fixture(), today=self.TODAY)
        assert "2026_02_PLAYED" not in games
        assert "2026_02_UPCOMING_PICK" in games

    def test_include_played_flag_includes_them(self):
        games = mls.select_candidate_games(
            _mixed_gameday_fixture(), picks_only=True, include_played=True, today=self.TODAY
        )
        assert "2026_02_PLAYED" in games

    def test_picks_only_filters_out_no_signal_games(self):
        games = mls.select_candidate_games(
            _mixed_gameday_fixture(), picks_only=True, include_played=True, today=self.TODAY
        )
        assert "2026_02_UPCOMING_NO_SIGNAL" not in games

    def test_picks_only_false_includes_no_signal_games(self):
        games = mls.select_candidate_games(
            _mixed_gameday_fixture(), picks_only=False, include_played=True, today=self.TODAY
        )
        assert "2026_02_UPCOMING_NO_SIGNAL" in games

    def test_season_week_filter(self):
        games = mls.select_candidate_games(
            _mixed_gameday_fixture(), season=2026, week=2, picks_only=False,
            include_played=True, today=self.TODAY,
        )
        assert "2026_03_OTHER_WEEK" not in games
        assert set(games) == {"2026_02_PLAYED", "2026_02_UPCOMING_PICK", "2026_02_UPCOMING_NO_SIGNAL"}

    def test_empty_predictions_returns_empty(self):
        assert mls.select_candidate_games(pd.DataFrame(columns=PRED_COLS), today=self.TODAY) == []
