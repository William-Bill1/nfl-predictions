"""scripts/spread_value_finder.compute_book_edges - price-adjusted fair-value
edges for one tracked sportsbook's spread lines.

Highest priority: the labeling regression this module exists to fix - output
must use standard bettor-facing spread notation (favorite negative), read
straight from the log's own home_point/away_point columns, never re-derived
from spread_tracker.py's home-favorite-positive convention (that mismatch
mislabeled a side during a manual version of this analysis, 2026-09-17).
"""

import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "spread_value_finder",
    Path(__file__).resolve().parent.parent / "scripts" / "spread_value_finder.py",
)
svf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(svf)

COLS = [
    "season", "week", "game_id", "gameday", "home_team", "away_team",
    "book_key", "book_title", "region",
    "home_point", "home_price", "away_point", "away_price",
    "home_spread_normalized",
    "home_implied_prob_devigged", "away_implied_prob_devigged",
    "nflverse_spread_line", "deviation_pts",
    "fetched_at", "source_week_snapshot",
]


def _row(week, game_id, home_team, away_team, book_key, book_title,
         home_point, home_price, home_prob_devigged):
    away_point = -home_point
    return dict(
        season=2026, week=week, game_id=game_id, gameday="2026-09-20",
        home_team=home_team, away_team=away_team,
        book_key=book_key, book_title=book_title, region="us",
        home_point=home_point, home_price=home_price,
        away_point=away_point, away_price=-home_price,
        home_spread_normalized=-home_point,  # nflverse convention: -1 * book's own home_point
        home_implied_prob_devigged=home_prob_devigged,
        away_implied_prob_devigged=1 - home_prob_devigged,
        nflverse_spread_line=None, deviation_pts=None,
        fetched_at="2026-09-17T00:00:00", source_week_snapshot="market_spreads_week2_2026.csv",
    )


# ---------------------------------------------------------------------------
# _normal_cdf / _fair_prob_home_covers
# ---------------------------------------------------------------------------

class TestNormalCdf:
    def test_matches_known_values(self):
        assert svf._normal_cdf(0.0) == pytest.approx(0.5)
        assert svf._normal_cdf(1.959964) == pytest.approx(0.975, abs=1e-4)
        assert svf._normal_cdf(-1.959964) == pytest.approx(0.025, abs=1e-4)


class TestNormalPpf:
    def test_is_inverse_of_cdf(self):
        for p in (0.025, 0.1, 0.3, 0.5, 0.591285, 0.9, 0.975):
            x = svf._normal_ppf(p)
            assert svf._normal_cdf(x) == pytest.approx(p, abs=1e-6)

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            svf._normal_ppf(0.0)
        with pytest.raises(ValueError):
            svf._normal_ppf(1.0)


class TestFairProbHomeCovers:
    def test_book_line_matches_field_is_exactly_half(self):
        # When the book's own line equals the field's median, that line is by
        # definition the 50/50 point of its own distribution.
        assert svf._fair_prob_home_covers(mu_field=-3.5, book_home_spread_normalized=-3.5) == pytest.approx(0.5)

    def test_matches_hand_computed_gb_nyj_example(self):
        # From the live GB@NYJ analysis: field fair home (NYJ) line -3.5,
        # PlayNow's home line -7.0 -> fair prob NYJ covers ~60.2%.
        p = svf._fair_prob_home_covers(mu_field=-3.5, book_home_spread_normalized=-7.0, sigma=13.5)
        assert p == pytest.approx(0.602, abs=0.001)


# ---------------------------------------------------------------------------
# compute_book_edges
# ---------------------------------------------------------------------------

def _gb_nyj_fixture():
    # 3 field books clustered at NYJ -3.5 (home favored... i.e. GB favored by
    # 3.5, home_point convention: home team is NYJ, so home_point=+3.5 means
    # NYJ getting 3.5, i.e. away GB is favorite) + PlayNow at NYJ +7.0 / GB -7.0.
    return pd.DataFrame([
        _row(2, "2026_02_GB_NYJ", "NYJ", "GB", "draftkings", "DraftKings", 3.5, -108, 0.4957),
        _row(2, "2026_02_GB_NYJ", "NYJ", "GB", "fanduel", "FanDuel", 3.5, -106, 0.4913),
        _row(2, "2026_02_GB_NYJ", "NYJ", "GB", "betmgm", "BetMGM", 3.5, -110, 0.5022),
        _row(2, "2026_02_GB_NYJ", "NYJ", "GB", "playnow_ca", "PlayNow (CA)", 7.0, -227, 0.6344),
    ], columns=COLS)


class TestComputeBookEdges:
    def test_edges_match_hand_computed_gb_nyj_example(self):
        edges = svf.compute_book_edges(_gb_nyj_fixture(), "playnow_ca")
        assert len(edges) == 2  # home + away rows for the one game

        home = edges[edges["side"] == "home"].iloc[0]
        away = edges[edges["side"] == "away"].iloc[0]

        assert home["team"] == "NYJ"
        assert home["fair_prob"] == pytest.approx(0.602, abs=0.001)
        assert home["required_prob"] == pytest.approx(0.6344, abs=1e-4)
        assert home["edge_pts"] == pytest.approx(-3.2, abs=0.1)

        assert away["team"] == "GB"
        assert away["fair_prob"] == pytest.approx(0.398, abs=0.001)
        assert away["required_prob"] == pytest.approx(0.3656, abs=1e-4)
        assert away["edge_pts"] == pytest.approx(3.2, abs=0.1)

    def test_labels_match_raw_bettor_facing_columns(self):
        # Regression: output must be read straight from home_point/away_point
        # (standard favorite-negative notation), not re-derived from
        # home_spread_normalized (nflverse's favorite-positive convention) -
        # that mismatch mislabeled a side during a manual version of this
        # analysis. Assert the edge-finder's "line" column matches the raw
        # log columns exactly, and is the sign-flip of home_spread_normalized.
        df = _gb_nyj_fixture()
        pn_row = df[df["book_key"] == "playnow_ca"].iloc[0]
        edges = svf.compute_book_edges(df, "playnow_ca")

        home = edges[edges["side"] == "home"].iloc[0]
        away = edges[edges["side"] == "away"].iloc[0]
        assert home["line"] == pytest.approx(pn_row["home_point"])
        assert away["line"] == pytest.approx(pn_row["away_point"])
        assert home["line"] == pytest.approx(-pn_row["home_spread_normalized"])

    def test_insufficient_field_books_skips_game(self):
        # Only 1 other book quoted this game - below MIN_FIELD_BOOKS (2).
        df = pd.DataFrame([
            _row(2, "2026_02_X_Y", "Y", "X", "draftkings", "DraftKings", 3.5, -110, 0.5),
            _row(2, "2026_02_X_Y", "Y", "X", "playnow_ca", "PlayNow (CA)", 7.0, -227, 0.6344),
        ], columns=COLS)
        edges = svf.compute_book_edges(df, "playnow_ca")
        assert edges.empty

    def test_missing_devigged_prob_skips_row(self):
        df = _gb_nyj_fixture()
        df.loc[df["book_key"] == "playnow_ca", "home_implied_prob_devigged"] = None
        edges = svf.compute_book_edges(df, "playnow_ca")
        assert edges.empty

    def test_unknown_book_returns_empty(self):
        edges = svf.compute_book_edges(_gb_nyj_fixture(), "nonexistent_book")
        assert edges.empty

    def test_empty_log_returns_empty(self):
        edges = svf.compute_book_edges(pd.DataFrame(columns=COLS), "playnow_ca")
        assert edges.empty

    def test_week_filter(self):
        week3_rows = pd.DataFrame([
            _row(3, "2026_03_A_B", "B", "A", "draftkings", "DraftKings", 3.5, -110, 0.5),
            _row(3, "2026_03_A_B", "B", "A", "fanduel", "FanDuel", 3.0, -110, 0.5),
            _row(3, "2026_03_A_B", "B", "A", "playnow_ca", "PlayNow (CA)", 5.0, -150, 0.6),
        ], columns=COLS)
        df = pd.concat([_gb_nyj_fixture(), week3_rows], ignore_index=True)
        edges_wk3 = svf.compute_book_edges(df, "playnow_ca", week=3)
        assert set(edges_wk3["week"]) == {3}
