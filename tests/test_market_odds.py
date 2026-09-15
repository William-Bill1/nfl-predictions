"""player_props/market_odds.py - the opt-in DK/FanDuel prop-odds fetcher.

No live network calls anywhere in here: requests.get is either never reached
(unset key / cache hit) or monkeypatched to a canned response.
"""
import pandas as pd
import pytest

from player_props import market_odds as mo


# ---------------------------------------------------------------------------
# Odds math
# ---------------------------------------------------------------------------

class TestImpliedProb:
    def test_negative_odds(self):
        assert mo._implied_prob(-110) == pytest.approx(110 / 210)

    def test_positive_odds(self):
        assert mo._implied_prob(150) == pytest.approx(100 / 250)

    def test_invalid_returns_nan(self):
        assert pd.isna(mo._implied_prob(None))
        assert pd.isna(mo._implied_prob("nope"))


class TestDevig:
    def test_symmetric_minus110_both_sides(self):
        # -110/-110 is the standard no-edge book price; devigged should land at 0.5.
        p = mo._devig_over_prob(-110, -110)
        assert p == pytest.approx(0.5, abs=1e-9)

    def test_asymmetric_sides(self):
        p = mo._devig_over_prob(-150, 130)
        raw_over = mo._implied_prob(-150)
        raw_under = mo._implied_prob(130)
        assert p == pytest.approx(raw_over / (raw_over + raw_under))

    def test_missing_side_returns_none(self):
        assert mo._devig_over_prob(-110, None) is None


class TestNormalizeName:
    def test_strips_suffix_and_punctuation(self):
        assert mo._normalize_name("Patrick Mahomes II") == "patrick mahomes"
        assert mo._normalize_name("D'Andre Swift Jr.") == "dandre swift"

    def test_case_and_whitespace(self):
        assert mo._normalize_name("  Nick   Mullens ") == "nick mullens"


# ---------------------------------------------------------------------------
# find_market_line / attach_market_odds
# ---------------------------------------------------------------------------

def _odds_df():
    return pd.DataFrame([
        {"season": 2026, "week": 1, "game_id": "2026_01_JAX_DEN", "home_team": "DEN",
         "away_team": "JAX", "player_name": "Nick Mullens", "prop_type": "passing_yards",
         "book": "fanduel", "line": 210.5, "over_odds": -115, "under_odds": -105},
        {"season": 2026, "week": 1, "game_id": "2026_01_JAX_DEN", "home_team": "DEN",
         "away_team": "JAX", "player_name": "Nick Mullens", "prop_type": "passing_yards",
         "book": "draftkings", "line": 209.5, "over_odds": -110, "under_odds": -110},
        {"season": 2026, "week": 1, "game_id": "2026_01_JAX_DEN", "home_team": "DEN",
         "away_team": "JAX", "player_name": "Travis Etienne Jr.", "prop_type": "rushing_yards",
         "book": "draftkings", "line": 55.5, "over_odds": -120, "under_odds": 100},
    ], columns=mo.ODDS_COLUMNS)


class TestFindMarketLine:
    def test_exact_match(self):
        r = mo.find_market_line("Nick Mullens", "JAX", "passing_yards", _odds_df())
        assert r is not None and r["line"] in (209.5, 210.5)

    def test_prefers_draftkings_over_fanduel(self):
        # Both books have Mullens' passing_yards line - DK is first in BOOKMAKERS.
        r = mo.find_market_line("Nick Mullens", "JAX", "passing_yards", _odds_df())
        assert r["book"] == "draftkings"
        assert r["line"] == 209.5

    def test_normalized_suffix_match(self):
        # Query without "Jr." should still hit the "Travis Etienne Jr." row.
        r = mo.find_market_line("Travis Etienne", "JAX", "rushing_yards", _odds_df())
        assert r is not None and r["book"] == "draftkings"

    def test_wrong_prop_type_unmatched(self):
        assert mo.find_market_line("Nick Mullens", "JAX", "receptions", _odds_df()) is None

    def test_unknown_player_unmatched(self):
        assert mo.find_market_line("Nobody Fakename", "JAX", "passing_yards", _odds_df()) is None

    def test_empty_or_none_df(self):
        assert mo.find_market_line("Nick Mullens", "JAX", "passing_yards", None) is None
        assert mo.find_market_line("Nick Mullens", "JAX", "passing_yards", pd.DataFrame()) is None


class TestAttachMarketOdds:
    def test_with_match(self):
        pred = {"prob_over": 0.60, "line_value": 200.0}
        info = {"line": 209.5, "book": "draftkings", "over_odds": -110,
                "under_odds": -110, "market_implied_prob": 0.5}
        out = mo.attach_market_odds(dict(pred), info)
        assert out["market_line"] == 209.5
        assert out["market_book"] == "draftkings"
        assert out["market_edge"] == pytest.approx(0.10)
        assert out["market_line_available"] is True
        # fixed-tier fields untouched
        assert out["line_value"] == 200.0 and out["prob_over"] == 0.60

    def test_without_match_marks_unavailable(self):
        pred = {"prob_over": 0.60, "line_value": 200.0}
        out = mo.attach_market_odds(dict(pred), None)
        assert out["market_line_available"] is False
        assert out["market_line"] is None and out["market_edge"] is None
        assert out["line_value"] == 200.0 and out["prob_over"] == 0.60


# ---------------------------------------------------------------------------
# fetch_market_odds - no network unless explicitly mocked
# ---------------------------------------------------------------------------

class TestFetchMarketOdds:
    def test_no_key_returns_empty_without_network(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mo, "ODDS_API_KEY", "")
        monkeypatch.setattr(mo, "DATA_DIR", tmp_path)

        def _boom(*a, **kw):
            raise AssertionError("requests.get should not be called with no API key")
        monkeypatch.setattr(mo.requests, "get", _boom)

        schedule = pd.DataFrame([{"home_team": "DEN", "away_team": "JAX"}])
        out = mo.fetch_market_odds(2026, 1, schedule)
        assert out.empty
        assert list(out.columns) == mo.ODDS_COLUMNS

    def test_cache_hit_returns_cached_without_network(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mo, "ODDS_API_KEY", "fake-key")
        monkeypatch.setattr(mo, "DATA_DIR", tmp_path)
        cached = pd.DataFrame([{
            "season": 2026, "week": 1, "game_id": "2026_01_JAX_DEN", "home_team": "DEN",
            "away_team": "JAX", "player_name": "Nick Mullens", "prop_type": "passing_yards",
            "book": "draftkings", "line": 209.5, "over_odds": -110, "under_odds": -110,
        }])
        cached.to_csv(mo._cache_path(2026, 1), index=False)

        def _boom(*a, **kw):
            raise AssertionError("requests.get should not be called on a cache hit")
        monkeypatch.setattr(mo.requests, "get", _boom)

        schedule = pd.DataFrame([{"home_team": "DEN", "away_team": "JAX"}])
        out = mo.fetch_market_odds(2026, 1, schedule)
        assert len(out) == 1
        assert out.iloc[0]["player_name"] == "Nick Mullens"

    def test_credit_floor_stops_mid_run(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mo, "ODDS_API_KEY", "fake-key")
        monkeypatch.setattr(mo, "DATA_DIR", tmp_path)
        monkeypatch.setattr(mo, "ODDS_API_MIN_REMAINING", 20)

        events = [
            {"id": "ev1", "home_team": "Denver Broncos", "away_team": "Jacksonville Jaguars"},
            {"id": "ev2", "home_team": "Green Bay Packers", "away_team": "New York Jets"},
        ]
        monkeypatch.setattr(mo, "_list_events", lambda: events)

        calls = []
        def _fetch(event_id):
            calls.append(event_id)
            # First call already reports we're under the floor -> loop must stop
            # before issuing a second call.
            return {"bookmakers": []}, 5
        monkeypatch.setattr(mo, "_fetch_event_odds", _fetch)

        schedule = pd.DataFrame([
            {"home_team": "DEN", "away_team": "JAX"},
            {"home_team": "GB", "away_team": "NYJ"},
        ])
        out = mo.fetch_market_odds(2026, 1, schedule)
        assert calls == ["ev1"]  # stopped after the first call, not both
        assert out.empty  # no outcomes in the canned response, but no crash


# ---------------------------------------------------------------------------
# Team map
# ---------------------------------------------------------------------------

def test_team_map_covers_prop_predictions_teams():
    import pandas as pd
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "data_files" / "player_props_predictions.csv"
    if not path.exists():
        pytest.skip("player_props_predictions.csv not present in this checkout")
    df = pd.read_csv(path, usecols=["team", "opponent"])
    teams = set(df["team"].dropna()) | set(df["opponent"].dropna())
    missing = teams - set(mo.TEAM_FULL_NAME)
    assert not missing, f"team abbreviations with no TEAM_FULL_NAME entry: {missing}"
