"""spread_tracker.py - the opt-in season-long US+CA sportsbook spread-line
tracker.

No live network calls anywhere in here: requests.get is either never reached
(unset key / cache hit / under credit floor) or monkeypatched to a canned
response.
"""
import pandas as pd
import pytest

import spread_tracker as st


# ---------------------------------------------------------------------------
# Sign-convention conversion - highest-risk piece, a bug here silently
# corrupts every deviation number without ever raising an error.
# ---------------------------------------------------------------------------

class TestNormalizeSpread:
    def test_home_favorite_book_quote_becomes_positive_spread_line(self):
        # Empirical anchor: home favored by 9.5 <=> spread_line == 9.5 <=>
        # book quotes home at -9.5.
        assert st._normalize_spread(-9.5) == pytest.approx(9.5)

    def test_away_favorite_book_quote_becomes_negative_spread_line(self):
        # Empirical anchor: spread_line == -3.0 <=> away favored <=> book
        # quotes home (the underdog) at +3.0.
        assert st._normalize_spread(3.0) == pytest.approx(-3.0)

    def test_pick_em_is_zero(self):
        assert st._normalize_spread(0.0) == pytest.approx(0.0)


class TestRegionForBook:
    @pytest.mark.parametrize("key,expected", [
        ("draftkings", "us"),
        ("fanduel", "us"),
        ("betmgm", "us"),
        ("betmgm_ca_on", "ca"),
        ("playnow_ca", "ca"),
        ("sportsinteraction_ca_on", "ca"),
        ("proline_ca_on", "ca"),
        ("pointsbetca", "us"),  # no literal "_ca" substring - documents the heuristic's limit
        ("", "us"),
    ])
    def test_classification(self, key, expected):
        assert st._region_for_book(key) == expected


# ---------------------------------------------------------------------------
# _parse_bulk_spreads
# ---------------------------------------------------------------------------

def _bulk_events_json():
    return [
        {
            "home_team": "Baltimore Ravens", "away_team": "New Orleans Saints",
            "bookmakers": [
                {
                    "key": "draftkings", "title": "DraftKings",
                    "markets": [{"key": "spreads", "outcomes": [
                        {"name": "Baltimore Ravens", "price": -110, "point": -8.5},
                        {"name": "New Orleans Saints", "price": -110, "point": 8.5},
                    ]}],
                },
                {
                    "key": "playnow_ca", "title": "PlayNow",
                    "markets": [{"key": "spreads", "outcomes": [
                        {"name": "Baltimore Ravens", "price": -115, "point": -7.0},
                        {"name": "New Orleans Saints", "price": -105, "point": 7.0},
                    ]}],
                },
                {
                    # No spreads market for this book - should be skipped, not error.
                    "key": "some_book", "title": "Some Book",
                    "markets": [{"key": "h2h", "outcomes": []}],
                },
            ],
        },
        {
            # Game not in our schedule map - should be dropped entirely.
            "home_team": "Kansas City Chiefs", "away_team": "Denver Broncos",
            "bookmakers": [{"key": "draftkings", "title": "DraftKings",
                             "markets": [{"key": "spreads", "outcomes": [
                                 {"name": "Kansas City Chiefs", "price": -110, "point": -3.0},
                                 {"name": "Denver Broncos", "price": -110, "point": 3.0},
                             ]}]}],
        },
    ]


def _game_id_by_teams():
    return {("Baltimore Ravens", "New Orleans Saints"): "2026_02_NO_BAL"}


class TestParseBulkSpreads:
    def test_row_count_and_matching(self):
        rows = st._parse_bulk_spreads(_bulk_events_json(), 2026, 2, _game_id_by_teams())
        # 2 books with a spreads market on the matched game; unmatched game and
        # the h2h-only book are excluded.
        assert len(rows) == 2
        assert {r["book_key"] for r in rows} == {"draftkings", "playnow_ca"}
        assert all(r["game_id"] == "2026_02_NO_BAL" for r in rows)
        assert all(r["home_team"] == "BAL" and r["away_team"] == "NO" for r in rows)

    def test_normalized_values_and_region(self):
        rows = st._parse_bulk_spreads(_bulk_events_json(), 2026, 2, _game_id_by_teams())
        dk = next(r for r in rows if r["book_key"] == "draftkings")
        pn = next(r for r in rows if r["book_key"] == "playnow_ca")
        assert dk["home_spread_normalized"] == pytest.approx(8.5)
        assert dk["region"] == "us"
        assert pn["home_spread_normalized"] == pytest.approx(7.0)
        assert pn["region"] == "ca"


# ---------------------------------------------------------------------------
# attach_nflverse_comparison
# ---------------------------------------------------------------------------

def _raw_spreads_df():
    return pd.DataFrame([
        {"season": 2026, "week": 2, "game_id": "2026_02_NO_BAL", "home_team": "BAL",
         "away_team": "NO", "book_key": "draftkings", "book_title": "DraftKings",
         "region": "us", "home_point": -8.5, "home_price": -110, "away_point": 8.5,
         "away_price": -110, "home_spread_normalized": 8.5, "fetched_at": "2026-09-16T00:00:00"},
    ], columns=st.RAW_COLUMNS)


class TestAttachNflverseComparison:
    def test_deviation_zero_when_book_matches_nflverse(self):
        preds = pd.DataFrame([
            {"game_id": "2026_02_NO_BAL", "gameday": "2026-09-21", "spread_line": 8.5},
        ])
        out = st.attach_nflverse_comparison(_raw_spreads_df(), preds)
        assert len(out) == 1
        assert out.iloc[0]["nflverse_spread_line"] == pytest.approx(8.5)
        assert out.iloc[0]["deviation_pts"] == pytest.approx(0.0)
        assert out.iloc[0]["home_implied_prob_devigged"] == pytest.approx(0.5, abs=1e-9)

    def test_unmatched_game_id_yields_nan_not_error(self):
        preds = pd.DataFrame([
            {"game_id": "2026_02_SOMETHING_ELSE", "gameday": "2026-09-21", "spread_line": 3.0},
        ])
        out = st.attach_nflverse_comparison(_raw_spreads_df(), preds)
        assert len(out) == 1
        assert pd.isna(out.iloc[0]["nflverse_spread_line"])
        assert pd.isna(out.iloc[0]["deviation_pts"])

    def test_empty_input_returns_empty_with_log_columns(self):
        out = st.attach_nflverse_comparison(pd.DataFrame(columns=st.RAW_COLUMNS), pd.DataFrame())
        assert out.empty
        assert list(out.columns) == st.LOG_COLUMNS


# ---------------------------------------------------------------------------
# upsert_weekly_spreads
# ---------------------------------------------------------------------------

def _log_row(week=2, book_key="draftkings", home_point_normalized=8.5):
    return pd.DataFrame([{
        "season": 2026, "week": week, "game_id": "2026_02_NO_BAL", "gameday": "2026-09-21",
        "home_team": "BAL", "away_team": "NO", "book_key": book_key, "book_title": book_key,
        "region": "us", "home_point": -home_point_normalized, "home_price": -110,
        "away_point": home_point_normalized, "away_price": -110,
        "home_spread_normalized": home_point_normalized,
        "home_implied_prob_devigged": 0.5, "away_implied_prob_devigged": 0.5,
        "nflverse_spread_line": home_point_normalized, "deviation_pts": 0.0,
        "fetched_at": "2026-09-16T00:00:00", "source_week_snapshot": "market_spreads_week2_2026.csv",
    }], columns=st.LOG_COLUMNS)


class TestUpsertWeeklySpreads:
    def test_first_run_inserts(self, tmp_path):
        log_path = tmp_path / "spread_tracker_log.csv"
        n = st.upsert_weekly_spreads(_log_row(), log_path=log_path)
        assert n == 1
        out = pd.read_csv(log_path)
        assert len(out) == 1

    def test_rerun_same_key_overwrites_not_duplicates(self, tmp_path):
        log_path = tmp_path / "spread_tracker_log.csv"
        st.upsert_weekly_spreads(_log_row(home_point_normalized=8.5), log_path=log_path)
        st.upsert_weekly_spreads(_log_row(home_point_normalized=9.0), log_path=log_path)
        out = pd.read_csv(log_path)
        assert len(out) == 1  # same (season, week, game_id, book_key) -> replaced, not appended
        assert out.iloc[0]["home_spread_normalized"] == pytest.approx(9.0)

    def test_new_book_same_game_appends(self, tmp_path):
        log_path = tmp_path / "spread_tracker_log.csv"
        st.upsert_weekly_spreads(_log_row(book_key="draftkings"), log_path=log_path)
        st.upsert_weekly_spreads(_log_row(book_key="fanduel"), log_path=log_path)
        out = pd.read_csv(log_path)
        assert len(out) == 2
        assert set(out["book_key"]) == {"draftkings", "fanduel"}

    def test_empty_input_is_noop(self, tmp_path):
        log_path = tmp_path / "spread_tracker_log.csv"
        n = st.upsert_weekly_spreads(pd.DataFrame(columns=st.LOG_COLUMNS), log_path=log_path)
        assert n == 0
        assert not log_path.exists()


# ---------------------------------------------------------------------------
# fetch_weekly_spreads - no network unless explicitly mocked
# ---------------------------------------------------------------------------

class TestFetchWeeklySpreads:
    def test_no_key_returns_empty_without_network(self, monkeypatch, tmp_path):
        monkeypatch.setattr(st, "ODDS_API_KEY", "")
        monkeypatch.setattr(st, "DATA_DIR", tmp_path)

        def _boom(*a, **kw):
            raise AssertionError("requests.get should not be called with no API key")
        monkeypatch.setattr(st.requests, "get", _boom)

        schedule = pd.DataFrame([{"home_team": "BAL", "away_team": "NO"}])
        out = st.fetch_weekly_spreads(2026, 2, schedule)
        assert out.empty
        assert list(out.columns) == st.RAW_COLUMNS

    def test_cache_hit_returns_cached_without_network(self, monkeypatch, tmp_path):
        monkeypatch.setattr(st, "ODDS_API_KEY", "fake-key")
        monkeypatch.setattr(st, "DATA_DIR", tmp_path)
        cached = pd.DataFrame([{
            "season": 2026, "week": 2, "game_id": "2026_02_NO_BAL", "home_team": "BAL",
            "away_team": "NO", "book_key": "draftkings", "book_title": "DraftKings",
            "region": "us", "home_point": -8.5, "home_price": -110, "away_point": 8.5,
            "away_price": -110, "home_spread_normalized": 8.5, "fetched_at": "2026-09-16T00:00:00",
        }])
        cached.to_csv(st._cache_path(2026, 2), index=False)

        def _boom(*a, **kw):
            raise AssertionError("requests.get should not be called on a cache hit")
        monkeypatch.setattr(st.requests, "get", _boom)

        schedule = pd.DataFrame([{"home_team": "BAL", "away_team": "NO"}])
        out = st.fetch_weekly_spreads(2026, 2, schedule)
        assert len(out) == 1
        assert out.iloc[0]["book_key"] == "draftkings"

    def test_credit_floor_skips_call_entirely(self, monkeypatch, tmp_path):
        monkeypatch.setattr(st, "ODDS_API_KEY", "fake-key")
        monkeypatch.setattr(st, "DATA_DIR", tmp_path)
        monkeypatch.setattr(st, "ODDS_API_MIN_REMAINING", 20)
        # Simulate a prior run that already observed low remaining credits -
        # unlike market_odds.py (many calls, can stop mid-loop), this module
        # makes exactly one call per run, so the floor must be checked BEFORE
        # that call using what a previous run last saw.
        monkeypatch.setattr(st, "_last_known_remaining", lambda: 5)

        def _boom(*a, **kw):
            raise AssertionError("requests.get should not be called under the credit floor")
        monkeypatch.setattr(st.requests, "get", _boom)

        schedule = pd.DataFrame([{"home_team": "BAL", "away_team": "NO"}])
        out = st.fetch_weekly_spreads(2026, 2, schedule)
        assert out.empty

    def test_empty_schedule_returns_empty_without_network(self, monkeypatch, tmp_path):
        monkeypatch.setattr(st, "ODDS_API_KEY", "fake-key")
        monkeypatch.setattr(st, "DATA_DIR", tmp_path)

        def _boom(*a, **kw):
            raise AssertionError("requests.get should not be called with an empty schedule")
        monkeypatch.setattr(st.requests, "get", _boom)

        out = st.fetch_weekly_spreads(2026, 2, pd.DataFrame())
        assert out.empty


# ---------------------------------------------------------------------------
# Credits state persistence
# ---------------------------------------------------------------------------

class TestCreditsState:
    def test_round_trip(self, monkeypatch, tmp_path):
        monkeypatch.setattr(st, "DATA_DIR", tmp_path)
        assert st._last_known_remaining() is None
        st._persist_remaining(42)
        assert st._last_known_remaining() == 42

    def test_persist_none_is_noop(self, monkeypatch, tmp_path):
        monkeypatch.setattr(st, "DATA_DIR", tmp_path)
        st._persist_remaining(None)
        assert st._last_known_remaining() is None

    def test_corrupt_state_file_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(st, "DATA_DIR", tmp_path)
        st._credits_state_path().parent.mkdir(parents=True, exist_ok=True)
        st._credits_state_path().write_text("not-a-number")
        assert st._last_known_remaining() is None
