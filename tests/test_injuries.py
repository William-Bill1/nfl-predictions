"""player_props/injuries.py - ESPN JSON injury feed parsing and player matching.

No live network calls: requests.get is monkeypatched to canned responses.
"""
import pandas as pd
import pytest
import requests

from player_props import injuries as inj


def _payload():
    return {
        "injuries": [
            {"displayName": "Washington Commanders", "injuries": [
                {"status": "Out", "details": {"type": "Elbow"},
                 "athlete": {"displayName": "Jayden Daniels", "position": {"abbreviation": "QB"}}},
            ]},
            {"displayName": "Carolina Panthers", "injuries": [
                {"status": "Injured Reserve", "details": {"type": "Knee"},
                 "athlete": {"displayName": "Jonathon Brooks", "position": {"abbreviation": "RB"}}},
                {"status": "Active", "details": {},
                 "athlete": {"displayName": "James Cook III", "position": {"abbreviation": "RB"}}},
                {"status": "Out", "athlete": {"displayName": ""}},  # no name -> dropped
            ]},
            {"displayName": "Buffalo Bills", "injuries": [
                {"status": "Questionable", "details": {"type": "Ankle"},
                 "athlete": {"displayName": "Khalil Shakir", "position": {"abbreviation": "WR"}}},
            ]},
        ]
    }


def _injuries_df():
    return inj.clean_injury_data(inj._parse_espn_injuries(_payload()))


class TestParse:
    def test_schema_and_rows(self):
        df = inj._parse_espn_injuries(_payload())
        assert list(df.columns) == inj.INJURY_COLUMNS
        assert len(df) == 4  # nameless entry dropped
        row = df[df["player_name"] == "Jayden Daniels"].iloc[0]
        assert row["team"] == "Washington Commanders"
        assert row["position"] == "QB"
        assert row["injury_type"] == "Elbow"
        assert row["status"] == "Out"

    def test_missing_details_is_blank_not_error(self):
        df = inj._parse_espn_injuries(_payload())
        assert df[df["player_name"] == "James Cook III"].iloc[0]["injury_type"] == ""

    def test_empty_payload(self):
        df = inj._parse_espn_injuries({})
        assert df.empty and list(df.columns) == inj.INJURY_COLUMNS

    def test_clean_maps_injured_reserve_to_ir(self):
        assert _injuries_df().set_index("player_name").loc["Jonathon Brooks", "status"] == "IR"


class TestFindPlayerInjury:
    def test_exact_match(self):
        assert inj.find_player_injury("Jayden Daniels", _injuries_df())["status"] == "Out"

    def test_suffix_normalized_match(self):
        assert inj.find_player_injury("James Cook", _injuries_df())["player_name"] == "James Cook III"

    @pytest.mark.parametrize("name", ["Tahj Brooks", "Rasheen Ali", "Daniels"])
    def test_no_last_name_or_substring_false_positives(self, name):
        # Regression: a last-name substring fallback matched "Tahj Brooks" to
        # an IR'd "Jonathon Brooks" (deleting a healthy player's prediction)
        # and "Rasheen Ali" to "Khalil Shakir".
        assert inj.find_player_injury(name, _injuries_df()) is None

    def test_empty_or_bad_input(self):
        assert inj.find_player_injury("Jayden Daniels", pd.DataFrame()) is None
        assert inj.find_player_injury("", _injuries_df()) is None
        assert inj.find_player_injury(None, _injuries_df()) is None


class _Resp:
    def __init__(self, payload=None, bad_json=False):
        self._payload, self._bad = payload, bad_json

    def raise_for_status(self):
        pass

    def json(self):
        if self._bad:
            raise ValueError("not json")
        return self._payload


class TestScrapeEspnInjuries:
    def test_no_spoofed_user_agent(self):
        # Regression: the old hard-coded Chrome 91 User-Agent gets a 403 from
        # the ESPN JSON endpoint; requests' default UA is accepted.
        assert "User-Agent" not in inj.HEADERS

    def test_parses_mocked_response(self, monkeypatch):
        monkeypatch.setattr(inj.requests, "get", lambda *a, **k: _Resp(_payload()))
        df = inj.scrape_espn_injuries()
        assert len(df) == 4

    def test_request_error_returns_empty(self, monkeypatch):
        def _boom(*a, **k):
            raise requests.ConnectionError("down")
        monkeypatch.setattr(inj.requests, "get", _boom)
        assert inj.scrape_espn_injuries().empty

    def test_bad_json_returns_empty(self, monkeypatch):
        monkeypatch.setattr(inj.requests, "get", lambda *a, **k: _Resp(bad_json=True))
        assert inj.scrape_espn_injuries().empty

    def test_empty_feed_returns_empty(self, monkeypatch):
        monkeypatch.setattr(inj.requests, "get", lambda *a, **k: _Resp({"injuries": []}))
        assert inj.scrape_espn_injuries().empty
