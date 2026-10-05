"""Ontario sportsbook spread captures (ontario_spreads.py).

Every network call is mocked; every artifact goes to a temporary directory.
"""

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
import requests

import ontario_spreads as on
import pregame_snapshots as ps
from test_team_features import ROOT

UTC = timezone.utc
KEY = "SECRETKEY1234567890abcdef"

# Wednesday 2026-10-07 12:00 America/Toronto (EDT, UTC-4) = 16:00 UTC.
WED_SLOT = datetime(2026, 10, 7, 16, 0, tzinfo=UTC)
# Sunday 2026-10-11 09:00 Toronto = 13:00 UTC; London kickoff 09:30 ET = 13:30 UTC.
SUN_SLOT = datetime(2026, 10, 11, 13, 0, tzinfo=UTC)

SCHEDULE_ROWS = [
    # game_id, season, week, gameday, gametime, away, home, away_score, home_score
    ("2026_04_ATL_NO", 2026, 4, "2026-10-05", "20:15", "ATL", "NO", 17, 24),     # last week, done
    ("2026_05_TB_DAL", 2026, 5, "2026-10-08", "20:15", "TB", "DAL", None, None),  # Thursday
    ("2026_05_PHI_JAX", 2026, 5, "2026-10-11", "09:30", "PHI", "JAX", None, None),  # London
    ("2026_05_CHI_GB", 2026, 5, "2026-10-11", "13:00", "CHI", "GB", None, None),
    ("2026_05_DEN_LAC", 2026, 5, "2026-10-11", "16:05", "DEN", "LAC", None, None),
    ("2026_05_BUF_LA", 2026, 5, "2026-10-12", "20:15", "BUF", "LA", None, None),  # Monday
    ("2026_06_DAL_NYG", 2026, 6, "2026-10-15", "20:15", "DAL", "NYG", None, None),  # next week
]


def _kickoff(game_id):
    row = next(r for r in SCHEDULE_ROWS if r[0] == game_id)
    return ps.kickoff_utc(row[3], row[4])[0]


@pytest.fixture
def schedule_path(tmp_path):
    df = pd.DataFrame(SCHEDULE_ROWS, columns=["game_id", "season", "week", "gameday", "gametime",
                                              "away_team", "home_team", "away_score", "home_score"])
    df["game_type"] = "REG"
    path = tmp_path / "nfl_games_historical.csv"
    df.to_csv(path, sep="\t", index=False)
    return path


def _book(key, home_point, home_price, away_point, away_price, updated, home, away, market=True):
    full_home, full_away = on.TEAM_FULL_NAME[home], on.TEAM_FULL_NAME[away]
    bk = {"key": key, "title": key, "last_update": updated, "markets": []}
    if market:
        bk["markets"].append({"key": "spreads", "last_update": updated, "outcomes": [
            {"name": full_home, "price": home_price, "point": home_point},
            {"name": full_away, "price": away_price, "point": away_point}]})
    return bk


def _event(game_id, books, commence=None, swapped=False, event_id=None):
    row = next(r for r in SCHEDULE_ROWS if r[0] == game_id)
    away, home = row[5], row[6]
    commence = commence or ps._iso(_kickoff(game_id))
    h, a = (away, home) if swapped else (home, away)
    return {"id": event_id or f"ev_{game_id}", "sport_key": "americanfootball_nfl",
            "commence_time": commence, "home_team": on.TEAM_FULL_NAME[h],
            "away_team": on.TEAM_FULL_NAME[a],
            "bookmakers": [_book(k, *v, home=home, away=away) for k, v in books.items()]}


def _quotes(updated, keys=("betano_ca_on", "betmgm_ca_on", "betrivers_ca_on", "pointsbetca",
                           "proline_ca_on", "sportsinteraction_ca_on", "fanduel")):
    # home -3.5 at -110, away +3.5 at -110
    return {k: (-3.5, -110, 3.5, -110, updated) for k in keys}


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None, text=""):
        self.status_code, self._payload, self.headers, self.text = status, payload, headers or {}, text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} for url: https://x/?apiKey={KEY}")


class FakeAPI:
    """Stands in for requests.get; records every call."""

    def __init__(self, events=None, remaining=480, odds_status=200, sports_headers=None,
                 odds_exc=None):
        self.calls, self.events, self.remaining = [], events or [], remaining
        self.odds_status, self.sports_headers, self.odds_exc = odds_status, sports_headers, odds_exc

    def __call__(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if url.endswith("/sports"):
            headers = self.sports_headers if self.sports_headers is not None else {
                "x-requests-remaining": str(self.remaining), "x-requests-used": "20",
                "x-requests-last": "0"}
            return FakeResponse(200, [], headers)
        if self.odds_exc:
            raise self.odds_exc
        return FakeResponse(self.odds_status, self.events,
                            {"x-requests-remaining": str(self.remaining - 1),
                             "x-requests-used": "21", "x-requests-last": "1"},
                            text=f"error body mentioning apiKey={KEY}")

    @property
    def paid_calls(self):
        return [c for c in self.calls if c[0].endswith("/odds")]


@pytest.fixture
def dirs(tmp_path):
    return {"capture_dir": tmp_path / "captures", "snapshot_dir": tmp_path / "snapshots"}


def _capture(monkeypatch, api, schedule_path, dirs, now, slot="auto", key=KEY, reserve=20):
    monkeypatch.setattr(on.requests, "get", api)
    return on.capture(now=now, slot=slot, api_key=key, capture_dir=dirs["capture_dir"],
                      schedule_path=schedule_path, snapshot_dir=dirs["snapshot_dir"],
                      reserve=reserve, repo_dir=ROOT)


def _wed_events(updated="2026-10-07T15:55:00Z"):
    return [_event(g, _quotes(updated)) for g in
            ("2026_05_TB_DAL", "2026_05_PHI_JAX", "2026_05_CHI_GB", "2026_05_DEN_LAC",
             "2026_05_BUF_LA")]


def _game(doc, game_id):
    return next(g for g in doc["games"] if g["game_id"] == game_id)


def _quote(doc, game_id, book):
    return next(q for q in _game(doc, game_id)["quotes"] if q["book_key"] == book)


# ------------------------------------------------------------ jurisdictions --

class TestOntarioMapping:
    def test_documented_ontario_keys(self):
        assert [b.key for b in on.ONTARIO_BOOKS] == [
            "betano_ca_on", "betmgm_ca_on", "betrivers_ca_on", "pointsbetca",
            "proline_ca_on", "sportsinteraction_ca_on"]
        assert all(b.jurisdiction == "CA-ON" and b.role == "ontario" for b in on.ONTARIO_BOOKS)
        assert all(b.title.endswith("(CA - ON)") for b in on.ONTARIO_BOOKS)

    def test_bet99_is_optional_paid_tier(self):
        assert on.BET99.key == "bet99_ca_on" and on.BET99.role == "ontario_paid_tier"
        assert on.BET99 not in on.REQUESTED_BOOKS                 # opt-in
        assert on.BET99 in on.requested_books(include_bet99=True)

    def test_api_fanduel_is_us_reference_never_ontario(self):
        assert on.FANDUEL_US.key == "fanduel"
        assert on.FANDUEL_US.jurisdiction == "US" and on.FANDUEL_US.role == "us_reference"
        ontario_keys = {b.key for b in on.REQUESTED_BOOKS if b.jurisdiction == "CA-ON"}
        assert "fanduel" not in ontario_keys

    def test_fanduel_ontario_is_manual_only(self):
        fd = on.FANDUEL_ONTARIO_MANUAL
        assert fd.source == "manual" and fd.jurisdiction == "CA-ON"
        assert fd not in on.REQUESTED_BOOKS and fd.key != "fanduel"

    def test_not_every_canadian_feed_is_ontario(self):
        assert "playnow_ca" in on.NON_ONTARIO_CANADIAN
        assert "playnow_ca" not in {b.key for b in on.REQUESTED_BOOKS}

    def test_capture_labels_fanduel_us_and_keeps_unrequested_books_out(
            self, monkeypatch, schedule_path, dirs):
        events = _wed_events()
        events[0]["bookmakers"].append(_book("playnow_ca", -3.0, -115, 3.0, -105,
                                             "2026-10-07T15:55:00Z", "DAL", "TB"))
        doc = _capture(monkeypatch, FakeAPI(events), schedule_path, dirs, WED_SLOT).doc
        fd = _quote(doc, "2026_05_TB_DAL", "fanduel")
        assert fd["jurisdiction"] == "US" and fd["role"] == "us_reference"
        assert "NOT FanDuel Ontario" in doc["coverage"]["fanduel"]["note"]
        g = _game(doc, "2026_05_TB_DAL")
        assert "playnow_ca" not in {q["book_key"] for q in g["quotes"]}
        assert g["unrequested_books"] == ["playnow_ca"]


# ------------------------------------------------- matching and handicaps --

class TestEventMatching:
    GAME = {"game_id": "2026_05_CHI_GB", "home_team": "GB", "away_team": "CHI",
            "kickoff_utc": ps._iso(_kickoff("2026_05_CHI_GB"))}

    def test_teams_and_kickoff_match(self):
        ev, orient, problem = on.match_event(self.GAME, [_event("2026_05_CHI_GB", {})])
        assert ev["id"] == "ev_2026_05_CHI_GB" and orient == "same" and problem is None

    def test_teams_alone_are_not_enough(self):
        # Same teams, different kickoff (e.g. a later meeting or a stale listing).
        other = _event("2026_05_CHI_GB", {}, commence="2026-12-20T18:00:00Z")
        assert on.match_event(self.GAME, [other]) == (None, "", "kickoff_mismatch")

    def test_within_tolerance_matches(self):
        near = _event("2026_05_CHI_GB", {}, commence="2026-10-11T17:05:00Z")
        assert on.match_event(self.GAME, [near])[0] is not None

    def test_no_event(self):
        assert on.match_event(self.GAME, []) == (None, "", "no_provider_event")

    def test_two_candidate_events_are_ambiguous(self):
        evs = [_event("2026_05_CHI_GB", {}, event_id="a"), _event("2026_05_CHI_GB", {}, event_id="b")]
        assert on.match_event(self.GAME, evs)[2] == "ambiguous_provider_event"

    def test_swapped_listing_keeps_each_teams_own_handicap(self, monkeypatch, schedule_path, dirs):
        # Provider lists the London game with home/away the other way round.
        events = _wed_events()
        events[1] = _event("2026_05_PHI_JAX", {"betano_ca_on": (3.0, 105, -3.0, -125,
                                                                "2026-10-07T15:55:00Z")},
                           swapped=True)
        doc = _capture(monkeypatch, FakeAPI(events), schedule_path, dirs, WED_SLOT).doc
        g = _game(doc, "2026_05_PHI_JAX")
        assert g["orientation"] == "swapped"
        q = _quote(doc, "2026_05_PHI_JAX", "betano_ca_on")
        # JAX (schedule home) +3 at +105; PHI (away) -3 at -125 - by team, not position.
        assert (q["home_point"], q["home_price"], q["away_point"], q["away_price"]) == (3.0, 105, -3.0, -125)

    def test_home_favourite_sign(self, monkeypatch, schedule_path, dirs):
        doc = _capture(monkeypatch, FakeAPI(_wed_events()), schedule_path, dirs, WED_SLOT).doc
        q = _quote(doc, "2026_05_TB_DAL", "betmgm_ca_on")
        assert q["home_point"] == -3.5 and q["away_point"] == 3.5     # DAL (home) favoured


# ----------------------------------------------------------------- odds --

class TestOddsValidation:
    @pytest.mark.parametrize("price,ok", [(-110, True), (100, True), (-100, True), (250, True),
                                          (99, False), (-99, False), (0, False), (110.5, False),
                                          (float("nan"), False), (None, False), ("-110", False),
                                          (True, False)])
    def test_american_prices(self, price, ok):
        assert on.valid_american(price) is ok

    @pytest.mark.parametrize("point,ok", [(-3.5, True), (0, True), (7.0, True), (-3.25, False),
                                          (61, False), (float("inf"), False), (None, False)])
    def test_handicaps(self, point, ok):
        assert on.valid_handicap(point) is ok

    def _q(self, home_point=-3.5, home_price=-110, away_point=3.5, away_price=-110,
           updated="2026-10-07T15:55:00Z"):
        bk = _book("betano_ca_on", home_point, home_price, away_point, away_price, updated,
                   "DAL", "TB")
        return on.parse_quote(bk, on.ONTARIO_BOOKS[0], {"home_team": "DAL", "away_team": "TB"},
                              WED_SLOT)

    def test_valid_quote(self):
        q = self._q()
        assert q["status"] == "quoted" and q["age_minutes"] == 5.0
        assert q["market_last_update"] == "2026-10-07T15:55:00Z"

    def test_non_mirrored_handicaps_invalid(self):
        assert self._q(away_point=3.0)["status"] == "invalid"

    def test_bad_price_invalid(self):
        assert self._q(home_price=-50)["status"] == "invalid"

    def test_future_update_invalid(self):
        assert "after the capture" in self._q(updated="2026-10-07T17:00:00Z")["problem"]

    def test_missing_update_invalid(self):
        bk = _book("betano_ca_on", -3.5, -110, 3.5, -110, None, "DAL", "TB")
        q = on.parse_quote(bk, on.ONTARIO_BOOKS[0], {"home_team": "DAL", "away_team": "TB"}, WED_SLOT)
        assert q["status"] == "invalid" and "last_update" in q["problem"]

    def test_stale_quote_kept_and_labelled(self):
        q = self._q(updated="2026-10-07T13:00:00Z")
        assert q["status"] == "stale" and q["home_point"] == -3.5 and q["age_minutes"] == 180.0

    def test_offset_timestamps_parse(self):
        assert on.parse_provider_time("2026-10-07T11:55:00-04:00") == datetime(2026, 10, 7, 15, 55, tzinfo=UTC)


# ---------------------------------------------------------- slots and DST --

class TestSlots:
    def test_wednesday_noon_edt(self):
        s = on.resolve_slot(WED_SLOT)
        assert s["slot_id"] == "2026-10-07_wednesday_noon" and s["status"] == "on_time"
        assert s["intended_utc"] == "2026-10-07T16:00:00Z"
        assert s["intended_local"] == "2026-10-07T12:00:00-04:00"

    def test_wednesday_noon_est_after_dst_ends(self):
        # DST ends 2026-11-01: noon Toronto is 17:00 UTC.
        assert on.resolve_slot(datetime(2026, 11, 4, 16, 0, tzinfo=UTC)) is None  # 11:00 local
        s = on.resolve_slot(datetime(2026, 11, 4, 17, 0, tzinfo=UTC))
        assert s["intended_utc"] == "2026-11-04T17:00:00Z" and s["status"] == "on_time"

    def test_sunday_morning_both_offsets(self):
        assert on.resolve_slot(SUN_SLOT)["slot_id"] == "2026-10-11_sunday_morning"
        assert on.resolve_slot(datetime(2026, 11, 8, 13, 0, tzinfo=UTC)) is None   # 08:00 EST
        assert on.resolve_slot(datetime(2026, 11, 8, 14, 0, tzinfo=UTC))["status"] == "on_time"

    def test_late_run_is_labelled(self):
        s = on.resolve_slot(WED_SLOT + timedelta(minutes=95))
        assert s["status"] == "late" and s["delay_minutes"] == 95.0

    def test_after_window_is_not_due(self):
        # Wednesday counts until 15:00 Toronto, Sunday until 11:00.
        assert on.resolve_slot(WED_SLOT + timedelta(hours=3))["status"] == "late"
        assert on.resolve_slot(WED_SLOT + timedelta(hours=3, minutes=1)) is None
        assert on.resolve_slot(SUN_SLOT + timedelta(hours=2))["status"] == "late"
        assert on.resolve_slot(SUN_SLOT + timedelta(hours=2, minutes=1)) is None

    def test_much_later_runs_are_never_labelled_as_the_slot(self):
        # Thursday morning, or Sunday after the early kickoffs.
        assert on.resolve_slot(datetime(2026, 10, 8, 13, 0, tzinfo=UTC)) is None
        assert on.resolve_slot(datetime(2026, 10, 11, 16, 59, tzinfo=UTC)) is None
        with pytest.raises(KeyError):
            on.resolve_slot(WED_SLOT, "thursday")

    def test_ad_hoc(self):
        s = on.resolve_slot(WED_SLOT + timedelta(days=1), "ad_hoc")
        assert s["status"] == "ad_hoc" and s["intended_utc"] is None

    def test_coverage_marks_missed_and_late(self, monkeypatch, schedule_path, dirs):
        _capture(monkeypatch, FakeAPI(_wed_events("2026-10-07T17:30:00Z")), schedule_path, dirs,
                 WED_SLOT + timedelta(minutes=95))
        cov = on.slot_coverage(datetime(2026, 10, 6, tzinfo=UTC), datetime(2026, 10, 12, tzinfo=UTC),
                               dirs["capture_dir"])
        assert [(c["slot_id"], c["status"]) for c in cov] == [
            ("2026-10-07_wednesday_noon", "late"), ("2026-10-11_sunday_morning", "missed")]

    def test_coverage_starts_at_first_capture(self, monkeypatch, schedule_path, dirs):
        assert on.slot_coverage(datetime(2026, 9, 1, tzinfo=UTC), WED_SLOT, dirs["capture_dir"]) == []
        _capture(monkeypatch, FakeAPI(_wed_events()), schedule_path, dirs, WED_SLOT)
        cov = on.slot_coverage(datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 10, 12, tzinfo=UTC),
                               dirs["capture_dir"])
        # Slots before the first capture weren't tracked, so they aren't "missed".
        assert [c["slot_id"] for c in cov] == ["2026-10-07_wednesday_noon",
                                               "2026-10-11_sunday_morning"]


# --------------------------------------------------------- kickoff rules --

class TestKickoffExclusion:
    def test_sunday_capture_excludes_games_underway(self, monkeypatch, schedule_path, dirs):
        # A delayed Sunday run at 09:40 ET: the 09:30 London game has started.
        now = SUN_SLOT + timedelta(minutes=40)
        events = [_event(g, _quotes("2026-10-11T13:35:00Z"))
                  for g in ("2026_05_CHI_GB", "2026_05_DEN_LAC", "2026_05_BUF_LA")]
        api = FakeAPI(events)
        doc = _capture(monkeypatch, api, schedule_path, dirs, now).doc
        assert {g["game_id"] for g in doc["games"]} == {"2026_05_CHI_GB", "2026_05_DEN_LAC",
                                                        "2026_05_BUF_LA"}
        reasons = {e["game_id"]: e["reason"] for e in doc["excluded_games"]}
        assert reasons["2026_05_PHI_JAX"] == "kicked_off" and reasons["2026_05_TB_DAL"] == "kicked_off"
        assert api.paid_calls[0][1]["commenceTimeFrom"] == ps._iso(now)

    def test_provider_reported_start_is_excluded(self, monkeypatch, schedule_path, dirs):
        events = _wed_events()
        # Provider says CHI@GB starts 5 minutes before capture (e.g. a moved game).
        events[2]["commence_time"] = "2026-10-07T15:55:00Z"
        sched = pd.read_csv(schedule_path, sep="\t")
        sched.loc[sched.game_id == "2026_05_CHI_GB", ["gameday", "gametime"]] = ["2026-10-07", "12:30"]
        sched.to_csv(schedule_path, sep="\t", index=False)
        doc = _capture(monkeypatch, FakeAPI(events), schedule_path, dirs, WED_SLOT).doc
        assert "2026_05_CHI_GB" not in {g["game_id"] for g in doc["games"]}

    def test_next_weeks_games_are_out_of_scope(self, monkeypatch, schedule_path, dirs):
        doc = _capture(monkeypatch, FakeAPI(_wed_events()), schedule_path, dirs, WED_SLOT).doc
        assert "2026_06_DAL_NYG" not in {g["game_id"] for g in doc["games"]}
        assert all(parse > WED_SLOT for parse in
                   (on.parse_utc(g["kickoff_utc"]) for g in doc["games"]))

    def test_no_pregame_games_makes_no_call(self, monkeypatch, schedule_path, dirs):
        api = FakeAPI()
        r = _capture(monkeypatch, api, schedule_path, dirs, datetime(2027, 3, 3, 17, 0, tzinfo=UTC),
                     slot="ad_hoc")
        assert r.status == "no_games" and api.calls == []


# ------------------------------------------------------- immutable writes --

class TestImmutability:
    def test_capture_is_sealed_and_valid(self, monkeypatch, schedule_path, dirs):
        r = _capture(monkeypatch, FakeAPI(_wed_events()), schedule_path, dirs, WED_SLOT)
        doc = json.loads(r.path.read_text())
        on.validate_capture(doc)
        assert doc["schema_version"] == 1 and doc["kind"] == "ontario_spread_capture"
        assert doc["run_id"] == r.path.stem and doc["code_revision"]
        assert doc["schedule"]["sha256"] == ps.sha256_file(schedule_path)

    def test_same_run_retry_is_idempotent(self, monkeypatch, schedule_path, dirs):
        r = _capture(monkeypatch, FakeAPI(_wed_events()), schedule_path, dirs, WED_SLOT)
        before = r.path.read_bytes()
        path, created = on.write_artifact(r.doc, dirs["capture_dir"], on.validate_capture)
        assert (path, created) == (r.path, False) and r.path.read_bytes() == before

    def test_conflicting_reuse_is_an_error(self, monkeypatch, schedule_path, dirs):
        r = _capture(monkeypatch, FakeAPI(_wed_events()), schedule_path, dirs, WED_SLOT)
        before = r.path.read_bytes()
        changed = json.loads(before)
        changed["games"][0]["quotes"][0]["home_price"] = -105
        changed = on._seal(changed)
        with pytest.raises(on.CaptureConflictError, match="different content"):
            on.write_artifact(changed, dirs["capture_dir"], on.validate_capture)
        assert r.path.read_bytes() == before

    def test_corrupt_existing_file_is_not_overwritten(self, monkeypatch, schedule_path, dirs):
        r = _capture(monkeypatch, FakeAPI(_wed_events()), schedule_path, dirs, WED_SLOT)
        r.path.write_text("{partial")
        with pytest.raises(on.CaptureConflictError, match="isn't a valid artifact"):
            on.write_artifact(r.doc, dirs["capture_dir"], on.validate_capture)
        assert r.path.read_text() == "{partial"

    def test_tampered_capture_fails_validation(self, monkeypatch, schedule_path, dirs):
        r = _capture(monkeypatch, FakeAPI(_wed_events()), schedule_path, dirs, WED_SLOT)
        doc = json.loads(r.path.read_text())
        doc["captured_at"] = "2026-10-07T15:00:00Z"
        with pytest.raises(on.ValidationError, match="checksum"):
            on.validate_capture(doc)

    def test_concurrent_identical_writers_make_one_file(self, monkeypatch, schedule_path, dirs):
        r = _capture(monkeypatch, FakeAPI(_wed_events()), schedule_path, dirs, WED_SLOT)
        other = dirs["capture_dir"].parent / "race"
        results, errors = [], []

        def write():
            try:
                results.append(on.write_artifact(r.doc, other, on.validate_capture)[1])
            except Exception as exc:     # pragma: no cover - reported below
                errors.append(exc)

        threads = [threading.Thread(target=write) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert errors == [] and sorted(results) == [False] * 7 + [True]
        assert [p.name for p in other.iterdir()] == [r.path.name]     # no temp files left

    def test_slot_already_captured_makes_no_call(self, monkeypatch, schedule_path, dirs):
        _capture(monkeypatch, FakeAPI(_wed_events()), schedule_path, dirs, WED_SLOT)
        api = FakeAPI(_wed_events())
        r = _capture(monkeypatch, api, schedule_path, dirs, WED_SLOT + timedelta(hours=1))
        assert r.status == "already_captured" and api.calls == []
        assert len(list(dirs["capture_dir"].glob("*.json"))) == 1

    def test_concurrent_capture_of_a_slot_is_refused(self, monkeypatch, schedule_path, dirs):
        dirs["capture_dir"].mkdir(parents=True)
        (dirs["capture_dir"] / ".2026-10-07_wednesday_noon.lock").write_text("other-run")
        api = FakeAPI(_wed_events())
        with pytest.raises(on.CaptureError, match="another capture holds"):
            _capture(monkeypatch, api, schedule_path, dirs, WED_SLOT)
        assert api.calls == []

    def test_existing_tracker_files_untouched(self, monkeypatch, schedule_path, dirs, tmp_path):
        _capture(monkeypatch, FakeAPI(_wed_events()), schedule_path, dirs, WED_SLOT)
        written = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
        assert all(p.startswith("captures/") or p == "nfl_games_historical.csv" for p in written)


# ---------------------------------------------------------- missing/stale --

class TestCoverage:
    def test_absent_books_recorded_as_absent(self, monkeypatch, schedule_path, dirs):
        events = _wed_events()
        events[0]["bookmakers"] = [b for b in events[0]["bookmakers"] if b["key"] != "proline_ca_on"]
        doc = _capture(monkeypatch, FakeAPI(events), schedule_path, dirs, WED_SLOT).doc
        q = _quote(doc, "2026_05_TB_DAL", "proline_ca_on")
        assert q["status"] == "absent" and q["home_point"] is None and q["home_price"] is None

    def test_bet99_not_requested_by_default_and_reported(self, monkeypatch, schedule_path, dirs):
        api = FakeAPI(_wed_events())
        doc = _capture(monkeypatch, api, schedule_path, dirs, WED_SLOT).doc
        assert "bet99_ca_on" not in api.paid_calls[0][1]["bookmakers"]
        cov = doc["coverage"]["bet99_ca_on"]
        assert cov["requested"] is False and "not requested" in cov["note"]
        assert all(q["book_key"] != "bet99_ca_on" for g in doc["games"] for q in g["quotes"])

    def test_bet99_opt_in_absence_reported(self, monkeypatch, schedule_path, dirs):
        monkeypatch.setenv("ONTARIO_SPREADS_INCLUDE_BET99", "1")
        api = FakeAPI(_wed_events())
        doc = _capture(monkeypatch, api, schedule_path, dirs, WED_SLOT).doc
        assert "bet99_ca_on" in api.paid_calls[0][1]["bookmakers"].split(",")
        assert doc["request"]["estimated_cost"] == 1           # 8 books still = 1 region
        cov = doc["coverage"]["bet99_ca_on"]
        assert cov["requested"] is True and cov["by_status"] == {"absent": 5}
        assert "paid-tier" in cov["note"]

    def test_missing_event_marks_every_book(self, monkeypatch, schedule_path, dirs):
        events = [e for e in _wed_events() if e["id"] != "ev_2026_05_BUF_LA"]
        doc = _capture(monkeypatch, FakeAPI(events), schedule_path, dirs, WED_SLOT).doc
        g = _game(doc, "2026_05_BUF_LA")
        assert g["match_problem"] == "no_provider_event"
        assert {q["status"] for q in g["quotes"]} == {"event_not_matched"}

    def test_book_without_spreads_market(self, monkeypatch, schedule_path, dirs):
        events = _wed_events()
        events[0]["bookmakers"] = [b for b in events[0]["bookmakers"] if b["key"] != "proline_ca_on"]
        events[0]["bookmakers"].append(_book("proline_ca_on", 0, 0, 0, 0, "2026-10-07T15:55:00Z",
                                             "DAL", "TB", market=False))
        doc = _capture(monkeypatch, FakeAPI(events), schedule_path, dirs, WED_SLOT).doc
        assert _quote(doc, "2026_05_TB_DAL", "proline_ca_on")["status"] == "no_spreads_market"

    def test_absent_quote_is_never_carried_forward(self, monkeypatch, schedule_path, dirs):
        _capture(monkeypatch, FakeAPI(_wed_events()), schedule_path, dirs, WED_SLOT)
        events = [_event(g, _quotes("2026-10-11T12:55:00Z", keys=("betano_ca_on",)))
                  for g in ("2026_05_PHI_JAX", "2026_05_CHI_GB", "2026_05_DEN_LAC", "2026_05_BUF_LA")]
        doc = _capture(monkeypatch, FakeAPI(events), schedule_path, dirs, SUN_SLOT).doc
        q = _quote(doc, "2026_05_CHI_GB", "betmgm_ca_on")
        assert q["status"] == "absent" and q["home_point"] is None   # Wednesday's -3.5 not reused

    def test_unmatched_provider_events_listed(self, monkeypatch, schedule_path, dirs):
        events = _wed_events() + [{"id": "mystery", "commence_time": "2026-10-09T00:15:00Z",
                                   "home_team": "Nowhere", "away_team": "Elsewhere", "bookmakers": []}]
        doc = _capture(monkeypatch, FakeAPI(events), schedule_path, dirs, WED_SLOT).doc
        assert [e["id"] for e in doc["unmatched_provider_events"]] == ["mystery"]


# ----------------------------------------------------------- model link --

class TestModelSnapshotLink:
    INDEX = [
        {"run_id": "20261006T030000Z-aaaaaaaaaaaa", "captured_at": "2026-10-06T03:05:00Z",
         "payload_sha256": "a" * 64, "file": "a.json", "games": {"2026_05_CHI_GB": "predicted"}},
        {"run_id": "20261007T030000Z-bbbbbbbbbbbb", "captured_at": "2026-10-07T03:05:00Z",
         "payload_sha256": "b" * 64, "file": "b.json", "games": {"2026_05_CHI_GB": "no_line"}},
        # Captured AFTER the Wednesday quotes - must never be used for them.
        {"run_id": "20261008T030000Z-cccccccccccc", "captured_at": "2026-10-08T03:05:00Z",
         "payload_sha256": "c" * 64, "file": "c.json", "games": {"2026_05_CHI_GB": "predicted"}},
    ]

    def test_latest_snapshot_at_or_before_capture(self):
        link = on.link_model_snapshot("2026_05_CHI_GB", WED_SLOT, self.INDEX, None)
        assert link["status"] == "linked" and link["run_id"].endswith("bbbbbbbbbbbb")
        assert link["prediction_status"] == "no_line"

    def test_never_a_later_snapshot(self):
        early = datetime(2026, 10, 6, 0, 0, tzinfo=UTC)
        assert on.link_model_snapshot("2026_05_CHI_GB", early, self.INDEX, None) == {
            "status": "no_eligible_snapshot"}

    def test_game_not_in_any_snapshot(self):
        assert on.link_model_snapshot("2026_05_BUF_LA", WED_SLOT, self.INDEX, None)["status"] == \
            "no_eligible_snapshot"

    def test_unreadable_snapshots_recorded(self):
        link = on.link_model_snapshot("2026_05_CHI_GB", WED_SLOT, None, "bad file")
        assert link == {"status": "unavailable", "reason": "bad file"}

    def test_capture_stores_link_but_no_probabilities(self, monkeypatch, schedule_path, dirs):
        monkeypatch.setattr(on, "model_snapshot_index", lambda d: self.INDEX)
        doc = _capture(monkeypatch, FakeAPI(_wed_events()), schedule_path, dirs, WED_SLOT).doc
        link = _quote(doc, "2026_05_CHI_GB", "betano_ca_on")["model_snapshot"]
        assert link["file"] == "b.json" and link["basis"] == "market_last_update"
        assert link["as_of"] == "2026-10-07T15:55:00Z"
        assert _quote(doc, "2026_05_TB_DAL", "betano_ca_on")["model_snapshot"]["status"] == \
            "no_eligible_snapshot"
        assert "model_snapshot" not in _game(doc, "2026_05_CHI_GB")
        text = json.dumps(doc)
        assert "prob_underdog" not in text and "probability" not in text.replace(
            "no model probabilities", "")

    def test_real_snapshot_index(self, tmp_path):
        from test_pregame_snapshots import PROBS, ROWS, SIGNALS, T0, _manifest, _write_inputs
        data = tmp_path / "data_files"
        _write_inputs(data, ROWS, PROBS, SIGNALS)
        _manifest(data)
        ps.capture(data, now=T0)
        index = on.model_snapshot_index(data / "pregame_snapshots")
        assert len(index) == 1 and "2026_04_NYJ_CHI" in index[0]["games"]
        assert on.link_model_snapshot("2026_04_NYJ_CHI", T0 - timedelta(seconds=1), index, None)[
            "status"] == "no_eligible_snapshot"
        assert on.link_model_snapshot("2026_04_NYJ_CHI", T0, index, None)["status"] == "linked"


# ------------------------------------------------------- manual quotes --

class TestManualQuotes:
    def _enter(self, tmp_path, schedule_path, **kw):
        args = dict(game_id="2026_05_TB_DAL", team="DAL", handicap=-3.5, price=-112,
                    observed_at="2026-10-07T12:05:00-04:00", now=WED_SLOT + timedelta(minutes=10),
                    schedule_path=schedule_path, manual_dir=tmp_path / "manual", repo_dir=ROOT)
        args.update(kw)
        return on.manual_quote(**args)

    def test_records_a_manual_fanduel_ontario_quote(self, tmp_path, schedule_path):
        path, created, doc = self._enter(tmp_path, schedule_path, opponent_price=-108,
                                         entered_by="tester")
        assert created and path.exists()
        on.validate_manual(json.loads(path.read_text()))
        assert (doc["source"], doc["book_key"], doc["jurisdiction"]) == (
            "manual", "fanduel_on_manual", "CA-ON")
        assert doc["observed_at"] == "2026-10-07T16:05:00Z"
        assert doc["entered_at"] == "2026-10-07T16:10:00Z"
        assert (doc["team"], doc["team_side"], doc["handicap"], doc["opponent_handicap"]) == (
            "DAL", "home", -3.5, 3.5)
        assert doc["schedule"]["sha256"] == ps.sha256_file(schedule_path) and doc["code_revision"]

    def test_duplicate_entry_not_recorded_twice(self, tmp_path, schedule_path):
        first = self._enter(tmp_path, schedule_path)
        second = self._enter(tmp_path, schedule_path, now=WED_SLOT + timedelta(minutes=20))
        assert second[1] is False and second[0] == first[0]
        assert len(list((tmp_path / "manual").glob("*.json"))) == 1

    @pytest.mark.parametrize("kw,match", [
        ({"observed_at": "2026-10-07T16:30:00Z"}, "future"),
        ({"observed_at": "2026-10-07T12:05:00"}, "timezone"),
        ({"game_id": "2026_05_TB_DAL", "observed_at": "2026-10-09T01:00:00Z",
          "now": datetime(2026, 10, 9, 2, tzinfo=UTC)}, "before kickoff"),
        ({"team": "NYG"}, "not playing"),
        ({"handicap": -3.25}, "half-point"),
        ({"price": -90}, "American"),
        ({"game_id": "2026_05_XXX_YYY"}, "not found"),
    ])
    def test_rejects_bad_entries(self, tmp_path, schedule_path, kw, match):
        with pytest.raises(on.ValidationError, match=match):
            self._enter(tmp_path, schedule_path, **kw)
        assert not (tmp_path / "manual").exists() or not list((tmp_path / "manual").glob("*.json"))

    def test_pickem_has_no_negative_zero(self, tmp_path, schedule_path):
        path, _, doc = self._enter(tmp_path, schedule_path, handicap=-0.0)
        assert "-0.0" not in path.read_text() and doc["opponent_handicap"] == 0.0

    def test_cli_requires_every_field(self, capsys):
        with pytest.raises(SystemExit):
            on.main(["manual-quote", "--game", "2026_05_TB_DAL", "--team", "DAL", "--spread", "-3.5"])

    def test_manual_quote_cannot_claim_api_source(self, tmp_path, schedule_path):
        path, _, doc = self._enter(tmp_path, schedule_path)
        doc = dict(doc, source="the_odds_api")
        with pytest.raises(on.ValidationError):
            on.validate_manual(on._seal(doc))


# ------------------------------------------------- budget and failures --

class TestBudgetAndFailures:
    def test_request_cost(self):
        assert on.request_cost() == 1                      # 8 bookmakers = 1 region equivalent
        assert on.request_cost(n_books=11) == 2 and on.request_cost(n_markets=2, n_books=8) == 2

    def test_missing_key_makes_no_calls(self, monkeypatch, schedule_path, dirs):
        api = FakeAPI(_wed_events())
        r = _capture(monkeypatch, api, schedule_path, dirs, WED_SLOT, key="")
        assert r.status == "no_key" and api.calls == []

    def test_not_due_makes_no_calls(self, monkeypatch, schedule_path, dirs):
        api = FakeAPI(_wed_events())
        r = _capture(monkeypatch, api, schedule_path, dirs, datetime(2026, 10, 9, 15, 0, tzinfo=UTC))
        assert r.status == "not_due" and api.calls == []

    def test_one_paid_call_with_explicit_books_and_no_key_stored(self, monkeypatch, schedule_path, dirs):
        api = FakeAPI(_wed_events())
        r = _capture(monkeypatch, api, schedule_path, dirs, WED_SLOT)
        assert len(api.paid_calls) == 1
        params = api.paid_calls[0][1]
        assert params["bookmakers"].split(",") == [b.key for b in on.REQUESTED_BOOKS]
        assert "regions" not in params and params["markets"] == "spreads"
        req = r.doc["request"]
        assert req["usage"] == {"x-requests-remaining": 479, "x-requests-used": 21, "x-requests-last": 1}
        assert req["credits_before"] == 480 and req["estimated_cost"] == 1
        assert KEY not in r.path.read_text() and "apiKey" not in r.path.read_text()

    def test_reserve_blocks_paid_call(self, monkeypatch, schedule_path, dirs):
        api = FakeAPI(_wed_events(), remaining=20)
        with pytest.raises(on.BudgetSkip, match="reserve of 20"):
            _capture(monkeypatch, api, schedule_path, dirs, WED_SLOT)
        assert api.paid_calls == [] and not list(dirs["capture_dir"].glob("*.json"))

    def test_unreadable_credits_block_paid_call(self, monkeypatch, schedule_path, dirs):
        api = FakeAPI(_wed_events(), sports_headers={})
        with pytest.raises(on.BudgetSkip, match="couldn't be read"):
            _capture(monkeypatch, api, schedule_path, dirs, WED_SLOT)
        assert api.paid_calls == []

    def test_http_error_is_a_redacted_failure(self, monkeypatch, schedule_path, dirs):
        api = FakeAPI(_wed_events(), odds_status=401)
        with pytest.raises(on.CaptureError, match="HTTP 401") as exc:
            _capture(monkeypatch, api, schedule_path, dirs, WED_SLOT)
        assert KEY not in str(exc.value) and "apiKey=***" in str(exc.value)
        assert not list(dirs["capture_dir"].glob("*.json"))

    def test_network_error_is_redacted(self, monkeypatch, schedule_path, dirs):
        api = FakeAPI(odds_exc=requests.ConnectionError(f"failed https://x/odds?apiKey={KEY}&m=1"))
        with pytest.raises(on.CaptureError) as exc:
            _capture(monkeypatch, api, schedule_path, dirs, WED_SLOT)
        assert KEY not in str(exc.value)

    def test_non_list_response_fails(self, monkeypatch, schedule_path, dirs):
        api = FakeAPI({"message": "quota exceeded"})
        with pytest.raises(on.CaptureError, match="expected a list"):
            _capture(monkeypatch, api, schedule_path, dirs, WED_SLOT)

    def test_lock_released_after_failure(self, monkeypatch, schedule_path, dirs):
        with pytest.raises(on.CaptureError):
            _capture(monkeypatch, FakeAPI(odds_status=500), schedule_path, dirs, WED_SLOT)
        assert not list(dirs["capture_dir"].glob(".*.lock"))

    def test_sanitize_strips_credential_urls(self):
        data = [{"id": "e", "link": f"https://book/?apiKey={KEY}", "nested": [KEY, "ok"]}]
        assert on.sanitize(data, KEY) == [{"id": "e", "nested": ["***", "ok"]}]

    @pytest.mark.parametrize("exc,code,status", [
        (on.BudgetSkip("low"), 2, "budget_skip"),
        (on.CaptureError(f"boom apiKey={KEY}"), 1, "failed"),
    ])
    def test_cli_exit_codes_and_summary(self, monkeypatch, tmp_path, capsys, exc, code, status):
        summary, output = tmp_path / "summary.md", tmp_path / "out.txt"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))
        monkeypatch.setenv("ODDS_API_KEY", KEY)

        def boom(**kw):
            raise exc
        monkeypatch.setattr(on, "capture", boom)
        assert on.main(["capture"]) == code
        out = capsys.readouterr().out
        assert "::error::" in out and KEY not in out
        assert f"status={status}" in output.read_text()
        assert KEY not in summary.read_text(encoding="utf-8")

    def test_cli_success_paths_exit_zero(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(on, "capture", lambda **kw: on.CaptureResult("not_due", "nothing due"))
        monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "o"))
        assert on.main(["capture"]) == 0
        assert "status=not_due" in (tmp_path / "o").read_text()


# ------------------------------------------------------------ workflow --

@pytest.fixture(scope="module")
def workflow():
    import yaml
    return yaml.safe_load((ROOT / ".github" / "workflows" / "ontario-spread-capture.yml")
                          .read_text(encoding="utf-8"))


class TestWorkflow:
    def test_cron_covers_both_utc_offsets(self, workflow):
        crons = {c["cron"] for c in workflow[True]["schedule"]}
        # Wednesday 12:00 Toronto = 16:00 UTC (EDT) / 17:00 UTC (EST);
        # Sunday 09:00 Toronto = 13:00 UTC (EDT) / 14:00 UTC (EST).
        assert crons == {"0 16 * 9-12,1-2 3", "0 17 * 9-12,1-2 3",
                         "0 13 * 9-12,1-2 0", "0 14 * 9-12,1-2 0"}

    def test_runs_are_serialized(self, workflow):
        assert workflow["concurrency"]["cancel-in-progress"] is False

    def test_capture_failure_is_not_swallowed(self, workflow):
        steps = workflow["jobs"]["capture"]["steps"]
        cap = next(s for s in steps if s.get("id") == "capture")
        assert "continue-on-error" not in cap
        assert "set -o pipefail" in cap["run"] and "ontario_spreads.py capture" in cap["run"]
        assert cap["env"]["ODDS_API_KEY"] == "${{ secrets.ODDS_API_KEY }}"
        others = [s for s in steps if s is not cap]
        assert all("ODDS_API_KEY" not in json.dumps(s.get("env", {})) for s in others)

    def test_commit_only_new_capture_files(self, workflow):
        steps = workflow["jobs"]["capture"]["steps"]
        commit = next(s for s in steps if "Commit" in s.get("name", ""))
        assert "data_files/ontario_spreads/captures" in commit["run"]
        assert "spread_tracker" not in commit["run"]


def test_cli_invalid_stored_capture_fails_visibly(monkeypatch, tmp_path, capsys):
    bad = tmp_path / "captures"
    bad.mkdir()
    (bad / "20261007T160000Z-aaaaaaaaaaaa.json").write_text("{not json")
    monkeypatch.setattr(on, "CAPTURE_DIR", bad)
    monkeypatch.setattr(on.capture, "__kwdefaults__", {**on.capture.__kwdefaults__, "capture_dir": bad,
                                                       "now": WED_SLOT})
    assert on.main(["capture"]) == 1
    assert "::error::" in capsys.readouterr().out



# ------------------------------------------------- review fixes (Oct 2026) --

class TestQuoteTimeModelLink:
    INDEX = TestModelSnapshotLink.INDEX

    def _doc(self, monkeypatch, schedule_path, dirs, events):
        monkeypatch.setattr(on, "model_snapshot_index", lambda d: self.INDEX)
        return _capture(monkeypatch, FakeAPI(events), schedule_path, dirs, WED_SLOT).doc

    def test_link_uses_quote_timestamp_not_fetch_time(self, monkeypatch, schedule_path, dirs):
        # Fetched Wednesday 16:00, but this book's market last changed at 02:00
        # Wednesday - before the 03:05 Wednesday model snapshot existed.
        events = _wed_events()
        for bk in events[2]["bookmakers"]:
            if bk["key"] == "betmgm_ca_on":
                bk["markets"][0]["last_update"] = "2026-10-07T02:00:00Z"
        doc = self._doc(monkeypatch, schedule_path, dirs, events)
        old = _quote(doc, "2026_05_CHI_GB", "betmgm_ca_on")
        assert old["status"] == "stale"
        assert old["model_snapshot"]["file"] == "a.json"                 # not b.json
        assert old["model_snapshot"]["as_of"] == "2026-10-07T02:00:00Z"
        assert _quote(doc, "2026_05_CHI_GB", "betano_ca_on")["model_snapshot"]["file"] == "b.json"

    def test_missing_quote_timestamp_gets_no_time_aligned_link(self, monkeypatch, schedule_path, dirs):
        events = _wed_events()
        for bk in events[2]["bookmakers"]:
            if bk["key"] == "betrivers_ca_on":
                del bk["markets"][0]["last_update"]          # only the bookmaker time remains
        doc = self._doc(monkeypatch, schedule_path, dirs, events)
        q = _quote(doc, "2026_05_CHI_GB", "betrivers_ca_on")
        assert q["status"] == "quoted" and q["market_last_update"] is None
        assert q["model_snapshot"]["status"] == "quote_time_unknown"
        assert "run_id" not in q["model_snapshot"]

    def test_no_quote_no_link(self, monkeypatch, schedule_path, dirs):
        events = [e for e in _wed_events() if e["id"] != "ev_2026_05_CHI_GB"]
        doc = self._doc(monkeypatch, schedule_path, dirs, events)
        assert {q["model_snapshot"]["status"] for q in _game(doc, "2026_05_CHI_GB")["quotes"]} == {
            "no_quote"}

    def test_validation_rejects_a_link_after_the_quote(self, monkeypatch, schedule_path, dirs):
        doc = self._doc(monkeypatch, schedule_path, dirs, _wed_events())
        q = _quote(doc, "2026_05_CHI_GB", "betano_ca_on")
        q["model_snapshot"] = dict(q["model_snapshot"], captured_at="2026-10-07T15:59:00Z")
        with pytest.raises(on.ValidationError, match="postdates the quote"):
            on.validate_capture(on._seal(doc))


class TestEmptyCaptureRetry:
    def test_empty_response_keeps_evidence_and_leaves_slot_open(self, monkeypatch, schedule_path, dirs):
        first = _capture(monkeypatch, FakeAPI([]), schedule_path, dirs, WED_SLOT)
        assert first.status == "empty" and first.path.exists() and first.doc["usable"] is False
        empty_bytes = first.path.read_bytes()
        # A retry later in the window captures normally...
        api = FakeAPI(_wed_events("2026-10-07T16:55:00Z"))
        second = _capture(monkeypatch, api, schedule_path, dirs, WED_SLOT + timedelta(hours=1))
        assert second.status == "captured" and len(api.paid_calls) == 1
        assert second.doc["slot"]["slot_id"] == first.doc["slot"]["slot_id"]
        # ...the empty capture is untouched, and the slot counts as captured.
        assert first.path.read_bytes() == empty_bytes
        cov = on.slot_coverage(WED_SLOT - timedelta(hours=1), WED_SLOT + timedelta(hours=2),
                               dirs["capture_dir"])
        assert [(c["slot_id"], c["status"], c["run_id"]) for c in cov] == [
            ("2026-10-07_wednesday_noon", "late", second.doc["run_id"])]
        # A third run now finds the slot filled and makes no call.
        api3 = FakeAPI(_wed_events())
        assert _capture(monkeypatch, api3, schedule_path, dirs,
                        WED_SLOT + timedelta(hours=2)).status == "already_captured"
        assert api3.calls == []

    def test_slot_with_only_empty_captures_reported_empty(self, monkeypatch, schedule_path, dirs):
        _capture(monkeypatch, FakeAPI([]), schedule_path, dirs, WED_SLOT)
        cov = on.slot_coverage(WED_SLOT - timedelta(hours=1), WED_SLOT + timedelta(hours=2),
                               dirs["capture_dir"])
        assert [c["status"] for c in cov] == ["empty"]

    def test_failed_request_writes_nothing_and_allows_retry(self, monkeypatch, schedule_path, dirs):
        with pytest.raises(on.CaptureError):
            _capture(monkeypatch, FakeAPI(odds_status=503), schedule_path, dirs, WED_SLOT)
        assert not list(dirs["capture_dir"].glob("*.json"))
        assert _capture(monkeypatch, FakeAPI(_wed_events()), schedule_path, dirs,
                        WED_SLOT + timedelta(minutes=30)).status == "captured"

    def test_cli_empty_fails_the_run(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "o"))
        monkeypatch.setattr(on, "capture", lambda **kw: on.CaptureResult("empty", "no quotes"))
        assert on.main(["capture"]) == 1
        assert "::error::" in capsys.readouterr().out
        assert "status=empty" in (tmp_path / "o").read_text()


class TestManualSeparation:
    def test_capture_cannot_carry_manual_or_unknown_books(self, monkeypatch, schedule_path, dirs):
        doc = _capture(monkeypatch, FakeAPI(_wed_events()), schedule_path, dirs, WED_SLOT).doc
        q = _quote(doc, "2026_05_TB_DAL", "betano_ca_on")
        q.update(book_key="fanduel_on_manual", jurisdiction="CA-ON")
        with pytest.raises(on.ValidationError, match="isn't an automated API feed"):
            on.validate_capture(on._seal(doc))

    def test_capture_quote_must_be_api_sourced(self, monkeypatch, schedule_path, dirs):
        doc = _capture(monkeypatch, FakeAPI(_wed_events()), schedule_path, dirs, WED_SLOT).doc
        _quote(doc, "2026_05_TB_DAL", "betano_ca_on")["source"] = "manual"
        with pytest.raises(on.ValidationError, match="isn't an automated API feed"):
            on.validate_capture(on._seal(doc))

    def test_manual_quotes_are_not_loaded_as_captures(self, monkeypatch, tmp_path, schedule_path):
        on.manual_quote(game_id="2026_05_TB_DAL", team="DAL", handicap=-3.5, price=-110,
                        observed_at="2026-10-07T16:05:00Z", now=WED_SLOT + timedelta(minutes=10),
                        schedule_path=schedule_path, manual_dir=tmp_path / "manual", repo_dir=ROOT)
        with pytest.raises(on.ValidationError):
            on.load_captures(tmp_path / "manual")


def test_workflow_commits_empty_evidence_and_uploads_files(workflow):
    steps = workflow["jobs"]["capture"]["steps"]
    commit = next(s for s in steps if "Commit" in s.get("name", ""))
    assert "always()" in commit["if"] and "'empty'" in commit["if"] and "'captured'" in commit["if"]
    assert "git pull --rebase" in commit["run"] and "--force" not in commit["run"]
    upload = next(s for s in steps if s.get("uses", "").startswith("actions/upload-artifact"))
    assert upload["if"] == "always()"


def test_credit_check_errors_are_redacted(monkeypatch, schedule_path, dirs):
    def failing(url, params=None, timeout=None):
        raise requests.ConnectionError(f"GET {url}?apiKey={params['apiKey']} failed")
    with pytest.raises(on.CaptureError) as exc:
        _capture(monkeypatch, failing, schedule_path, dirs, WED_SLOT)
    assert KEY not in str(exc.value) and "apiKey=***" in str(exc.value)
    assert not list(dirs["capture_dir"].glob("*.json"))
