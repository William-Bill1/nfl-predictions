"""Immutable injury-report captures (injury_reports.py).

Every source response is mocked; every capture goes to a temporary directory.
Nothing here reads or writes data_files/.
"""

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
import requests

import injury_reports as ir
import ontario_spreads as on
import pregame_snapshots as ps
from test_ontario_spreads import _bash, _git
from test_team_features import ROOT

UTC = timezone.utc
KEY = "SECRETKEY1234567890abcdef"

WED_SLOT = datetime(2026, 10, 7, 16, 0, tzinfo=UTC)     # Wed 12:00 EDT
SUN_SLOT = datetime(2026, 10, 11, 13, 0, tzinfo=UTC)    # Sun 09:00 EDT
SUN_EST = datetime(2026, 11, 1, 14, 0, tzinfo=UTC)      # Sun 09:00 EST (DST ended 02:00 that day)
WED_EST = datetime(2026, 11, 4, 17, 0, tzinfo=UTC)      # Wed 12:00 EST

SCHEDULE_ROWS = [
    ("2026_04_ATL_NO", 2026, 4, "2026-10-05", "20:15", "ATL", "NO", 17, 24),
    ("2026_05_TB_DAL", 2026, 5, "2026-10-08", "20:15", "TB", "DAL", None, None),
    ("2026_05_PHI_JAX", 2026, 5, "2026-10-11", "09:30", "PHI", "JAX", None, None),
    ("2026_05_CHI_GB", 2026, 5, "2026-10-11", "13:00", "CHI", "GB", None, None),
    ("2026_05_DEN_LAC", 2026, 5, "2026-10-11", "16:05", "DEN", "LAC", None, None),
    ("2026_05_BUF_LA", 2026, 5, "2026-10-12", "20:15", "BUF", "LA", None, None),
    ("2026_09_KC_MIA", 2026, 9, "2026-11-08", "13:00", "KC", "MIA", None, None),
]
WEEK5_TEAMS = ["TB", "DAL", "PHI", "JAX", "CHI", "GB", "DEN", "LAC", "BUF", "LA"]


@pytest.fixture
def schedule_path(tmp_path):
    df = pd.DataFrame(SCHEDULE_ROWS, columns=["game_id", "season", "week", "gameday", "gametime",
                                              "away_team", "home_team", "away_score", "home_score"])
    df["game_type"] = "REG"
    path = tmp_path / "nfl_games_historical.csv"
    df.to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture
def cdir(tmp_path):
    return tmp_path / "injury_captures"


@pytest.fixture(autouse=True)
def _no_ci_env(monkeypatch):
    for var in ("GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY", "GITHUB_SHA"):
        monkeypatch.delenv(var, raising=False)


def _entry(name, status="Questionable", pid="4241", injury="Ankle", date="2026-10-07T14:30Z",
           **extra):
    athlete = {"displayName": name, "position": {"abbreviation": "WR"}}
    if pid is not None:
        athlete["id"] = pid
    e = {"athlete": athlete, "details": {"type": injury} if injury else {},
         "type": {"name": f"INJURY_STATUS_{(status or 'x').upper()}", "description": status,
                  "abbreviation": (status or "x")[0]}}
    if status is not None:
        e["status"] = status
    if date is not None:
        e["date"] = date
    e.update(extra)
    return e


def _team(abbr, entries=None, name=None):
    full = name or on.TEAM_FULL_NAME[abbr]
    if entries is None:
        entries = [_entry(f"{abbr} Player One", pid=f"{abs(hash(abbr)) % 10**6}"),
                   _entry(f"{abbr} Player Two", status="Out", pid=None, date=None)]
    return {"id": str(len(full)), "displayName": full, "injuries": entries}


def _payload(teams=WEEK5_TEAMS, extra=()):
    return {"injuries": [_team(t) for t in teams] + list(extra)}


class Resp:
    def __init__(self, payload=None, status=200, body=None, headers=None):
        self.status_code = status
        self.content = body if body is not None else json.dumps(payload).encode()
        self.headers = headers or {"Date": "Wed, 07 Oct 2026 16:00:01 GMT"}


class Source:
    """Mock of requests.get for the ESPN feed; records every call."""

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, url, headers=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return r


def clock_at(*times):
    """clock() returning the given times in order (then the last forever)."""
    seq = list(times)

    def clock():
        return seq.pop(0) if len(seq) > 1 else seq[0]
    return clock


def run(schedule_path, cdir, source, now, received=None, slot="auto", started=None):
    started = started or now
    received = received or started + timedelta(seconds=2)
    return ir.capture(now=now, slot=slot, clock=clock_at(started, received), get=source,
                      capture_dir=cdir, schedule_path=schedule_path, repo_dir=ROOT)


def files(cdir):
    return sorted(cdir.glob("*.json")) if cdir.exists() else []


# --------------------------------------------------------------- outcomes --

class TestOutcomes:
    def test_full_coverage(self, schedule_path, cdir):
        src = Source(Resp(_payload()))
        r = run(schedule_path, cdir, src, WED_SLOT)
        assert r.status == "captured" and r.written
        d = r.doc
        assert d["outcome"] == "full_team_coverage" and d["usable"]
        assert d["scope"]["season"] == 2026 and d["scope"]["week"] == 5
        assert {g["game_id"] for g in d["scope"]["games"]} == {
            "2026_05_TB_DAL", "2026_05_PHI_JAX", "2026_05_CHI_GB", "2026_05_DEN_LAC",
            "2026_05_BUF_LA"}
        assert all(c["state"] == "rows_listed" for c in d["coverage"]["in_scope_teams"].values())
        assert set(d["coverage"]["in_scope_teams"]) == set(WEEK5_TEAMS)
        assert d["coverage"]["player_rows"] == 20
        assert src.calls[0]["url"] == ir.ESPN_INJURIES_URL
        assert src.calls[0]["timeout"] == ir.REQUEST_TIMEOUT           # bounded
        ir.load_captures(cdir)

    def test_partial_when_an_in_scope_team_is_absent(self, schedule_path, cdir):
        r = run(schedule_path, cdir, Source(Resp(_payload(WEEK5_TEAMS[:-1]))), WED_SLOT)
        assert r.status == "partial" and r.doc["outcome"] == "partial_team_coverage"
        assert r.doc["usable"]
        assert r.doc["coverage"]["in_scope_teams"]["LA"] == {"state": "absent_from_feed", "rows": 0}

    def test_team_listed_without_rows_is_not_no_injuries(self, schedule_path, cdir):
        teams = [_team(t) for t in WEEK5_TEAMS[:-1]] + [_team("LA", entries=[])]
        r = run(schedule_path, cdir, Source(Resp({"injuries": teams})), WED_SLOT)
        cov = r.doc["coverage"]
        assert cov["in_scope_teams"]["LA"] == {"state": "listed_without_rows", "rows": 0}
        assert "never means the team has no injuries" in cov["note"]
        assert not any(p["team"] == "LA" for p in r.doc["players"])     # nothing invented

    def test_unmatched_team_and_bye_team_are_explicit(self, schedule_path, cdir):
        extra = [_team("NYG"), _team("XX", name="Gotham Rogues",
                                     entries=[_entry("Mystery Man")])]
        r = run(schedule_path, cdir, Source(Resp(_payload(extra=extra))), WED_SLOT)
        d = r.doc
        assert d["outcome"] == "partial_team_coverage"
        assert d["coverage"]["unmatched_source_teams"] == ["Gotham Rogues"]
        assert d["coverage"]["rows_for_teams_without_game_in_scope"] == 2      # NYG on bye
        states = {p["player_name"]: (p["team"], p["game_id"], p["game_state"]) for p in d["players"]}
        assert states["Mystery Man"] == (None, None, "unmatched_team")
        assert states["NYG Player One"] == ("NYG", None, "no_game_in_scope")
        assert states["TB Player One"] == ("TB", "2026_05_TB_DAL", "pregame")

    def test_malformed_entry_is_recorded_not_dropped_silently(self, schedule_path, cdir):
        bad = _team("TB", entries=[_entry("TB Player One"), {"status": "Out", "athlete": {}}])
        teams = [bad] + [_team(t) for t in WEEK5_TEAMS[1:]]
        r = run(schedule_path, cdir, Source(Resp({"injuries": teams})), WED_SLOT)
        assert r.doc["outcome"] == "partial_team_coverage"
        [m] = r.doc["coverage"]["malformed_entries"]
        assert m["reason"] == "entry has no athlete display name" and m["entry_index"] == 1

    def test_empty_response_is_evidence_not_usable(self, schedule_path, cdir):
        r = run(schedule_path, cdir, Source(Resp({"injuries": []})), WED_SLOT)
        assert r.status == "empty" and not r.doc["usable"] and r.written
        assert all(c["state"] == "absent_from_feed" for c in
                   r.doc["coverage"]["in_scope_teams"].values())

    @pytest.mark.parametrize("response,kind", [
        (requests.Timeout("read timed out"), "timeout"),
        (requests.ConnectionError("no route"), "request_error"),
        (Resp(status=503, body=b"busy"), "http_status"),
        (Resp(body=b"<html>not json"), "invalid_json"),
        (Resp({"teams": []}), "unexpected_shape"),
        (Resp(body=b"x" * (ir.MAX_RESPONSE_BYTES + 1)), "too_large"),
    ])
    def test_failed_responses_are_unavailable(self, schedule_path, cdir, response, kind):
        r = run(schedule_path, cdir, Source(response), WED_SLOT)
        assert r.status == "unavailable" and r.doc["outcome"] == "unavailable"
        assert r.doc["response"]["error"]["type"] == kind
        assert not r.doc["usable"] and r.doc["players"] == []
        ir.load_captures(cdir)                                   # valid evidence file

    def test_never_reads_the_cached_csv(self, schedule_path, cdir, monkeypatch):
        import player_props.injuries as pinj

        def boom(*a, **k):
            raise AssertionError("cached injury CSV must not be used as an observation")
        monkeypatch.setattr(pinj, "load_cached_injuries", boom)
        monkeypatch.setattr(pinj, "get_injury_report", boom)
        monkeypatch.setattr(pinj, "INJURIES_FILE", Path("/nonexistent/espn_injuries.csv"))
        assert run(schedule_path, cdir, Source(Resp(_payload())), WED_SLOT).status == "captured"


# ------------------------------------------------------- raw status values --

class TestRawValues:
    @pytest.fixture
    def rows(self, schedule_path, cdir):
        entries = [
            _entry("A Out", status="Out"),
            _entry("B Questionable", status="Questionable"),
            _entry("C IR", status="Injured Reserve"),
            _entry("D Active", status="Active", injury=None),
            _entry("E DayToDay", status="Day-To-Day"),
            _entry("F NoStatus", status=None),
            _entry("G LinkId", pid=None),
            _entry("H NoDate", date=None),
            _entry("I BadDate", date="sometime Tuesday"),
            _entry("J DNPComment", status="Questionable", shortComment="DNP on Wednesday"),
        ]
        entries[6]["athlete"]["links"] = [{"href": "https://www.espn.com/nfl/player/_/id/3916387/x"}]
        teams = [_team("TB", entries=entries)] + [_team(t) for t in WEEK5_TEAMS[1:]]
        doc = run(schedule_path, cdir, Source(Resp({"injuries": teams})), WED_SLOT).doc
        return {p["player_name"]: p for p in doc["players"]}, doc

    def test_raw_status_kept_and_classified_lexically(self, rows):
        r, _ = rows
        assert (r["A Out"]["source_status"], r["A Out"]["status_class"]) == ("Out", "game_designation_word")
        assert r["C IR"]["source_status"] == "Injured Reserve"            # not rewritten to "IR"
        assert r["C IR"]["status_class"] == "roster_status"
        assert r["D Active"]["status_class"] == "listed_active"
        assert (r["E DayToDay"]["source_status"], r["E DayToDay"]["status_class"]) == \
            ("Day-To-Day", "unrecognized")
        assert (r["F NoStatus"]["source_status"], r["F NoStatus"]["status_class"]) == (None, "missing")

    def test_practice_participation_never_inferred(self, rows):
        r, doc = rows
        assert all(p["practice_participation"] is None for p in r.values())
        assert doc["source"]["practice_participation"] == "not supplied by this source"
        # A DNP comment stays a comment: the status is ESPN's, unchanged.
        j = r["J DNPComment"]
        assert j["short_comment"] == "DNP on Wednesday" and j["source_status"] == "Questionable"

    def test_missing_injury_is_unavailable_not_healthy(self, rows):
        r, _ = rows
        assert r["D Active"]["injury"] == {"type": None, "location": None, "detail": None,
                                           "side": None}

    def test_source_entry_dates_kept_raw_with_unverified_meaning(self, rows):
        r, doc = rows
        assert (r["A Out"]["source_entry_date_raw"], r["A Out"]["source_entry_date_utc"]) == \
            ("2026-10-07T14:30Z", "2026-10-07T14:30:00Z")
        assert (r["H NoDate"]["source_entry_date_raw"], r["H NoDate"]["source_entry_date_utc"]) == \
            (None, None)
        assert (r["I BadDate"]["source_entry_date_raw"], r["I BadDate"]["source_entry_date_utc"]) == \
            ("sometime Tuesday", None)
        ed = doc["source_entry_dates"]
        assert ed["meaning"] == "unverified"
        assert "not status-update times or evidence of freshness" in ed["note"]
        assert (ed["rows_with_unparseable_date"], ed["latest_utc"]) == (1, "2026-10-07T14:30:00Z")
        assert "unverified" in doc["source"]["entry_date"]

    def test_no_update_time_claims_anywhere(self, rows):
        # Regression: the entry date used to be stored as source_updated_at /
        # latest_source_update, a last-update claim that was never verified.
        _, doc = rows
        text = json.dumps(doc)
        for claim in ("source_updated", "latest_source_update", "source_timing", "update_time"):
            assert claim not in text

    def test_no_source_dates_at_all_is_stated(self, schedule_path, cdir):
        teams = [_team(t, entries=[_entry(f"{t} P", date=None)]) for t in WEEK5_TEAMS]
        doc = run(schedule_path, cdir, Source(Resp({"injuries": teams})), WED_SLOT).doc
        ed = doc["source_entry_dates"]
        assert (ed["latest_utc"], ed["rows_with_date"]) == (None, 0)
        assert ed["note"] == "the source supplied no entry dates; unavailable"
        assert all(p["source_entry_date_raw"] is None and p["source_entry_date_utc"] is None
                   for p in doc["players"])                               # never filled in

    def test_http_headers_kept_separate_from_entry_dates(self, schedule_path, cdir):
        teams = [_team(t, entries=[_entry(f"{t} P", date=None)]) for t in WEEK5_TEAMS]
        hdr = {"Date": "Wed, 07 Oct 2026 16:00:01 GMT",
               "Last-Modified": "Tue, 06 Oct 2026 10:00:00 GMT"}
        doc = run(schedule_path, cdir, Source(Resp({"injuries": teams}, headers=hdr)), WED_SLOT).doc
        assert doc["response"]["http_last_modified"] == "Tue, 06 Oct 2026 10:00:00 GMT"
        assert doc["response"]["http_date"] == "Wed, 07 Oct 2026 16:00:01 GMT"
        assert doc["source_entry_dates"]["latest_utc"] is None              # not borrowed
        assert "not player-status updates" in doc["source"]["http_headers"]

    def test_player_identifiers(self, rows):
        r, _ = rows
        assert (r["A Out"]["player_id"], r["A Out"]["player_id_source"]) == ("4241", "athlete.id")
        assert (r["G LinkId"]["player_id"], r["G LinkId"]["player_id_source"]) == \
            ("3916387", "athlete.links")

    def test_missing_identifier_is_explicit(self, schedule_path, cdir):
        teams = [_team(t, entries=[_entry(f"{t} P", pid=None)]) for t in WEEK5_TEAMS]
        doc = run(schedule_path, cdir, Source(Resp({"injuries": teams})), WED_SLOT).doc
        assert {(p["player_id"], p["player_id_source"]) for p in doc["players"]} == {(None, None)}


# ---------------------------------------------------------------- timing --

class TestTiming:
    def test_times_are_actual_not_slot(self, schedule_path, cdir):
        start = WED_SLOT + timedelta(minutes=7, seconds=3, microseconds=120)
        got = start + timedelta(seconds=4, microseconds=5)
        d = run(schedule_path, cdir, Source(Resp(_payload())), start, received=got).doc
        assert d["request_started_at"] == "2026-10-07T16:07:03.000120Z"
        assert d["received_at"] == d["observed_at"] == "2026-10-07T16:07:07.000125Z"
        assert d["slot"]["intended_utc"] == "2026-10-07T16:00:00Z"
        assert d["slot"]["delay_minutes"] == 7.1 and d["slot"]["status"] == "on_time"
        assert d["slot"]["judged_at"] == "received_at"

    @pytest.mark.parametrize("slot_utc,local", [
        (WED_SLOT, "2026-10-07T12:00:00-04:00"),
        (SUN_SLOT, "2026-10-11T09:00:00-04:00"),
        (SUN_EST, "2026-11-01T09:00:00-05:00"),
        (WED_EST, "2026-11-04T12:00:00-05:00"),
    ])
    def test_toronto_slots_across_dst(self, schedule_path, cdir, slot_utc, local):
        d = run(schedule_path, cdir, Source(Resp(_payload())), slot_utc).doc
        assert d["slot"]["intended_utc"] == ps._iso(slot_utc)
        assert d["slot"]["intended_local"] == local

    def test_dst_hour_early_is_not_due(self, schedule_path, cdir):
        # 13:00 UTC on 2026-11-01 is 08:00 EST: the EDT cron entry, an hour early.
        src = Source(Resp(_payload()))
        r = run(schedule_path, cdir, src, SUN_EST - timedelta(hours=1))
        assert r.status == "not_due" and src.calls == [] and files(cdir) == []

    @pytest.mark.parametrize("offset,status", [
        (timedelta(seconds=-1), "not_due"),
        (timedelta(0), "on_time"),
        (timedelta(minutes=30), "on_time"),
        (timedelta(minutes=30, seconds=1), "late"),
        (timedelta(hours=3), "late"),                    # window end is inclusive
        (timedelta(hours=3, seconds=1), "not_due"),
    ])
    def test_slot_boundaries(self, schedule_path, cdir, offset, status):
        src = Source(Resp(_payload()))
        t = WED_SLOT + offset
        r = run(schedule_path, cdir, src, t, received=t)
        if status == "not_due":
            assert r.status == "not_due" and src.calls == []
        else:
            assert r.doc["slot"]["status"] == status and r.doc["slot"]["fills_slot"]

    def test_received_after_window_is_labelled_and_does_not_fill(self, schedule_path, cdir):
        start = WED_SLOT + timedelta(hours=3) - timedelta(seconds=1)
        got = WED_SLOT + timedelta(hours=3, seconds=20)
        r = run(schedule_path, cdir, Source(Resp(_payload())), start, received=got)
        s = r.doc["slot"]
        assert (s["status"], s["fills_slot"]) == ("outside_window", False)
        assert "does not fill the slot" in r.message
        assert ir._already_captured(on.resolve_slot(start, "auto"), cdir) is None

    def test_ad_hoc_is_labelled_and_never_fills(self, schedule_path, cdir):
        t = datetime(2026, 10, 9, 2, 30, tzinfo=UTC)          # Thu 22:30 Toronto
        r = run(schedule_path, cdir, Source(Resp(_payload())), t, slot="ad_hoc")
        s = r.doc["slot"]
        assert s == {"name": "ad_hoc", "slot_id": "2026-10-08_ad_hoc", "timezone": "America/Toronto",
                     "intended_utc": None, "intended_local": None, "delay_minutes": None,
                     "status": "ad_hoc", "fills_slot": False, "judged_at": "received_at"}

    def test_ad_hoc_in_window_does_not_block_the_slot(self, schedule_path, cdir):
        run(schedule_path, cdir, Source(Resp(_payload())), WED_SLOT, slot="ad_hoc")
        src = Source(Resp(_payload()))
        r = run(schedule_path, cdir, src, WED_SLOT + timedelta(minutes=5))
        assert r.status == "captured" and len(src.calls) == 1

    def test_no_retrospective_capture_of_a_past_slot(self, schedule_path, cdir):
        src = Source(Resp(_payload()))
        r = run(schedule_path, cdir, src, datetime(2026, 10, 8, 16, 0, tzinfo=UTC),
                slot="wednesday_noon")
        assert r.status == "not_due" and src.calls == []

    def test_scope_follows_the_observation_time(self, schedule_path, cdir):
        # Sunday morning: the Thursday game has kicked off and is kept as such.
        d = run(schedule_path, cdir, Source(Resp(_payload())), SUN_SLOT).doc
        states = {g["game_id"]: g["state"] for g in d["scope"]["games"]}
        assert states["2026_05_TB_DAL"] == "kicked_off" and states["2026_05_CHI_GB"] == "pregame"


# ------------------------------------------------- repeats and immutability --

class TestRepeatsAndImmutability:
    def test_repeat_after_success_makes_no_request(self, schedule_path, cdir):
        run(schedule_path, cdir, Source(Resp(_payload())), WED_SLOT)
        before = {p: p.read_bytes() for p in files(cdir)}
        src = Source(Resp(_payload()))
        r = run(schedule_path, cdir, src, WED_SLOT + timedelta(hours=1))
        assert r.status == "already_captured" and src.calls == [] and not r.written
        assert {p: p.read_bytes() for p in files(cdir)} == before

    def test_partial_fills_the_slot(self, schedule_path, cdir):
        run(schedule_path, cdir, Source(Resp(_payload(WEEK5_TEAMS[:3]))), WED_SLOT)
        src = Source(Resp(_payload()))
        assert run(schedule_path, cdir, src, WED_SLOT + timedelta(hours=1)).status == "already_captured"

    @pytest.mark.parametrize("first", [Resp({"injuries": []}), requests.Timeout("t")])
    def test_failure_leaves_slot_open_and_retry_keeps_both(self, schedule_path, cdir, first):
        r1 = run(schedule_path, cdir, Source(first), WED_SLOT)
        kept = r1.path.read_bytes()
        src = Source(Resp(_payload()))
        r2 = run(schedule_path, cdir, src, WED_SLOT + timedelta(minutes=61))
        assert r2.status == "captured" and len(src.calls) == 1
        assert len(files(cdir)) == 2 and r1.path.read_bytes() == kept
        assert r2.doc["slot"]["status"] == "late"

    def test_same_content_is_a_noop_and_different_content_conflicts(self, schedule_path, cdir):
        r = run(schedule_path, cdir, Source(Resp(_payload())), WED_SLOT)
        original = r.path.read_bytes()
        assert ir.write_capture(r.doc, cdir) == (r.path, False)
        other = json.loads(original)
        other["players"] = other["players"][:-1]
        other["coverage"]["player_rows"] -= 1
        other["status_counts"] = ir._counts(p["status_class"] for p in other["players"])
        other["source_entry_dates"] = ir.entry_date_summary(other["players"])
        ir._seal(other)
        ir.validate_capture(other)
        with pytest.raises(ir.CaptureConflictError):
            ir.write_capture(other, cdir)
        assert r.path.read_bytes() == original

    def test_lock_held_means_no_request(self, schedule_path, cdir):
        cdir.mkdir(parents=True)
        (cdir / ".2026-10-07_wednesday_noon.lock").write_text("{}")
        src = Source(Resp(_payload()))
        with pytest.raises(on.CaptureError):
            run(schedule_path, cdir, src, WED_SLOT)
        assert src.calls == [] and files(cdir) == []


# --------------------------------------------------------- fail closed --

class TestFailClosed:
    @pytest.fixture
    def stored(self, schedule_path, cdir):
        r = run(schedule_path, cdir, Source(Resp(_payload())), WED_SLOT)
        return r.path

    def _rewrite(self, path, mutate, reseal=False):
        doc = json.loads(path.read_text(encoding="utf-8"))
        mutate(doc)
        if reseal:
            ir._seal(doc)
        path.write_text(json.dumps(doc), encoding="utf-8")

    def test_edited_row_fails_checksum(self, stored, cdir):
        self._rewrite(stored, lambda d: d["players"][0].update(source_status="Out"))
        with pytest.raises(ir.ValidationError, match="checksum"):
            ir.load_captures(cdir)

    def test_unknown_schema_version_rejected_even_if_resealed(self, stored, cdir):
        self._rewrite(stored, lambda d: d.update(schema_version=2), reseal=True)
        with pytest.raises(ir.ValidationError, match="schema_version"):
            ir.load_captures(cdir)

    def test_bad_json_and_wrong_name_rejected(self, stored, cdir):
        bad = cdir / "20261007T160000Z-aaaaaaaaaaaa.json"
        bad.write_text("{not json")
        with pytest.raises(ir.ValidationError, match="not valid JSON"):
            ir.load_captures(cdir)
        bad.unlink()
        stored.rename(cdir / "20261007T160000Z-bbbbbbbbbbbb.json")
        with pytest.raises(ir.ValidationError, match="file name"):
            ir.load_captures(cdir)

    def test_malformed_nested_value_is_a_validation_error(self, stored, cdir):
        self._rewrite(stored, lambda d: d.update(coverage=[]), reseal=True)
        with pytest.raises(ir.ValidationError, match="malformed"):
            ir.load_captures(cdir)

    @pytest.mark.parametrize("mutate,match", [
        (lambda d: d.update(received_at="2026-10-07T15:59:00.000000Z",
                            observed_at="2026-10-07T15:59:00.000000Z",
                            request_started_at="2026-10-07T15:58:59.000000Z"), "before the slot"),
        (lambda d: d["slot"].update(delay_minutes=5.0), "delay_minutes"),
        (lambda d: d.update(observed_at="2026-10-07T16:00:00.000000Z"), "observed_at"),
        (lambda d: d.update(received_at="2026-10-07T16:45:00.000000Z",
                            observed_at="2026-10-07T16:45:00.000000Z"), "delay_minutes"),
        (lambda d: d["slot"].update(fills_slot=False), "fills_slot"),
        (lambda d: d["players"][0].update(practice_participation="DNP"), "practice"),
        (lambda d: d["players"][0].update(status_class="game_designation_word",
                                          source_status="Day-To-Day"), "status_class"),
        (lambda d: d["coverage"]["in_scope_teams"].update(
            LA={"state": "absent_from_feed", "rows": 0}), "outcome should be partial"),
        (lambda d: d["source_entry_dates"].update(meaning="last status update"), "entry dates"),
        (lambda d: d["players"][0].update(source_entry_date_raw=None), "entry date"),
    ])
    def test_inconsistent_resealed_captures_rejected(self, stored, cdir, mutate, match):
        self._rewrite(stored, mutate, reseal=True)
        with pytest.raises(ir.ValidationError, match=match):
            ir.load_captures(cdir)

    def test_late_relabelled_on_time_rejected(self, schedule_path, cdir):
        t = WED_SLOT + timedelta(minutes=45)
        p = run(schedule_path, cdir, Source(Resp(_payload())), t, received=t).path
        self._rewrite(p, lambda d: d["slot"].update(status="on_time"), reseal=True)
        with pytest.raises(ir.ValidationError, match="should be late"):
            ir.load_captures(cdir)

    def test_unkeyed_checksum_limit_is_real(self, stored, cdir):
        # Documented limitation: a deliberate, internally consistent edit
        # followed by recomputing the checksum is NOT detected.
        def edit(d):
            d["players"][0]["short_comment"] = "edited"
        self._rewrite(stored, edit, reseal=True)
        assert ir.load_captures(cdir)[0]["players"][0]["short_comment"] == "edited"

    @pytest.mark.parametrize("slot", ["auto", "ad_hoc"])
    def test_capture_refuses_when_store_is_corrupt(self, stored, cdir, schedule_path, slot):
        self._rewrite(stored, lambda d: d.update(outcome="empty"))
        src = Source(Resp(_payload()))
        with pytest.raises(ir.ValidationError):
            run(schedule_path, cdir, src, WED_SLOT + timedelta(minutes=5), slot=slot)
        assert src.calls == []


# ------------------------------------------------------- secrets --

def test_no_secrets_in_stored_payload_or_errors(schedule_path, cdir, monkeypatch, capsys):
    monkeypatch.setenv("ODDS_API_KEY", KEY)
    leak = requests.ConnectionError(f"GET https://x/?apiKey={KEY} failed with {KEY}")
    r = run(schedule_path, cdir, Source(leak), WED_SLOT)
    text = r.path.read_text(encoding="utf-8")
    assert KEY not in text and KEY not in r.message and "apiKey=***" in text
    entries = [_entry("Leaky", shortComment=f"see https://x/?apiKey={KEY}")]
    teams = [_team("TB", entries=entries)] + [_team(t) for t in WEEK5_TEAMS[1:]]
    r2 = run(schedule_path, cdir, Source(Resp({"injuries": teams})), WED_SLOT + timedelta(hours=1),
             slot="ad_hoc")
    assert KEY not in r2.path.read_text(encoding="utf-8")
    assert KEY not in capsys.readouterr().out
    assert "Authorization" not in json.dumps(ir.HEADERS)          # no credentials are sent


# ------------------------------------------------ isolation from sportsbooks --

class TestSportsbookIsolation:
    def test_injury_capture_touches_only_its_own_store(self, schedule_path, tmp_path, monkeypatch):
        odds_calls = []
        monkeypatch.setattr(on.requests, "get", lambda *a, **k: odds_calls.append(a) or 1 / 0)
        sports = tmp_path / "ontario_captures"
        monkeypatch.setattr(on, "CAPTURE_DIR", sports)
        for resp in (requests.Timeout("t"), Resp(_payload())):
            run(schedule_path, tmp_path / "inj", Source(resp), WED_SLOT, slot="ad_hoc")
        assert odds_calls == [] and not sports.exists()

    @pytest.mark.parametrize("injury_first", [None, requests.Timeout("down"),
                                              Resp({"injuries": []}), Resp(_payload())])
    def test_sportsbook_capture_unchanged_by_injury_outcome(self, schedule_path, tmp_path,
                                                            monkeypatch, injury_first):
        import test_ontario_spreads as tos

        def sportsbook(tag):
            dirs = {"capture_dir": tmp_path / tag / "cap", "snapshot_dir": tmp_path / tag / "snap"}
            api = tos.FakeAPI(tos._wed_events())
            result = tos._capture(monkeypatch, api, schedule_path, dirs, WED_SLOT)
            return result, api, dirs
        base, base_api, _ = sportsbook("baseline")
        if injury_first is not None:
            run(schedule_path, tmp_path / "inj", Source(injury_first), WED_SLOT)
        after, api, dirs = sportsbook("after")
        assert after.status == base.status == "captured"
        assert len(api.paid_calls) == len(base_api.paid_calls) == 1
        assert api.calls == base_api.calls
        for key in ("slot", "games", "coverage", "usable", "excluded_games"):
            assert after.doc[key] == base.doc[key]
        assert [p.name for p in dirs["capture_dir"].glob("*.json")] == [after.path.name]


# ------------------------------------------------------- CLI and summary --

def _cli(monkeypatch, tmp_path, schedule_path, cdir, source, now, received=None, argv=("capture",)):
    out, summary = tmp_path / "gh_output", tmp_path / "gh_summary"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(ir, "CAPTURE_DIR", cdir)
    kw = dict(ir.capture.__kwdefaults__)
    kw.update(now=now, capture_dir=cdir, schedule_path=schedule_path, get=source,
              clock=clock_at(now, received or now + timedelta(seconds=1)), repo_dir=ROOT)
    monkeypatch.setattr(ir.capture, "__kwdefaults__", kw)
    code = ir.main(list(argv))
    read = lambda p: p.read_text(encoding="utf-8") if p.exists() else ""  # noqa: E731
    return code, read(out), read(summary)


class TestCli:
    def test_success_reports_status_and_written(self, monkeypatch, tmp_path, schedule_path, cdir,
                                                capsys):
        code, out, summ = _cli(monkeypatch, tmp_path, schedule_path, cdir,
                               Source(Resp(_payload())), WED_SLOT + timedelta(minutes=3))
        assert code == 0 and "status=captured" in out and "written=true" in out
        [path] = files(cdir)
        assert f"sha256={ir.file_sha256(path)}" in out and f"file={path.as_posix()}" in out
        assert "Injury report capture (evidence only)" in summ and "**captured**" in summ
        assert "observed (received) 2026-10-07T16:03:01" in summ
        assert "Wed 2026-10-07 12:03:01 EDT" in summ

    def test_unavailable_fails_step_visibly_and_keeps_evidence(self, monkeypatch, tmp_path,
                                                                 schedule_path, cdir, capsys):
        code, out, summ = _cli(monkeypatch, tmp_path, schedule_path, cdir,
                               Source(requests.ConnectionError("dns")), WED_SLOT)
        assert code == 1 and "status=unavailable" in out and "written=true" in out
        assert "❌" in summ and "sportsbook capture steps are unaffected" in summ
        assert "::error::" in capsys.readouterr().out and len(files(cdir)) == 1

    def test_partial_warns(self, monkeypatch, tmp_path, schedule_path, cdir, capsys):
        code, out, summ = _cli(monkeypatch, tmp_path, schedule_path, cdir,
                               Source(Resp(_payload(WEEK5_TEAMS[:4]))), WED_SLOT)
        assert code == 0 and "status=partial" in out and "⚠️" in summ
        assert "::warning::" in capsys.readouterr().out

    def test_not_due_writes_nothing(self, monkeypatch, tmp_path, schedule_path, cdir):
        src = Source(Resp(_payload()))
        code, out, _ = _cli(monkeypatch, tmp_path, schedule_path, cdir, src,
                            WED_SLOT - timedelta(hours=1))
        assert code == 0 and "status=not_due" in out and "written=false" in out and src.calls == []

    def test_corrupt_store_fails_closed(self, monkeypatch, tmp_path, schedule_path, cdir, capsys):
        cdir.mkdir(parents=True)
        (cdir / "20261007T160000Z-aaaaaaaaaaaa.json").write_text("{}")
        src = Source(Resp(_payload()))
        code, out, summ = _cli(monkeypatch, tmp_path, schedule_path, cdir, src, WED_SLOT)
        assert code == 1 and "status=failed" in out and "written=false" in out and src.calls == []
        assert "**failed**" in summ

    def test_list_and_validate(self, monkeypatch, tmp_path, schedule_path, cdir, capsys):
        run(schedule_path, cdir, Source(requests.Timeout("t")), WED_SLOT)
        run(schedule_path, cdir, Source(Resp(_payload(WEEK5_TEAMS[:-1]))),
            WED_SLOT + timedelta(minutes=40))
        monkeypatch.setattr(ir, "CAPTURE_DIR", cdir)
        capsys.readouterr()
        assert ir.main(["list"]) == 0
        text = capsys.readouterr().out
        assert "2 injury-report capture(s), all valid" in text
        assert "unavailable (timeout: t)" in text and "late (+40.0 min)" in text
        assert "source ESPN espn_injuries_json" in text and "season 2026 week 5" in text
        assert "source entry dates (meaning unverified)" in text and "update" not in text
        assert "absent_from_feed" in text and "EDT" in text
        assert ir.main(["validate"]) == 0 and "2 capture(s) valid" in capsys.readouterr().out
        next(iter(files(cdir))).write_text("{}")
        assert ir.main(["list"]) == 1 and "::error::" in capsys.readouterr().out


# ------------------------------------------------------- workflow --

@pytest.fixture(scope="module")
def workflow():
    import yaml
    return yaml.safe_load((ROOT / ".github" / "workflows" / "ontario-spread-capture.yml")
                          .read_text(encoding="utf-8"))


INJURY_IDS = ("injury", "injury_commit", "injury_upload")


class TestWorkflow:
    def _steps(self, workflow):
        return workflow["jobs"]["capture"]["steps"]

    def test_injury_steps_follow_every_sportsbook_step(self, workflow):
        steps = self._steps(workflow)
        names = [s.get("name") for s in steps]
        first_injury = names.index("Capture injury report")
        assert names.index("Report persistence") < first_injury
        assert names.index("Capture Ontario spreads") < names.index("Commit new capture files") \
            < first_injury
        injury = [s for s in steps if s.get("id") in INJURY_IDS
                  or s.get("name") == "Report injury capture persistence"]
        assert len(injury) == 4
        assert all(s.get("continue-on-error") is True for s in injury)
        assert all("ODDS_API_KEY" not in json.dumps(s) for s in injury)

    def test_sportsbook_steps_unchanged(self, workflow):
        steps = self._steps(workflow)
        cap = next(s for s in steps if s.get("id") == "capture")
        assert "continue-on-error" not in cap
        assert cap["run"].strip().splitlines()[-1] == \
            'python ontario_spreads.py capture --slot "$SLOT" | tee "$RUNNER_TEMP/ontario_capture.txt"'
        commit = next(s for s in steps if s.get("id") == "commit")
        assert "injury" not in commit["run"] and "injury" not in commit["if"]

    def test_injury_capture_guarded_and_slot_shared(self, workflow):
        cap = next(s for s in self._steps(workflow) if s.get("id") == "injury")
        assert cap["if"] == "always() && steps.refresh.outcome != 'failure'"
        assert cap["env"] == {"SLOT": "${{ github.event.inputs.slot || 'auto' }}"}
        assert "injury_reports.py capture" in cap["run"] and "set -o pipefail" in cap["run"]

    def test_injury_commit_scope(self, workflow):
        c = next(s for s in self._steps(workflow) if s.get("id") == "injury_commit")
        assert "steps.injury.outputs.written == 'true'" in c["if"]
        assert c["env"] == {"FILE": "${{ steps.injury.outputs.file }}"}
        assert 'git add -- "$FILE"' in c["run"]                    # only this run's file
        assert "ontario_spreads" not in c["run"] and "--force" not in c["run"]
        assert "@{u}..HEAD" in c["run"]

    def test_upload_holds_exactly_this_runs_file(self, workflow):
        up = next(s for s in self._steps(workflow) if s.get("id") == "injury_upload")
        assert up["with"]["path"] == "${{ steps.injury.outputs.file }}"
        assert up["with"]["if-no-files-found"] == "error"          # success means uploaded

    def test_persistence_verified_against_remote(self, workflow):
        rep = next(s for s in self._steps(workflow)
                   if s.get("name") == "Report injury capture persistence")
        assert "injury_reports.py persistence" in rep["run"]
        assert "git status" not in rep["run"]                      # a clean tree isn't proof
        assert rep["env"]["FILE"] == "${{ steps.injury.outputs.file }}"
        assert rep["env"]["SHA256"] == "${{ steps.injury.outputs.sha256 }}"
        assert rep["env"]["UPLOAD"] == "${{ steps.injury_upload.outcome }}"
        assert rep["env"]["STATUS"] == "${{ steps.injury.outputs.status }}"
        assert rep["if"] == "always() && steps.injury.outcome != 'skipped'"


def _commit_script(workflow):
    step = next(s for s in workflow["jobs"]["capture"]["steps"] if s.get("id") == "injury_commit")
    return step["run"]


ARTIFACT = "injury-report-capture-1-1"
REL = "data_files/injury_reports/captures/20261007T160001Z-aaaaaaaaaaaa.json"


@pytest.mark.skipif(_bash() is None, reason="bash not available")
class TestPersistence:
    """The workflow's injury commit script, then the persistence report, run
    against a local bare 'origin' (no network)."""

    def _clone(self, tmp_path):
        bare, seed, work = tmp_path / "origin.git", tmp_path / "seed", tmp_path / "work"
        _git(tmp_path, "init", "--bare", str(bare))
        _git(tmp_path, "init", str(seed))
        (seed / "README").write_text("x\n")
        _git(seed, "add", "README")
        _git(seed, "commit", "-m", "init")
        _git(seed, "push", str(bare), "HEAD:main")
        _git(tmp_path, "clone", str(bare), str(work))
        _git(work, "config", "core.autocrlf", "false")
        return bare, work

    def _commit(self, work, workflow, file=REL):
        env = {**os.environ, "FILE": file,
               "GIT_AUTHOR_NAME": "t", "GIT_COMMITTER_NAME": "t",
               "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
        script = "sleep() { :; }\n" + _commit_script(workflow)       # no real back-off waits
        return subprocess.run([_bash(), "-c", script], cwd=work, capture_output=True, text=True,
                              env=env)

    def _capture_file(self, work, content='{"capture": 1}\n'):
        path = work / REL
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode())
        return ir.file_sha256(path)

    def _reject_pushes(self, bare):
        hook = bare / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\necho rejected by test >&2\nexit 1\n", newline="\n")
        hook.chmod(0o755)

    def _report(self, monkeypatch, tmp_path, work, sha, upload="success", status="captured",
                written="true", file=REL):
        summary = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        code = ir.main(["persistence", "--status", status, "--written", written, "--file", file,
                        "--sha256", sha, "--upload", upload, "--artifact", ARTIFACT,
                        "--repo", str(work)])
        return code, summary.read_text(encoding="utf-8") if summary.exists() else ""

    def test_commit_and_push_succeed(self, tmp_path, workflow, monkeypatch, capsys):
        bare, work = self._clone(tmp_path)
        sha = self._capture_file(work)
        (work / "other.txt").write_text("untracked, not committed\n")
        r = self._commit(work, workflow)
        assert r.returncode == 0, r.stdout + r.stderr
        shown = _git(bare, "show", "--name-only", "--format=%s", "main")
        assert "Injury report capture" in shown and REL in shown and "other.txt" not in shown
        code, summ = self._report(monkeypatch, tmp_path, work, sha)
        assert code == 0
        assert "persisted: on remote main with identical content ✅" in summ
        assert "artifact" not in summ and "::error::" not in capsys.readouterr().out

    @pytest.mark.parametrize("upload", ["success", "failure"])
    def test_push_fails_with_clean_tree_is_not_persisted(self, tmp_path, workflow, monkeypatch,
                                                          capsys, upload):
        # Regression: the old report read `git status`, which is clean here.
        bare, work = self._clone(tmp_path)
        self._reject_pushes(bare)
        sha = self._capture_file(work)
        r = self._commit(work, workflow)
        assert r.returncode == 1 and "committed locally only" in r.stdout
        assert _git(work, "status", "--porcelain") == ""                 # clean tree
        assert REL not in _git(bare, "ls-tree", "-r", "--name-only", "main")
        code, summ = self._report(monkeypatch, tmp_path, work, sha, upload=upload)
        assert code == 1 and "✅" not in summ
        assert "committed locally but not on remote main" in summ
        out = capsys.readouterr().out
        if upload == "success":
            assert f"workflow artifact **{ARTIFACT}**" in summ and f"artifact {ARTIFACT}" in out
        else:
            assert "**not persisted anywhere**" in summ and "not persisted anywhere" in out

    @pytest.mark.parametrize("upload", ["success", "failure", ""])
    def test_uncommitted_capture(self, tmp_path, monkeypatch, upload):
        _, work = self._clone(tmp_path)
        sha = self._capture_file(work)
        code, summ = self._report(monkeypatch, tmp_path, work, sha, upload=upload)
        assert code == 1 and "NOT persisted: never committed" in summ
        assert (f"**{ARTIFACT}**" in summ) is (upload == "success")
        assert ("not persisted anywhere" in summ) is (upload != "success")

    def test_remote_verification_failure_is_unverified(self, tmp_path, workflow, monkeypatch):
        _, work = self._clone(tmp_path)
        sha = self._capture_file(work)
        assert self._commit(work, workflow).returncode == 0
        _git(work, "remote", "set-url", "origin", str(tmp_path / "gone.git"))
        code, summ = self._report(monkeypatch, tmp_path, work, sha, upload="failure")
        assert code == 1 and "persistence UNVERIFIED" in summ and "✅" not in summ
        assert "remote check error" in summ and "not persisted anywhere" in summ

    def test_remote_has_different_content(self, tmp_path, monkeypatch):
        bare, work = self._clone(tmp_path)
        other = tmp_path / "other"
        _git(tmp_path, "clone", str(bare), str(other))
        self._capture_file(other, '{"capture": "someone else"}\n')
        _git(other, "add", REL)
        _git(other, "commit", "-m", "x")
        _git(other, "push", "origin", "HEAD:main")
        sha = self._capture_file(work)
        code, summ = self._report(monkeypatch, tmp_path, work, sha)
        assert code == 1 and "different content" in summ

    def test_unpushed_sportsbook_commit_blocks_injury_commit(self, tmp_path, workflow, monkeypatch):
        bare, work = self._clone(tmp_path)
        (work / "sportsbook.json").write_text("{}\n")
        _git(work, "add", "sportsbook.json")
        _git(work, "commit", "-m", "Ontario spread capture [skip ci]")
        before = _git(bare, "rev-parse", "main")
        sha = self._capture_file(work)
        r = self._commit(work, workflow)
        assert r.returncode == 1 and "unpushed sportsbook commits present" in r.stdout
        assert _git(bare, "rev-parse", "main") == before                 # nothing pushed
        assert "sportsbook.json" not in _git(bare, "ls-tree", "-r", "--name-only", "main")
        code, summ = self._report(monkeypatch, tmp_path, work, sha)
        assert code == 1 and "never committed" in summ and f"**{ARTIFACT}**" in summ

    def test_commit_refuses_other_paths(self, tmp_path, workflow):
        bare, work = self._clone(tmp_path)
        before = _git(bare, "rev-parse", "main")
        (work / "data_files").mkdir()
        (work / "data_files" / "x.json").write_text("{}")
        r = self._commit(work, workflow, file="data_files/x.json")
        assert r.returncode == 1 and "unexpected injury capture path" in r.stdout
        assert _git(bare, "rev-parse", "main") == before

    def test_no_status_and_no_new_file(self, tmp_path, monkeypatch, capsys):
        _, work = self._clone(tmp_path)
        code, summ = self._report(monkeypatch, tmp_path, work, "", status="")
        assert code == 1 and "ended without a status" in summ
        code, summ = self._report(monkeypatch, tmp_path, work, "", status="already_captured",
                                  written="false")
        assert code == 0 and "no new injury capture file this run" in summ
        code, summ = self._report(monkeypatch, tmp_path, work, "not-a-sha")
        assert code == 1 and "UNVERIFIED" in summ


# --------------------------------------------- final window check (under lock)

class TestFinalWindowCheck:
    CLOSE = WED_SLOT + timedelta(hours=3)                 # Wed 15:00 Toronto, inclusive

    def _run(self, schedule_path, cdir, started, received=None):
        src = Source(Resp(_payload()))
        r = run(schedule_path, cdir, src, self.CLOSE - timedelta(seconds=1), started=started,
                received=received or started)
        return r, src

    @pytest.mark.parametrize("late_by", [timedelta(microseconds=1), timedelta(seconds=1),
                                         timedelta(minutes=5)])
    def test_window_closes_before_the_request(self, schedule_path, cdir, late_by):
        # Resolved inside the window, but schedule loading / the lock took us past it.
        r, src = self._run(schedule_path, cdir, self.CLOSE + late_by)
        assert r.status == "window_closed" and src.calls == [] and files(cdir) == []
        assert "window closed at 2026-10-07T19:00:00Z" in r.message and "no request" in r.message

    def test_exactly_at_close_is_inside(self, schedule_path, cdir):
        r, src = self._run(schedule_path, cdir, self.CLOSE)
        assert r.status == "captured" and len(src.calls) == 1
        assert (r.doc["slot"]["status"], r.doc["slot"]["fills_slot"]) == ("late", True)
        assert r.doc["request_started_at"] == "2026-10-07T19:00:00.000000Z"

    def test_response_after_close_is_evidence_only(self, schedule_path, cdir):
        r, src = self._run(schedule_path, cdir, self.CLOSE, self.CLOSE + timedelta(seconds=3))
        assert len(src.calls) == 1
        s = r.doc["slot"]
        assert (s["status"], s["fills_slot"]) == ("outside_window", False)
        assert r.doc["request_started_at"] == "2026-10-07T19:00:00.000000Z"
        assert r.doc["received_at"] == "2026-10-07T19:00:03.000000Z"

    def test_never_switches_slot_or_to_ad_hoc(self, schedule_path, cdir):
        r, _ = self._run(schedule_path, cdir, self.CLOSE + timedelta(hours=1))
        assert r.status == "window_closed" and r.doc is None and files(cdir) == []

    def test_sunday_window(self, schedule_path, cdir):
        close = SUN_SLOT + timedelta(hours=2)
        src = Source(Resp(_payload()))
        r = run(schedule_path, cdir, src, close, started=close + timedelta(microseconds=1))
        assert r.status == "window_closed" and src.calls == []

    def test_ad_hoc_has_no_window(self, schedule_path, cdir):
        src = Source(Resp(_payload()))
        t = self.CLOSE + timedelta(hours=5)
        r = run(schedule_path, cdir, src, t, slot="ad_hoc", started=t + timedelta(minutes=2))
        assert r.status == "captured" and r.doc["slot"]["status"] == "ad_hoc"

    def test_duplicate_found_under_the_lock(self, schedule_path, cdir, tmp_path, monkeypatch):
        # Another run's capture lands between the first check and the lock.
        other = run(schedule_path, tmp_path / "other", Source(Resp(_payload())), WED_SLOT)
        real_load = on.load_schedule

        def load_and_race(path):
            cdir.mkdir(parents=True, exist_ok=True)
            (cdir / other.path.name).write_bytes(other.path.read_bytes())
            return real_load(path)
        monkeypatch.setattr(on, "load_schedule", load_and_race)
        src = Source(Resp(_payload()))
        r = run(schedule_path, cdir, src, WED_SLOT + timedelta(minutes=10))
        assert r.status == "already_captured" and src.calls == []
        assert [p.name for p in files(cdir)] == [other.path.name]

    def test_cli_reports_window_closed(self, monkeypatch, tmp_path, schedule_path, cdir, capsys):
        out_f, summary = tmp_path / "o", tmp_path / "s"
        monkeypatch.setenv("GITHUB_OUTPUT", str(out_f))
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        kw = dict(ir.capture.__kwdefaults__)
        src = Source(Resp(_payload()))
        kw.update(now=self.CLOSE - timedelta(seconds=1), capture_dir=cdir, get=src,
                  schedule_path=schedule_path, repo_dir=ROOT,
                  clock=clock_at(self.CLOSE + timedelta(seconds=1)))
        monkeypatch.setattr(ir.capture, "__kwdefaults__", kw)
        assert ir.main(["capture"]) == 1
        assert "status=window_closed" in out_f.read_text() and "written=false" in out_f.read_text()
        assert "::error::" in capsys.readouterr().out and "❌" in summary.read_text(encoding="utf-8")
        assert src.calls == []
