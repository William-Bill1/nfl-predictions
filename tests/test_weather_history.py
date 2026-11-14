"""Focused tests for scripts/weather_history.py (synthetic responses, temporary
schedules and venue files; no network)."""
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import football_repository as fr  # noqa: E402
import weather_history as wh  # noqa: E402

UTC = timezone.utc
NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
COLS = ("game_id", "season", "week", "gameday", "gametime", "away_team", "away_score", "home_team",
        "home_score", "location", "roof", "stadium_id", "stadium")
GAMES = [
    ("2026_08_MIN_GB", "2026", "8", "2026-10-25", "13:00", "MIN", "", "GB", "", "Home", "outdoors",
     "GNB00", "Lambeau Field"),
    ("2026_09_BUF_NYJ", "2026", "9", "2026-11-01", "13:00", "BUF", "", "NYJ", "", "Home", "outdoors",
     "NYC01", "MetLife Stadium"),
    # GB "home" at a neutral site: must not borrow Lambeau's coordinates.
    ("2026_09_DET_GB", "2026", "9", "2026-11-01", "09:30", "DET", "", "GB", "", "Neutral", "outdoors",
     "LON00", "Tottenham Hotspur Stadium"),
    ("2026_09_LA_ARI", "2026", "9", "2026-11-01", "16:05", "LA", "", "ARI", "", "Home", "",
     "PHO00", "State Farm Stadium"),
    ("2026_09_SEA_LV", "2026", "9", "2026-11-01", "16:05", "SEA", "", "LV", "", "Home", "dome",
     "VEG00", "Allegiant Stadium"),
    ("2026_09_X_CLE", "2026", "9", "2026-11-01", "13:00", "HOU", "", "CLE", "", "Home", "outdoors",
     "CLE00", "Huntington Bank Field"),
    ("2026_09_AMBIG", "2026", "9", "2026-11-01", "01:30", "AAA", "", "BBB", "", "Home", "outdoors",
     "GNB00", "Lambeau Field"),
    ("2026_07_DONE", "2026", "7", "2026-10-18", "13:00", "CHI", "17", "GB", "20", "Home", "outdoors",
     "GNB00", "Lambeau Field"),
    ("2026_08_PAST", "2026", "8", "2026-10-19", "20:15", "CHI", "", "GB", "", "Home", "outdoors",
     "GNB00", "Lambeau Field"),
    ("2026_09_ROOFCONF", "2026", "9", "2026-11-01", "13:00", "AAA", "", "NYG", "", "Home", "dome",
     "NYC01", "MetLife Stadium"),
]


def _venue(sid, name, lat, lon, exposure="outdoor"):
    return {"stadium_id": sid, "stadium": name, "latitude": lat, "longitude": lon,
            "field_exposure": exposure, "roof_context": "test", "coordinate_basis": "synthetic",
            "references": [{"url": "https://example.test/v", "accessed_on": "2026-11-14"}],
            "reviewed_on": "2026-11-14"}


VENUES = [_venue("GNB00", "Lambeau Field", 44.5009705, -88.0619681),
          _venue("NYC01", "MetLife Stadium", 40.8135071, -74.0743442),
          _venue("PHO00", "State Farm Stadium", 33.5277, -112.2626, "retractable_roof"),
          _venue("VEG00", "Allegiant Stadium", 36.0909, -115.1833, "fixed_roof"),
          _venue("CLE00", "Cleveland Browns Stadium", 41.5061, -81.6995)]   # stale name


@pytest.fixture
def env(tmp_path):
    sched = tmp_path / "schedule.tsv"
    sched.write_text("\t".join(COLS) + "\n" + "\n".join("\t".join(g) for g in GAMES) + "\n",
                     encoding="utf-8")
    venues = tmp_path / "venues.json"
    venues.write_text(json.dumps({"venues_version": 1, "venues": VENUES}), encoding="utf-8")
    out = tmp_path / "captures"
    return {"schedule": sched, "venues": venues, "out": out, "db": tmp_path / "w.sqlite3", "tmp": tmp_path}


class FakeResp:
    def __init__(self, status, body, headers=None, url="https://api.open-meteo.com/v1/forecast?x"):
        self.status_code, self._body, self.url = status, body, url
        self.headers = headers or {"Date": "Tue, 20 Oct 2026 12:00:01 GMT", "Content-Type": "application/json"}

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def close(self):
        pass


class Getter:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    def __call__(self, url, **kw):
        self.calls.append((url, kw))
        if self.exc:
            raise self.exc
        return self.resp


def hours_for(start, n=8):
    return [(start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(n)]


def body(start=datetime(2026, 10, 25, 14, tzinfo=UTC), n=8, *, temp=None, units=None, drop=(),
         extra=None, raw_override=None):
    times = hours_for(start, n)
    hourly = {"time": times,
              "temperature_2m": temp if temp is not None else [40.0 + i for i in range(n)],
              "wind_speed_10m": [10.0] * n, "wind_gusts_10m": [20.0] * n,
              "precipitation": [0.0] * n, "precipitation_probability": [5] * n}
    u = {"time": "iso8601", "temperature_2m": "°F", "wind_speed_10m": "mp/h", "wind_gusts_10m": "mp/h",
         "precipitation": "mm", "precipitation_probability": "%"}
    u.update(units or {})
    for d in drop:
        hourly.pop(d)
        u.pop(d)
    doc = {"latitude": 44.5, "longitude": -88.06, "generationtime_ms": 0.5, "utc_offset_seconds": 0,
           "timezone": "GMT", "timezone_abbreviation": "GMT", "elevation": 195.0,
           "hourly_units": u, "hourly": hourly}
    doc.update(extra or {})
    return raw_override if raw_override is not None else json.dumps(doc).encode("utf-8")


def clock_at(*times):
    it = iter(times)
    return lambda: next(it)


def do_capture(env, game_id="2026_08_MIN_GB", resp=None, exc=None, at=NOW):
    get = Getter(resp if resp is not None or exc else FakeResp(200, body()), exc)
    out = wh.capture(game_id, venues_path=env["venues"], schedule_path=env["schedule"], out_dir=env["out"],
                     get=get, clock=clock_at(at, at + timedelta(seconds=1), at + timedelta(seconds=2)),
                     repo_dir=env["tmp"])
    return out, get


def imported(env, *paths):
    conn = wh.connect_for_import(env["db"])
    try:
        return wh.import_captures(conn, paths, now=NOW)
    finally:
        conn.close()


def ro(env_or_db):
    return wh.open_read_only(env_or_db["db"] if isinstance(env_or_db, dict) else env_or_db)


# ---------------------------------------------------------------- timestamps


def test_kickoff_utc_across_dst_and_request_window(env):
    # 2026-10-25 13:00 EDT = 17:00Z; 2026-11-01 13:00 EST (after the fall change) = 18:00Z.
    out, get = do_capture(env, "2026_08_MIN_GB")
    doc = json.loads(Path(out["metadata_file"]).read_text(encoding="utf-8"))
    assert doc["game"]["kickoff_utc"] == "2026-10-25T17:00:00.000000Z"
    assert get.calls[0][1]["params"]["start_hour"] == "2026-10-25T14:00"
    assert get.calls[0][1]["params"]["end_hour"] == "2026-10-25T21:00"
    out2, get2 = do_capture(env, "2026_09_BUF_NYJ",
                            resp=FakeResp(200, body(start=datetime(2026, 11, 1, 15, tzinfo=UTC))))
    doc2 = json.loads(Path(out2["metadata_file"]).read_text(encoding="utf-8"))
    assert doc2["game"]["kickoff_utc"] == "2026-11-01T18:00:00.000000Z"
    assert get2.calls[0][1]["params"]["start_hour"] == "2026-11-01T15:00"
    assert out2["outcome"] == "ok" and out2["hours"] == 8 and doc2["parsed"]["missing_requested_hours"] == []


def test_three_times_kept_separate(env):
    out, _ = do_capture(env, at=datetime(2026, 10, 20, 12, 0, 0, 250000, tzinfo=UTC))
    imported(env, out["metadata_file"])
    conn = ro(env)
    try:
        c = wh._rows(conn, "SELECT * FROM weather_captures")[0]
        hrs = wh._hours(conn, c["capture_import_id"])
    finally:
        conn.close()
    assert c["request_started_at_utc"] == "2026-10-20T12:00:01.250000Z"
    assert c["received_at_utc"] == "2026-10-20T12:00:02.250000Z"
    assert c["provider_issue_time_utc"] is None
    assert c["provider_generationtime_ms"] == 0.5          # duration, not an issue time
    assert "generationtime_ms" in c["provider_issue_time_basis"]
    assert [h["valid_time_utc"] for h in hrs][0] == "2026-10-25T14:00:00Z"
    assert hrs[0]["valid_time_raw"] == "2026-10-25T14:00"


def test_non_utc_response_times_are_refused(env):
    out, _ = do_capture(env, resp=FakeResp(200, body(extra={"utc_offset_seconds": -18000,
                                                             "timezone": "America/Chicago"})))
    assert out["outcome"] == "malformed_response" and out["hours"] == 0


# ---------------------------------------------------------------- identity


def test_exact_game_and_reviewed_coordinates(env):
    _, get = do_capture(env, "2026_09_BUF_NYJ",
                        resp=FakeResp(200, body(start=datetime(2026, 11, 1, 15, tzinfo=UTC))))
    params = get.calls[0][1]["params"]
    assert (params["latitude"], params["longitude"]) == (40.8135071, -74.0743442)
    assert params["temperature_unit"] == "fahrenheit" and params["wind_speed_unit"] == "mph"
    assert params["precipitation_unit"] == "mm" and params["timezone"] == "GMT"


@pytest.mark.parametrize("game_id, outcome", [
    ("2026_09_DET_GB", "venue_unresolved"),          # neutral site; never joined by home team
    ("2026_09_X_CLE", "venue_unresolved"),           # reviewed name differs from the schedule name
    ("2026_09_LA_ARI", "outside_pilot_coverage"),    # retractable: roof state not assumed
    ("2026_09_SEA_LV", "outside_pilot_coverage"),    # fixed roof
    ("2026_09_ROOFCONF", "venue_conflict"),          # record says outdoor, schedule says dome
    ("2026_99_NOPE", "game_not_found"),
    ("2026_09_AMBIG", "kickoff_unusable"),           # 01:30 on the fall-back date is ambiguous
    ("2026_07_DONE", "game_completed"),
    ("2026_08_PAST", "game_started"),
])
def test_refusals_make_no_request_and_write_nothing(env, game_id, outcome):
    out, get = do_capture(env, game_id)
    assert out["outcome"] == outcome and out["requested"] is False
    assert get.calls == []
    assert not env["out"].exists() or list(env["out"].iterdir()) == []


def test_duplicate_game_rows_are_ambiguous(env):
    with open(env["schedule"], "a", encoding="utf-8") as fh:
        fh.write("\t".join(GAMES[0]) + "\n")
    out, get = do_capture(env)
    assert out["outcome"] == "game_ambiguous" and get.calls == []


@pytest.mark.parametrize("mutate", [
    lambda v: v.pop("references"),
    lambda v: v.update(latitude=None),
    lambda v: v.update(field_exposure="probably_open"),
    lambda v: v.update(reviewed_on="soon"),
])
def test_invalid_venue_records_are_refused(env, mutate):
    vs = json.loads(json.dumps(VENUES))
    mutate(vs[0])
    env["venues"].write_text(json.dumps({"venues_version": 1, "venues": vs}), encoding="utf-8")
    with pytest.raises(wh.VenuesRefused):
        do_capture(env)


# ---------------------------------------------------------------- units, alignment, missing data


def test_units_raw_response_and_hours_are_preserved(env):
    raw = body()
    out, _ = do_capture(env, resp=FakeResp(200, raw))
    assert Path(out["raw_file"]).read_bytes() == raw
    imported(env, out["metadata_file"])
    conn = ro(env)
    try:
        c = wh._rows(conn, "SELECT * FROM weather_captures")[0]
        hrs = wh._hours(conn, c["capture_import_id"])
    finally:
        conn.close()
    assert c["raw_response"] == raw and c["raw_response_sha256"] == hashlib.sha256(raw).hexdigest()
    assert json.loads(c["hourly_units_json"]) == {"temperature_2m": "°F", "wind_speed_10m": "mp/h",
                                                  "wind_gusts_10m": "mp/h", "precipitation": "mm",
                                                  "precipitation_probability": "%"}
    assert (c["requested_latitude"], c["returned_latitude"]) == (44.5009705, 44.5)
    assert json.loads(c["request_params_json"])["hourly"] == ",".join(wh.VARIABLES)
    assert [h["temperature_2m"] for h in hrs] == [40.0 + i for i in range(8)]
    assert c["schedule_sha256"] == hashlib.sha256(env["schedule"].read_bytes()).hexdigest()
    assert json.loads(c["venue_json"])["stadium_id"] == "GNB00"


def test_missing_values_stay_null_and_missing_variable_is_explicit(env):
    temp = [40.0, None, 42.0, None, 44.0, 45.0, 46.0, 47.0]
    out, _ = do_capture(env, resp=FakeResp(200, body(temp=temp[:7], drop=("wind_gusts_10m",), n=7)))
    assert out["outcome"] == "ok_missing_variables" and out["usable"]
    doc = json.loads(Path(out["metadata_file"]).read_text(encoding="utf-8"))
    assert doc["parsed"]["missing_variables"] == ["wind_gusts_10m"]
    assert doc["parsed"]["missing_requested_hours"] == ["2026-10-25T21:00:00Z"]
    imported(env, out["metadata_file"])
    conn = ro(env)
    try:
        rows = conn.execute("SELECT temperature_2m, wind_gusts_10m FROM weather_forecast_hours "
                            "ORDER BY valid_time_utc").fetchall()
    finally:
        conn.close()
    assert [r[0] for r in rows] == temp[:7] and all(r[1] is None for r in rows)


def _bad(kind):
    good = body()
    return {
        "misaligned": FakeResp(200, body(temp=[1.0, 2.0])),
        "non_numeric": FakeResp(200, body(temp=["cold"] * 8)),
        "wrong_units": FakeResp(200, body(units={"temperature_2m": "°C"})),
        "truncated": FakeResp(200, good[: len(good) // 2]),
        "http_400": FakeResp(400, b'{"error":true,"reason":"Parameter start_hour out of range"}'),
        "too_large": FakeResp(200, b" " * (wh.MAX_RESPONSE_BYTES + 10)),
    }[kind]


@pytest.mark.parametrize("kind, outcome", [
    ("misaligned", "malformed_response"), ("non_numeric", "malformed_response"),
    ("wrong_units", "units_unexpected"), ("truncated", "malformed_response"),
    ("http_400", "http_error"), ("too_large", "response_too_large")])
def test_failed_responses_are_kept_but_contribute_no_rows(env, kind, outcome):
    out, _ = do_capture(env, resp=_bad(kind))
    assert out["outcome"] == outcome and out["usable"] is False and out["requested"]
    if kind == "http_400":
        assert "start_hour out of range" in out["detail"]
    imported(env, out["metadata_file"])
    conn = ro(env)
    try:
        c = wh._rows(conn, "SELECT outcome, usable FROM weather_captures")[0]
        n = conn.execute("SELECT COUNT(*) FROM weather_forecast_hours").fetchone()[0]
        res = wh.as_of(conn, "2026_08_MIN_GB", NOW + timedelta(days=1), now=NOW)
    finally:
        conn.close()
    assert c == {"outcome": outcome, "usable": 0} and n == 0
    assert res["capture"] is None and res["unusable_received_by_cutoff"] == [out["capture_id"]]


def test_transport_failure_is_recorded_without_raw_file(env):
    out, _ = do_capture(env, exc=requests.ConnectionError("connection refused"))
    assert out["outcome"] == "transport_error" and out["raw_file"] is None
    assert imported(env, out["metadata_file"])[0]["hours"] == 0


# ---------------------------------------------------------------- imports


def test_reimport_is_a_noop(env):
    out, _ = do_capture(env)
    first = imported(env, out["metadata_file"])
    again = imported(env, out["metadata_file"], out["metadata_file"])
    assert first[0]["created"] and [r["created"] for r in again] == [False, False]
    conn = ro(env)
    try:
        assert conn.execute("SELECT COUNT(*) FROM weather_captures").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM weather_forecast_hours").fetchone()[0] == 8
    finally:
        conn.close()


def test_identical_payload_at_different_times_are_separate_observations(env):
    a, _ = do_capture(env, at=NOW)
    b, _ = do_capture(env, at=NOW + timedelta(days=3))
    assert a["capture_id"] != b["capture_id"]
    imported(env, a["metadata_file"], b["metadata_file"])
    conn = ro(env)
    try:
        assert conn.execute("SELECT COUNT(*) FROM weather_captures").fetchone()[0] == 2
        cmp = wh.compare(conn, a["capture_id"], b["capture_id"], now=NOW + timedelta(days=4))
    finally:
        conn.close()
    assert cmp["counts"]["values_changed"] == 0 and cmp["counts"]["values_unchanged"] == 40


def test_batch_rollback_on_invalid_second_file(env):
    a, _ = do_capture(env, at=NOW)
    b, _ = do_capture(env, at=NOW + timedelta(hours=1))
    raw_b = Path(b["raw_file"])
    raw_b.write_bytes(raw_b.read_bytes().replace(b"40.0", b"99.0"))
    with pytest.raises(wh.ImportRefused, match="hash"):
        imported(env, a["metadata_file"], b["metadata_file"])
    meta_b = Path(b["metadata_file"])
    doc = json.loads(meta_b.read_text(encoding="utf-8"))
    doc["received_at_utc"] = "2026-10-19T00:00:00.000000Z"          # edited after capture
    meta_b.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(wh.ImportRefused, match="mismatch"):
        imported(env, a["metadata_file"], b["metadata_file"])
    conn = ro(env)
    try:
        assert conn.execute("SELECT COUNT(*) FROM weather_captures").fetchone()[0] == 0
    finally:
        conn.close()


def test_rollback_when_an_insert_fails_midway(env, monkeypatch):
    a, _ = do_capture(env, at=NOW)
    b, _ = do_capture(env, at=NOW + timedelta(hours=1))
    real, calls = wh._insert, []

    def flaky(*args, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise sqlite3.OperationalError("disk I/O error (simulated)")
        return real(*args, **kw)
    monkeypatch.setattr(wh, "_insert", flaky)
    with pytest.raises(sqlite3.OperationalError):
        imported(env, a["metadata_file"], b["metadata_file"])
    conn = ro(env)
    try:
        assert conn.execute("SELECT COUNT(*) FROM weather_captures").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM weather_forecast_hours").fetchone()[0] == 0
    finally:
        conn.close()


# ---------------------------------------------------------------- comparison and as-of


def _pair(env, later_body, later_at=NOW + timedelta(days=3)):
    a, _ = do_capture(env, at=NOW)
    b, _ = do_capture(env, resp=FakeResp(200, later_body), at=later_at)
    imported(env, a["metadata_file"], b["metadata_file"])
    return a, b


def test_compare_matching_valid_hours_only(env):
    later = body(start=datetime(2026, 10, 25, 15, tzinfo=UTC), temp=[50.0] * 8)   # shifted one hour
    a, b = _pair(env, later)
    conn = ro(env)
    try:
        cmp = wh.compare(conn, a["capture_id"], b["capture_id"], now=NOW)
    finally:
        conn.close()
    assert cmp["counts"]["hours_matched"] == 7
    assert cmp["counts"]["hours_earlier_only"] == 1 and cmp["counts"]["hours_later_only"] == 1
    by_t = {h["valid_time_utc"]: h for h in cmp["hours"]}
    h17 = by_t["2026-10-25T17:00:00Z"]
    assert h17["minutes_from_kickoff"] == 0
    assert h17["temperature_2m"] == {"earlier": 43.0, "later": 50.0, "state": "changed", "difference": 7.0}
    assert by_t["2026-10-25T14:00:00Z"] == {"valid_time_utc": "2026-10-25T14:00:00Z",
                                            "minutes_from_kickoff": -180, "presence": "earlier_only"}
    assert "kickoff" not in json.dumps(cmp["counts"])          # no interpolated kickoff value
    assert cmp["earlier"]["provider_issue_time_utc"] is None


def test_compare_units_mismatch_and_missing_variable_not_compared(env):
    later = body(units={"wind_speed_10m": "mph"}, drop=("precipitation",))
    a, b = _pair(env, later)
    conn = ro(env)
    try:
        cmp = wh.compare(conn, a["capture_id"], b["capture_id"], now=NOW)
    finally:
        conn.close()
    assert cmp["variables"]["wind_speed_10m"] == {"status": "units_mismatch", "earlier_unit": "mp/h",
                                                  "later_unit": "mph"}
    assert cmp["variables"]["precipitation"] == {"status": "variable_missing", "missing_in": ["later"]}
    assert all("wind_speed_10m" not in h and "precipitation" not in h for h in cmp["hours"])


def test_compare_missing_value_is_not_a_change(env):
    later = body(temp=[None] + [40.0 + i for i in range(1, 8)])
    a, b = _pair(env, later)
    conn = ro(env)
    try:
        cmp = wh.compare(conn, a["capture_id"], b["capture_id"], now=NOW)
    finally:
        conn.close()
    first = cmp["hours"][0]["temperature_2m"]
    assert first == {"earlier": 40.0, "later": None, "state": "missing_value_later"}
    assert cmp["counts"]["values_missing"] == 1 and cmp["counts"]["values_changed"] == 0


def test_compare_refusals(env):
    a, b = _pair(env, body())
    c, _ = do_capture(env, "2026_09_BUF_NYJ", resp=FakeResp(200, body(start=datetime(2026, 11, 1, 15, tzinfo=UTC))),
                      at=NOW + timedelta(days=1))
    f, _ = do_capture(env, resp=FakeResp(500, b"oops"), at=NOW + timedelta(days=4))
    imported(env, c["metadata_file"], f["metadata_file"])
    conn = ro(env)
    try:
        for args, kw, msg in [
            ((a["capture_id"], a["capture_id"]), {}, "same capture"),
            ((b["capture_id"], a["capture_id"]), {}, "not received before"),
            ((a["capture_id"], c["capture_id"]), {}, "different games"),
            ((b["capture_id"], f["capture_id"]), {}, "not usable"),
            ((a["capture_id"], b["capture_id"]), {"as_of_cutoff": NOW + timedelta(days=1)}, "after the cutoff"),
        ]:
            with pytest.raises(wh.CompareRefused, match=msg):
                wh.compare(conn, *args, **kw)
    finally:
        conn.close()


def test_as_of_excludes_captures_received_after_cutoff(env):
    a, b = _pair(env, body(temp=[60.0] * 8))
    conn = ro(env)
    try:
        received_a = wh.parse_ts(wh._rows(conn, "SELECT received_at_utc FROM weather_captures "
                                                "WHERE capture_id = ?", (a["capture_id"],))[0]["received_at_utc"])
        exact = wh.as_of(conn, "2026_08_MIN_GB", received_a, now=NOW)
        before = wh.as_of(conn, "2026_08_MIN_GB", received_a - timedelta(microseconds=1), now=NOW)
        between = wh.as_of(conn, "2026_08_MIN_GB", NOW + timedelta(days=1), now=NOW)
        after = wh.as_of(conn, "2026_08_MIN_GB", NOW + timedelta(days=5), now=NOW)
        listed = wh.list_captures(conn, as_of=NOW + timedelta(days=1), now=NOW + timedelta(days=1))
    finally:
        conn.close()
    assert exact["capture"]["capture_id"] == a["capture_id"]
    assert before["capture"] is None and before["excluded_received_after_cutoff"] == 2
    assert between["capture"]["capture_id"] == a["capture_id"] and between["excluded_received_after_cutoff"] == 1
    assert between["hours"][0]["temperature_2m"] == 40.0
    assert after["capture"]["capture_id"] == b["capture_id"] and after["hours"][0]["temperature_2m"] == 60.0
    assert [c["capture_id"] for c in listed["captures"]] == [a["capture_id"]]
    assert listed["captures"][0]["age_since_capture_hours"] == 24.0


# ---------------------------------------------------------------- read-only SQL and database files


def test_read_only_sql_refuses_writes_and_attach(env, capsys):
    out, _ = do_capture(env)
    imported(env, out["metadata_file"])
    before = hashlib.sha256(env["db"].read_bytes()).hexdigest()
    cols, rows = fr.run_sql(env["db"], "SELECT COUNT(*) AS n FROM weather_forecast_hours")
    assert cols == ["n"] and rows == [(8,)]
    for stmt in ("INSERT INTO weather_captures (capture_id) VALUES ('x')",
                 "DELETE FROM weather_forecast_hours", "DROP TABLE weather_captures",
                 f"ATTACH DATABASE '{env['tmp'] / 'other.db'}' AS o", "PRAGMA journal_mode = DELETE",
                 "UPDATE weather_forecast_hours SET temperature_2m = 0"):
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            fr.run_sql(env["db"], stmt)
        assert wh.main(["sql", "--db", str(env["db"]), stmt]) == 2
    assert wh.main(["sql", "--db", str(env["db"]), "SELECT 1"]) == 0
    assert "FORECASTS_NOT_OBSERVATIONS" in capsys.readouterr().err
    assert hashlib.sha256(env["db"].read_bytes()).hexdigest() == before
    assert not (env["tmp"] / "other.db").exists()


def test_foreign_database_is_refused_unchanged(env):
    conn = sqlite3.connect(env["db"])
    conn.execute("CREATE TABLE something_else (x)")
    conn.commit()
    conn.close()
    before = env["db"].read_bytes()
    out, _ = do_capture(env)
    with pytest.raises(fr.SchemaRefused):
        imported(env, out["metadata_file"])
    assert wh.main(["list", "--db", str(env["db"])]) == 2
    assert env["db"].read_bytes() == before


def test_cli_import_list_asof_compare(env, capsys):
    a, b = _pair(env, body(temp=[30.0] * 8))
    db = str(env["db"])
    assert wh.main(["import", "--db", db, a["metadata_file"]]) == 0
    assert json.loads(capsys.readouterr().out)[0]["created"] is False
    assert wh.main(["list", "--db", db, "--as-of", "2026-10-21T00:00:00Z"]) == 0
    assert len(json.loads(capsys.readouterr().out)["captures"]) == 1
    assert wh.main(["asof", "--db", db, "--game-id", "2026_08_MIN_GB", "--cutoff", "2026-10-30T00:00:00Z"]) == 0
    assert json.loads(capsys.readouterr().out)["capture"]["capture_id"] == b["capture_id"]
    assert wh.main(["compare", "--db", db, "--earlier", a["capture_id"], "--later", b["capture_id"]]) == 0
    assert json.loads(capsys.readouterr().out)["counts"]["values_changed"] == 8
    assert wh.main(["compare", "--db", db, "--earlier", b["capture_id"], "--later", a["capture_id"]]) == 2


# ---------------------------------------------------------------- attempt identity


def _same_second(env, make_get, offsets=(100000, 200000)):
    """Two attempts received within one second; returns outputs and the first
    attempt's file hashes taken before the second attempt."""
    outs, first = [], None
    for us in offsets:
        at = NOW + timedelta(microseconds=us)
        outs.append(wh.capture("2026_08_MIN_GB", venues_path=env["venues"], schedule_path=env["schedule"],
                               out_dir=env["out"], get=make_get(), repo_dir=env["tmp"],
                               clock=clock_at(at, at, at + timedelta(microseconds=1))))
        if first is None:
            first = {p.name: p.read_bytes() for p in env["out"].iterdir()}
    return outs, first


def _check_both_kept(env, outs, first, files_per_attempt):
    assert outs[0]["capture_id"] != outs[1]["capture_id"]
    assert len(list(env["out"].iterdir())) == 2 * files_per_attempt
    for name, data in first.items():                       # first attempt's bytes untouched
        assert (env["out"] / name).read_bytes() == data
    res = imported(env, *[o["metadata_file"] for o in outs])
    assert [r["created"] for r in res] == [True, True]
    conn = ro(env)
    try:
        assert conn.execute("SELECT COUNT(*) FROM weather_captures").fetchone()[0] == 2
    finally:
        conn.close()


def test_identical_successful_responses_within_one_second_are_two_attempts(env):
    raw = body()
    outs, first = _same_second(env, lambda: Getter(FakeResp(200, raw)))
    assert all(o["outcome"] == "ok" for o in outs)
    _check_both_kept(env, outs, first, files_per_attempt=2)
    assert all(Path(o["raw_file"]).read_bytes() == raw for o in outs)


def test_two_transport_failures_within_one_second_are_two_attempts(env):
    outs, first = _same_second(env, lambda: Getter(exc=requests.exceptions.SSLError("synthetic cert failure")),
                               offsets=(0, 0))                  # even the same microsecond
    assert all(o["outcome"] == "transport_error" and o["raw_file"] is None for o in outs)
    _check_both_kept(env, outs, first, files_per_attempt=1)


def test_existing_name_is_never_reopened_a_new_identity_is_drawn(env, monkeypatch):
    tokens = iter(["aaaaaaaaaaaa", "aaaaaaaaaaaa", "bbbbbbbbbbbb"])
    monkeypatch.setattr(wh, "_attempt_token", lambda: next(tokens))
    outs, first = _same_second(env, lambda: Getter(FakeResp(200, body())), offsets=(0, 0))
    assert outs[0]["capture_id"].endswith("-aaaaaaaaaaaa") and outs[1]["capture_id"].endswith("-bbbbbbbbbbbb")
    _check_both_kept(env, outs, first, files_per_attempt=2)


def test_identity_exhaustion_raises_and_keeps_first_attempt(env, monkeypatch):
    monkeypatch.setattr(wh, "_attempt_token", lambda: "cccccccccccc")
    at = NOW
    first = do_capture(env, at=at)[0]
    before = {p.name: p.read_bytes() for p in env["out"].iterdir()}
    with pytest.raises(FileExistsError, match="no unused capture_id"):
        do_capture(env, at=at)
    assert {p.name: p.read_bytes() for p in env["out"].iterdir()} == before
    assert first["capture_id"].endswith("-cccccccccccc")
