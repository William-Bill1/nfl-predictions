"""Focused tests for the read-only "Stored game context" panel
(stored_context.py, stored_context_panel.py and its wiring in predictions.py).

Synthetic injury/weather research databases in temporary folders; isolated
AppTest fixtures; the full home page reuses test_home_downloads' fixture with
.env loading disabled, the startup log writes stubbed and the network blocked.
"""
import hashlib
import json
import socket
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import injury_history as ih  # noqa: E402
import stored_context as sc  # noqa: E402
import weather_history as wh  # noqa: E402
from test_weather_history import FakeResp, Getter, body, _venue  # noqa: E402

UTC = timezone.utc
G = "2026_10_BUF_KC"              # this week's game
OLD = "2026_03_BUF_KC"            # an earlier meeting of the same teams
KICKOFF = datetime(2026, 11, 15, 18, 0, tzinfo=UTC)      # 13:00 ET
NOW = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)

GAMES = pd.DataFrame([
    {"game_id": G, "gameday": "2026-11-15", "gametime": "13:00", "home_team": "KC", "away_team": "BUF",
     "stadium_id": "KAN00", "stadium": "Fixture Arrowhead", "prob_underdogCovered": 0.61, "spread_line": -1.5},
    {"game_id": OLD, "gameday": "2026-09-20", "gametime": "13:00", "home_team": "KC", "away_team": "BUF",
     "stadium_id": "KAN00", "stadium": "Fixture Arrowhead", "prob_underdogCovered": 0.55, "spread_line": 3.0},
])


def us(dt):
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def tree(d):
    return {p.relative_to(d).as_posix(): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(Path(d).rglob("*")) if p.is_file()}


# ---------------------------------------------------------------- synthetic injury database


def player(name, pid, status, team, injury=None, entry_date="2026-11-10T12:00Z"):
    return {"player_name": name, "player_id": pid, "player_id_source": "athlete.id" if pid else None,
            "source_status": status, "team": team, "injury_type": injury, "source_entry_date_raw": entry_date}


def add_capture(conn, cid, received, *, players=(), games=((G, "KC", "BUF", KICKOFF, "pregame"),),
                usable=True, outcome="full_team_coverage"):
    """Insert one synthetic capture through a WRITABLE test-only connection."""
    rec = us(received)
    imp = conn.execute(
        "INSERT INTO injury_captures (capture_id, payload_sha256, transformation_version, file_name, source_path, "
        "file_sha256, file_bytes, imported_at_utc, request_started_at_utc, received_at_utc, slot_name, slot_id, "
        "slot_status, slot_fills, outcome, usable, source_name, source_json, teams_in_feed, player_rows, "
        "malformed_entries_json, unmatched_source_teams_json, source_entry_dates_json) "
        "VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, 'wednesday_noon', ?, 'on_time', 1, ?, ?, 'espn', '{}', 2, ?, "
        "'[]', '[]', '{}')",
        (cid, "p" + cid, ih.TRANSFORMATION_VERSION, cid + ".json", cid, "f" + cid, rec, rec, rec,
         f"{received:%Y-%m-%d}_slot", outcome, int(usable), len(players))).lastrowid
    for g, home, away, ko, state in games:
        conn.execute("INSERT INTO injury_capture_games (capture_import_id, game_id, season, week, home_team, "
                     "away_team, kickoff_utc, state) VALUES (?, ?, 2026, 10, ?, ?, ?, ?)",
                     (imp, g, home, away, ko.strftime("%Y-%m-%dT%H:%M:%SZ"), state))
    if usable:
        for i, team in enumerate(("KC", "BUF")):
            n = sum(1 for p in players if p["team"] == team)
            conn.execute("INSERT INTO injury_capture_feed_teams VALUES (?, ?, ?, ?, ?, ?)",
                         (imp, i, team, str(i), team, n))
            conn.execute("INSERT INTO injury_capture_teams VALUES (?, ?, 'rows_listed', ?)", (imp, team, n))
    for i, p in enumerate(players):
        conn.execute("INSERT INTO injury_observations (capture_import_id, entry_index, team, player_name, player_id, "
                     "player_id_source, source_status, status_class, injury_type, source_entry_date_raw, "
                     "raw_row_json) VALUES (?, ?, ?, ?, ?, ?, ?, 'game_designation_word', ?, ?, '{}')",
                     (imp, i, p["team"], p["player_name"], p["player_id"], p["player_id_source"],
                      p["source_status"], p["injury_type"], p["source_entry_date_raw"]))


WED = [player("Patrick M", "1", "Questionable", "KC", "Ankle"), player("Travis K", "2", "Out", "KC", "Knee"),
       player("No Id Guy", None, "Questionable", "KC"), player("Dup Five", "5", "Active", "KC"),
       player("Josh A", "10", "Active", "BUF")]
SAT = [player("Patrick M", "1", "Out", "KC", "Ankle"), player("No Id Guy", None, "Questionable", "KC"),
       player("Dup Five", "5", "Active", "KC"), player("Dup Five B", "5", "Active", "KC"),
       player("Josh A", "10", "Active", "BUF")]


def injury_db(path, extra=()):
    conn = ih.connect_for_import(path)
    try:
        conn.execute("BEGIN")
        add_capture(conn, "old-week3", datetime(2026, 9, 17, 16, tzinfo=UTC),
                    players=[player("Old Week Three", "99", "Out", "KC")],
                    games=((OLD, "KC", "BUF", datetime(2026, 9, 20, 17, tzinfo=UTC), "pregame"),))
        add_capture(conn, "wed", datetime(2026, 11, 11, 17, tzinfo=UTC), players=WED)
        add_capture(conn, "sat", datetime(2026, 11, 14, 10, tzinfo=UTC), players=SAT)
        for args, kw in extra:
            add_capture(conn, *args, **kw)
        conn.execute("COMMIT")
    finally:
        conn.close()
    return path


# ---------------------------------------------------------------- synthetic weather database


SCHED_COLS = ("game_id", "season", "week", "gameday", "gametime", "away_team", "away_score", "home_team",
              "home_score", "location", "roof", "stadium_id", "stadium")


def weather_setup(tmp, rows=((G, "2026", "10", "2026-11-15", "13:00", "BUF", "", "KC", "", "Home", "outdoors",
                              "KAN00", "Fixture Arrowhead"),)):
    """Temporary schedule plus a reviewed venue record for every row's exact venue."""
    sched = tmp / "schedule.tsv"
    sched.write_text("\t".join(SCHED_COLS) + "\n" + "\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")
    venues = tmp / "venues.json"
    records = {r[11]: _venue(r[11], r[12], 39.0489, -94.4839) for r in rows}
    venues.write_text(json.dumps({"venues_version": 1, "venues": list(records.values())}), encoding="utf-8")
    return sched, venues


def weather_capture(tmp, at, resp, game_id=G, rows=None):
    sched, venues = weather_setup(tmp, rows) if rows else weather_setup(tmp)
    ticks = iter([at, at, at + timedelta(seconds=1)])
    out = wh.capture(game_id, venues_path=venues, schedule_path=sched, out_dir=tmp / "wcaps",
                     get=Getter(resp), clock=lambda: next(ticks), repo_dir=tmp)
    return out


def weather_import(db, *outs):
    conn = wh.connect_for_import(db)
    try:
        wh.import_captures(conn, [o["metadata_file"] for o in outs], now=NOW)
    finally:
        conn.close()


START = KICKOFF - timedelta(hours=3)


def ok_body(**kw):
    return FakeResp(200, body(start=START, **kw))


def config(tmp, **kw):
    p = tmp / "context.json"
    p.write_text(json.dumps({"stored_context_version": 1, **{k: str(v) for k, v in kw.items()}}), encoding="utf-8")
    return {sc.CONFIG_ENV: str(p)}


@pytest.fixture
def stores(tmp_path):
    """Context databases outside the capture folders, plus their config."""
    d = tmp_path / "stores"
    d.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    injury_db(d / "injury.sqlite3")
    w1 = weather_capture(work, datetime(2026, 11, 13, 12, tzinfo=UTC),
                         ok_body(temp=[40.0, None, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0], drop=("precipitation",)))
    w2 = weather_capture(work, datetime(2026, 11, 14, 11, tzinfo=UTC), FakeResp(500, b"oops"))
    w3 = weather_capture(work, datetime(2026, 11, 14, 13, tzinfo=UTC), ok_body(temp=[99.0] * 8))  # after NOW
    weather_import(d / "weather.sqlite3", w1, w2, w3)
    env = config(d, injury_db=d / "injury.sqlite3", weather_db=d / "weather.sqlite3")
    return {"dir": d, "work": work, "env": env, "w": (w1, w2, w3)}


# ---------------------------------------------------------------- 1. identity


def test_exact_game_identity_and_another_weeks_meeting_is_refused(stores):
    ctx = sc.build(GAMES, G, environ=stores["env"], now=NOW)
    inj = ctx["injuries"]
    assert inj["latest"]["capture_id"] == "sat" and inj["previous"]["capture_id"] == "wed"
    names = {r["player_name"] for t in inj["teams"].values() for r in t["rows"]}
    assert "Old Week Three" not in names                       # same teams, earlier meeting: never borrowed
    old = sc.build(GAMES, OLD, environ=stores["env"], now=datetime(2026, 9, 18, tzinfo=UTC))
    assert old["injuries"]["latest"]["capture_id"] == "old-week3"
    assert old["weather"]["state"] == "none"                   # weather captures are all for G


@pytest.mark.parametrize("games, msg", [
    (pd.concat([GAMES, GAMES.iloc[[0]]]), "appears 2 times"),
    (GAMES.iloc[[1]], "not in the predictions table"),
    (GAMES.assign(gametime=["01:30", "13:00"], gameday=["2026-11-01", "2026-09-20"]), "no usable kickoff"),
    (GAMES.assign(home_team=["", "KC"]), "no home/away team"),
])
def test_missing_or_ambiguous_identity_is_refused(stores, games, msg):
    ctx = sc.build(games, G, environ=stores["env"], now=NOW)
    assert ctx["state"] == "identity_error" and msg in ctx["message"]


def test_capture_whose_game_id_has_other_teams_or_kickoff_is_excluded(tmp_path):
    db = injury_db(tmp_path / "i.sqlite3", extra=[
        (("swapped", datetime(2026, 11, 14, 11, tzinfo=UTC)),
         {"players": [player("Swap", "7", "Out", "KC")], "games": ((G, "BUF", "KC", KICKOFF, "pregame"),)}),
        (("moved", datetime(2026, 11, 14, 11, 30, tzinfo=UTC)),
         {"players": [player("Moved", "8", "Out", "KC")],
          "games": ((G, "KC", "BUF", KICKOFF + timedelta(hours=3), "pregame"),)})])
    inj = sc.build(GAMES, G, environ=config(tmp_path, injury_db=db), now=NOW)["injuries"]
    assert inj["latest"]["capture_id"] == "sat"
    assert any("swapped" in n and "not BUF @ KC" in n for n in inj["notices"])
    assert any("moved" in n and "kicking off" in n for n in inj["notices"])


def test_weather_capture_for_mismatched_game_is_refused_not_shown(tmp_path):
    w = weather_capture(tmp_path, datetime(2026, 11, 13, tzinfo=UTC), ok_body(),
                        rows=((G, "2026", "10", "2026-11-15", "13:00", "NYJ", "", "KC", "", "Home", "outdoors",
                               "KAN00", "Fixture Arrowhead"),))
    weather_import(tmp_path / "w.sqlite3", w)
    wx = sc.build(GAMES, G, environ=config(tmp_path, weather_db=tmp_path / "w.sqlite3"), now=NOW)["weather"]
    assert wx["state"] == "error" and "doesn't match this game" in wx["message"] and wx["hours"] == []


# ---------------------------------------------------------------- venue identity (weather only)


def test_matching_venue_shows_weather(stores):
    ctx = sc.build(GAMES, G, environ=stores["env"], now=NOW)
    assert (ctx["game"]["stadium_id"], ctx["game"]["stadium"]) == ("KAN00", "Fixture Arrowhead")
    assert ctx["weather"]["state"] == "ok" and ctx["weather"]["capture"]["stadium"] == "Fixture Arrowhead"
    assert len(ctx["weather"]["hours"]) == 8


@pytest.mark.parametrize("stadium_id, stadium, msg", [
    ("NYC01", "MetLife Stadium", "is for venue KAN00 (Fixture Arrowhead)"),     # moved venue
    ("NYC01", "Fixture Arrowhead", "is for venue KAN00"),                      # id differs only
    ("KAN00", "Arrowhead Stadium", "is for venue KAN00 (Fixture Arrowhead)"),   # name differs only (no fuzzy match)
    ("", "Fixture Arrowhead", "no complete venue identity"),                   # missing id
    ("KAN00", None, "no complete venue identity"),                             # missing name
])
def test_changed_or_missing_venue_hides_weather_but_keeps_injuries(stores, stadium_id, stadium, msg):
    games = GAMES.copy()
    games.loc[games.game_id == G, "stadium_id"] = stadium_id
    games.loc[games.game_id == G, "stadium"] = stadium
    ctx = sc.build(games, G, environ=stores["env"], now=NOW)
    assert ctx["injuries"]["state"] == "ok" and ctx["injuries"]["latest"]["capture_id"] == "sat"
    wx = ctx["weather"]
    assert wx["state"] == "error" and msg in wx["message"]
    assert wx["hours"] == [] and wx["capture"] is None


def test_missing_venue_columns_hide_weather(stores):
    ctx = sc.build(GAMES.drop(columns=["stadium_id", "stadium"]), G, environ=stores["env"], now=NOW)
    assert ctx["injuries"]["state"] == "ok"
    assert ctx["weather"]["state"] == "error" and "no complete venue identity" in ctx["weather"]["message"]
    assert ctx["weather"]["hours"] == []


def test_panel_shows_venue_notice_without_forecast_table(stores, tmp_path, monkeypatch):
    csv = tmp_path / "games.csv"
    GAMES.assign(stadium=["Somewhere Else", "Fixture Arrowhead"]).to_csv(csv, index=False)
    monkeypatch.setenv(sc.CONFIG_ENV, stores["env"][sc.CONFIG_ENV])
    at = AppTest.from_function(panel_app, args=(str(csv), NOW.isoformat()), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("the forecast is not shown" in e.value for e in at.error)
    frames = [e.value for e in at.dataframe]
    assert not [f for f in frames if "Valid time (UTC)" in f.columns]
    assert any("Reported status" in f.columns for f in frames)          # injuries still shown


# ---------------------------------------------------------------- 2. cutoff


def test_captures_received_after_now_are_excluded(stores):
    ctx = sc.build(GAMES, G, environ=stores["env"], now=NOW)
    assert ctx["cutoff"] == NOW and ctx["cutoff_basis"] == "current time"
    wx = ctx["weather"]
    assert wx["capture"]["capture_id"] == stores["w"][0]["capture_id"]      # not the 13:00 one
    assert wx["excluded_received_after_cutoff"] == 1
    assert all(h["temperature_2m"] != 99.0 for h in wx["hours"])


def test_post_kickoff_captures_excluded_and_entry_dates_are_not_capture_times(tmp_path):
    db = injury_db(tmp_path / "i.sqlite3", extra=[
        (("sun-am", datetime(2026, 11, 15, 14, tzinfo=UTC)),
         {"players": [player("Sunday Morning", "20", "Out", "KC", entry_date="2026-11-01T00:00Z")]}),
        (("post-kick", datetime(2026, 11, 15, 19, tzinfo=UTC)),       # after kickoff, early source date
         {"players": [player("After Kickoff", "21", "Out", "KC", entry_date="2026-11-02T00:00Z")]})])
    ctx = sc.build(GAMES, G, environ=config(tmp_path, injury_db=db), now=datetime(2026, 11, 16, tzinfo=UTC))
    assert ctx["cutoff"] == KICKOFF and ctx["cutoff_basis"] == "kickoff"
    inj = ctx["injuries"]
    assert inj["latest"]["capture_id"] == "sun-am" and inj["excluded_received_after_cutoff"] == 1
    names = {r["player_name"] for t in inj["teams"].values() for r in t["rows"]}
    assert "After Kickoff" not in names


def test_capture_exactly_at_cutoff_is_included(tmp_path):
    db = injury_db(tmp_path / "i.sqlite3")
    inj = sc.build(GAMES, G, environ=config(tmp_path, injury_db=db),
                   now=datetime(2026, 11, 14, 10, tzinfo=UTC))["injuries"]
    assert inj["latest"]["capture_id"] == "sat"
    inj = sc.build(GAMES, G, environ=config(tmp_path, injury_db=db),
                   now=datetime(2026, 11, 14, 10, tzinfo=UTC) - timedelta(microseconds=1))["injuries"]
    assert inj["latest"]["capture_id"] == "wed"


# ---------------------------------------------------------------- 3. configuration and databases


def test_not_configured():
    assert sc.load_config({})["state"] == "not_configured"
    assert sc.build(GAMES, G, environ={}, now=NOW) == {"state": "not_configured", "message": sc.NOT_CONFIGURED}


@pytest.mark.parametrize("text, msg", [
    ("{not json", "not valid JSON"), ("[1]", "JSON object"),
    ('{"stored_context_version": 1, "injury_db": "a", "extra": 1}', "unknown keys"),
    ('{"stored_context_version": 2, "injury_db": "a"}', "stored_context_version"),
    ('{"stored_context_version": 1}', "names no database"),
    ('{"stored_context_version": 1, "injury_db": ""}', "non-empty path"),
])
def test_malformed_configuration_is_reported(tmp_path, text, msg):
    p = tmp_path / "c.json"
    p.write_text(text, encoding="utf-8")
    cfg = sc.load_config({sc.CONFIG_ENV: str(p)})
    assert cfg["state"] == "error" and msg in cfg["message"]


def test_missing_config_file_is_reported(tmp_path):
    cfg = sc.load_config({sc.CONFIG_ENV: str(tmp_path / "nope.json")})
    assert cfg["state"] == "error" and "not found" in cfg["message"]


def test_missing_empty_and_incompatible_databases_are_notices_and_never_created(tmp_path, stores):
    (tmp_path / "empty.sqlite3").write_bytes(b"")
    cases = {"missing": tmp_path / "missing.sqlite3", "empty": tmp_path / "empty.sqlite3",
             "wrong_schema": stores["dir"] / "weather.sqlite3"}      # a weather DB given as injury DB
    for name, db in cases.items():
        before = db.read_bytes() if db.exists() else None
        inj = sc.build(GAMES, G, environ=config(tmp_path, injury_db=db), now=NOW)["injuries"]
        assert inj["state"] == "error" and "Injury database unavailable" in inj["message"], name
        assert (db.read_bytes() if db.exists() else None) == before, name
    assert not (tmp_path / "missing.sqlite3").exists()
    wx = sc.build(GAMES, G, environ=config(tmp_path, weather_db=stores["dir"] / "injury.sqlite3"), now=NOW)["weather"]
    assert wx["state"] == "error" and "Weather database unavailable" in wx["message"]


def test_relative_paths_resolve_against_the_config_folder(stores):
    p = stores["dir"] / "rel.json"
    p.write_text('{"stored_context_version": 1, "injury_db": "injury.sqlite3"}', encoding="utf-8")
    cfg = sc.load_config({sc.CONFIG_ENV: str(p)})
    assert cfg["injury_db"] == stores["dir"] / "injury.sqlite3" and cfg["weather_db"] is None


# ---------------------------------------------------------------- 4. injury identities and absence


def test_comparison_keeps_unresolved_inferred_and_absence_semantics(stores):
    inj = sc.build(GAMES, G, environ=stores["env"], now=NOW)["injuries"]
    kc = inj["comparisons"]["KC"]
    assert kc["counts"] == {"newly_observed": 0, "changed": 1, "unchanged": 1, "absent_from_later": 1,
                            "identity_unresolved": 3}
    assert kc["changed"][0]["player_name"] == "Patrick M" and kc["changed"][0]["identity_basis"] == "source_player_id"
    assert kc["identity"]["matched_by_inferred_name"] == 1           # No Id Guy, labelled inferred
    assert [a["player_name"] for a in kc["absent_from_later"]] == ["Travis K"]
    assert "NOT mean recovered" in kc["absent_from_later_meaning"]
    unresolved = kc["identity_unresolved"]["earlier"] + kc["identity_unresolved"]["later"]
    assert {u["player_id"] for u in unresolved} == {"5"}


def test_newer_failed_injury_capture_is_surfaced(tmp_path):
    db = injury_db(tmp_path / "i.sqlite3", extra=[
        (("failed-sat", datetime(2026, 11, 14, 11, tzinfo=UTC)), {"usable": False, "outcome": "unavailable"}),
        (("failed-noscope", datetime(2026, 11, 14, 11, 30, tzinfo=UTC)),
         {"usable": False, "outcome": "unavailable", "games": ()}),
        (("failed-oldweek", datetime(2026, 11, 14, 11, 40, tzinfo=UTC)),
         {"usable": False, "outcome": "unavailable",
          "games": ((OLD, "KC", "BUF", datetime(2026, 9, 20, 17, tzinfo=UTC), "completed"),)})])
    inj = sc.build(GAMES, G, environ=config(tmp_path, injury_db=db), now=NOW)["injuries"]
    assert inj["latest"]["capture_id"] == "sat"
    assert [(f["capture_id"], f["scope"]) for f in inj["newer_failed"]] == [
        ("failed-sat", "this game"), ("failed-noscope", "unknown (no scope week recorded)")]


# ---------------------------------------------------------------- 5. weather


def test_weather_missing_values_failures_and_labels(stores):
    wx = sc.build(GAMES, G, environ=stores["env"], now=NOW)["weather"]
    cap = wx["capture"]
    assert cap["missing_variables"] == ["precipitation"]
    assert [h["temperature_2m"] for h in wx["hours"]][:2] == [40.0, None]       # null stays null
    assert all(h["precipitation"] is None for h in wx["hours"])
    assert wx["hours"][0]["valid_time_utc"] == "2026-11-15T15:00:00Z"
    assert cap["units"]["temperature_2m"] == "°F" and "precipitation" not in cap["units"]
    assert [f["outcome"] for f in wx["newer_failed"]] == ["http_error"]
    assert cap["age_since_capture_hours"] == 24.0                                 # since capture (+1 s)


def test_no_weather_capture_is_unavailable_not_neutral(tmp_path):
    db = tmp_path / "w.sqlite3"
    wh.connect_for_import(db).close()
    wx = sc.build(GAMES, G, environ=config(tmp_path, weather_db=db), now=NOW)["weather"]
    assert wx["state"] == "none" and wx["hours"] == [] and "not mean indoor or neutral" in wx["message"]


# ---------------------------------------------------------------- panel rendering (isolated AppTest)


def panel_app(games_csv, now_iso):
    from datetime import datetime
    import pandas as pd
    import stored_context_panel
    stored_context_panel.render(pd.read_csv(games_csv, dtype=str), now=datetime.fromisoformat(now_iso))


def run_panel(tmp_path, env, monkeypatch, now=NOW):
    csv = tmp_path / "games.csv"
    GAMES.to_csv(csv, index=False)
    for k in (sc.CONFIG_ENV,):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    at = AppTest.from_function(panel_app, args=(str(csv), now.isoformat()), default_timeout=60)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def all_text(at):
    out = []
    for kind in ("markdown", "caption", "info", "warning", "error"):
        out += [e.value for e in getattr(at, kind)]
    return "\n".join(out)


def test_panel_not_configured(tmp_path, monkeypatch):
    at = run_panel(tmp_path, {}, monkeypatch)
    assert [e.value for e in at.info] == [sc.NOT_CONFIGURED] and not at.selectbox


def test_panel_malformed_config_is_visible(tmp_path, monkeypatch):
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    at = run_panel(tmp_path, {sc.CONFIG_ENV: str(tmp_path / "bad.json")}, monkeypatch)
    assert any("not valid JSON" in e.value for e in at.error)


def test_panel_shows_labels_changes_and_warnings(stores, tmp_path, monkeypatch):
    at = run_panel(tmp_path, stores["env"], monkeypatch)
    text = all_text(at)
    assert at.selectbox[0].value == G
    for needle in ("not the official NFL injury report", "'Active' entries are often news notes",
                   "25 entries per team", "Showing capture sat", "Changes since wed",
                   "This is not a recovery or clearance", "inferred_name_match",
                   "unresolved identity were not compared", "Captured model forecasts", "not observed stadium weather",
                   "Forecast issue time: not available", "h since capture", "Not returned in this capture: precipitation",
                   "A newer forecast capture failed", "Open-Meteo.com", "Cutoff Sat Nov 14, 2026 07:00 AM EST",
                   "1 capture(s) received after the cutoff are excluded"):
        assert needle in text, needle
    frames = [e.value for e in at.dataframe]
    weather = [f for f in frames if "Valid time (UTC)" in f.columns][0]
    assert "Temperature [°F]" in weather.columns and "Precipitation (prev. hour) [not returned]" in weather.columns
    assert pd.isna(weather["Temperature [°F]"].iloc[1])
    injuries = [f for f in frames if "Identity" in f.columns and "Reported status" in f.columns]
    assert any("no source ID" in set(f["Identity"]) for f in injuries)
    at.selectbox[0].select(OLD).run()
    assert not at.exception
    # The earlier meeting: cutoff is its kickoff, and only its own capture is shown.
    text = all_text(at)
    assert "Showing capture old-week3" in text and "Showing capture sat" not in text


def test_newly_imported_capture_appears_on_rerun(tmp_path, monkeypatch):
    d = tmp_path / "s"
    d.mkdir()
    db = d / "w.sqlite3"
    wh.connect_for_import(db).close()
    at = run_panel(tmp_path, config(d, weather_db=db), monkeypatch)
    assert sc.WEATHER_MISSING_NOTE in [e.value for e in at.info]
    w = weather_capture(tmp_path, datetime(2026, 11, 13, 12, tzinfo=UTC), ok_body())
    weather_import(db, w)
    at.run()
    assert not at.exception
    assert w["capture_id"] in all_text(at)


# ---------------------------------------------------------------- 6. coverage and receipt ages

G11 = "2026_11_NYJ_KC"           # a later week, nothing captured
NOVENUE = "2026_10_MIA_DEN"      # no venue identity: weather invalid, injuries merely missing
DUP = "2026_10_SEA_LV"           # two rows for one game_id: identity error for both contexts


def _g(gid, week, day, home, away, sid="KAN00", stadium="Fixture Arrowhead"):
    return {"game_id": gid, "week": week, "gameday": day, "gametime": "13:00", "home_team": home,
            "away_team": away, "stadium_id": sid, "stadium": stadium, "prob_underdogCovered": 0.6, "spread_line": 2.5}


MIXED = pd.DataFrame([_g(G, 10, "2026-11-15", "KC", "BUF"), _g(G11, 11, "2026-11-22", "KC", "NYJ"),
                      _g(NOVENUE, 10, "2026-11-15", "DEN", "MIA", sid="", stadium=""),
                      _g(DUP, 10, "2026-11-15", "LV", "SEA", "VEG00", "Allegiant"),
                      _g(DUP, 10, "2026-11-15", "LV", "SEA", "VEG00", "Allegiant")])
MIXED_IDS = [G, G11, NOVENUE, DUP]


def test_coverage_counts_accepted_context_once_per_game_and_type(stores):
    res = sc.build_all(MIXED, MIXED_IDS + [G], environ=stores["env"], now=NOW)   # G repeated: counted once
    cov = sc.coverage(res, MIXED_IDS + [G])
    assert cov["total"] == 4
    assert cov["injury"] == {"usable": 1, "missing": 2, "error": 1, "not_configured": 0}
    assert cov["weather"] == {"usable": 1, "missing": 1, "error": 2, "not_configured": 0}
    states = {g: (sc.coverage_state(res["contexts"][g], "injuries"), sc.coverage_state(res["contexts"][g], "weather"))
              for g in MIXED_IDS}
    assert states == {G: ("usable", "usable"), G11: ("missing", "missing"),
                      NOVENUE: ("missing", "error"), DUP: ("error", "error")}


def test_coverage_database_error_is_error_not_missing_and_types_are_independent(stores, tmp_path):
    env = config(tmp_path, injury_db=tmp_path / "absent.sqlite3", weather_db=stores["dir"] / "weather.sqlite3")
    cov = sc.coverage(sc.build_all(MIXED, MIXED_IDS, environ=env, now=NOW), MIXED_IDS)
    assert cov["injury"] == {"usable": 0, "missing": 0, "error": 4, "not_configured": 0}
    assert cov["weather"] == {"usable": 1, "missing": 1, "error": 2, "not_configured": 0}
    only_weather = config(tmp_path, weather_db=stores["dir"] / "weather.sqlite3")
    cov = sc.coverage(sc.build_all(MIXED, MIXED_IDS, environ=only_weather, now=NOW), MIXED_IDS)
    assert cov["injury"]["not_configured"] == 3 and cov["injury"]["error"] == 1      # DUP is still an identity error


def test_shared_read_opens_each_database_once_and_matches_single_game_build(stores, monkeypatch):
    opened = {"injury": 0, "weather": 0}
    for mod, key in ((ih, "injury"), (wh, "weather")):
        real = mod.open_read_only
        monkeypatch.setattr(mod, "open_read_only",
                            lambda p, _r=real, _k=key: (opened.__setitem__(_k, opened[_k] + 1), _r(p))[1])
    res = sc.build_all(MIXED, MIXED_IDS, environ=stores["env"], now=NOW)
    assert opened == {"injury": 1, "weather": 1}
    single = sc.build(MIXED, G, environ=stores["env"], now=NOW)
    assert res["contexts"][G]["injuries"] == single["injuries"]
    assert res["contexts"][G]["weather"] == single["weather"]


def run_mixed_panel(tmp_path, env, monkeypatch, games=MIXED, now=NOW):
    csv = tmp_path / "mixed.csv"
    games.to_csv(csv, index=False)
    monkeypatch.delenv(sc.CONFIG_ENV, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    at = AppTest.from_function(panel_app, args=(str(csv), now.isoformat()), default_timeout=60)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def coverage_frame(at):
    return [e.value for e in at.dataframe if "Games in selector" in e.value.columns][0].set_index("Context")


def test_panel_coverage_uses_the_selectors_own_games_and_span(stores, tmp_path, monkeypatch):
    at = run_mixed_panel(tmp_path, stores["env"], monkeypatch)
    n = len(at.selectbox(key="stored_context_game").options)
    assert n == 4
    text = all_text(at)
    assert "Coverage across the 4 game(s) in this selector** (weeks 10, 11; 2026-11-15 to 2026-11-22)" in text
    assert "this week" not in text.lower()
    f = coverage_frame(at)
    assert f.loc["Injury reports", "Usable"] == 1 and f.loc["Weather forecasts", "Usable"] == 1
    assert f.loc["Injury reports", "Missing (nothing usable by cutoff)"] == 2
    assert f.loc["Weather forecasts", "Invalid or error"] == 2
    assert set(f["Games in selector"]) == {4}
    assert "does not mean healthy, indoor or neutral weather" in text


def test_malformed_week_metadata_is_unknown_and_hides_nothing(stores, tmp_path, monkeypatch):
    import stored_context_panel as scp
    for bad in ("unknown", "10.5", None, float("nan"), "", "inf", "0", "23", "-1", "1e400"):
        assert scp._week(bad) is None, bad
    assert [scp._week(v) for v in (10, "10", "10.0", 10.0, " 11 ", 1, 22)] == [10, 10, 10, 10, 11, 1, 22]
    one = {G: ""}
    for bad in ("unknown", "10.5", None):
        frame = MIXED[MIXED["game_id"] == G].assign(week=bad)
        assert scp._span(frame, one) == "week unknown for 1 game(s); 2026-11-15"   # never "week 10"
    assert scp._span(MIXED.drop(columns="week"), dict.fromkeys(MIXED_IDS)) == \
        "week unknown for 4 game(s); 2026-11-15 to 2026-11-22"
    # Mixed valid/unknown in the panel: G malformed, NOVENUE fractional, DUP's rows disagree (one missing).
    bad = MIXED.astype({"week": object}).copy()
    bad.loc[bad["game_id"] == G, "week"] = "unknown"
    bad.loc[bad["game_id"] == NOVENUE, "week"] = 10.5
    bad.loc[bad.index[bad["game_id"] == DUP][0], "week"] = None
    good = run_mixed_panel(tmp_path, stores["env"], monkeypatch)
    at = run_mixed_panel(tmp_path, stores["env"], monkeypatch, games=bad)
    text = all_text(at)
    assert "Coverage across the 4 game(s) in this selector** (week 11; week unknown for 3 game(s); " \
           "2026-11-15 to 2026-11-22)" in text
    assert "week 10" not in text
    assert at.selectbox(key="stored_context_game").options == good.selectbox(key="stored_context_game").options
    assert coverage_frame(at).equals(coverage_frame(good))                    # same counts, same denominator
    assert "Injury report: received Sat Nov 14, 2026 05:00 AM EST" in text    # G's context still shown
    assert "; 24.0 h since receipt" in text


def test_panel_receipt_ages_and_missing_or_error_states(stores, tmp_path, monkeypatch):
    at = run_mixed_panel(tmp_path, stores["env"], monkeypatch)
    text = all_text(at)
    # sat injury capture received 10:00, forecast received 12:00 the day before; NOW is 12:00.
    assert "Injury report: received Sat Nov 14, 2026 05:00 AM EST (2026-11-14 10:00:00 UTC); 2.0 h since receipt" in text
    assert "Weather forecast: received Fri Nov 13, 2026 07:00 AM EST" in text and "; 24.0 h since receipt" in text
    assert sc.AGE_NOTE in text and sc.AVAILABILITY_NOTE in text
    assert sc.INJURY_SOURCE_NOTE in text and sc.ATTRIBUTION in text           # existing notices kept
    at.selectbox(key="stored_context_game").select(G11).run()
    text = all_text(at)
    assert "Injury report: missing: no usable capture was received by the cutoff" in text
    assert "Weather forecast: missing: no usable capture was received by the cutoff" in text
    assert sc.WEATHER_MISSING_NOTE in text
    at.selectbox(key="stored_context_game").select(NOVENUE).run()
    text = all_text(at)
    assert "Injury report: missing" in text
    assert "Weather forecast: invalid or error: The selected game has no complete venue identity" in text
    at.selectbox(key="stored_context_game").select(DUP).run()
    assert any("appears 2 times" in e.value for e in at.error)
    assert coverage_frame(at).loc["Injury reports", "Usable"] == 1             # summary unchanged by selection


def test_panel_coverage_reflects_new_import_on_next_rerun(tmp_path, monkeypatch):
    d = tmp_path / "s"
    d.mkdir()
    db = d / "w.sqlite3"
    wh.connect_for_import(db).close()
    at = run_mixed_panel(tmp_path, config(d, weather_db=db), monkeypatch)
    assert coverage_frame(at).loc["Weather forecasts", "Usable"] == 0
    assert "Weather forecast: missing" in all_text(at)
    weather_import(db, weather_capture(tmp_path, datetime(2026, 11, 13, 12, tzinfo=UTC), ok_body()))
    at.run()
    assert not at.exception
    assert coverage_frame(at).loc["Weather forecasts", "Usable"] == 1
    assert "24.0 h since receipt" in all_text(at)


# ---------------------------------------------------------------- 7. no writes


def test_reads_and_rendering_write_nothing(stores, tmp_path, monkeypatch):
    before = tree(stores["dir"])
    sc.build(GAMES, G, environ=stores["env"], now=NOW)
    sc.build(GAMES, OLD, environ=stores["env"], now=NOW)
    run_panel(tmp_path, stores["env"], monkeypatch)
    assert tree(stores["dir"]) == before                       # same files, bytes and mtimes; no journals


def test_panel_never_imports_captures_or_makes_requests(stores, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the panel must not capture, import, initialize or request")
    for name in ("capture", "import_captures", "connect_for_import", "fetch"):
        monkeypatch.setattr(wh, name, boom)
    for name in ("import_captures", "connect_for_import"):
        monkeypatch.setattr(ih, name, boom)
    monkeypatch.setattr(socket.socket, "connect", boom)
    ctx = sc.build(GAMES, G, environ=stores["env"], now=NOW)
    assert ctx["injuries"]["state"] == "ok" and ctx["weather"]["state"] == "ok"


# ---------------------------------------------------------------- 8. the real home page


from test_home_downloads import UPCOMING, links, write_home_fixture  # noqa: E402


PANEL_COLUMNS = {"Valid time (UTC)", "Identity", "Change", "Reason", "Games in selector"}


def home_stores(tmp_path, game_id, gameday, kickoff, stadium_id, stadium):
    """Context databases for the home fixture's first upcoming game."""
    d = tmp_path / "home_stores"
    d.mkdir()
    now = datetime.now(UTC)
    conn = ih.connect_for_import(d / "injury.sqlite3")
    try:
        conn.execute("BEGIN")
        add_capture(conn, "home-inj", now - timedelta(hours=2), players=WED,
                    games=((game_id, "KC", "BUF", kickoff, "pregame"),))
        conn.execute("COMMIT")
    finally:
        conn.close()
    work = tmp_path / "home_work"
    work.mkdir()
    row = (game_id, str(kickoff.year), "7", gameday, "13:00", "BUF", "",
           "KC", "", "Home", "outdoors", stadium_id, stadium)
    w = weather_capture(work, now - timedelta(hours=1),
                        FakeResp(200, body(start=kickoff - timedelta(hours=3))), game_id=game_id, rows=(row,))
    weather_import(d / "weather.sqlite3", w)
    return d, w


class TestHomePage:
    """The full home page, without and with the panel configured (the same
    fixture and stubs as test_home_downloads.TestHomePageWiring)."""

    @pytest.fixture
    def home_dir(self, tmp_path):
        d = tmp_path / "home" / "data_files"
        d.mkdir(parents=True)
        write_home_fixture(d)
        return d

    @pytest.fixture
    def home(self, home_dir, monkeypatch):
        import betting_log
        import dotenv
        monkeypatch.chdir(home_dir.parent)
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)   # never read the real .env
        monkeypatch.setattr(betting_log, "append_recommendations", lambda *a, **k: 0)
        monkeypatch.setattr(betting_log, "grade_pending", lambda *a, **k: 0)
        monkeypatch.setattr(sys, "excepthook", sys.excepthook)

        def refuse(*a, **k):
            raise OSError("network blocked in this test")
        monkeypatch.setattr(socket.socket, "connect", refuse)
        monkeypatch.setattr(socket, "create_connection", refuse)
        return AppTest.from_file(str(ROOT / "predictions.py"), default_timeout=120)

    def loaded(self, at):
        at.run()
        if f"nfl_predictions_{date.today():%Y%m%d}.csv" not in links(at):
            at.run()
        return at

    def _frames(self, at):
        return [e.value for e in at.dataframe if not (PANEL_COLUMNS & set(e.value.columns))]

    def test_page_with_and_without_context_and_values_unchanged(self, home_dir, home, monkeypatch, tmp_path):
        monkeypatch.delenv(sc.CONFIG_ENV, raising=False)
        before = tree(home_dir)
        plain = self.loaded(home)
        assert not plain.exception, [e.message for e in plain.exception]
        assert not plain.error and not plain.warning
        assert sc.NOT_CONFIGURED in [e.value for e in plain.info]
        plain_frames = self._frames(plain)

        preds = pd.read_csv(home_dir / "nfl_games_historical_with_predictions.csv", sep="\t")
        g = preds[(preds.home_team == UPCOMING[0][0]) & (preds.away_team == UPCOMING[0][1])].iloc[-1]
        import pregame_snapshots as ps
        kickoff, _ = ps.kickoff_utc(g.gameday, g.gametime)
        stores_dir, w = home_stores(tmp_path, g.game_id, g.gameday, kickoff, str(g.stadium_id), str(g.stadium))
        cfg = config(stores_dir, injury_db=stores_dir / "injury.sqlite3", weather_db=stores_dir / "weather.sqlite3")
        stores_before = tree(stores_dir)
        monkeypatch.setenv(sc.CONFIG_ENV, cfg[sc.CONFIG_ENV])
        at = AppTest.from_file(str(ROOT / "predictions.py"), default_timeout=120)
        at = self.loaded(at)
        assert not at.exception, [e.message for e in at.exception]
        assert not at.error, [e.value for e in at.error]
        assert at.selectbox(key="stored_context_game").value == g.game_id
        text = all_text(at)
        assert "Showing capture home-inj" in text and w["capture_id"] in text
        assert "Stored context not configured." not in text
        # Model outputs, picks and every other table are identical; nothing was written.
        configured = self._frames(at)
        assert len(configured) == len(plain_frames)
        for a, b in zip(plain_frames, configured):
            pd.testing.assert_frame_equal(a, b)
        assert tree(home_dir) == before
        assert tree(stores_dir) == stores_before
