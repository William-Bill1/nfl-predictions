"""Focused synthetic tests for scripts/injury_history.py (injury-history research pilot).

Captures are produced by the real injury_reports.capture() writer from a mocked
feed and a temporary schedule, so they pass the existing capture validator. No
network, no retained files."""
import hashlib
import importlib.util
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("injury_history", ROOT / "scripts" / "injury_history.py")
ih = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ih)
ir, fr, on = ih.ir, ih.fr, ih.on

UTC = timezone.utc
WED = datetime(2026, 11, 4, 17, 0, tzinfo=UTC)      # Wed 12:00 EST
SUN = datetime(2026, 11, 8, 14, 0, tzinfo=UTC)      # Sun 09:00 EST
WED2 = datetime(2026, 11, 11, 17, 0, tzinfo=UTC)    # next Wed 12:00 EST
NOW = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
SCHEDULE_ROWS = [
    ("2026_09_KC_MIA", 2026, 9, "2026-11-08", "13:00", "KC", "MIA"),
    ("2026_10_KC_DEN", 2026, 10, "2026-11-15", "13:00", "KC", "DEN"),
]


def entry(name, pid, status="Questionable", injury="Ankle", date="2026-11-04T15:00Z", comment=None):
    athlete = {"displayName": name, "position": {"abbreviation": "WR"}}
    if pid is not None:
        athlete["id"] = pid
    e = {"athlete": athlete, "details": {"type": injury} if injury else {},
         "type": {"name": f"INJURY_STATUS_{status.upper()}", "description": status,
                  "abbreviation": status[0]}, "status": status}
    if date is not None:
        e["date"] = date
    if comment is not None:
        e["shortComment"] = comment
    return e


def team(abbr, entries):
    return {"id": "1", "displayName": on.TEAM_FULL_NAME[abbr], "injuries": entries}


class Resp:
    def __init__(self, payload, status=200):
        self.status_code = status
        self.content = json.dumps(payload).encode()
        self.headers = {"Date": "Wed, 04 Nov 2026 17:00:02 GMT"}

    def iter_content(self, chunk_size=1, decode_unicode=False):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]

    def close(self):
        pass


@pytest.fixture
def env(tmp_path):
    df = pd.DataFrame(SCHEDULE_ROWS, columns=["game_id", "season", "week", "gameday", "gametime",
                                              "away_team", "home_team"])
    df["away_score"] = None
    df["home_score"] = None
    df["game_type"] = "REG"
    sched = tmp_path / "nfl_games_historical.csv"
    df.to_csv(sched, sep="\t", index=False)
    cdir = tmp_path / "captures"

    def make(at, teams, status=200):
        times = [at, at + timedelta(seconds=2)]
        res = ir.capture(now=at, clock=lambda: times.pop(0) if len(times) > 1 else times[0],
                         get=lambda *a, **k: Resp({"injuries": teams}, status), capture_dir=cdir,
                         schedule_path=sched, repo_dir=ROOT)
        assert res.path is not None, res
        return Path(res.path)
    return {"make": make, "db": tmp_path / "injury.sqlite3", "dir": tmp_path}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def imp(db, *paths):
    conn = ih.connect_for_import(db)
    try:
        return ih.import_captures(conn, paths, now=NOW)
    finally:
        conn.close()


def ro(db):
    return ih.open_read_only(db)


def dump(db):
    conn = sqlite3.connect(db)
    try:
        return {t: conn.execute(f"SELECT * FROM {t} ORDER BY 1, 2").fetchall()
                for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()


WED_TEAMS = [team("KC", [entry("Alpha One", "101", "Questionable", "Ankle"),
                         entry("Bravo Two", "102", "Out", "Knee"),
                         entry("Charlie Three", "103", "Questionable", "Hamstring"),
                         entry("Solo Name", None, "Questionable", None, date=None),
                         entry("Echo Five", "105", "Active", None, comment="Caught 3 passes Sunday.")]),
             team("MIA", [entry("Miami Guy", "201", "Out", "Shoulder")])]
SUN_TEAMS = [team("KC", [entry("Alpha One", "101", "Out", "Ankle", date="2026-11-07T20:00Z"),
                         entry("Bravo Two", "102", "Out", "Knee"),
                         entry("Delta Four", "104", "Doubtful", "Groin", date="2026-11-07T21:00Z"),
                         entry("Solo Name", None, "Questionable", None, date=None),
                         entry("Echo Five", "105", "Active", None, comment="Caught 3 passes Sunday.")]),
             team("MIA", [entry("Miami Guy", "201", "Out", "Shoulder"),
                          entry("Charlie Three", "103", "Questionable", "Hamstring")])]


def test_import_preserves_times_raw_values_and_missing_values(env):
    path = env["make"](WED, WED_TEAMS)
    doc = json.loads(path.read_text(encoding="utf-8"))
    res = imp(env["db"], path)
    assert res[0]["created"] is True and res[0]["observations"] == 6
    conn = ro(env["db"])
    cap = ih._rows(conn, "SELECT * FROM injury_captures")[0]
    assert cap["received_at_utc"] == doc["received_at"] and cap["request_started_at_utc"] == doc["request_started_at"]
    assert cap["imported_at_utc"] == "2026-11-14T12:00:00+00:00" != cap["received_at_utc"]
    assert cap["file_sha256"] == sha(path) and cap["payload_sha256"] == doc[ir.CHECKSUM_FIELD]
    assert cap["http_last_modified"] is None and cap["http_date"] == "Wed, 04 Nov 2026 17:00:02 GMT"
    assert cap["source_official_nfl_report"] == 0 and json.loads(cap["source_json"]) == doc["source"]
    obs = ih._rows(conn, "SELECT * FROM injury_observations ORDER BY entry_index")
    for o, r in zip(obs, doc["players"]):
        assert json.loads(o["raw_row_json"]) == r
        assert (o["source_status"], o["player_id"], o["source_entry_date_raw"], o["source_entry_date_utc"]) == \
            (r["source_status"], r["player_id"], r["source_entry_date_raw"], r["source_entry_date_utc"])
    solo = [o for o in obs if o["player_name"] == "Solo Name"][0]
    assert solo["player_id"] is None and solo["player_id_source"] is None
    assert solo["source_entry_date_raw"] is None and solo["source_entry_date_utc"] is None
    assert solo["injury_type"] is None and solo["practice_participation"] is None
    alpha = [o for o in obs if o["player_id"] == "101"][0]
    assert alpha["source_entry_date_raw"] == "2026-11-04T15:00Z"        # source time, not capture time
    assert alpha["source_entry_date_utc"] != cap["received_at_utc"]
    conn.close()


def test_duplicate_imports_add_nothing(env):
    path = env["make"](WED, WED_TEAMS)
    imp(env["db"], path)
    before = dump(env["db"])
    again = imp(env["db"], path, path)
    assert [r["created"] for r in again] == [False, False]
    assert all(r["same_file_bytes"] for r in again)
    assert dump(env["db"]) == before


def test_multiple_observations_for_one_player_are_kept(env):
    a, b = env["make"](WED, WED_TEAMS), env["make"](SUN, SUN_TEAMS)
    imp(env["db"], a, b)
    cols, rows = fr.run_sql(env["db"], "SELECT c.capture_id, o.source_status FROM injury_observations o "
                            "JOIN injury_captures c USING (capture_import_id) WHERE o.player_id = '101' "
                            "ORDER BY c.received_at_utc")
    assert [r[1] for r in rows] == ["Questionable", "Out"] and rows[0][0] != rows[1][0]


def test_compare_categories_and_absence_is_not_recovery(env):
    a, b = env["make"](WED, WED_TEAMS), env["make"](SUN, SUN_TEAMS)
    imp(env["db"], a, b)
    conn = ro(env["db"])
    out = ih.compare_team(conn, "KC", a.stem, b.stem)
    conn.close()
    assert out["counts"] == {"newly_observed": 1, "changed": 1, "unchanged": 3,
                             "absent_from_later": 1, "identity_unresolved": 0}
    ch = out["changed"][0]
    assert ch["player_id"] == "101" and ch["identity_basis"] == "source_player_id"
    assert ch["changed_fields"]["source_status"] == {"earlier": "Questionable", "later": "Out"}
    assert "game_id" not in ch["changed_fields"]
    assert out["newly_observed"][0]["player_id"] == "104"
    gone = out["absent_from_later"][0]
    assert gone["player_id"] == "103" and gone["source_status"] == "Questionable"   # last reported, unchanged
    assert gone["listed_for_other_team_in_later"] == ["MIA"]
    assert "does NOT mean recovered, healthy, active or cleared" in out["absent_from_later_meaning"]
    solo = [u for u in out["unchanged"] if u["player_name"] == "Solo Name"][0]
    assert solo["identity_basis"] == "inferred_name_match" and solo["player_id"] is None
    assert out["identity"]["matched_by_inferred_name"] == 1
    assert out["freshness"] == "not checked"
    assert out["earlier"]["captured_at_utc (response received)"] < out["later"]["captured_at_utc (response received)"]
    assert out["earlier"]["team_game_in_scope"]["game_id"] == "2026_09_KC_MIA"
    assert any("not the official NFL injury report" in x for x in out["limitations"])


def test_ambiguous_names_and_missing_ids_are_never_joined(env):
    wed = [team("KC", [entry("Same Name", None, "Out"), entry("Same Name", None, "Questionable"),
                       entry("Later Has Id", None, "Questionable"), entry("Kept", "301")]),
           team("MIA", [entry("Miami Guy", "201")])]
    sun = [team("KC", [entry("Same Name", None, "Out"), entry("Later Has Id", "302", "Out"),
                       entry("Kept", "301")]),
           team("MIA", [entry("Miami Guy", "201")])]
    a, b = env["make"](WED, wed), env["make"](SUN, sun)
    imp(env["db"], a, b)
    conn = ro(env["db"])
    out = ih.compare_team(conn, "KC", a.stem, b.stem)
    conn.close()
    un = out["identity_unresolved"]
    assert sorted(u["player_name"] for u in un["earlier"]) == ["Later Has Id", "Same Name", "Same Name"]
    assert [u["player_name"] for u in un["later"]] == ["Same Name"]
    assert "no ID is inferred" in [u for u in un["earlier"] if u["player_name"] == "Later Has Id"][0]["reason"]
    assert out["identity"]["matched_by_inferred_name"] == 0
    # The ID'd later entry is reported as newly observed under its own ID, not joined to the name.
    assert [n["player_id"] for n in out["newly_observed"]] == ["302"]
    assert out["counts"]["unchanged"] == 1 and out["counts"]["absent_from_later"] == 0


@pytest.mark.parametrize("duplicate_side", ["earlier", "later", "both"])
def test_duplicated_id_is_unresolved_on_both_sides(env, duplicate_side):
    single = [entry("Player A", "101"), entry("Other", "150", "Out")]
    doubled = [entry("Player A", "101"), entry("Player A", "101", "Out"), entry("Other", "150", "Out")]
    mia = team("MIA", [entry("Player M", "201")])
    a = env["make"](WED, [team("KC", doubled if duplicate_side in ("earlier", "both") else single), mia])
    b = env["make"](SUN, [team("KC", doubled if duplicate_side in ("later", "both") else single), mia])
    imp(env["db"], a, b)
    conn = ro(env["db"])
    out = ih.compare_team(conn, "KC", a.stem, b.stem)
    conn.close()
    un = out["identity_unresolved"]
    n_earlier = 2 if duplicate_side in ("earlier", "both") else 1
    n_later = 2 if duplicate_side in ("later", "both") else 1
    assert [u["player_id"] for u in un["earlier"]] == ["101"] * n_earlier
    assert [u["player_id"] for u in un["later"]] == ["101"] * n_later
    assert all("ambiguous on both sides" in u["reason"] for u in un["earlier"] + un["later"])
    for cat in ("newly_observed", "changed", "unchanged", "absent_from_later"):
        assert "101" not in [x["player_id"] for x in out[cat]], cat
    assert out["counts"] == {"newly_observed": 0, "changed": 0, "unchanged": 1, "absent_from_later": 0,
                             "identity_unresolved": n_earlier + n_later}
    assert out["unchanged"][0]["player_id"] == "150"                # unaffected players still compare
    assert out["identity"]["matched_by_source_player_id"] == 1


def test_malformed_input_refuses_the_whole_import(env):
    good, other = env["make"](WED, WED_TEAMS), env["make"](SUN, SUN_TEAMS)
    imp(env["db"], good)
    before_db, before_files = dump(env["db"]), {p: sha(p) for p in (good, other)}
    doc = json.loads(other.read_text(encoding="utf-8"))
    doc["players"][0]["source_status"] = "Healthy"                 # edited without resealing
    tampered = env["dir"] / "tampered" / other.name
    tampered.parent.mkdir()
    tampered.write_text(json.dumps(doc), encoding="utf-8")
    bad_json = env["dir"] / "bad" / other.name
    bad_json.parent.mkdir()
    bad_json.write_text("{not json", encoding="utf-8")
    renamed = env["dir"] / "renamed.json"
    renamed.write_bytes(other.read_bytes())
    for bad in (tampered, bad_json, renamed, env["dir"] / "captures"):
        with pytest.raises(ih.ImportRefused):
            imp(env["db"], other, bad)                              # valid file first: still nothing
        assert dump(env["db"]) == before_db
    assert {p: sha(p) for p in (good, other)} == before_files


def test_conflicting_content_for_an_imported_capture_is_refused(env, tmp_path):
    path = env["make"](WED, WED_TEAMS)
    imp(env["db"], path)
    before = dump(env["db"])
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["players"][0]["short_comment"] = "changed later"
    ir._seal(doc)                                                    # validly resealed, same capture_id
    alt = tmp_path / "alt" / path.name
    alt.parent.mkdir()
    alt.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ih.ImportRefused, match="different content"):
        imp(env["db"], alt)
    assert dump(env["db"]) == before


def test_compare_refusals(env):
    a, b = env["make"](WED, WED_TEAMS), env["make"](SUN, SUN_TEAMS)
    down = env["make"](WED2, [], status=503)                        # unavailable capture
    imp(env["db"], a, b, down)
    conn = ro(env["db"])
    for args in (("KC", a.stem, a.stem), ("KC", b.stem, a.stem), ("KC", a.stem, down.stem),
                 ("BUF", a.stem, b.stem), ("KC", a.stem, "20991231T000000Z-000000000000")):
        with pytest.raises(ih.CompareRefused):
            ih.compare_team(conn, *args)
    conn.close()


def test_list_reports_coverage_and_slot_gaps(env):
    a, c = env["make"](WED, WED_TEAMS), env["make"](WED2, [team("KC", [entry("Kept", "301")])])
    imp(env["db"], a, c)
    conn = ro(env["db"])
    out = ih.list_captures(conn)
    conn.close()
    assert [x["capture_id"] for x in out["captures"]] == [a.stem, c.stem]
    assert out["captures"][0]["entries_per_feed_team"] == [1, 5]
    assert out["freshness"] == "not checked"
    slots = {s["slot_id"]: s["status"] for s in out["slots_in_imported_range"]}
    assert slots == {"2026-11-04_wednesday_noon": "filled",
                     "2026-11-08_sunday_morning": "no usable imported capture",
                     "2026-11-11_wednesday_noon": "filled"}


def test_read_only_sql_rejects_writes_and_attach(env, capsys):
    path = env["make"](WED, WED_TEAMS)
    imp(env["db"], path)
    before = sha(env["db"])
    probe = env["dir"] / "attached.sqlite3"
    for q in ("INSERT INTO injury_captures (capture_id) VALUES ('x')",
              "DELETE FROM injury_observations", "DROP TABLE injury_observations",
              f"ATTACH DATABASE '{probe.as_posix()}' AS x", "PRAGMA query_only = OFF",
              "CREATE TEMP TABLE t (a)"):
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            fr.run_sql(env["db"], q)
        assert ih.main(["sql", "--db", str(env["db"]), q]) == 2
    assert not probe.exists() and sha(env["db"]) == before
    assert ih.main(["sql", "--db", str(env["db"]), "SELECT COUNT(*) FROM injury_observations"]) == 0
    assert capsys.readouterr().out.strip().endswith("6")


def test_other_schemas_are_refused_unchanged(env):
    path = env["make"](WED, WED_TEAMS)
    football = env["dir"] / "football.sqlite3"
    conn = sqlite3.connect(football)
    conn.executescript(fr.SCHEMA)
    conn.close()
    extra = env["dir"] / "extra.sqlite3"
    conn = sqlite3.connect(extra)
    conn.executescript(ih.SCHEMA + "CREATE TABLE extra (a);")
    conn.close()
    for db in (football, extra):
        before = sha(db)
        with pytest.raises(fr.SchemaRefused):
            imp(db, path)
        assert ih.main(["import", "--db", str(db), str(path)]) == 2
        assert ih.main(["list", "--db", str(db)]) == 2
        assert sha(db) == before


def test_prior_contents_and_sources_unchanged_by_later_work(env, capsys):
    a, b = env["make"](WED, WED_TEAMS), env["make"](SUN, SUN_TEAMS)
    src_before = {p: sha(p) for p in (a, b)}
    imp(env["db"], a)
    first = dump(env["db"])
    imp(env["db"], b)
    after = dump(env["db"])
    for t, rows in first.items():
        assert after[t][:len(rows)] == rows                         # earlier rows untouched
    db_before = sha(env["db"])
    assert ih.main(["compare", "--db", str(env["db"]), "--team", "KC", "--earlier", a.stem,
                    "--later", b.stem]) == 0
    assert json.loads(capsys.readouterr().out)["counts"]["changed"] == 1
    assert ih.main(["list", "--db", str(env["db"])]) == 0
    assert sha(env["db"]) == db_before
    assert {p: sha(p) for p in (a, b)} == src_before
