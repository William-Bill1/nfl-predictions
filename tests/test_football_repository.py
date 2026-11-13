"""Synthetic tests for the football-repository pilot (scripts/football_repository.py)."""
import csv
import hashlib
import importlib.util
import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("football_repository",
                                              ROOT / "scripts" / "football_repository.py")
fr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fr)

NOW = datetime(2026, 11, 13, 17, 0, tzinfo=timezone.utc)
DAY1, DAY2 = date(2026, 11, 13), date(2026, 11, 14)      # consecutive Toronto dates
COLUMNS = list(fr.REQUIRED_COLUMNS) + ["temp"]


def game(game_id, home, away, hs, as_, *, season=2025, game_type="REG", gameday="2025-10-05",
         roof="outdoors", surface="grass", overtime=None, location="Home"):
    scored = hs is not None and as_ is not None
    return {"game_id": game_id, "season": str(season), "game_type": game_type, "week": "5",
            "gameday": gameday, "gametime": "13:00", "away_team": away,
            "away_score": "" if as_ is None else f"{as_}.0", "home_team": home,
            "home_score": "" if hs is None else f"{hs}.0", "location": location,
            "result": f"{hs - as_}.0" if scored else "", "total": f"{hs + as_}.0" if scored else "",
            "overtime": "" if not scored else (overtime or "0.0"), "roof": roof, "surface": surface,
            "stadium_id": "STA00", "stadium": "Synthetic Field", "temp": ""}


BASE = [
    game("2025_01_BBB_AAA", "AAA", "BBB", 24, 17),                           # AAA home W, BBB away L
    game("2025_02_AAA_CCC", "CCC", "AAA", 30, 20, surface="grass "),         # AAA away L, padded grass
    game("2025_03_DDD_AAA", "AAA", "DDD", 20, 20, overtime="1.0"),           # tie
    game("2025_04_AAA_BBB", "BBB", "AAA", 10, 27, roof="open"),              # open roof
    game("2025_05_EEE_AAA", "AAA", "EEE", 31, 3, surface="fieldturf"),       # known artificial
    game("2025_06_FFF_AAA", "AAA", "FFF", 21, 14, roof=""),                  # roof unknown, grass
    game("2025_07_GGG_AAA", "AAA", "GGG", 17, 13, surface=""),               # outdoors, surface unknown
    game("2025_08_HHH_AAA", "AAA", "HHH", 28, 0, surface="hybrid"),          # unrecognized surface
    game("2025_09_AAA_III", "III", "AAA", 14, 21, roof="dome", surface="fieldturf"),
    game("2025_10_AAA_JJJ", "JJJ", "AAA", 7, 10, roof="closed", surface="grass"),
    game("2025_11_KKK_AAA", "AAA", "KKK", None, None, gameday="2026-11-20"),  # scheduled
    game("2025_12_LLL_AAA", "AAA", "LLL", 21, 7, gameday="2026-11-13"),       # scored, same day as DAY1
    game("2025_13_MMM_AAA", "AAA", "MMM", 0, 0),                              # 0-0 placeholder
    game("2025_14_NNN_AAA", "AAA", "NNN", 20, 10, gameday=""),                # no valid game day
    game("2025_15_AAA_PPP", "PPP", "AAA", 24, 3, roof="dome", surface=""),    # dome, surface missing
    game("2025_20_OOO_AAA", "AAA", "OOO", 35, 31, game_type="WC", gameday="2026-01-10"),
    game("2024_01_BBB_AAA", "AAA", "BBB", 13, 10, season=2024, gameday="2024-09-08"),
]


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return path


@pytest.fixture
def db(tmp_path):
    conn = fr.connect(tmp_path / "pilot.sqlite3")
    yield conn
    conn.close()


@pytest.fixture
def source(tmp_path):
    return write(tmp_path / "games.csv", BASE)


def imported(db, source, day=DAY1):
    iid, _ = fr.import_source(db, source, now=NOW)
    aid, _ = fr.assess_completion(db, iid, day, now=NOW)
    return iid, aid


def counts(conn):
    return tuple(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in
                 ("source_imports", "game_observations", "completion_assessments", "game_completions"))


def ids(items):
    return [i["game_id"] for i in items]


def test_outdoor_grass_record_home_away_and_ties(db, source):
    _, aid = imported(db, source)
    r = fr.team_record(db, aid, "AAA", 2025)
    assert (r["W"], r["L"], r["T"], r["games"], r["W:L"]) == (1, 1, 1, 3, "1:1")
    assert r["included_game_ids"] == ["2025_01_BBB_AAA", "2025_02_AAA_CCC", "2025_03_DDD_AAA"]
    assert r["win_pct_excluding_ties"] == 0.5 and r["nfl_standings_pct"] == pytest.approx(0.5)
    assert (r["evaluation_toronto_date"], r["completion_rule_version"]) == ("2026-11-13",
                                                                           fr.COMPLETION_RULE_VERSION)
    assert "retrospective" in r["conditions_basis"]
    bbb = fr.team_record(db, aid, "BBB", 2025)        # away loss; its home open-roof loss excluded
    assert (bbb["W"], bbb["L"], bbb["T"], bbb["included_game_ids"]) == (0, 1, 0, ["2025_01_BBB_AAA"])
    assert bbb["win_pct_excluding_ties"] == 0.0
    ddd = fr.team_record(db, aid, "DDD", 2025)        # tie only: W/(W+L) undefined
    assert (ddd["W"], ddd["L"], ddd["T"], ddd["win_pct_excluding_ties"]) == (0, 0, 1, None)
    rows = db.execute("SELECT team, opponent, is_home, outcome FROM team_game_results "
                      "WHERE assessment_id = ? AND game_id = '2025_01_BBB_AAA' ORDER BY team", (aid,)).fetchall()
    assert rows == [("AAA", "BBB", 1, "W"), ("BBB", "AAA", 0, "L")]


def test_open_roof_postseason_and_season_are_explicit_choices(db, source):
    _, aid = imported(db, source)
    with_open = fr.team_record(db, aid, "AAA", 2025, roofs=("outdoors", "open"))
    assert (with_open["W"], with_open["L"], with_open["T"]) == (2, 1, 1)
    assert "2025_04_AAA_BBB" in with_open["included_game_ids"]
    assert "2025_10_AAA_JJJ" in ids(with_open["excluded_by_known_conditions"])   # closed never outdoor
    assert fr.team_record(db, aid, "AAA", 2025, game_type="WC")["included_game_ids"] == ["2025_20_OOO_AAA"]
    assert fr.team_record(db, aid, "AAA", 2024)["included_game_ids"] == ["2024_01_BBB_AAA"]


def test_exclusion_categories_are_disjoint_and_complete(db, source):
    _, aid = imported(db, source)
    r = fr.team_record(db, aid, "AAA", 2025)
    unfinished = {e["game_id"]: e["reason"] for e in r["excluded_unfinished_or_unverifiable"]}
    assert set(unfinished) == {"2025_11_KKK_AAA", "2025_12_LLL_AAA", "2025_13_MMM_AAA", "2025_14_NNN_AAA"}
    assert unfinished["2025_12_LLL_AAA"] == "game day is not before today"
    assert unfinished["2025_13_MMM_AAA"] == "0-0 placeholder"
    assert ids(r["excluded_by_known_conditions"]) == ["2025_04_AAA_BBB", "2025_05_EEE_AAA",
                                                      "2025_09_AAA_III", "2025_10_AAA_JJJ", "2025_15_AAA_PPP"]
    assert ids(r["eligibility_undetermined_unknown_condition"]) == ["2025_06_FFF_AAA", "2025_07_GGG_AAA",
                                                                    "2025_08_HHH_AAA"]
    groups = [r["included_game_ids"], list(unfinished), ids(r["excluded_by_known_conditions"]),
              ids(r["eligibility_undetermined_unknown_condition"])]
    flat = [g for grp in groups for g in grp]
    assert len(flat) == len(set(flat)) == r["games_in_scope"] == 15     # no double counting


def test_dome_with_missing_surface_is_excluded_and_a_quality_gap(db, source):
    _, aid = imported(db, source)
    r = fr.team_record(db, aid, "AAA", 2025)
    assert "2025_15_AAA_PPP" in ids(r["excluded_by_known_conditions"])
    assert "2025_15_AAA_PPP" not in ids(r["eligibility_undetermined_unknown_condition"])
    q = r["data_quality_missing_conditions"]
    assert q == {"roof_unknown_only": ["2025_06_FFF_AAA"],
                 "surface_unknown_only": ["2025_07_GGG_AAA", "2025_08_HHH_AAA", "2025_15_AAA_PPP"],
                 "roof_and_surface_unknown": []}


def test_raw_conditions_are_kept_beside_categories(db, source):
    iid, _ = imported(db, source)
    raw = dict(db.execute("SELECT game_id, surface_raw || '|' || surface_category || '|' || "
                          "roof_raw || '|' || roof_category FROM game_observations WHERE import_id = ?",
                          (iid,)).fetchall())
    assert raw["2025_02_AAA_CCC"] == "grass |natural_grass|outdoors|outdoors"
    assert raw["2025_04_AAA_BBB"] == "grass|natural_grass|open|open"
    assert raw["2025_06_FFF_AAA"] == "grass|natural_grass||unknown"
    assert raw["2025_08_HHH_AAA"] == "hybrid|unknown|outdoors|outdoors"


def test_consecutive_dates_one_import_and_reproducible_assessments(db, source):
    iid, day1 = imported(db, source, DAY1)
    snapshot = db.execute("SELECT * FROM game_completions WHERE assessment_id = ? ORDER BY game_id",
                          (day1,)).fetchall()
    first = fr.team_record(db, day1, "AAA", 2025)

    iid_again, created = fr.import_source(db, source, now=NOW)           # same bytes next day
    assert (iid_again, created) == (iid, False)
    day2, created2 = fr.assess_completion(db, iid, DAY2, now=NOW)
    assert created2 and day2 != day1
    assert counts(db)[:3] == (1, len(BASE), 2)

    second = fr.team_record(db, day2, "AAA", 2025)
    status = dict(db.execute("SELECT assessment_id || ':' || game_id, completion_status FROM "
                             "game_completions WHERE game_id IN ('2025_12_LLL_AAA', '2025_11_KKK_AAA')"))
    assert status[f"{day1}:2025_12_LLL_AAA"] == "excluded"                # same day on DAY1
    assert status[f"{day2}:2025_12_LLL_AAA"] == "final"                   # eligible the next day
    assert status[f"{day1}:2025_11_KKK_AAA"] == status[f"{day2}:2025_11_KKK_AAA"] == "excluded"
    assert "2025_12_LLL_AAA" in second["included_game_ids"]
    assert (second["W"], second["L"], second["T"]) == (2, 1, 1)
    assert second["evaluation_toronto_date"] == "2026-11-14"

    # The DAY1 assessment is unchanged and still gives the same answer.
    assert db.execute("SELECT * FROM game_completions WHERE assessment_id = ? ORDER BY game_id",
                      (day1,)).fetchall() == snapshot
    assert fr.team_record(db, day1, "AAA", 2025) == first
    assert fr.find_assessment(db, iid, DAY1) == day1 and fr.find_assessment(db, iid, DAY2) == day2


def test_repeating_an_assessment_adds_nothing(db, source):
    iid, aid = imported(db, source)
    before = counts(db)
    assert fr.assess_completion(db, iid, DAY1, now=NOW) == (aid, False)
    assert counts(db) == before == (1, len(BASE), 1, len(BASE))


def test_one_assessment_never_multiplies_through_others(db, source):
    iid, day1 = imported(db, source, DAY1)
    fr.assess_completion(db, iid, DAY2, now=NOW)
    fr.assess_completion(db, iid, date(2026, 11, 15), now=NOW)
    n = db.execute("SELECT COUNT(*) FROM team_game_results WHERE assessment_id = ? "
                   "AND game_id = '2025_01_BBB_AAA'", (day1,)).fetchone()[0]
    assert n == 2                                                       # one row per team
    assert fr.team_record(db, day1, "AAA", 2025)["games"] == 3
    with pytest.raises(LookupError):
        fr.find_assessment(db, iid, date(2026, 11, 16))                 # never falls back to another date


def test_changed_source_is_a_separate_import(db, source, tmp_path):
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    first, aid1 = imported(db, source)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before    # source only read
    changed = [dict(g) for g in BASE]
    changed[1] = game("2025_02_AAA_CCC", "CCC", "AAA", 20, 30, surface="grass ")   # corrected score
    second, created = fr.import_source(db, write(tmp_path / "games_v2.csv", changed), now=NOW)
    assert created and second != first
    aid2, _ = fr.assess_completion(db, second, DAY1, now=NOW)
    assert counts(db)[:2] == (2, 2 * len(BASE))
    assert fr.team_record(db, aid1, "AAA", 2025)["W:L"] == "1:1"        # earlier import intact
    assert fr.team_record(db, aid2, "AAA", 2025)["W:L"] == "2:0"


@pytest.mark.parametrize("bad, message", [
    (game("2025_01_BBB_AAA", "AAA", "ZZZ", 3, 0), "duplicate game_id"),
    (game("2025_99_XXX_AAA", "AAA", "", 3, 0), "invalid teams"),
    (game("2025_99_XXX_AAA", "AAA", "AAA", 3, 0), "invalid teams"),
    (game("2025_99_XXX_AAA", "AAA", "XXX", 3, 0, season=""), "not a valid NFL season"),
    (game(" 2025_99_XXX_AAA", "AAA", "XXX", 3, 0), "invalid game_id"),
])
def test_refused_import_leaves_no_partial_records(db, source, tmp_path, bad, message):
    imported(db, source)
    before = counts(db)
    bad_file = write(tmp_path / "bad.csv", BASE[:5] + [bad] + BASE[5:])   # fails after 5 inserts
    with pytest.raises(fr.ImportRefused, match=message):
        fr.import_source(db, bad_file, now=NOW)
    assert counts(db) == before


def test_failed_assessment_rolls_back(db, source, monkeypatch):
    iid, _ = imported(db, source)
    before = counts(db)
    original, calls = fr.bj.completion_evidence, []

    def failing(schedule, game_id, now):            # fails after 4 games were written
        calls.append(game_id)
        if len(calls) == 5:
            raise RuntimeError("simulated failure mid-assessment")
        return original(schedule, game_id, now)

    monkeypatch.setattr(fr.bj, "completion_evidence", failing)
    with pytest.raises(RuntimeError, match="simulated"):
        fr.assess_completion(db, iid, DAY2, now=NOW)
    assert counts(db) == before


def test_missing_required_column_is_refused(db, tmp_path):
    path = tmp_path / "nosurface.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[c for c in COLUMNS if c != "surface"], delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(BASE[:2])
    with pytest.raises(fr.ImportRefused, match="surface"):
        fr.import_source(db, path, now=NOW)
    assert counts(db) == (0, 0, 0, 0)


def test_ordinary_sql_beyond_the_example_is_read_only(tmp_path, source):
    path = tmp_path / "pilot.sqlite3"
    conn = fr.connect(path)
    _, aid = imported(conn, source)
    conn.close()
    cols, rows = fr.run_sql(path, "SELECT roof_category, outcome, COUNT(*) FROM team_game_results "
                                  "WHERE assessment_id = ? AND team = 'AAA' AND season = 2025 "
                                  "GROUP BY 1, 2 ORDER BY 1, 2", (aid,))
    assert cols == ["roof_category", "outcome", "COUNT(*)"]
    assert ("dome", "W", 1) in rows and ("closed", "W", 1) in rows
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        fr.run_sql(path, "DELETE FROM game_observations")


# ---- read-only ad hoc SQL -------------------------------------------------

def file_db(tmp_path, source, name="pilot.sqlite3"):
    path = tmp_path / name
    conn = fr.connect(path)
    imported(conn, source)
    conn.close()
    return path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_attach_is_refused_and_creates_nothing(tmp_path, source):
    path = file_db(tmp_path, source)
    target = tmp_path / "created_by_attach.sqlite3"
    before = digest(path)
    for sql, params in [("ATTACH DATABASE ? AS other", (str(target),)),
                        (f"ATTACH '{target}' AS other", ())]:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            fr.run_sql(path, sql, params)
    assert not target.exists()
    assert digest(path) == before


@pytest.mark.parametrize("sql", [
    "INSERT INTO source_imports VALUES (9, 'x', 'y', 'z', 0, 'v')",
    "UPDATE game_observations SET roof_raw = 'dome'",
    "DELETE FROM completion_assessments",
    "CREATE TABLE extra (x)",
    "DROP VIEW team_game_results",
    "ALTER TABLE source_imports ADD COLUMN extra TEXT",
    "PRAGMA user_version = 7",
    "PRAGMA query_only = OFF",
    "CREATE TEMP TABLE scratch (x)",
    "BEGIN IMMEDIATE",
    "VACUUM",
    "SELECT load_extension('x')",
])
def test_writes_and_side_effects_are_refused_and_database_unchanged(tmp_path, source, sql):
    path = file_db(tmp_path, source)
    before = digest(path)
    with pytest.raises(sqlite3.DatabaseError):
        fr.run_sql(path, sql)
    assert digest(path) == before
    assert sorted(tmp_path.iterdir()) == sorted([path, tmp_path / "games.csv"])


def test_ordinary_read_only_queries_still_work(tmp_path, source):
    path = file_db(tmp_path, source)
    before = digest(path)
    _, rows = fr.run_sql(path, "SELECT COUNT(*) FROM game_observations o JOIN game_completions c "
                               "ON c.game_id = o.game_id WHERE c.completion_status = 'final'")
    assert rows == [(13,)]                                    # 17 games, 4 not final on DAY1
    _, rows = fr.run_sql(path, "WITH w AS (SELECT team, SUM(outcome = 'W') AS wins FROM team_game_results "
                               "WHERE assessment_id = 1 GROUP BY team) SELECT wins FROM w WHERE team = 'AAA'")
    assert rows == [(10,)]                                    # AAA wins, any type/season/conditions
    _, rows = fr.run_sql(path, "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 3) "
                               "SELECT SUM(i), MAX(LENGTH(game_id)) FROM n, source_imports s "
                               "JOIN game_observations o ON o.import_id = s.import_id")
    assert rows == [(3 * len(BASE) * 2, 15)]
    assert digest(path) == before


# ---- schema refusal ---------------------------------------------------------

V1_SCHEMA = """
CREATE TABLE source_imports (import_id INTEGER PRIMARY KEY, source_path TEXT NOT NULL,
    source_sha256 TEXT NOT NULL, imported_at_utc TEXT NOT NULL,
    completion_evaluated_toronto_date TEXT NOT NULL, row_count INTEGER NOT NULL,
    transformation_version TEXT NOT NULL, UNIQUE (source_sha256, transformation_version));
CREATE TABLE game_observations (import_id INTEGER NOT NULL, game_id TEXT NOT NULL,
    season INTEGER NOT NULL, game_type TEXT NOT NULL, week INTEGER, gameday TEXT, gametime TEXT,
    home_team TEXT NOT NULL, away_team TEXT NOT NULL, home_score INTEGER, away_score INTEGER,
    location TEXT, stadium_id TEXT, stadium TEXT, roof_raw TEXT, surface_raw TEXT,
    roof_category TEXT NOT NULL, surface_category TEXT NOT NULL, completion_status TEXT NOT NULL,
    completion_reason TEXT NOT NULL, PRIMARY KEY (import_id, game_id));
CREATE VIEW team_game_results AS SELECT import_id, game_id FROM game_observations
    WHERE completion_status = 'final';
INSERT INTO source_imports VALUES (1, 'x', 'abc', '2026-11-13T15:20:22+00:00', '2026-11-13', 0, 'football-repo-pilot-1');
"""


def synthetic_db(path, script):
    conn = sqlite3.connect(str(path))
    conn.executescript(script)
    conn.close()
    return path


def test_new_database_initializes_and_current_schema_reopens_unchanged(tmp_path):
    path = tmp_path / "new.sqlite3"
    assert fr.schema_state(path) == "empty"
    fr.connect(path).close()
    assert fr.schema_state(path) == "current"
    before = digest(path)
    fr.connect(path).close()                                  # reopen: no schema statements run
    fr.open_read_only(path).close()
    assert digest(path) == before
    empty = tmp_path / "zero_bytes.sqlite3"
    empty.write_bytes(b"")
    fr.connect(empty).close()                                 # an empty file counts as new
    assert fr.schema_state(empty) == "current"


@pytest.mark.parametrize("script", [
    V1_SCHEMA,                                                           # the first pilot's schema
    fr.SCHEMA.split("CREATE TABLE IF NOT EXISTS completion_assessments")[0],   # partial: 2 tables only
    fr.SCHEMA.replace("transformation_version TEXT NOT NULL,", "transformation_version TEXT,", 1),
    fr.SCHEMA + "CREATE TABLE notes (x);",                               # current plus an extra table
])
def test_unsupported_schema_is_refused_unchanged(tmp_path, script):
    path = synthetic_db(tmp_path / "other.sqlite3", script)
    before, files = digest(path), sorted(tmp_path.iterdir())
    with pytest.raises(fr.SchemaRefused, match="unsupported database schema"):
        fr.connect(path)
    with pytest.raises(fr.SchemaRefused):
        fr.open_read_only(path)
    assert fr.main(["assess", "--db", str(path), "--import-id", "1", "--date", "2026-11-13"]) == 2
    assert fr.main(["record", "--db", str(path), "--import-id", "1", "--date", "2026-11-13",
                    "--team", "AAA", "--season", "2025"]) == 2
    assert digest(path) == before and sorted(tmp_path.iterdir()) == files


def test_record_command_leaves_a_supported_database_unchanged(tmp_path, source, capsys):
    path = file_db(tmp_path, source)
    before = digest(path)
    assert fr.main(["record", "--db", str(path), "--import-id", "1", "--date", "2026-11-13",
                    "--team", "AAA", "--season", "2025"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["coverage_status"] == "incomplete" and out["complete_record"] is None
    subset = out["source_subset"]
    assert (subset["W"], subset["L"], subset["T"], subset["assessment_id"]) == (1, 1, 1, 1)
    assert digest(path) == before
    assert sorted(tmp_path.iterdir()) == sorted([path, tmp_path / "games.csv"])
