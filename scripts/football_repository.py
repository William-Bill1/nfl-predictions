"""Football-repository pilot (docs/SPREAD_ENGINE_AND_DATA_REPOSITORY.md, section 6.1).

A small local SQLite database built from the tab-separated
data_files/nfl_games_historical.csv. It answers "for a selected team and NFL
season, what is its outright W/L/T record outdoors on natural grass?", and
any other question that ordinary SQL can ask of the same rows.

Source identity and completion are stored separately:
  source_imports          one row per distinct source (SHA-256 of the exact
                          bytes + transformation version). Re-importing
                          identical bytes returns the existing import.
  game_observations       one row per (import_id, game_id): identity, teams,
                          schedule, raw score text and game-specific
                          conditions (raw roof/surface plus derived categories).
  completion_assessments  one row per (import_id, Toronto evaluation date,
                          completion rule version). Repeating one is a no-op;
                          earlier assessments are never changed.
  game_completions        one row per (assessment_id, game_id): final or
                          excluded, with the helper's reason.
  team_game_results       view: two rows per game that is final in an
                          assessment, one from each team's perspective.

Every query selects one assessment (which also fixes the import), so results
are never combined across assessments and never depend on today's date.

Completion reuses bet_journal.completion_evidence, the conservative
Toronto-day rule the pipeline also uses through final_score_mask. Its rules
are not copied here. COMPLETION_RULE_VERSION names the rule an assessment
used; change it by hand whenever completion_evidence's semantics change, so
older assessments stay distinguishable and reproducible.

Roof and surface are retrospective source records for each game. They are not
confirmed pregame knowledge or a confirmed roof state, and they are never
inferred from a team's usual stadium.

Usage: see docs/FOOTBALL_REPOSITORY_PILOT.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sqlite3
import sys
from datetime import date, datetime, time, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import bet_journal as bj  # noqa: E402
import ontario_spreads as on  # noqa: E402

TRANSFORMATION_VERSION = "football-repo-pilot-2"
# Assigned by hand. Bump it whenever bet_journal.completion_evidence's
# semantics change (what counts as final, the same-day rule, score checks).
COMPLETION_RULE_VERSION = "bet_journal.completion_evidence@2026-11-13"
DEFAULT_SOURCE = ROOT / "data_files" / "nfl_games_historical.csv"
CONDITIONS_BASIS = ("retrospective source record; not confirmed pregame knowledge "
                    "or confirmed roof state")

ROOF_CATEGORIES = ("outdoors", "open", "closed", "dome")
NATURAL_GRASS = "natural_grass"
ARTIFICIAL = "artificial"
UNKNOWN = "unknown"
# Explicit list only: a surface that is not on it stays unknown, and is never
# assumed to be grass because it is not called turf.
ARTIFICIAL_SURFACES = ("fieldturf", "astroturf", "a_turf", "sportturf", "matrixturf")

REQUIRED_COLUMNS = ("game_id", "season", "game_type", "week", "gameday", "gametime",
                    "away_team", "away_score", "home_team", "home_score", "location",
                    "result", "total", "overtime", "roof", "surface", "stadium_id", "stadium")
# Source fields the completion helper reads; kept as raw text so any later
# assessment sees exactly what was imported.
COMPLETION_FIELDS = ("game_id", "gameday", "home_score", "away_score", "result", "total", "overtime")

SCHEMA = """
CREATE TABLE IF NOT EXISTS source_imports (
    import_id INTEGER PRIMARY KEY,
    source_path TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    imported_at_utc TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    transformation_version TEXT NOT NULL,
    UNIQUE (source_sha256, transformation_version)
);
CREATE TABLE IF NOT EXISTS game_observations (
    import_id INTEGER NOT NULL REFERENCES source_imports(import_id),
    game_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    game_type TEXT NOT NULL,
    week INTEGER,
    gameday TEXT,
    gametime TEXT,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_score_raw TEXT,
    away_score_raw TEXT,
    result_raw TEXT,
    total_raw TEXT,
    overtime_raw TEXT,
    location TEXT,
    stadium_id TEXT,
    stadium TEXT,
    roof_raw TEXT,
    surface_raw TEXT,
    roof_category TEXT NOT NULL,
    surface_category TEXT NOT NULL,
    PRIMARY KEY (import_id, game_id)
);
CREATE TABLE IF NOT EXISTS completion_assessments (
    assessment_id INTEGER PRIMARY KEY,
    import_id INTEGER NOT NULL REFERENCES source_imports(import_id),
    evaluation_toronto_date TEXT NOT NULL,
    completion_rule_version TEXT NOT NULL,
    assessed_at_utc TEXT NOT NULL,
    UNIQUE (import_id, evaluation_toronto_date, completion_rule_version)
);
CREATE TABLE IF NOT EXISTS game_completions (
    assessment_id INTEGER NOT NULL REFERENCES completion_assessments(assessment_id),
    game_id TEXT NOT NULL,
    completion_status TEXT NOT NULL CHECK (completion_status IN ('final', 'excluded')),
    completion_reason TEXT NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    PRIMARY KEY (assessment_id, game_id)
);
CREATE VIEW IF NOT EXISTS team_game_results AS
WITH f AS (
    SELECT a.assessment_id, a.evaluation_toronto_date, a.completion_rule_version,
           o.*, c.home_score, c.away_score
    FROM completion_assessments a
    JOIN game_completions c ON c.assessment_id = a.assessment_id
    JOIN game_observations o ON o.import_id = a.import_id AND o.game_id = c.game_id
    WHERE c.completion_status = 'final')
SELECT assessment_id, evaluation_toronto_date, completion_rule_version, import_id, game_id,
       season, game_type, week, gameday, home_team AS team, away_team AS opponent,
       1 AS is_home, location, roof_raw, surface_raw, roof_category, surface_category,
       home_score AS team_score, away_score AS opponent_score,
       CASE WHEN home_score > away_score THEN 'W'
            WHEN home_score < away_score THEN 'L' ELSE 'T' END AS outcome
FROM f
UNION ALL
SELECT assessment_id, evaluation_toronto_date, completion_rule_version, import_id, game_id,
       season, game_type, week, gameday, away_team, home_team,
       0, location, roof_raw, surface_raw, roof_category, surface_category,
       away_score, home_score,
       CASE WHEN away_score > home_score THEN 'W'
            WHEN away_score < home_score THEN 'L' ELSE 'T' END
FROM f;
"""


class ImportRefused(ValueError):
    """The source was refused; nothing from it was written."""


class SchemaRefused(RuntimeError):
    """The database file has an unsupported schema; it was not changed."""


def roof_category(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    return value if value in ROOF_CATEGORIES else UNKNOWN


def surface_category(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    if value == "grass":
        return NATURAL_GRASS
    return ARTIFICIAL if value in ARTIFICIAL_SURFACES else UNKNOWN


def _schema_objects(conn: sqlite3.Connection) -> set:
    return set(conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master").fetchall())


def _expected_schema() -> set:
    ref = sqlite3.connect(":memory:")
    try:
        ref.executescript(SCHEMA)
        return _schema_objects(ref)
    finally:
        ref.close()


def _read_only_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def schema_state(path: Path) -> str:
    """'empty' for a missing or empty database, 'current' for exactly this
    pilot's schema; raises SchemaRefused otherwise. Reads through a read-only
    connection, so a refused file is left byte-identical."""
    if not path.exists():
        return "empty"
    conn = _read_only_connection(path)
    try:
        found = _schema_objects(conn)
    finally:
        conn.close()
    if not found:
        return "empty"
    expected = _expected_schema()
    if found == expected:
        return "current"
    missing = sorted(name for _, name, _, _ in expected - found)
    unexpected = sorted(name for _, name, _, _ in found - expected)
    raise SchemaRefused(
        f"{path}: unsupported database schema (old, partial or incompatible); refusing to open it. "
        f"Missing or different: {', '.join(missing) or 'none'}; unexpected or different: "
        f"{', '.join(unexpected) or 'none'}. Use a new database file; this pilot has no migrations.")


def connect(db_path) -> sqlite3.Connection:
    """Open a pilot database for import/assessment. A missing or empty file is
    initialized with the schema; a file with exactly the current schema is
    opened as is; anything else raises SchemaRefused before any change."""
    path = Path(db_path)
    state = schema_state(path)
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    if state == "empty":
        conn.executescript(SCHEMA)
    return conn


_READ_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                 sqlite3.SQLITE_RECURSIVE}


def _read_only_authorizer(action, arg1, arg2, db_name, trigger):
    """SQLite authorizer: allow only reading. ATTACH, DETACH, PRAGMA, writes,
    schema changes and transactions are denied when the statement is
    prepared, before anything runs."""
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() == "load_extension":
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK if action in _READ_ACTIONS else sqlite3.SQLITE_DENY


def open_read_only(db_path, *, require_current_schema: bool = True) -> sqlite3.Connection:
    """A read-only connection: mode=ro, query_only, and an authorizer that
    permits only SELECT-type reads. Never initializes or changes the schema.
    By default the file must have exactly the current pilot schema."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"database not found: {path}")
    if require_current_schema and schema_state(path) != "current":
        raise SchemaRefused(f"{path}: not an initialized pilot database (no tables)")
    conn = _read_only_connection(path)
    conn.execute("PRAGMA query_only = ON")
    conn.set_authorizer(_read_only_authorizer)
    return conn


def _utc(now: datetime | None) -> str:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(timezone.utc).isoformat(timespec="seconds")


def _season(text: str, game_id: str) -> int:
    value = bj._int_field(text)  # the same integer parsing the completion rule uses
    if value is None or not 1900 <= value <= 2200:
        raise ImportRefused(f"{game_id}: season {text!r} is not a valid NFL season year")
    return value


def import_source(conn: sqlite3.Connection, source_path=DEFAULT_SOURCE, *,
                  now: datetime | None = None) -> tuple[int, bool]:
    """Import one tab-separated schedule file. Returns (import_id, created).

    Identical source bytes under the same transformation version return the
    existing import (created=False) and write nothing. Any refusal (missing
    column, duplicate game ID, invalid required identity) rolls back the
    whole import. The source file is only read. No completion is assessed
    here; see assess_completion."""
    imported_at = _utc(now)
    source_path = Path(source_path).resolve()
    data = source_path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    existing = conn.execute(
        "SELECT import_id FROM source_imports WHERE source_sha256 = ? AND transformation_version = ?",
        (sha, TRANSFORMATION_VERSION)).fetchone()
    if existing:
        return existing[0], False

    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""), delimiter="\t")
    missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise ImportRefused(f"source is missing required columns: {', '.join(missing)}")
    rows = list(reader)

    conn.execute("BEGIN IMMEDIATE")
    try:
        import_id = conn.execute(
            "INSERT INTO source_imports (source_path, source_sha256, imported_at_utc, row_count, "
            "transformation_version) VALUES (?, ?, ?, ?, ?)",
            (str(source_path), sha, imported_at, len(rows), TRANSFORMATION_VERSION)).lastrowid
        for row in rows:
            game_id = (row["game_id"] or "").strip()
            home, away = (row["home_team"] or "").strip(), (row["away_team"] or "").strip()
            game_type = (row["game_type"] or "").strip()
            if not game_id or game_id != row["game_id"]:
                raise ImportRefused(f"invalid game_id {row['game_id']!r}")
            if not home or not away or home == away:
                raise ImportRefused(f"{game_id}: invalid teams home={row['home_team']!r} "
                                    f"away={row['away_team']!r}")
            if not game_type:
                raise ImportRefused(f"{game_id}: missing game_type")
            season = _season(row["season"], game_id)
            try:
                conn.execute(
                    "INSERT INTO game_observations VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (import_id, game_id, season, game_type, bj._int_field(row["week"]),
                     row["gameday"], row["gametime"] or None, home, away,
                     row["home_score"], row["away_score"], row["result"], row["total"],
                     row["overtime"], row["location"] or None, row["stadium_id"] or None,
                     row["stadium"] or None, row["roof"], row["surface"],
                     roof_category(row["roof"]), surface_category(row["surface"])))
            except sqlite3.IntegrityError as exc:
                if "UNIQUE" in str(exc) or "PRIMARY KEY" in str(exc):
                    raise ImportRefused(f"duplicate game_id {game_id!r} in source") from exc
                raise
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return import_id, True


def assess_completion(conn: sqlite3.Connection, import_id: int, evaluation_date: date, *,
                      now: datetime | None = None) -> tuple[int, bool]:
    """Assess completion for every game of one import as of a Toronto date.
    Returns (assessment_id, created). Repeating the same (import, date, rule
    version) returns the existing assessment and writes nothing."""
    if not isinstance(evaluation_date, date) or isinstance(evaluation_date, datetime):
        raise TypeError("evaluation_date must be a datetime.date (a Toronto calendar date)")
    assessed_at = _utc(now)
    if conn.execute("SELECT 1 FROM source_imports WHERE import_id = ?", (import_id,)).fetchone() is None:
        raise ValueError(f"no import with import_id {import_id}")
    key = (import_id, evaluation_date.isoformat(), COMPLETION_RULE_VERSION)
    existing = conn.execute(
        "SELECT assessment_id FROM completion_assessments WHERE import_id = ? AND "
        "evaluation_toronto_date = ? AND completion_rule_version = ?", key).fetchone()
    if existing:
        return existing[0], False

    raw = conn.execute(
        "SELECT game_id, gameday, home_score_raw, away_score_raw, result_raw, total_raw, overtime_raw "
        "FROM game_observations WHERE import_id = ?", (import_id,)).fetchall()
    schedule = pd.DataFrame(raw, columns=list(COMPLETION_FIELDS), dtype=object)
    # Midday in Toronto on the evaluation date: the helper uses only its Toronto date.
    as_of = datetime.combine(evaluation_date, time(12), tzinfo=on.TORONTO)

    conn.execute("BEGIN IMMEDIATE")
    try:
        assessment_id = conn.execute(
            "INSERT INTO completion_assessments (import_id, evaluation_toronto_date, "
            "completion_rule_version, assessed_at_utc) VALUES (?, ?, ?, ?)",
            key + (assessed_at,)).lastrowid
        for (game_id,) in conn.execute("SELECT game_id FROM game_observations WHERE import_id = ?",
                                       (import_id,)).fetchall():
            evidence, reason = bj.completion_evidence(schedule, game_id, as_of)
            conn.execute(
                "INSERT INTO game_completions VALUES (?, ?, ?, ?, ?, ?)",
                (assessment_id, game_id, "final" if evidence else "excluded",
                 reason or "conservative final evidence",
                 evidence["home_score"] if evidence else None,
                 evidence["away_score"] if evidence else None))
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return assessment_id, True


def find_assessment(conn: sqlite3.Connection, import_id: int, evaluation_date: date,
                    rule_version: str = COMPLETION_RULE_VERSION) -> int:
    """The assessment_id for exactly this import, date and rule version. Never
    creates one and never falls back to another date."""
    row = conn.execute(
        "SELECT assessment_id FROM completion_assessments WHERE import_id = ? AND "
        "evaluation_toronto_date = ? AND completion_rule_version = ?",
        (import_id, evaluation_date.isoformat(), rule_version)).fetchone()
    if row is None:
        raise LookupError(f"no completion assessment for import {import_id}, Toronto date "
                          f"{evaluation_date.isoformat()}, rule {rule_version!r}; run 'assess' first")
    return row[0]


def team_record(conn: sqlite3.Connection, assessment_id: int, team: str, season: int, *,
                game_type: str = "REG", roofs=("outdoors",), surfaces=(NATURAL_GRASS,)) -> dict:
    """Outright W/L/T for one team and NFL season within one completion assessment.

    Defaults: regular season, roof 'outdoors' only (pass roofs=("outdoors",
    "open") to include open retractable roofs), natural grass. Every game of
    the team/season/game type falls in exactly one of: included, unfinished or
    unverifiable, definitely excluded by a known condition, or eligibility
    undetermined because a required condition is unknown."""
    meta = conn.execute("SELECT import_id, evaluation_toronto_date, completion_rule_version "
                        "FROM completion_assessments WHERE assessment_id = ?", (assessment_id,)).fetchone()
    if meta is None:
        raise LookupError(f"no completion assessment with assessment_id {assessment_id}")
    import_id, evaluation_date, rule_version = meta
    rows = conn.execute(
        "SELECT o.game_id, o.home_team, o.roof_category, o.surface_category, c.completion_status, "
        "c.completion_reason, c.home_score, c.away_score FROM game_observations o "
        "JOIN game_completions c ON c.game_id = o.game_id AND c.assessment_id = ? "
        "WHERE o.import_id = ? AND o.season = ? AND o.game_type = ? AND (o.home_team = ? OR o.away_team = ?) "
        "ORDER BY o.gameday, o.game_id",
        (assessment_id, import_id, season, game_type, team, team)).fetchall()

    included, unfinished, by_conditions, undetermined = [], [], [], []
    quality = {"roof_unknown_only": [], "surface_unknown_only": [], "roof_and_surface_unknown": []}
    for game_id, home, roof, surface, status, reason, hs, as_ in rows:
        if roof == UNKNOWN and surface == UNKNOWN:
            quality["roof_and_surface_unknown"].append(game_id)
        elif roof == UNKNOWN:
            quality["roof_unknown_only"].append(game_id)
        elif surface == UNKNOWN:
            quality["surface_unknown_only"].append(game_id)
        if status != "final":
            unfinished.append({"game_id": game_id, "reason": reason})
            continue
        known_fail = (roof != UNKNOWN and roof not in roofs) or (surface != UNKNOWN and surface not in surfaces)
        if known_fail:
            by_conditions.append({"game_id": game_id, "roof": roof, "surface": surface})
        elif roof == UNKNOWN or surface == UNKNOWN:
            undetermined.append({"game_id": game_id, "roof": roof, "surface": surface})
        else:
            mine, theirs = (hs, as_) if home == team else (as_, hs)
            included.append((game_id, "W" if mine > theirs else "L" if mine < theirs else "T"))

    w = sum(1 for _, o in included if o == "W")
    lost = sum(1 for _, o in included if o == "L")
    t = sum(1 for _, o in included if o == "T")
    return {
        "assessment_id": assessment_id, "import_id": import_id,
        "evaluation_toronto_date": evaluation_date, "completion_rule_version": rule_version,
        "team": team, "season": season, "game_type": game_type,
        "roof_categories": list(roofs), "surface_categories": list(surfaces),
        "W": w, "L": lost, "T": t, "games": len(included), "W:L": f"{w}:{lost}",
        "win_pct_excluding_ties": (w / (w + lost)) if (w + lost) else None,
        "nfl_standings_pct": ((w + 0.5 * t) / len(included)) if included else None,
        "included_game_ids": [g for g, _ in included],
        "excluded_unfinished_or_unverifiable": unfinished,
        "excluded_by_known_conditions": by_conditions,
        "eligibility_undetermined_unknown_condition": undetermined,
        "games_in_scope": len(rows),
        "data_quality_missing_conditions": quality,
        "conditions_basis": CONDITIONS_BASIS,
    }


def run_sql(db_path, sql: str, params=()) -> tuple[list[str], list[tuple]]:
    """Run one ad hoc read-only query. Anything other than reading (ATTACH,
    writes, schema changes, PRAGMA) raises sqlite3.DatabaseError "not
    authorized" before it runs. The schema is not checked here, so a retained
    older pilot database can still be read safely."""
    conn = open_read_only(db_path, require_current_schema=False)
    try:
        cur = conn.execute(sql, params)
        return [d[0] for d in cur.description or ()], cur.fetchall()
    finally:
        conn.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("import", help="import the schedule file (no completion assessment)")
    pi.add_argument("--db", required=True)
    pi.add_argument("--source", default=str(DEFAULT_SOURCE))
    pa = sub.add_parser("assess", help="assess completion for one import as of a Toronto date")
    pa.add_argument("--db", required=True)
    pa.add_argument("--import-id", type=int, required=True)
    pa.add_argument("--date", type=date.fromisoformat, required=True, help="Toronto date, YYYY-MM-DD")
    pr = sub.add_parser("record", help="team/season W/L/T by conditions, for one assessment")
    pr.add_argument("--db", required=True)
    pr.add_argument("--import-id", type=int, required=True)
    pr.add_argument("--date", type=date.fromisoformat, required=True,
                    help="the assessment's Toronto evaluation date, YYYY-MM-DD")
    pr.add_argument("--rule-version", default=COMPLETION_RULE_VERSION)
    pr.add_argument("--team", required=True)
    pr.add_argument("--season", type=int, required=True)
    pr.add_argument("--game-type", default="REG")
    pr.add_argument("--include-open", action="store_true", help="also count open retractable roofs")
    ps_ = sub.add_parser("sql", help="run a read-only SQL query")
    ps_.add_argument("--db", required=True)
    ps_.add_argument("query")
    a = p.parse_args(argv)
    if a.cmd == "sql":
        try:
            cols, rows = run_sql(a.db, a.query)
        except (sqlite3.DatabaseError, FileNotFoundError) as exc:
            print(f"error: {exc} (the sql command only runs read-only queries)", file=sys.stderr)
            return 2
        print("\t".join(cols))
        for r in rows:
            print("\t".join("" if v is None else str(v) for v in r))
        return 0
    if a.cmd != "import" and not Path(a.db).exists():
        p.error(f"database not found: {a.db}")
    try:
        # record only reads: it never initializes or changes the database.
        conn = open_read_only(a.db) if a.cmd == "record" else connect(a.db)
    except SchemaRefused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        if a.cmd == "import":
            import_id, created = import_source(conn, a.source)
            out = {"import_id": import_id, "created": created, "source_import": conn.execute(
                "SELECT * FROM source_imports WHERE import_id = ?", (import_id,)).fetchone()}
        elif a.cmd == "assess":
            assessment_id, created = assess_completion(conn, a.import_id, a.date)
            out = {"assessment_id": assessment_id, "created": created, "assessment": conn.execute(
                "SELECT * FROM completion_assessments WHERE assessment_id = ?", (assessment_id,)).fetchone(),
                "status_counts": dict(conn.execute(
                    "SELECT completion_status, COUNT(*) FROM game_completions WHERE assessment_id = ? "
                    "GROUP BY 1", (assessment_id,)).fetchall())}
        else:
            try:
                assessment_id = find_assessment(conn, a.import_id, a.date, a.rule_version)
            except LookupError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            roofs = ("outdoors", "open") if a.include_open else ("outdoors",)
            out = team_record(conn, assessment_id, a.team, a.season, game_type=a.game_type, roofs=roofs)
    finally:
        conn.close()
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
