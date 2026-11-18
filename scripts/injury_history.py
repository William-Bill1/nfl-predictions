"""Injury-history research pilot: retained injury captures in a separate SQLite file.

Research only. Nothing in the app, the probability engine or the pipeline reads
this database. It imports explicitly selected, already-retained capture files
written by ``injury_reports.py`` (``data_files/injury_reports/captures/``) and
answers read-only questions such as "what changed for KC between Wednesday's and
Sunday's captures?".

Tables
  injury_captures            one row per capture (capture_id + content checksum):
                             file hashes, capture times, slot, outcome, source
                             attribution, HTTP metadata and provenance.
  injury_capture_feed_teams  the team blocks the source listed, with entry counts.
  injury_capture_teams       coverage state of each in-scope team (from the capture).
  injury_capture_games       the scope week's games and their state at capture time,
                             with the kickoff's provenance (kickoff_basis and the
                             source schedule's own kickoff; see KICKOFF_BASES).
  injury_observations        one row per listed entry, with every stored field
                             and the exact row JSON.

What the source is: ESPN's general injury feed. Its status mixes game-designation
words, roster statuses and "Active" news notes; it is not the official NFL
report and has no practice participation. The per-entry date's meaning is
unverified; it is stored as supplied and never used as an update time.
Every usable retained capture lists exactly 25 entries per team, which suggests
a per-team cap (not a confirmed provider contract), so a player missing from a
capture may just have been pushed out. See docs/INJURY_HISTORY_PILOT.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import football_repository as fr  # noqa: E402  (read-only connection + authorizer)
import injury_reports as ir  # noqa: E402  (capture validator, field definitions)
import ontario_spreads as on  # noqa: E402  (slot rules)

TRANSFORMATION_VERSION = "injury-history-pilot-1"
CHECKSUM_FIELD = ir.CHECKSUM_FIELD

# Reported fields compared between two observations of the same entry. game_id and
# game_state describe the capture's scope week, not the report, so they are shown
# as context rather than compared.
COMPARED_FIELDS = ("player_name", "position", "source_status", "status_type_name",
                   "status_type_description", "status_type_abbreviation", "injury_type",
                   "injury_location", "injury_detail", "injury_side", "short_comment",
                   "source_return_date", "source_entry_date_raw")

ABSENCE_MEANING = ("not listed for this team in the later capture. This does NOT mean recovered, "
                   "healthy, active or cleared: the source may have dropped the entry (every "
                   "usable retained capture lists exactly 25 entries per team, suggesting a cap), "
                   "or the player may be listed for another team")
FRESHNESS = "not checked"
LIMITATIONS = (
    "ESPN's general injury feed, not the official NFL injury report; no practice participation",
    "'Active' entries are mostly news notes, not official availability designations",
    "every team being present does not mean complete injury lists or complete rosters",
    "a missing entry may reflect the source's apparent per-team entry cap (unconfirmed)",
    "source entry dates are stored as supplied; their meaning is unverified",
    "no betting or probability impact is claimed or computed",
)

SCHEMA_V1 = """
CREATE TABLE injury_captures (
    capture_import_id          INTEGER PRIMARY KEY,
    capture_id                 TEXT NOT NULL,
    payload_sha256             TEXT NOT NULL,
    transformation_version     TEXT NOT NULL,
    file_name                  TEXT NOT NULL,
    source_path                TEXT NOT NULL,
    file_sha256                TEXT NOT NULL,
    file_bytes                 INTEGER NOT NULL,
    imported_at_utc            TEXT NOT NULL,
    request_started_at_utc     TEXT NOT NULL,
    received_at_utc            TEXT NOT NULL,
    slot_name                  TEXT NOT NULL,
    slot_id                    TEXT NOT NULL,
    slot_status                TEXT NOT NULL,
    slot_fills                 INTEGER NOT NULL,
    slot_intended_utc          TEXT,
    slot_delay_minutes         REAL,
    scope_season               INTEGER,
    scope_week                 INTEGER,
    outcome                    TEXT NOT NULL,
    usable                     INTEGER NOT NULL,
    source_name                TEXT NOT NULL,
    source_official_nfl_report INTEGER,
    source_json                TEXT NOT NULL,
    http_status                INTEGER,
    http_date                  TEXT,
    http_last_modified         TEXT,
    response_bytes             INTEGER,
    response_error_json        TEXT,
    teams_in_feed              INTEGER NOT NULL,
    player_rows                INTEGER NOT NULL,
    malformed_entries_json     TEXT NOT NULL,
    unmatched_source_teams_json TEXT NOT NULL,
    source_entry_dates_json    TEXT NOT NULL,
    schedule_path              TEXT,
    schedule_sha256            TEXT,
    code_revision              TEXT,
    code_dirty                 INTEGER,
    UNIQUE (capture_id, transformation_version)
);
CREATE TABLE injury_capture_feed_teams (
    capture_import_id INTEGER NOT NULL REFERENCES injury_captures(capture_import_id),
    team_index        INTEGER NOT NULL,
    source_team_name  TEXT,
    source_team_id    TEXT,
    team              TEXT,
    entries           INTEGER NOT NULL,
    PRIMARY KEY (capture_import_id, team_index)
);
CREATE TABLE injury_capture_teams (
    capture_import_id INTEGER NOT NULL REFERENCES injury_captures(capture_import_id),
    team              TEXT NOT NULL,
    coverage_state    TEXT NOT NULL,
    coverage_rows     INTEGER NOT NULL,
    PRIMARY KEY (capture_import_id, team)
);
CREATE TABLE injury_capture_games (
    capture_import_id INTEGER NOT NULL REFERENCES injury_captures(capture_import_id),
    game_id           TEXT NOT NULL,
    season            INTEGER,
    week              INTEGER,
    home_team         TEXT,
    away_team         TEXT,
    kickoff_utc       TEXT,
    state             TEXT,
    PRIMARY KEY (capture_import_id, game_id)
);
CREATE TABLE injury_observations (
    capture_import_id        INTEGER NOT NULL REFERENCES injury_captures(capture_import_id),
    entry_index              INTEGER NOT NULL,
    source_team_name         TEXT,
    team                     TEXT,
    player_name              TEXT NOT NULL,
    player_id                TEXT,
    player_id_source         TEXT,
    position                 TEXT,
    source_status            TEXT,
    status_class             TEXT NOT NULL,
    status_type_name         TEXT,
    status_type_description  TEXT,
    status_type_abbreviation TEXT,
    injury_type              TEXT,
    injury_location          TEXT,
    injury_detail            TEXT,
    injury_side              TEXT,
    short_comment            TEXT,
    source_return_date       TEXT,
    practice_participation   TEXT,
    source_entry_date_raw    TEXT,
    source_entry_date_utc    TEXT,
    game_id                  TEXT,
    game_state               TEXT,
    raw_row_json             TEXT NOT NULL,
    PRIMARY KEY (capture_import_id, entry_index)
);
CREATE INDEX injury_observations_team_player ON injury_observations (team, player_id);
"""

# Kickoff provenance (2026 Week 11 reviewed corrections). Added to the frozen
# first schema by ALTER, so a new database and an upgraded copy of a first-schema
# database end with identical schema text. Rows carried over by `upgrade` keep
# their stored values and get the pre-provenance defaults: they were imported
# before provenance was stored and are never reinterpreted.
PRE_PROVENANCE = "imported_before_kickoff_provenance"
KICKOFF_BASES = {
    "source": "the source schedule's kickoff (game not under kickoff review)",
    "reviewed_correction": "reviewed correction applied during collection; source_kickoff_utc is the source's time",
    "source_already_corrected": "the source schedule already had the reviewed kickoff",
    "capture_without_kickoff_review": "capture made before kickoff review existed; kickoff as captured",
    PRE_PROVENANCE: "row imported before kickoff provenance was stored; kickoff as captured",
}
DOCUMENTED_REVIEW_BASES = ("reviewed_correction", "source_already_corrected")
PROVENANCE_DDL = f"""
ALTER TABLE injury_captures ADD COLUMN schedule_provenance TEXT NOT NULL DEFAULT '{PRE_PROVENANCE}';
ALTER TABLE injury_captures ADD COLUMN schedule_sha256_basis TEXT;
ALTER TABLE injury_captures ADD COLUMN schedule_kickoff_review_json TEXT;
ALTER TABLE injury_capture_games ADD COLUMN source_kickoff_utc TEXT;
ALTER TABLE injury_capture_games ADD COLUMN kickoff_basis TEXT NOT NULL DEFAULT '{PRE_PROVENANCE}';
"""
SCHEMA = SCHEMA_V1 + PROVENANCE_DDL


class ImportRefused(ValueError):
    """The selected captures can't be imported. Nothing was written."""


class CompareRefused(ValueError):
    """The comparison can't be made as requested."""


# --------------------------------------------------------------------------
# Database files (same exact-schema rule as the football pilot)
# --------------------------------------------------------------------------

def _expected_schema(script: str = SCHEMA) -> set:
    ref = sqlite3.connect(":memory:")
    try:
        ref.executescript(script)
        return fr._schema_objects(ref)
    finally:
        ref.close()


def schema_state(path: Path) -> str:
    """'empty' (missing or no objects), 'current', or 'legacy' (exactly the
    first schema, without kickoff provenance: readable, but imports need an
    explicit `upgrade` to a new file). Anything else raises fr.SchemaRefused;
    the file is only read, through a read-only connection."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return "empty"
    conn = fr._read_only_connection(path)
    try:
        found = fr._schema_objects(conn)
    except sqlite3.DatabaseError as exc:
        raise fr.SchemaRefused(f"{path}: not a readable SQLite database ({exc})") from None
    finally:
        conn.close()
    if not found:
        return "empty"
    if found == _expected_schema():
        return "current"
    if found == _expected_schema(SCHEMA_V1):
        return "legacy"
    raise fr.SchemaRefused(f"{path}: not an injury-history pilot database (schema differs); "
                           "use a new database file - this pilot has no migrations")


def connect_for_import(db_path) -> sqlite3.Connection:
    path = Path(db_path)
    state = schema_state(path)
    if state == "legacy":
        raise fr.SchemaRefused(f"{path}: injury-history database without kickoff provenance; importing "
                               "would lose it. Run `upgrade --from <this file> --to <new file>` and "
                               "import into the new file")
    conn = sqlite3.connect(path, isolation_level=None)
    if state == "empty":
        conn.executescript(SCHEMA)
    return conn


def open_read_only(db_path) -> sqlite3.Connection:
    """The football pilot's guarded connection (mode=ro, query_only, read-only
    authorizer), after checking this pilot's schema (current, or legacy without
    kickoff provenance; see has_kickoff_provenance)."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"database not found: {path}")
    if schema_state(path) not in ("current", "legacy"):
        raise fr.SchemaRefused(f"{path}: not an initialized injury-history database")
    return fr.open_read_only(path, require_current_schema=False)


def has_kickoff_provenance(conn: sqlite3.Connection) -> bool:
    """True when this connection's database stores kickoff provenance columns."""
    return "kickoff_basis" in {r[1] for r in conn.execute("PRAGMA table_info(injury_capture_games)")}


# --------------------------------------------------------------------------
# Upgrade (explicit, to a new file; the original is only read)
# --------------------------------------------------------------------------

_TABLE_KEYS = {"injury_captures": "capture_import_id",
               "injury_capture_feed_teams": "capture_import_id, team_index",
               "injury_capture_teams": "capture_import_id, team",
               "injury_capture_games": "capture_import_id, game_id",
               "injury_observations": "capture_import_id, entry_index"}


def _table_rows(conn, table: str, columns: list[str]) -> list[tuple]:
    return conn.execute(f"SELECT {', '.join(columns)} FROM {table} ORDER BY {_TABLE_KEYS[table]}").fetchall()


def upgrade_database(src, dst) -> dict:
    """Copy a legacy (first-schema) database to the NEW file `dst` and add the
    kickoff-provenance columns there. `src` is opened read-only and never
    changed. Every carried-over capture and game row gets the
    imported_before_kickoff_provenance basis; no stored value is changed or
    reinterpreted. Verifies every original column of every row and returns the
    row counts. On any failure `dst` is removed and nothing else is touched."""
    src, dst = Path(src), Path(dst)
    if schema_state(src) != "legacy":
        raise fr.SchemaRefused(f"{src}: only a legacy injury-history database (first schema) can be upgraded")
    if dst.exists():
        raise fr.SchemaRefused(f"{dst}: upgrade writes a new file; it already exists")
    original = fr._read_only_connection(src)
    target = None
    try:
        tables = {t: [r[1] for r in original.execute(f"PRAGMA table_info({t})")] for t in _TABLE_KEYS}
        before = {t: _table_rows(original, t, cols) for t, cols in tables.items()}
        target = sqlite3.connect(dst, isolation_level=None)
        original.backup(target)
        target.executescript("BEGIN IMMEDIATE;" + PROVENANCE_DDL + "COMMIT;")
        if fr._schema_objects(target) != _expected_schema():
            raise fr.SchemaRefused(f"{dst}: upgraded schema differs from the current schema")
        for t, cols in tables.items():
            if _table_rows(target, t, cols) != before[t]:
                raise fr.SchemaRefused(f"{dst}: rows of {t} changed during upgrade")
        for t, col in (("injury_captures", "schedule_provenance"), ("injury_capture_games", "kickoff_basis")):
            if target.execute(f"SELECT COUNT(*) FROM {t} WHERE {col} != ?", (PRE_PROVENANCE,)).fetchone()[0]:
                raise fr.SchemaRefused(f"{dst}: carried-over {t} rows lack the pre-provenance basis")
        for conn in (original, target):
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise fr.SchemaRefused("integrity check failed")
        counts = {t: len(rows) for t, rows in before.items()}
    except BaseException:
        if target is not None:
            target.close()
            target = None
        dst.unlink(missing_ok=True)
        raise
    finally:
        original.close()
        if target is not None:
            target.close()
    return {"from": str(src), "to": str(dst), "rows": counts, "carried_over_basis": PRE_PROVENANCE}


# --------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------

def _utc(now: datetime | None) -> str:
    return fr._utc(now)


def _dumps(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _read_capture(path: Path) -> tuple[dict, bytes]:
    path = Path(path)
    if not path.is_file():
        raise ImportRefused(f"{path}: not a capture file (select capture files explicitly)")
    data = path.read_bytes()
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ImportRefused(f"{path}: not valid UTF-8 JSON ({exc})") from None
    try:
        ir.validate_capture(doc, str(path))
    except ir.ValidationError as exc:
        raise ImportRefused(str(exc)) from None
    if path.stem != doc["capture_id"]:
        raise ImportRefused(f"{path}: file name doesn't match capture_id {doc['capture_id']!r}")
    return doc, data


def import_captures(conn: sqlite3.Connection, paths, *, now: datetime | None = None) -> list[dict]:
    """Import explicitly selected capture files in one transaction.

    Every file is validated with injury_reports.validate_capture before anything
    is written; any refusal (malformed file, checksum mismatch, a capture_id
    already imported with different content) writes nothing for any file.
    A capture already imported with the same content returns its existing
    capture_import_id with created=False and adds no observations."""
    imported_at = _utc(now)
    loaded = []
    seen = {}
    for p in paths:
        doc, data = _read_capture(Path(p))
        if doc["capture_id"] in seen and seen[doc["capture_id"]] != doc[CHECKSUM_FIELD]:
            raise ImportRefused(f"{p}: capture_id {doc['capture_id']} selected twice with different content")
        seen[doc["capture_id"]] = doc[CHECKSUM_FIELD]
        loaded.append((Path(p).resolve(), doc, data))

    results = []
    conn.execute("BEGIN IMMEDIATE")
    try:
        for path, doc, data in loaded:
            file_sha = hashlib.sha256(data).hexdigest()
            row = conn.execute(
                "SELECT capture_import_id, payload_sha256, file_sha256 FROM injury_captures "
                "WHERE capture_id = ? AND transformation_version = ?",
                (doc["capture_id"], TRANSFORMATION_VERSION)).fetchone()
            if row:
                if row[1] != doc[CHECKSUM_FIELD]:
                    raise ImportRefused(f"{path}: capture {doc['capture_id']} was already imported "
                                        "with different content; refusing")
                results.append({"capture_id": doc["capture_id"], "capture_import_id": row[0],
                                "created": False, "file_sha256": file_sha,
                                "same_file_bytes": row[2] == file_sha})
                continue
            cid = _insert_capture(conn, path, doc, data, file_sha, imported_at)
            results.append({"capture_id": doc["capture_id"], "capture_import_id": cid,
                            "created": True, "file_sha256": file_sha,
                            "observations": len(doc["players"])})
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return results


def _game_provenance(doc: dict) -> tuple[str, dict]:
    """(capture schedule_provenance, {game_id: (kickoff_basis, source_kickoff_utc)})
    for an already validated capture. A legacy capture's kickoffs are kept as
    captured, with no source kickoff claimed."""
    games = doc["scope"].get("games") or []
    if ir.schedule_kind(doc) == "legacy":
        return "capture_without_kickoff_review", {g["game_id"]: ("capture_without_kickoff_review", None)
                                                  for g in games}
    records = {r["game_id"]: r for r in doc["schedule"]["kickoff_review"]["games"]}
    out = {}
    for g in games:
        rec = records.get(g["game_id"])
        if rec is None:
            out[g["game_id"]] = ("source", g.get("kickoff_utc"))
        else:
            out[g["game_id"]] = ("reviewed_correction" if rec["status"] == "applied" else
                                 "source_already_corrected", rec["source_kickoff_utc"])
    return "reviewed", out


def _insert_capture(conn, path: Path, doc: dict, data: bytes, file_sha: str, imported_at: str) -> int:
    slot, resp, cov, src = doc["slot"], doc["response"], doc["coverage"], doc["source"]
    provenance, game_basis = _game_provenance(doc)
    sched = doc["schedule"]
    cid = conn.execute(
        "INSERT INTO injury_captures (capture_id, payload_sha256, transformation_version, file_name, "
        "source_path, file_sha256, file_bytes, imported_at_utc, request_started_at_utc, received_at_utc, "
        "slot_name, slot_id, slot_status, slot_fills, slot_intended_utc, slot_delay_minutes, "
        "scope_season, scope_week, outcome, usable, source_name, source_official_nfl_report, "
        "source_json, http_status, http_date, http_last_modified, response_bytes, response_error_json, "
        "teams_in_feed, player_rows, malformed_entries_json, unmatched_source_teams_json, "
        "source_entry_dates_json, schedule_path, schedule_sha256, code_revision, code_dirty, "
        "schedule_provenance, schedule_sha256_basis, schedule_kickoff_review_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (doc["capture_id"], doc[CHECKSUM_FIELD], TRANSFORMATION_VERSION, path.name, str(path),
         file_sha, len(data), imported_at, doc["request_started_at"], doc["received_at"],
         slot["name"], slot["slot_id"], slot["status"], int(slot["fills_slot"]),
         slot["intended_utc"], slot["delay_minutes"], doc["scope"].get("season"),
         doc["scope"].get("week"), doc["outcome"], int(doc["usable"]), src["name"],
         None if src.get("official_nfl_report") is None else int(src["official_nfl_report"]),
         _dumps(src), resp["http_status"], resp["http_date"], resp["http_last_modified"],
         resp["bytes"], None if resp["error"] is None else _dumps(resp["error"]),
         cov["teams_in_feed"], cov["player_rows"], _dumps(cov["malformed_entries"]),
         _dumps(cov["unmatched_source_teams"]), _dumps(doc["source_entry_dates"]),
         doc["schedule"].get("path"), doc["schedule"].get("sha256"), doc["code_revision"],
         None if doc["code_dirty"] is None else int(doc["code_dirty"]), provenance,
         sched.get("sha256_basis"),
         None if "kickoff_review" not in sched else _dumps(sched["kickoff_review"]))).lastrowid
    for i, t in enumerate(doc["teams"]):
        conn.execute("INSERT INTO injury_capture_feed_teams VALUES (?, ?, ?, ?, ?, ?)",
                     (cid, i, t["source_team_name"], t["source_team_id"], t["team"], t["entries"]))
    for team, c in sorted(cov["in_scope_teams"].items()):
        conn.execute("INSERT INTO injury_capture_teams VALUES (?, ?, ?, ?)",
                     (cid, team, c["state"], c["rows"]))
    for g in doc["scope"].get("games") or []:
        basis, source_kickoff = game_basis[g.get("game_id")]
        conn.execute("INSERT INTO injury_capture_games (capture_import_id, game_id, season, week, home_team, "
                     "away_team, kickoff_utc, state, source_kickoff_utc, kickoff_basis) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (cid, g.get("game_id"), g.get("season"), g.get("week"), g.get("home_team"),
                      g.get("away_team"), g.get("kickoff_utc"), g.get("state"), source_kickoff, basis))
    for i, r in enumerate(doc["players"]):
        st, inj = r["source_status_type"] or {}, r["injury"] or {}
        conn.execute(
            "INSERT INTO injury_observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (cid, i, r["source_team_name"], r["team"], r["player_name"], r["player_id"],
             r["player_id_source"], r["position"], r["source_status"], r["status_class"],
             st.get("name"), st.get("description"), st.get("abbreviation"), inj.get("type"),
             inj.get("location"), inj.get("detail"), inj.get("side"), r["short_comment"],
             r["source_return_date"], r["practice_participation"], r["source_entry_date_raw"],
             r["source_entry_date_utc"], r["game_id"], r["game_state"], _dumps(r)))
    return cid


# --------------------------------------------------------------------------
# List
# --------------------------------------------------------------------------

def _rows(conn, sql, params=()) -> list[dict]:
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def list_captures(conn) -> dict:
    caps = _rows(conn, "SELECT * FROM injury_captures ORDER BY received_at_utc, capture_id")
    out = []
    for c in caps:
        counts = sorted({r[0] for r in conn.execute(
            "SELECT entries FROM injury_capture_feed_teams WHERE capture_import_id = ?",
            (c["capture_import_id"],))})
        dates = json.loads(c["source_entry_dates_json"])
        out.append({
            "capture_import_id": c["capture_import_id"], "capture_id": c["capture_id"],
            "captured_at_utc (response received)": c["received_at_utc"],
            "slot_id": c["slot_id"], "slot_status": c["slot_status"], "fills_slot": bool(c["slot_fills"]),
            "scope": {"season": c["scope_season"], "week": c["scope_week"]},
            "outcome": c["outcome"], "usable": bool(c["usable"]),
            "response_error": json.loads(c["response_error_json"]) if c["response_error_json"] else None,
            "teams_in_feed": c["teams_in_feed"], "observations": c["player_rows"],
            "entries_per_feed_team": counts,
            "source_entry_dates (meaning unverified)": {k: dates.get(k) for k in ("earliest_utc", "latest_utc")},
            "http_last_modified": c["http_last_modified"],
            "file_sha256": c["file_sha256"], "payload_sha256": c["payload_sha256"],
            "provenance": {"schedule_sha256": c["schedule_sha256"], "code_revision": c["code_revision"],
                           "code_dirty": None if c["code_dirty"] is None else bool(c["code_dirty"])},
        })
    return {"captures": out, "slots_in_imported_range": _slot_gaps(caps),
            "freshness": FRESHNESS, "limitations": list(LIMITATIONS)}


def _slot_gaps(caps: list[dict]) -> list[dict]:
    """Wednesday-noon / Sunday-morning slots between the first and last imported
    capture (Toronto dates), with the imported usable captures that fill each.
    Only the imported range is judged; this says nothing about other files."""
    if not caps:
        return []
    days = [ir.parse_utc_us(c["received_at_utc"]).astimezone(on.TORONTO).date() for c in caps]
    filled: dict[str, list[str]] = {}
    for c in caps:
        if c["usable"] and c["slot_fills"]:
            filled.setdefault(c["slot_id"], []).append(c["capture_id"])
    out, day = [], min(days)
    while day <= max(days):
        for rule in on.SLOTS:
            if day.weekday() == rule.weekday:
                sid = f"{day.isoformat()}_{rule.name}"
                out.append({"slot_id": sid, "filled_by_usable_imported_capture": filled.get(sid, []),
                            "status": "filled" if sid in filled else "no usable imported capture"})
        day += timedelta(days=1)
    return out


# --------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------

def _capture(conn, capture_id: str) -> dict:
    rows = _rows(conn, "SELECT * FROM injury_captures WHERE capture_id = ? AND transformation_version = ?",
                 (capture_id, TRANSFORMATION_VERSION))
    if not rows:
        raise CompareRefused(f"capture {capture_id!r} is not imported in this database")
    return rows[0]


def _context(conn, cap: dict, team: str) -> dict:
    cid = cap["capture_import_id"]
    cov = conn.execute("SELECT coverage_state, coverage_rows FROM injury_capture_teams "
                       "WHERE capture_import_id = ? AND team = ?", (cid, team)).fetchone()
    feed = conn.execute("SELECT SUM(entries), COUNT(*) FROM injury_capture_feed_teams "
                        "WHERE capture_import_id = ? AND team = ?", (cid, team)).fetchone()
    counts = sorted({r[0] for r in conn.execute(
        "SELECT entries FROM injury_capture_feed_teams WHERE capture_import_id = ?", (cid,))})
    game = _rows(conn, "SELECT game_id, kickoff_utc, state FROM injury_capture_games WHERE "
                       "capture_import_id = ? AND (home_team = ? OR away_team = ?)", (cid, team, team))
    dates = conn.execute("SELECT MIN(source_entry_date_utc), MAX(source_entry_date_utc) FROM "
                         "injury_observations WHERE capture_import_id = ? AND team = ?", (cid, team)).fetchone()
    return {
        "capture_id": cap["capture_id"], "capture_import_id": cid,
        "captured_at_utc (response received)": cap["received_at_utc"],
        "slot_id": cap["slot_id"], "slot_status": cap["slot_status"], "fills_slot": bool(cap["slot_fills"]),
        "outcome": cap["outcome"], "usable": bool(cap["usable"]),
        "team_coverage_state": cov[0] if cov else ("listed_in_feed_no_game_in_scope" if feed[1] else None),
        "team_entries": feed[0] or 0,
        "entries_per_feed_team_in_capture": counts,
        "team_game_in_scope": game[0] if game else None,
        "team_source_entry_dates (meaning unverified)": {"earliest_utc": dates[0], "latest_utc": dates[1]},
        "http_last_modified": cap["http_last_modified"],
        "file_sha256": cap["file_sha256"],
    }


def _entries(conn, cid: int, team: str) -> list[dict]:
    return _rows(conn, "SELECT * FROM injury_observations WHERE capture_import_id = ? AND team = ? "
                       "ORDER BY entry_index", (cid, team))


def _duplicated_ids(earlier: list[dict], later: list[dict]) -> dict[str, dict]:
    """Source player IDs listed more than once for the team in either capture,
    with the count on each side. Such an ID is unresolved on BOTH sides."""
    out: dict[str, dict] = {}
    for side, entries in (("earlier", earlier), ("later", later)):
        counts: dict[str, int] = {}
        for e in entries:
            if e["player_id"] is not None:
                counts[e["player_id"]] = counts.get(e["player_id"], 0) + 1
        for pid, n in counts.items():
            if n > 1:
                out.setdefault(pid, {})[side] = n
    return out


def _key_entries(entries: list[dict], other: list[dict], duplicated: dict[str, dict]):
    """Identity keys for one side. Returns ({key: entry}, unresolved list).
    Source player IDs are used as given, except an ID duplicated for the team in
    either capture: every row with it, on both sides, is unresolved and never
    compared. An entry without an ID is matched by exact name only when that
    name is unique on both sides and the other side's entry also has no ID; that
    match is labelled inferred. Never infers an ID."""
    keyed, unresolved = {}, []
    for e in entries:
        pid = e["player_id"]
        if pid is None:
            continue
        if pid in duplicated:
            where = ", ".join(f"{n} times in the {s} capture" for s, n in sorted(duplicated[pid].items()))
            unresolved.append({**_brief(e), "reason": f"source player ID {pid} is listed {where} for "
                                                      "this team; ambiguous on both sides - not compared"})
        else:
            keyed[("id", pid)] = e
    names_here = {}
    for e in entries:
        names_here[e["player_name"]] = names_here.get(e["player_name"], 0) + 1
    for e in entries:
        if e["player_id"] is not None:
            continue
        name = e["player_name"]
        there = [o for o in other if o["player_name"] == name]
        if names_here[name] > 1 or len(there) > 1:
            unresolved.append({**_brief(e), "reason": "no source ID and the name is not unique; "
                                                      "not matched"})
        elif there and there[0]["player_id"] is not None:
            unresolved.append({**_brief(e), "reason": "no source ID; the same name has a source ID "
                                                      "in the other capture - not joined (no ID is inferred)"})
        else:
            keyed[("name", name)] = e
    return keyed, unresolved


def _brief(e: dict) -> dict:
    return {"player_id": e["player_id"], "player_id_source": e["player_id_source"],
            "player_name": e["player_name"], "position": e["position"],
            "source_status": e["source_status"], "status_class": e["status_class"],
            "injury_type": e["injury_type"], "short_comment": e["short_comment"],
            "source_entry_date_raw": e["source_entry_date_raw"]}


def _basis(key) -> str:
    return "source_player_id" if key[0] == "id" else "inferred_name_match"


def compare_team(conn, team: str, earlier_id: str, later_id: str) -> dict:
    if earlier_id == later_id:
        raise CompareRefused("select two different captures")
    a, b = _capture(conn, earlier_id), _capture(conn, later_id)
    if ir.parse_utc_us(a["received_at_utc"]) >= ir.parse_utc_us(b["received_at_utc"]):
        raise CompareRefused(f"{earlier_id} was not captured before {later_id}; pass the earlier "
                             "capture as --earlier")
    for cap in (a, b):
        if not cap["usable"]:
            raise CompareRefused(f"capture {cap['capture_id']} is not usable (outcome {cap['outcome']}); "
                                 "its missing entries would be meaningless")
    ctx_a, ctx_b = _context(conn, a, team), _context(conn, b, team)
    for ctx in (ctx_a, ctx_b):
        if ctx["team_coverage_state"] in (None, "absent_from_feed"):
            raise CompareRefused(f"{team} is not in the feed of capture {ctx['capture_id']}; not comparable")

    ea, eb = _entries(conn, a["capture_import_id"], team), _entries(conn, b["capture_import_id"], team)
    duplicated = _duplicated_ids(ea, eb)
    ka, ua = _key_entries(ea, eb, duplicated)
    kb, ub = _key_entries(eb, ea, duplicated)
    later_ids_elsewhere = _ids_by_team(conn, b["capture_import_id"], team)
    earlier_ids_elsewhere = _ids_by_team(conn, a["capture_import_id"], team)

    new, changed, same, absent = [], [], [], []
    for key, e in kb.items():
        if key not in ka:
            new.append({**_brief(e), "identity_basis": _basis(key) if key[0] == "id" else "no_source_id",
                        "listed_for_other_team_in_earlier": earlier_ids_elsewhere.get(e["player_id"], [])})
    for key, e in ka.items():
        if key not in kb:
            absent.append({**_brief(e), "identity_basis": _basis(key) if key[0] == "id" else "no_source_id",
                           "listed_for_other_team_in_later": later_ids_elsewhere.get(e["player_id"], [])})
            continue
        f = kb[key]
        diff = {c: {"earlier": e[c], "later": f[c]} for c in COMPARED_FIELDS if e[c] != f[c]}
        item = {"player_id": f["player_id"], "player_name": f["player_name"],
                "identity_basis": _basis(key)}
        if diff:
            changed.append({**item, "changed_fields": diff,
                            "earlier_status": e["source_status"], "later_status": f["source_status"]})
        else:
            same.append({**item, "source_status": f["source_status"]})

    notes = []
    for ctx in (ctx_a, ctx_b):
        g = ctx["team_game_in_scope"]
        if g is None:
            notes.append(f"{ctx['capture_id']}: {team} had no game in the capture's scope week")
        elif g["state"] != "pregame":
            notes.append(f"{ctx['capture_id']}: captured with {team}'s game {g['game_id']} in state "
                         f"{g['state']!r}; not pregame evidence for that game")
        if ctx["slot_status"] not in ("on_time", "late"):
            notes.append(f"{ctx['capture_id']}: slot status {ctx['slot_status']!r}; does not fill a slot")
        if len(ctx["entries_per_feed_team_in_capture"]) == 1:
            notes.append(f"{ctx['capture_id']}: every feed team has exactly "
                         f"{ctx['entries_per_feed_team_in_capture'][0]} entries, which suggests a per-team "
                         "cap (not a confirmed provider contract); older entries may be omitted")
    ga, gb = ctx_a["team_game_in_scope"], ctx_b["team_game_in_scope"]
    if (ga or {}).get("game_id") != (gb or {}).get("game_id"):
        notes.append("the two captures refer to different scope games/weeks for this team")
    inferred = sum(1 for k in kb if k[0] == "name" and k in ka)
    return {
        "team": team, "earlier": ctx_a, "later": ctx_b,
        "freshness": FRESHNESS,
        "evidence_notes": notes,
        "identity": {"matched_by_source_player_id": sum(1 for k in kb if k[0] == "id" and k in ka),
                     "matched_by_inferred_name": inferred,
                     "identity_unresolved": len(ua) + len(ub)},
        "counts": {"newly_observed": len(new), "changed": len(changed), "unchanged": len(same),
                   "absent_from_later": len(absent), "identity_unresolved": len(ua) + len(ub)},
        "compared_fields": list(COMPARED_FIELDS),
        "newly_observed": new, "changed": changed, "unchanged": same,
        "absent_from_later": absent, "absent_from_later_meaning": ABSENCE_MEANING,
        "identity_unresolved": {"earlier": ua, "later": ub},
        "limitations": list(LIMITATIONS),
    }


def _ids_by_team(conn, cid: int, exclude_team: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for pid, t in conn.execute("SELECT player_id, team FROM injury_observations WHERE capture_import_id = ? "
                               "AND player_id IS NOT NULL AND (team IS NULL OR team != ?)",
                               (cid, exclude_team)):
        out.setdefault(pid, []).append(t)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("import", help="import explicitly selected capture files")
    p.add_argument("--db", required=True)
    p.add_argument("captures", nargs="+", help="capture JSON files (no directories)")
    p = sub.add_parser("list", help="imported captures, their coverage and slot gaps")
    p.add_argument("--db", required=True)
    p = sub.add_parser("compare", help="compare one team between two imported captures")
    p.add_argument("--db", required=True)
    p.add_argument("--team", required=True)
    p.add_argument("--earlier", required=True, help="capture_id of the earlier capture")
    p.add_argument("--later", required=True, help="capture_id of the later capture")
    p = sub.add_parser("upgrade", help="copy a legacy database to a NEW file with kickoff provenance columns")
    p.add_argument("--from", dest="src", required=True, help="legacy database (only read)")
    p.add_argument("--to", dest="dst", required=True, help="new database file to create")
    p = sub.add_parser("sql", help="one read-only query (football pilot safeguards)")
    p.add_argument("--db", required=True)
    p.add_argument("query")
    args = ap.parse_args(argv)

    try:
        if args.cmd == "import":
            conn = connect_for_import(args.db)
            try:
                out = import_captures(conn, args.captures)
            finally:
                conn.close()
        elif args.cmd == "upgrade":
            out = upgrade_database(args.src, args.dst)
        elif args.cmd == "sql":
            if not Path(args.db).exists():
                raise FileNotFoundError(f"database not found: {args.db}")
            cols, rows = fr.run_sql(args.db, args.query)
            print("COVERAGE_NOT_CHECKED: raw SQL does not establish complete injury lists, official "
                  "status or freshness.", file=sys.stderr)
            print("\t".join(cols))
            for r in rows:
                print("\t".join("" if v is None else str(v) for v in r))
            return 0
        else:
            conn = open_read_only(args.db)
            try:
                out = list_captures(conn) if args.cmd == "list" else \
                    compare_team(conn, args.team, args.earlier, args.later)
            finally:
                conn.close()
    except (ImportRefused, CompareRefused, fr.SchemaRefused, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except sqlite3.DatabaseError as exc:
        print(f"error: {exc} (the sql command only runs read-only queries)", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
