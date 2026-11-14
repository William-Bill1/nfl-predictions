"""Read-only stored game context (injury and weather research databases) for one game.

Data adapter for the home page's "Stored game context" panel. It reads the
injury-history and weather-history pilot databases through their guarded
read-only connections and never writes, initializes, captures, imports or
requests anything. Nothing here feeds the model, picks or probabilities.

Configuration is explicit: the environment variable NFL_STORED_CONTEXT_CONFIG
names a JSON file
    {"stored_context_version": 1,
     "injury_db": "<path to injury_history.sqlite3>",
     "weather_db": "<path to weather_history.sqlite3>"}
(either database may be omitted; relative paths resolve against the config
file's folder). Unset means "not configured"; anything malformed is reported.

Identity and time: context is selected by the exact game_id and checked against
that game's teams and kickoff; weather is also checked against the game's exact
stadium_id and stadium name (missing or different venue: no forecast shown). The cutoff is the earlier of now and kickoff;
captures RECEIVED after it are excluded. Source entry dates are never used as
capture times.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import football_repository as fr  # noqa: E402
import injury_history as ih  # noqa: E402
import pregame_snapshots as ps  # noqa: E402
import weather_history as wh  # noqa: E402

CONFIG_ENV = "NFL_STORED_CONTEXT_CONFIG"
CONFIG_VERSION = 1
CONFIG_KEYS = {"stored_context_version", "injury_db", "weather_db"}
TORONTO = ZoneInfo("America/Toronto")
NOT_CONFIGURED = "Stored context not configured."

INJURY_SOURCE_NOTE = (
    "ESPN's general injury/news feed, not the official NFL injury report. 'Active' entries are "
    "often news notes, not availability designations. Every retained capture lists at most about "
    "25 entries per team (an apparent cap, not a confirmed provider rule), so coverage may be "
    "incomplete. A player missing from a later capture is NOT recovered or cleared.")
WEATHER_SOURCE_NOTE = (
    "Captured model forecasts for a grid cell near the venue, not observed stadium weather. "
    "No forecast issue time is available; age is time since capture. Values are the provider's "
    "hourly valid times only (no kickoff interpolation, no daily averages). Gusts and "
    "precipitation describe the preceding hour.")
WEATHER_MISSING_NOTE = ("No usable forecast was captured by the cutoff. This means unavailable or "
                        "not captured; it does not mean indoor or neutral weather.")
ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0, https://open-meteo.com/en/license)"

ENTRY_FIELDS = ("player_name", "position", "player_id", "player_id_source", "source_status",
                "status_class", "injury_type", "injury_location", "injury_detail", "short_comment",
                "source_return_date", "source_entry_date_raw")

_DB_ERRORS = (FileNotFoundError, fr.SchemaRefused, sqlite3.DatabaseError, OSError)


class GameIdentityError(ValueError):
    """The selected game can't be identified exactly; no context is shown."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(text) -> datetime | None:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(timezone.utc) if dt.tzinfo else None


def toronto(dt: datetime | None) -> str:
    return "unknown" if dt is None else dt.astimezone(TORONTO).strftime("%a %b %d, %Y %I:%M %p %Z")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def load_config(environ=None) -> dict:
    """{'state': 'not_configured'} | {'state': 'error', 'message'} |
    {'state': 'ok', 'path', 'injury_db', 'weather_db'} (each db a Path or None).
    Only reads the config file; never checks or creates the databases."""
    environ = os.environ if environ is None else environ
    raw = environ.get(CONFIG_ENV)
    if raw is None or not str(raw).strip():
        return {"state": "not_configured", "message": NOT_CONFIGURED}
    path = Path(raw)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"state": "error", "message": f"Stored-context configuration file not found: {path}"}
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return {"state": "error", "message": f"Stored-context configuration {path} is not valid JSON ({exc})"}
    if not isinstance(doc, dict):
        return {"state": "error", "message": f"Stored-context configuration {path} must be a JSON object"}
    unknown = sorted(set(doc) - CONFIG_KEYS)
    if unknown:
        return {"state": "error", "message": f"Stored-context configuration {path} has unknown keys {unknown}"}
    if doc.get("stored_context_version") != CONFIG_VERSION:
        return {"state": "error", "message": f"Stored-context configuration {path} needs "
                                             f"\"stored_context_version\": {CONFIG_VERSION}"}
    out = {"state": "ok", "path": path}
    for key in ("injury_db", "weather_db"):
        value = doc.get(key)
        if value is None:
            out[key] = None
        elif not isinstance(value, str) or not value.strip():
            return {"state": "error", "message": f"Stored-context configuration {path}: {key} must be a "
                                                 "non-empty path string"}
        else:
            p = Path(value)
            out[key] = p if p.is_absolute() else (path.parent / p)
    if out["injury_db"] is None and out["weather_db"] is None:
        return {"state": "error", "message": f"Stored-context configuration {path} names no database"}
    return out


# --------------------------------------------------------------------------
# Game identity and cutoff
# --------------------------------------------------------------------------

def _text(v) -> str:
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return ""
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.strftime("%Y-%m-%d")
    return str(v).strip()


def game_identity(games: pd.DataFrame, game_id: str) -> dict:
    """The one row with exactly this game_id: teams and scheduled kickoff (UTC).
    Missing, duplicated or incomplete identities raise GameIdentityError."""
    if games is None or "game_id" not in games.columns:
        raise GameIdentityError("no game table to identify the game from")
    rows = games[games["game_id"].astype(str) == str(game_id)]
    if len(rows) == 0:
        raise GameIdentityError(f"game {game_id!r} is not in the predictions table")
    if len(rows) > 1:
        raise GameIdentityError(f"game {game_id!r} appears {len(rows)} times in the predictions table; "
                                "refusing to guess which row is meant")
    r = rows.iloc[0]
    home, away = _text(r.get("home_team")), _text(r.get("away_team"))
    if not home or not away:
        raise GameIdentityError(f"game {game_id!r} has no home/away team")
    kickoff, problem = ps.kickoff_utc(_text(r.get("gameday"))[:10] or None, _text(r.get("gametime")) or None)
    if kickoff is None:
        raise GameIdentityError(f"game {game_id!r} has no usable kickoff time ({problem})")
    # Venue identity is kept as given (possibly empty); only weather needs it,
    # so a missing venue never blocks the injury section.
    return {"game_id": str(game_id), "home_team": home, "away_team": away, "kickoff_utc": kickoff,
            "stadium_id": _text(r.get("stadium_id")), "stadium": _text(r.get("stadium"))}


def cutoff_for(kickoff: datetime, now: datetime) -> datetime:
    return min(now, kickoff)


# --------------------------------------------------------------------------
# Injuries
# --------------------------------------------------------------------------

def _rows(conn, sql, params=()) -> list[dict]:
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _summary(cap: dict) -> dict:
    return {"capture_id": cap["capture_id"], "received_at_utc": _parse(cap["received_at_utc"]),
            "outcome": cap["outcome"], "usable": bool(cap["usable"]), "slot_id": cap["slot_id"],
            "slot_status": cap["slot_status"]}


def injury_context(db_path: Path, game: dict, cutoff: datetime) -> dict:
    try:
        conn = ih.open_read_only(db_path)
    except _DB_ERRORS as exc:
        return {"state": "error", "message": f"Injury database unavailable: {exc}"}
    try:
        return _injury(conn, game, cutoff)
    except sqlite3.DatabaseError as exc:
        return {"state": "error", "message": f"Injury database could not be read: {exc}"}
    finally:
        conn.close()


def _injury(conn, game: dict, cutoff: datetime) -> dict:
    gid, home, away, kickoff = game["game_id"], game["home_team"], game["away_team"], game["kickoff_utc"]
    caps = _rows(conn, "SELECT * FROM injury_captures WHERE transformation_version = ?",
                 (ih.TRANSFORMATION_VERSION,))
    caps.sort(key=lambda c: (_parse(c["received_at_utc"]) or datetime.min.replace(tzinfo=timezone.utc),
                             c["capture_id"]))
    notices, eligible, failed = [], [], []
    after_cutoff = 0
    for c in caps:
        received = _parse(c["received_at_utc"])
        if received is None:
            notices.append(f"capture {c['capture_id']} has no readable receipt time; ignored")
            continue
        if received > cutoff:
            after_cutoff += 1
            continue
        scoped = _rows(conn, "SELECT * FROM injury_capture_games WHERE capture_import_id = ? AND game_id = ?",
                       (c["capture_import_id"], gid))
        has_scope = conn.execute("SELECT COUNT(*) FROM injury_capture_games WHERE capture_import_id = ?",
                                 (c["capture_import_id"],)).fetchone()[0] > 0
        if not scoped:
            if not c["usable"] and not has_scope:
                failed.append({**_summary(c), "scope": "unknown (no scope week recorded)"})
            continue                              # another week's capture: never borrowed
        g = scoped[0]
        if len(scoped) > 1 or (g["home_team"], g["away_team"]) != (home, away):
            notices.append(f"capture {c['capture_id']} lists {gid} with teams "
                           f"{g['away_team']} @ {g['home_team']}, not {away} @ {home}; excluded")
            continue
        if _parse(g["kickoff_utc"]) != kickoff:
            notices.append(f"capture {c['capture_id']} lists {gid} kicking off {g['kickoff_utc']}, not "
                           f"{kickoff:%Y-%m-%dT%H:%M}Z; excluded")
            continue
        if not c["usable"]:
            failed.append({**_summary(c), "scope": "this game"})
        elif g["state"] != "pregame":
            notices.append(f"capture {c['capture_id']} recorded {gid} as {g['state']!r}, not pregame; excluded")
        else:
            eligible.append(c)
    latest = eligible[-1] if eligible else None
    newest = _parse(latest["received_at_utc"]) if latest else None
    newer_failed = [f for f in failed if newest is None or f["received_at_utc"] > newest]
    out = {"state": "ok" if latest else "none", "notices": notices, "newer_failed": newer_failed,
           "excluded_received_after_cutoff": after_cutoff, "eligible_count": len(eligible),
           "latest": None, "previous": None, "teams": {}, "comparisons": {}}
    if latest is None:
        out["message"] = "No usable pregame injury capture for this game was received by the cutoff."
        return out
    out["latest"] = _summary(latest)
    for team in (away, home):
        ctx = ih._context(conn, latest, team)
        out["teams"][team] = {
            "coverage_state": ctx["team_coverage_state"], "entries": ctx["team_entries"],
            "entries_per_feed_team": ctx["entries_per_feed_team_in_capture"],
            "rows": [{k: e[k] for k in ENTRY_FIELDS} for e in ih._entries(conn, latest["capture_import_id"], team)],
        }
    if len(eligible) >= 2:
        prev = eligible[-2]
        out["previous"] = _summary(prev)
        for team in (away, home):
            try:
                out["comparisons"][team] = ih.compare_team(conn, team, prev["capture_id"], latest["capture_id"])
            except ih.CompareRefused as exc:
                out["comparisons"][team] = {"refused": str(exc)}
    return out


# --------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------

def weather_context(db_path: Path, game: dict, cutoff: datetime, now: datetime) -> dict:
    try:
        conn = wh.open_read_only(db_path)
    except _DB_ERRORS as exc:
        return {"state": "error", "message": f"Weather database unavailable: {exc}"}
    try:
        return _weather(conn, game, cutoff, now)
    except sqlite3.DatabaseError as exc:
        return {"state": "error", "message": f"Weather database could not be read: {exc}"}
    finally:
        conn.close()


def _venue_problem(game: dict, chosen: dict | None) -> str | None:
    """Why weather can't be shown for this venue, or None. Exact comparison of
    stadium_id AND stadium name; no fuzzy matching, no fallback."""
    if not game["stadium_id"] or not game["stadium"]:
        return (f"The selected game has no complete venue identity (stadium_id {game['stadium_id'] or 'missing'!r}, "
                f"stadium {game['stadium'] or 'missing'!r}); weather is not shown.")
    if chosen is not None and ((chosen["stadium_id"] or "").strip(), (chosen["stadium"] or "").strip()) \
            != (game["stadium_id"], game["stadium"]):
        return (f"Weather capture {chosen['capture_id']} is for venue {chosen['stadium_id']} "
                f"({chosen['stadium']}), but the selected game is at {game['stadium_id']} ({game['stadium']}); "
                "the forecast is not shown.")
    return None


def _weather(conn, game: dict, cutoff: datetime, now: datetime) -> dict:
    res = wh.as_of(conn, game["game_id"], cutoff, now=now)
    failed_ids = res["unusable_received_by_cutoff"]
    failed = []
    if failed_ids:
        q = ",".join("?" * len(failed_ids))
        failed = [{"capture_id": r["capture_id"], "received_at_utc": _parse(r["received_at_utc"]),
                   "outcome": r["outcome"], "detail": r["outcome_detail"]}
                  for r in _rows(conn, f"SELECT capture_id, received_at_utc, outcome, outcome_detail "
                                       f"FROM weather_captures WHERE capture_id IN ({q})", tuple(failed_ids))]
    out = {"state": "none", "notices": [], "excluded_received_after_cutoff": res["excluded_received_after_cutoff"],
           "capture": None, "hours": [], "newer_failed": [], "variable_meaning": res["variable_meaning"]}
    cap = res["capture"]
    chosen = None
    if cap is not None:
        chosen = _rows(conn, "SELECT * FROM weather_captures WHERE capture_id = ?", (cap["capture_id"],))[0]
        if (chosen["home_team"], chosen["away_team"]) != (game["home_team"], game["away_team"]) \
                or _parse(chosen["kickoff_utc"]) != game["kickoff_utc"]:
            out.update(state="error", message=(
                f"Weather capture {cap['capture_id']} is for {chosen['away_team']} @ {chosen['home_team']} "
                f"kicking off {chosen['kickoff_utc']}, which doesn't match this game; not shown."))
            return out
    venue = _venue_problem(game, chosen)
    if venue:
        out.update(state="error", message=venue)
        return out
    received = _parse(chosen["received_at_utc"]) if chosen else None
    out["newer_failed"] = sorted((f for f in failed if received is None or f["received_at_utc"] > received),
                                 key=lambda f: f["received_at_utc"])
    if chosen is None:
        out["message"] = WEATHER_MISSING_NOTE
        return out
    out.update(state="ok", hours=res["hours"], capture={
        "capture_id": chosen["capture_id"], "received_at_utc": received,
        "age_since_capture_hours": cap["age_since_capture_hours"], "outcome": chosen["outcome"],
        "units": cap["units"], "missing_variables": cap["missing_variables"],
        "missing_requested_hours": cap["missing_requested_hours"], "stadium": chosen["stadium"],
        "field_exposure": chosen["field_exposure"], "schedule_roof": chosen["schedule_roof"],
        "requested": cap["requested"], "returned_grid": cap["returned_grid"]})
    return out


# --------------------------------------------------------------------------
# One or several games
# --------------------------------------------------------------------------

AVAILABILITY_NOTE = ("The ESPN injury/news feed does not establish official game-day availability: "
                     "official inactives are announced separately before kickoff and are not stored here.")
AGE_NOTE = ("Age is the time since this system received the capture. It is not a forecast issue time "
            "and does not show that the source itself was current.")


def _section(conns: dict, errors: dict, key: str, label: str, read) -> dict:
    """One database's result for one game, from the shared connection."""
    if key in errors:
        return {"state": "error", "message": errors[key]}
    if key not in conns:
        return {"state": "not_configured", "message": NOT_CONFIGURED}
    try:
        return read(conns[key])
    except sqlite3.DatabaseError as exc:
        return {"state": "error", "message": f"{label} database could not be read: {exc}"}


def _one(games: pd.DataFrame, game_id: str, conns: dict, errors: dict, now: datetime) -> dict:
    try:
        game = game_identity(games, game_id)
    except GameIdentityError as exc:
        return {"state": "identity_error", "message": str(exc), "now": now}
    cutoff = cutoff_for(game["kickoff_utc"], now)
    return {
        "state": "ok", "game": game, "now": now, "cutoff": cutoff,
        "cutoff_basis": "kickoff" if cutoff == game["kickoff_utc"] else "current time",
        "injuries": _section(conns, errors, "injury_db", "Injury", lambda c: _injury(c, game, cutoff)),
        "weather": _section(conns, errors, "weather_db", "Weather", lambda c: _weather(c, game, cutoff, now)),
    }


def build_all(games: pd.DataFrame, game_ids, *, environ=None, now: datetime | None = None) -> dict:
    """Context for several games from one read of each database: one guarded
    read-only connection per database for the whole call, each game evaluated
    once with its own cutoff, all against the same `now`. Nothing is cached
    between calls, so a newly imported capture shows on the next call.
    Returns {'state', 'message'?, 'now', 'contexts': {game_id: context}}."""
    now = now or utc_now()
    cfg = load_config(environ)
    if cfg["state"] != "ok":
        return {"state": cfg["state"], "message": cfg["message"], "now": now, "contexts": {}}
    conns, errors = {}, {}
    for key, opener, label in (("injury_db", ih.open_read_only, "Injury"),
                               ("weather_db", wh.open_read_only, "Weather")):
        if cfg[key] is not None:
            try:
                conns[key] = opener(cfg[key])
            except _DB_ERRORS as exc:
                errors[key] = f"{label} database unavailable: {exc}"
    try:
        contexts = {str(g): _one(games, str(g), conns, errors, now) for g in game_ids}
    finally:
        for conn in conns.values():
            conn.close()
    return {"state": "ok", "now": now, "contexts": contexts}


def build(games: pd.DataFrame, game_id: str, *, environ=None, now: datetime | None = None) -> dict:
    """Everything the panel shows for one game. Never raises for data problems."""
    res = build_all(games, [game_id], environ=environ, now=now)
    if res["state"] != "ok":
        return {"state": res["state"], "message": res["message"]}
    ctx = res["contexts"][str(game_id)]
    return {k: v for k, v in ctx.items() if k != "now"} if ctx["state"] != "ok" else ctx


# --------------------------------------------------------------------------
# Coverage and capture age
# --------------------------------------------------------------------------

COVERAGE_STATES = ("usable", "missing", "error", "not_configured")


def coverage_state(ctx: dict | None, key: str) -> str:
    """'usable' only for context accepted by the identity, venue and cutoff
    checks; 'missing' when nothing usable was received by the cutoff; 'error'
    for an unidentifiable game or a capture/database that failed a check."""
    if ctx is None or ctx.get("state") != "ok":
        return "error"
    return {"ok": "usable", "none": "missing", "not_configured": "not_configured"}.get(ctx[key]["state"], "error")


def coverage(result: dict, game_ids) -> dict:
    """Counts over exactly these games; each game once per context type."""
    ids = list(dict.fromkeys(str(g) for g in game_ids))
    out = {"total": len(ids)}
    for kind, key in (("injury", "injuries"), ("weather", "weather")):
        counts = dict.fromkeys(COVERAGE_STATES, 0)
        for gid in ids:
            counts[coverage_state(result["contexts"].get(gid), key)] += 1
        out[kind] = counts
    return out


def receipt_age_hours(received: datetime | None, now: datetime) -> float | None:
    """Hours since this system received a capture (not a source or issue time)."""
    return None if received is None else (now - received).total_seconds() / 3600
