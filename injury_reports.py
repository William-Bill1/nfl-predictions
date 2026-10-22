"""Immutable injury-report captures around the Ontario decision slots.

Phase 1: evidence only. When the Ontario Spread Capture workflow runs for the
Wednesday 12:00 or Sunday 09:00 America/Toronto slot (or an ad-hoc run), this
records what the application's existing injury source - ESPN's public
injuries JSON feed, the same one ``player_props/injuries.py`` reads - showed at
that moment. Nothing here adjusts probabilities, scores games or recommends
bets, and nothing reads or writes the nightly ``espn_injuries.csv`` cache.

What the source is (and isn't)
-------------------------------
* ESPN's per-player ``status`` is ESPN's general injury status. It mixes
  game-designation words (Questionable, Doubtful, Out), roster statuses
  (Injured Reserve, Suspension, ...) and "Active". It is **not** the official
  NFL game-status report and is stored exactly as supplied; ``status_class``
  is only a lexical grouping of that raw word.
* The feed has **no practice-participation field**. Practice participation
  is stored as unavailable, never inferred (a DNP is not "out"; a blank is not
  "healthy").
* Per-entry ``date`` (when present) is stored as the source's **entry date**
  (raw text, plus UTC when parseable). What that date means - a status update,
  the injury, or something else - is **unverified**, so it is never treated as
  a last-update time or as evidence of freshness. Absent or unparseable dates
  are recorded explicitly and never filled from the capture time. The HTTP
  ``Date`` / ``Last-Modified`` headers describe the response, not any player's
  status, and are stored separately.

Timing rules
------------
* ``received_at`` (UTC) - when the response arrived - is the observation time.
  ``request_started_at`` is also stored. Neither is ever moved to the slot time.
* The slot follows ontario_spreads' rules (Toronto wall clock, same windows),
  judged at ``received_at``: on time up to 30 minutes after the slot, "late"
  inside the window, and "outside_window" after it. An outside-window or
  ad-hoc capture never fills a scheduled slot. A capture is never made for a
  past slot (no retrospective reconstruction).
* The clock is read again under the slot lock immediately before the request:
  if that slot's window has closed by then, no request is made
  (``window_closed``); the run never switches to another slot or to ad hoc.

Storage
-------
Each capture is a new checksummed JSON file in
``data_files/injury_reports/captures/``, created exclusively and never
overwritten. Unavailable or empty responses are kept as evidence but leave the
slot open for a later run inside the window. The checksum is an unkeyed
SHA-256: it detects accidental changes, not a deliberate edit followed by
recomputing the checksum.

See docs/INJURY_REPORT_ARCHIVE.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import requests

import ontario_spreads as on
import pregame_snapshots as ps
from player_props.injuries import ESPN_INJURIES_URL, HEADERS
from player_props.market_odds import _ABBR_BY_FULL_NAME

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_files"
STORE_DIR = DATA_DIR / "injury_reports"
CAPTURE_DIR = STORE_DIR / "captures"
SCHEDULE_PATH = on.SCHEDULE_PATH

SCHEMA_VERSION = 1
CAPTURE_KIND = "injury_report_capture"
CHECKSUM_FIELD = ps.CHECKSUM_FIELD
TORONTO = on.TORONTO

REQUEST_TIMEOUT = (5, 20)             # (connect, per-read) seconds
# Decoded (decompressed) body limit. The first live response (2026-10-21) was
# 8,726,788 bytes, over the original 8 MiB; 32 MiB leaves ~3.8x headroom for a
# growing in-season feed. It bounds the accumulated body bytes (at most the
# limit plus one chunk), not total process memory: the bytes copy,
# decompression buffers and the parsed JSON need memory on top of that.
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
READ_CHUNK = 64 * 1024                # decoded bytes per read (urllib3 2.x returns at most this)
READ_DEADLINE = 90.0                  # seconds to read the body, enforced by a watchdog that
                                      # shuts the transport down (see _read_bounded)
MAX_TEXT = 500                        # per stored free-text field / error message

SOURCE = {
    "name": "espn_injuries_json",
    "publisher": "ESPN",
    "url": ESPN_INJURIES_URL,
    "official_nfl_report": False,
    "practice_participation": "not supplied by this source",
    "status_meaning": ("ESPN's general injury status as supplied: a mixture of game-designation "
                       "words, roster statuses and 'Active'; not the official NFL game-status "
                       "report"),
    "entry_date": ("per-entry 'date' as supplied (raw, plus UTC when parseable); its meaning is "
                   "unverified - not treated as a status-update time or as evidence of freshness"),
    "http_headers": "HTTP Date / Last-Modified describe the response, not player-status updates",
}
ENTRY_DATE_MEANING = "unverified"

# Lexical grouping of ESPN's raw status word (lower-cased). Not a claim about
# the official report; the raw word is always stored alongside.
STATUS_CLASSES = {
    "questionable": "game_designation_word", "doubtful": "game_designation_word",
    "out": "game_designation_word", "probable": "game_designation_word",
    "injured reserve": "roster_status", "ir": "roster_status",
    "physically unable to perform": "roster_status", "pup": "roster_status",
    "non-football injury": "roster_status", "nfi": "roster_status",
    "suspension": "roster_status", "suspended": "roster_status",
    "active": "listed_active",
}
STATUS_CLASS_VALUES = ("game_designation_word", "roster_status", "listed_active",
                       "unrecognized", "missing")

OUTCOMES = ("full_team_coverage", "partial_team_coverage", "empty", "unavailable")
USABLE_OUTCOMES = ("full_team_coverage", "partial_team_coverage")
TEAM_COVERAGE = ("rows_listed", "listed_without_rows", "absent_from_feed")
SLOT_STATUSES = ("on_time", "late", "outside_window", on.AD_HOC)
ID_SOURCES = ("athlete.id", "athlete.links", None)
_ESPN_ID_RE = re.compile(r"/id/(\d+)(?:/|$)")


class CaptureConflictError(on.CaptureConflictError):
    pass


class ValidationError(ValueError):
    pass


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def iso(dt: datetime) -> str:
    return on.iso(dt)


def _iso_us(dt: datetime) -> str:
    """UTC with microseconds: the observation and request times are kept at
    full precision so the order of start and receipt is exact."""
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


_UTC_US_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")


def parse_utc_us(text) -> datetime:
    if not isinstance(text, str) or not _UTC_US_RE.match(text):
        raise ValueError(f"{text!r} is not a UTC timestamp of the form YYYY-MM-DDTHH:MM:SS.ffffffZ")
    return ps._parse_iso(text)


def _text(value) -> str | None:
    """A supplied non-empty string, bounded; anything else is unavailable."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value[:MAX_TEXT] if value else None


def status_class(raw: str | None) -> str:
    if raw is None:
        return "missing"
    return STATUS_CLASSES.get(raw.strip().lower(), "unrecognized")


def _player_id(athlete: dict) -> tuple[str | None, str | None]:
    """ESPN athlete id, from athlete.id or (failing that) an athlete link URL."""
    raw = athlete.get("id")
    if isinstance(raw, (str, int)) and not isinstance(raw, bool) and str(raw).strip().isdigit():
        return str(raw).strip(), "athlete.id"
    for link in athlete.get("links") or []:
        href = link.get("href") if isinstance(link, dict) else None
        m = _ESPN_ID_RE.search(href) if isinstance(href, str) else None
        if m:
            return m.group(1), "athlete.links"
    return None, None


def _entry_date(raw) -> tuple[str | None, str | None]:
    """(raw text as supplied, its UTC form when parseable). Absent -> (None, None).

    The source entry date's meaning is unverified: it is kept as supplied and
    never used as a status-update time or a freshness signal."""
    if raw is None:
        return None, None
    text = _text(str(raw))
    if text is None:
        return None, None
    try:
        return text, iso(on.parse_provider_time(text))
    except ValueError:
        return text, None


# --------------------------------------------------------------------------
# Parse
# --------------------------------------------------------------------------

def parse_entries(payload, api_key: str | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """ESPN injuries payload -> (team blocks, player rows, malformed entries).

    Raises ValueError if the payload doesn't have the expected top-level
    shape. Every supplied value is kept as supplied; anything missing is None.
    """
    payload = on.sanitize(payload, api_key)
    if not isinstance(payload, dict) or not isinstance(payload.get("injuries"), list):
        raise ValueError("response has no 'injuries' list")
    teams, rows, malformed = [], [], []
    for t_index, team in enumerate(payload["injuries"]):
        if not isinstance(team, dict):
            malformed.append({"team_index": t_index, "entry_index": None,
                              "reason": "team block is not an object"})
            continue
        name = _text(team.get("displayName"))
        abbr = _ABBR_BY_FULL_NAME.get(name) if name else None
        items = team.get("injuries")
        if not isinstance(items, list):
            malformed.append({"team_index": t_index, "entry_index": None,
                              "reason": "team block has no 'injuries' list"})
            items = []
        teams.append({"source_team_name": name, "source_team_id": _text(str(team["id"]))
                      if isinstance(team.get("id"), (str, int)) else None,
                      "team": abbr, "entries": len(items)})
        for e_index, item in enumerate(items):
            athlete = item.get("athlete") if isinstance(item, dict) else None
            player = _text(athlete.get("displayName")) if isinstance(athlete, dict) else None
            if not player:
                malformed.append({"team_index": t_index, "entry_index": e_index,
                                  "source_team_name": name,
                                  "reason": "entry has no athlete display name"})
                continue
            pid, id_source = _player_id(athlete)
            position = athlete.get("position")
            details = item.get("details") if isinstance(item.get("details"), dict) else {}
            injury = {k: _text(details.get(k)) for k in ("type", "location", "detail", "side")}
            stype = item.get("type") if isinstance(item.get("type"), dict) else {}
            raw_status = _text(item.get("status"))
            date_raw, date_utc = _entry_date(item.get("date"))
            rows.append({
                "source_team_name": name, "team": abbr,
                "player_name": player, "player_id": pid, "player_id_source": id_source,
                "position": _text(position.get("abbreviation")) if isinstance(position, dict) else None,
                "source_status": raw_status, "status_class": status_class(raw_status),
                "source_status_type": {k: _text(stype.get(k)) for k in
                                       ("name", "description", "abbreviation")},
                "injury": injury,
                "short_comment": _text(item.get("shortComment")),
                "source_return_date": _text(details.get("returnDate")),
                "practice_participation": None,
                "source_entry_date_raw": date_raw, "source_entry_date_utc": date_utc,
            })
    return teams, rows, malformed


# --------------------------------------------------------------------------
# Scope, matching and coverage
# --------------------------------------------------------------------------

def scope_games(schedule, at: datetime) -> tuple[dict | None, list[dict]]:
    """(season/week scope, that week's games with their pregame state) at `at`,
    using ontario_spreads' scope rule (the week of the next game to kick off)."""
    eligible, excluded = on.games_in_scope(schedule, at)
    if not eligible:
        return None, []
    games = [{**g, "state": "pregame"} for g in eligible] + \
            [{"game_id": g["game_id"], "season": g["season"], "week": g["week"],
              "home_team": g["home_team"], "away_team": g["away_team"],
              "kickoff_utc": g["kickoff_utc"], "state": g["reason"]} for g in excluded]
    games.sort(key=lambda g: g["game_id"])
    return {"season": eligible[0]["season"], "week": eligible[0]["week"]}, games


def match_rows(rows: list[dict], games: list[dict]) -> None:
    """Attach game_id / game_state per row, from the row's team only."""
    by_team = {}
    for g in games:
        by_team[g["home_team"]] = g
        by_team[g["away_team"]] = g
    for r in rows:
        g = by_team.get(r["team"]) if r["team"] else None
        r["game_id"] = g["game_id"] if g else None
        r["game_state"] = g["state"] if g else ("unmatched_team" if not r["team"]
                                               else "no_game_in_scope")


def team_coverage(teams: list[dict], rows: list[dict], games: list[dict]) -> dict:
    """Per in-scope team: rows_listed / listed_without_rows / absent_from_feed.
    None of these says a team has no injuries."""
    in_scope = sorted({t for g in games for t in (g["home_team"], g["away_team"])})
    listed = {t["team"] for t in teams if t["team"]}
    with_rows = {}
    for r in rows:
        if r["team"]:
            with_rows[r["team"]] = with_rows.get(r["team"], 0) + 1
    out = {}
    for team in in_scope:
        if with_rows.get(team):
            out[team] = {"state": "rows_listed", "rows": with_rows[team]}
        elif team in listed:
            out[team] = {"state": "listed_without_rows", "rows": 0}
        else:
            out[team] = {"state": "absent_from_feed", "rows": 0}
    return out


def outcome_for(teams: list[dict], rows: list[dict], malformed: list[dict],
                coverage: dict) -> str:
    if not teams or not rows:
        return "empty"
    unmatched = any(t["team"] is None for t in teams)
    absent = any(c["state"] == "absent_from_feed" for c in coverage.values())
    return "partial_team_coverage" if unmatched or absent or malformed else "full_team_coverage"


# --------------------------------------------------------------------------
# Slot timing
# --------------------------------------------------------------------------

def slot_record(resolved: dict, received_at: datetime) -> dict:
    """The capture's slot, judged at the observation time (received_at).

    A scheduled slot is on_time up to ON_TIME_TOLERANCE after the slot, late
    inside its window and outside_window after it; only on_time/late fill the
    slot. Ad-hoc captures never fill a slot."""
    if resolved["name"] == on.AD_HOC:
        return {"name": on.AD_HOC, "slot_id": resolved["slot_id"], "timezone": "America/Toronto",
                "intended_utc": None, "intended_local": None, "delay_minutes": None,
                "status": on.AD_HOC, "fills_slot": False, "judged_at": "received_at"}
    rule = on.SLOTS_BY_NAME[resolved["name"]]
    intended = on.parse_utc(resolved["intended_utc"])
    delay = received_at - intended
    if not on.in_slot_window(rule, intended, received_at):
        status = "outside_window"
    else:
        status = "on_time" if delay <= on.ON_TIME_TOLERANCE else "late"
    return {"name": resolved["name"], "slot_id": resolved["slot_id"],
            "timezone": "America/Toronto", "intended_utc": resolved["intended_utc"],
            "intended_local": resolved["intended_local"],
            "delay_minutes": round(delay.total_seconds() / 60, 1), "status": status,
            "fills_slot": status in ("on_time", "late"), "judged_at": "received_at"}


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------

class _TooLarge(Exception):
    pass


class _Deadline(Exception):
    pass


_SOCKET_PATHS = (
    ("_connection", "sock"),            # urllib3 2.x and urllib3-future HTTPResponse
    ("_fp", "fp", "raw", "_sock"),      # urllib3 2.x wrapping http.client.HTTPResponse
    ("fp", "raw", "_sock"),             # a bare http.client.HTTPResponse
)


def _socket_of(resp) -> socket.socket | None:
    raw = getattr(resp, "raw", None)
    for path in _SOCKET_PATHS:
        obj = raw
        for name in path:
            obj = getattr(obj, name, None)
        if isinstance(obj, socket.socket):
            return obj
    return None


def _abort_transport(resp) -> None:
    """Unblock a body read in progress on `resp`, from another thread.

    The per-read socket timeout only measures inactivity, so a server that
    trickles bytes can keep a chunk read blocked indefinitely (standard
    urllib3 / http.client block until the whole chunk arrives). Shutting the
    connection's socket down makes that blocked recv return (or fail) in the
    reading thread itself. urllib3-future's HTTPResponse.shutdown() is a
    no-op, so the socket is reached directly; without one, close() is the
    best effort."""
    sock = _socket_of(resp)
    try:
        if sock is not None:
            sock.shutdown(socket.SHUT_RDWR)
        else:
            resp.close()
    except Exception:                            # already closed or never connected
        pass


def _read_bounded(resp, limit: int, deadline: float, monotonic: Callable) -> bytes:
    """The decoded body, read in chunks: stops as soon as more than `limit`
    decoded bytes have arrived (_TooLarge) or `deadline` seconds have passed
    (_Deadline). Content-Length is never trusted for the size actually read;
    compressed bodies are counted after decoding.

    The deadline is enforced by a watchdog timer that shuts the transport down
    if it fires, so it holds even while a chunk read is blocked on a slow
    trickle. Reading stays in this thread; the timer is cancelled and joined
    before returning, so nothing keeps running afterwards."""
    started = monotonic()
    body = bytearray()
    expired = threading.Event()

    def on_deadline():
        expired.set()
        _abort_transport(resp)

    watchdog = threading.Timer(deadline, on_deadline)
    watchdog.name = "injury-read-deadline"
    watchdog.daemon = True
    watchdog.start()
    try:
        try:
            for chunk in resp.iter_content(chunk_size=READ_CHUNK):
                if chunk:
                    body += chunk
                    if len(body) > limit:
                        raise _TooLarge(len(body))
                if expired.is_set() or monotonic() - started > deadline:
                    raise _Deadline(len(body))
        except (_TooLarge, _Deadline):
            raise
        except Exception:
            if expired.is_set():             # the abort broke the read: report the deadline
                raise _Deadline(len(body)) from None
            raise
        if expired.is_set():                 # the abort ended the stream early: not a body
            raise _Deadline(len(body))
        return bytes(body)
    finally:
        watchdog.cancel()
        watchdog.join()


def fetch(get: Callable = requests.get, *, limit: int = MAX_RESPONSE_BYTES,
          deadline: float = READ_DEADLINE, monotonic: Callable = time.monotonic
          ) -> tuple[dict | None, dict]:
    """(payload or None, response/error metadata). Never raises for network,
    HTTP or JSON problems; they become an 'error' entry. The body is streamed
    and bounded (`limit` decoded bytes, `deadline` seconds in total, on top of
    the per-read timeout), and the response is always closed."""
    meta = {"http_status": None, "http_last_modified": None, "http_date": None,
            "bytes": None, "error": None}
    resp = None
    try:
        resp = get(ESPN_INJURIES_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT, stream=True)
        meta["http_status"] = resp.status_code
        headers = getattr(resp, "headers", {}) or {}
        meta["http_last_modified"] = _text(headers.get("Last-Modified"))
        meta["http_date"] = _text(headers.get("Date"))
        declared = headers.get("Content-Length")
        try:
            declared = int(declared) if declared is not None else None
        except (TypeError, ValueError):
            declared = None                       # unusable header: rely on counting
        if resp.status_code == 200 and declared is not None and declared > limit:
            meta["error"] = {"type": "too_large",
                             "message": f"declared Content-Length {declared} is over {limit} "
                                        "bytes; body not read"}
            return None, meta
        body = _read_bounded(resp, limit, deadline, monotonic)
        meta["bytes"] = len(body)
        if resp.status_code != 200:
            meta["error"] = {"type": "http_status", "message": f"HTTP {resp.status_code}"}
            return None, meta
        return json.loads(body), meta
    except _TooLarge as exc:
        meta["bytes"] = exc.args[0]
        meta["error"] = {"type": "too_large",
                         "message": f"response over {limit} bytes (stopped reading at "
                                    f"{exc.args[0]} decoded bytes)"}
    except _Deadline as exc:
        meta["bytes"] = exc.args[0]
        meta["error"] = {"type": "timeout",
                         "message": f"body not complete within {deadline:g} s "
                                    f"({exc.args[0]} bytes read)"}
    except requests.Timeout as exc:
        meta["error"] = {"type": "timeout", "message": on.redact(exc)[:MAX_TEXT]}
    except requests.RequestException as exc:
        meta["error"] = {"type": "request_error", "message": on.redact(exc)[:MAX_TEXT]}
    except ValueError as exc:
        meta["error"] = {"type": "invalid_json", "message": on.redact(exc)[:MAX_TEXT]}
    finally:
        if resp is not None:
            try:
                resp.close()
            except Exception:                     # closing must never mask the outcome
                pass
    return None, meta


# --------------------------------------------------------------------------
# Build, seal, validate
# --------------------------------------------------------------------------

def _seal(doc: dict) -> dict:
    doc[CHECKSUM_FIELD] = ps.payload_checksum(doc)
    return doc


def build_capture(*, capture_id: str, resolved: dict, request_started_at: datetime,
                  received_at: datetime, payload, meta: dict, schedule, schedule_sha256: str,
                  code: tuple[str, bool | None], api_key: str | None = None) -> dict:
    scope, games = scope_games(schedule, received_at)
    teams, rows, malformed, error = [], [], [], meta.get("error")
    if error is None:
        try:
            teams, rows, malformed = parse_entries(payload, api_key)
        except ValueError as exc:
            error = {"type": "unexpected_shape", "message": on.redact(exc, api_key)[:MAX_TEXT]}
    match_rows(rows, games)
    coverage = team_coverage(teams, rows, games) if error is None else \
        {t: {"state": "absent_from_feed", "rows": 0} for t in
         sorted({x for g in games for x in (g["home_team"], g["away_team"])})}
    outcome = "unavailable" if error is not None else outcome_for(teams, rows, malformed, coverage)
    doc = {
        "schema_version": SCHEMA_VERSION, "kind": CAPTURE_KIND, "capture_id": capture_id,
        "slot": slot_record(resolved, received_at),
        "request_started_at": _iso_us(request_started_at),
        "received_at": _iso_us(received_at), "observed_at": _iso_us(received_at),
        "source": dict(SOURCE),
        "response": {**{k: meta.get(k) for k in ("http_status", "http_last_modified",
                                                 "http_date", "bytes")},
                     "error": error},
        "source_entry_dates": entry_date_summary(rows),
        "scope": {**(scope or {"season": None, "week": None}), "games": games},
        "outcome": outcome, "usable": outcome in USABLE_OUTCOMES,
        "coverage": {
            "teams_in_feed": len(teams), "player_rows": len(rows),
            "malformed_entries": malformed,
            "unmatched_source_teams": sorted({t["source_team_name"] or "(no name)"
                                              for t in teams if t["team"] is None}),
            "rows_for_teams_without_game_in_scope": sum(
                1 for r in rows if r["game_state"] == "no_game_in_scope"),
            "in_scope_teams": coverage,
            "note": "no rows for a team never means the team has no injuries",
        },
        "status_counts": _counts(r["status_class"] for r in rows),
        "teams": teams, "players": rows,
        "schedule": {"path": ps.SCHEDULE_NAME, "sha256": schedule_sha256},
        "code_revision": code[0], "code_dirty": code[1],
    }
    return _seal(doc)


def entry_date_summary(rows: list[dict]) -> dict:
    """Counts and range of the source entry dates. Their meaning is unverified."""
    raw = [r["source_entry_date_raw"] for r in rows if r["source_entry_date_raw"] is not None]
    utc = [r["source_entry_date_utc"] for r in rows if r["source_entry_date_utc"] is not None]
    return {
        "meaning": ENTRY_DATE_MEANING,
        "rows_with_date": len(raw), "rows_without_date": len(rows) - len(raw),
        "rows_with_unparseable_date": len(raw) - len(utc),
        "earliest_utc": min(utc) if utc else None, "latest_utc": max(utc) if utc else None,
        "note": ("source entry dates are kept as supplied; their meaning is unverified, so they "
                 "are not status-update times or evidence of freshness")
        if raw else ("the source supplied no entry dates; unavailable" if rows else None),
    }


def _counts(values) -> dict:
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


_FIELDS = ("schema_version", "kind", "capture_id", "slot", "request_started_at", "received_at",
           "observed_at", "source", "response", "source_entry_dates", "scope", "outcome", "usable",
           "coverage", "status_counts", "teams", "players", "schedule", "code_revision",
           "code_dirty", CHECKSUM_FIELD)
_ROW_FIELDS = ("source_team_name", "team", "player_name", "player_id", "player_id_source",
               "position", "source_status", "status_class", "source_status_type", "injury",
               "short_comment", "source_return_date", "practice_participation",
               "source_entry_date_raw", "source_entry_date_utc", "game_id", "game_state")
_ID_RE = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{12}$")
_SLOT_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_(wednesday_noon|sunday_morning|ad_hoc)$")


def _need(cond, where, msg):
    if not cond:
        raise ValidationError(f"{where}: {msg}")


def _check_slot(slot, started: datetime, received: datetime, where: str) -> None:
    _need(isinstance(slot, dict) and set(slot) == {
        "name", "slot_id", "timezone", "intended_utc", "intended_local", "delay_minutes",
        "status", "fills_slot", "judged_at"}, where, "slot fields")
    _need(slot["judged_at"] == "received_at" and slot["timezone"] == "America/Toronto",
          where, "slot basis")
    _need(isinstance(slot["slot_id"], str) and _SLOT_ID_RE.match(slot["slot_id"]), where, "slot_id")
    if slot["name"] == on.AD_HOC:
        _need(slot["status"] == on.AD_HOC and slot["fills_slot"] is False
              and slot["intended_utc"] is None and slot["delay_minutes"] is None, where,
              "an ad-hoc capture never claims a slot time")
        _need(slot["slot_id"] == f"{started.astimezone(TORONTO):%Y-%m-%d}_{on.AD_HOC}", where,
              "ad-hoc slot_id must be the Toronto date of the request")
        return
    _need(slot["name"] in on.SLOTS_BY_NAME, where, "unknown slot name")
    rule = on.SLOTS_BY_NAME[slot["name"]]
    intended = on.parse_utc(slot["intended_utc"])
    local = intended.astimezone(TORONTO)
    _need(local.weekday() == rule.weekday and local.time() == rule.local_time, where,
          "intended time isn't the slot's Toronto wall-clock time")
    _need(slot["slot_id"] == f"{local:%Y-%m-%d}_{rule.name}", where, "slot_id/intended mismatch")
    _need(slot["intended_local"] == local.isoformat(), where, "intended_local")
    # Truthfulness: information can't have been observed before the slot opened,
    # and the label must follow from the actual receipt time.
    _need(intended <= started <= received, where,
          "request/receipt before the slot time: not an observation at the slot")
    delay = received - intended
    _need(slot["delay_minutes"] == round(delay.total_seconds() / 60, 1), where, "delay_minutes")
    expected = ("outside_window" if not on.in_slot_window(rule, intended, received) else
                "on_time" if delay <= on.ON_TIME_TOLERANCE else "late")
    _need(slot["status"] == expected, where, f"slot status should be {expected}")
    _need(slot["fills_slot"] is (expected in ("on_time", "late")), where, "fills_slot")


def validate_capture(doc, where: str = "capture") -> None:
    """Schema, checksum and internal consistency. Fails closed: any problem,
    including a malformed nested value, is a ValidationError."""
    try:
        _validate(doc, where)
    except ValidationError:
        raise
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValidationError(f"{where}: malformed capture ({type(exc).__name__}: {exc})") from None


def _validate(doc, where: str) -> None:
    _need(isinstance(doc, dict), where, "not an object")
    _need(doc.get("schema_version") == SCHEMA_VERSION and not isinstance(doc.get("schema_version"), bool),
          where, f"unsupported schema_version {doc.get('schema_version')!r}")
    _need(doc.get("kind") == CAPTURE_KIND, where, "wrong kind")
    _need(set(doc) == set(_FIELDS), where,
          f"fields differ from schema: {sorted(set(doc) ^ set(_FIELDS))}")
    try:
        ok = doc[CHECKSUM_FIELD] == ps.payload_checksum(doc)
    except ValueError:
        ok = False
    _need(ok, where, "checksum mismatch")
    _need(isinstance(doc["capture_id"], str) and _ID_RE.match(doc["capture_id"]), where, "capture_id")
    started = parse_utc_us(doc["request_started_at"])
    received = parse_utc_us(doc["received_at"])
    _need(doc["observed_at"] == doc["received_at"], where, "observed_at must be received_at")
    _need(started <= received, where, "received before the request started")
    _check_slot(doc["slot"], started, received, where)
    _need(doc["capture_id"].startswith(f"{started:%Y%m%dT%H%M%SZ}"), where,
          "capture_id must carry the request start time")
    _need(doc["source"] == SOURCE, where, "source attribution differs")
    _need(doc["outcome"] in OUTCOMES, where, "outcome")
    _need(doc["usable"] is (doc["outcome"] in USABLE_OUTCOMES), where, "usable")
    resp = doc["response"]
    _need(isinstance(resp, dict) and set(resp) == {"http_status", "http_last_modified",
                                                   "http_date", "bytes", "error"}, where, "response")
    _need((resp["error"] is None) is (doc["outcome"] != "unavailable"), where,
          "an unavailable outcome must carry its error, and only it")
    rows, cov = doc["players"], doc["coverage"]
    _need(isinstance(rows, list) and isinstance(doc["teams"], list), where, "rows")
    for i, r in enumerate(rows):
        w = f"{where}: players[{i}]"
        _need(isinstance(r, dict) and set(r) == set(_ROW_FIELDS), w, "row fields")
        _need(isinstance(r["player_name"], str) and r["player_name"], w, "player_name")
        _need(r["status_class"] == status_class(r["source_status"]), w,
              "status_class doesn't follow from the raw status")
        _need(r["practice_participation"] is None, w,
              "this source supplies no practice participation")
        _need(r["player_id_source"] in ID_SOURCES and (r["player_id"] is None) is
              (r["player_id_source"] is None), w, "player id source")
        if r["source_entry_date_utc"] is not None:
            _need(r["source_entry_date_raw"] is not None, w, "entry date UTC without raw text")
            on.parse_utc(r["source_entry_date_utc"])
    _need(cov["player_rows"] == len(rows) and cov["teams_in_feed"] == len(doc["teams"]), where,
          "coverage counts")
    _need(all(c["state"] in TEAM_COVERAGE for c in cov["in_scope_teams"].values()), where,
          "team coverage state")
    _need(doc["status_counts"] == _counts(r["status_class"] for r in rows), where, "status_counts")
    _need(doc["source_entry_dates"] == entry_date_summary(rows), where,
          "source entry dates don't follow from the rows")
    if doc["outcome"] != "unavailable":
        expected = outcome_for(doc["teams"], rows, cov["malformed_entries"], cov["in_scope_teams"])
        _need(doc["outcome"] == expected, where, f"outcome should be {expected}")


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------

def write_capture(doc: dict, directory: Path = CAPTURE_DIR) -> tuple[Path, bool]:
    """Exclusive create of <capture_id>.json; never overwrites. Identical
    content again is a no-op; different content under the same id conflicts."""
    validate_capture(doc, "new capture")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{doc['capture_id']}.json"
    text = json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n"

    def accept_existing():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            validate_capture(existing, str(path))
        except (ValueError, KeyError, TypeError) as exc:
            raise CaptureConflictError(f"{path} exists but isn't a valid capture ({exc}); "
                                       "refusing to overwrite it") from None
        if ps._canonical(existing) != ps._canonical(doc):
            raise CaptureConflictError(f"{path} already exists with different content; "
                                       "refusing to overwrite it")

    if path.exists():
        accept_existing()
        return path, False
    if ps._exclusive_write(path, text):
        return path, True
    accept_existing()
    return path, False


def load_captures(directory: Path = CAPTURE_DIR) -> list[dict]:
    """Every capture, validated; any invalid file fails the whole read."""
    out = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValidationError(f"{path}: not valid JSON ({exc})") from None
        validate_capture(doc, str(path))
        _need(path.stem == doc["capture_id"], str(path), "file name doesn't match capture_id")
        out.append(doc)
    return out


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------

@dataclass
class CaptureResult:
    status: str        # captured | partial | empty | unavailable | already_captured | not_due |
    #                    no_games | window_closed
    message: str
    path: Path | None = None
    doc: dict | None = None
    written: bool = False


def _already_captured(resolved: dict, capture_dir: Path) -> CaptureResult | None:
    """already_captured if a usable capture fills this scheduled slot. Reads
    (and so validates) the whole store first, for ad-hoc runs too: an invalid
    store stops every capture before any request."""
    docs = load_captures(capture_dir)
    if resolved["name"] == on.AD_HOC:
        return None
    for doc in docs:
        if doc["slot"]["slot_id"] == resolved["slot_id"] and doc["usable"] and doc["slot"]["fills_slot"]:
            return CaptureResult("already_captured",
                                 f"injury report for slot {resolved['slot_id']} already captured "
                                 f"({doc['capture_id']}); no request made")
    return None


def capture(*, now: datetime | None = None, slot: str = "auto", clock: Callable = ps.utc_now,
            get: Callable = requests.get, capture_dir: Path = CAPTURE_DIR,
            schedule_path: Path = SCHEDULE_PATH, repo_dir: Path = ROOT) -> CaptureResult:
    """One injury-report capture for the due slot (or ad hoc).

    `now` is when the run starts (default: clock()); the request start and
    receipt times come from `clock`, never from the slot."""
    now = now or clock()
    resolved = on.resolve_slot(now, slot)
    if resolved is None:
        return CaptureResult("not_due", f"no capture slot is due at {iso(now)} "
                                        f"({now.astimezone(TORONTO):%a %H:%M} Toronto); "
                                        "no request made")
    done = _already_captured(resolved, capture_dir)
    if done:
        return done
    schedule, schedule_sha = on.load_schedule(schedule_path)
    if scope_games(schedule, now)[0] is None:
        return CaptureResult("no_games", "no upcoming games in scope; no request made")
    with on.SlotLock(capture_dir, resolved["slot_id"], "injury-" + on.new_id(now),
                     what="injury capture"):
        done = _already_captured(resolved, capture_dir)
        if done:
            return done
        # Last check, immediately before the request: schedule loading and the
        # lock take time, and the slot's window may have closed meanwhile.
        started = max(now, clock())
        if resolved["name"] == on.AD_HOC:      # dated by the actual request start
            resolved = on.resolve_slot(started, on.AD_HOC)
        else:
            rule = on.SLOTS_BY_NAME[resolved["name"]]
            intended = on.parse_utc(resolved["intended_utc"])
            if not on.in_slot_window(rule, intended, started):
                closes = on.slot_window(rule, intended)[1]
                return CaptureResult(
                    "window_closed",
                    f"slot {resolved['slot_id']} window closed at {iso(closes)} "
                    f"({closes.astimezone(TORONTO):%H:%M} Toronto) before the request could start "
                    f"({_iso_us(started)}); no request made")
        payload, meta = fetch(get)
        received = max(started, clock())
        doc = build_capture(capture_id=on.new_id(started), resolved=resolved,
                            request_started_at=started, received_at=received, payload=payload,
                            meta=meta, schedule=schedule, schedule_sha256=schedule_sha,
                            code=ps.code_revision(repo_dir))
        path, created = write_capture(doc, capture_dir)
    slot_text = f"slot {doc['slot']['slot_id']} ({doc['slot']['status']})"
    if doc["outcome"] == "unavailable":
        err = doc["response"]["error"]
        return CaptureResult("unavailable", f"injury source unavailable for {slot_text}: "
                                            f"{err['type']}: {err['message']}; kept as {path.name}",
                             path, doc, created)
    if doc["outcome"] == "empty":
        return CaptureResult("empty", f"injury source returned no player rows for {slot_text}; "
                                      f"kept as {path.name}", path, doc, created)
    status = "captured" if doc["outcome"] == "full_team_coverage" else "partial"
    note = "" if doc["slot"]["fills_slot"] or doc["slot"]["name"] == on.AD_HOC else \
        " - received after the slot window closed, so it does not fill the slot"
    return CaptureResult(status, f"{doc['outcome']} for {slot_text}: "
                                 f"{doc['coverage']['player_rows']} rows from "
                                 f"{doc['coverage']['teams_in_feed']} teams as {path.name}{note}",
                         path, doc, created)


# --------------------------------------------------------------------------
# Listing
# --------------------------------------------------------------------------

def _toronto(text: str) -> str:
    return f"{parse_utc_us(text).astimezone(TORONTO):%a %Y-%m-%d %H:%M:%S %Z}"


def describe(doc: dict) -> list[str]:
    s, cov, scope = doc["slot"], doc["coverage"], doc["scope"]
    states = _counts(c["state"] for c in cov["in_scope_teams"].values())
    ed = doc["source_entry_dates"]
    source_time = (f"source entry dates (meaning unverified) {ed['earliest_utc']} to "
                   f"{ed['latest_utc']}; {ed['rows_without_date']} rows without one"
                   if ed["latest_utc"] else
                   "source entry dates unavailable" if not ed["rows_with_date"] else
                   "source entry dates present but unparseable (meaning unverified)")
    err = doc["response"]["error"]
    return [
        f"{doc['capture_id']}  {s['slot_id']}  {s['status']}"
        + (f" (+{s['delay_minutes']} min)" if s["delay_minutes"] is not None else "")
        + ("" if s["fills_slot"] or s["name"] == on.AD_HOC else "  [does not fill slot]"),
        f"    observed (received) {doc['received_at']} = {_toronto(doc['received_at'])}",
        f"    source {doc['source']['publisher']} {doc['source']['name']}; {source_time}",
        f"    scope season {scope['season']} week {scope['week']}, {len(scope['games'])} games; "
        f"outcome {doc['outcome']}"
        + (f" ({err['type']}: {err['message']})" if err else ""),
        f"    {cov['player_rows']} rows, {cov['teams_in_feed']} teams in feed; in-scope teams "
        f"{states}; unmatched feed teams {cov['unmatched_source_teams'] or 'none'}; "
        f"malformed entries {len(cov['malformed_entries'])}",
        f"    status classes {doc['status_counts']}",
    ]


# --------------------------------------------------------------------------
# Persistence check (workflow)
# --------------------------------------------------------------------------

GIT_TIMEOUT = 60


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_bytes(repo: Path, *args) -> bytes:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True,
                          timeout=GIT_TIMEOUT).stdout


def _blob_state(repo: Path, rev: str, rel: str, sha256: str) -> str:
    """identical | content_differs | absent, for `rel` at `rev`."""
    try:
        data = _git_bytes(repo, "show", f"{rev}:{rel}")
    except subprocess.CalledProcessError:
        return "absent"
    return "identical" if hashlib.sha256(data).hexdigest() == sha256 else "content_differs"


def verify_persistence(*, repo_dir: Path, rel_path: str, sha256: str, upload: str,
                       artifact: str, remote: str = "origin", branch: str = "main") -> dict:
    """Where this run's capture file actually is.

    Checks remote `branch` (freshly fetched) for the file with identical
    content, then the local HEAD, then the working tree, and whether the
    artifact upload succeeded. Only an identical file on the remote is
    'persisted_remotely'; if the remote can't be read, persistence is
    'remote_unverified', never assumed."""
    repo = Path(repo_dir)
    out = {"file": rel_path, "sha256": sha256, "artifact": artifact,
           "artifact_uploaded": upload == "success", "remote": None, "remote_error": None,
           "local_commit": None, "working_tree": None}
    try:
        _git_bytes(repo, "fetch", "--quiet", "--no-tags", remote, branch)
        rev = _git_bytes(repo, "rev-parse", "FETCH_HEAD").decode().strip()
        out["remote"] = _blob_state(repo, rev, rel_path, sha256)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        detail = getattr(exc, "stderr", b"") or b""
        detail = detail.decode("utf-8", "replace") if isinstance(detail, bytes) else str(detail)
        out["remote"] = "unverified"
        out["remote_error"] = on.redact(f"{type(exc).__name__}: {detail.strip() or exc}")[:MAX_TEXT]
    try:
        out["local_commit"] = _blob_state(repo, "HEAD", rel_path, sha256)
    except (subprocess.TimeoutExpired, OSError):
        out["local_commit"] = "unknown"
    path = repo / rel_path
    out["working_tree"] = ("identical" if file_sha256(path) == sha256 else "content_differs") \
        if path.is_file() else "absent"
    if out["remote"] == "identical":
        out["state"] = "persisted_remotely"
    elif out["remote"] == "unverified":
        out["state"] = "remote_unverified"
    elif out["remote"] == "content_differs":
        out["state"] = "remote_content_differs"
    elif out["local_commit"] == "identical":
        out["state"] = "committed_locally_not_pushed"
    elif out["working_tree"] == "identical":
        out["state"] = "uncommitted"
    else:
        out["state"] = "missing_locally"
    out["persisted"] = out["state"] == "persisted_remotely"
    return out


_STATE_TEXT = {
    "persisted_remotely": "persisted: on remote main with identical content",
    "remote_unverified": "persistence UNVERIFIED: remote main could not be read",
    "remote_content_differs": "NOT persisted: remote main has this path with different content",
    "committed_locally_not_pushed": "NOT persisted: committed locally but not on remote main "
                                    "(push failed or skipped)",
    "uncommitted": "NOT persisted: never committed",
    "missing_locally": "NOT persisted: the file is no longer in the runner's checkout",
}


def persistence_lines(v: dict) -> list[str]:
    lines = ["### Injury capture persistence",
             f"- file: `{v['file']}` (sha256 {v['sha256'][:12]}...)",
             f"- {_STATE_TEXT[v['state']]}"
             + (" ✅" if v["persisted"] else " ❌"),
             f"- remote main: {v['remote']}; local commit: {v['local_commit']}; "
             f"working tree: {v['working_tree']}"]
    if v["remote_error"]:
        lines.append(f"- remote check error: {v['remote_error']}")
    if not v["persisted"]:
        if v["artifact_uploaded"]:
            lines.append(f"- a copy is in workflow artifact **{v['artifact']}** (kept 90 days); "
                         "re-commit it by hand")
        else:
            lines.append("- the artifact upload did not succeed either: this capture is **not "
                         "persisted anywhere** and is lost when the runner ends")
    return lines


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

EXIT_OK, EXIT_FAILED = 0, 1


def _summary(lines: list[str]) -> None:
    on._summary(lines)


def _output(**kv) -> None:
    on._output(**kv)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Immutable injury-report captures (evidence only)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="capture the due slot's injury report (no request if not due)")
    c.add_argument("--slot", default="auto", choices=["auto", *on.SLOTS_BY_NAME, on.AD_HOC])
    lst = sub.add_parser("list", help="validate and list captures (times, source, scope, coverage)")
    lst.add_argument("--last", type=int, default=None, help="only the N most recent captures")
    sub.add_parser("validate", help="validate every stored capture")
    pr = sub.add_parser("persistence", help="(workflow) report where this run's capture file is")
    pr.add_argument("--status", default="", help="the capture step's status output")
    pr.add_argument("--written", default="", help="the capture step's written output")
    pr.add_argument("--file", default="", help="repo-relative path of the file this run wrote")
    pr.add_argument("--sha256", default="", help="sha256 of that file as written")
    pr.add_argument("--upload", default="", help="outcome of the artifact upload step")
    pr.add_argument("--artifact", default="", help="artifact name")
    pr.add_argument("--repo", default=".", help="repository directory")
    args = ap.parse_args(argv)

    if args.cmd == "persistence":
        return _persistence_cli(args)

    if args.cmd == "capture":
        try:
            result = capture(slot=args.slot)
        except (on.CaptureError, CaptureConflictError, ValueError, KeyError, TypeError,
                OSError) as exc:
            msg = on.redact(exc)[:MAX_TEXT]
            print(f"::error::Injury report capture failed: {msg}")
            _output(status="failed", written="false")
            _summary(["### Injury report capture (evidence only)", "- status: **failed** ❌",
                      f"- {msg}", "- sportsbook capture steps are unaffected"])
            return EXIT_FAILED
        print(f"[injury_reports] {result.status}: {result.message}")
        out = {"status": result.status, "written": "true" if result.written else "false"}
        if result.written:
            out.update(file=result.path.resolve().relative_to(ROOT.resolve()).as_posix()
                       if result.path.resolve().is_relative_to(ROOT.resolve())
                       else result.path.as_posix(),
                       sha256=file_sha256(result.path))
        _output(**out)
        lines = ["### Injury report capture (evidence only)", f"- status: **{result.status}**",
                 f"- {result.message}"]
        if result.doc:
            lines += [f"  {x}" for x in describe(result.doc)]
        if result.status in ("unavailable", "empty"):
            print(f"::error::Injury report capture {result.status}: {result.message}")
            lines[1] += " ❌"
            lines.append("- sportsbook capture steps are unaffected; the slot stays open for a "
                         "later run inside its window")
            _summary(lines)
            return EXIT_FAILED
        if result.status == "window_closed":
            print(f"::error::Injury report not captured: {result.message}")
            lines[1] += " ❌"
            _summary(lines)
            return EXIT_FAILED
        if result.status == "partial":
            print(f"::warning::Injury report coverage is partial: {result.message}")
            lines[1] += " ⚠️"
        _summary(lines)
        return EXIT_OK
    try:
        docs = load_captures(CAPTURE_DIR)
    except (ValueError, OSError) as exc:
        print(f"::error::Injury report captures failed validation: {on.redact(exc)[:MAX_TEXT]}")
        return EXIT_FAILED
    if args.cmd == "validate":
        print(f"[injury_reports] {len(docs)} capture(s) valid")
        return EXIT_OK
    docs.sort(key=lambda d: d["received_at"])
    if args.last is not None:
        docs = docs[-args.last:] if args.last > 0 else []
    print(f"{len(docs)} injury-report capture(s), all valid; times in UTC and America/Toronto")
    for d in docs:
        print("\n".join(describe(d)))
    return EXIT_OK


def _persistence_cli(args) -> int:
    if not args.status:
        print("::error::injury capture step ended without a status")
        _summary(["### Injury capture persistence",
                  "- the injury capture step ended without a status ❌"])
        return EXIT_FAILED
    if args.written != "true":
        _summary(["### Injury capture persistence",
                  f"- no new injury capture file this run (status {args.status})"])
        return EXIT_OK
    if not args.file or not re.fullmatch(r"[0-9a-f]{64}", args.sha256 or ""):
        print("::error::injury capture wrote a file but its path or sha256 is missing")
        _summary(["### Injury capture persistence",
                  "- the capture wrote a file but its path or sha256 is missing; persistence "
                  "UNVERIFIED ❌"])
        return EXIT_FAILED
    v = verify_persistence(repo_dir=Path(args.repo), rel_path=args.file, sha256=args.sha256,
                           upload=args.upload, artifact=args.artifact)
    lines = persistence_lines(v)
    _summary(lines)
    print("\n".join(lines))
    if v["persisted"]:
        return EXIT_OK
    where = (f"only in workflow artifact {args.artifact}" if v["artifact_uploaded"]
             else "not persisted anywhere")
    print(f"::error::Injury capture {args.file}: {_STATE_TEXT[v['state']]}; {where}")
    return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
