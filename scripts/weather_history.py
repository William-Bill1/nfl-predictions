"""Weather-history research pilot: manually captured hourly forecasts in a separate SQLite file.

Research only. Nothing in the app, the probability engine or the pipeline reads
this. It keeps Open-Meteo hourly *forecasts* captured for explicitly selected
games, so you can ask what had been captured by a decision time and what
changed between two captures. A forecast is not an observation of stadium
conditions; this pilot stores no observed weather.

Commands
  capture  one manual forecast request for one exact game_id, using a reviewed
           venue file (no default location, no home-team join, no fuzzy match).
           Writes <capture_id>.json (metadata) + <capture_id>.response (exact bytes).
  import   explicitly selected capture metadata files, in one transaction.
  list     imported captures (optionally only those received by --as-of).
  asof     the latest usable capture of a game received by a cutoff.
  compare  two captures of the same game at matching valid hours and units.
  sql      one read-only query (the football pilot's guarded connection).

Times are kept apart and never substituted for each other:
  received_at_utc        when this system received the response (capture time)
  valid_time_utc         the forecast hour a value is for
  provider_issue_time    null: the forecast response supplies no model run /
                         issue time. generationtime_ms is processing duration.
See docs/WEATHER_HISTORY_PILOT.md.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import secrets
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import football_repository as fr  # noqa: E402  (read-only connection + authorizer)
import injury_reports as ir  # noqa: E402  (bounded body reader)
import pregame_snapshots as ps  # noqa: E402  (kickoff conversion, checksum, code revision)

TRANSFORMATION_VERSION = "weather-history-pilot-1"
CAPTURE_KIND = "weather_forecast_capture"
SCHEMA_VERSION = 1
VENUES_VERSION = 1
CHECKSUM_FIELD = ps.CHECKSUM_FIELD
DEFAULT_SCHEDULE = ROOT / "data_files" / ps.SCHEDULE_NAME
DEFAULT_CAPTURE_DIR = ROOT / "reports" / "football_repository_pilot" / "weather_history" / "captures"

ENDPOINT = "https://api.open-meteo.com/v1/forecast"
USER_AGENT = "nfl-predictions-weather-history-pilot"
REQUEST_TIMEOUT = (5, 20)
READ_DEADLINE = 30.0
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_TEXT = 500
HOURS_BEFORE_KICKOFF, HOURS_AFTER_KICKOFF = 3, 4

VARIABLES = ("temperature_2m", "wind_speed_10m", "wind_gusts_10m", "precipitation",
             "precipitation_probability")
# Meaning per Open-Meteo's documentation (https://open-meteo.com/en/docs, checked 2026-11-14).
VARIABLE_MEANING = {
    "temperature_2m": "air temperature at 2 m, instantaneous at the valid time",
    "wind_speed_10m": "sustained wind speed at 10 m, instantaneous at the valid time",
    "wind_gusts_10m": "maximum gust at 10 m during the hour preceding the valid time",
    "precipitation": "total precipitation summed over the hour preceding the valid time",
    "precipitation_probability": "probability of more than 0.1 mm in the hour preceding the valid time",
}
UNIT_PARAMS = {"temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "mm",
               "timezone": "GMT", "timeformat": "iso8601"}
# Returned unit strings accepted as matching the requested units. Anything else
# makes the capture unusable (units_unexpected) rather than being converted.
EXPECTED_UNITS = {"temperature_2m": ("°F",), "wind_speed_10m": ("mp/h", "mph"),
                  "wind_gusts_10m": ("mp/h", "mph"), "precipitation": ("mm",),
                  "precipitation_probability": ("%",)}

SOURCE = {
    "name": "open_meteo_forecast",
    "publisher": "Open-Meteo",
    "endpoint": ENDPOINT,
    "attribution": "Weather data by Open-Meteo.com",
    "licence": "CC BY 4.0",
    "licence_url": "https://open-meteo.com/en/license",
    "terms_url": "https://open-meteo.com/en/terms",
    "use": "free non-commercial tier; rate limited; availability not guaranteed",
    "nature": "model forecast for a model grid cell; not measured stadium conditions",
}
ISSUE_TIME_BASIS = ("not supplied: the forecast response has no model run or issue time; "
                    "generationtime_ms is server processing duration, not an issue timestamp")
KICKOFF_VALUE = "not interpolated: only the provider's hourly valid times are stored and compared"

USABLE_OUTCOMES = ("ok", "ok_missing_variables")
FAILED_OUTCOMES = ("http_error", "transport_error", "timeout", "response_too_large",
                   "malformed_response", "units_unexpected")
# Refusals before any request: no request is made and no capture file is written.
PRE_REQUEST_OUTCOMES = ("game_not_found", "game_ambiguous", "game_completed", "kickoff_unusable",
                        "game_started", "venue_unresolved", "venue_conflict", "outside_pilot_coverage")
FIELD_EXPOSURES = ("outdoor", "fixed_roof", "retractable_roof", "open_sided_fixed_roof")
LIMITATIONS = (
    "captured forecasts, not observed weather; values are for a model grid cell near the venue",
    "no provider issue time is available; age is shown since capture, not since issuance",
    "gusts and precipitation describe the preceding hour, not an instant at kickoff",
    "no kickoff value is interpolated; no daily averages are computed",
    "only reviewed outdoor venues are covered; roofed and retractable venues are outside the pilot",
    "no betting or probability impact is claimed or computed",
)

_TIME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})(Z|\+00:00)?$")

SCHEMA = """
CREATE TABLE weather_captures (
    capture_import_id           INTEGER PRIMARY KEY,
    capture_id                  TEXT NOT NULL,
    payload_sha256              TEXT NOT NULL,
    transformation_version      TEXT NOT NULL,
    metadata_file_name          TEXT NOT NULL,
    metadata_path               TEXT NOT NULL,
    metadata_sha256             TEXT NOT NULL,
    raw_response_file           TEXT,
    raw_response_sha256         TEXT,
    raw_response_bytes          INTEGER,
    raw_response                BLOB,
    imported_at_utc             TEXT NOT NULL,
    request_started_at_utc      TEXT NOT NULL,
    received_at_utc             TEXT NOT NULL,
    provider_issue_time_utc     TEXT,
    provider_issue_time_basis   TEXT NOT NULL,
    provider_generationtime_ms  REAL,
    game_id                     TEXT NOT NULL,
    season                      TEXT,
    week                        TEXT,
    home_team                   TEXT,
    away_team                   TEXT,
    schedule_gameday            TEXT,
    schedule_gametime           TEXT,
    kickoff_utc                 TEXT NOT NULL,
    schedule_path               TEXT NOT NULL,
    schedule_sha256             TEXT NOT NULL,
    stadium_id                  TEXT NOT NULL,
    stadium                     TEXT NOT NULL,
    schedule_roof               TEXT,
    venue_json                  TEXT NOT NULL,
    venues_file_sha256          TEXT NOT NULL,
    field_exposure              TEXT NOT NULL,
    requested_latitude          REAL NOT NULL,
    requested_longitude         REAL NOT NULL,
    returned_latitude           REAL,
    returned_longitude          REAL,
    returned_elevation          REAL,
    returned_timezone           TEXT,
    returned_utc_offset_seconds INTEGER,
    request_url                 TEXT NOT NULL,
    request_params_json         TEXT NOT NULL,
    http_status                 INTEGER,
    http_date                   TEXT,
    response_error_json         TEXT,
    outcome                     TEXT NOT NULL,
    usable                      INTEGER NOT NULL,
    outcome_detail              TEXT,
    missing_variables_json      TEXT NOT NULL,
    missing_requested_hours_json TEXT NOT NULL,
    hourly_units_json           TEXT NOT NULL,
    source_json                 TEXT NOT NULL,
    code_revision               TEXT,
    code_dirty                  INTEGER,
    UNIQUE (capture_id, transformation_version)
);
CREATE TABLE weather_forecast_hours (
    capture_import_id         INTEGER NOT NULL REFERENCES weather_captures(capture_import_id),
    valid_time_utc            TEXT NOT NULL,
    hour_index                INTEGER NOT NULL,
    valid_time_raw            TEXT NOT NULL,
    temperature_2m            REAL,
    wind_speed_10m            REAL,
    wind_gusts_10m            REAL,
    precipitation             REAL,
    precipitation_probability REAL,
    PRIMARY KEY (capture_import_id, valid_time_utc)
);
"""


class ImportRefused(ValueError):
    """The selected captures can't be imported. Nothing was written."""


class CompareRefused(ValueError):
    """The comparison can't be made as requested."""


class VenuesRefused(ValueError):
    """The reviewed-venue file is invalid; no request was made."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def ts(dt: datetime) -> str:
    """Canonical stored UTC timestamp, to the microsecond (cutoffs compare exactly)."""
    if dt.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_ts(text: str) -> datetime:
    dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"{text!r}: a timezone is required (e.g. 2026-11-15T12:00:00Z)")
    return dt.astimezone(timezone.utc)


def _dumps(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rows(conn, sql, params=()) -> list[dict]:
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


# --------------------------------------------------------------------------
# Reviewed venues
# --------------------------------------------------------------------------

def load_venues(path) -> tuple[list[dict], str]:
    """Reviewed venue records and the sha256 of the bytes parsed. Each record
    needs the exact schedule stadium_id and stadium name, coordinates, field
    exposure, roof context, at least one reference and a review date."""
    data = Path(path).read_bytes()
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise VenuesRefused(f"{path}: not valid UTF-8 JSON ({exc})") from None
    if not isinstance(doc, dict) or doc.get("venues_version") != VENUES_VERSION \
            or not isinstance(doc.get("venues"), list):
        raise VenuesRefused(f"{path}: expected {{'venues_version': {VENUES_VERSION}, 'venues': [...]}}")
    seen = set()
    for i, v in enumerate(doc["venues"]):
        where = f"{path}: venues[{i}]"
        if not isinstance(v, dict):
            raise VenuesRefused(f"{where}: not an object")
        for key in ("stadium_id", "stadium", "roof_context", "coordinate_basis", "reviewed_on"):
            if not isinstance(v.get(key), str) or not v[key].strip():
                raise VenuesRefused(f"{where}: {key} must be a non-empty string")
        lat, lon = v.get("latitude"), v.get("longitude")
        if not (_number(lat) and -90 <= lat <= 90 and _number(lon) and -180 <= lon <= 180):
            raise VenuesRefused(f"{where}: latitude/longitude must be finite decimal degrees")
        if v.get("field_exposure") not in FIELD_EXPOSURES:
            raise VenuesRefused(f"{where}: field_exposure must be one of {FIELD_EXPOSURES}")
        refs = v.get("references")
        if not isinstance(refs, list) or not refs or not all(
                isinstance(r, dict) and isinstance(r.get("url"), str) and r["url"].startswith("https://")
                and isinstance(r.get("accessed_on"), str) for r in refs):
            raise VenuesRefused(f"{where}: references must list at least one {{url (https), accessed_on}}")
        try:
            date.fromisoformat(v["reviewed_on"])
        except ValueError:
            raise VenuesRefused(f"{where}: reviewed_on must be YYYY-MM-DD") from None
        if v["stadium_id"] in seen:
            raise VenuesRefused(f"{where}: stadium_id {v['stadium_id']} listed twice")
        seen.add(v["stadium_id"])
    return doc["venues"], _sha(data)


# --------------------------------------------------------------------------
# Game selection
# --------------------------------------------------------------------------

def select_game(schedule_bytes: bytes, game_id: str) -> tuple[dict | None, str | None]:
    """(row as text, None) for exactly one row with this game_id, else (None, outcome)."""
    df = pd.read_csv(io.BytesIO(schedule_bytes), sep="\t", dtype=str, keep_default_na=False)
    rows = df[df["game_id"] == game_id]
    if len(rows) == 0:
        return None, "game_not_found"
    if len(rows) > 1:
        return None, "game_ambiguous"
    return rows.iloc[0].to_dict(), None


def resolve(game: dict, venues: list[dict], now: datetime) -> tuple[dict, str | None, str]:
    """(context, refusal outcome or None, detail). No request happens on a refusal."""
    kickoff, problem = ps.kickoff_utc(game.get("gameday") or None, game.get("gametime") or None)
    ctx = {"game_id": game["game_id"], "season": game.get("season"), "week": game.get("week"),
           "home_team": game.get("home_team"), "away_team": game.get("away_team"),
           "schedule_gameday": game.get("gameday"), "schedule_gametime": game.get("gametime"),
           "schedule_time_zone": str(ps.SOURCE_TZ), "kickoff_utc": None if kickoff is None else ts(kickoff),
           "stadium_id": game.get("stadium_id"), "stadium": game.get("stadium"),
           "schedule_roof": game.get("roof") or None, "schedule_location": game.get("location") or None}
    if (game.get("home_score") or "").strip() and (game.get("away_score") or "").strip():
        return ctx, "game_completed", "the schedule lists a final score"
    if kickoff is None:
        return ctx, "kickoff_unusable", problem
    if kickoff <= now:
        return ctx, "game_started", f"kickoff {ts(kickoff)} is not after {ts(now)}"
    matches = [v for v in venues if v["stadium_id"] == game.get("stadium_id")]
    if not matches:
        return ctx, "venue_unresolved", f"no reviewed venue record for stadium_id {game.get('stadium_id')!r}"
    venue = matches[0]
    ctx["venue"] = venue
    if venue["stadium"] != game.get("stadium"):
        return ctx, "venue_unresolved", (f"reviewed record names {venue['stadium']!r} but the schedule "
                                         f"names {game.get('stadium')!r}; not matched")
    if venue["field_exposure"] != "outdoor":
        return ctx, "outside_pilot_coverage", (f"{venue['stadium']} is {venue['field_exposure']}; outdoor "
                                               "forecasts are not field conditions and roof state is not assumed")
    if (game.get("roof") or "") != "outdoors":
        return ctx, "venue_conflict", (f"reviewed record says outdoor but the schedule roof is "
                                       f"{game.get('roof') or 'missing'!r}; not requested")
    return ctx, None, "reviewed outdoor venue"


def request_params(venue: dict, kickoff: datetime) -> dict:
    base = kickoff.replace(minute=0, second=0, microsecond=0)
    fmt = "%Y-%m-%dT%H:%M"
    return {"latitude": venue["latitude"], "longitude": venue["longitude"],
            "hourly": ",".join(VARIABLES), **UNIT_PARAMS,
            "start_hour": (base - timedelta(hours=HOURS_BEFORE_KICKOFF)).strftime(fmt),
            "end_hour": (base + timedelta(hours=HOURS_AFTER_KICKOFF)).strftime(fmt)}


# --------------------------------------------------------------------------
# Request and response
# --------------------------------------------------------------------------

def fetch(params: dict, get: Callable = requests.get, *, limit: int = MAX_RESPONSE_BYTES,
          deadline: float = READ_DEADLINE, monotonic: Callable = time.monotonic
          ) -> tuple[bytes | None, dict]:
    """(exact body bytes or None, metadata). Never raises for network problems."""
    meta = {"http_status": None, "http_date": None, "content_type": None, "url": None, "error": None}
    resp = None
    try:
        resp = get(ENDPOINT, params=params, headers={"User-Agent": USER_AGENT},
                   timeout=REQUEST_TIMEOUT, stream=True)
        meta["http_status"] = resp.status_code
        meta["url"] = getattr(resp, "url", None)
        headers = getattr(resp, "headers", {}) or {}
        meta["http_date"] = headers.get("Date")
        meta["content_type"] = headers.get("Content-Type")
        return ir._read_bounded(resp, limit, deadline, monotonic), meta
    except ir._TooLarge as exc:
        meta["error"] = {"type": "too_large", "message": f"over {limit} bytes ({exc.args[0]} read)"}
    except ir._Deadline as exc:
        meta["error"] = {"type": "timeout", "message": f"body incomplete after {deadline:g} s "
                                                       f"({exc.args[0]} bytes read)"}
    except requests.Timeout as exc:
        meta["error"] = {"type": "timeout", "message": str(exc)[:MAX_TEXT]}
    except requests.RequestException as exc:
        meta["error"] = {"type": "request_error", "message": str(exc)[:MAX_TEXT]}
    finally:
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
    return None, meta


def _expected_hours(params: dict) -> list[str]:
    start = datetime.strptime(params["start_hour"], "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
    end = datetime.strptime(params["end_hour"], "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
    out = []
    while start <= end:
        out.append(start.strftime("%Y-%m-%dT%H:%M:%SZ"))
        start += timedelta(hours=1)
    return out


def parse_response(http_status, body: bytes | None, error: dict | None, params: dict) -> dict:
    """Deterministic interpretation of one response. Failures carry no hours;
    missing values stay None; nothing is defaulted or converted."""
    out = {"outcome": None, "usable": False, "detail": None, "hours": [], "units": {},
           "missing_variables": [], "missing_requested_hours": [], "returned": {}}

    def fail(outcome, detail):
        out.update(outcome=outcome, detail=str(detail)[:MAX_TEXT], hours=[], usable=False)
        return out

    if error is not None:
        kind = {"too_large": "response_too_large", "timeout": "timeout"}.get(error.get("type"),
                                                                             "transport_error")
        return fail(kind, error.get("message"))
    if body is None:
        return fail("transport_error", "no response body")
    if http_status != 200:
        reason = None
        try:
            doc = json.loads(body.decode("utf-8"))
            reason = doc.get("reason") if isinstance(doc, dict) else None
        except (UnicodeDecodeError, ValueError):
            pass
        return fail("http_error", f"HTTP {http_status}" + (f": {reason}" if reason else ""))
    try:
        doc = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        return fail("malformed_response", f"not complete JSON (truncated or invalid): {exc}")
    if not isinstance(doc, dict):
        return fail("malformed_response", "response is not a JSON object")
    if doc.get("error") is True:
        return fail("http_error", f"provider error: {doc.get('reason')}")
    returned = {k: doc.get(k) for k in ("latitude", "longitude", "elevation", "timezone",
                                        "utc_offset_seconds", "generationtime_ms")}
    out["returned"] = returned
    if not (_number(returned["latitude"]) and _number(returned["longitude"])):
        return fail("malformed_response", "returned latitude/longitude missing")
    if returned["utc_offset_seconds"] != 0 or returned["timezone"] not in ("GMT", "UTC"):
        return fail("malformed_response", f"times not in UTC (timezone {returned['timezone']!r}, "
                                          f"offset {returned['utc_offset_seconds']!r})")
    hourly, units = doc.get("hourly"), doc.get("hourly_units")
    if not isinstance(hourly, dict) or not isinstance(units, dict):
        return fail("malformed_response", "hourly or hourly_units missing")
    if units.get("time") != "iso8601" or not isinstance(hourly.get("time"), list) or not hourly["time"]:
        return fail("malformed_response", "hourly time array missing, empty or not iso8601")
    times = []
    for raw in hourly["time"]:
        m = _TIME_RE.match(raw) if isinstance(raw, str) else None
        if not m or not m.group(1).endswith(":00"):
            return fail("malformed_response", f"unexpected valid time {raw!r}")
        times.append(m.group(1) + ":00Z")
    if times != sorted(set(times)):
        return fail("malformed_response", "valid times repeated or out of order")
    present = []
    for var in VARIABLES:
        in_h, in_u = var in hourly, var in units
        if not in_h and not in_u:
            out["missing_variables"].append(var)
            continue
        if in_h != in_u:
            return fail("malformed_response", f"{var} has values or units but not both")
        if not isinstance(hourly[var], list) or len(hourly[var]) != len(times):
            return fail("malformed_response", f"{var} is not aligned with the time array")
        if not all(v is None or _number(v) for v in hourly[var]):
            return fail("malformed_response", f"{var} has a non-numeric value")
        out["units"][var] = units[var]
        present.append(var)
    if not present:
        return fail("malformed_response", "none of the requested variables were returned")
    bad = {v: out["units"][v] for v in present if out["units"][v] not in EXPECTED_UNITS[v]}
    if bad:
        return fail("units_unexpected", f"returned units {bad} don't match the requested units")
    for i, (raw, valid) in enumerate(zip(hourly["time"], times)):
        row = {"hour_index": i, "valid_time_raw": raw, "valid_time_utc": valid}
        for var in VARIABLES:
            row[var] = hourly[var][i] if var in present else None
        out["hours"].append(row)
    out["missing_requested_hours"] = [h for h in _expected_hours(params) if h not in set(times)]
    out["usable"] = True
    out["outcome"] = "ok" if not out["missing_variables"] else "ok_missing_variables"
    out["detail"] = None if not out["missing_variables"] else \
        f"not returned: {', '.join(out['missing_variables'])}"
    return out


# --------------------------------------------------------------------------
# Capture (one manual request for one game)
# --------------------------------------------------------------------------

def _summary(parsed: dict) -> dict:
    return {"outcome": parsed["outcome"], "usable": parsed["usable"], "detail": parsed["detail"],
            "hours": len(parsed["hours"]), "units": parsed["units"],
            "missing_variables": parsed["missing_variables"],
            "missing_requested_hours": parsed["missing_requested_hours"], "returned": parsed["returned"]}


ID_ATTEMPTS = 5


def _attempt_token() -> str:
    """48 random bits per attempt (monkeypatched in tests)."""
    return secrets.token_hex(6)


def new_capture_id(received: datetime, game_id: str) -> str:
    """Unique per attempt: receipt time to the microsecond + 48 random bits.
    Never derived from the response, so identical bodies (or two failures with
    no body) received in the same second still get distinct identities."""
    return f"{received.astimezone(timezone.utc):%Y%m%dT%H%M%S%f}Z-{game_id}-{_attempt_token()}"


def _write_new(out_dir: Path, received: datetime, game_id: str, body: bytes | None,
               build: Callable[[str, str | None], dict]) -> tuple[str, str | None]:
    """Write the metadata (and raw body) under a fresh capture_id with exclusive
    creation. The metadata file is created first and reserves the identity; if
    the name already exists a new identity is drawn, so an earlier attempt's
    files are never opened for writing."""
    for _ in range(ID_ATTEMPTS):
        capture_id = new_capture_id(received, game_id)
        raw_name = f"{capture_id}.response" if body is not None else None
        try:
            meta = open(out_dir / f"{capture_id}.json", "x", encoding="utf-8", newline="\n")
        except FileExistsError:
            continue
        with meta:
            if body is not None:
                with open(out_dir / raw_name, "xb") as fh:      # never overwrite
                    fh.write(body)
            meta.write(json.dumps(build(capture_id, raw_name), indent=1, ensure_ascii=False,
                                  allow_nan=False) + "\n")
        return capture_id, raw_name
    raise FileExistsError(f"no unused capture_id after {ID_ATTEMPTS} attempts in {out_dir}")


def capture(game_id: str, *, venues_path, schedule_path=DEFAULT_SCHEDULE, out_dir=DEFAULT_CAPTURE_DIR,
            get: Callable = requests.get, clock: Callable = ps.utc_now, repo_dir: Path = ROOT,
            tls_trust: str = "not recorded") -> dict:
    """One manual capture. Returns a result dict with 'outcome'. Pre-request
    refusals make no request and write nothing; any response (or failure) after a
    request is written: metadata JSON plus the exact body bytes when there are any.
    Every attempt gets its own capture_id (see new_capture_id)."""
    venues, venues_sha = load_venues(venues_path)
    schedule_bytes, schedule_sha = ps.read_bytes_once(schedule_path)
    now = clock()
    game, refusal = select_game(schedule_bytes, game_id)
    if refusal:
        return {"outcome": refusal, "requested": False, "game_id": game_id,
                "detail": f"{game_id!r}: {'no' if refusal == 'game_not_found' else 'more than one'} schedule row"}
    ctx, refusal, detail = resolve(game, venues, now)
    if refusal:
        return {"outcome": refusal, "requested": False, "game_id": game_id, "detail": detail, "game": ctx}
    venue = ctx.pop("venue")
    params = request_params(venue, parse_ts(ctx["kickoff_utc"]))
    started = clock()
    body, meta = fetch(params, get)
    received = clock()
    parsed = parse_response(meta["http_status"], body, meta["error"], params)
    raw_sha = _sha(body) if body is not None else None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rev, dirty = ps.code_revision(repo_dir)

    def build(capture_id: str, raw_name: str | None) -> dict:
        doc = {
            "kind": CAPTURE_KIND, "schema_version": SCHEMA_VERSION, "capture_id": capture_id,
            "request_started_at_utc": ts(started), "received_at_utc": ts(received),
            "provider_issue_time_utc": None, "provider_issue_time_basis": ISSUE_TIME_BASIS,
            "game": ctx, "schedule": {"path": str(Path(schedule_path)), "sha256": schedule_sha},
            "venue": venue, "venues_file": {"path": str(Path(venues_path)), "sha256": venues_sha},
            "request": {"endpoint": ENDPOINT, "params": params, "url": meta["url"],
                        "tls_trust": tls_trust},
            "response": {"http_status": meta["http_status"], "http_date": meta["http_date"],
                         "content_type": meta["content_type"], "error": meta["error"],
                         "raw_file": raw_name, "raw_sha256": raw_sha,
                         "raw_bytes": None if body is None else len(body)},
            "parsed": _summary(parsed), "variable_meaning": VARIABLE_MEANING, "source": SOURCE,
            "code_revision": rev, "code_dirty": dirty,
        }
        doc[CHECKSUM_FIELD] = ps.payload_checksum(doc)
        return doc

    capture_id, raw_name = _write_new(out_dir, received, game_id, body, build)
    return {"outcome": parsed["outcome"], "usable": parsed["usable"], "requested": True,
            "capture_id": capture_id, "metadata_file": str(out_dir / f"{capture_id}.json"),
            "raw_file": None if raw_name is None else str(out_dir / raw_name),
            "detail": parsed["detail"], "hours": len(parsed["hours"])}


# --------------------------------------------------------------------------
# Database files (same exact-schema rule as the football pilot)
# --------------------------------------------------------------------------

def _expected_schema() -> set:
    ref = sqlite3.connect(":memory:")
    try:
        ref.executescript(SCHEMA)
        return fr._schema_objects(ref)
    finally:
        ref.close()


def schema_state(path: Path) -> str:
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
    raise fr.SchemaRefused(f"{path}: not a weather-history pilot database (schema differs); "
                           "use a new database file - this pilot has no migrations")


def connect_for_import(db_path) -> sqlite3.Connection:
    path = Path(db_path)
    state = schema_state(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)
    if state == "empty":
        conn.executescript(SCHEMA)
    return conn


def open_read_only(db_path) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"database not found: {path}")
    if schema_state(path) != "current":
        raise fr.SchemaRefused(f"{path}: not an initialized weather-history database")
    return fr.open_read_only(path, require_current_schema=False)


# --------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------

_REQUIRED = ("kind", "schema_version", "capture_id", "request_started_at_utc", "received_at_utc",
             "provider_issue_time_utc", "provider_issue_time_basis", "game", "schedule", "venue",
             "venues_file", "request", "response", "parsed", "source", CHECKSUM_FIELD)


def _read_capture(path: Path) -> tuple[dict, bytes, bytes | None, dict]:
    path = Path(path)
    if not path.is_file() or path.suffix != ".json":
        raise ImportRefused(f"{path}: not a capture metadata file (select <capture_id>.json files)")
    data = path.read_bytes()
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ImportRefused(f"{path}: not valid UTF-8 JSON ({exc})") from None
    if not isinstance(doc, dict) or any(k not in doc for k in _REQUIRED):
        raise ImportRefused(f"{path}: not a weather capture (missing fields)")
    if doc["kind"] != CAPTURE_KIND or doc["schema_version"] != SCHEMA_VERSION:
        raise ImportRefused(f"{path}: unsupported kind/schema_version")
    if ps.payload_checksum(doc) != doc[CHECKSUM_FIELD]:
        raise ImportRefused(f"{path}: {CHECKSUM_FIELD} mismatch (file changed after capture)")
    if path.stem != doc["capture_id"]:
        raise ImportRefused(f"{path}: file name doesn't match capture_id {doc['capture_id']!r}")
    if doc["provider_issue_time_utc"] is not None:
        raise ImportRefused(f"{path}: this pilot never records a provider issue time")
    for key in ("request_started_at_utc", "received_at_utc"):
        try:
            parse_ts(doc[key])
        except (ValueError, TypeError):
            raise ImportRefused(f"{path}: {key} is not a UTC timestamp") from None
    resp = doc["response"]
    raw = None
    if resp.get("raw_file") is not None:
        raw_path = path.parent / resp["raw_file"]
        if Path(resp["raw_file"]).name != resp["raw_file"] or not raw_path.is_file():
            raise ImportRefused(f"{path}: raw response file {resp['raw_file']!r} not found beside it")
        raw = raw_path.read_bytes()
        if _sha(raw) != resp.get("raw_sha256") or len(raw) != resp.get("raw_bytes"):
            raise ImportRefused(f"{raw_path}: raw response hash/size doesn't match the capture")
    parsed = parse_response(resp.get("http_status"), raw, resp.get("error"), doc["request"]["params"])
    if _summary(parsed) != doc["parsed"]:
        raise ImportRefused(f"{path}: the raw response no longer parses to the recorded outcome")
    return doc, data, raw, parsed


def import_captures(conn: sqlite3.Connection, paths, *, now: datetime | None = None) -> list[dict]:
    """Import explicitly selected capture files in one transaction. Every file is
    validated first; any refusal or error writes nothing for any file. A capture
    already imported with the same content is a no-op (created=False)."""
    imported_at = ts(now or datetime.now(timezone.utc))
    loaded, seen = [], {}
    for p in paths:
        doc, data, raw, parsed = _read_capture(Path(p))
        if seen.get(doc["capture_id"], doc[CHECKSUM_FIELD]) != doc[CHECKSUM_FIELD]:
            raise ImportRefused(f"{p}: capture_id {doc['capture_id']} selected twice with different content")
        seen[doc["capture_id"]] = doc[CHECKSUM_FIELD]
        loaded.append((Path(p).resolve(), doc, data, raw, parsed))
    results = []
    conn.execute("BEGIN IMMEDIATE")
    try:
        for path, doc, data, raw, parsed in loaded:
            row = conn.execute("SELECT capture_import_id, payload_sha256 FROM weather_captures "
                               "WHERE capture_id = ? AND transformation_version = ?",
                               (doc["capture_id"], TRANSFORMATION_VERSION)).fetchone()
            if row:
                if row[1] != doc[CHECKSUM_FIELD]:
                    raise ImportRefused(f"{path}: capture {doc['capture_id']} was already imported "
                                        "with different content; refusing")
                results.append({"capture_id": doc["capture_id"], "capture_import_id": row[0],
                                "created": False})
                continue
            cid = _insert(conn, path, doc, data, raw, parsed, imported_at)
            results.append({"capture_id": doc["capture_id"], "capture_import_id": cid, "created": True,
                            "outcome": parsed["outcome"], "hours": len(parsed["hours"])})
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return results


def _insert(conn, path: Path, doc: dict, data: bytes, raw, parsed: dict, imported_at: str) -> int:
    g, v, resp, rq, ret = doc["game"], doc["venue"], doc["response"], doc["request"], parsed["returned"]
    values = {
        "capture_id": doc["capture_id"], "payload_sha256": doc[CHECKSUM_FIELD],
        "transformation_version": TRANSFORMATION_VERSION, "metadata_file_name": path.name,
        "metadata_path": str(path), "metadata_sha256": _sha(data), "raw_response_file": resp["raw_file"],
        "raw_response_sha256": resp["raw_sha256"], "raw_response_bytes": resp["raw_bytes"],
        "raw_response": raw, "imported_at_utc": imported_at,
        "request_started_at_utc": doc["request_started_at_utc"], "received_at_utc": doc["received_at_utc"],
        "provider_issue_time_utc": None, "provider_issue_time_basis": doc["provider_issue_time_basis"],
        "provider_generationtime_ms": ret.get("generationtime_ms"),
        "game_id": g["game_id"], "season": g.get("season"), "week": g.get("week"),
        "home_team": g.get("home_team"), "away_team": g.get("away_team"),
        "schedule_gameday": g.get("schedule_gameday"), "schedule_gametime": g.get("schedule_gametime"),
        "kickoff_utc": g["kickoff_utc"], "schedule_path": doc["schedule"]["path"],
        "schedule_sha256": doc["schedule"]["sha256"], "stadium_id": g["stadium_id"], "stadium": g["stadium"],
        "schedule_roof": g.get("schedule_roof"), "venue_json": _dumps(v),
        "venues_file_sha256": doc["venues_file"]["sha256"], "field_exposure": v["field_exposure"],
        "requested_latitude": rq["params"]["latitude"], "requested_longitude": rq["params"]["longitude"],
        "returned_latitude": ret.get("latitude"), "returned_longitude": ret.get("longitude"),
        "returned_elevation": ret.get("elevation"), "returned_timezone": ret.get("timezone"),
        "returned_utc_offset_seconds": ret.get("utc_offset_seconds"),
        "request_url": rq.get("url") or rq["endpoint"], "request_params_json": _dumps(rq["params"]),
        "http_status": resp.get("http_status"), "http_date": resp.get("http_date"),
        "response_error_json": None if resp.get("error") is None else _dumps(resp["error"]),
        "outcome": parsed["outcome"], "usable": int(parsed["usable"]), "outcome_detail": parsed["detail"],
        "missing_variables_json": _dumps(parsed["missing_variables"]),
        "missing_requested_hours_json": _dumps(parsed["missing_requested_hours"]),
        "hourly_units_json": _dumps(parsed["units"]), "source_json": _dumps(doc["source"]),
        "code_revision": doc.get("code_revision"),
        "code_dirty": None if doc.get("code_dirty") is None else int(doc["code_dirty"]),
    }
    cols = ", ".join(values)
    cid = conn.execute(f"INSERT INTO weather_captures ({cols}) VALUES ({', '.join('?' * len(values))})",
                       tuple(values.values())).lastrowid
    for h in parsed["hours"]:
        conn.execute("INSERT INTO weather_forecast_hours VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (cid, h["valid_time_utc"], h["hour_index"], h["valid_time_raw"],
                      *(h[var] for var in VARIABLES)))
    return cid


# --------------------------------------------------------------------------
# List, as-of, compare
# --------------------------------------------------------------------------

def _received_by(caps: list[dict], cutoff: datetime | None) -> tuple[list[dict], list[dict]]:
    if cutoff is None:
        return caps, []
    kept = [c for c in caps if parse_ts(c["received_at_utc"]) <= cutoff]
    return kept, [c for c in caps if c not in kept]


def _brief(c: dict, now: datetime) -> dict:
    return {"capture_id": c["capture_id"], "game_id": c["game_id"], "stadium": c["stadium"],
            "kickoff_utc": c["kickoff_utc"], "received_at_utc (capture time)": c["received_at_utc"],
            "age_since_capture_hours": round((now - parse_ts(c["received_at_utc"])).total_seconds() / 3600, 2),
            "provider_issue_time_utc": None, "outcome": c["outcome"], "usable": bool(c["usable"]),
            "outcome_detail": c["outcome_detail"], "http_status": c["http_status"],
            "hours": c.get("n_hours"), "units": json.loads(c["hourly_units_json"]),
            "missing_variables": json.loads(c["missing_variables_json"]),
            "missing_requested_hours": json.loads(c["missing_requested_hours_json"]),
            "requested": [c["requested_latitude"], c["requested_longitude"]],
            "returned_grid": [c["returned_latitude"], c["returned_longitude"]],
            "raw_response_sha256": c["raw_response_sha256"]}


_CAPS_SQL = ("SELECT c.*, (SELECT COUNT(*) FROM weather_forecast_hours h WHERE h.capture_import_id = "
             "c.capture_import_id) AS n_hours FROM weather_captures c")


def list_captures(conn, *, game_id: str | None = None, as_of: datetime | None = None,
                  now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    caps = _rows(conn, _CAPS_SQL + (" WHERE c.game_id = ?" if game_id else ""),
                 (game_id,) if game_id else ())
    caps.sort(key=lambda c: (parse_ts(c["received_at_utc"]), c["capture_id"]))
    kept, later = _received_by(caps, as_of)
    return {"as_of": None if as_of is None else ts(as_of), "captures": [_brief(c, now) for c in kept],
            "excluded_received_after_cutoff": len(later), "provider_issue_time_basis": ISSUE_TIME_BASIS,
            "source": SOURCE, "limitations": list(LIMITATIONS)}


def _hours(conn, cid: int) -> list[dict]:
    return _rows(conn, "SELECT * FROM weather_forecast_hours WHERE capture_import_id = ? "
                       "ORDER BY valid_time_utc", (cid,))


def as_of(conn, game_id: str, cutoff: datetime, *, now: datetime | None = None) -> dict:
    """The latest usable capture of the game received at or before the cutoff.
    Captures received later are excluded even if they're the only ones."""
    now = now or datetime.now(timezone.utc)
    caps = _rows(conn, _CAPS_SQL + " WHERE c.game_id = ?", (game_id,))
    kept, later = _received_by(caps, cutoff)
    usable = sorted((c for c in kept if c["usable"]), key=lambda c: parse_ts(c["received_at_utc"]))
    chosen = usable[-1] if usable else None
    return {"game_id": game_id, "cutoff": ts(cutoff),
            "capture": None if chosen is None else _brief(chosen, now),
            "status": "found" if chosen else "no usable capture received by the cutoff",
            "hours": [] if chosen is None else [
                {k: h[k] for k in ("valid_time_utc", *VARIABLES)} for h in _hours(conn, chosen["capture_import_id"])],
            "unusable_received_by_cutoff": [c["capture_id"] for c in kept if not c["usable"]],
            "excluded_received_after_cutoff": len(later), "kickoff_value": KICKOFF_VALUE,
            "variable_meaning": VARIABLE_MEANING}


def _capture_row(conn, capture_id: str) -> dict:
    rows = _rows(conn, _CAPS_SQL + " WHERE c.capture_id = ?", (capture_id,))
    if not rows:
        raise CompareRefused(f"capture {capture_id!r} is not imported")
    return rows[0]


def compare(conn, earlier_id: str, later_id: str, *, as_of_cutoff: datetime | None = None,
            now: datetime | None = None) -> dict:
    """Two usable captures of the same game, value by value at identical valid
    hours. A variable is compared only when both units are identical."""
    now = now or datetime.now(timezone.utc)
    if earlier_id == later_id:
        raise CompareRefused("the same capture on both sides")
    a, b = _capture_row(conn, earlier_id), _capture_row(conn, later_id)
    if a["game_id"] != b["game_id"]:
        raise CompareRefused(f"different games ({a['game_id']} vs {b['game_id']})")
    if parse_ts(a["received_at_utc"]) >= parse_ts(b["received_at_utc"]):
        raise CompareRefused(f"{earlier_id} was not received before {later_id}")
    for c in (a, b):
        if not c["usable"]:
            raise CompareRefused(f"{c['capture_id']} is not usable ({c['outcome']}: {c['outcome_detail']})")
        if as_of_cutoff is not None and parse_ts(c["received_at_utc"]) > as_of_cutoff:
            raise CompareRefused(f"{c['capture_id']} was received after the cutoff {ts(as_of_cutoff)}")
    if (a["requested_latitude"], a["requested_longitude"]) != (b["requested_latitude"], b["requested_longitude"]):
        raise CompareRefused("the captures requested different coordinates")
    ua, ub = json.loads(a["hourly_units_json"]), json.loads(b["hourly_units_json"])
    variables = {}
    for var in VARIABLES:
        if var not in ua or var not in ub:
            variables[var] = {"status": "variable_missing",
                              "missing_in": [s for s, u in (("earlier", ua), ("later", ub)) if var not in u]}
        elif ua[var] != ub[var]:
            variables[var] = {"status": "units_mismatch", "earlier_unit": ua[var], "later_unit": ub[var]}
        else:
            variables[var] = {"status": "compared", "unit": ua[var], "meaning": VARIABLE_MEANING[var]}
    ha = {h["valid_time_utc"]: h for h in _hours(conn, a["capture_import_id"])}
    hb = {h["valid_time_utc"]: h for h in _hours(conn, b["capture_import_id"])}
    kickoff = parse_ts(a["kickoff_utc"])
    counts = {"hours_matched": 0, "hours_earlier_only": 0, "hours_later_only": 0,
              "values_changed": 0, "values_unchanged": 0, "values_missing": 0}
    hours = []
    for t in sorted(set(ha) | set(hb)):
        entry = {"valid_time_utc": t,
                 "minutes_from_kickoff": round((parse_ts(t) - kickoff).total_seconds() / 60)}
        if t not in hb or t not in ha:
            key = "earlier_only" if t not in hb else "later_only"
            entry["presence"] = key
            counts[f"hours_{key}"] += 1
            hours.append(entry)
            continue
        entry["presence"] = "both"
        counts["hours_matched"] += 1
        for var, info in variables.items():
            if info["status"] != "compared":
                continue
            x, y = ha[t][var], hb[t][var]
            if x is None or y is None:
                state = "missing_value_" + ("both" if x is None and y is None else
                                            "earlier" if x is None else "later")
                counts["values_missing"] += 1
                entry[var] = {"earlier": x, "later": y, "state": state}
            else:
                changed = x != y
                counts["values_changed" if changed else "values_unchanged"] += 1
                entry[var] = {"earlier": x, "later": y, "state": "changed" if changed else "unchanged",
                              "difference": round(y - x, 4)}
        hours.append(entry)
    side = lambda c: {"capture_id": c["capture_id"], "received_at_utc (capture time)": c["received_at_utc"],
                      "age_since_capture_hours": round((now - parse_ts(c["received_at_utc"])).total_seconds() / 3600, 2),
                      "provider_issue_time_utc": None,
                      "returned_grid": [c["returned_latitude"], c["returned_longitude"]],
                      "missing_requested_hours": json.loads(c["missing_requested_hours_json"])}
    notes = []
    if (a["returned_latitude"], a["returned_longitude"]) != (b["returned_latitude"], b["returned_longitude"]):
        notes.append("the provider returned different model-grid coordinates")
    if a["kickoff_utc"] != b["kickoff_utc"]:
        notes.append(f"kickoff differs between captures ({a['kickoff_utc']} vs {b['kickoff_utc']}); "
                     "minutes_from_kickoff uses the earlier capture's kickoff")
    return {"game_id": a["game_id"], "stadium": a["stadium"], "kickoff_utc": a["kickoff_utc"],
            "earlier": side(a), "later": side(b), "variables": variables, "counts": counts, "hours": hours,
            "notes": notes, "provider_issue_time_basis": ISSUE_TIME_BASIS, "kickoff_value": KICKOFF_VALUE,
            "source": {"attribution": SOURCE["attribution"], "licence_url": SOURCE["licence_url"]},
            "limitations": list(LIMITATIONS)}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("capture", help="one manual forecast request for one exact game_id")
    p.add_argument("--game-id", required=True)
    p.add_argument("--venues", required=True, help="reviewed venue JSON file")
    p.add_argument("--schedule", default=str(DEFAULT_SCHEDULE))
    p.add_argument("--out-dir", default=str(DEFAULT_CAPTURE_DIR))
    p = sub.add_parser("import", help="import explicitly selected capture metadata files")
    p.add_argument("--db", required=True)
    p.add_argument("captures", nargs="+")
    p = sub.add_parser("list", help="imported captures")
    p.add_argument("--db", required=True)
    p.add_argument("--game-id")
    p.add_argument("--as-of", type=parse_ts, help="only captures received at or before this UTC time")
    p = sub.add_parser("asof", help="latest usable capture received by a cutoff")
    p.add_argument("--db", required=True)
    p.add_argument("--game-id", required=True)
    p.add_argument("--cutoff", type=parse_ts, required=True)
    p = sub.add_parser("compare", help="two captures of one game at matching valid hours")
    p.add_argument("--db", required=True)
    p.add_argument("--earlier", required=True)
    p.add_argument("--later", required=True)
    p.add_argument("--as-of", type=parse_ts)
    p = sub.add_parser("sql", help="one read-only query (football pilot safeguards)")
    p.add_argument("--db", required=True)
    p.add_argument("query")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "capture":
            # Process-scoped: verify TLS against the Windows trust store (a local
            # security product may re-sign HTTPS with a root only Windows trusts).
            # Verification stays on; nothing system-wide is changed.
            from player_props.rehearsal_real import use_system_trust_store
            out = capture(args.game_id, venues_path=args.venues, schedule_path=args.schedule,
                          out_dir=args.out_dir, tls_trust=use_system_trust_store())
            print(json.dumps(out, indent=1, ensure_ascii=False))
            return 0 if out.get("usable") else (3 if not out["requested"] else 4)
        if args.cmd == "import":
            conn = connect_for_import(args.db)
            try:
                out = import_captures(conn, args.captures)
            finally:
                conn.close()
        elif args.cmd == "sql":
            if not Path(args.db).exists():
                raise FileNotFoundError(f"database not found: {args.db}")
            cols, rows = fr.run_sql(args.db, args.query)
            print("FORECASTS_NOT_OBSERVATIONS: rows are captured provider forecasts, not observed "
                  "weather; filter by received_at_utc for anything known by a time.", file=sys.stderr)
            print("\t".join(cols))
            for r in rows:
                print("\t".join("" if v is None else str(v) for v in r))
            return 0
        else:
            conn = open_read_only(args.db)
            try:
                if args.cmd == "list":
                    out = list_captures(conn, game_id=args.game_id, as_of=args.as_of)
                elif args.cmd == "asof":
                    out = as_of(conn, args.game_id, args.cutoff)
                else:
                    out = compare(conn, args.earlier, args.later, as_of_cutoff=args.as_of)
            finally:
                conn.close()
    except (ImportRefused, CompareRefused, VenuesRefused, fr.SchemaRefused, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except sqlite3.DatabaseError as exc:
        print(f"error: {exc} (only read-only queries are allowed)", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
