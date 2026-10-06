"""Immutable Ontario sportsbook spread captures (The Odds API, bulk endpoint).

Phase 1: collection and storage only. Twice a week - Wednesday 12:00 and
Sunday 09:00 America/Toronto - this records every NFL spread quote that
Ontario-licensed sportsbooks show for the upcoming games, so Wednesday and
Sunday-morning lines and prices can later be compared. Every capture is a new,
immutable, checksummed JSON file; a later capture never replaces or edits an
earlier one, and nothing is carried forward from an earlier capture.

Separate from ``spread_tracker.py`` (US + Canada regions, one upserted
season log, Wednesday only), which keeps working unchanged. Nothing here reads,
rewrites or backfills that tracker's files.

Scope
-----
* Bookmakers are requested by key (``bookmakers=``), not by region, so the
  feed list is explicit. Each key carries a jurisdiction. Only keys The Odds
  API documents as Ontario feeds ("(CA - ON)") are ``CA-ON``; ``playnow_ca``
  (British Columbia) is a Canadian feed that is NOT requested or treated as
  Ontario. ``bet99_ca_on`` is paid-tier coverage: requested, and its absence
  is reported, never treated as an error.
* The API's ``fanduel`` key is FanDuel **US**. It is requested only as a
  ``US``-jurisdiction reference and is never used as FanDuel Ontario. FanDuel
  Ontario quotes come only from the manual entry command (source ``manual``).

Safety
------
* No ``ODDS_API_KEY`` -> no network calls at all.
* A free call to ``/v4/sports`` reads the remaining credits first; the paid
  call is skipped if it would leave fewer than the reserve
  (``ODDS_API_MIN_REMAINING``, shared with the other Odds API features), or
  if the remaining credits can't be read.
* The key never appears in logs, errors or stored artifacts.
* Games that have kicked off (per the schedule or the provider) are excluded;
  quotes are pregame only. Per-game closing captures are a later phase.

See docs/ONTARIO_SPREAD_TRACKING.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import secrets
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

try:  # ODDS_API_KEY from a local .env for CLI runs
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pregame_snapshots as ps
from player_props.market_odds import ODDS_API_BASE, ODDS_API_SPORT, TEAM_FULL_NAME, _ABBR_BY_FULL_NAME

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_files"
STORE_DIR = DATA_DIR / "ontario_spreads"
CAPTURE_DIR = STORE_DIR / "captures"
MANUAL_DIR = STORE_DIR / "manual"
SCHEDULE_PATH = DATA_DIR / ps.SCHEDULE_NAME
MODEL_SNAPSHOT_DIR = DATA_DIR / "pregame_snapshots"

SCHEMA_VERSION = 1
CAPTURE_KIND = "ontario_spread_capture"
MANUAL_KIND = "ontario_manual_spread_quote"
CHECKSUM_FIELD = ps.CHECKSUM_FIELD
TORONTO = ZoneInfo("America/Toronto")

MARKET = "spreads"
ENDPOINT = f"/sports/{ODDS_API_SPORT}/odds"
REQUEST_TIMEOUT = 20
CREDIT_RESERVE = int(os.getenv("ODDS_API_MIN_REMAINING", "20"))
STALE_AFTER = timedelta(minutes=60)          # market last_update older than this -> "stale"
KICKOFF_MATCH_TOLERANCE = timedelta(minutes=60)
ON_TIME_TOLERANCE = timedelta(minutes=30)


# --------------------------------------------------------------------------
# Bookmakers and jurisdictions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Book:
    key: str
    title: str             # as documented by The Odds API
    jurisdiction: str      # "CA-ON" or "US"
    role: str              # "ontario", "ontario_paid_tier", "us_reference", "manual_ontario"
    source: str            # "the_odds_api" or "manual"


# Verified against https://the-odds-api.com/sports-odds-data/bookmaker-apis.html
# (Canada region) on 2026-10-05. Titles ending "(CA - ON)" are Ontario feeds.
ONTARIO_BOOKS = (
    Book("betano_ca_on", "Betano (CA - ON)", "CA-ON", "ontario", "the_odds_api"),
    Book("betmgm_ca_on", "BetMGM (CA - ON)", "CA-ON", "ontario", "the_odds_api"),
    Book("betrivers_ca_on", "BetRivers (CA - ON)", "CA-ON", "ontario", "the_odds_api"),
    Book("pointsbetca", "PointsBet (CA - ON)", "CA-ON", "ontario", "the_odds_api"),
    Book("proline_ca_on", "PROLINE (CA - ON)", "CA-ON", "ontario", "the_odds_api"),
    Book("sportsinteraction_ca_on", "Sports Interaction (CA - ON)", "CA-ON", "ontario", "the_odds_api"),
)
# "Only available on paid subscriptions" per the provider docs.
BET99 = Book("bet99_ca_on", "BET99 (CA - ON)", "CA-ON", "ontario_paid_tier", "the_odds_api")
# The API's `fanduel` is FanDuel US - a reference only, never FanDuel Ontario.
FANDUEL_US = Book("fanduel", "FanDuel", "US", "us_reference", "the_odds_api")
# FanDuel Ontario has no API feed; quotes are entered by hand.
FANDUEL_ONTARIO_MANUAL = Book("fanduel_on_manual", "FanDuel Ontario (manual entry)", "CA-ON",
                              "manual_ontario", "manual")

# Every API feed this module knows. BET99 is only requested when enabled
# (ONTARIO_SPREADS_INCLUDE_BET99=1), because a free-plan request naming a
# paid-only bookmaker hasn't been verified to succeed.
API_BOOKS = ONTARIO_BOOKS + (BET99, FANDUEL_US)
BOOKS_BY_KEY = {b.key: b for b in API_BOOKS + (FANDUEL_ONTARIO_MANUAL,)}
API_BOOK_KEYS = frozenset(b.key for b in API_BOOKS)


def requested_books(include_bet99: bool | None = None) -> tuple[Book, ...]:
    """The bookmakers one capture requests, in a fixed order."""
    if include_bet99 is None:
        include_bet99 = os.getenv("ONTARIO_SPREADS_INCLUDE_BET99", "") == "1"
    return tuple(b for b in API_BOOKS if b is not BET99 or include_bet99)


REQUESTED_BOOKS = requested_books(include_bet99=False)    # the default request
# Documented Canadian feeds that are deliberately NOT treated as Ontario.
NON_ONTARIO_CANADIAN = {"playnow_ca": "PlayNow (CA) - British Columbia, not Ontario"}


def request_cost(n_markets: int = 1, n_books: int = len(API_BOOKS)) -> int:
    """Credits for one bulk /odds call: markets x region-equivalents, where every
    10 bookmakers count as one region (provider docs)."""
    return n_markets * max(1, math.ceil(n_books / 10))


# --------------------------------------------------------------------------
# Errors, redaction
# --------------------------------------------------------------------------

class CaptureError(RuntimeError):
    """A capture couldn't be made (request failed, invalid response, ...)."""


class BudgetSkip(RuntimeError):
    """The paid call was skipped to protect the credit reserve."""


class CaptureConflictError(RuntimeError):
    """An artifact with this identity exists with different content."""


class ValidationError(ValueError):
    pass


_KEY_PARAM_RE = re.compile(r"(apiKey=)[^&\s'\"]+", re.IGNORECASE)


def redact(text, api_key: str | None = None) -> str:
    """Remove the API key (and any apiKey= URL parameter) from text."""
    text = str(text)
    key = api_key if api_key is not None else os.getenv("ODDS_API_KEY", "")
    if key:
        text = text.replace(key, "***")
    return _KEY_PARAM_RE.sub(r"\1***", text)


def sanitize(obj, api_key: str | None = None):
    """Copy of provider data with any key or credential-bearing URL removed."""
    key = api_key if api_key is not None else os.getenv("ODDS_API_KEY", "")
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, str) and ((key and key in v) or "apikey=" in v.lower()):
                continue
            out[k] = sanitize(v, key)
        return out
    if isinstance(obj, list):
        return [sanitize(v, key) for v in obj]
    if isinstance(obj, str) and ((key and key in obj) or "apikey=" in obj.lower()):
        return "***"
    return obj


# --------------------------------------------------------------------------
# Time, slots
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SlotRule:
    name: str
    weekday: int           # Monday=0
    local_time: time
    window: timedelta      # how long after the slot a run still counts for it


# Windows are kept short so a much later quote can't be labelled as the slot's:
# a run after the window makes no capture and the slot shows as missed.
SLOTS = (
    SlotRule("wednesday_noon", 2, time(12, 0), timedelta(hours=3)),      # until 15:00
    SlotRule("sunday_morning", 6, time(9, 0), timedelta(hours=2)),       # until 11:00
)
SLOTS_BY_NAME = {s.name: s for s in SLOTS}
AD_HOC = "ad_hoc"


def iso(dt: datetime) -> str:
    return ps._iso(dt)


def parse_utc(text) -> datetime:
    return ps._parse_utc(text)


def parse_provider_time(text) -> datetime:
    """Provider ISO timestamps ('...Z' or with offset) -> aware UTC datetime."""
    if not isinstance(text, str) or not text:
        raise ValueError(f"missing timestamp {text!r}")
    return ps._parse_iso(text)


def slot_intended(rule: SlotRule, now_utc: datetime) -> datetime:
    """The most recent occurrence of the slot at or before now, as aware UTC.
    Built in America/Toronto wall-clock time, so DST is handled by zoneinfo."""
    local = now_utc.astimezone(TORONTO)
    days_back = (local.weekday() - rule.weekday) % 7
    day = local.date() - timedelta(days=days_back)
    intended = datetime.combine(day, rule.local_time, tzinfo=TORONTO)
    if intended > local:
        intended = datetime.combine(day - timedelta(days=7), rule.local_time, tzinfo=TORONTO)
    return intended.astimezone(timezone.utc)


def resolve_slot(now_utc: datetime, requested: str = "auto") -> dict | None:
    """Which slot a run at `now_utc` belongs to, or None if no slot is due.

    `requested` = "auto" (scheduled runs), a slot name, or "ad_hoc" (manual run).
    A named slot must still be inside its window.
    """
    if requested == AD_HOC:
        return {"name": AD_HOC, "slot_id": f"{now_utc.astimezone(TORONTO):%Y-%m-%d}_{AD_HOC}",
                "timezone": "America/Toronto", "intended_utc": None, "intended_local": None,
                "delay_minutes": None, "status": AD_HOC}
    rules = SLOTS if requested == "auto" else (SLOTS_BY_NAME[requested],)
    for rule in rules:
        intended = slot_intended(rule, now_utc)
        delay = now_utc - intended
        if timedelta(0) <= delay <= rule.window:
            local = intended.astimezone(TORONTO)
            return {"name": rule.name, "slot_id": f"{local:%Y-%m-%d}_{rule.name}",
                    "timezone": "America/Toronto", "intended_utc": iso(intended),
                    "intended_local": local.isoformat(),
                    "delay_minutes": round(delay.total_seconds() / 60, 1),
                    "status": "on_time" if delay <= ON_TIME_TOLERANCE else "late"}
    return None


def expected_slots(start_utc: datetime, end_utc: datetime) -> list[dict]:
    """Every scheduled slot whose time falls in [start, end]."""
    out = []
    day = start_utc.astimezone(TORONTO).date() - timedelta(days=1)
    last = end_utc.astimezone(TORONTO).date()
    while day <= last:
        for rule in SLOTS:
            if day.weekday() == rule.weekday:
                local = datetime.combine(day, rule.local_time, tzinfo=TORONTO)
                if start_utc <= local.astimezone(timezone.utc) <= end_utc:
                    out.append({"name": rule.name, "slot_id": f"{local:%Y-%m-%d}_{rule.name}",
                                "intended_utc": iso(local)})
        day += timedelta(days=1)
    return out


# --------------------------------------------------------------------------
# Odds validation
# --------------------------------------------------------------------------

def valid_american(price) -> bool:
    """American odds: an integer-valued number with |price| >= 100."""
    if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price):
        return False
    return float(price).is_integer() and abs(price) >= 100


def valid_handicap(point) -> bool:
    """A spread: finite, a multiple of 0.5, at most 60 points."""
    if isinstance(point, bool) or not isinstance(point, (int, float)) or not math.isfinite(point):
        return False
    return float(point * 2).is_integer() and abs(point) <= 60


# --------------------------------------------------------------------------
# Schedule and event matching
# --------------------------------------------------------------------------

def load_schedule(path: Path = SCHEDULE_PATH) -> tuple[pd.DataFrame, str]:
    """The nflverse schedule (read once) and the sha256 of the bytes parsed."""
    data, digest = ps.read_bytes_once(path)
    return ps.parse_tsv(data), digest


def games_in_scope(schedule: pd.DataFrame, captured_at: datetime) -> tuple[list[dict], list[dict]]:
    """The upcoming week's games for a capture at `captured_at`.

    Scope is the (season, week) of the next game to kick off; within it, games
    that have already kicked off are excluded (with a reason), as are games
    with unusable kickoff times. Returns (eligible, excluded).
    """
    rows = []
    for g in schedule.itertuples(index=False):
        if pd.notna(getattr(g, "home_score", None)) and pd.notna(getattr(g, "away_score", None)):
            continue                                   # completed
        kickoff, problem = ps.kickoff_utc(g.gameday, g.gametime)
        rows.append((g, kickoff, problem))
    upcoming = [(g, k) for g, k, p in rows if k is not None and k > captured_at]
    if not upcoming:
        return [], []
    g0 = min(upcoming, key=lambda gk: gk[1])[0]
    season, week = int(g0.season), int(g0.week)
    eligible, excluded = [], []
    for g, kickoff, problem in rows:
        if int(g.season) != season or int(g.week) != week:
            continue
        base = {"game_id": g.game_id, "season": season, "week": week,
                "home_team": g.home_team, "away_team": g.away_team,
                "kickoff_utc": iso(kickoff) if kickoff else None}
        if kickoff is None:
            excluded.append({**base, "reason": problem})
        elif kickoff <= captured_at:
            excluded.append({**base, "reason": "kicked_off"})
        else:
            eligible.append(base)
    # Played games of that week (both scores in) are also underway/over.
    for g in schedule.itertuples(index=False):
        if int(g.season) == season and int(g.week) == week and pd.notna(getattr(g, "home_score", None)) \
                and pd.notna(getattr(g, "away_score", None)):
            excluded.append({"game_id": g.game_id, "season": season, "week": week,
                             "home_team": g.home_team, "away_team": g.away_team,
                             "kickoff_utc": None, "reason": "completed"})
    return sorted(eligible, key=lambda x: (x["kickoff_utc"], x["game_id"])), \
        sorted(excluded, key=lambda x: x["game_id"])


def match_event(game: dict, events: list[dict]) -> tuple[dict | None, str, str | None]:
    """Find the provider event for a scheduled game by BOTH teams and kickoff.

    Returns (event, orientation, problem). orientation is "same" or "swapped"
    (neutral-site games can list home/away the other way round). A team match
    whose commence_time is more than KICKOFF_MATCH_TOLERANCE from the scheduled
    kickoff is not a match.
    """
    home_full, away_full = TEAM_FULL_NAME.get(game["home_team"]), TEAM_FULL_NAME.get(game["away_team"])
    if not home_full or not away_full:
        return None, "", "unknown_team_abbreviation"
    kickoff = parse_utc(game["kickoff_utc"])
    team_matches = []
    for ev in events:
        pair = (ev.get("home_team"), ev.get("away_team"))
        if pair == (home_full, away_full):
            team_matches.append((ev, "same"))
        elif pair == (away_full, home_full):
            team_matches.append((ev, "swapped"))
    if not team_matches:
        return None, "", "no_provider_event"
    timed = []
    for ev, orient in team_matches:
        try:
            commence = parse_provider_time(ev.get("commence_time"))
        except ValueError:
            continue
        if abs(commence - kickoff) <= KICKOFF_MATCH_TOLERANCE:
            timed.append((ev, orient))
    if not timed:
        return None, "", "kickoff_mismatch"
    if len(timed) > 1:
        return None, "", "ambiguous_provider_event"
    return timed[0][0], timed[0][1], None


# --------------------------------------------------------------------------
# Quotes
# --------------------------------------------------------------------------

def parse_quote(bookmaker: dict | None, book: Book, game: dict, captured_at: datetime) -> dict:
    """One requested book's spread quote for one matched game.

    Handicaps and prices are taken by TEAM NAME from the provider outcomes, so
    a swapped home/away listing can't flip a sign. Status:
    quoted | stale | absent | no_spreads_market | incomplete | invalid.
    Absent quotes are recorded as absent - never filled from an earlier capture.
    """
    base = {"book_key": book.key, "book_title": book.title, "jurisdiction": book.jurisdiction,
            "role": book.role, "source": book.source,
            "home_point": None, "home_price": None, "away_point": None, "away_price": None,
            "bookmaker_last_update": None, "market_last_update": None, "age_minutes": None,
            "problem": None}
    if bookmaker is None:
        return {**base, "status": "absent"}
    base["bookmaker_last_update"] = bookmaker.get("last_update")
    market = next((m for m in bookmaker.get("markets") or [] if m.get("key") == MARKET), None)
    if market is None:
        return {**base, "status": "no_spreads_market"}
    base["market_last_update"] = market.get("last_update")
    outcomes = {o.get("name"): o for o in market.get("outcomes") or []}
    home_o = outcomes.get(TEAM_FULL_NAME[game["home_team"]])
    away_o = outcomes.get(TEAM_FULL_NAME[game["away_team"]])
    if home_o is None or away_o is None:
        return {**base, "status": "incomplete", "problem": "missing one side's outcome"}
    hp, hpr, ap, apr = home_o.get("point"), home_o.get("price"), away_o.get("point"), away_o.get("price")
    base.update(home_point=hp, home_price=hpr, away_point=ap, away_price=apr)
    problems = []
    if not valid_handicap(hp) or not valid_handicap(ap):
        problems.append("handicap not a finite half-point value")
    elif hp != -ap:
        problems.append(f"handicaps don't mirror ({hp} vs {ap})")
    if not valid_american(hpr) or not valid_american(apr):
        problems.append("price not valid American odds")
    stamp = base["market_last_update"] or base["bookmaker_last_update"]
    try:
        updated = parse_provider_time(stamp)
    except ValueError:
        problems.append("missing or unparseable last_update")
        updated = None
    if updated is not None:
        if updated > captured_at + timedelta(minutes=5):
            problems.append("last_update is after the capture time")
        base["age_minutes"] = round((captured_at - updated).total_seconds() / 60, 1)
    if problems:
        return {**base, "status": "invalid", "problem": "; ".join(problems)}
    if captured_at - updated > STALE_AFTER:
        return {**base, "status": "stale"}
    return {**base, "status": "quoted"}


# --------------------------------------------------------------------------
# Model snapshot link (no probabilities copied)
# --------------------------------------------------------------------------

def model_snapshot_index(snapshot_dir: Path = MODEL_SNAPSHOT_DIR) -> list[dict]:
    """Validated pregame model snapshots: run_id, captured_at, checksum, file,
    and the game IDs each one captured."""
    out = []
    for path in sorted(Path(snapshot_dir).glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        ps.validate_snapshot(doc, str(path))
        out.append({"run_id": doc["run_id"], "captured_at": doc["captured_at"],
                    "payload_sha256": doc[ps.CHECKSUM_FIELD], "file": path.name,
                    "games": {g["game_id"]: g["prediction_status"] for g in doc["games"]}})
    return out


def link_model_snapshot(game_id: str, as_of: datetime, index: list[dict] | None,
                        index_error: str | None) -> dict:
    """The latest model snapshot captured at or before `as_of` that contains
    this game. Never a later one. No match is recorded explicitly."""
    if index_error is not None:
        return {"status": "unavailable", "reason": index_error}
    eligible = [s for s in index or []
                if game_id in s["games"] and parse_utc(s["captured_at"]) <= as_of]
    if not eligible:
        return {"status": "no_eligible_snapshot"}
    best = max(eligible, key=lambda s: (s["captured_at"], s["run_id"]))
    return {"status": "linked", "run_id": best["run_id"], "captured_at": best["captured_at"],
            "payload_sha256": best["payload_sha256"], "file": best["file"],
            "prediction_status": best["games"][game_id]}


def link_quote_model(game_id: str, quote: dict, captured_at: datetime,
                     index: list[dict] | None, index_error: str | None) -> dict:
    """Model snapshot link for one quote, aligned to the QUOTE's provider
    timestamp (the market's last_update), not to when the API was called.

    The snapshot must have been captured at or before that timestamp (and the
    capture). A quote without a usable timestamp gets no time-aligned link.
    """
    if quote["status"] not in ("quoted", "stale"):
        return {"status": "no_quote" if quote["home_point"] is None else "not_linked_invalid_quote"}
    try:
        quote_time = parse_provider_time(quote.get("market_last_update"))
    except ValueError:
        return {"status": "quote_time_unknown",
                "reason": "the market has no last_update, so no time-aligned model comparison"}
    as_of = min(quote_time, captured_at)
    return {**link_model_snapshot(game_id, as_of, index, index_error),
            "basis": "market_last_update", "as_of": iso(as_of)}


# --------------------------------------------------------------------------
# Network (mocked in tests)
# --------------------------------------------------------------------------

def _usage(resp) -> dict:
    def as_int(name):
        try:
            return int(float(resp.headers.get(name)))
        except (TypeError, ValueError):
            return None
    return {"x-requests-remaining": as_int("x-requests-remaining"),
            "x-requests-used": as_int("x-requests-used"),
            "x-requests-last": as_int("x-requests-last")}


def check_budget(api_key: str, cost: int, reserve: int = CREDIT_RESERVE) -> dict:
    """Free /v4/sports call to read the remaining credits before paying.

    Raises BudgetSkip if the remaining credits can't be read, or if the paid
    call would leave fewer than `reserve`."""
    try:
        resp = requests.get(f"{ODDS_API_BASE}/sports", params={"apiKey": api_key},
                            timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise CaptureError(f"credit check failed: {redact(exc, api_key)}") from None
    usage = _usage(resp)
    remaining = usage["x-requests-remaining"]
    if remaining is None:
        raise BudgetSkip("remaining credits couldn't be read; refusing to spend credits blind")
    if remaining - cost < reserve:
        raise BudgetSkip(f"{remaining} credits remaining; a {cost}-credit call would go below "
                         f"the reserve of {reserve}")
    return usage


def fetch_odds(api_key: str, params: dict) -> tuple[list, dict, int]:
    """The one paid call. Returns (events, usage headers, HTTP status)."""
    try:
        resp = requests.get(f"{ODDS_API_BASE}{ENDPOINT}", params={"apiKey": api_key, **params},
                            timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        raise CaptureError(f"odds request failed: {redact(exc, api_key)}") from None
    usage = _usage(resp)
    if resp.status_code != 200:
        body = redact(getattr(resp, "text", "")[:300], api_key)
        raise CaptureError(f"odds request returned HTTP {resp.status_code}: {body} "
                           f"(usage: {usage})")
    try:
        events = resp.json()
    except ValueError:
        raise CaptureError("odds response was not JSON") from None
    if not isinstance(events, list):
        raise CaptureError(f"odds response was {type(events).__name__}, expected a list of events")
    return events, usage, resp.status_code


# --------------------------------------------------------------------------
# Building, validating and writing artifacts
# --------------------------------------------------------------------------

def new_id(at: datetime) -> str:
    return f"{at.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(6)}"


def _seal(doc: dict) -> dict:
    doc[CHECKSUM_FIELD] = ps.payload_checksum(doc)
    return doc


def build_capture(*, run_id: str, slot: dict, captured_at: datetime, received_at: datetime,
                  schedule: pd.DataFrame, schedule_sha256: str, events: list, usage: dict,
                  credit_check: dict, params: dict, http_status: int, api_key: str,
                  snapshot_index: list[dict] | None, snapshot_error: str | None,
                  code: tuple[str, bool | None], books: tuple[Book, ...] = REQUESTED_BOOKS) -> dict:
    eligible, excluded = games_in_scope(schedule, captured_at)
    events = sanitize(events, api_key)
    matched_ids, games = set(), []
    for game in eligible:
        event, orientation, problem = match_event(game, events)
        entry = {**game, "provider_event_id": None, "provider_commence_time": None,
                 "orientation": None, "match_problem": problem}
        if event is not None:
            matched_ids.add(event.get("id"))
            commence = parse_provider_time(event.get("commence_time"))
            entry.update(provider_event_id=event.get("id"),
                         provider_commence_time=iso(commence), orientation=orientation)
            if commence <= captured_at:
                excluded.append({**game, "reason": "provider_reports_started"})
                continue
            by_key = {b.get("key"): b for b in event.get("bookmakers") or []}
            entry["quotes"] = [parse_quote(by_key.get(b.key), b, game, captured_at)
                               for b in books]
            for q in entry["quotes"]:
                q["model_snapshot"] = link_quote_model(game["game_id"], q, captured_at,
                                                       snapshot_index, snapshot_error)
            entry["unrequested_books"] = sorted(k for k in by_key if k not in BOOKS_BY_KEY)
        else:
            entry["quotes"] = [{"book_key": b.key, "book_title": b.title,
                                "jurisdiction": b.jurisdiction, "role": b.role, "source": b.source,
                                "status": "event_not_matched", "home_point": None,
                                "home_price": None, "away_point": None, "away_price": None,
                                "bookmaker_last_update": None, "market_last_update": None,
                                "age_minutes": None, "problem": problem,
                                "model_snapshot": {"status": "no_quote"}}
                               for b in books]
            entry["unrequested_books"] = []
        games.append(entry)

    coverage = {}
    for b in API_BOOKS:
        if b not in books:
            coverage[b.key] = {"jurisdiction": b.jurisdiction, "role": b.role, "requested": False,
                               "games": 0, "by_status": {},
                               "note": "paid-tier coverage, not requested: no quotes exist for it "
                                       "in this capture (set ONTARIO_SPREADS_INCLUDE_BET99=1 on a "
                                       "paid plan)"}
            continue
        statuses = [q["status"] for g in games for q in g["quotes"] if q["book_key"] == b.key]
        counts = {s: statuses.count(s) for s in sorted(set(statuses))}
        coverage[b.key] = {"jurisdiction": b.jurisdiction, "role": b.role, "requested": True,
                           "games": len(statuses), "by_status": counts,
                           "note": ("paid-tier coverage: absence is expected on the free plan"
                                    if b is BET99 else
                                    "FanDuel US reference - NOT FanDuel Ontario"
                                    if b is FANDUEL_US else None)}

    doc = {
        "schema_version": SCHEMA_VERSION, "kind": CAPTURE_KIND, "run_id": run_id,
        "slot": slot, "captured_at": iso(captured_at), "received_at": iso(received_at),
        "code_revision": code[0], "code_dirty": code[1],
        "request": {"endpoint": ENDPOINT, "params": params, "http_status": http_status,
                    "estimated_cost": request_cost(n_books=len(books)), "credit_reserve": credit_check.get("reserve"),
                    "credits_before": credit_check.get("x-requests-remaining"),
                    "usage": usage},
        "books_requested": [{"key": b.key, "title": b.title, "jurisdiction": b.jurisdiction,
                             "role": b.role} for b in books],
        "not_ontario": NON_ONTARIO_CANADIAN,
        "schedule": {"path": ps.SCHEDULE_NAME, "sha256": schedule_sha256},
        "conventions": {
            "handicap": "each side's own handicap as quoted (favourite negative); "
                        "home_point is the home team's, away_point the away team's",
            "price": "American odds", "times": "UTC, YYYY-MM-DDTHH:MM:SSZ",
            "stale_after_minutes": STALE_AFTER.total_seconds() / 60,
            "model_link": "per quote: the latest pregame model snapshot captured at or "
                          "before the quote's market last_update (never later); a reference "
                          "only - no model probabilities are stored here",
        },
        "games": games,
        # False when no ONTARIO book quoted any in-scope game (an empty
        # response, or only the US FanDuel reference). Such a capture is kept
        # as evidence but doesn't fill its slot, so a retry can still capture it.
        "usable": has_ontario_quote(games),
        "excluded_games": sorted(excluded, key=lambda x: x["game_id"]),
        "unmatched_provider_events": sorted(
            [{"id": e.get("id"), "home_team": e.get("home_team"), "away_team": e.get("away_team"),
              "commence_time": e.get("commence_time")}
             for e in events if e.get("id") not in matched_ids], key=lambda x: str(x["id"])),
        "coverage": coverage,
        "provider_response": events,
    }
    return _seal(doc)


def has_ontario_quote(games: list[dict]) -> bool:
    """True if at least one Ontario (CA-ON) feed quoted an in-scope game.
    Quotes from the US FanDuel reference never count."""
    return any(q["status"] in ("quoted", "stale") and q["jurisdiction"] == "CA-ON"
               and BOOKS_BY_KEY.get(q["book_key"]) in ONTARIO_BOOKS + (BET99,)
               for g in games for q in g["quotes"])


_CAPTURE_FIELDS = ("schema_version", "kind", "run_id", "slot", "captured_at", "received_at",
                   "code_revision", "code_dirty", "request", "books_requested", "not_ontario",
                   "schedule", "conventions", "games", "usable", "excluded_games",
                   "unmatched_provider_events", "coverage", "provider_response", CHECKSUM_FIELD)
_QUOTE_STATUSES = {"quoted", "stale", "absent", "no_spreads_market", "incomplete", "invalid",
                   "event_not_matched"}
_ID_RE = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{12}$")


def _need(cond, where, msg):
    if not cond:
        raise ValidationError(f"{where}: {msg}")


def validate_capture(doc, where="capture") -> None:
    _need(isinstance(doc, dict), where, "not an object")
    _need(set(doc) == set(_CAPTURE_FIELDS), where,
          f"fields differ from schema: {sorted(set(doc) ^ set(_CAPTURE_FIELDS))}")
    _need(doc["schema_version"] == SCHEMA_VERSION, where, "unsupported schema_version")
    _need(doc["kind"] == CAPTURE_KIND, where, "wrong kind")
    _need(doc[CHECKSUM_FIELD] == ps.payload_checksum(doc), where, "checksum mismatch")
    _need(isinstance(doc["run_id"], str) and _ID_RE.match(doc["run_id"]), where, "bad run_id")
    captured = parse_utc(doc["captured_at"])
    _need(parse_utc(doc["received_at"]) >= captured, where, "received before captured")
    _need("apiKey" not in json.dumps(doc["request"]), where, "request contains a credential")
    key = os.getenv("ODDS_API_KEY", "")
    _need(not key or key not in json.dumps(doc), where, "artifact contains the API key")
    for g in doc["games"]:
        gw = f"{where}: game {g.get('game_id')}"
        _need(parse_utc(g["kickoff_utc"]) > captured, gw, "kickoff not after capture")
        if g["provider_commence_time"] is not None:
            _need(parse_utc(g["provider_commence_time"]) > captured, gw, "provider start not after capture")
        for q in g["quotes"]:
            _need(q["status"] in _QUOTE_STATUSES, gw, f"unknown status {q['status']}")
            _need(q["book_key"] in API_BOOK_KEYS and q["source"] == "the_odds_api", gw,
                  f"{q['book_key']} isn't an automated API feed (manual quotes live elsewhere)")
            book = BOOKS_BY_KEY[q["book_key"]]
            _need(q["jurisdiction"] == book.jurisdiction, gw, f"{q['book_key']} jurisdiction")
            if q["status"] in ("quoted", "stale"):
                _need(valid_handicap(q["home_point"]) and q["home_point"] == -q["away_point"],
                      gw, f"{q['book_key']} handicaps")
                _need(valid_american(q["home_price"]) and valid_american(q["away_price"]),
                      gw, f"{q['book_key']} prices")
            elif q["status"] in ("absent", "event_not_matched", "no_spreads_market"):
                _need(q["home_point"] is None and q["home_price"] is None, gw,
                      f"{q['book_key']} has values but status {q['status']}")
            link = q["model_snapshot"]
            if link.get("status") == "linked":
                _need(q["status"] in ("quoted", "stale") and q["market_last_update"], gw,
                      f"{q['book_key']} model link without a quote timestamp")
                as_of = parse_utc(link["as_of"])
                _need(as_of <= captured
                      and as_of <= parse_provider_time(q["market_last_update"])
                      and parse_utc(link["captured_at"]) <= as_of, gw,
                      f"{q['book_key']} model snapshot postdates the quote")
    _need(doc["usable"] == has_ontario_quote(doc["games"]), where, "usable flag")


def _accept_existing(path: Path, text: str, validate) -> dict:
    existing = path.read_text(encoding="utf-8")
    try:
        doc = json.loads(existing)
        validate(doc, str(path))
    except (ValueError, KeyError, TypeError) as exc:
        raise CaptureConflictError(f"{path} exists but isn't a valid artifact ({exc}); "
                                   "refusing to overwrite it") from None
    if ps._canonical(doc) != ps._canonical(json.loads(text)):
        raise CaptureConflictError(f"{path} already exists with different content; "
                                   "refusing to overwrite it")
    return doc


def write_artifact(doc: dict, directory: Path, validate) -> tuple[Path, bool]:
    """Exclusive, atomic create of <run_id>.json. Same content again -> no-op
    (idempotent retry); different content under the same id -> conflict."""
    validate(doc, "new artifact")
    directory.mkdir(parents=True, exist_ok=True)
    ident = doc.get("run_id") or doc.get("quote_id")
    path = directory / f"{ident}.json"
    text = json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        _accept_existing(path, text, validate)
        return path, False
    if ps._exclusive_write(path, text):
        return path, True
    _accept_existing(path, text, validate)     # lost a race to an identical writer
    return path, False


def load_captures(directory: Path = CAPTURE_DIR) -> list[dict]:
    out = []
    for path in sorted(Path(directory).glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        validate_capture(doc, str(path))
        out.append(doc)
    return out


class SlotLock:
    """Exclusive lock file (e.g. per capture slot, or for manual-quote saves) so
    two runs can't do the same write at once. Not committed (see .gitignore).

    The file records who holds it (purpose, id, process ID, host, start time
    and a random token). A conflict names the recorded owner and the lock's
    age. A lock up to EXPECTED_MAX_AGE old is reported as in progress; an older
    one as "older than expected" - which does not establish that its owner has
    stopped. Manual deletion is advised only after confirming the recorded
    process on the recorded host is no longer running, or after investigating
    when the owner can't be determined. The lock is never deleted
    automatically. On exit the file is removed only if it still carries this
    holder's token, so a lock taken over by someone else is never deleted.
    """

    EXPECTED_MAX_AGE = timedelta(minutes=10)

    def __init__(self, directory: Path, slot_id: str, run_id: str, what: str = "capture"):
        self.path = Path(directory) / f".{slot_id}.lock"
        self.run_id = run_id
        self.what = what
        self.token = secrets.token_hex(8)

    def _holder(self) -> tuple[dict, timedelta | None]:
        try:
            raw = self.path.read_text(encoding="utf-8")
            info = json.loads(raw) if raw.strip().startswith("{") else {"id": raw.strip()}
        except (OSError, ValueError):
            info = {}
        try:
            started = parse_utc(info["started_at"]) if "started_at" in info else \
                datetime.fromtimestamp(self.path.stat().st_mtime, timezone.utc)
            age = ps.utc_now() - started
        except (OSError, ValueError, KeyError):
            age = None
        return info, age

    def _conflict_message(self, info: dict, age: timedelta | None) -> str:
        """Recovery instructions for a lock held by someone else. Never
        suggests deleting it on the strength of its age alone."""
        pid, host = info.get("pid"), info.get("host")
        owner_known = bool(pid) and bool(host)
        if age is None:
            age_text = "its age could not be determined"
        else:
            minutes = int(age.total_seconds() // 60)
            age_text = (f"taken {minutes} minute{'s' if minutes != 1 else ''} ago" if minutes
                        else f"taken {int(age.total_seconds())} seconds ago")
        owner = (f"recorded owner: process {pid} on host {host}"
                 + (f", {self.what} {info['id']}" if info.get("id") else "")
                 if owner_known else "owner not recorded in the lock file")
        parts = [f"another {self.what} holds lock file {self.path} ({owner}; {age_text})."]
        if age is not None and age > self.EXPECTED_MAX_AGE:
            parts.append(f"The lock is older than expected (a {self.what} normally finishes in "
                         "seconds), but age alone does not establish that its owner has stopped.")
        else:
            parts.append("It appears to be in progress; try again in a moment.")
        if owner_known:
            elsewhere = "" if host == platform.node() else \
                f" (it was taken on host {host}, not this one, so check there)"
            parts.append(f"Delete {self.path} only after confirming that process {pid} on host "
                         f"{host} is no longer running{elsewhere}.")
        else:
            parts.append(f"Its owner can't be determined from the lock file, so investigate "
                         f"before deleting it: check that no {self.what} is running on any "
                         f"machine or process that uses {self.path.parent}, then delete "
                         f"{self.path}.")
        return " ".join(parts)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise CaptureError(self._conflict_message(*self._holder())) from None
        except PermissionError:
            # Windows reports an existing but unopenable lock path (e.g. a
            # directory, or a file with restricted access) this way.
            if os.path.lexists(self.path):
                raise CaptureError(self._conflict_message(*self._holder())) from None
            raise
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"what": self.what, "id": self.run_id, "pid": os.getpid(),
                       "host": platform.node(), "started_at": iso(ps.utc_now()),
                       "token": self.token}, f)
        return self

    def __exit__(self, *exc):
        try:
            info = json.loads(self.path.read_text(encoding="utf-8"))
            if info.get("token") == self.token:
                self.path.unlink()
        except (OSError, ValueError):
            pass
        return False


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------

@dataclass
class CaptureResult:
    status: str        # captured | empty | already_captured | not_due | no_key | no_games
    message: str
    path: Path | None = None
    doc: dict | None = None


def capture(*, now: datetime | None = None, slot: str = "auto", api_key: str | None = None,
            capture_dir: Path = CAPTURE_DIR, schedule_path: Path = SCHEDULE_PATH,
            snapshot_dir: Path = MODEL_SNAPSHOT_DIR, reserve: int = CREDIT_RESERVE,
            repo_dir: Path = ROOT) -> CaptureResult:
    """Make one capture for the due slot. Raises CaptureError / BudgetSkip /
    CaptureConflictError on failure; every non-raising outcome is a result."""
    api_key = os.getenv("ODDS_API_KEY", "") if api_key is None else api_key
    now = now or ps.utc_now()
    resolved = resolve_slot(now, slot)
    if resolved is None:
        return CaptureResult("not_due", f"no capture slot is due at {iso(now)} "
                                        f"({now.astimezone(TORONTO):%a %H:%M} Toronto)")
    if resolved["name"] != AD_HOC:
        for doc in load_captures(capture_dir):
            if doc["slot"]["slot_id"] == resolved["slot_id"] and doc["usable"]:
                return CaptureResult("already_captured",
                                     f"slot {resolved['slot_id']} already captured by run "
                                     f"{doc['run_id']}; no API call made")
    if not api_key:
        return CaptureResult("no_key", "ODDS_API_KEY not set - no API calls made")

    schedule, schedule_sha = load_schedule(schedule_path)
    eligible, _ = games_in_scope(schedule, now)
    if not eligible:
        return CaptureResult("no_games", "no upcoming pregame games in scope; no API call made")

    run_id = new_id(now)
    books = requested_books()
    cost = request_cost(n_books=len(books))
    with SlotLock(capture_dir, resolved["slot_id"], run_id):
        credit = check_budget(api_key, cost, reserve)
        credit["reserve"] = reserve
        captured_at = ps.utc_now() if now is None else now
        kickoffs = [parse_utc(g["kickoff_utc"]) for g in eligible]
        params = {"markets": MARKET, "bookmakers": ",".join(b.key for b in books),
                  "oddsFormat": "american", "dateFormat": "iso",
                  "commenceTimeFrom": iso(captured_at),
                  "commenceTimeTo": iso(max(kickoffs) + KICKOFF_MATCH_TOLERANCE)}
        events, usage, status = fetch_odds(api_key, params)
        received_at = max(captured_at, ps.utc_now()) if now is None else captured_at
        try:
            index, index_error = model_snapshot_index(snapshot_dir), None
        except (OSError, ValueError) as exc:
            index, index_error = None, f"model snapshots couldn't be read: {exc}"
        doc = build_capture(run_id=run_id, slot=resolved, captured_at=captured_at,
                            received_at=received_at, schedule=schedule,
                            schedule_sha256=schedule_sha, events=events, usage=usage,
                            credit_check=credit, params=params, http_status=status,
                            api_key=api_key, snapshot_index=index, snapshot_error=index_error,
                            code=ps.code_revision(repo_dir), books=books)
        path, created = write_artifact(doc, capture_dir, validate_capture)
    if not doc["usable"]:
        return CaptureResult("empty", f"no Ontario book quoted any in-scope game for slot "
                                      f"{resolved['slot_id']} (US reference quotes don't count); "
                                      f"kept as {path.name}, but the slot stays open for a retry "
                                      "within its window", path, doc)
    return CaptureResult("captured", f"captured slot {resolved['slot_id']} "
                                     f"({resolved['status']}) as {path.name}", path, doc)


def slot_coverage(start_utc: datetime, end_utc: datetime,
                  capture_dir: Path = CAPTURE_DIR) -> list[dict]:
    """Each scheduled slot in the range: captured (on_time/late) or missed.

    Slots before the first scheduled capture are left out: tracking hadn't
    started, so they weren't missed. With no scheduled captures yet, the list
    is empty.
    """
    by_slot = {}
    for doc in load_captures(capture_dir):
        by_slot.setdefault(doc["slot"]["slot_id"], []).append(doc)
    scheduled = [d["slot"]["intended_utc"] for docs in by_slot.values() for d in docs
                 if d["slot"]["intended_utc"]]
    if not scheduled:
        return []
    start_utc = max(start_utc, parse_utc(min(scheduled)))
    out = []
    for s in expected_slots(start_utc, end_utc):
        docs = sorted(by_slot.get(s["slot_id"], []), key=lambda d: d["captured_at"])
        usable = [d for d in docs if d["usable"]]
        if usable:
            d = usable[0]
            out.append({**s, "status": d["slot"]["status"], "run_id": d["run_id"],
                        "captured_at": d["captured_at"], "delay_minutes": d["slot"]["delay_minutes"]})
        elif docs:
            # Only empty captures: the slot has no quotes.
            out.append({**s, "status": "empty", "run_id": docs[-1]["run_id"],
                        "captured_at": docs[-1]["captured_at"],
                        "delay_minutes": docs[-1]["slot"]["delay_minutes"]})
        else:
            out.append({**s, "status": "missed", "run_id": None, "captured_at": None,
                        "delay_minutes": None})
    return out


# --------------------------------------------------------------------------
# Manual FanDuel Ontario quotes
# --------------------------------------------------------------------------

_MANUAL_FIELDS = ("schema_version", "kind", "quote_id", "source", "book_key", "book_title",
                  "jurisdiction", "entered_at", "observed_at", "entered_by", "code_revision",
                  "game_id", "season", "week", "home_team", "away_team", "kickoff_utc",
                  "team", "team_side", "handicap", "price", "opponent_handicap",
                  "opponent_price", "note", "schedule", CHECKSUM_FIELD)


def validate_manual(doc, where="manual quote") -> None:
    _need(isinstance(doc, dict), where, "not an object")
    _need(set(doc) == set(_MANUAL_FIELDS), where,
          f"fields differ from schema: {sorted(set(doc) ^ set(_MANUAL_FIELDS))}")
    _need(doc["schema_version"] == SCHEMA_VERSION and doc["kind"] == MANUAL_KIND, where, "schema")
    _need(doc[CHECKSUM_FIELD] == ps.payload_checksum(doc), where, "checksum mismatch")
    _need(doc["source"] == "manual" and doc["book_key"] == FANDUEL_ONTARIO_MANUAL.key
          and doc["jurisdiction"] == "CA-ON", where, "must be a manual FanDuel Ontario quote")
    _need(isinstance(doc["quote_id"], str) and _ID_RE.match(doc["quote_id"]), where, "bad quote_id")
    observed, entered = parse_utc(doc["observed_at"]), parse_utc(doc["entered_at"])
    _need(observed <= entered, where, "observed_at is after entered_at")
    _need(observed < parse_utc(doc["kickoff_utc"]), where, "observed at or after kickoff")
    _need(doc["team"] in (doc["home_team"], doc["away_team"]), where, "team not in game")
    _need(valid_handicap(doc["handicap"]), where, "handicap")
    _need(doc["opponent_handicap"] == -doc["handicap"], where, "opponent handicap must mirror")
    _need(valid_american(doc["price"]), where, "price")
    _need(doc["opponent_price"] is None or valid_american(doc["opponent_price"]), where,
          "opponent price")


def _parse_observed(text: str) -> datetime:
    dt = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValidationError("--observed-at needs a timezone, e.g. 2026-10-07T12:05-04:00 or ...Z")
    return dt.astimezone(timezone.utc).replace(microsecond=0)


def toronto_local_to_utc(day: date, at: time) -> datetime:
    """An America/Toronto wall-clock date and time -> aware UTC datetime.

    Rejects the two times that don't map to exactly one instant: the repeated
    hour when DST ends (ambiguous) and the skipped hour when it starts
    (nonexistent)."""
    naive = datetime.combine(day, at.replace(second=0, microsecond=0))
    first, second = naive.replace(tzinfo=TORONTO, fold=0), naive.replace(tzinfo=TORONTO, fold=1)
    if first.utcoffset() != second.utcoffset():
        roundtrip = first.astimezone(timezone.utc).astimezone(TORONTO).replace(tzinfo=None)
        if roundtrip == naive:
            raise ValidationError(f"{naive:%Y-%m-%d %H:%M} happens twice in Toronto (DST ends); "
                                  "enter the time in UTC instead")
        raise ValidationError(f"{naive:%Y-%m-%d %H:%M} doesn't exist in Toronto (DST starts)")
    return first.astimezone(timezone.utc)


_MANUAL_IDENTITY = ("game_id", "team", "observed_at", "handicap", "price", "opponent_price")


def prepare_manual_quote(*, game_id: str, team: str, handicap: float, price: int,
                         observed_at: str, opponent_price: int | None = None,
                         note: str | None = None, entered_by: str | None = None,
                         now: datetime | None = None, schedule_path: Path = SCHEDULE_PATH,
                         repo_dir: Path = ROOT) -> dict:
    """Validate one observed FanDuel Ontario quote and build its sealed,
    validated document - without writing anything (used for previews).

    Raises ValidationError for: a time without a timezone, an observation in
    the future or at/after kickoff, an unknown game, a team not in the game,
    a handicap that isn't a half-point value within 60, or invalid American
    odds."""
    entered = (now or ps.utc_now()).replace(microsecond=0)
    observed = _parse_observed(observed_at)
    if observed > entered:
        raise ValidationError("observed_at is in the future")
    schedule, schedule_sha = load_schedule(schedule_path)
    rows = schedule[schedule["game_id"] == game_id]
    if len(rows) != 1:
        raise ValidationError(f"game {game_id!r} not found in the schedule")
    g = rows.iloc[0]
    kickoff, problem = ps.kickoff_utc(g["gameday"], g["gametime"])
    if kickoff is None:
        raise ValidationError(f"game {game_id} has no usable kickoff ({problem})")
    if observed >= kickoff:
        raise ValidationError(f"observed_at {iso(observed)} is not before kickoff {iso(kickoff)}; "
                              "only pregame quotes are recorded")
    team = team.strip().upper()
    if team not in (g["home_team"], g["away_team"]):
        raise ValidationError(f"team {team} is not playing in {game_id}")
    if not valid_handicap(handicap):
        raise ValidationError(f"spread {handicap} isn't a half-point value within 60")
    if not valid_american(price) or (opponent_price is not None and not valid_american(opponent_price)):
        raise ValidationError("prices must be American odds (e.g. -110, +105)")
    handicap = float(handicap) + 0.0           # -0.0 -> 0.0
    doc = {
        "schema_version": SCHEMA_VERSION, "kind": MANUAL_KIND, "quote_id": new_id(entered),
        "source": "manual", "book_key": FANDUEL_ONTARIO_MANUAL.key,
        "book_title": FANDUEL_ONTARIO_MANUAL.title, "jurisdiction": "CA-ON",
        "entered_at": iso(entered), "observed_at": iso(observed), "entered_by": entered_by,
        "code_revision": ps.code_revision(repo_dir)[0],
        "game_id": game_id, "season": int(g["season"]), "week": int(g["week"]),
        "home_team": g["home_team"], "away_team": g["away_team"], "kickoff_utc": iso(kickoff),
        "team": team, "team_side": "home" if team == g["home_team"] else "away",
        "handicap": handicap, "price": int(price),
        "opponent_handicap": 0.0 - handicap,      # never -0.0 for a pick'em
        "opponent_price": None if opponent_price is None else int(opponent_price),
        "note": note, "schedule": {"path": ps.SCHEDULE_NAME, "sha256": schedule_sha},
    }
    doc = _seal(doc)
    validate_manual(doc, "new manual quote")
    return doc


def find_manual_duplicate(doc: dict, manual_dir: Path = MANUAL_DIR) -> tuple[Path, dict] | None:
    """An already-recorded quote for the same observation (same game, team,
    time, spread and prices), if any."""
    for path in sorted(Path(manual_dir).glob("*.json")):
        old = json.loads(path.read_text(encoding="utf-8"))
        if all(old.get(k) == doc[k] for k in _MANUAL_IDENTITY):
            return path, old
    return None


def save_manual_quote(doc: dict, manual_dir: Path = MANUAL_DIR) -> tuple[Path, bool, dict]:
    """Persist a prepared manual quote. The duplicate check and the write run
    under an exclusive lock, so two near-simultaneous submissions of the same
    observation can't both be written. Returns (path, created, document);
    created=False means the observation was already recorded (that file is
    returned unchanged)."""
    manual_dir = Path(manual_dir)
    with SlotLock(manual_dir, "manual_entry", doc["quote_id"], what="manual-quote save"):
        existing = find_manual_duplicate(doc, manual_dir)
        if existing is not None:
            return existing[0], False, existing[1]
        path, created = write_artifact(doc, manual_dir, validate_manual)
    return path, created, doc


def manual_quote(*, game_id: str, team: str, handicap: float, price: int, observed_at: str,
                 opponent_price: int | None = None, note: str | None = None,
                 entered_by: str | None = None, now: datetime | None = None,
                 schedule_path: Path = SCHEDULE_PATH, manual_dir: Path = MANUAL_DIR,
                 repo_dir: Path = ROOT) -> tuple[Path, bool, dict]:
    """Record one observed FanDuel Ontario spread quote, source "manual".
    The same observation entered twice is refused rather than duplicated."""
    doc = prepare_manual_quote(game_id=game_id, team=team, handicap=handicap, price=price,
                               observed_at=observed_at, opponent_price=opponent_price,
                               note=note, entered_by=entered_by, now=now,
                               schedule_path=schedule_path, repo_dir=repo_dir)
    return save_manual_quote(doc, manual_dir)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

EXIT_OK, EXIT_FAILED, EXIT_BUDGET = 0, 1, 2


def _summary(lines: list[str]) -> None:
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def _output(**kv) -> None:
    path = os.getenv("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            for k, v in kv.items():
                f.write(f"{k}={v}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Immutable Ontario sportsbook spread captures")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="capture the due slot (no API call if not due)")
    c.add_argument("--slot", default="auto", choices=["auto", *SLOTS_BY_NAME, AD_HOC])

    m = sub.add_parser("manual-quote", help="record an observed FanDuel Ontario quote")
    m.add_argument("--game", required=True, help="nflverse game_id, e.g. 2026_05_TB_DAL")
    m.add_argument("--team", required=True, help="team the handicap belongs to, e.g. DAL")
    m.add_argument("--spread", required=True, type=float, help="that team's handicap, e.g. -3.5")
    m.add_argument("--price", required=True, type=int, help="that team's American price, e.g. -110")
    m.add_argument("--observed-at", required=True, help="when you saw it, ISO with timezone")
    m.add_argument("--opponent-price", type=int, default=None)
    m.add_argument("--note", default=None)
    m.add_argument("--entered-by", default=None)

    cov = sub.add_parser("coverage", help="scheduled slots: captured on time, late, or missed")
    cov.add_argument("--days", type=int, default=28)

    sub.add_parser("validate", help="validate every stored artifact")
    args = ap.parse_args(argv)

    try:
        if args.cmd == "capture":
            result = capture(slot=args.slot)
            print(f"[ontario_spreads] {result.status}: {result.message}")
            _output(status=result.status)
            lines = ["### Ontario spread capture", f"- status: **{result.status}**",
                     f"- {result.message}"]
            if result.doc:
                req = result.doc["request"]
                lines += [f"- credits before: {req['credits_before']}, usage after: {req['usage']}",
                          f"- games: {len(result.doc['games'])}, "
                          f"excluded: {len(result.doc['excluded_games'])}"]
                for key, cv in result.doc["coverage"].items():
                    lines.append(f"  - {key} ({cv['jurisdiction']}): {cv['by_status']}")
            if result.status == "no_key":
                print("::warning::ODDS_API_KEY not set - Ontario spread capture skipped")
            if result.status == "empty":
                # Kept as evidence and committed, but the run fails so it's seen;
                # the slot stays open and the next run inside the window retries.
                print(f"::error::Ontario spread capture returned no quotes: {result.message}")
                lines[1] += " ❌"
                _summary(lines)
                return EXIT_FAILED
            _summary(lines)
            return EXIT_OK
        if args.cmd == "manual-quote":
            path, created, doc = manual_quote(
                game_id=args.game, team=args.team, handicap=args.spread, price=args.price,
                observed_at=args.observed_at, opponent_price=args.opponent_price,
                note=args.note, entered_by=args.entered_by)
            print(f"[ontario_spreads] {'recorded' if created else 'already recorded'}: "
                  f"{doc['team']} {doc['handicap']:+g} ({doc['price']:+d}) "
                  f"observed {doc['observed_at']} -> {path.name}")
            return EXIT_OK
        if args.cmd == "coverage":
            end = ps.utc_now()
            rows = slot_coverage(end - timedelta(days=args.days), end)
            if not rows:
                print("no scheduled captures yet - coverage starts at the first one")
            for s in rows:
                print(f"{s['slot_id']:32} {s['status']:9} {s['run_id'] or ''}")
            return EXIT_OK
        if args.cmd == "validate":
            n = len(load_captures())
            for path in sorted(MANUAL_DIR.glob("*.json")):
                validate_manual(json.loads(path.read_text(encoding="utf-8")), str(path))
            print(f"[ontario_spreads] {n} capture(s) and "
                  f"{len(list(MANUAL_DIR.glob('*.json')))} manual quote(s) valid")
            return EXIT_OK
    except BudgetSkip as exc:
        msg = redact(exc)
        print(f"::error::Ontario spread capture skipped to protect the credit reserve: {msg}")
        _output(status="budget_skip")
        _summary(["### Ontario spread capture", f"- status: **budget_skip** ❌", f"- {msg}"])
        return EXIT_BUDGET
    except (CaptureError, CaptureConflictError, ValueError, KeyError, TypeError, OSError) as exc:
        # ValueError covers ValidationError, SnapshotValidationError and bad JSON.
        msg = redact(exc)
        print(f"::error::Ontario spreads {args.cmd} failed: {msg}")
        _output(status="failed")
        _summary(["### Ontario spread capture", f"- status: **failed** ❌", f"- {msg}"])
        return EXIT_FAILED
    return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
