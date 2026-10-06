"""Display logic for the "Ontario Line Timing" page (Phase 3).

Read-only. Builds the Phase 2 Wednesday-vs-Sunday comparison in memory from
the validated capture (and manual-quote) artifacts and shapes it for display.
It makes no API calls, writes nothing, and never falls back to sample or
regenerated data. Kept out of the page script so it can be tested directly.
"""
from __future__ import annotations

import hashlib
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import ontario_spreads as on

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ontario_spread_report",
                                               ROOT / "scripts" / "ontario_spread_report.py")
rpt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rpt)

TORONTO = on.TORONTO

VIEWS = {
    rpt.GROUP_ONTARIO: "Ontario sportsbooks (automated feeds)",
    rpt.GROUP_MANUAL: "FanDuel Ontario (manual entries)",
    rpt.GROUP_US_REFERENCE: "FanDuel US (reference only)",
}

OUTCOME_LABELS = {
    "unchanged": "Unchanged",
    "equivalent": "Equivalent payoff",
    "sunday_dominates": "Sunday dominates",
    "wednesday_dominates": "Wednesday dominates",
    "trade_off": "Trade-off (not ranked)",
}

KEY_LABELS = {
    "through": "Crossed", "onto": "Moved onto", "off": "Moved off",
    "on_both_sides": "On it both days (favourite flipped)", "stays_on": "Stayed on",
    "unchanged": "", "none": "", None: "",
}

QUOTE_STATUS_LABELS = {
    "quoted": "fresh", "stale": "stale", "absent": "not offered", "invalid": "invalid",
    "no_spreads_market": "no spread market", "incomplete": "incomplete",
    "event_not_matched": "game not matched", None: "",
}


class IntegrityError(RuntimeError):
    """A source artifact is unreadable or fails validation."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def source_fingerprint(capture_dir: Path, manual_dir: Path) -> tuple:
    """Content hash of every capture and manual artifact (name + SHA-256).

    Used as the cache key: adding, removing or changing any file gives a new
    fingerprint, so a refreshed capture is never hidden by a cached report."""
    out = []
    for label, d in (("capture", Path(capture_dir)), ("manual", Path(manual_dir))):
        for p in sorted(d.glob("*.json")) if d.exists() else []:
            out.append((label, p.name, hashlib.sha256(p.read_bytes()).hexdigest()))
    return tuple(out)


def build(capture_dir: Path, manual_dir: Path) -> dict:
    """The full report (all three groups), built in memory. Raises
    IntegrityError naming the problem for any unreadable or invalid file."""
    try:
        return rpt.build_report(capture_dir, manual_dir, include_manual=True,
                                include_us_reference=True)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise IntegrityError(str(exc)) from None


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

def toronto(ts: str | None) -> str:
    """'2026-10-07T16:00:00Z' -> 'Wed Oct 7, 12:00 EDT' (America/Toronto)."""
    if not ts:
        return ""
    try:
        dt = on.parse_provider_time(ts).astimezone(TORONTO)
    except ValueError:
        return ts
    return f"{dt:%a %b} {dt.day}, {dt:%H:%M %Z}"


def signed_handicap(h) -> str:
    """Signed handicap label: '+3.5', '-3', 'PK' (0)."""
    if h is None or pd.isna(h):
        return ""
    h = float(h)
    if h == 0:
        return "PK"
    text = f"{h:+.1f}"
    return text[:-2] if text.endswith(".0") else text


def signed_price(p) -> str:
    """American price label: '+105', '-110'."""
    if p is None or pd.isna(p):
        return ""
    return f"{int(p):+d}"


def signed_change(x, unit: str = "") -> str:
    if x is None or pd.isna(x):
        return ""
    if x == 0:
        return f"0{unit}"
    text = f"{x:+.1f}"
    return (text[:-2] if text.endswith(".0") else text) + unit


# --------------------------------------------------------------------------
# Slots
# --------------------------------------------------------------------------

def _slot_window(slot_id: str) -> tuple[datetime, datetime]:
    day, name = slot_id.split("_", 1)
    rule = on.SLOTS_BY_NAME[name]
    start = datetime.combine(datetime.strptime(day, "%Y-%m-%d").date(), rule.local_time,
                             tzinfo=TORONTO)
    return start.astimezone(timezone.utc), (start + rule.window).astimezone(timezone.utc)


def slot_state(slot_id: str | None, ref: dict | None, report: dict, now: datetime) -> dict:
    """State of one intended slot of a week:
    captured | pending (window not over, no usable capture yet) |
    empty (only captures without Ontario quotes) | missed | not_determined."""
    if slot_id is None:
        return {"state": "not_determined", "label": "Not determined (week anchor ambiguous)",
                "delay_minutes": None}
    if ref is not None:
        delay = (on.parse_utc(ref["captured_at"]) - on.parse_utc(ref["intended_utc"])) \
            .total_seconds() / 60
        late = ref["slot_status"] == "late"
        return {"state": "captured", "delay_minutes": round(delay, 1),
                "label": (f"Captured {toronto(ref['captured_at'])}"
                          + (f" (late, {delay:.0f} min after the slot)" if late else " (on time)"))}
    start, end = _slot_window(slot_id)
    if now < end:
        return {"state": "pending", "delay_minutes": None,
                "label": f"Pending: slot {toronto(on.iso(start))}, window closes {toronto(on.iso(end))}"}
    unusable = [n for n in report.get("capture_notes", [])
                if n["slot_id"] == slot_id and n["reason"] == "not_usable_no_ontario_quote"]
    if unusable:
        return {"state": "empty", "delay_minutes": None,
                "label": "Captured, but no Ontario sportsbook quoted any game"}
    return {"state": "missed", "delay_minutes": None,
            "label": f"Missed: no capture in the window ending {toronto(on.iso(end))}"}


def week_slots(report: dict, season: int, week: int, now: datetime) -> dict:
    w = next((x for x in report["weeks"] if (x["season"], x["week"]) == (season, week)), None)
    if w is None:
        return {}
    return {"wednesday": slot_state(w["intended_wednesday"], w["wednesday"], report, now),
            "sunday": slot_state(w["intended_sunday"], w["sunday"], report, now),
            "intended_wednesday": w["intended_wednesday"], "intended_sunday": w["intended_sunday"]}


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def weeks_available(report: dict) -> list[tuple[int, int]]:
    return sorted({(w["season"], w["week"]) for w in report["weeks"]})


def default_week(report: dict) -> tuple[int, int] | None:
    """The most recent week that actually has a capture (or, failing that,
    any rows) - never an empty future week."""
    captured = [(w["season"], w["week"]) for w in report["weeks"]
                if w["wednesday"] is not None or w["sunday"] is not None]
    if captured:
        return max(captured)
    with_rows = {(r["season"], r["week"]) for r in report["rows"]}
    return max(with_rows) if with_rows else None


# --------------------------------------------------------------------------
# Rows for display
# --------------------------------------------------------------------------

_REASON_TEXT = {
    "no_sunday_observation_kicked_off": "Kicked off before the Sunday slot",
    "no_wednesday_observation_kicked_off": "Kicked off before the Wednesday slot",
    "no_sunday_observation_completed": "Already played at the Sunday slot",
    "no_sunday_observation_game_not_in_capture": "Not in the Sunday capture",
    "no_wednesday_observation_game_not_in_capture": "Not in the Wednesday capture",
    "no_wednesday_manual_observation_in_intended_slot": "No manual Wednesday quote in the slot window",
    "no_sunday_manual_observation_in_intended_slot": "No manual Sunday quote in the slot window",
    rpt.ANCHOR_DATES_DISAGREE: "Kickoff moved to another week (e.g. postponed)",
    rpt.ANCHOR_NOT_SUNDAY_WEEK: "Kickoff is not in this Sunday's week",
    rpt.ANCHOR_WEEK_DISAGREES: "Week's games disagree on their Sunday",
    "team_mismatch": "Team mismatch", "jurisdiction_mismatch": "Jurisdiction mismatch",
}


def _reason_text(reason: str) -> str:
    if reason in _REASON_TEXT:
        return _REASON_TEXT[reason]
    for prefix, day in (("wednesday_", "Wednesday"), ("sunday_", "Sunday")):
        if reason.startswith(prefix + "quote_"):
            status = reason[len(prefix + "quote_"):]
            return f"{day} quote {QUOTE_STATUS_LABELS.get(status, status.replace('_', ' '))}"
        if reason == prefix + "out_of_window":
            return f"{day} quote outside its slot window"
        if reason == prefix + "observed_at_or_after_kickoff":
            return f"{day} observation not before kickoff"
    return reason.replace("_", " ")


def display_status(row: dict, slots: dict) -> str:
    if row["status"] == "compared":
        return "Compared"
    reasons = set(row["reasons"])
    for day in ("sunday", "wednesday"):
        if f"no_{day}_capture_for_week" in reasons:
            state = slots.get(day, {}).get("state")
            label = day.capitalize()
            if state == "pending":
                return f"{label} comparison pending"
            if state == "empty":
                return f"{label} capture had no Ontario quotes"
            if state == "not_determined":
                return "Week not anchored"
            return f"{label} slot missed"
    if any(r.startswith("no_anchor_") for r in reasons):
        return "Not compared: week can't be determined"
    if row["status"] == "unmatched":
        return "Unmatched"
    if row["status"] == "missing_quote":
        return "Quote missing"
    return "Not compared (quote quality)"


DISPLAY_COLUMNS = (
    "Matchup", "Team", "Sportsbook", "Source", "Wed spread", "Wed price", "Wed break-even",
    "Sun spread", "Sun price", "Sun break-even", "Spread change", "Break-even change",
    "Key 3", "Key 7", "Outcome", "Status", "Reasons",
    "Wed captured (Toronto)", "Wed provider update (Toronto)", "Wed quote", "Wed slot",
    "Sun captured (Toronto)", "Sun provider update (Toronto)", "Sun quote", "Sun slot",
    "_game_id", "_book_key",
)


def display_rows(report: dict, group: str, season: int, week: int, now: datetime,
                 books: list[str] | None = None, game_id: str | None = None,
                 team: str | None = None) -> pd.DataFrame:
    """Filtered, display-ready rows for one group and week."""
    slots = week_slots(report, season, week, now)
    out = []
    for r in report["rows"]:
        if (r["group"], r["season"], r["week"]) != (group, season, week):
            continue
        if books and r["book_key"] not in books:
            continue
        if game_id and r["game_id"] != game_id:
            continue
        if team and r["comparison_team"] != team:
            continue
        out.append({
            "Matchup": r["matchup"], "Team": r["comparison_team"],
            "Sportsbook": r["book_title"] or r["book_key"],
            "Source": ("manual entry" if r["source"] == "manual" else
                       "US reference feed" if r["jurisdiction"] == "US" else "automated feed"),
            "Wed spread": signed_handicap(r["wed_handicap"]), "Wed price": signed_price(r["wed_price"]),
            "Wed break-even": None if r["wed_break_even"] is None else r["wed_break_even"] * 100,
            "Sun spread": signed_handicap(r["sun_handicap"]), "Sun price": signed_price(r["sun_price"]),
            "Sun break-even": None if r["sun_break_even"] is None else r["sun_break_even"] * 100,
            "Spread change": signed_change(r["spread_change"]),
            "Break-even change": signed_change(
                None if r["break_even_change"] is None else r["break_even_change"] * 100, " pp"),
            "Key 3": KEY_LABELS.get(r["key_3"], r["key_3"]),
            "Key 7": KEY_LABELS.get(r["key_7"], r["key_7"]),
            "Outcome": OUTCOME_LABELS.get(r["outcome"], "") if r["outcome"] else "",
            "Status": display_status(r, slots),
            "Reasons": "; ".join(_reason_text(x) for x in r["reasons"]),
            "Wed captured (Toronto)": toronto(r["wed_observed_at"]),
            "Wed provider update (Toronto)": toronto(r["wed_provider_update"]),
            "Wed quote": _freshness(r, "wed"),
            "Wed slot": r["wed_slot_status"] or "",
            "Sun captured (Toronto)": toronto(r["sun_observed_at"]),
            "Sun provider update (Toronto)": toronto(r["sun_provider_update"]),
            "Sun quote": _freshness(r, "sun"),
            "Sun slot": r["sun_slot_status"] or "",
            "_game_id": r["game_id"], "_book_key": r["book_key"],
        })
    # Always the same columns, even when no row matches the filters.
    return pd.DataFrame(out, columns=list(DISPLAY_COLUMNS))


def _freshness(r: dict, prefix: str) -> str:
    status = r[f"{prefix}_quote_status"]
    if status is None:
        return ""
    label = QUOTE_STATUS_LABELS.get(status, status)
    age = r[f"{prefix}_age_minutes"]
    if age is not None and status in ("quoted", "stale"):
        label += f", {age:.0f} min old at capture"
    if r[f"{prefix}_slot_status"] == "manual":
        label = "manual observation"
    return label


def summarize(df: pd.DataFrame) -> dict:
    """Counts for the filtered rows. Sides, game/book pairs and games are
    counted separately (each game/book pair has two sides)."""
    if df.empty:
        return {"sides": 0, "compared_sides": 0, "pairs": 0, "games": 0, "outcomes": {}}
    compared = df[df["Status"] == "Compared"]
    return {
        "sides": len(df), "compared_sides": len(compared),
        "pairs": df[["_game_id", "_book_key"]].drop_duplicates().shape[0],
        "games": df["_game_id"].nunique(),
        "outcomes": {label: int((compared["Outcome"] == label).sum())
                     for label in OUTCOME_LABELS.values()},
    }
