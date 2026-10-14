"""Read-only "Was waiting worthwhile?" analysis of Ontario sportsbook spreads.

Three separate questions, answered from archived artifacts only:

1. Offers - for the same game, team and automated Ontario sportsbook, did the
   week's Sunday-morning quote offer better or worse terms than its
   Wednesday-noon quote? Rows are exactly the Phase 2 comparison's eligible
   pairs (scripts/ontario_spread_report.py: intended slots, eligibility,
   freshness, kickoff and source separation); nothing is re-paired here.
2. Realized results - once the schedule gives conservative evidence of a
   final score, each offer is settled (betting_log.spread_result) and
   simulated at equal stakes (bet_journal.win_profit, cent-rounded). This is
   hindsight: the offer that happened to win was not necessarily the better
   offer before the game.
3. Model decisions - for each game, the frozen pregame model snapshot that was
   current at each intended decision time (see DECISION_RULE), and whether the
   bet signal was kept, withdrawn, added or switched sides.

Nothing here writes to a source file, calls an API or reads the bet journal.
Reports go to a git-ignored output directory (default
reports/ontario_waiting/), refused anywhere that overlaps source data.
Corrupt inputs fail the run. Injury information is not archived, so no change
is attributed to injuries.

    python scripts/ontario_waiting_report.py [--season 2026] [--week 5] [--as-of 2026-10-14]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402

import bet_journal as bj  # noqa: E402
import betting_log  # noqa: E402
import frozen_spread_report as fr  # noqa: E402
import ontario_spread_report as osr  # noqa: E402
import ontario_spreads as on  # noqa: E402
import pregame_snapshots as ps  # noqa: E402

REPORT_KIND = "ontario_waiting_analysis"
REPORT_SCHEMA_VERSION = 1
SNAPSHOT_DIR, SCHEDULE_PATH = fr.SNAPSHOT_DIR, fr.SCHEDULE_PATH
CAPTURE_DIR, MANUAL_DIR = on.CAPTURE_DIR, on.MANUAL_DIR
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "ontario_waiting"
JSON_NAME = "ontario_waiting_analysis.json"
CSV_NAME = "ontario_waiting_offers.csv"
STAKE = fr.STAKE                         # equal simulated stake on each offer (units)

# Game result status for an offer row
FINAL, PENDING, UNRESOLVED = "final", "pending", "unresolved"

# Model decision availability (per game and decision time)
AVAILABLE = "available"
NOT_ANCHORED = "week_not_anchored"
KICKOFF_UNKNOWN = "kickoff_unavailable"
KICKED_OFF = "kicked_off_before_decision_time"
NO_SNAPSHOT = "no_snapshot_at_or_before_decision_time"
UNAVAILABLE_REASONS = {
    NOT_ANCHORED: "the week's Wednesday/Sunday slots can't be determined",
    KICKOFF_UNKNOWN: "no single usable scheduled kickoff for this game",
    KICKED_OFF: "the game kicked off before this decision time",
    NO_SNAPSHOT: "no validated model snapshot captured at or before this decision time",
    ps.NO_LINE: "the selected snapshot had no line (not a no-bet decision)",
    ps.PICKEM: "the selected snapshot had a pick'em line (no underdog to price)",
    ps.NO_PROBABILITY: "the selected snapshot had no probability (not a no-bet decision)",
}

# Sunday's decision relative to Wednesday's (per game)
RETAINED, WITHDRAWN, ADDED = "signal_retained", "signal_withdrawn", "signal_added"
SIDE_CHANGED, NEITHER, UNAVAILABLE = "side_changed", "neither_signalled", "unavailable"
DECISION_CLASSES = (RETAINED, WITHDRAWN, ADDED, SIDE_CHANGED, NEITHER, UNAVAILABLE)

# Checks against the CURRENT schedule on top of Phase 2's own (which uses the
# kickoffs recorded in the captures). A pair needs exactly one schedule row for
# the game with a usable kickoff, and both quotes strictly before it; otherwise
# it fails closed - the capture's older kickoff is never substituted.
TIMING_UNVERIFIED = "timing_unverified_current_kickoff"      # a quote at/after kickoff
TIMING_STATUS = "quote_not_before_current_kickoff"
KICKOFF_UNVERIFIED_STATUS = "current_kickoff_unavailable"
KICKOFF_UNAVAILABLE_PREFIX = "current_kickoff_unavailable_"
KICKOFF_NOT_IN_SCHEDULE = KICKOFF_UNAVAILABLE_PREFIX + "game_not_in_schedule"
KICKOFF_DUPLICATED = KICKOFF_UNAVAILABLE_PREFIX + "game_listed_more_than_once"
KICKOFF_TEXT = {
    KICKOFF_NOT_IN_SCHEDULE: "the game isn't in the current schedule",
    KICKOFF_DUPLICATED: "the game is listed more than once in the current schedule",
    KICKOFF_UNAVAILABLE_PREFIX + ps.MISSING_KICKOFF:
        "the current schedule has no kickoff date or time for the game",
    KICKOFF_UNAVAILABLE_PREFIX + ps.INVALID_KICKOFF:
        "the current schedule's kickoff date or time can't be read",
    KICKOFF_UNAVAILABLE_PREFIX + ps.AMBIGUOUS_KICKOFF:
        "the current kickoff time is in the repeated autumn hour (ambiguous)",
    KICKOFF_UNAVAILABLE_PREFIX + ps.NONEXISTENT_KICKOFF:
        "the current kickoff time is in the skipped spring hour (doesn't exist)",
}

DECISION_RULE = (
    "Decision times are the week's intended slot times: Wednesday 12:00 and Sunday 09:00 "
    "America/Toronto (the slots Phase 2 pairs), not the time a quote capture happened to run. "
    "For each game and decision time, the model decision is the LATEST validated pregame "
    "snapshot captured at or before that time and strictly before the kickoff it recorded "
    "(pregame_snapshots.select_from); the decision time must itself be strictly before the "
    "game's current scheduled kickoff. The snapshot is chosen first and its prediction read "
    "afterwards: a snapshot without a line or probability is 'unavailable', never replaced "
    "by an older snapshot that had one, and never read as a deliberate no-bet decision.")
PAIRING_RULE = (
    "Exactly the Ontario Line Timing comparison's eligible pairs (ontario_spread_report): "
    "automated Ontario sportsbook feeds only; the same season, week, game, sportsbook, "
    "jurisdiction and team; the week's intended Wednesday-noon slot against its "
    "Sunday-morning slot (four days apart); both quotes fresh ('quoted'), in-window and "
    "before kickoff. Never another week's capture, an ad-hoc capture, a manual observation, "
    "the US reference feed or a different sportsbook on the other day. Additionally, the "
    "CURRENT schedule must list the game exactly once with a usable kickoff, and both "
    "observations must be strictly before it; otherwise there is no pair (fail closed - the "
    "capture's older kickoff is never substituted).")
HINDSIGHT_NOTE = (
    "Realized profit is hindsight. It shows which quote happened to pay more for the final "
    "score; a winning offer was not automatically the better offer before the game. The "
    "payoff classification (dominates / equivalent / trade-off) is the pre-game comparison.")
INDEPENDENCE_NOTE = (
    "Rows are not independent bets: the same game appears once per sportsbook and once per "
    "side, and the two sides of a game settle in opposite directions. Counts are shown "
    "with their own denominators; nothing is summed into a single return.")
INJURY_NOTE = ("Injury information was not archived at both decision times; this analysis "
               "cannot determine whether injuries caused a change.")
NOT_CLAIMED = ("No preferred betting time, staking strategy or parlay is recommended, and no "
               "conclusion is drawn from a small sample.")

OFFER_CSV_FIELDS = (
    "season", "week", "game_id", "matchup", "team", "opponent", "side", "book_key", "book_title",
    "wed_observed_at", "wed_provider_update", "wed_handicap", "wed_price", "wed_slot_id",
    "wed_run_id", "sun_observed_at", "sun_provider_update", "sun_handicap", "sun_price",
    "sun_slot_id", "sun_run_id", "classification", "kickoff_utc", "kickoff_changed",
    "result_status", "result_reason", "home_score", "away_score", "wed_result", "wed_profit",
    "sun_result", "sun_profit", "profit_difference",
)


class ReportError(RuntimeError):
    """An input is missing, corrupt or inconsistent, or the output is refused."""


# --------------------------------------------------------------------------
# Loading (file-dependent only; no clock)
# --------------------------------------------------------------------------

def load_snapshots(snapshot_dir: Path) -> pd.DataFrame:
    """Every validated snapshot row. No snapshot directory = no history (an
    empty frame); any invalid file fails the run."""
    if not Path(snapshot_dir).is_dir():
        return pd.DataFrame()
    try:
        return ps.load_snapshots(snapshot_dir)
    except (ps.SnapshotValidationError, ValueError, OSError) as exc:
        raise ReportError(f"snapshot input is invalid: {exc}") from None


def load_model_inputs(snapshot_dir: Path = SNAPSHOT_DIR,
                      schedule_path: Path = SCHEDULE_PATH) -> dict:
    """The validated schedule (frozen_spread_report's loader and column check)
    and every validated snapshot row."""
    try:
        schedule, sha = fr.load_schedule(Path(schedule_path))
    except fr.ReportError as exc:
        raise ReportError(str(exc)) from None
    return {"schedule": schedule, "schedule_sha": sha,
            "snapshots": load_snapshots(Path(snapshot_dir))}


def load_sources(capture_dir: Path = CAPTURE_DIR, manual_dir: Path = MANUAL_DIR,
                 snapshot_dir: Path = SNAPSHOT_DIR, schedule_path: Path = SCHEDULE_PATH) -> dict:
    """The validated inputs: the Phase 2 comparison (built exactly as the Ontario
    Line Timing page builds it), the schedule and every snapshot row."""
    try:
        ontario = osr.build_report(Path(capture_dir), Path(manual_dir), include_manual=True,
                                   include_us_reference=True)
    except (osr.ReportError, ValueError, KeyError, TypeError, OSError) as exc:
        raise ReportError(f"Ontario capture input is invalid: {exc}") from None
    return {"ontario": ontario, **load_model_inputs(snapshot_dir, schedule_path)}


# --------------------------------------------------------------------------
# Results and settlement
# --------------------------------------------------------------------------

def game_result(schedule: pd.DataFrame, game_id: str, home: str, away: str,
                now: datetime) -> dict:
    """final (conservative completion evidence), pending or unresolved."""
    rows = schedule[schedule["game_id"] == game_id]
    if len(rows) == 1 and (rows.iloc[0]["home_team"], rows.iloc[0]["away_team"]) != (home, away):
        return {"status": UNRESOLVED, "reason": "schedule teams differ from the capture",
                "home_score": None, "away_score": None}
    evidence, why = bj.completion_evidence(schedule, game_id, now)
    if evidence is None:
        return {"status": PENDING if why.startswith(fr.PENDING_REASONS) else UNRESOLVED,
                "reason": why, "home_score": None, "away_score": None}
    return {"status": FINAL, "reason": "", "home_score": evidence["home_score"],
            "away_score": evidence["away_score"]}


def settle_offer(team: str, home: str, away: str, handicap: float, price: int,
                 home_score: int, away_score: int) -> tuple[str, Decimal]:
    """(win/loss/push, simulated profit at STAKE): win = bet_journal.win_profit
    (cent-rounded), push = 0.00, loss = -STAKE."""
    result = betting_log.spread_result(team, home, away, handicap, home_score, away_score)
    profit = bj.win_profit(int(price), STAKE) if result == "win" else \
        -STAKE if result == "loss" else Decimal("0.00")
    return result, profit


# --------------------------------------------------------------------------
# Offers (Phase 2 pairs)
# --------------------------------------------------------------------------

def _in_scope(r: dict, season: int | None, week: int | None) -> bool:
    return (season is None or r["season"] == season) and (week is None or r["week"] == week)


def current_kickoff(schedule: pd.DataFrame, game_id: str) -> tuple[datetime | None, str | None]:
    """(kickoff, None) for exactly one schedule row with a usable kickoff, else
    (None, reason). Any duplicate - identical or conflicting - is ambiguous."""
    rows = schedule[schedule["game_id"] == game_id]
    if rows.empty:
        return None, KICKOFF_NOT_IN_SCHEDULE
    if len(rows) > 1:
        return None, KICKOFF_DUPLICATED
    ko, why = ps.kickoff_utc(rows.iloc[0].get("gameday"), rows.iloc[0].get("gametime"))
    return (ko, None) if ko is not None else (None, KICKOFF_UNAVAILABLE_PREFIX + why)


def offers(ontario: dict, schedule: pd.DataFrame, now: datetime, season: int | None = None,
           week: int | None = None) -> tuple[list[dict], list[dict]]:
    """(paired offer rows, Ontario team-sides without an eligible pair)."""
    paired, missing = [], []
    for r in ontario["rows"]:
        if r["group"] != osr.GROUP_ONTARIO or not _in_scope(r, season, week):
            continue
        if r["status"] != "compared":
            missing.append(_missing(r, r["status"], r["reasons"]))
            continue
        ko, why = current_kickoff(schedule, r["game_id"])
        if ko is None:                                     # fail closed
            missing.append(_missing(r, KICKOFF_UNVERIFIED_STATUS, [why]))
            continue
        if not all(on.parse_utc(r[f"{p}_observed_at"]) < ko for p in ("wed", "sun")):
            missing.append(_missing(r, TIMING_STATUS, [TIMING_UNVERIFIED]))
            continue
        home, away = (r["comparison_team"], r["opponent"]) if r["side"] == "home" else \
            (r["opponent"], r["comparison_team"])
        res = game_result(schedule, r["game_id"], home, away, now)
        o = {
            "season": r["season"], "week": r["week"], "game_id": r["game_id"],
            "matchup": r["matchup"], "team": r["comparison_team"], "opponent": r["opponent"],
            "side": r["side"], "home_team": home, "away_team": away,
            "book_key": r["book_key"], "book_title": r["book_title"],
            **{f"{p}_{k}": r[f"{p}_{k}"] for p in ("wed", "sun")
               for k in ("observed_at", "provider_update", "handicap", "price", "slot_id",
                         "run_id")},
            "classification": r["outcome"], "kickoff_utc": r["kickoff_utc"],
            "kickoff_changed": r["kickoff_changed"],
            "result_status": res["status"], "result_reason": res["reason"],
            "home_score": res["home_score"], "away_score": res["away_score"],
            "wed_result": None, "wed_profit": None, "sun_result": None, "sun_profit": None,
            "profit_difference": None,
        }
        if res["status"] == FINAL:
            diff = Decimal("0.00")
            for p, sign in (("wed", -1), ("sun", 1)):
                result, profit = settle_offer(o["team"], home, away, o[f"{p}_handicap"],
                                              o[f"{p}_price"], res["home_score"],
                                              res["away_score"])
                o[f"{p}_result"], o[f"{p}_profit"] = result, str(profit)
                diff += sign * profit
            o["profit_difference"] = str(diff)
        paired.append(o)
    key = lambda x: (x["season"], x["week"], x["game_id"], x["book_key"], x["side"])  # noqa: E731
    return sorted(paired, key=key), sorted(missing, key=key)


def _missing(r: dict, status: str, reasons: list[str]) -> dict:
    return {"season": r["season"], "week": r["week"], "game_id": r["game_id"],
            "matchup": r["matchup"], "team": r["comparison_team"], "side": r["side"],
            "book_key": r["book_key"], "book_title": r["book_title"], "status": status,
            "reasons": list(reasons)}


# --------------------------------------------------------------------------
# Model decisions
# --------------------------------------------------------------------------

def decision_times(week_entry: dict) -> tuple[datetime | None, datetime | None]:
    """(Wednesday, Sunday) intended decision times for an anchored week."""
    out = []
    for key in ("intended_wednesday", "intended_sunday"):
        slot_id = week_entry.get(key)
        out.append(on.parse_utc(on.slot_from_id(slot_id)["intended_utc"]) if slot_id else None)
    return out[0], out[1]


def select_at(snapshots: pd.DataFrame, decision: datetime) -> dict[str, dict]:
    """game_id -> the latest snapshot row captured AT OR BEFORE `decision` (and
    strictly before its recorded kickoff). Snapshot times are validated
    whole-second UTC stamps, so 'strictly before decision + 1 s' is exactly
    'at or before decision'. No status filter: chosen first, read after."""
    if snapshots.empty:
        return {}
    picks = ps.select_from(snapshots, "latest", before=decision + timedelta(seconds=1))
    return {r["game_id"]: r for r in picks.to_dict("records")}


def decision(sel: dict | None, at: datetime | None, kickoff: datetime | None,
             anchored: bool, kickoff_reason: str | None = None) -> dict:
    """The model decision for one game at one decision time. Needs the same
    unique, usable current kickoff as an offer pair (kickoff_reason says why
    there isn't one)."""
    d = {"decision_time": ps._iso(at) if at else None, "status": None, "reason": "",
         "run_id": None, "snapshot_file": None, "captured_at": None, "age_minutes": None,
         "code_revision": None, "team": None, "handicap": None, "probability": None,
         "signal": None}
    if not anchored or at is None:
        d["status"] = NOT_ANCHORED
    elif kickoff is None:
        d["status"] = KICKOFF_UNKNOWN
    elif not at < kickoff:
        d["status"] = KICKED_OFF
    elif sel is None:
        d["status"] = NO_SNAPSHOT
    else:
        captured = ps._parse_utc(sel["captured_at"])
        d.update(run_id=sel["run_id"], snapshot_file=sel["snapshot_file"],
                 captured_at=sel["captured_at"], code_revision=sel["code_revision"],
                 age_minutes=int((at - captured).total_seconds() // 60))
        if sel["prediction_status"] != ps.PREDICTED:
            d["status"] = sel["prediction_status"]
        else:
            d.update(status=AVAILABLE, team=sel["underdog_team"],
                     handicap=abs(float(sel["spread_line"])),
                     probability=fr._none(sel["prob_underdog_covers"]),
                     signal=bool(sel["bet_signal"]))
    if d["status"] != AVAILABLE:
        d["reason"] = UNAVAILABLE_REASONS[d["status"]]
        if d["status"] == KICKOFF_UNKNOWN and kickoff_reason in KICKOFF_TEXT:
            d["reason"] += f" ({KICKOFF_TEXT[kickoff_reason]})"
    return d


def classify_decisions(wed: dict, sun: dict) -> str:
    """Sunday's decision relative to Wednesday's. Unavailable inputs (missing
    snapshot, line or probability) are never read as 'no signal'."""
    if wed["status"] != AVAILABLE or sun["status"] != AVAILABLE:
        return UNAVAILABLE
    if wed["signal"] and sun["signal"]:
        return RETAINED if wed["team"] == sun["team"] else SIDE_CHANGED
    if wed["signal"]:
        return WITHDRAWN
    if sun["signal"]:
        return ADDED
    return NEITHER


def probability_change(wed: dict, sun: dict) -> dict:
    """The probabilities are directly comparable only for the same proposition:
    the same team at the same handicap."""
    if wed["status"] != AVAILABLE or sun["status"] != AVAILABLE:
        return {"comparable": False, "change": None, "note": "a decision is unavailable"}
    if wed["team"] != sun["team"]:
        return {"comparable": False, "change": None,
                "note": f"different team ({wed['team']} then {sun['team']}): the probabilities "
                        "describe different propositions"}
    if wed["handicap"] != sun["handicap"]:
        return {"comparable": False, "change": None,
                "note": f"different handicap (+{wed['handicap']:g} then +{sun['handicap']:g}): "
                        "the probabilities describe different propositions"}
    note = "same snapshot at both times" if wed["run_id"] == sun["run_id"] else \
        "same team and handicap"
    return {"comparable": True, "change": round(sun["probability"] - wed["probability"], 6),
            "note": note}


def model_decisions(ontario: dict, schedule: pd.DataFrame, snapshots: pd.DataFrame,
                    season: int | None = None, week: int | None = None) -> list[dict]:
    """One row per scheduled game of each analysed week."""
    sched_season = pd.to_numeric(schedule["season"], errors="coerce")
    sched_week = pd.to_numeric(schedule["week"], errors="coerce")
    out, cache = [], {}
    for w in ontario["weeks"]:
        if not _in_scope(w, season, week):
            continue
        wed_at, sun_at = decision_times(w)
        anchored = w["intended_wednesday"] is not None and w["intended_sunday"] is not None
        for at in (wed_at, sun_at):
            if at is not None and at not in cache:
                cache[at] = select_at(snapshots, at)
        games = schedule[(sched_season == w["season"]) & (sched_week == w["week"])]
        for gid in sorted(set(games["game_id"])):
            g = games[games["game_id"] == gid].iloc[0]
            ko, ko_why = current_kickoff(schedule, gid)
            wed = decision(cache.get(wed_at, {}).get(gid), wed_at, ko, anchored, ko_why)
            sun = decision(cache.get(sun_at, {}).get(gid), sun_at, ko, anchored, ko_why)
            out.append({"season": w["season"], "week": w["week"], "game_id": gid,
                        "matchup": f"{g['away_team']} @ {g['home_team']}",
                        "kickoff_utc": ps._iso(ko) if ko else None,
                        "wednesday": wed, "sunday": sun,
                        "classification": classify_decisions(wed, sun),
                        "probability": probability_change(wed, sun)})
    return out


# --------------------------------------------------------------------------
# Rollups (separate denominators)
# --------------------------------------------------------------------------

def rollups(paired: list[dict], missing: list[dict], decisions: list[dict]) -> dict:
    settled = [o for o in paired if o["result_status"] == FINAL]
    diffs = [Decimal(o["profit_difference"]) for o in settled]
    reasons: dict[str, int] = {}
    for m in missing:
        for r in m["reasons"]:
            reasons[r] = reasons.get(r, 0) + 1
    return {
        "denominators": {
            "games_with_paired_captures": len({(o["season"], o["week"], o["game_id"])
                                               for o in paired}),
            "game_book_pairs": len({(o["season"], o["week"], o["game_id"], o["book_key"])
                                    for o in paired}),
            "team_side_comparisons": len(paired),
            "games_in_model_table": len(decisions),
            "games_with_model_decisions_at_both_times":
                sum(d["classification"] != UNAVAILABLE for d in decisions),
            "settled_comparisons": len(settled),
        },
        "offer_classifications": {
            "denominator": "team-side comparisons",
            **{c: sum(o["classification"] == c for o in paired) for c in osr.OUTCOMES}},
        "realized_profit_difference": {
            "denominator": "settled team-side comparisons (hindsight; not independent bets)",
            "sunday_higher": sum(d > 0 for d in diffs),
            "wednesday_higher": sum(d < 0 for d in diffs),
            "equal": sum(d == 0 for d in diffs),
            "games_represented": len({(o["season"], o["week"], o["game_id"]) for o in settled}),
            "game_book_pairs_represented": len({(o["season"], o["week"], o["game_id"],
                                                 o["book_key"]) for o in settled}),
        },
        "model_classifications": {
            "denominator": "games in the model table",
            **{c: sum(d["classification"] == c for d in decisions) for c in DECISION_CLASSES}},
        "missing_pairs": {"team_sides_without_eligible_pair": len(missing),
                          "by_reason": dict(sorted(reasons.items()))},
        "independence": INDEPENDENCE_NOTE,
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def analyze(src: dict, *, now: datetime, season: int | None = None,
            week: int | None = None) -> dict:
    """The analysis for this clock (results depend on the date; pairing and
    decisions don't). Pure: reads only `src`."""
    ontario, schedule = src["ontario"], src["schedule"]
    try:
        paired, missing = offers(ontario, schedule, now, season, week)
        decisions = model_decisions(ontario, schedule, src["snapshots"], season, week)
    except (ValueError, KeyError, TypeError) as exc:
        raise ReportError(f"inputs are inconsistent: {exc}") from None
    weeks = []
    for w in ontario["weeks"]:
        if _in_scope(w, season, week):
            wed_at, sun_at = decision_times(w)
            weeks.append({"season": w["season"], "week": w["week"],
                          "intended_wednesday": w["intended_wednesday"],
                          "intended_sunday": w["intended_sunday"],
                          "wednesday_capture": w["wednesday"], "sunday_capture": w["sunday"],
                          "wednesday_decision_time": ps._iso(wed_at) if wed_at else None,
                          "sunday_decision_time": ps._iso(sun_at) if sun_at else None})
    return {
        "schema_version": REPORT_SCHEMA_VERSION, "kind": REPORT_KIND,
        "as_of": ps._iso(now.replace(microsecond=0)),
        "filters": {"season": season, "week": week},
        "status": "no_eligible_pairs" if not paired else "ok",
        "rules": {"pairing": PAIRING_RULE, "decision_time": DECISION_RULE,
                  "settlement": f"betting_log.spread_result at each offer's own signed handicap "
                                f"once bet_journal.completion_evidence finds a final score; "
                                f"{STAKE} units per offer: win = bet_journal.win_profit "
                                f"(cent-rounded), push = 0.00, loss = -{STAKE}; "
                                "profit_difference = Sunday - Wednesday",
                  "classification": "ontario_spread_report payoff dominance, decided before "
                                    "the game",
                  "hindsight": HINDSIGHT_NOTE, "independence": INDEPENDENCE_NOTE,
                  "injuries": INJURY_NOTE, "not_claimed": NOT_CLAIMED},
        "inputs": {"ontario_report_sha256": ontario["report_sha256"],
                   "schedule_sha256": src["schedule_sha"],
                   "snapshot_files": sorted(set(src["snapshots"]["snapshot_file"]))
                   if not src["snapshots"].empty else []},
        "weeks": weeks,
        "rollups": rollups(paired, missing, decisions),
        "offers": paired,
        "missing_pairs": missing,
        "model_decisions": decisions,
    }


def build_report(capture_dir: Path = CAPTURE_DIR, manual_dir: Path = MANUAL_DIR,
                 snapshot_dir: Path = SNAPSHOT_DIR, schedule_path: Path = SCHEDULE_PATH, *,
                 season: int | None = None, week: int | None = None,
                 as_of: date | None = None) -> dict:
    as_of = as_of or bj.now_utc().astimezone(on.TORONTO).date()
    now = datetime.combine(as_of, time(12, 0), tzinfo=on.TORONTO)
    return analyze(load_sources(capture_dir, manual_dir, snapshot_dir, schedule_path),
                   now=now, season=season, week=week)


def to_csv(report: dict) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=OFFER_CSV_FIELDS, lineterminator="\n",
                       extrasaction="ignore")
    w.writeheader()
    for o in report["offers"]:
        w.writerow({k: ("" if o.get(k) is None else o.get(k)) for k in OFFER_CSV_FIELDS})
    return buf.getvalue()


def write_report(report: dict, output_dir: Path, protected: tuple = ()) -> tuple[Path, Path]:
    """Write JSON + CSV. Refuses any output directory that is, contains, or
    resolves (symlinks, junctions, '..', letter case) into source data."""
    output_dir = Path(output_dir)
    roots = (fr.DATA_DIR, on.STORE_DIR, bj.JOURNAL_DIR, ROOT / ".git") + \
        tuple(Path(p) for p in protected)
    for root in roots:
        if osr._inside(output_dir, root) or osr._inside(root, output_dir):
            raise ReportError(f"refusing to write reports to {output_dir}: it overlaps source "
                              f"data at {root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = output_dir / JSON_NAME, output_dir / CSV_NAME
    for target in (json_path, csv_path):        # re-check: links created meanwhile
        if any(osr._inside(target, root) for root in roots):
            raise ReportError(f"refusing to write {target}: it resolves into source data")
    osr._atomic_write(json_path, (json.dumps(report, indent=1, sort_keys=True, allow_nan=False)
                                  + "\n").encode("utf-8"))
    osr._atomic_write(csv_path, to_csv(report).encode("utf-8"))
    return json_path, csv_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Read-only 'Was waiting worthwhile?' analysis")
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--as-of", type=date.fromisoformat,
                    help="result cutoff (Toronto date, default today): only games dated before "
                         "it count as final. Inputs are the CURRENT files")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--capture-dir", type=Path, default=CAPTURE_DIR)
    ap.add_argument("--manual-dir", type=Path, default=MANUAL_DIR)
    ap.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_DIR)
    ap.add_argument("--schedule", type=Path, default=SCHEDULE_PATH)
    args = ap.parse_args(argv)
    try:
        report = build_report(args.capture_dir, args.manual_dir, args.snapshot_dir,
                              args.schedule, season=args.season, week=args.week,
                              as_of=args.as_of)
        json_path, csv_path = write_report(
            report, args.output_dir, (args.capture_dir, args.manual_dir, args.snapshot_dir,
                                      Path(args.schedule).parent))
    except (ReportError, OSError, ValueError) as exc:
        print(f"[ontario_waiting_report] failed: {exc}")
        return 1
    d = report["rollups"]["denominators"]
    print(f"[ontario_waiting_report] {report['status']}: {d['team_side_comparisons']} team-side "
          f"comparisons in {d['games_with_paired_captures']} games; model decisions at both "
          f"times for {d['games_with_model_decisions_at_both_times']} of "
          f"{d['games_in_model_table']} games -> {json_path}, {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
