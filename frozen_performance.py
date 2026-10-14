"""Display logic for the "Frozen Performance" page.

Read-only. Builds the frozen-pregame spread report (scripts/
frozen_spread_report.py) in memory and shapes it for display: no API calls,
no writes, and none of the report's selection, settlement, coverage or market
logic is re-implemented here - every number shown comes from the report.
Kept out of the page script so it can be tested directly.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

import ontario_line_timing as olt
import ontario_spreads as on
from season_utils import completed_weeks

ROOT = Path(__file__).resolve().parent
if "frozen_spread_report" in sys.modules:
    fr = sys.modules["frozen_spread_report"]
else:
    _spec = importlib.util.spec_from_file_location("frozen_spread_report",
                                                   ROOT / "scripts" / "frozen_spread_report.py")
    fr = importlib.util.module_from_spec(_spec)
    sys.modules["frozen_spread_report"] = fr
    _spec.loader.exec_module(fr)

TORONTO = on.TORONTO
# Read by the page at run time (tests point them at temporary fixtures).
SNAPSHOT_DIR, SCHEDULE_PATH, CAPTURE_DIR = fr.SNAPSHOT_DIR, fr.SCHEDULE_PATH, fr.CAPTURE_DIR

SELECTIONS = {"latest": "Latest pregame snapshot (default)",
              "earliest": "Earliest pregame snapshot"}

STATUS_LABELS = {
    fr.EVALUATED: "Evaluated",
    fr.PUSH: "Push - excluded from probability metrics",
    fr.PENDING: "Pending - no final result yet",
    fr.ps.NO_LINE: "Excluded - no line in the frozen snapshot",
    fr.ps.PICKEM: "Excluded - pick'em line",
    fr.ps.NO_PROBABILITY: "Excluded - no frozen probability",
    fr.INVALID_OUTCOME: "Excluded - result unresolved",
    fr.TIMING_UNVERIFIED: "Excluded - kickoff timing unverified",
}
EXCLUDED = (fr.ps.NO_LINE, fr.ps.PICKEM, fr.ps.NO_PROBABILITY, fr.INVALID_OUTCOME,
            fr.TIMING_UNVERIFIED)
KICKOFF_NOTES = {"moved_later": "kickoff moved later after the capture",
                 "moved_earlier": "kickoff moved earlier (still after the capture)",
                 "current_kickoff_unavailable": "current kickoff time unavailable"}


class IntegrityError(RuntimeError):
    """A source artifact is missing, unreadable or fails validation."""


# --------------------------------------------------------------------------
# Loading (file-dependent only; cached by the page on the fingerprint)
# --------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise IntegrityError(f"{path} can't be read: {exc}") from None


def source_fingerprint(snapshot_dir: Path, schedule_path: Path, capture_dir: Path) -> tuple:
    """Name + SHA-256 of every snapshot, capture and the schedule. Adding,
    changing or deleting any of them gives a new fingerprint (the cache key).
    A file or folder that exists but can't be read (e.g. permission denied)
    raises IntegrityError, like an invalid one - never a crash."""
    out = []
    try:
        for label, d in (("snapshot", Path(snapshot_dir)), ("capture", Path(capture_dir))):
            out.append((label + "_dir", d.name, d.is_dir()))
            for p in sorted(d.glob("*.json")) if d.is_dir() else []:
                out.append((label, p.name, _sha256(p)))
        sp = Path(schedule_path)
        out.append(("schedule", sp.name, _sha256(sp) if sp.is_file() else None))
    except OSError as exc:                      # e.g. a folder that can't be listed
        raise IntegrityError(f"source files can't be read: {exc}") from None
    return tuple(out)


# Schedule columns the page needs before it builds the season/week choices
# (season_utils.completed_weeks filters on game_type). The report's own
# schedule check doesn't require game_type, so it is checked here.
PAGE_SCHEDULE_COLUMNS = ("season", "week", "game_type", "home_score", "away_score")


def load(snapshot_dir: Path, schedule_path: Path, capture_dir: Path, which: str) -> dict:
    """The report's validated, selected sources (no clock involved)."""
    try:
        src = fr.load_sources(Path(snapshot_dir), Path(schedule_path), Path(capture_dir),
                              which=which)
    except (fr.ReportError, ValueError, KeyError, TypeError, OSError) as exc:
        raise IntegrityError(str(exc)) from None
    missing = [c for c in PAGE_SCHEDULE_COLUMNS if c not in src["schedule"].columns]
    if missing:
        raise IntegrityError(f"schedule {schedule_path} lacks column(s) {missing} needed for "
                             "the season and week choices")
    return src


def today_toronto() -> date:
    return datetime.now(timezone.utc).astimezone(TORONTO).date()


def build(src: dict, season: int, week: int | None, as_of: date) -> dict:
    """The report for this outcome cutoff - recomputed on every render,
    outside the file cache, because completion depends on the date."""
    try:
        return fr.report_from_sources(src, season=season, week=week, as_of=as_of)
    except (fr.ReportError, ValueError, KeyError, TypeError) as exc:
        raise IntegrityError(str(exc)) from None


# --------------------------------------------------------------------------
# Season and week choices
# --------------------------------------------------------------------------

def season_options(schedule: pd.DataFrame) -> list[int]:
    """Seasons with at least one completed regular-season game, newest first."""
    seasons = sorted({int(s) for s in pd.to_numeric(schedule["season"], errors="coerce").dropna()},
                     reverse=True)
    return [s for s in seasons if completed_weeks(schedule, s)[0]]


def week_options(schedule: pd.DataFrame, season: int) -> tuple[list[tuple[int, bool]], int | None]:
    """([(week, fully_completed)], default week). Weeks with any result are
    offered; the default is the latest fully completed week (season_utils)."""
    weeks, default = completed_weeks(schedule, season)
    reg = schedule[(schedule["season"] == season) & (schedule["game_type"] == "REG")]
    done = reg["home_score"].notna() & reg["away_score"].notna()
    full = set(int(w) for w, ok in done.groupby(reg["week"]).all().items() if ok)
    return [(w, w in full) for w in weeks], default


def provisional(report: dict, week_complete: bool | None) -> bool:
    """A view is provisional when its week isn't fully played in the schedule,
    the report still has pending or unresolved games in it, or it includes
    games dated on or after the outcome cutoff (not played yet - e.g. future
    weeks whose frozen snapshot had no line, which count as excluded)."""
    c = report["coverage"]
    open_games = c["by_status"].get(fr.PENDING, 0) + c["by_status"].get(fr.INVALID_OUTCOME, 0)
    unplayed = c["selected_games"] - c["games_dated_before_as_of"]
    return week_complete is False or open_games > 0 or unplayed > 0


# --------------------------------------------------------------------------
# Shaping
# --------------------------------------------------------------------------

def _pct(p) -> str:
    return "" if p is None else f"{p:.1%}"


def results_rows(report: dict) -> pd.DataFrame:
    """One row per selected game, from the report's frozen values only."""
    rows = []
    for g in report["games"]:
        final = "" if g["home_score"] is None else \
            f"{g['away_team']} {g['away_score']} - {g['home_score']} {g['home_team']}"
        reason = "" if g["status"] == fr.EVALUATED else g["status_reason"]
        note = KICKOFF_NOTES.get(g["kickoff_check"])
        if note:
            reason = f"{reason}; {note}" if reason else note
        rows.append({
            "Game": f"{g['away_team']} @ {g['home_team']}",
            "Kickoff (Toronto)": olt.toronto(g["current_kickoff_utc"] or g["frozen_kickoff_utc"]),
            "Frozen capture (Toronto)": olt.toronto(g["captured_at"]),
            "Team": g["underdog_team"] or "",
            "Handicap": "" if g["underdog_handicap"] is None
            else olt.signed_handicap(g["underdog_handicap"]),
            "Frozen P(cover)": _pct(g["prob_underdog_covers"]),
            "Frozen signal": "Yes" if g["bet_signal"] else "No",
            "Final score": final,
            "Spread result": g["underdog_result"] or "",
            "Status": STATUS_LABELS.get(g["status"], g["status"]),
            "Reason": reason,
        })
    return pd.DataFrame(rows, columns=["Game", "Kickoff (Toronto)", "Frozen capture (Toronto)",
                                       "Team", "Handicap", "Frozen P(cover)", "Frozen signal",
                                       "Final score", "Spread result", "Status", "Reason"])


def status_counts(games: list[dict]) -> dict:
    """Evaluated / push / pending / excluded counts for a list of report games."""
    by = {s: sum(g["status"] == s for g in games) for s in fr.STATUS_ORDER}
    return {"selected": len(games), "evaluated": by[fr.EVALUATED], "push": by[fr.PUSH],
            "pending": by[fr.PENDING], "excluded": sum(by[s] for s in EXCLUDED),
            "excluded_by_reason": {s: by[s] for s in EXCLUDED if by[s]}}


def summaries(report: dict) -> dict:
    """The two prediction summaries: every frozen prediction, and the frozen
    bet-signal subset - each with the report's own metrics."""
    pm = report["probability_metrics"]
    signal_games = [g for g in report["games"] if g["bet_signal"]]
    return {"all": {**status_counts(report["games"]), "metrics": pm["all_valid_probabilities"]},
            "signal": {**status_counts(signal_games), "metrics": pm["signal_only"]}}


def coverage_rows(report: dict) -> pd.DataFrame:
    c = report["coverage"]
    cw = c["completed_without_snapshot"]
    rows = [("Selected frozen predictions", c["selected_games"])]
    rows += [(STATUS_LABELS.get(s, s), n) for s, n in c["by_status"].items() if n]
    rows += [("Completed games in scope", cw["completed_games_in_scope"]),
             ("- with a selected frozen snapshot", cw["completed_with_selected_snapshot"]),
             ("- no snapshot history at all", cw[fr.NO_HISTORY]),
             ("- kicked off before the first snapshot", cw[fr.BEFORE_FIRST]),
             ("- missing a snapshot after tracking began", cw[fr.AFTER_FIRST]),
             ("- kickoff time unavailable", cw[fr.KICKOFF_UNKNOWN])]
    return pd.DataFrame(rows, columns=["Coverage", "Games"])


def metric_text(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"
