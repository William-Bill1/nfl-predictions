"""Frozen-pregame spread performance report (read-only).

Evaluates the spread model only on predictions that were frozen BEFORE
kickoff in ``data_files/pregame_snapshots/`` (see docs/PREGAME_SNAPSHOTS.md).
It never regenerates predictions, never reads today's probabilities or lines,
and never treats the rewritten predictions CSV as prospective evidence.

Rules (see docs/FROZEN_SPREAD_REPORT.md):

* Selection - one observation per game, chosen by pregame_snapshots'
  ``select_captures`` (every snapshot validated; captured strictly before the
  kickoff the snapshot recorded): the latest such capture by default, or the
  earliest with ``--which earliest``. The capture is chosen FIRST and its
  prediction status inspected afterwards - no searching other captures for a
  probability or a signal. Games without a capture are never backfilled.
* Timing cross-check - the chosen capture must also be strictly before the
  game's kickoff in the CURRENT schedule. If the game was moved earlier past
  the capture, the game is ``timing_unverified`` and excluded (no fallback).
  Scheduled kickoffs are the only timing evidence stored; actual start times
  are not.
* Outcome - the frozen underdog at its frozen handicap (+|spread_line|),
  settled with betting_log.spread_result once bet_journal.completion_evidence
  finds conservative evidence of a completed game. Anything ambiguous,
  duplicated or incomplete stays unresolved.
* Metrics - Brier score and log loss vs a 50% baseline on exactly the same
  non-push games; calibration bins; per season/week and per code revision;
  signal-only results as a separate subset. Pushes are reported separately.
* Market - only automated Ontario sportsbook feeds (CA-ON jurisdiction,
  Ontario role) in archived, validated Ontario spread captures
  (data_files/ontario_spreads/captures) taken strictly before kickoff, whose
  quote matches the same game, the frozen underdog and its exact handicap,
  with both prices (devigged). The US reference feed and manual quotes are
  excluded before any line, probability or price is read. Anything else is a
  coverage gap, never a substitute.
* Scope - the explicit --season, otherwise the latest season in the
  schedule (never derived from which predictions were selected). --as-of is an
  outcome cutoff over the CURRENT files, not a historical reconstruction.
* Simulated returns - signal-only, frozen terms, 100 units staked per bet, at
  an archived matched price; an assumed -110 scenario is reported separately
  and labelled as such. Actual wagers (the bet journal) are never read.

Outputs (deterministic for the same inputs and --as-of date): a JSON report
and a per-game CSV in a git-ignored directory (default reports/frozen_spread/),
refused anywhere that overlaps source data. Corrupt inputs fail the run.

    python scripts/frozen_spread_report.py [--which latest|earliest]
        [--season 2026] [--week 5] [--as-of 2026-10-09] [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402

import bet_journal as bj  # noqa: E402
import betting_log  # noqa: E402
import ontario_spread_report as osr  # noqa: E402
import ontario_spreads as on  # noqa: E402
import pregame_snapshots as ps  # noqa: E402

REPORT_KIND = "frozen_pregame_spread_report"
REPORT_SCHEMA_VERSION = 1
DATA_DIR = ROOT / "data_files"
SNAPSHOT_DIR = DATA_DIR / "pregame_snapshots"
SCHEDULE_PATH = DATA_DIR / ps.SCHEDULE_NAME
CAPTURE_DIR = on.CAPTURE_DIR
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "frozen_spread"
JSON_NAME = "frozen_spread_report.json"
CSV_NAME = "frozen_spread_games.csv"

LOG_LOSS_EPS = 1e-15            # probabilities clipped to [eps, 1 - eps] for log loss only
CALIBRATION_BINS = 10           # equal-width bins over [0, 1]
ASSUMED_PRICE = -110            # the separately labelled scenario
STAKE = Decimal("100.00")       # simulated stake per bet (units)
ROUND = 6

# Per-game status (exactly one per selected game)
EVALUATED = "evaluated"                  # underdog won or lost at the frozen handicap
PUSH = "push"
PENDING = "pending"                      # no conservative completion evidence yet
INVALID_OUTCOME = "invalid_outcome"      # duplicated / inconsistent / unmatched result
TIMING_UNVERIFIED = "timing_unverified"  # capture not before the current scheduled kickoff
STATUS_ORDER = (EVALUATED, PUSH, PENDING, ps.NO_LINE, ps.PICKEM, ps.NO_PROBABILITY,
                INVALID_OUTCOME, TIMING_UNVERIFIED)

# Market match status (per game)
MATCHED = "matched"
NOT_APPLICABLE = "not_applicable"                  # no frozen probability at a valid line
NO_CAPTURE = "no_archived_capture_before_kickoff"
NO_ONTARIO_QUOTE = "no_fresh_ontario_quote"          # capture has no usable Ontario-feed quote
NO_EXACT_QUOTE = "no_quote_at_frozen_handicap"
MARKET_ORDER = (MATCHED, NO_EXACT_QUOTE, NO_ONTARIO_QUOTE, NO_CAPTURE, NOT_APPLICABLE)
ONTARIO_ROLES = ("ontario", "ontario_paid_tier")

# Columns of the per-game CSV holding identifiers or hashes: read them as text
# (read_games_csv), for the reason given at pregame_snapshots.SELECTION_TEXT_COLUMNS.
TEXT_COLUMNS = ps.SELECTION_TEXT_COLUMNS + ("market_capture_run_id", "market_capture_file")

CSV_FIELDS = (
    "season", "week", "game_id", "home_team", "away_team", "status", "status_reason",
    "run_id", "snapshot_file", "captured_at", "code_revision", "config_id", "feature_set_id",
    "artifact_sha256", "frozen_kickoff_utc", "current_kickoff_utc", "kickoff_check",
    "prediction_status", "spread_line", "underdog_team", "underdog_handicap",
    "prob_underdog_covers", "bet_signal", "home_score", "away_score", "underdog_result",
    "underdog_covered", "market_status", "market_status_detail", "market_capture_run_id",
    "market_capture_file", "market_captured_at", "market_books", "market_prob_underdog",
    "archived_price", "archived_price_book", "sim_profit_archived", "sim_profit_assumed_110",
)

PENDING_REASONS = ("no final score yet", "game day is not before today")
SPREAD_TRACKER_NOTE = ("data_files/market_spreads_week*_*.csv (spread tracker) is a re-fetchable "
                       "cache (overwritten with --no-cache) with no integrity checksum, so it is "
                       "not treated as an archived pregame quote")


class ReportError(RuntimeError):
    """An input is missing, corrupt or inconsistent, or the output is refused."""


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def load_schedule(path: Path) -> tuple[pd.DataFrame, str]:
    try:
        data, sha = ps.read_bytes_once(path)
        df = ps.parse_tsv(data)
    except (OSError, ValueError) as exc:
        raise ReportError(f"schedule {path} can't be read: {exc}") from None
    need = {"game_id", "season", "week", "gameday", "gametime", "home_team", "away_team",
            "home_score", "away_score", "result", "total", "overtime"}
    missing = sorted(need - set(df.columns))
    if missing:
        raise ReportError(f"schedule {path} lacks column(s) {missing}")
    df["game_id"] = df["game_id"].astype(str)
    return df, sha


def load_market_captures(capture_dir: Path) -> list[dict]:
    """Every Ontario capture, each validated; a corrupt file fails the run."""
    out = []
    for path in sorted(Path(capture_dir).glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            on.validate_capture(doc, str(path))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ReportError(f"market capture {path} is invalid: {exc}") from None
        out.append({**doc, "_file": path.name})
    return out


def select_frozen(snapshot_dir: Path, which: str) -> pd.DataFrame:
    """One capture per game (pregame_snapshots.select_captures, no status
    filter - the capture is chosen before its prediction is looked at)."""
    if not Path(snapshot_dir).is_dir():
        raise ReportError(f"snapshot directory {snapshot_dir} not found")
    try:
        return ps.select_captures(snapshot_dir, which=which)
    except (ps.SnapshotValidationError, ValueError, OSError) as exc:
        raise ReportError(f"snapshot input is invalid: {exc}") from None


def read_games_csv(path) -> pd.DataFrame:
    """Read this report's per-game CSV with identifier/hash columns as text."""
    return pd.read_csv(path, dtype={c: str for c in TEXT_COLUMNS})


# --------------------------------------------------------------------------
# Per-game evaluation
# --------------------------------------------------------------------------

def _none(v):
    """JSON-safe scalar (NaN/NA -> None, numpy -> Python)."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v.item() if hasattr(v, "item") else v


def current_kickoff(rows: pd.DataFrame) -> datetime | None:
    if len(rows) != 1:
        return None
    ko, _ = ps.kickoff_utc(rows.iloc[0].get("gameday"), rows.iloc[0].get("gametime"))
    return ko


def kickoff_check(frozen: datetime, current: datetime | None, captured: datetime) -> str:
    if current is None:
        return "current_kickoff_unavailable"
    if not captured < current:
        return "capture_not_before_current_kickoff"
    return "unchanged" if current == frozen else \
        "moved_later" if current > frozen else "moved_earlier"


def evaluate_game(sel: dict, schedule: pd.DataFrame, now: datetime) -> dict:
    """Status, frozen terms and (when evidenced) the outcome of one selected game."""
    line = _none(sel["spread_line"])
    prob = _none(sel["prob_underdog_covers"])
    status_pred = sel["prediction_status"]
    handicap = abs(float(line)) if sel["line_status"] == "valid" else None
    frozen_ko, captured = ps._parse_utc(sel["kickoff_utc"]), ps._parse_utc(sel["captured_at"])
    rows = schedule[schedule["game_id"] == sel["game_id"]]
    cur = current_kickoff(rows)
    g = {
        "season": int(sel["season"]), "week": int(sel["week"]), "game_id": sel["game_id"],
        "home_team": sel["home_team"], "away_team": sel["away_team"],
        "run_id": sel["run_id"], "snapshot_file": sel["snapshot_file"],
        "captured_at": sel["captured_at"], "code_revision": sel["code_revision"],
        "config_id": sel["config_id"], "feature_set_id": sel["feature_set_id"],
        "artifact_sha256": sel["artifact_sha256"], "frozen_kickoff_utc": sel["kickoff_utc"],
        "current_kickoff_utc": ps._iso(cur) if cur else None,
        "kickoff_check": kickoff_check(frozen_ko, cur, captured),
        "prediction_status": status_pred, "spread_line": line,
        "underdog_team": _none(sel["underdog_team"]), "underdog_handicap": handicap,
        "prob_underdog_covers": prob, "bet_signal": bool(sel["bet_signal"]),
        "home_score": None, "away_score": None, "underdog_result": None,
        "underdog_covered": None, "status": None, "status_reason": "",
    }
    if g["kickoff_check"] == "capture_not_before_current_kickoff":
        g["status"] = TIMING_UNVERIFIED
        g["status_reason"] = (f"captured {sel['captured_at']} is not before the current scheduled "
                              f"kickoff {g['current_kickoff_utc']}; no earlier capture substituted")
        return g
    if len(rows) == 1 and (rows.iloc[0]["home_team"], rows.iloc[0]["away_team"]) != \
            (sel["home_team"], sel["away_team"]):
        g["status"], g["status_reason"] = INVALID_OUTCOME, "schedule teams differ from the snapshot"
        return g
    if status_pred != ps.PREDICTED:
        g["status"], g["status_reason"] = status_pred, "no frozen probability at a valid line"
        return g
    evidence, why = bj.completion_evidence(schedule, sel["game_id"], now)
    if evidence is None:
        g["status"] = PENDING if why.startswith(PENDING_REASONS) else INVALID_OUTCOME
        g["status_reason"] = why
        return g
    h, a = evidence["home_score"], evidence["away_score"]
    result = betting_log.spread_result(g["underdog_team"], g["home_team"], g["away_team"],
                                       handicap, h, a)
    g.update(home_score=h, away_score=a, underdog_result=result,
             underdog_covered=None if result == "push" else int(result == "win"),
             status=PUSH if result == "push" else EVALUATED,
             status_reason=f"underdog {g['underdog_team']} +{handicap:g} settled {result}")
    return g


# --------------------------------------------------------------------------
# Market (archived Ontario captures only)
# --------------------------------------------------------------------------

def implied(price: int) -> float:
    return -price / (-price + 100) if price < 0 else 100 / (price + 100)


def devig(dog_price: int, fav_price: int) -> float:
    d, f = implied(dog_price), implied(fav_price)
    return d / (d + f)


def _decimal_odds(price: int) -> Decimal:
    return Decimal(1) + (Decimal(100) / Decimal(-price) if price < 0 else Decimal(price) / 100)


def is_ontario_feed(q: dict) -> bool:
    """An automated Ontario sportsbook feed quote: the Ontario report's own
    grouping (registered CA-ON book with an Ontario role, source the_odds_api)
    AND the quote's stored jurisdiction/role agreeing with it. US reference
    (FanDuel US) and manual quotes are never part of the market baseline."""
    return (osr._group_for(q) == osr.GROUP_ONTARIO and q.get("jurisdiction") == "CA-ON"
            and q.get("role") in ONTARIO_ROLES)


def market_for(g: dict, captures: list[dict], which: str) -> dict:
    """The archived market view of the frozen underdog at its exact handicap.

    One capture is chosen first (latest/earliest strictly before both the
    frozen and the current kickoff, by (captured_at, run_id)); only its quotes
    are used. A quote counts only if 'quoted' (fresh), its last update is
    before kickoff, the underdog's point equals the frozen handicap and the
    other side mirrors it, with both prices present. No other capture is
    searched for a better match."""
    empty = {"market_status": None, "market_status_detail": "", "market_capture_run_id": None,
             "market_capture_file": None, "market_captured_at": None, "market_books": None,
             "market_prob_underdog": None, "archived_price": None, "archived_price_book": None}
    if g["prediction_status"] != ps.PREDICTED or g["status"] == TIMING_UNVERIFIED:
        return {**empty, "market_status": NOT_APPLICABLE}
    limits = [ps._parse_utc(g["frozen_kickoff_utc"])]
    if g["current_kickoff_utc"]:
        limits.append(ps._parse_utc(g["current_kickoff_utc"]))
    cands = []
    for c in captures:
        entry = next((x for x in c["games"] if x["game_id"] == g["game_id"]), None)
        if entry is None:
            continue
        taken = max(on.parse_utc(c["captured_at"]), on.parse_utc(c["received_at"]))
        if all(taken < k for k in limits + [on.parse_utc(entry["kickoff_utc"])]):
            cands.append((taken, c["run_id"], c, entry))
    if not cands:
        return {**empty, "market_status": NO_CAPTURE,
                "market_status_detail": "no archived capture listing this game before kickoff"}
    cands.sort(key=lambda t: (t[0], t[1]))
    taken, run_id, cap, entry = cands[-1] if which == "latest" else cands[0]
    base = {**empty, "market_capture_run_id": run_id, "market_capture_file": cap["_file"],
            "market_captured_at": ps._iso(taken)}
    dog_home = g["underdog_team"] == g["home_team"]
    # Restrict to fresh automated Ontario feeds FIRST: nothing from the US
    # reference or manual entries reaches the lines, probabilities or prices.
    ontario = [q for q in entry["quotes"] if is_ontario_feed(q) and q["status"] == "quoted"]
    if not ontario:
        return {**base, "market_status": NO_ONTARIO_QUOTE,
                "market_status_detail": "the chosen capture has no fresh automated Ontario quote "
                                        "for this game (US reference and manual quotes excluded)"}
    matched, seen_points = [], set()
    for q in ontario:
        dog_pt, dog_pr = (q["home_point"], q["home_price"]) if dog_home else \
            (q["away_point"], q["away_price"])
        fav_pt, fav_pr = (q["away_point"], q["away_price"]) if dog_home else \
            (q["home_point"], q["home_price"])
        seen_points.add(dog_pt)
        stamp = q["market_last_update"] or q["bookmaker_last_update"]
        try:
            updated = on.parse_provider_time(stamp)
        except (TypeError, ValueError):
            continue
        if not all(updated < k for k in limits) or updated > taken:
            continue
        if dog_pt == g["underdog_handicap"] and fav_pt == -g["underdog_handicap"] \
                and on.valid_american(dog_pr) and on.valid_american(fav_pr):
            matched.append((q["book_key"], int(dog_pr), int(fav_pr)))
    if not matched:
        pts = ", ".join(f"{p:+g}" for p in sorted(p for p in seen_points if p is not None))
        return {**base, "market_status": NO_EXACT_QUOTE,
                "market_status_detail": f"underdog quoted at [{pts or 'none'}], frozen handicap "
                                        f"+{g['underdog_handicap']:g}"}
    matched.sort()
    prob = sum(devig(d, f) for _, d, f in matched) / len(matched)
    # Simulated price: the LOWEST-paying matched price (conservative), ties by book key.
    book, price, _ = min(matched, key=lambda m: (_decimal_odds(m[1]), m[0]))
    return {**base, "market_status": MATCHED, "market_books": len(matched),
            "market_status_detail": "devigged two-way prices, mean over Ontario books: "
                                    + ", ".join(b for b, _, _ in matched),
            "market_prob_underdog": round(prob, ROUND), "archived_price": price,
            "archived_price_book": book}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def prob_metrics(pairs: list[tuple[float, int]]) -> dict:
    """Brier score and log loss of (probability, covered) pairs vs a 50% baseline."""
    n = len(pairs)
    out = {"n": n, "covers": sum(y for _, y in pairs), "observed_cover_rate": None,
           "mean_probability": None, "brier": None, "brier_baseline_50": 0.25 if n else None,
           "log_loss": None, "log_loss_baseline_50": round(math.log(2), ROUND) if n else None,
           "n_probability_0_or_1": sum(p in (0.0, 1.0) for p, _ in pairs),
           "n_clipped_for_log_loss": sum(not LOG_LOSS_EPS <= p <= 1 - LOG_LOSS_EPS for p, _ in pairs)}
    if not n:
        return out
    clip = [(min(max(p, LOG_LOSS_EPS), 1 - LOG_LOSS_EPS), y) for p, y in pairs]
    out.update(
        observed_cover_rate=round(out["covers"] / n, ROUND),
        mean_probability=round(sum(p for p, _ in pairs) / n, ROUND),
        brier=round(sum((p - y) ** 2 for p, y in pairs) / n, ROUND),
        log_loss=round(-sum(y * math.log(p) + (1 - y) * math.log(1 - p) for p, y in clip) / n,
                       ROUND))
    return out


def calibration(pairs: list[tuple[float, int]], bins: int = CALIBRATION_BINS) -> list[dict]:
    rows = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        inside = [(p, y) for p, y in pairs if min(int(p * bins + 1e-9), bins - 1) == i]
        n = len(inside)
        rows.append({"bin": f"[{lo:.1f}, {hi:.1f}{']' if i == bins - 1 else ')'}", "n": n,
                     "mean_probability": round(sum(p for p, _ in inside) / n, ROUND) if n else None,
                     "observed_cover_rate": round(sum(y for _, y in inside) / n, ROUND) if n else None})
    return rows


def _pairs(games: list[dict]) -> list[tuple[float, int]]:
    return [(g["prob_underdog_covers"], g["underdog_covered"]) for g in games
            if g["status"] == EVALUATED and g["prob_underdog_covers"] is not None]


def grouped_metrics(games: list[dict]) -> dict:
    evaluated = [g for g in games if g["status"] == EVALUATED]
    by_week, by_rev = {}, {}
    for g in evaluated:
        by_week.setdefault(f"{g['season']}-W{g['week']:02d}", []).append(g)
        by_rev.setdefault(g["code_revision"], []).append(g)
    return {"overall": prob_metrics(_pairs(evaluated)),
            "pushes_excluded": sum(g["status"] == PUSH for g in games),
            "by_season_week": {k: prob_metrics(_pairs(v)) for k, v in sorted(by_week.items())},
            "by_code_revision": {k: prob_metrics(_pairs(v)) for k, v in sorted(by_rev.items())},
            "calibration": calibration(_pairs(evaluated))}


def market_comparison(games: list[dict]) -> dict:
    sub = [g for g in games if g["status"] == EVALUATED and g["market_status"] == MATCHED]
    counts = {s: sum(g["market_status"] == s for g in games) for s in MARKET_ORDER}
    if not sub:
        reason = ("no evaluated (non-push, completed) game has an archived pregame quote at its "
                  "exact frozen handicap; " + SPREAD_TRACKER_NOTE)
        return {"status": "unavailable", "reason": reason, "matched_games": 0,
                "market_status_counts": counts, "source": "data_files/ontario_spreads/captures"}
    model = [(g["prob_underdog_covers"], g["underdog_covered"]) for g in sub]
    market = [(g["market_prob_underdog"], g["underdog_covered"]) for g in sub]
    return {"status": "available", "reason": "", "matched_games": len(sub),
            "market_status_counts": counts, "source": "data_files/ontario_spreads/captures",
            "note": SPREAD_TRACKER_NOTE,
            "model": prob_metrics(model), "market": prob_metrics(market),
            "baseline_50": prob_metrics([(0.5, y) for _, y in model])}


def simulate(games: list[dict], archived: bool) -> dict:
    """Signal-only simulated returns at frozen terms (100 units per bet)."""
    signals = [g for g in games if g["bet_signal"]]
    settled = [g for g in signals if g["status"] in (EVALUATED, PUSH)]
    bets, no_price = [], 0
    for g in settled:
        price = g["archived_price"] if archived else ASSUMED_PRICE
        if price is None:
            no_price += 1
            continue
        result = g["underdog_result"]
        profit = bj.win_profit(int(price), STAKE) if result == "win" else \
            -STAKE if result == "loss" else Decimal("0.00")
        bets.append((g, profit))
        g["sim_profit_archived" if archived else "sim_profit_assumed_110"] = str(profit)
    staked = STAKE * len(bets)
    net = sum((p for _, p in bets), Decimal("0.00"))
    return {"bets": len(bets),
            "wins": sum(g["underdog_result"] == "win" for g, _ in bets),
            "losses": sum(g["underdog_result"] == "loss" for g, _ in bets),
            "pushes": sum(g["underdog_result"] == "push" for g, _ in bets),
            "staked_units": str(staked), "net_units": str(net),
            "roi": round(float(net / staked), ROUND) if bets else None,
            "signals_selected": len(signals),
            "signals_not_settled": len(signals) - len(settled),
            "settled_without_archived_price": no_price if archived else None}


# --------------------------------------------------------------------------
# Coverage of completed games
# --------------------------------------------------------------------------

def first_snapshot_capture(snapshot_dir: Path) -> datetime | None:
    """captured_at of the earliest snapshot file (each validated), or None when
    no snapshot history exists at all. Independent of any game's selection."""
    first = None
    for path in sorted(Path(snapshot_dir).glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            ps.validate_snapshot(doc, str(path))
        except (OSError, ValueError) as exc:
            raise ReportError(f"snapshot input is invalid: {path}: {exc}") from None
        cap = ps._parse_utc(doc["captured_at"])
        first = cap if first is None else min(first, cap)
    return first


def scope_seasons(schedule: pd.DataFrame, season: int | None) -> tuple[list[int], str]:
    """Evaluation seasons, defined WITHOUT looking at predictions: the explicit
    --season, otherwise the latest season in the schedule."""
    if season is not None:
        return [season], "explicit --season"
    seasons = pd.to_numeric(schedule["season"], errors="coerce").dropna()
    if seasons.empty:
        raise ReportError("the schedule lists no seasons; pass --season")
    return [int(seasons.max())], ("default: the latest season in the schedule (max season); "
                                  "pass --season for another")


NO_HISTORY = "no_snapshot_history"
BEFORE_FIRST = "before_first_snapshot"
AFTER_FIRST = "after_first_snapshot"
KICKOFF_UNKNOWN = "kickoff_unavailable"


def completed_without_snapshot(schedule: pd.DataFrame, selected: set, seasons: list[int],
                               first_capture: datetime | None, now: datetime,
                               week: int | None) -> dict:
    """Completed games (conservative evidence) in scope that have no selected
    frozen capture - never backfilled, only counted and listed:

    * no_snapshot_history   - there are no snapshot files at all;
    * before_first_snapshot - kickoff at or before the first snapshot's capture;
    * after_first_snapshot  - kickoff after it: a genuine coverage gap;
    * kickoff_unavailable   - the current schedule gives no usable kickoff."""
    buckets = {NO_HISTORY: [], BEFORE_FIRST: [], AFTER_FIRST: [], KICKOFF_UNKNOWN: []}
    completed = with_snapshot = 0
    in_scope = schedule[pd.to_numeric(schedule["season"], errors="coerce").isin(seasons)]
    for gid in sorted(set(in_scope["game_id"])):
        rows = schedule[schedule["game_id"] == gid]
        if week is not None and int(rows.iloc[0]["week"]) != week:
            continue
        evidence, _ = bj.completion_evidence(schedule, gid, now)
        if evidence is None:
            continue
        completed += 1
        if gid in selected:
            with_snapshot += 1
            continue
        ko = current_kickoff(rows)
        bucket = NO_HISTORY if first_capture is None else KICKOFF_UNKNOWN if ko is None else \
            AFTER_FIRST if ko > first_capture else BEFORE_FIRST
        buckets[bucket].append(gid)
    return {"completed_games_in_scope": completed,
            "completed_with_selected_snapshot": with_snapshot,
            "completed_without_snapshot": completed - with_snapshot,
            "first_snapshot_captured_at": ps._iso(first_capture) if first_capture else None,
            **{k: len(v) for k, v in buckets.items()},
            "after_first_snapshot_games": buckets[AFTER_FIRST],
            "kickoff_unavailable_games": buckets[KICKOFF_UNKNOWN],
            "note": "completed games in scope with no selected frozen capture; never evaluated "
                    "or backfilled. no_snapshot_history: no snapshot files exist; "
                    "before_first_snapshot: kickoff at or before the first snapshot's capture; "
                    "after_first_snapshot: a coverage gap after snapshots began."}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

AS_OF_SEMANTICS = (
    "Outcome cutoff only, not a historical reconstruction: a game counts as completed only "
    "if its game day is before this date (America/Toronto). The inputs are the snapshot, "
    "capture and schedule files as they exist when the report runs, so snapshots or captures "
    "written after this date and later score corrections may be present.")

LIMITATIONS = (
    "Small sample: the report covers only games with a frozen pregame capture. It is not the "
    "whole season, and no statistical significance is claimed for any difference.",
    "No profitability claim: simulated returns use frozen terms and archived or assumed prices "
    "for a handful of games; they are not evidence of an edge.",
    "Timing: captures are nightly. Eligibility uses scheduled kickoffs (recorded in the "
    "snapshot and in the current schedule); actual start times are not stored, so a game "
    "started early without a schedule change can't be detected.",
    "Line source: the frozen line is nflverse's schedule line at capture time, not a sportsbook "
    "price; market comparison needs an archived quote at that exact handicap.",
    "Market timing: the archived market capture and the frozen model capture are chosen "
    "independently (both strictly pregame) and can be hours or days apart; compare "
    "market_captured_at with captured_at per game.",
    "Outcomes: completion uses the schedule's conservative evidence (consistent integer scores, "
    "game day before the as-of date), not an authoritative final-status flag.",
)


def build_report(snapshot_dir: Path = SNAPSHOT_DIR, schedule_path: Path = SCHEDULE_PATH,
                 capture_dir: Path = CAPTURE_DIR, *, which: str = "latest",
                 season: int | None = None, week: int | None = None,
                 as_of: date | None = None) -> dict:
    if which not in ("latest", "earliest"):
        raise ReportError("which must be 'latest' or 'earliest'")
    as_of = as_of or bj.now_utc().astimezone(on.TORONTO).date()
    now = datetime.combine(as_of, time(12, 0), tzinfo=on.TORONTO)
    schedule, schedule_sha = load_schedule(Path(schedule_path))
    picks = select_frozen(Path(snapshot_dir), which)
    first_capture = first_snapshot_capture(Path(snapshot_dir))
    captures = load_market_captures(Path(capture_dir)) if Path(capture_dir).is_dir() else []
    snapshot_files = sorted(p.name for p in Path(snapshot_dir).glob("*.json"))
    seasons, scope_rule = scope_seasons(schedule, season)

    games, outside_scope = [], 0
    for sel in (picks.to_dict("records") if len(picks) else []):
        if int(sel["season"]) not in seasons or (week is not None and int(sel["week"]) != week):
            outside_scope += 1
            continue
        g = evaluate_game(sel, schedule, now)
        g.update(market_for(g, captures, which))
        g["sim_profit_archived"] = g["sim_profit_assumed_110"] = None
        games.append(g)
    games.sort(key=lambda g: (g["frozen_kickoff_utc"], g["game_id"]))

    signal_games = [g for g in games if g["bet_signal"]]
    archived = simulate(games, archived=True)
    assumed = simulate(games, archived=False)
    status_counts = {s: sum(g["status"] == s for g in games) for s in STATUS_ORDER}
    started = [g for g in games
               if ps._parse_utc(g["current_kickoff_utc"] or g["frozen_kickoff_utc"])
               .astimezone(on.TORONTO).date() < as_of]
    started_counts = {s: sum(g["status"] == s for g in started) for s in STATUS_ORDER}
    n_eval = status_counts[EVALUATED]
    return {
        "schema_version": REPORT_SCHEMA_VERSION, "kind": REPORT_KIND,
        "as_of_date": as_of.isoformat(),
        "as_of_semantics": AS_OF_SEMANTICS,
        "scope": {"seasons": seasons, "week": week, "rule": scope_rule,
                  "selected_outside_scope": outside_scope,
                  "snapshot_history": "none" if first_capture is None else
                  f"first snapshot captured {ps._iso(first_capture)}"},
        "selection": {
            "which": which, "season": season, "week": week,
            "rule": (f"{which} snapshot captured strictly before the kickoff it recorded "
                     "(pregame_snapshots.select_captures, no status filter); the capture is "
                     "chosen first, then its prediction inspected; it must also be before the "
                     "current scheduled kickoff, otherwise timing_unverified (no fallback)"),
        },
        "inputs": {"snapshot_files": snapshot_files, "schedule_sha256": schedule_sha,
                   "market_capture_files": sorted(c["_file"] for c in captures),
                   "excluded_market_sources": [SPREAD_TRACKER_NOTE]},
        "coverage": {
            "selected_games": len(games), "by_status": status_counts,
            "games_dated_before_as_of": len(started),
            "by_status_games_dated_before_as_of": started_counts,
            "bet_signals": len(signal_games),
            "completed_without_snapshot": completed_without_snapshot(
                schedule, {g["game_id"] for g in games}, seasons, first_capture, now, week),
            "sample_note": "no snapshot history exists: nothing can be evaluated"
            if first_capture is None else
            (f"{n_eval} evaluated game(s) with a frozen pregame probability; a "
                            "captured sample, not the whole season, and too small for "
                            "conclusions" if n_eval < 100 else
                            f"{n_eval} evaluated games; a captured sample, not the whole season"),
        },
        "probability_metrics": {
            "all_valid_probabilities": grouped_metrics(games),
            "signal_only": grouped_metrics(signal_games),
            "conventions": {"target": "underdog covers the frozen handicap (1) or not (0)",
                            "pushes": "excluded from probability metrics, counted separately",
                            "baseline": "constant 50% on exactly the same games",
                            "log_loss_clipping": f"probabilities clipped to [{LOG_LOSS_EPS:g}, "
                                                 f"1 - {LOG_LOSS_EPS:g}] for log loss only",
                            "calibration": f"{CALIBRATION_BINS} equal-width bins; 1.0 falls in "
                                           "the last bin"},
        },
        "market_comparison": market_comparison(games),
        "simulated_returns": {
            "conventions": {"bets": "signal-only (bet_signal true in the selected frozen capture)",
                            "terms": "frozen underdog at the frozen handicap",
                            "stake": f"{STAKE} units per bet",
                            "push": "stake returned, profit 0, counted as a bet",
                            "roi": "net units / units staked on settled bets (pushes included)",
                            "actual_wagers": "never read (bet journal excluded)"},
            "archived_price": {**archived, "label": "lowest-paying matched archived Ontario "
                                                    "quote at the exact frozen handicap; bets "
                                                    "without one are excluded and counted"},
            "assumed_minus_110": {**assumed, "label": "SCENARIO ONLY: assumed -110 for every "
                                                      "signal; not an offered price"},
        },
        "limitations": list(LIMITATIONS),
        "games": games,
    }


def to_csv(report: dict) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for g in report["games"]:
        w.writerow({k: ("" if g.get(k) is None else g.get(k)) for k in CSV_FIELDS})
    return buf.getvalue()


def protected_roots(snapshot_dir: Path, schedule_path: Path, capture_dir: Path) -> tuple:
    return (DATA_DIR, Path(snapshot_dir), Path(schedule_path).parent, Path(capture_dir),
            bj.JOURNAL_DIR, on.STORE_DIR, ROOT / ".git")


def write_report(report: dict, output_dir: Path, protected: tuple = ()) -> tuple[Path, Path]:
    """Write JSON + CSV. Refuses any output directory that is, contains, or
    resolves (symlinks, junctions, '..', letter case) into source data."""
    output_dir = Path(output_dir)
    roots = (DATA_DIR,) + tuple(Path(p) for p in protected)
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
    ap = argparse.ArgumentParser(description="Read-only frozen-pregame spread performance report")
    ap.add_argument("--which", choices=("latest", "earliest"), default="latest")
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--as-of", type=date.fromisoformat,
                    help="outcome cutoff (Toronto date, default today): only games dated before "
                         "it count as completed. Inputs are the CURRENT files - not a historical "
                         "reconstruction")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_DIR)
    ap.add_argument("--schedule", type=Path, default=SCHEDULE_PATH)
    ap.add_argument("--capture-dir", type=Path, default=CAPTURE_DIR)
    args = ap.parse_args(argv)
    try:
        report = build_report(args.snapshot_dir, args.schedule, args.capture_dir,
                              which=args.which, season=args.season, week=args.week,
                              as_of=args.as_of)
        json_path, csv_path = write_report(
            report, args.output_dir, protected_roots(args.snapshot_dir, args.schedule,
                                                     args.capture_dir))
    except (ReportError, OSError, ValueError) as exc:
        print(f"[frozen_spread_report] failed: {exc}")
        return 1
    c = report["coverage"]
    print(f"[frozen_spread_report] {c['selected_games']} selected games "
          f"({', '.join(f'{k} {v}' for k, v in c['by_status'].items() if v)}); market "
          f"{report['market_comparison']['status']} -> {json_path}, {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
