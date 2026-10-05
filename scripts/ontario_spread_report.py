"""Phase 2: read-only Wednesday-vs-Sunday comparison of Ontario spread captures.

Reads the validated, immutable captures written by ``ontario_spreads.py``
(and, only when asked, manual FanDuel Ontario quotes) and compares each
side's Wednesday-noon quote with its Sunday-morning quote for the same NFL
week, sportsbook and team. It describes how the quotes moved. It doesn't say
when to bet, call Sunday a closing line, or claim any ROI.

Nothing here writes to, re-seals or deletes a source file. Reports go to an
output directory (default ``reports/ontario_spreads/``, git-ignored), and
the same inputs always give byte-identical reports: there are no wall-clock
timestamps, and rows are sorted.

Selection rules (see docs/ONTARIO_SPREAD_TRACKING.md, "Comparison report"):

* Only scheduled slots (``wednesday_noon``, ``sunday_morning``) are compared.
  ``ad_hoc`` captures are listed but never used.
* Each slot is represented by its earliest **usable** capture (ties broken by
  ``run_id``), the same capture Phase 1's ``slot_coverage`` reports. Empty and
  US-only captures never represent a slot. Later usable duplicates are listed
  as superseded, and their quotes are never used to fill gaps.
* A Sunday slot pairs only with the Wednesday slot of the same NFL week
  (four days earlier, Toronto date); an earlier Wednesday never stands in.
* A quote is only ever compared with the same game, sportsbook,
  jurisdiction, team and spread market at the other slot. Ontario feeds, the
  US FanDuel reference and manual FanDuel Ontario quotes are separate groups
  and are never merged.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tempfile
from datetime import timedelta
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ontario_spreads as on  # noqa: E402

REPORT_KIND = "ontario_spread_comparison"
REPORT_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "ontario_spreads"
JSON_NAME = "ontario_spread_comparison.json"
CSV_NAME = "ontario_spread_comparison.csv"

WEDNESDAY, SUNDAY = "wednesday_noon", "sunday_morning"
KEY_NUMBERS = (3, 7)

GROUP_ONTARIO = "ontario_api"              # automated Ontario feeds (default)
GROUP_US_REFERENCE = "us_reference_api"    # FanDuel US, only with --include-us-reference
GROUP_MANUAL = "fanduel_ontario_manual"    # manual entries, only with --include-manual
GROUPS = (GROUP_ONTARIO, GROUP_US_REFERENCE, GROUP_MANUAL)

COMPARABLE = ("quoted",)                   # stale/invalid/missing never produce a winner
OUTCOMES = ("unchanged", "equivalent", "sunday_dominates", "wednesday_dominates", "trade_off")
# Integer final margins (team score minus opponent score) checked when comparing
# payoffs; wide enough for any valid handicap (|h| <= 60).
MARGINS = range(-90, 91)

CSV_FIELDS = (
    "group", "season", "week", "game_id", "matchup", "comparison_team", "opponent", "side",
    "book_key", "book_title", "jurisdiction", "source",
    "wed_handicap", "wed_price", "wed_break_even", "wed_quote_status", "wed_slot_id",
    "wed_slot_status", "wed_observed_at", "wed_provider_update", "wed_age_minutes",
    "wed_provider_update_basis", "wed_run_id", "wed_file",
    "sun_handicap", "sun_price", "sun_break_even", "sun_quote_status", "sun_slot_id",
    "sun_slot_status", "sun_observed_at", "sun_provider_update", "sun_age_minutes",
    "sun_provider_update_basis", "sun_run_id", "sun_file",
    "spread_change", "break_even_change", "key_3", "key_7",
    "kickoff_utc", "kickoff_changed", "status", "outcome", "reasons",
)


class ReportError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Odds arithmetic
# --------------------------------------------------------------------------

def break_even(price) -> float | None:
    """Win rate needed to break even at an American price, pushes excluded."""
    if not on.valid_american(price):
        return None
    p = float(price)
    return round(-p / (-p + 100.0) if p < 0 else 100.0 / (p + 100.0), 6)


def key_number_move(wed: float, sun: float, key: int) -> str:
    """How a team's handicap moved relative to key number `key` (either sign).

    unchanged | stays_on | on_both_sides | onto | off | through | none
    ("through" = passed strictly over +key or -key without landing on it).
    """
    if wed == sun:
        return "stays_on" if abs(wed) == key else "unchanged"
    wed_on, sun_on = abs(wed) == key, abs(sun) == key
    if wed_on and sun_on:
        return "on_both_sides"          # e.g. -3 -> +3 (favourite flipped)
    if sun_on:
        return "onto"
    if wed_on:
        return "off"
    lo, hi = min(wed, sun), max(wed, sun)
    if any(lo < k < hi for k in (key, -key)):
        return "through"
    return "none"


def _profit(price) -> Fraction:
    """Profit per 1 unit staked on a win at an American price (exact)."""
    p = int(price)
    return Fraction(100, -p) if p < 0 else Fraction(p, 100)


def settle(handicap: float, price, margin: int) -> Fraction:
    """Result per 1 unit staked, for a final margin (team minus opponent):
    win = profit at the price, push = 0, loss = -1."""
    x = Fraction(margin) + Fraction(handicap).limit_denominator(2)
    return _profit(price) if x > 0 else Fraction(0) if x == 0 else Fraction(-1)


def classify(wed_h: float, wed_price, sun_h: float, sun_price) -> str:
    """Sunday vs Wednesday for the SAME team, decided by payoffs.

    Both bets are settled for every integer final margin (win, push or loss).
    If Sunday's result is at least as good for every margin and better for
    some, Sunday dominates (and vice versa). If each is better for some
    margins, it's a trade-off. Identical quotes are unchanged; different quotes
    with identical payoffs (e.g. -100 vs +100) are equivalent.

    This agrees with the simple rule - larger signed handicap is better, lower
    break-even (pushes excluded) is a better payout - and also covers
    whole-number lines with pushes and prices paying less than the stake.
    """
    if (float(wed_h), int(wed_price)) == (float(sun_h), int(sun_price)):
        return "unchanged"
    diffs = [settle(sun_h, sun_price, m) - settle(wed_h, wed_price, m) for m in MARGINS]
    better, worse = any(d > 0 for d in diffs), any(d < 0 for d in diffs)
    if not better and not worse:
        return "equivalent"
    if better and not worse:
        return "sunday_dominates"
    if worse and not better:
        return "wednesday_dominates"
    return "trade_off"


# --------------------------------------------------------------------------
# Loading (read-only)
# --------------------------------------------------------------------------

def load_inputs(capture_dir: Path, manual_dir: Path | None) -> tuple[list[dict], list[dict], list[dict]]:
    """Validated captures and manual quotes, with their source file names.

    Files are only read. Any file that fails Phase 1 validation stops the
    report (it is never skipped silently)."""
    captures = []
    for path in sorted(Path(capture_dir).glob("*.json")):
        doc = json.loads(path.read_bytes().decode("utf-8"))
        on.validate_capture(doc, str(path))
        captures.append({**doc, "_file": path.name})
    manual = []
    if manual_dir is not None:
        for path in sorted(Path(manual_dir).glob("*.json")):
            doc = json.loads(path.read_bytes().decode("utf-8"))
            on.validate_manual(doc, str(path))
            manual.append({**doc, "_file": path.name})
    inputs = [{"file": c["_file"], "kind": c["kind"], "payload_sha256": c[on.CHECKSUM_FIELD]}
              for c in captures] + \
             [{"file": m["_file"], "kind": m["kind"], "payload_sha256": m[on.CHECKSUM_FIELD]}
              for m in manual]
    return captures, manual, inputs


def select_slot_captures(captures: list[dict]) -> tuple[dict[str, dict], list[dict]]:
    """slot_id -> the capture representing it, plus notes on every capture
    that doesn't represent its slot (and why)."""
    by_slot: dict[str, list[dict]] = {}
    notes = []
    for c in captures:
        if c["slot"]["name"] not in (WEDNESDAY, SUNDAY):
            notes.append({"file": c["_file"], "run_id": c["run_id"], "slot_id": c["slot"]["slot_id"],
                          "reason": "ad_hoc_capture_not_compared"})
            continue
        by_slot.setdefault(c["slot"]["slot_id"], []).append(c)
    chosen = {}
    for slot_id, docs in sorted(by_slot.items()):
        docs = sorted(docs, key=lambda d: (d["captured_at"], d["run_id"]))
        usable = [d for d in docs if d["usable"]]
        if usable:
            chosen[slot_id] = usable[0]
        for d in docs:
            if usable and d is usable[0]:
                continue
            notes.append({"file": d["_file"], "run_id": d["run_id"], "slot_id": slot_id,
                          "reason": "not_usable_no_ontario_quote" if not d["usable"]
                          else "superseded_by_earlier_usable_capture"})
    return chosen, sorted(notes, key=lambda n: (n["slot_id"], n["file"]))


def _local_date(c: dict):
    return on.parse_utc(c["slot"]["intended_utc"]).astimezone(on.TORONTO).date()


def week_pairs(chosen: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """(season, week) -> its Wednesday and Sunday representative captures.

    The Wednesday must be the one in the same NFL week as the Sunday: exactly
    four days earlier (Toronto dates). A Wednesday capture from an earlier
    week - e.g. the one eight days before a season opener, which also
    contains week-1 games - is never used in its place. Returns the pairs
    and notes on Wednesday captures left out for that reason."""
    weeks: dict[tuple[int, int], dict[str, list[dict]]] = {}
    for c in chosen.values():
        seen = {(g["season"], g["week"]) for g in c["games"]} | \
               {(g["season"], g["week"]) for g in c["excluded_games"]}
        for sw in seen:
            weeks.setdefault(sw, {}).setdefault(c["slot"]["name"], []).append(c)
    pairs, notes = [], []
    for (season, week), slots in sorted(weeks.items()):
        suns = sorted(slots.get(SUNDAY, []), key=lambda c: c["slot"]["intended_utc"])
        sun = suns[-1] if suns else None
        weds = sorted(slots.get(WEDNESDAY, []), key=lambda c: c["slot"]["intended_utc"])
        if sun is not None:
            same_week = _local_date(sun) - timedelta(days=4)
            for w in weds:
                if _local_date(w) != same_week:
                    notes.append({"file": w["_file"], "run_id": w["run_id"],
                                  "slot_id": w["slot"]["slot_id"],
                                  "reason": f"wednesday_not_in_same_nfl_week_as_sunday_"
                                            f"{sun['slot']['slot_id']}"})
            weds = [w for w in weds if _local_date(w) == same_week]
        wed = weds[-1] if weds else None
        pairs.append({"season": season, "week": week, "wednesday": wed, "sunday": sun})
    return pairs, notes


# --------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------

def _group_for(quote: dict) -> str | None:
    book = on.BOOKS_BY_KEY.get(quote["book_key"])
    if book is None or quote["source"] != "the_odds_api":
        return None
    if book.jurisdiction == "CA-ON" and book.role in ("ontario", "ontario_paid_tier"):
        return GROUP_ONTARIO
    if book is on.FANDUEL_US:
        return GROUP_US_REFERENCE
    return None


def _api_observations(capture: dict | None) -> tuple[dict, dict, dict]:
    """From one capture: {(group, game_id, book, side): obs}, {game_id: game},
    {game_id: excluded reason}."""
    obs, games, excluded = {}, {}, {}
    if capture is None:
        return obs, games, excluded
    for e in capture["excluded_games"]:
        excluded[e["game_id"]] = e["reason"]
    for g in capture["games"]:
        games[g["game_id"]] = g
        for q in g["quotes"]:
            group = _group_for(q)
            if group is None:
                continue
            for side in ("home", "away"):
                team = g[f"{side}_team"]
                obs[(group, g["game_id"], q["book_key"], side)] = {
                    "team": team, "handicap": q[f"{side}_point"], "price": q[f"{side}_price"],
                    "quote_status": q["status"], "book_title": q["book_title"],
                    "jurisdiction": q["jurisdiction"], "source": q["source"],
                    "slot_id": capture["slot"]["slot_id"], "slot_status": capture["slot"]["status"],
                    "observed_at": capture["captured_at"],
                    "provider_update": q["market_last_update"] or q["bookmaker_last_update"],
                    "provider_update_basis": "market_last_update" if q["market_last_update"]
                    else ("bookmaker_last_update" if q["bookmaker_last_update"] else None),
                    "age_minutes": q["age_minutes"], "run_id": capture["run_id"],
                    "file": capture["_file"], "kickoff_utc": g["kickoff_utc"],
                }
    return obs, games, excluded


def assign_manual_to_slots(manual: list[dict]) -> tuple[dict, list[dict]]:
    """Manual quotes -> {(slot_name, slot_id, game_id, team): quote}, plus notes.

    A manual observation belongs to a slot only if its observed_at falls inside
    that slot's Phase 1 window (on.resolve_slot). Within a slot, the earliest
    observation (then quote_id) is used; later ones are noted, never merged."""
    assigned, notes = {}, []
    for m in sorted(manual, key=lambda m: (m["observed_at"], m["quote_id"])):
        slot = on.resolve_slot(on.parse_utc(m["observed_at"]))
        if slot is None:
            notes.append({"file": m["_file"], "quote_id": m["quote_id"], "game_id": m["game_id"],
                          "reason": "manual_observation_outside_slot_windows"})
            continue
        key = (slot["name"], slot["slot_id"], m["game_id"], m["team"])
        if key in assigned:
            notes.append({"file": m["_file"], "quote_id": m["quote_id"], "game_id": m["game_id"],
                          "reason": "superseded_by_earlier_manual_observation_in_slot"})
            continue
        assigned[key] = {**m, "_slot": slot}
    return assigned, sorted(notes, key=lambda n: (n["game_id"], n["file"]))


def _manual_observations(assigned: dict, slot_name: str, season: int, week: int) -> dict:
    """Manual quotes for one slot of one week: {(group, game_id, book, side): obs}.

    The quoted team's side always exists; the opponent's side only when its
    price was entered (its handicap is the mirror)."""
    obs = {}
    for (name, slot_id, game_id, team), m in sorted(assigned.items()):
        if name != slot_name or m["season"] != season or m["week"] != week:
            continue
        sides = [(m["team_side"], m["team"], m["handicap"], m["price"])]
        if m["opponent_price"] is not None:
            other = "away" if m["team_side"] == "home" else "home"
            sides.append((other, m[f"{other}_team"], m["opponent_handicap"], m["opponent_price"]))
        for side, t, handicap, price in sides:
            obs[(GROUP_MANUAL, game_id, m["book_key"], side)] = {
                "team": t, "handicap": handicap, "price": price, "quote_status": "quoted",
                "book_title": m["book_title"], "jurisdiction": m["jurisdiction"],
                "source": m["source"], "slot_id": slot_id, "slot_status": "manual",
                "observed_at": m["observed_at"], "provider_update": None,
                "provider_update_basis": "manual_observed_at", "age_minutes": None,
                "run_id": m["quote_id"], "file": m["_file"], "kickoff_utc": m["kickoff_utc"],
            }
    return obs


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

def _row(group, season, week, game, book_key, side, wed, sun, missing_reason=None) -> dict:
    team = game[f"{side}_team"]
    other = game["away_team" if side == "home" else "home_team"]
    any_obs = wed or sun
    row = {
        "group": group, "season": season, "week": week, "game_id": game["game_id"],
        "matchup": f"{game['away_team']} @ {game['home_team']}", "comparison_team": team,
        "opponent": other, "side": side, "book_key": book_key,
        "book_title": any_obs["book_title"] if any_obs else None,
        "jurisdiction": any_obs["jurisdiction"] if any_obs else None,
        "source": any_obs["source"] if any_obs else None,
    }
    for prefix, o in (("wed", wed), ("sun", sun)):
        row.update({
            f"{prefix}_handicap": o["handicap"] if o else None,
            f"{prefix}_price": o["price"] if o else None,
            f"{prefix}_break_even": break_even(o["price"]) if o and o["price"] is not None else None,
            f"{prefix}_quote_status": o["quote_status"] if o else None,
            f"{prefix}_slot_id": o["slot_id"] if o else None,
            f"{prefix}_slot_status": o["slot_status"] if o else None,
            f"{prefix}_observed_at": o["observed_at"] if o else None,
            f"{prefix}_provider_update": o["provider_update"] if o else None,
            f"{prefix}_provider_update_basis": o["provider_update_basis"] if o else None,
            f"{prefix}_age_minutes": o["age_minutes"] if o else None,
            f"{prefix}_run_id": o["run_id"] if o else None,
            f"{prefix}_file": o["file"] if o else None,
        })
    kickoffs = sorted({o["kickoff_utc"] for o in (wed, sun) if o})
    row["kickoff_utc"] = kickoffs[0] if kickoffs else game.get("kickoff_utc")
    row["kickoff_changed"] = len(kickoffs) > 1
    row.update(spread_change=None, break_even_change=None, key_3=None, key_7=None,
               status=None, outcome=None, reasons=[])

    reasons = []
    if missing_reason:
        reasons.append(missing_reason)
    for prefix, o in (("wednesday", wed), ("sunday", sun)):
        if o is None:
            continue
        if o["quote_status"] not in COMPARABLE:
            reasons.append(f"{prefix}_quote_{o['quote_status']}")
        elif o["handicap"] is None or o["price"] is None:
            reasons.append(f"{prefix}_quote_missing_values")
        # Both observations strictly before kickoff (re-checked; Phase 1 also enforces it).
        for k in kickoffs:
            if on.parse_utc(o["observed_at"]) >= on.parse_utc(k):
                reasons.append(f"{prefix}_observed_at_or_after_kickoff")
        if o["slot_status"] not in ("on_time", "late", "manual"):
            reasons.append(f"{prefix}_out_of_window")
    if wed and sun and wed["team"] != sun["team"]:
        reasons.append("team_mismatch")
    if wed and sun and wed["jurisdiction"] != sun["jurisdiction"]:
        reasons.append("jurisdiction_mismatch")
    row["reasons"] = sorted(set(reasons))

    if row["reasons"]:
        row["status"] = _status_for(row["reasons"])
        return row
    w_h, s_h = float(wed["handicap"]), float(sun["handicap"])
    row["spread_change"] = s_h - w_h + 0.0
    row["break_even_change"] = round(row["sun_break_even"] - row["wed_break_even"], 6) + 0.0
    row["key_3"] = key_number_move(w_h, s_h, 3)
    row["key_7"] = key_number_move(w_h, s_h, 7)
    row["outcome"] = classify(w_h, wed["price"], s_h, sun["price"])
    row["status"] = "compared"
    return row


_MISSING_QUOTE = ("_quote_absent", "_quote_no_spreads_market", "_quote_incomplete",
                  "_quote_event_not_matched", "_quote_missing_values")


def _status_for(reasons: list[str]) -> str:
    """unmatched: a slot has no observation of the game at all (e.g. a
    Thursday game at the Sunday slot); missing_quote: the game was observed
    but this book had no usable quote; quality_excluded: a quote exists but is
    stale, invalid, out of window or otherwise ineligible."""
    if any(r.startswith("no_") for r in reasons):
        return "unmatched"
    if any(r.endswith(_MISSING_QUOTE) for r in reasons):
        return "missing_quote"
    return "quality_excluded"


def _no_obs_reason(prefix: str, capture: dict | None, excluded: dict, game_id: str,
                   slot_present: bool) -> str:
    if not slot_present:
        return f"no_{prefix}_capture_for_week"
    if game_id in excluded:
        return f"no_{prefix}_observation_{excluded[game_id]}"     # e.g. kicked_off
    return f"no_{prefix}_observation_game_not_in_capture"


def _has_values(o: dict | None) -> bool:
    return o is not None and o["handicap"] is not None and o["price"] is not None


def compare_week(pair: dict, groups: tuple[str, ...], books: set[str] | None,
                 manual_assigned: dict) -> tuple[list[dict], list[dict]]:
    """Rows for one week, plus the game/book pairs a requested book quoted in
    neither slot (counted, but not listed as rows)."""
    season, week = pair["season"], pair["week"]
    wed_obs, wed_games, wed_excl = _api_observations(pair["wednesday"])
    sun_obs, sun_games, sun_excl = _api_observations(pair["sunday"])
    if GROUP_MANUAL in groups:
        wed_obs.update(_manual_observations(manual_assigned, WEDNESDAY, season, week))
        sun_obs.update(_manual_observations(manual_assigned, SUNDAY, season, week))
    games = {**sun_games, **wed_games}
    rows, not_quoted = [], set()
    keys = sorted({k for k in set(wed_obs) | set(sun_obs) if k[0] in groups
                   and (books is None or k[2] in books)})
    for group, game_id, book_key, side in keys:
        if not (_has_values(wed_obs.get((group, game_id, book_key, side)))
                or _has_values(sun_obs.get((group, game_id, book_key, side)))):
            not_quoted.add((group, season, week, game_id, book_key))
            continue
        # A manual quote can be for a game in neither capture.
        game = games.get(game_id) or _manual_game(manual_assigned, game_id)
        wed, sun = wed_obs.get((group, game_id, book_key, side)), sun_obs.get((group, game_id, book_key, side))
        missing = None
        if wed is None:
            missing = _no_obs_reason("wednesday", pair["wednesday"], wed_excl, game_id,
                                     pair["wednesday"] is not None or group == GROUP_MANUAL)
            if group == GROUP_MANUAL:
                missing = "no_wednesday_manual_observation_in_slot_window"
        elif sun is None:
            missing = _no_obs_reason("sunday", pair["sunday"], sun_excl, game_id,
                                     pair["sunday"] is not None or group == GROUP_MANUAL)
            if group == GROUP_MANUAL:
                missing = "no_sunday_manual_observation_in_slot_window"
        rows.append(_row(group, season, week, game, book_key, side, wed, sun, missing))
    return rows, [{"group": g, "season": s, "week": w, "game_id": gid, "book_key": b}
                  for g, s, w, gid, b in sorted(not_quoted)]


def _manual_game(assigned: dict, game_id: str) -> dict | None:
    for (_, _, gid, _), m in sorted(assigned.items()):
        if gid == game_id:
            return {"game_id": gid, "home_team": m["home_team"], "away_team": m["away_team"],
                    "kickoff_utc": m["kickoff_utc"]}
    return None


# --------------------------------------------------------------------------
# Rollups
# --------------------------------------------------------------------------

def _counts(rows: list[dict], not_quoted: list[dict]) -> dict:
    compared = [r for r in rows if r["status"] == "compared"]
    pairs: dict[tuple, list[dict]] = {}
    for r in rows:
        pairs.setdefault((r["season"], r["week"], r["game_id"], r["book_key"]), []).append(r)
    pair_status = {}
    for k, rs in pairs.items():
        st = {r["status"] for r in rs}
        pair_status[k] = ("compared" if st == {"compared"} else
                          "partially_compared" if "compared" in st else
                          "unmatched" if "unmatched" in st else
                          "missing_quote" if "missing_quote" in st else
                          "quality_excluded")
    games = {}
    for (s, w, g, _), st in pair_status.items():
        games.setdefault((s, w, g), set()).add(st)
    outcome_counts = {o: sum(r["outcome"] == o for r in compared) for o in OUTCOMES}
    reasons: dict[str, int] = {}
    for r in rows:
        for reason in r["reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "sides": {"total": len(rows), "compared": len(compared),
                  "unmatched": sum(r["status"] == "unmatched" for r in rows),
                  "missing_quote": sum(r["status"] == "missing_quote" for r in rows),
                  "quality_excluded": sum(r["status"] == "quality_excluded" for r in rows),
                  "outcomes": outcome_counts,
                  "outcome_denominator": "compared sides (two per compared game/book pair)"},
        "game_book_pairs": {"total": len(pair_status),
                            **{s: sum(v == s for v in pair_status.values())
                               for s in ("compared", "partially_compared", "unmatched",
                                         "missing_quote", "quality_excluded")}},
        "games": {"total": len(games),
                  "with_any_compared_pair": sum("compared" in v or "partially_compared" in v
                                                for v in games.values()),
                  "with_no_compared_pair": sum(not ({"compared", "partially_compared"} & v)
                                               for v in games.values())},
        "reasons_by_side": dict(sorted(reasons.items())),
        # Requested books with no quote values at either slot (absent both
        # times): coverage gaps, not comparisons, so they have no rows.
        "game_book_pairs_not_quoted_in_either_slot": len(not_quoted),
    }


def rollups(rows: list[dict], not_quoted: list[dict], groups: tuple[str, ...]) -> dict:
    out = {}
    for group in groups:
        g_rows = [r for r in rows if r["group"] == group]
        g_nq = [n for n in not_quoted if n["group"] == group]
        per_book = {}
        for book in sorted({r["book_key"] for r in g_rows} | {n["book_key"] for n in g_nq}):
            per_book[book] = _counts([r for r in g_rows if r["book_key"] == book],
                                     [n for n in g_nq if n["book_key"] == book])
        out[group] = {"overall": _counts(g_rows, g_nq), "per_book": per_book,
                      "note": "Per-book counts cover different games and sample sizes; no "
                              "overall best book is declared."}
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def build_report(capture_dir: Path = on.CAPTURE_DIR, manual_dir: Path = on.MANUAL_DIR, *,
                 season: int | None = None, week: int | None = None,
                 books: list[str] | None = None, include_manual: bool = False,
                 include_us_reference: bool = False) -> dict:
    captures, manual, inputs = load_inputs(capture_dir, manual_dir if include_manual else None)
    groups = tuple(g for g in GROUPS if g == GROUP_ONTARIO
                   or (g == GROUP_US_REFERENCE and include_us_reference)
                   or (g == GROUP_MANUAL and include_manual))
    chosen, capture_notes = select_slot_captures(captures)
    manual_assigned, manual_notes = assign_manual_to_slots(manual)
    pairs, pairing_notes = week_pairs(chosen)
    capture_notes = sorted(capture_notes + pairing_notes, key=lambda n: (n["slot_id"], n["file"]))
    if include_manual:   # weeks with manual observations but no captures
        known = {(p["season"], p["week"]) for p in pairs}
        for (_, _, _, _), m in sorted(manual_assigned.items()):
            if (m["season"], m["week"]) not in known:
                known.add((m["season"], m["week"]))
                pairs.append({"season": m["season"], "week": m["week"],
                              "wednesday": None, "sunday": None})
        pairs.sort(key=lambda p: (p["season"], p["week"]))
    if season is not None:
        pairs = [p for p in pairs if p["season"] == season]
    if week is not None:
        pairs = [p for p in pairs if p["week"] == week]
    book_filter = set(books) if books else None

    rows, weeks, not_quoted = [], [], []
    for p in pairs:
        week_rows, week_nq = compare_week(p, groups, book_filter, manual_assigned)
        rows.extend(week_rows)
        not_quoted.extend(week_nq)
        weeks.append({"season": p["season"], "week": p["week"],
                      "wednesday": _slot_ref(p["wednesday"]), "sunday": _slot_ref(p["sunday"])})
    rows.sort(key=lambda r: (r["season"], r["week"], GROUPS.index(r["group"]), r["game_id"],
                             r["book_key"], r["side"]))
    status = "no_observations_yet" if not captures and not manual else (
        "no_comparable_observations" if not any(r["status"] == "compared" for r in rows)
        else "ok")
    report = {
        "schema_version": REPORT_SCHEMA_VERSION, "kind": REPORT_KIND, "status": status,
        "filters": {"season": season, "week": week, "books": sorted(book_filter) if book_filter else None,
                    "include_manual": include_manual, "include_us_reference": include_us_reference},
        "groups": list(groups),
        "rules": {
            "comparison": "same season, week, game_id, sportsbook, jurisdiction, team and "
                          "spread market; Wednesday-noon vs Sunday-morning slot of that week",
            "slot_selection": "earliest usable capture per scheduled slot (ties by run_id); "
                              "empty/US-only, ad_hoc and later duplicate captures are not used",
            "handicap": "each team's own handicap; larger is better for that team; "
                        "spread_change = Sunday - Wednesday",
            "price": "break-even = win rate needed at the American price, pushes excluded; "
                     "lower is a better payout",
            "outcome": "sunday_dominates / wednesday_dominates = at least as good on both "
                       "handicap and payout and strictly better on one; trade_off = better "
                       "on one, worse on the other; unchanged = identical",
            "eligibility": "only 'quoted' observations from in-window scheduled slots, both "
                           "before kickoff; stale, missing, invalid or out-of-window quotes "
                           "never produce an outcome",
            "manual": "manual FanDuel Ontario observations only when requested, assigned to a "
                      "slot only if observed inside its window, compared only with manual",
            "not_claimed": "Sunday 09:00 is not a closing line; no best betting time, edge "
                           "or ROI is claimed",
        },
        "inputs": sorted(inputs, key=lambda i: i["file"]),
        "weeks": weeks,
        "capture_notes": capture_notes,
        "manual_notes": manual_notes if include_manual else [],
        "rollups": rollups(rows, not_quoted, groups),
        "rows": rows,
    }
    report["report_sha256"] = _report_hash(report)      # of everything above
    return report


def _report_hash(report: dict) -> str:
    import hashlib
    return hashlib.sha256(on.ps._canonical(report).encode()).hexdigest()


def _slot_ref(c: dict | None) -> dict | None:
    if c is None:
        return None
    return {"slot_id": c["slot"]["slot_id"], "slot_status": c["slot"]["status"],
            "intended_utc": c["slot"]["intended_utc"], "captured_at": c["captured_at"],
            "run_id": c["run_id"], "file": c["_file"]}


def to_csv(report: dict) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for r in report["rows"]:
        w.writerow({**r, "reasons": ";".join(r["reasons"])})
    return buf.getvalue()


def _norm(path: Path) -> str:
    """Absolute, symlink-resolved, case-normalized path for comparisons."""
    return os.path.normcase(os.path.realpath(path))


def _inside(path: Path, root: Path) -> bool:
    p, r = _norm(path), _norm(root)
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def _atomic_write(path: Path, data: bytes) -> None:
    """Write via a temp file and os.replace, which swaps the directory entry:
    an existing symlink or hard link at `path` is replaced, never written
    through, so whatever it pointed to is untouched."""
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_report(report: dict, output_dir: Path,
                 protected: tuple[Path, ...] = ()) -> tuple[Path, Path]:
    """Write the JSON and CSV reports. Refuses any output directory that is,
    or resolves (through symlinks, junctions, `..` or letter case) to, a
    location inside the source data: data_files/ and any capture/manual
    directories passed in `protected`."""
    output_dir = Path(output_dir)
    roots = (on.DATA_DIR, on.STORE_DIR) + tuple(Path(p) for p in protected)
    for root in roots:
        if _inside(output_dir, root) or _inside(root, output_dir):
            raise ReportError(f"refusing to write reports to {output_dir}: it overlaps "
                              f"source data at {root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = output_dir / JSON_NAME, output_dir / CSV_NAME
    for target in (json_path, csv_path):        # re-check after mkdir (links created meanwhile)
        if any(_inside(target, root) for root in roots):
            raise ReportError(f"refusing to write {target}: it resolves into source data")
    _atomic_write(json_path, (json.dumps(report, indent=1, sort_keys=True, allow_nan=False)
                              + "\n").encode("utf-8"))
    _atomic_write(csv_path, to_csv(report).encode("utf-8"))
    return json_path, csv_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Read-only Wednesday-vs-Sunday Ontario spread report")
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--book", action="append", dest="books",
                    help="limit to a bookmaker key (repeatable), e.g. --book betmgm_ca_on")
    ap.add_argument("--include-manual", action="store_true",
                    help="also compare manual FanDuel Ontario observations (separate group)")
    ap.add_argument("--include-us-reference", action="store_true",
                    help="also compare the US FanDuel reference feed (separate group)")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--capture-dir", type=Path, default=on.CAPTURE_DIR)
    ap.add_argument("--manual-dir", type=Path, default=on.MANUAL_DIR)
    args = ap.parse_args(argv)
    try:
        report = build_report(args.capture_dir, args.manual_dir, season=args.season,
                              week=args.week, books=args.books,
                              include_manual=args.include_manual,
                              include_us_reference=args.include_us_reference)
        json_path, csv_path = write_report(report, args.output_dir,
                                           protected=(args.capture_dir, args.manual_dir))
    except (ReportError, ValueError, KeyError, OSError) as exc:
        print(f"[ontario_spread_report] failed: {exc}")
        return 1
    o = report["rollups"].get(GROUP_ONTARIO, {}).get("overall", {})
    print(f"[ontario_spread_report] {report['status']}: "
          f"{o.get('sides', {}).get('compared', 0)} compared sides, "
          f"{o.get('game_book_pairs', {}).get('compared', 0)} compared game/book pairs "
          f"-> {json_path}, {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
