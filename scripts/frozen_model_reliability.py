"""Frozen-model reliability report (read-only, evidence only).

Assesses the accumulated FROZEN pregame spread predictions against two
baselines - a constant 50% and the archived Ontario market - with fixed
calibration bins and paired, week-clustered uncertainty. It evaluates evidence
only: no retraining, no calibration fitting, no threshold or signal changes,
no betting recommendations.

Everything about WHICH games count and HOW they settle comes from
scripts/frozen_spread_report.py (selection, timing cross-check, completion,
settlement, coverage and the exact-line Ontario market match). This module
adds only the reliability view on top of that report's per-game rows.

Protocol (version PROTOCOL_VERSION; see docs/FROZEN_MODEL_RELIABILITY.md),
fixed for subsequent evaluation. It was not registered before the games
already evaluated were played, so results on those games are exploratory:

* One observation per game: the latest eligible pregame snapshot, chosen
  before its probability or signal is inspected; no fallback to an older one.
  The frozen underdog, frozen handicap and frozen probability decide the game.
* Pushes are excluded from probability metrics and counted separately.
* Samples, each reported for all eligible predictions and for the frozen
  bet-signal subset separately:
    overall - every evaluated game: model vs 50%;
    matched - evaluated games with a matched Ontario quote: model vs market
              vs 50% on exactly the same games.
* Calibration bins BIN_EDGES, left-closed/right-open, the last bin closed at
  1.0; 95% Wilson intervals (binomial reference intervals that ignore
  dependence between games); bins with fewer than SPARSE_BELOW games are
  flagged descriptive only.
* Paired per-game score differences (model minus baseline; negative = model
  scored better), with a paired season/week cluster bootstrap
  (BOOTSTRAP_REPLICATES replicates, seed BOOTSTRAP_SEED, percentile 95%
  interval). Fewer than MIN_CLUSTERS weeks -> interval unavailable;
  reaching MIN_CLUSTERS only permits an exploratory interval.
* --earliest-sensitivity adds a separate earliest-snapshot view. It is never
  pooled with the latest view and never ranked against it.

Outputs go to a git-ignored directory (default reports/frozen_reliability/),
refused anywhere overlapping source data. Corrupt inputs fail the run. No
source writes, no network, no bet-journal access.

    python scripts/frozen_model_reliability.py [--season 2026] [--as-of 2026-10-21]
        [--earliest-sensitivity] [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import frozen_spread_report as fsr  # noqa: E402
import ontario_spread_report as osr  # noqa: E402

REPORT_KIND = "frozen_model_reliability_report"
PROTOCOL_VERSION = 1
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "frozen_reliability"
JSON_NAME = "frozen_model_reliability.json"
GAMES_CSV = "frozen_model_reliability_games.csv"
CALIBRATION_CSV = "frozen_model_reliability_calibration.csv"
PLOT_NAME = "frozen_model_calibration.svg"

TRACKING_START = "2026-10-05"            # first pregame snapshot (2026-10-05T09:54:41Z)
BIN_EDGES = (0.0, 0.40, 0.50, 0.60, 0.70, 1.0)
SPARSE_BELOW = 10                        # display flag only, not a reliability threshold
WILSON_Z = 1.959963984540054             # two-sided 95% normal quantile
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261005
MIN_CLUSTERS = 5
LOG_LOSS_EPS = fsr.LOG_LOSS_EPS
ROUND = fsr.ROUND
EXCLUDED_STATUSES = (fsr.ps.NO_LINE, fsr.ps.PICKEM, fsr.ps.NO_PROBABILITY,
                     fsr.INVALID_OUTCOME, fsr.TIMING_UNVERIFIED)

ReportError = fsr.ReportError

PROTOCOL = {
    "version": PROTOCOL_VERSION,
    "status": ("fixed for subsequent evaluation; not prospectively registered before the games "
               "already evaluated were played, so results on those games are exploratory"),
    "selection": ("one observation per game: the latest pregame snapshot captured strictly "
                  "before the kickoff it recorded (frozen_spread_report / "
                  "pregame_snapshots.select_captures), chosen before its probability or signal "
                  "is inspected; it must also precede the current scheduled kickoff, otherwise "
                  "timing_unverified; no fallback to an older prediction"),
    "terms": "the frozen underdog, frozen handicap (+|spread_line|) and frozen probability",
    "target": "1 if the frozen underdog covered the frozen handicap, 0 if it did not",
    "completion_and_settlement": ("bet_journal.completion_evidence + betting_log.spread_result, "
                                  "as in frozen_spread_report"),
    "pushes": "excluded from probability metrics and counted separately",
    "subsets": "all eligible predictions and the frozen bet-signal subset, reported separately",
    "samples": {
        "overall": "every evaluated (completed, non-push) game: model vs constant 50%",
        "matched_market": ("evaluated games with a matched archived Ontario quote at the exact "
                           "frozen handicap: model vs Ontario market vs 50% on the same games"),
    },
    "market": ("frozen_spread_report.market_for: one archived, validated Ontario capture per "
               "game (latest strictly pregame), automated Ontario feeds only (US reference and "
               "manual quotes excluded first), fresh quotes at the exact frozen handicap with both "
               "prices; devigged two-way probability averaged over the matched books, so one "
               "game is one observation however many books quote it"),
    "calibration_bins": {"edges": list(BIN_EDGES),
                         "endpoints": "left-closed, right-open; the last bin [0.70, 1.0] is "
                                      "closed at 1.0; 0.40 belongs to [0.40, 0.50)",
                         "interval": ("95% Wilson score interval for the observed cover "
                                      "rate: a binomial reference interval that treats games "
                                      "as independent; it does not account for dependence "
                                      "between games and is distinct from the week-cluster "
                                      "bootstrap intervals"),
                         "sparse": f"bins with fewer than {SPARSE_BELOW} games are flagged "
                                   "descriptive only; the flag is not a reliability threshold",
                         "fixed": "fixed by protocol v1 for subsequent evaluation; not "
                                  "tuned to results"},
    "paired_differences": {
        "definition": "per game: model score minus baseline score; negative = model better",
        "scores": f"Brier (p - y)^2 and log loss -[y ln p + (1-y) ln(1-p)], p clipped to "
                  f"[{LOG_LOSS_EPS:g}, 1 - {LOG_LOSS_EPS:g}] for log loss only",
        "interval": (f"paired season/week cluster bootstrap: resample whole weeks with "
                     f"replacement, each game's model/baseline pair kept together, mean over the "
                     f"resampled games; {BOOTSTRAP_REPLICATES} replicates, numpy default_rng seed "
                     f"{BOOTSTRAP_SEED}, 2.5th/97.5th percentiles (linear interpolation)"),
        "minimum_clusters": (f"fewer than {MIN_CLUSTERS} weeks: interval unavailable; "
                             f"{MIN_CLUSTERS} or more only permits an exploratory interval and "
                             "does not establish that it is reliable"),
    },
    "sensitivity": ("--earliest-sensitivity: the earliest pregame snapshot per game, reported "
                    "separately; never pooled with the latest view and not a policy choice"),
    "no_post_hoc_selection": ("weeks, books, bins and snapshot policy are fixed by this "
                              "protocol; none is chosen after seeing outcomes"),
}

UNCERTAINTY_NOTE = (
    "Exploratory: with few weekly clusters a cluster bootstrap understates uncertainty and "
    "relies on weeks being exchangeable. An interval that includes zero is not evidence that "
    "the model and baseline are equivalent; one that excludes zero is not by itself evidence of "
    "a betting edge or of profitability. Reaching the minimum number of weeks only permits an "
    "exploratory interval; it does not make the interval reliable.")

LIMITATIONS = (
    f"Short sample: frozen tracking began {TRACKING_START}; only games with a frozen pregame "
    "snapshot are evaluated, a few weeks of one season.",
    "Changing thresholds, bins or signals on the strength of this sample would require a new, "
    "untouched evaluation period; this report must not be used to tune them.",
    "Games are not independent: games in a week share conditions, and the two sides of a game "
    "settle in opposite directions. One game is one observation here.",
    "Market comparison covers only games with an archived Ontario quote at the exact frozen "
    "handicap. The market capture and the model snapshot are taken at different times (compare "
    "market_captured_at with captured_at), so the comparison is not simultaneous.",
    "The market probability is the mean devigged two-way probability across matched Ontario "
    "books in one capture; it is a reference, not an achievable price.",
    "--as-of is an outcome cutoff over the current files, not a historical reconstruction.",
    "Protocol v1 is fixed for subsequent evaluation; it was not registered before the games "
    "already evaluated were played, so results on those games are exploratory.",
    "Calibration-bin Wilson intervals are binomial reference intervals that treat games as "
    "independent; they ignore dependence between games and are not the week-cluster "
    "bootstrap intervals.",
    "No profitability, edge or significance claim is made.",
)


# --------------------------------------------------------------------------
# Scores, bins, intervals
# --------------------------------------------------------------------------

def brier(p: float, y: int) -> float:
    return (p - y) ** 2


def log_loss(p: float, y: int) -> float:
    q = min(max(p, LOG_LOSS_EPS), 1 - LOG_LOSS_EPS)
    return -(y * math.log(q) + (1 - y) * math.log(1 - q))


def bin_index(p: float) -> int:
    """Index into BIN_EDGES: left-closed, right-open, last bin closed at 1.0."""
    if not isinstance(p, (int, float)) or isinstance(p, bool) or not 0.0 <= p <= 1.0 \
            or math.isnan(p):
        raise ReportError(f"probability {p!r} is not in [0, 1]")
    last = len(BIN_EDGES) - 2
    for i in range(last):
        if p < BIN_EDGES[i + 1]:
            return i
    return last


def bin_label(i: int) -> str:
    lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
    return f"[{lo:.2f}, {hi:.2f}{']' if i == len(BIN_EDGES) - 2 else ')'}"


def wilson(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float] | None:
    """95% Wilson score interval for k successes in n trials (None when n = 0)."""
    if n == 0:
        return None
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def calibration_table(pairs: list[tuple[float, int]]) -> list[dict]:
    rows = []
    for i in range(len(BIN_EDGES) - 1):
        inside = [(p, y) for p, y in pairs if bin_index(p) == i]
        n, k = len(inside), sum(y for _, y in inside)
        ci = wilson(k, n)
        rows.append({
            "bin": bin_label(i), "lower": BIN_EDGES[i], "upper": BIN_EDGES[i + 1],
            "upper_closed": i == len(BIN_EDGES) - 2, "n": n, "covers": k,
            "mean_predicted": round(sum(p for p, _ in inside) / n, ROUND) if n else None,
            "observed_cover_rate": round(k / n, ROUND) if n else None,
            "wilson_low": round(ci[0], ROUND) if ci else None,
            "wilson_high": round(ci[1], ROUND) if ci else None,
            "note": "empty: rate and interval unavailable" if n == 0 else
                    f"sparse (fewer than {SPARSE_BELOW} games): descriptive only"
                    if n < SPARSE_BELOW else "descriptive; not a reliability claim",
        })
    return rows


# --------------------------------------------------------------------------
# Paired differences and the week-cluster bootstrap
# --------------------------------------------------------------------------

def cluster_of(g: dict) -> str:
    return f"{g['season']}-W{g['week']:02d}"


def paired_differences(games: list[dict], model_key: str, baseline) -> list[dict]:
    """Per game: model minus baseline Brier and log loss. `baseline` is a
    probability key or a constant."""
    out = []
    for g in games:
        y, p = g["underdog_covered"], g[model_key]
        b = baseline if isinstance(baseline, float) else g[baseline]
        out.append({"game_id": g["game_id"], "cluster": cluster_of(g),
                    "d_brier": brier(p, y) - brier(b, y),
                    "d_log_loss": log_loss(p, y) - log_loss(b, y)})
    return out


def bootstrap_distribution(diffs: list[dict], key: str, replicates: int = BOOTSTRAP_REPLICATES,
                           seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Replicate means of the paired differences under week-cluster resampling.

    Each replicate draws as many clusters as there are, with replacement;
    a drawn cluster contributes all of its games' paired differences (the
    model/baseline pair is never split), and the replicate value is the mean
    over every drawn game."""
    clusters = sorted({d["cluster"] for d in diffs})
    sums = np.array([sum(d[key] for d in diffs if d["cluster"] == c) for c in clusters])
    counts = np.array([sum(1 for d in diffs if d["cluster"] == c) for c in clusters])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(clusters), size=(replicates, len(clusters)))
    return sums[idx].sum(axis=1) / counts[idx].sum(axis=1)


def cluster_bootstrap(diffs: list[dict], key: str, replicates: int = BOOTSTRAP_REPLICATES,
                      seed: int = BOOTSTRAP_SEED, min_clusters: int = MIN_CLUSTERS) -> dict:
    """Mean paired difference with a paired week-cluster bootstrap interval.

    Whole clusters are drawn with replacement (as many as there are); every
    game in a drawn cluster contributes its paired difference; the replicate
    statistic is the mean over all drawn games."""
    n = len(diffs)
    clusters = sorted({d["cluster"] for d in diffs})
    out = {"n_games": n, "n_clusters": len(clusters), "clusters": clusters,
           "mean_difference": round(sum(d[key] for d in diffs) / n, ROUND) if n else None,
           "interval_95": None, "interval_status": None, "replicates": replicates,
           "seed": seed, "method": "paired season/week cluster bootstrap, percentile"}
    if n == 0:
        out["interval_status"] = "unavailable: no games"
        return out
    if len(clusters) < min_clusters:
        out["interval_status"] = (f"unavailable: {len(clusters)} week cluster(s), fewer than the "
                                  f"{min_clusters} required; a bootstrap over so few weeks would "
                                  "give a falsely precise interval")
        return out
    stats = bootstrap_distribution(diffs, key, replicates, seed)
    lo, hi = np.quantile(stats, [0.025, 0.975])
    out["interval_95"] = [round(float(lo), ROUND), round(float(hi), ROUND)]
    out["interval_status"] = "available (exploratory)"
    return out


def comparison(games: list[dict], model_key: str, baseline, label: str) -> dict:
    diffs = paired_differences(games, model_key, baseline)
    return {"comparison": label,
            "sign": "model score minus baseline score; negative means the first scored better",
            "brier": cluster_bootstrap(diffs, "d_brier"),
            "log_loss": cluster_bootstrap(diffs, "d_log_loss"),
            "note": UNCERTAINTY_NOTE}


# --------------------------------------------------------------------------
# Samples
# --------------------------------------------------------------------------

def _pairs(games: list[dict], key: str = "prob_underdog_covers") -> list[tuple[float, int]]:
    return [(g[key], g["underdog_covered"]) for g in games]


def evaluated(games: list[dict]) -> list[dict]:
    return [g for g in games if g["status"] == fsr.EVALUATED
            and g["prob_underdog_covers"] is not None]


def matched(games: list[dict]) -> list[dict]:
    return [g for g in evaluated(games) if g["market_status"] == fsr.MATCHED
            and g["market_prob_underdog"] is not None]


def status_counts(games: list[dict]) -> dict:
    out = {"evaluated": sum(g["status"] == fsr.EVALUATED for g in games),
           "pushes": sum(g["status"] == fsr.PUSH for g in games),
           "pending": sum(g["status"] == fsr.PENDING for g in games),
           "excluded": {s: sum(g["status"] == s for g in games) for s in EXCLUDED_STATUSES}}
    out["excluded_total"] = sum(out["excluded"].values())
    return out


def overall_sample(games: list[dict]) -> dict:
    ev = evaluated(games)
    pairs = _pairs(ev)
    return {"definition": PROTOCOL["samples"]["overall"], "n_games": len(ev),
            "model": fsr.prob_metrics(pairs),
            "baseline_50": fsr.prob_metrics([(0.5, y) for _, y in pairs]),
            "model_minus_50": comparison(ev, "prob_underdog_covers", 0.5, "model vs 50%"),
            "calibration": calibration_table(pairs)}


def matched_sample(games: list[dict]) -> dict:
    mt = matched(games)
    pairs = _pairs(mt)
    ev = evaluated(games)
    unmatched = {}
    for g in ev:
        if g["market_status"] != fsr.MATCHED:
            unmatched.setdefault(g["market_status"], []).append(
                {"game_id": g["game_id"], "detail": g["market_status_detail"]})
    return {"definition": PROTOCOL["samples"]["matched_market"], "n_games": len(mt),
            "evaluated_without_market": sum(len(v) for v in unmatched.values()),
            "unavailable_market_by_reason": {k: {"n": len(v), "games": v}
                                             for k, v in sorted(unmatched.items())},
            "books_per_game": {"min": min((g["market_books"] for g in mt), default=None),
                               "max": max((g["market_books"] for g in mt), default=None),
                               "note": "books are averaged into one probability per game"},
            "model": fsr.prob_metrics(pairs),
            "market": fsr.prob_metrics(_pairs(mt, "market_prob_underdog")),
            "baseline_50": fsr.prob_metrics([(0.5, y) for _, y in pairs]),
            "model_minus_market": comparison(mt, "prob_underdog_covers", "market_prob_underdog",
                                             "model vs Ontario market"),
            "model_minus_50": comparison(mt, "prob_underdog_covers", 0.5, "model vs 50%"),
            "market_minus_50": comparison(mt, "market_prob_underdog", 0.5, "market vs 50%"),
            "calibration_model": calibration_table(pairs),
            "calibration_market": calibration_table(_pairs(mt, "market_prob_underdog"))}


def by_week(games: list[dict]) -> dict:
    weeks = {}
    for g in games:
        weeks.setdefault(cluster_of(g), []).append(g)
    out = {}
    for k in sorted(weeks):
        wk = weeks[k]
        mt = matched(wk)
        out[k] = {"counts": status_counts(wk),
                  "overall_model": fsr.prob_metrics(_pairs(evaluated(wk))),
                  "matched_n": len(mt),
                  "matched_model": fsr.prob_metrics(_pairs(mt)),
                  "matched_market": fsr.prob_metrics(_pairs(mt, "market_prob_underdog"))}
    return out


def view(games: list[dict]) -> dict:
    """The full reliability view of one set of selected games, for all
    eligible predictions and the frozen bet-signal subset separately."""
    signal = [g for g in games if g["bet_signal"]]
    return {sub: {"counts": status_counts(gs), "overall": overall_sample(gs),
                  "matched_market": matched_sample(gs), "by_week": by_week(gs)}
            for sub, gs in (("all_predictions", games), ("signal_only", signal))}


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

def fingerprint(snapshot_dir: Path, schedule_path: Path, capture_dir: Path) -> dict:
    def files(d):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(d).glob("*.json"))} if Path(d).is_dir() else {}
    try:
        out = {"schedule": {Path(schedule_path).name:
                            hashlib.sha256(Path(schedule_path).read_bytes()).hexdigest()},
               "pregame_snapshots": files(snapshot_dir),
               "ontario_captures": files(capture_dir)}
    except OSError as exc:
        raise ReportError(f"source files can't be read: {exc}") from None
    out["combined_sha256"] = hashlib.sha256(
        json.dumps(out, sort_keys=True).encode()).hexdigest()
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def build_report(snapshot_dir: Path = fsr.SNAPSHOT_DIR, schedule_path: Path = fsr.SCHEDULE_PATH,
                 capture_dir: Path = fsr.CAPTURE_DIR, *, season: int | None = None,
                 as_of: date | None = None, earliest_sensitivity: bool = False,
                 generated_at: datetime | None = None) -> dict:
    before = fingerprint(snapshot_dir, schedule_path, capture_dir)
    base = fsr.build_report(snapshot_dir, schedule_path, capture_dir, which="latest",
                            season=season, as_of=as_of)
    sens = fsr.build_report(snapshot_dir, schedule_path, capture_dir, which="earliest",
                            season=season, as_of=as_of) if earliest_sensitivity else None
    if fingerprint(snapshot_dir, schedule_path, capture_dir) != before:
        raise ReportError("source files changed while the report was running; run it again")
    games = base["games"]
    generated_at = generated_at or datetime.now(timezone.utc)
    report = {
        "kind": REPORT_KIND, "protocol_version": PROTOCOL_VERSION, "protocol": PROTOCOL,
        "generated_at_utc": generated_at.astimezone(timezone.utc)
        .isoformat(timespec="seconds").replace("+00:00", "Z"),
        "as_of_date": base["as_of_date"], "as_of_semantics": fsr.AS_OF_SEMANTICS,
        "scope": base["scope"], "tracking_start": TRACKING_START,
        "settings": {"bin_edges": list(BIN_EDGES), "sparse_below": SPARSE_BELOW,
                     "wilson_z": WILSON_Z, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                     "bootstrap_seed": BOOTSTRAP_SEED, "min_clusters": MIN_CLUSTERS,
                     "log_loss_eps": LOG_LOSS_EPS, "earliest_sensitivity": earliest_sensitivity},
        "sources": {"fingerprints": before, "built_on": "scripts/frozen_spread_report.py "
                    f"(schema {fsr.REPORT_SCHEMA_VERSION})",
                    "excluded_market_sources": base["inputs"]["excluded_market_sources"]
                    + ["manual Ontario quotes and the US reference feed (excluded per quote)"],
                    "bet_journal": "never read"},
        "coverage": {"selected_games": base["coverage"]["selected_games"],
                     "by_status": base["coverage"]["by_status"],
                     "completed_without_snapshot": base["coverage"]["completed_without_snapshot"],
                     "market_status_counts": base["market_comparison"]["market_status_counts"]},
        "latest": view(games),
        "earliest_sensitivity": None if sens is None else {
            "label": ("SENSITIVITY ONLY: earliest pregame snapshot per game; separate from the "
                      "latest view, never pooled with it, and not a candidate policy"),
            **view(sens["games"])},
        "uncertainty_note": UNCERTAINTY_NOTE,
        "limitations": list(LIMITATIONS),
        "games": [per_game_row(g) for g in games],
    }
    return report


def per_game_row(g: dict) -> dict:
    row = {k: g.get(k) for k in fsr.CSV_FIELDS}
    ev = g["status"] == fsr.EVALUATED and g["prob_underdog_covers"] is not None
    mt = ev and g["market_status"] == fsr.MATCHED and g["market_prob_underdog"] is not None
    y = g["underdog_covered"]
    row.update(
        week_cluster=cluster_of(g), in_overall_sample=ev, in_matched_sample=mt,
        brier_model=round(brier(g["prob_underdog_covers"], y), ROUND) if ev else None,
        brier_50=round(brier(0.5, y), ROUND) if ev else None,
        brier_market=round(brier(g["market_prob_underdog"], y), ROUND) if mt else None,
        log_loss_model=round(log_loss(g["prob_underdog_covers"], y), ROUND) if ev else None,
        log_loss_50=round(log_loss(0.5, y), ROUND) if ev else None,
        log_loss_market=round(log_loss(g["market_prob_underdog"], y), ROUND) if mt else None,
        calibration_bin=bin_label(bin_index(g["prob_underdog_covers"])) if ev else None)
    return row


GAME_FIELDS = fsr.CSV_FIELDS + ("week_cluster", "in_overall_sample", "in_matched_sample",
                                "brier_model", "brier_50", "brier_market", "log_loss_model",
                                "log_loss_50", "log_loss_market", "calibration_bin")
CAL_FIELDS = ("view", "subset", "sample", "probability", "bin", "lower", "upper", "upper_closed",
              "n", "covers", "mean_predicted", "observed_cover_rate", "wilson_low",
              "wilson_high", "note")
TEXT_COLUMNS = fsr.TEXT_COLUMNS


def read_games_csv(path) -> pd.DataFrame:
    """Read the per-game CSV with identifier/hash columns as text."""
    return pd.read_csv(path, dtype={c: str for c in TEXT_COLUMNS})


def _csv(fields, rows) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    return buf.getvalue()


def calibration_rows(report: dict) -> list[dict]:
    rows = []
    for view_name in ("latest", "earliest_sensitivity"):
        v = report[view_name]
        if not v:
            continue
        for subset in ("all_predictions", "signal_only"):
            s = v[subset]
            for sample, prob, table in (
                    ("overall", "model", s["overall"]["calibration"]),
                    ("matched_market", "model", s["matched_market"]["calibration_model"]),
                    ("matched_market", "market", s["matched_market"]["calibration_market"])):
                rows += [{"view": view_name, "subset": subset, "sample": sample,
                          "probability": prob, **r} for r in table]
    return rows


# --------------------------------------------------------------------------
# Plot (standalone SVG, no plotting dependency)
# --------------------------------------------------------------------------

# The plot's footer note, one string per line so nothing runs past the canvas.
PLOT_FOOTER = (f"Dashed: perfect calibration. Red: fewer than {SPARSE_BELOW} games (descriptive only).",
               "Wilson bars ignore dependence. Not a reliability claim.")


def title_lines(title: str, limit: int = 60) -> list[str]:
    """The title as at most two lines, split before its parenthesised details
    when it is too long for one line; the words are unchanged."""
    if len(title) <= limit or " (" not in title:
        return [title]
    head, tail = title.split(" (", 1)
    return [head, "(" + tail]


def calibration_svg(table: list[dict], title: str) -> str:
    W, H, L, R, T, B = 560, 614, 70, 20, 64, 110
    pw, ph = W - L - R, H - T - B

    def x(v):
        return L + v * pw

    def y(v):
        return T + (1 - v) * ph
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>']
    for i, line in enumerate(title_lines(title)):
        out.append(f'<text class="title" x="{W / 2}" y="{22 + 18 * i}" text-anchor="middle" '
                   f'font-size="14">{_esc(line)}</text>')
    out.append(f'<rect x="{L}" y="{T}" width="{pw}" height="{ph}" fill="none" stroke="#999"/>')
    for e in BIN_EDGES:
        out.append(f'<line x1="{x(e):.1f}" y1="{T}" x2="{x(e):.1f}" y2="{T + ph}" '
                   f'stroke="#eee"/>')
        out.append(f'<text x="{x(e):.1f}" y="{T + ph + 16}" text-anchor="middle">{e:.2f}</text>')
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<text x="{L - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{t:.2f}</text>')
    out.append(f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" stroke="#888" '
               f'stroke-dasharray="5,4"><title>perfect calibration</title></line>')
    for r in table:
        if not r["n"]:
            mid = (r["lower"] + r["upper"]) / 2
            out.append(f'<text x="{x(mid):.1f}" y="{T + ph - 6}" text-anchor="middle" '
                       f'fill="#999">n=0</text>')
            continue
        px, py = x(r["mean_predicted"]), y(r["observed_cover_rate"])
        lo, hi = y(r["wilson_low"]), y(r["wilson_high"])
        colour = "#c0392b" if r["n"] < SPARSE_BELOW else "#1f4e9c"
        out.append(f'<line x1="{px:.1f}" y1="{lo:.1f}" x2="{px:.1f}" y2="{hi:.1f}" '
                   f'stroke="{colour}" stroke-width="2"/>')
        out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{colour}">'
                   f'<title>{r["bin"]}: n={r["n"]}, mean predicted {r["mean_predicted"]}, '
                   f'observed {r["observed_cover_rate"]}, Wilson 95% '
                   f'[{r["wilson_low"]}, {r["wilson_high"]}]</title></circle>')
        # count label to the right of the point, or to its left near the right edge
        right = r["mean_predicted"] <= 0.9
        out.append(f'<text x="{px + 8 if right else px - 8:.1f}" y="{py - 8:.1f}" '
                   f'text-anchor="{"start" if right else "end"}" fill="{colour}">n={r["n"]}</text>')
    out += [f'<text x="{L + pw / 2}" y="{T + ph + 40}" text-anchor="middle">mean predicted '
            f'underdog cover probability (fixed bins)</text>',
            f'<text transform="translate(18,{T + ph / 2}) rotate(-90)" text-anchor="middle">'
            f'observed cover rate (95% Wilson)</text>']
    for i, line in enumerate(PLOT_FOOTER):
        out.append(f'<text class="footer" x="{W / 2}" y="{H - 30 + 16 * i}" text-anchor="middle" '
                   f'fill="#555" font-size="11">{_esc(line)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------
# Write
# --------------------------------------------------------------------------

def write_outputs(report: dict, output_dir: Path, protected: tuple = ()) -> list[Path]:
    """Write JSON, both CSVs and the SVG. Refuses any output directory that
    is, contains, or resolves (symlinks, junctions, '..', letter case) into
    source data - the frozen-spread report's own rule."""
    output_dir = Path(output_dir)
    roots = (fsr.DATA_DIR,) + tuple(Path(p) for p in protected)
    for root in roots:
        if osr._inside(output_dir, root) or osr._inside(root, output_dir):
            raise ReportError(f"refusing to write reports to {output_dir}: it overlaps source "
                              f"data at {root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    table = report["latest"]["all_predictions"]["overall"]["calibration"]
    n = report["latest"]["all_predictions"]["overall"]["n_games"]
    contents = {
        JSON_NAME: json.dumps(report, indent=1, sort_keys=True, allow_nan=False) + "\n",
        GAMES_CSV: _csv(GAME_FIELDS, report["games"]),
        CALIBRATION_CSV: _csv(CAL_FIELDS, calibration_rows(report)),
        PLOT_NAME: calibration_svg(table, f"Frozen model calibration: all eligible predictions "
                                          f"(n={n}, latest snapshot, as of {report['as_of_date']})"),
    }
    paths = [output_dir / name for name in contents]
    for target in paths:                       # re-check: links created meanwhile
        if any(osr._inside(target, root) for root in roots):
            raise ReportError(f"refusing to write {target}: it resolves into source data")
    for target in paths:
        osr._atomic_write(target, contents[target.name].encode("utf-8"))
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Read-only frozen-model reliability report")
    ap.add_argument("--season", type=int)
    ap.add_argument("--as-of", type=date.fromisoformat,
                    help="outcome cutoff (Toronto date, default today); current files, not a "
                         "historical reconstruction")
    ap.add_argument("--earliest-sensitivity", action="store_true",
                    help="add a separate earliest-snapshot sensitivity view (never pooled)")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--snapshot-dir", type=Path, default=fsr.SNAPSHOT_DIR)
    ap.add_argument("--schedule", type=Path, default=fsr.SCHEDULE_PATH)
    ap.add_argument("--capture-dir", type=Path, default=fsr.CAPTURE_DIR)
    args = ap.parse_args(argv)
    try:
        report = build_report(args.snapshot_dir, args.schedule, args.capture_dir,
                              season=args.season, as_of=args.as_of,
                              earliest_sensitivity=args.earliest_sensitivity)
        paths = write_outputs(report, args.output_dir,
                              fsr.protected_roots(args.snapshot_dir, args.schedule,
                                                  args.capture_dir))
    except (ReportError, OSError, ValueError) as exc:
        print(f"[frozen_model_reliability] failed: {exc}")
        return 1
    a = report["latest"]["all_predictions"]
    print(f"[frozen_model_reliability] protocol v{PROTOCOL_VERSION}, as of "
          f"{report['as_of_date']}: overall n={a['overall']['n_games']}, matched-market "
          f"n={a['matched_market']['n_games']}, pushes {a['counts']['pushes']}, pending "
          f"{a['counts']['pending']} -> {', '.join(str(p) for p in paths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
