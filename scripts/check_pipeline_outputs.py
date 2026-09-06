"""Post-run sanity checks for `python nfl-gather-data.py`.

Used by the `pipeline-smoke` CI job. Exits non-zero (with a message) if any
expected output is missing or obviously wrong. Not a correctness proof - just
enough to catch "the pipeline crashed / stopped writing a column / started
emitting nonsense probabilities" before it lands on main.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

DATA = Path("data_files")
PRED = DATA / "nfl_games_historical_with_predictions.csv"
METRICS = DATA / "model_metrics.json"
FEATS = DATA / "best_features_spread.txt"
SRC = DATA / "nfl_games_historical.csv"

REQUIRED_PRED_COLS = {
    "prob_underdogCovered", "pred_spreadCovered_optimal", "spreadPush",
    "ev_spread", "edge_underdog_spread", "gameday", "season", "week",
}
REQUIRED_METRIC_KEYS = {
    "Spread Accuracy", "Spread MAE", "Optimal Spread Threshold",
    "Spread_EV_Analysis", "Spread_OOS_Test",
}

errors: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        errors.append(msg)


# --- predictions csv -------------------------------------------------------
check(PRED.exists(), f"missing {PRED}")
if PRED.exists():
    df = pd.read_csv(PRED, sep="\t")
    missing = REQUIRED_PRED_COLS - set(df.columns)
    check(not missing, f"{PRED.name} missing columns: {sorted(missing)}")

    if SRC.exists():
        n_src = len(pd.read_csv(SRC, sep="\t"))
        check(len(df) == n_src,
              f"{PRED.name} has {len(df)} rows, expected {n_src} (same as source)")

    if "prob_underdogCovered" in df.columns:
        p = df["prob_underdogCovered"].dropna()
        check(len(p) > 0, "prob_underdogCovered is all-NaN")
        check(p.between(0.0, 1.0).all(),
              f"prob_underdogCovered out of [0,1]: min={p.min()}, max={p.max()}")
        check(p.nunique() > 10,
              f"prob_underdogCovered nearly constant ({p.nunique()} unique values)")

    if "pred_spreadCovered_optimal" in df.columns:
        sig = int((df["pred_spreadCovered_optimal"] == 1).sum())
        check(sig > 0, "no spread bet signals at all (pred_spreadCovered_optimal)")
        check(sig < len(df), "every game is a spread signal (threshold broken?)")

# --- metrics json --------------------------------------------------------
check(METRICS.exists(), f"missing {METRICS}")
if METRICS.exists():
    m = json.loads(METRICS.read_text())
    missing = REQUIRED_METRIC_KEYS - set(m)
    check(not missing, f"{METRICS.name} missing keys: {sorted(missing)}")

    acc = m.get("Spread Accuracy")
    check(isinstance(acc, (int, float)) and 0.0 <= acc <= 1.0,
          f"Spread Accuracy not a probability: {acc!r}")

    thr = m.get("Optimal Spread Threshold")
    check(isinstance(thr, (int, float)) and 0.4 <= thr <= 0.95,
          f"Optimal Spread Threshold implausible: {thr!r}")

    oos = m.get("Spread_OOS_Test") or {}
    check(oos.get("slice") == "test", f"Spread_OOS_Test.slice != 'test': {oos!r}")
    ev = m.get("Spread_EV_Analysis") or {}
    check(ev.get("slice") == "validation",
          f"Spread_EV_Analysis.slice != 'validation': {ev!r}")

# --- feature file ------------------------------------------------------
check(FEATS.exists() and FEATS.stat().st_size > 0, f"missing/empty {FEATS}")

if errors:
    print("PIPELINE SMOKE FAILED:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)

print("pipeline smoke OK:", PRED.name, "+", METRICS.name, "look sane")
