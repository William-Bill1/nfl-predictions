"""Roll up `betting_recommendations_log.csv` into a season-to-date spread scorecard.

Writes `data_files/spread_performance.json` (overall + per-week + per-tier
record / profit / ROI) and prints a short summary. Run by
`weekly-model-performance.yml` after `betting_log.py` has graded finished bets;
also fine to run by hand.

Only `bet_type == 'spread'` rows count (moneyline / totals are disabled). A
`-110` bet: win -> +90.91, loss -> -100, push -> 0. ROI = profit / (settled * 100).
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

DATA_DIR = "data_files"
LOG_PATH = os.path.join(DATA_DIR, "betting_recommendations_log.csv")
OUT_PATH = os.path.join(DATA_DIR, "spread_performance.json")

SETTLED = ("win", "loss", "push")


def _bucket(rows: pd.DataFrame) -> dict:
    """Record / profit / ROI for a slice of settled+pending spread rows."""
    settled = rows[rows["bet_result"].isin(SETTLED)]
    graded = rows[rows["bet_result"].isin(("win", "loss"))]  # ROI denominator excludes pushes
    wins = int((settled["bet_result"] == "win").sum())
    losses = int((settled["bet_result"] == "loss").sum())
    pushes = int((settled["bet_result"] == "push").sum())
    profit = float(pd.to_numeric(settled.get("bet_profit"), errors="coerce").fillna(0).sum())
    n_graded = len(graded)
    return {
        "pending": int((rows["bet_result"] == "pending").sum()),
        "settled": wins + losses + pushes,
        "win": wins,
        "loss": losses,
        "push": pushes,
        "win_rate": round(wins / n_graded, 4) if n_graded else None,
        "profit": round(profit, 2),
        "roi_pct": round(profit / (n_graded * 100) * 100, 2) if n_graded else None,
    }


def build_report(log_path: str = LOG_PATH) -> dict:
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": os.path.basename(log_path),
        "note": ("Spread bets only, -110 (win +90.91 / loss -100 / push 0). "
                 "ROI = profit / (settled non-push bets * 100)."),
        "overall": _bucket(pd.DataFrame(columns=["bet_result", "bet_profit"])),
        "by_week": [],
        "by_tier": {},
    }
    if not (os.path.exists(log_path) and os.path.getsize(log_path) > 0):
        return report

    df = pd.read_csv(log_path)
    if "bet_type" in df.columns:
        df = df[df["bet_type"].astype(str) == "spread"]
    if df.empty:
        return report

    report["overall"] = _bucket(df)

    if "week" in df.columns:
        for wk, grp in df.groupby("week", dropna=True):
            try:
                wk_label = int(wk)
            except (TypeError, ValueError):
                wk_label = str(wk)
            report["by_week"].append({"week": wk_label, **_bucket(grp)})
        report["by_week"].sort(key=lambda r: (isinstance(r["week"], str), r["week"]))

    if "confidence_tier" in df.columns:
        for tier in ("Elite", "Strong", "Good", "Lean"):
            grp = df[df["confidence_tier"].astype(str) == tier]
            if len(grp):
                report["by_tier"][tier] = _bucket(grp)

    return report


def main() -> None:
    report = build_report()
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    o = report["overall"]
    print(f"[spread_report] wrote {OUT_PATH}")
    if o["settled"]:
        wr = "n/a" if o["win_rate"] is None else f"{o['win_rate']:.1%}"
        roi = "n/a" if o["roi_pct"] is None else f"{o['roi_pct']:+.1f}%"
        push_str = f"-{o['push']}P" if o["push"] else ""
        print(f"[spread_report] season to date: {o['win']}-{o['loss']}{push_str}  "
              f"{wr} win, {roi} ROI, ${o['profit']:+.2f}  ({o['pending']} pending)")
        for wk in report["by_week"]:
            if wk["settled"]:
                wk_roi = "n/a" if wk["roi_pct"] is None else f"{wk['roi_pct']:+.1f}%"
                print(f"   week {wk['week']:>2}: {wk['win']}-{wk['loss']}  {wk_roi}")
    else:
        print(f"[spread_report] no settled spread bets yet ({o['pending']} pending)")


if __name__ == "__main__":
    main()
