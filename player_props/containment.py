"""Containment notice for player-prop outputs (Player Props, Parlay Builder,
Model Performance).

A read-only audit of the committed artifacts found two input problems:

* the player game logs behind every prop (data_files/player_*_stats.csv) were
  last rebuilt in January 2026 and are not refreshed by the nightly job, so
  2026 props use no 2026 games;
* the historical home/away values the saved prop models were trained on were
  assigned at random (np.random.choice in the January aggregator), matching
  the real home team only about half the time.

Until the pipeline is repaired and the models retrained, prop outputs stay
visible for inspection but are labelled as unvalidated. Nothing here changes a
probability, filter or stored figure; it only changes how they are described.
The game spread model is a separate pipeline and is not covered.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

STATS_FILES = ("player_passing_stats.csv", "player_rushing_stats.csv",
               "player_receiving_stats.csv")
HOLDOUT_PASSED = "passed 2025 holdout check"
HOLDOUT_NOT_PASSED = "did not pass holdout check"


def latest_stats_date(data_dir: Path | str) -> date | None:
    """Latest player game date across the prop stats files, or None if none can be read."""
    latest = None
    for name in STATS_FILES:
        try:
            dates = pd.to_datetime(pd.read_csv(Path(data_dir) / name, usecols=["game_date"])["game_date"],
                                   errors="coerce").dropna()
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        if not dates.empty:
            d = dates.max().date()
            latest = d if latest is None or d > latest else latest
    return latest


def _stats_line(stats_date: date | None, season: int | None) -> str:
    if stats_date is None:
        return ("- **Player statistics could not be checked.** The player game logs behind "
                "these props could not be read, so their freshness is unknown.")
    if season is not None and stats_date < date(season, 8, 1):
        return (f"- **Stale player statistics.** The player game logs behind these props end on "
                f"{stats_date.isoformat()}, before the {season} season began, so every "
                f"{season} prop (including the Last 3/5/10 averages) uses no {season} games.")
    return (f"- **Player statistics** end on {stats_date.isoformat()}. They are not refreshed "
            f"by the nightly job, so check that date against the games shown.")


def notice_markdown(stats_date: date | None, season: int | None = None) -> str:
    """The prominent warning shown at the top of every page that displays prop outputs."""
    return "\n".join([
        "**⚠️ Player prop outputs have unresolved data and training problems. Do not use "
        "them to guide wagers.**",
        "",
        _stats_line(stats_date, season),
        "- **Random home/away training values.** In the historical data the saved prop "
        "models were trained on, each game's home/away value was assigned at random (it "
        "matches the real home team only about half the time). The models learned from "
        "that noise, while predictions supply the real home/away value.",
        "- **Holdout labels describe a different scorer.** The stored props' holdout-check "
        "labels come from metrics measured on an XGBoost + LightGBM average, but the props "
        "were scored by XGBoost alone, so the labels don't describe the model that produced them.",
        "- **Some lines don't match their model.** About 1 in 12 stored props (rushing and "
        "receiving yards, and the elite TD tiers) show a line other than the one their model "
        "was trained on, so their probability is for a different line.",
        "",
        "Probabilities, probability bands, holdout-check labels, and past hit rates or ROI "
        "describe these model outputs only; they are **not validated betting confidence**. "
        "Everything stays visible for inspection, and no figure has been recalculated.",
        "",
        "**Scope:** player props only. The game spread model on the main page is a separate "
        "model and pipeline and is not covered by this notice. Checked against the committed "
        "model files in `player_props/models/`. They have no provenance bundle, so they are no "
        "longer used for new predictions.",
    ])


def band_label(probability: float) -> str:
    """Neutral probability band (formerly Elite/Strong/Good/Lean); display only."""
    if probability >= 0.65:
        return "≥65% band"
    if probability >= 0.60:
        return "60–65% band"
    if probability >= 0.55:
        return "55–60% band"
    return "<55% band"


def holdout_label(model_reliable) -> str:
    """The model_reliable flag, described as what it is: a 2025 out-of-time check."""
    return HOLDOUT_PASSED if str(model_reliable).strip().lower() in ("true", "1") else HOLDOUT_NOT_PASSED


def single_season(predictions: pd.DataFrame | None) -> int | None:
    """The one season the predictions name, if exactly one valid whole-number season."""
    if predictions is None or predictions.empty or "season" not in predictions.columns:
        return None
    values = set()
    for v in predictions["season"]:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if f != f or not f.is_integer():
            return None
        values.add(int(f))
    return values.pop() if len(values) == 1 else None
