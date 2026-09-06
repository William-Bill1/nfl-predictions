"""
scripts/export_best_bets.py -- NFL

Reads data_files/nfl_games_historical_with_predictions.csv (the pipeline output)
and writes data_files/best_bets_today.json in the Sports Picks Grid schema:
today's games that carry a spread bet signal (pred_spreadCovered_optimal == 1).

This is deliberately independent of the Streamlit app / betting_recommendations_log.csv
so the nightly workflow can produce the feed on its own.
"""
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

SPORT = "NFL"
MODEL_VERSION = "1.1.0"
SEASON = str(date.today().year)
OUT_PATH = Path("data_files/best_bets_today.json")
PRED_PATH = Path("data_files/nfl_games_historical_with_predictions.csv")


def _write(bets: list, notes: str = "") -> None:
    payload: dict = {
        "meta": {
            "sport": SPORT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
            "season": SEASON,
        },
        "bets": bets,
    }
    if notes:
        payload["meta"]["notes"] = notes
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[{SPORT}] Wrote {len(bets)} bets -> {OUT_PATH}")


def _tier(prob: float) -> str:
    if prob >= 0.60:
        return "Elite"
    if prob >= 0.55:
        return "Strong"
    if prob >= 0.52:
        return "Good"
    return "Lean"


def _safe_float(val):
    try:
        f = float(val)
        return f if f == f else None  # drop NaN
    except (TypeError, ValueError):
        return None


def _safe_int(val):
    try:
        return int(round(float(val)))
    except (TypeError, ValueError):
        return None


def main() -> None:
    today = date.today()

    # Off-season: still write an empty feed so the aggregator doesn't error.
    if not (today.month >= 9 or today.month <= 2):
        _write([], "NFL off-season")
        return

    if not PRED_PATH.exists():
        _write([], f"Source file not found: {PRED_PATH}")
        return

    try:
        df = pd.read_csv(PRED_PATH, sep="\t")
    except Exception as e:  # noqa: BLE001
        _write([], f"Failed to read source: {e}")
        return

    needed = {"gameday", "pred_spreadCovered_optimal", "spread_line",
              "home_team", "away_team", "prob_underdogCovered"}
    if not needed.issubset(df.columns):
        _write([], f"Predictions file missing columns: {sorted(needed - set(df.columns))}")
        return

    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    slate = df[
        (df["gameday"].dt.date == today)
        & (df["pred_spreadCovered_optimal"] == 1)
        & (df["spread_line"].fillna(0) != 0)
    ].copy()

    if slate.empty:
        _write([], f"No spread signals for {today}")
        return

    bets = []
    for _, row in slate.sort_values("gameday").iterrows():
        spread = float(row["spread_line"])          # +ve => home favored (nflverse)
        home, away = str(row["home_team"]), str(row["away_team"])
        if spread > 0:                               # home favored -> bet the away underdog
            underdog, dog_line, dog_odds = away, spread, row.get("away_spread_odds")
        else:                                        # away favored -> bet the home underdog
            underdog, dog_line, dog_odds = home, abs(spread), row.get("home_spread_odds")

        prob = _safe_float(row["prob_underdogCovered"]) or 0.0
        bets.append({
            "game_date": str(today),
            "game_time": (str(row["gametime"]).strip() or None) if "gametime" in row else None,
            "game": f"{away} @ {home}",
            "home_team": home,
            "away_team": away,
            "bet_type": "Spread",
            "pick": f"{underdog} +{dog_line:g}",
            "confidence": prob,
            "edge": _safe_float(row.get("edge_underdog_spread")),
            "tier": _tier(prob),
            "odds": _safe_int(dog_odds) or -110,
            "line": dog_line,
            "notes": None,
        })

    _write(bets)


if __name__ == "__main__":
    main()
