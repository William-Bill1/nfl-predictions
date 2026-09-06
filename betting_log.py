"""Betting-recommendation log: append new spread signals, grade finished ones.

`data_files/betting_recommendations_log.csv` is the running record of every
spread bet the model flagged and how it turned out. Two operations:

* append_recommendations - add rows for upcoming games where
  pred_spreadCovered_optimal == 1 (deduped against still-pending rows).
* grade_pending - for pending rows whose game now has a final score, fill in
  actual_home_score / actual_away_score / bet_result (win|loss|push) /
  bet_profit, using the underdogCovered / spreadPush labels that
  nfl-gather-data.py already computed.

Both take the predictions DataFrame
(data_files/nfl_games_historical_with_predictions.csv) so this works headlessly
in the nightly workflow - it does not depend on the Streamlit app running.

Moneyline and totals are disabled, so only `spread` bets are logged/graded.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd

DATA_DIR = "data_files"
LOG_PATH = os.path.join(DATA_DIR, "betting_recommendations_log.csv")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "nfl_games_historical_with_predictions.csv")

LOG_COLUMNS = [
    "log_date", "season", "week", "game_id", "gameday", "home_team", "away_team",
    "bet_type", "recommended_team", "spread_line", "total_line", "moneyline_odds",
    "model_probability", "edge", "confidence_tier",
    "actual_home_score", "actual_away_score", "bet_result", "bet_profit",
]

# -110 spread bet: risk 100 to win ~90.91.
WIN_PROFIT = 90.91
LOSS_PROFIT = -100.0


def _spread_tier(prob: float) -> str:
    """Mirror of predictions.py::SPREAD_TIER_CUTS (kept local so this stays
    importable without the Streamlit module)."""
    if prob >= 0.65:
        return "Elite"
    if prob >= 0.59:
        return "Strong"
    if prob >= 0.55:
        return "Good"
    return "Lean"


def _underdog_team(row) -> str:
    sl = row.get("spread_line")
    if pd.isna(sl):
        return "Unknown"
    if sl < 0:
        return str(row.get("home_team", ""))   # away favored -> home is the dog
    if sl > 0:
        return str(row.get("away_team", ""))   # home favored -> away is the dog
    return "Pick"


def append_recommendations(preds_df: pd.DataFrame, log_path: str = LOG_PATH) -> int:
    """Append rows for upcoming games with a spread signal. Returns rows added."""
    if preds_df is None or "pred_spreadCovered_optimal" not in preds_df.columns:
        return 0

    df = preds_df.copy()
    df["gameday"] = pd.to_datetime(df.get("gameday"), errors="coerce")
    today = pd.to_datetime(datetime.now().date())
    # Only log games in the next ~10 days so each week's recorded signal
    # reflects that week's model state (the pipeline re-trains weekly).
    horizon = pd.to_datetime((datetime.now() + timedelta(days=10)).date())
    signal = df[(df["gameday"] > today) & (df["gameday"] <= horizon)
                & (df["pred_spreadCovered_optimal"] == 1)]
    if signal.empty:
        return 0

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    records = [{
        "log_date": now,
        "season": row.get("season", ""),
        "week": row.get("week", ""),
        "game_id": row.get("game_id", ""),
        "gameday": row.get("gameday", ""),
        "home_team": row.get("home_team", ""),
        "away_team": row.get("away_team", ""),
        "bet_type": "spread",
        "recommended_team": _underdog_team(row),
        "spread_line": row.get("spread_line", ""),
        "total_line": row.get("total_line", ""),
        "moneyline_odds": "",
        "model_probability": row.get("prob_underdogCovered", ""),
        "edge": row.get("edge_underdog_spread", ""),
        "confidence_tier": _spread_tier(float(row.get("prob_underdogCovered", 0) or 0)),
        "actual_home_score": "", "actual_away_score": "",
        "bet_result": "pending", "bet_profit": "",
    } for _, row in signal.iterrows()]

    new_df = pd.DataFrame(records, columns=LOG_COLUMNS)

    existing = None
    if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
        existing = pd.read_csv(log_path)

    if existing is not None and len(existing):
        pending = existing[existing["bet_result"] == "pending"]
        keyed = set(zip(pending["game_id"].astype(str), pending["bet_type"].astype(str)))
        new_df = new_df[~new_df.apply(
            lambda x: (str(x["game_id"]), str(x["bet_type"])) in keyed, axis=1
        )]
        if new_df.empty:
            return 0
        pd.concat([existing, new_df], ignore_index=True).to_csv(log_path, index=False)
    else:
        new_df.to_csv(log_path, index=False)
    return len(new_df)


def grade_pending(preds_df: pd.DataFrame, log_path: str = LOG_PATH) -> int:
    """Grade pending rows whose game has a final score. Returns rows graded."""
    if preds_df is None or not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return 0

    log_df = pd.read_csv(log_path)
    if "bet_result" not in log_df.columns or (log_df["bet_result"] == "pending").sum() == 0:
        return 0

    need = {"game_id", "gameday", "home_score", "away_score", "underdogCovered", "spreadPush"}
    if not need.issubset(preds_df.columns):
        return 0

    # A game is gradable only once its date is fully past AND it has a real
    # (non 0-0) final score. nfl-gather-data.py fillna(0)s unplayed rows, so
    # "scores present" alone is not enough.
    gd = pd.to_datetime(preds_df["gameday"], errors="coerce")
    today = pd.to_datetime(datetime.now().date())
    total = preds_df["home_score"].fillna(0) + preds_df["away_score"].fillna(0)
    played = preds_df[(gd < today) & (total > 0)]
    by_id = {str(r["game_id"]): r for _, r in played.iterrows()}

    graded = 0
    for i in log_df.index[log_df["bet_result"] == "pending"]:
        row = log_df.loc[i]
        g = by_id.get(str(row.get("game_id")))
        if g is None or pd.isna(g["home_score"]) or pd.isna(g["away_score"]):
            continue
        log_df.at[i, "actual_home_score"] = int(g["home_score"])
        log_df.at[i, "actual_away_score"] = int(g["away_score"])
        if str(row.get("bet_type")) != "spread":
            continue  # moneyline/totals disabled - nothing to grade
        if int(g.get("spreadPush", 0)) == 1:
            log_df.at[i, "bet_result"], log_df.at[i, "bet_profit"] = "push", 0.0
        elif int(g.get("underdogCovered", 0)) == 1:
            log_df.at[i, "bet_result"], log_df.at[i, "bet_profit"] = "win", WIN_PROFIT
        else:
            log_df.at[i, "bet_result"], log_df.at[i, "bet_profit"] = "loss", LOSS_PROFIT
        graded += 1

    if graded:
        log_df.to_csv(log_path, index=False)
    return graded


def main() -> None:
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"[betting_log] predictions file not found: {PREDICTIONS_PATH}")
        return
    preds = pd.read_csv(PREDICTIONS_PATH, sep="\t")
    added = append_recommendations(preds)
    graded = grade_pending(preds)
    print(f"[betting_log] appended {added} new spread rec(s), graded {graded} finished bet(s)")
    if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0:
        log = pd.read_csv(LOG_PATH)
        done = log[log["bet_result"].isin(["win", "loss"])]
        if len(done):
            wr = (done["bet_result"] == "win").mean()
            roi = pd.to_numeric(done["bet_profit"], errors="coerce").sum() / (len(done) * 100) * 100
            print(f"[betting_log] {len(done)} settled: {wr:.1%} win, {roi:+.1f}% ROI "
                  f"({int((log['bet_result'] == 'pending').sum())} pending, "
                  f"{int((log['bet_result'] == 'push').sum())} push)")


if __name__ == "__main__":
    main()
