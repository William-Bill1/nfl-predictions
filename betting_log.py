"""Betting-recommendation log: append new spread signals, grade finished ones.

`data_files/betting_recommendations_log.csv` is the running record of every
spread bet the model flagged and how it turned out. Two operations:

* append_recommendations - add rows for upcoming games where
  pred_spreadCovered_optimal == 1 (deduped against all existing rows).
* grade_pending - for pending rows whose game now has a final score, fill in
  actual_home_score / actual_away_score / bet_result (win|loss|push) /
  bet_profit, using the originally recorded team handicap and American price.
  Rows with no recorded team (legacy "Pick") become `unresolved` and are
  excluded from totals.

Both take the predictions DataFrame
(data_files/nfl_games_historical_with_predictions.csv) so this works headlessly
in the nightly workflow - it does not depend on the Streamlit app running.

Moneyline and totals are disabled, so only `spread` bets are logged/graded.
"""

from __future__ import annotations

import os
import math
import shutil
import tempfile
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
    "bet_spread", "bet_odds", "odds_source",
]

# -110 spread bet: risk 100 to win ~90.91.
WIN_PROFIT = 90.91
LOSS_PROFIT = -100.0

# Rows that never recorded a team (legacy "Pick" rows) can't be settled.
# They are kept in the log for the record but excluded from every total.
UNRESOLVED = "unresolved"
ASSUMED_LEGACY_ODDS = "assumed -110 (legacy row, no price recorded)"
AUDIT_COLUMNS = [
    "game_id", "recommended_team", "spread_line", "previous_result", "previous_profit",
    "corrected_result", "corrected_profit", "corrected_at", "reason",
]


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
    signal = signal[signal["spread_line"].notna() & (signal["spread_line"] != 0)]
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
        "bet_spread": abs(float(row["spread_line"])),
        "bet_odds": -110,
        "odds_source": "assumed -110 model recommendation",
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
        keyed = set(zip(existing["game_id"].astype(str), existing["bet_type"].astype(str)))
        new_df = new_df[~new_df.apply(
            lambda x: (str(x["game_id"]), str(x["bet_type"])) in keyed, axis=1
        )]
        if new_df.empty:
            return 0
        pd.concat([existing, new_df], ignore_index=True).to_csv(log_path, index=False)
    else:
        new_df.to_csv(log_path, index=False)
    return len(new_df)


def grade_pending(preds_df: pd.DataFrame, log_path: str = LOG_PATH, *, regrade: bool = False,
                  audit_path: str | None = None) -> int:
    """Grade pending rows whose game has a final score. Returns rows graded.

    With regrade=True, already-settled and unresolved rows are re-settled too.
    Every change to a previously settled result is appended to `audit_path`
    (default: `default_audit_path(log_path)`, a dated file next to the log)
    BEFORE the log is rewritten. Each file is replaced atomically on its own
    (temp file + os.replace). If the audit write raises, the log is left
    untouched; if the log write raises, the audit is restored to its prior
    bytes. This is rollback for handled exceptions, not a crash-safe two-file
    transaction: a process kill between the two replaces leaves audit rows for
    corrections the log doesn't yet have (a later regrade would record them
    again). A regrade that changes nothing writes no audit rows, so repeated
    regrades are idempotent.
    """
    if preds_df is None or not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return 0

    log_df = pd.read_csv(log_path)
    if "bet_result" not in log_df.columns:
        return 0

    need = {"game_id", "gameday", "home_score", "away_score"}
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

    for col in ("bet_spread", "bet_odds", "odds_source"):
        if col not in log_df.columns:
            log_df[col] = pd.Series(dtype="object")
        log_df[col] = log_df[col].astype("object")

    graded = 0
    corrections = []
    states = ["pending", "win", "loss", "push", UNRESOLVED] if regrade else ["pending"]
    for i in log_df.index[log_df["bet_result"].isin(states)]:
        row = log_df.loc[i]
        g = by_id.get(str(row.get("game_id")))
        if g is None or pd.isna(g["home_score"]) or pd.isna(g["away_score"]):
            continue
        if str(row.get("bet_type")) != "spread":
            continue  # moneyline/totals disabled - nothing to grade
        result, profit, reason = _settle(row, g)
        if result is None:
            continue
        previous = (row.get("bet_result"), row.get("bet_profit"))
        log_df.at[i, "actual_home_score"] = int(g["home_score"])
        log_df.at[i, "actual_away_score"] = int(g["away_score"])
        log_df.at[i, "bet_result"], log_df.at[i, "bet_profit"] = result, profit
        if result != UNRESOLVED and pd.isna(row.get("bet_odds")):
            log_df.at[i, "bet_odds"] = -110
            log_df.at[i, "odds_source"] = ASSUMED_LEGACY_ODDS
        graded += 1
        if previous[0] in ("win", "loss", "push", UNRESOLVED) and not _same_settlement(previous, (result, profit)):
            corrections.append({
                "game_id": row.get("game_id"), "recommended_team": row.get("recommended_team"),
                "spread_line": row.get("spread_line"),
                "previous_result": previous[0], "previous_profit": previous[1],
                "corrected_result": result, "corrected_profit": profit,
                "corrected_at": datetime.now().isoformat(timespec="seconds"), "reason": reason,
            })

    if not graded:
        return 0
    restore_audit = None
    if corrections:
        # Audit first: a changed historical settlement is never written
        # without its correction record.
        restore_audit = _append_audit(audit_path or default_audit_path(log_path), corrections)
    try:
        _atomic_replace(log_path, lambda tmp: log_df.to_csv(tmp, index=False))
    except BaseException:
        if restore_audit:
            restore_audit()
        raise
    return graded


def default_audit_path(log_path: str = LOG_PATH, when: datetime | None = None) -> str:
    """Dated corrections file next to the log: settlement_corrections_YYYYMMDD.csv."""
    when = when or datetime.now()
    return os.path.join(os.path.dirname(log_path) or ".", f"settlement_corrections_{when:%Y%m%d}.csv")


def _atomic_replace(path: str, write) -> None:
    """Call write(tmp_path) on a temp file beside `path`, then swap it in."""
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=os.path.dirname(path) or ".")
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _append_audit(audit_path: str, corrections: list[dict]):
    """Append correction rows atomically, keeping existing entries.

    Returns a callable that restores the audit file to its prior state.
    """
    original = None                     # prior bytes, or None if the file didn't exist
    if os.path.exists(audit_path):
        with open(audit_path, "rb") as f:
            original = f.read()
    existed = bool(original)            # non-empty file -> append without a header
    audit = pd.DataFrame(corrections, columns=AUDIT_COLUMNS)

    def write(tmp):
        if existed:
            shutil.copyfile(audit_path, tmp)
        audit.to_csv(tmp, mode="a" if existed else "w", header=not existed, index=False)

    _atomic_replace(audit_path, write)

    def restore():
        if original is None:
            if os.path.exists(audit_path):
                os.remove(audit_path)
        else:
            def put_back(tmp):
                with open(tmp, "wb") as f:
                    f.write(original)
            _atomic_replace(audit_path, put_back)
    return restore


def spread_result(team: str, home: str, away: str, handicap: float,
                  home_score: float, away_score: float) -> str:
    """Win, loss or push for a spread bet on `team` at its own signed handicap
    (e.g. -3.5 lays 3.5), given the final score. Shared with bet_journal.py."""
    margin = float(home_score) - float(away_score)
    covered = (margin if team == home else -margin) + float(handicap)
    return "push" if abs(covered) < 1e-9 else "win" if covered > 0 else "loss"


def _settle(row, g) -> tuple[str | None, float | None, str]:
    """Settle one spread row from final scores and the bet as recorded.

    Returns (result, profit per $100 risk, reason). result is UNRESOLVED when
    the row never named one of the two teams (legacy "Pick" rows logged
    before a line existed); (None, None, "") when the row can't be graded.
    """
    team, home, away = row.get("recommended_team"), row.get("home_team"), row.get("away_team")
    if pd.isna(team) or team not in (home, away):
        return UNRESOLVED, None, f"no team recorded (recommended_team={team!r}); excluded from totals"
    handicap = row.get("bet_spread")
    reason = "graded against recorded team handicap and odds"
    if pd.isna(handicap):
        original = row.get("spread_line")
        if pd.isna(original):
            return None, None, ""
        # Convert the original nflverse line (home-favored positive) for the recorded team.
        handicap = -float(original) if team == home else float(original)
        reason = "graded against line recorded at log time (not closing line), assumed -110"
    odds = row.get("bet_odds")
    odds = -110.0 if pd.isna(odds) else float(odds)
    if not math.isfinite(odds) or abs(odds) < 100 or not math.isfinite(float(handicap)):
        return None, None, ""
    result = spread_result(team, home, away, handicap, g["home_score"], g["away_score"])
    profit = 0.0 if result == "push" else LOSS_PROFIT
    if result == "win":
        profit = round(100 * (100 / -odds if odds < 0 else odds / 100), 2)
    return result, profit, reason


def _same_settlement(a, b) -> bool:
    if a[0] != b[0]:
        return False
    pa, pb = a[1], b[1]
    if pd.isna(pa) or pd.isna(pb):
        return pd.isna(pa) and pd.isna(pb)
    return abs(float(pa) - float(pb)) < 0.005


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regrade", action="store_true", help="Correct settled bets using original recorded lines")
    args = parser.parse_args()
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"[betting_log] predictions file not found: {PREDICTIONS_PATH}")
        return
    preds = pd.read_csv(PREDICTIONS_PATH, sep="\t")
    added = append_recommendations(preds)
    graded = grade_pending(preds, regrade=args.regrade, audit_path=default_audit_path(LOG_PATH))
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
        unresolved = log[log["bet_result"] == UNRESOLVED]
        if len(unresolved):
            print(f"[betting_log] {len(unresolved)} unresolved (excluded from totals): "
                  + ", ".join(unresolved["game_id"].astype(str)))


if __name__ == "__main__":
    main()
