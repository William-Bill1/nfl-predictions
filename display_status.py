"""Presentation helpers for the home page (predictions.py). Display only.

* Game completion. The predictions CSV is regenerated nightly and stores 0-0
  scores (and 0 outcome flags) for games that have not been played, so a
  score of 0 or a past date is not evidence of a result. A game is shown as
  final only when
    1. the nflverse schedule (nfl_games_historical.csv) passes the repo's
       conservative final-score rule, bet_journal.completion_evidence: exactly
       one row; both scores finite non-negative integers, not 0-0; `result`
       and `total` consistent with them; `overtime` 0 or 1; game day before
       today (Toronto). Only that function is reused: it is pure and reads
       only the schedule frame passed to it, never journal files; and
    2. the predictions file's own home/away scores equal those final scores.
  Otherwise:
    * PENDING: the schedule row has no scores yet, or the game is today;
    * UNAVAILABLE: no schedule, no row, a duplicated game_id, or malformed
      evidence (text, negative, fractional or infinite scores, 0-0,
      inconsistent result/total/overtime, bad game day);
    * INCONSISTENT: the schedule is final but the predictions file shows
      different scores, or the schedule lists the game without scores while
      the predictions file shows one other than its 0-0 placeholder.
  Results of any game that is not final are shown blank and left out of
  performance counts. Stored scores, flags and settlement are not changed.
* Neutral labels for the spread model's probability bands. The stored tier
  values ("Elite", "Strong", ... in the betting log, "Elite"/"Good" in
  best_bets_today.json) are data and stay as they are; only what the page
  shows changes. Cut points, thresholds and sorting are untouched.
* Wording for model-implied edges and for predictions regenerated after the
  results were known.

Nothing here changes a probability, a threshold, a recommendation, a stored
file or settlement (betting_log / nfl-gather-data.py).
"""
from __future__ import annotations

from datetime import datetime, timezone
from os import path

import pandas as pd

import bet_journal as bj

SCHEDULE_NAME = "nfl_games_historical.csv"
EVIDENCE_COLUMNS = ("game_id", "gameday", "home_score", "away_score", "result", "total", "overtime")

FINAL = "Final"
PENDING = "Pending - no final score yet"
UNAVAILABLE = "Result unavailable"
INCONSISTENT = "Unavailable - sources disagree"

# bet_journal.completion_evidence's reason for a scored game dated today or later.
_NOT_BEFORE_TODAY = "game day is not before today"
_UNSCORED = ()   # schedule lists the game with no scores yet

# Columns that describe a game's result. Blanked for any game that isn't final.
RESULT_COLUMNS = ("home_score", "away_score", "underdogCovered", "overHit")
FLAG_COLUMNS = ("underdogCovered", "overHit")   # 0/1 outcomes shown as checkboxes

# Stored tier value -> label shown on the page. Keys cover both spellings in
# use: betting_log._spread_tier / export_best_bets ("Elite") and the old
# home-page labels ("🔥 Elite").
NEUTRAL_TIER_LABELS = {
    "Elite": "Model ≥65%",
    "Strong": "Model 59–65%",
    "Good": "Model 55–59%",
    "Lean": "Model 50–55%",
}
for _old, _new in list(NEUTRAL_TIER_LABELS.items()):
    NEUTRAL_TIER_LABELS[{"Elite": "🔥 Elite", "Strong": "⭐ Strong",
                         "Good": "📈 Good", "Lean": "⚖️ Lean"}[_old]] = _new

TIER_NOTE = ("Bands group the spread model's probability that the underdog covers. They are "
             "model-output ranges, not validated confidence levels or expected win rates: the "
             "model's calibration and betting value are not established.")

EDGE_HELP = ("Model-implied difference: the model's probability minus the probability implied by "
             "the price. It is only meaningful if the model is calibrated, which is not "
             "established. Not a demonstrated betting edge.")

REGENERATED_NOTE = ("These predictions come from the latest nightly run, which regenerates "
                    "probabilities and picks for every game, including games already played, "
                    "after their results were known. They are not the forecasts made before "
                    "kickoff and are not evidence of historical predictive success. For "
                    "predictions frozen before kickoff, see the Frozen Performance page.")

FROZEN_PAGE = "pages/9_Frozen_Performance.py"


def tier_label(value) -> str:
    """Neutral display label for a stored tier value; other values unchanged."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return NEUTRAL_TIER_LABELS.get(str(value), str(value))


def load_completion_evidence(data_dir: str) -> pd.DataFrame | None:
    """EVIDENCE_COLUMNS from the nflverse schedule, or None if the file is
    missing, unreadable or lacks any of them. Values are kept as read, so
    malformed scores reach the validation below rather than being coerced."""
    p = path.join(data_dir, SCHEDULE_NAME)
    if not path.exists(p):
        return None
    try:
        return pd.read_csv(p, sep="\t", usecols=list(EVIDENCE_COLUMNS),
                           dtype={"game_id": str, "gameday": str})
    except (OSError, ValueError):
        return None


def _schedule_verdicts(game_ids, evidence, now) -> dict:
    """game_id -> (FINAL | PENDING | UNAVAILABLE, scores): scores is the
    schedule's (home, away) when FINAL, _UNSCORED when the schedule lists
    the game without scores, else None."""
    ids = set(map(str, game_ids))
    if evidence is None or evidence.empty or not set(EVIDENCE_COLUMNS) <= set(evidence.columns):
        return {g: (UNAVAILABLE, None) for g in ids}
    ev = evidence.copy()
    ev["game_id"] = ev["game_id"].astype(str)
    ev = ev[ev["game_id"].isin(ids)]
    groups = {g: rows for g, rows in ev.groupby("game_id", sort=False)}
    out = {}
    for g in ids:
        rows = groups.get(g)
        if rows is None:
            out[g] = (UNAVAILABLE, None)
            continue
        if len(rows) == 1 and pd.isna(rows["home_score"].iloc[0]) and pd.isna(rows["away_score"].iloc[0]):
            out[g] = (PENDING, _UNSCORED)                   # listed, not played yet
            continue
        found, why = bj.completion_evidence(rows, g, now)  # also rejects duplicates
        if found is not None:
            out[g] = (FINAL, (found["home_score"], found["away_score"]))
        else:
            out[g] = (PENDING if why == _NOT_BEFORE_TODAY else UNAVAILABLE, None)
    return out


def completion_status(game_ids: pd.Series, evidence: pd.DataFrame | None,
                      now: datetime | None = None) -> pd.Series:
    """FINAL / PENDING / UNAVAILABLE from the schedule alone, aligned to
    game_ids. Dates and 0 scores in the predictions file are never used."""
    v = _schedule_verdicts(game_ids, evidence, now or datetime.now(timezone.utc))
    return pd.Series([v[str(g)][0] for g in game_ids], index=game_ids.index, dtype=object)


def result_status(df: pd.DataFrame, evidence: pd.DataFrame | None,
                  now: datetime | None = None) -> pd.Series:
    """Status for each row of a predictions frame: the schedule's verdict,
    cross-checked with the home/away scores the frame itself holds. A final
    game whose predictions scores differ, or a game the schedule lists
    without scores whose predictions file shows a real score, is INCONSISTENT."""
    if "game_id" not in df.columns:
        return pd.Series(UNAVAILABLE, index=df.index, dtype=object)
    v = _schedule_verdicts(df["game_id"], evidence, now or datetime.now(timezone.utc))
    has_scores = {"home_score", "away_score"} <= set(df.columns)
    out = []
    for idx, g in df["game_id"].items():
        status, scores = v[str(g)]
        shown = ((bj._int_field(df.at[idx, "home_score"]), bj._int_field(df.at[idx, "away_score"]))
                 if has_scores else (None, None))
        if status == FINAL and shown != scores:
            status = INCONSISTENT
        elif scores == _UNSCORED and None not in shown and shown != (0, 0):
            status = INCONSISTENT
        out.append(status)
    return pd.Series(out, index=df.index, dtype=object)


def blank_unsettled(df: pd.DataFrame, status: pd.Series,
                    columns=RESULT_COLUMNS) -> pd.DataFrame:
    """Copy of df with result columns set to missing wherever status != FINAL.
    Final games keep their values exactly."""
    out = df.copy()
    not_final = status.reindex(out.index) != FINAL
    for c in columns:
        if c in out.columns:
            col = pd.to_numeric(out[c], errors="coerce")
            if c in FLAG_COLUMNS:
                # 0/1 outcome flags: nullable Int64 renders as an empty checkbox.
                out[c] = col.astype("Int64").mask(not_final)
            else:
                # Scores stay float (as stored); st.dataframe shows a missing value with its
                # grey missing-value marker. The Status column says why it is missing.
                out[c] = col.astype("float64").mask(not_final)
    return out


def final_only(df: pd.DataFrame, evidence: pd.DataFrame | None,
               now: datetime | None = None) -> tuple[pd.DataFrame, int]:
    """A copy of the rows whose game is FINAL (result_status), and how many
    rows were left out."""
    keep = result_status(df, evidence, now) == FINAL
    return df[keep].copy(), int((~keep).sum())
