"""Rolling weekly spread backtest - a read-only audit of the spread model.

Answers one question honestly: had the production spread model been retrained
every week using only what was known before kickoff, how would its
P(underdog covers) have scored against simple market baselines?

Two reports, never merged:

* RETROSPECTIVE (`retrospective_report.json`) - rebuilt from today's nflverse
  schedule. Lines and odds are nflverse CLOSING values, so this is an estimate
  of what the model would have said, not what it did say.
* FROZEN PREGAME (`frozen_pregame_report.json`) - the probabilities the
  pipeline actually logged before kickoff (`betting_recommendations_log.csv`),
  settled on the originally recorded line. Only bet signals were logged, so
  this is a biased sample of the model's output.

Method (per evaluated NFL week W, whole weeks, chronological):

1. Features: every team aggregate (win %, blowout %, cover %, ...) comes from
   team_features.py - the same code production uses - i.e. the mean over that
   team's COMPLETED games from strictly earlier weeks, with production's
   home-games-only / away-games-only split. A game's own outcome, same-week
   outcomes and every later outcome never touch its features.
2. Training pool: completed games with a line from weeks before W. Asserted:
   every pool game kicked off before W's earliest kickoff.
3. Models compared on W's games:
   - production_config: production's estimator exactly (XGBoost + LightGBM,
     each CalibratedClassifierCV(isotonic, cv=5), averaged; target = favorite
     covers; P(underdog) = 1 - P(favorite)), refit on the whole pool. Its
     calibration is sklearn's internal cross-fitted folds, which are not
     chronological - that is production's behaviour and is reported as such.
   - production_holdout_platt: the same two uncalibrated base models fitted on
     the FIT period only (pool minus its most recent whole weeks), then a Platt
     (logistic) calibrator fitted on their predictions for the CALIBRATION
     period (the most recent whole weeks holding >= --cal-min-games games). The
     base models are never refit on the calibration weeks, so the calibrator is
     applied to exactly the model it was fitted against.
   - logistic_abs_spread: logistic regression on |spread_line|, fit on the pool.
   - constant_50: 0.5 for every game.
   - devig_closing: the underdog's closing spread odds devigged against the
     favorite's. Only where both prices exist; reported with its own counts.
4. Bets: underdog when p >= 0.5438 (production's rule: -110 break-even 52.38%
   plus a 2-point edge). Profit per $100 risk at the underdog's nflverse
   closing price (price_source=closing_odds_nflverse) or, if missing, an
   assumed -110 (price_source=assumed_-110). Pushes return 0 profit.
5. Metrics: Brier, log loss, accuracy and calibration bins exclude pushes. ROI =
   profit / (100 x non-push bets). Missing / pick'em lines are excluded and
   counted. 95% intervals come from a bootstrap that resamples whole weeks.

Usage:
    python scripts/rolling_spread_backtest.py --smoke
    python scripts/rolling_spread_backtest.py --start 2023-1 --end 2026-22

Outputs go only to backtest_output/ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from team_features import TEAM_FEATURES, compute_team_features  # noqa: E402

SCHEDULE_PATH = ROOT / "data_files" / "nfl_games_historical.csv"
LOG_PATH = ROOT / "data_files" / "betting_recommendations_log.csv"
FEATURES_PATH = ROOT / "data_files" / "best_features_spread.txt"
OUTPUT_DIR = ROOT / "backtest_output"

BREAKEVEN_110 = 110 / 210
BET_THRESHOLD = 0.5438          # production: BREAKEVEN (0.5238) + min_edge 0.02
EPS = 1e-6
CAL_EDGES = np.linspace(0.0, 1.0, 11)
MODELS = ["production_config", "production_holdout_platt", "logistic_abs_spread",
          "constant_50", "devig_closing"]

# Pregame columns that may be passed straight through as features. Anything
# score-derived (total, result, ...) is deliberately absent.
PREGAME_COLUMNS = {
    "spread_line", "away_moneyline", "home_moneyline", "away_spread_odds",
    "home_spread_odds", "total_line", "under_odds", "over_odds", "div_game",
    "temp", "wind", "away_rest", "home_rest", "week", "season",
}
@dataclass
class Config:
    start: tuple[int, int] = (2023, 1)
    end: tuple[int, int] = (2100, 99)
    seed: int = 42
    n_boot: int = 1000
    cal_min_games: int = 256
    min_fit_games: int = 300
    n_estimators: int = 150
    max_weeks: int | None = None
    features: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def prepare_games(raw: pd.DataFrame) -> pd.DataFrame:
    """Add week key, kickoff, completion flag and outcome/label columns."""
    df = raw.copy().reset_index(drop=True)
    df["key"] = df["season"].astype(int) * 100 + df["week"].astype(int)
    time = df["gametime"].fillna("00:00") if "gametime" in df else "00:00"
    df["kickoff"] = pd.to_datetime(df["gameday"].astype(str).str[:10] + " " + time,
                                   format="%Y-%m-%d %H:%M", errors="coerce")
    df["completed"] = df["home_score"].notna() & df["away_score"].notna()
    hs, as_, sl = df["home_score"], df["away_score"], df["spread_line"]
    margin = hs - as_
    df["has_line"] = sl.notna() & (sl != 0)
    df["missing_line"] = sl.isna()
    df["pickem"] = sl == 0

    def done(values):
        return pd.Series(values, index=df.index).where(df["completed"])

    df["homeWin"] = done((margin > 0).astype(float))
    df["awayWin"] = done((margin < 0).astype(float))
    df["isCloseGame"] = done((margin.abs() <= 3).astype(float))
    df["isBlowout"] = done((margin.abs() >= 20).astype(float))
    df["pointDiff"] = margin.abs()                       # production's (absolute) definition
    df["totalScore"] = hs + as_
    df["homeFavored"] = (sl > 0).astype(float).where(sl.notna())
    df["awayFavored"] = (sl < 0).astype(float).where(sl.notna())
    fav_margin = np.where(sl > 0, margin, -margin)
    df["spreadCovered"] = done(np.where(df["has_line"], (fav_margin > sl.abs()).astype(float), 0.0))
    if "total_line" in df:
        df["overHit"] = done((df["totalScore"] > df["total_line"]).astype(float))
        df["underHit"] = done((df["totalScore"] < df["total_line"]).astype(float))
        df["totalHit"] = done((df["totalScore"] == df["total_line"]).astype(float))

    # Underdog-side labels (the shipped convention). spread_line > 0 means the
    # home team is favored, so the away team is the underdog.
    df["underdog_is_home"] = sl < 0
    ud_result = np.where(sl > 0, -margin + sl, margin - sl)  # underdog margin + its points
    df["ud_result"] = pd.Series(ud_result, index=df.index).where(df["completed"] & df["has_line"])
    df["push"] = df["ud_result"] == 0
    df["ud_covered"] = (df["ud_result"] > 0).astype(float).where(df["ud_result"].notna())
    return df


def build_features(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Leak-free feature matrix for `names` (production feature names).

    Team aggregates come from `team_features.py` - the same code the
    production pipeline uses - so backtest and production features agree.
    """
    team_names = [n for n in names if n not in PREGAME_COLUMNS]
    for name in team_names:
        if name not in TEAM_FEATURES:
            raise ValueError(f"Feature {name!r} has no leak-free definition in this backtest")
    team = compute_team_features(df, team_names) if team_names else pd.DataFrame(index=df.index)
    out = pd.DataFrame(index=df.index)
    for name in names:
        out[name] = (pd.to_numeric(df[name], errors="coerce").fillna(0.0)
                     if name in PREGAME_COLUMNS else team[name])
    return out


def load_production_features(path: Path = FEATURES_PATH) -> list[str]:
    lines = [l.strip() for l in path.read_text().splitlines()]
    return sorted(l for l in lines if l and not l.lower().startswith("best mean"))


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------

def eval_week_keys(df: pd.DataFrame, cfg: Config) -> list[int]:
    lo, hi = cfg.start[0] * 100 + cfg.start[1], cfg.end[0] * 100 + cfg.end[1]
    ok = df["completed"] & df["has_line"] & df["key"].between(lo, hi)
    keys = sorted(df.loc[ok, "key"].unique().tolist())
    return keys[: cfg.max_weeks] if cfg.max_weeks else keys


def training_pool(df: pd.DataFrame, key: int) -> tuple[pd.Index, pd.Timestamp]:
    """Completed, lined games from weeks before `key`, all kicked off before its first game."""
    cutoff = df.loc[df["key"] == key, "kickoff"].min()
    pool = df.index[(df["key"] < key) & df["completed"] & df["has_line"]]
    if len(pool) and not (df.loc[pool, "kickoff"] < cutoff).all():
        raise RuntimeError(f"training game kicked off after week {key} began ({cutoff})")
    return pool, cutoff


def split_fit_calibration(df: pd.DataFrame, pool: pd.Index, cal_min_games: int) -> tuple[pd.Index, pd.Index]:
    """Most recent whole weeks with >= cal_min_games games -> calibration; earlier -> fit."""
    keys = df.loc[pool, "key"]
    counts = keys.value_counts().sort_index(ascending=False)
    cal_keys, total = [], 0
    for k, n in counts.items():
        if total >= cal_min_games:
            break
        cal_keys.append(k)
        total += n
    is_cal = keys.isin(cal_keys)
    return pool[~is_cal.to_numpy()], pool[is_cal.to_numpy()]


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def _base_models(n_estimators: int, seed: int):
    from xgboost import XGBClassifier
    import lightgbm as lgb
    xgb = XGBClassifier(eval_metric="logloss", n_estimators=n_estimators, max_depth=6,
                        learning_rate=0.1, random_state=seed, n_jobs=1)
    lgbm = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=6, learning_rate=0.1,
                              verbosity=-1, random_state=seed, n_jobs=1)
    return xgb, lgbm


def fit_production_config(X, y_fav, cfg: Config):
    from sklearn.calibration import CalibratedClassifierCV
    return [CalibratedClassifierCV(m, method="isotonic", cv=5).fit(X, y_fav)
            for m in _base_models(cfg.n_estimators, cfg.seed)]


def predict_fav(models, X) -> np.ndarray:
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_holdout_platt(X_fit, y_fit, X_cal, y_cal, cfg: Config):
    """Base models on the fit period; Platt calibrator on the (untouched) calibration period."""
    from sklearn.linear_model import LogisticRegression
    base = [m.fit(X_fit, y_fit) for m in _base_models(cfg.n_estimators, cfg.seed)]
    raw_cal = predict_fav(base, X_cal)
    platt = LogisticRegression().fit(_logit(raw_cal).reshape(-1, 1), y_cal)
    return base, platt


def predict_holdout_platt(fitted, X) -> np.ndarray:
    base, platt = fitted
    return platt.predict_proba(_logit(predict_fav(base, X)).reshape(-1, 1))[:, 1]


def american_to_prob(odds) -> np.ndarray:
    o = np.asarray(odds, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(o < 0, -o / (100 - o), 100 / (100 + o))
    return np.where(np.isfinite(o) & (np.abs(o) >= 100), p, np.nan)


def devig_underdog(df: pd.DataFrame) -> np.ndarray:
    ud_home = df["underdog_is_home"].to_numpy()
    ud = american_to_prob(np.where(ud_home, df["home_spread_odds"], df["away_spread_odds"]))
    fav = american_to_prob(np.where(ud_home, df["away_spread_odds"], df["home_spread_odds"]))
    return ud / (ud + fav)


def profit_per_100(result: str, odds: float) -> float:
    if result == "push":
        return 0.0
    if result == "loss":
        return -100.0
    return round(100 * (100 / -odds if odds < 0 else odds / 100), 2)


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------

def run_backtest(raw: pd.DataFrame, cfg: Config, progress=print) -> tuple[pd.DataFrame, list[dict], dict]:
    """Returns (per-game predictions, per-week sample counts, exclusion counts)."""
    from sklearn.linear_model import LogisticRegression

    df = prepare_games(raw)
    feats = cfg.features
    X_all = build_features(df, feats)
    weeks = eval_week_keys(df, cfg)
    lo, hi = cfg.start[0] * 100 + cfg.start[1], cfg.end[0] * 100 + cfg.end[1]
    in_range = df["completed"] & df["key"].between(lo, hi) & df["key"].isin(weeks)
    excluded = {"missing_line": int((in_range & df["missing_line"]).sum()),
                "pickem_line": int((in_range & df["pickem"]).sum())}

    rows, windows = [], []
    for key in weeks:
        pool, cutoff = training_pool(df, key)
        fit_idx, cal_idx = split_fit_calibration(df, pool, cfg.cal_min_games)
        ev = df.index[(df["key"] == key) & df["completed"] & df["has_line"]]
        win = {"season": key // 100, "week": key % 100, "cutoff": str(cutoff),
               "pool_games": len(pool), "fit_games": len(fit_idx), "cal_games": len(cal_idx),
               "fit_weeks": f"{df.loc[fit_idx, 'key'].min()}..{df.loc[fit_idx, 'key'].max()}" if len(fit_idx) else "",
               "cal_weeks": f"{df.loc[cal_idx, 'key'].min()}..{df.loc[cal_idx, 'key'].max()}" if len(cal_idx) else "",
               "eval_games": len(ev)}
        if len(fit_idx) < cfg.min_fit_games or len(cal_idx) < cfg.cal_min_games:
            win["skipped"] = "insufficient history"
            windows.append(win)
            continue
        windows.append(win)

        y_fav = df["spreadCovered"]
        prod = fit_production_config(X_all.loc[pool], y_fav[pool], cfg)
        platt = fit_holdout_platt(X_all.loc[fit_idx], y_fav[fit_idx], X_all.loc[cal_idx], y_fav[cal_idx], cfg)
        lr_rows = pool[~df.loc[pool, "push"].to_numpy()]
        lr = LogisticRegression().fit(df.loc[lr_rows, ["spread_line"]].abs(), df.loc[lr_rows, "ud_covered"])

        e = df.loc[ev]
        preds = {
            "production_config": 1 - predict_fav(prod, X_all.loc[ev]),
            "production_holdout_platt": 1 - predict_holdout_platt(platt, X_all.loc[ev]),
            "logistic_abs_spread": lr.predict_proba(e[["spread_line"]].abs())[:, 1],
            "constant_50": np.full(len(ev), 0.5),
            "devig_closing": devig_underdog(e),
        }
        ud_price = np.where(e["underdog_is_home"], e["home_spread_odds"], e["away_spread_odds"]).astype(float)
        price_ok = np.isfinite(ud_price) & (np.abs(ud_price) >= 100)
        for name, p in preds.items():
            for i, gid in enumerate(e.index):
                r = e.loc[gid]
                result = "push" if r["push"] else "win" if r["ud_covered"] == 1 else "loss"
                bet = bool(np.isfinite(p[i]) and p[i] >= BET_THRESHOLD)
                price = float(ud_price[i]) if price_ok[i] else -110.0
                rows.append({
                    "model": name, "season": int(r["season"]), "week": int(r["week"]), "key": int(key),
                    "game_id": r["game_id"], "spread_line": float(r["spread_line"]),
                    "p_underdog": float(p[i]) if np.isfinite(p[i]) else np.nan,
                    "ud_covered": float(r["ud_covered"]), "push": bool(r["push"]), "result": result,
                    "bet": bet, "price": price,
                    "price_source": "closing_odds_nflverse" if price_ok[i] else "assumed_-110",
                    "profit": profit_per_100(result, price) if bet else 0.0,
                    "evaluation": "retrospective_closing_line",
                })
        progress(f"[backtest] {key // 100} wk {key % 100:>2}: pool {len(pool)} "
                 f"(fit {len(fit_idx)}, cal {len(cal_idx)}), eval {len(ev)}")
    return pd.DataFrame(rows), windows, excluded


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def binary_metrics(p, y) -> dict:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    if len(y) == 0:
        return {"n": 0, "brier": None, "log_loss": None, "accuracy": None}
    return {"n": int(len(y)),
            "brier": round(float(((p - y) ** 2).mean()), 6),
            "log_loss": round(float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), 6),
            "accuracy": round(float(((p >= 0.5) == (y == 1)).mean()), 6)}


def calibration_bins(p, y) -> list[dict]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    out = []
    for lo, hi in zip(CAL_EDGES[:-1], CAL_EDGES[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                        "mean_pred": round(float(p[m].mean()), 4), "actual_rate": round(float(y[m].mean()), 4)})
    return out


def betting_summary(d: pd.DataFrame) -> dict:
    b = d[d["bet"]]
    w, l, pu = int((b["result"] == "win").sum()), int((b["result"] == "loss").sum()), int((b["result"] == "push").sum())
    profit = float(b["profit"].sum())
    return {"bets": int(len(b)), "win": w, "loss": l, "push": pu, "profit": round(profit, 2),
            "roi_pct": round(profit / ((w + l) * 100) * 100, 2) if w + l else None,
            "price_sources": {k: int(v) for k, v in b["price_source"].value_counts().items()}}


def summarize(d: pd.DataFrame) -> dict:
    scored = d[~d["push"] & d["p_underdog"].notna()]
    return {**binary_metrics(scored["p_underdog"], scored["ud_covered"]),
            "pushes_excluded": int(d["push"].sum()),
            "unavailable": int(d["p_underdog"].isna().sum()),
            "calibration": calibration_bins(scored["p_underdog"], scored["ud_covered"]),
            **betting_summary(d)}


def bootstrap_by_week(preds: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """95% percentile intervals, resampling whole weeks (paired across models)."""
    if n_boot <= 0 or preds.empty:
        return {}
    keys = np.array(sorted(preds["key"].unique()))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    count_per_draw = np.stack([np.bincount(d, minlength=len(keys)) for d in draws])  # B x W

    def per_week(model):
        d = preds[preds["model"] == model]
        s = d[~d["push"] & d["p_underdog"].notna()]
        p = np.clip(s["p_underdog"].to_numpy(), EPS, 1 - EPS)
        y = s["ud_covered"].to_numpy()
        frame = pd.DataFrame({"key": s["key"], "sq": (p - y) ** 2,
                              "ll": -(y * np.log(p) + (1 - y) * np.log(1 - p)),
                              "hit": ((p >= 0.5) == (y == 1)).astype(float), "n": 1.0})
        agg = frame.groupby("key")[["sq", "ll", "hit", "n"]].sum().reindex(keys, fill_value=0.0)
        bets = d[d["bet"] & (d["result"] != "push")].groupby("key").agg(profit=("profit", "sum"), nb=("profit", "size"))
        bets = bets.reindex(keys, fill_value=0.0)
        return {c: count_per_draw @ agg[c].to_numpy() for c in ("sq", "ll", "hit", "n")} | {
            "profit": count_per_draw @ bets["profit"].to_numpy(), "nb": count_per_draw @ bets["nb"].to_numpy()}

    def ci(x):
        x = x[np.isfinite(x)]
        return [round(float(np.percentile(x, 2.5)), 5), round(float(np.percentile(x, 97.5)), 5)] if len(x) else None

    base = per_week("constant_50")
    out = {"method": "percentile bootstrap over evaluation weeks (paired across models)",
           "n_boot": n_boot, "seed": seed, "n_weeks": int(len(keys)), "models": {}}
    for model in preds["model"].unique():
        s = per_week(model)
        with np.errstate(invalid="ignore", divide="ignore"):
            brier, ll = s["sq"] / s["n"], s["ll"] / s["n"]
            out["models"][model] = {
                "brier_95ci": ci(brier), "log_loss_95ci": ci(ll), "accuracy_95ci": ci(s["hit"] / s["n"]),
                "roi_pct_95ci": ci(np.where(s["nb"] > 0, s["profit"] / (s["nb"] * 100) * 100, np.nan)),
                "brier_minus_constant50_95ci": ci(brier - base["sq"] / base["n"]),
            }
    return out


def retrospective_report(preds: pd.DataFrame, windows: list[dict], excluded: dict, cfg: Config) -> dict:
    report = {
        "evaluation": "RETROSPECTIVE - rebuilt from the current nflverse schedule using closing lines/odds; "
                      "not the predictions the pipeline actually made before kickoff.",
        "config": {**asdict(cfg), "bet_threshold": BET_THRESHOLD},
        "methods": {
            "features": "team aggregates over strictly earlier completed games (earlier season/week), "
                        "production home-only/away-only split; no game's own or later outcome is used",
            "production_config": "XGB+LGBM, each CalibratedClassifierCV(isotonic, cv=5) (non-chronological internal folds, "
                                 "as production), refit weekly on all pool games",
            "production_holdout_platt": "same uncalibrated base models fit on fit weeks only; Platt calibrator fit on the "
                                        "later calibration weeks; base models never refit on calibration weeks",
            "logistic_abs_spread": "logistic regression of underdog cover on |spread_line|, pool games, pushes excluded",
            "constant_50": "0.5 for every game",
            "devig_closing": "nflverse closing spread odds, underdog price devigged against favorite price",
            "metrics": "Brier/log loss/accuracy/calibration exclude pushes; ROI = profit / (100 x non-push bets); "
                       "profit per $100 risk at the underdog's closing price, else assumed -110 (see price_sources)",
        },
        "excluded_games": excluded,
        "windows": windows,
        "per_season": {}, "overall": {},
    }
    if preds.empty:
        return report
    for model in MODELS:
        d = preds[preds["model"] == model]
        report["overall"][model] = summarize(d)
        for season, ds in d.groupby("season"):
            report["per_season"].setdefault(str(season), {})[model] = summarize(ds)
    report["uncertainty"] = bootstrap_by_week(preds, cfg.n_boot, cfg.seed)
    return report


# --------------------------------------------------------------------------
# Frozen pregame report
# --------------------------------------------------------------------------

def frozen_report(log: pd.DataFrame, raw: pd.DataFrame) -> dict:
    """Score the probabilities actually logged before kickoff, on the recorded line."""
    import betting_log as bl

    scores = raw.set_index("game_id")[["home_score", "away_score"]]
    rows, unresolved, pending = [], [], []
    for _, r in log[log["bet_type"].astype(str) == "spread"].iterrows():
        gid = str(r["game_id"])
        g = scores.loc[gid] if gid in scores.index else None
        if g is None or pd.isna(g["home_score"]) or pd.isna(g["away_score"]):
            pending.append(gid)
            continue
        result, profit, _ = bl._settle(r, g)
        if result in (None, bl.UNRESOLVED):
            unresolved.append({"game_id": gid, "recommended_team": str(r.get("recommended_team"))})
            continue
        src = r.get("odds_source")
        recorded = isinstance(src, str) and not src.lower().startswith("assumed")
        rows.append({"season": int(r["season"]), "game_id": gid, "p": float(r["model_probability"]),
                     "covered": 1.0 if result == "win" else 0.0, "push": result == "push", "result": result,
                     "profit": float(profit), "bet": True,
                     "price_source": "recorded" if recorded else "assumed_-110"})
    d = pd.DataFrame(rows)

    def summ(x):
        s = x[~x["push"]]
        return {**binary_metrics(s["p"], s["covered"]), "pushes_excluded": int(x["push"].sum()),
                "calibration": calibration_bins(s["p"], s["covered"]), **betting_summary(x)}

    return {
        "evaluation": "FROZEN PREGAME - model_probability as logged before kickoff, settled on the originally "
                      "recorded team and line. Only bet signals were logged (selection-biased sample); the "
                      "probability is the first logged value, not the last pregame value.",
        "price_note": "assumed_-110 rows have no sportsbook price on record; they are not actual prices",
        "unresolved_excluded": unresolved,
        "pending_or_unscored": len(pending),
        "overall": summ(d) if len(d) else {},
        "per_season": {str(s): summ(x) for s, x in d.groupby("season")} if len(d) else {},
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _season_week(text: str) -> tuple[int, int]:
    season, _, week = text.partition("-")
    return int(season), int(week or 1)


def _print_overall(report: dict) -> None:
    print(f"\n{'model':26} {'n':>4} {'brier':>7} {'logloss':>7} {'acc':>6} {'bets':>5} {'W-L-P':>9} {'ROI%':>7}  brier 95% CI")
    for model, s in report["overall"].items():
        ci = report.get("uncertainty", {}).get("models", {}).get(model, {}).get("brier_95ci")
        roi = "n/a" if s.get("roi_pct") is None else f"{s['roi_pct']:+.1f}"
        print(f"{model:26} {s['n']:>4} {s['brier'] or 0:>7.4f} {s['log_loss'] or 0:>7.4f} {s['accuracy'] or 0:>6.3f} "
              f"{s['bets']:>5} {s['win']:>3}-{s['loss']}-{s['push']:<3} {roi:>7}  {ci}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2023-1", help="first evaluated season-week, e.g. 2023-1")
    ap.add_argument("--end", default="2100-99", help="last evaluated season-week (default: latest completed)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--cal-min-games", type=int, default=256)
    ap.add_argument("--min-fit-games", type=int, default=300)
    ap.add_argument("--smoke", action="store_true", help="3 weeks, 25 trees, 50 bootstrap draws")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = ap.parse_args(argv)

    cfg = Config(start=_season_week(args.start), end=_season_week(args.end), seed=args.seed,
                 n_boot=args.n_boot, cal_min_games=args.cal_min_games, min_fit_games=args.min_fit_games,
                 features=load_production_features())
    if args.smoke:
        cfg.n_estimators, cfg.n_boot, cfg.max_weeks = 25, 50, 3

    raw = pd.read_csv(SCHEDULE_PATH, sep="\t")
    preds, windows, excluded = run_backtest(raw, cfg)
    retro = retrospective_report(preds, windows, excluded, cfg)
    frozen = frozen_report(pd.read_csv(LOG_PATH), raw) if LOG_PATH.exists() else {}

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = {"generated_at": datetime.now().isoformat(timespec="seconds"), "smoke": args.smoke}
    (out / "retrospective_report.json").write_text(json.dumps({**stamp, **retro}, indent=2, default=str))
    (out / "frozen_pregame_report.json").write_text(json.dumps({**stamp, **frozen}, indent=2, default=str))
    preds.to_csv(out / "retrospective_predictions.csv", index=False)

    print(f"\nRETROSPECTIVE (closing lines), weeks {cfg.start} .. {cfg.end}; features: {', '.join(cfg.features)}")
    if not preds.empty:
        _print_overall(retro)
    if frozen.get("overall"):
        f = frozen["overall"]
        print(f"\nFROZEN PREGAME (logged signals): n={f['n']} brier={f['brier']} log_loss={f['log_loss']} "
              f"bets={f['bets']} {f['win']}-{f['loss']}-{f['push']} ROI={f['roi_pct']}% "
              f"prices={f['price_sources']}; unresolved excluded: {len(frozen['unresolved_excluded'])}")
    print(f"\nwrote {out / 'retrospective_report.json'}, {out / 'frozen_pregame_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
