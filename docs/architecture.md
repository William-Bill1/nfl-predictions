# NFL Predictions — Architecture

## Overview
Multi-page Streamlit app for NFL betting analytics. Three XGBoost+LightGBM
game-outcome models (spread / moneyline / totals) and a per-stat player-prop
system, all batch-computed into `data_files/`; the app only reads those files.
**Only the spread model currently drives a bet signal** — moneyline and totals
show market-implied probabilities (their models have no out-of-time edge).

## Pipeline
```
Step 1 — build/train  (python build_and_train_pipeline.py):
    nfl_data_py
        ↓  update_schedule.py         → data_files/nfl_schedule_<year>.csv
        ↓  create-nfl-historical.py   → data_files/nfl_games_historical.csv
                                        (all games incl. the unplayed schedule)
        ↓  nfl-gather-data.py  (~90s, no network, deterministic)
             feature engineering (all rows) → temporal train/test on PLAYED rows
             → data_files/nfl_games_historical_with_predictions.csv   (all rows)
               data_files/model_metrics.json
               data_files/model_feature_importances.csv
               data_files/best_features_spread.txt  (fixed point)

Player props  (python player_props/train_models.py ; python player_props/predict.py):
    PBP → aggregators.py → player_{passing,rushing,receiving}_stats.csv
        → models.py            → player_props/models/*.json
        → predict.py           → player_props_predictions.csv  (latest)
                               + player_props_predictions_week{W}_{season}.csv  (frozen, write-once)

Step 2 — UI:
    predictions.py                 (7 tabs; Underdog/Over-Under tabs removed)
    pages/1_Historical_Data.py
    pages/2_Player_Props.py        [includes the DK Pick 6 calculator]
    pages/3_Parlay_Builder.py
    pages/4_Model_Performance.py
```

## Determinism & mid-season
- Every XGB/LGBM estimator takes `RANDOM_STATE=42` + `n_jobs=1` (via `_XGB_KW` /
  `_LGBM_KW`); feature lists are `sorted()` on load and `best_features_spread.txt`
  is written sorted. `python nfl-gather-data.py` byte-reproduces its own outputs.
- `nfl-gather-data.py` computes a `_played` mask (rows with both final scores).
  Features and the probability write cover **all** rows so upcoming games get
  predictions; the train/test split, EV threshold, accuracy/MAE and season-long
  team-rate features use **played rows only**.
- Train/test is a temporal split (`temporal_split`): last 20% of played rows by
  (season, week). Not random — the earlier random split inflated every metric.

## ML Models
Three XGBoost classifiers (binary):
| Model | Target | Threshold | Notes |
|-------|--------|-----------|-------|
| Spread | trained on `spreadCovered` (favorite covers); ships `prob_underdogCovered = 1 - that` | EV-based | the only model that drives a bet signal |
| Moneyline | trained on `underdogWon`; **not shipped** — ships market implied prob (no out-of-time edge) | — | — |
| Totals | trained on `overHit`; **not shipped** — ships market implied P(over) (coin flip out-of-time) | — | — |

Spread confidence tiers (`add_spread_confidence_tiers`): Elite ≥0.60, Strong
0.55–0.60, Good 0.52–0.55, Lean 0.50–0.52. The EV threshold (`spread_ev_threshold`,
in P(underdog covers) space, pushes excluded) is ~0.545.

### Spread convention (Aug 2026 — was the "inversion fix")

`model_spread` predicts `P(favorite covers)`. The dashboard and EV code work in
`P(underdog covers)`, so `nfl-gather-data.py` computes the complement **once**:
`prob_underdogCovered = 1 - _blend_proba(model_spread, ...)`. A push
(`spreadPush = 1`, favorite's margin lands exactly on the line) is neither a
cover nor an underdog cover and is excluded from accuracy/ROI. The EV threshold,
`Spread Accuracy`, `Spread MAE` and `predictedSpreadCovered` are all in
underdog-covers space. This is a change of convention, not a fix for a
"backwards" model — the earlier "-90% → +60% ROI" story was a variable mix-up
(favorite-covers probability fed into underdog-covers bet logic).

### Player Props (`player_props/`)
XGBoost + LightGBM soft-voting ensembles per (stat, player-tier) at training
time; **inference loads the XGB `.json` only** (the `_lgbm.txt` sidecars are
git-ignored). `predict.py` targets `--season`/`--week` (default: next upcoming
week of the current schedule) and writes a **write-once** frozen snapshot
`player_props_predictions_week{W}_{season}.csv`; `backtest.py` scores that
snapshot, so `run_weekly_accuracy_check` is a genuine prospective test.
`models.py` uses a **temporal hold-out** (most recent season) — each metrics
record carries `base_rate`, `roc_auc` and a `reliable` flag; only ~5/26 models
clear an out-of-time bar (AUC ≥ 0.58 and accuracy above the majority base rate).
All TD props are force-flagged `reliable = False` (every tier collapses to the
same 0.5 line and they are ~coin-flip out-of-time); `predict.py` carries the
flag through as `model_reliable` and the UI marks unreliable rows "display
only". `opponent_def_rank` was removed (a dataset-wide leaky average that
clipped to a constant). Prop lines are still fixed tiers (275, 250, …) not
market lines — the weekly hit rate reflects line placement, not betting edge.

## Feature Engineering
All features are pre-game only (zero data leakage):
- **Momentum** (8): Last 3 games win%, scoring, point differential
- **Rest** (5): Rest day differences, well-rested ≥10d / short-rest ≤6d flags
- **Weather** (3): Cold ≤32°F, windy ≥15mph, extreme conditions
- Rolling stats: `prior_games = df[(team) & ((season < s) | (season == s & week < w))]`

## API Integrations
| Source | Purpose | Notes |
|--------|---------|-------|
| nfl_data_py | Schedule, play-by-play | Local, no key needed |
| ESPN scores | Completed game scores | Runtime, public API |
| SMTP email | Bet notifications | `emailer.py`, Gmail App Passwords |

No runtime API calls except ESPN scores for completed games.

## Key Components
- `build_and_train_pipeline.py` — `update_schedule` → `create-nfl-historical` → `nfl-gather-data`
- `nfl-gather-data.py` — feature engineering + train (played rows) + predict (all rows)
- `create-nfl-historical.py` — schedule + game fetch via nfl_data_py
- `season_utils.py` — `upcoming_or_current_season()` (schedules) / `latest_pbp_season()` (PBP); one source of truth for the season year
- `player_props/train_models.py` — prop model training
- `player_props/predict.py` — prop predictions + frozen weekly snapshot
- `scripts/export_best_bets.py` — reads the predictions CSV (`pred_spreadCovered_optimal == 1`, today's games) → `best_bets_today.json`; independent of the app
- `scripts/send_rich_email_now.py` — SMTP email sender
- `scripts/generate_rss.py` — `alerts_feed.xml` RSS feed

## Storage
All data in `data_files/` (committed to git):
- `nfl_games_historical_with_predictions.csv` — games (played + upcoming) + spread/market probabilities
- `model_metrics.json`, `model_feature_importances.csv`, `best_features_spread.txt` — model eval + selected features
- `player_props_predictions.csv` — latest prop feed; `player_props_predictions_week{W}_{season}.csv` — frozen weekly snapshots
- `betting_recommendations_log.csv` — spread recs, appended by the running app
- `best_bets_today.json` — Sports Picks Grid feed
- `data_files/exports/` — PDF exports

## Memory Optimisation (Streamlit Cloud)
- All numeric columns → `float32` (50% reduction vs `float64`)
- Use DataFrame views not `.copy()`
- All data loading via `@st.cache_data` — NEVER at module level (causes silent crashes)
