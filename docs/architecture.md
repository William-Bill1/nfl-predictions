# NFL Predictions — Architecture

## Overview
Multi-page Streamlit app for NFL betting predictions using XGBoost models. Predicts outcomes for spread, moneyline, over/under markets, and player props. All data and models are pre-computed locally.

## Two-Step Pipeline
```
Step 1 (~5 min, run first):
    nfl_data_py library
        ↓
    create-nfl-historical.py → data_files/nfl_games_historical.csv
        ↓
    nfl-gather-data.py → Feature Engineering + XGBoost Training
        ↓
    data_files/nfl_games_historical_with_predictions.csv
    data_files/model_feature_importances.csv
    data_files/model_metrics.json

    (Run both via: python build_and_train_pipeline.py)

Step 2 — UI:
    predictions.py (main dashboard)
    pages/1_📊_Historical_Data.py
    pages/2_🎯_Player_Props.py  [includes DK Pick 6 Calculator]
    pages/3_🎲_Parlay_Builder.py
    pages/4_📈_Model_Performance.py
```

## ML Models
Three XGBoost classifiers (binary):
| Model | Target | Threshold | Notes |
|-------|--------|-----------|-------|
| Spread | trained on `spreadCovered` (favorite covers); ships `prob_underdogCovered = 1 - that` | EV-based | the only model that drives a bet signal |
| Moneyline | trained on `underdogWon`; **not shipped** — ships market implied prob (no out-of-time edge) | — | — |
| Totals | trained on `overHit`; **not shipped** — ships market implied P(over) (coin flip out-of-time) | — | — |

Confidence tiers: Elite ≥65%, Strong 60–65%, Good 55–60%, Standard 50–55%

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
XGBoost + LightGBM soft-voting ensembles per stat category. Models in `player_props/models/*.json`. DK Pick 6 Calculator in `pages/2_🎯_Player_Props.py`.

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
- `build_and_train_pipeline.py` — runs both pipeline steps (~5 min)
- `nfl-gather-data.py` — feature engineering + XGBoost training
- `create-nfl-historical.py` — historical data fetch via nfl_data_py
- `player_props/train_models.py` — player prop model training
- `scripts/export_best_bets.py` — `best_bets_today.json` writer
- `scripts/send_rich_email_now.py` — SMTP email sender
- `scripts/generate_rss.py` — `alerts_feed.xml` RSS feed

## Storage
All data in `data_files/` (committed to git):
- `nfl_games_historical_with_predictions.csv` — main dataset + predictions
- `model_feature_importances.csv`, `model_metrics.json` — model eval
- `best_bets_today.json` — Sports Picks Grid feed
- `data_files/exports/` — PDF exports

## Memory Optimisation (Streamlit Cloud)
- All numeric columns → `float32` (50% reduction vs `float64`)
- Use DataFrame views not `.copy()`
- All data loading via `@st.cache_data` — NEVER at module level (causes silent crashes)
