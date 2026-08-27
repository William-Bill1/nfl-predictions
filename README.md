# 🏈 NFL Betting Analytics & Predictions Dashboard

<p align="left">
  <img src="data_files/gridiron-oracle-transparent.png" alt="NFL Predictions Logo" width="260" />
</p>

A multi-page Streamlit app that trains machine-learning models on historical NFL
data (2020–present) to predict game outcomes — **spread** and **over/under**
betting signals, plus an underdog-moneyline view — and a **player-props** system
(passing / rushing / receiving yards and TDs) for DraftKings Pick 6-style markets. Everything is batch-computed: scripts write
CSV/JSON into `data_files/`, and the app reads those files, so no build step or
API keys are needed to run the dashboard.

- 📍 Roadmap: [`docs/ROADMAP.md`](docs/ROADMAP.md) · Player props: [`docs/PLAYER_PROPS_ROADMAP.md`](docs/PLAYER_PROPS_ROADMAP.md)
- 🏗️ Architecture detail: [`docs/architecture.md`](docs/architecture.md)
- 📜 Change history: [`CHANGELOG.md`](CHANGELOG.md)

---

## Quick start

**Requirements:** Python **3.12 or 3.13** (3.11 is *not* supported — the code
uses PEP 701 f-strings). All data and trained models are committed, so the app
runs without retraining.

```bash
python -m venv venv
# Windows:  .\venv\Scripts\Activate.ps1     macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt

python smoke_test.py            # sanity check: expect "SMOKE OK: rows=..."
streamlit run predictions.py    # opens http://localhost:8501
```

On Windows you can instead run `./run-local.ps1`. If port 8501 is busy:
`streamlit run predictions.py --server.port 8502`.

To run the test suite: `pip install pytest && pytest -q`.

---

## What it does

### Dashboard (`predictions.py` + `pages/`)

| Page / tab | Contents |
|---|---|
| **Predictions** (main) | Model predictions vs. actual results; upcoming-game probabilities & edges (model % − implied %); betting-performance metrics |
| **Underdog Bets** | Market implied probability that the underdog wins outright. The moneyline *model* is disabled — no out-of-time edge — so no bets are generated here. |
| **Spread Bets** | High-confidence spread picks, confidence-tiered |
| **Over/Under Bets** | Totals picks sorted by value edge |
| **Betting Log** | Auto-logged recommendations with timestamps and outcomes |
| **Historical Data** page | Filter-driven browser over ~290k play-by-play rows (2020–present); 12+ filters, quick presets, pagination |
| **Player Props** page | Per-player predictions + interactive **DK Pick 6 calculator** (enter a line, get OVER/UNDER with confidence tier; ML model or Laplace-smoothed historical fallback) |
| **Parlay Builder** page | Combine bets, compute parlay odds |
| **Model Performance** page | Accuracy, calibration, weekly tracking |

### Models

Three binary classifiers in `nfl-gather-data.py`, each a
`CalibratedClassifierCV(XGBClassifier, isotonic)` optionally soft-voted with a
LightGBM twin. Training is seeded and single-threaded, so the pipeline is
byte-reproducible.

| Target | Predicts | Ships? |
|---|---|---|
| `spreadCovered` | favorite covers the spread | yes — EV-based threshold |
| `overHit` | total goes over | yes — F1-optimized threshold |
| `underdogWon` | underdog wins outright | **no** — model trained for diagnostics only; `prob_underdogWon` ships the market implied probability instead |

Performance on the **temporal** hold-out (last 20% of games by date, 339 games):

| | test accuracy | notes |
|---|---|---|
| Spread | ~0.55 | +EV bet subset (~13 games) backtests around +17% theoretical ROI |
| Totals | ~0.51 | near coin-flip |
| Moneyline | — | model AUC ≈ 0.56, worse-calibrated than the 33% base rate, negative backtest ROI → **disabled** (see the Underdog Bets note) |

The spread model is trained on "favorite covers" and its probability is then
inverted (`1 − p`); see
[`docs/SPREAD_MODEL_INVESTIGATION.md`](docs/SPREAD_MODEL_INVESTIGATION.md) for
what that fix actually does and what remains fragile.

### Feature engineering

~75 candidate features: rolling team win/scoring/differential rates, last-3-game
momentum, rest-day advantage, weather flags, spread-size buckets. All are
computed with a strict "prior games only" filter
(`season < s OR (season == s AND week < w)`), so the **features** contain no
future information. Best-feature subsets per target are cached in
`data_files/best_features_*.txt`.

---

## Data

| Source | Used for | Notes |
|---|---|---|
| **nflverse** (`nfl_data_py`) | schedules, play-by-play | local, no key |
| **ESPN** scores API | completed-game scores/odds | runtime, in-app only (`predictions.py::update_completed_games`) |
| **Open-Meteo** | player-prop weather adjustments | `player_props/weather.py` |
| **ESPN** injury page | player-prop injury adjustments | scraped in `player_props/injuries.py` |

All artifacts live in `data_files/` and are committed. The big one,
`nfl_play_by_play_historical.csv.gz` (~116 MB, **tab-separated**), is tracked
with **Git LFS** — run `git lfs pull` if it comes down as a pointer. The app
degrades gracefully if it's missing.

---

## Developer notes

### Run the full pipeline locally

```bash
python build_and_train_pipeline.py     # update_schedule → create-nfl-historical → nfl-gather-data  (~5 min)
```

Or step by step:

```bash
python update_schedule.py              # refresh data_files/nfl_schedule_<year>.csv
python create-nfl-historical.py        # nflverse schedules → data_files/nfl_games_historical.csv
python nfl-gather-data.py              # feature engineering + train + predict → *_with_predictions.csv, model_metrics.json
python update_pbp_smart.py             # refresh the play-by-play LFS file (only downloads when stale)
```

### Retrain models

- **Game models:** `python nfl-gather-data.py` (reads the committed
  `nfl_games_historical.csv`; no network needed). Writes
  `nfl_games_historical_with_predictions.csv`, `model_metrics.json`,
  `model_feature_importances.csv`.
- **Player-prop models:** `python player_props/train_models.py`
  (add `--skip-aggregation` to reuse cached per-player game logs). Writes
  `player_props/models/*.json` + `model_metrics.json`.

### Add a feature or data source

1. Add the column in `nfl-gather-data.py` (compute it from **prior games only** —
   no leakage) and append its name to the `features` list.
2. Note that `select_dtypes` drops non-numeric columns before training, so
   categoricals need encoding first.
3. Retrain (`python nfl-gather-data.py`) so `best_features_*.txt` and the
   predictions CSV regenerate.
4. If the dashboard reads the new column, add it there too — `predictions.py`
   and `pages/` load the same predictions CSV.

### Conventions

- Season-year logic is centralized in [`season_utils.py`](season_utils.py)
  (`upcoming_or_current_season()` for schedules, `latest_pbp_season()` for PBP) —
  don't re-derive it inline.
- Data loading in the app must go through `@st.cache_data`, never at module
  scope (Streamlit Cloud OOMs otherwise).
- Put new helper/diagnostic scripts in `scripts/`, import-safe (no heavy loads
  at import), with a header comment and an `if __name__ == '__main__'` entry.
- On Windows, run pipeline scripts with `PYTHONUTF8=1` — emoji `print()`
  otherwise crashes the cp1252 console.

---

## Automation (GitHub Actions)

| Workflow | Schedule | Purpose |
|---|---|---|
| `nightly-update.yml` | 03:00 UTC, Sep–Feb | refresh PBP, run pipeline, retrain prop models, commit results |
| `weekly-model-performance.yml` | Mondays 06:00 UTC, Sep–Feb | backtest last week, persist `accuracy_results_*.json` |
| `update-schedule.yml` | daily 06:00 UTC | refresh `nfl_schedule_<year>.csv` |
| `send_predictions_schedule.yml` | Wed evenings (in season) | email predictions |
| `rss_test.yml` | on push | regenerate + link-check `alerts_feed.xml` |
| `keep-alive.yml` | twice daily | ping the deployed app so Streamlit Cloud doesn't sleep |

All pin Python 3.12 and check out with `lfs: false` (data is regenerated in the
run). Trigger any of them manually from the Actions tab.

---

## Configuration

Optional features (email, RSS) read environment variables. For local dev, copy
`.env.example` to `.env` (git-ignored) — `python-dotenv` loads it automatically.

| Variable | For |
|---|---|
| `EMAIL_FROM`, `EMAIL_TO`, `EMAIL_PASSWORD`, `SMTP_SERVER`, `SMTP_PORT` | email notifications (Gmail App Password) |
| `ALERTS_SITE_URL` | base URL for RSS per-alert links |

Never commit real secrets. On Streamlit Cloud use the platform's secrets
manager (`st.secrets`), not a `.env` file.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pip install` fails building numpy | Use Python 3.12/3.13 and the pinned `requirements.txt`; `nfl-data-py` must be `==0.3.2` (0.3.3 pins `numpy<2`, which has no modern wheel). |
| `ModuleNotFoundError: bs4` on the Player Props page | `pip install beautifulsoup4` (it's in `requirements.txt`). |
| `SyntaxError` on startup | You're on Python 3.11. Use 3.12 or 3.13. |
| `Missing file: nfl_play_by_play_historical.csv.gz` | `git lfs pull`. The app still runs without it, with reduced Historical Data features. |
| `KeyError: Features not in index` | Feature lists drifted between training and the app — rerun `python nfl-gather-data.py`. |
| Historical Data page shows garbage / no columns | The PBP file is **tab-separated**; every `read_csv` of it needs `sep='\t'`. |
| Dashboard won't load / port in use | `streamlit run predictions.py --server.port 8502` (check for stragglers: `tasklist | findstr streamlit`). |
| Betting sections show 2020 games | Fixed — each section reloads fresh data and filters `gameday > today`. Rerun with latest code. |
| Slow load / high memory | Expected ~1.5 GB and 10–30 s on first load; use the Historical Data filters, and `@st.cache_data` handles re-runs. |

---

**Built with** Python · Streamlit · XGBoost · LightGBM · scikit-learn · nflverse data.
