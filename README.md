# 🏈 NFL Betting Analytics & Predictions Dashboard

<p align="left">
  <img src="data_files/gridiron-oracle-transparent.png" alt="NFL Predictions Logo" width="260" />
</p>

A multi-page Streamlit app that trains machine-learning models on historical NFL
data (2020–present) for game outcomes — currently a (roughly break-even) **spread** betting signal,
plus market-implied moneyline and over/under views — and a **player-props**
system (passing / rushing / receiving yards; TD props kept but flagged
unreliable) for DraftKings Pick 6-style markets. Everything is batch-computed: scripts write
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

To run the test suite: `pip install -r requirements-dev.txt && pytest -q`.

---

## What it does

### Dashboard `predictions.py` — 7 tabs

The dashboard opens with a standing reality-check banner: the spread signal is
~break-even out-of-sample, and season-to-date record/ROI once bets settle.

| Tab | Contents |
|---|---|
| Model Predictions | model vs. actual results for completed games |
| Probabilities & Edges | upcoming-game probabilities (spread / market-implied moneyline & totals) and edges (model % − implied %) |
| Betting Performance | win-rate / ROI on the spread signal, out-of-sample test slice only |
| Spread Bets | confidence-tiered upcoming spread picks (the only live bet signal) |
| Betting Log | spread recommendations + graded outcomes (`betting_recommendations_log.csv`) |
| Model Performance | accuracy, calibration, the out-of-sample spread warning, weekly tracking from the betting log |
| Bankroll Management | Kelly-style position sizing for the higher-confidence spread bets |

The Underdog Bets and Over/Under Bets tabs were removed — the moneyline and
totals models are disabled (see below); their probabilities still show, as
market-implied columns, on the Probabilities & Edges tab.

### `pages/`

| Page | Contents |
|---|---|
| `1_Historical_Data.py` | filter-driven browser over ~290k play-by-play rows (2020–present); 12+ filters, quick presets, pagination |
| `2_Player_Props.py` | per-player prop predictions (defaults to a "tested (reliable) models only" view — see below) + interactive **DK Pick 6 calculator** (enter a line → OVER/UNDER + confidence tier; ML model or Laplace-smoothed historical fallback) |
| `3_Parlay_Builder.py` | combine bets, compute parlay odds |
| `4_Model_Performance.py` | model evaluation and calibration metrics |

### Models

Three binary classifiers in `nfl-gather-data.py`, each a
`CalibratedClassifierCV(XGBClassifier, isotonic)` optionally soft-voted with a
LightGBM twin. Every estimator is seeded (`RANDOM_STATE=42`) and single-threaded
(`n_jobs=1`) and the feature lists are `sorted()`, so `python nfl-gather-data.py`
**byte-reproduces its own artifacts**. The split is a **three-way temporal**
split (`temporal_split_3way`, ordered by season+week): earliest ~60% trains the
models, the next ~20% is a **validation** slice that fits the EV / F1 betting
thresholds, the last ~20% is a **test** slice used only for the reported
metrics. The current season's unplayed schedule flows through for prediction but
is never trained, tuned, or scored on.

| Target | Predicts | Ships a bet signal? |
|---|---|---|
| `spreadCovered` | favorite covers the spread | **yes** — `prob_underdogCovered = 1 − P(favorite covers)`, EV-based threshold |
| `underdogWon` | underdog wins outright | **no** — `prob_underdogWon` ships the market implied probability |
| `overHit` | total goes over | **no** — `prob_overHit` ships the market implied P(over) |

All three still train (for the diagnostics on the Model Performance page); only
the spread model drives a bet.

**Honest out-of-sample result** — EV threshold fitted on the validation slice,
scored on the untouched test slice (`model_metrics.json` → `Spread_OOS_Test` /
`Spread_EV_Analysis`):

| | out-of-sample | verdict |
|---|---|---|
| Spread | 141 test bets, 76–65, **53.9%** correct, **+2.9% ROI** (breakeven 52.4%); −5.2% on the validation slice; raw directional accuracy 48.2% | ships, but **~break-even** — treat picks as market context, not a proven edge |
| Moneyline | AUC ~0.56, "edges" anti-predictive, backtest −4% ROI | **disabled** — ships market-implied prob |
| Totals | AUC ~0.50, coin flip, backtest −5% ROI | **disabled** — ships market-implied P(over) |

`model_spread` is trained on "favorite covers"; the underdog probability is the
complement, computed once. This is a change of convention, not a fix for a
"backwards" model — see
[`docs/SPREAD_MODEL_INVESTIGATION.md`](docs/SPREAD_MODEL_INVESTIGATION.md)
(resolved; also logs a rejected QB-feature experiment).

**Player-prop models** (`player_props/`) also use a temporal hold-out (most
recent season). Each metrics record carries `base_rate` / `roc_auc` /
`reliable`; only ~5 of 26 clear an out-of-time bar (AUC ≥ 0.58 **and** accuracy
above the majority base rate). Every TD prop is force-flagged unreliable (all
tiers collapse to the same 0.5 line). `predict.py` writes a `model_reliable`
column and the Player Props page defaults to showing only the reliable subset.
Prop lines are still fixed tiers (275, 250, …), not market lines — the weekly
hit rate measures line placement, not betting edge.

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
| **nflverse** (`nfl_data_py`) | schedules, play-by-play, final scores | local, no key — completed-game scores come from the regenerated predictions CSV, not a runtime call |
| **Open-Meteo** | player-prop weather adjustments | `player_props/weather.py` (nightly runs `--no-weather`) |
| **ESPN** injury page | player-prop injury adjustments | scraped in `player_props/injuries.py` (nightly runs `--no-injuries`) |

All artifacts live in `data_files/` and are committed. The big one,
`nfl_play_by_play_historical.csv.gz` (~116 MB, **tab-separated**), is tracked
with **Git LFS** — run `git lfs pull` if it comes down as a pointer. The app
degrades gracefully if it's missing.

---

## Developer notes

### Run the full pipeline locally

```bash
python build_and_train_pipeline.py     # update_schedule → create-nfl-historical → nfl-gather-data
```

Steps individually:

```bash
python update_schedule.py              # refresh data_files/nfl_schedule_<year>.csv  (network)
python create-nfl-historical.py        # nflverse schedules → data_files/nfl_games_historical.csv  (network)
python nfl-gather-data.py              # feature engineering + train + predict  (~90s, no network)
python update_pbp_smart.py             # refresh the play-by-play LFS file (only downloads when stale)
```

### Retrain models

- **Game models:** `python nfl-gather-data.py` — reads the committed
  `nfl_games_historical.csv`, no network. Writes
  `nfl_games_historical_with_predictions.csv`, `model_metrics.json`,
  `model_feature_importances.csv`, `best_features_spread.txt`. Deterministic:
  running it twice produces byte-identical output.
- **Player-prop models:** `python player_props/train_models.py`
  (`--skip-aggregation` reuses cached per-player game logs). Writes
  `player_props/models/*.json` + `model_metrics.csv/json`.
- **Player-prop predictions:** `python player_props/predict.py`
  — predicts the next upcoming week of the current schedule and writes a
  **write-once** frozen snapshot `player_props_predictions_week{W}_{season}.csv`
  that `player_props/backtest.py` scores. Flags: `--week N` / `--season YYYY`,
  `--no-freeze`, and `--no-injuries` / `--no-weather` (the ESPN scrape and
  per-player Open-Meteo lookups are slow and network-fragile). Set env
  `PROP_ROSTER_FILTER=1` to drop players no longer on an NFL roster (opt-in —
  the pre-season roster feed is unreliable, so it is off by default).
- **Results tracking:** `python betting_log.py` — headless; appends the next
  ~10 days of spread signals to `betting_recommendations_log.csv` and grades any
  whose game now has a real final score. `python scripts/weekly_spread_report.py`
  rolls that log into `data_files/spread_performance.json` (season-to-date +
  per-week + per-tier record / profit / ROI). Both run in the nightly / weekly
  Actions, so the log fills without anyone opening the app.

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
| `nightly-update.yml` | 03:00 UTC, Sep–Feb | refresh PBP, run pipeline, retrain prop models, **freeze the week's prop snapshot**, `betting_log.py` (grade finished spread bets), `export_best_bets.py`, commit |
| `weekly-model-performance.yml` | Mondays 06:00 UTC, Sep–Feb | prop backtest, `betting_log.py`, `weekly_spread_report.py` → `spread_performance.json`; persist `accuracy_results_*.json` |
| `update-schedule.yml` | daily 06:00 UTC | refresh `nfl_schedule_<year>.csv` |
| `tests.yml` | on push / PR | `pytest -q` on Python 3.12 and 3.13, plus a `pipeline-smoke` job (runs `nfl-gather-data.py`, `check_pipeline_outputs.py`, asserts a 2nd run byte-reproduces) |
| `send_predictions_schedule.yml` | Wed evenings (in season) | email predictions |
| `rss_test.yml` | on push | regenerate + link-check `alerts_feed.xml` |
| `keep-alive.yml` | twice daily | ping the deployed app so Streamlit Cloud doesn't sleep |

Most pin Python 3.12 and check out with `lfs: false` (data is regenerated in the
run). Trigger any manually from the Actions tab. The nightly commits back to
`main` — `git fetch` before pushing local work.

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
| `model_metrics.json` shows terrible mid-season numbers | Fixed — `nfl-gather-data.py` now excludes the unplayed schedule (`_played` mask). Rerun with latest code. |
| "Generate Predictions" button errors with `ModuleNotFoundError` | Fixed — it now runs `sys.executable`, not bare `python`. Rerun with latest code. |
| No moneyline / totals bets anywhere | Expected — both models are disabled. Only the Spread Bets tab produces signals. |
| Player-prop predictions all "high confidence" | Expected — prop lines are fixed tiers, not market lines. The weekly hit rate measures line placement, not edge. The page defaults to the reliable subset; untick "tested models only" to see the rest. |
| Spread Bets banner says "roughly break-even" | Working as intended — the spread model has no demonstrated out-of-sample edge (`Spread_OOS_Test` in `model_metrics.json`). |
| Slow load / high memory | Expected ~1.5 GB and 10–30 s on first load; use the Historical Data filters, and `@st.cache_data` handles re-runs. |

---

**Built with** Python · Streamlit · XGBoost · LightGBM · scikit-learn · nflverse data.
