# Changelog

History moved out of `README.md`. Newest first. Dates are as recorded in the
original notes; undated dashboard/infra work from late 2025 is grouped at the
bottom.

---

## August 2026

- **Pipeline reproducibility.** Seeded every XGBoost/LightGBM estimator with
  `RANDOM_STATE=42` and `n_jobs=1`, and sort the feature lists on load so
  `best_features_spread.txt` (rewritten each run by the Monte-Carlo step) is a
  fixed point. `python nfl-gather-data.py` now byte-reproduces its own artifacts.
- **Temporal train/test split** replaces the random one — test games are now the
  last 20% by date, so reported metrics are out-of-time. `nfl-gather-data.py`
  body wrapped in `main()` + `__main__` guard.
- **Moneyline model disabled.** On the temporal hold-out it had no edge
  (AUC ≈ 0.56, worse-calibrated than the base rate, negative backtest ROI; its
  "edges" were anti-predictive). `prob_underdogWon` now ships the market implied
  probability; `ev_moneyline` ≈ 0 for every game so no moneyline bets are
  generated. `model_moneyline` is still trained for the reported diagnostics.
  The Underdog Bets tab carries a note explaining this.
- Betting-simulation prints restricted to the held-out test set (were scoring
  training games). String columns dropped from the model `features` list (were
  always ignored). New `tests.yml` CI workflow; `pytest.ini` scopes collection
  to `tests/`.
- Dependencies pinned (`requirements.txt` + `requirements-dev.txt`);
  `beautifulsoup4` added. Correctness fixes: `isWindy` uses wind not temp; PBP
  files read as tab-separated; season-year logic centralised in `season_utils.py`.
- README trimmed 795 → ~200 lines; this CHANGELOG and
  `docs/SPREAD_MODEL_INVESTIGATION.md` added.

## April 2026

- Added `lightgbm` to `requirements.txt`; XGBoost + LightGBM soft-voting
  ensembles for both game-level and player-prop models.
- Added `player_props/train_models.py` - dedicated player-prop training pipeline
  (aggregation, rolling features, matchup prep).
- Nightly workflow now also retrains player-prop models and uploads
  `player_props/models/model_metrics.json`.
- Added `.github/workflows/weekly-model-performance.yml` - weekly backtests,
  accuracy reports persisted to `data_files/accuracy_results_*.json`.
- Added `docs/LSTM_TRANSFORMER_ROADMAP.md` (off-season planning).
- Removed an unsupported `st.switch_page()` call in `pages/1_Historical_Data.py`.

## December 29, 2025 - Emailing predictions

Automated HTML email notifications with clear, actionable recommendations:
readable bet lines ("**TEN +2.5** to cover (69.1%)"), per-bet confidence tier
badges, full bet names, threshold filtering (Spread >=50%, Moneyline >=28%,
Totals >=50%), team colour markers. Setup via `EMAIL_FROM` / `EMAIL_TO` /
`EMAIL_PASSWORD` / `SMTP_SERVER` / `SMTP_PORT`; preview with
`python scripts/preview_email.py`, send with
`python scripts/send_rich_email_now.py`. Uses SMTP via `emailer.py`
(Gmail App Passwords).

## December 13, 2025 - Critical model fix & new features

- **Spread prediction inversion fix.** A mislabeled training target made the
  spread model's confidence run backwards. Corrected with
  `prob_underdogCovered = 1 - prob_underdogCovered` right after prediction in
  `nfl-gather-data.py`. Reported impact: betting ROI -90% -> +60%, 62/63
  remaining games flagged profitable, max confidence -> 89.5%, calibration
  error 45% -> 28%. (See `docs/SPREAD_MODEL_INVESTIGATION.md` for a later
  analysis of what this fix actually did and what is still fragile.)
- **18 new leak-free features:** momentum (8), rest-advantage (5),
  weather-impact (3). See `docs/NEW_FEATURES_DEC13.md`.
- UI: EV explanation expander, spread bets sorted date-ascending, unicode/icon
  fixes, PDF/CSV export UX.
- Docs: `docs/MODEL_FIX_PLAN.md`.

## December 11, 2025

- **Per-game detail page** at `?game=<game_id>` - matchup summary, model
  predictions, shareable link, lazy loading (no full PBP load).
- Underdog labelling in the per-game header (spread-first, moneyline fallback).
- Schedule/table links use path-relative `?game=` params with `target="_self"`
  for subpath-deployment compatibility.
- Schedule -> prediction matching tightened to prefer the same season.
- Sidebar download buttons render from placeholders and populate once data is
  loaded.
- Away/home QB names in the per-game header; full team names before logos;
  `00:00:00` gameday times hidden.
- **Export downloads / sidebar:** always-visible sidebar controls for
  Predictions CSV, Betting Log, and on-demand Predictions PDF, with embedded
  `csv_icon.png` / `pdf_icon.png` (fallback `favicon.ico`). Buttons render
  after data finishes loading.

## November 26, 2025

- Per-game UI polish: left-aligned metrics, re-aligned spread/total and
  probability groups under the `@` marker.
- Team names ~30px bold with responsive CSS; extra spacing on QB lines;
  `.team-name` / `.team-qb` classes + mobile media query.
- Per-game page no longer loads the large PBP dataset or the betting-log CSV
  during initial render.
- Betting-log table + per-game CSV download removed from the per-game view
  (the Performance dashboard still uses the central betting log).
- Fixed a `NameError` from UI columns being used before creation.

## November 2025 - Major performance breakthrough

- Spread model "fixed" (inverted predictions corrected) - reported 3.6% ->
  91.9% win rate on the selective high-confidence subset (~33% of games).
- Both spread and moneyline betting reported profitable.
- Framed as data-leakage-free with "strict temporal boundaries" (note: rolling
  *features* are leak-free; the train/test split is still random).

## October 2025

- **Data-leakage elimination (critical).** Historical stats had been computed
  over all-time data (including future games) during training. Switched to
  strict "prior games only" rolling stats. Accuracy dropped to a realistic
  56-64% but reported ROI rose from 27.8% to 60.9%.
- **Optimal XGBoost params.** 300 estimators, lr 0.05, depth 6, L1/L2
  regularization; lighter params (100 estimators, depth 4) for Monte Carlo
  feature selection.
- **Monte Carlo feature selection.** 8- -> 15-feature subsets, 100 -> 200
  iterations.
- **Dashboard:** "Next 10 Underdog Bets" section with real payout math;
  "Favored" column; corrected favorite/underdog identification from
  `spread_line` sign.
- Threshold documentation corrected to the actual F1-optimized value (28%
  after the leakage fix).
- Streamlit compatibility: removed deprecated `use_container_width` /
  `width='stretch'` usages.
- Date-filtering bug fix: betting sections were showing 2020 games because
  `predictions_df` was mutated by earlier sections; each section now reloads
  fresh data and filters `gameday > today`.
- **Git LFS** for `nfl_play_by_play_historical.csv.gz`.
- Feature engineering: current-season form, prior-season records,
  head-to-head history (all leak-free).
- Reliability: synchronized feature lists between `nfl-gather-data.py` and
  `predictions.py`; Monte Carlo samples only available numeric features;
  graceful fallbacks for missing features/data.

## 2025 - Dashboard & infrastructure (undated notes)

- **Three-model system:** added over/under (totals) predictions alongside
  spread and moneyline, with F1-optimized thresholds, value-edge analysis,
  confidence tiers, and an "Over/Under Bets" tab (top 15 by value edge).
- **Multi-page app:** dedicated Historical Data page for the ~290k play-by-play
  records; 12+ filter controls; quick presets (Red Zone, 3rd & Short, Pass
  Attempts Only); pagination 50-500 rows; session-state filter reset.
- **In-app notifications:** `st.toast()` alerts for Elite (>=65%) and Strong
  (60-65%) bets, deduplicated via `st.session_state`; per-alert pages at
  `?alert=<guid>`; detected public base URL persisted to
  `data_files/app_config.json`.
- **RSS feed:** `scripts/generate_rss.py` -> `data_files/alerts_feed.xml`,
  using `app_base_url` from `app_config.json` or `ALERTS_SITE_URL`; sidebar
  "Rebuild RSS" button.
- **Bankroll management tab:** bankroll input, risk tolerance
  (Conservative 1% / Moderate 2% / Aggressive 3% / Very Aggressive 5%),
  Kelly-inspired position sizing for elite bets, exposure tracking.
- **Model Performance tab:** total bets, win rate, ROI, units won; breakdown
  by confidence tier; weekly line charts. Reads
  `betting_recommendations_log.csv`.
- **Memory optimization for Streamlit Cloud:** `float32` / `Int8` dtypes,
  DataFrame views instead of `.copy()`, `@st.cache_data` lazy loading,
  pagination, `.streamlit/config.toml` with raised message-size limits.
- **Loading progress indicators**, **cache-management UI**
  (`st.cache_data.clear()`), compact header layout, `smoke_test.py`.
- Bug fixes: `pred_totalsProb` -> `prob_overHit`; added `moneyline_bet_return`;
  nested-tab indentation errors; column-existence guards before dataframe
  access.
