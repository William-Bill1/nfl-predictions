# Changelog

History moved out of `README.md`. Newest first. Dates are as recorded in the
original notes; undated dashboard/infra work from late 2025 is grouped at the
bottom.

---

## September 2026

- **Removed dead `st.experimental_rerun()` + deprecation cleanup.**
  `st.experimental_rerun()` was removed from Streamlit in 1.37; the app pins
  1.62, so all 8 call sites (7 in `predictions.py`, 1 in
  `pages/1_Historical_Data.py`) were an `AttributeError` waiting on a button
  press — swapped to `st.rerun()`. Replaced `pd.Timedelta(days=7)` /
  `pd.Timedelta(hours=12)` with `datetime.timedelta` (the pandas form raises a
  numpy "generic unit" `DeprecationWarning` with numpy 2.x).

- **CI pipeline smoke test.** New `pipeline-smoke` job in `tests.yml` runs
  `python nfl-gather-data.py` against the committed
  `nfl_games_historical.csv` (no network), then `scripts/check_pipeline_outputs.py`
  sanity-checks the artifacts (required columns, probabilities in [0,1], a
  non-empty + non-degenerate signal set, the `Spread_EV_Analysis` /
  `Spread_OOS_Test` keys), and finally asserts the run is **deterministic** —
  a second run must byte-reproduce `nfl_games_historical_with_predictions.csv`,
  `model_metrics.json` and `best_features_spread.txt`. `pytest -q` never
  exercised the batch pipeline.

- **Pre-season readiness sweep.**
  - `nfl-gather-data.py` now masks to `_played` games for training / the temporal
    split / metrics / season-long team rates. The unplayed schedule (272 rows
    once the season is set) had been landing in the test set and tanking every
    metric (spread acc 0.55 → 0.40); the nightly had committed polluted
    artifacts. Upcoming games still get probabilities written.
  - "🔄 Generate Predictions" button + `?run_pipeline` trigger now run
    `sys.executable` with a UTF-8 env (was bare `python` → wrong interpreter
    under a venv-launched app → `ModuleNotFoundError`).
  - `scripts/export_best_bets.py` reads `nfl_games_historical_with_predictions.csv`
    directly (was reading a log only the running app writes → the nightly feed
    was empty all season).
  - Betting Performance tab no longer `UnboundLocalError`s when moneyline/totals
    produce zero bets; `betting_recommendations_log.csv` truncated to header for
    a clean 2026 start; Spread Bets tab filters `spread_line != 0`.
  - Removed the disabled **Underdog Bets** and **Over/Under Bets** tabs (9 → 7).
  - `nfl_schedule_2026.csv` populated (272 games).

- **Results tracking actually works now (`betting_log.py`).** New headless module
  owns `betting_recommendations_log.csv`: `append_recommendations` logs spread
  signals for games in the next ~10 days (so each week's recorded edge reflects
  that week's model), `grade_pending` fills `actual_*_score` / `bet_result` /
  `bet_profit` from the `underdogCovered` / `spreadPush` labels once a game has a
  real (non 0-0) final score and its date is past. `predictions.py`'s
  `log_betting_recommendations` and `update_completed_games` are now thin
  delegators — the latter was dead code (a `continue` made the grading block
  unreachable and it only ever handled moneyline). The nightly workflow runs
  `python betting_log.py` after the pipeline, so the Model Performance tab and
  weekly backtest get data without anyone opening the app.

- **Honest spread backtest — threshold and evaluation are now separate
  slices.** `nfl-gather-data.py` moved from a 2-way temporal split to
  `temporal_split_3way` (60% train / 20% validation / 20% test). The EV
  threshold and the moneyline/totals F1 thresholds are fitted on the
  **validation** slice; Spread Accuracy/MAE and the betting simulation are
  reported on the **test** slice the tuning never touched. `model_metrics.json`
  now carries `Spread_EV_Analysis` (validation) *and* `Spread_OOS_Test`
  (test). The result: the spread edge is **~break-even out-of-sample** — 141
  bets, 76–65, 53.9% accuracy, +2.9% ROI (breakeven 52.4%), vs −5.2% on the
  validation slice it fits and vs the +25%+ the old same-slice split implied.
  Raw directional accuracy at a 0.5 cutoff is 48.2% on the test slice. The
  Model Performance tab shows this as an `st.warning` ("treat spread bets as
  roughly break-even, not a proven edge"). Models retrain on 60% now, so all
  shipped probabilities / feature importances regenerated; pipeline still
  byte-reproduces (`best_features_spread.txt` converged).

- **Spread confidence tiers recalibrated + copy sweep.** New cutoffs
  (`SPREAD_TIER_CUTS`) Elite ≥0.65 / Strong 0.59–0.65 / Good 0.55–0.59 / Lean
  0.50–0.55, anchored to the real `prob_underdogCovered` signal distribution
  (median ≈0.57) instead of round numbers. The old Good/Lean split (0.52/0.50)
  covered almost no live bets — the EV threshold means signals rarely sit below
  ~0.545. `betting_log._spread_tier` and `emailer.py` now mirror the same cuts
  (they had drifted apart: 0.60/0.55/0.52 vs 0.65/0.60/0.55). Spread-tab tier
  box changed from a green "PERFORMANCE BY CONFIDENCE LEVEL … Expected 60%+ win
  rate" `st.success` to a neutral `st.info` that says these are model
  probabilities, not promised win rates (out-of-time AUC ~0.58).

- **Prop roster filter (opt-in, `PROP_ROSTER_FILTER=1`).** `predict.py` can now
  drop players who are no longer on an NFL roster before it picks "recent
  starters", via `nfl_data_py.import_seasonal_rosters`. Left **off by default**:
  the pre-season nflverse roster feed is unreliable this early (players listed
  on the wrong team, veterans like DeAndre Hopkins / Tyler Lockett missing
  entirely), so an always-on filter would cut real Week 1 starters. The
  plumbing (`load_active_roster`, `get_recent_starters(roster_ids=...)`, a
  size-sanity guard, tests) is ready for when the real rosters publish. Week 1
  2026 snapshot left as-is (unfiltered).

- **Player-prop honesty pass.** `player_props/models.py` now holds out the most
  recent season (temporal split) instead of a random one, so `model_metrics.csv`
  is out-of-time. Each record gains `base_rate` / `roc_auc` / `reliable`; only
  ~5/26 models clear the bar (AUC ≥ 0.58 **and** accuracy above the majority
  base rate) — the skewed-line tiers that used to report 65-75% "accuracy" were
  mostly just predicting the majority class. All TD props are force-flagged
  `reliable = False` (every tier collapses to the same 0.5 line; ~coin-flip
  out-of-time). `predict.py` carries the flag through as a `model_reliable`
  column; the Player Props / Parlay pages mark unreliable rows "display only"
  and no longer show a "Defense Rank" column. `opponent_def_rank` deleted
  end-to-end — its aggregator averaged a stat over the whole dataset (leaked
  future games) and then clipped to a constant `1` for every row, so it was
  pure noise. Dropped two dead model files (`passing_tds_high/over.json`) and
  git-ignored the `_lgbm.txt` sidecars (inference only uses the XGB `.json`).

- **Player-prop weekly snapshots.** `player_props/predict.py` now targets a
  season/week (`--season` / `--week`, default: next upcoming week of the current
  schedule) instead of the hard-coded 2025 file, and writes a write-once frozen
  snapshot `player_props_predictions_week{W}_{season}.csv` alongside the latest
  feed. `backtest.py` prefers that frozen file, so the weekly accuracy check is a
  genuine prospective test instead of scoring the current (possibly
  hindsight-retrained) predictions. Added `--no-injuries` / `--no-weather`
  (the ESPN scrape and per-player Open-Meteo lookups are slow/flaky); the nightly
  runs with both off. Week 1 2026 frozen.
  Caveat unchanged: prop lines are fixed tiers, not market lines, so the
  ~65-70% weekly "accuracy" measures line placement, not betting edge, and the
  confidence distribution skews high.

## August 2026

- **Pipeline reproducibility.** Seeded every XGBoost/LightGBM estimator with
  `RANDOM_STATE=42` and `n_jobs=1`, and sort the feature lists on load so
  `best_features_spread.txt` (rewritten each run by the Monte-Carlo step) is a
  fixed point. `python nfl-gather-data.py` now byte-reproduces its own artifacts.
- **Temporal train/test split** replaces the random one — test games are now the
  last 20% by date, so reported metrics are out-of-time. `nfl-gather-data.py`
  body wrapped in `main()` + `__main__` guard.
- **Spread model: one convention.** `model_spread` predicts P(favorite covers);
  `nfl-gather-data.py` now takes the complement once
  (`prob_underdogCovered = 1 - that`) with an honest comment instead of a
  "predictions are backwards" narrative. The EV threshold, `Spread Accuracy`,
  `Spread MAE` and `predictedSpreadCovered` are all in P(underdog covers) space
  now. New `spreadPush` column; a push is no longer counted as an underdog
  cover and is refunded (return 0) in the backtest, not scored as a loss. Model
  training is unchanged (byte-identical feature importances). See
  `docs/SPREAD_MODEL_INVESTIGATION.md` (now marked resolved).

- **Moneyline and totals models disabled.** On the temporal hold-out neither
  has an out-of-time edge — moneyline AUC ≈ 0.56 (its "edges" anti-predictive,
  −4% backtest ROI); totals AUC ≈ 0.50 (a coin flip, −5% backtest ROI). Both are
  worse-calibrated than their base rates. `prob_underdogWon` / `prob_overHit`
  now ship the **market implied** probabilities, `pred_*_optimal` are forced to
  0, so no moneyline or totals bets are generated. Both models are still trained
  for the diagnostics on the Model Performance page; `model_metrics.json` carries
  a note for each. The Underdog Bets and Over/Under Bets tabs explain this.
  **Only the spread model currently drives a bet signal.**
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
