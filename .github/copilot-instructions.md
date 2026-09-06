# Copilot Instructions: NFL Predictions Project

## Overview
Multi-page Streamlit app for NFL betting analytics. XGBoost+LightGBM models for
spread / moneyline / totals plus a per-stat player-prop system. All data/models
are batch-computed into `data_files/`; the dashboard makes **no runtime API
calls** (final scores come from the regenerated predictions CSV). **Only the
spread model drives a bet signal** — and out-of-sample it is ~break-even
(`model_metrics.json` → `Spread_OOS_Test`); moneyline and totals show
market-implied probabilities (their models have no out-of-time edge and are kept
for diagnostics only).

## Architecture & Data Flow
**Data Pipeline (`python build_and_train_pipeline.py`, ~90s, deterministic)**:
1. `update_schedule.py` → `nfl_schedule_<year>.csv`
2. `create-nfl-historical.py` → `nfl_games_historical.csv` (all games incl. the unplayed schedule)
3. `nfl-gather-data.py` → features on all rows, **3-way temporal split** on **played** rows (`temporal_split_3way`, 60 train / 20 validation / 20 test), predict all rows → `nfl_games_historical_with_predictions.csv`, `model_metrics.json`, `best_features_spread.txt`
4. `betting_log.py` (nightly) → append/grade `betting_recommendations_log.csv`; `scripts/weekly_spread_report.py` (weekly) → `spread_performance.json`

**UI Layer**:
- `predictions.py` → Main dashboard, 7 tabs (Underdog Bets / Over-Under Bets removed with the disabled models)
- `pages/1_Historical_Data.py` → filtering over ~290k play-by-play records
- `pages/2_Player_Props.py` → player prop predictions (yards, TDs, receptions)
  - **New**: includes a `DK Pick 6 Calculator` tab for entering DraftKings Pick 6 over/under lines and receiving an OVER/UNDER recommendation. The calculator uses cached ensemble models located in `player_props/models` and falls back to a Laplace-smoothed historical hit rate when a model/tier is unavailable.
  - The player props system now supports XGBoost + LightGBM soft-voting ensembles and includes usage features like `target_share` to improve receiving predictions.
- `pages/3_Parlay_Builder.py` → Multi-bet parlay construction
- `pages/4_Model_Performance.py` → Model evaluation and calibration metrics
- All data loaded via `@st.cache_data` decorators (never at module level)

**Critical Constraints**:
- Features must be pre-game only (rolling stats exclude the current game).
- Every XGB/LGBM estimator gets `**_XGB_KW` / `**_LGBM_KW` (seed + `n_jobs=1`);
  keep `sorted()` on the feature lists. Otherwise the pipeline stops being
  byte-reproducible.
- Train/score on `_played` rows only; never on the unplayed schedule.

## Critical Conventions

### Memory Optimization (Streamlit Cloud)
```python
# Dtype pattern - apply to all DataFrames
df['numeric_col'] = df['numeric_col'].astype('float32')  # 50% memory reduction
df['boolean_col'] = df['boolean_col'].astype('Int8')     # vs float64
# Use DataFrame views, not .copy(), to avoid duplication
filtered_df = df[df['season'] == 2025]  # Good: creates view
```

### Lazy Data Loading (MANDATORY)
```python
# WRONG - causes silent crashes on Streamlit Cloud
predictions_df = pd.read_csv('data_files/predictions.csv')  # Module-level = BAD

# CORRECT - use caching
@st.cache_data
def load_predictions_csv():
    return pd.read_csv('data_files/predictions.csv', sep='\t')

# In function/page context only
predictions_df = load_predictions_csv()
```

### Spread model - one convention (Aug 2026)
`model_spread` is trained on `spreadCovered` = P(**favorite** covers). Everything
downstream works in P(**underdog** covers), so `nfl-gather-data.py` takes the
complement exactly once:
```python
prob_underdogCovered = 1.0 - _blend_proba(model_spread, lgbm_spread, X_spread)
```
Pushes get their own `spreadPush` column and are excluded from accuracy/ROI. The
EV threshold, Spread Accuracy/MAE and `predictedSpreadCovered` are all in
underdog-covers space. Do **not** re-invert anywhere else. (The old "predictions
are backwards / -90%→+60% ROI" framing was a variable mix-up, not a model bug.)

### Bet signals
- **Spread** is the only live signal. `pred_spreadCovered_optimal == 1` when
  `prob_underdogCovered >= optimal_spread_threshold` (EV-based, ~0.545, computed
  by `spread_ev_threshold` in P(underdog covers) space with pushes excluded) AND
  `ev_spread > 0`. Threshold is fitted on the **validation** slice of the
  three-way `temporal_split_3way` (60/20/20); the reported ROI/accuracy come
  from the **test** slice. `model_metrics.json` → `Spread_OOS_Test` is the
  honest number (~break-even on current data).
- **Moneyline / Totals**: `pred_underdogWon_optimal` and `pred_overHit_optimal`
  are hard-coded to 0. `prob_underdogWon` / `prob_overHit` are the market
  implied probabilities, not model output.
- Spread confidence tiers (`add_spread_confidence_tiers` / `SPREAD_TIER_CUTS`):
  Elite ≥0.65, Strong 0.59–0.65, Good 0.55–0.59, Lean 0.50–0.55 (anchored to the
  signal distribution; Lean is below the ~0.545 EV threshold). `betting_log`
  and `emailer.py` mirror these.

### UI Patterns
```python
# HTML Download Buttons with embedded icons
with open('data_files/pdf_icon.png', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode('ascii')
img_tag = f'<img src="data:image/png;base64,{img_b64}" style="width:36px;height:36px;...">'
html = f'<a download="{filename}" href="{data_uri}">{img_tag}<span>Download</span></a>'
st.markdown(html, unsafe_allow_html=True)

# Session state for notifications (avoid duplicates)
if 'notified_games' not in st.session_state:
    st.session_state.notified_games = set()
if game_id not in st.session_state.notified_games:
    st.toast("🔥 New elite bet!", icon="🔥")
    st.session_state.notified_games.add(game_id)

# Tab structure
tab1, tab2 = st.tabs(["Spread Bets", "Moneyline Bets"])
with tab1:
    if predictions_df is not None:
        st.dataframe(predictions_df, width='stretch')

# In-app pipeline trigger — the "🔄 Generate Predictions" button. Must use
# sys.executable (not bare "python" - that picks a different interpreter under a
# venv-launched Streamlit) and a UTF-8 env (emoji prints crash a cp1252 pipe).
result = subprocess.run(
    [sys.executable, "build_and_train_pipeline.py"],
    capture_output=True, text=True, timeout=1200,
    encoding="utf-8", errors="replace",
    env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
)
```


### PDF Export Pattern
```python
# Generate on-demand only (not at module load)
def generate_pdf_bytes(df_upcoming) -> bytes:
    buffer = BytesIO()
    # Use ReportLab, landscape letter, compact headers
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
    # Filter to upcoming games only, save to data_files/exports/
    return buffer.getvalue()
```

## Developer Workflow
- **Run app**: `streamlit run predictions.py`
- **Build & train**: `python build_and_train_pipeline.py` (~90s)
  - training only: `python nfl-gather-data.py` (no network; byte-reproducible)
  - retrain prop models: `python player_props/train_models.py --skip-aggregation`
  - prop predictions + weekly freeze: `python player_props/predict.py`
    (`--week N` / `--season YYYY` / `--no-injuries --no-weather` / `--no-freeze`;
    env `PROP_ROSTER_FILTER=1` to drop off-roster players, opt-in)
  - results tracking: `python betting_log.py` then `python scripts/weekly_spread_report.py`
- **Tests**: `pip install -r requirements-dev.txt && pytest -q` (only `tests/`;
  `pytest.ini` keeps `scripts/test_*.py` out).
- **Python**: 3.12 or 3.13. Not 3.11 (PEP 701 f-strings).
- **Windows**: run pipeline scripts with `PYTHONUTF8=1` (emoji prints).
- **Deployment**: Streamlit Cloud, all data files committed. The fork's nightly
  Action commits to `main` — `git fetch` before pushing.

### Developer Scripts & Checks
- When creating new helper or check scripts (for model diagnostics, calibration checks, or data validation), create them as Python files and place them in the `scripts/` folder (e.g., `scripts/check_moneyline_calibration.py`, `scripts/analyze_underdog_impact.py`).
- All new scripts must follow the project's lazy-loading and dtype guidelines and be import-safe (should not perform heavy data loads at module import time).
- Add a one-line description at the top of each script and include a simple `if __name__ == '__main__':` runner so they can be executed directly by CI or from the command line.


## Patterns & Examples
- **Adding features**: Update feature lists in both `nfl-gather-data.py` and `predictions.py`, validate no leakage, retrain models, update best features files
- **Adding tabs**:
  ```python
  with tab_name:
      st.write("### Section Title")
      if predictions_df is not None:
          st.dataframe(predictions_df[...])
      else:
          st.warning("Data not available")
  ```
- **DK Pick 6 Calculator**: `pages/2_🎯_Player_Props.py` now contains an interactive calculator where users can:
   - Search/select a player, choose stat category (auto-suggested by position), and enter the DraftKings Pick 6 line.
   - See both the ML-model probability (when available) and a historical hit-rate fallback (Laplace smoothing).
   - View the prediction source in the UI: `🤖 ML Model` (model chosen by season-average tier) or `📊 Historical` (fallback).
   - Models and feature logic live under `player_props/predict.py` and model files are JSONs in `player_props/models/`.

  Developer notes:
  - Models are loaded with `@st.cache_data` to avoid repeated heavy loads.
  - Feature extraction uses L3/L5/L10 rolling stats plus auxiliary features (TDs, attempts, completions, targets) and matchup defaults (`is_home`, `days_rest`). `opponent_def_rank` was removed (leaky + constant).
  - Retrain prop models with `python player_props/train_models.py`; generate/freeze predictions with `python player_props/predict.py`.
  - **Tier Selection Pattern**: Choose model tier based on season average performance (e.g., elite_qb for ≥280 passing yards).
  - **Fallback Pattern**: Use Laplace-smoothed historical hit rate (games_over + 1) / (total_games + 2) when model unavailable.
  - **Position-Based UI**: Auto-suggest stat categories based on player position (QB: Passing Yards/TDs, RB: Rushing/Receiving, WR/TE: Receiving).
- **Parlay Builder**: In `pages/3_🎲_Parlay_Builder.py`, combine bets from predictions_df, calculate parlay odds = product of individual probabilities
- **Betting logic**: spread confidence tiers (`SPREAD_TIER_CUTS`, mirrored by `betting_log._spread_tier` / `emailer.py`) — Elite ≥0.65, Strong 0.59-0.65, Good 0.55-0.59, Lean 0.50-0.55 (Lean sits below the ~0.545 EV threshold).
- **File Path Handling**: Use `from pathlib import Path; root = Path(__file__).parent.parent; sys.path.append(str(root))` for project-relative imports.
- **DataFrame Optimization**: Immediately convert dtypes after CSV loads: `df['float_col'].astype('float32')`, `df['int_col'].astype('Int32')` to reduce memory 50%.
- **UI Layout Patterns**: Use `col1, col2 = st.columns([2,1])` for asymmetric inputs, dynamic dataframe heights with `height=get_dataframe_height(df)`.
- **Feature Engineering**: Rolling stats exclude current game: `prior_games = df[(df['team']==team) & ((df['season']<season) | ((df['season']==season) & (df['week']<week)))]`.
- **Adding an estimator**: pass `**_XGB_KW` / `**_LGBM_KW` or it reintroduces
  run-to-run drift.
- **Season year**: `from season_utils import upcoming_or_current_season, latest_pbp_season` — don't compute it inline.
- **PBP file** (`nfl_play_by_play_historical.csv.gz`) is TAB-separated — every `read_csv` needs `sep='\t'`.

## Integration Points
- **External data**: All historical/play-by-play data is pre-fetched and stored in `data_files/`. The dashboard makes no runtime API calls; completed-game scores come from the regenerated predictions CSV (nflverse), graded by `betting_log.grade_pending`.
- **Feature importances/metrics**: Stored in `model_feature_importances.csv` and `model_metrics.json` (spread eval: `Spread_EV_Analysis` = validation slice, `Spread_OOS_Test` = held-out test slice).
- **Automated workflows**: `nightly-update.yml` (Sept-Feb) — smart PBP update, run the pipeline, retrain prop models, freeze the week's prop snapshot, `betting_log.py`, `export_best_bets.py`, commit. `weekly-model-performance.yml` (Mondays) — prop backtest + `betting_log.py` + `weekly_spread_report.py` → `spread_performance.json`. `tests.yml` — `pytest -q` on 3.12/3.13 plus a `pipeline-smoke` job (runs `nfl-gather-data.py`, `check_pipeline_outputs.py`, asserts a 2nd run byte-reproduces).
- **Email notifications**: `scripts/preview_email.py` / `scripts/send_rich_email_now.py`, SMTP via `emailer.py` (Gmail App Passwords). Spread bets only now.
- **RSS feed**: `scripts/generate_rss.py` → `alerts_feed.xml`, base URL from `app_config.json`.
- **best_bets_today.json**: `scripts/export_best_bets.py` reads the predictions
  CSV directly (`pred_spreadCovered_optimal == 1`, today's games) — independent
  of the app / `betting_recommendations_log.csv`.

## Known Issues / gotchas
- Module-level data loading → silent Cloud crashes; always `@st.cache_data`.
- PBP file is TAB-separated.
- `pytest` writes nothing to `data_files/` now (`pytest.ini` scopes to `tests/`).
  `pytest.ini` also sets `pythonpath = .` so a bare `pytest` collects (without it
  only `python -m pytest` worked — this failed CI for 10 days).
- `nfl-gather-data.py` has a hyphen → import via `runpy`/`importlib`, not `import`.
- `predictions.py` re-executes top-to-bottom on every `st.rerun()`, which resets
  module-level `historical_game_level_data`/`predictions_df = None` — non-button
  `st.rerun()` must be one-shot via `st.session_state`, or it loops forever.
- Prop lines are fixed tiers, not market lines, so prop "accuracy" ≠ edge. Only
  ~5/26 prop models are `reliable`; every TD prop is flagged unreliable.

## References
- `README.md` — setup + current model status
- `CHANGELOG.md` — dated history (Sep 2026: honest 3-way-split backtest,
  headless `betting_log.py` results tracking, weekly spread scorecard, spread
  tier recalibration, player-prop reliability flags, rerun-loop + CI fixes)
- `docs/architecture.md`, `docs/SPREAD_MODEL_INVESTIGATION.md` (resolved; carries
  a "Signal experiments log" of rejected features)
