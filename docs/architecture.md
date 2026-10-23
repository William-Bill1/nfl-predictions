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
             feature engineering (all rows) → 3-way temporal split on PLAYED rows
             (train ~60% / validation ~20% / test ~20%): models on train, EV+F1
             thresholds on validation, reported metrics on test
             → data_files/nfl_games_historical_with_predictions.csv   (all rows)
               data_files/model_metrics.json  (incl. Spread_EV_Analysis /
                                               Spread_OOS_Test)
               data_files/model_feature_importances.csv
               data_files/best_features_spread.txt  (fixed point)
               data_files/pipeline_run_manifest.json  (provenance of this run:
                 unique run_id, revision, config/feature ids, cutoffs, artifact
                 + schedule sha256 - schedule read once, hash of the parsed
                 bytes; written last, deleted at start; NOT a deterministic
                 artifact)

Pregame capture  (python pregame_snapshots.py capture - nightly, after step 1):
    manifest hashes verified against the predictions CSV + schedule
        → data_files/pregame_snapshots/<run_id>.json   (immutable, one per run,
          schema-validated + whole-payload checksum; eligible upcoming games
          with a usable kickoff after capture time; no outcomes; a failed
          capture fails the nightly run after data is published)

Results tracking  (python betting_log.py ; python scripts/weekly_spread_report.py):
    nfl_games_historical_with_predictions.csv
        → betting_log.py            → data_files/betting_recommendations_log.csv
                                      (append next ~10d of signals; grade finished)
        → weekly_spread_report.py   → data_files/spread_performance.json
                                      (season-to-date / per-week / per-tier ROI)

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
    pages/4_Model_Performance.py   [weeks with completed games only; defaults to the latest fully completed week]
    pages/5_Spread_Tracker.py      [opt-in; display-only, reads spread_tracker_report.json]
    pages/6_Value_Finder.py        [opt-in; display-only, book-vs-field + model-vs-book edges]
    pages/7_Ontario_Line_Timing.py [Wed-vs-Sun Ontario spread comparison built in memory from validated captures via ontario_line_timing.py; plus "Was waiting worthwhile?" via scripts/ontario_waiting_report.py; only write: manual FanDuel Ontario quote form]
    pages/8_Bet_Journal.py         [actual placed Ontario spread wagers via bet_journal.py; writes only data_files/bet_journal/ (git-ignored); grades on an explicit click]
```

## Determinism & mid-season
- Every XGB/LGBM estimator takes `RANDOM_STATE=42` + `n_jobs=1` (via `_XGB_KW` /
  `_LGBM_KW`); feature lists are `sorted()` on load and `best_features_spread.txt`
  is written sorted. `python nfl-gather-data.py` byte-reproduces its own outputs.
- `pipeline_run_manifest.json` holds the run's timestamp/run ID and is kept
  out of the determinism comparison; everything else above is byte-identical
  across runs (the manifest's non-volatile fields are identical too).
- The manifest hashes the schedule and predictions CSVs' exact bytes, so
  `.gitattributes` marks both `-text` (no `core.autocrlf` conversion on
  checkout). See "Line endings" in `docs/PREGAME_SNAPSHOTS.md`.
- `nfl-gather-data.py` computes a `_played` mask (rows with both final scores).
  Features and the probability write cover **all** rows so upcoming games get
  predictions; the train/test split, EV threshold, accuracy/MAE and season-long
  team-rate features use **played rows only**.
- Split is a **three-way** temporal split (`temporal_split_3way`), ordered by
  (season, week): earliest ~60% trains the models, next ~20% is the
  **validation** slice that fits the EV / F1 betting thresholds, last ~20% is
  the **test** slice for all reported metrics. The threshold is no longer picked
  on the same games it is scored on — `model_metrics.json` carries both
  `Spread_EV_Analysis` (validation) and `Spread_OOS_Test` (test). On current
  data the honest out-of-sample spread result is ~break-even (~+3% ROI over
  ~140 bets, 54% accuracy vs 52.4% breakeven), not the +25%+ the old 2-way
  split implied.

## ML Models
Three XGBoost classifiers (binary):
| Model | Target | Threshold | Notes |
|-------|--------|-----------|-------|
| Spread | trained on `spreadCovered` (favorite covers); ships `prob_underdogCovered = 1 - that` | EV-based | the only model that drives a bet signal |
| Moneyline | trained on `underdogWon`; **not shipped** — ships market implied prob (no out-of-time edge) | — | — |
| Totals | trained on `overHit`; **not shipped** — ships market implied P(over) (coin flip out-of-time) | — | — |

Spread confidence tiers (`add_spread_confidence_tiers`, `SPREAD_TIER_CUTS`):
Elite ≥0.65, Strong 0.59–0.65, Good 0.55–0.59, Lean 0.50–0.55. Anchored to the
actual `prob_underdogCovered` signal distribution (median ≈0.57, p90 ≈0.70).
The EV threshold (`spread_ev_threshold`, in P(underdog covers) space, pushes
excluded) is ~0.545, so the Lean band sits mostly below it — shown for context,
not a bet. `betting_log._spread_tier` and `emailer.py` mirror these cutoffs.

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

### Spread model audit: rolling backtest (Oct 2026)

`scripts/rolling_spread_backtest.py` is a read-only, week-by-week backtest of
the spread model with leak-free features. It never writes production files.
Use it before trusting any change to the spread model. Method, command and how
to read the output are in [ROLLING_SPREAD_BACKTEST.md](ROLLING_SPREAD_BACKTEST.md).

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

**Schedule dates.** `nfl_schedule_<season>.csv` `date` is nflverse `gameday`
as `YYYY-MM-DD`: a calendar date with no kickoff time. The Player Props page's
remaining-games banner (`player_props/season_schedule.py`) takes the season
from the displayed predictions, never the calendar year: every prediction must
record the same whole-number season (a missing value is an error, not
skipped), and every prediction's complete game date must be a recorded
calendar date (`YYYY-MM-DD`, or that date at midnight with no offset or UTC)
inside that season's window (1 August to the end of February, so January
playoff games belong to the previous year's season). It then loads only
`nfl_schedule_<season>.csv`, which must be readable and have
`update_schedule.py`'s columns (`week`, `date`, `home_team`, `away_team`,
`venue`, `status`). Every row needs a `YYYY-MM-DD` date inside the same window,
a whole-number week from 1 to 22, and two distinct 2-3 letter team
abbreviations, and no team may appear twice on one date (a duplicate game).
The page parses prediction game dates with `errors='coerce'`, so a bad stored
value shows the warning instead of breaking the page.
Otherwise the page shows a warning and omits the count, with no fallback to
another season; the predictions stay visible. The count covers the rest of the
season and uses `player_props/game_dates.py`: a calendar date counts
through that whole day in America/Toronto, and only an offset-aware
`commence_time` is treated as a kickoff instant (a zone-less time is never
assumed UTC). `predict.load_schedule` keeps its own rule: a game dated D stays
selectable until 12:00 UTC on D (08:00 EDT / 07:00 EST), so the nightly run the
evening before still targets that week, and the Monday-evening run moves on to
the next week. Its `game_date` column stores the calendar date as UTC midnight
(`2026-10-26 00:00:00+00:00`); use its date part and never convert it to local
time.

**Containment (October 2026).** A read-only audit of the committed artifacts
found that player-prop outputs rest on bad inputs, so they must not guide
wagers until the pipeline is repaired and the models retrained.

The audit found:
- **Stale stats.** `data_files/player_*_stats.csv` were last written on
  2026-01-07 by the January aggregator and end on 2026-01-04. The nightly job
  trains with `--skip-aggregation` and never rebuilds them, so 2026 props use
  no 2026 games.
- **Random `is_home`.** That January aggregator set `is_home` with
  `np.random.choice`, which matches the schedule on about 50% of rows. All 26
  committed models split on it.
- **Constant `days_rest`.** It is 7 in every training row and 1 at
  prediction time (`predict.calculate_days_rest` measures previous game to
  now). It has 0 splits in every committed model, so it has no effect today;
  it becomes a train/serve mismatch once the stats are rebuilt.

`player_props/containment.py` supplies the warning shown first on Player
Props, Parlay Builder and Model Performance, plus neutral probability bands
and "holdout check" labels in place of Elite/Strong/Good and "tested".
No probability, filter, stored output or historical figure changes; Parlay
Builder's "Edge" (always 0%) is no longer shown. The claims cover only the
committed models in `player_props/models/`. Models retrained during nightly
runs are not saved and are unverified. The game spread model is a separate
pipeline.

**Stage 1 feature generation (code done; not deployed).**
`aggregators.build_player_stats` builds all three stat tables, and
`write_player_stats` validates and writes them. Both the standalone
aggregator and `train_models.py`'s full-aggregation path now use them, so the
matchup features exist before anything is saved.
- **Column contract:** the output is exactly `output_columns(stat_type)`, the
  columns the stats CSVs already carry minus the removed `opponent_def_rank`.
  Receiving `target_share` is still computed but not written, so a retrain
  can't pick it up silently. A test checks that every committed model's
  features are in the contract.
- **`is_home`:** comes from the row's unique schedule game in
  `nfl_games_historical.csv`. The season and week must agree, the team must be
  exactly one side, and the opponent must be the other side. A missing,
  duplicate or conflicting schedule row, or an unreadable schedule file,
  raises `FeatureValidationError`; there is no default and nothing random.
- **Rolling L3/L5/L10:** an exponentially weighted mean of the player's
  appearances strictly before the game. Rows are ordered by calendar date,
  across seasons and teams. A player in two games on one date is a failure.
- **`days_rest`:** calendar days since the player's previous appearance *in
  that stat table*, capped at 21, with 7 for a first-ever appearance; history
  carries across seasons and teams. This measures appearance gaps, not
  physiological rest: a game with no row in that table (no targets, no
  carries) lengthens the gap. Prediction-time `days_rest` (always 1) is
  unchanged until the parity stage.
- **Duplicates:** a player's plays in one game always become one row. Name
  spellings that differ within a game are unified, and every play is summed.
  A player listed for two teams, opponents or dates in one game is a failure.
  Plays missing a grouping label are excluded and counted.
- **Ratios:** `completion_pct`, `ypc`, `ypr` and `catch_rate` are NaN rather
  than infinite on a zero denominator.
- **Validation before writing:** the exact columns, numeric types, no
  infinities, no missing identities or counts, unique player/game rows, valid
  dates, `is_home` of 0 or 1, and `days_rest` of 1-21. A rolling value may be
  missing only when the player has no earlier value of that stat. A
  validation failure writes nothing.
- **Writing the three files:** each table is written to a temporary file,
  and each existing destination is copied to a `.bak` file beside it. Then
  each file is replaced atomically with `os.replace`, one at a time.
  - If a replacement raises, the previous set is restored: replaced files
    get their backup back, and files that didn't exist before are removed.
    The original error is then re-raised.
  - If the restore fails too, `PersistenceRollbackError` names the files it
    couldn't restore and keeps every `.bak` file, which may be the only copy
    of the old data.
  - This covers errors Python can catch; it is **not** a transaction across
    the three files. A crash or power loss between two replacements can
    still leave a mixed set, with the `.bak` files left for manual recovery.
- **Offline rebuild (October 2026):** built from the local play-by-play file
  into a temporary directory. It passed validation, and `is_home` matched the
  schedule on 100% of rows (production: about 50%). Row counts were 4,181 /
  14,042 / 26,984: 27 player-games from 2026-01-04 that the January CSVs
  missed, plus 2 receiving duplicates merged.
- **Not current:** the local play-by-play file ends on 2026-01-04 and lacks
  the 2025 playoffs and every 2026 game (`pbp_metadata.json` overstates it as
  "2020-2026").

**These rebuilt stats must not replace the production CSVs yet.** The
committed models were trained on the January features (simple rolling means,
random `is_home`, constant `days_rest`), and the rebuilt rolling values differ
on roughly 84-88% of rows.

**Remaining stages (not implemented):**
2. **Source refresh and freshness checks.** Persist or refresh the
   play-by-play data the nightly job fetches (today it isn't committed and
   the metadata overstates coverage). Rebuild the stats nightly, and before
   predicting, warn or fail when the latest stats date is behind the target
   week; record that date and the checksums with each prediction.
3. **Train/serve parity.** One feature builder for training and prediction:
   - compute prediction-time `days_rest` from the upcoming game date and the
     player's latest appearance, or drop the feature;
   - match the rolling definition the models are trained on;
   - decide on `target_share` explicitly;
   - add a test that the prediction path reproduces the training row for past
     games.
4. **Model provenance.** Save or commit the models the nightly run actually
   uses, with a manifest (data checksums, feature list, code SHA, training
   window, metrics), and refuse to predict on a mismatch.
5. **Retraining.** Retrain on the rebuilt features only after stages 2-4.
6. **Leakage-safe evaluation.** Score the held-out season, comparing with and
   without `days_rest` and the real `is_home`, then require prospective frozen
   weeks before removing the containment notice. No accuracy improvement is
   claimed until then.

**Real market lines** (`player_props/market_odds.py`, live since `ODDS_API_KEY`
was added 2026-09-16): pulls DK/FanDuel lines for the four reliable prop
types from The Odds API and adds `market_line`/`market_book`/
`market_implied_prob`/`market_edge`/`market_line_available` columns
(additive — `line_value` unchanged). The Player Props page shows a "Market" /
"Market Edge" column and sorts by a *directional* edge (flipped to align with
the recommendation, since raw `market_edge` is always in P(over) terms — a
strongly negative value on an UNDER pick means a strong edge, not a weak one).
`market_edge` is only computed when the book's line equals the model's own
tier line (`line_value`) - otherwise the two probabilities are for different
lines and aren't comparable, so the row shows the book line but no edge.
Coverage is partial (only games DK/FanDuel have posted props for). Details +
rollout history in `docs/ODDS_API_INTEGRATION_PLAN.md`.

## Feature Engineering
- **Team aggregates** (38: win/close/blowout %, scoring, point diff, line and
  total averages, favored/cover/over/under/push-total %, last-3 form, point-diff
  trend): `team_features.compute_team_features`, shared by `nfl-gather-data.py`
  and `scripts/rolling_spread_backtest.py`.
  - **Availability rule:** only COMPLETED games (both scores present) from
    strictly EARLIER week keys (`season*100 + week`). The whole current week is
    excluded, because the schedule has no "result posted" timestamp and a
    kickoff time doesn't prove a game had finished at prediction time.
  - **Role split:** home stats use the team's earlier home games, and away
    stats its earlier away games. `PointDiffTrend` uses both.
  - **Order:** results don't depend on row order. Last-N is ordered by
    (week, kickoff, game_id).
  - **Cold start:** 0.0.
  - **History:** until Oct 2026, FavoredPct, SpreadCoveredPct and the
    over/under hit rates were full-history means that included each game's
    own result. The loop-based stats also counted unplayed earlier games as
    0-results for games 2+ weeks out.
- **Momentum** (8): Last 3 games win%, scoring, point differential
- **Rest** (5): Rest day differences, well-rested ≥10d / short-rest ≤6d flags
- **Weather** (3): Cold ≤32°F, windy ≥15mph, extreme conditions
- **Missing lines:** `_valid_line` (non-missing, non-zero `spread_line`) is
  captured before `fillna(0)`. Games without a valid line get NaN
  `prob_underdogCovered` / `ev_spread` / `edge_underdog_spread` and
  `pred_spreadCovered_optimal = 0`, and are never trained on. The CSV's
  `spread_line` is still written as 0 for those games (schema unchanged);
  read a missing probability as "no line".

## API Integrations
| Source | Purpose | Notes |
|--------|---------|-------|
| nfl_data_py | Schedule, play-by-play, final scores | Local, no key; final scores arrive via the regenerated predictions CSV |
| Open-Meteo | Player-prop weather adjustments | `player_props/weather.py` (nightly: `--no-weather`) |
| ESPN injuries JSON feed | Player-prop injury adjustments | `player_props/injuries.py` (on in the nightly since 2026-09-26); was an HTML scrape that silently returned 0 rows by 2026 |
| ESPN injuries JSON feed | Immutable injury-report captures at the Ontario slots (evidence only) | `injury_reports.py`, run as continue-on-error steps of `.github/workflows/ontario-spread-capture.yml` after the sportsbook steps; no key; never reads `espn_injuries.csv`; see `docs/INJURY_REPORT_ARCHIVE.md` |
| The Odds API | Real DK/FanDuel player-prop lines | `player_props/market_odds.py`; opt-in design (no-ops without `ODDS_API_KEY`), live in this fork since 2026-09-16 |
| The Odds API | Immutable Ontario sportsbook spread captures (Wed 12:00 / Sun 09:00 Toronto) + manual FanDuel Ontario quotes | `ontario_spreads.py` + `.github/workflows/ontario-spread-capture.yml`; opt-in (same `ODDS_API_KEY`), bulk endpoint by bookmaker key, 1 credit per capture; see `docs/ONTARIO_SPREAD_TRACKING.md` |
| The Odds API | Season-long US+CA sportsbook game-spread tracker vs. nflverse's line, a price-adjusted per-book value finder, and a model-vs-book line shopper | `spread_tracker.py` + `scripts/spread_tracker_report.py` + `scripts/spread_value_finder.py` + `scripts/model_line_shop.py` + `pages/5_Spread_Tracker.py` + `pages/6_Value_Finder.py`; opt-in (same `ODDS_API_KEY`), bulk endpoint, live since 2026-09-16; see `docs/MARKET_SPREAD_TRACKER_PLAN.md` |
| SMTP email | Bet notifications | `emailer.py`, Gmail App Passwords |

The dashboard makes **no runtime API calls**. `update_completed_games` is now a
thin delegator to `betting_log.grade_pending`, which grades against the
nflverse-sourced scores already in the predictions CSV — the old ESPN runtime
score fetch is gone.

## Key Components
- `build_and_train_pipeline.py` — `update_schedule` → `create-nfl-historical` → `nfl-gather-data`
- `nfl-gather-data.py` — feature engineering + 3-way temporal split + train + predict (all rows)
- `create-nfl-historical.py` — schedule + game fetch via nfl_data_py
- `season_utils.py` — `upcoming_or_current_season()` (schedules) / `latest_pbp_season()` (PBP); one source of truth for the season year
- `betting_log.py` — headless owner of `betting_recommendations_log.csv` (`append_recommendations`, `grade_pending`); `predictions.py` delegates to it. Its read-only display helpers (`recommendation_text`, `odds_text`, `RAW_SPREAD_LABEL`) are used by the Betting Log tab, the alert view and `scripts/generate_rss.py`. They show the team at its recorded `bet_spread` (or "Recommendation unavailable"), label `spread_line` as the raw source spread (home-favourite positive), and label -110 as assumed. `game_time_text` formats the alert header's game date: date-only values keep their calendar date, and only offset-aware timestamps are converted to America/Toronto. `spread_result` (win/loss/push for a team at its signed handicap) is shared with `bet_journal.py`
- `bet_journal.py` — actual-bet journal behind `pages/8_Bet_Journal.py`. It keeps append-only, checksummed records (wager / amendment / void / grade / invalidation) in the git-ignored `data_files/bet_journal/`. Two chains of explicit links order the history: a terms chain through `previous` and a settlement chain through `supersedes`. Writes happen under an exclusive lock, must name the current tip, and run duplicate and second-wager checks (for new wagers and amendments). Grading is explicit and audited: it needs conservative completion evidence from the schedule (consistent integer scores, not 0–0, game day before today), settles in Decimal CAD via `betting_log.spread_result`, and invalidates grades the evidence no longer supports. Current state and totals exclude voided, superseded and unverified settlements. A damaged or ambiguous history (forks, cycles, dangling links, duplicate voids, out-of-order timestamps, inconsistent grades) raises an integrity error that stops the page. `python bet_journal.py validate` checks a journal directory. It never reads or writes the recommendations log, captures or snapshots
- `player_props/train_models.py` — prop model training (temporal hold-out)
- `player_props/predict.py` — prop predictions + frozen weekly snapshot; `model_reliable` flag; opt-in `PROP_ROSTER_FILTER`
- `player_props/market_odds.py` — opt-in DK/FanDuel prop-odds fetch (`ODDS_API_KEY`); `market_odds_week{W}_{season}.csv` doubles as its own cache, filled in incrementally (cached games never re-fetched; uncovered games re-tried each run - free until props are posted)
- `spread_tracker.py` — opt-in season-long US+CA sportsbook game-spread tracker (`ODDS_API_KEY`, bulk endpoint); upserts `spread_tracker_log.csv` against nflverse's `spread_line`
- `ontario_spreads.py` — opt-in Ontario sportsbook spread captures (`capture`, `manual-quote`, `coverage`, `validate`); separate from `spread_tracker.py`, whose files it never touches. Owns the one expected-slot coverage calculation (`expected_coverage`: every Wed 12:00 / Sun 09:00 Toronto slot from `TRACKING_START` = 2026-10-07 through the next Sunday; pending / awaiting capture / missed / captured, with Phase 1's inclusive window rule `in_slot_window`), shared by the `coverage` CLI and the Ontario Line Timing page
- `frozen_performance.py` — display logic for `pages/9_Frozen_Performance.py`. It loads `scripts/frozen_spread_report.py`'s sources (`load_sources`, cached on a SHA-256 fingerprint of every snapshot, capture and the schedule) and evaluates them per render for today's outcome cutoff (`report_from_sources`). It shapes the report's own numbers into separate all-prediction / signal / market / simulated summaries, coverage rows and a per-game table, re-implementing none of its logic. Read-only; integrity errors stop the page
- `injury_reports.py` — evidence-only injury-report captures at the Ontario slots (`capture`, `list`, `validate`). Its workflow report (`persistence`) verifies this run's file against a fresh fetch of remote main by sha256. It stores ESPN's raw statuses with a lexical `status_class`, practice participation as unavailable (the feed has none), per-entry source entry dates as supplied (meaning unverified; never treated as update times or freshness), and the request and receipt times. The slot is judged at receipt, with `ontario_spreads`' slot rules. Per-team coverage (rows listed / listed without rows / absent) and outcomes (full / partial / empty / unavailable) are recorded. Files are created exclusively, checksummed and validated on read. No probabilities, scores or recommendations
- `ontario_line_timing.py` — display logic for `pages/7_Ontario_Line_Timing.py`: builds the Phase 2 report in memory (cached on a SHA-256 fingerprint of every capture/manual file), Toronto-time and signed labels, slot state (captured / pending / missed / empty) and default week; integrity errors stop the page
- `scripts/ontario_waiting_report.py` — read-only "Was waiting worthwhile?" analysis: Phase 2's eligible automated-Ontario Wednesday/Sunday pairs (same team and sportsbook, never re-paired), hindsight settlement at equal stakes (`betting_log.spread_result`, `bet_journal.win_profit`), and the latest pregame snapshot at or before each intended decision time (`pregame_snapshots.select_from`) with signal retained/withdrawn/added/side-changed classification; separate denominators; writes JSON/CSV only to the git-ignored `reports/ontario_waiting/`. Shown as a section of the Ontario Line Timing page (file inputs cached on a SHA-256 fingerprint of every snapshot and the schedule; results recomputed per render)
- `scripts/ontario_spread_report.py` — read-only Wednesday-vs-Sunday comparison of Ontario spread captures (per side: spread change, break-even change, key numbers 3/7, dominance vs trade-off); writes deterministic JSON/CSV to the git-ignored `reports/ontario_spreads/`; never modifies captures
- `scripts/frozen_spread_report.py` — read-only evaluation of the spread model on predictions frozen before kickoff (`docs/FROZEN_SPREAD_REPORT.md`):
  - **Selection:** one `pregame_snapshots.select_captures` observation per game (latest by default, chosen before its prediction is inspected, cross-checked against the current scheduled kickoff, no fallback or backfill).
  - **Outcomes:** the frozen underdog at its frozen handicap via `betting_log.spread_result` and `bet_journal.completion_evidence`.
  - **Coverage and metrics:** coverage for every status, Brier/log loss vs 50% with calibration by week and code revision.
  - **Market and returns:** market comparison only against automated Ontario-feed quotes (US reference and manual excluded) in archived captures at the exact handicap; signal-only simulated returns at archived vs assumed −110 prices.
  - **Scope:** `--season`, or by default the latest schedule season, independent of selections; coverage distinguishes no snapshot history from games before the first snapshot. `--as-of` is an outcome cutoff over current files, not a reconstruction.
  - **Output:** deterministic JSON/CSV to the git-ignored `reports/frozen_spread/`. It refuses output paths overlapping `data_files/`, never reads the bet journal, and corrupt inputs fail the run.
- `scripts/frozen_model_reliability.py` — read-only, evidence-only reliability view built on `frozen_spread_report` (no competing selection, settlement or market matching; `docs/FROZEN_MODEL_RELIABILITY.md`):
  - **Protocol v1** (fixed for subsequent evaluation; results on the games already evaluated are exploratory): the latest pregame snapshot, chosen first; pushes counted separately; all predictions and the signal subset separately.
  - **Samples:** overall (model vs 50%) and matched Ontario market (model vs market vs 50%, one game = one observation).
  - **Calibration:** fixed bins [0,.40) / [.40,.50) / [.50,.60) / [.60,.70) / [.70,1] with 95% Wilson intervals (binomial reference intervals that ignore dependence between games); sparse bins are descriptive only.
  - **Uncertainty:** paired per-game Brier/log-loss differences with a paired season/week cluster bootstrap (10,000 replicates, seed 20261005; unavailable below 5 weeks).
  - **Sensitivity:** an optional separate earliest-snapshot view.
  - **Output:** JSON/CSV/SVG to the git-ignored `reports/frozen_reliability/`, with source fingerprints. Protected output paths; no source writes, network or journal access.
- `scripts/spread_tracker_report.py` — rolls `spread_tracker_log.csv` up into `spread_tracker_report.json` (per-book ranking, best-line-per-game, anomaly flags)
- `scripts/spread_value_finder.py` — reads `spread_tracker_log.csv`; for one book, ranks sides by fair-value edge (normal approx of margin of victory vs. the book's own devigged price) — point-divergence alone isn't the same as a favorable price
- `scripts/model_line_shop.py` — reads `spread_tracker_log.csv` + `nfl_games_historical_with_predictions.csv`; extends the spread model's own probability (evaluated against nflverse's line only) to every tracked book's specific line, ranked by edge vs. the model instead of vs. the field
- `scripts/export_best_bets.py` — reads the predictions CSV (`pred_spreadCovered_optimal == 1`, today's games) → `best_bets_today.json`; independent of the app
- `scripts/weekly_spread_report.py` — `betting_recommendations_log.csv` → `spread_performance.json` rollup
- `scripts/run_weekly_backtest.py` — finds the most recently fully-completed NFL week (`gameday < today` and a real score for every game that week) and runs `player_props.backtest.run_weekly_accuracy_check` against it; replaces a broken inline CI snippet that called that function with no `week` argument
- `scripts/check_pipeline_outputs.py` — post-run sanity checks for the `pipeline-smoke` CI job
- `scripts/send_rich_email_now.py` — SMTP email sender
- `scripts/generate_rss.py` — `alerts_feed.xml` RSS feed

## Storage
All data in `data_files/` (committed to git):
- `nfl_games_historical_with_predictions.csv` — games (played + upcoming) + spread/market probabilities
- `model_metrics.json` (incl. `Spread_EV_Analysis` / `Spread_OOS_Test`), `model_feature_importances.csv`, `best_features_spread.txt` — model eval + selected features
- `player_props_predictions.csv` — latest prop feed; `player_props_predictions_week{W}_{season}.csv` — frozen weekly snapshots
- `market_odds_week{W}_{season}.csv` — opt-in DK/FanDuel prop lines (empty/absent unless `ODDS_API_KEY` is set); incremental per-game cache (a game is fetched once, once its props exist)
- `market_spreads_week{W}_{season}.csv` — opt-in raw US+CA sportsbook game-spread quotes (write-once cache); `spread_tracker_log.csv` — accumulating season-long spread-line-vs-nflverse comparison, upserted weekly by `spread_tracker.py`
- `spread_tracker_report.json` — season-to-date per-book ranking + best-line-per-game + anomaly flags, rolled up from `spread_tracker_log.csv` by `scripts/spread_tracker_report.py`
- `betting_recommendations_log.csv` — spread recs + graded outcomes; owned by `betting_log.py` (nightly / weekly), no longer the running app
- `pipeline_run_manifest.json` — provenance of the latest successful pipeline run (written by `nfl-gather-data.py`)
- `ontario_spreads/captures/<run_id>.json` — immutable Ontario spread captures, one per scheduled slot (quotes by bookmaker with jurisdiction, provider timestamps, coverage, model-snapshot link; no model probabilities); `ontario_spreads/manual/<quote_id>.json` — manual FanDuel Ontario quotes (source `manual`)
- `injury_reports/captures/<capture_id>.json` — immutable injury-report captures (ESPN feed, evidence only), at most one usable capture per scheduled slot; unusable (empty / unavailable) and ad-hoc captures are kept and labelled
- `bet_journal/<record_id>.json` — **not committed** (git-ignored, personal data): actual placed wagers and their amendments, voids, grades and invalidations, append-only (`bet_journal.py`; see `docs/BET_JOURNAL.md`)
- `pregame_snapshots/<run_id>.json` — immutable pregame spread predictions, one file per nightly run (`pregame_snapshots.py`; see `docs/PREGAME_SNAPSHOTS.md`); evaluated read-only by `scripts/frozen_spread_report.py`, whose output goes to the git-ignored `reports/frozen_spread/`, never `data_files/`
- `spread_performance.json` — season-to-date spread scorecard from `weekly_spread_report.py`
- `best_bets_today.json` — Sports Picks Grid feed
- `data_files/exports/` — PDF exports

## Memory Optimisation (Streamlit Cloud)
- All numeric columns → `float32` (50% reduction vs `float64`)
- Use DataFrame views not `.copy()`
- All data loading via `@st.cache_data` — NEVER at module level (causes silent crashes)
