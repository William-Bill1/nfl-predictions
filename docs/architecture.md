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

**Stage 2 freshness and provenance (code done; blocks nightly prop
predictions until the stats are rebuilt).**

*Source chain as found (October 2026):*
- **Nightly runner:** checks out without LFS, so
  `nfl_play_by_play_historical.csv.gz` is only a pointer.
  `update_pbp_smart.py` then combines the cached 2020-2024 file with a fresh
  download of the current season (2026), so **the runner's file has no 2025
  season**. Its committed `pbp_metadata.json` (262,769 plays) is consistent
  with that, but says "2020-2026" and records only the run time. The file
  itself is never committed (the commit step adds `data_files/*.csv` and
  `*.json`).
- **Nothing in the nightly job reads that play-by-play.**
  `train_models.py --skip-aggregation` trains from the committed stats CSVs,
  and the retrained models stay on the runner. `predict.py` uses those
  models and the same CSVs, and its predictions files are committed.
- **The committed stats CSVs** (January 2026 build, no manifest) end on
  2026-01-04 but miss BAL@PIT from that same day and all 13 games of the 2025
  playoffs, besides every 2026 game.
- **This checkout's play-by-play file** (Git LFS) is a different file:
  seasons 2020-2025, 292,670 plays, 1,680 games, last game 2026-01-04.

*What Stage 2 adds (`player_props/provenance.py`):*
- **Source observation:** `read_source` reads the play-by-play once and
  records the content's SHA-256, size, actual seasons, game counts and date
  range, plus `observed_at` (when it was read). It also records, per game,
  whether the `END GAME` play is present and its score
  (`backtest.pbp_game_completion`). File modification times and
  `pbp_metadata.json` are not used, and no generation time is claimed for a
  file.
- **Manifest:** `data_files/player_props_stats_manifest.json`, written by
  `write_player_stats(..., manifest_for=stats_manifest(source))` in the same
  replace-and-rollback set as the three CSVs. It records each file's hash,
  columns and actual coverage; the source observation; the aggregation code
  (`aggregators.py` hash, git commit and whether that file had uncommitted
  changes); `FEATURE_DEFINITION`; and the rolling windows. Both rebuild
  entry points write it. The existing CSVs have none, so they are
  unverified, and none is created for them.

*Freshness policy for target season S, week W.* The game schedule
(`nfl_games_historical.csv`, read-only) records results, but they are not
an authoritative final status, so they count only when corroborated (see
"Completed" below).
- **Cutoff:** the date of the target week's first game. The feature source
  is only stats rows from completed games with (season, week) before (S, W)
  *and* dated before the cutoff. Rows from the target week (for example a
  Thursday game already played) or later are removed and counted.
- **Expected games:** every scheduled game before the target week, in season
  S, season S-1 (including its playoffs) and every earlier season the stats
  contain.
- **Completed** (`provenance.completion_status`), checked in this order.
  Every condition must hold:
  - both scores are present, whole, non-negative and possible (a score of 1
    is impossible);
  - the scores are not 0-0, which is treated as a placeholder (unplayed rows
    are filled with 0 elsewhere);
  - the recorded `result` and `total` equal home - away and home + away;
  - the game is dated before today in Toronto;
  - the play-by-play the stats were built from has that game's `END GAME`
    play with the same score.

  These follow the reviewed conventions already in use: `total > 0` and
  `gameday < today` (betting log, weekly backtest), the journal's
  "consistent integer scores, not 0-0, game day before today", and the
  weekly backtest's `END GAME` rule. Without a valid manifest nothing can be
  corroborated, so no game counts as completed.
- **Game-level coverage:** each completed expected game must appear in all
  three stats tables, with rows for both teams. The latest date alone proves
  nothing; a recent maximum date can hide missing games.
- **Reported, never guessed** (each fails the check):
  - an earlier game dated before the cutoff that isn't completed, with the
    reason: no recorded score (postponed, in progress or not yet recorded),
    one score, invalid scores, 0-0, a missing or inconsistent result/total,
    dated today or later, or no matching `END GAME` play in the source;
  - an invalid date;
  - a duplicate `game_id`, or a team twice in one week;
  - no target-week games, or a prediction schedule that disagrees with the
    game schedule;
  - stats games missing from the schedule;
  - stats rows for a game without completion evidence.

  A game moved to on or after the cutoff is not expected, and this is noted.
- **Provenance:** the manifest must exist and parse, and its structure is
  checked before any value is used. This covers the source, its coverage
  and per-game `END GAME` evidence, the aggregation code, and each stats
  entry. It also covers 64-character hashes, time-zone-aware timestamps
  that aren't in the future (and source read before stats written), and
  coverage fields. A malformed manifest is a reported failure, never a
  traceback.

  Every stats file must match its recorded hash, size and coverage. The
  columns, feature definition and rolling windows must match the current
  code. A changed `aggregators.py`, a changed source file or a build from
  uncommitted code is reported as a note.
- **No substitution:** nothing falls back to another season, the latest
  feed or older stats. Week 1 is valid only with the whole previous season.

*Enforcement:*
- **Command:** `python player_props/provenance.py [--season S --week W]
  [--data-dir DIR]`. It is read-only, prints the diagnostics, and exits 0
  (valid), 1 (not established) or 2 (could not run). Without a season and
  week it uses the week `predict.py` would target.
- **Prediction gate:** `predict.generate_predictions` runs the same check for
  its target week before loading models, fetching odds or writing anything.
  On failure, including an error inside the check, it returns 1 (the
  script exits 1). Every predictions file and frozen snapshot is left
  byte-identical. On success the models see only the feature source.
- **Backtest guard:** the weekly backtest matches predictions on player name
  only. When a week had no frozen snapshot, it used to fall back to
  `player_props_predictions.csv` whatever week that held.
  - Now the file it picks (frozen snapshot or fallback) must have `week` and
    `season` columns, at least one row, and every row equal to the requested
    week and season. Anything else is refused.
  - A file name alone is not proof, and the committed snapshots for Weeks
    1-7 2026 all pass.

*Effect on the nightly job:*
- The "Freeze player-prop predictions" step fails (exit 1) with the
  diagnostics in its log. It has `continue-on-error`, so the job continues
  and the run is not marked failed.
- No new `player_props_predictions.csv` or weekly snapshot is written, so
  the existing files (Week 7 2026 and earlier) stay as committed and keep
  their containment notice.
- Model retraining still runs (unchanged), and so do the spread steps.
- Weekly backtests for Week 8 onward report "no predictions" instead of
  scoring Week 7's predictions.
- This holds until the stats are rebuilt from a complete source with a
  manifest (source refresh, below). Local verification against these files
  showed: no manifest, so no game's completion can be corroborated; 108
  games with recorded results missing for Week 8 (14 from 2025, 94 from
  2026); and Week 7's games not yet scored.

*Offline rebuild with manifest (local source, temporary directory):*
- **Passes:** 2025 Weeks 18 and 19, the latest weeks this source supports.
- **Fails 2026 Week 1:** the 13 playoff games are missing.
- **Fails 2026 Weeks 7 and 8:** 106 and 107 missing games.

*Limits:* this checks the stats, not the models: the models used at
prediction time are still unversioned (model provenance, below). The
manifest is replaced with the stats but is not crash-atomic, like the CSVs.

**Stage 3 train/serve parity (code done; generation stays blocked).**

*Mismatches found (October 2026).* Training used each stored stats row's own
features; prediction rebuilt them differently:
- **Rolling stats:** prediction reused the player's latest stored row, whose
  L3/L5/L10 exclude that latest game, so they were one game stale. On the
  2025 rows of a local rebuild, L3 differed from training on about 95%.
- **`days_rest`:** prediction used (latest game date - now) clamped to
  1-14, so it was always 1. Training uses the gap to the previous
  appearance, capped at 21.
- **`is_home`:** prediction took it from the prediction schedule's team
  order, with no unique-game check. Training uses the unique
  `nfl_games_historical.csv` game.
- **Player identity:** prediction matched history, tier and position by
  short name (and team). Short names aren't unique: 4/40/64 names are shared
  by more than one player ID in passing/rushing/receiving. The team filter
  also lost players after a trade (28/75/119 such 2025 rows).
- **Feature lists:** training built them by silently skipping absent
  columns (receiving `target_share` was skipped). Prediction swallowed any
  model-input error with "Prediction failed" and moved on.
- **Parsing:** pandas' default CSV float parser is not exactly round-trip,
  so stored features could differ from recomputed ones in the last digit.

*Design (`player_props/features.py`):*
- **One definition:** `MODEL_FEATURES` gives the exact, ordered inputs per
  prop type. They are the names the committed models were trained on, and
  `target_share` is deliberately excluded.
- **Training rows** are the Stage 1 stats rows (`features.training_rows`).
- **Prediction rows:** `features.inference_rows` appends the target game to
  the player's eligible history and runs the same Stage 1 functions:
  `calculate_rolling_averages` (exponential weighting of strictly earlier
  appearances, by date then `game_id`), `derive_days_rest` and
  `derive_is_home`. A stored historical game rebuilt this way equals its
  training row exactly.
- **Eligible history** for a target game is the player's appearances in
  that stat table, by player ID, on every team and season, dated strictly
  before the target game. For prediction this is Stage 2's feature source:
  completed games before the target week, dated before its first game. A
  result already recorded in the target week (a Thursday game) never enters
  that week's features, and history containing the target game itself is
  refused.
- **History is checked before any feature is computed**
  (`features.history_problems`), and failures raise:
  - a repeated player-game row (it would be averaged twice);
  - rows of one game disagreeing on season, week or date, or not naming the
    same two teams as team and opponent;
  - a player in two games on one date;
  - missing identity or date values, or missing columns.

  Nothing is deduplicated or summed. Appearances on other teams and in other
  seasons are valid and kept.
- **`days_rest`:** calendar days from the previous appearance in that table
  to the target game, capped at 21. A first-ever appearance gets 7.
- **`is_home`:** the target game must resolve to exactly one schedule game
  (`features.resolve_game`), and the team must be one side of it. Missing,
  duplicate or conflicting evidence raises.
- **No history:** a player-game with no earlier appearance in the table has
  nothing to average. It is excluded from training and from prediction alike,
  and counted. A row with only some rolling values breaks the Stage 1
  contract and raises.
- **Input checks:** `features.model_matrix` builds the exact model input.
  It rejects missing or duplicated columns, non-numeric, boolean or
  non-finite values, `is_home` outside 0/1, and `days_rest` outside 1-21.
  Prediction feature errors stop the whole run before anything is written.
- **Player identity:** stable `player_id` for history, tiers and position
  (`players.csv` `gsis_id`). A recent player whose latest appearance was for
  another team is excluded and reported: the stats can't show his current
  team, and a trade after his last game isn't visible without rosters.
- **Parsing:** stats are read with round-trip float parsing
  (`aggregators.read_stats_csv`) on both paths.

*Compatibility boundary (the flat `model_metrics.json` records below are superseded by
Stage 4 bundles; the contract itself is unchanged):*
- **The contract:** `FEATURE_CONTRACT` = `FEATURE_SCHEMA_VERSION`
  (`props-features-1`) plus Stage 1's `FEATURE_DEFINITION`.
- **Training** records the contract, the exact feature list and the stats
  manifest hash in `model_metrics.json`. It does so only when the stats it
  trains on have a valid Stage 2 manifest; otherwise it records none.
- **Prediction** loads a model only when all three of these hold:
  - its recorded contract matches;
  - its recorded feature list is an explicit list of unique names in exactly
    `MODEL_FEATURES` order (a missing, null, non-list, repeated or reordered
    list is refused);
  - the booster's own feature names agree.

  Correct booster names alone are not enough. Otherwise it refuses the model
  and reports why, and with no usable model it exits 1 before writing.
- **The committed models** were trained before Stage 3 on the January
  features, so all of them are refused. No contract is created for them,
  and generation stays blocked (Stage 2 blocks it first).

*Offline reconciliation:* a local rebuild was compared week by week, using
the Stage 2 cutoff for each week's history.
- 42,924 player-game rows across all three tables were rebuilt through
  `inference_rows`: 612,606 values (rolling stats, `is_home`, `days_rest`),
  with 0 differing.
- 2,283 first appearances were excluded on both paths.
- The local source has no player with two games in one week, so the Stage 2
  week cutoff and "strictly before the game" give the same history there.

*Effect on nightly retraining.* This is also a training change.
`train_models.py --skip-aggregation` → `models.train_all_models` now:
- **Builds training rows explicitly** from `MODEL_FEATURES` via
  `features.training_rows`, instead of dropping NaN rows and silently
  skipping absent columns. On the current CSVs every prop type trains on
  exactly the same rows as before (verified read-only: 3,888 / 13,184 /
  25,827).
- **Reads the stats with exact float parsing.** Some feature values differ
  from the old parse by at most about 6e-14, so a retrained model may differ
  marginally where a split threshold falls in between. Not measured: no
  retrain was run.
- **Fails instead of skipping** when a feature column is missing, a row has
  only some rolling values, or a value is non-finite or out of range. The
  step has `continue-on-error`, so the run continues with whatever models
  are present.
- **Records `features` and `feature_contract`** in `model_metrics.json`. On
  the current stats (no manifest) the contract is null and a warning is
  printed, so the retrained models are refused at prediction. Generation is
  already blocked by Stage 2.

*Remaining incompatibilities (not features):*
- **Ensemble:** training metrics are for an XGBoost + LightGBM average, but
  prediction uses XGBoost only, and no LightGBM files are committed.
- **Tier and line:** tier selection (raw last-8 average) and the
  forced-OVER/UNDER clamps happen only at prediction.
- **Adjustments:** injury and weather adjustments are applied after the
  model.
- **No accuracy claim:** parity makes the inputs consistent; it is not
  evidence that predictions are better.

**Stage 4 model provenance and consistent scoring (code done; generation
stays blocked).**

*Findings (October 2026):*
- **Ensemble vs served scorer:** training fitted XGBoost and, when
  installed, LightGBM, and reported metrics (and the `reliable` flag) for
  their average. Prediction loads only the XGBoost file, and the LightGBM
  files are gitignored and were never committed. So the holdout labels
  described a scorer that never served a prediction.
- **How far off:** rescoring the committed XGBoost models on the 2025 rows
  of the committed stats (read-only) moves AUC by at most 0.009, and none
  of the 26 `reliable` flags changes. This is inspection on flawed January
  data, not validation.
- **Two `PROP_LINES` copies:** prediction's copy differed from training's
  for 8 tiers. Rushing and receiving yards (3 tiers each) and the elite TD
  tiers showed a line other than the one the model was trained on. About 1
  in 12 rows in every stored snapshot (53-74 per file) carry a probability
  for a different line than the one shown.
- **Silent fallbacks:** `receiving_yards_elite_wr` and `receptions_elite_wr`
  have no model file. `load_models` gave those names the `star_wr` model,
  under the elite name and line.
- **Persistence:** training wrote each model file, then the metrics, one at
  a time into the committed `player_props/models/`, with no manifest. Nightly
  retraining overwrote them on the runner.
- **Page calculator:** the Player Props "DraftKings Pick 6" calculator
  loaded the model files unverified, built its own features (simple means,
  `is_home=1`, `days_rest=7`), and reported the model's probability for its
  tier line as if it were the line entered. Failures fell back silently to
  a historical rate.

*Scorer choice: XGBoost only.* Production prediction has only ever loaded
XGBoost. Nothing showed a reason to serve the ensemble, and serving it would
mean committing and verifying a second artifact per model. LightGBM is no
longer trained for props; the spread pipeline is unchanged.

*Design (`player_props/model_bundle.py`):*
- **One scorer:** `score_probabilities` is used in training evaluation and
  in prediction. It requires the exact ordered feature matrix and a
  `binary:logistic` XGBoost classifier with classes [0, 1] trained on those
  names. It returns P(class 1) = P(stat > the model's line), and rejects
  any output that isn't a finite (n, 2) probability summing to 1. Nothing is
  clipped or substituted.
- **Training (`models.train_prop_model`):** fits, serialises, reloads and
  scores the reloaded model, so the metrics describe the exact stored bytes.
  The protocol is a temporal hold-out: train on seasons before H, evaluate
  on season H, raw model probability, OVER at P >= 0.5.
- **Bundle:** `player_props/models/bundles/<bundle_id>/` holds the model
  files, `model_metrics.json` and `manifest.json`. The manifest records:
  - the scorer and its version, and the Stage 3 feature contract;
  - per model: file hash and size, prop, line type and line value, target
    (stat > line), ordered features, training and evaluation windows
    (seasons, rows, positives, dates), the evaluation protocol, and metrics
    computed for that scorer (accuracy, AUC, precision, recall, F1, and the
    exact-probability Brier score and log loss);
  - the training source: the stats manifest hash, stats file hashes, the
    feature definition, the last game date and the play-by-play hash;
  - the code (file hashes, git commit, uncommitted-changes flag) and the
    library versions;
  - `built_at`, the build time, separate from the data windows.

  `player_props/models/current_bundle.json` names the bundle in use and its
  manifest hash.
- **Publication (`publish_bundle`):** the manifest is validated, everything
  is written to `bundles/.<id>.partial` and verified from disk exactly as
  prediction will load it, the directory is renamed, and then the pointer is
  switched with one `os.replace`.
  - Until then the previous bundle stays current. A catchable failure
    removes the unfinished directory; if that fails, the error names what
    was left.
  - Bundles are immutable and old ones are kept.
  - **Not handled:** a crash or power loss mid-publish (the pointer still
    names the old bundle, and a leftover directory is ignored), durability
    without fsync, and concurrent publishers (no lock: the last pointer
    switch wins, and each bundle stays self-consistent).
- **Loading (`load_bundle`):** verifies, before any model is used:
  - the pointer, and the manifest hash, structure, version and scorer;
  - the feature contract and the xgboost version;
  - every model file's hash, name, prop and line identity, feature list,
    booster feature names and objective;
  - the metrics file's hash and its agreement with the manifest;
  - training windows not overlapping evaluation, and evaluation within the
    training source.

  Anything wrong raises, and prediction exits 1 before writing.
- **Existing models:** flat files have no bundle and are refused, the
  committed ones included. No retrospective manifest was created.
- **Training vs feature source:** these are recorded separately. Training
  needs stats with a valid Stage 2 manifest and the current feature
  definition, but not the latest feature source. Prediction requires the
  bundle's training data to end before the target week's first game, so a
  model trained on that week or later is refused. A maximum model age is
  not enforced yet; that belongs to evaluation (stage 6).
- **No fallbacks:** each bundle model has its own file, metrics and
  identity. A tier without a model gets no prediction.
- **One `PROP_LINES`:** the training lines, in `features.py`. Prediction
  shows each model's own line.

*Post-processing inventory (unchanged behaviour, now recorded):*
- **Tier selection:** by the raw last-8 average.
- **Clamps:** a P floor of 0.75 for elite/star RB/WR averaging over 1.3x the
  line, and a P cap of 0.25 when L5 is under 0.6x the line.
- **Confidence filter:** the 0.55 MIN_CONFIDENCE filter.
- **Injuries:** Out is dropped; Questionable and Doubtful scale P by 0.8 and
  0.7.
- **Weather:** wind and cold penalties.
- **Market:** the market edge is computed from the final P.
- **Recorded in outputs:** `player_id` (the stable ID that was scored; see
  "Stable player identity" below), `raw_model_prob_over` (the scorer's output,
  untouched by every later step), `prob_over` (after adjustments),
  `adjustments`, `model_name` and `model_bundle`.
  - `adjustments` lists, in order, each step that changed the probability:
    `clamp_star_floor_0.75` or `clamp_low_average_cap_0.25`, then
    `injury_questionable`, `injury_doubtful` or `injury_limited_practice`
    (by the injury helper's own precedence), then `weather`. It is `none`
    when nothing changed it.
  - The injury helper scales the prediction in place, so the probability is
    captured before it runs. An injury entry that changes nothing is not
    listed. `model_reliable` and the holdout labels
  describe the raw model only, never the adjusted probability.
- **Page calculator:** now shows only the historical hit rate. It can't
  build contract features without a target game, and no model is trained
  for an arbitrary entered line.

*Effect on nightly retraining.* `train_models.py --skip-aggregation` now
verifies the stats first. Today they have no manifest, so it trains nothing,
writes nothing, and exits 1, and the step shows as failed (it has
`continue-on-error`). It no longer overwrites `player_props/models/*.json`
on the runner. With verified stats it would publish
`player_props/models/bundles/<id>/` and `current_bundle.json` on the runner.
The commit step adds only `data_files/`, so the bundle would not be
committed; workflow uploads are unchanged. Prediction stays blocked by
Stage 2 first, then by the missing bundle.

**Stage 5A complete-source refresh and bundle storage (mechanisms only; not
deployed).** Nothing here runs on a schedule. No live data was fetched, no
production file or model was changed, and generation stays blocked.

*What was found:*
- **No 2025 on the runner:** the nightly runner checks out without LFS, so
  `update_pbp_smart.py` builds its file from a cached 2020-2024 file plus the
  current season only.
- **Metadata isn't evidence:** `pbp_metadata.json` records the requested
  seasons and the run time, not the content.
- **Shared contract:** both are part of the spread pipeline's contract, so
  this stage adds a separate prop-specific path rather than changing them.
- **Bundles weren't durable:** they lived only on the runner's disk, with no
  lock between publishers.

*Source refresh (`player_props/source_refresh.py`):*
- **Explicit seasons:** every season from `FIRST_SEASON` through the newest
  season with play-by-play, with no gaps (`required_seasons`).
- **Validated per season:** each season is downloaded (the fetcher is
  injectable; tests are offline) and checked before use. The checks:
  - Git LFS pointers are refused, plain or gzipped;
  - empty, truncated or unreadable data, and missing required columns;
  - season labels and `game_id` prefixes must match the season;
  - dates must be valid, one per game, and inside the season;
  - duplicate (`game_id`, `play_id`) plays;
  - **the schedule itself** must be well formed (fail closed): required
    values present, valid `game_id`s for the season, valid dates, no
    duplicate `game_id`, and no team in two games in one week;
  - **each downloaded game** must match its unique schedule row's date,
    season and week, and no game may be absent from the schedule;
  - **game-level coverage**: every completed game in the schedule, by the
    Stage 2 completion predicate, must be present with an `END GAME` play
    matching the recorded score;
  - **unfinished vs malformed**: a game counts as unfinished only when no
    score is recorded or it is dated today or later. Malformed evidence
    (one score, 0-0, an impossible score, a missing or inconsistent
    result/total, an invalid date) fails the refresh and is never treated as
    an unfinished game.
- **Staged install:** a complete candidate is written to a staging directory
  (tab-separated gzip with a fixed gzip timestamp, so equal content hashes
  equal). It is re-read with `provenance.read_source`, checked to cover
  exactly the required seasons, and installed with its manifest as one set:
  backed-up replace, with rollback on a catchable failure. A backup leaves
  the recovery list only after its restore succeeds. If a restore fails, the
  error names each destination left wrong and the backup still holding its
  previous content (or says it didn't exist before), and those backups are
  kept.
- **Failure:** any problem leaves the previous source and manifest untouched
  and lists every problem, including the missing coverage.
- **Lock:** a refresh lock (`ontario_spreads.SlotLock`) excludes concurrent
  refreshes of the same directory.
- **Manifest** (`player_props_source_manifest.json`):
  - the required seasons;
  - each download's SHA-256 and size;
  - the installed file's SHA-256, size and actual coverage;
  - per-season coverage counts, including games present without completion
    evidence;
  - the schedule file's hash;
  - `refreshed_at`, distinct from the data's coverage.
- **Output location:** `--out-dir` is required and nothing else is written.
  The shared `nfl_play_by_play_historical.csv.gz` and `pbp_metadata.json` are
  never read or written.
- **Score corrections:** every refresh replaces each season wholesale, so a
  correction arrives with the next successful refresh. If the play-by-play's
  `END GAME` score and the schedule's recorded score disagree (one source
  corrected first), the refresh fails and the previous source stays until
  they agree.
- **Incomplete games:** a game present without completion evidence (in
  progress, or not yet scored) is kept but listed. Stage 2 excludes it from
  features until it is corroborated. Unplayed games aren't required.

*Bundle storage (`player_props/bundle_store.py`, `model_bundle.py`):*
- **Export:** one verified bundle goes to a tar containing `export.json`
  (bundle id, manifest hash, and each file's SHA-256 and size) and
  `<bundle_id>/<file>.json`. Entries have fixed metadata, so the same bundle
  exports byte-identically, and the archive is written to a temporary file
  and renamed.
- **Import refuses, before writing anything:**
  - unreadable or truncated archives;
  - links, devices, directories and other non-regular entries;
  - absolute, traversing, backslash or drive-letter paths;
  - unexpected or duplicate names, and entries over the size caps;
  - index/entry mismatches, and hash, size or manifest-hash mismatches;
  - a bundle id already installed with different content.
- **Import then, under the publisher lock:** stages the files, verifies the
  staged bundle exactly as prediction loads it, renames it into place, and
  only then switches the pointer. Re-importing an identical bundle is a
  no-op. A failure removes the staged copy and leaves the previous current
  bundle in use.
- **Pointer switch (`model_bundle.switch_pointer`):** re-verifies the bundle
  from disk and replaces `current_bundle.json` with one `os.replace`.
- **Publisher coordination (`model_bundle.publisher_lock`):** the existing
  `ontario_spreads.SlotLock` (`.model-bundle.lock` in the models directory)
  covers publication, import and pointer switches.
  - The lock records its owner. A conflict names that owner and the lock's
    age, and the lock is never removed on age alone.
  - On exit it is removed only if it still carries the holder's token.
  - It coordinates processes sharing one filesystem, not separate runners.
- **Retention:**
  - Bundles are immutable, and nothing deletes them automatically.
  - `inventory` lists every bundle, the current one, whether each verifies,
    and leftover staging directories, and changes nothing.
  - The current bundle, and the only verified copy of any bundle, must never
    be removed. Pruning is a manual decision made from the inventory.

*Implemented vs operationally verified:* every mechanism above is
implemented and tested offline (`tests/test_player_props_stage5a.py`, with
synthetic data, injected downloads and no network). None has been exercised
against real nflverse downloads, a GitHub runner, or artifact storage.

*Later deployment procedure (stage 5B; not enabled):*
1. **Refresh the source** in the nightly job, before retraining, as its own
   step without `continue-on-error`, so a failure is visible:
   `python player_props/source_refresh.py --out-dir data_files/player_props_source`.
   It needs network access to nflverse (about 7 season downloads).
2. **Aggregate from the prop source** (a small code change in 5B: point
   `train_models.py`'s full aggregation at the prop source). Then train,
   which writes stats with a Stage 2 manifest and publishes a bundle.
3. **Persist the bundle durably.** Recommended: commit
   `player_props/models/bundles/<id>/` and `current_bundle.json` in the
   existing commit step (the job already has `contents: write`). A bundle is
   on the order of a few megabytes and is immutable.
   - As a secondary copy, upload the export as an artifact:
     `bundle_store.py export --out $RUNNER_TEMP/prop-bundle.tar`, then
     `actions/upload-artifact@v4` with `retention-days` set explicitly.
4. **Restore before predicting** on a fresh runner: either from the
   committed bundle (no action needed), or with `actions/download-artifact@v4`
   from the last successful run (needs `actions: read` and a `github-token`
   with `run-id`), then
   `python player_props/bundle_store.py import --models-dir player_props/models --archive prop-bundle.tar`.
5. **Serialise runs** with workflow
   `concurrency: {group: nightly-update, cancel-in-progress: false}`, since
   the file lock doesn't span runners.

*Corrections to this procedure (runner and storage audit, 2026-10-29;
nothing is implemented):*
- **Immutability must be established, not assumed.** A bundle directory is
  written once, but its stored copy is only as immutable as its storage.
  - Git history can be rewritten.
  - A GitHub Release's assets can be deleted or replaced, and its tag moved,
    unless immutable releases are enabled for the repository.
  - Before any Release copy counts as immutable, the plan must enable
    immutable releases and confirm the protection is in effect.
  - Separately, each asset's SHA-256 should be recorded in a committed
    pointer and every download verified against it, treating a mismatch as
    unusable. These hashes only detect a changed asset. They don't make a
    Release immutable, and they don't prevent an asset from being replaced
    or deleted.
  - "Immutable" is not claimed for any copy until that protection is in
    place and tested.
- **Publishing needs its own job.** Writing a bundle, release or pointer
  needs `contents: write`. That permission belongs to a separate publish job
  that only uploads already-verified outputs, not to the steps that refresh,
  train or run tests. Those run with `contents: read` and
  `persist-credentials: false`. Today the whole nightly job has
  `contents: write`, so step 3 as written would give every step that
  permission.
- **One definition of blocked and failed.** Today the provenance command
  exits 1 for "not established" and 2 for "could not run", but `predict.py`
  and `train_models.py` exit 1 in both cases. In the nightly job, both
  outcomes therefore look the same, and the run stays green. Every prop
  entry point should use one shared classification:
  - **blocked**: a check ran and deliberately refused, for example inputs
    not established, verification failed or a publication checkpoint
    tripped. Existing outputs are unchanged.
  - **failed**: something could not run, for example an exception, a
    missing dependency, an unreadable input or a crash mid-step.
  - **succeeded**: verified outputs were published.

  Workflows must report the three separately, and from the spread steps.

*Storage and retention limits:*
- **Artifacts** expire after their retention period (90 days by default;
  configurable per repository), so they can't be the only recovery copy.
- **The Actions cache** can be evicted at any time (unused entries after 7
  days; a per-repository size cap), so the play-by-play cache is a
  convenience. A restored cache must pass `verify_installed` or be
  re-downloaded.
- **The prop source** (a column subset, all seasons) is tens of megabytes:
  too large to commit nightly without LFS. Plan to re-download it, or cache
  it and verify it on restore.

**Stage 5B preparation: isolated end-to-end rehearsal (offline; not
deployed).** `python player_props/rehearsal.py --workspace DIR` runs the real
production functions in order on a synthetic league, with the network blocked:

1. **Fixture downloads** (`rehearsal_fixtures.py`): eight teams, five seasons,
   nflverse-style plays, and `END GAME` rows matching the schedule's scores.
2. **Source refresh** (`source_refresh.refresh_source`), run twice: a
   **training** source through the last training season, and a newer
   **inference** source through the weeks already played in the target season.
3. **Aggregation** (`build_player_stats`, `write_player_stats` with
   `stats_manifest`) for each source.
4. **Training** (`models.train_all_models`) with the production minimums and
   out-of-time evaluation unchanged. It publishes a verified bundle.
5. **Export** (`bundle_store.export_bundle`), then **restore**
   (`import_bundle`) into an empty runtime models directory. Prediction is
   pointed only there; the training folder can be deleted after restore.
6. **Input validation** (`provenance.check_inputs`), then **prediction**
   (`predict.generate_predictions`, now taking explicit `data_dir` /
   `models_dir`, restored after the call).
7. **Independent verification**, of the latest feed **and** the frozen
   snapshot, before anything is published:
   - each row is recomputed: the tier and model production would pick, the
     restored model's raw score, the one permitted clamp (star floor 0.75 or
     low-average cap 0.25, restated in `rehearsal.py` rather than imported),
     its exact label (`none` when unclamped; any other label is refused), and
     `prob_over`, `prob_under`, `confidence` and `recommendation`;
   - the line, model, tier, recent averages, season, week and this run's
     bundle ID must match; weather must read `Skipped` and every market field
     must be empty;
   - the snapshot must hold exactly the latest feed's rows and values;
   - every model's stored metrics equal a recomputation with the restored
     model on its evaluation rows.

   The verified files' hashes are recorded, and publishing refuses a file
   that changed afterwards.
8. **Publish** to `<workspace>/published/`. A week's frozen snapshot is never
   overwritten: a different one makes the run `conflict` and publishes
   nothing, while an identical one is accepted. `snapshots.json` records which
   run and bundle froze each snapshot.

*Guarantees:*
- **Paths:** everything lives under `<workspace>/runs/<run_id>/`. The
  workspace, and each directory a run creates, is resolved (symlinks and
  Windows junctions included) and refused if it is, contains or lies inside
  the repository, `data_files/`, `player_props/models/` or another protected
  path.
- **Training cutoff:** the training source must end strictly before the
  target week's first game, and prediction refuses a bundle trained on that
  week anyway (Stage 4).
- **Failure handling:** a failure stops every later stage. The run keeps its
  partial outputs, a `summary.json` with status `failed`, and
  `diagnostics/traceback.txt`; nothing is published. A retry is a new run.
  Writing the summary never replaces the stage's own error:
  - **Early failures:** a run that stops before `runtime/data` exists (for
    example invalid `train_seasons`) still saves `summary.json`.
  - **Reference hash:** if the runtime players reference is missing or can't
    be read, the hash is recorded as unavailable, with the reason. No other
    file is substituted.
  - **Unreadable files:** any other file that can't be hashed is listed under
    `artifacts_unreadable`.
  - **Diagnostics:** if the traceback can't be saved, the error records
    `traceback_not_saved`.
- **Summary:** `summary.json` (machine-readable) records:
  - stage statuses;
  - every artifact's SHA-256;
  - training and inference source coverage;
  - the feature cutoff and training end date;
  - per-model training and evaluation windows;
  - the bundle ID, feature contract and scorer;
  - skipped models and skipped external inputs;
  - the output and verification counts, and the limitations.

  It claims no accuracy or edge.

*What it does not cover:*
- **External inputs:** injury, weather and market-odds inputs are skipped, so
  only the model clamps can appear in `adjustments`, and their rules are not
  verified here.
- **Which rows exist:** now checked; see "Expected-row completeness" below.
- **Label order:** under these rules at most one clamp applies, so label order
  is enforced only as an exact string.
- **Synthetic only:** the data is synthetic, so real nflverse formats and
  volumes, real schedules, a GitHub runner, and artifact or cache storage are
  untested.
- **Real date at prediction:** Stage 2's completion check at prediction uses
  the real current date.
- **Training volume:** one full run takes about 20 seconds locally (26 of 28
  models train; 2 are skipped by the real imbalance rule).

*Deployment implications (nothing enabled):* the rehearsal is the rehearsal
of the 5B workflow order. Going live still needs:
1. Live verification of the source refresh against real nflverse downloads
   and the real schedule, including postseason, 0-0 and `gameday` vs
   `game_date` edge cases.
2. A decision on where the inference stats and the bundle persist between
   runs (see Stage 5A).
3. Workflow steps with `concurrency`, permissions and artifact retention.
4. A first live, supervised run with containment kept until stage 6
   evaluation.

**Real-data compatibility rehearsal (2026-10-25; isolated, not deployed).**
The same rehearsal now also runs on real nflverse data, in two separate steps:

1. **Download** (network, writes only under the workspace):
   `python player_props/rehearsal_real.py --workspace DIR --seasons 2020-2026`
   fetches each season's play-by-play with the source refresh's own
   `nflverse_fetch`, plus the nflverse schedule
   (`nflverse/nfldata/.../games.csv`).
   - **One directory per attempt:** each attempt writes to its own
     `downloads/<UTC time>-<token>/`, created exclusively, so a previous
     attempt is never reused or overwritten, even within the same second.
   - **Completion record:** `downloads.json` holds each file's URL, size,
     SHA-256 and the TLS trust source. It is written only after the attempt
     succeeds, so a failed attempt has none and can't be used. TLS is verified
   against the operating system's trust store (`truststore`), because a local
   security product here re-signs HTTPS with a root only Windows trusts.
2. **Rehearse** (offline, network blocked):
   `python player_props/rehearsal.py --workspace DIR --real-inputs DOWNLOADS
   --players PLAYERS_CSV --train-seasons 2020-2025 --target 2026:7`. Every
   downloaded file is re-verified against its recorded hash. The players
   reference isn't part of the source-refresh contract, so it's named
   explicitly (the committed `data_files/players.csv`) and its hash recorded.
   `summary.json` now records the inputs used.

*First run (real data, 2026 week 7):*
- **Source:** 2020-2026 validated by `refresh_source`, judged against
  today's date (2026-10-25).
  - **Downloaded:** 1,787 games and 311,540 plays, from 2020-09-10 to
    2026-10-22, postseason included.
  - **Eligible as completed:** every downloaded game (1,787), each with an
    `END GAME` score matching the schedule. Eligible means a recorded score
    corroborated by that score, dated before the judging day.
  - **2026:** 272 games scheduled, of which 94 were downloaded, all
    eligible. The other 178 were not yet completed, including 3 already
    scored on the judging day, which therefore count as unfinished.
  - **The cancelled 2022 BUF-CIN game** (week 17) is in neither the nflverse
    schedule nor the play-by-play: 2022 lists 271 regular-season and 13
    postseason games, and BUF and CIN 16 regular-season games each. It is
    never required as completed and contributes no evidence.
- **Stage 2:** provenance and freshness passed for 2026 week 7, with a
  feature cutoff of 2026-10-22. The 4 week-7 games already scored were
  excluded.
- **Training:** 26 models were trained on 2020-2024 and evaluated on 2025.
  Two were skipped by the imbalance rule. The bundle exported and restored
  byte-identically.
- **Prediction:** 750 rows for 14 games. All of them were verified, and the
  frozen snapshot matched.
- **Publishing:** run `20261025T201417Z-e7d13275` published its week-7
  snapshot. A later run on the final code (`20261025T202042Z-7afa9654`)
  verified its own 750 rows, then correctly refused to replace that
  snapshot, because it came from a different bundle. Nothing was
  overwritten.
- **Run cost:** about 4 minutes wall time, about 1,700 CPU-seconds and a
  peak working set of about 760 MB.

*Found and fixed:* real `players.csv` repeats display names (retired
namesakes), and the prediction output carried names, not player IDs. The
verifier's name-to-ID lookup therefore rebuilt 34 rows for the wrong player
and failed. Interim fix: resolve a shared name by team. Since superseded by
`player_id` in the outputs (below).

**Stable player identity (`player_id` in prediction outputs).**
- **Source of the ID:** each new prediction row carries `player_id`, the
  stats ID that selected its feature row and scored its model (from
  `predict.candidate_players`). It is written as text in the latest feed and
  in newly frozen snapshots, and is never derived from a name.
- **Rehearsal verifier** (fails closed, no name lookup). A row's own
  `player_id` must:
  - be present, and appear in the player reference exactly once (production
    uses only an ID's single record);
  - be one of the team's candidates under production's rule;
  - agree with the row's short name, display name, position and prop
    eligibility (`POSITION_MAP`).
- **Duplicates and file comparison:**
  - duplicates are found on `(player_id, team, opponent, prop_type,
    line_value)`, so different players sharing a name are not duplicates;
  - the latest feed and the snapshot must carry the same identities.
- **A file without a `player_id` column** is refused as unverifiable.
- **Compatibility:**
  - Existing production feeds and frozen snapshots are unchanged (none has
    `player_id`; they already vary between 23 and 29 columns).
  - They remain readable: the Player Props and Parlay Builder pages show
    named columns, and backtest grading joins on `player_name`, so an extra or
    absent `player_id` changes nothing for them.
  - IDs are not backfilled by name, and historical files are not treated as
    identity-verified.
  - New production snapshots would carry `player_id` once generation is
    unblocked (it stays blocked). A week's frozen snapshot is never rewritten,
    so older weeks keep their original schema.
- **Not changed:** probabilities, selection rules, the feature contract,
  bundle checks and the generation gates.
- **Limitations:**
  - Backtest grading still matches by `player_name`; using `player_id` there
    needs ID-keyed actual results.
  - Output completeness and 2026 rookie coverage remain separate, unverified
    tasks.

*Still not established:*
- **Output completeness:** now checked in the rehearsal, relative to its inputs
  and rules (see "Expected-row completeness" below).
- **Rookies:** the committed players reference has no 2026 rookies, so they
  get no position and no props. A candidate reference that adds them is built
  offline (see "Players-reference candidate builder" below) but not adopted.
- **Deployment readiness:** unverified. A GitHub runner, artifact or cache
  storage, the workflow steps, and injury, weather and market adjustments
  were not exercised.

**Expected-row completeness (rehearsal verifier).**
`player_props/rehearsal_expected.py` derives, from the inputs alone and before
either prediction file is read, the rows production should write for the
target week. Rows are identified by `(player_id, team, opponent, prop_type,
line_value)`. The latest feed and the frozen snapshot are then each compared
with that set separately, so a missing or extra prediction is caught even
when the two files agree.

*Inclusion rules,* as production applies them, with injury, weather and
market inputs disabled:
1. **Games:** the target week's rows of `nfl_schedule_<season>.csv`. Each
   must resolve to exactly one game in `nfl_games_historical.csv`, with its
   own date a valid `YYYY-MM-DD` equal to that game's calendar date. The two
   schedules must list the same games. A missing, malformed or different date
   fails; the authoritative date is never substituted.
2. **Candidates per team:** players appearing in any feature table in one of
   the team's 3 most recent games, whose latest appearance in any table (by
   date, then `game_id`) was for this team.
3. **Reference:** exactly one `players.csv` record for the ID, with an
   eligibility group (`position_group`) in `POSITION_MAP` (QB, RB, WR, TE).
   The group must be present, unpadded and consistent with the position label
   (see "Prop eligibility by position group" below), or the check fails.
4. **Props:** that group's props. The position label isn't used to select
   props.
5. **History:** an earlier appearance in the prop's stat table.
6. **Tier:** the mean of the last 8 appearances in that table (every team), or
   the lowest tier with fewer than 3.
7. **Model:** the first restored bundle model for (prop, tier).
8. **Confidence:** the final probability (after the permitted clamp) is at
   least 0.55 or at most 0.45.

*Independence:*
- **Restated, not called:** the rules above are restated in the module.
  Production's `candidate_players`, `get_player_performance_tier`,
  `get_player_position` and its inclusion loop are not used, and the expected
  set is never derived from either output.
- **Shared, and why:** the constants `POSITION_MAP`, `TIER_RULES` and
  `PROP_STAT`; the Stage 2-validated feature tables; and the feature builder
  and scorer (`inference_rows`, `model_matrix`, `score_probabilities`), which
  produce the probability the confidence band applies to. A slip in one of
  these shared pieces would not be caught here.

*Results:*
- **Failing rows:** a missing, unexpected or duplicate identity, or a wrong
  home/away side, fails verification. Nothing is published, and previously
  published files stay byte-identical.
- **Summary:** `summary.json` records `completeness`: expected and per-file
  observed, distinct, missing, unexpected, duplicate and wrong-side counts,
  plus the exclusions by rule. On a derivation failure it records the
  problems.
- **Status values:**
  - `complete`: both files were checked and have zero missing, unexpected,
    duplicate and wrong-side identities;
  - `not checked`: a file couldn't be checked (no `player_id` column);
  - `incomplete`: any other failure;
  - `expected set not derived`: the inputs were invalid or ambiguous.

  The status is decided before the row checks run, so a later failure
  can't leave it stale.
- **Existing checks:** every existing identity, probability, bundle,
  feed/snapshot and publication-hash check is unchanged.

*Exclusions:* the existing rules are unchanged, and each exclusion is counted:
- `latest_appearance_elsewhere`;
- `not_in_player_reference`;
- `position_not_eligible`;
- `no_history`;
- `no_model` (for example, a model skipped at training);
- `confidence_band`.

*Invalid or ambiguous input fails* rather than shrinking the set:
- a game that doesn't resolve, schedules that disagree, a game listed twice,
  or a prediction-schedule date that is missing, malformed or different from
  the game's;
- a missing feature table;
- two games for one team on one date;
- an ambiguous latest team;
- a player ID repeated in the reference (production silently gives such a
  player no rows);
- the `PROP_ROSTER_FILTER` roster rule switched on (not restated).

*What completeness covers:* every eligible prediction for the players in the
inputs, under these rules. It does **not** cover players missing from the
inputs:
- **Not in the stats:** players without recent stats.
- **Not in the reference:** for example 2026 rookies, who are absent from the
  current `players.csv` and are counted as `not_in_player_reference`.
- **Rosters and depth charts:** no real NFL roster or depth-chart coverage is
  claimed.

That reference and rookie coverage remains a separate limitation.

**Players-reference candidate builder** (`player_props/reference_builder.py`,
offline). Builds a *candidate* players reference and reports; nothing is
adopted and `data_files/players.csv` is never written.

*Why:* the committed `players.csv` is a one-time nflverse snapshot (blob
`9572b3d`, committed 2025-10-26) that no workflow refreshes. Its newest
`rookie_season` is 2025, and 6,391 of its `gsis_id` values are not GSIS IDs
at all: they equal the record's `esb_id` (placeholders nflverse used before
a GSIS ID was assigned). Play-by-play uses GSIS IDs, so those players can
never match. In the 2026 Week 7 real-data rehearsal, 57 of 355 candidates
were `not_in_player_reference`.

*Inputs:* the existing reference, and a downloads directory whose
`manifest.json` records each file's URL, size and SHA-256 (`players.csv`
from the nflverse `players` release; optionally `roster_weekly_<season>.csv`
from the `weekly_rosters` release). Every file is checked against the
manifest before use. The output directory must be outside the repository and
production paths. It must also not be, contain or lie inside the downloads
directory, and must not be or contain the existing-reference or candidates
file, since replacing an output removes it. Paths are compared fully
resolved, with symlinks and Windows junctions followed, and the check runs
before anything is read.

*Policy:*
1. **Fresh records** are taken as they are, in the existing column order,
   every value as text (no numeric re-formatting).
2. **Historical IDs** absent from the fresh source are retained unchanged.
3. **Placeholder correction:** a placeholder ID is replaced by a GSIS ID only
   when its `esb_id` and its `smart_id` each match exactly one fresh record,
   both the same one, whose `gsis_id` is GSIS-format, is not already an
   existing ID, and is claimed by no other placeholder. Otherwise the result
   is `no_match` or `rejected_*` (ambiguous, single ID only, conflicting,
   target not GSIS, target already existing, shared target), and the
   placeholder is retained.
4. **No roster filtering:** records are never dropped for roster membership
   or status (traded, injured, inactive, reserve, cut). The weekly roster is
   evidence only, for one explicit season and week; rows from other weeks
   are counted and not used.
5. **No name joins:** records sharing an `esb_id` or `smart_id` are reported
   as source aliases (one person, several records), never merged.

*Refused:* a missing required column (`gsis_id`, `display_name`,
`short_name`, `last_name`, `position`, `position_group`, `esb_id`,
`smart_id`), a fresh file lacking an existing column, a download that differs
from its manifest, and:
- a `gsis_id` that is missing, blank, padded, malformed or repeated;
- a `position` that is missing, blank or padded;
- a `position_group` that is missing, blank, padded, or contradicts the
  position label (QB, RB, WR and TE must have their own group, FB must have
  RB, and any other label must not have an eligible group).

The report lists eligibility-group changes (`eligibility_group_changes`,
`eligibility_flips`) separately from label-only changes within one group
(`position_label_changes_same_group`). Candidate eligibility is counted by
group.

*Definitions:*
- **Blank:** empty or whitespace-only.
- **Padded:** leading or trailing whitespace. A canonical ID and its padded
  spelling are therefore refused, not treated as two people.
- **Valid `gsis_id`:** either a GSIS ID (`00-00NNNNN`) or a supported legacy
  placeholder, which is three capital letters and six digits equal to the
  record's own `esb_id`. Every non-GSIS ID in the 2025 and 2026 nflverse
  files has that form. Anything else is malformed.

These checks run on the existing reference, the fresh source and the
assembled candidate, as written: no value is trimmed or rewritten.

Because a supported placeholder is its own `esb_id`, validated inputs can't
reach two of the crosswalk outcomes: a non-GSIS target, and two placeholders
claiming one target. The crosswalk still refuses both. As a backstop, the
candidate may not lose any existing ID except a documented correction.
Output is written to a staging directory and swapped in only when complete,
so a failure leaves any previous output unchanged.

*Outputs:*
- `players_candidate.csv`: the candidate reference;
- `crosswalk.csv`: every placeholder the source no longer has, with its
  result;
- `identities.csv`: each record's category (retained in source, added,
  corrected placeholder, retained historical) and its alias group;
- `field_changes.csv`: every changed field for retained and corrected
  identities, `meaningful` or `format_only` (whitespace, numeric form, or a
  listed equivalent such as the headshot URL template);
- `candidates.csv`: with `--candidates`, each week candidate's coverage and
  eligibility before and after, and the week's roster evidence;
- `report.json`: source hashes, the counts above as records and as people,
  field changes with the consumers that read each field, and eligibility
  flips.

*First run (2026-10-26, offline, on hashed downloads from 17:45 UTC):*
- **Sources:**
  - existing: committed `players.csv` at `1a16695`, `f3fd7d45…`;
  - fresh players: `72b6b09c…`;
  - roster: `46cff94a…`, 2026 weeks 1–7, with only Week 7 used.
- **Identities (records / people):**
  - retained in source: 24,133 / 24,133;
  - added: 513 / 512;
  - corrected placeholders: 215 / 215, all through both crosswalk IDs;
  - retained historical: 161 / 161 (59 GSIS-format IDs, plus 102
    placeholders without any crosswalk match);
  - one source alias: `00-0040793` and `EKE080143`.
- **Week 7 candidates:**
  - in the reference: 298 → 355 of 355;
  - eligible: 291 → 347, which is 291 + 57 newly covered (43 added, 14
    corrected) − 2 lost + 1 gained;
  - lost: `00-0036727` RB→FB and `00-0037304` TE→FB;
  - gained: `00-0033376` FB→RB.
  - The Week 7 roster lists all three as `position` RB with
    `depth_chart_position` FB. That disagreement is reported, not resolved.
- **Field changes:** the consumers of the changed fields are:
  - `position`, 276 meaningful: prediction eligibility and the pages;
  - `display_name`, 16: prediction output and the pages;
  - `short_name`, 253: the pages' name-merge key;
  - `last_name`, 1: read, not used.

  No other changed field is read by a consumer.

*Not done:* adoption. The candidate rehearsal (2026-10-27) and the pages'
`short_name` merges (see below) are done.

**Prop eligibility by position group** (policy `position-group-v1`;
`player_props/predict.py`, restated in `player_props/rehearsal_expected.py`).

*Why:* NFL's position label is inconsistent for fullbacks. Of the 17 Week 7
depth-chart fullbacks in 2026, nflverse's `position` labels 7 as FB and 10 as
RB, so eligibility by label included some fullbacks and excluded others.
`position_group` puts every fullback (all 210 FB records) in RB, and it is
the only field where position and group disagree. The prop models are
trained and evaluated on every row of their stat table regardless of
position, so the policy changes which (player, prop) pairs are predicted,
not the models.

*Rule:*
- **Eligibility source:** prop families come from the players reference's
  `position_group`, found by `gsis_id`, through `POSITION_MAP`:
  - QB → passing;
  - RB → rushing and receptions;
  - WR and TE → receiving and receptions.
- **Fullbacks:** an FB (group RB) gets the RB families, subject to every
  existing history, tier, model and confidence check.
- **Labels:** `position` stays the source label, shown as written (FB stays
  FB) and recorded in the output.
- **Recorded group:** each row records the group that selected it, as
  `eligibility_group`.
- **No other inputs:** no roster or depth-chart filtering and no per-player
  overrides.

*Fails closed:*
- **Reference without the column:** a reference without `position_group` is
  refused (`load_players` raises).
- **Unusable record in production:** a missing, blank or padded group or
  label, or a group that contradicts the label, gives that player no props,
  and the run prints which IDs and why. It never falls back to the position.
- **Unusable record in the rehearsal's restatement:** the expected set isn't
  derived, so verification fails and nothing is published.

*Rehearsal:* the restatement implements its own group check
(`restated_group`, `SKILL_LABEL_GROUPS`), importing only the shared
`POSITION_MAP` and `TIER_RULES`. The verifier checks, for each row:
- the recorded `position` against the reference label;
- `eligibility_group` against the reference's group;
- the prop against that group's families.

Verification refuses to run if production's `ELIGIBILITY_POLICY` and the
restated `RESTATED_POLICY` differ. `summary.json` records `eligibility`:
the policy, the restated policy, the field and the SHA-256 of the players
reference used (`runtime/data/players.csv`).
- **One read:** verification reads that file once and checks those exact
  bytes.
- **Re-checked:** the file's hash is checked again at the end of verification
  and before publishing. If it has changed or become unreadable, publication
  is refused and already-published files are untouched.
- **Not atomic (concurrent-writer race):** these are checkpoints, not a lock.
  A process that rewrites the reference after the pre-publish check, while
  the files are being copied, isn't detected. Nothing locks or snapshots the
  file, and the reference isn't published with the predictions. The
  protection assumes that nothing else writes into the run directory during
  a run.
- **What's recorded:** the summary records the hash verification read
  (`players_reference_source` `verification`; `verified` says whether
  verification passed), and separately the file's state when the summary is
  written (`players_reference_current_sha256` or
  `players_reference_current_unavailable`).
- **After publication:** a reference that becomes unreadable keeps the
  verified hash and doesn't change the run's status.
- **No verification:** a run that never reached verification hashes the
  file as it is (`current file`). If the file doesn't exist yet or can't be
  read, the SHA-256 is `None` and `players_reference_unavailable` says why;
  the fixtures copy and the configured source are never used instead.

*Unchanged:* model features, probabilities, trained artifacts, the UI's
identity safeguards (`reference_join`, which shows the label) and the
containment warnings. Older prediction files without `eligibility_group`
stay readable and are never rewritten.

*Not established:* unchanged model features don't establish accuracy,
calibration or betting value for any position subgroup, fullbacks included.
Prospective evaluation should report results by eligibility group.

*Effect (2026 Week 7 candidates, offline, with the candidate reference):*
- **Eligible players:** 351, against 347 when eligibility was by the
  candidate's label.
- **Newly eligible:** four players, all FB-labelled: Juszczyk, Burton,
  Prentice and Heyward.
- **Reference-wide:** the change report shows 5 real prop-family changes. A
  label-based rule would also count 6 pure FB/RB relabellings.
- **Not yet confirmed:** no real-data rehearsal has been run under this
  policy.

**Player-reference joins in the pages** (`player_props/reference_join.py`;
`pages/2_Player_Props.py`, `pages/3_Parlay_Builder.py`).

*Why:* the pages attached display name and position by merging on
`short_name`, which is not unique. Several players share a name (for
example "J.Smith"), so the merge could multiply prediction or season-total
rows, or attach another player's details. Several pickers had the same
weakness:
- the Pick 6 search keyed its options by "name - position";
- the Player Search tab mapped a chosen name back with `.index(name)`;
- the Parlay Builder mapped a chosen label back with `.index(label)`.

Each of these silently resolved two players to the first.

*Join rules:*
- **IDs are text:** the predictions, stats and reference files are read with
  `player_id` / `gsis_id` as text.
- **Many-to-one by ID:** rows are matched `player_id` → `gsis_id`. A row is
  never multiplied, and row count, order and existing values are kept.
- **Repeated reference IDs:** a `gsis_id` on more than one reference record
  is reported on the page and never used. Its records are kept, not dropped.
- **Unknown IDs:** a row whose ID is not in the reference stays unenriched.
  It never falls back to its name.
- **Rows without an ID (older files):** only these may use their name, and
  only when exactly one reference record has that `short_name` and its ID is
  unambiguous. Such matches are labelled as inferred, not verified.
- **Stored values kept:** fields are only filled where the input lacks them.
  When nothing can be attached, `display_name` falls back to the row's own
  `player_name`.

Every enriched row carries `reference_match`, with one of these values:
- `id`;
- `name_unique`;
- `unknown_id`;
- `duplicate_reference_id`;
- `ambiguous_name`;
- `name_not_found`;
- `no_identity`;
- `reference_unavailable`: the reference is absent, unreadable or lacks
  columns. This is kept distinct from a legacy name that simply isn't in a
  usable reference.

The pages show a warning summarising reference problems and rows that are
inferred or left unenriched.

*Where it applies:*
- **Prediction loaders (both pages):** every row's identity is classified,
  even when `display_name` and `position` are stored. Stored non-blank values
  are kept. Blank cells are filled only from an ID match (or, for rows
  without an ID, a unique-name match); otherwise `display_name` shows the
  row's own `player_name`. Unknown and repeated IDs are reported without any
  name fallback.
- **Pick 6 market-line prefill:** this player's rows by ID. Rows without an
  ID match only through an unambiguous display name.
- **Season leaders** (`reference_join.season_totals`):
  - rows with a `player_id` are totalled per ID (with name and team);
  - rows without one are totalled per name and team on their own. They are
    never merged into an identified player, even with the same name, and no
    row is lost.
  - Position is attached by ID, and only inferred for an ID-less group whose
    name is unique.
  - An `Identity` column appears when any row isn't verified by ID ("inferred
    by name", "ambiguous name", "not in reference", ...).
  - Season filters are unchanged.
- **Pick 6 search and Player Search:** one option per ID, labels made unique
  (the ID is appended to shared labels), and details looked up by ID. A
  repeated reference ID is shown with a warning instead of details.
- **Parlay Builder:**
  - leg labels are made unique, so each maps to exactly one row;
  - the same-player correlation warning is keyed by ID, so namesakes aren't
    treated as one player.

*Unchanged:* probabilities, calculations, filters, selections and stored
files. The dead first copy of `load_player_props_predictions` (replaced at
import by the second) was removed.

*Limits:*
- `get_recent_form` (unused) still filters by name.
- A season leader's `player_name` is still the stats file's short name.
- Older prediction files without IDs can only be enriched where a name is
  unique.

**Python 3.12 compatibility job (`player-props-py312` in
`.github/workflows/tests.yml`; added, not yet run).**
- **Why:** the nightly runner uses Python 3.12 (CPython 3.12.15 on
  2026-10-29). The `pytest` and `pipeline-smoke` jobs use 3.14, and the
  local rehearsals ran on 3.13. On 3.12 only three things have been seen to
  work: the pins install, and `train_models.py` and `predict.py` import and
  stop at the Stage 2 gate (nightly log). The rest of the prop path (source
  refresh, aggregation, training, bundles, the verifier and the rehearsal)
  is unverified on 3.12.
- **What it runs:** the files in `tests/player_props_py312_suites.txt`, on
  Python 3.12 with `requirements-dev.txt` (the production pins plus pytest
  and PyYAML). The list covers:
  - features, freshness and provenance;
  - train/serve parity and bundles;
  - source refresh and storage;
  - the synthetic and injected-download rehearsals, with completeness and
    early-failure summaries;
  - the reference builder and position groups;
  - UI identity joins and containment;
  - dates and seasons, the roster filter, and mocked injury and odds feeds;
  - Pick 6 wording, Model Performance week selection and backtest console
    encoding.

  `tests/test_player_props_suites.py` requires every
  `tests/test_player_props_*.py` to be listed.
- **Isolation:** synthetic fixtures and temporary workspaces only.
  - `tests/no_network_plugin.py` is a Python socket guard, not OS-level
    network isolation. For the whole session it patches `socket.getaddrinfo`,
    `gethostbyname`, `gethostbyname_ex`, `gethostbyaddr`,
    `create_connection` and `socket.socket.connect`/`connect_ex`, and restores
    them when pytest finishes.
    - **Allowed:** `localhost`, literal loopback addresses (recognized with
      `ipaddress`, including `::1` and IPv4-mapped forms), the unspecified
      addresses and Unix sockets.
    - **Refused:** every other hostname, before any DNS resolution, so
      lookalikes such as `127.example.com` fail too; and every other address.
    - **Not covered:** code that kept its own reference to a socket function
      before the plugin loaded, C extensions with their own networking, and
      subprocesses.
  - The job has `contents: read`, and checks out with
    `persist-credentials: false`.
- **Results:**
  - **Kept:** the full pytest log, JUnit XML, pytest's exit code, the suite
    list and the Python and library versions are uploaded as an artifact
    even on failure (30 days).
  - **Not hidden:** a failure fails the job; there is no
    `continue-on-error`. pytest's output goes straight to the log (no pipe),
    the version step runs with `pipefail`, and an empty suite list fails the
    job rather than letting pytest collect the default tree.
- **Not covered:**
  - live nflverse downloads;
  - real-data training on 3.12;
  - the nightly workflow itself, which is unchanged;
  - byte-for-byte equality of models trained on 3.12 and 3.13.

  The job passed on PR #50 (Python 3.12.15, Linux): the synthetic suites
  above are compatible with 3.12. Live data, real-data training, storage and
  deployment remain unverified on 3.12.

**Release manifest (S1; `player_props/release_manifest.py`; offline, nothing
released).** It builds and verifies one candidate release package: a
directory describing one coherent source, stats, reference and model-bundle
set. Nothing uploads, tags, activates or creates a release, and no workflow
runs it.
- **Package:** `release_manifest.json` plus 19 assets, each at one fixed path
  for its role:
  - the training and inference play-by-play sources and their source
    manifests;
  - the training and inference stats, with their Stage 2 manifests and
    schedules;
  - the candidate players reference (the builder's output) and the runtime
    reference (the bytes prediction read);
  - the exported bundle archive;
  - the rehearsal `summary.json` that verified the set, and the predictions
    snapshot it verified.
- **Manifest (`player_props_release`, version 1):**
  - every asset's role, name, size and SHA-256;
  - the reference derivation (`byte_copy` or `reserialized`);
  - the builder's code (file hashes, git commit, dirty flag) and runtime
    (Python, libraries, platform);
  - an optional parent release (`release_id` and manifest SHA-256);
  - an external tag and pointer, descriptive only;
  - the declared claims;
  - six *derived* sections, extracted from the assets:
    - `identities`: source, stats, schedule and reference hashes;
    - `bundle`: id, manifest hash, feature contract, scorer, training source,
      code and libraries;
    - `eligibility`: policy, field and the reference hash that verification
      read;
    - `windows`: per-model training and evaluation windows, training end and
      feature cutoff;
    - `inference_target`;
    - `verification`: the rehearsal's run id, stages, completeness and
      snapshot hash.
- **Training versus inference inputs:** they are different files, and their
  hashes are never required to match. They are linked explicitly:
  - The bundle's `training_source` must name this package's training stats
    manifest, every training stats file hash and the training play-by-play.
  - Each stats manifest must name its own source: training stats the
    training source, inference stats the inference source.
  - Both use the bundle's feature definition and rolling windows.
  - The training data must end before the inference feature cutoff.
  - The inference source must cover every training season.
- **Candidate versus runtime reference:** two assets with two hashes, never
  interchangeable, and bytes are never normalised.
  - **`byte_copy`:** the two files must be identical.
  - **`reserialized`:** the bytes must differ, but every identity column
    (`gsis_id`, `short_name`, `display_name`, `last_name`, `position`,
    `position_group`) must match as text, row by row.
  - The eligibility evidence must name the runtime reference's exact hash.
    Evidence naming the candidate is refused.
- **Evidence is cross-checked, not accepted:** a summary saying
  `verified: true` isn't enough.
  - The summary must list every run asset in the package under its run path
    with the same hash, including both copies of the bundle manifest.
  - It must name this bundle, contract, training end, sources and windows.
  - Its output verification and publish stage must name this snapshot.
  - Its status must be `succeeded`, with complete rows.
  - Its stage records must be exactly the supported rehearsal stages
    (`rehearsal.STAGES`): each once, in order, each with status `ok` and no
    error. Missing, duplicate, unknown, malformed, extra-field or
    non-successful records are refused, each with its index, even when every
    hash and derived section has been updated to match. A summary with only
    `workspace: ok` used to pass.
  - The snapshot's path in the summary fixes the target week, and the Stage 2
    gate (`provenance.check_inputs`) is re-run for that week on the
    inference stats. It runs as of the rehearsal's finish date, in Toronto
    time.
- **Verification (`verify_package`):**
  - **Reading:** every asset is read once.
  - **Refused before any content check:** missing, unlisted, duplicate,
    conflicting, unexpected or unsafe entries; links or junctions; size or
    hash mismatches; malformed JSON or schema. JSON errors give the line and
    column; schema errors give the field. An asset's `role` and `name` must be
    strings: a list, object, null or number is reported for that field
    (previously a list or object raised `TypeError`).
  - **Content checks:** the existing validators run on a private temporary
    copy of exactly those bytes:
    - `source_refresh.verify_installed`;
    - Stage 2 `check_provenance` and `check_inputs`;
    - `bundle_store.read_archive`, the safe tar reader (no general
      extraction);
    - `model_bundle.verify_bundle_dir`.
  - **Derived sections:** each is recomputed and compared with the manifest.
- **Building (`build_package`):** each input is read once and staged in a
  sibling directory.
  - **Refused locations:** an output that overlaps the repository, a
    production path, any input file, the run directory or the `--parent`
    package (paths compared fully resolved, links and junctions included).
    - The parent is an input. The output may not be it, lie inside it or
      contain it: replacing the output would otherwise delete or change the
      package the new manifest names as its parent.
    - These checks run before the parent is read and before anything is
      created; on refusal the parent, the inputs and any previous output are
      left byte-identical.
  - **Install:** the package must verify before it replaces the output, with
    a rename-aside swap.
  - **On a catchable failure:** the previous package is kept (or restored)
    and the staging directory is removed.
  - **Not handled:** a crash or power loss mid-swap can leave a `.staging-`
    or `.previous-` sibling, and nothing is fsynced.
- **Verified versus declared:** the report lists both.
  - **Verified:** the hashes and the relationships above.
  - **Declared, not provable from the bytes:**
    - that the downloads came from nflverse at the recorded times;
    - that the rehearsal ran the stages it reports;
    - that the runtime reference was produced from the candidate (only
      identity-column equality is checked);
    - git commits, dirty flags and library versions;
    - the parent release, unless `--parent` is given;
    - the external tag and pointer.
- **Immutability:** hashes detect a changed asset. They don't make a package
  or any stored copy of it immutable, and don't prevent replacement or
  deletion.
- **CLI** (exit 0 valid, 1 refused, 2 couldn't run):

  ```
  python player_props/release_manifest.py build --run-dir RUN --candidate-reference CSV --reference-derivation reserialized --out DIR
  python player_props/release_manifest.py verify --package DIR [--parent DIR]
  ```
- **First real-artifact check (2026-10-30; in a temporary directory outside
  the repository):** a package was assembled from the retained 917-row
  position-group-v1 rehearsal (run `20261029T202848Z-957eda08`) and its
  candidate reference. Nothing was retrained, regenerated or downloaded.
  - **Result:** it built and verified.
  - **Linked training and inference inputs:**
    - training source `81ed2ec7…` (2020–2025) → training stats `277bda96…`,
      which the bundle names;
    - inference source `235b8a9e…` (2020–2026) → inference stats
      `5f56dc26…`.
  - **Separate references:** candidate `2d3aad21…` and runtime `8e634557…`
    (reserialized; identity columns identical).
  - **Evidence:** the eligibility evidence names `8e634557…`, and 917 of 917
    rows are verified.
  - **Negative check:** declaring the references a byte copy was refused.
  - **Model performance:** the check measures nothing about it.

**Remaining stages (not implemented):**
2. **Deploy the source refresh (5B).** Run the Stage 5A refresh on the
   runner, aggregate from the prop source, rebuild the stats with their
   manifest, and record the manifest's hashes with each prediction. The
   Stage 2 checks block predictions until this exists.
4. **Deploy bundle storage (5B).** Commit or upload the published bundle,
   restore it before predicting, and serialise runs (see the procedure
   above). Export, import, locking and inventory are done (Stage 5A).
5. **Retraining.** Retrain on rebuilt, manifested stats (after stage 2),
   publishing a verified bundle.
6. **Leakage-safe and prospective evaluation.** Score the held-out season,
   comparing with and without `days_rest` and the real `is_home`. Then
   require prospective frozen weeks before removing the containment notice.
   No accuracy improvement is claimed until then, including from parity.

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
