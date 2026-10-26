# Changelog

History moved out of `README.md`. Newest first. Dates are as recorded in the
original notes; undated dashboard/infra work from late 2025 is grouped at the
bottom.

---

## October 2026

- **Player props: stable `player_id` in prediction outputs** (`player_props/predict.py`,
  `player_props/rehearsal.py`).
  - **New rows:** each carries the stats ID that selected its features and
    scored its model, as text, in the latest feed and newly frozen snapshots.
  - **Rehearsal verifier:** uses that ID directly, with no name lookup. It
    fails closed on a missing, unknown, repeated or inconsistent ID (names,
    team, position or prop eligibility). Duplicates and feed/snapshot
    comparisons include the ID.
  - **Existing files:** production feeds and snapshots are unchanged and still
    readable. Historical files are not backfilled by name or treated as
    identity-verified.
  - **Unchanged:** probabilities, selection, the feature contract, bundle
    checks and generation gates. Generation stays blocked.
  - **Limitations:** historical grading (`backtest.calculate_hit_rate`) still
    joins predictions to results by `player_name`. Output completeness and
    2026 rookie coverage remain separate and unverified.

- **Player props: real-data compatibility rehearsal** (`player_props/rehearsal_real.py`,
  `player_props/rehearsal.py`). Isolated; not deployed. Generation stays blocked
  and the containment warnings are unchanged.
  - **How it runs:** the Stage 5B rehearsal can now run on real nflverse data.
    A separate download step fetches the play-by-play and schedule into the
    workspace with recorded hashes. The rehearsal then runs offline on those
    exact bytes (`--real-inputs`, `--players`, `--train-seasons`, `--target`).
  - **First run (2026 week 7):** passed source validation (2020-2026), the
    Stage 2 checks, training, bundle restore, prediction and output
    verification.
  - **Fix:** the verifier mis-identified players who share a display name.
    It now resolves them by team and refuses ambiguity.
  - **Summary:** `summary.json` records the inputs used.
  - **Download attempts:** each gets its own exclusively created directory, so
    one in the same second can no longer reuse or overwrite a previous one.
    The completion record is written only after success.
  - **Not claimed:** completeness of the output rows, accuracy or edge.

- **Historical Data: clearer CSV exports** (`historical_exports.py`,
  `pages/1_Historical_Data.py`). "Export Filtered Data" downloaded only the
  visible page (typically 50 rows) although its label suggested every
  filtered row.
  - **Export current page (N rows):** the rows shown, as before; file
    `nfl_historical_data_<date>_page<p>_of_<n>.csv`.
  - **Export all filtered rows (N rows):** every row matching the active
    filters, across all pages; file
    `nfl_historical_data_<date>_all_filtered_<N>_rows.csv`. It is built only
    when clicked (a deferred Streamlit download), never on a rerun, and
    nothing is kept in session state, so it always reflects the filters and
    data in effect. Clicking it doesn't rerun the page: in a real browser a
    rerun replaced the deferred download mid-fetch and no file arrived.
  - **Before the full dataset is loaded** the page works from a small sample,
    so that button reads **Export filtered sample (N rows)** and its file is
    `nfl_historical_data_<date>_sample_filtered_<N>_rows.csv`. A caption says
    only the loaded sample is included and that "Load full historical
    play-by-play data" is needed to export the full filtered dataset. The
    page passes the loaded-data scope explicitly; it isn't inferred from the
    row count.
  - **Known limitation (unchanged):** filters set on the sample are cleared
    when the full dataset is loaded, because that button reruns the page
    before the filters render. Re-apply them after loading.
  - **Unchanged:** columns, UTF-8 encoding, display order and no index
    column. Filtering and pagination are untouched.
  - **Edge cases:** no matching rows says there is nothing to export. An
    export failure is shown instead of hiding the buttons.

- **Home page: the sidebar downloads are back** (`home_downloads.py`,
  `predictions.py`). Download Predictions, Download Betting Log and Generate
  Predictions PDF never appeared: `home_page()` created the three sidebar
  placeholders as locals but filled them only `if '...' in globals()`, which
  is always false inside a function, and every error was swallowed.
  - **Fix:** `home_page()` now passes its placeholders to
    `home_downloads.render_home_downloads()`. The CSV, PDF and link code moved
    there unchanged, so file names, contents and the on-demand PDF (also saved
    under `data_files/exports/`) are as before.
  - **Missing inputs are explained:** predictions not loaded yet, no
    recommendation log yet, or the PDF library (now imported optionally)
    unavailable. A failure is shown in the sidebar instead of hiding the
    control.
  - "Betting log" is the model's recommendation log
    (`betting_recommendations_log.csv`), not the Bet Journal.

- **Player props: Stage 5B preparation, an isolated end-to-end rehearsal**
  (`player_props/rehearsal.py`, `player_props/rehearsal_fixtures.py`,
  `predict.py`). Offline only: synthetic downloads, network blocked; no
  workflow, dependency or production change; generation stays blocked.
  - **Pipeline:** the real production functions run in order: source refresh
    (separate training and inference sources), aggregation with the stats
    manifest, training under the production minimums, a verified bundle,
    export, restore into an empty runtime, input validation, prediction,
    independent verification, and snapshot-immutable publishing.
  - **Output verification:** before publishing, every row of both the latest
    feed and the frozen snapshot is recomputed from the raw score and the
    permitted clamp: final probability, complement, confidence, side, exact
    adjustment label, tier, model, line and bundle ID. The two files must
    match each other, and a file changed after verification is not published.
  - **Paths:** everything lives under an explicit workspace. Paths that
    resolve into or around the repository or production data are refused,
    including via symlinks and junctions.
  - **Failures:** a failure stops later stages and keeps diagnostics. A
    different frozen snapshot is never overwritten.
  - **Summary:** a machine-readable summary records stage statuses, artifact
    hashes, coverage, training and evaluation windows, the bundle identity,
    exclusions and limitations. No accuracy is claimed.
  - **`predict.generate_predictions`** accepts explicit `data_dir` /
    `models_dir` for one call; default behaviour is unchanged.

- **Player props: Stage 5A source refresh and bundle storage (mechanisms;
  not deployed)** (`player_props/source_refresh.py`,
  `player_props/bundle_store.py`, `model_bundle.py`). No live data fetched,
  no workflow changed, no production file or model changed; generation stays
  blocked.
  - **Source refresh:**
    - The required seasons are explicit (no "cache plus current year"), and
      every season is validated.
    - Validation covers LFS pointers, empty or truncated data, season
      identity, game IDs, dates, duplicate plays, and game-level coverage
      against the Stage 2 completion predicate.
    - Each downloaded game must match its schedule row's date, season and
      week.
    - A malformed schedule (invalid dates, duplicates, conflicts,
      contradictory scores) fails closed, rather than being read as
      unfinished games.
    - A failed rollback names every unrestored file and the backup that
      still holds its previous content.
    - A complete candidate is staged and installed with a manifest of what
      was actually read; any failure keeps the previous source.
    - It is separate from the spread pipeline's file and metadata.
  - **Bundle storage:**
    - Deterministic export and fully validated import (unsafe entries,
      hash mismatches and conflicting IDs are refused).
    - The pointer switches only after the staged bundle verifies.
    - The existing `SlotLock` coordinates publishers and is never removed on
      age alone.
    - A read-only inventory; retention never deletes the current bundle or
      the only verified copy.
  - **Deployment:** a later procedure (5B) is documented in
    `docs/architecture.md`, including workflow changes, permissions and
    artifact/cache limits. None of it is enabled.

- **Player props: Pick 6 calculator wording** (`pages/2_Player_Props.py`,
  `player_props/pick6.py`). The calculator has scored no model since Stage 4;
  its wording now says so.
  - **Wording:** the percentage is labelled **Smoothed historical hit rate**
    (was "Model output"). It is explained as Laplace-smoothed from historical
    games ((over + 1) / (games + 2)), not a model prediction, a calibrated
    win probability or a betting recommendation.
  - **Counts:** the card shows the sample, split into games strictly above,
    strictly below and exactly at the line, plus the observed games behind
    the side shown and that side's formula with its numbers.
  - **Equality and ties:** with a whole-number line, games exactly at the line
    still count as not over (unchanged), so the other side is labelled
    **NOT OVER (below or equal)** with (games not over + 1) / (games + 2).
    Before, ten games all equal to the line showed "91.7% UNDER". It reads
    "UNDER" only when no game equalled the line.
    - A note says this counting convention does not describe sportsbook
      settlement rules; check the operator's rules separately.
    - Tied sides are flagged, and the heading is now "Side with the higher
      smoothed rate".
  - **Removed:** the intro's "model's prediction", the "Recommendation"
    heading, and the obsolete Pro Tips (model tier selection, weighting
    claims, defense rank). The how-to guide's model example is replaced.
  - **Unchanged:** the calculation (now `pick6.historical_hit_rate`, tested
    against the original formula), the containment notice and the
    stale-data warning.

- **Player props: Stage 4 model provenance and consistent scoring**
  (`player_props/model_bundle.py`, `models.py`, `predict.py`, `features.py`,
  `train_models.py`, `containment.py`, `pages/2_Player_Props.py`). Code
  only: no model retrained, replaced or regenerated. Generation stays
  blocked, and the containment notice stays and gains two items.
  - **Found:**
    - Training reported XGBoost + LightGBM ensemble metrics, but prediction
      scored with XGBoost only (the LightGBM files were never committed).
      Rescoring the committed models read-only changed AUC by at most 0.009
      and no `reliable` flag.
    - Prediction kept its own `PROP_LINES`, which differed from training for
      8 tiers, so about 1 in 12 stored props show a line their model wasn't
      trained on.
    - Two elite tiers were silently served by the `star_wr` model.
    - Model files were written one at a time with no manifest.
    - The Player Props calculator scored unverified models on improvised
      features.
  - **Now:**
    - XGBoost only, scored by one shared, validated `score_probabilities`
      in evaluation and prediction.
    - Models, metrics and a versioned manifest form one bundle under
      `player_props/models/bundles/`. It is verified from disk before an
      atomic pointer switch (`current_bundle.json`) and re-verified before
      every use.
    - The manifest records hashes, identity, the feature contract, the
      training source, code, libraries, windows and the evaluation protocol.
    - No fallbacks, and one `PROP_LINES`.
    - Outputs gain `raw_model_prob_over`, `adjustments`, `model_name` and
      `model_bundle`. `adjustments` names every clamp, injury rule
      (questionable, doubtful, limited practice) and weather change that
      altered the probability.
    - The page calculator shows only the historical rate.
  - **Existing models are refused**, with no retrospective manifest.
    Nightly retraining now refuses to train on unverified stats (exit 1) and
    no longer overwrites the flat model files.

- **Player props: Stage 3 train/serve feature parity**
  (`player_props/features.py`, `models.py`, `predict.py`, `aggregators.py`,
  `provenance.py`). Code only: no data refreshed, no model retrained or
  replaced. Generation stays blocked, and the containment notice stays.
  - **Found:** prediction used each player's latest stored row, whose
    rolling stats exclude that game, so they were one game stale (about 95%
    of 2025 rows differed in L3). Its `days_rest` was always 1. It took
    `is_home` from the team order, and matched players by short name and
    team, which mixes players who share a name and loses traded players.
    Model-input errors were skipped silently.
  - **Now:** one definition (`MODEL_FEATURES`, exact order).
    `inference_rows` rebuilds a target game's features with the Stage 1
    functions from the player's eligible history: by player ID, every team
    and season, strictly before the game, within Stage 2's feature source.
    - `days_rest` is the gap to the previous appearance, capped at 21.
    - `is_home` comes from the unique schedule game.
    - First appearances are excluded on both paths.
    - History is validated first: repeated player-games, rows of a game
      that disagree on labels, or a player in two games on one date raise.
      Nothing is deduplicated.
    - Inputs are validated (`model_matrix`), and stats are read with
      round-trip float parsing.
  - **Compatibility boundary:** models record `FEATURE_CONTRACT` and their
    feature list, but only when trained on stats with a valid manifest. The
    recorded list is required: an exact list of unique names in order.
    Prediction refuses any other model, so all committed models are
    refused, with no provenance created for them.
  - **Training changes too:** nightly retraining builds rows from the
    shared list, which gives the same rows on the current CSVs. It reads
    stats with exact float parsing (value differences up to about 6e-14). It
    fails on invalid stats instead of skipping columns. It records a null
    contract, so those models are refused.
  - **Offline check:** 42,924 rebuilt rows (612,606 values) equal their
    training rows exactly; 2,283 first appearances are excluded on both
    paths. No accuracy improvement is claimed.

- **Player props: Stage 2 freshness and provenance**
  (`player_props/provenance.py`, `predict.py`, `aggregators.py`,
  `train_models.py`, `backtest.py`). Code only: no data refreshed, no stats
  rebuilt in production, no model retrained; the containment notice stays.
  - **Found:** the nightly runner's play-by-play is 2020-2024 plus 2026, with
    no 2025, and nothing nightly reads it. The committed stats CSVs end on
    2026-01-04 but miss BAL@PIT from that day, the 2025 playoffs and all of
    2026. `pbp_metadata.json` claims "2020-2026".
  - **Source observation:** the actual seasons, games, dates, per-game
    `END GAME` scores and SHA-256 of the play-by-play content, with the time
    it was read. Modification times and metadata are not used.
  - **Manifest:** `player_props_stats_manifest.json` is written with the
    three stats CSVs, in the same rollback set. It links each file's hash and
    coverage to the source, the aggregation code revision and the feature
    definition. The existing CSVs have no manifest and stay unverified.
    Malformed or incomplete manifests (any nested structure, hash, timestamp
    or coverage field) are reported failures, not tracebacks.
  - **Freshness policy:** for a target week, every completed earlier game
    (this season, last season including its playoffs, and earlier seasons in
    the stats) must be in all three tables, with rows for both teams.
    Missing or ambiguous completion evidence fails. Rows from the target
    week or later are removed from the feature source.
  - **Completed** means all of these hold:
    - whole, possible scores, not 0-0;
    - a recorded result and total that agree with the scores;
    - dated before today in Toronto;
    - the source's `END GAME` play shows the same score.

    Recorded schedule scores alone are not treated as a final status.
  - **Command:** `python player_props/provenance.py` (read-only; exit 1 when
    freshness or provenance can't be established).
  - **Gate:** `predict.py` runs the check first and exits 1 without touching
    any predictions file when it fails. With the current files it fails, so
    nightly prop predictions stop being refreshed until the stats are rebuilt.
  - **Backtest:** a predictions file is scored only when every row's week
    and season equal the requested ones. Files that lack the columns, are
    empty, or hold mixed or other weeks are refused; the file name alone
    proves nothing.

- **Player props: Stage 1 feature generation** (`player_props/aggregators.py`,
  `player_props/train_models.py`). Code only: no stats were rebuilt in
  production, no model was retrained, and the containment notice stays.
  - **Root causes fixed in code:**
    - The full-aggregation training path saved the stats CSVs before adding
      `is_home`/`days_rest`.
    - The committed CSVs came from a January version with random `is_home`
      and constant `days_rest`.
    - Name-spelling variants split one player-game into two rows (2 in
      receiving).
    - Rolling stats were ordered by season/week, not by date.
  - **Now:** `build_player_stats` adds every feature before
    `write_player_stats` validates the exact column contract.
    - A validation failure writes nothing.
    - Each file is replaced atomically. If a later replacement fails, the
      earlier ones are rolled back from `.bak` copies, and files that didn't
      exist before are removed.
    - If that rollback fails, `PersistenceRollbackError` keeps the backups
      and names them.
    - A crash or power loss between replacements can still leave a mixed
      set; this is not a transaction across the three files.
    - `is_home` comes from the unique matching schedule game; missing,
      duplicate or conflicting evidence fails.
    - Rolling stats use only earlier appearances, by date.
    - `days_rest` = days since the previous appearance in that stat table,
      capped at 21 (7 for a first appearance), across seasons and teams. It
      measures appearance gaps, not physiological rest.
    - Duplicates are merged with every play summed, and ratios are never
      infinite.
  - **Offline check:** a rebuild from local sources into a temporary
    directory validated, with `is_home` matching the schedule on 100% of
    rows (production: about 50%). The local play-by-play ends 2026-01-04, so
    the output is **not current**.
  - **Unchanged:** prediction-time `days_rest`, production stats, models and
    predictions.
  - **Remaining stages:** source refresh, train/serve parity, model
    provenance, retraining and leakage-safe evaluation (see
    `docs/architecture.md`).
  - **Tests:** `tests/test_player_props_stage1.py`.

- **Player props: containment notice** (`player_props/containment.py`, Player
  Props, Parlay Builder, Model Performance).
  - **Why:** a read-only audit of the committed artifacts found two problems.
    - The player stats behind every prop end on 2026-01-04 and are not
      refreshed, so 2026 props use no 2026 games.
    - The historical home/away values the saved prop models were trained on
      were assigned at random. It also found `days_rest` constant (7 in
      training, 1 at prediction time), with 0 splits in every committed model.
  - **Now:** every page that shows prop outputs leads with a warning not to use
    them for wagers. The warning explains both issues, says the spread model
    is separate, and says nightly-retrained models are unverified.
  - **Wording changes:**
    - Elite/Strong/Good/Lean become neutral probability bands.
    - "tested (reliable)" becomes "passed the 2025 holdout check".
    - "Top Picks/Recommendations" become model outputs.
    - Model Performance's "Best Strategy: Bet on…" and "Profitable
      Thresholds" become past results, not advice.
    - Parlay Builder's betting tips are replaced with neutral guidance.
    - The stale ROC-AUC claims in the Player Props help text are removed.
  - **Removed:** Parlay Builder's "Edge" metric and "Our Edge" / "Implied
    Win Rate" lines. Edge compared the combined probability with odds derived
    from that same probability, so it was always 0%.
  - **Unchanged:** probabilities, filters and their defaults, stored
    predictions, models, stats and past performance figures.
  - **Repair plan:** documented in `docs/architecture.md`, not yet
    implemented.
  - **Tests:** `tests/test_player_props_containment.py`, plus a Model
    Performance render test.

- **Player Props: remaining-games banner uses the predictions' season**
  (`player_props/season_schedule.py`, `pages/2_Player_Props.py`).
  - **Before:** the page loaded the hard-coded `nfl_schedule_2025.csv`, so in
    2026 it found no games and the banner never appeared. It also said
    "upcoming games this week" while counting every remaining game.
  - **Season:** taken from the displayed predictions' `season` column, never
    the calendar year. Every prediction must record the same whole-number
    season (a missing value is an error, not skipped). Every prediction's
    complete game date must be a recorded calendar date (`YYYY-MM-DD`, or that
    date at midnight with no offset or UTC) between 1 August and the end of
    February, so January playoff games belong to the previous season.
  - **Schedule:** only `nfl_schedule_<season>.csv`. If it is missing,
    unreadable or malformed, or the season is ambiguous or invalid, the page
    shows a warning and omits the count. Malformed means any of:
    - missing any of `update_schedule.py`'s six columns;
    - no games;
    - a date that isn't `YYYY-MM-DD` or is outside the season;
    - a week that isn't a whole number from 1 to 22;
    - a missing, malformed or identical home/away team;
    - a team listed twice on one date (a duplicate game).
    There is no fallback to another season, and the predictions stay visible:
    the page now parses prediction game dates with `errors='coerce'`, so a bad
    stored value no longer breaks the page.
  - **Wording:** "N remaining scheduled games this season (2026)", counted
    from today's date in Toronto. Today's games stay included because the
    schedule records dates, not kickoff times.
  - **Unchanged:** prediction calculations, week selection, stored data and
    `days_rest`, which remains a separate investigation.
  - **Tests:** `tests/test_player_props_season.py`.

- **Player Props: upcoming games keep their game day** (`player_props/game_dates.py`,
  `pages/2_Player_Props.py`).
  - **The bug:** `get_upcoming_games` read the schedule's calendar `date` as UTC
    midnight and compared it with the current instant. A game dropped out of
    the "upcoming games" count at 8 PM Toronto (EDT) the evening before it
    was played, and was missing all of game day.
  - **Now:** a calendar date counts as upcoming through that whole day in
    America/Toronto. Only an offset-aware `commence_time` is a kickoff instant,
    converted to America/Toronto. A time with no zone keeps its date and is
    never assumed UTC. Missing or invalid dates are excluded.
  - **Reviewed and left unchanged:**
    - `predict.load_schedule`: its UTC-midnight cutoff never shifts a date. It
      selects a week until 08:00 Toronto on the game day, which fits the
      nightly run.
    - `predict.calculate_days_rest`: the timezone cannot change its result.
  - **Unchanged:** prop predictions, week selection and stored data.
  - **Limitation:** the page still loads the hard-coded
    `nfl_schedule_2025.csv`, so in the 2026 season it finds no upcoming games
    and the real banner does not appear. This fix has no visible effect
    until the page loads the current season's schedule.
  - **Follow-ups:**
    - Load the current season's schedule, and make the banner wording
      accurate: it says "this week" but counts all remaining games.
    - Investigate `days_rest`: prediction always passes 1, while training
      uses the real gap between games. Keeping it unchanged here does not
      mean it is correct.
  - **Tests:** `tests/test_player_props_dates.py`.

- **Alert header: correct game date** (`betting_log.game_time_text`, alert view,
  Betting Log note).
  - **The bug:** the alert view read the logged game day as UTC midnight and
    converted it to Eastern time, so every alert showed the previous day:
    `2026-10-26 00:00:00` (a Monday game) showed "Sun, Oct 25 2026 08:00 PM",
    and `2026-09-13` showed "Sat, Sep 12 2026".
  - **Now:** date-only values, including the log's midnight form, show that
    calendar date with "(date only; kickoff time not recorded)". Only a
    timestamp with an explicit offset is shown as a kickoff time, in
    America/Toronto with EDT/EST. Missing or invalid values show "Game date
    unavailable". The alert's detail table uses the same text.
  - **Betting Log note:** the outdated "Automatic Updates" note (ESPN scores,
    Nov 16, 2025) now describes current grading: nflverse final scores, after
    the game date, in the nightly update and on page load, with no promised
    run times.
  - **Unchanged:** the log's records, schema, recommendation text and
    settlement.
  - **Tests:** `tests/test_alert_game_date.py`.

- **Recommendation log: clearer display** (`betting_log.py` display helpers, Betting
  Log tab, alert view, RSS).
  - **The confusion:** a row stored `spread_line=-4.5`, `recommended_team=PHI`
    and `bet_spread=4.5`. Shown together, the raw nflverse line
    (home-favourite positive) read as "PHI -4.5" for a PHI +4.5
    recommendation.
  - **Now:** a **Recommendation** column or badge shows `PHI +4.5`, built from
    `recommended_team` and `bet_spread` only. `spread_line` is labelled
    **Raw source spread**, with its sign convention explained. Odds read
    `-110 (assumed; not a sportsbook offer)`.
  - **Legacy rows:** rows without a team or a recorded handicap show
    "Recommendation unavailable". No handicap is derived for display.
  - **Unchanged:** the CSV schema, records, append and dedup, grading and
    settlement.
  - **Tests:** `tests/test_recommendation_display.py`.

- **Injury archive: streamed, bounded ESPN response** (`injury_reports.py`).
  - **What happened:** the first live capture (2026-10-21) received HTTP 200
    with 8,726,788 bytes. That was over the 8 MiB cap, so it was archived as
    `unavailable` / `too_large`. It is unchanged, and the slot is not
    backfilled.
  - **New size limit:** 32 MiB of decoded body, about 3.8x the observed size.
  - **Streaming:** the response is streamed and counted in 64 KiB chunks,
    stopping as soon as the limit is passed. Compressed bodies are counted
    after decoding. A declared Content-Length over the limit is rejected
    unread; otherwise the header isn't trusted.
  - **Time limits:** a 90 s body-read deadline is added to the existing 5 s
    connect and 20 s per-read (inactivity) timeouts. A watchdog timer shuts
    the connection's socket down when the deadline fires, so a trickling
    server can't hold a chunk read open. Reading stays in the calling thread,
    and the timer is joined before returning.
  - **Memory wording:** the limit bounds the accumulated body bytes (at most
    the limit plus one chunk), not total process memory.
  - **Clean-up and evidence:** the response is closed on every path. Error
    evidence and the observation time (receipt, after the body is read) are
    unchanged.
  - **Unchanged:** sportsbook collection, persistence, retries and credit
    use.
  - **Still pending:** compatibility with the real payload, since no raw
    response has been parsed or archived yet.

- **Home page: "Predictions last generated" shows the real generation time**
  (`prediction_provenance.py`).
  - **Was:** the predictions CSV's file modification time, in the machine's local
    zone. A git checkout or fast-forward resets that time, so the caption showed
    09:42 EDT when the nightly had run at 04:12 EDT.
  - **Now:** the pipeline run manifest's `generated_at` and run ID, used only when
    the manifest validates (`pregame_snapshots.validate_manifest`) and its recorded
    sha256 matches the predictions file on disk. It is shown in America/Toronto
    with EDT or EST.
  - **Otherwise:** "Generation time unavailable" with a short reason (missing,
    malformed or mismatched manifest). File time, checkout time and the current
    time are never substituted.
  - **Display only:** the manifest, predictions and pipeline are unchanged.

- **Reliability plot: nothing clipped** (`scripts/frozen_model_reliability.py`). The
  calibration SVG's footer note ran past the 560 px canvas. It now wraps onto two
  lines with the same wording, and the canvas is taller. A long title wraps
  before its details, and a count label near the right edge sits left of its
  point. Layout only: report metrics, JSON and CSVs are unchanged.

- **Frozen-model reliability report (read-only, evidence only)**
  (`scripts/frozen_model_reliability.py`, `docs/FROZEN_MODEL_RELIABILITY.md`).
  - **Built on the frozen-spread report:** it uses that report's selection,
    timing cross-check, settlement, coverage and exact-line Ontario market
    match unchanged.
  - **Protocol v1, fixed for subsequent evaluation** (not registered before
    the games already evaluated, whose results are exploratory):
    - the latest pregame snapshot, chosen first, with no fallback;
    - pushes counted separately;
    - all predictions and the signal subset reported separately;
    - two explicit samples: overall (model vs 50%) and matched Ontario
      market (model vs market vs 50%; books averaged, so one game is one
      observation).
  - **Calibration:** fixed bins [0,.40) / [.40,.50) / [.50,.60) / [.60,.70) /
    [.70,1] with 95% Wilson intervals. These are binomial reference
    intervals that ignore dependence between games, separate from the
    bootstrap. Empty bins are kept; sparse bins are descriptive only.
  - **Uncertainty:** paired per-game Brier and log-loss differences (model
    minus baseline) with a paired season/week cluster bootstrap (10,000
    replicates, seed 20261005). The interval is unavailable with fewer than
    5 weeks. Reaching 5 only permits an exploratory interval, not a reliable
    one, and no edge is claimed.
  - **Sensitivity:** an optional, separate earliest-snapshot view, never
    pooled.
  - **Output:** JSON, two CSVs and a standalone SVG calibration plot in
    the git-ignored `reports/frozen_reliability/`, with source fingerprints
    and protected output paths. Corrupt inputs fail the run; no source
    writes, network calls or journal access.
  - **Sample:** frozen tracking began 2026-10-05, so the sample is short.
    Changing thresholds on it would need a new, untouched evaluation
    period.
  - **Tests:** `tests/test_frozen_model_reliability.py`.

- **Injury report archive, phase 1 (evidence only)** (`injury_reports.py`,
  `docs/INJURY_REPORT_ARCHIVE.md`).
  - **What it does:** around each Ontario decision slot, the Ontario Spread
    Capture run also saves ESPN's injury report as an immutable, checksummed
    capture in `data_files/injury_reports/captures/`.
  - **Source semantics:** raw statuses are kept as supplied. ESPN's status
    isn't the official game report, and the feed has no practice
    participation, so that is stored as unavailable. ESPN's per-entry
    `date` is kept raw (plus UTC when parseable) as a source *entry date*
    whose meaning is unverified. It is never a last-update time or
    freshness evidence, and HTTP `Date` / `Last-Modified` stay separate.
  - **Truthful times:** request and receipt times are stored, and the slot
    is judged at receipt with the sportsbook capture's Toronto rules: on
    time, late, or outside the window (which doesn't fill the slot). Ad-hoc
    captures are labelled, and past slots are never reconstructed. Under
    the slot lock, the duplicate check is repeated and the clock re-read
    immediately before the request. If the window has closed by then,
    nothing is requested (`window_closed`), and the run never switches slot
    or to ad hoc.
  - **Coverage:** per in-scope team (rows listed / listed without rows /
    absent). Outcomes are full / partial / empty / unavailable, and empty
    or unavailable captures are kept as evidence while the slot stays open.
  - **Isolation:** the workflow steps run after every sportsbook step, never
    see `ODDS_API_KEY` and are `continue-on-error`, so sportsbook capture
    behaviour and the run's conclusion are unchanged. Failures show as
    annotations and in the run summary. The commit stages only this run's
    file and refuses if a sportsbook commit is unpushed. Persistence is
    verified against a fresh fetch of remote `main` by sha256: persisted /
    committed locally only / never committed / unverified, plus whether the
    artifact holds a copy. A clean working tree is never taken as proof.
  - **Not included:** no injury model, adjustments, scores, recommendations
    or page. `espn_injuries.csv` and the dispatcher are untouched.
    `python injury_reports.py list` validates and lists captures.
  - **Tests:** `tests/test_injury_reports.py`.

- **Bet Journal: moneyline wagers and bonus bets** (schema 2; cash spreads
  unchanged).
  - **Markets:** a wager is a spread or a moneyline. A moneyline has no
    handicap (`null`, never +0) and settles on the outright winner. A tie
    follows the wager's recorded tie rule from the sportsbook's terms
    (refunded as a push, or a loss); it is never assumed to be a loss.
  - **Funding:** entirely cash, or entirely one bonus bet with its face
    value and whether the stake is returned as cash on a win (reissued
    bonus credit is not cash). Mixed funding,
    parlays, boosts and cash-outs are rejected or out of scope.
  - **Bonus payouts:** a win pays bonus × multiplier in cash (plus the
    stake only when returned). A loss is $0.00 cash. A push leaves the
    bonus outcome unresolved. Example: $25 at +320, stake not returned:
    win $80.00, loss $0.00.
  - **Totals:** ROI covers cash-funded wagers only, and bonus face value
    never counts as cash staked. Bonus stake used, cash won from bonus
    bets, unresolved outcomes and pending bonus stake are separate. *Net
    cash incl. bonus winnings* is labelled with its components.
  - **Compatibility:** existing records and checksums are untouched.
    Schema 1 keeps encoding cash spreads (legacy defaults: spread, cash).
    Schema 2 is written only for moneyline or bonus terms and their
    grades. Unknown versions are rejected.
  - **Grades and duplicates:** grades record market, funding and bonus
    outcome, checked against their terms. Duplicate detection includes
    market and funding.
  - **Page and Excel:** market and funding controls, the tie-rule and
    bonus-return choices, and Preview/Save agreement on all of them. A
    bonus-bet totals row. The Excel export has market, funding,
    cash/bonus stake, tie-rule and bonus-outcome columns and the bonus
    Summary rows.
  - **Tests:** `tests/test_bet_journal_moneyline_bonus.py`.

- **Bet Journal: on-demand Excel export** (`bet_journal_export.py`; an
  *Excel export (viewing copy)* section on the Bet Journal page).
  - **Flow:** *Prepare Excel export* builds an .xlsx in memory from one fresh,
    validated read of the journal. *Download Excel* serves it as
    `bet_journal_YYYYMMDD_HHMMSS.xlsx` (Toronto time).
  - **Sheets:**
    - **Summary:** totals from `bet_journal.totals()`, plus what each total
      includes and which wagers are excluded and why;
    - **Wagers:** one row per wager on its current terms; pending,
      unverified and voided wagers have a blank profit, never a settled $0;
    - **History:** every record, with precise timestamps and links.
  - **Integrity:**
    - a corrupt, tampered, stray or broken-link journal refuses the export;
    - a prepared workbook reflects the journal at preparation time (shown on
      the page); after a journal change it is discarded on the page's next
      rerun, and the page asks you to prepare again;
    - nothing is written to disk, and no network is used;
    - it's a viewing copy, not an editable replacement or a backup.
  - **Excel safety:** text is never stored as a formula (`=`, `+`, `-` and `@`
    stay literal). IDs and timestamps are text; amounts, ROI, handicap and
    odds are numeric with CAD, percentage and signed formats.
  - **Dependency:** `openpyxl==3.1.5` added to `requirements.txt`.
  - **Tests:** `tests/test_bet_journal_export.py` reads the generated .xlsx
    independently, with zipfile and ElementTree.

- **Fix: Bet Journal rejected its own timestamps** ("Not saved: new wager:
  '…:38.086319Z' is not a UTC timestamp…").
  - **Cause:** `bet_journal.iso()` writes `recorded_at` with microseconds
    whenever the clock has them, as designed for ordering. Validation used
    the snapshot/Ontario parser, which accepts whole seconds only, so every
    real-clock save failed.
  - **Fix:** a journal-only parser (`bet_journal.parse_ts`) accepts
    canonical whole-second and six-digit microsecond UTC timestamps, and
    only the spelling the journal writes. Validation, the page and display
    use it. History is now ordered by time rather than text, since
    `…:38.5Z` sorts before `…:38Z` as text.
  - **Preserved:** existing records, checksums, record IDs, duplicate
    protection, history chains, settlement and the placement rules
    (before kickoff, not after recording). Snapshot and Ontario capture
    validation is unchanged.
  - **Tests:** real-clock regression tests: Preview → Save, reload and
    validation, late entry, amendment/void/grade/invalidation, same-second
    ordering, and rejection of non-canonical forms.
- **Fix: model decisions wait for their decision time** ("Was waiting
  worthwhile?").
  - **Bug:** a decision time still in the future was filled with the latest
    snapshot that existed. After the Week 6 Wednesday capture, the Sunday
    decision showed as available for 13 games, and 4 counted as "signal
    retained".
  - **Fix:** it is now `decision_time_not_reached`, and the game is a new
    `pending` class, kept out of "model decisions at both times".
  - **Unchanged:** permanent unavailability (kicked off before the decision
    time, no snapshot or line) still wins, and past decision times are not
    affected.
- **Frozen Performance page** (`pages/9_Frozen_Performance.py`, display logic
  `frozen_performance.py`). A read-only app view of the frozen-pregame spread
  report: the spread model scored only on predictions frozen before kickoff.
  - **No recomputation:** every number comes from the report.
  - **Report split:** `scripts/frozen_spread_report.py` gained
    `load_sources` / `report_from_sources` so the page can cache file loading
    and still evaluate the date on every render. `build_report`'s behaviour is
    unchanged.
  - **Controls:** season and week, defaulting to the latest fully completed
    week. Partly played weeks, pending games and unplayed games are labelled
    provisional. Snapshot choice: latest pregame (default) or earliest.
  - **Separate summaries:**
    - all frozen predictions;
    - the frozen bet-signal subset;
    - the automated-Ontario market comparison on its own matched sample;
    - simulated returns (observed archived price vs an assumed −110
      scenario, labelled "not placed").

    Actual wagers stay in the Bet Journal.
  - **Coverage and results:** coverage counts include the reasons for missing
    snapshots. A per-game table shows the frozen underdog, signed handicap,
    probability, signal, final score, win/loss/push and status.
  - **Caching and failures:** sources are cached on a SHA-256 fingerprint of
    every snapshot, capture and the schedule. Corrupt artifacts stop the page
    with an integrity error, and so do unreadable files (e.g. permission denied,
    previously a crash) and a schedule missing `game_type` or the other columns
    the season/week choices need (rejected before any control is built). The
    page writes nothing and makes no API calls.
  - **Tests:** `tests/test_frozen_performance.py`.
  - **Sample:** 2026 snapshots began Oct 5, so the evaluable sample is small.
    The page makes no accuracy or profitability claim.

- **"Was waiting worthwhile?" for Ontario spreads**
  (`scripts/ontario_waiting_report.py`; a new section of the Ontario Line
  Timing page). Read-only, automated Ontario feeds only.
  - **Offers:** exactly Phase 2's eligible Wednesday/Sunday pairs for the same
    game, team and sportsbook, plus a check against the current schedule.
    - **Fails closed on the current kickoff.** A pair needs exactly one
      schedule row for the game, with a usable kickoff, and both quotes
      strictly before it.
    - A missing game, duplicated rows (even identical ones), or a missing,
      unreadable or DST-ambiguous kickoff leave the team-sides in the gaps as
      `current_kickoff_unavailable_*`. A kickoff moved to or before a quote is
      `timing_unverified_current_kickoff`.
    - The capture's older kickoff is never substituted, and excluded rows
      count in no rollup.
    - Model decisions need the same unique kickoff.
    - Nothing is substituted: no other week's Wednesday, ad-hoc capture,
      manual entry, US feed or other sportsbook. Missing pairs stay listed
      with reasons (week 5 of 2026 has no scheduled Wednesday capture).
    - Each offer shows its timestamps, signed handicap, price and payoff
      classification.
    - Once a final score is evidenced, each offer is settled with
      `betting_log.spread_result` and `bet_journal.win_profit` at 100 units,
      with the Sunday − Wednesday profit difference. This is labelled as
      hindsight.
  - **Model decisions:** one row per game.
    - At each intended decision time (Wednesday 12:00 and Sunday 09:00
      Toronto), the latest validated snapshot captured at or before it and
      before kickoff is used, chosen before its prediction is read.
    - Sunday is classified as signal retained, withdrawn, added, side
      changed, neither, or unavailable. A missing snapshot, line or
      probability is unavailable, never "no signal".
    - Probabilities are compared only for the same team and handicap.
  - **Summary:** separate denominators: games with paired captures,
    game/sportsbook pairs, team-side comparisons, games with decisions at
    both times, and settled comparisons. Nothing is summed into one return,
    and no betting time, staking or parlay is recommended. The injury
    limitation is stated.
  - **Integrity:** corrupt or unreadable inputs stop the analysis with an
    integrity error. The page's capture fingerprint now reports an
    unreadable file as an integrity error instead of crashing. File inputs
    are cached on content fingerprints, and results are recomputed per
    render.
  - **Other changes:** `pregame_snapshots.select_from` (select_captures' rule
    on already-loaded rows; behaviour unchanged). The CLI writes only to the
    git-ignored `reports/ontario_waiting/`.
- **Frozen-pregame spread performance report** (`scripts/frozen_spread_report.py`,
  read-only CLI). It evaluates the spread model only on predictions frozen
  before kickoff in `data_files/pregame_snapshots/`, never on regenerated
  ones.
  - **Selection:** one snapshot per game, reusing
    `pregame_snapshots.select_captures` (latest eligible by default,
    `--which earliest` as an alternative).
    - The capture is chosen before its prediction is inspected: no fallback to
      an earlier probability or signal, and no backfill.
    - Each game records its run ID, capture time, code revision and frozen
      terms.
    - A capture not before the *current* scheduled kickoff is
      `timing_unverified` and excluded.
  - **Outcomes:** the frozen underdog at its frozen handicap, settled with
    `betting_log.spread_result` once `bet_journal.completion_evidence` finds
    conservative evidence. Pending, pushes and invalid results are reported
    separately.
  - **Coverage:** every selected game has a status.
    - Completed games without a selected snapshot are listed separately.
      They're computed from the schedule over the scope, independently of the
      selections, and split into: no snapshot history, before the first
      snapshot, after it (a gap), and kickoff unavailable.
    - Scope is `--season`, or by default the latest schedule season. Selected
      games outside it are counted.
  - **`--as-of`:** an outcome cutoff over the current files, not a historical
    reconstruction. The report says so.
  - **Metrics:** Brier score and log loss vs a 50% baseline on the same
    non-push games, with clipping documented. Calibration bins, grouping by
    season/week and code revision, and signal-only results as a subset.
  - **Market comparison:** only automated Ontario sportsbook feeds in
    archived, validated captures taken before kickoff.
    - The US FanDuel reference feed and manual quotes are excluded before any
      line, probability or price is read.
    - The quote must match the exact frozen handicap, with both prices
      devigged. Otherwise "unavailable", with the reason.
    - The spread tracker's re-fetchable cache isn't used.
  - **Simulated returns:** signal-only, 100 units per bet, at the
    lowest-paying matched archived Ontario-feed price. Assumed −110 is a separate,
    labelled scenario. The ROI denominator and push treatment are stated, and
    actual wagers are never read.
  - **Output:** deterministic JSON and per-game CSV in the git-ignored
    `reports/frozen_spread/`.
    - Output paths overlapping `data_files/` (including through symlinks,
      `..` and letter case) are refused.
    - Corrupt snapshots, captures or schedules fail the run.
    - `read_games_csv` keeps identifier and hash columns as text.
  - **Docs and tests:** `docs/FROZEN_SPREAD_REPORT.md` and
    `tests/test_frozen_spread_report.py` (synthetic snapshots, temporary
    schedules and captures, network blocked).

- **`player_props/backtest.py` diagnostics are safe under Windows redirected
  output.**
  - **Before:** with stdout redirected on Windows (e.g. the app started with
    its output sent to a log file), Python writes in cp1252, and the module's
    emoji status lines raised `UnicodeEncodeError`:
    - The first one skipped the pre-aggregated stats entirely.
    - A successful play-by-play collection fell into its failure path.
    - The failure report itself then crashed the Model Performance page,
      hiding the original error.
  - **Now:**
    - The module's console output is plain ASCII, written through `_say()`.
    - On an encoding error, `_say()` writes the message again with the
      characters escaped (e.g. text from an exception, a path or a player's
      name).
    - If the output is unavailable (`OSError`/`ValueError`, e.g. a closed
      stream), it drops the message.
    - Original errors are reported and returned unchanged.
  - **Scope:**
    - This covers `backtest.py`'s diagnostics only, not every output path of
      the Model Performance page or the app.
    - Predictions, calculations, caching, settlement and the displayed
      results are unchanged.
    - `PYTHONUTF8=1` is no longer needed for this module's diagnostics.
  - **Tests:** regression tests run the success paths, the original-error
    path, the weekly check and the history loaders against a strict cp1252
    stream.

- **Windows dispatcher for Ontario spread captures** (installed separately;
  nothing is registered by default).
  - **Why:** GitHub's cron ran 4.7–5 h late on 2026-10-07 and missed the
    Wednesday slot.
  - **What it does:** `scripts/ontario_dispatch.py`, run by Windows Task
    Scheduler, dispatches the **existing** Ontario Spread Capture workflow on
    `main` with `slot=wednesday_noon` or `slot=sunday_morning`:
    - at Wed 12:05 and Sun 09:05 Toronto, retrying at 12:30 / 09:30;
    - with a status report at 13:00 / 10:00;
    - with recovery at startup, on resume from sleep, on leaving Modern
      Standby and when a network connects.
  - **Guards:**
    - **Where and when:** it dispatches only inside the slot window, and
      not in its last 15 minutes.
    - **Rechecked just before sending:** the clock is read again after the
      GitHub checks and immediately before the request is recorded and sent.
      If the window closed or the cutoff passed during the checks, nothing is
      sent.
    - **Only if needed:** only without a validated usable capture on main,
      read at one pinned commit, and without a queued or running run.
    - **Limits:** at most 2 runs per slot (from GitHub) and at most 3
      requests per slot from the computer.
  - **Dispatch requests are never retried automatically.**
    - A timeout or 5xx is reported as uncertain.
    - A `workflow_dispatch` run created afterwards is followed only as a
      *candidate*. GitHub returns no run ID, so it may be another dispatch's
      run.
    - Dispatch is not exactly-once.
  - **Outside a window:** a missed slot is logged once, never captured ad
    hoc.
  - **Outcomes:** dispatch accepted, run queued, run failed and capture
    persisted are distinct; only the last is success.
  - **Auth and logs:**
    - The token comes from Windows Credential Manager.
    - `ODDS_API_KEY` is kept out of the dispatcher process.
    - Logs are redacted and kept outside AppData, because Store Python
      redirects AppData writes.
  - **Local state:**
    - The ledger is written atomically. A corrupt ledger is moved aside and
      reported.
    - **A lock left by a killed run** is taken over only if three things
      hold:
      - its owner record is complete;
      - it was taken on this computer;
      - it predates the last start, or its process no longer exists.
    - **Takeover is guarded:** it happens under a separate recovery lock,
      after re-checking the lock's token.
    - **Never removed automatically:** locks with an unknown or corrupt
      owner, or from another host. They are reported with what to check
      before deleting them.
  - **Scripts:** `scripts/windows/` has idempotent install (with
    `-PlanOnly`), status, uninstall and token scripts.
    - The installer checks the real interpreter with `ontario_dispatch.py
      runtime` (python.org CPython 3.12–3.14, `requirements-dispatch.txt`).
    - It verifies "Log on as a batch job" from the effective policy.
  - **Backup:** GitHub cron stays configured.
  - **Docs:** `docs/ONTARIO_DISPATCH_WINDOWS.md` covers setup, the
    unattended test, the task settings, recovery behaviour, the power,
    network and battery limitations, and what is not yet verified.

- **Ontario capture: no second paid capture of a slot from a queued run.**
  - **Before:** a run that waited behind another one checked out the older
    commit it was triggered on, so it didn't see the first run's capture. It
    could spend a credit capturing the slot again.
  - **Fix:**
    - The workflow now fast-forwards to the latest `main` before capturing.
      If it can't fetch or fast-forward, the run fails before the capture
      step, so no Odds API request is made from a stale commit. Upload and
      the persistence report still run.
    - `capture()` checks the slot again while holding the slot lock, before
      the credit check.

- **Ontario expected-slot coverage.** Coverage now describes the expected
  Wednesday 12:00 and Sunday 09:00 (Toronto) slots, independently of what was
  captured or which weeks the comparison has.
  - **Tracking start:** a fixed `TRACKING_START` of 2026-10-07, no longer
    inferred from the first capture. Nothing earlier is listed.
  - **States:** pending, awaiting capture, missed or captured (on time or
    late). They use Phase 1's own inclusive window rule, now shared with
    `capture`.
  - **Display range:** from the tracking start through the next Sunday slot,
    with history kept. Windows follow DST.
  - **What counts:** only usable scheduled captures fill a slot. Empty and
    US-only captures are listed as evidence, and `ad_hoc` captures and
    manual quotes never count.
  - **One calculation** (`ontario_spreads.expected_coverage`) serves both
    `python ontario_spreads.py coverage` and a new, always-visible
    "Scheduled slots" table on the Ontario Line Timing page. The page's week
    slot cards and row reasons use it too, replacing its own window logic.
    States are recomputed from the clock on every rerun.
  - **Today's data:** 2026-10-07 Wednesday shows as **missed** and
    2026-10-11 Sunday as **pending**.
  - **CLI:** `--days` now defaults to the full tracked history; the workflow
    still passes `--days 28`.
  - **Comparison unchanged:** comparison rows and metrics still use only
    eligible observations.
  - **Capture validation:** a capture's slot block must now match its
    calendar slot exactly (slot ID, intended time, delay inside the window,
    on-time/late, `captured_at`). An inconsistent block is an integrity error,
    so no capture can fill another slot. Capture eligibility itself is
    unchanged.
  - **What "missed" means:** rows are configured capture opportunities, not
    evidence that games or markets existed. **Missed** means no usable
    scheduled capture was stored; it doesn't by itself show a failed run (for
    example outside the season, or with no games in scope).
- **Pre-game snapshot `select` CSV: identifiers and hashes read as text.**
  - **The crash:** pandas 2.3.3's C CSV parser segfaults on hex strings shaped
    like scientific notation (pandas-dev/pandas#62617, #62740). The `select`
    CSV's `code_revision` is the git SHA, which is `GITHUB_SHA` in CI, so a
    commit SHA like `85e6046866697474…` crashed the test run that read the
    CSV back.
  - **The fix:** `pregame_snapshots.read_selection_csv` reads exactly the
    columns in `SELECTION_TEXT_COLUMNS` as text: `run_id`, `game_id`,
    `snapshot_file`, `code_revision`, `config_id`, `feature_set_id` and
    `artifact_sha256`. Numeric columns keep their types, and no provenance
    value changes.
  - **Tests:** they pin `GITHUB_SHA`, so they no longer depend on the commit
    being tested. Regression tests read the reported dangerous strings in a
    bounded child process and check they come back as the exact text.
  - **Scope:** no other committed or generated CSV has hash columns, so
    other readers are unchanged. External readers of `picks.csv` should keep
    these columns as text (see `docs/PREGAME_SNAPSHOTS.md`).

- **Ontario Line Timing: no crash when no comparable week exists.**
  - **The crash:** the first stored capture (2026-10-07) was `ad_hoc`,
    because the scheduled Wednesday run arrived after its window. The report
    then had no weeks, and the page raised `IndexError` while choosing a
    default season.
  - **Now:** it shows "No Wednesday/Sunday comparison yet", and a new
    "Captures stored but not compared" section lists such captures. Each row
    has the actual capture time, the reason, and fresh-quote counts given
    separately for Ontario feeds and the US reference. Ad-hoc captures still
    never fill a slot or enter the comparison.
  - **Test fix:** a manual-entry test assumed `data_files/ontario_spreads/`
    never exists; it now checks that the real files are unchanged.
  - **Still open:** slot status needs a report week, so the missed October 7
    noon slot isn't shown as missed and October 11 can't show as pending yet.
- **Bet journal.** A new **Bet Journal** page records single-game spread
  wagers actually placed at Ontario sportsbooks, on their accepted terms. It
  records wagers and never places them. See `docs/BET_JOURNAL.md`.
  - **Entry:** the sportsbook, game and team, the signed spread, accepted
    American odds, a positive CAD stake, and the placement time in
    America/Toronto, plus an optional reference and note.
    - **Placement time:** it must not be in the future and must be strictly
      before kickoff. Late entry of an earlier pregame wager is allowed.
      DST-ambiguous and nonexistent times are rejected.
  - **Preview, then Save:** Preview writes nothing. Save writes exactly the
    previewed terms, and any change in between requires a new preview.
  - **Records** (`bet_journal.py`) are append-only, checksummed JSON records
    in `data_files/bet_journal/`, which is git-ignored because it's personal
    data.
    - **Kinds:** wager, amendment, void, grade and invalidation.
    - **Corrections:** amendments and voids need a reason. The originals stay
      in history and drop out of the totals.
    - **Explicit links:** history is linked explicitly, never ordered by
      timestamp or file name.
      - **Terms chain:** `wager → amendment → … [→ void]`, through
        `previous`.
      - **Settlement chain:** grades and invalidations, through
        `supersedes`.
      - **Terminal transitions:** settlements not timestamped strictly before
        a void, or recorded after their terms were superseded, are rejected.
      - **Saves:** must name the current tip, checked under the lock, so a
        correction based on an outdated Preview is refused.
  - **Duplicates:** reruns, double clicks and concurrent saves, including
    from separate processes, can't add a duplicate. A genuine second wager on
    identical terms needs explicit confirmation, and so do amended terms that
    match another current wager. The confirmed set is re-checked under the
    journal lock.
  - **Grading:** it runs only from the **Grade settled wagers** button and
    reuses the reviewed spread rule, now exposed as
    `betting_log.spread_result`.
    - **Evidence:** nflverse has no final-status flag, so a game is graded
      only on conservative evidence of completion:
      - one schedule row;
      - whole, non-negative scores that aren't 0–0;
      - `result` = home − away, `total` = home + away, `overtime` 0 or 1;
      - a game day before today in Toronto.

      Live partial scores, placeholders, inconsistent fields and
      duplicated rows stay pending.
    - **Profit:** scaled to the CAD stake at the accepted odds. Wins earn net
      winnings, losses lose the stake and pushes earn $0.
    - **Changes:** a score correction or amended terms writes a superseding
      grade. Evidence that no longer supports a grade writes an
      **invalidation**: the wager shows as *unverified* and leaves profit and
      ROI, and restored evidence grades it again. Earlier grades stay in
      history, and grading again with unchanged data writes nothing.
    - **Provenance:** each grade records its evidence, the SHA-256 of the
      exact schedule bytes parsed, and the exact terms record graded.
  - **Totals:** W-L-P, net profit, pending stake, and ROI over graded
    current wagers. Pushes are included; pending, unverified and voided
    wagers are excluded.
  - **Integrity:** a damaged or ambiguous history shows an integrity error
    instead of totals, and saving and grading refuse. That includes
    malformed records, forks, cycles, dangling links, duplicate voids,
    out-of-order timestamps, invalid confirmations of non-duplicate wagers,
    settlements after voids or superseding amendments, and grades that don't
    follow from their terms and evidence.
  - **CLI:** `python bet_journal.py validate [--dir PATH]` checks a journal,
    e.g. after restoring a backup. The guide has backup and restore steps.
  - **Page details:**
    - **Tabs:** the selected tab stays selected across reruns.
    - **Dollar amounts:** they're escaped, so two amounts in one message
      don't render as LaTeX math.
    - **Selection labels:** the wager and game selectors show labels that
      don't change when a wager is amended. A stale selection is refused
      with "choose … again" rather than acted on.
- **Shared lock fix (Windows/Linux).** `SlotLock` is shared by Ontario captures, manual
  quotes and the bet journal. It could leave its own lock file behind when
  deleting it coincided with another process reading it to report the
  owner, and then every later save was reported as busy. Creating a lock
  while a deletion was pending also raised a raw `PermissionError`. Both
  transient cases now retry for a bounded time (at most 5 s). Contention is
  reported only for a readable lock record; unrelated permission errors are
  not described as another writer. A lock is still never deleted on another
  holder's behalf.

- **FanDuel Ontario manual-quote form.** The Ontario Line Timing page has a
  "Record a FanDuel Ontario quote (manual observation)" section.
  - **Entry:** pick an upcoming game and team, then enter the bettor-facing
    signed spread, American odds (and optionally the opponent's) and the
    observation date and time in America/Toronto. DST-ambiguous and
    nonexistent times are rejected.
  - **Preview:** e.g. "Arizona Cardinals +4.5 at −110", labelled a manually
    observed quote, not a placed bet. It shows whether the time falls in the
    game's Wednesday or Sunday slot window, another week's slot or no slot.
  - **Save:** happens only on an explicit Save, a button of the same form.
    It writes exactly the previewed values: any change after Preview,
    including the note, requires a new preview. It re-validates with the
    current clock, so a game that kicks off between Preview and Save is
    refused. Reruns, double clicks and identical resubmissions write at most
    one file. Quote identity is game, team, observation time, handicap,
    price and opponent price; the note and the submission don't count.
  - **Locks:** they record their owner (process ID and host). A busy lock is
    reported with its owner and age as "in progress" or "older than
    expected", noting that age alone doesn't show the owner has stopped.
    Manual deletion is advised only after confirming that recorded process
    on that host is no longer running; with an unknown or unreadable owner,
    the message says to investigate first. Locks are removed after a failed
    save and are never deleted automatically or on someone else's behalf.
    A valid existing lock is reported as contention; unrelated permission
    errors are not mislabeled as a busy lock. Afterwards the page
    switches to the manual view, with its filters cleared. The view's new
    "Recorded manual observations" table shows each observation's times,
    source and whether it's in the comparison.
  - **Shared code:** `ontario_spreads.manual_quote` is now split into
    `prepare_manual_quote` (validate only) and `save_manual_quote`, which
    runs the duplicate check and the write under an exclusive lock. The CLI
    and the form share them, and CLI behaviour is unchanged.
  - **Also:** manual rows whose Sunday slot hasn't closed now read "Sunday
    comparison pending". No API calls, captures, recommendations or bet
    records.

- **Ontario Line Timing page (Phase 3).** New read-only Streamlit page
  `pages/7_Ontario_Line_Timing.py`, registered in `predictions.py`'s
  navigation, with display logic in `ontario_line_timing.py`.
  - **Data:** it builds the Wednesday-vs-Sunday comparison in memory from the
    validated captures. The cache is keyed on a SHA-256 fingerprint of every
    source file, so a new or changed capture is never hidden by a stale
    cache. It makes no API calls.
  - **Views:** Ontario automated feeds by default; manual FanDuel Ontario and
    the US FanDuel reference are separate, opt-in views. The US view labels
    them "US reference quotes; not verified as available in Ontario."
  - **Filters:** season, week, sportsbook, game and team. The default is the
    most recent captured week.
  - **Table:** signed handicaps and prices, break-even percentages, key
    numbers 3 and 7, payoff-dominance outcomes, and Toronto-time capture and
    provider-update times with freshness.
  - **Slots:** captured, pending, missed or empty. Wednesday-only weeks show
    "Sunday comparison pending".
  - **Empty or corrupt data:** with no captures it says "No observations
    yet" and explains the schedule. A file that fails validation shows an
    integrity error instead of partial or substituted data.

  Trade-offs aren't ranked; Sunday isn't a closing line; no best time, edge
  or ROI is claimed.

- **Ontario spread comparison report (Phase 2).** New
  `scripts/ontario_spread_report.py` is a read-only report comparing each
  side's Wednesday-noon quote with its Sunday-morning quote, for the same NFL
  week, game, sportsbook, jurisdiction and team.
  - **Input:** Phase 1-validated captures only. Each slot uses its earliest
    usable capture; empty, US-only, `ad_hoc` and duplicate captures are never
    used, and missing quotes are never filled from elsewhere. A Sunday pairs
    only with the Wednesday of the same NFL week (four days earlier); an
    earlier Wednesday never stands in for a missing one.
  - **Per side:** both quotes with provenance, spread change, break-even
    change (pushes excluded), key numbers 3 and 7 crossed, landed on or left,
    and an outcome: unchanged, equivalent, Sunday or Wednesday dominates, or
    trade-off. The outcome is decided by settling both bets for every integer
    margin (win, push, loss).
    Only fresh, valid, in-window quotes from before kickoff are compared;
    Thursday games and early-Sunday kickoffs stay unmatched.
  - **Groups:** Ontario feeds by default. The US FanDuel reference and manual
    FanDuel Ontario quotes are separate, opt-in groups. A manual quote is
    used only if observed inside the game week's intended Wednesday or Sunday
    slot (four days apart); quotes from other calendar slots, such as an
    earlier Wednesday, are noted and never substituted. Games whose recorded
    kickoffs imply different weeks (e.g. postponed), or whose week can't be
    pinned to one Sunday, are reported as unmatched rather than guessed.
  - **Rollups** count sides, game/book pairs and games separately, and
    declare no best book.
  - **Output:** deterministic JSON/CSV in the git-ignored
    `reports/ontario_spreads/`, written atomically. Locations that resolve
    into source data are refused, including via symlinks or alternate paths.
    An empty history gives a `no_observations_yet` report.

  This describes movement only: Sunday isn't called a closing line, and
  there's no best-time or ROI claim. Docs: "Comparison report" in
  `docs/ONTARIO_SPREAD_TRACKING.md`.

- **Ontario sportsbook spread captures (Phase 1: collection and storage).**
  New `ontario_spreads.py` and `ontario-spread-capture.yml` record NFL
  spreads and prices from Ontario feeds at Wednesday 12:00 and Sunday 09:00
  America/Toronto. The feeds are `betano_ca_on`, `betmgm_ca_on`,
  `betrivers_ca_on`, `pointsbetca`, `proline_ca_on` and
  `sportsinteraction_ca_on`, with `bet99_ca_on` opt-in as paid tier.
  - **Request:** one bulk `/odds` call per slot, by bookmaker key (1 credit),
    after a free credit check against the shared reserve. No key, no calls;
    the key is redacted everywhere.
  - **Storage:** each capture is a new immutable, checksummed file with run
    ID, slot (intended vs actual time, on time or late), code revision,
    request scope and usage headers, the sanitized provider response, and
    per-book quotes with jurisdiction and provider timestamps.
  - **Explicit gaps:** missing, stale and invalid quotes are labelled and
    never carried forward. Events match on teams **and** kickoff; games
    underway are excluded.
  - **FanDuel:** the API's `fanduel` is labelled a US reference. FanDuel
    Ontario comes only from the new `manual-quote` command (source
    `manual`).
  - **Model link:** each quote links to the latest pregame model snapshot
    captured no later than the quote's own provider timestamp; without one,
    no time-aligned link is claimed. No probabilities are copied.
  - **Windows:** Wednesday 12:00–15:00 and Sunday 09:00–11:00 Toronto, so a
    much later run can't be labelled as the slot. An empty response, or one
    where only the US FanDuel reference quoted, is kept as evidence but
    leaves the slot open for a retry inside the window.
  - **Persistence:** capture files are uploaded as a run artifact even when
    the capture or commit fails. A final step fails the run if any capture
    didn't reach `main`, naming the artifact that holds it.
  - **Failures** fail the workflow. `spread_tracker.py` and its files are
    unchanged.

  Docs: `docs/ONTARIO_SPREAD_TRACKING.md`.

- **Model Performance page: default week.** The page treated
  `selected_season == 2025` as the current season, so from 2026 it offered
  Weeks 1–18 and opened on Week 18, a week not yet played ("No play-by-play
  data found"). Weeks now come from completed games in
  `nfl_games_historical.csv` (`season_utils.completed_weeks`). Only
  regular-season weeks with results are listed, and the default is the
  latest week whose games have all finished. A week with a Monday night game
  still to come isn't picked, or cached, early. The season defaults to the
  newest one with results. If the selected season has none (preseason), the
  page says so and skips the week analyses, instead of opening an unplayed
  week. The unused date-based `get_current_nfl_week` /
  `get_season_for_week` helpers were removed.
- **Player-prop accuracy results are only cached when final.**
  `player_props/backtest.py` saved every analysis it ran, including a week
  with a game still to play or play-by-play missing a game. The page then
  served that saved file as the week's result. Completeness is now checked
  game by game: `collect_actual_results` records, for each play-by-play
  `game_id`, whether its `END GAME` play is present and the score on that
  play, both totals read from the same row (`pbp_game_completion`).
  Per-column maxima aren't used: running totals aren't monotonic (a reversed
  score dips and recovers; 737 of 1,680 games in 2020–2025 do), so maxima
  could pair scores that never stood together and hide a corrected scoring
  event. `week_results_status` calls results final
  only when every scheduled regular-season game has final scores in the
  schedule, has play-by-play ending in `END GAME`, and that play's score
  matches the schedule. A game with no `END GAME` play, or with ambiguous or
  null `END GAME` scores, stays provisional. Seeing every team in the stats isn't
  enough, since a game whose play-by-play stops early still lists both
  teams. Provisional results are shown with a warning and never saved.
  Checked against the local 2020–2025 play-by-play, all 107 regular-season
  weeks verify as final. Limitations: pre-aggregated weekly stats have no
  game IDs, so their completeness can't be shown and results from them are
  never cached (they're recalculated on each view). The check can't detect
  non-scoring plays missing from the middle of a game whose `END GAME` play
  and final score are both present.
  Saved files now record their season, and
  `load_accuracy_results_for_week` only serves files for the requested
  season. Older files that don't record a season are recalculated rather
  than served. The loader also picked the "newest" file by time of day
  alone; it now uses the date too. `scripts/run_weekly_backtest.py` no
  longer saves a second copy of each result.
- **Byte-exact manifest-hashed files.** `.gitattributes` marks
  `data_files/nfl_games_historical.csv` and
  `nfl_games_historical_with_predictions.csv` `-text`. Git for Windows'
  `core.autocrlf=true` had converted them to CRLF on checkout, so committed,
  correct data failed the pregame-snapshot provenance check. Verification is
  unchanged and still hashes exact bytes. Checkouts made before the rule
  need a one-time refresh (see "Line endings" in
  `docs/PREGAME_SNAPSHOTS.md`); new attributes alone don't rewrite existing
  files.
- **Pregame spread snapshots.** `nfl-gather-data.py` now ends a successful
  run by writing `data_files/pipeline_run_manifest.json`, recording the code
  revision, config and feature-set ids, cutoffs, a unique run ID assigned
  once, and SHA-256 of the exact schedule bytes it parsed (read once) and
  the predictions it wrote. A stale manifest is deleted when a run starts.
  The nightly job then runs `pregame_snapshots.py capture`, which reads each
  input once, verifies it against the manifest, and writes an immutable,
  schema-validated `data_files/pregame_snapshots/<run_id>.json` for the
  eligible upcoming games: not completed, with a usable kickoff time
  strictly after the capture time. That covers valid predictions with or
  without a bet signal, and lineless or pick'em games with explicit
  statuses and no probability. Kickoffs are US Eastern converted to UTC;
  missing, ambiguous and nonexistent times are skipped with a reason.
  Outcomes are never stored. Files are created with an exclusive link; a
  retry is accepted only if it validates and matches every provenance field,
  so a changed provenance is refused even with identical predictions. An
  unkeyed whole-payload checksum detects accidental changes, but doesn't
  authenticate a file. A failed capture marks the nightly run failed after
  the data is published. `pregame_snapshots.py select` returns the earliest
  or latest eligible capture per game, read-only and after validating every
  file. `check_pipeline_outputs.py` verifies the manifest. Training,
  calibration, thresholds and predictions are unchanged, and the
  determinism check is unaffected. Docs: `docs/PREGAME_SNAPSHOTS.md`.
- `nfl-gather-data.py`'s check that played games are in (season, week)
  order is now an explicit `ValueError` (`require_chronological`) naming the
  first out-of-order game. It was an `assert`, which `python -O` silently
  skips; the temporal train/validation/test split depends on row order.
  It first rejects any played game whose season or week is missing,
  non-numeric, infinite (`inf`, `-inf`, `"inf"`) or fractional (e.g. week
  1.5), or a missing column, naming the game, field and original value.
  Missing values compare False and slipped through the order check; `inf`
  sorted after every real week; an out-of-order `inf` crashed the message
  with `OverflowError`. Values must also fall inside documented bounds
  (`SEASON_BOUNDS = (1920, 2100)`, `WEEK_BOUNDS = (1, 22)`, weeks 1-18
  regular season plus playoffs through Super Bowl week 22), checked before
  the int64 conversion. Values of 2**63 or more (e.g. season 1e20) used to
  wrap silently, accepting out-of-order input. Numeric strings ("2020",
  "1.0"), same-week ties, playoff weeks and season boundaries (week resetting
  to 1) still pass.
  New tests cover genuine pick'em lines (`spread_line == 0`): no spread
  probability, EV, edge or recommendation, never in training, and
  `check_pipeline_outputs.py` fails if one carries a probability or signal.
  No model settings or predictions changed.
- **Production team features no longer leak results.** Team aggregates
  move to a shared `team_features.py`, used by both `nfl-gather-data.py`
  and the rolling backtest. Each game now uses only completed games from
  strictly earlier weeks. Two production bugs are removed:
  - Full-history team rates (FavoredPct, SpreadCoveredPct, Over/Under/
    TotalHitPct) averaged every played game, including the game being
    predicted. They differed from a correct earlier-games-only value by up
    to 0.91.
  - The per-row loops counted unplayed earlier games as 0-results for games
    2+ weeks out, and ordered "last 3" by row position.

  Feature names, home/away definitions and the 0 cold-start value are
  unchanged. Games without a valid spread line (193 upcoming games) now get
  no spread probability and no signal; 58 lineless games had carried a
  spread signal. Measured on the production validation/test split, accuracy
  barely changes and is still no better than a 50% guess (test Brier 0.2527
  → 0.2539, interval vs 50% includes 0). Model, calibration, thresholds and
  feature selection are unchanged.
- Added `scripts/rolling_spread_backtest.py`, a read-only weekly backtest
  of the spread model. Each week it retrains using only earlier games, with
  team stats built from earlier games only, and keeps the fit and calibration
  periods separate. It compares the model with a 50% guess, devigged closing
  odds and a logistic model on |spread|. Results are reported per season with
  week-resampled confidence intervals, and frozen pregame predictions are
  scored separately. See `docs/ROLLING_SPREAD_BACKTEST.md`. Production
  training and predictions are unchanged.
- Spread Value Finder and model line shopping now use the offered American
  price for break-even probability. A -110 quote requires 52.38%, not the
  margin-free 50%. The page labels this column "Break-even %".
- Spread recommendations freeze the team handicap, assumed -110 odds and
  odds source. Settlement uses final scores and the original bet, never
  the latest line or cover labels. Legacy rows use their recorded nflverse
  line and a -110 assumption, labelled in `odds_source`. Legacy "Pick" rows
  (logged before a line existed, no team recorded) are marked `unresolved`,
  excluded from every total and listed in `spread_performance.json` under
  `unresolved_games`; no team is inferred. `python betting_log.py --regrade`
  corrects settled results and appends each change to
  `data_files/settlement_corrections_YYYYMMDD.csv`. Profits are per $100 risk
  at recorded odds. Repeat logging deduplicates settled rows too, and games
  without a posted line are no longer logged.
- 2 Oct regrade: NE +3.5 (wk 1) push -> win and TEN +5.5 (wk 3) loss -> win
  (graded against the recorded line, not the later closing line); SEA@ARI
  (wk 2) and PIT@CLE (wk 4) "Pick" rows -> unresolved. Season record moves
  from 10-8-3 to 12-6-2 with 2 unresolved.
- Regrades always leave an audit record. `grade_pending(regrade=True)` now
  defaults to a dated `settlement_corrections_YYYYMMDD.csv` beside the log
  when no audit path is given; before, corrections were silently dropped.
  The audit is written before the log, and each file is replaced atomically.
  If the audit write fails the log is unchanged; if the log write fails the
  audit is restored. This covers errors Python can catch, not a crash
  between the two writes. Existing audit entries are kept, and repeating a
  regrade adds no duplicate rows.

## September 2026

- **Prop odds no longer freeze early in the week.** `market_odds.py`'s
  per-week odds file was write-once, and the first fetch of a week happens
  the night after the previous week ends - when books have posted props for
  only a couple of games. Week 3 2026 froze at 2 of 16 games (4 matched
  props) until a manual refresh (204). The file is now filled in per game:
  cached games are never re-fetched, and uncovered upcoming games are re-tried
  on every run. Checked live that a call for a game with no props posted
  costs 0 credits, so re-trying is free and each game is paid for once
  (~4 credits). Week 4 is the first week this applies to. 3 new tests.

- **Player Props "Market Edge" fixed: it compared probabilities at different
  lines.** Refreshing Week 3 prop odds (the write-once snapshot had been taken
  Monday, when only 2 of 16 games had props posted - 4 matched props; the
  refresh gives 204) exposed that `market_edge = prob_over -
  market_implied_prob` subtracts the model's P(over) at its own fixed tier
  line from the book's P(over) at the book's line. 180 of 204 matches had
  different lines, producing bogus 40-50 point "edges" at the top of the page
  (P(Derrick Henry > 75 rush yds)=93% vs the book's P(> 90.5)=50%;
  "UNDER 0.5 receptions" edges from a 3.5-reception model line). Edge is now
  only computed when the lines match (24 props this week); other matches
  keep the book line and price with a blank edge. The reliable (yardage)
  models use round tiers that almost never equal a book line, so the default
  reliable-only view currently shows book lines but no edges. Two existing
  tests encoded the cross-line behavior and were corrected; 2 new tests.

- **Nightly now applies injury adjustments to player props.** Dropped
  `--no-injuries` from `nightly-update.yml`'s `predict.py` step, now that
  the ESPN injury fetch works (below). Out/IR players' props are removed;
  Questionable/Doubtful lower confidence. A local dry run on the Week 3
  slate adjusted 17 players and removed 10 - every removal verified as a
  genuine Out/IR (including Nico Collins and Jaxson Dart, who had been
  published as live props). A failed fetch still degrades to no
  adjustments rather than failing the step. Weather stays off.

- **Player-prop injury data fixed (it had been silently empty).**
  `player_props/injuries.py` scraped ESPN's HTML injury page, which had
  stopped yielding parseable tables - every run returned 0 rows, and the
  committed `espn_injuries.csv` cache was ~8.5 months stale. It now reads
  ESPN's public JSON injuries feed (800 entries across all 32 teams on
  first run). Two follow-on fixes were needed to make that safe:
  (1) the hard-coded Chrome 91 `User-Agent` got a 403 from the JSON
  endpoint, so it was dropped; (2) `find_player_injury`'s last-name
  substring fallback, dormant while the scrape returned nothing, would have
  matched the wrong player for 58 of 317 prop players against a
  league-wide feed (e.g. "Tahj Brooks" -> an IR'd "Jonathon Brooks",
  deleting a healthy player's prediction). Matching is now exact or
  suffix/punctuation-normalized full name only ("James Cook" still matches
  "James Cook III"). Zero false matches on the current slate. 15 new tests
  (`tests/test_injuries.py`). The nightly still runs `--no-injuries`.

- **Fixed `weekly-model-performance.yml`'s silent no-op, caught during
  routine "run app and verify" checks.** The Monday job had shown "Success"
  on every run for weeks, but its inline backtest step called
  `run_weekly_accuracy_check()` with no arguments - the function requires a
  `week` argument, so it raised `TypeError` on every run, immediately
  swallowed by that step's `continue-on-error: true` (the follow-up
  `save_accuracy_results(results)` call was missing its own required `week`
  argument too, so even a working backtest would have failed there
  instead). Net effect: `data_files/spread_performance.json` on `main` was
  stuck at 9/14 numbers through two more weeks of real results, with
  nothing visibly wrong in the Actions UI. New
  `scripts/run_weekly_backtest.py` replaces the inline snippet: finds the
  most recently fully-completed week (mirrors `betting_log.grade_pending`'s
  exact "is this game played" convention - `gameday < today` and a real,
  non-0-0 score) and calls the backtest/save functions correctly. Verified
  end-to-end against real data: genuine accuracy results for the first time
  in weeks (65% overall hit rate, 63.4% on the reliable-only subset, +24.1%
  ROI on fixed-tier props), plus a fresh `spread_performance.json` (8-4-1,
  66.7% win rate, +27.3% ROI season-to-date). 7 new tests (134 total).

- **Spread tracker: fixed a stale-pick bug caught during routine verification.**
  A daily "run the app and verify" check surfaced that a Thursday game
  (`2026_02_DET_BUF`, final BUF 41-31) was still showing up as a "current
  model pick" the next day, in both `model_line_shop.py` and
  `pages/6_Value_Finder.py`. Root cause: `pred_spreadCovered_optimal == 1`
  reflects the model's read at prediction time, not whether the game is
  still upcoming - filtering on it alone can surface an already-played game
  as a live recommendation. New `select_candidate_games()` (shared by the
  CLI and the page, so there's one filter definition, not two that could
  drift) excludes `gameday <= today` by default, mirroring
  `betting_log.append_recommendations`'s exact convention; `--include-played`
  / an "Include played games" checkbox is the explicit escape hatch. 6 new
  tests (127 total).

- **Spread tracker: Value Finder UI page.** New `pages/6_Value_Finder.py`
  surfaces the price-adjusted analysis (below) in the app, two tabs: "Book
  vs Field" (a book picker + week filter over `spread_value_finder.py`) and
  "Model vs Books" (season/week pickers + a "model picks only" toggle over
  `model_line_shop.py`, one table per qualifying game). Kept on its own
  page rather than folded into `pages/5_Spread_Tracker.py` - this is the
  actively-iterated, prescriptive half of the feature ("what to bet" vs.
  "what happened"), so isolating it means refining it can't destabilize the
  already-verified tracker page. Both tabs import the scripts' functions
  directly rather than reimplementing the math. Registered in
  `predictions.py`'s `st.navigation()` list at the same time it was added -
  this app doesn't use Streamlit's automatic `pages/` folder discovery,
  which is exactly what made `5_Spread_Tracker.py` invisible when that
  registration was missed the first time.

- **Spread tracker: model-vs-book line shopping.** New
  `scripts/model_line_shop.py`. A different baseline than the value finder
  below: that one asks whether a book's price is good relative to the other
  tracked books, this one asks whether it's good relative to our OWN
  model's read. The spread model's `prob_underdogCovered` is computed once
  against nflverse's own consensus line - it has zero awareness of
  individual sportsbook lines. Since a book offering more points than that
  line is strictly easier to cover, `extrapolate_prob()` estimates the
  model's implied probability at any book's specific line (same normal
  margin-of-victory approximation, reusing `_normal_cdf` from
  `spread_value_finder.py` plus a new `_normal_ppf` sibling) and compares it
  to that book's own price. Every result is explicitly labeled
  "extrapolated" - not literally the model's output, since it was never
  evaluated at that exact line. Prompted directly by a chat exchange
  confirming "more points at the nflverse baseline increases the odds when
  extrapolated, correct?" and asking for that to be reusable rather than
  hand-computed per book. Live-verified: exactly reproduced hand-computed
  numbers for ARI @ DraftKings (+11.3pt) and FanDuel (+10.1pt), and found
  PlayNow has the single biggest edge on that game (+13.1pt) despite fewer
  points than the field - its plus-money price compensates. 14 new tests
  (121 total).

- **Spread tracker: price-adjusted value finder.** New
  `scripts/spread_value_finder.py`. Prompted by a chat request to recommend
  parlay legs from a book (PlayNow) the spread tracker had flagged as
  divergent from the field - point-only divergence turned out to be the
  wrong question, since a book can move the points and shade the price to
  compensate, netting out to a fair (or worse) bet. Computes a fair win
  probability per side (normal approximation of NFL margin of victory,
  sigma=13.5, evaluated against the field median) and compares it to what
  the book's own price requires to break even, ranked by edge - for any
  tracked book, not just PlayNow. A first manual pass at this math mislabeled
  a side by reading the tracker's internal home-favorite-positive convention
  directly instead of the log's own bettor-facing columns; the shipped
  version reads labels straight from `home_point`/`away_point` and has a
  regression test guarding against that exact bug class. Live-verified
  against real Week 2 2026 data: exactly reproduced the original manual
  analysis, and also surfaced a real edge (+5.6pt) that the existing
  point-only anomaly detector had missed because only the price, not the
  point number, was unusual. 10 new tests (107 total).

- **Season-long spread-line tracker, Phase 3: UI page (all 3 phases done).**
  New `pages/5_Spread_Tracker.py` surfaces the Phase 2 report in the app: a
  per-book season-to-date ranking table + bar chart (which sportsbook is
  closest to nflverse's line), a "closest book" callout, a best-line-per-game
  table, an anomalies table, a week selector, and a raw-log expander. Purely
  a display layer - reads the already-generated JSON/CSV, never calls The
  Odds API itself; shows an explanatory `st.info` when no report exists yet
  (fresh clone, or `ODDS_API_KEY` unset). All 3 phases of
  `docs/MARKET_SPREAD_TRACKER_PLAN.md` are now built.

- **Season-long spread-line tracker, Phase 2: comparison rollup.** New
  `scripts/spread_tracker_report.py` rolls `spread_tracker_log.csv` up into
  `data_files/spread_tracker_report.json`: a per-book season-to-date ranking
  by mean-absolute deviation from nflverse's line (the direction-agnostic
  "which book is closest to nflverse" answer), a best-line-per-game callout
  (which book gives the most points to each side), and a field-median-relative
  anomaly list generalizing the PlayNow divergence first spotted by hand in
  chat (flags any book more than 1.5pt off that game's field median).
  Live-verified against the real Week 3 2026 log: DraftKings ranked closest to
  nflverse (mean|dev|=0.00pt across 16 games), FanDuel furthest of the
  mainstream books (0.34pt), zero anomalies for that single week. Wired into
  `spread-tracker.yml` right after the fetch step. See
  `docs/MARKET_SPREAD_TRACKER_PLAN.md`.

- **Season-long spread-line tracker, Phase 1 (opt-in, off by default).** New
  root-level `spread_tracker.py` pulls real US + Canadian sportsbook
  game-spread lines (DraftKings, FanDuel, BetMGM, BetRivers, PROLINE, Sports
  Interaction, PlayNow, and others) via The Odds API's bulk `/odds` endpoint
  (one call, both `us`/`ca` regions, 2 credits total), normalizes them into
  nflverse's `spread_line` sign convention, and upserts them into a new
  accumulating `data_files/spread_tracker_log.csv` alongside a
  `nflverse_spread_line`/`deviation_pts` comparison - so which sportsbook(s)
  consistently offer a better number than nflverse's line can be answered
  from real season data instead of one-off manual pulls. Reuses the existing
  `ODDS_API_KEY` secret; no new secret needed. Live-verified against real
  Week 3 2026 data (144 game/book rows, sane ±1pt deviations) - and caught a
  real bug along the way: the predictions CSV read used a plain
  `pd.read_csv()`, but `nfl_games_historical_with_predictions.csv` is
  tab-separated despite the `.csv` extension, so every join silently came
  back 100% `NaN` with no error; fixed with `sep='\t'` (matching
  `betting_log.py`'s own read of the same file). See
  `docs/MARKET_SPREAD_TRACKER_PLAN.md`.

- **Market-odds integration complete (Phase 3): DK Pick 6 pre-fill.** The
  Pick 6 Calculator's line input now pre-fills from a real matched
  DraftKings/FanDuel line when one exists for the selected player/stat
  (labeled as a sportsbook line to confirm against the actual Pick 6 board,
  not the Pick 6 number itself - still fully editable). The widget's `key`
  is scoped to `(player, stat)` rather than a fixed string: Streamlit ignores
  a new `value=` once a fixed key already has a session_state entry, so
  switching players wouldn't otherwise refresh the shown default after the
  first render. All 3 phases of `docs/ODDS_API_INTEGRATION_PLAN.md` are now
  built.

- **Market-odds live + Phase 2 UI + a real bug caught and fixed.**
  `ODDS_API_KEY` was added as a GitHub Actions secret; the Sep 16 nightly
  confirmed it live, matching 30 real DraftKings/FanDuel props against the
  cached Week 2 fetch. That run also exposed a bug: `attach_market_odds()`
  gated on `isinstance(prob_over, (int, float))`, but `model.predict_proba()`
  returns `numpy.float32` - not a `float` subclass - so `market_edge` came
  back `NaN` for 29 of the 30 matches. Fixed with a `float()` cast instead of
  a type-gate; regression-tested with an actual `numpy.float32` input.
  Player Props page (Phase 2): new "Market" / "Market Edge" columns on the
  main props table. Sorting/display use a *directional* edge (flipped to
  align with the recommendation) rather than raw `market_edge`, which is
  always in P(over) terms - a strongly negative value on an UNDER pick means
  a *strong* edge in that direction, not a weak one; showing it unflipped
  would have ranked a good UNDER pick as if it were bad. Falls back to the
  existing confidence sort when no market data is matched yet (most rows,
  today - coverage grows as kickoff approaches).

- **Market-odds fetcher, Phase 1 (opt-in, off by default).** New
  `player_props/market_odds.py` pulls real DraftKings/FanDuel lines for the
  four reliable prop types (passing/rushing/receiving yards, receptions) from
  [The Odds API](https://the-odds-api.com/), so `market_edge = model_prob -
  market_implied_prob` can eventually be a real number instead of accuracy
  against an arbitrary fixed tier. Wired into `predict.py` the same way as
  `skip_injuries`/`skip_weather` (`--no-market-odds`, `ODDS_API_KEY` env var);
  a genuine zero-cost no-op until the key is set - no network call, existing
  fixed-tier behavior unchanged. New `market_line`/`market_book`/
  `market_implied_prob`/`market_edge`/`market_line_available` columns on the
  predictions CSV (additive - `line_value` and everything else untouched). New
  frozen artifact `market_odds_week{W}_{season}.csv`, doubling as the
  once-per-week cache that keeps this inside the free tier's 500 credits/month.
  Nightly workflow passes `secrets.ODDS_API_KEY` through (unset today, so
  still a no-op in production). Design + credit-budget math in
  `docs/ODDS_API_INTEGRATION_PLAN.md`; UI wiring (Player Props page columns,
  DK Pick 6 pre-fill) is Phase 2/3, not built yet. Tests:
  `tests/test_market_odds.py` (54 total pass; no live API calls).

- **CI `Tests` workflow was red for 10 days — fixed.** `pytest -q` (the bare
  console script, which is what CI runs) errored at collection with
  `ModuleNotFoundError: No module named 'season_utils'` on every run since the
  workflow was added — only `python -m pytest` worked (that form prepends CWD to
  `sys.path`). Fix: `pythonpath = .` in `pytest.ini`. Also verified a full
  `build_and_train_pipeline.py` run end-to-end (schedule + historical fetch +
  train): clean, artifacts byte-identical to committed apart from an odds
  refresh on ~10 games.

- **Fixed an infinite rerun loop.** The `st.experimental_rerun()` → `st.rerun()`
  swap activated a latent loop: `streamlit run predictions.py` re-executes the
  whole file each rerun, resetting the module-level
  `historical_game_level_data = None` / `predictions_df = None`; the startup
  poll block then reloaded the data and called `st.rerun()` again, forever
  (~0.6 s/cycle). The two non-button rerun sites (post-background-load
  auto-refresh; `?run_pipeline=1` URL trigger) are now one-shot, guarded by
  `st.session_state` (which survives reruns; module globals do not).

- **Spread signal experiment (rejected).** Tried QB new-starter flags
  (`home/awayTeamNewStarterQB`, `qbNewStarterEdge`, leak-free from
  `home_qb_name` / `away_qb_name`) as a spread feature. Converged out-of-sample:
  179 bets, 53.1% acc, **+1.3% ROI** vs the +2.9% baseline; validation ROI
  worse (−12.8% vs −5.2%). No durable edge, and it is 0 for every upcoming game
  anyway. Not merged; logged in `docs/SPREAD_MODEL_INVESTIGATION.md`.

- **Player props: reliable-only by default + cleared stale history.** The
  Player Props page now defaults to "Show only props from tested (reliable)
  models" (uncheck to see everything) - so the top of the list is the ~240
  yards props whose models cleared the out-of-time bar, not the ~640 TD /
  skewed-line props. `player_props/backtest.py` carries `model_reliable` into
  each graded result and reports a reliable-only hit rate + ROI
  (`reliable_accuracy`, `by_reliable`, `roi_analysis_reliable`). Deleted the 12
  committed `accuracy_results_week*_20260110_*.json` files - those were 2025
  playoff backtests that would have contaminated the 2026 weekly accuracy
  history until enough real weeks accumulated.

- **Weekly spread scorecard + up-front honesty banner.** `weekly-model-performance.yml`
  only backtested player props — it never graded the spread betting log or
  summarised it. It now runs `betting_log.py` (grade finished bets) then
  `scripts/weekly_spread_report.py`, which rolls `betting_recommendations_log.csv`
  into `data_files/spread_performance.json` (overall + per-week + per-tier
  record / profit / ROI, pushes excluded from the ROI denominator) and commits
  it. The main dashboard now opens with an `st.warning` stating the
  out-of-sample reality — spread model is ~break-even (`Spread_OOS_Test`:
  141 bets, 53.9% correct, +2.9% ROI) — plus season-to-date from
  `spread_performance.json` once bets settle. Also fixed the tracking-log tab's
  per-tier breakdown, which iterated `['Elite','Strong','Good','Standard']` and
  silently dropped every `Lean` bet (`betting_log` writes `Lean`, not
  `Standard`).

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
