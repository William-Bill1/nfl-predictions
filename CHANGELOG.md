# Changelog

History moved out of `README.md`. Newest first. Dates are as recorded in the
original notes; undated dashboard/infra work from late 2025 is grouped at the
bottom.

---

## October 2026

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
    much later run can't be labelled as the slot. An empty response is kept
    as evidence but leaves the slot open for a retry.
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
