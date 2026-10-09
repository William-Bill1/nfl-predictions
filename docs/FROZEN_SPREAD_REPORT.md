# Frozen-pregame spread report

`scripts/frozen_spread_report.py` evaluates the spread model **only on
predictions frozen before kickoff** in `data_files/pregame_snapshots/` (see
[`PREGAME_SNAPSHOTS.md`](PREGAME_SNAPSHOTS.md)).

It's read-only:

- It never regenerates predictions.
- It never reads the rewritten predictions CSV or today's lines.
- It never reads actual wagers.

It writes a JSON report and a per-game CSV to a git-ignored directory.

```bash
python scripts/frozen_spread_report.py                     # latest pregame capture per game
python scripts/frozen_spread_report.py --which earliest    # earliest instead
python scripts/frozen_spread_report.py --season 2026 --week 5 --as-of 2026-10-14
python scripts/frozen_spread_report.py --output-dir reports/frozen_spread_alt
```

- **Outputs:** `frozen_spread_report.json` and `frozen_spread_games.csv`.
- **Default location:** `reports/frozen_spread/` (git-ignored).
- **Deterministic:** the same inputs and the same `--as-of` date give
  byte-identical files, with sorted rows and no wall-clock times.
- **`--as-of`** (America/Toronto date, default today) is an **outcome cutoff
  only**. A game counts as completed only if its game day is before that date.
  - **Current files:** the inputs are the snapshot, capture and schedule files
    as they exist when the report runs. Snapshots or captures written after the
    date, and later score corrections, may be present.
  - **Not a historical reconstruction:** it does **not** show what the report
    would have said on that date. The report repeats this in `as_of_semantics`.
- **Scope** (`scope` in the report):
  - **`--season`:** the explicit season.
  - **Otherwise:** the latest season in the schedule.
  - **Never from selections:** scope is never derived from which predictions
    were selected.
  - **`--week`:** narrows it further.
  - **Outside scope:** selected games outside the scope aren't evaluated, and
    are counted in `selected_outside_scope`.

Tests: `tests/test_frozen_spread_report.py`.

## 1. Selection: one frozen observation per game

- **The rule:** `pregame_snapshots.select_captures`, with **no status filter**.
  Every snapshot is validated, and a capture is eligible only if it was taken
  strictly before the kickoff it recorded.
  - **Default:** the latest eligible capture.
  - **`--which earliest`:** the earliest. Ties are broken by `run_id`.
- **The capture is chosen first.** Its prediction is inspected afterwards. If
  the chosen capture has no probability (`no_probability`), no line or a
  pick'em, that is the game's status. An earlier capture with a probability
  or a signal is **never** used instead.
- **Timing cross-check:** the chosen capture must also be strictly before the
  game's kickoff in the **current** schedule (`kickoff_check`).
  - **Passes:** `unchanged`, `moved_later`, `moved_earlier` (still after the
    capture), or `current_kickoff_unavailable`.
  - **Fails:** `capture_not_before_current_kickoff`, when the game was moved
    to or before the capture time. The game is then **`timing_unverified`**:
    excluded, with no fallback to an earlier capture.
- **Recorded per game:** snapshot run ID and file, capture time, code
  revision, config and feature-set IDs, artifact hash, and the frozen terms
  (line, underdog, handicap, probability, signal).
- **No backfill:** games without a pregame capture are never evaluated. They
  are listed as coverage gaps instead (section 3).

**Timing evidence.** Only *scheduled* kickoffs are stored: the one recorded in
the snapshot and the one in today's schedule. Actual start times aren't
archived, so a game that started early without a schedule change can't be
detected.

## 2. Outcomes

- **The bet evaluated:** the frozen underdog at its frozen handicap,
  `+|spread_line|` from the snapshot. Never today's line.
- **Settlement:** the reviewed `betting_log.spread_result`.
- **Completion evidence:** `bet_journal.completion_evidence`. It requires
  exactly one schedule row, finite non-negative integer scores (not 0-0),
  `result == home - away`, `total == home + away`, `overtime` of 0 or 1, and a
  game day before the as-of date.
  - Missing scores, or a game day not yet past, give **`pending`**.
  - Duplicated, inconsistent or placeholder results, or schedule teams that
    differ from the snapshot, give **`invalid_outcome`**.

## 3. Coverage

Every selected game has exactly one status:

| Status | Meaning |
|---|---|
| `evaluated` | the underdog won or lost at the frozen handicap |
| `push` | landed exactly on the handicap; reported separately, never in probability metrics |
| `pending` | no conservative completion evidence yet |
| `no_line` / `pickem` / `no_probability` | the selected capture had no probability at a valid line |
| `invalid_outcome` | the result is ambiguous, duplicated or inconsistent |
| `timing_unverified` | the capture wasn't strictly before the current scheduled kickoff |

**`coverage` in the report also gives:**

- `by_status`;
- the same counts for games dated before the as-of date (most `no_line` games
  are future weeks without posted lines);
- `completed_without_snapshot`: completed games **in scope** that have no
  selected capture. This is computed from the schedule, independently of the
  selections, so it's meaningful even with no snapshots at all. They are listed,
  never evaluated:
  - `no_snapshot_history`: no snapshot files exist at all;
  - `before_first_snapshot`: kickoff at or before the first snapshot's capture;
  - `after_first_snapshot`: kickoff after it, a genuine coverage gap, listed
    by game ID;
  - `kickoff_unavailable`: the current schedule gives no usable kickoff;
  - plus `completed_games_in_scope`, `completed_with_selected_snapshot` and
    `first_snapshot_captured_at`;
- `sample_note`: the evaluated count and a statement that this is a captured
  sample, not the whole season, or that no snapshot history exists.

## 4. Probability metrics

The metrics use every evaluated game with a frozen probability, with
signal-only results as a separate subset.

- **Brier score and log loss,** each against a constant 50% baseline (0.25
  and ln 2) on **exactly the same** non-push games.
- **Clipping:** for log loss only, probabilities are clipped to
  `[1e-15, 1 - 1e-15]`. In floating point, the upper bound is
  `0.999999999999999`.
  - `n_probability_0_or_1` counts exact 0s and 1s.
  - `n_clipped_for_log_loss` counts the clipped values.
  - The Brier score always uses the unclipped probability.
- **Calibration:** 10 equal-width bins with count, mean predicted probability
  and observed cover rate. A probability of 1.0 falls in the last bin.
- **Groups:** `overall`, `by_season_week` (e.g. `2026-W05`) and
  `by_code_revision`.
- **No significance tests or profitability claims** are made; the
  `sample_note` states how small the sample is.

## 5. Market comparison

- **Source:** only the archived, validated Ontario spread captures
  (`data_files/ontario_spreads/captures/`), and only a capture taken strictly
  before both the frozen and the current kickoff.
- **One capture per game:** latest (or earliest, following `--which`) by
  `(captured_at, run_id)`. Only that capture's quotes are used, and no other
  capture is searched for a match.
- **Automated Ontario feeds only.** A quote is used only if the Ontario
  report's own grouping (`_group_for`) classes it as an automated Ontario feed:
  a registered CA-ON book with an Ontario role, from `the_odds_api`. The
  quote's stored `jurisdiction` (`CA-ON`) and `role` must agree.
  - **Excluded:** the US FanDuel reference feed and manual quotes.
  - **Applied first:** this filter runs before any point is collected,
    probability devigged or price chosen.
  - **None left:** a capture with no fresh Ontario quote for the game is
    `no_fresh_ontario_quote`.
- **A quote counts only if all of these hold:**
  - it is `quoted` (fresh), not `stale`, `invalid` or similar;
  - its last update is before kickoff and no later than the capture;
  - the frozen underdog's handicap equals the frozen handicap exactly, and the
    other side mirrors it;
  - both prices are valid American odds.
- **Devigging:** each matched book is devigged as
  `implied(dog) / (implied(dog) + implied(fav))`. The market probability is
  the mean over matched books.
- **Gaps:** `no_quote_at_frozen_handicap` (it lists the Ontario points that
  were quoted), `no_fresh_ontario_quote` and
  `no_archived_capture_before_kickoff`. They are never filled from
  another line, another capture, current odds or postgame odds.
- **Same games for all three:** model, market and the 50% baseline are scored
  on exactly the matched, evaluated games.
- **Unavailable:** with no matched evaluated game, the comparison reports
  `"status": "unavailable"` with the reason.
- **Not used:** the spread-tracker weekly CSVs
  (`market_spreads_week*_*.csv`). They are a re-fetchable cache, overwritten
  with `--no-cache`, with no integrity checksum.
- **Timing caveat:** the market capture and the frozen model capture are
  chosen independently, and can be days apart. Compare `market_captured_at`
  with `captured_at` per game.

## 6. Simulated returns (signal-only)

These are never actual wagers; the bet journal is not read.

- **Which bets:** games whose selected capture had `bet_signal: true`, at the
  frozen underdog and handicap, 100 units staked per bet.
- **Settlement:** a win pays at the price (`bet_journal.win_profit`), a loss
  is −100, and a push returns the stake with profit 0 but still counts as a
  bet.
- **ROI:** net units divided by units staked on settled bets, pushes
  included. Pending or unresolved signals aren't bets yet; they're counted in
  `signals_not_settled`.
- **Archived price:** the **lowest-paying** matched archived **Ontario-feed**
  quote at the exact frozen handicap, the conservative choice. US-reference
  and manual prices are never used. A signal without one is
  excluded and counted (`settled_without_archived_price`). A price is never
  inferred from later data.
- **Assumed −110:** reported separately and labelled as a scenario, not an
  offered price.

## 7. Inputs, outputs and failure

- **Corrupt inputs fail the run** (exit code 1). That covers:
  - an invalid or tampered snapshot (schema, checksum, a capture at or after
    its recorded kickoff);
  - an invalid market capture;
  - an unreadable schedule, or one missing a required column;
  - a missing snapshot directory.

  Nothing is skipped silently.
- **Protected locations:** output directories are refused if they are, contain,
  or resolve into (through symlinks, junctions, `..` or letter case):
  - `data_files/`, which holds the snapshots, captures, journal and production
    inputs;
  - the snapshot, capture or schedule locations given;
  - `.git`.

  Files are written by atomic replace, so a link at the target is replaced,
  never written through.
- **CSV identifiers:** the per-game CSV's identifier and hash columns (for
  example `code_revision` and `run_id`) must be read as text. Use
  `frozen_spread_report.read_games_csv(path)`; a hex SHA such as
  `85e60468...` can crash pandas' CSV parser otherwise (see
  `PREGAME_SNAPSHOTS.md`).
- **JSON:** no `NaN`; missing values are `null`.

## Limitations

- **Small, partial sample:** captures began on 2026-10-05. Most completed
  games of 2026 predate them, and many future games have no line yet.
- **Nightly captures:** a line or probability that moved during the day before
  kickoff isn't recorded.
- **Line source:** the frozen line is nflverse's schedule line at capture time,
  not a sportsbook offer.
- **Completion evidence:** it's conservative but not an authoritative final
  status.
- **`--as-of` is not a reconstruction:** it doesn't restrict inputs to what
  existed on that date.
- **No edge claim:** none of this shows a betting edge.
