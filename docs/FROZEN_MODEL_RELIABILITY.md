# Frozen-model reliability report

`scripts/frozen_model_reliability.py` compares the spread model's **frozen
pregame predictions** with two baselines: a constant 50%, and the archived
Ontario market. It uses fixed calibration bins and paired, week-clustered
uncertainty.

**Evidence only, read-only:**
- no retraining, calibration fitting, parameter search or threshold change;
- no change to any model, feature, probability or signal;
- no betting recommendation;
- no source writes, no network calls, no bet-journal access.

```bash
python scripts/frozen_model_reliability.py                        # latest snapshot per game
python scripts/frozen_model_reliability.py --earliest-sensitivity # adds a separate view
python scripts/frozen_model_reliability.py --season 2026 --as-of 2026-10-21 \
    --output-dir reports/frozen_reliability_alt
```

**Outputs** go to the git-ignored `reports/frozen_reliability/`:

| File | Contents |
|---|---|
| `frozen_model_reliability.json` | the full report |
| `frozen_model_reliability_games.csv` | one row per selected game: the frozen-spread report's columns, plus sample membership and per-game scores |
| `frozen_model_reliability_calibration.csv` | every calibration table |
| `frozen_model_calibration.svg` | a standalone calibration plot |

## Built on the frozen-spread report

This report doesn't select, settle, cover or match games itself. It calls
`scripts/frozen_spread_report.py` ([`FROZEN_SPREAD_REPORT.md`](FROZEN_SPREAD_REPORT.md))
and uses that report's per-game rows unchanged:
- **Selection:** `pregame_snapshots.select_captures`, with the timing
  cross-check against the current schedule.
- **Settlement:** `bet_journal.completion_evidence` and
  `betting_log.spread_result`.
- **Coverage and market:** the completed-games-without-snapshot coverage, and
  the exact-line Ontario market match.

Its cumulative, per-week, signal-only and market metrics are the same numbers
that report gives.

## Evaluation protocol (version 1)

**Status:** protocol v1 is fixed for **subsequent** evaluation, and encoded
as `PROTOCOL` in the report. It was **not** registered before the games
already evaluated were played; it was written with this sample in existence.
So results on those games are **exploratory**, and only games evaluated
after the protocol was fixed are a cleaner test.

**The rules:**

1. **Selection:** one observation per game, the **latest** pregame snapshot
   captured strictly before the kickoff it recorded.
   - It is chosen before its probability or signal is looked at.
   - It must also precede the current scheduled kickoff, otherwise the game
     is `timing_unverified`.
   - There is no fallback to an older prediction.
2. **Terms:** the frozen underdog, the frozen handicap (+|spread_line|) and
   the frozen probability decide the game. The target is whether the
   underdog covered.
3. **Statuses:** `evaluated`, `push`, `pending`, or excluded (`no_line`,
   `pickem`, `no_probability`, `invalid_outcome`, `timing_unverified`), all
   counted.
   - **Pushes** are excluded from probability metrics and counted
     separately.
4. **Samples:** each is reported for **all eligible predictions** and for the
   **frozen bet-signal subset**, separately.
   - **Overall:** every evaluated game, model vs 50%.
   - **Matched market:** evaluated games with a matched Ontario quote. Model
     vs market vs 50%, on exactly the same games.
5. **Calibration bins** (below), fixed by protocol v1.
6. **No outcome-driven choices:** weeks, books, bins and snapshot policy are
   never chosen after seeing outcomes.
7. **Earliest-snapshot sensitivity** (`--earliest-sensitivity`): a separate
   block, `earliest_sensitivity`. It is never pooled with the latest view,
   never ranked against it, and is not a candidate policy. The latest view is
   identical with or without it.
8. **Outcome cutoff:** `--as-of` is a cutoff over the **current** files, not
   a historical reconstruction.

## Metrics

| Metric | How |
|---|---|
| Brier score | (p − y)² |
| Log loss | −[y ln p + (1−y) ln(1−p)], with p clipped to [1e-15, 1 − 1e-15] for log loss only (as in the frozen-spread report) |
| Mean predicted probability | the mean frozen cover probability |
| Observed cover rate | covers ÷ evaluated games. It is a rate, not classification accuracy. |

Each sample also reports counts (evaluated, pushes, pending, excluded by
reason) cumulatively and per season/week. The coverage section shows
completed games without a snapshot, and the market match status of every
selected game.

## Market baseline

- **Matching:** `frozen_spread_report.market_for`. Per game, it chooses one
  archived, validated Ontario capture, the latest strictly pregame.
- **Which quotes count:** automated Ontario feeds only. The US reference feed
  and manual Ontario quotes are excluded before any price is read. Only fresh
  quotes at the **exact frozen handicap**, with both prices, are used.
- **One game, one observation:** the market probability is the devigged
  two-way probability **averaged over the matched books**. `books_per_game`
  shows the range.
- **Not simultaneous:** the market capture and the model snapshot are taken at
  different times. Compare `market_captured_at` with `captured_at` per game.
- **Unavailable comparisons:** evaluated games without a match are listed in
  `unavailable_market_by_reason` with each reason:
  - no archived capture before kickoff;
  - no fresh Ontario quote;
  - no quote at the frozen handicap.

## Calibration

**Bins (fixed):** `[0, 0.40)`, `[0.40, 0.50)`, `[0.50, 0.60)`, `[0.60, 0.70)`,
`[0.70, 1.00]`.
- **Endpoints:** each bin is left-closed and right-open, except the last,
  which is closed at 1.0. For example, 0.40 is in `[0.40, 0.50)`, and 0.70
  and 1.0 are in the last bin.
- **Different from the frozen-spread report:** that report keeps its own 10
  equal-width bins. These five are this protocol's; they are not tuned.

**Each bin shows:**
- the number of games and covers;
- the mean predicted probability and the observed cover rate;
- a **95% Wilson score interval** for the observed rate (z = 1.959964).
  This is a **binomial reference interval**: it treats the bin's games as
  independent, so it does **not** account for dependence between games
  (shared weeks, the two sides of a game). It is distinct from the
  week-cluster bootstrap intervals below, and should be read as a rough
  guide, not a calibrated uncertainty.

**Empty and sparse bins:**
- **Empty bins** stay listed, with the rate and interval unavailable.
- **Sparse bins** (fewer than 10 games) are flagged *descriptive only*. That
  flag is a display aid, **not** a reliability threshold. No count turns a
  bin into evidence of good calibration.

**Plot:** `frozen_model_calibration.svg` shows all eligible predictions
(latest view):
- the diagonal for perfect calibration;
- one point per non-empty bin, with its Wilson interval and `n`;
- sparse bins in red, and empty bins marked `n=0`.

## Uncertainty: paired, week-clustered

- **Paired difference:** for each game, *model score − baseline score*, for
  Brier and log loss. **Negative means the model scored better.**
- **Comparisons:**
  - overall: model − 50%;
  - matched sample: model − market, model − 50% and market − 50%.
- **Interval:** a **paired season/week cluster bootstrap**.
  - Whole weeks are drawn with replacement, as many as there are. Every game
    in a drawn week contributes its paired difference, so a model/baseline
    pair is never split.
  - Each replicate is the mean over all drawn games.
  - 10,000 replicates, `numpy.random.default_rng(20261005)`, with the 2.5th
    and 97.5th percentiles (linear interpolation).
  - Weeks are the clusters because games in a week share conditions (injury
    news, weather, line moves).
- **Minimum:** with **fewer than 5 week clusters**, the interval is reported
  as unavailable with the reason; no precise interval is invented. The mean
  difference is still shown. Reaching 5 clusters only **permits** an
  exploratory interval; it does **not** establish that the interval is
  reliable.

**Reading the intervals:**
- **Exploratory:** with few weeks, a cluster bootstrap understates
  uncertainty and assumes weeks are exchangeable.
- **Includes zero:** that is **not** evidence that the model and baseline are
  equivalent.
- **Excludes zero:** that is **not**, by itself, evidence of a betting edge
  or of profitability.
- **No guarantees:** the intervals don't remove model or dependence
  assumptions.

## Provenance and failure

The JSON records:
- the protocol version and full protocol, including its status (fixed for
  subsequent evaluation);
- every setting;
- the generation time (UTC), the `as_of_date` and its meaning, and the scope;
- `tracking_start`;
- the source fingerprints: the sha256 of the schedule, every pregame snapshot
  and every Ontario capture, plus a combined hash.

The fingerprint is taken before and after the inputs are read. If they
differ, the run fails.

**Failure handling:**
- **Corrupt inputs** fail the run, with no partial report: a bad snapshot
  checksum or schema, an unreadable capture, or a probability outside [0, 1].
- **Output paths:** output directories overlapping source data are refused,
  using the frozen-spread report's rule: `data_files/`, the snapshot,
  schedule and capture folders, the journal and `.git`. This covers aliases
  (`..`), letter-case variants, symlinks and junctions.
- **CSV reading:** `read_games_csv` keeps identifier and hash columns
  (`run_id`, `code_revision`, ...) as text.

## Sample, and what it can't do

- **Short sample:** frozen tracking began on **2026-10-05** (first snapshot
  `2026-10-05T09:54:41Z`). Only games with a frozen pregame snapshot are
  evaluated, a few weeks of one season. Completed games before then are
  counted as `before_first_snapshot` and never backfilled.
- **No tuning from this sample:** changing thresholds, bins or signals because
  of it would require a **new, untouched evaluation period**. This report must
  not be used to tune them.
- **Not independent:** games in a week share conditions, and a game's two
  sides settle in opposite directions. One game is one observation here.
- **Market reference only:** the market probability isn't an achievable
  price, and it isn't observed at the same moment as the model.
- **No claims:** no profitability, edge or significance claim is made.
