# Ontario sportsbook spread tracking (Phase 1: collection and storage)

`ontario_spreads.py` records the NFL spreads and prices that Ontario-licensed
sportsbooks show at two fixed times a week:

- **Wednesday 12:00** America/Toronto
- **Sunday 09:00** America/Toronto

The goal is a later comparison of Wednesday versus Sunday-morning lines and
prices. This phase only collects and stores them; there is no analysis, UI or
betting rule. Nothing here shows that either time is better to bet, or that
any book or strategy has an edge.

Each capture is a new, immutable, checksummed JSON file. A later capture never
replaces or edits an earlier one, and nothing is carried forward between
captures: a quote missing at one capture is recorded as missing, not as
unchanged.

This is separate from `spread_tracker.py`, which keeps working unchanged. It
pulls the US and Canada regions every Wednesday at 12:00 UTC and upserts
`spread_tracker_log.csv`. Nothing here reads, rewrites or backfills that
tracker's files, and its old quotes are never re-labelled as pregame captures.

## Bookmakers and jurisdictions

Bookmakers are requested **by key**, not by region, so the feed list is
explicit. The keys and titles were checked against The Odds API's
[bookmaker list](https://the-odds-api.com/sports-odds-data/bookmaker-apis.html)
(Canada region) on 2026-10-05.

| Key | Provider title | Jurisdiction | Role |
|---|---|---|---|
| `betano_ca_on` | Betano (CA - ON) | `CA-ON` | Ontario |
| `betmgm_ca_on` | BetMGM (CA - ON) | `CA-ON` | Ontario |
| `betrivers_ca_on` | BetRivers (CA - ON) | `CA-ON` | Ontario |
| `pointsbetca` | PointsBet (CA - ON) | `CA-ON` | Ontario |
| `proline_ca_on` | PROLINE (CA - ON) | `CA-ON` | Ontario |
| `sportsinteraction_ca_on` | Sports Interaction (CA - ON) | `CA-ON` | Ontario |
| `bet99_ca_on` | BET99 (CA - ON) | `CA-ON` | Ontario, **paid-tier only**; opt-in |
| `fanduel` | FanDuel | `US` | **US reference only** |
| `fanduel_on_manual` | FanDuel Ontario (manual entry) | `CA-ON` | manual quotes, source `manual` |

- **Only Ontario feeds are labelled Ontario.** A key is `CA-ON` only when the
  provider documents it as an Ontario feed ("(CA - ON)"). `playnow_ca`
  ("PlayNow (CA)", British Columbia) is a Canadian feed but not an Ontario
  one; it isn't requested. Any other bookmaker that appears in a response is
  listed under `unrequested_books` and never treated as Ontario.
- **BET99** is "only available on paid subscriptions". It isn't requested
  unless `ONTARIO_SPREADS_INCLUDE_BET99=1`, because it hasn't been verified
  that a free-plan request naming a paid-only bookmaker succeeds. Either way,
  each capture's `coverage.bet99_ca_on` says whether it was requested and how
  many games it quoted. Absence is reported, never treated as an error.
- **FanDuel.** The API's `fanduel` key is FanDuel **US**. It's kept as a
  `US`-jurisdiction reference only, and is never used as, or substituted for,
  FanDuel Ontario. FanDuel Ontario has no API feed; its quotes come only from
  the manual entry command below.

## Capture (`python ontario_spreads.py capture`)

1. **Slot.** From the America/Toronto wall clock, works out whether a slot is
   due. If none is (see "Scheduling"), it stops with `not_due` and makes no
   call. If the slot already has a capture, it stops with `already_captured`
   and makes no call.
2. **No key, no calls.** Without `ODDS_API_KEY` it stops with `no_key` before
   any network access.
3. **Scope.** Reads the nflverse schedule (`nfl_games_historical.csv`; its
   sha256 is recorded) and takes the upcoming week: the (season, week) of the
   next game to kick off. Within that week, any game that has kicked off,
   finished, or has an unusable kickoff time is excluded with a reason. If no
   pregame games remain, it stops with `no_games` and makes no call.
4. **Credit check.** A free `GET /v4/sports` call reads `x-requests-remaining`.
   The provider documents that "Calls to the /sports endpoint will not affect
   the quota usage" and that they return `x-requests-remaining`,
   `x-requests-used` and `x-requests-last`. The paid call is skipped
   (`budget_skip`, exit code 2) if it would leave fewer credits than the
   reserve (`ODDS_API_MIN_REMAINING`, default 20, shared with the other Odds
   API features). It is also skipped if the remaining credits can't be read:
   the check fails closed.
5. **One paid call:**
   `GET /v4/sports/americanfootball_nfl/odds?markets=spreads&bookmakers=<keys>&oddsFormat=american&dateFormat=iso&commenceTimeFrom=<capture time>&commenceTimeTo=<last kickoff + 1h>`.
   Up to 10 bookmakers cost the same as one region, so a capture costs **1
   credit**. `commenceTimeFrom` asks the provider to leave out games that have
   started.
6. **Event matching** uses both teams **and** the kickoff time. An event
   matches a scheduled game only if its teams match and its `commence_time`
   is within 60 minutes of the scheduled kickoff. Teams alone aren't enough.
   A neutral-site game listed with home and away the other way round matches
   with `orientation: "swapped"`. Handicaps and prices are always read by
   **team name**, so the swap can't flip a sign. An event whose
   `commence_time` has already passed is excluded
   (`provider_reports_started`), even if the schedule says otherwise.
7. **Write.** The capture is sealed with a checksum, validated, and written
   once (see "Immutability").
8. **Empty captures.** If no **Ontario** book quoted any in-scope game, the
   file is still written as evidence, with `usable: false`. This covers an
   empty provider response, and also a response where only the US FanDuel
   reference quoted: US quotes never make a capture usable. The result is `empty`: the run fails, and
   the slot stays open, so the next run inside the window can capture it. The
   provider doesn't charge for an `/odds` request that returns no events.

### Quote status

Each in-scope game has one entry per requested book:

| status | meaning |
|---|---|
| `quoted` | both sides present; handicaps mirror (home = −away, half-point steps); prices are valid American odds; updated within 60 minutes of the capture |
| `stale` | as `quoted`, but the market's `last_update` is more than 60 minutes before the capture. The values are kept and labelled. |
| `absent` | the provider returned the event but not this book |
| `no_spreads_market` | the book is present but has no spreads market for the game |
| `incomplete` | one side's outcome is missing |
| `invalid` | values fail validation: non-mirrored handicaps, a bad price, or a missing or future `last_update`. The raw values are kept with a `problem` |
| `event_not_matched` | the game itself wasn't matched (`match_problem`: `no_provider_event`, `kickoff_mismatch`, `ambiguous_provider_event`) |

Absent, missing and stale quotes are always explicit. They are never filled
from an earlier capture.

## Capture file: `data_files/ontario_spreads/captures/<run_id>.json`

| Field | Contents |
|---|---|
| `schema_version`, `kind` | `1`, `ontario_spread_capture` |
| `run_id` | `<UTC timestamp>-<12 random hex>`, assigned once per run |
| `slot` | `name` (`wednesday_noon`, `sunday_morning` or `ad_hoc`), `slot_id` (e.g. `2026-10-07_wednesday_noon`), `intended_utc`, `intended_local`, `delay_minutes`, `status` (`on_time` ≤ 30 min, `late`, or `ad_hoc`) |
| `captured_at`, `received_at` | when the paid request was made and its response received (UTC) |
| `code_revision`, `code_dirty` | `GITHUB_SHA` in Actions, otherwise `git rev-parse HEAD` |
| `request` | endpoint, parameters **without the API key**, HTTP status, estimated cost, the credit reserve, credits before the call, and the usage headers `x-requests-remaining`, `x-requests-used` and `x-requests-last` |
| `books_requested`, `not_ontario` | the feed list with jurisdictions; Canadian feeds that aren't Ontario |
| `schedule` | path and sha256 of the schedule bytes used |
| `games[]` | `game_id`, `season`, `week`, teams, scheduled `kickoff_utc`, `provider_event_id`, `provider_commence_time`, `orientation`, `match_problem`, `quotes[]`, `unrequested_books` |
| `quotes[]` | `book_key`, `book_title`, `jurisdiction`, `role`, `source` (always `the_odds_api`), `status`, `home_point`, `home_price`, `away_point`, `away_price`, `bookmaker_last_update`, `market_last_update`, `age_minutes`, `problem`, `model_snapshot` |
| `usable` | `true` if at least one **Ontario** (`CA-ON`) feed quoted (or stale-quoted) an in-scope game; US reference quotes don't count. An unusable capture doesn't fill its slot |
| `excluded_games`, `unmatched_provider_events` | games left out, with reasons; provider events that matched no in-scope game |
| `coverage` | per book: whether it was requested, game count, counts by status, and a note (BET99 paid tier, FanDuel US reference) |
| `provider_response` | the provider's events, as returned. Any string containing the key or an `apiKey=` URL is removed |
| `payload_sha256` | SHA-256 of the canonical document without this field |

**Handicap convention.** Each side's own handicap as quoted, with the
favourite negative: `home_point` is the home team's and `away_point` the away
team's. This is **not** nflverse's `spread_line` convention, which is home
favourite positive.

**Checksum.** The checksum is unkeyed, the same as for pregame snapshots. It
detects accidental changes but not a deliberate edit with the checksum
recomputed. Git history is the record of changes.

## Model snapshot link

Each **quote** has a `model_snapshot` entry linking it to a frozen pregame
model snapshot (`data_files/pregame_snapshots/`, see
`docs/PREGAME_SNAPSHOTS.md`). The link is aligned to the **quote's own
provider timestamp**, the market's `last_update` (`basis:
"market_last_update"`), not to when the API was called. A price last changed
at 02:00 is compared with the model as it stood at 02:00, even if it was
fetched at noon.

- **`linked`** names the latest snapshot that contains the game and was
  captured at or before `as_of`, which is the earlier of the quote's
  `last_update` and the capture time. It records the snapshot's `run_id`,
  `captured_at`, `payload_sha256`, `file`, and the game's `prediction_status`.
  A snapshot captured after the quote is never used, and validation rejects a
  file that claims one.
- **`quote_time_unknown`** means the market has no `last_update`, so no
  time-aligned model comparison is claimed (even if the bookmaker-level
  timestamp exists).
- **`no_eligible_snapshot`** means no such snapshot exists.
- **`no_quote`** is used for absent, missing or unmatched quotes, and
  **`not_linked_invalid_quote`** for invalid ones.
- **`unavailable`** means the snapshots couldn't be read or failed validation.
  The reason is recorded and the quotes are still captured.

The link is a reference only. **No model probabilities are copied into
capture files.** Actual quote history and model output, including any
probability extrapolated to a book's line (`scripts/model_line_shop.py`), are
kept separate.

## Manual FanDuel Ontario quotes

```bash
python ontario_spreads.py manual-quote --game 2026_05_TB_DAL --team DAL \
    --spread -3.5 --price -112 --observed-at 2026-10-07T12:05-04:00 \
    [--opponent-price -108] [--note "..."] [--entered-by NAME]
```

- **Required:** the game (nflverse `game_id`), the team, that team's spread
  and American price, and the observation time. The observation time must
  include a timezone; it may not be in the future or at or after kickoff.
- **Validation:** the team must play in the game, the spread must be a
  half-point value, and prices must be valid American odds. Anything else is
  rejected and nothing is written.
- **Written to** `data_files/ontario_spreads/manual/<quote_id>.json`, with
  `source: "manual"`, `book_key: "fanduel_on_manual"` and
  `jurisdiction: "CA-ON"`.
- **Recorded with it:** `entered_at`, `observed_at`, `entered_by`, the code
  revision, the schedule hash, and a checksum.
- **Duplicates:** entering the same observation twice (same game, team, time,
  spread and prices) returns the existing file rather than creating a second.
- **Kept separate from automated quotes.** Manual quotes have their own
  directory, `kind` and `source`. A capture file is rejected by validation if
  any quote in it isn't an API feed with `source: "the_odds_api"`, so a manual
  FanDuel Ontario quote can't appear as an automated one.

## Immutability, retries and concurrency

- **New files only.** Files are written to a temp file, then hard-linked to
  the final name (the same mechanism as pregame snapshots). A partial file is
  never visible, and an existing file is never replaced.
- **Same-run retry:** writing an identical document again is a no-op.
- **Conflicts:** the same `run_id` with different content, or an existing file
  that isn't a valid artifact, raises `CaptureConflictError`. Nothing is
  written.
- **One capture per slot.** A slot that already has a usable capture (one
  with an Ontario quote) is skipped without an API call. A failed request
  writes nothing, and an empty or Ontario-less capture doesn't fill the slot,
  so a later run inside the window can still capture it. After the window,
  the slot stays `empty` (or `missed`); a later run never fills it. Completed
  captures are never changed. Two local runs for the same slot at once are
  blocked by an exclusive lock file (`.<slot_id>.lock`, git-ignored); the
  second run fails without calling the API. In Actions, a `concurrency` group
  serializes the runs.
- **Validation on read:** `python ontario_spreads.py validate` checks every
  stored capture and manual quote. A slot check reads every capture, so an
  invalid file makes the next capture fail visibly rather than be skipped.

## Scheduling (`.github/workflows/ontario-spread-capture.yml`)

GitHub's cron runs in UTC, so each slot has two cron entries, one for each
offset:

| Slot | EDT (UTC−4) | EST (UTC−5) |
|---|---|---|
| Wednesday 12:00 Toronto | 16:00 UTC | 17:00 UTC |
| Sunday 09:00 Toronto | 13:00 UTC | 14:00 UTC |

The script decides from the Toronto wall clock, using `zoneinfo`. The entry
an hour early finds no slot due (`not_due`); the entry an hour late finds the
slot already captured. Neither makes a paid call.

- **Windows.** A run counts for a slot from the slot time until the end of
  its window: Wednesday until **15:00** and Sunday until **11:00** Toronto. A
  run more than 30 minutes after the slot is labelled `late`, and its actual
  capture time is always recorded. A run after the window captures nothing,
  so a Thursday or Sunday-afternoon quote can never be labelled as the
  Wednesday or Sunday-morning slot; the slot shows as `missed`. Each quote
  also carries the provider's own `last_update`, and old quotes are labelled
  `stale`.
- **Missed slots.** From the first scheduled capture onward, a slot with no
  capture is reported as `missed` by `python ontario_spreads.py coverage`
  (also printed in each workflow run's summary). Earlier slots weren't
  tracked, so they aren't listed. Until the first capture, `coverage` says
  there are no captures yet, so a failure before then is visible only as a
  failed run.
- **Sunday and early games.** Games underway at capture time, including the
  09:30 ET London games when a run is delayed, are excluded.
- **Manual runs.** `workflow_dispatch` takes a `slot` input. `ad_hoc` captures
  outside the schedule and is labelled `ad_hoc`, still pregame games only.
- **Later phase.** Per-game closing-line captures aren't part of this phase.

### Failure visibility

| Outcome | Exit code | Workflow |
|---|---|---|
| `captured` | 0 | commits the new file (on `main`); a failed push fails the run |
| `empty` | 1 | **fails**; the file is still committed as evidence, and the slot stays open for a retry |
| `not_due`, `already_captured`, `no_games` | 0 | nothing to commit |
| `no_key` | 0 | `::warning::` annotation; no calls made |
| `budget_skip` | 2 | **fails**, with `::error::` and a summary section |
| request, validation or write failure | 1 | **fails**, with `::error::` and a summary section |

The capture step has no `continue-on-error`. Errors are redacted: the key and
any `apiKey=` parameter are replaced with `***` in every message. The secret
is passed only to the capture step.

**Concurrent pushes.** The commit step adds only new files under
`data_files/ontario_spreads/captures/`. It uses `git pull --rebase`, then a
plain push, retried 3 times; it never force-pushes. Other workflows (the
nightly, the US+CA tracker) also push plain commits to `main`, so neither side
can drop the other's commits; a racing push fails rather than overwrites.
Every run also uploads the capture files as a workflow artifact (kept 90
days), including runs where the capture or commit step failed, so a capture
survives a failed push.

A final **Report persistence** step runs on every run. It fails the run with
an `::error::` if any capture file is still uncommitted or unpushed, naming
the artifact that holds it, or saying it is lost if the upload also failed.
Re-committing from the artifact is a manual step. A manual run on a branch
other than `main` doesn't commit, so it is reported the same way.

## Credit budget

These estimates were made by reading the code that makes requests. No live
calls were made to measure them.

| Feature | Requests | Credits |
|---|---|---|
| Player-prop odds (`player_props/market_odds.py`, nightly) | per event, 4 markets × 1 region; each game paid once, after its props are posted | ≤ 4 × ~16 games ≈ 64/week |
| US+CA spread tracker (`spread_tracker.py`, Wednesdays) | bulk, 1 market × 2 regions | 2/week |
| **Ontario captures (this)** | bulk, 1 market, 7 bookmakers (8 with BET99), so 1 region equivalent; 2 slots a week | **2/week** |
| Credit checks (`/v4/sports`) | free | 0 |

That's roughly 68 credits a week, or about 295 a month during the regular
season, against the free tier's 500. This is an **approximate estimate, not a
guaranteed cap**: manual runs, refetches (`--no-cache`), retries after empty
captures, and other ad-hoc API use all add to it, and the prop figure depends
on how many games have posted props. The reserve check is what protects the
quota. The Ontario captures add about 9 a month,
plus 1 for each manual `workflow_dispatch` run. The duplicate DST cron entries
cost nothing. The Wednesday Ontario capture and the Wednesday US+CA tracker
are separate requests: they ask for different scopes, and folding them
together would change the existing tracker, which this phase leaves alone.

## Limitations

- **Two snapshots a week.** These are Wednesday and Sunday-morning quotes,
  not closing lines; lines that move in between aren't seen. Per-game closing
  captures are a later phase.
- **Coverage depends on the provider.** A book missing from a response is
  recorded as absent; the reason (not offered, not yet posted, or a feed
  problem) isn't known.
- **FanDuel Ontario** exists only as manually entered quotes, which are only
  as accurate as the person entering them.
- **BET99** isn't requested on the free plan. Its behaviour on a paid plan is
  untested here.
- **Event matching** needs the schedule's kickoff to be within 60 minutes of
  the provider's. A game moved by more than that shows as
  `event_not_matched` (`kickoff_mismatch`) until the schedule catches up.
- **The credit check** relies on `/v4/sports` returning usage headers; if it
  doesn't, every capture is skipped and the run fails, which is visible.
- **Lock files.** A process killed mid-capture can leave a local lock file
  behind. The next run says so and names the file to delete.
- **Unkeyed checksum:** see above.
- **No analysis.** This phase collects data only. It doesn't show that
  Wednesday or Sunday lines are better, and it doesn't validate any
  betting-time strategy.
