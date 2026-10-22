# Injury report archive (`injury_reports.py`)

**Phase 1: evidence only.** Around each Ontario decision slot (Wednesday 12:00
and Sunday 09:00, America/Toronto), the Ontario Spread Capture workflow also
saves what the app's injury source showed at that moment. Each capture is an
immutable, checksummed JSON file.

**What this phase doesn't do:**
- no injury-impact model;
- no probability adjustment;
- no injury score;
- no betting recommendation;
- no new page.

The player-prop injury adjustment (`player_props/injuries.py`) and its nightly
cache, `data_files/espn_injuries.csv`, are unchanged. This module never reads
that cache, so a cached file is never presented as a fresh observation.

## Source and what it means

**Source:** ESPN's public injuries JSON feed (`ESPN_INJURIES_URL`), the same
feed `player_props/injuries.py` reads. No key or credential is sent.

| Field | What ESPN supplies | Stored as |
|---|---|---|
| Status | ESPN's general injury status. It mixes game-designation words (Questionable, Doubtful, Out), roster statuses (Injured Reserve, Suspension, ...) and "Active". It is **not** the official NFL game-status report. | `source_status`: the raw word, unchanged |
| Status group | (derived) | `status_class`: `game_designation_word`, `roster_status`, `listed_active`, `unrecognized` or `missing`. This is only a grouping of the raw word, never a claim about the official report. |
| Practice participation | **Not supplied** by this feed | `practice_participation: null` on every row, and the capture's `source` record says "not supplied by this source" |
| Injury | `details.type` / `location` / `detail` / `side`, when present | `injury`: each field `null` when absent |
| Comment, return date | `shortComment`, `details.returnDate`, when present | `short_comment`, `source_return_date`: `null` when absent |
| Player | athlete display name; ESPN id in `athlete.id`, or failing that in an athlete link | `player_name`; `player_id` with `player_id_source` (`athlete.id`, `athlete.links` or `null`) |
| Team | ESPN team display name | `source_team_name`; `team` (nflverse abbreviation) only on an exact full-name match, otherwise `null` |
| Entry date | per-entry `date`, when present. **What it means is unverified**: it might be when the status changed, when the injury happened, or something else. | `source_entry_date_raw` (as supplied) and `source_entry_date_utc` (only when parseable). When absent, both are `null`. Never used as a last-update time or as evidence of freshness. |

**Nothing is inferred:**
- a DNP mentioned in a comment stays a comment and never becomes "out";
- a blank or missing field is unavailable, never "healthy".

## Capture record

Each capture is `data_files/injury_reports/captures/<capture_id>.json`.

- **Identity:** `schema_version` (1), `kind`, and `capture_id` (the request's
  UTC start second plus a random suffix).
- **Times:**

  | Field | Meaning |
  |---|---|
  | `request_started_at` | when the request was sent (UTC, microseconds) |
  | `received_at` | when the response arrived (UTC, microseconds) |
  | `observed_at` | always equals `received_at`; this is the observation time |
  | `slot.intended_utc` / `slot.intended_local` | the slot the run was for, if any |
  | `source_entry_dates` | the source's per-entry dates: `meaning: "unverified"`, how many rows have one, how many are unparseable, and the earliest and latest. When the source supplied none, `note` says they are unavailable. These are **not** status-update times and say nothing about freshness. |
  | `response.http_last_modified`, `response.http_date` | HTTP headers as received. They describe the response, not any player's status, and are never used as an entry date. |

- **Slot:** `name`, `slot_id`, `delay_minutes`, `status` (`on_time`, `late`,
  `outside_window` or `ad_hoc`) and `fills_slot`. The slot is judged at
  `received_at` (`judged_at`).
- **Scope:** the season and week, using the same rule as the sportsbook
  capture (the week of the next game to kick off at the observation time).
  Each game in that week is listed with its state: `pregame`, `kicked_off`,
  `completed` or a schedule problem.
- **Rows:** one per listed player, with the fields above plus `game_id` and
  `game_state`. A team with no game in scope gets `no_game_in_scope`; a team
  name that didn't match gets `unmatched_team`.
- **Coverage and outcome:** see the next section.
- **Provenance:** source attribution, the response's HTTP status, size and
  error (if any), the schedule's sha256, `code_revision` and
  `payload_sha256`.

## Coverage, outcomes and failures

**Each in-scope team gets one of three coverage states. None of them means the
team has no injuries:**

| State | Meaning |
|---|---|
| `rows_listed` | the feed listed players for the team |
| `listed_without_rows` | the feed listed the team with no players. That is what the source showed, not a confirmation the team is healthy. |
| `absent_from_feed` | the team doesn't appear in the response |

**Capture outcomes:**

| Outcome | When | Usable | Fills the slot |
|---|---|---|---|
| `full_team_coverage` | every in-scope team is in the feed, every feed team name matched, and no entry was malformed | yes | yes (when on time or late) |
| `partial_team_coverage` | an in-scope team is absent, a feed team name didn't match, or an entry was malformed | yes | yes (when on time or late) |
| `empty` | valid response with no player rows | no | no: kept as evidence; a later run inside the window retries |
| `unavailable` | timeout (including the total read deadline), connection error or interrupted stream, non-200, response over 32 MiB (declared or counted), invalid JSON or unexpected shape | no | no: kept as evidence with its error; retried like `empty` |

**Failure handling:**
- **Recorded, not dropped:** malformed entries (for example, no player name)
  are listed in `coverage.malformed_entries`, as are unmatched feed team
  names. Rows for teams without an in-scope game (bye weeks) are counted.
- **Bounded requests:** see *Response limits* below.

## Response limits

The response is **streamed** (`stream=True`) and read in 64 KiB chunks. It
is bounded four ways, and closed on every path (success, error, early
rejection):

| Bound | Value | Enforced how |
|---|---|---|
| Connect timeout | 5 s | `requests` |
| Per-read timeout | 20 s between chunks | `requests` |
| Total read deadline | 90 s to read the whole body | a watchdog timer shuts the connection's socket down when it fires, so even a read blocked inside one chunk returns; also checked after every chunk |
| Body size | 32 MiB (33,554,432 bytes) of **decoded** body | counted chunk by chunk; reading stops as soon as the limit is passed |

- **Content-Length:** a declared length over the limit is rejected before
  the body is read. Otherwise the header is never trusted; whether it's
  missing, wrong or unparseable, the bytes actually received are counted.
- **Compression:** with gzip or deflate, `requests` decodes as it streams.
  `urllib3` 2.x returns at most one chunk of decoded bytes per read, so the
  count is of decompressed bytes.
- **What the limit bounds:** the accumulated body bytes, at most the limit
  plus one chunk. It does **not** bound total process memory: converting to
  `bytes`, decompression buffers and the parsed JSON all need memory on top
  of that.
- **Why a watchdog:** the 20 s socket timeout only measures inactivity. A
  server trickling a byte every few seconds never trips it. With standard
  `urllib3` (which wraps `http.client`), one 64 KiB chunk read then blocks
  until the whole chunk arrives, so a check after each chunk never runs.
  - **How it works:** the watchdog shuts the socket down at the deadline,
    which makes the blocked read return in the reading thread itself. The
    capture records a `timeout` ("body not complete within 90 s"), and a
    truncated body is never parsed as if complete.
  - **Clean-up:** the timer is cancelled and joined before `fetch` returns, so
    no background thread or reader remains.
  - **Transports:** the socket is found under `urllib3` 2.x, urllib3-future
    (whose `HTTPResponse.shutdown()` is a no-op) and a bare `http.client`
    response.
- **Worst-case wall time** for one request: 5 s to connect, up to 20 s for
  the headers, then at most 90 s for the body (plus JSON parsing).
- **Why 32 MiB:** the first live response (2026-10-21) was **8,726,788
  bytes**, over the original 8 MiB cap, and was correctly archived as
  `unavailable` / `too_large`. 32 MiB gives about 3.8x headroom for a feed
  that grows during the season, while still bounding a misbehaving server.
  That historical capture is unchanged, and its Wednesday slot is not
  backfilled.
- **Evidence kept:** `response.bytes` is the decoded byte count read, up to
  the point reading stopped. A declared-length rejection leaves it `null`
  and puts the declared length in the error message.
- **No credentials stored:** error messages are capped at 500 characters and
  redacted the same way as the sportsbook capture, so no key or `apiKey=`
  value is ever stored or printed.

## Timing rules

The injury capture uses the sportsbook capture's slot rules
(`ontario_spreads.resolve_slot`, `in_slot_window`), in Toronto wall-clock time
via `zoneinfo`, so DST is handled. **Wednesday 12:00** is 16:00 UTC in EDT
and 17:00 UTC in EST; **Sunday 09:00** is 13:00 UTC in EDT and 14:00 UTC in
EST.

| Run time (Toronto) | Result |
|---|---|
| before the slot | `not_due`: no request |
| slot time to +30 min | `on_time` |
| up to the window end (Wed 15:00, Sun 11:00, inclusive) | `late` |
| after the window | `not_due`: no request. A past slot is never captured retrospectively. |
| resolved inside the window, but the window closed before the request could start (schedule loading or the lock took time) | `window_closed`: **no request**, nothing written; the run never switches to another slot or to ad hoc |
| request inside the window, response received after it | `outside_window`: kept, labelled, does **not** fill the slot |
| manual `--slot ad_hoc` | `ad_hoc`: no intended time, never fills a slot |

**Final check before the request:** the slot is resolved when the run starts.
Under the slot lock, the store is re-read: if a usable capture for the slot
appeared meanwhile, the run stops (`already_captured`). The clock is then read
again, immediately before the request, and must still be inside that same
slot's window. The closing instant counts as inside, like the sportsbook rule.
That reading becomes `request_started_at`.

**What a capture can't claim:** information received at 12:07 is labelled as
received at 12:07, with a 7.1-minute delay. A capture never claims to show
what was known at exactly 12:00, or at 09:00 for Sunday. Validation enforces
this:
- the request start must be at or after the slot time;
- the receipt must be at or after the request start;
- `delay_minutes`, `status` and `fills_slot` must follow from `received_at`;
- `observed_at` must equal `received_at`.

## Integration with the existing workflow

There is no new scheduler, and the Windows dispatcher is unchanged. The
injury steps run inside the existing `ontario-spread-capture.yml` job, **after**
every sportsbook step has finished. So they can't change the sportsbook
capture's eligibility, idempotency, retries, commit, persistence report or
Odds API credit use.

| Step | Runs when | Does |
|---|---|---|
| Capture injury report | the run reached the latest main (skipped when that refresh failed, like the capture) | `python injury_reports.py capture --slot "$SLOT"`, using the same `slot` input |
| Commit injury report capture | on `main`, when a new file was written | stages only **this run's file**, named in the capture step's `file` output; refuses if the sportsbook steps left an unpushed commit, so it never pushes one for them; `git pull --rebase` and a plain push, 3 tries |
| Upload injury report capture | a new file was written | artifact `injury-report-capture-<run>-<attempt>` with exactly that file (`if-no-files-found: error`, so "success" means it was uploaded), kept 90 days |
| Report injury capture persistence | the capture step ran | `python injury_reports.py persistence` (below) |

**Persistence is verified, not assumed.** A clean working tree isn't proof:
if the local commit succeeds but its push fails, the tree is clean but the
file never reached `main`. The capture step outputs this run's file path and
its sha256. The report step then fetches remote `main`, compares that file's
content with the sha256, and reports one state:

| State | Meaning | Exit |
|---|---|---|
| persisted | on remote `main` with identical content | 0 ✅ |
| persistence unverified | remote `main` couldn't be fetched or read; never treated as success | 1 |
| remote content differs | remote `main` has that path with other content | 1 |
| committed locally only | in the local commit but not on remote `main` (push failed or was refused) | 1 |
| never committed | only in the runner's working tree (for example, the commit was refused because of an unpushed sportsbook commit) | 1 |
| missing locally | not even in the working tree | 1 |

Unless the file is persisted, the summary names the artifact holding a copy,
or says the capture is **not persisted anywhere** if the upload failed too.

**Failure visibility and its trade-off:**
- **Never fails the run:** every injury step is `continue-on-error`, and none
  receives `ODDS_API_KEY`. That keeps the run's conclusion exactly as the
  sportsbook steps leave it, because the Windows dispatcher reads that
  conclusion.
- **Shown instead:**
  - an `unavailable` or `empty` capture, or a step failure, gives an
    `::error::` annotation and a ❌ summary section;
  - `partial_team_coverage` gives a `::warning::`.
- **The cost:** a green run can hide an injury failure, so check the summary.

**Retries:** an unusable injury capture leaves the slot open. A later run of
the workflow inside the same window retries it: the second cron entry, or a
manual run with the slot name. A successful (full or partial) capture for the
slot makes later runs skip without a request.

**Retry limit:** the dispatcher dispatches only for missing *sportsbook*
captures. Once the sportsbook slot is captured, an injury failure isn't
re-dispatched, so that slot may have no usable injury capture. Each such gap
stays visible (`python injury_reports.py list`), and nothing is backfilled.

## Commands

```bash
python injury_reports.py capture --slot auto   # the due slot; no request if none is due
python injury_reports.py capture --slot ad_hoc # labelled ad hoc; never fills a slot
python injury_reports.py list [--last N]       # validate all, then list times, source, scope, coverage
python injury_reports.py validate              # validate every stored capture
python injury_reports.py persistence ...       # workflow only: verify this run's file reached main
```

`python injury_reports.py persistence ...` is the workflow's report step;
it takes the capture step's outputs and the upload outcome.

`list` and `validate` read every capture and fail closed (`::error::`, exit 1)
if any file is invalid: bad JSON, an unknown schema, a missing field, a
checksum mismatch, inconsistent timing or coverage, or a file name that
doesn't match its `capture_id`. `capture` also refuses to run (no request)
while the store is invalid.

## Immutability

Files are created exclusively (temp file plus hard link) and never replaced.
Writing identical content again is a no-op. Different content under an
existing name is a conflict, and the existing file is left untouched. A
per-slot lock file (`.<slot_id>.lock`, git-ignored) stops two runs from
capturing the same slot at once.

**Checksum limit:** `payload_sha256` is an unkeyed SHA-256 of the canonical
document. It detects accidental changes and corruption. It does **not**
detect a deliberate edit followed by recomputing the checksum. Git history
and the workflow artifacts are the record of what was originally written.

**Two different hashes:**
- **The workflow's file sha256:** a hash of the file's exact bytes as
  committed (LF line endings). The persistence check compares it with the
  blob on remote `main`.
- **`payload_sha256`:** a hash of the canonical JSON content. It doesn't
  depend on line endings or formatting.
- **On Windows:** a checkout with `core.autocrlf=true` converts the file to
  CRLF, so its raw-file hash differs from the committed blob, while the JSON,
  `payload_sha256` and validation are identical. Compare against the
  committed blob (`git show origin/main:<path> | sha256sum`), not the
  working-tree file.

## How later analysis could use this, and its limits

A later phase could pair each Wednesday and Sunday Ontario sportsbook capture
with the injury capture for the same slot. It could then describe which
listed statuses changed between the two decision times, next to how the
spreads moved.

**Limits on any such analysis:**
- **Different times:** the sportsbook quote, the injury capture and the
  nightly model snapshot are observed at different moments. Injury news can
  arrive between them, so co-occurrence is never causation.
- **Not the official report:** the feed is ESPN's status, not the official
  NFL report, and it has no practice data. Missing rows aren't "healthy".
- **No verified update times:** the meaning of ESPN's entry `date` is
  unverified, so the archive can't say how fresh any status was.
- **Payload compatibility still pending:** as of the first live run, no
  capture has parsed a real response. That run was rejected as too large
  before parsing, and no raw ESPN response is archived anywhere in the
  repository. So the field handling (status words, athlete ids, comments,
  entry dates) is tested only against hand-written fixtures, until a live
  capture with rows exists. The nightly `espn_injuries.csv` is a flattened
  extract, not evidence of the raw JSON schema.
- **No history before deployment:** slots before this phase have no injury
  capture, and nothing is reconstructed from a later feed. Until the archive
  covers both decision times of a week,
  `scripts/ontario_waiting_report.py` keeps saying injury effects can't be
  determined.
- **Few observations:** two per week, with books and sides that aren't
  independent.
