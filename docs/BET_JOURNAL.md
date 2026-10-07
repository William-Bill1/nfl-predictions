# Bet journal

The **Bet Journal** page (`pages/8_Bet_Journal.py`, logic in `bet_journal.py`)
records single-game spread wagers **you have actually placed** at Ontario
sportsbooks, on the terms the sportsbook accepted. It records wagers; it
never places them.

It is kept separate from everything else:

- the model's simulated recommendations (`betting_recommendations_log.csv`);
- sportsbook line observations (`data_files/ontario_spreads/`);
- frozen predictions (`data_files/pregame_snapshots/`).

It reads only `data_files/nfl_games_historical.csv`, for the game list and the
completion evidence. It makes no API calls, creates no recommendations and
writes nowhere but `data_files/bet_journal/`.

**Scope:** single-game spread bets only. Parlays, cash-outs, promotions,
links to model snapshots and line-timing analysis are not covered.

## Using it

1. **Record a wager.**
   - Pick the game and the team you bet on.
   - Enter what the bet slip shows:
     - the Ontario sportsbook (choose "Other Ontario-licensed" and type its
       name if it isn't listed);
     - the **signed spread** for your team (`-3.5` lays 3.5, `+3.5` gets 3.5,
       `0` is pick'em);
     - the accepted **American odds**;
     - the **stake in CAD** (dollars and cents, more than $0, at most
       $100,000);
     - the placement date and time in **America/Toronto**;
     - optionally, the sportsbook's bet reference and a note.

   A pregame wager can be recorded later ("late entry"): games from the past
   180 days are offered as well as the next 14 days.
2. **Preview.** It shows the exact team, signed handicap, odds, stake and
   placement time, labelled a *user-confirmed placed wager*. Nothing is
   written.
3. **Save.** Save writes exactly the previewed terms. If you change anything
   after Preview, Save refuses and asks you to preview again.
4. **Grade settled wagers** (button). This settles every current wager the
   schedule shows as completed (see
   [Grading evidence](#grading-evidence-conservative)). Opening the page
   never grades anything.
5. **Correct or void.**
   - Wagers are never edited or deleted.
   - **Amend terms** appends new terms for the same game, with a required
     reason. To change the game, void the wager and record a new one.
   - **Void** (e.g. entered by mistake) appends a void record, also with a
     required reason.
   - The original and every correction stay in **History (audit)**.
   - **Changed since Preview:** if the wager changed after your Preview (for
     example in another tab or process), Save refuses and asks you to preview
     again.

### Placement time rules

The time is entered in America/Toronto and converted to UTC.

- **DST:** a time that happens twice (when DST ends) or doesn't exist (when
  DST starts) is rejected.
- **Not in the future:** the placement must not be later than now. It is
  recorded to the minute.
- **Before kickoff:** the placement must be strictly before the scheduled
  kickoff. The kickoff minute itself is rejected.

## Grading evidence (conservative)

The nflverse schedule has **no authoritative "final" flag**, and kickoff
having passed proves nothing. So a game is graded only when its schedule row
gives all of this **conservative evidence of completion**:

- **One row:** exactly one row for the game. A missing or duplicated game
  stays pending.
- **Scores:** `home_score` and `away_score` are finite, non-negative whole
  numbers, and **not 0–0** (unplayed games can carry 0–0 placeholders).
- **Consistency:**
  - `result` = home − away;
  - `total` = home + away;
  - `overtime` is 0 or 1.

  In nflverse these columns are filled together only once a game is complete.
- **Game day:** strictly before today in America/Toronto. A game is never
  graded on its own day, which rules out live partial scores.

Anything else stays **pending**, with the reason recorded. This is
deliberately conservative evidence, not an official final status. A score the
schedule later corrects is handled through supersession and invalidation
(below).

## Settlement, totals and ROI

- **Result.** The result uses `betting_log.spread_result`, the same reviewed
  rule as the recommendations log: your team's final margin plus your signed
  handicap. Above 0 wins, exactly 0 pushes, below 0 loses.
- **Profit (CAD), exact to the cent:**

  | Result | Net profit |
  |---|---|
  | Win at negative odds | stake × 100 / \|odds\| |
  | Win at positive odds | stake × odds / 100 |
  | Loss | −stake |
  | Push | $0.00 |

  Wins are rounded half-up to the cent.
- **ROI** = net profit ÷ total stake of **graded current** wagers.
  - **Graded** means wins, losses and pushes, so a push adds its stake to
    the denominator and $0 to profit.
  - **Pending** (no evidence yet, or graded only on terms since amended),
    **unverified** and **voided** wagers are excluded.
  - With nothing graded, ROI is "n/a".
- **Pending stake** includes pending and unverified wagers.
- **Voided wagers** are excluded from every total and shown only in History
  (or in the current table with "Include voided wagers").
- **Amended wagers** count once, on their latest terms.

### Grading records, corrections and invalidation

Each wager has a **settlement chain** of grade and invalidation records.
Every record names the record it supersedes. Each click of **Grade settled
wagers** compares the chain's latest record with the current schedule
evidence:

| Situation | Writes |
|---|---|
| Evidence, and no grade yet for the current terms | **grade** ("initial grade", "terms amended since the last grade" or "evidence restored after invalidation") |
| A grade exists, and the evidence changed | **grade** superseding it ("score correction: …") |
| A grade exists for the current terms, but the evidence no longer supports it (scores removed, inconsistent fields, a duplicated row, 0–0) | **invalidation** superseding it |
| Anything else | nothing |

Running it again with unchanged data writes nothing.

**Grade contents:**

- the result and profit;
- the evidence: `gameday`, both scores, `result`, `total`, `overtime`;
- the SHA-256 of the **exact schedule bytes parsed**;
- the code revision;
- `terms_record_id`: the exact wager or amendment it graded, with a hash of
  those terms.

When the journal loads, the result and profit are recomputed from those terms
and that evidence, and they must match.

**Invalidation:**

- **Effect:** an invalidated wager shows as **unverified**, with the reason,
  and the page warns about it. It is out of profit and ROI until the evidence
  supports a fresh grade.
- **History:** the withdrawn grade stays in History.

## Duplicates vs. a genuine second wager

**Identical** means the same:

- sportsbook (and its name);
- game;
- team;
- signed handicap;
- odds;
- stake;
- placement minute;

and the two wagers' sportsbook references don't differ. If both have a
reference and the references differ, they are different wagers.

- **New wagers:** an identical entry is refused as a duplicate unless you tick
  **"This is a separate, second wager…"** and preview again. That covers a
  rerun, double click, resubmission, or another session or process saving
  concurrently. The record lists the wager IDs you confirmed it is separate
  from (`confirmed_existing`).
- **Amendments:** corrected terms identical to another current wager need the
  same explicit confirmation, via the separate-wager box in the correction
  form. The amendment records `confirmed_existing` too. If this is really the
  same wager entered twice, void one instead.
- **Check at save:** under the journal lock, Save requires the identical
  wagers on record to be **exactly** the set confirmed at Preview. Otherwise
  it refuses and asks you to preview again.
- **References:** a sportsbook reference already recorded at the same
  sportsbook is always refused, for new wagers and amendments.
- **Double saves:** each preview also has a one-time token, so the same
  preview is never saved twice.

## History model and integrity

Every record is one file, `data_files/bet_journal/<record_id>.json`.

- **IDs:** like `W-20261007T170000Z-<12 hex>`. The prefix is `W` wager,
  `A` amendment, `V` void, `G` grade or `X` invalidation. The ID timestamp
  matches the UTC second of `recorded_at`; `recorded_at` retains subsecond
  precision to validate chronology between independent chains.
- **Written once:**
  - via a temporary file and an exclusive link, so a record is never
    overwritten;
  - under the journal lock `.journal_write.lock`, with the same owner and
    recovery rules as the Ontario locks.
- **Fields on every record:**
  - `schema_version`, `kind`, `record_id`, `recorded_at`, `code_revision`;
  - `payload_sha256`: SHA-256 of the canonical record without that field.
- **Unkeyed checksum:** it detects **accidental** damage, not a deliberate
  edit followed by recomputing the checksum.

### Explicit links

History is ordered by **explicit links**, never by timestamps or file names:

- **Terms chain:** `wager → amendment → … [→ void]`. Each amendment and the
  void name their predecessor in `previous`.
- **Settlement chain:** `grade → grade | invalidation → …`, linked through
  `supersedes`.

A save must link to the **current tip**, checked under the lock. When two
corrections are made from the same Preview state, only the first succeeds;
the second is refused, not silently ordered.

### What loading rejects

Loading rejects, with an **integrity error** naming the file, and nothing
repaired:

- **Bad files:** unreadable JSON, NaN or Infinity, a schema or checksum
  mismatch, a wrong file name, or a stray file.
- **Bad terms:** invalid values, or season/week/teams that don't match the
  `game_id`.
- **Bad links:**
  - a link to a missing record, a record of the wrong kind, or another
    wager's chain;
  - a **fork** (two records naming the same predecessor);
  - a **cycle**, or any record not reachable along the chain;
  - a **duplicate void**, or anything after the void;
  - a settlement not timestamped strictly before its wager's void, or after
    its terms were superseded;
  - a `confirmed_existing` reference that doesn't name an active, identical
    earlier wager at the time of the confirmation;
  - more than one first grade;
  - an invalidation that doesn't supersede a grade.
- **Bad timestamps:** a record recorded before the record it links to,
  including confirmed second-wager references to later or unknown wagers.
- **Bad grades:** evidence that isn't consistent; a `terms_sha256`, result or
  profit that doesn't follow from the referenced terms and evidence; or a
  grade naming terms outside its own wager's chain.

**Effect:** the page shows the error and **no totals**. Saving and grading
also refuse, so a damaged history is never extended or silently resolved.

## Backup and restore

`data_files/bet_journal/` is **git-ignored** because these are personal
wagers. It lives only on this machine and isn't deployed to Streamlit Cloud.
**Back it up yourself.**

The history is a chain of links across files, so always copy the **whole
directory**.

- **Broken links are caught:** a copy that drops a record which another
  record links to fails validation.
- **A missing latest record is not:** if a copy drops only the *latest*
  record of a chain (e.g. the newest grade or amendment), nothing links to
  it, so the copy still validates. There is no manifest.

Validation alone cannot detect that a terminal record was already missing
before the backup. Keep an **external inventory** from a known-complete
journal (record count plus sorted record IDs and per-file SHA-256 hashes) to
detect that loss; a hash comparison between the current source and its copy
only proves the copy matches the current source.

### Backup

1. Stop the app and every process that can save or grade. Keep all writers
  stopped through the copy and its verification. Check that
  `data_files/bet_journal/.journal_write.lock` doesn't exist.
2. Validate the journal:

   ```bash
   python bet_journal.py validate
   ```

   It's read-only. It prints `OK: <records>, <wagers> …` (exit 0), or
   `INTEGRITY ERROR: …` (exit 1).
3. Copy the directory to a dated backup outside the repo, e.g. PowerShell:

   ```powershell
   Copy-Item -Recurse data_files\bet_journal "$HOME\Backups\bet_journal-$(Get-Date -Format yyyyMMdd-HHmm)"
   ```

   or bash:

   ```bash
   cp -a data_files/bet_journal ~/Backups/bet_journal-$(date +%Y%m%d-%H%M)
   ```

4. Validate the copy:

   ```bash
   python bet_journal.py validate --dir <backup dir>
   ```

   It must print `OK` with the **same record and wager counts** as step 2.

5. Compare every relative file name and SHA-256. The commands below should
   produce no differences. Keep the app stopped until this check is done.

   PowerShell:

   ```powershell
   function Get-JournalHashes($dir) {
       $root = (Resolve-Path -LiteralPath $dir).Path.TrimEnd('\') + '\'
       Get-ChildItem -LiteralPath $dir -File -Recurse | ForEach-Object {
           $relative = $_.FullName.Substring($root.Length)
           $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
           "$relative $hash"
       } | Sort-Object
   }
   $source = Get-JournalHashes 'data_files\bet_journal'
   $copy = Get-JournalHashes "$HOME\Backups\bet_journal-YYYYMMDD-HHmm"
   Compare-Object $source $copy
   ```

   Bash (Git Bash or Linux):

   ```bash
   diff -u \
     <(cd data_files/bet_journal && find . -type f -print0 | sort -z | xargs -0 sha256sum) \
     <(cd "$HOME/Backups/bet_journal-YYYYMMDD-HHmm" && find . -type f -print0 | sort -z | xargs -0 sha256sum)
   ```

### Restore

1. Stop the app.
2. Validate the backup:

   ```bash
   python bet_journal.py validate --dir <backup dir>
   ```

3. Move the current journal **aside**; don't delete it:

   ```powershell
   Rename-Item data_files\bet_journal bet_journal.before-restore
   ```

4. Copy the **whole** backup directory into place:

   ```powershell
   Copy-Item -Recurse <backup dir> data_files\bet_journal
   ```

   If a `.journal_write.lock` came along with the backup, delete it, but only
   after confirming that the process it names is no longer running.
5. Validate the restored journal: `python bet_journal.py validate`. It must
   print `OK` with the same counts as the backup in step 2.
6. Start the app and check the totals and History. Keep
   `bet_journal.before-restore` until you're satisfied.

A single damaged record can be restored from the backup on its own only if it
is byte-for-byte the same file. Run `validate` afterwards in every case.

## Developer notes

- The page caches the parsed journal on a SHA-256 fingerprint of every file
  in the directory, so any new, changed or stray file is re-read and
  re-validated.
- Tests are in `tests/test_bet_journal.py`. They use temporary directories,
  block sockets and `requests`, fix the clock, and render the page with
  AppTest. Cross-process tests run independent Python processes on a shared
  temporary journal, with a start barrier, bounded waits and guaranteed
  cleanup.
