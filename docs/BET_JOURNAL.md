# Bet journal

The **Bet Journal** page (`pages/8_Bet_Journal.py`, logic in `bet_journal.py`)
records single-game spread and moneyline wagers **you have actually placed**
at Ontario sportsbooks - cash-funded or as a sportsbook bonus bet - on the
terms the sportsbook accepted. It records wagers; it
never places them.

It is kept separate from everything else:

- the model's simulated recommendations (`betting_recommendations_log.csv`);
- sportsbook line observations (`data_files/ontario_spreads/`);
- frozen predictions (`data_files/pregame_snapshots/`).

It reads only `data_files/nfl_games_historical.csv`, for the game list and the
completion evidence. It makes no API calls, creates no recommendations and
writes nowhere but `data_files/bet_journal/`.

**Scope:** single-game spread or moneyline bets, funded entirely with cash
or entirely with one bonus bet. **Not covered:** parlays, odds boosts,
cash-outs, mixed cash and bonus stakes, other promotions, links to model
snapshots and line-timing analysis. The journal never creates moneyline or
any other recommendations.

## Using it

1. **Record a wager.**
   - Pick the game and the team you bet on.
   - Choose the **market** (Spread or Moneyline) and the **funding** (Cash or
     Bonus bet).
   - Enter what the bet slip shows:
     - the Ontario sportsbook (choose "Other Ontario-licensed" and type its
       name if it isn't listed);
     - for a spread: the **signed spread** for your team (`-3.5` lays 3.5,
       `+3.5` gets 3.5, `0` is pick'em). A moneyline has no spread field;
     - for a moneyline: what the sportsbook does **if the game ends in a
       tie**, from its rules: refunded (push) or settled as a loss. It is
       required and never assumed;
     - the accepted **American odds**;
     - the **stake in CAD**, or for a bonus bet its **face value**, in
       dollars and cents, more than $0, at most $100,000;
     - for a bonus bet: whether the promotion **returns the bonus stake as
       cash with the winnings** (most don't), from its terms. It is required.
       A stake reissued as new bonus credit is not cash: record it as not
       returned;
     - the placement date and time in **America/Toronto**;
     - optionally, the sportsbook's bet reference and a note.

   A pregame wager can be recorded later ("late entry"): games from the past
   180 days are offered as well as the next 14 days.
2. **Preview.** It shows the exact team, market (signed handicap or
   "moneyline"), odds, stake or bonus amount, bonus-return terms, tie rule and
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

- **Result, by market:**
  - **Spread:** `betting_log.spread_result`, the same reviewed rule as the
    recommendations log: your team's final margin plus your signed handicap.
    Above 0 wins, exactly 0 pushes, below 0 loses. This is unchanged.
  - **Moneyline:** your team wins outright → win, loses → loss. A **tie**
    follows the wager's recorded tie rule: refunded → push, settles as a loss
    → loss. A tie is never assumed to be a loss.
- **Win multiplier** at the accepted odds: 100 / |odds| for negative odds,
  odds / 100 for positive odds. Wins are rounded half-up to the cent.
- **Cash result (CAD), exact to the cent:**

  | Funding | Win | Loss | Push / refunded tie |
  |---|---|---|---|
  | Cash | stake × multiplier | −stake | $0.00 |
  | Bonus bet, stake **not** returned | bonus × multiplier | $0.00 | $0.00, bonus outcome **unresolved** |
  | Bonus bet, stake returned as cash | bonus × multiplier + bonus | $0.00 | $0.00, bonus outcome **unresolved** |

  Example: a $25 bonus bet at +320, stake not returned, wins **$80.00**
  cash (not $105), and loses **$0.00** cash (not −$25). "Returned" means
  returned **as cash**; promotional credit reissued by the sportsbook is never
  counted as cash winnings.
- **Bonus outcome** (stored on each bonus-bet grade): `used` after a win or
  loss. After a push or refunded tie it is `unresolved`: whether the
  sportsbook reissues the bonus isn't recorded, so it is never counted as
  cash or as used.
- **ROI** (cash bets only) = net profit ÷ total cash stake of **graded
  current cash-funded** wagers. Bonus bets never enter ROI, and bonus face
  value is never counted as cash staked.
  - **Graded** means wins, losses and pushes, so a push adds its stake to
    the denominator and $0 to profit.
  - **Pending** (no evidence yet, or graded only on terms since amended),
    **unverified** and **voided** wagers are excluded.
  - With nothing graded, ROI is "n/a".
- **Pending stake** is the cash stake of pending and unverified wagers;
  pending bonus stake is shown separately.
- **Bonus-bet totals**, kept apart from cash:
  - **bonus stake used** (face value of graded bonus wins and losses);
  - **cash won from bonus bets**;
  - **unresolved bonus outcomes** (count and face value);
  - **pending bonus stake**.
- **Net cash incl. bonus winnings** = cash-bet net profit + cash won from
  bonus bets. It is a clearly labelled combination; ROI is never computed on
  it.
- **W-L-P and pending counts** cover every current wager, of both markets
  and both fundings.
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

## Markets, funding and schema versions

Records carry `schema_version`. An unknown version is rejected, never
guessed.

- **Schema 1:** the original records. A wager's terms are a **cash-funded
  spread** bet (`stake_cad`, signed `handicap`).
  - **Unchanged:** every existing file stays valid with its checksum, and
    nothing is rewritten.
  - **Legacy defaults:** market `spread`, funding `cash`, the stake as the
    cash stake, no bonus stake, no tie rule.
  - **Still used for new records:** cash-funded spread wagers, their
    amendments and grades, and every void and invalidation. That keeps one
    encoding per wager.
- **Schema 2:** written only for terms schema 1 can't express (a moneyline
  or a bonus bet).
  - **Terms fields:** `market` (`spread` / `moneyline`), `handicap` (`null`
    for a moneyline, never 0), `tie_rule` (`push` / `loss` for a moneyline,
    otherwise `null`), `cash_stake_cad`, `bonus_stake_cad` and
    `bonus_stake_returned` (`true` / `false` for a bonus bet, otherwise
    `null`; `true` only when the stake is returned as cash on a win).
  - **Funding:** exactly one of the two stakes is above zero; mixed funding
    is rejected.
  - **Grades:** a schema-2 grade adds `market`, `funding` and
    `bonus_outcome`. It must match its terms' schema, market and funding, and
    its result, cash result and bonus outcome must follow from the terms and
    scores.
- **Amendments:** an amendment may change the market or funding (same
  game). A grade of the earlier terms then no longer applies, and the next
  grading writes a superseding grade with the new market and funding.
  Invalidations keep the same chain rules for every market and funding.

## Duplicates vs. a genuine second wager

**Identical** means the same:

- sportsbook (and its name);
- game;
- market;
- team;
- signed handicap (none for a moneyline);
- odds;
- cash stake, bonus stake and the bonus-return terms;
- moneyline tie rule;
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
- **Timestamp format:** UTC with `Z`, either whole seconds
  (`2026-10-07T17:00:00Z`) or exactly six fractional digits
  (`2026-10-19T11:25:38.086319Z`).
  - **Which fields:** `recorded_at` keeps the clock's microseconds;
    `placed_at` (whole minute) and `kickoff_utc` are whole seconds.
  - **One spelling per instant:** only the form the journal writes is
    accepted. `.000000`, other digit counts and offsets are rejected.
  - **Older records:** whole-second records remain valid.
  - **Comparison:** records are compared and listed by time, not as text.
  - **Not relaxed elsewhere:** pregame snapshots and Ontario captures keep
    their own whole-second rule.
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

## Excel export (viewing copy)

On the page, open **Excel export (viewing copy)** and click **Prepare Excel
export**, then **Download Excel**. The file is named after the time it was
built in Toronto, for example `bet_journal_20261019_143000.xlsx`.

**It's a view, not the journal.** The JSON records stay authoritative:

- Editing the workbook never changes the journal.
- It isn't a complete backup, because checksums and other record fields are
  left out. Back up `data_files/bet_journal/` itself (see below).
- It's built in memory only when you click. No spreadsheet is kept beside
  the journal, and nothing is written to disk.

**Consistent as of preparation.** All three sheets come from one validated
read of the journal (`bet_journal_export.read_journal`), and never from the
page's cache.

- If files change during the read, it is repeated.
- A prepared workbook shows the journal as it was at preparation time; the
  page shows when it was prepared. It isn't updated afterwards.
- If the journal changes later (a save, correction or grading), the page
  discards that workbook on its next rerun and asks you to prepare again.
  There's no background polling, so prepare again after any change before
  downloading.
- Any invalid record, broken link or stray file refuses the export with an
  integrity error. A partial workbook is never produced.

**No new calculations.** Totals, statuses, grades and history order come
from `totals()`, `current_state()` and `history_rows()`.

### Sheets

- **Summary:**
  - the export time (America/Toronto and UTC), the currency (CAD) and the
    journal totals: cash-bet totals and ROI, and the bonus-bet totals
    separately;
  - which wagers enter each total;
  - the wagers left out of profit and ROI, with the reason: void, pending
    or unverified.
- **Wagers:** one row per wager, on its current effective terms:
  - game, season and week, team, sportsbook, market, signed handicap (blank
    for a moneyline), American odds, funding, cash stake and bonus stake,
    the bonus-return terms and the moneyline tie rule;
  - placement time (Toronto and UTC) and recording time (precise UTC);
  - status, result, profit and how the profit was settled;
  - grade verification, the exclusion note, reference and note.

  Profit is **blank** for pending, unverified and voided wagers, never a
  settled $0.00. A push shows $0.00 with the result `push`.
- **History:** every record (wagers, amendments, voids, grades and
  invalidations), in the History tab's order:
  - precise recording times;
  - the record each one follows (previous or superseded);
  - the terms record, result, profit, stake, details, reason and code
    revision.

### Formatting and safety

- **Header rows** are frozen and filtered, and columns have readable widths.
- **Amounts** are numeric CAD cells, and ROI is a numeric percentage.
- **Handicap and odds** are numeric, with a format that always shows the
  sign (`+3.5`, `-110`).
- **Identifiers, references, notes and precise timestamps** are text cells,
  so nothing is truncated and microseconds are kept.
- **No formulas:** text is never stored as a formula. Notes, references and
  sportsbook names beginning with `=`, `+`, `-` or `@` stay literal, with
  Excel's quote prefix so they also stay literal when edited.
- **Dependency:** `openpyxl` (pinned in `requirements.txt`).

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
