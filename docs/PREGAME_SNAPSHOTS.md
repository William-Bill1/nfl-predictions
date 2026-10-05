# Pregame spread snapshots

An auditable, append-only record of what the spread model predicted **before
kickoff**. The nightly pipeline rewrites
`nfl_games_historical_with_predictions.csv` for every game, played ones
included, so that CSV can't show what was predicted at the time. Snapshots
freeze each run's predictions for the **eligible upcoming games** (defined
below), including games with no bet signal, no line, or a pick'em line.

Snapshots record predictions only. They are not evidence that the model has a
betting edge; see `docs/ROLLING_SPREAD_BACKTEST.md` for how it scores.

Code: [`pregame_snapshots.py`](../pregame_snapshots.py). Tests:
`tests/test_pregame_snapshots.py`.

## How it fits the nightly run

```
nfl-gather-data.py
  start: delete data_files/pipeline_run_manifest.json
         read the schedule bytes ONCE -> hash them -> parse the same bytes
  ...train / predict as before (unchanged)...
  end:   assign run_id, write data_files/pipeline_run_manifest.json   (only if the run finished)
pregame_snapshots.py capture        (nightly step right after the pipeline)
  read schedule + predictions bytes ONCE each -> check hashes vs manifest -> parse those bytes
  write data_files/pregame_snapshots/<run_id>.json      (new file, never replaced)
```

### Failure reporting in the nightly workflow

The capture step (`id: pregame_capture`) runs only after the pipeline step
succeeds. On success it writes its result line to the run's step summary.

It keeps `continue-on-error: true` **deliberately**. Failing immediately would
skip committing that night's predictions, grading the betting log and
exporting best bets. A failure is not silent, though. The last step, "Fail the
run if the pregame capture failed", runs under `always()` whenever capture
failed. It emits an `::error::` annotation, writes a "❌ capture FAILED"
section to the step summary, and exits 1. The run is therefore marked
**failed**, after the data has been published. The Summary step reports the
capture outcome; it doesn't claim the nightly update succeeded.

## Commands

```bash
python pregame_snapshots.py capture                       # after a pipeline run
python pregame_snapshots.py select --which earliest       # one row per game, printed
python pregame_snapshots.py select --which latest --status predicted --output picks.csv
python pregame_snapshots.py select --which latest --before 2026-10-04T12:00:00Z
```

`select` only reads. It validates every snapshot file first and refuses to
write its output inside the snapshot directory.

## Run manifest: `data_files/pipeline_run_manifest.json`

Written by `nfl-gather-data.py` as the last step of a successful run. It's kept
separate from the deterministic artifacts (`nfl_games_historical_with_predictions.csv`,
`model_metrics.json`, `best_features_spread.txt`), so its timestamped fields
don't affect the CI determinism check.

| Field | Meaning |
|---|---|
| `run_id` | the generating run's identity, assigned **once** when the manifest is written: `<generated_at %Y%m%dT%H%M%SZ>-<12 random hex>`. It's unique even for identical inputs. Capture retries reuse it and never mint a new one |
| `generated_at` | UTC time the run finished writing predictions (whole seconds) |
| `code_revision`, `code_dirty` | `GITHUB_SHA` in Actions, otherwise `git rev-parse HEAD` plus whether tracked files were modified; `unknown` / null without git |
| `code_sha256` | hash of `nfl-gather-data.py` and `team_features.py` as executed |
| `config`, `config_id` | the fitted spread models' `get_params(deep=True)`, the split defaults, the EV min edge, library versions; `config_id` = first 16 hex of its canonical-JSON sha256 |
| `features`, `feature_set_id` | the spread model's training columns (sorted) and their hash |
| `training_cutoff` / `data_cutoff` | last (season, week, gameday) and game count of the model-fitting slice and of all played, lined games |
| `spread_threshold` | the bet-signal probability threshold the run used |
| `artifact.sha256` / `schedule.sha256` | SHA-256 of the predictions CSV it wrote and of the **exact schedule bytes it parsed** |
| `ci_run`, `platform` | `GITHUB_RUN_ID` / `GITHUB_RUN_ATTEMPT` / `GITHUB_WORKFLOW` when present; Python version |

Provenance always comes from the run that produced the predictions. A capture
never stamps the current checkout's revision onto an older CSV. The manifest is
validated for structure and internal consistency: `config_id` and
`feature_set_id` must match the recorded values, and the `run_id` timestamp
must match `generated_at`. The predictions CSV and schedule are each read once,
hash-checked, and parsed from those same bytes. Any mismatch raises
`ProvenanceError` and nothing is written. A run that fails leaves **no**
manifest, because the stale one was deleted when the run started.

### Line endings: hashed files are byte-exact

The manifest hashes the **exact bytes** of the schedule and predictions files,
line endings included, and verification never normalizes them. Git for
Windows defaults to `core.autocrlf=true`, which converts LF to CRLF on
checkout. That gave a correct checkout different bytes from the ones the
nightly committed, so the provenance check failed. `.gitattributes` therefore
marks exactly the two hashed files `-text`, so Git checks them out (and
commits them) byte-for-byte:

```
data_files/nfl_games_historical.csv -text
data_files/nfl_games_historical_with_predictions.csv -text
```

Other files keep normal line-ending handling. Snapshot files don't need the
rule: their checksum is computed over the parsed, canonical JSON, not the
file bytes.

**Refreshing a checkout made before the rule.** New attributes don't rewrite
files that are already checked out. Git's cached file metadata may also still
mark them as clean, so `git status` can show nothing and a plain
`git checkout -- <file>` can leave the converted bytes in place. Check each of
the two files first:

```
git diff --ignore-cr-at-eol --quiet -- <file>
```

| Exit code | Meaning | What to do |
|---|---|---|
| `0` | No content differences from the committed file once CRs at line ends are ignored. This does **not** show that a line-ending difference exists; the file may already be byte-identical. | Safe to restore: delete the file and check it out again. That always writes the committed bytes, and nothing is lost because there are no content edits. |
| `1` | Real content differences: local edits, or data regenerated locally. | **Leave the file alone.** Don't check it out, and don't delete it. Edited data won't match the committed manifest anyway, even after converting its line endings, so there's nothing to gain. If you want the committed file back, save a copy of your version elsewhere first and decide yourself. |
| `>1` | The command failed (not a repository, bad path, and so on). | Change nothing; fix the error and rerun. |

PowerShell (Windows):

```powershell
foreach ($f in 'data_files/nfl_games_historical_with_predictions.csv',
               'data_files/nfl_games_historical.csv') {
    git diff --ignore-cr-at-eol --quiet -- $f
    $rc = $LASTEXITCODE
    if ($rc -eq 1) {
        Write-Host "$f has content changes - left untouched"
    } elseif ($rc -ne 0) {
        Write-Host "git diff failed for $f (exit $rc) - nothing changed"
    } else {
        try {
            Remove-Item -LiteralPath $f -ErrorAction Stop
        } catch {
            Write-Host "could not delete $f - nothing changed: $($_.Exception.Message)"
            continue
        }
        git checkout -- $f
        if ($LASTEXITCODE -eq 0) {
            Write-Host "$f restored to the committed bytes"
        } else {
            Write-Host "git checkout failed for $f (exit $LASTEXITCODE) - run: git checkout -- $f"
        }
    }
}
```

Bash (Git Bash, Linux, macOS):

```bash
for f in data_files/nfl_games_historical_with_predictions.csv data_files/nfl_games_historical.csv; do
    git diff --ignore-cr-at-eol --quiet -- "$f"; rc=$?
    case $rc in
        0) if ! rm -- "$f"; then echo "could not delete $f - nothing changed"
           elif git checkout -- "$f"; then echo "$f restored to the committed bytes"
           else echo "git checkout failed for $f - run: git checkout -- $f"; fi ;;
        1) echo "$f has content changes - left untouched" ;;
        *) echo "git diff failed for $f (exit $rc) - nothing changed" ;;
    esac
done
```

Both scripts report every outcome. A failed delete leaves the file as it
was. On Windows, deletes can be refused while another program (an editor,
the app, antivirus) has the file open; close it and rerun. If the checkout
fails after a successful delete, nothing is lost: the file had no content
edits, so `git checkout -- <file>` brings it back.

Afterwards, `git ls-files --eol <file>` shows `w/lf` for a restored file. A
fresh clone, or any later pull that updates these files, gets the exact bytes
automatically. `python scripts/check_pipeline_outputs.py` then passes for
committed, unedited data.

**Running the pipeline on Windows.** pandas writes CRLF on Windows. A local
run therefore produces CRLF files, and its manifest records the hashes of
those CRLF bytes, so local capture and verification stay consistent. Because
the files are `-text`, Git now shows them as changed in full against the
committed LF versions, and committing them would commit CRLF bytes. Before,
autocrlf converted them to LF on commit, which silently broke their match
with the manifest. Don't commit locally generated data. The nightly on Linux
produces the committed versions.

## Snapshot file: `data_files/pregame_snapshots/<run_id>.json`

Top level: `schema_version` (1), `kind` (`pregame_spread_snapshot`),
`run_id`, `captured_at` (UTC), `run` (the full manifest), `kickoff_timezone`,
`spread_convention`, `probability`, `counts`, `games`, `skipped`, and
`payload_sha256`.

One `games` entry per captured game, with exactly these fields:

| Field | Meaning |
|---|---|
| `season`, `week`, `game_id`, `home_team`, `away_team` | from the schedule; `game_id` must equal `{season}_{week:02d}_{away}_{home}` |
| `kickoff_utc` | scheduled kickoff, `YYYY-MM-DDTHH:MM:SSZ`, converted from the schedule's US Eastern time |
| `kickoff_source` | the original `gameday gametime America/New_York` |
| `spread_line` | nflverse convention: points the **home** team is favored by (positive = home favored, negative = away favored, 0 = pick'em); null if no line posted. Taken from the raw schedule, because the predictions CSV writes 0 for a missing line |
| `line_status` | `valid` / `missing` / `pickem` |
| `underdog_team` | away team if the line is positive, home team if negative; null without a valid line |
| `prob_underdog_covers` | model P(underdog covers the line), pushes excluded; a float in [0, 1] for `predicted`, otherwise null |
| `bet_signal` | boolean; can be true only for `predicted` |
| `prediction_status` | `predicted` (valid line and probability, with or without a bet) / `no_line` / `pickem` / `no_probability` (valid line but the pipeline wrote no probability) |

There are no scores, results, profits or other outcome-derived fields.

### Validation and the checksum

`validate_snapshot` runs before a snapshot is written, before an existing file
is accepted on a retry, and on every file before selection. It checks:

- **Format:** the supported schema version and kind, all required fields, and
  exactly the game fields above.
- **Types and values:** integer season and week in range, a well-formed
  `game_id`, and strict UTC timestamps (`...Z`).
- **Times:** `captured_at` is no earlier than `generated_at`, and every
  kickoff is strictly after `captured_at`.
- **Consistency:** status, line status, line, underdog, probability and signal
  agree with each other. There are no duplicate games, `counts` match the
  lists, and skip reasons are known.
- **Run provenance:** the manifest checks described above.

`payload_sha256` is the SHA-256 of the **entire** canonical document except
the checksum field itself, so capture time, kickoff times and provenance are
all covered. It is **unkeyed**: it detects accidental, partial or careless
changes, but anyone who edits a file deliberately can recompute it. It does not
authenticate a file. Tamper-evidence comes from the committed snapshots' Git
history, not from the checksum.

## What gets captured: eligible games

A game is captured only if it has **usable schedule data** at capture time:

1. **Not completed.** Completed games (both final scores present) are never
   captured, only counted (`counts.completed_excluded`). Probabilities
   regenerated after a game is played are never backfilled as "pregame".
2. **Usable kickoff.** `gameday` + `gametime` are listed by nflverse in US
   Eastern time (London games appear as 09:30) and converted with the IANA
   zone `America/New_York`, so daylight saving is handled. Otherwise the game
   is **skipped** and listed in `skipped` with a reason:
   - `missing_kickoff`: no date or time;
   - `invalid_kickoff`: the date or time can't be parsed;
   - `ambiguous_kickoff`: the time falls in the repeated hour when DST ends;
   - `nonexistent_kickoff`: the time falls in the skipped hour when DST
     starts.
3. **Kickoff strictly after the capture time.** Otherwise the game is skipped
   as `capture_not_before_kickoff`. A capture at the exact kickoff instant is
   excluded.

Every eligible game is captured with its status. A valid prediction without a
bet is `predicted` with `bet_signal: false`, which is different from
`no_line` and `pickem`.

### Coverage is not guaranteed

A game only has a snapshot if a capture ran successfully between the
schedule listing it and its kickoff. The nightly runs at 03:00 UTC, which is
before every usual Thursday, Sunday and Monday window, but:

- **Outages:** if the nightly is skipped or the pipeline fails, or if the
  capture fails (the run goes red), no snapshot exists for that night. A game
  that kicks off before the next successful run has none at all.
- **Kickoff changes:** eligibility uses the scheduled kickoff in the schedule
  **at capture time**. If a game is moved later (flexed), later captures carry
  the new time. If a game is moved **earlier** after a capture, that capture
  stays valid against the time it recorded but could postdate the real
  kickoff. Selection can't detect this; check a later capture's `kickoff_utc`
  if it matters.
- **Unusable data:** games with a missing, unparseable, ambiguous or
  nonexistent kickoff time aren't captured until the schedule fixes it.

## Immutability, retries and concurrency

- **New files only.** A snapshot is written to a temp file in the snapshot
  directory, then hard-linked to `<run_id>.json`. The link fails if the name
  exists, so a partial file is never visible and an existing file is never
  replaced. The temp file is always removed.
- **Retry of the same run.** The manifest and its `run_id` are reused. An
  existing file is accepted only if it validates **and** its `run` equals the
  current manifest field for field: hashes, code, configuration, feature set,
  cutoffs, threshold and identity. The original file, with its earlier
  `captured_at` and possibly more games, is kept byte-for-byte, even when the
  retry happens after some or all kickoffs.
- **Conflicts.** The same `run_id` with any different provenance, even when the
  prediction bytes are identical, raises `SnapshotConflictError`. So does an
  existing file that fails validation. Nothing is written.
- **Later run.** A new `run_id` produces a new file. Earlier files are never
  touched.
- **Concurrent writers of the same run:** exactly one link succeeds. The others
  validate the winning file and no-op, or raise if it conflicts.
- **Write failure** (disk full, permissions): the error is raised, and no
  snapshot or temp file is left behind.
- Hard links need a local filesystem that supports them, such as ext4 on
  GitHub's runners or NTFS.

## Selection rules (`select_captures`)

`select_captures(snapshot_dir, which, statuses=None, before=None)` returns one
row per game, with the game fields plus `run_id`, `captured_at`,
`code_revision`, `config_id`, `feature_set_id`, `artifact_sha256` and
`snapshot_file`.

- **Validation:** every file is validated first. An invalid file raises
  `SnapshotValidationError` naming it rather than being silently skipped.
- **Eligible:** captured strictly before the game's `kickoff_utc` (re-checked),
  strictly before `before` if given, and with `prediction_status` in
  `statuses` if given.
- **Order:** `(captured_at, run_id)`. `earliest` takes the first and `latest`
  takes the last, so ties are broken by `run_id`, independent of file order.
- **Read-only:** source files are never modified.

## Limitations

- **Coverage:** see "Coverage is not guaranteed" above. Captures are nightly,
  so a line or probability that changed during the day before kickoff isn't
  recorded.
- **Line source:** the line is nflverse's line in the schedule at run time. It
  isn't a sportsbook price, and it can differ from lines in the betting log or
  spread tracker.
- **Checksum:** the unkeyed `payload_sha256` can't authenticate a file against
  a deliberate re-sealed edit. Use Git history.
- **No history before the first nightly with this change.** The betting log
  (`betting_recommendations_log.csv`) remains a separate, legacy record of
  logged bet signals.
- **Storage:** about 250 bytes per game per run, for all remaining games of the
  season, which is a few MB per season.
