# Injury history pilot

This is a small research database of the injury captures already kept in `data_files/injury_reports/captures/`. It answers read-only questions such as "what changed for KC between Wednesday's and Sunday's captures?".

It's for research only. The app, the probability engine and the nightly pipeline don't read it, and it computes no injury effects.

- **Code:** `scripts/injury_history.py`, which uses only the standard library plus existing modules:
  - `injury_reports.validate_capture` checks each capture file;
  - the football pilot's read-only connection runs `list`, `compare` and `sql`;
  - the Ontario slot rules give the slot list.
- **Tests:** `tests/test_injury_history.py`.
- **Database location:** keep databases under the git-ignored `reports/football_repository_pilot/`. The demonstration database is `reports/football_repository_pilot/injury_history/injury_history.sqlite3`.

## What the source is and isn't

The captures come from `injury_reports.py`, which reads ESPN's public injuries feed; see `docs/INJURY_REPORT_ARCHIVE.md`.

- **Status:** ESPN's general status, which mixes game-designation words (Questionable, Out), roster statuses (Injured Reserve) and "Active". It is **not the official NFL injury report**.
  - Most "Active" rows are **news notes**, such as a stat line from the last game. They are not official availability designations.
  - In the retained captures, 589–616 of each capture's 800 rows are "Active".
- **Practice participation:** the source has no practice-participation field, so it is always null.
- **Entry dates:** the per-entry date is stored as supplied. Its meaning is **unverified**, so it is never used as a status-update time or as evidence of freshness.
- **HTTP Last-Modified:** missing in every retained capture.
- **25 entries per team:** every usable retained capture lists exactly **25 entries for each of the 32 teams** (800 rows).
  - Within each team, the entries are in descending entry-date order, and the oldest date differs from team to team.
  - That is consistent with a cap on the most recent entries, but it is **not a confirmed provider contract**.
  - Entries also disappear for other reasons. In the 11-08 → 11-11 KC comparison, an entry that was 15th-oldest of 25 vanished.

So:
- all 32 teams being present does not mean complete injury lists or complete rosters;
- a player missing from a capture may simply have been pushed out of it;
- nothing here is an official injury report or a betting signal.

## Schema

| Table | One row per | Contents |
|---|---|---|
| `injury_captures` | capture (`capture_id`, under `transformation_version` `injury-history-pilot-1`) | file name, path, SHA-256 and size; the capture's own `payload_sha256`; request and receipt times; when it was imported; slot (name, id, status, whether it fills the slot, intended time, delay); scope season and week; outcome and whether it's usable; source attribution as JSON; HTTP status, Date, Last-Modified and size, plus any error; feed and row counts; malformed and unmatched entries; summary of entry dates; schedule SHA-256 of the raw source file and its stated basis; `schedule_provenance`; the capture's kickoff review as JSON (null for a capture without one); code revision and whether it was dirty |
| `injury_capture_feed_teams` | team block listed by the source | source team name and ID, nflverse abbreviation, entry count |
| `injury_capture_teams` | in-scope team | coverage state (`rows_listed`, `listed_without_rows`, `absent_from_feed`) and row count |
| `injury_capture_games` | game in the capture's scope week | teams, kickoff used by the capture, state at capture time, `kickoff_basis` and `source_kickoff_utc` (see below) |
| `injury_observations` | listed entry (`capture_import_id`, `entry_index`) | every stored field, plus the exact row as `raw_row_json` |

The fields in `injury_observations` are:
- team: source team name and abbreviation;
- player: name, ESPN ID and the ID's source, position;
- status: raw status, its group, and the source's status type (name, description, abbreviation);
- injury: type, location, detail and side;
- comment and return date;
- practice participation (always null);
- entry date, as raw text and as UTC when parseable;
- the scope game and its state.

**Two separate times.** The capture time is `received_at_utc`, when the response arrived. The source's own time is `source_entry_date_raw` / `source_entry_date_utc`. They're never merged, and missing values stay null.

**Kickoff provenance.** Since the 2026 Week 11 schedule change, injury collection corrects two kickoffs in memory (see `docs/INJURY_REPORT_ARCHIVE.md`). Each game row says why its kickoff is what it is:

| `kickoff_basis` | Meaning | `source_kickoff_utc` |
|---|---|---|
| `source` | game not under kickoff review; the source schedule's time | same as `kickoff_utc` |
| `reviewed_correction` | reviewed correction applied during collection | the source schedule's (stale) time |
| `source_already_corrected` | the source schedule already had the reviewed time | same as `kickoff_utc` |
| `capture_without_kickoff_review` | capture written before kickoff review existed | null (not claimed) |
| `imported_before_kickoff_provenance` | row carried over by `upgrade` from a database without these columns | null (not claimed) |

`injury_captures.schedule_provenance` is `reviewed`, `capture_without_kickoff_review` or `imported_before_kickoff_provenance`. `schedule_sha256` always identifies the raw source bytes; it is never a hash of corrected data.

**Imports**
- **Repeats:** a repeat of an identical capture returns the existing row and adds nothing.
- **Conflicts:** the same `capture_id` with different content is refused.
- **Whole-import refusal:** every selected file is checked with `injury_reports.validate_capture` (schema, checksum, consistency) and must be named after its `capture_id`. If any file fails, none of them is imported.
- **Unchanged files:** source files are only read.

**Database files.** These follow the football pilot's rule:
- a missing or empty file is initialized;
- a file with exactly the current schema is opened as is;
- a file with exactly the first schema (before kickoff provenance, "legacy") can still be read by `list`, `compare`, `sql` and the stored-context panel, but `import` refuses it: importing there would drop provenance;
- anything else is refused, unchanged, with exit code 2.

**Upgrading a legacy database.** There is no in-place migration. `upgrade` copies a legacy database to a new file and adds the provenance columns there:
- the original is opened read-only and never changed; the target must not exist;
- every original column of every row is compared after the copy, and both files pass `PRAGMA integrity_check`;
- carried-over capture and game rows get `imported_before_kickoff_provenance` and null provenance values. They are never reinterpreted: re-importing the same capture file afterwards is a no-op, so those rows keep that basis;
- on any failure the new file is removed.

Alternatively, rebuild: import the retained capture files into a new database. That gives every capture its full provenance. The stored-context panel's reviewed-game rule (`docs/INJURY_REPORT_ARCHIVE.md`) means rows without documented provenance are never shown for the two reviewed games, so a rebuild is the way to see captures for them.

**Line endings.** Windows checkouts with `core.autocrlf=true` keep the capture JSON with CRLF line endings, so their file hashes differ from LF copies even though the content and `payload_sha256` are identical. Import one consistent copy, as the demonstration did with the worktree's LF files.

## Commands

Run these from the repository root.

```
# Import explicitly selected capture files (no directories, no globbing inside the tool)
python scripts/injury_history.py import --db reports/football_repository_pilot/injury_history/injury_history.sqlite3 data_files/injury_reports/captures/20261104T170633Z-2f4233f6de94.json data_files/injury_reports/captures/20261108T140610Z-1f36feff3f03.json

# Imported captures, coverage, and Wednesday/Sunday slots in the imported range without a usable capture
python scripts/injury_history.py list --db reports/football_repository_pilot/injury_history/injury_history.sqlite3

# One team between two explicitly selected captures (earlier first)
python scripts/injury_history.py compare --db reports/football_repository_pilot/injury_history/injury_history.sqlite3 --team KC --earlier 20261104T170633Z-2f4233f6de94 --later 20261108T140610Z-1f36feff3f03

# Copy a legacy database to a NEW file with kickoff provenance columns (the original is only read)
python scripts/injury_history.py upgrade --from <legacy.sqlite3> --to <new.sqlite3>

# One read-only query
python scripts/injury_history.py sql --db reports/football_repository_pilot/injury_history/injury_history.sqlite3 "SELECT c.received_at_utc, o.source_status, o.injury_type FROM injury_observations o JOIN injury_captures c USING (capture_import_id) WHERE o.player_id = '3139477' ORDER BY c.received_at_utc"
```

`list`, `compare` and `sql` use the football pilot's read-only connection:
- the file is opened with `mode=ro`;
- `PRAGMA query_only` is on;
- an authorizer allows only reads.

`ATTACH`, writes, `PRAGMA` and schema changes are refused with `not authorized` and exit code 2. `sql` also prints `COVERAGE_NOT_CHECKED`.

## Comparison semantics

`compare` refuses (exit 2):
- the same capture on both sides;
- an "earlier" capture that wasn't captured first;
- a capture that isn't usable (for example the 2026-10-21 `unavailable` capture);
- a team that isn't in either feed.

**Identity**
- **ESPN player ID:** the identity, used exactly as supplied. An ID is never inferred.
- **Entries without an ID:** matched by exact name only when the name is unique for that team in both captures and the other entry also has no ID. Such matches are labelled `inferred_name_match`.
- **Anything else is `identity_unresolved` and isn't compared:** a repeated name, a name that has an ID in the other capture, or an ID listed more than once for the team in either capture. In that last case, every row with that ID is unresolved on **both** sides, so a one-sided duplicate never shows up as newly observed, changed, unchanged or absent.

Every row in the retained captures has an ESPN ID.

Each entry of the team falls into exactly one category:

| Category | Meaning |
|---|---|
| `newly_observed` | in the later capture only |
| `changed` | in both; lists every compared field that differs, earlier and later value |
| `unchanged` | in both; compared fields identical |
| `absent_from_later` | in the earlier capture only. **This never means recovered, healthy, active or cleared.** The source may have dropped the entry, or the player may be listed for another team (`listed_for_other_team_in_later`, by ID). The earlier status is shown as last reported. |
| `identity_unresolved` | could not be matched safely (see above) |

The compared fields are:
- name and position;
- raw status and the source's status type;
- injury type, location, detail and side;
- comment and return date;
- entry date.

The scope game is shown as context and isn't compared, because it changes each week.

**Output**
- **Capture details for each side:** capture time, slot status, outcome, the team's coverage state and entry count, the team's scope game and its state at capture time, the team's entry-date range (labelled unverified), and HTTP Last-Modified.
- **Notes when they apply:**
  - every team has the same entry count (a possible cap);
  - the capture was taken after the team's kickoff;
  - the capture doesn't fill a slot;
  - the two captures cover different weeks.
- **Freshness:** reported as **`not checked`**. No freshness threshold is used.

`list` shows the Wednesday-noon and Sunday-morning slots between the first and last imported capture that have no usable imported capture. It judges only that range.

## Not covered

The pilot doesn't cover:
- a player reference or cross-source identity;
- the official NFL report or practice participation;
- download, scheduling or any app, pipeline or probability integration.

Use a new database file if the capture format changes.
