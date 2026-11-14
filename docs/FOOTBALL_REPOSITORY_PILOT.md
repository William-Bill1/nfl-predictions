# Football repository pilot

A small local SQLite database for ad hoc team-record questions, built from `data_files/nfl_games_historical.csv`. It implements section 6.1 of the spread-engine and data-repository plan. It's a research tool only: the app and the probability engine don't use it.

The code is in `scripts/football_repository.py`, using the Python standard library plus the existing completion rule. Write databases under the git-ignored `reports/` folder.

## How the data is stored

Two things are kept separately: what the source file said, and which games counted as complete on a given date.

| Table | One row per | Notes |
|---|---|---|
| `source_imports` | distinct source file (SHA-256 of its exact bytes, plus the transformation version) | Re-importing identical bytes returns the existing import and writes nothing. A changed file becomes a new import, and earlier imports are kept. |
| `game_observations` | `(import_id, game_id)` | Identity, teams, schedule, raw score text, raw roof and surface text, and the derived roof and surface categories. |
| `completion_assessments` | `(import_id, evaluation_toronto_date, completion_rule_version)` | Repeating the same assessment returns the existing one. Earlier assessments are never changed, so their results can be reproduced. |
| `game_completions` | `(assessment_id, game_id)` | `final` or `excluded`, the completion rule's reason, and the final scores. |
| `team_game_results` (view) | team per game that is final in an assessment | Two rows per final game, one from each team's perspective. |

Completion uses `bet_journal.completion_evidence`, the conservative Toronto-day rule that the pipeline also uses. It reads only the schedule rows it's given, never journal files.

Its rules aren't copied into the pilot. Because they depend on the date, every assessment is made for an explicit Toronto calendar date. A game dated on or after that date is never treated as final.

### Rule version

`COMPLETION_RULE_VERSION` in `scripts/football_repository.py` names the completion rule an assessment used. It is currently `bet_journal.completion_evidence@2026-11-13`. It's assigned by hand.

**Change it whenever `completion_evidence`'s meaning changes.** That includes what counts as final, the same-day rule and the score checks. Assessments made under the old rule then remain distinct and reproducible, and the new rule gets its own assessments. Nothing detects such a change automatically.

## Commands

Run these from the repository root. `--date` is always a Toronto date in `YYYY-MM-DD` format, and it's never filled in automatically.

```
# 1. Import the source file. This prints the import_id and assesses nothing.
python scripts/football_repository.py import --db reports/football_repository_pilot/pilot.sqlite3

# 2. Assess completion for that import as of a Toronto date. This prints the assessment_id.
python scripts/football_repository.py assess --db reports/football_repository_pilot/pilot.sqlite3 --import-id 1 --date 2026-11-13

# 3. Get a team record for exactly that assessment.
#    This fails if no matching assessment exists; it never falls back to another date.
python scripts/football_repository.py record --db reports/football_repository_pilot/pilot.sqlite3 --import-id 1 --date 2026-11-13 --team KC --season 2025
```

`record` defaults to the regular season, an `outdoors` roof and natural grass.
- **Open roofs:** add `--include-open` to count open retractable roofs as well.
- **Postseason:** use `--game-type WC|DIV|CON|SB`.
- **Older assessments:** use `--rule-version` to select one made under an older rule.

The output records the assessment ID, import ID, evaluation date and rule version, and shows:
- W, L, T, the number of games, and W:L;
- win % excluding ties, `W / (W + L)`, which is null when there are no decisions;
- the NFL standings percentage, labelled separately as `(W + 0.5*T) / games`;
- the game IDs in each category below.

### Database files and schema checks

`import` and `assess` check an existing database file before running any schema statement. The check uses a read-only connection, and compares the file's schema exactly against the schema in `scripts/football_repository.py`.

| File | Result |
|---|---|
| Missing or empty | Initialized with the current schema |
| Exactly the current schema | Opened as is, with no schema statements run |
| Anything else (the first pilot's schema, a partial schema, changed columns, or extra objects) | Refused with `unsupported database schema …` and exit code 2. The file is left byte-identical. |

There are no migrations. Use a new database file instead. The first pilot's `reports/football_repository_pilot/pilot.sqlite3` uses the old schema and is refused.

`record` only reads. It never initializes or changes a database, and it refuses a file that doesn't have the current schema.

### Ordinary SQL

The `sql` command runs one read-only query. It uses SQLite's own controls:
- the file is opened with `mode=ro`;
- `PRAGMA query_only` is turned on;
- an authorizer allows only reads (`SELECT`, reading tables and views, functions and recursive CTEs).

Anything else is denied when the statement is prepared, before it runs, and the command reports `not authorized` with exit code 2. That includes `ATTACH` and `DETACH`, writes, schema changes, `PRAGMA`, transactions, temporary tables and `load_extension`. So `ATTACH` can't create a file, and the database is unchanged.

Joins, aggregates and CTEs work as usual. The schema isn't checked here, so an older pilot database can still be read safely.

Always filter on one `assessment_id`. That also fixes the import, so results are never combined across assessments.

```
python scripts/football_repository.py sql --db reports/football_repository_pilot/pilot.sqlite3 "SELECT outcome, COUNT(*) AS games FROM team_game_results WHERE assessment_id = 2 AND team = 'KC' AND season = 2025 AND game_type = 'REG' AND roof_category = 'outdoors' AND surface_category = 'natural_grass' GROUP BY outcome"
```

The view's columns are:

| Column | Values |
|---|---|
| `assessment_id`, `evaluation_toronto_date`, `completion_rule_version`, `import_id` | Which assessment the row belongs to |
| `team`, `opponent` | Team codes |
| `is_home` | The source's designated home team. Check `location` for `Neutral` sites |
| `season` | The NFL season, not the calendar year |
| `game_type`, `week`, `gameday` | Schedule fields |
| `roof_raw`, `surface_raw` | The exact source text |
| `roof_category` | `outdoors`, `open`, `closed`, `dome` or `unknown` |
| `surface_category` | `natural_grass`, `artificial` or `unknown` |
| `team_score`, `opponent_score` | Final scores |
| `outcome` | `W`, `L` or `T` |

## Exclusion categories

For the selected team, season and game type within one assessment, every game falls into exactly one category. The four counts add up to `games_in_scope`.

1. **Included:** final, and both the roof and the surface are known and match the filter.
2. **`excluded_unfinished_or_unverifiable`:** not final in this assessment. That covers scheduled games, same-day games, 0–0 placeholders, inconsistent scores and invalid dates, each with the rule's reason. These games are never counted as losses.
3. **`excluded_by_known_conditions`:** final, but at least one known value rules the game out. For example, a dome, a closed roof, an open roof unless `--include-open` is used, or artificial turf. A dome game with a blank surface belongs here: it is definitely not outdoors.
4. **`eligibility_undetermined_unknown_condition`:** final, and no known value rules it out, but a required condition is unknown. For example, `outdoors` with a blank surface, or a blank roof on grass. It's not counted as a match or a non-match.

`data_quality_missing_conditions` lists games in scope with missing or unrecognized values. The three lists don't overlap: roof unknown only, surface unknown only, or both unknown. This is data-quality information only, and a game can appear in it and in one of the categories above.

## Limits

- **Surface:** only trimmed, case-insensitive `grass` counts as natural grass. Surfaces are classed as artificial only from an explicit list, and anything else, or a blank, is `unknown`.
- **No inference:** roof and surface are never inferred from a team's usual stadium.
- **What the conditions mean:** roof and surface are retrospective source records. They are not confirmed roof state or confirmed pregame knowledge.
- **What completion means:** an assessment states what the source file showed, judged as of the evaluation date. It is not an authoritative final status. It doesn't show when upstream published a score.
- **Neutral sites:** "home" is the source's designation.


## Coverage-first analysis (uncommitted follow-up)

A database containing every scheduled row may still have missing or incorrect fields. Raw SQL results do not establish a complete record. The `sql` command now prints `COVERAGE_NOT_CHECKED`; the original `record` command labels its output as an unverified `source_subset` (or incomplete where conditions are missing), with `complete_record: null`. Its underlying source-based calculation remains available for inspection.

For the separately retained 2026 audit dataset, use `scripts/football_query.py`. This command reads the audit metadata and game-level evidence view; it never imports, creates or modifies a database. It requires an explicit assessment ID. Example:

```powershell
python scripts/football_query.py --db reports/football_repository_pilot/captures/20261113T203523Z/pilot_2026_roofresolved.sqlite3 --assessment-id 1 --team KC --season 2026
```

By default it checks verified scores, roof and surface before presenting an outdoor-natural-grass record. `--all-conditions` requests a scores-only record and does not require irrelevant condition fields. Regular season is the default; `--game-type` selects another game type.

- `complete_for_snapshot`: all required fields for the query are resolved within the captured schedule; exit 0. This is not an independent guarantee that upstream listed every game.
- `incomplete`: at least one relevant game has missing, conflicting or unverified required evidence. The complete record is null; any known subset is labelled separately; exit 3.
- `no_data`: no games in the selected scope; no complete record; exit 3.
- `not_checked`: absent/incompatible audit data or another read error; no fallback to raw results; exit 2.

Future and same-day games remain pending. Past or undated games without final evidence block a complete-so-far claim. A known nonmatching condition can exclude a game even when another condition is unresolved (for example artificial turf excludes a game from a natural-grass query despite an unresolved roof). Unverified scores are never graded as losses.

The audit retains original source values and PDF evidence. It prefers game-specific official gamebooks for surface classification, including five disagreements with populated nflverse values. The later roof-resolved snapshot records the reviewed roof classifications described below; unresolved future disagreements must remain flagged rather than guessed. Melbourne's U.S./Australian date difference is reconciled explicitly by time zone, not by permitting arbitrary date mismatches.

This command supports the named audit dataset's fixed evidence view, not arbitrary third-party databases or automatic SQL meaning validation. Weather, injury, officiating and forecast queries are not covered by these core checks. The audit files stay outside Git. There is no app or probability-engine integration.


### Reviewed roof classifications and owner-supplied facts

The newer retained `pilot_2026_roofresolved.sqlite3` resolves the three international-game roof classifications using game-specific gamebooks plus primary venue/structural evidence. A roof over spectator stands does not make the uncovered playing field indoors. `open` refers to an open retractable playing-field roof; an open-air fixed venue uses `outdoors`.

SoFi retains the source's fixed-roof `dome` code with its open-sided enclosure documented separately in the resolution evidence. The gamebook phrase "Open Stadium" is not used alone to infer an uncovered field or a retractable roof. The fixed-roof code does not imply full enclosure or immunity to wind/weather. All original strings and prior issues remain retained.

Manual information is an accepted fallback when electronic evidence is insufficient. The owner can send the specific game ID, field, observed value, units, observation date/time and a source/reference in chat; no direct SQLite editing is needed. Store the submission separately as user-supplied, preserve the original provider value, validate identity/units/evidence, and only then clear any affected coverage blocker. An unverified manual assertion does not make the record complete. This documents a curation procedure, not an automatic manual-entry importer.


## Repeatable research refresh

`scripts/football_refresh.py` captures the current season into a new timestamped
folder below `reports/football_repository_pilot/refreshes/`. It preserves the
exact downloaded bytes, a season-only TSV, a normalized copy of the comparison
source, a new SQLite database, and `report.json` written last. An attempt without
`report.json` did not finish; do not use it as a successful snapshot. Failed
attempts are left in place for inspection. Existing snapshots are never replaced.

```powershell
python scripts/football_refresh.py --download --previous-source reports/football_repository_pilot/captures/20261113T203523Z/games_2026.tsv --season 2026 --date 2026-11-14
```

Alternatively use `--source path/to/saved/games.csv` instead of `--download`.
Input can be comma- or tab-separated. Select the season and Toronto evaluation
date explicitly. For the next comparison, pass the previous successful refresh's
`games.tsv` as `--previous-source`; do not automatically select the newest folder.
The download fetches only the fixed public nflverse schedule, with TLS defaults,
a 45-second timeout and an 8 MiB size limit. No live odds or injury API is called.

The report lists added/removed games, changed raw fields (including formatting
changes), newly completed games, games no longer assessed as complete, and
completed games missing recognized roof/surface values. Both source versions
are assessed on the **same** specified date, so the completion delta measures
source differences, not a comparison against an earlier assessment date.
Raw hashes identify the inputs; the database records the normalized TSV hashes.
Unchanged input still creates a separate refresh attempt for the requested date.

A successful refresh exits 0 but is **captured_not_audited**, not verified complete.
Errors exit 2. This command does not copy gamebook corrections or manual facts,
create the audited query view, promote a new approved snapshot, update the app,
or schedule refreshes. Missing or changed facts need review before audited
queries can use them. Referee, injury and weather coverage is not established.
The existing recurring automation checks freshness; it does not run this writer.
