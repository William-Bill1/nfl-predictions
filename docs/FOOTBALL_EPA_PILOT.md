# EPA repository pilot

Research only. No prediction engine, application, scheduled collection or probability changes.
The separate SQLite snapshot accepts a supplied nflverse CSV (or CSV.gz) and a
specified assessment from the audited football repository. Existing files are never overwritten.

## Definitions and limits

Only audited, score-verified REG games from the requested season are included.
Eligible plays have `play_type` pass/run; kneels and spikes are excluded.
Dropbacks include sacks and scrambles (`qb_dropback=1`); remaining rush attempts
are designed rushes. Penalties classified as non-plays are excluded. Two-point
conversion attempts are pass/run plays and are included with the provider's EPA
(71 attempts in the 2026-11-16 snapshot); the snapshot stores no two-point flag,
so they cannot be separated in ad hoc SQL. EPA is the
provider's estimate, not a newly fitted model. Success means EPA > 0.
Defense is opponent offensive EPA allowed (lower is better). Net EPA is offensive
EPA/play minus defensive EPA allowed/play. Aggregate rates are weighted by plays,
not averages of game rates. Missing EPA remains NULL, with counts reported.

Coverage checks require eligible plays for both offenses in every audited game
and no missing EPA. They do not prove that every play was supplied. Never present
an incomplete report as a complete ranking. No opponent adjustment or conversion
to points, betting edge, win probability or a synthetic 100-point score is provided.

Scorecard coverage is evaluated over the selected team's selected games, not
the entire database. An incomplete selection retains sample counts but withholds
all rates and net EPA. An empty date range says `no_data`; an unknown team is
refused. Raw ad hoc SQL exposes underlying observations and does not automatically
apply these scorecard coverage protections.

The before-date cutoff is exclusive. Last four means the team's last four audited
completed games, including any with missing PBP; missing games are not replaced
with older ones. Sources are retrospective and may include corrections made after
the selected cutoff: this is NOT proof of historical pre-bet availability.

## Commands (PowerShell, from repository root)

Use `venv\Scripts\python.exe -B scripts/football_epa.py --help`.
Build requires `--pbp`, `--audited-db`, `--assessment-id`, `--season`, and a new `--output`.
The `metadata` table records both source hashes and the definition version.

Example scorecard:

```powershell
venv\Scripts\python.exe -B scripts/football_epa.py scorecard --db reports/football_repository_pilot/epa_20261116/epa.sqlite3 --teams LAC BAL --before 2026-11-16
```

It returns season-to-date and last-four-game metrics with sample counts.

Ad hoc SQL is a first-class interface. Tables: `games`, `plays`, `metadata`.
View: `team_game_epa` (one row per game, team, side and play kind).
Use the existing guarded read-only connection; SQL cannot write, ATTACH or run PRAGMA.

```powershell
venv\Scripts\python.exe -B scripts/football_epa.py sql --db reports/football_repository_pilot/epa_20261116/epa.sqlite3 --query "SELECT team,SUM(total_epa)/SUM(measured_plays) AS offensive_epa, SUM(measured_plays) AS plays FROM team_game_epa WHERE side='offense' GROUP BY team ORDER BY offensive_epa DESC"
```

For conditions analysis, query/export the existing audited context database and
join results in memory by exact game ID. Do not bypass the SQL guard with ATTACH.
Do not combine different assessments or silently merge revised snapshots.

This first card contains efficiency metrics only. Market prices, injury/weather
display, opponent adjustment and historical meetings are future extensions.

## Readable matchup report

The optional report wrapper formats the same query results as side-by-side
Markdown, including season and last-four-game samples, selected game IDs,
coverage notices and source hashes. It does not change the database or calculation.

```powershell
venv\Scripts\python.exe -B scripts/football_scorecard.py --db reports/football_repository_pilot/epa_20261116/epa.sqlite3 --teams LAC BAL --before 2026-11-16 --output reports/football_repository_pilot/epa_20261116/LAC_BAL.md
```

Output files must be new; existing files are refused. Omit `--output` to print the
report, or use `--format json` for structured results. The database's latest game
date describes the retained snapshot, not proof that it includes every game
completed before a future requested cutoff. Use the source coverage audit when
deciding whether a snapshot is sufficiently current for the question.

The head-to-head table shows measured offensive/defensive EPA, offensive success,
recent net EPA and descriptive edges. Optional `--context-db` adds verified score
point differential/game; its bytes must match the EPA snapshot's audit hash.
Passing/rushing offense and opposing defense are shown separately. No weights,
combined score, point adjustment or betting recommendation is assigned.
Strength of schedule, turnovers, injury impact, game site and a composite matchup
rating are explicitly uncalculated or unavailable. Turnovers need a separately
verified rule for return fumbles and multiple possession changes.

With `--context-db`, records also include overall and last-four W-L-T, home,
away, neutral and unknown-location splits, current streak, latest result and the
last included game date. Source-reported `Home`/`Neutral` comes from the exact
assessment's game observations; it is not independently verified home advantage.
Absent or unfamiliar locations remain unknown. A tie ends a win/loss streak;
bye weeks do not interrupt it. Zero-game splits show their zero sample explicitly.
Optional `--game-id` selects an on/after-cutoff game for the same two teams and
highlights the applicable site record, without adding that game to past results.

## Turnover supplement

`scripts/football_turnovers.py --source <weekly-team-stats.csv> --epa-db <epa.sqlite3>
--output <new-turnovers.sqlite3>` builds a separate research database. It requires
exactly one row for each team in every EPA-snapshot game, matching teams, opponent,
season and REG type. Missing, duplicate or invalid counts refuse the build.
The source is nflverse `stats_team_week_<season>.csv`; giveaways are interceptions
thrown plus `fumbles_lost_total`, including special teams. Takeaways are the
opponent's giveaways, which correctly includes end-zone fumble touchbacks where
no player receives recovery credit. These are provider totals, not an independent
gamebook audit or turnover-adjusted power rating.

Pass `--turnover-db <path>` to the scorecard. Source hashes, definition and the
exact EPA database hash are retained. Season/last-four summaries follow the EPA
selection. Missing selections withhold totals; no supplied supplement displays
Unavailable. Ad hoc read-only SQL can query `team_game_turnovers`, including
giveaways, takeaways and differential by exact game/team. No engine integration.

## Opponent-record schedule context

Supplying the pinned `--context-db` also calculates schedule context without new
downloads or database writes. Each selected opponent's REG record uses all
score-verified finals in that assessment, same season, strictly before the cutoff,
excluding every game against the subject team. Repeated opponents are counted
once per meeting. The pooled fraction is `(wins + 0.5 * ties) / games`, weighted
by available opponent-game observations. Last-four selects the subject's last
four games; opponent records still cover their full season before the cutoff.

No opponent sample means an unavailable percentage, never zero. Samples, combined
W-L-T and status are shown. This describes past schedule difficulty, not a team
edge, league ranking, opponent-adjusted EPA, power rating or betting value.
The retained snapshot can omit games; coverage is relative to that snapshot.
Per-opponent detail and the definition version are in JSON `opponent_records`.
For ad hoc reproduction, retain the read-only SQL alongside the research report.

The scorecard now checks known scheduled games in that assessment before showing
the opponent win fraction. A relevant past unresolved/unverified result, an
unknown/malformed date, or a verified subject game absent from the EPA selection
sets `incomplete` and withholds the percentage. JSON `coverage_gaps` and Markdown
list game IDs and reasons; retained W-L-T counts are partial observations only.
Same-day/future games are excluded, and unrelated games do not block the measure.
Subject-game completeness is checked season-wide even for last-four summaries,
because missing subject results can change that selection. It cannot establish
completeness against games absent from the retained schedule itself, and it does
not change the existing overall-record or EPA coverage rules.

## Weekly batch generation

`scripts/football_weekly_scorecards.py` wraps the existing report without downloads,
ranking or model work. Supply `--epa-db`, `--context-db`, optional `--turnover-db`,
`--season`, `--week`, `--before` and a new `--output` directory. Repeat `--game-id`
to select exact matchups for picks; omit it for every REG game in the selected
week. A selected game before the exclusive cutoff is refused rather than silently
skipped. Repository output must stay under `reports/`.

The tool validates source identity and all reports before creating the directory.
It writes Markdown/JSON cards and README, then INDEX.json last. A failed write may
leave an incomplete folder; absence of INDEX.json means incomplete. Existing
folders are never overwritten or removed, and the operation is not crash-atomic.
Index notices identify incomplete EPA, schedule context and supplied turnover
coverage; `generated_with_coverage_gaps` is not a complete-data or approval claim.
Missing injury/weather context remains disclosed by the reports. Ad hoc SQL over
the same local databases is unchanged. No app, engine or scheduler integration.
