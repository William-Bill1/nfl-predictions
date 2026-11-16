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
