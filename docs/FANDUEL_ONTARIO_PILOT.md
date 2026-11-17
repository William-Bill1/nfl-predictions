# FanDuel Ontario shortlist pilot

Manual research observations only. Reuse `ontario_spreads.py`'s existing manual
recorder. The automated `fanduel` feed is US reference data and is never substituted.
No new API, scraping, scheduler, engine or live-app integration is included.

## First capture

Owner supplies game/team, handicap, American odds and the actual observation
date/time (Toronto with offset, or UTC). Opponent odds and a screenshot/reference
are optional. Never use model lines or invented observations as sportsbook quotes.
Record a reference in the existing note field; do not store login/session details.
Missing opponent odds stay null; the mirrored opponent handicap is derived by the
existing schema and is not a separate observation of that side's offer.

Example command shape (placeholders, not an actual quote):

```text
venv\Scripts\python.exe -B ontario_spreads.py manual-quote --game GAME_ID --team TEAM --spread SPREAD --price ODDS --observed-at ISO_WITH_OFFSET --note REFERENCE
```

That existing command writes the actual manual artifact under data_files. Run it
only for explicitly supplied real observations. Use temporary directories for tests.
Plan checkpoints Tuesday, Wednesday, Friday, Sunday morning and before each
shortlisted game's kickoff; adapt for Thursday/international/Monday kickoffs.
No automation was installed. Absence of a quote is not proof a checkpoint was missed.

## Timeline and ad hoc queries

`scripts/fanduel_ontario_timeline.py --manual-dir PATH --schedule TSV --as-of
ISO_WITH_OFFSET [--output NEW_REPORT_FOLDER]` validates artifacts with the existing
validator and checks exact game identity/current kickoff. Only quotes observed
AND entered at or before the as-of cutoff are eligible. Late transcription never becomes
historical knowledge at the earlier observation time. Files are read without edits.

The report groups by exact game/team, retains all observations and compares first
with last observed pregame quote. Same-time conflicting terms prevent comparison;
the tool never arbitrarily selects one. First/last observed are not verified
opening/closing prices. Positive spread change means more points for that team.
Margins 3/7 and -3/-7 are flagged when the endpoint interval reaches or crosses them.
Price-only change is reported as one-side implied-probability percentage points
only at unchanged handicap. This includes bookmaker margin and is not a fair
probability estimate, ROI or betting edge. No movement is attributed to injury news.

Optional output must be a new folder below repository reports/. It contains
timeline.json, quotes.sqlite3 and COMPLETE.json written last. Failed writes can
leave partial output; no crash-atomicity or durable-storage claim. Existing output
is refused. Original validated artifacts remain the source of truth.

Use the existing guarded read-only SQL tool on the new `quotes` table, for example:

```sql
SELECT game_id,team,observed_at,handicap,price,opponent_price
FROM quotes ORDER BY game_id,team,observed_at,quote_id;
```

Real observations are recorded only when the owner supplies prices and their
time; they stay local and are not committed with this tool. Historical chat
prices without a confirmed date/time are not imported.

## Schedule time changes

Quotes keep the original schedule hash and kickoff and are never rewritten.
Default timeline behaviour still refuses a changed kickoff. To intentionally
reassess historical quotes, supply a reviewed corrected schedule and pass
`--reassess-kickoffs`. Game ID, teams, season, week, artifact integrity and both
as-of cutoffs remain enforced. Observations at or after the evaluated kickoff
are excluded, including the exact kickoff instant. Other games still report.

In this mode `timeline.json` records `kickoff_evaluation`, with each changed
quote's original and evaluated kickoff and pregame decision. Included observations
and SQLite `raw_json` remain the original documents. Excluded quotes remain in the
source directory and are listed with their reason in the report. Querying SQLite
alone omits these exclusions and reassessment metadata; consult timeline.json too.
This is retrospective schedule reassessment, not proof the corrected time was
known when a quote was entered.

New quotes must use the same reviewed corrected schedule via the recorder's
`--schedule` argument. Reassessment does not change the recorder's default schedule,
fetch schedule updates or protect a command given a stale schedule. Do not use the
production schedule for near-kickoff recording until its times have been checked.

For Week 11 2026, the NFL's November 9 announcement moves PIT-PHI to November 22
13:00 ET and JAX-NYG to November 22 16:25 ET:
https://media.nfl.com/content/dam/communications/documents/2026/Week%2011%20Schedule%20Change.pdf
A research-only corrected snapshot and provenance can be kept under reports/;
production data and immutable quotes need not be edited.
