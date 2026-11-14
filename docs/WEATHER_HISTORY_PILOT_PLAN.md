# Weather-history pilot: bounded implementation plan

Prepared 2026-11-14, from main at `70271666330e29db9ec2b6844cab89623b282d7a`.
This is a plan, not an implemented capture service or an approval to change predictions.

## Inventory and first use

The `data_files` inventory found no files named for retained weather forecasts.
The schedule has retrospective `temp` and `wind` fields, with incomplete coverage;
these do not establish forecasts captured before kickoff. Gamebooks are later
observations and must remain distinct from forecasts.

`player_props/weather.py` already calls Open-Meteo but is unsuitable as an archive:
it returns assumed neutral values after failures and for distant dates, substitutes
a generic location for unknown venues, and aggregates a portion of the day rather
than preserving hourly forecasts. Its venue/roof mapping also needs independent
validation (for example Highmark currently shares Pittsburgh's coordinates).
Leave this helper and all adjustment code unchanged.

First use: preserve hourly forecasts captured for explicitly selected games,
then ask what was captured by a decision time and what changed between captures.
Start with one or two outdoor games, not every stadium or a historical backfill.

## Source proposal

Use the public Open-Meteo forecast JSON endpoint:
`https://api.open-meteo.com/v1/forecast`.

Official references checked on 2026-11-14:
- Parameters and units: https://open-meteo.com/en/docs
- Terms: https://open-meteo.com/en/terms
- Attribution: https://open-meteo.com/en/license

The free service is for non-commercial use and is rate limited. The current
terms specify fewer than 10,000 daily requests, 5,000 hourly and 600 per minute.
The pilot needs only one request per explicitly selected game per manual capture.
No paid account, API key, commercial-use assurance or new dependency is proposed.
Include provider attribution and the licence link with outputs. Recheck terms
before adding scheduling or broader use; provider availability is not guaranteed.

Request hourly temperature, sustained wind, gusts, precipitation amount and
precipitation probability. Set UTC time, Fahrenheit, mph and millimetres explicitly.
Retain returned units and the complete raw response. Provider hourly values are
forecasts, not measured stadium conditions. Gust and precipitation fields can
represent a preceding interval; preserve their documented meaning and actual
valid timestamps rather than labelling them instantaneous kickoff measurements.

## Venue and game identity

Use the selected schedule row's exact game ID, teams, venue and kickoff, with the
schedule hash. Reuse the established kickoff time-zone conversion.

Before a real request, the selected game needs a reviewed venue coordinate record:
exact venue name, latitude/longitude, authoritative reference, review date and
roof context. Keep requested coordinates separate from returned model-grid
coordinates. Do not join by home team alone, perform fuzzy venue matching, use
a default location or reuse the current helper's coordinates without review.
Unknown coordinates mean `venue_unresolved`, with no API request.

Roof context is separate from outdoor weather. A roof over spectator stands is
not a roof over the field. A retractable roof is not assumed open or closed.
Open-sided fixed-roof venues need their own context. Outdoor forecasts are not
estimates of indoor temperature. Start with reviewed outdoor games; leave other
cases explicitly outside this pilot's coverage.

## Minimal storage and commands

Use one separate research SQLite file under the existing ignored
`reports/football_repository_pilot/` tree. Reuse the guarded read-only SQL helper.
Two tables are sufficient initially:
- `weather_captures`: capture identity, game/schedule/venue evidence, request and
  response times, exact request parameters, raw response path and hash, HTTP
  outcome, source attribution, requested/returned coordinates and returned units.
- `weather_forecast_hours`: capture ID, valid UTC timestamp and nullable raw
  hourly values. Do not store invented neutral conditions.

A separate provider issue time/model run is null unless the response explicitly
supplies it. `generationtime_ms` is processing duration, not an issue timestamp.
Capture time, forecast valid time and any issue time must never be substituted
for each other. Show age since capture as such, not forecast age since issuance.

A repeat import of the same capture is a no-op. A fresh capture at another time
is a new observation even when its weather payload is unchanged. Preserve both
Wednesday and Sunday captures; do not deduplicate by weather values alone.

Initial commands: manual capture for one explicit game, import an existing
capture, list, compare two explicit captures for the same game and read-only SQL.
Comparison uses matching valid timestamps and units, not daily averages. Show
missing hours and units mismatches. As-of queries require receipt by the cutoff;
a later download is not evidence that this system knew it earlier. Preserve the
selected hourly timestamps; do not interpolate a kickoff value in this pilot.

Capture failures must be visible and contribute no usable forecast rows. Retain
bounded response/error evidence where available. Missing values remain null.
An API failure, an unknown venue, a missing field and an indoor limitation are
different outcomes. Keep the tool small; no generic ingestion framework.

## Focused acceptance and stop point

Use synthetic responses and temporary schedules. Cover UTC conversion including
DST, exact game/venue selection, missing coordinates, correct units and aligned
hourly arrays, missing values, failed/truncated responses, repeated imports,
unchanged payloads at different capture times, comparison at matching valid
hours, as-of cutoffs, batch rollback and read-only SQL refusing writes/ATTACH.
Retain full focused logs, JUnit and actual exit codes in the existing pilot
evidence area. No local full suite, mutation campaign or browser tests.

After offline tests pass, at most two public forecast requests may demonstrate
the pilot, only for independently reviewed outdoor venue mappings. Save the raw
responses and all selection evidence under the existing report tree. If a mapping
cannot be verified, report that limitation instead of guessing or broadening scope.

Stop with uncommitted code, tests and short usage documentation for independent
review. Do not schedule captures, display them in the app, integrate probabilities,
train, regenerate predictions, restart 8501, change dependencies/workflows or
access the Bet Journal, credentials or real Model Performance. Keep all current
production files, approved databases and user edits unchanged.
