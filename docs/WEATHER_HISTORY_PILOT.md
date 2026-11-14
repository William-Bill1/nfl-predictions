# Weather history pilot

A small research database of hourly weather **forecasts** that were captured by hand for explicitly selected games. It answers questions such as:
- what forecast had been captured for this game by a given time?
- what changed between two captures?

It's for research only. The app, the probability engine, `player_props/weather.py` and the nightly pipeline don't read it. Nothing is scheduled, and no weather effect is computed. The plan is in `docs/WEATHER_HISTORY_PILOT_PLAN.md`.

- **Code:** `scripts/weather_history.py`. It uses no new dependencies and reuses existing helpers:
  - `pregame_snapshots.kickoff_utc` converts kickoff times; it refuses ambiguous and nonexistent DST wall-clock times.
  - `injury_reports._read_bounded` reads responses with a size limit and a deadline.
  - `pregame_snapshots.payload_checksum` seals each capture.
  - The football pilot's guarded read-only connection runs the read commands.
- **Tests:** `tests/test_weather_history.py`, using synthetic responses and temporary schedules only.
- **Data location:** keep captures and databases under the git-ignored `reports/football_repository_pilot/weather_history/`.

## What is stored and what isn't

- **Forecasts only.** Values come from the [Open-Meteo](https://open-meteo.com/) forecast API for the model grid cell nearest the requested point. They are not measured stadium conditions, and the pilot stores no observed weather.
- **Attribution:** "Weather data by Open-Meteo.com", licensed [CC BY 4.0](https://open-meteo.com/en/license). The free tier is for non-commercial use and is rate limited; see the [terms](https://open-meteo.com/en/terms). Recheck the terms before any broader use.
- **Variables:** these are requested in °F, mph and mm, with times in UTC (`timezone=GMT`):

  | Variable | What a value covers |
  |---|---|
  | `temperature_2m` | an instant (the valid time) |
  | `wind_speed_10m` | an instant (the valid time) |
  | `wind_gusts_10m` | the maximum over the **preceding hour** |
  | `precipitation` | the total over the **preceding hour** |
  | `precipitation_probability` | the chance of more than 0.1 mm in the **preceding hour** |

- **Request window:** the three hours before kickoff's hour through four hours after it.
- **Three separate times,** never substituted for each other:
  - `received_at_utc` is when this system received the response (the capture time), stored to the microsecond. `request_started_at_utc` is also kept.
  - `valid_time_utc` is the forecast hour a value applies to. The provider's raw string is kept in `valid_time_raw`.
  - `provider_issue_time_utc` is always **null**, because the forecast response supplies no model run or issue time. `generationtime_ms` is server processing time; it's stored as `provider_generationtime_ms` and is not an issue time. Ages are shown as **time since capture**.
- **Kept exactly:**
  - the raw response bytes (a `.response` file, plus a blob in the database);
  - SHA-256 hashes of the raw response, the metadata file, the schedule and the venue file;
  - the request parameters and URL;
  - the returned units, and the returned grid coordinates and elevation, separate from the requested coordinates.

## Venues and games

`capture` needs `--game-id` (an exact schedule `game_id`) and `--venues`, a reviewed venue file. Each venue record contains:
- the exact schedule `stadium_id` and stadium name;
- latitude and longitude;
- `field_exposure`: `outdoor`, `fixed_roof`, `retractable_roof` or `open_sided_fixed_roof`;
- the roof context and coordinate basis;
- at least one https reference with the date it was accessed;
- the review date.

The game's `stadium_id` and stadium name must both match the record exactly. There is no join by home team, no fuzzy matching and no default location. The helper's coordinates aren't used.

These conditions are refused **before any request**. No file is written and the exit code is 3:

| Outcome | When |
|---|---|
| `game_not_found` | the `game_id` isn't in the schedule |
| `game_ambiguous` | the `game_id` appears on more than one schedule row |
| `game_completed` | the schedule already has the final score |
| `kickoff_unusable` | the kickoff time is missing, ambiguous or nonexistent |
| `game_started` | kickoff has already passed |
| `venue_unresolved` | there is no reviewed record, or the name differs |
| `outside_pilot_coverage` | the venue isn't outdoor; a retractable roof's state is never assumed, and an outdoor forecast isn't an indoor estimate |
| `venue_conflict` | the record says outdoor but the schedule's roof field doesn't |

Once a request is made, the result is always written:

| Outcome | Usable | Meaning |
|---|---|---|
| `ok` | yes | every requested variable was returned |
| `ok_missing_variables` | yes | some requested variables weren't returned; they are listed and stored as null |
| `http_error` | no | the provider returned an HTTP error |
| `transport_error` | no | the connection failed |
| `timeout` | no | the response didn't finish within the deadline |
| `response_too_large` | no | the response was over the size limit |
| `malformed_response` | no | truncated or invalid JSON, arrays that don't line up with the times, non-numeric values, non-UTC times, or repeated or out-of-order hours |
| `units_unexpected` | no | the returned units don't match the requested ones; nothing is converted |

Failed captures have **no forecast hours**. Missing values stay null, and neutral weather is never substituted. Hours inside the requested window that weren't returned are listed as `missing_requested_hours`.

The capture exit code is 0 when usable and 4 when the capture was written but isn't usable.

**One identity per attempt.** Each attempt's `capture_id` is the receipt time to the microsecond, the game ID and 48 random bits, for example `20261114T162450909882Z-2026_10_MIN_GB-7381cb45b30f`. It never comes from the response.
- Identical responses, or two failures without a body, received in the same second are still separate attempts.
- Files are created exclusively. The metadata file reserves the identity; if a name already exists, a new identity is drawn, so an earlier attempt's files are never reopened for writing.
- Captures from before 2026-11-14 16:20 UTC use an older second-plus-body-hash ID. They remain importable.

**TLS.** The `capture` command checks certificates against the Windows trust store through `truststore`, for this process only, as `player_props/rehearsal_real.py` does. This handles a local security product that re-signs HTTPS with a root only Windows trusts. Certificate verification stays on, and no system setting is changed. Each capture records `request.tls_trust`.

## Database

There are two tables:
- **`weather_captures`:** one row per capture, with identity, the three times, game, schedule, venue, the request and response, outcome, units, source attribution and code revision.
- **`weather_forecast_hours`:** one row per capture and valid hour, holding the five raw values, which may be null.

**Imports**
- **One transaction for all files:** every selected `<capture_id>.json` is checked (checksum, file name, raw-file hash and size). Its raw response is parsed again and must reproduce the recorded outcome. If any file is refused, or an insert fails, nothing is written.
- **Repeats:** re-importing the same capture adds nothing.
- **Conflicts:** the same `capture_id` with different content is refused.
- **Separate captures:** two captures taken at different times are separate observations even when their weather values are identical.

**Database files** follow the football pilot's rule:
- a missing or empty file is initialized;
- a file with exactly this schema is opened as is;
- anything else is refused unchanged, with exit code 2. There are no migrations.

## Commands

Run these from the repository root.

```
# One manual capture (one request) for one game
python scripts/weather_history.py capture --game-id 2026_10_MIN_GB --venues reports/football_repository_pilot/evidence/weather_history/demo/reviewed_venues.json

# Import explicitly selected capture metadata files
python scripts/weather_history.py import --db reports/football_repository_pilot/weather_history/weather_history.sqlite3 reports/football_repository_pilot/weather_history/captures/<capture_id>.json

# Captures (optionally only those received by a time)
python scripts/weather_history.py list --db <db> [--game-id G] [--as-of 2026-11-15T17:00:00Z]

# What had been captured by a decision time
python scripts/weather_history.py asof --db <db> --game-id G --cutoff 2026-11-15T17:00:00Z

# Two captures of one game, earlier first
python scripts/weather_history.py compare --db <db> --earlier <capture_id> --later <capture_id> [--as-of T]

# One read-only query
python scripts/weather_history.py sql --db <db> "SELECT valid_time_utc, temperature_2m FROM weather_forecast_hours"
```

- **`asof`** returns the latest usable capture received at or before the cutoff. Captures received later are excluded and counted: a later download isn't evidence of what was known earlier.
- **`compare`** refuses (exit 2) when:
  - the same capture is on both sides, or the captures are for different games;
  - the earlier capture wasn't received first;
  - either capture is unusable or was received after `--as-of`;
  - the captures requested different coordinates.

  Values are compared only at identical valid hours, and only when both units are identical. Otherwise it reports `units_mismatch` or `variable_missing`, hours present in only one capture, and missing values (which are not counted as changes). Each hour shows `minutes_from_kickoff` as context. No kickoff value is interpolated and no daily average is computed.
- **`sql`** uses the read-only connection (`mode=ro`, `query_only` and a read-only authorizer). Writes, `ATTACH`, `PRAGMA` and schema changes are refused with exit code 2. It prints `FORECASTS_NOT_OBSERVATIONS` to stderr.

## Not covered

The pilot doesn't cover:
- scheduling or bulk or historical requests;
- app display or any probability or prediction use;
- roofed and retractable venues;
- observed weather or gamebooks;
- interpolation to kickoff;
- forecast issue times.
