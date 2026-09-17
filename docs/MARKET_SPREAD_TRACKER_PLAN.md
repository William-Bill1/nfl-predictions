# Season-Long Sportsbook Spread Tracker — Design

**Status:** **All 3 phases built, plus a post-launch value-finder addition.**
`spread_tracker.py` fetches/normalizes/upserts real US + Canadian sportsbook
game-spread lines into `data_files/spread_tracker_log.csv`, joined against
nflverse's `spread_line`; `scripts/spread_tracker_report.py` rolls that log
up into `data_files/spread_tracker_report.json` (per-book season-to-date
ranking, best-line-per-game callouts, anomaly flags); `pages/5_Spread_Tracker.py`
surfaces that report in the Streamlit app; `scripts/spread_value_finder.py`
(added 2026-09-17, see Phase 4 below) computes price-adjusted fair-value
edges for one book's lines, since point-divergence alone isn't the same
question as "is this side priced favorably." Live-verified 2026-09-16 against
real Week 3 2026 data: 144 game/book rows (16 games × up to 10 books across
`us`/`ca`), all `deviation_pts` values landing in a sane ±1 point range,
cache-hit and upsert-idempotency both confirmed against the real API; the
Phase 2 report ranked DraftKings closest to nflverse's line (mean|dev|=0.00pt)
with zero anomalies flagged for that single week. Caught and fixed a real bug
along the way: the CLI's predictions-CSV read used a plain `pd.read_csv()`,
but `nfl_games_historical_with_predictions.csv` is **tab-separated** despite
the `.csv` extension (see `betting_log.py`'s own `PREDICTIONS_PATH` read) —
this silently produced 100% `NaN` `nflverse_spread_line`/`deviation_pts`
values with no error; fixed with `sep='\t'`.

**Provider:** [The Odds API](https://the-odds-api.com/), same account/key
already live for `player_props/market_odds.py`. This module uses the
**bulk** `/v4/sports/{sport}/odds` endpoint instead of the per-event one —
one call returns every game and every bookmaker in the response's `regions`
at once, at a flat `markets × regions` credit cost regardless of slate size.

**Goal:** answer "which sportsbook(s), US or Canadian, consistently offer a
different (better) number than nflverse's `spread_line` — the line the spread
model is trained and evaluated against?" from real accumulated season data,
rather than one-off manual pulls. This was prompted directly by three live,
ad-hoc comparisons made in chat (DK vs. FanDuel a full point apart on one
game; a Canadian book, PlayNow, diverging 3+ points from the field on two of
four games) that were worth tracking over time instead of re-derived by hand
each week.

## Non-goals

- **Totals/moneylines.** Spreads only, for now — same reasoning as the
  player-prop module's scoping, extend later if it proves valuable.
- **Open-vs-close intraweek line movement.** Phase 1 captures **one snapshot
  per week** (upserted, so a same-week rerun replaces rather than
  duplicates). Tracking a line's movement from open to close would need a
  `line_type` column added to the key — a real schema change, not built
  speculatively before there's a concrete need for it.
- **Feeding `deviation_pts` back into the prediction models.** This is a
  research/reporting feature answering a human question about book quality,
  not a new model input.
- **This supersedes `ODDS_API_INTEGRATION_PLAN.md`'s "spread/moneyline/totals
  stay on the nflverse consensus line" non-goal** — that exclusion was scoped
  to the player-prop-edge use case (the spread model has no proven
  out-of-sample edge, so precise game odds don't help prop predictions). This
  is a different use case entirely: comparing *sportsbooks against each
  other and against nflverse*, independent of whether the spread model has
  edge.

## Module: `spread_tracker.py`

Root-level, peer to `betting_log.py` — it owns a new accumulating CSV
artifact the same way `betting_log.py` owns
`betting_recommendations_log.csv`, rather than living under `scripts/`
(which holds tools that *consume* an artifact, not own one) or
`player_props/` (a distinct, player-level concern).

Reuses from `player_props/market_odds.py` directly (generic two-way-market
math and team-name maps, no player-prop-specific assumptions):
`_devig_over_prob`, `TEAM_FULL_NAME`, `_ABBR_BY_FULL_NAME`, `_read_remaining`,
`ODDS_API_BASE`, `ODDS_API_SPORT`, and the dotenv-load-at-import pattern.

```python
ODDS_API_REGIONS = "us,ca"        # 1 market x 2 regions = 2 credits/call
ODDS_API_MARKET = "spreads"

def _region_for_book(book_key: str) -> str:
    """'ca' if '_ca' in book_key else 'us' - every Canadian book key observed
    (betmgm_ca_on, playnow_ca, proline_ca_on, sportsinteraction_ca_on,
    betrivers_ca_on, betano_ca_on) contains '_ca'; no hardcoded book list."""

def _normalize_spread(home_point: float) -> float:
    """Sportsbook home-team quote (favorite-negative, e.g. -8.5) -> nflverse
    spread_line convention (home-favorite-positive): -1 * home_point."""

def fetch_weekly_spreads(season, week, schedule, use_cache=True) -> pd.DataFrame:
    """ONE bulk call for the whole week's slate across both regions. Never
    raises; empty DataFrame on unset key, cache hit, empty schedule, under
    the credit floor, or a failed request."""

def attach_nflverse_comparison(spreads_df, predictions_df) -> pd.DataFrame:
    """Left-join on game_id -> nflverse_spread_line, deviation_pts. Unmatched
    games get NaN, not dropped/errored."""

def upsert_weekly_spreads(rows_df, log_path=LOG_PATH) -> int:
    """Upsert keyed on (season, week, game_id, book_key) - a rerun replaces
    the existing row rather than duplicating it."""
```

### Credit-budget guard

Unlike `market_odds.py` (many per-event calls per run, can check the floor
*between* calls), this module makes exactly **one** call per run, so there's
no "mid-run" to stop at. Instead, the last-observed `x-requests-remaining`
value is persisted to a small local state file
(`data_files/.odds_api_last_remaining`, gitignored — ephemeral run state, not
a data artifact) after every successful call, and read back **before** the
next call: if it's already known to be under `ODDS_API_MIN_REMAINING`, the
call is skipped entirely rather than made and then regretted.

## Where it plugs into the pipeline

Standalone CLI, not threaded through `predict.py` or `nfl-gather-data.py` —
this tracks *sportsbook* data independent of the model pipeline. Run weekly
via its own GitHub Actions workflow (see below) or by hand:
`python spread_tracker.py --week 3`.

## Schema

**`data_files/market_spreads_week{W}_{season}.csv`** — raw per-week
write-once cache (audit trail of exactly what the API returned that week,
before the nflverse join): `season, week, game_id, home_team, away_team,
book_key, book_title, region, home_point, home_price, away_point, away_price,
home_spread_normalized, fetched_at`.

**`data_files/spread_tracker_log.csv`** — the accumulating season-long
artifact:

| Column | Notes |
|---|---|
| `season, week, game_id, gameday, home_team, away_team` | standard |
| `book_key, book_title, region` | e.g. `draftkings`/`DraftKings`/`us`, `playnow_ca`/`PlayNow`/`ca` |
| `home_point, home_price, away_point, away_price` | raw book quote (favorite-negative) |
| `home_spread_normalized` | `-1 * home_point`, nflverse convention |
| `home_implied_prob_devigged, away_implied_prob_devigged` | via `_devig_over_prob(home_price, away_price)` and its complement |
| `nflverse_spread_line` | joined from the predictions CSV; `NaN` if the game isn't in it yet |
| `deviation_pts` | `home_spread_normalized - nflverse_spread_line`; positive = book gives the home side more points than nflverse |
| `fetched_at, source_week_snapshot` | timestamp + which raw cache file this row came from |

Sign-convention anchor (tested explicitly in `tests/test_spread_tracker.py`):
home favored by 9.5 ⇔ `spread_line = 9.5` ⇔ book quotes home at `-9.5` ⇔
`_normalize_spread(-9.5) == 9.5` ⇔ `deviation_pts == 0` when a book agrees
exactly with nflverse.

## Matching logic

Same `game_id = f"{season}_{week:02d}_{away}_{home}"` construction as the
rest of the pipeline, keyed off `TEAM_FULL_NAME`/`_ABBR_BY_FULL_NAME` (full
team name ↔ abbreviation), reused as-is from `market_odds.py`. Games the API
returns that aren't in the current week's schedule (e.g. a future week
already posted) are silently dropped, not an error.

## Config / secrets

Reuses the existing `ODDS_API_KEY` secret/env var — no new secret needed.
Same `ODDS_API_MIN_REMAINING` env var (shared name, shared meaning) as
`market_odds.py`.

## Free-tier budget math

Bulk endpoint: 1 market × 2 regions = **2 credits/call**, once per week ⇒
~38 credits for a full ~19-week season. Combined with the player-prop
feature's own ~1,150 credits/season on the same key, total draw stays well
inside the 500-credit/month free tier (~2,250 available across a season).

## Error handling

| Failure | Behavior |
|---|---|
| `ODDS_API_KEY` unset | empty DF immediately, one log line, zero API calls |
| HTTP error / timeout | caught, logged, empty DF — `continue-on-error: true` in the workflow |
| Unmatched game | dropped from that row, not blocking |
| Credit floor (from last-known state) | call skipped entirely, before it would have been made |
| Cache present | skip the network call entirely |
| Predictions CSV missing/unavailable | `nflverse_spread_line`/`deviation_pts` are `NaN` for every row, not an error |

## GitHub Actions

New dedicated workflow, `.github/workflows/spread-tracker.yml`, **Wednesday
12:00 UTC** (in-season months) rather than appended to
`weekly-model-performance.yml` (Monday 6am UTC, timed for backtesting *last*
week — NFL books don't post the full next-week board until Tuesday, so
Monday would grab stale/partial numbers; the two workflows also have
unrelated failure semantics). Reuses the existing `ODDS_API_KEY` secret.

## Tests

`tests/test_spread_tracker.py` — no live network calls, mirrors
`tests/test_market_odds.py`'s discipline. 28 tests covering:
`_normalize_spread`'s sign conversion (the highest-risk piece — a bug here
would silently corrupt every deviation number), `_region_for_book`
classification, `_parse_bulk_spreads` against a canned bulk-JSON fixture,
`attach_nflverse_comparison`'s join/NaN behavior, `upsert_weekly_spreads`'s
insert/overwrite/append semantics, and `fetch_weekly_spreads`'s
no-key/cache-hit/credit-floor/empty-schedule short-circuits.

`tests/test_spread_tracker_report.py` — mirrors
`tests/test_weekly_spread_report.py`'s `importlib.util` loading pattern
(the script lives in `scripts/`, which `pytest.ini` excludes from
collection). 10 tests covering: `_per_book_bucket`'s mean/mean-abs
deviation math (including the case where positive and negative deviations
average toward zero but `mean_abs` doesn't), `_best_lines_per_game`'s
max-point-per-side selection, `_anomalies`'s threshold flagging and
worst-first ordering, and `build_report`'s overall/by-week aggregation plus
JSON-serializability (numpy int64/float64 leaking into the report would
break `json.dumps`).

## Rollout phases

1. **✅ Done.** `spread_tracker.py` + both CSV artifacts + full test suite +
   this doc. Live-verified against real Week 3 2026 data (144 rows, 16
   games, 10 books across `us`/`ca`, sane deviations, confirmed idempotent
   cache-hit and upsert behavior).
2. **✅ Done.** `scripts/spread_tracker_report.py` rolls `spread_tracker_log.csv`
   up into `data_files/spread_tracker_report.json`: per-book season-to-date
   `mean_deviation_pts`/`mean_abs_deviation_pts` (the latter is the
   direction-agnostic "closest to nflverse" ranking), a `best_line_per_game`
   callout (which book gives the most points to each side, per game), and a
   field-median-relative `anomalies` list generalizing the PlayNow
   divergence first spotted by hand (flags any book more than
   `ANOMALY_THRESHOLD_PTS` = 1.5pt off that game's field median that week).
   Wired into `spread-tracker.yml` right after the fetch step. Live-verified
   against the real Week 3 2026 log: DraftKings ranked closest to nflverse
   (mean|dev|=0.00pt across 16 games), FanDuel furthest of the mainstream
   books (mean|dev|=0.34pt), zero anomalies for that single week (expected —
   the anomaly list only gets interesting once a genuinely mispriced book
   shows up, as PlayNow did in the original ad-hoc chat comparison).
3. **✅ Done.** `pages/5_Spread_Tracker.py` surfaces Phase 2's report: a
   per-book season-to-date ranking table + bar chart (sortable by mean
   absolute deviation from nflverse), a "closest book" callout, a
   best-line-per-game table, an anomalies table, a week selector (defaults
   to season-to-date, can narrow to one week), and a raw-log expander. Purely
   a display layer — reads the already-generated
   `spread_tracker_report.json`/`spread_tracker_log.csv`, never calls The
   Odds API itself. Shows an `st.info` explaining the opt-in feature and how
   to populate it when no report exists yet (e.g. a fresh clone or
   `ODDS_API_KEY` unset).
4. **✅ Done (post-launch addition).** `scripts/spread_value_finder.py` —
   Phase 2's `anomalies` list flags books whose *point number* diverges from
   the field, but a book can move the points and shade the price to
   compensate, netting out to a fair (or even worse) bet; point-only
   divergence isn't the same question as "is this side actually priced
   favorably." Added after a live example (2026-09-17): a chat request to
   recommend PlayNow parlay legs from the Phase 2 anomaly list turned out to
   need price-adjusted fair-value math, not just point-shopping — and a first
   manual pass at that math mislabeled a side by reading the internal
   home-favorite-positive convention directly instead of the log's own
   bettor-facing `home_point`/`away_point` columns. This script is the
   corrected, tested, reusable version: for one target book, computes a fair
   win probability per side (normal approximation of NFL margin of victory,
   `MOV_SIGMA` = 13.5, evaluated at the book's own line against the field
   median) and compares it to what the book's own price requires to break
   even, ranked by edge. Output is always read from the raw bettor-facing
   columns, never re-derived from `home_spread_normalized` — the exact bug
   class `TestComputeBookEdges::test_labels_match_raw_bettor_facing_columns`
   guards against. CLI: `python scripts/spread_value_finder.py --book
   playnow_ca --week 2`. Live-verified against real Week 2 2026 data: exactly
   reproduced every number from the original manual analysis, and also
   surfaced a real edge (DEN -4.0 @ +130 vs JAX, +5.6pt) that Phase 2's
   point-only anomaly detector had missed entirely because the point number
   itself wasn't unusual — only the price was.

## Open questions

1. Open-vs-close line capture — deferred; would need a `line_type` column
   added to the key (see Non-goals). Revisit only if within-week movement
   turns out to matter.
2. `bet99_ca_on` requires a paid Odds API tier and doesn't appear on the
   free plan's `ca` region response — not worth upgrading for one extra
   Canadian book unless the free-tier books turn out to be systematically
   uninformative.
3. Anomaly flags (Phase 2) triggering anything beyond a report line (e.g. a
   notification) — out of scope; this is a hobby research feature, not an
   alerting system.
