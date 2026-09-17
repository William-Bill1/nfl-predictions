# Sportsbook Player-Prop Odds Integration — Design

**Status:** **All 3 phases built.** `ODDS_API_KEY` is set as a GitHub Actions
secret (added 2026-09-16) and confirmed live: the Sep 16 nightly matched 30
real props to DK/FanDuel lines. Caught and fixed a real bug along the way -
`attach_market_odds` gated on `isinstance(prob_over, (int, float))`, but
`model.predict_proba()` returns `numpy.float32` (not a `float` subclass), so
`market_edge` was `NaN` for nearly every match; fixed with a `float()` cast
instead of a type-gate. Phase 2 (`pages/2_Player_Props.py` main table):
"Market" / "Market Edge" columns, sorted by a *directional* edge (flipped to
align with the recommendation - `market_edge` itself is always in P(over)
terms, so a strongly negative value on an UNDER pick means a *strong* edge,
not a weak one; sorting/display use the flipped, always-"bigger is better"
version). Phase 3 (DK Pick 6 Calculator tab): the line input pre-fills from
a matched market line when one exists (still fully editable, labeled as a
sportsbook line to confirm against the Pick 6 board, not the Pick 6 number
itself).
**Provider:** [The Odds API](https://the-odds-api.com/) (see chat discussion — free tier is
real, DK + FanDuel covered by name, player-prop market keys line up with what
`player_props/models.py` already predicts).
**Goal:** replace the fixed internal prop-line tiers (275 passing yards, 7.5
receptions, …) with real DraftKings/FanDuel lines for the four **reliable**
prop types, so `edge = model_prob − market_implied_prob` becomes a real number
instead of "how often a player clears an arbitrary round line."

## Non-goals (this pass)

- **Spread/moneyline/totals** stay on the nflverse consensus line **for
  player-prop edge purposes** — the spread model has no proven out-of-sample
  edge (`Spread_OOS_Test` in `model_metrics.json`), so precise live game odds
  don't fix that. Superseded for a different purpose (comparing sportsbooks
  against each other and against nflverse, independent of model edge) by
  `spread_tracker.py` — see `docs/MARKET_SPREAD_TRACKER_PLAN.md`.
- **TD props** (`passing_tds`, `rushing_tds`, `receiving_tds`) are already
  force-flagged `reliable = False` in `player_props/models.py` (every tier
  collapses to the same 0.5 line, ~coin-flip out-of-time). Don't spend credits
  pulling market odds for markets the app already tells users to ignore.
- **Historical backtest** — The Odds API's historical endpoint only covers
  "featured markets" (moneyline/spread/total), not player props, so this can't
  retroactively validate past weeks. It's a going-forward integration only,
  same shape as the existing write-once weekly snapshot.
- **DK Pick 6 specifically** — Pick 6 is DraftKings' own DFS-style product
  with its own line-setting, not the same feed as DraftKings' regular
  sportsbook player props. This integration pulls **regular sportsbook**
  DK/FanDuel prop lines, which are usually close to Pick 6's lines but not
  guaranteed identical. The DK Pick 6 Calculator (`pages/2_Player_Props.py`)
  keeps manual line entry; market odds would only pre-fill it as a *starting
  point*, with a note that it's the sportsbook line, not the Pick 6 line.

## New module: `player_props/market_odds.py`

Mirrors the existing `player_props/injuries.py` shape (`get_injury_report` /
`find_player_injury` / `adjust_prediction_for_injury`) so it fits the
established pattern of an optional, cache-backed enrichment step:

```python
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_REGION = "us"
# Only the reliable prop types - see Non-goals.
ODDS_API_MARKETS = {
    "passing_yards": "player_pass_yds",
    "rushing_yards": "player_rush_yds",
    "receiving_yards": "player_reception_yds",
    "receptions": "player_receptions",
}
BOOKMAKERS = ("draftkings", "fanduel")

def fetch_market_odds(season, week, schedule: pd.DataFrame,
                       use_cache: bool = True) -> pd.DataFrame:
    """One row per (game, player, prop_type, book): line + over/under price.

    Returns an empty DataFrame - never raises - when ODDS_API_KEY is unset,
    every request fails, or the response can't be parsed. Callers must treat
    "no market odds" as a normal, expected state (same contract as
    get_injury_report's empty-DataFrame-on-failure).
    """

def find_market_line(display_name: str, team: str, prop_type: str,
                      odds_df: pd.DataFrame | None) -> dict | None:
    """Best-available (BOOKMAKERS preference order) line for one player/prop.
    Returns {'line': float, 'book': str, 'over_odds': int, 'under_odds': int,
             'market_implied_prob': float} or None if unmatched.
    """

def attach_market_odds(prediction: dict, market_info: dict | None) -> dict:
    """Adds market_line / market_book / market_implied_prob / market_edge /
    market_line_available to a prediction dict in place. No-ops (leaves the
    fixed-tier fields as-is) when market_info is None.
    """
```

**Implementation simplification vs. the original sketch above:** rather than
two separate caching layers (a short-TTL cache plus the frozen weekly
artifact), the frozen artifact itself *is* the cache —
`fetch_market_odds` checks `data_files/market_odds_week{W}_{season}.csv` first
and returns it unfetched if present, matching the write-once philosophy the
prop snapshot already uses (`predict.py::generate_predictions`) and giving a
*stronger* budget guarantee than a time-based TTL: once fetched for a week, a
week is never re-fetched, full stop, regardless of how many times `predict.py`
is rerun that week.

### Credit-budget guard (built in, not bolted on)

`fetch_market_odds` reads the response headers The Odds API returns on every
call — `x-requests-remaining` / `x-requests-used` — and:
- logs remaining credits after every call (`print(f"[market_odds] {remaining} credits left")`),
- stops issuing further per-event calls **mid-run** if remaining drops below a
  floor (`ODDS_API_MIN_REMAINING`, default 20) and returns whatever it already
  fetched, rather than erroring out or burning the account to zero,
- this floor is a constant, not a tier-specific hardcode — it behaves
  identically on the free tier and the $30 tier, it just gets hit less often
  on the paid one.

## Where it plugs into `predict.py`

`generate_predictions(season, week, freeze, skip_injuries, skip_weather)` at
`player_props/predict.py:1015` already threads an optional enrichment step
through per-game prediction (`predict_props_for_game`, line 822, takes
`skip_injuries`/`skip_weather` and calls `get_injury_report()` /
`get_weather_for_game()` internally). Add a third, same-shaped flag:

```python
def generate_predictions(season=None, week=None, freeze=True,
                          skip_injuries=False, skip_weather=False,
                          skip_market_odds=False):
    ...
    market_odds_df = (pd.DataFrame() if skip_market_odds
                       else fetch_market_odds(season, resolved_week))
```

Then in `predict_props_for_game` (or in the prediction-assembly loop it
feeds), after a prediction dict is built from the fixed-tier line, call
`attach_market_odds(prediction, find_market_line(...))` — same insertion
point as the existing `adjust_prediction_for_injury` / `adjust_for_weather`
calls.

New CLI flag on the existing `argparse` block: `--no-market-odds` (mirrors
`--no-injuries` / `--no-weather` / `--no-freeze`).

## Schema changes

**`player_props_predictions.csv` / `..._week{W}_{season}.csv`** gain columns
(additive — nothing existing changes):

| Column | Meaning |
|---|---|
| `market_line` | best available DK/FanDuel line for this player+prop |
| `market_book` | which book it came from (`draftkings` / `fanduel`) |
| `market_implied_prob` | vig-adjusted implied P(over) from `market_over_odds`/`market_under_odds`, same `implied_prob()` math already in `nfl-gather-data.py` |
| `market_edge` | `prob_over − market_implied_prob` (the *real* edge, parallel to `edge_underdog_spread` on the game side) |
| `market_line_available` | bool — False when unmatched/no key/quota hit, so the UI can distinguish "no edge" from "no market data" |

`line_value` (the existing fixed-tier column) is **kept**, not replaced — it's
still what the trained model's threshold was calibrated against, and losing
it would break `backtest.py`'s existing hit-rate logic. `market_line` is a
new, independent field for display/edge purposes.

**New frozen artifact** (mirrors `betting_log.py` → `spread_performance.json`
and the write-once prop snapshot): `data_files/market_odds_week{W}_{season}.csv`
— raw fetched rows (one per player/prop/book, before matching/aggregation),
written once per week alongside the prop snapshot. This is the audit trail:
if a market line looks wrong later, you can check what was actually returned
that week without re-querying (impossible anyway, since historical player
props aren't available from the API).

## Name & team matching

The prop stats' join key is **`display_name`** (full name: "Nick Mullens",
"Ollie Gordon II" — see `player_props_predictions_week2_2026.csv`), not the
abbreviated `player_name` ("N.Mullens") used for rolling-stat lookups. The
Odds API returns full names too, so match on `display_name` directly, with:
- exact match first,
- fallback: normalize both sides (strip `Jr.`/`Sr.`/`II`/`III`, lowercase,
  strip punctuation) and retry,
- log every unmatched player at the end of a run (`print(f"[market_odds]
  {n} players had no market line: {names[:10]}...")`) — same "tell me what's
  missing" discipline as `⚠️ Model {name} not found, skipping` in
  `predict.py::load_models`. Don't fail silently on a miss; just leave
  `market_line_available = False` for that row.

Team codes: The Odds API returns full team names ("Kansas City Chiefs"); the
prop stats use abbreviations ("KC"). `predictions.py:2908` already has a
`team_full_name_map` (abbr → full) defined inline inside a function — not
importable as-is. Cheapest option: a small **inverted copy** local to
`market_odds.py` (matches the codebase's existing style of small inline maps
rather than a new shared module for one dict). Optional cleanup if this ever
gets built: hoist the one true map into `season_utils.py` and have both call
sites import it — flagging as a nice-to-have, not a blocker.

## Config / secrets

- New GitHub Actions secret: `ODDS_API_KEY`.
- New env var read by `market_odds.py`: `ODDS_API_KEY` (same name locally via
  `.env`, consistent with `EMAIL_PASSWORD` etc. in `.env.example`).
- **Unset by default everywhere** — local dev, CI, and a fresh clone all work
  with zero market-odds calls until someone opts in by setting the key. This
  is the same posture as `PROP_ROSTER_FILTER=1` (opt-in, not opt-out) from
  the roster-filter work, but for a different reason: that one was gated
  because the underlying data was bad; this one is gated because it costs
  real money once the free tier is exceeded, so silently-always-on is the
  wrong default for a hobby project.

## Free-tier budget math (design target)

Player props require **one call per game** (`/v4/sports/{sport}/events/{id}/odds`,
not the bulk `/odds` endpoint), costing `markets × regions` credits per call:

- 4 markets (the reliable subset) × 1 region (`us`) = **4 credits/game**
- 16-game week ≈ **64 credits**
- Pulled **once per week** (matching the existing write-once frozen-snapshot
  cadence — `predict.py` already only generates each week's snapshot once) ⇒
  a full ~18-week season ≈ 1,150 credits total, spread across ~4.5 months of
  free-tier resets (500/month × 4.5 ≈ 2,250 available). **Fits the free tier
  with room to spare**, as long as nothing calls it more than once per week.

The caching layer (6h TTL) is what enforces "once per week" in practice even
if `predict.py` gets rerun by hand or the nightly retries.

## The $30/mo (20,000 credits/month) upgrade path

Nothing in the design above hardcodes the free tier — the upgrade is a
**cadence change, not a code change**:

- Add a step to `nightly-update.yml` calling `predict.py` with market odds
  enabled on nights it currently skips it (right now props only freeze once
  a week on the nightly that lands the new upcoming week) — i.e. refresh
  `market_line` **daily** as lines move, instead of once at freeze time.
  64 credits/day × 7 ≈ 450/week — trivial against 20,000/month.
- Or: widen `ODDS_API_MARKETS` to include TD props too, or add a second
  region, without hitting the budget guard.
- The `ODDS_API_MIN_REMAINING` floor and the per-call credit logging mean
  you'd *see* the free tier getting tight (via the nightly Action logs)
  before it ever silently failed — that's the signal to flip the plan.

## Error handling / graceful degradation

Every failure mode falls back to **today's behavior** (fixed-tier lines, no
`market_*` columns populated / `market_line_available=False`), never a hard
failure of the prediction pipeline:

| Failure | Behavior |
|---|---|
| `ODDS_API_KEY` unset | `fetch_market_odds` returns empty DF immediately, one log line, zero API calls |
| HTTP error / timeout | caught, logged, empty DF for that call — same `continue-on-error: true` posture as the injuries/weather steps already have in the nightly workflow |
| Player/team unmatched | that row's `market_line_available = False`; doesn't block other rows |
| Credit floor hit mid-run | stop fetching, keep what's already fetched, log it |
| Cache present and fresh | skip the network call entirely |

## UI changes (separate follow-up PR, not required to land with the fetcher)

- **Player Props page** (`pages/2_Player_Props.py`): add a "Market Line"
  column (from `market_line`/`market_book`) next to the existing prop table,
  and prefer sorting/filtering by `market_edge` when available, falling back
  to the current `confidence` sort when it's not (unmatched player, no key).
- **DK Pick 6 Calculator**: pre-fill the line input from `market_line` when
  available (labeled "DraftKings sportsbook line — confirm against the Pick 6
  board" per the Non-goals caveat above), still fully editable.

## Tests (`tests/test_market_odds.py`)

No live API calls in CI — mock `requests.get` / feed canned JSON fixtures,
matching how `tests/test_betting_log.py` and `tests/test_prop_roster_filter.py`
build small in-memory DataFrames rather than touching real data:

- `find_market_line`: exact match, suffix-normalized match, unmatched → `None`.
- `attach_market_odds`: populates fields when given a match; leaves the
  prediction dict's existing fixed-tier fields untouched when `None`.
- `fetch_market_odds`: unset key → empty DF, no network attempted (patch
  `requests.get` with a `Mock` that raises if called, to prove it); credit
  floor stops mid-run (feed a fake header sequence).
- Team-map round-trip: every abbreviation in `player_props_predictions` `team`
  column resolves through the inverted map.

## Docs to update once this is actually built

- `README.md` — Data table gets a new row (`The Odds API — player-prop lines
  — opt-in, ODDS_API_KEY`); troubleshooting row for "market lines missing."
- `docs/architecture.md` — API Integrations table + Player Props section
  (note `market_*` columns, the credit-budget guard, weekly cadence).
- `.github/copilot-instructions.md` — new env var, new opt-in pattern to list
  alongside `PROP_ROSTER_FILTER`.
- `CHANGELOG.md` — standard dated entry when it ships.
- `.env.example` — add `ODDS_API_KEY=`.

## Rollout phases

1. **✅ Done.** `market_odds.py` + the frozen artifact + tests, wired into
   `predict.py` behind `--no-market-odds`. Live-verified against a real
   nightly run (30/880 props matched, 14/500 free credits used).
2. **✅ Done.** `market_line`/`market_edge`/etc. columns were already on the
   predictions CSV from Phase 1; added the Player Props page table columns
   ("Market", "Market Edge") plus edge-aware sorting.
3. **✅ Done.** DK Pick 6 Calculator tab's line input pre-fills from a
   matched `market_line` when the selected player/stat has one; falls back
   to the original static default (100.5) otherwise. Widget `key` is scoped
   to `(player, stat)` rather than a fixed string, since Streamlit ignores a
   new `value=` once a fixed key already has a stored session_state entry -
   without that, switching players wouldn't actually change the shown
   default after the first render.

## Open questions - resolved / still open

1. ~~Account + `ODDS_API_KEY`~~ — **done.** Key added as a GitHub Actions
   secret 2026-09-16, confirmed live against the nightly.
2. Weekly cadence (matches the write-once snapshot, safest for the free
   tier) vs. daily-refresh-of-market-line-only-while-keeping-model-predictions-weekly
   — still on weekly by default. Revisit only if fresher in-week line movement
   turns out to matter for the props that get built on top of this.
3. ~~Commit the frozen artifact?~~ — **yes**, going with committed
   (`market_odds_week2_2026.csv` is in git) — same audit-trail reasoning as
   the player-prop weekly snapshots.
4. ~~Phase 3 (DK Pick 6 pre-fill)~~ — **done.** No open questions remain;
   this doc is now a historical design record rather than a plan.
