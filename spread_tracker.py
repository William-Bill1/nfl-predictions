"""Season-long sportsbook game-spread tracker (The Odds API, bulk endpoint).

Distinct from player_props/market_odds.py, which pulls PLAYER PROP lines via
the per-event endpoint. This module pulls GAME SPREAD lines via the bulk
/v4/sports/{sport}/odds endpoint, across both US and Canadian regions, so the
season-to-date question "which sportsbook(s) consistently beat nflverse's
spread_line?" can be answered from an accumulating CSV instead of one-off
manual pulls. See docs/MARKET_SPREAD_TRACKER_PLAN.md for the full design.

Off by default everywhere (local dev, CI, a fresh clone) until ODDS_API_KEY is
set - every public function degrades to an empty DataFrame on any failure,
never raises, so the rest of the pipeline is unaffected when this is
unconfigured or the API is unreachable.

Phase 1 (this module): fetch + upsert into a season-long CSV. No comparison
report and no UI yet.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

try:  # picks up ODDS_API_KEY from a local .env for CLI/standalone runs
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from player_props.market_odds import (
    ODDS_API_BASE,
    ODDS_API_SPORT,
    TEAM_FULL_NAME,
    _ABBR_BY_FULL_NAME,
    _devig_over_prob,
    _read_remaining,
)

DATA_DIR = Path(__file__).parent / 'data_files'

ODDS_API_KEY = os.getenv('ODDS_API_KEY', '')
ODDS_API_REGIONS = 'us,ca'  # 1 market x 2 regions = 2 credits per call
ODDS_API_MARKET = 'spreads'
# Same floor/semantics as player_props/market_odds.py, but this module only
# ever makes ONE call per run (the bulk endpoint), so the check happens
# before that single call rather than mid-loop.
ODDS_API_MIN_REMAINING = int(os.getenv('ODDS_API_MIN_REMAINING', '20'))
REQUEST_TIMEOUT = 15

RAW_COLUMNS = [
    'season', 'week', 'game_id', 'home_team', 'away_team',
    'book_key', 'book_title', 'region',
    'home_point', 'home_price', 'away_point', 'away_price',
    'home_spread_normalized', 'fetched_at',
]

LOG_COLUMNS = [
    'season', 'week', 'game_id', 'gameday', 'home_team', 'away_team',
    'book_key', 'book_title', 'region',
    'home_point', 'home_price', 'away_point', 'away_price',
    'home_spread_normalized',
    'home_implied_prob_devigged', 'away_implied_prob_devigged',
    'nflverse_spread_line', 'deviation_pts',
    'fetched_at', 'source_week_snapshot',
]
LOG_PATH = DATA_DIR / 'spread_tracker_log.csv'
KEY_COLUMNS = ['season', 'week', 'game_id', 'book_key']


def _credits_state_path() -> Path:
    """Ephemeral local state (not committed - see .gitignore): the last
    remaining-credits count this module observed, so a single-call fetch can
    skip BEFORE making that call when we're already known to be under the
    floor. Unlike market_odds.py (many calls per run, can check mid-loop),
    this module makes exactly one call per run, so there is no "mid-run" to
    stop at - the only place to act on a low count is before the next run's
    call, using what the previous run last saw.
    """
    return DATA_DIR / '.odds_api_last_remaining'


def _last_known_remaining() -> int | None:
    try:
        return int(_credits_state_path().read_text().strip())
    except (OSError, ValueError):
        return None


def _persist_remaining(remaining: int | None) -> None:
    if remaining is None:
        return
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _credits_state_path().write_text(str(remaining))
    except OSError:
        pass


def _cache_path(season, week) -> Path:
    """Raw per-week fetch cache - write-once, same philosophy as
    market_odds.py's per-week file: the audit trail of exactly what the API
    returned that week, before any normalization/join.
    """
    return DATA_DIR / f'market_spreads_week{int(week)}_{int(season)}.csv'


def _region_for_book(book_key: str, book_title: str | None = None) -> str:
    """Most Canadian province-licensed books The Odds API returns key its
    bookmaker with a '_ca' substring (betmgm_ca_on, playnow_ca, proline_ca_on,
    sportsinteraction_ca_on, betano_ca_on, betrivers_ca_on, ...) - US books
    never contain '_ca' (draftkings, fanduel, betmgm, ...; plain 'betmgm'
    doesn't collide with 'betmgm_ca_on' under a substring check.

    `pointsbetca` is a real, live-observed exception: no '_ca' substring in
    the key, so the key-only check misses it (caught 2026-09-17 - it showed
    up as region 'us' in a real report despite its own title reading
    "PointsBet (CA - ON)"). The API's `bookmakers[].title` reliably carries a
    literal '(CA' marker for every Canadian book observed (including this
    one), so it's used as a second signal when supplied. Anything neither
    signal catches defaults to 'us' (the overwhelming majority), so a future
    new US book is never miscategorized as CA.
    """
    if '_ca' in (book_key or ''):
        return 'ca'
    if book_title and '(CA' in book_title:
        return 'ca'
    return 'us'


def _normalize_spread(home_point: float) -> float:
    """Sportsbook home-team point number (favorite-negative, e.g. -8.5) ->
    nflverse spread_line convention (home-favorite-positive).

    spread_line convention (confirmed empirically against
    nfl_games_historical_with_predictions.csv): positive = home favored by
    that many points, negative = away favored. Sportsbook convention: the
    home team's own quoted number is negative when home is favored (e.g.
    "Baltimore -8.5"). => spread_line_equivalent = -1 * home_point.
    """
    return -1.0 * float(home_point)


def _fetch_bulk_spreads():
    """One paid call: 1 market x 2 regions = 2 credits, ALL games + all
    books at once. Returns (events_json, remaining_credits)."""
    resp = requests.get(
        f"{ODDS_API_BASE}/sports/{ODDS_API_SPORT}/odds",
        params={
            'apiKey': ODDS_API_KEY,
            'regions': ODDS_API_REGIONS,
            'markets': ODDS_API_MARKET,
            'oddsFormat': 'american',
            'dateFormat': 'iso',
        },
        timeout=REQUEST_TIMEOUT,
    )
    remaining = _read_remaining(resp)
    resp.raise_for_status()
    return resp.json(), remaining


def _parse_bulk_spreads(events_json, season, week, game_id_by_teams: dict) -> list[dict]:
    """Bulk /odds response -> one row per (game, book) that matches a game in
    game_id_by_teams (keyed by (home_full_name, away_full_name)).
    """
    now = datetime.now().isoformat(timespec='seconds')
    rows: list[dict] = []
    for ev in events_json or []:
        home_full, away_full = ev.get('home_team'), ev.get('away_team')
        game_id = game_id_by_teams.get((home_full, away_full))
        if game_id is None:
            continue
        home_abbr = _ABBR_BY_FULL_NAME.get(home_full, home_full)
        away_abbr = _ABBR_BY_FULL_NAME.get(away_full, away_full)
        for bk in ev.get('bookmakers', []) or []:
            mkt = next(
                (m for m in bk.get('markets', []) or [] if m.get('key') == ODDS_API_MARKET),
                None,
            )
            if mkt is None:
                continue
            outcomes = {o.get('name'): o for o in mkt.get('outcomes', []) or []}
            home_o, away_o = outcomes.get(home_full), outcomes.get(away_full)
            if not home_o or not away_o or home_o.get('point') is None:
                continue
            home_point = home_o.get('point')
            book_key = bk.get('key', '')
            book_title = bk.get('title')
            rows.append({
                'season': season, 'week': week, 'game_id': game_id,
                'home_team': home_abbr, 'away_team': away_abbr,
                'book_key': book_key, 'book_title': book_title,
                'region': _region_for_book(book_key, book_title),
                'home_point': home_point, 'home_price': home_o.get('price'),
                'away_point': away_o.get('point'), 'away_price': away_o.get('price'),
                'home_spread_normalized': _normalize_spread(home_point),
                'fetched_at': now,
            })
    return rows


def fetch_weekly_spreads(season, week, schedule: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    """Raw sportsbook spread quotes for one week's games, across US + CA
    books. NOT yet joined to nflverse's spread_line or upserted into the
    season log - see attach_nflverse_comparison / upsert_weekly_spreads.

    `schedule` is the week's games (needs home_team/away_team abbreviations -
    the same DataFrame player_props.predict.load_schedule() returns).

    Returns a DataFrame with columns RAW_COLUMNS. Returns an empty one - never
    raises - when ODDS_API_KEY is unset, this week was already fetched (cache
    hit), remaining credits are under the floor, schedule is empty, or the
    request fails.
    """
    cache_file = _cache_path(season, week)
    if use_cache and cache_file.exists():
        print(f"[spread_tracker] using cached {cache_file.name} (write-once - delete it to refetch)")
        return pd.read_csv(cache_file)

    if not ODDS_API_KEY:
        print("[spread_tracker] ODDS_API_KEY not set - skipping spread tracking")
        return pd.DataFrame(columns=RAW_COLUMNS)

    if schedule is None or schedule.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)

    remaining_known = _last_known_remaining()
    if remaining_known is not None and remaining_known < ODDS_API_MIN_REMAINING:
        print(f"[spread_tracker] skipping - last known {remaining_known} credits remaining "
              f"is under the floor {ODDS_API_MIN_REMAINING}")
        return pd.DataFrame(columns=RAW_COLUMNS)

    game_id_by_teams = {
        (TEAM_FULL_NAME.get(row['home_team']), TEAM_FULL_NAME.get(row['away_team'])):
            f"{int(season)}_{int(week):02d}_{row['away_team']}_{row['home_team']}"
        for _, row in schedule.iterrows()
    }

    try:
        events_json, remaining = _fetch_bulk_spreads()
    except requests.RequestException as e:
        print(f"[spread_tracker] bulk spreads request failed: {e}")
        return pd.DataFrame(columns=RAW_COLUMNS)

    _persist_remaining(remaining)
    if remaining is not None:
        print(f"[spread_tracker] {remaining} credits remaining after this call")

    rows = _parse_bulk_spreads(events_json, season, week, game_id_by_teams)
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    print(f"[spread_tracker] fetched {len(df)} game/book spread rows "
          f"({df['game_id'].nunique() if not df.empty else 0} games, "
          f"{df['book_key'].nunique() if not df.empty else 0} books)")

    if use_cache:
        # Write even when empty, so a week with zero matched games doesn't
        # get re-queried (for credits) on every subsequent run this week.
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False)
        print(f"[spread_tracker] wrote {cache_file.name}")

    return df


def attach_nflverse_comparison(spreads_df: pd.DataFrame, predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join spreads_df.game_id -> predictions_df.spread_line/gameday, add
    nflverse_spread_line and deviation_pts = home_spread_normalized -
    nflverse_spread_line. Rows for games not yet in predictions_df get NaN in
    both - not dropped, not an error.
    """
    if spreads_df is None or spreads_df.empty:
        return pd.DataFrame(columns=LOG_COLUMNS)

    if predictions_df is not None and not predictions_df.empty and 'game_id' in predictions_df.columns:
        cols = [c for c in ('game_id', 'gameday', 'spread_line') if c in predictions_df.columns]
        ref = predictions_df[cols].drop_duplicates(subset='game_id').rename(
            columns={'spread_line': 'nflverse_spread_line'}
        )
    else:
        ref = pd.DataFrame(columns=['game_id', 'gameday', 'nflverse_spread_line'])

    merged = spreads_df.merge(ref, on='game_id', how='left')
    if 'gameday' not in merged.columns:
        merged['gameday'] = None
    if 'nflverse_spread_line' not in merged.columns:
        merged['nflverse_spread_line'] = pd.NA

    merged['home_implied_prob_devigged'] = merged.apply(
        lambda r: _devig_over_prob(r['home_price'], r['away_price']), axis=1
    )
    merged['away_implied_prob_devigged'] = merged['home_implied_prob_devigged'].apply(
        lambda p: (1.0 - p) if p is not None else None
    )
    merged['deviation_pts'] = merged['home_spread_normalized'] - pd.to_numeric(
        merged['nflverse_spread_line'], errors='coerce'
    )
    merged['source_week_snapshot'] = merged.apply(
        lambda r: _cache_path(r['season'], r['week']).name, axis=1
    )

    return merged.reindex(columns=LOG_COLUMNS)


def upsert_weekly_spreads(rows_df: pd.DataFrame, log_path: Path = LOG_PATH) -> int:
    """Upsert into the season-long log, keyed on
    (season, week, game_id, book_key). A rerun for the same key replaces the
    existing row (current known line) rather than duplicating it - unlike
    betting_log.py's "append once, dedupe pending-only" pattern, this tracks
    "current known line for a book/game/week," and duplicates would
    double-count in any season-to-date aggregate. Returns rows written.
    """
    if rows_df is None or rows_df.empty:
        return 0

    if log_path.exists() and log_path.stat().st_size > 0:
        existing = pd.read_csv(log_path)
        new_keys = set(map(tuple, rows_df[KEY_COLUMNS].values))
        existing_keys = existing[KEY_COLUMNS].apply(tuple, axis=1)
        existing = existing[~existing_keys.isin(new_keys)]
        combined = pd.concat([existing, rows_df], ignore_index=True)
    else:
        combined = rows_df

    combined = combined.sort_values(KEY_COLUMNS).reset_index(drop=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(log_path, index=False, columns=LOG_COLUMNS)
    return len(rows_df)


if __name__ == '__main__':
    import argparse
    try:
        from season_utils import upcoming_or_current_season
        from player_props.predict import load_schedule
    except ImportError:  # run as `python spread_tracker.py` from elsewhere
        import sys as _sys
        _sys.path.append(str(Path(__file__).parent))
        from season_utils import upcoming_or_current_season
        from player_props.predict import load_schedule

    ap = argparse.ArgumentParser(description="Fetch/track US+CA sportsbook game-spread lines for one week")
    ap.add_argument('--season', type=int, default=None)
    ap.add_argument('--week', type=int, default=None, help="default: next upcoming week")
    ap.add_argument('--no-cache', action='store_true', help="ignore/overwrite an existing frozen raw-week file")
    args = ap.parse_args()

    season = args.season or upcoming_or_current_season()
    sched = load_schedule(season=season, week=args.week)
    if sched is None or sched.empty:
        print("No schedule/week resolved - nothing to fetch")
        raise SystemExit(1)
    week = args.week or int(sched['week'].iloc[0])

    raw = fetch_weekly_spreads(season, week, sched, use_cache=not args.no_cache)

    predictions_path = DATA_DIR / 'nfl_games_historical_with_predictions.csv'
    # Despite the .csv extension this file is TAB-separated (see betting_log.py's
    # own PREDICTIONS_PATH read) - a plain pd.read_csv() silently parses it as
    # one giant unsplit column, so every join against it below would look like
    # a 100% miss (all-NaN nflverse_spread_line) without ever raising.
    predictions_df = pd.read_csv(predictions_path, sep='\t') if predictions_path.exists() else pd.DataFrame()

    joined = attach_nflverse_comparison(raw, predictions_df)
    n_written = upsert_weekly_spreads(joined)
    print(f"[spread_tracker] upserted {n_written} rows for season {season} week {week} "
          f"into {LOG_PATH.name}")
