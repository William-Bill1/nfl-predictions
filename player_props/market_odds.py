"""
Sportsbook player-prop odds (The Odds API).

Optional, opt-in enrichment: fetches real DraftKings/FanDuel prop lines so
`market_edge = model_prob - market_implied_prob` is a real number instead of
"how often a player clears an arbitrary internal round-number tier." See
docs/ODDS_API_INTEGRATION_PLAN.md for the full design, the credit-budget math
this module is built to respect, and why spread/totals odds and TD props are
deliberately out of scope.

Off by default everywhere (local dev, CI, a fresh clone) until ODDS_API_KEY is
set - every public function degrades to an empty DataFrame / None on any
failure, never raises, so the prediction pipeline is unaffected when this is
unconfigured or the API is unreachable.

Phase 1 (this module): fetch + freeze one CSV per week, no UI wiring yet.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:  # picks up ODDS_API_KEY from a local .env for CLI/standalone runs;
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATA_DIR = Path(__file__).parent.parent / 'data_files'

ODDS_API_KEY = os.getenv('ODDS_API_KEY', '')
ODDS_API_BASE = 'https://api.the-odds-api.com/v4'
ODDS_API_SPORT = 'americanfootball_nfl'
ODDS_API_REGION = 'us'
# Stop issuing further per-event calls once remaining credits drop below this,
# rather than draining the account or erroring out. Same floor on the free
# tier and the $30/mo tier - it just gets hit less often on the paid one.
ODDS_API_MIN_REMAINING = int(os.getenv('ODDS_API_MIN_REMAINING', '20'))
REQUEST_TIMEOUT = 15

# Only the prop types models.py has flagged reliable (temporal hold-out
# AUC >= 0.58 and accuracy above the base rate). TD props are force-flagged
# unreliable there (every tier collapses to the same 0.5 line) - not worth
# spending credits on. Keys are our prop_type; values are the API's market key.
MARKET_KEYS = {
    'passing_yards': 'player_pass_yds',
    'rushing_yards': 'player_rush_yds',
    'receiving_yards': 'player_reception_yds',
    'receptions': 'player_receptions',
}
_PROP_TYPE_BY_MARKET = {v: k for k, v in MARKET_KEYS.items()}

# Preference order when more than one book has a line for the same player/prop.
BOOKMAKERS = ('draftkings', 'fanduel')

ODDS_COLUMNS = [
    'season', 'week', 'game_id', 'home_team', 'away_team',
    'player_name', 'prop_type', 'book', 'line', 'over_odds', 'under_odds',
]

# nflverse team abbreviation -> The Odds API's full team name.
TEAM_FULL_NAME = {
    'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills', 'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars',
    'KC': 'Kansas City Chiefs', 'LV': 'Las Vegas Raiders', 'LAC': 'Los Angeles Chargers',
    'LA': 'Los Angeles Rams', 'MIA': 'Miami Dolphins', 'MIN': 'Minnesota Vikings',
    'NE': 'New England Patriots', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
    'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers',
    'SF': 'San Francisco 49ers', 'SEA': 'Seattle Seahawks', 'TB': 'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders',
}
_ABBR_BY_FULL_NAME = {v: k for k, v in TEAM_FULL_NAME.items()}


def _cache_path(season, week) -> Path:
    """Same write-once file the props snapshot uses as its own cache: if it's
    already there, we already spent the credits for this week - don't refetch.
    """
    return DATA_DIR / f'market_odds_week{int(week)}_{int(season)}.csv'


def _implied_prob(american_odds):
    """American odds -> raw (vig-included) implied probability."""
    try:
        o = float(american_odds)
    except (TypeError, ValueError):
        return np.nan
    if o == 0 or np.isnan(o):
        return np.nan
    return (-o) / ((-o) + 100) if o < 0 else 100 / (o + 100)


def _devig_over_prob(over_odds, under_odds):
    """Two-way devig: normalize both raw implied probs so they sum to 1,
    removing the book's built-in vig. None if either side is missing/invalid.
    """
    p_over = _implied_prob(over_odds)
    p_under = _implied_prob(under_odds)
    if np.isnan(p_over) or np.isnan(p_under) or (p_over + p_under) <= 0:
        return None
    return p_over / (p_over + p_under)


def _read_remaining(resp) -> int | None:
    try:
        return int(resp.headers.get('x-requests-remaining'))
    except (TypeError, ValueError):
        return None


def _list_events():
    """Free call (doesn't count against the usage quota) - all upcoming NFL
    events with their event IDs, home/away team full names, and kickoff time.
    """
    resp = requests.get(
        f"{ODDS_API_BASE}/sports/{ODDS_API_SPORT}/events",
        params={'apiKey': ODDS_API_KEY, 'dateFormat': 'iso'},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _fetch_event_odds(event_id):
    """One paid call: markets x regions credits (4 markets x 1 region = 4)."""
    resp = requests.get(
        f"{ODDS_API_BASE}/sports/{ODDS_API_SPORT}/events/{event_id}/odds",
        params={
            'apiKey': ODDS_API_KEY,
            'regions': ODDS_API_REGION,
            'markets': ','.join(MARKET_KEYS.values()),
            'bookmakers': ','.join(BOOKMAKERS),
            'oddsFormat': 'american',
        },
        timeout=REQUEST_TIMEOUT,
    )
    remaining = _read_remaining(resp)
    resp.raise_for_status()
    return resp.json(), remaining


def _parse_event_odds(event_json, season, week, game_id) -> list[dict]:
    """Event odds JSON -> one row per (book, prop_type, player): pairs the
    Over/Under outcomes (same `point`) into a single line/over_odds/under_odds
    row. Outcomes missing one side of the pair are dropped (can't devig).
    """
    rows: dict[tuple, dict] = {}
    for bk in event_json.get('bookmakers', []) or []:
        book = bk.get('key')
        for mkt in bk.get('markets', []) or []:
            prop_type = _PROP_TYPE_BY_MARKET.get(mkt.get('key'))
            if not prop_type:
                continue
            for outcome in mkt.get('outcomes', []) or []:
                player_name = outcome.get('description')
                side = str(outcome.get('name', '')).lower()
                if not player_name or side not in ('over', 'under'):
                    continue
                key = (book, prop_type, player_name, outcome.get('point'))
                row = rows.setdefault(key, {
                    'season': season, 'week': week, 'game_id': game_id,
                    'home_team': event_json.get('home_team'),
                    'away_team': event_json.get('away_team'),
                    'player_name': player_name, 'prop_type': prop_type,
                    'book': book, 'line': outcome.get('point'),
                    'over_odds': None, 'under_odds': None,
                })
                row[f'{side}_odds'] = outcome.get('price')
    return [r for r in rows.values() if r['over_odds'] is not None and r['under_odds'] is not None]


def fetch_market_odds(season, week, schedule: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    """DK/FanDuel player-prop lines for one week's games.

    `schedule` is the week's games (needs home_team/away_team abbreviations -
    the same DataFrame player_props.predict.load_schedule() already returns),
    used only to filter the sport-wide event list down to this week's games.

    Returns a DataFrame with columns ODDS_COLUMNS. Returns an empty one -
    never raises - when ODDS_API_KEY is unset, this week was already fetched
    (cache hit), or every request fails.
    """
    cache_file = _cache_path(season, week)
    if use_cache and cache_file.exists():
        print(f"[market_odds] using cached {cache_file.name} (write-once - delete it to refetch)")
        return pd.read_csv(cache_file)

    if not ODDS_API_KEY:
        print("[market_odds] ODDS_API_KEY not set - skipping market odds (fixed tiers only)")
        return pd.DataFrame(columns=ODDS_COLUMNS)

    if schedule is None or schedule.empty:
        return pd.DataFrame(columns=ODDS_COLUMNS)

    # Build game_id the same way the rest of the pipeline does:
    # SEASON_WEEK_AWAY_HOME, e.g. 2026_01_NE_SEA.
    game_id_by_teams = {
        (row['home_team'], row['away_team']): f"{int(season)}_{int(week):02d}_{row['away_team']}_{row['home_team']}"
        for _, row in schedule.iterrows()
    }
    target_full_names = {
        (TEAM_FULL_NAME.get(h), TEAM_FULL_NAME.get(a)): gid
        for (h, a), gid in game_id_by_teams.items()
    }

    try:
        events = _list_events()
    except requests.RequestException as e:
        print(f"[market_odds] failed to list events: {e}")
        return pd.DataFrame(columns=ODDS_COLUMNS)

    matched = [
        (ev, target_full_names[(ev.get('home_team'), ev.get('away_team'))])
        for ev in events
        if (ev.get('home_team'), ev.get('away_team')) in target_full_names
    ]
    print(f"[market_odds] {len(matched)}/{len(schedule)} week-{week} games matched to book events")

    all_rows: list[dict] = []
    remaining = None
    for ev, game_id in matched:
        if remaining is not None and remaining < ODDS_API_MIN_REMAINING:
            print(f"[market_odds] stopping early - {remaining} credits left "
                  f"(floor {ODDS_API_MIN_REMAINING}); {len(matched) - len(all_rows)} games not fetched")
            break
        try:
            event_json, remaining = _fetch_event_odds(ev['id'])
        except requests.RequestException as e:
            print(f"[market_odds] event {ev.get('id')} ({ev.get('away_team')} @ {ev.get('home_team')}) failed: {e}")
            continue
        all_rows.extend(_parse_event_odds(event_json, season, week, game_id))
        if remaining is not None:
            print(f"[market_odds]   ...{remaining} credits remaining")

    df = pd.DataFrame(all_rows, columns=ODDS_COLUMNS)
    print(f"[market_odds] fetched {len(df)} player/prop/book rows")

    if use_cache:
        # Write even when empty, so a week with zero matched games doesn't
        # get re-queried (for credits) on every subsequent run this week.
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False)
        print(f"[market_odds] wrote {cache_file.name}")

    return df


def _normalize_name(name: str) -> str:
    import re
    name = re.sub(r"\b(Jr|Sr|II|III|IV)\.?\b", "", str(name), flags=re.IGNORECASE)
    name = re.sub(r"[^a-z0-9 ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


def find_market_line(display_name: str, team: str, prop_type: str,
                      odds_df: pd.DataFrame | None) -> dict | None:
    """Best-available (BOOKMAKERS preference order) line for one player/prop.

    Returns {'line': float, 'book': str, 'over_odds': int, 'under_odds': int,
    'market_implied_prob': float} or None if unmatched / no data.
    """
    if odds_df is None or odds_df.empty or not isinstance(display_name, str) or not display_name:
        return None

    candidates = odds_df[odds_df['prop_type'] == prop_type]
    if candidates.empty:
        return None

    exact = candidates[candidates['player_name'].str.strip().str.lower() == display_name.strip().lower()]
    if exact.empty:
        target = _normalize_name(display_name)
        norm_names = candidates['player_name'].map(_normalize_name)
        exact = candidates[norm_names == target]
    if exact.empty:
        return None

    for book in BOOKMAKERS:
        row = exact[exact['book'] == book]
        if not row.empty:
            r = row.iloc[0]
            implied = _devig_over_prob(r['over_odds'], r['under_odds'])
            if implied is None:
                continue
            return {
                'line': float(r['line']), 'book': book,
                'over_odds': r['over_odds'], 'under_odds': r['under_odds'],
                'market_implied_prob': implied,
            }
    return None


def attach_market_odds(prediction: dict, market_info: dict | None) -> dict:
    """Adds market_line / market_book / market_implied_prob / market_edge /
    market_line_available to a prediction dict in place. The existing
    fixed-tier fields (line_value, prob_over, confidence, ...) are never
    touched - market_* is purely additive. No-ops when market_info is None.
    """
    if not market_info:
        prediction['market_line'] = None
        prediction['market_book'] = None
        prediction['market_implied_prob'] = None
        prediction['market_edge'] = None
        prediction['market_line_available'] = False
        return prediction

    prob_over = prediction.get('prob_over')
    prediction['market_line'] = market_info['line']
    prediction['market_book'] = market_info['book']
    prediction['market_implied_prob'] = market_info['market_implied_prob']
    prediction['market_edge'] = (
        (prob_over - market_info['market_implied_prob'])
        if isinstance(prob_over, (int, float)) else None
    )
    prediction['market_line_available'] = True
    return prediction


if __name__ == '__main__':
    import argparse
    try:
        from season_utils import upcoming_or_current_season
    except ImportError:  # run as `python player_props/market_odds.py`
        import sys as _sys
        _sys.path.append(str(Path(__file__).parent.parent))
        from season_utils import upcoming_or_current_season

    ap = argparse.ArgumentParser(description="Fetch DK/FanDuel player-prop odds for one week")
    ap.add_argument('--season', type=int, default=None)
    ap.add_argument('--week', type=int, required=True)
    ap.add_argument('--no-cache', action='store_true', help="ignore/overwrite an existing frozen file")
    args = ap.parse_args()

    season = args.season or upcoming_or_current_season()
    schedule_path = DATA_DIR / f'nfl_schedule_{season}.csv'
    if not schedule_path.exists():
        print(f"Schedule not found: {schedule_path}")
        raise SystemExit(1)
    sched = pd.read_csv(schedule_path)
    sched = sched[sched['week'] == args.week]

    result = fetch_market_odds(season, args.week, sched, use_cache=not args.no_cache)
    if not result.empty:
        print(result.head(20).to_string(index=False))
