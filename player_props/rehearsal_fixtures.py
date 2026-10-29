"""
Deterministic synthetic league for the Stage 5B rehearsal (player_props/rehearsal.py).

Eight teams play a round robin of WEEKS weeks per season. Every game has
nflverse-style plays (passes to three receivers and a back, runs, touchdowns,
interceptions) and an END GAME row whose score matches the schedule's recorded
result, so the production completion and coverage checks run unchanged. Sizes
are chosen so the production training minimums (100 rows, 40 held-out rows,
both classes at 10% or more) are met for most prop lines without weakening them.

Nothing here touches the network or the repository's data_files/.
"""
from __future__ import annotations

import gzip
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TEAMS = ['KC', 'BUF', 'DEN', 'LV', 'NE', 'NYJ', 'MIA', 'BAL']
WEEKS = 8
ROLES = ['QB', 'RB', 'WR1', 'WR2', 'TE']
POSITION = {'QB': 'QB', 'RB': 'RB', 'WR1': 'WR', 'WR2': 'WR', 'TE': 'TE'}
# nflverse position_group for each label (a fullback's group is RB).
POSITION_GROUP = {'QB': 'QB', 'RB': 'RB', 'FB': 'RB', 'WR': 'WR', 'TE': 'TE'}
TARGET_SHARE = {'WR1': 0.35, 'WR2': 0.25, 'TE': 0.2, 'RB': 0.2}


def player_id(team: str, role: str) -> str:
    return f"{team}-{role}"


def round_robin(week: int) -> List[Tuple[str, str]]:
    """(home, away) pairs for one week: the circle method, no team twice."""
    n = len(TEAMS)
    r = (week - 1) % (n - 1)
    rot = [TEAMS[0]] + [TEAMS[1 + (i + r) % (n - 1)] for i in range(n - 1)]
    pairs = [(rot[i], rot[n - 1 - i]) for i in range(n // 2)]
    return [(h, a) if (week + i) % 2 else (a, h) for i, (h, a) in enumerate(pairs)]


def game_date(season: int, week: int) -> str:
    return (date(season, 9, 7) + timedelta(days=7 * (week - 1))).isoformat()


def game_id(season: int, week: int, home: str, away: str) -> str:
    return f"{season}_{week:02d}_{away}_{home}"


def _team_plays(rng, base: dict, team: str, start: int) -> Tuple[List[dict], int]:
    plays, play = [], start
    attempts = int(rng.integers(26, 40))
    receivers = list(TARGET_SHARE)
    weights = np.array([TARGET_SHARE[r] for r in receivers])
    for _ in range(attempts):
        play += 1
        target = receivers[int(rng.choice(len(receivers), p=weights))]
        complete = rng.random() < 0.64
        yards = float(max(0, round(rng.normal(11.5, 8)))) if complete else 0.0
        td = int(complete and rng.random() < 0.06)
        pick = int(not complete and rng.random() < 0.04)
        plays.append({**base, 'play_id': play, 'posteam': team,
                      'passer_player_id': player_id(team, 'QB'), 'passer_player_name': f"{team}.QB",
                      'receiver_player_id': player_id(team, target), 'receiver_player_name': f"{team}.{target}",
                      'passing_yards': yards, 'receiving_yards': yards, 'complete_pass': int(complete),
                      'pass_attempt': 1, 'pass_touchdown': td, 'interception': pick})
    for _ in range(int(rng.integers(10, 24))):
        play += 1
        plays.append({**base, 'play_id': play, 'posteam': team,
                      'rusher_player_id': player_id(team, 'RB'), 'rusher_player_name': f"{team}.RB",
                      'rushing_yards': float(round(rng.normal(4.3, 3.5))), 'rush_attempt': 1,
                      'rush_touchdown': int(rng.random() < 0.035)})
    return plays, play


EMPTY = dict(passer_player_id=None, passer_player_name=None, rusher_player_id=None,
             rusher_player_name=None, receiver_player_id=None, receiver_player_name=None,
             passing_yards=0.0, rushing_yards=0.0, receiving_yards=0.0, pass_touchdown=0,
             rush_touchdown=0, complete_pass=0, pass_attempt=0, interception=0, rush_attempt=0,
             total_home_score=None, total_away_score=None, desc='play', season_type='REG')


def league(seasons: List[int], played_weeks: Dict[int, int], seed: int = 7
           ) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    """({season: plays}, schedule).

    ``played_weeks[season]`` is how many weeks have been played (default all
    WEEKS); later weeks are scheduled with no score and no plays. The schedule
    is in nfl_games_historical.csv form (scores, result, total)."""
    plays_by_season: Dict[int, pd.DataFrame] = {}
    sched = []
    for season in seasons:
        rng = np.random.default_rng(seed * 10007 + season)
        rows = []
        for week in range(1, WEEKS + 1):
            day = game_date(season, week)
            played = week <= played_weeks.get(season, WEEKS)
            for home, away in round_robin(week):
                gid = game_id(season, week, home, away)
                h, a = (int(rng.integers(10, 38)), int(rng.integers(10, 38))) if played else (None, None)
                if played and h == a:
                    h += 3
                sched.append(dict(game_id=gid, season=season, game_type='REG', week=week, gameday=day,
                                  home_team=home, away_team=away,
                                  home_score=float(h) if played else None,
                                  away_score=float(a) if played else None,
                                  result=float(h - a) if played else None,
                                  total=float(h + a) if played else None))
                if not played:
                    continue
                base = {**EMPTY, 'game_id': gid, 'season': season, 'week': week, 'game_date': day}
                play = 0
                for team, opp in ((home, away), (away, home)):
                    team_plays, play = _team_plays(rng, {**base, 'defteam': opp}, team, play)
                    rows += team_plays
                rows.append({**base, 'play_id': play + 1, 'posteam': None, 'defteam': None,
                             'desc': 'END GAME', 'total_home_score': h, 'total_away_score': a})
        plays_by_season[season] = pd.DataFrame(rows)
    return plays_by_season, pd.DataFrame(sched)


def season_download(plays: pd.DataFrame) -> bytes:
    """One season as nflverse publishes it: comma-separated, gzipped."""
    return gzip.compress(plays.to_csv(index=False).encode('utf-8'), mtime=0)


def players() -> pd.DataFrame:
    return pd.DataFrame([dict(gsis_id=player_id(t, r), short_name=f"{t}.{r}",
                              display_name=f"{t} {r}", last_name=r, position=POSITION[r],
                              position_group=POSITION_GROUP[POSITION[r]])
                         for t in TEAMS for r in ROLES])


def prediction_schedule(schedule: pd.DataFrame, season: int) -> pd.DataFrame:
    """nfl_schedule_<season>.csv form used by predict.load_schedule."""
    s = schedule[schedule['season'] == season]
    return pd.DataFrame({'week': s['week'], 'date': s['gameday'], 'home_team': s['home_team'],
                         'away_team': s['away_team'], 'venue': 'Synthetic Field', 'status': 'REG'})


def fetcher(plays_by_season: Dict[int, pd.DataFrame], drop_seasons=(),
            drop_games: Optional[set] = None):
    """A download function for source_refresh: synthetic bytes, never the network."""
    def fetch(season: int) -> bytes:
        if season in drop_seasons or season not in plays_by_season:
            raise OSError(f"404: no synthetic download for season {season}")
        df = plays_by_season[season]
        if drop_games:
            df = df[~df['game_id'].isin(drop_games)]
        return season_download(df)
    return fetch
