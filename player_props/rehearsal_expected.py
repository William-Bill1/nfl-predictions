"""
Independent expected-row set for the Player Props rehearsal (completeness).

From the inputs alone, before any prediction file is read, this derives which
rows production should write for a target week, identified by
(player_id, team, opponent, prop_type, line_value). The rehearsal then compares
the latest feed and the frozen snapshot with it separately, so a missing or
extra prediction is caught even when the two files agree.

Production's inclusion rules (predict.generate_predictions /
predict_props_for_game) are RESTATED here, not called. Production's
candidate_players, get_player_performance_tier and get_player_position are not
used, so a change or slip in them shows up as a difference. Shared pieces are
the definitions both sides must agree on: the constants POSITION_MAP,
TIER_RULES and PROP_STAT; the Stage 2-validated feature tables; the feature
builder and the scorer (features.inference_rows / model_matrix,
model_bundle.score_probabilities), which compute the probability the
confidence band applies to; and the restated post-processing passed in by the
rehearsal.

Inclusion rules, in order (with the injury, weather and market inputs
disabled, as in the rehearsal):
 1. Games: the target week's rows of nfl_schedule_<season>.csv. Each must
    resolve to exactly one game in the authoritative schedule
    (nfl_games_historical.csv) for that season and week, with its own date a
    valid YYYY-MM-DD equal to that game's calendar date, and the two must list
    the same games. A missing, malformed or different date fails; the
    authoritative date is never substituted.
 2. Candidates per team: players appearing in any feature table in one of the
    team's 3 most recent games, whose latest appearance in any table (by date,
    then game_id) was for this team. Otherwise: excluded,
    ``latest_appearance_elsewhere``.
 3. Reference: exactly one players.csv record for the ID. None: excluded,
    ``not_in_player_reference`` (rookies or anyone else missing from the
    reference; a coverage limit, not a completeness check). More than one:
    invalid input, the check fails.
 4. Props (policy position-group-v1): POSITION_MAP[position_group]; the
    position label is not used to select props. The group must be present,
    unpadded and consistent with the label (QB/RB/WR/TE their own group, FB
    group RB, any other label not an eligible group); otherwise the input is
    invalid and the check fails. An ineligible group: excluded,
    ``position_not_eligible``.
 5. History: an earlier appearance in that prop's stat table. None: excluded,
    ``no_history``.
 6. Tier: the mean of the player's last 8 appearances in that table (every
    team), or the lowest tier with fewer than 3.
 7. Model: the first restored bundle model for (prop, tier). None (for
    example, a model skipped at training): excluded, ``no_model``.
 8. Confidence: the final probability (after the permitted clamp) must be at
    least 0.55 or at most 0.45. Otherwise: excluded, ``confidence_band``.

Invalid or ambiguous input raises ExpectedSetError instead of shrinking the
set: a game that doesn't resolve, schedules that disagree, a missing feature
table, a tie deciding a team's recent games or a player's latest team, a
repeated reference ID, an unusable eligibility group (missing, padded or
inconsistent with the position label), or the roster filter switched on (its
rule isn't restated here).

Completeness is relative to these inputs and rules. It says nothing about
players absent from the inputs: no NFL roster, depth chart or rookie coverage
is claimed.
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd

from player_props.features import PROP_STAT, TARGET_COLUMNS, inference_rows, model_matrix
from player_props.model_bundle import score_probabilities
from player_props.predict import POSITION_MAP, TIER_RULES

RECENT_GAMES = 3            # predict.candidate_players(n_games=3)
TIER_WINDOW = 8             # the last 8 appearances set the tier
TIER_MIN_APPEARANCES = 3    # fewer: the lowest tier
EXAMPLES = 5

# Restated eligibility policy (production: predict.ELIGIBILITY_POLICY). The
# rehearsal refuses to verify when the two versions differ.
RESTATED_POLICY = 'position-group-v1'
# The group each skill label must carry, restated (not imported).
SKILL_LABEL_GROUPS = {'QB': 'QB', 'RB': 'RB', 'FB': 'RB', 'WR': 'WR', 'TE': 'TE'}


def restated_group(position, group):
    """(group, None) or (None, problem), restating the policy's checks on a
    reference record as written (no trimming, no fall back to position)."""
    if not isinstance(group, str) or group.strip() == '':
        return None, "no position_group"
    if group.strip() != group:
        return None, f"position_group {group!r} is padded"
    if not isinstance(position, str) or position.strip() == '':
        return None, "no position"
    if position.strip() != position:
        return None, f"position {position!r} is padded"
    want = SKILL_LABEL_GROUPS.get(position)
    if want is not None and group != want:
        return None, f"position {position!r} has position_group {group!r}, not {want!r}"
    if want is None and group in POSITION_MAP:
        return None, f"position {position!r} has the eligible position_group {group!r}"
    return group, None

Identity = Tuple[str, str, str, str, float]


class ExpectedSetError(RuntimeError):
    """The expected set can't be derived from these inputs (invalid or ambiguous)."""

    def __init__(self, problems: List[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def identity(player_id, team, opponent, prop_type, line_value) -> Identity:
    return (str(player_id), str(team), str(opponent), str(prop_type), float(line_value))


@dataclass
class ExpectedSet:
    rows: Dict[Identity, dict] = field(default_factory=dict)
    exclusions: Counter = field(default_factory=Counter)
    games: List[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {'expected': len(self.rows), 'games': len(self.games),
                'exclusions': dict(sorted(self.exclusions.items()))}


def _day(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors='coerce')


def _calendar_date(value):
    """A strict YYYY-MM-DD calendar date, or None for a missing or malformed one."""
    if not isinstance(value, str) or len(value) != 10:
        return None
    try:
        return datetime.strptime(value, '%Y-%m-%d').date()
    except ValueError:
        return None


def _week_games(data_dir: Path, games: pd.DataFrame, season: int, week: int) -> List[dict]:
    path = Path(data_dir) / f'nfl_schedule_{season}.csv'
    if not path.exists():
        raise ExpectedSetError([f"{path.name} is missing"])
    sched = pd.read_csv(path, dtype={'date': str})
    sched = sched[pd.to_numeric(sched['week'], errors='coerce') == week]
    if sched.empty:
        raise ExpectedSetError([f"{path.name} lists no games for week {week}"])
    pairs = list(zip(sched['home_team'], sched['away_team']))
    problems = []
    if len(set(pairs)) != len(pairs):
        problems.append(f"{path.name} lists a week-{week} game twice")
    teams = [t for p in pairs for t in p]
    if len(set(teams)) != len(teams):
        problems.append(f"{path.name} puts a team in two week-{week} games")
    official = games[(games['season'] == season) & (games['week'] == week)]
    official_pairs = set(zip(official['home_team'], official['away_team']))
    if set(pairs) != official_pairs:
        problems.append(f"{path.name} and the schedule disagree for {season} week {week}: "
                        f"only in {path.name} {sorted(set(pairs) - official_pairs)[:EXAMPLES]}, "
                        f"only in the schedule {sorted(official_pairs - set(pairs))[:EXAMPLES]}")
    if 'date' not in sched.columns:
        problems.append(f"{path.name} has no date column")
    out = []
    for row in sched.to_dict('records'):
        home, away = row['home_team'], row['away_team']
        g = official[(official['home_team'] == home) & (official['away_team'] == away)]
        if len(g) != 1:
            problems.append(f"{away} @ {home} matches {len(g)} schedule games for {season} week {week}")
            continue
        g = g.iloc[0]
        # The row's own date must be a valid YYYY-MM-DD equal to the game's
        # calendar date; the schedule's date is never substituted for it.
        listed = _calendar_date(row.get('date'))
        if listed is None:
            problems.append(f"{path.name}: {away} @ {home} has a missing or malformed date {row.get('date')!r}")
            continue
        if listed != _calendar_date(str(g['gameday'])):
            problems.append(f"{path.name}: {away} @ {home} is dated {listed}, but the schedule has "
                            f"{g['gameday']!r} for {g['game_id']}")
            continue
        out.append({'game_id': g['game_id'], 'home': home, 'away': away, 'gameday': str(g['gameday'])})
    if problems:
        raise ExpectedSetError(problems)
    return out


def _team_candidates(tables: Dict[str, pd.DataFrame], team: str, exclusions: Counter) -> Dict[str, str]:
    """{player_id: latest player_name} for the team, restating the candidate rule."""
    ids = set()
    for name, t in tables.items():
        rows = t[t['team'] == team]
        g = rows[['game_id', 'game_date']].drop_duplicates()
        if g['game_id'].duplicated().any():
            raise ExpectedSetError([f"{name}: a {team} game has more than one date"])
        g = g.assign(_d=_day(g['game_date'])).sort_values('_d', ascending=False)
        if g['_d'].isna().any() or g['_d'].duplicated().any():
            # Two games on one date (or no date) make "the most recent games" ambiguous.
            raise ExpectedSetError([f"{name}: {team}'s {RECENT_GAMES} most recent games are ambiguous "
                                    f"(a missing date, or two games on one date)"])
        recent = set(g['game_id'].head(RECENT_GAMES))
        ids |= set(rows.loc[rows['game_id'].isin(recent), 'player_id'])
    every = pd.concat([t[['player_id', 'player_name', 'team', 'game_id', 'game_date']] for t in tables.values()])
    every = every[every['player_id'].isin(ids)].assign(_d=lambda d: _day(d['game_date']))
    out = {}
    for pid, rows in every.groupby('player_id'):
        last = rows[rows['_d'] == rows['_d'].max()]
        last = last[last['game_id'] == last['game_id'].max()]
        teams, names = set(last['team']), set(last['player_name'])
        if len(teams) != 1 or len(names) != 1:
            raise ExpectedSetError([f"player {pid}'s latest appearance is ambiguous: teams {sorted(teams)}, "
                                    f"names {sorted(map(str, names))}"])
        if teams.pop() == team:
            out[str(pid)] = names.pop()
        else:
            exclusions['latest_appearance_elsewhere'] += 1
    return out


def _tier(history: pd.DataFrame, prop_type: str) -> str:
    stat_col, cuts, lowest = TIER_RULES[prop_type]
    if len(history) < TIER_MIN_APPEARANCES:
        return lowest
    avg = history.head(TIER_WINDOW)[stat_col].mean()
    for cut, tier in cuts:
        if avg >= cut:
            return tier
    return lowest


def expected_rows(data_dir: Path, tables: Dict[str, pd.DataFrame], models: Dict[str, dict],
                  games: pd.DataFrame, season: int, week: int, post_process: Callable,
                  players: pd.DataFrame) -> ExpectedSet:
    """The expected rows for (season, week), from the inputs only."""
    if os.getenv('PROP_ROSTER_FILTER', '').lower() in ('1', 'true', 'yes'):
        raise ExpectedSetError(["PROP_ROSTER_FILTER is on; the roster rule isn't restated, so "
                                "completeness can't be derived"])
    needed = sorted({PROP_STAT[p][0] for props in POSITION_MAP.values() for p in props})
    missing = [s for s in needed if s not in tables]
    if missing:
        raise ExpectedSetError([f"feature table(s) {missing} missing"])
    if 'position_group' not in players.columns:
        raise ExpectedSetError(["players reference has no position_group; eligibility (policy "
                                f"{RESTATED_POLICY}) can't be derived"])
    result = ExpectedSet(games=_week_games(data_dir, games, season, week))

    reference = players.assign(gsis_id=players['gsis_id'].astype(str))
    counts = reference['gsis_id'].value_counts()
    reference = reference.drop_duplicates('gsis_id', keep=False).set_index('gsis_id')

    # Per stat table, the (player, game) rows needing a feature row this week.
    wanted: Dict[str, List[dict]] = {s: [] for s in needed}
    plan = []
    for game in result.games:
        for team, opponent in ((game['home'], game['away']), (game['away'], game['home'])):
            for pid, short in _team_candidates(tables, team, result.exclusions).items():
                if counts.get(pid, 0) > 1:
                    raise ExpectedSetError([f"player_id {pid} has {counts[pid]} player-reference records"])
                if pid not in reference.index:
                    result.exclusions['not_in_player_reference'] += 1
                    continue
                group, problem = restated_group(reference.at[pid, 'position'],
                                                reference.at[pid, 'position_group'])
                if problem:
                    raise ExpectedSetError([f"player_id {pid}: {problem}"])
                if group not in POSITION_MAP:
                    result.exclusions['position_not_eligible'] += 1
                    continue
                for prop_type in POSITION_MAP[group]:
                    stat_type = PROP_STAT[prop_type][0]
                    t = tables[stat_type]
                    hist = t[t['player_id'] == pid]
                    if hist.empty:
                        result.exclusions['no_history'] += 1
                        continue
                    hist = hist.assign(_d=_day(hist['game_date'])).sort_values(['_d', 'game_id'], ascending=False)
                    tier = _tier(hist, prop_type)
                    model = next((n for n, i in models.items()
                                  if i['prop_type'] == prop_type and i['line_type'] == tier), None)
                    if model is None:
                        result.exclusions['no_model'] += 1
                        continue
                    target = dict(player_id=pid, player_name=short, game_id=game['game_id'], season=season,
                                  week=week, team=team, opponent=opponent, game_date=game['gameday'])
                    if target not in wanted[stat_type]:
                        wanted[stat_type].append(target)
                    plan.append((pid, team, opponent, prop_type, tier, model, game, stat_type))

    built = {s: inference_rows(tables[s], pd.DataFrame(w, columns=TARGET_COLUMNS), s, games)
             .set_index('player_id') for s, w in wanted.items() if w}
    for pid, team, opponent, prop_type, tier, model, game, stat_type in plan:
        info = models[model]
        row = built[stat_type].loc[[pid]]
        X = model_matrix(row, prop_type)
        raw = float(score_probabilities(info['model'], X, prop_type)[0])
        recent = X.iloc[0].to_dict().get(f"{info['stat_col']}_L5", 0)
        if not post_process(raw, tier, recent, info['line_value'])['included']:
            result.exclusions['confidence_band'] += 1
            continue
        key = identity(pid, team, opponent, prop_type, info['line_value'])
        if key in result.rows:
            raise ExpectedSetError([f"expected identity {key} derived twice"])
        result.rows[key] = {'is_home': team == game['home'], 'game_id': game['game_id'],
                            'model_name': model, 'tier': tier}
    return result


FAILURE_COUNTS = ('missing', 'unexpected', 'duplicates', 'wrong_side')


def completeness_status(per_file: Dict[str, dict]) -> str:
    """'complete' only when every file was checked and has no missing,
    unexpected, duplicate or wrong-side identity; 'not checked' when any file
    couldn't be checked; otherwise 'incomplete'."""
    if not per_file or any(not c.get('checked') for c in per_file.values()):
        return 'not checked'
    if all(c.get(k) == 0 for c in per_file.values() for k in FAILURE_COUNTS):
        return 'complete'
    return 'incomplete'


def _fmt(keys) -> str:
    keys = sorted(keys, key=str)
    more = f" (+{len(keys) - EXAMPLES} more)" if len(keys) > EXAMPLES else ""
    return ", ".join("{} {} vs {} {} {}".format(*k) for k in keys[:EXAMPLES]) + more


def completeness_problems(frame: pd.DataFrame, label: str, expected: ExpectedSet) -> Tuple[List[str], dict]:
    """One predictions file against the expected set: missing, unexpected and
    duplicate identities, and the home/away side of expected rows."""
    if 'player_id' not in frame.columns:
        return ([f"{label}: no player_id column, so completeness can't be checked"],
                {'checked': False, 'observed': int(len(frame))})
    keys = [identity(r['player_id'], r['team'], r['opponent'], r['prop_type'], r['line_value'])
            for r in frame.to_dict('records')]
    observed = Counter(keys)
    want = set(expected.rows)
    missing = want - set(observed)
    unexpected = set(observed) - want
    duplicates = [k for k, c in observed.items() if c > 1]
    side = [k for k, r in zip(keys, frame.to_dict('records'))
            if k in expected.rows and bool(r['is_home']) != expected.rows[k]['is_home']]
    problems = []
    if missing:
        problems.append(f"{label}: {len(missing)} expected prediction(s) missing: {_fmt(missing)}")
    if unexpected:
        problems.append(f"{label}: {len(unexpected)} unexpected prediction(s): {_fmt(unexpected)}")
    if duplicates:
        problems.append(f"{label}: {len(duplicates)} identit(ies) listed more than once: {_fmt(duplicates)}")
    if side:
        problems.append(f"{label}: home/away side differs from the schedule for {_fmt(set(side))}")
    counts = {'checked': True, 'observed': len(keys), 'distinct': len(observed), 'missing': len(missing),
              'unexpected': len(unexpected), 'duplicates': len(duplicates), 'wrong_side': len(set(side))}
    return problems, counts
