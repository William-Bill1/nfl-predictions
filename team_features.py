"""Leak-free team aggregate features for the game models.

One implementation, used by both the production pipeline
(`nfl-gather-data.py`) and the rolling backtest
(`scripts/rolling_spread_backtest.py`), so the two cannot drift apart.

Availability rule (conservative, no completion timestamps needed)
----------------------------------------------------------------
A game's features for week W use only games that

1. are COMPLETED - both final scores present in the schedule, and
2. belong to a strictly EARLIER week key (season * 100 + week).

Nothing from week W itself is used - not even a Thursday result for a
Sunday game - because the schedule has no reliable "result posted at"
timestamp, and a kickoff time alone doesn't prove a game had finished
when a prediction was made. Excluding the whole current week is the
conservative cutoff. A game's own outcome and all later outcomes therefore
never touch its features. Results depend only on game content, not on row
order; "last N games" are ordered by (week key, kickoff, game_id).

Definitions (unchanged from production, except for the rule above)
-------------------------------------------------------------------
* `homeTeam<Stat>` / `awayTeam<Stat>` averages use the team's earlier games
  in the SAME role only (home games for home*, away games for away*), as
  the production loops always did. `PointDiffTrend` is the exception: it
  uses the team's earlier games in either role.
* A game counts toward a stat only if that stat is defined for it (e.g.
  `FavoredPct` / `SpreadCoveredPct` need a posted line, the over/under hit
  rates need a total line).
* `SpreadCoveredPct` = rate at which the game's FAVORITE covered (the
  production `spreadCovered` label); a 0 line counts as not covered.
* `PointDiffTrend` = mean of the differences of the team's last 3 point
  differentials = (last - first) / 2; 0 with fewer than 3 earlier games.

Cold start
----------
When a team has no qualifying earlier game the feature is
`COLD_START_DEFAULT` (0.0) - the value production has always used
(`calc_rolling_stat` returned 0, and the pipeline `fillna(0)`s).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLD_START_DEFAULT = 0.0
LAST_N = 3

# Per-side stat -> how to get its value for a game, given the side ("home"/"away").
# A callable takes (outcomes, side) and returns a Series aligned with the games.
_SIDE_STATS = {
    "WinPct":          lambda o, s: o[f"{s}Win"],
    "CloseGamePct":    lambda o, s: o["isCloseGame"],
    "BlowoutPct":      lambda o, s: o["isBlowout"],
    "AvgScore":        lambda o, s: o[f"{s}_score"],
    "AvgScoreAllowed": lambda o, s: o["away_score" if s == "home" else "home_score"],
    "AvgPointDiff":    lambda o, s: o["pointDiff"],
    "AvgTotalScore":   lambda o, s: o["totalScore"],
    "AvgPointSpread":  lambda o, s: o["spread_line"],
    "AvgTotal":        lambda o, s: o["total"],
    "FavoredPct":      lambda o, s: o[f"{s}Favored"],
    "SpreadCoveredPct": lambda o, s: o["spreadCovered"],
    "OverHitPct":      lambda o, s: o["overHit"],
    "UnderHitPct":     lambda o, s: o["underHit"],
    "TotalHitPct":     lambda o, s: o["totalHit"],
}
_LAST3_STATS = {
    "Last3WinPct":          "WinPct",
    "Last3AvgScore":        "AvgScore",
    "Last3AvgScoreAllowed": "AvgScoreAllowed",
}

# Production column order (nfl-gather-data.py has always appended them in this order).
TEAM_FEATURES = [
    f"{side}Team{stat}"
    for stat in ("WinPct", "CloseGamePct", "BlowoutPct", "AvgScore", "AvgScoreAllowed",
                 "AvgPointDiff", "AvgTotalScore", "GamesPlayed", "AvgPointSpread", "AvgTotal",
                 "FavoredPct", "SpreadCoveredPct", "OverHitPct", "UnderHitPct", "TotalHitPct",
                 "Last3WinPct", "Last3AvgScore", "Last3AvgScoreAllowed", "PointDiffTrend")
    for side in ("home", "away")
]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def game_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Per-game outcome values, NaN wherever a game isn't completed (or the stat is undefined).

    Never writes to `df`; production keeps its own label columns.
    """
    hs, as_ = _num(df, "home_score"), _num(df, "away_score")
    sl, tl, total = _num(df, "spread_line"), _num(df, "total_line"), _num(df, "total")
    done = hs.notna() & as_.notna()
    margin = hs - as_
    has_sl, has_tl = done & sl.notna(), done & tl.notna()

    def when(mask, values):
        return pd.Series(np.asarray(values, dtype=float), index=df.index).where(mask)

    fav_margin = np.where(sl > 0, margin, -margin)
    return pd.DataFrame({
        "completed": done,
        "key": _num(df, "season") * 100 + _num(df, "week"),
        "home_score": hs.where(done), "away_score": as_.where(done),
        "homeWin": when(done, margin > 0), "awayWin": when(done, margin < 0),
        "isCloseGame": when(done, margin.abs() <= 3), "isBlowout": when(done, margin.abs() >= 20),
        "pointDiff": margin.abs().where(done), "totalScore": (hs + as_).where(done),
        "total": total.where(done),
        "spread_line": sl.where(has_sl),
        "homeFavored": when(has_sl, sl > 0), "awayFavored": when(has_sl, sl < 0),
        "spreadCovered": when(has_sl, (sl != 0) & (fav_margin > sl.abs())),
        "overHit": when(has_tl, hs + as_ > tl), "underHit": when(has_tl, hs + as_ < tl),
        "totalHit": when(has_tl, hs + as_ == tl),
    }, index=df.index)


def _order(df: pd.DataFrame) -> pd.Series:
    """Kickoff timestamp, used only to order games inside a week key (NaT sorts first)."""
    day = df["gameday"].astype(str).str[:10] if "gameday" in df else pd.Series("", index=df.index)
    time = df["gametime"].fillna("00:00").astype(str) if "gametime" in df else "00:00"
    return pd.to_datetime(day + " " + time, format="%Y-%m-%d %H:%M", errors="coerce")


def _prior_window(game_team, game_key, game_tiebreak, game_ids, values, query_team, query_key,
                  n: int | None = None, reducer: str = "mean") -> np.ndarray:
    """For each query (team, key): aggregate `values` over that team's eligible
    games with key < query key - all of them, or only the last `n`.

    Eligible games are those where `values` is not NaN (so NaN already encodes
    "not completed" / "stat undefined"). reducer: "mean", "count" or "trend".
    """
    ok = values.notna() & game_key.notna()
    games = pd.DataFrame({"team": game_team[ok], "key": game_key[ok],
                          "tb": game_tiebreak[ok], "gid": game_ids[ok].astype(str),
                          "v": values[ok].astype(float)})
    games = games.sort_values(["team", "key", "tb", "gid"], kind="mergesort", na_position="first")
    out = np.full(len(query_team), COLD_START_DEFAULT, dtype=float)
    qt, qk = query_team.to_numpy(), query_key.to_numpy(dtype=float)
    for team, g in games.groupby("team", sort=False):
        q = np.flatnonzero(qt == team)
        if not len(q):
            continue
        keys, vals = g["key"].to_numpy(dtype=float), g["v"].to_numpy(dtype=float)
        k = np.searchsorted(keys, qk[q], side="left")      # strictly earlier keys only
        if reducer == "count":
            out[q] = k
            continue
        if reducer == "trend":
            enough = k >= n
            idx = q[enough]
            last, first = vals[k[enough] - 1], vals[k[enough] - n]
            out[idx] = (last - first) / (n - 1)
            continue
        cs = np.concatenate([[0.0], np.cumsum(vals)])
        lo = np.zeros_like(k) if n is None else np.maximum(k - n, 0)
        cnt = k - lo
        has = cnt > 0
        out[q[has]] = (cs[k[has]] - cs[lo[has]]) / cnt[has]
    return out


def _feature(df: pd.DataFrame, outcomes: pd.DataFrame, tiebreak: pd.Series, name: str) -> np.ndarray:
    side = "home" if name.startswith("homeTeam") else "away" if name.startswith("awayTeam") else None
    if side is None:
        raise ValueError(f"{name!r} is not a team aggregate feature")
    stat = name[len(f"{side}Team"):]
    team, key = df[f"{side}_team"], outcomes["key"]
    gid = df["game_id"] if "game_id" in df else pd.Series(df.index.astype(str), index=df.index)
    if stat == "GamesPlayed":
        return _prior_window(team, key, tiebreak, gid, outcomes["homeWin"], team, key, reducer="count")
    if stat == "PointDiffTrend":
        # Either role: stack the home and away appearances of every game.
        diff = outcomes["home_score"] - outcomes["away_score"]
        game_team = pd.concat([df["home_team"], df["away_team"]], ignore_index=True)
        game_key = pd.concat([key, key], ignore_index=True)
        game_tb = pd.concat([tiebreak, tiebreak], ignore_index=True)
        game_ids = pd.concat([gid, gid], ignore_index=True)
        vals = pd.concat([diff, -diff], ignore_index=True)
        return _prior_window(game_team, game_key, game_tb, game_ids, vals, team.reset_index(drop=True),
                             key.reset_index(drop=True), n=LAST_N, reducer="trend")
    if stat in _LAST3_STATS:
        vals = _SIDE_STATS[_LAST3_STATS[stat]](outcomes, side)
        return _prior_window(team, key, tiebreak, gid, vals, team, key, n=LAST_N)
    if stat in _SIDE_STATS:
        return _prior_window(team, key, tiebreak, gid, _SIDE_STATS[stat](outcomes, side), team, key)
    raise ValueError(f"{name!r} has no leak-free definition")


def compute_team_features(df: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    """Leak-free team aggregates for every row of a schedule frame.

    `df` needs season, week, home_team, away_team, home_score, away_score and,
    for the line/total based stats, spread_line / total_line / total. Scores
    must be NaN (not 0) for games that haven't been played. Returns a frame
    indexed like `df` with one column per requested name (default: all of
    TEAM_FEATURES, in production column order).
    """
    names = list(TEAM_FEATURES if names is None else names)
    outcomes = game_outcomes(df)
    tiebreak = _order(df)
    return pd.DataFrame({name: _feature(df, outcomes, tiebreak, name) for name in names}, index=df.index)
