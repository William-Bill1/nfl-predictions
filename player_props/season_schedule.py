"""Season schedule for the Player Props page's remaining-games banner.

The season comes from the displayed predictions, never from the calendar
year: an NFL season is named after the year it starts, so January and
February games belong to the previous year's season. Every prediction must
record the same whole-number season, and when `game_date` is present every
prediction's complete recorded date must be valid and fall inside that
season's window (1 August to the end of February, matching season_utils'
1 March rollover). Only `nfl_schedule_<season>.csv` is used. It must be
readable, have update_schedule.py's columns, and every row needs a date-only
date inside the same window, a whole-number week from 1 to 22, and two
distinct team abbreviations; a team listed twice on one date is a duplicate
game. Anything else
produces a message and no count; there is no fallback to another season.
Read-only: nothing is written and nothing is fetched.
"""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from player_props.game_dates import _DATE_ONLY, upcoming
from season_utils import FIRST_SEASON

# update_schedule.py writes exactly these columns.
SCHEDULE_COLUMNS = ("week", "date", "home_team", "away_team", "venue", "status")
MAX_WEEK = 22   # nflverse weeks: 18 regular-season + 4 playoff rounds
_WEEK = re.compile(r"\d{1,2}")
_TEAM = re.compile(r"[A-Z]{2,3}")
# A stored prop game date: the date alone, or that date at midnight with no
# offset or a UTC offset ('2026-10-26 00:00:00+00:00').
_RECORDED = re.compile(r"(\d{4}-\d{2}-\d{2})(?:[ T]00:00(?::00(?:\.0+)?)?(?:Z|[+-]00:?00)?)?")
COUNT_OMITTED = "the remaining-games count is omitted."
DATE_ONLY_NOTE = ("Counted from today's date in Toronto; today's games stay included "
                  "because the schedule records dates, not kickoff times.")


def season_window(season: int) -> tuple[date, date]:
    """[1 August of `season`, 1 March of the next year): preseason to Super Bowl."""
    return date(season, 8, 1), date(season + 1, 3, 1)


def _recorded_date(value) -> date | None:
    """A prediction's calendar game date as recorded, or None if the whole value
    isn't one. Stored prop game dates are UTC midnight ('2026-10-26
    00:00:00+00:00'), so the date part is taken as is; converting to Toronto
    time would move it to the previous day. Any other time or offset is not a
    recorded calendar date and is rejected."""
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, datetime):
        offset = value.utcoffset()
        if value.time() != time(0) or offset not in (None, timedelta(0)):
            return None
        return value.date()
    if isinstance(value, date):
        return value
    m = _RECORDED.fullmatch(str(value).strip())
    try:
        return date.fromisoformat(m.group(1)) if m else None
    except ValueError:
        return None


def prediction_season(predictions: pd.DataFrame | None) -> tuple[int | None, str | None]:
    """(season, None) when the predictions name one valid season, else (None, message)."""
    if predictions is None or predictions.empty:
        return None, f"No player prop predictions are loaded, so {COUNT_OMITTED}"
    if "season" not in predictions.columns:
        return None, f"The predictions have no season column, so {COUNT_OMITTED}"
    # Every displayed prediction must name the season; missing values are not dropped.
    raw = list(predictions["season"])
    if any(v is None or (not isinstance(v, str) and pd.isna(v)) for v in raw):
        return None, f"Some predictions have no season recorded, so {COUNT_OMITTED}"
    values = set()
    for v in dict.fromkeys(raw):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None, f"The predictions' season {v!r} is not a year, so {COUNT_OMITTED}"
        if not f.is_integer() or f < FIRST_SEASON or f > 2100:
            return None, f"The predictions' season {v!r} is not a valid season, so {COUNT_OMITTED}"
        values.add(int(f))
    if len(values) > 1:
        names = ", ".join(str(s) for s in sorted(values))
        return None, f"The predictions mix seasons ({names}), so {COUNT_OMITTED}"
    season = values.pop()
    if "game_date" in predictions.columns:
        start, end = season_window(season)
        days = [_recorded_date(v) for v in predictions["game_date"]]
        if any(d is None for d in days):
            return None, (f"Some predictions have no valid game date, so their {season} "
                          f"season can't be confirmed and {COUNT_OMITTED}")
        if any(not start <= d < end for d in days):
            return None, (f"Some prediction game dates fall outside the {season} season, "
                          f"so {COUNT_OMITTED}")
    return season, None


def load_season_schedule(season: int, data_dir: Path | str) -> tuple[pd.DataFrame | None, str | None]:
    """(schedule, None) for a valid `nfl_schedule_<season>.csv`, else (None, message)."""
    path = Path(data_dir) / f"nfl_schedule_{season}.csv"
    name = f"data_files/{path.name}"
    if not path.is_file():
        return None, f"The {season} schedule ({name}) is missing, so {COUNT_OMITTED}"
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None, f"The {season} schedule ({name}) can't be read, so {COUNT_OMITTED}"
    missing = [c for c in SCHEDULE_COLUMNS if c not in df.columns]
    if missing:
        return None, (f"The {season} schedule ({name}) is missing {', '.join(missing)}, "
                      f"so {COUNT_OMITTED}")
    if df.empty:
        return None, f"The {season} schedule ({name}) has no games, so {COUNT_OMITTED}"
    start, end = season_window(season)
    days = []
    for v in df["date"]:
        s = v.strip()
        try:
            days.append(date.fromisoformat(s) if _DATE_ONLY.fullmatch(s) else None)
        except ValueError:
            days.append(None)
    if any(d is None for d in days):
        return None, (f"The {season} schedule ({name}) has dates that aren't YYYY-MM-DD, "
                      f"so {COUNT_OMITTED}")
    if any(not start <= d < end for d in days):
        return None, (f"The {season} schedule ({name}) has dates outside the {season} "
                      f"season, so {COUNT_OMITTED}")
    weeks = [w.strip() for w in df["week"]]
    if any(not _WEEK.fullmatch(w) or not 1 <= int(w) <= MAX_WEEK for w in weeks):
        return None, (f"The {season} schedule ({name}) has rows without a whole-number week "
                      f"from 1 to {MAX_WEEK}, so {COUNT_OMITTED}")
    home = [t.strip() for t in df["home_team"]]
    away = [t.strip() for t in df["away_team"]]
    if any(not _TEAM.fullmatch(h) or not _TEAM.fullmatch(a) or h == a
           for h, a in zip(home, away)):
        return None, (f"The {season} schedule ({name}) has rows with a missing, malformed "
                      f"or identical home/away team, so {COUNT_OMITTED}")
    # A team plays at most once per date; a repeat is a duplicate game (in
    # either home/away order) and would inflate the count.
    seen = set()
    for d, h, a in zip(days, home, away):
        if (d, h) in seen or (d, a) in seen:
            return None, (f"The {season} schedule ({name}) lists a team more than once on "
                          f"{d.isoformat()} (duplicate games), so {COUNT_OMITTED}")
        seen.update({(d, h), (d, a)})
    return df, None


def remaining_games_text(season: int, count: int) -> str:
    """Banner text whose wording matches its scope: the rest of the season."""
    if count == 0:
        return f"📅 No remaining scheduled games in the {season} season. {DATE_ONLY_NOTE}"
    games = "game" if count == 1 else "games"
    return f"📅 **{count} remaining scheduled {games} this season** ({season}). {DATE_ONLY_NOTE}"


def remaining_games_banner(predictions: pd.DataFrame | None, data_dir: Path | str,
                           now: datetime | None = None) -> tuple[str, str]:
    """('info', text) with the count, or ('warning', message) with no count."""
    season, problem = prediction_season(predictions)
    if problem:
        return "warning", problem
    schedule, problem = load_season_schedule(season, data_dir)
    if problem:
        return "warning", problem
    return "info", remaining_games_text(season, len(upcoming(schedule, "date", now)))
