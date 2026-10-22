"""Schedule dates for the Player Props page.

`nfl_schedule_<season>.csv` `date` is nflverse's `gameday` written as
YYYY-MM-DD: a calendar game date with no kickoff time or zone. Reading it as
UTC midnight and comparing with the current instant drops a game at 8 PM
Toronto (EDT) the evening before it is played. Here a calendar date stays
that date and is compared with today's date in America/Toronto. Only a
timestamp with an explicit offset (e.g. an Odds API `commence_time` ending in
Z) is a known kickoff instant; a time without a zone is never assumed UTC.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

TORONTO = ZoneInfo("America/Toronto")
_DATE_ONLY = re.compile(r"\d{4}-\d{2}-\d{2}")


def _parse(value) -> tuple[date | None, datetime | None]:
    """(calendar game day, kickoff instant in Toronto or None)."""
    try:
        if value is None or pd.isna(value):
            return None, None
    except (TypeError, ValueError):
        pass
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        return value, None
    else:
        s = str(value).strip()
        try:
            if _DATE_ONLY.fullmatch(s):
                return date.fromisoformat(s), None
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None, None
    if dt.tzinfo is not None and dt.utcoffset() is not None:
        local = dt.astimezone(TORONTO)
        return local.date(), local
    # Naive: a date (midnight) or a time with no zone - keep the date as recorded.
    return dt.date(), None


def game_day(value) -> date | None:
    """Calendar game date: a date as recorded, or an offset-aware timestamp's
    America/Toronto date. None for missing or invalid values."""
    return _parse(value)[0]


def kickoff(value) -> datetime | None:
    """Kickoff instant in America/Toronto, only for offset-aware timestamps."""
    return _parse(value)[1]


def upcoming(schedule_df: pd.DataFrame, column: str, now: datetime | None = None) -> pd.DataFrame:
    """Rows still to be played, sorted by game day (then kickoff).

    A row with a known kickoff is upcoming until that instant. A calendar date
    is upcoming through the whole of that day in Toronto, since its kickoff
    time is unknown. Missing or invalid dates are excluded."""
    now = (now or datetime.now(TORONTO)).astimezone(TORONTO)
    rows = [(i, d, k) for i, (d, k) in enumerate(map(_parse, schedule_df[column]))
            if (k >= now if k is not None else d is not None and d >= now.date())]
    rows.sort(key=lambda r: (r[1], r[2] is not None, r[2] or now))
    out = schedule_df.iloc[[r[0] for r in rows]].copy()
    out["game_day"] = pd.Series([r[1] for r in rows], index=out.index, dtype=object)
    out["kickoff_toronto"] = pd.Series([r[2] for r in rows], index=out.index, dtype=object)
    return out
