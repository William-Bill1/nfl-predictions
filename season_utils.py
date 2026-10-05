"""Shared NFL-season date helpers.

An NFL season is named after the calendar year it *starts* in: the "2025
season" runs from September 2025 through the Super Bowl in February 2026.

Two different notions of "current season" are needed in this project, so they
are two separate functions:

* :func:`upcoming_or_current_season` - for **schedules**. Once next season's
  schedule is published (spring), that is the season we want to fetch and
  predict, even though no games have been played yet.
* :func:`latest_pbp_season` - for **play-by-play**. PBP only exists for games
  that have actually been played, so during the off-season this is last
  season, and it only advances once the new regular season kicks off in
  September.

Before this module, each script (`create-nfl-historical.py`,
`create-play-by-play.py`, `update_pbp_smart.py`, `update_schedule.py`) computed
the season year with its own slightly different inline rule - including one
hard-coded ``2025`` - which broke every rollover.
"""

from __future__ import annotations

from datetime import date

# The first season with data in this project.
FIRST_SEASON = 2020


def upcoming_or_current_season(today: date | None = None) -> int:
    """Season year to use for **schedules**.

    Rolls over on 1 March: Jan/Feb still belong to the season that started the
    previous calendar year (playoffs); from March onward we look at the season
    that starts (or has started) this calendar year.

    >>> upcoming_or_current_season(date(2026, 1, 15))
    2025
    >>> upcoming_or_current_season(date(2026, 8, 27))
    2026
    """
    today = today or date.today()
    return today.year if today.month >= 3 else today.year - 1


def latest_pbp_season(today: date | None = None) -> int:
    """Season year that has **play-by-play** data available.

    Rolls over on 1 September (regular-season kickoff). Before September we are
    still in - or just past - the previous season, whose PBP is the newest that
    exists.

    >>> latest_pbp_season(date(2026, 1, 15))
    2025
    >>> latest_pbp_season(date(2026, 8, 27))
    2025
    >>> latest_pbp_season(date(2026, 10, 1))
    2026
    """
    today = today or date.today()
    return today.year if today.month >= 9 else today.year - 1


def season_range(today: date | None = None) -> range:
    """``range(FIRST_SEASON, upcoming_or_current_season() + 1)`` - the seasons to
    request when building the full historical schedule dataset."""
    return range(FIRST_SEASON, upcoming_or_current_season(today) + 1)


def completed_weeks(schedule, season: int) -> tuple[list[int], int | None]:
    """Regular-season weeks of ``season`` that have results, plus a default.

    ``schedule`` is the nflverse schedule (``nfl_games_historical.csv``); a game
    counts as completed when both final scores are present.

    Returns ``(weeks, default)``:

    * ``weeks`` - sorted weeks with at least one completed regular-season game,
      i.e. the weeks that have results to analyse;
    * ``default`` - the latest week in which *every* scheduled game is
      completed, so a partly played week (a Monday night game still to come)
      isn't picked, and cached, before its results are final. If no week is
      fully completed yet, the latest week with any results; ``None`` when the
      season has no completed games at all (preseason, or a season that isn't
      in the data).
    """
    reg = schedule[(schedule['season'] == season) & (schedule['game_type'] == 'REG')]
    if reg.empty:
        return [], None
    done = reg['home_score'].notna() & reg['away_score'].notna()
    by_week = done.groupby(reg['week']).agg(['any', 'all'])
    weeks = sorted(int(w) for w in by_week.index[by_week['any']])
    if not weeks:
        return [], None
    full = [int(w) for w in by_week.index[by_week['all']]]
    return weeks, max(full) if full else weeks[-1]
