"""Tests for season_utils - the shared NFL-season date logic."""

from datetime import date

from season_utils import (
    FIRST_SEASON,
    latest_pbp_season,
    season_range,
    upcoming_or_current_season,
)


class TestUpcomingOrCurrentSeason:
    def test_january_is_previous_calendar_year(self):
        # Jan 2026 -> still the 2025 season (playoffs)
        assert upcoming_or_current_season(date(2026, 1, 15)) == 2025

    def test_february_is_previous_calendar_year(self):
        assert upcoming_or_current_season(date(2026, 2, 28)) == 2025

    def test_march_rolls_to_current_calendar_year(self):
        # From March we look ahead to the season starting this year
        assert upcoming_or_current_season(date(2026, 3, 1)) == 2026

    def test_preseason_august_is_current_calendar_year(self):
        assert upcoming_or_current_season(date(2026, 8, 27)) == 2026

    def test_midseason_november(self):
        assert upcoming_or_current_season(date(2025, 11, 10)) == 2025


class TestLatestPbpSeason:
    def test_january_is_previous_season(self):
        assert latest_pbp_season(date(2026, 1, 15)) == 2025

    def test_august_still_previous_season(self):
        # New season hasn't kicked off yet - newest PBP is last year's
        assert latest_pbp_season(date(2026, 8, 27)) == 2025

    def test_september_rolls_over(self):
        assert latest_pbp_season(date(2026, 9, 1)) == 2026

    def test_october_is_current_season(self):
        assert latest_pbp_season(date(2026, 10, 1)) == 2026


class TestSeasonRange:
    def test_starts_at_first_season(self):
        assert season_range(date(2026, 8, 27)).start == FIRST_SEASON

    def test_includes_upcoming_season(self):
        r = season_range(date(2026, 8, 27))
        assert r.stop == 2027  # range end is exclusive -> 2026 included
        assert 2026 in r

    def test_offseason_january(self):
        r = season_range(date(2026, 1, 15))
        assert list(r)[-1] == 2025
