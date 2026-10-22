"""Alert header game date (betting_log.game_time_text). Date-only game days
keep their calendar date; only offset-aware timestamps are converted to
America/Toronto. In-memory values only; data_files/ is never read."""
from datetime import date, datetime, timezone

import pandas as pd
import pytest

import betting_log as bl
from test_team_features import ROOT

DATE_ONLY = bl.DATE_ONLY_NOTE


class TestObservedDates:
    def test_log_midnight_monday_stays_monday(self):
        # 2026_07_DAL_PHI is logged as '2026-10-26 00:00:00' (a Monday game).
        # The old code read midnight as UTC and showed 'Sun, Oct 25 2026 08:00 PM'.
        text = bl.game_time_text("2026-10-26 00:00:00")
        assert text == f"Mon, Oct 26 2026 {DATE_ONLY}"
        assert "Oct 25" not in text and "PM" not in text

    def test_date_only_sunday_stays_sunday(self):
        # 2026_01_WAS_PHI is logged as '2026-09-13'; the old code showed 'Sat, Sep 12 2026'.
        text = bl.game_time_text("2026-09-13")
        assert text == f"Sun, Sep 13 2026 {DATE_ONLY}"
        assert "Sep 12" not in text


class TestDateOnly:
    @pytest.mark.parametrize("value,day", [
        ("2026-01-01", "Thu, Jan 01 2026"),
        ("2026-03-08", "Sun, Mar 08 2026"),     # DST starts
        ("2026-11-01", "Sun, Nov 01 2026"),     # DST ends
        ("2026-12-31", "Thu, Dec 31 2026"),
        (" 2026-10-26 ", "Mon, Oct 26 2026"),
    ])
    def test_date_strings_keep_their_calendar_date(self, value, day):
        assert bl.game_time_text(value) == f"{day} {DATE_ONLY}"

    @pytest.mark.parametrize("value", [
        "2026-10-26 00:00:00", "2026-10-26T00:00:00", "2026-10-26 00:00",
        "2026-10-26 00:00:00.000", pd.Timestamp("2026-10-26"), datetime(2026, 10, 26),
        date(2026, 10, 26),
    ])
    def test_midnight_representations_are_date_only(self, value):
        text = bl.game_time_text(value)
        assert text == f"Mon, Oct 26 2026 {DATE_ONLY}"
        assert not any(z in text for z in ("EDT", "EST", "AM", "PM"))

    def test_naive_time_shows_its_date_without_inventing_a_zone(self):
        text = bl.game_time_text("2026-10-26 20:15")
        assert text == f"Mon, Oct 26 2026 {bl.NAIVE_TIME_NOTE}"
        assert not any(z in text for z in ("EDT", "EST", "PM"))


class TestKickoffTimestamps:
    @pytest.mark.parametrize("value,text", [
        # Monday-night kickoff: already Tuesday in UTC, still Monday in Toronto.
        ("2026-10-27T00:15:00Z", "Mon, Oct 26 2026 08:15 PM EDT"),
        ("2026-10-27T00:15:00+00:00", "Mon, Oct 26 2026 08:15 PM EDT"),
        ("2026-12-29T01:15:00+00:00", "Mon, Dec 28 2026 08:15 PM EST"),
        ("2026-10-26T20:15:00-04:00", "Mon, Oct 26 2026 08:15 PM EDT"),
        (pd.Timestamp("2026-10-27 00:15", tz="UTC"), "Mon, Oct 26 2026 08:15 PM EDT"),
        (datetime(2026, 9, 13, 20, 25, tzinfo=timezone.utc), "Sun, Sep 13 2026 04:25 PM EDT"),
    ])
    def test_converted_to_toronto(self, value, text):
        assert bl.game_time_text(value) == text

    @pytest.mark.parametrize("value,text", [
        ("2026-03-08T06:59:00Z", "Sun, Mar 08 2026 01:59 AM EST"),
        ("2026-03-08T07:00:00Z", "Sun, Mar 08 2026 03:00 AM EDT"),
        ("2026-11-01T05:30:00Z", "Sun, Nov 01 2026 01:30 AM EDT"),
        ("2026-11-01T06:30:00Z", "Sun, Nov 01 2026 01:30 AM EST"),
    ])
    def test_dst_transitions(self, value, text):
        assert bl.game_time_text(value) == text


@pytest.mark.parametrize("value", [
    None, float("nan"), pd.NaT, "", "   ", "nan", "NaT", "not a date", "2026-02-30",
    "2026-13-01", "2026-10-26 25:00", "26/10/2026",
])
def test_missing_or_invalid_is_unavailable(value):
    assert bl.game_time_text(value) == bl.GAME_DATE_UNAVAILABLE


def test_alert_view_uses_the_helper():
    src = (ROOT / "predictions.py").read_text(encoding="utf-8")
    alert = src[src.index("# Game date."):src.index("# Recommendation and quick stats")]
    assert "game_time_text(row.get('gameday'" in alert
    assert "tz_localize" not in alert and "America/New_York" not in alert


def test_results_note_describes_current_behaviour():
    src = (ROOT / "predictions.py").read_text(encoding="utf-8")
    note = src[src.index("**🔄 How results are updated:**"):]
    note = note[:note.index('""")')]
    assert "nflverse" in note and "nightly" in note and "pending" in note
    assert "ESPN" not in note and "2025" not in note and "Automatic Updates" not in src
    assert not any(t in note for t in ("AM", "PM", "UTC", ":00"))   # no exact run times
