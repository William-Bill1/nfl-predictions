"""Player Props schedule dates (player_props.game_dates, the page's
get_upcoming_games and predict.load_schedule). Temporary fixtures only; no
network and data_files/ is never read."""
import importlib.util
from datetime import date, datetime, timezone
from unittest import mock

import pandas as pd
import pytest

from player_props import game_dates as gd
from player_props import predict
from test_team_features import ROOT

TOR = gd.TORONTO


def toronto(s):
    return datetime.fromisoformat(s).replace(tzinfo=TOR)


def schedule():
    return pd.DataFrame([
        dict(week=7, date="2026-10-22", home_team="LAC", away_team="DEN"),
        dict(week=7, date="2026-10-25", home_team="BUF", away_team="NYJ"),
        dict(week=7, date="2026-10-26", home_team="PHI", away_team="DAL"),   # Monday
        dict(week=8, date="2026-10-29", home_team="KC", away_team="LV"),
    ])


@pytest.fixture(scope="module")
def page():
    spec = importlib.util.spec_from_file_location("player_props_page",
                                                  ROOT / "pages" / "2_Player_Props.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestGameDay:
    @pytest.mark.parametrize("value", [
        "2026-10-26", " 2026-10-26 ", "2026-10-26 00:00:00", "2026-10-26T00:00:00",
        pd.Timestamp("2026-10-26"), datetime(2026, 10, 26), date(2026, 10, 26),
    ])
    def test_dates_and_midnight_keep_their_date(self, value):
        assert gd.game_day(value) == date(2026, 10, 26)
        assert gd.kickoff(value) is None

    def test_naive_time_is_not_assumed_utc(self):
        # 20:15 with no zone: neither shifted nor treated as a known kickoff.
        assert gd.game_day("2026-10-26 20:15") == date(2026, 10, 26)
        assert gd.kickoff("2026-10-26 20:15") is None

    @pytest.mark.parametrize("value,day,local", [
        # Monday-night kickoff: Tuesday in UTC, Monday in Toronto.
        ("2026-10-27T00:15:00Z", date(2026, 10, 26), "2026-10-26 20:15 EDT"),
        ("2026-12-29T01:15:00+00:00", date(2026, 12, 28), "2026-12-28 20:15 EST"),
        (pd.Timestamp("2026-10-27 00:15", tz="UTC"), date(2026, 10, 26), "2026-10-26 20:15 EDT"),
        (datetime(2026, 9, 13, 20, 25, tzinfo=timezone.utc), date(2026, 9, 13), "2026-09-13 16:25 EDT"),
    ])
    def test_utc_timestamps_cross_toronto_midnight(self, value, day, local):
        assert gd.game_day(value) == day
        assert f"{gd.kickoff(value):%Y-%m-%d %H:%M %Z}" == local

    @pytest.mark.parametrize("value,day,local", [
        ("2026-03-08T04:59:00Z", date(2026, 3, 7), "2026-03-07 23:59 EST"),
        ("2026-03-08T07:00:00Z", date(2026, 3, 8), "2026-03-08 03:00 EDT"),
        ("2026-11-01T03:59:00Z", date(2026, 10, 31), "2026-10-31 23:59 EDT"),
        ("2026-11-01T05:30:00Z", date(2026, 11, 1), "2026-11-01 01:30 EDT"),
        ("2026-11-01T06:30:00Z", date(2026, 11, 1), "2026-11-01 01:30 EST"),
    ])
    def test_dst_transitions(self, value, day, local):
        assert gd.game_day(value) == day
        assert f"{gd.kickoff(value):%Y-%m-%d %H:%M %Z}" == local

    @pytest.mark.parametrize("value", [
        None, float("nan"), pd.NaT, "", "  ", "nan", "TBD", "2026-02-30", "2026-13-01",
        "26/10/2026",
    ])
    def test_missing_or_invalid(self, value):
        assert gd.game_day(value) is None and gd.kickoff(value) is None


class TestPageUpcomingGames:
    def test_monday_game_still_upcoming_sunday_evening(self, page):
        # Old code: '2026-10-26' as UTC midnight = Sun 20:00 EDT, so it vanished here.
        up = page.get_upcoming_games(schedule(), now=toronto("2026-10-25T20:01"))
        assert list(up["date"]) == ["2026-10-25", "2026-10-26", "2026-10-29"]

    @pytest.mark.parametrize("now,dates", [
        ("2026-10-25T00:00", ["2026-10-25", "2026-10-26", "2026-10-29"]),
        ("2026-10-25T23:59", ["2026-10-25", "2026-10-26", "2026-10-29"]),
        ("2026-10-26T23:59", ["2026-10-26", "2026-10-29"]),
        ("2026-10-27T00:00", ["2026-10-29"]),
    ])
    def test_calendar_day_counts_through_the_toronto_day(self, page, now, dates):
        assert list(page.get_upcoming_games(schedule(), now=toronto(now))["date"]) == dates

    def test_utc_now_is_converted_before_taking_the_date(self, page):
        # 2026-10-27 02:00 UTC is still Monday 22:00 in Toronto.
        now = datetime(2026, 10, 27, 2, 0, tzinfo=timezone.utc)
        assert "2026-10-26" in list(page.get_upcoming_games(schedule(), now=now)["date"])

    def test_known_kickoffs_use_the_instant(self, page):
        df = pd.DataFrame({"commence_time": ["2026-10-27T00:15:00Z", "2026-10-26T17:00:00Z",
                                             "2026-10-26 20:15", "bad", None]})
        up = page.get_upcoming_games(df, now=toronto("2026-10-26T20:00"))
        # 20:15 EDT kickoff still ahead; 13:00 EDT kickoff gone; the zone-less
        # 20:15 keeps its date only; invalid and missing rows are dropped.
        assert list(up["commence_time"]) == ["2026-10-26 20:15", "2026-10-27T00:15:00Z"]
        up = page.get_upcoming_games(df, now=toronto("2026-10-26T20:16"))
        assert list(up["commence_time"]) == ["2026-10-26 20:15"]

    def test_input_unchanged_and_none_handled(self, page):
        df = schedule()
        before = df.copy()
        page.get_upcoming_games(df, now=toronto("2026-10-25T12:00"))
        pd.testing.assert_frame_equal(df, before)
        assert page.get_upcoming_games(None) is None
        assert page.get_upcoming_games(pd.DataFrame({"x": [1]})) is None

    def test_page_no_longer_localizes_dates_to_utc(self):
        src = (ROOT / "pages" / "2_Player_Props.py").read_text(encoding="utf-8")
        fn = src[src.index("def get_upcoming_games"):src.index("def filter_stats_by_season")]
        assert "tz_localize" not in fn and "'UTC'" not in fn


class TestPredictScheduleUnchanged:
    """predict.load_schedule is left as is: a date-D game stays selected until
    12:00 UTC on D (08:00 EDT / 07:00 EST), which suits the nightly run the
    evening before; no calendar date is shifted."""

    def _select(self, tmp_path, when):
        schedule().to_csv(tmp_path / "nfl_schedule_2026.csv", index=False)
        inst = pd.Timestamp(when, tz=str(TOR))
        now = classmethod(lambda cls, tz=None: inst.tz_convert(tz) if tz else inst.tz_localize(None))
        with mock.patch.object(predict, "DATA_DIR", tmp_path), \
                mock.patch.object(pd.Timestamp, "now", now):
            return predict.load_schedule(season=2026)

    @pytest.mark.parametrize("when,week,dates", [
        ("2026-10-25 23:00", 7, ["2026-10-26"]),     # nightly run Sunday evening
        ("2026-10-26 07:59", 7, ["2026-10-26"]),
        ("2026-10-26 08:01", 8, ["2026-10-29"]),
        ("2026-10-26 23:00", 8, ["2026-10-29"]),     # nightly run Monday evening
    ])
    def test_week_selection(self, tmp_path, when, week, dates):
        sel = self._select(tmp_path, when)
        assert set(sel["week"]) == {week} and list(sel["date"]) == dates

    def test_game_date_keeps_the_calendar_date(self, tmp_path):
        sel = self._select(tmp_path, "2026-10-25 23:00")
        assert [str(d.date()) for d in sel["game_date"]] == list(sel["date"])
