"""Player Props remaining-games banner (player_props.season_schedule): season
from the predictions, that season's schedule only, Toronto date counting.
Temporary fixtures only; data_files/ is never read."""
import importlib.util
import os
import socket
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from player_props import season_schedule as ss
from player_props.game_dates import TORONTO
from test_team_features import ROOT

HEADER = "week,date,home_team,away_team,venue,status\n"


def toronto(s):
    return datetime.fromisoformat(s).replace(tzinfo=TORONTO)


def preds(season=2026, dates=("2026-10-25 00:00:00+00:00", "2026-10-26 00:00:00+00:00"),
          parse=True):
    df = pd.DataFrame({"player_name": ["A.Player"] * len(dates), "season": [season] * len(dates),
                       "week": 7, "game_date": list(dates)})
    if parse:   # as the page loads them: pd.to_datetime -> UTC-aware timestamps
        df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def write_schedule(d, season, rows):
    (Path(d) / f"nfl_schedule_{season}.csv").write_text(
        HEADER + "".join(f"{w},{dt},HOM,AWY,Field,REG\n" for w, dt in rows), encoding="utf-8")


SCHED_2026 = [(7, "2026-10-22"), (7, "2026-10-25"), (7, "2026-10-26"), (8, "2026-10-29"),
              (18, "2027-01-10")]
SCHED_2025 = [(1, "2025-09-04"), (22, "2026-02-08")]


@pytest.fixture
def data_dir(tmp_path):
    write_schedule(tmp_path, 2026, SCHED_2026)
    write_schedule(tmp_path, 2025, SCHED_2025)
    return tmp_path


@pytest.fixture(scope="module")
def page_module():
    spec = importlib.util.spec_from_file_location("player_props_page_season",
                                                  ROOT / "pages" / "2_Player_Props.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


class TestSeasonSelection:
    def test_2026_predictions_use_the_2026_schedule_despite_a_2025_file(self, data_dir):
        level, text = ss.remaining_games_banner(preds(), data_dir, now=toronto("2026-10-22T12:00"))
        assert level == "info"
        assert "**5 remaining scheduled games this season** (2026)" in text

    def test_january_predictions_belong_to_the_previous_season(self, data_dir):
        p = preds(season=2025, dates=("2026-01-17 00:00:00+00:00", "2026-02-08 00:00:00+00:00"))
        assert ss.prediction_season(p) == (2025, None)
        level, text = ss.remaining_games_banner(p, data_dir, now=toronto("2026-01-20T12:00"))
        assert level == "info" and "1 remaining scheduled game this season** (2025)" in text

    def test_january_game_labelled_with_the_calendar_year_is_rejected(self):
        p = preds(season=2026, dates=("2026-01-17 00:00:00+00:00",))
        season, msg = ss.prediction_season(p)
        assert season is None and "outside the 2026 season" in msg

    @pytest.mark.parametrize("dates", [("2026-10-26",), ("2026-10-26T00:00:00",)])
    def test_unparsed_game_dates_use_their_recorded_date(self, dates):
        assert ss.prediction_season(preds(dates=dates, parse=False)) == (2026, None)

    def test_utc_midnight_dates_are_not_shifted_out_of_the_window(self):
        # 2026-08-01 00:00 UTC is 31 July in Toronto; the recorded date is 1 August.
        p = preds(dates=("2026-08-01 00:00:00+00:00",))
        assert ss.prediction_season(p) == (2026, None)

    @pytest.mark.parametrize("season,fragment", [
        ([2025, 2026], "mix seasons (2025, 2026)"),
        (["abc", "abc"], "is not a year"),
        ([2026.5, 2026.5], "is not a valid season"),
        ([1999, 1999], "is not a valid season"),
        ([None, None], "no season recorded"),
    ])
    def test_ambiguous_or_invalid_season(self, data_dir, season, fragment):
        p = pd.DataFrame({"season": season, "game_date": pd.to_datetime(
            ["2026-10-25 00:00:00+00:00"] * 2)})
        level, text = ss.remaining_games_banner(p, data_dir)
        assert level == "warning" and fragment in text and "count is omitted" in text
        assert "remaining scheduled" not in text

    @pytest.mark.parametrize("missing", [None, float("nan"), pd.NA])
    def test_one_missing_season_is_not_dropped(self, data_dir, missing):
        p = pd.DataFrame({"season": pd.Series([2026, missing], dtype=object),
                          "game_date": ["2026-10-25", "2026-10-26"]})
        level, text = ss.remaining_games_banner(p, data_dir)
        assert level == "warning" and "Some predictions have no season recorded" in text

    @pytest.mark.parametrize("value", [
        "2026-10-25NOT-A-DATE", "2026-10-25 garbage", "2026-10-25T", "2026-10-25 20:15",
        "2026-10-25 00:00:00+05:00", "2026-10-25T00:00:00-04:00", "2026-10-25 00:00:00+00:00x",
        pd.Timestamp("2026-10-25 13:00", tz="UTC"),
        pd.Timestamp("2026-10-25 00:00", tz="America/Toronto"),
    ])
    def test_complete_game_date_value_is_validated(self, data_dir, value):
        p = pd.DataFrame({"season": [2026], "game_date": pd.Series([value], dtype=object)})
        level, text = ss.remaining_games_banner(p, data_dir)
        assert level == "warning" and "no valid game date" in text

    @pytest.mark.parametrize("value", [
        "2026-10-25", "2026-10-25 00:00:00+00:00", "2026-10-25T00:00:00Z", "2026-10-25 00:00:00",
        "2026-10-25 00:00", pd.Timestamp("2026-10-25", tz="UTC"), pd.Timestamp("2026-10-25"),
    ])
    def test_stored_utc_midnight_forms_keep_their_recorded_date(self, value):
        assert ss._recorded_date(value).isoformat() == "2026-10-25"

    def test_page_loader_keeps_bad_game_dates_visible(self, tmp_path, monkeypatch, page_module):
        (tmp_path / "data_files").mkdir()
        pd.DataFrame({"player_name": ["A.One", "B.Two"], "season": [2026, 2026],
                      "game_date": ["2026-10-25 00:00:00+00:00", "2026-10-25NOT-A-DATE"]}
                     ).to_csv(tmp_path / "data_files" / "player_props_predictions.csv", index=False)
        write_schedule(tmp_path / "data_files", 2026, SCHED_2026)
        monkeypatch.chdir(tmp_path)
        page_module.load_player_props_predictions.clear()
        try:
            p = page_module.load_player_props_predictions()
        finally:
            page_module.load_player_props_predictions.clear()
        assert list(p["player_name"]) == ["A.One", "B.Two"]          # still displayed
        level, text = ss.remaining_games_banner(p, Path("data_files"))
        assert level == "warning" and "no valid game date" in text

    def test_missing_season_column_or_bad_game_dates(self, data_dir):
        p = preds().drop(columns="season")
        assert "no season column" in ss.remaining_games_banner(p, data_dir)[1]
        p = preds(dates=("not a date", "2026-10-26"), parse=False)
        assert "no valid game date" in ss.remaining_games_banner(p, data_dir)[1]

    @pytest.mark.parametrize("p", [None, pd.DataFrame(), preds().iloc[0:0]])
    def test_empty_predictions(self, data_dir, p):
        level, text = ss.remaining_games_banner(p, data_dir)
        assert level == "warning" and "No player prop predictions are loaded" in text


class TestScheduleValidation:
    def test_missing_schedule_does_not_fall_back(self, tmp_path):
        write_schedule(tmp_path, 2025, SCHED_2025)   # only a stale season present
        level, text = ss.remaining_games_banner(preds(), tmp_path)
        assert level == "warning"
        assert "The 2026 schedule (data_files/nfl_schedule_2026.csv) is missing" in text

    @pytest.mark.parametrize("content,fragment", [
        (b"", "can't be read"),
        (b"\xff\xfe\x00bad", "can't be read"),
        (b'week,date\n1,"2026-09-10\n', "can't be read"),
        (b"week,date,home_team\n1,2026-09-10,SEA\n", "missing away_team"),
        (HEADER.encode(), "has no games"),
        ((HEADER + "1,2026-09-10 20:20,SEA,NE,F,REG\n").encode(), "aren't YYYY-MM-DD"),
        ((HEADER + "1,TBD,SEA,NE,F,REG\n").encode(), "aren't YYYY-MM-DD"),
        ((HEADER + "1,2026-02-30,SEA,NE,F,REG\n").encode(), "aren't YYYY-MM-DD"),
        ((HEADER + "1,2025-09-04,SEA,NE,F,REG\n").encode(), "outside the 2026 season"),
    ])
    def test_unreadable_or_malformed(self, tmp_path, content, fragment):
        (tmp_path / "nfl_schedule_2026.csv").write_bytes(content)
        level, text = ss.remaining_games_banner(preds(), tmp_path)
        assert level == "warning" and fragment in text and "count is omitted" in text

    @pytest.mark.parametrize("rows,fragment", [
        # The reproduced case: no week, no teams.
        ("INVALID,2026-10-25,,,F,REG\n", "whole-number week"),
        (",2026-10-25,SEA,NE,F,REG\n", "whole-number week"),
        ("7.5,2026-10-25,SEA,NE,F,REG\n", "whole-number week"),
        ("0,2026-10-25,SEA,NE,F,REG\n", "whole-number week"),
        ("23,2026-10-25,SEA,NE,F,REG\n", "whole-number week"),
        ("-1,2026-10-25,SEA,NE,F,REG\n", "whole-number week"),
        ("7,2026-10-25,,NE,F,REG\n", "home/away team"),
        ("7,2026-10-25,SEA,,F,REG\n", "home/away team"),
        ("7,2026-10-25,SEA,SEA,F,REG\n", "home/away team"),
        ("7,2026-10-25,Seattle,NE,F,REG\n", "home/away team"),
        ("7,2026-10-25,SEA,NE,F,REG\n7,2026-10-25,SEA,NE,F,REG\n", "duplicate games"),
        ("7,2026-10-25,SEA,NE,F,REG\n7,2026-10-25,NE,SEA,F,REG\n", "duplicate games"),
        ("7,2026-10-25,SEA,NE,F,REG\n7,2026-10-25,SEA,KC,F,REG\n", "duplicate games"),
    ])
    def test_row_values_and_duplicates(self, tmp_path, rows, fragment):
        (tmp_path / "nfl_schedule_2026.csv").write_text(HEADER + rows, encoding="utf-8")
        level, text = ss.remaining_games_banner(preds(), tmp_path, now=toronto("2026-10-22T12:00"))
        assert level == "warning" and fragment in text and "count is omitted" in text
        assert "remaining scheduled" not in text

    def test_reproduced_four_column_schedule_is_rejected(self, tmp_path):
        (tmp_path / "nfl_schedule_2026.csv").write_text(
            "week,date,home_team,away_team\nINVALID,2026-10-25,,\n", encoding="utf-8")
        level, text = ss.remaining_games_banner(preds(), tmp_path)
        assert level == "warning" and "missing venue, status" in text

    def test_same_teams_on_different_dates_are_fine(self, tmp_path):
        (tmp_path / "nfl_schedule_2026.csv").write_text(
            HEADER + "7,2026-10-25,SEA,NE,F,REG\n12,2026-11-29,NE,SEA,F,REG\n22,2027-02-07,SEA,NE,F,REG\n",
            encoding="utf-8")
        text = ss.remaining_games_banner(preds(), tmp_path, now=toronto("2026-10-22T12:00"))[1]
        assert "**3 remaining scheduled games this season**" in text

    def test_unreadable_directory_entry(self, tmp_path):
        (tmp_path / "nfl_schedule_2026.csv").mkdir()
        assert ss.remaining_games_banner(preds(), tmp_path)[0] == "warning"


class TestCountingAndWording:
    @pytest.mark.parametrize("now,count", [
        ("2026-10-22T00:00", 5),
        ("2026-10-22T23:59", 5),    # today's date-only game stays included
        ("2026-10-23T00:00", 4),
        ("2026-10-26T20:30", 3),    # Monday-night game still counted on its day
        ("2026-10-27T00:00", 2),
        ("2027-01-10T23:59", 1),
    ])
    def test_toronto_date_boundaries(self, data_dir, now, count):
        text = ss.remaining_games_banner(preds(), data_dir, now=toronto(now))[1]
        noun = "game" if count == 1 else "games"
        assert f"**{count} remaining scheduled {noun} this season** (2026)" in text

    def test_utc_now_counts_by_toronto_date(self, data_dir):
        # 2026-10-23 02:00 UTC is still Thursday 22 October in Toronto.
        now = datetime.fromisoformat("2026-10-23T02:00:00+00:00")
        assert "**5 remaining" in ss.remaining_games_banner(preds(), data_dir, now=now)[1]

    def test_zero_remaining_games(self, data_dir):
        level, text = ss.remaining_games_banner(preds(), data_dir, now=toronto("2027-01-11T00:00"))
        assert level == "info"
        assert text.startswith("📅 No remaining scheduled games in the 2026 season.")

    def test_wording_matches_the_scope(self):
        text = ss.remaining_games_text(2026, 12)
        assert "remaining scheduled games this season" in text
        assert "this week" not in text and "upcoming" not in text
        assert "today's games stay included" in text and "kickoff times" in text

    def test_page_uses_the_season_banner(self):
        src = (ROOT / "pages" / "2_Player_Props.py").read_text(encoding="utf-8")
        assert "nfl_schedule_2025" not in src and "upcoming games** this week" not in src
        assert "remaining_games_banner(predictions" in src


def test_reads_only_and_writes_nothing(data_dir):
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in data_dir.iterdir()}
    ss.remaining_games_banner(preds(), data_dir, now=toronto("2026-10-22T12:00"))
    after = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in data_dir.iterdir()}
    assert before == after and not any(n.endswith((".tmp", ".bak")) for n in os.listdir(data_dir))
