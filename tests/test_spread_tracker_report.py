"""scripts/spread_tracker_report.build_report - the season-to-date sportsbook
spread-line comparison rollup."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "spread_tracker_report",
    Path(__file__).resolve().parent.parent / "scripts" / "spread_tracker_report.py",
)
str_ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(str_)

COLS = [
    "season", "week", "game_id", "gameday", "home_team", "away_team",
    "book_key", "book_title", "region",
    "home_point", "home_price", "away_point", "away_price",
    "home_spread_normalized",
    "home_implied_prob_devigged", "away_implied_prob_devigged",
    "nflverse_spread_line", "deviation_pts",
    "fetched_at", "source_week_snapshot",
]


def _row(week, game_id, home_team, away_team, book_key, book_title, region,
         home_spread_normalized, nflverse_spread_line, deviation_pts):
    return dict(
        season=2026, week=week, game_id=game_id, gameday="2026-09-21",
        home_team=home_team, away_team=away_team,
        book_key=book_key, book_title=book_title, region=region,
        home_point=-home_spread_normalized, home_price=-110,
        away_point=home_spread_normalized, away_price=-110,
        home_spread_normalized=home_spread_normalized,
        home_implied_prob_devigged=0.5, away_implied_prob_devigged=0.5,
        nflverse_spread_line=nflverse_spread_line, deviation_pts=deviation_pts,
        fetched_at="2026-09-16T00:00:00", source_week_snapshot="market_spreads_week2_2026.csv",
    )


def _write(tmp_path, rows):
    p = tmp_path / "log.csv"
    pd.DataFrame(rows, columns=COLS).to_csv(p, index=False)
    return str(p)


def test_missing_log_is_empty_but_valid(tmp_path):
    r = str_.build_report(str(tmp_path / "nope.csv"))
    assert r["overall"] == {"n_weeks": 0, "n_games": 0, "n_quotes": 0, "per_book": {}}
    assert r["by_week"] == [] and r["best_line_per_game"] == [] and r["anomalies"] == []


def test_empty_log_is_empty_but_valid(tmp_path):
    r = str_.build_report(_write(tmp_path, []))
    assert r["overall"]["n_quotes"] == 0


class TestPerBookBucket:
    def test_mean_and_mean_abs_deviation(self):
        # DK agrees with nflverse (0), FanDuel is +1 over, -1 under across two games.
        df = pd.DataFrame([
            _row(2, "g1", "BAL", "NO", "draftkings", "DraftKings", "us", 8.5, 8.5, 0.0),
            _row(2, "g1", "BAL", "NO", "fanduel", "FanDuel", "us", 9.5, 8.5, 1.0),
            _row(2, "g2", "TB", "CLE", "fanduel", "FanDuel", "us", 7.5, 8.5, -1.0),
        ], columns=COLS)
        out = str_._per_book_bucket(df)
        assert out["draftkings"]["mean_deviation_pts"] == pytest.approx(0.0)
        assert out["draftkings"]["mean_abs_deviation_pts"] == pytest.approx(0.0)
        assert out["fanduel"]["mean_deviation_pts"] == pytest.approx(0.0)  # +1 and -1 average out
        assert out["fanduel"]["mean_abs_deviation_pts"] == pytest.approx(1.0)  # but abs doesn't
        assert out["fanduel"]["n_quotes"] == 2

    def test_missing_comparison_yields_none_not_error(self):
        df = pd.DataFrame([
            _row(2, "g1", "BAL", "NO", "draftkings", "DraftKings", "us", 8.5, float("nan"), float("nan")),
        ], columns=COLS)
        out = str_._per_book_bucket(df)
        assert out["draftkings"]["mean_deviation_pts"] is None
        assert out["draftkings"]["mean_abs_deviation_pts"] is None
        assert out["draftkings"]["n_with_comparison"] == 0
        assert out["draftkings"]["n_quotes"] == 1


class TestBestLinesPerGame:
    def test_picks_max_point_per_side(self):
        # Best for a home bettor = book that gives home the MOST points (least
        # negative / most positive home_point). Best for away = max away_point.
        df = pd.DataFrame([
            _row(2, "g1", "BAL", "NO", "draftkings", "DraftKings", "us", 8.5, 8.5, 0.0),
            _row(2, "g1", "BAL", "NO", "playnow_ca", "PlayNow", "ca", 7.0, 8.5, -1.5),
        ], columns=COLS)
        out = str_._best_lines_per_game(df)
        assert len(out) == 1
        row = out[0]
        # PlayNow's home_point is -7.0 (higher/less negative than DK's -8.5) -> best for home.
        assert row["best_home_book"] == "playnow_ca"
        assert row["best_home_point"] == pytest.approx(-7.0)
        # DK's away_point is 8.5 (higher than PlayNow's 7.0) -> best for away.
        assert row["best_away_book"] == "draftkings"
        assert row["best_away_point"] == pytest.approx(8.5)


class TestAnomalies:
    def test_flags_book_beyond_threshold(self):
        # Field median ~8.5; PlayNow at 11.5 is 3pt off -> flagged (> 1.5 default).
        df = pd.DataFrame([
            _row(2, "g1", "TB", "CLE", "draftkings", "DraftKings", "us", 8.5, 8.5, 0.0),
            _row(2, "g1", "TB", "CLE", "fanduel", "FanDuel", "us", 8.5, 8.5, 0.0),
            _row(2, "g1", "TB", "CLE", "playnow_ca", "PlayNow", "ca", 11.5, 8.5, 3.0),
        ], columns=COLS)
        anomalies = str_._anomalies(df)
        assert len(anomalies) == 1
        assert anomalies[0]["book_key"] == "playnow_ca"
        assert anomalies[0]["diff_from_field_median"] == pytest.approx(3.0)

    def test_no_anomaly_within_threshold(self):
        df = pd.DataFrame([
            _row(2, "g1", "TB", "CLE", "draftkings", "DraftKings", "us", 8.5, 8.5, 0.0),
            _row(2, "g1", "TB", "CLE", "fanduel", "FanDuel", "us", 8.0, 8.5, -0.5),
            _row(2, "g1", "TB", "CLE", "betmgm", "BetMGM", "us", 9.0, 8.5, 0.5),
        ], columns=COLS)
        assert str_._anomalies(df) == []

    def test_worst_first_ordering(self):
        # Median needs >=2 "normal" books per game so one outlier doesn't just
        # drag the median halfway to it.
        df = pd.DataFrame([
            _row(2, "g1", "TB", "CLE", "draftkings", "DraftKings", "us", 8.5, 8.5, 0.0),
            _row(2, "g1", "TB", "CLE", "fanduel", "FanDuel", "us", 8.5, 8.5, 0.0),
            _row(2, "g1", "TB", "CLE", "book_a", "Book A", "us", 11.5, 8.5, 3.0),
            _row(2, "g2", "GB", "NYJ", "draftkings", "DraftKings", "us", 3.5, 3.5, 0.0),
            _row(2, "g2", "GB", "NYJ", "fanduel", "FanDuel", "us", 3.5, 3.5, 0.0),
            _row(2, "g2", "GB", "NYJ", "book_b", "Book B", "us", 7.0, 3.5, 3.5),
        ], columns=COLS)
        anomalies = str_._anomalies(df)
        assert [a["book_key"] for a in anomalies] == ["book_b", "book_a"]


class TestBuildReport:
    def test_overall_and_by_week(self, tmp_path):
        rows = [
            _row(2, "g1", "BAL", "NO", "draftkings", "DraftKings", "us", 8.5, 8.5, 0.0),
            _row(3, "g2", "TB", "CLE", "draftkings", "DraftKings", "us", 8.5, 8.0, 0.5),
        ]
        report = str_.build_report(_write(tmp_path, rows))
        assert report["overall"]["n_weeks"] == 2
        assert report["overall"]["n_games"] == 2
        assert report["overall"]["n_quotes"] == 2
        weeks = {w["week"]: w for w in report["by_week"]}
        assert set(weeks) == {2, 3}
        assert weeks[2]["n_games"] == 1

    def test_json_serializable(self, tmp_path):
        import json
        rows = [_row(2, "g1", "BAL", "NO", "playnow_ca", "PlayNow", "ca", 11.5, 8.5, 3.0)]
        report = str_.build_report(_write(tmp_path, rows))
        json.dumps(report)  # must not raise (numpy int64/float64 leaking would)
