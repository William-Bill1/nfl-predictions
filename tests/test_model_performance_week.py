"""Model Performance page: season/week defaults come from completed games.

The page used to treat ``selected_season == 2025`` as "the current season", so
from 2026 on it offered Weeks 1-18 and defaulted to Week 18, a week not yet
played. It now lists only regular-season weeks with completed games
(``season_utils.completed_weeks``) and defaults to the latest fully completed
one, or shows a message when the season has none.

The page tests render the real page with ``streamlit.testing`` against a
temporary schedule file, with the backtest's data-fetching functions stubbed,
so no network or production data is used.
"""

import shutil
import sys

import pandas as pd
import pytest

from season_utils import completed_weeks
from test_team_features import ROOT

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAGE = ROOT / "pages" / "4_Model_Performance.py"


def _schedule(spec):
    """``spec``: {season: {week: (completed, scheduled)}} -> schedule rows.

    Also adds a played Wild Card game per season, which must be ignored.
    """
    rows = []
    for season, weeks in spec.items():
        for week, (completed, scheduled) in weeks.items():
            for i in range(scheduled):
                score = 20.0 if i < completed else None
                rows.append({"season": season, "game_type": "REG", "week": week,
                             "home_team": f"H{i}", "away_team": f"A{i}",
                             "home_score": score, "away_score": score})
        rows.append({"season": season, "game_type": "WC", "week": 19,
                     "home_team": "H0", "away_team": "A0",
                     "home_score": 24.0, "away_score": 17.0})
    return pd.DataFrame(rows)


FULL_SEASON = {week: (16, 16) for week in range(1, 19)}
# Monday morning of Week 4: Weeks 1-3 done, Monday night game still to come,
# Week 5 not started.
IN_PROGRESS = {1: (16, 16), 2: (16, 16), 3: (16, 16), 4: (15, 16), 5: (0, 15), 6: (0, 14)}
PRESEASON = {week: (0, 16) for week in range(1, 19)}


class TestCompletedWeeks:
    def test_current_season_defaults_to_latest_fully_completed_week(self):
        weeks, default = completed_weeks(_schedule({2026: IN_PROGRESS}), 2026)
        assert weeks == [1, 2, 3, 4]       # Week 4 has results; 5+ unplayed
        assert default == 3                # not the partly played Week 4

    def test_historical_season_defaults_to_week_18(self):
        weeks, default = completed_weeks(_schedule({2025: FULL_SEASON}), 2025)
        assert weeks == list(range(1, 19))  # playoff weeks excluded
        assert default == 18

    def test_preseason_has_no_weeks(self):
        assert completed_weeks(_schedule({2026: PRESEASON}), 2026) == ([], None)

    def test_season_missing_from_schedule(self):
        assert completed_weeks(_schedule({2025: FULL_SEASON}), 2027) == ([], None)

    def test_first_week_partly_played_falls_back_to_it(self):
        # Thursday night of Week 1: results exist but no week is complete yet.
        weeks, default = completed_weeks(_schedule({2026: {1: (1, 16), 2: (0, 16)}}), 2026)
        assert (weeks, default) == ([1], 1)

    def test_seasons_do_not_mix(self):
        sched = _schedule({2025: FULL_SEASON, 2026: IN_PROGRESS})
        assert completed_weeks(sched, 2026)[1] == 3
        assert completed_weeks(sched, 2025)[1] == 18

    def test_one_missing_score_is_not_completed(self):
        sched = _schedule({2026: {1: (16, 16), 2: (16, 16)}})
        sched.loc[(sched["week"] == 2) & (sched["game_type"] == "REG"), "away_score"] = [float("nan")] + [20.0] * 15
        assert completed_weeks(sched, 2026) == ([1, 2], 1)

    def test_empty_schedule(self):
        empty = pd.DataFrame(columns=["season", "game_type", "week", "home_score", "away_score"])
        assert completed_weeks(empty, 2026) == ([], None)


@pytest.fixture
def render_page(tmp_path, monkeypatch):
    """Copy the page into a temp tree with a fixture schedule and render it."""
    from streamlit.testing.v1 import AppTest
    import player_props.backtest as backtest
    import season_utils

    calls = []
    monkeypatch.setattr(backtest, "load_accuracy_results_for_week",
                        lambda week, season: calls.append((week, season)))

    def collect(week, season=None):
        calls.append(("collect", week, season))
        return render.actuals if render.actuals is not None else (pd.DataFrame(), "stubbed")

    monkeypatch.setattr(backtest, "collect_actual_results", collect)
    monkeypatch.setattr(backtest, "save_accuracy_results",
                        lambda metrics, week, filepath=None, season=None:
                        calls.append(("save", week, season)))
    monkeypatch.chdir(tmp_path)  # no prop predictions file unless a test writes one

    def render(spec, current_season):
        monkeypatch.setattr(season_utils, "upcoming_or_current_season",
                            lambda today=None: current_season)
        (tmp_path / "pages").mkdir(exist_ok=True)
        (tmp_path / "data_files").mkdir(exist_ok=True)
        shutil.copy(PAGE, tmp_path / "pages" / PAGE.name)
        _schedule(spec).to_csv(tmp_path / "data_files" / "nfl_games_historical.csv",
                               sep="\t", index=False)
        at = AppTest.from_file(str(tmp_path / "pages" / PAGE.name), default_timeout=60).run()
        assert not at.exception, [e.value for e in at.exception]
        return at

    render.calls = calls
    render.actuals = None
    return render


def _season_and_week(at):
    boxes = at.sidebar.selectbox
    season_box = boxes[0]
    season = int(season_box.options[season_box.index].split()[0])
    week_box = boxes[1] if len(boxes) > 1 else None
    return season, week_box


class TestModelPerformancePage:
    def test_current_season_defaults_to_completed_week(self, render_page):
        at = render_page({2024: FULL_SEASON, 2025: FULL_SEASON, 2026: IN_PROGRESS}, 2026)
        season, week_box = _season_and_week(at)
        assert season == 2026
        assert week_box.options == ["1", "2", "3", "4"]
        assert week_box.value == 3
        # The default analysis looked at Week 3, never an unplayed week.
        assert render_page.calls == [(3, 2026)]

    def test_historical_season_defaults_to_week_18(self, render_page):
        at = render_page({2024: FULL_SEASON, 2025: FULL_SEASON, 2026: IN_PROGRESS}, 2026)
        at.sidebar.selectbox[0].set_value(1).run()  # 2025
        season, week_box = _season_and_week(at)
        assert season == 2025
        assert week_box.options == [str(w) for w in range(1, 19)]
        assert week_box.value == 18

    def test_preseason_defaults_to_last_season_with_results(self, render_page):
        at = render_page({2024: FULL_SEASON, 2025: FULL_SEASON, 2026: PRESEASON}, 2026)
        season, week_box = _season_and_week(at)
        assert season == 2025
        assert week_box.value == 18

    def test_preseason_selected_shows_message_not_unplayed_week(self, render_page):
        at = render_page({2024: FULL_SEASON, 2025: FULL_SEASON, 2026: PRESEASON}, 2026)
        render_page.calls.clear()
        at.sidebar.selectbox[0].set_value(2).run()  # 2026
        assert not at.exception
        season, week_box = _season_and_week(at)
        assert season == 2026 and week_box is None
        assert any("no completed regular-season games" in i.value for i in at.info)
        assert not any(b.label.startswith("🔄 Run Fresh Analysis") for b in at.sidebar.button)
        assert render_page.calls == []  # no analysis of any week was attempted

    def test_no_season_has_results(self, render_page):
        at = render_page({2026: PRESEASON}, 2026)
        season, week_box = _season_and_week(at)
        assert season == 2026 and week_box is None
        assert render_page.calls == []


# ------------------------------------------- final vs provisional results --

def _actuals(teams):
    """Player results whose data covers ``teams``' games, as collect_actual_results returns."""
    import player_props.backtest as backtest
    df = pd.DataFrame({"player_name": ["QB One"], "passing_yards": [300.0],
                       "rushing_yards": [0.0], "receiving_yards": [0.0]})
    df.attrs[backtest.TEAMS_ATTR] = sorted(teams)
    return df


def _teams(n):
    return {f"H{i}" for i in range(n)} | {f"A{i}" for i in range(n)}


def _write_prop_predictions(data_dir):
    data_dir.mkdir(exist_ok=True)
    pd.DataFrame({"player_name": ["QB One"], "prop_type": ["passing_yards"],
                  "line_value": [250.5], "recommendation": ["OVER"],
                  "confidence": [0.7]}).to_csv(data_dir / "player_props_predictions.csv", index=False)


def _saves(calls):
    return [c for c in calls if c and c[0] == "save"]


class TestWeekResultsStatus:
    SCHED = _schedule({2026: IN_PROGRESS})

    def _status(self, actuals, week):
        from player_props.backtest import week_results_status
        return week_results_status(actuals, week, 2026, schedule=self.SCHED)

    def test_final_when_every_game_final_and_covered(self):
        assert self._status(_actuals(_teams(16)), 3) == (True, "")

    def test_delayed_play_by_play_is_provisional(self):
        # Scores are final but one game's play-by-play hasn't been published.
        final, reason = self._status(_actuals(_teams(15)), 3)
        assert not final
        assert "missing for A15, H15" in reason and "delayed" in reason

    def test_unfinished_game_is_provisional(self):
        final, reason = self._status(_actuals(_teams(16)), 4)
        assert not final and "1 of 16 Week 4 games aren't final" in reason

    def test_no_results_is_provisional(self):
        assert not self._status(pd.DataFrame(), 3)[0]

    def test_unknown_coverage_is_provisional(self):
        df = _actuals(_teams(16))
        df.attrs.clear()
        final, reason = self._status(df, 3)
        assert not final and "don't say which games" in reason

    def test_week_not_in_schedule_is_provisional(self):
        assert not self._status(_actuals(_teams(16)), 17)[0]

    def test_missing_schedule_is_provisional(self, tmp_path, monkeypatch):
        from player_props.backtest import week_results_status
        monkeypatch.chdir(tmp_path)
        final, reason = week_results_status(_actuals(_teams(16)), 3, 2026)
        assert not final and "couldn't be read" in reason


class TestAccuracyCache:
    @pytest.fixture
    def workdir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_prop_predictions(tmp_path / "data_files")
        _schedule({2026: IN_PROGRESS}).to_csv(
            tmp_path / "data_files" / "nfl_games_historical.csv", sep="\t", index=False)
        return tmp_path / "data_files"

    def _run(self, monkeypatch, teams, week=3):
        import player_props.backtest as backtest
        monkeypatch.setattr(backtest, "collect_actual_results",
                            lambda w, s=None: (_actuals(teams), ""))
        return backtest.run_weekly_accuracy_check(week, 2026)

    def test_provisional_results_are_returned_but_not_saved(self, workdir, monkeypatch):
        import player_props.backtest as backtest
        result = self._run(monkeypatch, _teams(15))
        assert result["total_predictions"] == 1 and result["final"] is False
        assert "missing for" in result["provisional_reason"]
        assert list(workdir.glob("accuracy_results_*.json")) == []
        assert backtest.load_accuracy_results_for_week(3, 2026) is None

    def test_partly_played_week_is_not_saved(self, workdir, monkeypatch):
        result = self._run(monkeypatch, _teams(16), week=4)
        assert result["final"] is False
        assert list(workdir.glob("accuracy_results_*.json")) == []

    def test_final_results_are_saved_with_season(self, workdir, monkeypatch):
        import player_props.backtest as backtest
        assert self._run(monkeypatch, _teams(16))["final"] is True
        cached = backtest.load_accuracy_results_for_week(3, 2026)
        assert cached["season"] == 2026 and cached["week"] == 3
        assert cached["total_predictions"] == 1
        assert backtest.load_accuracy_results_for_week(3, 2025) is None

    def test_cache_is_per_season_and_newest_by_date(self, workdir):
        import json
        import player_props.backtest as backtest

        def write(name, **extra):
            doc = {"overall_accuracy": 0.5, "total_predictions": 1, "by_confidence_tier": {},
                   "by_prop_type": {}, "detailed_results": [], **extra}
            (workdir / name).write_text(json.dumps(doc))

        write("accuracy_results_week3_20260924_235959.json", season=2026, week=3, tag="older")
        write("accuracy_results_week3_20260928_000001.json", season=2026, week=3, tag="newer")
        write("accuracy_results_week3_20260930_120000.json", season=2025, week=3, tag="2025")
        write("accuracy_results_week3_20261001_120000.json", tag="legacy, no season")
        # Newest by date, not by time of day; other seasons and legacy files skipped.
        assert backtest.load_accuracy_results_for_week(3, 2026)["tag"] == "newer"
        assert backtest.load_accuracy_results_for_week(3, 2025)["tag"] == "2025"
        assert backtest.load_accuracy_results_for_week(3, 2024) is None

    def test_weekly_script_does_not_save_again(self, workdir, monkeypatch):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_weekly_backtest", ROOT / "scripts" / "run_weekly_backtest.py")
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        assert not hasattr(script, "save_accuracy_results")
        monkeypatch.setattr(script, "run_weekly_accuracy_check",
                            lambda w, s: {"final": False, "provisional_reason": "stub"})
        monkeypatch.setattr(script, "_load_predictions", lambda: pd.DataFrame())
        monkeypatch.setattr(script, "latest_completed_week", lambda df, s: 3)
        monkeypatch.setattr(script.os.path, "exists", lambda p: True)
        script.main()
        assert list(workdir.glob("accuracy_results_*.json")) == []


class TestPageProvisional:
    def test_delayed_data_shows_warning_and_is_not_cached(self, render_page, tmp_path):
        _write_prop_predictions(tmp_path / "data_files")
        render_page.actuals = (_actuals(_teams(15)), "")
        at = render_page({2025: FULL_SEASON, 2026: IN_PROGRESS}, 2026)
        assert any("Provisional results, not cached" in w.value and "missing for" in w.value
                   for w in at.warning)
        assert _saves(render_page.calls) == []

    def test_complete_data_is_cached_with_season(self, render_page, tmp_path):
        _write_prop_predictions(tmp_path / "data_files")
        render_page.actuals = (_actuals(_teams(16)), "")
        at = render_page({2025: FULL_SEASON, 2026: IN_PROGRESS}, 2026)
        assert not any("Provisional" in w.value for w in at.warning)
        assert _saves(render_page.calls) == [("save", 3, 2026)]

    def test_missing_play_by_play_shows_clear_message(self, render_page, tmp_path):
        _write_prop_predictions(tmp_path / "data_files")
        render_page.actuals = (pd.DataFrame(), "No play-by-play data found for Week 3, Season 2026")
        at = render_page({2025: FULL_SEASON, 2026: IN_PROGRESS}, 2026)
        assert any("No play-by-play data found for Week 3" in e.value for e in at.error)
        assert _saves(render_page.calls) == []
