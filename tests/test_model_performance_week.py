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


FINAL = (24.0, 17.0)   # every completed fixture game ends home 24, away 17


def _game_id(season, week, i):
    return f"{season}_{week:02d}_A{i}_H{i}"


def _schedule(spec):
    """``spec``: {season: {week: (completed, scheduled)}} -> schedule rows.

    Also adds a played Wild Card game per season, which must be ignored.
    """
    rows = []
    for season, weeks in spec.items():
        for week, (completed, scheduled) in weeks.items():
            for i in range(scheduled):
                done = i < completed
                rows.append({"game_id": _game_id(season, week, i), "season": season,
                             "game_type": "REG", "week": week,
                             "home_team": f"H{i}", "away_team": f"A{i}",
                             "home_score": FINAL[0] if done else None,
                             "away_score": FINAL[1] if done else None})
        rows.append({"game_id": f"{season}_19_A0_H0", "season": season, "game_type": "WC",
                     "week": 19, "home_team": "H0", "away_team": "A0",
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

# One fixture game's play-by-play: running score after each play, ending with
# nflverse's "END GAME" row at the final score FINAL (home 24, away 17).
GAME_SCRIPT = [
    ("pass", (7, 0)), ("pass", (7, 7)), ("rush", (14, 7)), ("pass", (14, 10)),
    ("pass", (21, 10)), ("rush", (21, 17)), ("pass", (24, 17)), ("END GAME", (24, 17)),
]


def _pbp(season, week, n_games, keep=None):
    """nflverse-shaped play-by-play for ``n_games`` games of a week.

    ``keep`` maps a game index to how many of its plays are present, to model
    play-by-play that stops early. Both teams of every game still appear.
    """
    keep = keep or {}
    rows = []
    for i in range(n_games):
        script = GAME_SCRIPT[:keep.get(i, len(GAME_SCRIPT))]
        for play_id, (kind, (home, away)) in enumerate(script, start=1):
            is_pass, is_rush = kind == "pass", kind == "rush"
            rows.append({
                "game_id": _game_id(season, week, i), "play_id": play_id,
                "season": season, "week": week, "home_team": f"H{i}", "away_team": f"A{i}",
                "desc": "END GAME" if kind == "END GAME" else f"{kind} play",
                "total_home_score": home, "total_away_score": away,
                # nflverse copies the schedule's final score onto every play,
                # so these say nothing about whether the plays are complete.
                "home_score": FINAL[0], "away_score": FINAL[1],
                "pass": int(is_pass), "rush": int(is_rush),
                "passer_player_name": f"QB {i}" if is_pass else None,
                "receiver_player_name": f"WR {i}" if is_pass else None,
                "rusher_player_name": f"RB {i}" if is_rush else None,
                "passing_yards": 12.0 if is_pass else None,
                "receiving_yards": 12.0 if is_pass else None,
                "rushing_yards": 6.0 if is_rush else None,
                "pass_touchdown": 0, "rush_touchdown": 0,
                "complete_pass": int(is_pass),
            })
    return pd.DataFrame(rows)


from player_props.backtest import collect_actual_results as _REAL_COLLECT  # noqa: E402


def _collect(monkeypatch, pbp, week=3, season=2026):
    """Run the real collect_actual_results on fixture play-by-play (no network)."""
    import player_props.backtest as backtest

    def no_weekly(*a, **k):
        raise ValueError("pre-aggregated stats unavailable")

    monkeypatch.setattr(backtest.nfl, "import_weekly_data", no_weekly)
    monkeypatch.setattr(backtest.pd, "read_parquet", lambda url: pbp)
    # The unpatched function: some tests stub backtest.collect_actual_results.
    actuals, error = _REAL_COLLECT(week, season)
    assert error == "" and not actuals.empty
    return actuals


def _write_prop_predictions(data_dir):
    data_dir.mkdir(exist_ok=True)
    pd.DataFrame({"player_name": ["QB 0"], "prop_type": ["passing_yards"],
                  "line_value": [50.5], "recommendation": ["OVER"],
                  "confidence": [0.7]}).to_csv(data_dir / "player_props_predictions.csv", index=False)


def _saves(calls):
    return [c for c in calls if c and c[0] == "save"]


COMPLETE = {}                      # every game's play-by-play complete
ENDS_EARLY = {5: 5}                # game 5 stops at 14-10, before its last scores
NO_END_ROW = {5: 7}                # game 5 reaches 24-17 but the END GAME play is missing


class TestPbpGameCompletion:
    def test_summarizes_each_game(self):
        from player_props.backtest import pbp_game_completion
        done = pbp_game_completion(_pbp(2026, 3, 2, keep={1: 3}))
        assert done[_game_id(2026, 3, 0)] == {"end_game": True, "home_total": 24, "away_total": 17}
        # No END GAME play: no final state is claimed from the plays that exist.
        assert done[_game_id(2026, 3, 1)] == {"end_game": False, "home_total": None, "away_total": None}

    def test_final_state_is_one_paired_row_not_independent_maxima(self):
        from player_props.backtest import pbp_game_completion
        pbp = _pbp(2026, 3, 1)
        gid = _game_id(2026, 3, 0)
        # A scoring play was later reversed: home briefly 24 (then back to 21),
        # away reached 17 only at the end. Maxima are (24, 17); the real final
        # state is (21, 17).
        pbp.loc[(pbp["play_id"] >= 7), "total_home_score"] = 21
        pbp.loc[(pbp["play_id"] == 6), "total_home_score"] = 24
        assert pbp["total_home_score"].max() == 24 and pbp["total_away_score"].max() == 17
        assert pbp_game_completion(pbp)[gid] == {"end_game": True, "home_total": 21, "away_total": 17}

    def test_ambiguous_or_null_end_state_is_unprovable(self):
        from player_props.backtest import pbp_game_completion
        gid = _game_id(2026, 3, 0)
        two_ends = pd.concat([_pbp(2026, 3, 1), _pbp(2026, 3, 1).tail(1).assign(
            total_home_score=21, play_id=99)], ignore_index=True)
        assert pbp_game_completion(two_ends)[gid] == {"end_game": True, "home_total": None, "away_total": None}
        null_end = _pbp(2026, 3, 1)
        null_end.loc[null_end["desc"] == "END GAME", "total_away_score"] = None
        assert pbp_game_completion(null_end)[gid] == {"end_game": True, "home_total": None, "away_total": None}

    def test_row_order_does_not_matter(self):
        from player_props.backtest import pbp_game_completion
        pbp = _pbp(2026, 3, 1)
        assert pbp_game_completion(pbp.iloc[::-1]) == pbp_game_completion(pbp)

    def test_missing_columns_give_no_evidence(self):
        from player_props.backtest import pbp_game_completion
        assert pbp_game_completion(_pbp(2026, 3, 1).drop(columns="desc")) == {}


class TestWeekResultsStatus:
    SCHED = _schedule({2026: IN_PROGRESS})

    def _status(self, actuals, week=3):
        from player_props.backtest import week_results_status
        return week_results_status(actuals, week, 2026, schedule=self.SCHED)

    def test_complete_play_by_play_is_final(self, monkeypatch):
        actuals = _collect(monkeypatch, _pbp(2026, 3, 16, keep=COMPLETE))
        assert self._status(actuals) == (True, "")

    def test_every_team_present_but_one_game_ends_early(self, monkeypatch):
        # The regression: all 32 teams appear in the stats, yet game 5's
        # play-by-play stops at 14-10. Team coverage alone called this final.
        pbp = _pbp(2026, 3, 16, keep=ENDS_EARLY)
        assert set(pbp["home_team"]) | set(pbp["away_team"]) == (
            {f"H{i}" for i in range(16)} | {f"A{i}" for i in range(16)})
        final, reason = self._status(_collect(monkeypatch, pbp))
        assert not final
        assert f"ends before the final play for {_game_id(2026, 3, 5)}" in reason

    def test_missing_end_game_row_is_provisional(self, monkeypatch):
        final, reason = self._status(_collect(monkeypatch, _pbp(2026, 3, 16, keep=NO_END_ROW)))
        assert not final and "ends before the final play" in reason

    def test_score_mismatch_is_provisional(self, monkeypatch):
        # END GAME present, but a scoring play is missing: the totals disagree.
        pbp = _pbp(2026, 3, 16)
        gid = _game_id(2026, 3, 2)
        drop = pbp.index[(pbp["game_id"] == gid) & (pbp["play_id"].isin([7]))]
        pbp = pbp.drop(index=drop)
        pbp.loc[pbp["game_id"] == gid, "total_home_score"] = pbp.loc[
            pbp["game_id"] == gid, "total_home_score"].clip(upper=21)
        final, reason = self._status(_collect(monkeypatch, pbp))
        assert not final
        assert f"doesn't match the final score for {gid} (17-21 vs final 17-24)" in reason

    def test_corrected_scoring_event_is_not_hidden_by_maxima(self, monkeypatch):
        # Home hit 24 before a reversal and away reached 17, so the per-column
        # maxima equal the schedule's 24-17. The END GAME state is 21-17: not final.
        pbp = _pbp(2026, 3, 16)
        gid = _game_id(2026, 3, 4)
        in_game = pbp["game_id"] == gid
        pbp.loc[in_game & (pbp["play_id"] >= 7), "total_home_score"] = 21
        pbp.loc[in_game & (pbp["play_id"] == 6), "total_home_score"] = 24
        sub = pbp[in_game]
        assert (sub["total_home_score"].max(), sub["total_away_score"].max()) == (24, 17)
        final, reason = self._status(_collect(monkeypatch, pbp))
        assert not final
        assert f"doesn't match the final score for {gid} (17-21 vs final 17-24)" in reason

    def test_end_state_missing_a_score_is_provisional(self, monkeypatch):
        pbp = _pbp(2026, 3, 16)
        gid = _game_id(2026, 3, 6)
        pbp.loc[(pbp["game_id"] == gid) & (pbp["desc"] == "END GAME"), "total_home_score"] = None
        final, reason = self._status(_collect(monkeypatch, pbp))
        assert not final and f"{gid} (no valid final score state" in reason

    def test_missing_game_is_provisional(self, monkeypatch):
        pbp = _pbp(2026, 3, 16)
        pbp = pbp[pbp["game_id"] != _game_id(2026, 3, 9)]
        final, reason = self._status(_collect(monkeypatch, pbp))
        assert not final and f"no play-by-play for {_game_id(2026, 3, 9)}" in reason

    def test_unfinished_game_is_provisional(self, monkeypatch):
        final, reason = self._status(_collect(monkeypatch, _pbp(2026, 4, 16), week=4), week=4)
        assert not final and "1 of 16 Week 4 games aren't final" in reason

    def test_pre_aggregated_stats_cannot_be_verified(self):
        # Weekly stats carry no game IDs, so completeness can't be shown.
        weekly = pd.DataFrame({"player_name": ["QB 0"], "passing_yards": [300.0],
                               "rushing_yards": [0.0], "receiving_yards": [0.0]})
        final, reason = self._status(weekly)
        assert not final and "can't be verified game by game" in reason

    def test_no_results_is_provisional(self):
        assert not self._status(pd.DataFrame())[0]

    def test_week_not_in_schedule_is_provisional(self, monkeypatch):
        actuals = _collect(monkeypatch, _pbp(2026, 17, 16), week=17)
        assert not self._status(actuals, week=17)[0]

    def test_missing_schedule_is_provisional(self, tmp_path, monkeypatch):
        from player_props.backtest import week_results_status
        actuals = _collect(monkeypatch, _pbp(2026, 3, 16))
        monkeypatch.chdir(tmp_path)
        final, reason = week_results_status(actuals, 3, 2026)
        assert not final and "couldn't be read" in reason


class TestAccuracyCache:
    @pytest.fixture
    def workdir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_prop_predictions(tmp_path / "data_files")
        _schedule({2026: IN_PROGRESS}).to_csv(
            tmp_path / "data_files" / "nfl_games_historical.csv", sep="\t", index=False)
        return tmp_path / "data_files"

    def _run(self, monkeypatch, keep, week=3):
        """run_weekly_accuracy_check on fixture play-by-play, through the real
        collect_actual_results (network calls patched out)."""
        import player_props.backtest as backtest
        pbp = _pbp(2026, week, 16, keep=keep)
        monkeypatch.setattr(backtest.nfl, "import_weekly_data",
                            lambda *a, **k: (_ for _ in ()).throw(ValueError("no weekly stats")))
        monkeypatch.setattr(backtest.pd, "read_parquet", lambda url: pbp)
        return backtest.run_weekly_accuracy_check(week, 2026)

    def test_provisional_results_are_returned_but_not_saved(self, workdir, monkeypatch):
        import player_props.backtest as backtest
        result = self._run(monkeypatch, ENDS_EARLY)
        assert result["total_predictions"] == 1 and result["final"] is False
        assert "ends before the final play" in result["provisional_reason"]
        assert list(workdir.glob("accuracy_results_*.json")) == []
        assert backtest.load_accuracy_results_for_week(3, 2026) is None

    def test_partly_played_week_is_not_saved(self, workdir, monkeypatch):
        result = self._run(monkeypatch, COMPLETE, week=4)
        assert result["final"] is False
        assert list(workdir.glob("accuracy_results_*.json")) == []

    def test_final_results_are_saved_with_season(self, workdir, monkeypatch):
        import player_props.backtest as backtest
        assert self._run(monkeypatch, COMPLETE)["final"] is True
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
    def test_game_ending_early_shows_warning_and_is_not_cached(self, render_page, tmp_path,
                                                               monkeypatch):
        _write_prop_predictions(tmp_path / "data_files")
        render_page.actuals = (_collect(monkeypatch, _pbp(2026, 3, 16, keep=ENDS_EARLY)), "")
        at = render_page({2025: FULL_SEASON, 2026: IN_PROGRESS}, 2026)
        assert any("Provisional results, not cached" in w.value
                   and "ends before the final play" in w.value for w in at.warning)
        assert _saves(render_page.calls) == []

    def test_complete_data_is_cached_with_season(self, render_page, tmp_path, monkeypatch):
        _write_prop_predictions(tmp_path / "data_files")
        render_page.actuals = (_collect(monkeypatch, _pbp(2026, 3, 16)), "")
        at = render_page({2025: FULL_SEASON, 2026: IN_PROGRESS}, 2026)
        assert not any("Provisional" in w.value for w in at.warning)
        assert _saves(render_page.calls) == [("save", 3, 2026)]

    def test_missing_play_by_play_shows_clear_message(self, render_page, tmp_path):
        _write_prop_predictions(tmp_path / "data_files")
        render_page.actuals = (pd.DataFrame(), "No play-by-play data found for Week 3, Season 2026")
        at = render_page({2025: FULL_SEASON, 2026: IN_PROGRESS}, 2026)
        assert any("No play-by-play data found for Week 3" in e.value for e in at.error)
        assert _saves(render_page.calls) == []
