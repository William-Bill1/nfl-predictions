"""player_props/backtest.py console output under Windows redirected output.

With stdout redirected (e.g. the Streamlit app started with its output sent
to a log file), Python on Windows writes in the locale code page, cp1252,
with strict errors. Status lines with emoji used to raise UnicodeEncodeError:

* inside collect_actual_results' first ``try`` - so the pre-aggregated stats
  were never tried, and a successful play-by-play collection fell into the
  failure path;
* in that failure path itself - so the page crashed instead of reporting the
  original error.

Every test writes to a strict cp1252 stream. No network: the nflverse loaders
are patched with temporary fixtures.
"""
import io
import json
import sys

import pandas as pd
import pytest

import player_props.backtest as backtest
from test_model_performance_week import (IN_PROGRESS, _pbp, _schedule,
                                         _write_prop_predictions)
from test_team_features import ROOT

SNOWMAN = "☃"          # not in cp1252
E_ACUTE = "é"          # in cp1252


def strict_cp1252_stdout(monkeypatch):
    """Replace sys.stdout with a stream like the one Windows gives a redirected
    process (cp1252, strict errors); return a reader for what was written.

    Called inside each test: pytest re-installs its own capture stream after
    fixture setup, which would undo a replacement made in a fixture.
    """
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", write_through=True)
    monkeypatch.setattr(sys, "stdout", stream)

    def written():
        stream.flush()
        return raw.getvalue().decode("cp1252").replace("\r\n", "\n")
    return written


def _no_network(monkeypatch, weekly=None, pbp=None):
    """Patch both nflverse loaders. A value is returned; an exception raised."""
    def loader(result):
        def load(*args, **kwargs):
            if isinstance(result, BaseException):
                raise result
            return result
        return load
    monkeypatch.setattr(backtest.nfl, "import_weekly_data",
                        loader(weekly if weekly is not None else ValueError("no weekly stats")))
    monkeypatch.setattr(backtest.pd, "read_parquet",
                        loader(pbp if pbp is not None else AssertionError("PBP not expected")))


def test_the_stream_is_strict(monkeypatch):
    # Guard: the stream reproduces the failure the module used to hit.
    strict_cp1252_stdout(monkeypatch)
    with pytest.raises(UnicodeEncodeError):
        print("✅ done")


class TestCollectActualResults:
    def test_pre_aggregated_success_is_used_and_reported(self, monkeypatch):
        written = strict_cp1252_stdout(monkeypatch)
        weekly = pd.DataFrame({"player_name": [" QB 0 "], "week": [3], "season": [2026],
                               "passing_yards": [250.0]})
        _no_network(monkeypatch, weekly=weekly)          # PBP must not be needed
        actuals, error = backtest.collect_actual_results(3, 2026)
        assert error == ""
        assert actuals["player_name"].tolist() == ["QB 0"]
        out = written()
        assert "OK: collected pre-aggregated stats for 1 players in Week 3" in out
        assert "ERROR" not in out and "Falling back" not in out

    def test_play_by_play_success_is_not_reported_as_failure(self, monkeypatch):
        written = strict_cp1252_stdout(monkeypatch)
        _no_network(monkeypatch, pbp=_pbp(2026, 3, 16))
        actuals, error = backtest.collect_actual_results(3, 2026)
        assert error == "" and not actuals.empty
        assert set(actuals.attrs[backtest.COMPLETION_ATTR]) == {
            f"2026_03_A{i}_H{i}" for i in range(16)}
        out = written()
        assert "OK: collected play-by-play stats for" in out and "ERROR" not in out

    def test_original_errors_are_reported_not_masked(self, monkeypatch):
        # Both sources fail, with messages cp1252 can't fully encode: the
        # caller gets the original errors, and the report can't crash.
        written = strict_cp1252_stdout(monkeypatch)
        _no_network(monkeypatch,
                    weekly=RuntimeError(f"weekly feed down {SNOWMAN}"),
                    pbp=OSError(f"parquet unreachable {E_ACUTE}{SNOWMAN}"))
        actuals, error = backtest.collect_actual_results(3, 2026)
        assert actuals.empty
        assert f"weekly feed down {SNOWMAN}" in error
        assert f"parquet unreachable {E_ACUTE}{SNOWMAN}" in error
        out = written()
        assert "ERROR: could not collect actual results" in out
        assert f"parquet unreachable {E_ACUTE}\\u2603" in out          # escaped, not raised


class TestWeeklyCheck:
    @pytest.fixture
    def workdir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_prop_predictions(tmp_path / "data_files")
        _schedule({2026: IN_PROGRESS}).to_csv(
            tmp_path / "data_files" / "nfl_games_historical.csv", sep="\t", index=False)
        return tmp_path / "data_files"

    def test_successful_check_completes_and_saves(self, workdir, monkeypatch):
        written = strict_cp1252_stdout(monkeypatch)
        _no_network(monkeypatch, pbp=_pbp(2026, 3, 16))
        result = backtest.run_weekly_accuracy_check(3, 2026)
        assert result["final"] is True and result["total_predictions"] == 1
        [saved] = workdir.glob("accuracy_results_week3_*.json")
        assert json.loads(saved.read_text())["total_predictions"] == 1
        out = written()
        assert "Accuracy results saved to:" in out
        assert "OK: weekly accuracy check complete" in out

    def test_missing_results_are_reported(self, workdir, monkeypatch):
        written = strict_cp1252_stdout(monkeypatch)
        _no_network(monkeypatch, pbp=OSError("offline"))
        assert backtest.run_weekly_accuracy_check(3, 2026) == {}
        assert "ERROR: no actual results available for Week 3" in written()

    def test_unreadable_saved_results_warn_without_raising(self, workdir, monkeypatch):
        written = strict_cp1252_stdout(monkeypatch)
        (workdir / f"accuracy_results_week3_20261009_{E_ACUTE}.json").write_text(
            "{not json", encoding="utf-8")
        assert backtest.load_accuracy_results_for_week(3, 2026) is None
        assert backtest.load_accuracy_history().empty
        assert written().count("WARNING: could not load") == 2


class TestSay:
    def test_unencodable_text_is_escaped(self, monkeypatch):
        written = strict_cp1252_stdout(monkeypatch)
        backtest._say(f"Jos{E_ACUTE} {SNOWMAN} done")
        assert written() == f"Jos{E_ACUTE} \\u2603 done\n"

    def test_unwritable_stream_is_ignored(self, monkeypatch):
        stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        stream.close()
        monkeypatch.setattr(sys, "stdout", stream)
        backtest._say("still fine")                    # no ValueError

    def test_utf8_stream_is_unchanged(self, monkeypatch):
        raw = io.BytesIO()
        monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="utf-8",
                                                            write_through=True))
        backtest._say(f"Jos{E_ACUTE} {SNOWMAN}")
        assert raw.getvalue().decode("utf-8").replace("\r\n", "\n") == \
            f"Jos{E_ACUTE} {SNOWMAN}\n"


def test_module_output_is_ascii_and_goes_through_say():
    src = (ROOT / "player_props" / "backtest.py").read_text(encoding="utf-8")
    assert src.isascii()
    prints = [line.strip() for line in src.splitlines() if "print(" in line]
    assert prints == ["print(message)",
                      'print(message.encode(encoding, "backslashreplace").decode(encoding))']
