"""nfl-gather-data.py guards: chronological training input and genuine pick'em lines.

All pipeline runs use a synthetic schedule in a temporary DATA_DIR and a
fast stand-in for XGBoost; nothing under data_files/ is read or written.
"""

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_team_features import ROOT, _FastClassifier, _load, _schedule

CHECKER = ROOT / "scripts" / "check_pipeline_outputs.py"


def _pipeline_module(data_dir: Path, name: str):
    gd = _load(name, "nfl-gather-data.py")
    gd.DATA_DIR = str(data_dir) + "/"
    gd.XGBClassifier = _FastClassifier
    gd._LGBM_AVAILABLE = False
    return gd


# ------------------------------------------------- chronological order --

class TestChronologicalGuard:
    def test_sorted_input_passes(self):
        gd = _load("nfl_gather_data_guard_sorted", "nfl-gather-data.py")
        gd.require_chronological(_schedule())          # no exception

    def test_unsorted_input_is_rejected_with_clear_message(self):
        gd = _load("nfl_gather_data_guard_unsorted", "nfl-gather-data.py")
        raw = _schedule()
        swapped = pd.concat([raw.iloc[:20], raw.iloc[[40]], raw.iloc[20:40], raw.iloc[41:]])
        with pytest.raises(ValueError, match=r"ordered by \(season, week\).*row 21"):
            gd.require_chronological(swapped)

    def test_pipeline_rejects_unsorted_training_input(self, tmp_path):
        raw = _schedule(seasons=(2019, 2020), weeks=6).sample(frac=1, random_state=0)
        raw.to_csv(tmp_path / "nfl_games_historical.csv", sep="\t", index=False)
        gd = _pipeline_module(tmp_path, "nfl_gather_data_guard_pipeline")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="temporal"):
                gd.main()
        assert not (tmp_path / "nfl_games_historical_with_predictions.csv").exists()  # stopped before writing

    def test_guard_still_runs_under_python_O(self, tmp_path):
        raw = _schedule(seasons=(2020,), weeks=4).iloc[::-1]
        raw.to_csv(tmp_path / "unsorted.csv", sep="\t", index=False)
        code = (
            "import importlib.util, sys, pandas as pd\n"
            f"spec = importlib.util.spec_from_file_location('g', r'{ROOT / 'nfl-gather-data.py'}')\n"
            "g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)\n"
            "assert False, 'asserts are on'  # skipped under -O\n"
            f"g.require_chronological(pd.read_csv(r'{tmp_path / 'unsorted.csv'}', sep='\\t'))\n"
        )
        res = subprocess.run([sys.executable, "-O", "-c", code], cwd=ROOT, capture_output=True, text=True)
        assert res.returncode != 0
        assert "ValueError" in res.stderr and "ordered by (season, week)" in res.stderr
        assert "asserts are on" not in res.stderr          # proves -O really disabled asserts


# ------------------------------------------------------ genuine pick'em --

@pytest.fixture(scope="module")
def pickem_run(tmp_path_factory):
    """Pipeline on a schedule where some PLAYED and some UPCOMING games have spread_line == 0."""
    root = tmp_path_factory.mktemp("pickem")
    data = root / "data_files"
    data.mkdir()
    raw = _schedule(seasons=(2018, 2019, 2020, 2021), weeks=12, unplayed_from=(2021, 9))
    raw.loc[raw.index[5::7], "spread_line"] = 0.0          # genuine pick'em lines, played and upcoming
    raw.to_csv(data / "nfl_games_historical.csv", sep="\t", index=False)

    gd = _pipeline_module(data, "nfl_gather_data_pickem")
    split_inputs = []
    real_split = gd.temporal_split_3way

    def spy(X, y, *a, **kw):
        split_inputs.append(set(X.index))
        return real_split(X, y, *a, **kw)

    gd.temporal_split_3way = spy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gd.main()
    out = pd.read_csv(data / "nfl_games_historical_with_predictions.csv", sep="\t")
    return root, raw, out, split_inputs


class TestGenuinePickem:
    def test_pickem_has_no_probability_or_recommendation(self, pickem_run):
        _, raw, out, _ = pickem_run
        pk = (raw["spread_line"] == 0).to_numpy()
        played = raw["home_score"].notna().to_numpy()
        assert (pk & played).sum() >= 10 and (pk & ~played).sum() >= 1   # both kinds present
        for col in ("prob_underdogCovered", "ev_spread", "edge_underdog_spread"):
            assert out.loc[pk, col].isna().all(), col
        assert (out.loc[pk, "pred_spreadCovered_optimal"] == 0).all()
        assert out.loc[~pk & raw["spread_line"].notna().to_numpy(), "prob_underdogCovered"].notna().all()

    def test_pickem_games_never_reach_training(self, pickem_run):
        _, raw, _, split_inputs = pickem_run
        assert len(split_inputs) == 3                           # spread, moneyline, totals
        pickem_rows = set(raw.index[raw["spread_line"] == 0])
        for rows in split_inputs:
            assert rows and rows.isdisjoint(pickem_rows)
            assert raw.loc[sorted(rows), "home_score"].notna().all()   # played games only

    def _check(self, root):
        return subprocess.run([sys.executable, str(CHECKER)], cwd=root, capture_output=True, text=True)

    def test_checker_passes_clean_output(self, pickem_run):
        root, *_ = pickem_run
        res = self._check(root)
        assert res.returncode == 0, res.stdout + res.stderr

    @pytest.mark.parametrize("tamper,message", [
        ("prob", "games without a spread line have a prob_underdogCovered"),
        ("signal", "games without a spread line carry a spread bet signal"),
    ])
    def test_checker_rejects_pickem_probability_or_signal(self, pickem_run, tmp_path, tamper, message):
        root, raw, out, _ = pickem_run
        bad_root = tmp_path / "bad"
        (bad_root / "data_files").mkdir(parents=True)
        for f in (root / "data_files").iterdir():
            (bad_root / "data_files" / f.name).write_bytes(f.read_bytes())
        bad = out.copy()
        i = raw.index[raw["spread_line"] == 0][0]
        if tamper == "prob":
            bad.loc[i, "prob_underdogCovered"] = 0.6
        else:
            bad.loc[i, "pred_spreadCovered_optimal"] = 1
        bad.to_csv(bad_root / "data_files" / "nfl_games_historical_with_predictions.csv", sep="\t", index=False)
        res = self._check(bad_root)
        assert res.returncode == 1
        assert message in res.stdout
