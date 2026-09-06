"""scripts/weekly_spread_report.build_report - the season-to-date spread rollup."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "weekly_spread_report",
    Path(__file__).resolve().parent.parent / "scripts" / "weekly_spread_report.py",
)
wsr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wsr)

COLS = ["bet_type", "week", "confidence_tier", "bet_result", "bet_profit"]
PROFIT = {"win": 90.91, "loss": -100.0, "push": 0.0, "pending": ""}


def _row(week, tier, res):
    return dict(bet_type="spread", week=week, confidence_tier=tier,
               bet_result=res, bet_profit=PROFIT[res])


def _write(tmp_path, rows):
    p = tmp_path / "log.csv"
    pd.DataFrame(rows, columns=COLS).to_csv(p, index=False)
    return str(p)


def test_missing_log_is_empty_but_valid(tmp_path):
    r = wsr.build_report(str(tmp_path / "nope.csv"))
    assert r["overall"]["settled"] == 0
    assert r["by_week"] == [] and r["by_tier"] == {}


def test_only_spread_rows_count(tmp_path):
    log = _write(tmp_path, [
        _row(1, "Good", "win"),
        {**_row(1, "Good", "loss"), "bet_type": "moneyline"},
    ])
    assert wsr.build_report(log)["overall"]["settled"] == 1


def test_record_profit_and_roi(tmp_path):
    # 5 win, 3 loss, 1 push -> graded=8, profit=5*90.91-300=154.55, roi=19.32
    rows = ([_row(1, "Good", "win")] * 5 + [_row(1, "Good", "loss")] * 3
            + [_row(1, "Good", "push")] + [_row(2, "Lean", "pending")])
    o = wsr.build_report(_write(tmp_path, rows))["overall"]
    assert (o["win"], o["loss"], o["push"], o["pending"]) == (5, 3, 1, 1)
    assert o["settled"] == 9
    assert o["win_rate"] == pytest.approx(0.625)          # pushes excluded
    assert o["profit"] == pytest.approx(154.55, abs=0.01)
    assert o["roi_pct"] == pytest.approx(19.32, abs=0.01)  # / (8 * 100)


def test_by_week_and_by_tier(tmp_path):
    rows = [_row(1, "Elite", "win"), _row(1, "Good", "loss"), _row(2, "Good", "win")]
    r = wsr.build_report(_write(tmp_path, rows))
    weeks = {w["week"]: w for w in r["by_week"]}
    assert weeks[1]["win"] == 1 and weeks[1]["loss"] == 1
    assert weeks[2]["win"] == 1
    assert r["by_tier"]["Elite"]["win"] == 1
    assert "Lean" not in r["by_tier"]  # no Lean rows -> not emitted
