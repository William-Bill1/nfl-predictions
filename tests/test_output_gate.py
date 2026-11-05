"""The nightly publication gate: scripts/check_pipeline_outputs.py on outputs of
the real pipeline (synthetic schedules, fast stub classifier, temp dirs), and the
nightly workflow's wiring, simulated from the gate onward with stand-in `git`
and `python` commands. Nothing here touches data_files/ or the network."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from test_pending_settlement import _run, _settlement_schedule, _utc
from test_team_features import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import check_pipeline_outputs as cpo  # noqa: E402
import pregame_snapshots as ps  # noqa: E402

CHECKER = ROOT / "scripts" / "check_pipeline_outputs.py"
WORKFLOW = ROOT / ".github" / "workflows" / "nightly-update.yml"
PRED, SCHED, MANIFEST = ps.PREDICTIONS_NAME, ps.SCHEDULE_NAME, ps.MANIFEST_NAME
TORONTO = ZoneInfo("America/Toronto")
RUN = {"GITHUB_RUN_ID": "4242", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": "c0ffee" + "0" * 34,
       "GITHUB_WORKFLOW": "Nightly NFL Data Update"}
SECRETS = {"GITHUB_TOKEN": "ghs_not_a_real_token_1234567890", "ODDS_API_KEY": "odds-key-not-real-0987"}


def _pipeline(tmp_path_factory, now, name):
    """The real pipeline as this CI run would run it (GITHUB_* set, then restored)."""
    raw, invalid = _settlement_schedule()
    saved = {k: os.environ.get(k) for k in RUN}
    os.environ.update(RUN)
    try:
        gd, data, out, _ = _run(raw, tmp_path_factory.mktemp(name), now, name)
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return dict(gd=gd, raw=raw, invalid=invalid, data=data, out=out, now=now)


@pytest.fixture(scope="module")
def sameday(tmp_path_factory):
    """Run "as of" 2021 week 9's first game day: settled games, scored-but-same-day
    games, unplayed games and games with invalid evidence, all in one file."""
    raw, _ = _settlement_schedule()
    week9 = raw[(raw["season"] == 2021) & (raw["week"] == 9)]
    r = _pipeline(tmp_path_factory, _utc(f"{week9['gameday'].min()}T17:00:00"), "gate_sameday")
    r["week9_day1"] = week9.index[week9["gameday"] == week9["gameday"].min()]
    return r


@pytest.fixture(scope="module")
def tonight(tmp_path_factory):
    """Run at the real current time, as the nightly does (2021 week 10+ still unplayed)."""
    return _pipeline(tmp_path_factory, None, "gate_tonight")


# ------------------------------------------------------------- helpers --

def _copy(src: Path, tmp_path: Path) -> Path:
    dst = tmp_path / "data_files"
    shutil.copytree(src, dst)
    return dst


def _check(data, require=False, env=None):
    errors, info = cpo.check_outputs(data, require_current_run=require, env=RUN if env is None else env)
    return errors, info


def _edit_cells(path: Path, edits: dict):
    """Change cells in a TSV as text: {(row, column): value}; '' = missing."""
    lines = path.read_text(encoding="utf-8").split("\n")
    header = lines[0].split("\t")
    for (row, col), value in edits.items():
        cells = lines[row + 1].split("\t")
        cells[header.index(col)] = str(value)
        lines[row + 1] = "\t".join(cells)
    path.write_text("\n".join(lines), encoding="utf-8")


def _drop_column(path: Path, col: str):
    lines = path.read_text(encoding="utf-8").split("\n")
    j = lines[0].split("\t").index(col)
    path.write_text("\n".join("\t".join(c for k, c in enumerate(line.split("\t")) if k != j) if line else line
                              for line in lines), encoding="utf-8")


def _manifest(data) -> dict:
    return json.loads((data / MANIFEST).read_text(encoding="utf-8"))


def _reseal(data, **changes):
    """Re-record the files' hashes (and apply manifest changes), as a pipeline
    that wrote these exact bytes would have - so only the content checks can object."""
    m = _manifest(data)
    m["artifact"]["sha256"] = hashlib.sha256((data / PRED).read_bytes()).hexdigest()
    m["schedule"]["sha256"] = hashlib.sha256((data / SCHED).read_bytes()).hexdigest()
    for k, v in changes.items():
        if v is None:
            m.pop(k, None)
        else:
            m[k] = v
    (data / MANIFEST).write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _has(errors, *needles):
    text = "\n".join(errors)
    return all(n in text for n in needles)


def _final(r):
    return r["gd"].final_score_mask(r["raw"], r["now"] or datetime.fromisoformat(
        _manifest(r["data"])["settled_as_of"].replace("Z", "+00:00"))).to_numpy()


# --------------------------------------------------------- valid output --

class TestValidOutputs:
    def test_settled_and_pending_games_pass(self, sameday):
        errors, info = _check(sameday["data"])
        assert errors == []
        out, final = sameday["out"], _final(sameday)
        assert info["final"] == final.sum() > 0 and info["pending"] == (~final).sum() > 0
        assert info["flagged_pending"] >= 2                      # flagged games with no result yet
        assert out.loc[sameday["week9_day1"], "spread_bet_return"].isna().all()   # scored, same day

    def test_tonights_run_passes_as_the_current_run(self, tonight):
        m = _manifest(tonight["data"])
        assert m["ci_run"]["GITHUB_RUN_ID"] == RUN["GITHUB_RUN_ID"] and m["code_revision"] == RUN["GITHUB_SHA"]
        assert ps._parse_utc(m["settled_as_of"]) <= ps._parse_utc(m["generated_at"])
        errors, info = _check(tonight["data"], require=True)
        assert errors == [] and info["final"] > 0 and info["pending"] > 0

    def test_command_line_exit_code_and_report(self, tonight, tmp_path):
        report = tmp_path / "gate.md"
        env = {**os.environ, **RUN}
        r = subprocess.run([sys.executable, str(CHECKER), "--require-current-run", "--report", str(report)],
                           cwd=tonight["data"].parent, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "pipeline smoke OK" in r.stdout and "passed" in report.read_text(encoding="utf-8")


# ------------------------------------------------ pending-game outcomes --

class TestPendingGames:
    def test_the_former_fictitious_loss_is_rejected(self, sameday, tmp_path):
        """A flagged unplayed game stored as a 0-0 result and a -100 loss (the old
        fillna(0) bug), in a file whose hashes are otherwise in order."""
        data, out, final = _copy(sameday["data"], tmp_path), sameday["out"], _final(sameday)
        i = int(np.flatnonzero((out["pred_spreadCovered_optimal"] == 1).to_numpy() & ~final
                               & sameday["raw"]["home_score"].isna().to_numpy())[0])
        _edit_cells(data / PRED, {(i, "home_score"): 0.0, (i, "away_score"): 0.0, (i, "spreadCovered"): 0,
                                  (i, "spread_bet_return"): -100.0})
        _reseal(data)
        errors, _ = _check(data)
        assert _has(errors, "pending or unverifiable game(s) carry realized outcomes or returns",
                    "spread_bet_return", out.loc[i, "game_id"])

    @pytest.mark.parametrize("column,value", [("spread_bet_return", 0.0), ("underdogCovered", 1),
                                              ("moneyline_bet_return", 0)])
    def test_any_realized_value_on_a_pending_game_is_rejected(self, sameday, tmp_path, column, value):
        data, final = _copy(sameday["data"], tmp_path), _final(sameday)
        i = int(np.flatnonzero(~final)[0])
        _edit_cells(data / PRED, {(i, column): value})
        _reseal(data)
        errors, _ = _check(data)
        assert _has(errors, "carry realized outcomes", column)

    def test_same_day_game_with_scores_must_stay_pending(self, sameday, tmp_path):
        """Toronto same-day policy: scored, valid evidence, dated the settlement
        day -> pending. Settling it anyway is rejected."""
        data, raw = _copy(sameday["data"], tmp_path), sameday["raw"]
        i = int(sameday["week9_day1"][0])
        assert raw.loc[i, ["home_score", "away_score"]].notna().all()
        _edit_cells(data / PRED, {(i, "home_score"): raw.loc[i, "home_score"],
                                  (i, "away_score"): raw.loc[i, "away_score"]})
        _reseal(data)
        assert _has(_check(data)[0], "carry realized outcomes", raw.loc[i, "game_id"])

    def test_the_gate_uses_the_runs_own_clock_across_toronto_midnight(self, sameday, tmp_path):
        """The week-9 day-1 games were pending when the run settled. A clock one
        second before Toronto midnight agrees; one at midnight would have settled
        them, so the same file no longer matches its recorded clock."""
        day = datetime.fromisoformat(str(sameday["raw"].loc[sameday["week9_day1"][0], "gameday"]))
        midnight = datetime(day.year, day.month, day.day, tzinfo=TORONTO) + timedelta(days=1)
        before = _copy(sameday["data"], tmp_path / "before")
        _reseal(before, settled_as_of=ps._iso(midnight - timedelta(seconds=1)))
        assert _check(before)[0] == []
        after = _copy(sameday["data"], tmp_path / "after")
        _reseal(after, settled_as_of=ps._iso(midnight))
        errors, _ = _check(after)
        assert _has(errors, "final game(s) are missing outcome or return fields",
                    sameday["raw"].loc[sameday["week9_day1"][0], "game_id"])


# --------------------------------------------- missing / stale / mismatched --

class TestProvenance:
    @pytest.mark.parametrize("name,needle", [(PRED, "missing"), (SCHED, "missing"), (MANIFEST, "not found")])
    def test_missing_output(self, tonight, tmp_path, name, needle):
        data = _copy(tonight["data"], tmp_path)
        (data / name).unlink()
        errors, _ = _check(data)
        assert errors and _has(errors, needle)

    def test_truncated_predictions_fail_the_recorded_hash(self, tonight, tmp_path):
        data = _copy(tonight["data"], tmp_path)
        blob = (data / PRED).read_bytes()
        (data / PRED).write_bytes(blob[: len(blob) // 2])
        assert _has(_check(data)[0], "does not match the pipeline run")

    def test_truncated_predictions_fail_even_when_recorded(self, tonight, tmp_path):
        """A pipeline that wrote (and hashed) a short file: fewer rows than the schedule."""
        data = _copy(tonight["data"], tmp_path)
        lines = (data / PRED).read_text(encoding="utf-8").splitlines()
        (data / PRED).write_text("\n".join(lines[: len(lines) // 2]) + "\n", encoding="utf-8")
        _reseal(data)
        assert _has(_check(data)[0], "rows, expected")

    def test_malformed_predictions(self, tonight, tmp_path):
        data = _copy(tonight["data"], tmp_path)
        (data / PRED).write_bytes(b"\x00\x01not a table")
        _reseal(data)
        errors, _ = _check(data)
        assert errors and _has(errors, PRED)

    @pytest.mark.parametrize("name", [PRED, SCHED])
    def test_file_changed_after_the_run(self, tonight, tmp_path, name):
        data = _copy(tonight["data"], tmp_path)
        _edit_cells(data / name, {(3, "gametime"): "09:30"})
        assert _has(_check(data)[0], f"{name} does not match the pipeline run")

    def test_outputs_from_another_run_are_stale(self, tonight, tmp_path):
        """A consistent manifest + CSV pair that an earlier run wrote (e.g. the
        committed one) passes provenance, but isn't tonight's."""
        data = _copy(tonight["data"], tmp_path)
        assert _check(data)[0] == []                                   # internally consistent
        errors, _ = _check(data, require=True, env={**RUN, "GITHUB_RUN_ID": "4241"})
        assert _has(errors, "GITHUB_RUN_ID", "outputs from another run")
        errors, _ = _check(data, require=True, env={**RUN, "GITHUB_RUN_ATTEMPT": "2"})
        assert _has(errors, "GITHUB_RUN_ATTEMPT")
        errors, _ = _check(data, require=True, env={**RUN, "GITHUB_SHA": "f" * 40})
        assert _has(errors, "code_revision")

    def test_code_hash_must_match_the_checkout(self, tonight, tmp_path):
        data = _copy(tonight["data"], tmp_path)
        _reseal(data, code_sha256="0" * 64)
        assert _has(_check(data, require=True)[0], "code_sha256 doesn't match")

    def test_current_run_needs_the_ci_identity(self, tonight):
        assert _has(_check(tonight["data"], require=True, env={})[0], "GITHUB_RUN_ID", "not set")

    def test_manifest_without_settlement_clock_is_rejected(self, tonight, tmp_path):
        data = _copy(tonight["data"], tmp_path)
        _reseal(data, settled_as_of=None)
        assert _has(_check(data)[0], "no valid settled_as_of")

    def test_settlement_clock_must_precede_the_manifest(self, tonight, tmp_path):
        data = _copy(tonight["data"], tmp_path)
        m = _manifest(data)
        later = ps._parse_utc(m["generated_at"]) + timedelta(minutes=1)
        _reseal(data, settled_as_of=ps._iso(later))
        assert _has(_check(data, require=True)[0], "settled_as_of", "not within")

    def test_malformed_settlement_clock_is_an_invalid_manifest(self, tonight, tmp_path):
        data = _copy(tonight["data"], tmp_path)
        _reseal(data, settled_as_of="yesterday")
        assert _has(_check(data)[0], "invalid manifest", "settled_as_of")

    def test_rows_out_of_step_with_the_schedule(self, tonight, tmp_path):
        data = _copy(tonight["data"], tmp_path)
        lines = (data / PRED).read_text(encoding="utf-8").split("\n")
        lines[1], lines[2] = lines[2], lines[1]
        (data / PRED).write_text("\n".join(lines), encoding="utf-8")
        _reseal(data)
        assert _has(_check(data)[0], "games differ from the schedule")


# ------------------------------------------- probabilities and results --

class TestValuesAndResults:
    @pytest.mark.parametrize("column,value", [("prob_underdogCovered", 1.5), ("prob_underdogCovered", -0.1),
                                              ("prob_overHit", "inf"), ("prob_underdogWon", "abc")])
    def test_invalid_probability(self, tonight, tmp_path, column, value):
        data = _copy(tonight["data"], tmp_path)
        i = int(np.flatnonzero(tonight["out"][column].notna().to_numpy())[0])
        _edit_cells(data / PRED, {(i, column): value})
        _reseal(data)
        assert _has(_check(data)[0], column)

    def test_invalid_signal(self, tonight, tmp_path):
        data = _copy(tonight["data"], tmp_path)
        _edit_cells(data / PRED, {(0, "pred_spreadCovered_optimal"): 2})
        _reseal(data)
        assert _has(_check(data)[0], "pred_spreadCovered_optimal must be 0 or 1")

    def _final_row(self, r, flagged_return=None):
        out, final = r["out"], _final(r)
        mask = final.copy()
        if flagged_return is not None:
            mask &= (out["spread_bet_return"] == flagged_return).to_numpy()
        return int(np.flatnonzero(mask)[0])

    def test_score_that_disagrees_with_the_schedule(self, tonight, tmp_path):
        data, i = _copy(tonight["data"], tmp_path), self._final_row(tonight)
        _edit_cells(data / PRED, {(i, "home_score"): tonight["out"].loc[i, "home_score"] + 1})
        _reseal(data)
        assert _has(_check(data)[0], "home_score disagrees with the schedule's final-score evidence")

    def test_return_that_disagrees_with_the_result(self, tonight, tmp_path):
        data, i = _copy(tonight["data"], tmp_path), self._final_row(tonight, flagged_return=90.91)
        _edit_cells(data / PRED, {(i, "spread_bet_return"): -100.0})
        _reseal(data)
        assert _has(_check(data)[0], "spread_bet_return disagrees", tonight["out"].loc[i, "game_id"])

    def test_outcome_flag_that_disagrees_with_the_score(self, tonight, tmp_path):
        data, i = _copy(tonight["data"], tmp_path), self._final_row(tonight)
        _edit_cells(data / PRED, {(i, "homeWin"): 1 - int(tonight["out"].loc[i, "homeWin"])})
        _reseal(data)
        assert _has(_check(data)[0], "homeWin disagrees with the final score")

    def test_final_game_missing_its_return(self, tonight, tmp_path):
        data, i = _copy(tonight["data"], tmp_path), self._final_row(tonight)
        _edit_cells(data / PRED, {(i, "spread_bet_return"): ""})
        _reseal(data)
        assert _has(_check(data)[0], "final game(s) are missing outcome or return fields")

    def test_inconsistent_schedule_evidence_unsettles_a_graded_game(self, tonight, tmp_path):
        """The schedule's result no longer equals home - away: the game has no
        valid final evidence, so its stored grading must not stand."""
        data, i = _copy(tonight["data"], tmp_path), self._final_row(tonight)
        _edit_cells(data / SCHED, {(i, "result"): tonight["raw"].loc[i, "result"] + 1})
        _reseal(data)
        assert _has(_check(data)[0], "carry realized outcomes", tonight["out"].loc[i, "game_id"])


# ------------------------------- probabilities, teams and lines vs schedule --

def _valid_line(r):
    line = r["raw"]["spread_line"]
    return (line.notna() & (line != 0)).to_numpy()


def _row(r, mask) -> int:
    rows = np.flatnonzero(mask)
    assert len(rows), "fixture has no such game"
    return int(rows[0])


def _flagged_valid_row(r):
    return _row(r, _valid_line(r) & (r["out"]["pred_spreadCovered_optimal"] == 1).to_numpy())


def _gate(data):
    """The checks exactly as the nightly gate runs them."""
    return _check(data, require=True)[0]


# The three independently reproduced gaps: each output was resealed, so only a
# content check can object - and before the fix none did.
TAMPERS = {
    "blank_flagged_spread_probability": (lambda r: {(_flagged_valid_row(r), "prob_underdogCovered"): ""},
                                         "no finite prob_underdogCovered in [0,1], 1 of them flagged"),
    "spread_line_100": (lambda r: {(_flagged_valid_row(r), "spread_line"): 100.0},
                        "spread_line differs from the schedule"),
    "home_team_ZZZ": (lambda r: {(0, "home_team"): "ZZZ"}, "home_team differs from the schedule"),
}


class TestSourceFieldsAndMarkets:
    @pytest.mark.parametrize("tamper", sorted(TAMPERS))
    def test_reproduced_gap_is_rejected(self, tonight, tmp_path, tamper):
        edits, needle = TAMPERS[tamper]
        data = _copy(tonight["data"], tmp_path)
        cells = edits(tonight)
        _edit_cells(data / PRED, cells)
        _reseal(data)
        errors = _gate(data)
        (row, _), = cells
        assert len(errors) == 1 and needle in errors[0] and tonight["out"].loc[row, "game_id"] in errors[0], errors

    @pytest.mark.parametrize("column,value", [("prob_underdogCovered", "nan"), ("prob_underdogCovered", "inf"),
                                              ("prob_overHit", ""), ("prob_underdogWon", "")])
    def test_missing_probability_where_the_market_is_valid(self, tonight, tmp_path, column, value):
        r = tonight
        mask = _valid_line(r) if column == "prob_underdogCovered" else r["out"][column].notna().to_numpy()
        if column == "prob_underdogCovered":
            mask &= (r["out"]["pred_spreadCovered_optimal"] == 0).to_numpy()   # unflagged games too
        i = _row(r, mask)
        data = _copy(r["data"], tmp_path)
        _edit_cells(data / PRED, {(i, column): value})
        _reseal(data)
        assert _has(_gate(data), f"no finite {column}", r["out"].loc[i, "game_id"])

    @pytest.mark.parametrize("column,value", [("spread_line", "flip"), ("total_line", 99.5),
                                              ("home_moneyline", 999.0), ("away_moneyline", -999.0),
                                              ("home_spread_odds", -120.0), ("away_spread_odds", -120.0),
                                              ("over_odds", -125.0), ("under_odds", -125.0),
                                              ("spread_line", "")])
    def test_line_that_disagrees_with_the_schedule(self, tonight, tmp_path, column, value):
        """Including a missing line where the schedule has one: never filled in."""
        i = _flagged_valid_row(tonight)
        if value == "flip":
            value = -tonight["out"].loc[i, "spread_line"]
        data = _copy(tonight["data"], tmp_path)
        _edit_cells(data / PRED, {(i, column): value})
        _reseal(data)
        assert _has(_gate(data), f"{column} differs from the schedule", tonight["out"].loc[i, "game_id"])

    def test_swapped_teams_are_rejected(self, tonight, tmp_path):
        out = tonight["out"]
        data = _copy(tonight["data"], tmp_path)
        _edit_cells(data / PRED, {(3, "home_team"): out.loc[3, "away_team"], (3, "away_team"): out.loc[3, "home_team"]})
        _reseal(data)
        errors = _gate(data)
        assert _has(errors, "home_team differs", "away_team differs", out.loc[3, "game_id"])

    @pytest.mark.parametrize("column", ["home_team", "away_team", "spread_line", "over_odds", "home_moneyline"])
    def test_missing_output_column_is_rejected(self, tonight, tmp_path, column):
        data = _copy(tonight["data"], tmp_path)
        _drop_column(data / PRED, column)
        _reseal(data)
        assert _has(_gate(data), f"{PRED} missing team/line columns: ['{column}']")

    @pytest.mark.parametrize("column", ["away_team", "over_odds"])
    def test_missing_schedule_column_is_rejected(self, tonight, tmp_path, column):
        data = _copy(tonight["data"], tmp_path)
        _drop_column(data / SCHED, column)
        _reseal(data)
        assert _has(_gate(data), f"{SCHED} missing team/line columns: ['{column}']")

    def test_unposted_lines_are_written_as_zero_and_pass(self, tonight):
        """The pipeline's own normalization on tonight's real output: a line not
        posted yet is NaN in the schedule, 0 in the predictions, with no spread
        probability or signal - and the gate accepts it."""
        raw, out = tonight["raw"], tonight["out"]
        unposted = raw["spread_line"].isna().to_numpy()
        assert unposted.any()
        assert (out.loc[unposted, "spread_line"] == 0).all()
        assert out.loc[unposted, "prob_underdogCovered"].isna().all()
        assert (out.loc[unposted, "pred_spreadCovered_optimal"] == 0).all()
        assert _gate(tonight["data"]) == []

    def test_unposted_line_left_blank_in_the_output_is_rejected(self, tonight, tmp_path):
        i = _row(tonight, tonight["raw"]["spread_line"].isna().to_numpy())
        data = _copy(tonight["data"], tmp_path)
        _edit_cells(data / PRED, {(i, "spread_line"): ""})
        _reseal(data)
        assert _has(_gate(data), "spread_line differs from the schedule", tonight["out"].loc[i, "game_id"])

    def test_probability_on_an_unposted_line_is_rejected(self, tonight, tmp_path):
        i = _row(tonight, tonight["raw"]["spread_line"].isna().to_numpy())
        data = _copy(tonight["data"], tmp_path)
        _edit_cells(data / PRED, {(i, "prob_underdogCovered"): 0.6})
        _reseal(data)
        assert _has(_gate(data), "games without a spread line have a prob_underdogCovered")

    def _pickem(self, r, tmp_path, prob):
        """A pending game turned into a pick'em in both files, as the pipeline writes one."""
        i = _row(r, ~_final(r) & _valid_line(r) & (r["out"]["pred_spreadCovered_optimal"] == 0).to_numpy())
        data = _copy(r["data"], tmp_path)
        _edit_cells(data / SCHED, {(i, "spread_line"): 0.0})
        _edit_cells(data / PRED, {(i, "spread_line"): 0.0, (i, "prob_underdogCovered"): prob})
        _reseal(data)
        return data

    def test_pickem_without_probability_passes(self, tonight, tmp_path):
        assert _gate(self._pickem(tonight, tmp_path, "")) == []

    def test_pickem_with_probability_is_rejected(self, tonight, tmp_path):
        assert _has(_gate(self._pickem(tonight, tmp_path, 0.55)), "games without a spread line have a prob_underdogCovered")

    @pytest.mark.parametrize("written,ok", [(0.0, True), ("inf", False), ("", False)])
    def test_infinite_odds_are_written_as_zero(self, tonight, tmp_path, written, ok):
        """inf in the schedule -> 0 in the predictions, and then no market (no
        prob_overHit). Any other spelling of it is rejected."""
        i = _row(tonight, ~_final(tonight))
        data = _copy(tonight["data"], tmp_path)
        _edit_cells(data / SCHED, {(i, "over_odds"): "inf"})
        _edit_cells(data / PRED, {(i, "over_odds"): written, (i, "prob_overHit"): ""})
        _reseal(data)
        errors = _gate(data)
        assert (errors == []) if ok else _has(errors, "over_odds differs from the schedule"), errors

    def test_no_moneyline_odds_means_no_moneyline_probability(self, tonight, tmp_path):
        i = _row(tonight, ~_final(tonight))
        data = _copy(tonight["data"], tmp_path)
        _edit_cells(data / SCHED, {(i, "home_moneyline"): "", (i, "away_moneyline"): ""})
        _edit_cells(data / PRED, {(i, "home_moneyline"): 0.0, (i, "away_moneyline"): 0.0, (i, "prob_underdogWon"): ""})
        _reseal(data)
        assert _gate(data) == []
        _edit_cells(data / PRED, {(i, "prob_underdogWon"): 0.3})
        _reseal(data)
        assert _has(_gate(data), "without odds for prob_underdogWon carry one")


# --------------------------------------------------- workflow wiring --

PUBLISHING = ("Upload updated predictions", "Commit and push changes",
              "Export best bets for Sports Picks Grid", "Commit best_bets_today.json")
GATE = "Validate pipeline outputs before publishing"


@pytest.fixture(scope="module")
def workflow():
    import yaml
    steps = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["update-predictions"]["steps"]
    return steps, [s["name"] for s in steps]


class TestWorkflowGate:
    def test_gate_is_required_and_runs_before_every_publishing_step(self, workflow):
        steps, names = workflow
        gate = steps[names.index(GATE)]
        assert gate["id"] == "output_gate" and "continue-on-error" not in gate
        assert "check_pipeline_outputs.py --require-current-run" in gate["run"].replace("\\\n", "")
        g = names.index(GATE)
        # Everything that writes tonight's files runs first; everything that publishes after.
        for before in ("Run prediction pipeline", "Capture pregame spread snapshot",
                       "Update the betting-recommendations log"):
            assert names.index(before) < g
        for name in PUBLISHING:
            step = steps[names.index(name)]
            assert names.index(name) > g, name
            assert not re.search(r"\b(always|failure|cancelled)\(\)", step.get("if", "")), name

    def test_no_publishing_step_before_the_gate(self, workflow):
        steps, names = workflow
        for s in steps[: names.index(GATE)]:
            text = json.dumps(s)
            assert "git push" not in text and "upload-artifact" not in text, s["name"]


def _bash():
    if os.name != "nt":
        return shutil.which("bash")
    try:
        exec_path = Path(subprocess.run(["git", "--exec-path"], capture_output=True, text=True,
                                        check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None
    for base in exec_path.parents:
        for cand in (base / "bin" / "bash.exe", base / "usr" / "bin" / "bash.exe"):
            if cand.exists():
                return str(cand)
    return None


def _posix(p: Path) -> str:
    """A path as bash sees it (C:/x -> /c/x under Git for Windows)."""
    s = p.as_posix()
    return f"/{s[0].lower()}{s[2:]}" if os.name == "nt" and s[1:2] == ":" else s


def _condition(expr: str, outcomes: dict, job_failed: bool) -> bool:
    """GitHub's `if:` for the forms this workflow uses: terms joined by &&, and
    an implicit success() unless a status function is present."""
    if "||" in expr:
        raise AssertionError(f"unsupported condition {expr!r}")
    status_fn, ok = False, True
    for term in (t.strip() for t in expr.split("&&") if t.strip()):
        if term in ("always()", "failure()", "success()", "cancelled()"):
            status_fn = True
            ok &= {"always()": True, "failure()": job_failed, "success()": not job_failed,
                   "cancelled()": False}[term]
        elif term in ("steps.season_check.outputs.in_season == 'true'", "github.ref == 'refs/heads/main'"):
            pass
        elif m := re.fullmatch(r"steps\.(\w+)\.outcome == '(\w+)'", term):
            ok &= outcomes.get(m.group(1), "success") == m.group(2)
        else:
            raise AssertionError(f"unsupported condition term {term!r}")
    return ok and (status_fn or not job_failed)


def _simulate(workflow, data_src: Path, tmp_path: Path, env_run: dict):
    """Run the job from the gate onward as Actions would: the gate and both
    commit steps are executed by bash with stand-in `git` (records the call; the
    tree always has staged changes) and `python` (runs only the checker); other
    steps are recorded as run or skipped."""
    bash = _bash()
    if not bash:
        pytest.skip("bash not available")
    steps, names = workflow
    work = tmp_path / "workspace"
    shutil.copytree(data_src, work / "data_files")
    bin_, calls, summary, temp = tmp_path / "bin", tmp_path / "calls.log", tmp_path / "summary.md", tmp_path / "rt"
    bin_.mkdir(), temp.mkdir()
    py, checker = Path(sys.executable).as_posix(), CHECKER.as_posix()
    (bin_ / "python").write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = "scripts/check_pipeline_outputs.py" ]; then shift; exec "{py}" "{checker}" "$@"; fi\n'
        f'echo "python $*" >> "{calls.as_posix()}"\n', encoding="utf-8", newline="\n")
    (bin_ / "git").write_text(
        f'#!/bin/sh\necho "git $*" >> "{calls.as_posix()}"\ncase "$1" in diff) exit 1;; esac\nexit 0\n',
        encoding="utf-8", newline="\n")
    for f in bin_.iterdir():
        f.chmod(0o755)
    env = {**os.environ, **env_run, **SECRETS, "GITHUB_STEP_SUMMARY": summary.as_posix(),
           "RUNNER_TEMP": temp.as_posix(), "PATH": str(bin_) + os.pathsep + os.environ.get("PATH", "")}
    env.pop("PYTHONIOENCODING", None), env.pop("PYTHONUTF8", None)
    executed = {GATE, "Commit and push changes", "Commit best_bets_today.json"}
    outcomes, job_failed, ran, logs = {"pregame_capture": "success"}, False, [], ""
    for step in steps[names.index(GATE):]:
        if not _condition(step.get("if", ""), outcomes, job_failed):
            continue
        ran.append(step["name"])
        code = 0
        if step["name"] in executed:
            assert "${{" not in step["run"]
            script = tmp_path / "step.sh"
            # Git for Windows' bash launcher puts its own git first on PATH; put the
            # stand-ins first inside the script instead (the step body is unchanged).
            script.write_bytes((f'export PATH="{_posix(bin_)}:$PATH"\n'
                                + step["run"].replace("\r\n", "\n")).encode())
            r = subprocess.run([bash, "--noprofile", "--norc", "-e", str(script)], cwd=work, env=env,
                               capture_output=True, text=True, timeout=300)
            code, logs = r.returncode, logs + r.stdout + r.stderr
        if "id" in step:
            outcomes[step["id"]] = "failure" if code else "success"
        if code and not step.get("continue-on-error"):
            job_failed = True
    return dict(ran=ran, failed=job_failed, calls=calls.read_text() if calls.exists() else "",
                summary=summary.read_text(encoding="utf-8") if summary.exists() else "", logs=logs)


class TestWorkflowSimulation:
    def test_valid_outputs_continue_to_publication(self, workflow, tonight, tmp_path):
        r = _simulate(workflow, tonight["data"], tmp_path, RUN)
        assert not r["failed"], r["logs"]
        assert set(PUBLISHING) <= set(r["ran"]) and "Keep the rejected outputs for diagnosis" not in r["ran"]
        assert "git add" in r["calls"] and r["calls"].count("python scripts/publish_commit.py") == 2
        assert "Pipeline output gate: passed" in r["summary"]

    def test_failed_validation_publishes_nothing(self, workflow, tonight, tmp_path):
        """Tonight's own outputs (current run, hashes in order) with one pending
        game stored as a -100 loss: only the settlement check objects."""
        data = _copy(tonight["data"], tmp_path / "src")
        i = int(np.flatnonzero(~_final(tonight))[0])
        _edit_cells(data / PRED, {(i, "spread_bet_return"): -100.0})
        _reseal(data)
        errors, _ = _check(data, require=True)
        assert len(errors) == 1 and "carry realized outcomes" in errors[0]
        r = _simulate(workflow, data, tmp_path, RUN)
        assert r["failed"]
        assert not set(PUBLISHING) & set(r["ran"])
        assert "git add" not in r["calls"] and "publish_commit.py" not in r["calls"]
        assert "Keep the rejected outputs for diagnosis" in r["ran"]
        assert "Pipeline output gate: FAILED" in r["summary"] and "carry realized outcomes" in r["summary"]
        assert "::error title=Pipeline output gate failed::" in r["logs"]

    @pytest.mark.parametrize("tamper", sorted(TAMPERS))
    def test_reproduced_gap_publishes_nothing(self, workflow, tonight, tmp_path, tamper):
        edits, needle = TAMPERS[tamper]
        data = _copy(tonight["data"], tmp_path / "src")
        _edit_cells(data / PRED, edits(tonight))
        _reseal(data)
        r = _simulate(workflow, data, tmp_path, RUN)
        assert r["failed"] and not set(PUBLISHING) & set(r["ran"])
        assert "git add" not in r["calls"] and "publish_commit.py" not in r["calls"]
        assert needle in r["summary"]

    def test_stale_outputs_publish_nothing(self, workflow, tonight, tmp_path):
        r = _simulate(workflow, tonight["data"], tmp_path, {**RUN, "GITHUB_RUN_ID": "9999"})
        assert r["failed"] and "publish_commit.py" not in r["calls"]
        assert "outputs from another run" in r["summary"]

    def test_diagnostics_do_not_expose_credentials(self, workflow, tonight, tmp_path):
        r = _simulate(workflow, tonight["data"], tmp_path, {**RUN, "GITHUB_RUN_ID": "9999"})
        for secret in SECRETS.values():
            assert secret not in r["summary"] and secret not in r["logs"]
