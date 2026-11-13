"""Spread-only non-push fitting and explicit conditional output semantics.

Synthetic schedules and temporary directories only; no production data reads.
"""
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV

import pregame_snapshots as ps
from test_team_features import _FastClassifier, _load, _schedule
from test_pregame_snapshots import _write_inputs, _manifest, ROWS, PROBS, SIGNALS, T0


@pytest.mark.parametrize("played,valid,push,expected", [
    (True, True, 0, True), (True, True, 1, False),
    (False, True, 0, False), (True, False, 0, False),
    (False, False, 1, False), (True, True, np.nan, False),
])
def test_spread_eligibility(played, valid, push, expected):
    gd = _load("spread_nonpush_unit", "nfl-gather-data.py")
    actual = gd.spread_nonpush_mask(pd.Series([played]), pd.Series([valid]), pd.Series([push]))
    assert bool(actual.iloc[0]) is expected


@pytest.fixture(scope="module")
def nonpush_run(tmp_path_factory):
    data = tmp_path_factory.mktemp("spread_nonpush")
    raw = _schedule(seasons=(2018, 2019, 2020, 2021), weeks=12, unplayed_from=(2021, 9))
    # Deliberate pushes in several chronological slices, including an away favorite.
    raw.loc[[0, 20, 70, 110], ["spread_line", "home_score", "away_score"]] = [3.0, 24.0, 21.0]
    raw.loc[90, ["spread_line", "home_score", "away_score"]] = [-3.0, 21.0, 24.0]
    raw["total"] = raw.home_score + raw.away_score
    raw["result"] = raw.home_score - raw.away_score
    raw["overtime"] = np.where(raw.home_score.notna(), 0, np.nan)
    raw.to_csv(data / "nfl_games_historical.csv", sep="\t", index=False)
    gd = _load("spread_nonpush_pipeline", "nfl-gather-data.py")
    split_inputs, fit_inputs = [], []
    real_split = gd.temporal_split_3way
    def record_split(X, y, **kwargs):
        split_inputs.append(X.index.copy())
        return real_split(X, y, **kwargs)
    class RecordingCalibrator(CalibratedClassifierCV):
        def fit(self, X, y, **kwargs):
            fit_inputs.append(X.index.copy())
            return super().fit(X, y, **kwargs)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gd, "DATA_DIR", str(data) + "/")
        mp.setattr(gd, "XGBClassifier", _FastClassifier)
        mp.setattr(gd, "_LGBM_AVAILABLE", False)
        mp.setattr(gd, "temporal_split_3way", record_split)
        mp.setattr(gd, "CalibratedClassifierCV", RecordingCalibrator)
        # Feature-search scoring isn't under test; keep this fixture bounded.
        mp.setattr(gd, "cross_val_score", lambda *a, **k: np.array([0.5, 0.5, 0.5]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gd.main(now=datetime(2026, 11, 13, tzinfo=timezone.utc))
    out = pd.read_csv(data / "nfl_games_historical_with_predictions.csv", sep="\t")
    return raw, out, split_inputs, fit_inputs, data


def _expected(raw):
    base = raw.home_score.notna() & raw.away_score.notna() & raw.spread_line.notna() & raw.spread_line.ne(0)
    margin = raw.home_score - raw.away_score
    push = np.where(raw.spread_line > 0, margin, -margin) == raw.spread_line.abs()
    return base, pd.Series(push, index=raw.index)


def test_pushes_excluded_from_all_spread_splits_and_calibrator_fit(nonpush_run):
    raw, _, splits, fits, _ = nonpush_run
    base, push = _expected(raw)
    expected = raw.index[base & ~push]
    assert push.sum() >= 5
    assert splits[0].equals(expected)
    k = len(expected) - 2 * int(np.ceil(0.2 * len(expected)))
    assert fits[0].equals(expected[:k])
    assert not push.loc[fits[0]].any()


def test_other_targets_keep_pushes_and_previous_sample(nonpush_run):
    raw, _, splits, _, _ = nonpush_run
    base, push = _expected(raw)
    assert splits[1].equals(raw.index[base])
    assert splits[2].equals(raw.index[base])
    assert push.loc[splits[1]].any()


def test_pushes_still_have_predictions_and_correct_settlement(nonpush_run):
    raw, out, _, _, _ = nonpush_run
    base, push = _expected(raw)
    rows = out.loc[base & push]
    assert rows.prob_underdogCovered.notna().all()
    assert rows.spreadPush.eq(1).all()
    assert rows.underdogCovered.eq(0).all()
    assert rows.spread_bet_return.eq(0).all()


def test_output_labels_and_conditional_ev(nonpush_run):
    _, out, _, _, data = nonpush_run
    assert out.spread_probability_basis.eq("conditional_no_push").all()
    assert out.spread_ev_basis.eq("per_100_non_push_stakes_assumed_minus110").all()
    np.testing.assert_allclose(out.ev_spread, out.prob_underdogCovered * 90.91 - (1-out.prob_underdogCovered)*100, equal_nan=True)
    m = json.loads((data / ps.MANIFEST_NAME).read_text())
    metrics = json.loads((data / "model_metrics.json").read_text())
    assert m["config"]["spread_probability_basis"] == "conditional_no_push"
    assert m["config"]["spread_ev_basis"] == "per_100_non_push_stakes_assumed_minus110"
    assert metrics["Spread Probability Basis"] == "conditional_no_push"
    assert "no push" in metrics["Spread Note"]
    assert "non-push" in metrics["Spread EV Note"]


@pytest.mark.parametrize("conditional", [False, True])
def test_old_and_new_snapshot_meanings(tmp_path, conditional):
    sched, preds = _write_inputs(tmp_path, ROWS, PROBS, SIGNALS)
    m = _manifest(tmp_path)
    if conditional:
        m["config"]["spread_probability_basis"] = ps.CONDITIONAL_SPREAD_BASIS
        m["config_id"] = ps._short_hash(m["config"])
    doc = ps.build_snapshot(m, sched, preds, T0)
    assert doc["probability"] == (ps.CONDITIONAL_PROBABILITY_MEANING if conditional else ps.PROBABILITY_MEANING)
    ps.validate_snapshot(doc)
    if conditional:
        doc["probability"] = ps.PROBABILITY_MEANING
        doc[ps.CHECKSUM_FIELD] = ps.payload_checksum(doc)
        with pytest.raises(ps.SnapshotValidationError, match="probability meaning"):
            ps.validate_snapshot(doc)


def test_unknown_basis_is_refused(tmp_path):
    _write_inputs(tmp_path, ROWS, PROBS, SIGNALS)
    m = _manifest(tmp_path)
    m["config"]["spread_probability_basis"] = "invented"
    m["config_id"] = ps._short_hash(m["config"])
    with pytest.raises(ps.SnapshotValidationError, match="unsupported spread_probability_basis"):
        ps.validate_manifest(m)


def test_integer_and_half_point_push_eligibility():
    gd = _load("spread_nonpush_lines", "nfl-gather-data.py")
    lines = pd.Series([3.0, 3.5, -3.0, -3.5])
    home = pd.Series([24, 24, 21, 21])
    away = pd.Series([21, 21, 24, 24])
    favorite_margin = np.where(lines > 0, home-away, away-home)
    pushes = pd.Series((favorite_margin == lines.abs()).astype(int))
    result = gd.spread_nonpush_mask(pd.Series([True]*4), pd.Series([True]*4), pushes)
    assert result.tolist() == [False, True, False, True]
