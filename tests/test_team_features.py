"""team_features.py (shared leak-free team aggregates) and the production
pipeline's use of it in nfl-gather-data.py.

Covers: no current/future outcome can reach a game's features, row order
doesn't matter, same-week and unplayed games are excluded, cold-start and
hand-computed values, missing spread lines never become pick'em
recommendations, and production == backtest feature values.
"""

import importlib.util
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import team_features as tf

ROOT = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _schedule(seasons=(2019, 2020, 2021), weeks=10, teams=8, seed=0, unplayed_from=None):
    """Synthetic nflverse-like schedule. Games in (season, week) >= unplayed_from
    have NaN scores; every 5th of those also has no line."""
    rng = np.random.default_rng(seed)
    names = [f"T{i}" for i in range(teams)]
    rows, day = [], date(2019, 9, 5)
    for season in seasons:
        for week in range(1, weeks + 1):
            order = rng.permutation(names)
            for g in range(teams // 2):
                home, away = order[2 * g], order[2 * g + 1]
                spread = float(rng.choice([-7, -3.5, -2.5, 1.5, 3, 4.5, 6.5]))
                hs = float(rng.integers(3, 40))
                as_ = float(max(hs - round(spread + rng.normal(0, 13)), 0))
                total_line = float(rng.choice([40.5, 44.5, 47.5]))
                row = dict(
                    game_id=f"{season}_{week:02d}_{away}_{home}", season=season, week=week,
                    gameday=(day + timedelta(days=g % 2)).isoformat(),
                    gametime="20:15" if g == 0 else "13:00",
                    home_team=home, away_team=away, home_score=hs, away_score=as_,
                    total=hs + as_, spread_line=spread, total_line=total_line,
                    away_moneyline=150.0, home_moneyline=-170.0,
                    away_spread_odds=-110.0, home_spread_odds=-110.0,
                    under_odds=-110.0, over_odds=-110.0, div_game=0, temp=60.0, wind=5.0,
                    away_rest=7, home_rest=7)
                if unplayed_from and (season, week) >= unplayed_from:
                    row.update(home_score=np.nan, away_score=np.nan, total=np.nan)
                    if g == 0:
                        row["spread_line"] = np.nan   # line not posted yet
                rows.append(row)
            day += timedelta(days=7)
        day += timedelta(days=150)
    return pd.DataFrame(rows)


def _key(df):
    return df["season"] * 100 + df["week"]


# ------------------------------------------------------------- leakage --

class TestNoLeakage:
    def test_current_and_future_outcomes_cannot_change_earlier_features(self):
        raw = _schedule()
        before = tf.compute_team_features(raw)
        cut = 202005
        later = (_key(raw) >= cut).to_numpy()
        rng = np.random.default_rng(1)
        changed = raw.copy()
        changed.loc[later, "home_score"] = rng.integers(0, 50, later.sum()).astype(float)
        changed.loc[later, "away_score"] = rng.integers(0, 50, later.sum()).astype(float)
        changed.loc[later, "total"] = changed.loc[later, "home_score"] + changed.loc[later, "away_score"]
        changed.loc[later, "spread_line"] = -changed.loc[later, "spread_line"]
        after = tf.compute_team_features(changed)
        upto = (_key(raw) <= cut).to_numpy()          # includes the changed week itself
        pd.testing.assert_frame_equal(before[upto], after[upto])
        assert not before[~upto].equals(after[~upto])  # not vacuous: later weeks do move

    def test_same_week_result_is_not_used(self):
        # A wins away in week 1, loses away on Thursday of week 2, plays away again
        # on Sunday of week 2. The Thursday result is the same week -> excluded.
        raw = pd.DataFrame([
            dict(game_id="a", season=2020, week=1, gameday="2020-09-13", gametime="13:00",
                 home_team="B", away_team="A", home_score=0, away_score=30),
            dict(game_id="b", season=2020, week=2, gameday="2020-09-17", gametime="20:15",
                 home_team="C", away_team="A", home_score=30, away_score=0),
            dict(game_id="c", season=2020, week=2, gameday="2020-09-20", gametime="13:00",
                 home_team="D", away_team="A", home_score=np.nan, away_score=np.nan),
        ]).assign(spread_line=3.0)
        f = tf.compute_team_features(raw, ["awayTeamWinPct", "awayTeamGamesPlayed", "awayTeamLast3AvgScore"])
        assert f.loc[2, "awayTeamWinPct"] == 1.0          # 0.5 if Thursday leaked in
        assert f.loc[2, "awayTeamGamesPlayed"] == 1
        assert f.loc[2, "awayTeamLast3AvgScore"] == 30.0

    def test_unplayed_earlier_games_are_not_counted_as_results(self):
        raw = _schedule(unplayed_from=(2021, 4))
        target = raw.index[_key(raw) == 202108]
        gap = _key(raw).between(202104, 202107)           # unplayed weeks before the target week
        with_gap = tf.compute_team_features(raw).loc[target]
        without = raw[~gap]
        no_gap = tf.compute_team_features(without).loc[target]
        pd.testing.assert_frame_equal(with_gap, no_gap)


class TestOrderAndValues:
    def test_row_order_does_not_change_features(self):
        raw = _schedule(unplayed_from=(2021, 6))
        a = tf.compute_team_features(raw).set_index(raw["game_id"])
        shuffled = raw.sample(frac=1, random_state=3).reset_index(drop=True)
        b = tf.compute_team_features(shuffled).set_index(shuffled["game_id"]).loc[a.index]
        pd.testing.assert_frame_equal(a, b)

    def test_hand_computed_values_and_cold_start(self):
        raw = pd.DataFrame([
            dict(game_id="g1", season=2020, week=1, home_team="A", away_team="B", home_score=24, away_score=10, spread_line=3.0),
            dict(game_id="g2", season=2020, week=2, home_team="A", away_team="C", home_score=3, away_score=17, spread_line=-2.0),
            dict(game_id="g3", season=2020, week=3, home_team="A", away_team="D", home_score=30, away_score=0, spread_line=np.nan),
            dict(game_id="g4", season=2020, week=4, home_team="A", away_team="B", home_score=np.nan, away_score=np.nan, spread_line=np.nan),
        ]).assign(gameday="2020-09-10", gametime="13:00")
        f = tf.compute_team_features(raw, ["homeTeamWinPct", "homeTeamGamesPlayed", "homeTeamLast3AvgScore",
                                           "homeTeamFavoredPct", "homeTeamSpreadCoveredPct",
                                           "homeTeamPointDiffTrend"])
        assert f["homeTeamWinPct"].tolist() == [tf.COLD_START_DEFAULT, 1.0, 0.5, pytest.approx(2 / 3)]
        assert f["homeTeamGamesPlayed"].tolist() == [0, 1, 2, 3]
        assert f.loc[3, "homeTeamLast3AvgScore"] == pytest.approx((24 + 3 + 30) / 3)
        # Line-based stats skip g3 (no line). g1: A favored by 3, won by 14 -> favorite
        # covered. g2: C favored by 2, won by 14 -> favorite covered.
        assert f.loc[3, "homeTeamFavoredPct"] == pytest.approx(0.5)
        assert f.loc[3, "homeTeamSpreadCoveredPct"] == pytest.approx(1.0)
        # Trend over A's diffs +14, -14, +30 -> (30 - 14) / 2
        assert f.loc[3, "homeTeamPointDiffTrend"] == pytest.approx(8.0)
        assert f.loc[2, "homeTeamPointDiffTrend"] == 0.0                     # fewer than 3 games

    def test_unknown_or_score_derived_name_is_refused(self):
        with pytest.raises(ValueError):
            tf.compute_team_features(_schedule(), ["total"])
        assert len(tf.TEAM_FEATURES) == 38 and len(set(tf.TEAM_FEATURES)) == 38


# ------------------------------------------------- production pipeline --

class _FastClassifier(LogisticRegression):
    """Stands in for XGBClassifier so the real pipeline runs in seconds."""

    def __init__(self, eval_metric=None, n_estimators=None, max_depth=None, learning_rate=None,
                 random_state=None, n_jobs=None, scale_pos_weight=None):
        super().__init__(max_iter=500)
        self.eval_metric, self.n_estimators, self.max_depth = eval_metric, n_estimators, max_depth
        self.learning_rate, self.random_state, self.n_jobs = learning_rate, random_state, n_jobs
        self.scale_pos_weight = scale_pos_weight

    @property
    def feature_importances_(self):
        return np.abs(self.coef_[0])


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Run nfl-gather-data.main() on a synthetic schedule in a temp DATA_DIR."""
    data = tmp_path_factory.mktemp("pipeline")
    raw = _schedule(seasons=(2018, 2019, 2020, 2021), weeks=12, unplayed_from=(2021, 9))
    raw.to_csv(data / "nfl_games_historical.csv", sep="\t", index=False)
    gd = _load("nfl_gather_data_under_test", "nfl-gather-data.py")
    gd.DATA_DIR = str(data) + "/"
    gd.XGBClassifier = _FastClassifier
    gd._LGBM_AVAILABLE = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gd.main()
    out = pd.read_csv(data / "nfl_games_historical_with_predictions.csv", sep="\t")
    return raw, out


class TestProductionPipeline:
    def test_missing_lines_never_become_pickem_recommendations(self, pipeline_run):
        raw, out = pipeline_run
        no_line = raw["spread_line"].isna().to_numpy()
        assert no_line.sum() >= 3                                   # the fixture really has some
        assert out.loc[no_line, "prob_underdogCovered"].isna().all()
        assert out.loc[no_line, "ev_spread"].isna().all()
        assert out.loc[no_line, "edge_underdog_spread"].isna().all()
        assert (out.loc[no_line, "pred_spreadCovered_optimal"] == 0).all()
        assert (out.loc[no_line, "spread_line"] == 0).all()        # CSV schema unchanged (0-filled)
        lined = ~no_line
        assert out.loc[lined, "prob_underdogCovered"].notna().all()  # every lined game still predicted
        assert (out["pred_spreadCovered_optimal"] == 1).sum() > 0

    def test_production_and_backtest_features_agree(self, pipeline_run):
        raw, out = pipeline_run
        rsb = _load("rolling_spread_backtest_under_test", "scripts/rolling_spread_backtest.py")
        bt = rsb.build_features(rsb.prepare_games(raw), tf.TEAM_FEATURES)
        assert list(out.columns[out.columns.isin(tf.TEAM_FEATURES)]) == tf.TEAM_FEATURES  # column order kept
        np.testing.assert_allclose(out[tf.TEAM_FEATURES].to_numpy(), bt.to_numpy(), rtol=0, atol=1e-12)

    def test_production_features_ignore_current_and_future_outcomes(self, pipeline_run):
        raw, out = pipeline_run
        played = raw["home_score"].notna()
        cut = 202005
        changed = raw.copy()
        later = (_key(raw) >= cut) & played
        changed.loc[later, ["home_score", "away_score"]] = changed.loc[later, ["away_score", "home_score"]].to_numpy()
        f_new = tf.compute_team_features(changed)
        upto = (_key(raw) <= cut).to_numpy()
        np.testing.assert_allclose(out.loc[upto, tf.TEAM_FEATURES].to_numpy(),
                                   f_new.loc[upto, tf.TEAM_FEATURES].to_numpy(), rtol=0, atol=1e-12)
