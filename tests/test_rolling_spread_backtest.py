"""scripts/rolling_spread_backtest.py - leak-free features, chronological
windows, original-line settlement, missing lines / pushes, determinism.

All tests use a small synthetic schedule (no repo data, no network) and a
tiny tree count so the suite stays fast.
"""

import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "rolling_spread_backtest",
    Path(__file__).resolve().parent.parent / "scripts" / "rolling_spread_backtest.py",
)
rsb = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rsb   # @dataclass resolves annotations via sys.modules
_spec.loader.exec_module(rsb)

FEATS = ["awayTeamSpreadCoveredPct", "awayTeamWinPct", "homeTeamAvgPointDiff",
         "homeTeamSpreadCoveredPct", "homeTeamWinPct", "spread_line"]


def _schedule(seasons=(2019, 2020, 2021), weeks=8, teams=8, seed=0):
    """Round-robin-ish synthetic schedule: teams/2 games per week."""
    rng = np.random.default_rng(seed)
    names = [f"T{i}" for i in range(teams)]
    rows, day = [], date(2019, 9, 5)
    for season in seasons:
        for week in range(1, weeks + 1):
            order = rng.permutation(names)
            for g in range(teams // 2):
                home, away = order[2 * g], order[2 * g + 1]
                spread = float(rng.choice([-7, -3.5, -2.5, 1.5, 3, 4.5, 6.5]))
                margin = int(round(spread + rng.normal(0, 13)))
                rows.append(dict(
                    game_id=f"{season}_{week:02d}_{away}_{home}", season=season, week=week,
                    gameday=(day + timedelta(days=g % 2)).isoformat(),
                    gametime="13:00" if g % 2 else "20:15",
                    home_team=home, away_team=away,
                    home_score=float(20 + max(margin, -20)), away_score=20.0,
                    spread_line=spread, total_line=44.5,
                    away_spread_odds=-110.0, home_spread_odds=-110.0))
            day += timedelta(days=7)
        day += timedelta(days=150)
    return pd.DataFrame(rows)


def _cfg(**kw):
    base = dict(start=(2021, 1), end=(2021, 8), seed=7, n_boot=40, cal_min_games=20,
                min_fit_games=30, n_estimators=5, features=FEATS)
    base.update(kw)
    return rsb.Config(**base)


def _features(raw):
    return rsb.build_features(rsb.prepare_games(raw), FEATS)


# ---------------------------------------------------------------- leakage --

class TestLeakFreeFeatures:
    def test_current_and_future_outcomes_cannot_change_earlier_features(self):
        raw = _schedule()
        before = _features(raw)
        key = raw["season"] * 100 + raw["week"]
        cut = 202003
        changed = raw.copy()
        rng = np.random.default_rng(99)
        later = key >= cut
        changed.loc[later, "home_score"] = rng.integers(0, 50, later.sum()).astype(float)
        changed.loc[later, "away_score"] = rng.integers(0, 50, later.sum()).astype(float)
        after = _features(changed)
        upto = (key <= cut).to_numpy()   # includes the changed week itself
        pd.testing.assert_frame_equal(before[upto], after[upto])
        # Not vacuous: the change does propagate to strictly later weeks.
        assert not before[key > cut].equals(after[key > cut])

    def test_feature_is_mean_of_strictly_earlier_home_games(self):
        raw = pd.DataFrame([
            dict(game_id="g1", season=2020, week=1, home_team="A", away_team="B", home_score=24, away_score=10),
            dict(game_id="g2", season=2020, week=2, home_team="A", away_team="C", home_score=3, away_score=17),
            dict(game_id="g3", season=2020, week=3, home_team="A", away_team="D", home_score=30, away_score=0),
            dict(game_id="g4", season=2020, week=4, home_team="A", away_team="B", home_score=np.nan, away_score=np.nan),
        ]).assign(gameday="2020-09-10", gametime="13:00", spread_line=3.0)
        f = rsb.build_features(rsb.prepare_games(raw), ["homeTeamWinPct", "homeTeamGamesPlayed"])
        assert f["homeTeamWinPct"].tolist() == [0.0, 1.0, 0.5, pytest.approx(2 / 3)]
        assert f["homeTeamGamesPlayed"].tolist() == [0, 1, 2, 3]

    def test_score_derived_feature_is_refused(self):
        with pytest.raises(ValueError):
            rsb.build_features(rsb.prepare_games(_schedule()), ["total"])


# ---------------------------------------------------------------- windows --

class TestChronologicalWindows:
    def test_pool_and_calibration_precede_evaluation_week(self):
        df = rsb.prepare_games(_schedule())
        for key in (202001, 202105):
            pool, cutoff = rsb.training_pool(df, key)
            assert (df.loc[pool, "key"] < key).all()
            assert (df.loc[pool, "kickoff"] < cutoff).all()
            fit, cal = rsb.split_fit_calibration(df, pool, 20)
            assert set(fit).isdisjoint(cal) and set(fit) | set(cal) == set(pool)
            assert len(cal) >= 20
            assert df.loc[fit, "key"].max() < df.loc[cal, "key"].min()   # whole weeks, fit strictly earlier

    def test_models_never_see_evaluation_week_games(self, monkeypatch):
        raw = _schedule()
        df = rsb.prepare_games(raw)
        seen = []
        real_prod, real_platt = rsb.fit_production_config, rsb.fit_holdout_platt

        def spy_prod(X, y, cfg):
            seen.append(("prod", set(df.loc[X.index, "key"])))
            return real_prod(X, y, cfg)

        def spy_platt(X_fit, y_fit, X_cal, y_cal, cfg):
            fit_keys, cal_keys = set(df.loc[X_fit.index, "key"]), set(df.loc[X_cal.index, "key"])
            assert max(fit_keys) < min(cal_keys)
            assert set(X_fit.index).isdisjoint(X_cal.index)
            seen.append(("platt", fit_keys | cal_keys))
            return real_platt(X_fit, y_fit, X_cal, y_cal, cfg)

        monkeypatch.setattr(rsb, "fit_production_config", spy_prod)
        monkeypatch.setattr(rsb, "fit_holdout_platt", spy_platt)
        preds, windows, _ = rsb.run_backtest(raw, _cfg(), progress=lambda *_: None)
        evaluated = [w["season"] * 100 + w["week"] for w in windows if "skipped" not in w]
        assert len(evaluated) == 8 and len(seen) == 16
        for i, key in enumerate(evaluated):
            for _, keys in seen[2 * i: 2 * i + 2]:
                assert max(keys) < key
        assert set(preds["key"]) == set(evaluated)

    def test_window_sample_counts_are_recorded(self):
        _, windows, _ = rsb.run_backtest(_schedule(), _cfg(max_weeks=2), progress=lambda *_: None)
        w = windows[0]
        assert w["pool_games"] == w["fit_games"] + w["cal_games"]
        assert w["eval_games"] == 4 and w["cutoff"].startswith("20")


# ------------------------------------------------------- settlement/prices --

class TestSettlementAndPrices:
    @pytest.mark.parametrize("result,odds,profit", [
        ("win", -110, 90.91), ("win", -175, 57.14), ("win", 150, 150.0),
        ("loss", 150, -100.0), ("push", -110, 0.0),
    ])
    def test_profit_per_100_at_price(self, result, odds, profit):
        assert rsb.profit_per_100(result, odds) == pytest.approx(profit)

    def test_devig(self):
        df = rsb.prepare_games(_schedule())
        assert np.allclose(rsb.devig_underdog(df), 0.5)
        assert np.isnan(rsb.american_to_prob([np.nan, 50])).all()

    def test_frozen_report_uses_recorded_line_and_price(self):
        raw = pd.DataFrame([
            dict(game_id="a", home_score=13, away_score=10),   # NE +3.5 recorded, lost by 3 -> win
            dict(game_id="b", home_score=24, away_score=20),   # DAL +4 recorded, lost by 4 -> push
            dict(game_id="c", home_score=27, away_score=24),   # "Pick" row -> unresolved
            dict(game_id="d", home_score=np.nan, away_score=np.nan),
        ])
        log = pd.DataFrame([
            dict(game_id="a", season=2026, bet_type="spread", home_team="SEA", away_team="NE",
                 recommended_team="NE", spread_line=3.0, bet_spread=3.5, bet_odds=150,
                 odds_source="DraftKings quote at log time", model_probability=0.6),
            dict(game_id="b", season=2026, bet_type="spread", home_team="NYG", away_team="DAL",
                 recommended_team="DAL", spread_line=4.0, bet_spread=np.nan, bet_odds=np.nan,
                 odds_source=np.nan, model_probability=0.7),
            dict(game_id="c", season=2026, bet_type="spread", home_team="CLE", away_team="PIT",
                 recommended_team="Pick", spread_line=0.0, model_probability=0.55),
            dict(game_id="d", season=2026, bet_type="spread", home_team="X", away_team="Y",
                 recommended_team="Y", spread_line=3.0, model_probability=0.58),
        ])
        r = rsb.frozen_report(log, raw)
        o = r["overall"]
        assert (o["win"], o["loss"], o["push"], o["profit"]) == (1, 0, 1, 150.0)
        assert o["n"] == 1 and o["pushes_excluded"] == 1            # push out of the probability metrics
        assert o["price_sources"] == {"recorded": 1, "assumed_-110": 1}
        assert r["unresolved_excluded"] == [{"game_id": "c", "recommended_team": "Pick"}]
        assert r["pending_or_unscored"] == 1


# --------------------------------------------------- missing lines / pushes --

def test_missing_lines_excluded_and_pushes_scored_separately():
    raw = _schedule()
    wk = (raw.season == 2021) & (raw.week == 2)
    first, second = raw.index[wk][:2]
    raw.loc[first, "spread_line"] = np.nan                        # missing line
    raw.loc[second, ["spread_line", "home_score", "away_score"]] = [3.0, 23.0, 20.0]   # home -3 wins by 3: push
    preds, _, excluded = rsb.run_backtest(raw, _cfg(), progress=lambda *_: None)
    assert raw.loc[first, "game_id"] not in set(preds["game_id"])
    assert excluded["missing_line"] == 1
    push = preds[preds["game_id"] == raw.loc[second, "game_id"]]
    assert push["push"].all() and (push["profit"] == 0).all()
    s = rsb.summarize(preds[preds["model"] == "constant_50"])
    assert s["pushes_excluded"] >= 1
    assert s["n"] == int((~preds.loc[preds["model"] == "constant_50", "push"]).sum())


# ------------------------------------------------------------ determinism --

def test_deterministic_with_fixed_seed():
    raw, cfg = _schedule(), _cfg(max_weeks=3)
    runs = []
    for _ in range(2):
        preds, windows, excluded = rsb.run_backtest(raw, cfg, progress=lambda *_: None)
        runs.append((preds, rsb.retrospective_report(preds, windows, excluded, cfg)))
    pd.testing.assert_frame_equal(runs[0][0], runs[1][0])
    assert runs[0][1] == runs[1][1]
    other = rsb.bootstrap_by_week(runs[0][0], cfg.n_boot, seed=cfg.seed + 1)
    assert other != runs[0][1]["uncertainty"]
