"""Player-prop Stage 3 train/serve parity (player_props/features.py and its use in
models.py / predict.py). A target game's features built for prediction must equal
its stored training row. Synthetic data only; no network, data_files/ untouched."""
import json
import socket

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from player_props import aggregators as ag
from player_props import features as fx
from player_props import provenance as pv
from test_player_props_stage1 import GAMES, play, qb_season, schedule
from test_player_props_stage2 import _prediction_files, gid, league, snapshot


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


ROLLING = {s: [f"{c}_L{w}" for c in ag.ROLLING_STATS[s] for w in ag.ROLLING_WINDOWS]
           for s in ag.STATS_FILES}


def stage1_tables(pbp=None, sched=None):
    tables, _ = ag.build_player_stats(qb_season() if pbp is None else pbp,
                                      schedule() if sched is None else sched)
    return tables


def reconstruct(table, stat_type, sched, game_id, player_id):
    """The prediction-side row for a stored game: built from only the other
    games (the target withheld), as if the target hadn't been played."""
    target = table[(table["game_id"] == game_id) & (table["player_id"] == player_id)]
    history = table[table["game_id"] != game_id]
    return fx.inference_rows(history, target[fx.TARGET_COLUMNS], stat_type, sched).iloc[0]


def ewm(values, span):
    """Hand-rolled adjust=False exponential mean."""
    alpha, out = 2 / (span + 1), None
    for v in values:
        out = v if out is None else alpha * v + (1 - alpha) * out
    return out


# ---------------------------------------------------------------------------
# Hand-calculated values and parity
# ---------------------------------------------------------------------------

class TestHandCalculated:
    """Game 4 (2025_08_KC_NE, 2025-10-23) after 100, 200 and 300 passing yards."""

    def test_rolling_matches_hand_calculation(self):
        assert ewm([100, 200, 300], 3) == pytest.approx(225.0)
        assert ewm([100, 200, 300], 5) == pytest.approx(1700 / 9)       # 188.888...
        assert ewm([100, 200, 300], 10) == pytest.approx(151.2396694)

    @pytest.mark.parametrize("side", ["training", "inference"])
    def test_both_paths_give_the_hand_values(self, side):
        tables = stage1_tables()
        p = tables["passing"]
        row = (p[p["game_id"] == "2025_08_KC_NE"].iloc[0] if side == "training"
               else reconstruct(p, "passing", schedule(), "2025_08_KC_NE", "P1"))
        assert row["passing_yards_L3"] == pytest.approx(225.0, abs=1e-12)
        assert row["passing_yards_L5"] == pytest.approx(1700 / 9, abs=1e-12)
        assert row["passing_yards_L10"] == pytest.approx(ewm([100, 200, 300], 10), abs=1e-12)
        assert row["days_rest"] == 21          # 28 days since 2025-09-25, capped
        assert row["is_home"] == 1             # KC hosts NE

    @pytest.mark.parametrize("game,rest,home", [
        ("2025_02_KC_LV", 4, 0),      # Thursday: short week, away
        ("2025_04_KC_DEN", 14, 0),    # after a bye
        ("2025_08_KC_NE", 21, 1),     # missed appearances, capped
        ("2025_09_KC_NYJ", 6, 0),
        ("2026_01_KC_SF", 21, 1),     # new season, capped
        ("2026_02_LAR_SF", 10, 0),    # traded to LAR; history carries over
    ])
    def test_days_rest_and_home_on_both_paths(self, game, rest, home):
        p = stage1_tables()["passing"]
        stored = p[p["game_id"] == game].iloc[0]
        built = reconstruct(p, "passing", schedule(), game, "P1")
        assert (stored["days_rest"], stored["is_home"]) == (rest, home)
        assert (built["days_rest"], built["is_home"]) == (rest, home)


class TestParity:
    @pytest.mark.parametrize("stat_type", list(ag.STATS_FILES))
    def test_every_stored_row_is_reconstructed_exactly(self, stat_type):
        table = stage1_tables()[stat_type]
        cols = ROLLING[stat_type] + fx.MATCHUP
        compared = 0
        for r in table.itertuples():
            built = reconstruct(table, stat_type, schedule(), r.game_id, r.player_id)
            stored = table.loc[r.Index, cols].to_numpy(dtype=float)
            got = built[cols].to_numpy(dtype=float)
            assert ((stored == got) | (np.isnan(stored) & np.isnan(got))).all(), (r.game_id, cols)
            compared += 1
        assert compared == len(table) == 7

    def test_zero_denominator_ratios_match(self):
        # A run with no attempt makes ypc NaN; its rolling values must agree.
        plays = qb_season()
        rb = (plays["rusher_player_id"] == "R1") & (plays["game_id"] == "2025_02_KC_LV")
        plays.loc[rb, "rush_attempt"] = 0
        table = stage1_tables(plays)["rushing"]
        assert table.loc[table["game_id"] == "2025_02_KC_LV", "ypc"].isna().all()
        for game in ["2025_04_KC_DEN", "2025_08_KC_NE", "2026_02_LAR_SF"]:
            stored = table[table["game_id"] == game].iloc[0]
            built = reconstruct(table, "rushing", schedule(), game, "R1")
            for c in ROLLING["rushing"]:
                assert (stored[c] == built[c]) or (np.isnan(stored[c]) and np.isnan(built[c])), c

    def test_first_appearance_is_excluded_on_both_paths(self):
        p = stage1_tables()["passing"]
        first = p[p["game_id"] == "2025_01_BUF_KC"]
        rows, report = fx.training_rows(p, "passing_yards")
        assert report == {"rows": 7, "used": 6, "excluded_no_history": 1}
        assert "2025_01_BUF_KC" not in set(rows["game_id"])
        built = fx.inference_rows(p.iloc[0:0], first[fx.TARGET_COLUMNS], "passing", schedule())
        assert not fx.has_history(built, "passing_yards").iloc[0]
        assert built["days_rest"].iloc[0] == ag.DAYS_REST_FIRST

    def test_trade_keeps_pre_trade_history(self):
        p = stage1_tables()["passing"]
        built = reconstruct(p, "passing", schedule(), "2026_02_LAR_SF", "P1")
        kc_only = ewm([100, 200, 300, 250, 280, 310], 3)        # all six KC games
        assert built["passing_yards_L3"] == pytest.approx(kc_only, abs=1e-9)


class TestNoLeakage:
    TARGET = "2025_08_KC_NE"

    def _features(self, table):
        return reconstruct(table, "passing", schedule(), self.TARGET, "P1")[ROLLING["passing"] + fx.MATCHUP]

    def test_later_games_changed_or_deleted_change_nothing(self):
        p = stage1_tables()["passing"]
        base = self._features(p)
        changed = p.copy()
        later = pd.to_datetime(changed["game_date"]) > pd.Timestamp("2025-10-23")
        changed.loc[later, "passing_yards"] = 9999
        pd.testing.assert_series_equal(self._features(changed), base)
        pd.testing.assert_series_equal(self._features(p[~later.to_numpy()]), base)
        # and Stage 1's stored row for the target is unchanged by later games
        plays = qb_season(yards=(100, 200, 300, 250, 999, 999, 999))
        again = stage1_tables(plays)["passing"]
        cols = ROLLING["passing"] + fx.MATCHUP
        pd.testing.assert_series_equal(
            again[again["game_id"] == self.TARGET].iloc[0][cols],
            p[p["game_id"] == self.TARGET].iloc[0][cols])

    def test_target_result_changes_nothing(self):
        plays = qb_season(yards=(100, 200, 300, 5000, 280, 310, 220))
        p = stage1_tables(plays)["passing"]
        assert p.loc[p["game_id"] == self.TARGET, "passing_yards"].iloc[0] == 5000
        pd.testing.assert_series_equal(self._features(p), self._features(stage1_tables()["passing"]))

    def test_history_holding_the_target_result_is_refused(self):
        p = stage1_tables()["passing"]
        target = p[p["game_id"] == self.TARGET]
        with pytest.raises(fx.FeatureContractError, match="target game's result"):
            fx.inference_rows(p, target[fx.TARGET_COLUMNS], "passing", schedule())

    def test_history_on_or_after_the_target_date_is_ignored(self):
        p = stage1_tables()["passing"]
        target = p[p["game_id"] == self.TARGET][fx.TARGET_COLUMNS]
        others = p[p["game_id"] != self.TARGET]
        early = others[pd.to_datetime(others["game_date"]) < pd.Timestamp("2025-10-23")]
        a = fx.inference_rows(others, target, "passing", schedule())
        b = fx.inference_rows(early, target, "passing", schedule())
        pd.testing.assert_frame_equal(a, b)

    def test_thursday_result_does_not_reach_sunday_features(self, tmp_path):
        """Stage 2 policy: the feature source ends before the target week's
        first game, so a target-week result already recorded (a Thursday game)
        never reaches that week's features."""
        sched = pv.load_games(league(tmp_path, through=(2026, 3), completed=(2026, 3)))
        data = tmp_path / "data"
        tables = {s: pd.read_csv(data / n) for s, n in ag.STATS_FILES.items()}
        check = pv.check_inputs(2026, 3, data)
        assert check.ok, check.freshness_problems + check.provenance_problems
        source = check.feature_tables
        qb = "KC-QB"                        # played Thursday-equivalent week-3 game for KC
        p = tables["passing"]
        assert gid(2026, 3, "KC", "DEN") in set(p.loc[p["player_id"] == qb, "game_id"])
        assert gid(2026, 3, "KC", "DEN") not in set(source["passing"]["game_id"])
        # Rebuild KC-QB for week 3 from the feature source and from everything.
        target = pd.DataFrame([dict(player_id=qb, player_name="KC.QB", game_id=gid(2026, 3, "KC", "DEN"),
                                    season=2026, week=3, team="KC", opponent="DEN",
                                    game_date="2026-09-27")])
        from_source = fx.inference_rows(source["passing"], target, "passing", sched)
        with pytest.raises(fx.FeatureContractError, match="target game's result"):
            fx.inference_rows(p, target, "passing", sched)
        expected = p[(p["player_id"] == qb) & (p["game_id"] == gid(2026, 3, "KC", "DEN"))].iloc[0]
        for c in ROLLING["passing"] + fx.MATCHUP:
            assert from_source[c].iloc[0] == pytest.approx(expected[c], abs=1e-12), c


class TestBadInput:
    def _target(self, **change):
        p = stage1_tables()["passing"]
        t = p[p["game_id"] == "2025_08_KC_NE"][fx.TARGET_COLUMNS].copy()
        for k, v in change.items():
            t[k] = v
        return p[p["game_id"] != "2025_08_KC_NE"], t

    def test_duplicate_targets(self):
        h, t = self._target()
        with pytest.raises(fx.FeatureContractError, match="more than once"):
            fx.inference_rows(h, pd.concat([t, t]), "passing", schedule())

    def test_two_games_on_one_date(self):
        h, t = self._target()
        t2 = t.assign(game_id="2025_09_KC_NYJ")
        with pytest.raises(fx.FeatureContractError, match="two games on one date"):
            fx.inference_rows(h[h["game_id"] != "2025_09_KC_NYJ"], pd.concat([t, t2]), "passing", schedule())

    @pytest.mark.parametrize("date", ["2025-13-01", "", "TBD"])
    def test_missing_or_invalid_target_date(self, date):
        h, t = self._target(game_date=date)
        with pytest.raises(fx.FeatureContractError):
            fx.inference_rows(h, t, "passing", schedule())

    def test_invalid_history_date(self):
        h, t = self._target()
        h = h.copy()
        h.loc[h.index[0], "game_date"] = "not a date"
        with pytest.raises(ag.FeatureValidationError, match="invalid game dates"):
            fx.inference_rows(h, t, "passing", schedule())

    def test_missing_target_column(self):
        h, t = self._target()
        with pytest.raises(fx.FeatureContractError, match="missing column"):
            fx.inference_rows(h, t.drop(columns="opponent"), "passing", schedule())

    @pytest.mark.parametrize("edit,message", [
        (lambda s: s[s["game_id"] != "2025_08_KC_NE"], "no schedule game"),
        (lambda s: pd.concat([s, s[s["game_id"] == "2025_08_KC_NE"]]), "more than one row"),
        (lambda s: s.assign(home_team=s["home_team"].where(s["game_id"] != "2025_08_KC_NE", "NYG")),
         "not exactly one side"),
    ])
    def test_home_away_needs_unique_consistent_schedule_evidence(self, edit, message):
        h, t = self._target()
        with pytest.raises(ag.FeatureValidationError, match=message):
            fx.inference_rows(h, t, "passing", edit(schedule()))

    @pytest.mark.parametrize("edit,count", [
        (lambda s: s[s["game_id"] != "2025_08_KC_NE"], 0),
        (lambda s: pd.concat([s, s[s["game_id"] == "2025_08_KC_NE"]]), 2),
    ])
    def test_target_game_must_resolve_uniquely(self, edit, count):
        with pytest.raises(ag.FeatureValidationError, match=f"{count} matching schedule game"):
            fx.resolve_game(edit(schedule()), 2025, 8, "KC", "NE")


class TestIdentity:
    def test_shared_short_name_is_not_mixed(self):
        plays = qb_season()
        other = plays[plays["passer_player_id"] == "P1"].copy()
        other["passer_player_id"] = "P2"            # same name "P.Mahomes", another ID
        other["passing_yards"] = 10
        p = stage1_tables(pd.concat([plays, other]))["passing"]
        a = reconstruct(p, "passing", schedule(), "2025_08_KC_NE", "P1")
        b = reconstruct(p, "passing", schedule(), "2025_08_KC_NE", "P2")
        assert a["passing_yards_L3"] == pytest.approx(225.0)
        assert b["passing_yards_L3"] == pytest.approx(10.0)

    def test_candidates_by_id_and_latest_team(self):
        from player_props import predict
        p = stage1_tables()["passing"]
        # P1's latest appearance is for LAR: not a KC candidate, reported.
        names, report = predict.candidate_players({"passing": p}, "KC")
        assert names == {} and report["latest_elsewhere"] == ["P1"]
        names, report = predict.candidate_players({"passing": p}, "LAR")
        assert names == {"P1": "P.Mahomes"} and report["latest_elsewhere"] == []

    def test_tier_uses_every_team(self):
        from player_props import predict
        p = stage1_tables()["passing"]          # all 7 games (KC and LAR) average 237.1
        assert predict.get_player_performance_tier("P1", "passing_yards", {"passing": p}) == "good_qb"
        assert predict.get_player_performance_tier("nobody", "passing_yards", {"passing": p}) == "starter"


# ---------------------------------------------------------------------------
# Model input contract
# ---------------------------------------------------------------------------

def _rows(prop="passing_yards", n=3):
    return pd.DataFrame({c: np.arange(n, dtype=float) + 1 for c in fx.model_features(prop)}
                        ).assign(is_home=[0, 1, 1][:n], days_rest=[7, 4, 21][:n])


class TestMoreIdentity:
    def test_name_spelling_change_keeps_history(self):
        p = stage1_tables()["passing"]
        early = p["game_id"].isin(["2025_01_BUF_KC", "2025_02_KC_LV"])
        p.loc[early, "player_name"] = "Pat.Mahomes"          # same ID, older spelling
        built = reconstruct(p, "passing", schedule(), "2025_08_KC_NE", "P1")
        assert built["passing_yards_L3"] == pytest.approx(225.0, abs=1e-12)

    def test_tier_crosses_teams(self):
        from player_props import predict
        p = stage1_tables(qb_season(yards=(150, 150, 150, 150, 150, 150, 900)))["passing"]
        # KC games alone average 150 ("starter"); all seven average 257 ("star_qb").
        assert predict.get_player_performance_tier("P1", "passing_yards", {"passing": p}) == "star_qb"

    def test_another_game_on_the_target_date_is_refused(self):
        p = stage1_tables()["passing"]
        target = p[p["game_id"] == "2025_08_KC_NE"][fx.TARGET_COLUMNS]
        history = p[p["game_id"] != "2025_08_KC_NE"].copy()
        clash = history[history["game_id"] == "2025_04_KC_DEN"].assign(
            game_id="2025_08_AAA_KC", game_date="2025-10-23")
        with pytest.raises(fx.FeatureContractError, match="another game for a target player"):
            fx.inference_rows(pd.concat([history, clash]), target, "passing", schedule())


class TestModelMatrix:
    def test_exact_order_regardless_of_input_order(self):
        rows = _rows()
        shuffled = rows[list(reversed(rows.columns))].assign(extra=1)
        X = fx.model_matrix(shuffled, "passing_yards")
        assert list(X.columns) == fx.model_features("passing_yards")
        assert (X.to_numpy() == rows.to_numpy(dtype=float)).all()

    @pytest.mark.parametrize("edit,message", [
        (lambda r: r.drop(columns="pass_tds_L5"), "missing feature column"),
        (lambda r: pd.concat([r, r[["is_home"]]], axis=1), "duplicated column"),
        (lambda r: r.assign(passing_yards_L3=[1.0, np.nan, 3.0]), "non-finite"),
        (lambda r: r.assign(attempts_L5=[1.0, np.inf, 3.0]), "non-finite"),
        (lambda r: r.assign(completions_L3=["1", "x", "3"]), "non-numeric"),
        (lambda r: r.assign(is_home=[True, False, True]), "boolean"),
        (lambda r: r.assign(is_home=[0, 2, 1]), "is_home"),
        (lambda r: r.assign(days_rest=[0, 4, 21]), "days_rest"),
        (lambda r: r.assign(days_rest=[7, 4, 22]), "days_rest"),
        (lambda r: r.assign(days_rest=[7.5, 4, 21]), "days_rest"),
    ])
    def test_bad_input_is_refused(self, edit, message):
        with pytest.raises(fx.FeatureContractError, match=message):
            fx.model_matrix(edit(_rows()), "passing_yards")

    def test_unknown_prop(self):
        with pytest.raises(fx.FeatureContractError, match="unknown prop"):
            fx.model_matrix(_rows(), "passing_yard")

    def test_partial_history_breaks_the_contract(self):
        p = stage1_tables()["passing"]
        p.loc[p["game_id"] == "2025_01_BUF_KC", "passing_yards_L3"] = 5.0
        with pytest.raises(fx.FeatureContractError, match="some but not all rolling values"):
            fx.training_rows(p, "passing_yards")

    def test_training_rejects_non_finite_values(self):
        p = stage1_tables()["passing"]
        p.loc[p["game_id"] == "2025_04_KC_DEN", "attempts_L5"] = np.inf
        with pytest.raises(fx.FeatureContractError, match="non-finite"):
            fx.training_rows(p, "passing_yards")

    def test_feature_lists_match_the_committed_models(self):
        import glob
        from pathlib import Path
        root = Path(pv.ROOT) / "player_props" / "models"
        seen = 0
        for path in glob.glob(str(root / "*.json")):
            name = Path(path).stem
            if name.startswith("model_metrics"):
                continue
            prop = next(p for p in sorted(fx.MODEL_FEATURES, key=len, reverse=True)
                        if name.startswith(p + "_"))
            m = xgb.XGBClassifier()
            m.load_model(path)
            assert m.get_booster().feature_names == fx.model_features(prop), name
            seen += 1
        assert seen >= 20


# ---------------------------------------------------------------------------
# Compatibility boundary
# ---------------------------------------------------------------------------

def _tiny_model(path, prop="passing_yards", order=None):
    names = order or fx.model_features(prop)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((60, len(names))), columns=names)
    X["is_home"] = (X["is_home"] > 0.5).astype(int)
    m = xgb.XGBClassifier(n_estimators=3, max_depth=2)
    m.fit(X, (rng.random(60) > 0.5).astype(int))
    m.save_model(path)


def _metrics(models_dir, entries):
    (models_dir / "model_metrics.json").write_text(json.dumps({"models": entries}))


class TestCompatibility:
    @pytest.mark.parametrize("record,reason", [
        ({}, "no feature contract recorded"),
        ({"feature_contract": None}, "no feature contract recorded"),
        ({"feature_contract": {**fx.FEATURE_CONTRACT, "schema_version": "props-features-0"}},
         "schema_version"),
        ({"feature_contract": {**fx.FEATURE_CONTRACT, "feature_definition": "stage0"}},
         "feature_definition"),
        ({"feature_contract": fx.FEATURE_CONTRACT,
          "features": list(reversed(fx.model_features("passing_yards")))}, "features"),
    ])
    def test_models_without_the_current_contract_are_refused(self, tmp_path, monkeypatch, capsys,
                                                             record, reason):
        from player_props import predict
        _tiny_model(tmp_path / "passing_yards_starter.json")
        entry = {"model_name": "passing_yards_starter",
                 "features": fx.model_features("passing_yards"), **record}
        _metrics(tmp_path, [entry])
        monkeypatch.setattr(predict, "MODELS_DIR", tmp_path)
        assert predict.load_models() == {}
        assert reason in capsys.readouterr().out

    def test_booster_feature_order_must_match(self, tmp_path, monkeypatch):
        from player_props import predict
        names = fx.model_features("passing_yards")
        _tiny_model(tmp_path / "passing_yards_starter.json", order=names[1:] + names[:1])
        _metrics(tmp_path, [{"model_name": "passing_yards_starter", "features": names,
                             "feature_contract": fx.FEATURE_CONTRACT}])
        monkeypatch.setattr(predict, "MODELS_DIR", tmp_path)
        assert predict.load_models() == {}

    def test_compatible_model_is_loaded(self, tmp_path, monkeypatch):
        from player_props import predict
        _tiny_model(tmp_path / "passing_yards_starter.json")
        _metrics(tmp_path, [{"model_name": "passing_yards_starter",
                             "features": fx.model_features("passing_yards"),
                             "feature_contract": fx.FEATURE_CONTRACT}])
        monkeypatch.setattr(predict, "MODELS_DIR", tmp_path)
        assert list(predict.load_models()) == ["passing_yards_starter"]

    def test_committed_models_are_all_refused(self):
        from player_props import predict
        assert predict.load_models() == {}

    def test_training_records_no_contract_without_a_manifest(self, tmp_path):
        from player_props import models
        data = league(tmp_path)
        (data / pv.MANIFEST_FILE).unlink()
        assert models.training_feature_contract(data) is None

    def test_training_records_the_contract_from_a_valid_manifest(self, tmp_path):
        from player_props import models
        data = league(tmp_path)
        contract = models.training_feature_contract(data)
        assert {k: contract[k] for k in fx.FEATURE_CONTRACT} == fx.FEATURE_CONTRACT
        assert len(contract["stats_manifest_sha256"]) == 64

    def test_training_refuses_a_feature_list_that_differs(self):
        from player_props import models
        p = stage1_tables()["passing"]
        with pytest.raises(fx.FeatureContractError, match="features"):
            models.train_prop_model(p, fx.model_features("passing_yards")[:-1], "x",
                                    "passing_yards_starter")


# ---------------------------------------------------------------------------
# Prediction path
# ---------------------------------------------------------------------------

def _players(data):
    rows = [dict(gsis_id=f"{t}-{pos}", short_name=f"{t}.{pos}", display_name=f"{t} {pos}",
                 last_name=pos, position=pos)
            for t in ("KC", "BUF", "DEN", "LV") for pos in ("QB", "RB", "WR")]
    pd.DataFrame(rows).to_csv(data / "players.csv", index=False)


class TestPredictionPath:
    @pytest.fixture
    def setup(self, tmp_path, monkeypatch):
        from player_props import predict
        data = league(tmp_path)
        _prediction_files(data)
        _players(data)
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        monkeypatch.setattr(predict, "DATA_DIR", data)
        monkeypatch.setattr(predict, "MODELS_DIR", models_dir)
        return predict, data, models_dir

    def _run(self, predict, **kw):
        return predict.generate_predictions(season=2026, week=3, freeze=False, skip_injuries=True,
                                            skip_weather=True, skip_market_odds=True, **kw)

    def test_incompatible_models_block_before_any_write(self, setup):
        predict, data, models_dir = setup
        _tiny_model(models_dir / "passing_yards_starter.json")
        _metrics(models_dir, [{"model_name": "passing_yards_starter",
                               "features": fx.model_features("passing_yards")}])
        before = snapshot(data)
        assert self._run(predict) == 1
        assert snapshot(data) == before

    def test_feature_error_blocks_before_any_write(self, setup, monkeypatch):
        predict, data, models_dir = setup
        _tiny_model(models_dir / "passing_yards_starter.json")
        _metrics(models_dir, [{"model_name": "passing_yards_starter",
                               "features": fx.model_features("passing_yards"),
                               "feature_contract": fx.FEATURE_CONTRACT}])
        before = snapshot(data)
        monkeypatch.setattr(predict, "inference_rows",
                            lambda *a, **k: (_ for _ in ()).throw(fx.FeatureContractError("bad")))
        assert self._run(predict) == 1
        assert snapshot(data) == before

    def test_predictions_use_the_shared_feature_rows(self, setup, monkeypatch):
        predict, data, models_dir = setup
        _tiny_model(models_dir / "passing_yards_starter.json")
        _metrics(models_dir, [{"model_name": "passing_yards_starter",
                               "features": fx.model_features("passing_yards"),
                               "feature_contract": fx.FEATURE_CONTRACT}])
        seen = []
        real = fx.model_matrix

        def spy(rows, prop):
            X = real(rows, prop)
            seen.append((rows.index[0], X))
            return X
        monkeypatch.setattr(predict, "model_matrix", spy)
        self._run(predict)
        out = pd.read_csv(data / "player_props_predictions.csv")
        assert set(out["player_name"]) == {"KC.QB", "BUF.QB", "DEN.QB", "LV.QB"}
        # Each QB's features equal an independent reconstruction from the feature source.
        check = pv.check_inputs(2026, 3, data)
        games = pv.load_games(data)
        for player_id, X in seen:
            team = player_id.split("-")[0]
            game = [g for g in (gid(2026, 3, "KC", "DEN"), gid(2026, 3, "BUF", "LV")) if team in g][0]
            home, away = game.split("_")[3], game.split("_")[2]
            target = pd.DataFrame([dict(player_id=player_id, player_name=player_id.replace("-", "."),
                                        game_id=game, season=2026, week=3, team=team,
                                        opponent=away if team == home else home,
                                        game_date="2026-09-27")])
            want = fx.model_matrix(fx.inference_rows(check.feature_tables["passing"], target,
                                                     "passing", games), "passing_yards")
            assert (X.to_numpy() == want.to_numpy()).all(), player_id
            assert X["days_rest"].iloc[0] == 7 and X["is_home"].iloc[0] == (team == home)
        assert len(seen) == 4


def test_stats_round_trip_through_csv_exactly(tmp_path):
    # Stored features must equal the in-memory values bit for bit, or a model
    # would see different inputs in training and prediction.
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.random(5000) * 300, "y": rng.random(5000) / 7})
    df.to_csv(tmp_path / "s.csv", index=False)
    back = ag.read_stats_csv(tmp_path / "s.csv")
    assert (back.to_numpy() == df.to_numpy()).all()


# ---------------------------------------------------------------------------
# History validation (no duplicate or contradictory history rows)
# ---------------------------------------------------------------------------

TARGET_GAME = "2025_08_KC_NE"


def _split(table):
    target = table[table["game_id"] == TARGET_GAME][fx.TARGET_COLUMNS]
    return table[table["game_id"] != TARGET_GAME], target


class TestHistoryValidation:
    def test_clean_history_gives_the_hand_value(self):
        history, target = _split(stage1_tables()["passing"])
        row = fx.inference_rows(history, target, "passing", schedule()).iloc[0]
        assert row["passing_yards_L3"] == pytest.approx(225.0, abs=1e-12)

    def test_repeated_row_is_refused_not_counted_twice(self):
        # Counted twice, the repeated 2025_04_KC_DEN row (300 yards) would move L3 to 262.5.
        history, target = _split(stage1_tables()["passing"])
        dup = pd.concat([history, history[history["game_id"] == "2025_04_KC_DEN"]])
        with pytest.raises(ag.FeatureValidationError, match="repeating a player-game"):
            fx.inference_rows(dup, target, "passing", schedule())

    @pytest.mark.parametrize("stat_type", list(ag.STATS_FILES))
    def test_repeated_rows_are_refused_in_every_table(self, stat_type):
        table = stage1_tables()[stat_type]
        history, target = _split(table)
        with pytest.raises(ag.FeatureValidationError, match="repeating a player-game"):
            fx.inference_rows(pd.concat([history, history.iloc[[0]]]), target, stat_type, schedule())

    def test_repeat_with_different_values_is_refused(self):
        history, target = _split(stage1_tables()["passing"])
        extra = history[history["game_id"] == "2025_04_KC_DEN"].assign(passing_yards=1.0)
        with pytest.raises(ag.FeatureValidationError, match="repeating a player-game"):
            fx.inference_rows(pd.concat([history, extra]), target, "passing", schedule())

    @pytest.mark.parametrize("column,value,message", [
        ("game_date", "2025-09-26", "disagree on season, week or date"),
        ("season", 2024, "disagree on season, week or date"),
        ("week", 5, "disagree on season, week or date"),
        ("team", "NYG", "same two teams"),
        ("opponent", "KC", "same two teams"),
    ])
    def test_conflicting_labels_within_a_game(self, column, value, message):
        # Another player's row for 2025_04_KC_DEN contradicts P1's.
        history, target = _split(stage1_tables()["passing"])
        other = history[history["game_id"] == "2025_04_KC_DEN"].assign(player_id="P9")
        other[column] = value
        with pytest.raises(ag.FeatureValidationError, match=message):
            fx.inference_rows(pd.concat([history, other]), target, "passing", schedule())

    def test_two_games_on_one_date_in_history(self):
        history, target = _split(stage1_tables()["passing"])
        clash = history[history["game_id"] == "2025_04_KC_DEN"].assign(
            game_id="2025_04_KC_XXX", opponent="XXX")
        with pytest.raises(ag.FeatureValidationError, match="two games on one date"):
            fx.inference_rows(pd.concat([history, clash]), target, "passing", schedule())

    @pytest.mark.parametrize("column", ["player_id", "game_id", "team", "game_date"])
    def test_missing_identity_values(self, column):
        history, target = _split(stage1_tables()["passing"])
        history = history.copy()
        history[column] = history[column].astype(object)
        history.loc[history.index[0], column] = None
        with pytest.raises(ag.FeatureValidationError, match="missing identity or date"):
            fx.inference_rows(history, target, "passing", schedule())

    def test_missing_history_columns(self):
        history, target = _split(stage1_tables()["passing"])
        with pytest.raises(ag.FeatureValidationError, match="missing column"):
            fx.inference_rows(history.drop(columns="attempts"), target, "passing", schedule())

    def test_trades_and_seasons_remain_valid_history(self):
        p = stage1_tables()["passing"]
        assert fx.history_problems(p, "passing") == []         # KC and LAR, 2025 and 2026
        built = reconstruct(p, "passing", schedule(), "2026_02_LAR_SF", "P1")
        assert built["passing_yards_L3"] == pytest.approx(ewm([100, 200, 300, 250, 280, 310], 3))

    def test_duplicate_history_stops_prediction_before_writing(self, tmp_path, monkeypatch, capsys):
        from player_props import predict
        data = league(tmp_path)
        _prediction_files(data)
        _players(data)
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        _tiny_model(models_dir / "passing_yards_starter.json")
        _metrics(models_dir, [{"model_name": "passing_yards_starter",
                               "features": fx.model_features("passing_yards"),
                               "feature_contract": fx.FEATURE_CONTRACT}])
        monkeypatch.setattr(predict, "DATA_DIR", data)
        monkeypatch.setattr(predict, "MODELS_DIR", models_dir)
        real = predict.check_inputs

        def with_duplicate(*a, **k):
            result = real(*a, **k)
            assert result.ok
            p = result.feature_tables["passing"]
            result.feature_tables["passing"] = pd.concat([p, p.iloc[[0]]], ignore_index=True)
            return result
        monkeypatch.setattr(predict, "check_inputs", with_duplicate)
        before = snapshot(data)
        assert predict.generate_predictions(season=2026, week=3, freeze=True, skip_injuries=True,
                                            skip_weather=True, skip_market_odds=True) == 1
        assert snapshot(data) == before
        assert "repeating a player-game" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Recorded feature list is required
# ---------------------------------------------------------------------------

NAMES = fx.model_features("passing_yards")


class TestRecordedFeatureList:
    @pytest.mark.parametrize("features,message", [
        (None, "no feature list recorded"),
        ("passing_yards_L3", "must be a list of names"),
        (tuple(NAMES), "must be a list of names"),
        ([1, 2, 3], "must be a list of names"),
        (NAMES[:-1] + [None], "must be a list of names"),
        (NAMES + [NAMES[0]], "repeat names"),
        (NAMES[:-1] + [NAMES[0]], "repeat names"),
        (list(reversed(NAMES)), "!="),
        (NAMES[:-1], "!="),
        ([], "!="),
    ])
    def test_bad_lists_are_refused(self, features, message):
        problems = fx.contract_problems(fx.FEATURE_CONTRACT, features, "passing_yards")
        assert any(message in p for p in problems), problems

    def test_exact_list_is_accepted(self):
        assert fx.contract_problems(fx.FEATURE_CONTRACT, list(NAMES), "passing_yards") == []

    @pytest.mark.parametrize("record", [
        {"feature_contract": fx.FEATURE_CONTRACT},                       # no "features" key
        {"feature_contract": fx.FEATURE_CONTRACT, "features": None},
        {"feature_contract": fx.FEATURE_CONTRACT, "features": "x"},
        {"feature_contract": fx.FEATURE_CONTRACT, "features": NAMES + NAMES[:1]},
    ])
    def test_correct_booster_names_do_not_make_up_for_missing_metadata(self, tmp_path, monkeypatch,
                                                                       record):
        from player_props import predict
        _tiny_model(tmp_path / "passing_yards_starter.json")      # booster names are exactly right
        m = xgb.XGBClassifier()
        m.load_model(tmp_path / "passing_yards_starter.json")
        assert m.get_booster().feature_names == NAMES
        assert predict.model_incompatibility("passing_yards_starter", "passing_yards", record, m)
        _metrics(tmp_path, [{"model_name": "passing_yards_starter", **record}])
        monkeypatch.setattr(predict, "MODELS_DIR", tmp_path)
        assert predict.load_models() == {}

    def test_no_record_at_all(self, tmp_path):
        from player_props import predict
        _tiny_model(tmp_path / "m.json")
        m = xgb.XGBClassifier()
        m.load_model(tmp_path / "m.json")
        problems = predict.model_incompatibility("passing_yards_starter", "passing_yards", None, m)
        assert any("no feature contract" in p for p in problems)
        assert any("no feature list" in p for p in problems)
