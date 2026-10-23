"""Player-prop Stage 4: one scorer, model bundles with provenance, verified loading
(player_props/model_bundle.py, models.py, predict.py). Models are tiny synthetic
ones trained only inside these tests; no network; data_files/ and
player_props/models/ are never written."""
import json
import os
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from player_props import features as fx
from player_props import model_bundle as mb
from player_props import provenance as pv
from prop_bundle_helpers import publish_test_bundle, synthetic_rows, trained_entry, training_source
from test_player_props_stage2 import _prediction_files, gid, league, snapshot
from test_player_props_stage3 import _players

NAMES = fx.model_features("passing_yards")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _model(names=NAMES, objective="binary:logistic", n=120, seed=0, classes=2):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.random((n, len(names))), columns=names)
    y = rng.integers(0, classes, n) if classes > 2 else (X[names[0]] > 0.5).astype(int)
    m = xgb.XGBClassifier(n_estimators=5, max_depth=2, objective=objective)
    m.fit(X, y)
    return m, X


def _X(n=5, seed=1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.random((n, len(NAMES))), columns=NAMES)
    X["is_home"] = 1.0
    X["days_rest"] = 7.0
    return X


# ---------------------------------------------------------------------------
# The shared scorer
# ---------------------------------------------------------------------------

class TestScorer:
    def test_positive_class_is_over(self):
        # Trained so class 1 = the lead stat is high: high rows must score higher.
        m, X = _model()
        p = mb.score_probabilities(m, X, "passing_yards")
        hi, lo = X[NAMES[0]] > 0.8, X[NAMES[0]] < 0.2
        assert p[hi.to_numpy()].mean() > 0.5 > p[lo.to_numpy()].mean()

    def test_probabilities_are_class_one(self):
        m, X = _model()
        assert (mb.score_probabilities(m, X, "passing_yards") == m.predict_proba(X)[:, 1]).all()

    @pytest.mark.parametrize("edit,message", [
        (lambda X: X[list(reversed(NAMES))], "input columns"),
        (lambda X: X.drop(columns=NAMES[-1]), "input columns"),
        (lambda X: X.assign(extra=1.0), "input columns"),
        (lambda X: X.to_numpy(), "input columns"),
    ])
    def test_input_must_be_the_exact_ordered_matrix(self, edit, message):
        m, X = _model()
        with pytest.raises(mb.ScoringError, match=message):
            mb.score_probabilities(m, edit(X), "passing_yards")

    def test_model_trained_on_another_order_is_refused(self):
        m, _ = _model(names=NAMES[1:] + NAMES[:1])
        with pytest.raises(mb.ScoringError, match="feature names"):
            mb.score_probabilities(m, _X(), "passing_yards")

    def test_model_for_another_prop_is_refused(self):
        m, _ = _model(names=fx.model_features("receptions"))
        with pytest.raises(mb.ScoringError):
            mb.score_probabilities(m, _X(), "passing_yards")

    @pytest.mark.parametrize("objective", ["binary:logitraw", "binary:hinge"])
    def test_unsupported_objective(self, objective):
        m, X = _model(objective=objective)
        with pytest.raises(mb.ScoringError, match="objective"):
            mb.score_probabilities(m, X, "passing_yards")

    def test_multiclass_is_refused(self):
        m, X = _model(objective="multi:softprob", classes=3)
        with pytest.raises(mb.ScoringError):
            mb.score_probabilities(m, X, "passing_yards")

    @pytest.mark.parametrize("output,message", [
        (np.array([[0.2, 0.9]]), "sum to 1"),
        (np.array([[np.nan, 0.5]]), "outside"),
        (np.array([[-0.1, 1.1]]), "outside"),
        (np.array([[0.5, 0.5, 0.0]]), "shape"),
        (np.array([0.5]), "shape"),
    ])
    def test_invalid_outputs_are_refused(self, monkeypatch, output, message):
        m, _ = _model()
        monkeypatch.setattr(m, "predict_proba", lambda X: output)
        with pytest.raises(mb.ScoringError, match=message):
            mb.score_probabilities(m, _X(1), "passing_yards")

    def test_unreadable_model(self):
        with pytest.raises(mb.ScoringError, match="not a readable"):
            mb.score_probabilities(object(), _X(1), "passing_yards")


# ---------------------------------------------------------------------------
# Training uses the scorer prediction serves
# ---------------------------------------------------------------------------

class TestTraining:
    def test_metrics_are_for_the_exact_served_model(self):
        from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
        entry, rows = trained_entry("passing_yards", "starter")
        served = xgb.XGBClassifier()
        served.load_model(bytearray(entry["data"]))
        test = rows[rows["season"] == 2025]
        y = (test["passing_yards"] > 200.0).astype(int)
        p = mb.score_probabilities(served, fx.model_matrix(test, "passing_yards"), "passing_yards")
        assert entry["metrics"]["roc_auc"] == float(roc_auc_score(y, p))
        assert entry["metrics"]["accuracy"] == float(accuracy_score(y, (p >= 0.5).astype(int)))
        # exact-probability metrics: any change to the served probabilities shows here
        assert entry["metrics"]["brier"] == float(brier_score_loss(y, p))
        assert entry["metrics"]["log_loss"] == float(log_loss(y, p, labels=[0, 1]))
        assert entry["metrics"]["scorer"] == mb.SCORER["name"]
        assert "ensemble" not in entry["metrics"]

    def test_target_is_strictly_over_the_line(self):
        from player_props import models
        rows = synthetic_rows("passing_yards", line=200.0)
        rows.loc[rows.index[::4], "passing_yards"] = 200.0          # exactly on the line: not OVER
        entry, _ = models.train_prop_model(rows, NAMES, "passing_yards", "starter", 200.0)
        test = rows[rows["season"] == 2025]
        assert entry["evaluation"]["positives"] == int((test["passing_yards"] > 200.0).sum())
        train = rows[rows["season"] == 2024]
        assert entry["training"]["positives"] == int((train["passing_yards"] > 200.0).sum())

    def test_entry_records_windows_target_and_protocol(self):
        entry, rows = trained_entry("passing_yards", "starter")
        assert entry["training"]["seasons"] == [2024] and entry["training"]["rows"] == 200
        assert entry["evaluation"]["seasons"] == [2025] and entry["evaluation"]["rows"] == 100
        assert entry["training"]["last_date"] < entry["evaluation"]["first_date"]
        assert entry["target"] == {"stat": "passing_yards", "rule": "stat > line"}
        assert entry["line_value"] == fx.PROP_LINES["passing_yards"]["starter"]
        assert "raw model probability" in entry["evaluation"]["protocol"]

    def test_no_lightgbm_model_is_trained(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "lightgbm", None)       # import would fail
        entry, _ = trained_entry("receptions", "starter")
        assert entry["scorer"] == mb.SCORER

    def test_one_season_has_no_out_of_time_evaluation(self):
        from player_props import models
        rows = synthetic_rows("passing_yards", line=200.0)
        rows["season"] = 2025
        entry, reason = models.train_prop_model(rows, NAMES, "passing_yards", "starter", 200.0)
        assert entry is None and "two seasons" in reason

    def test_unverified_stats_train_and_write_nothing(self, tmp_path):
        from player_props import models
        data = league(tmp_path)
        (data / pv.MANIFEST_FILE).unlink()
        out = tmp_path / "models_out"
        result = models.train_all_models(data_dir=data, models_dir=out)
        assert result["published"] is False and "no provenance manifest" in result["problems"][0]
        assert not out.exists()

    def test_too_little_data_publishes_nothing(self, tmp_path):
        from player_props import models
        out = tmp_path / "models_out"
        result = models.train_all_models(data_dir=league(tmp_path), models_dir=out)
        assert result["published"] is False and result["problems"] == ["no model could be trained"]
        assert len(result["skipped"]) == 28 and not out.exists()

    def test_training_source_is_recorded_from_the_verified_stats(self, tmp_path):
        data = league(tmp_path)
        src = training_source(data)
        m = json.loads((data / pv.MANIFEST_FILE).read_text())
        assert src["stats_files"] == {e["file"]: e["sha256"] for e in m["stats"].values()}
        assert src["last_game_date"] == "2026-09-20"
        assert src["feature_definition"] == fx.FEATURE_CONTRACT["feature_definition"]


# ---------------------------------------------------------------------------
# Bundles: publish, load, tamper
# ---------------------------------------------------------------------------

@pytest.fixture
def bundle(tmp_path):
    data = league(tmp_path)
    models_dir = tmp_path / "models"
    path = publish_test_bundle(models_dir, data, specs=(("passing_yards", "starter"),
                                                         ("passing_yards", "good_qb")))
    return models_dir, path, data


def _manifest(path):
    return json.loads((path / mb.MANIFEST_FILE).read_text())


def _rewrite(models_dir, path, manifest):
    """Write a manifest and re-point the pointer at its hash (a forger's edit)."""
    data = json.dumps(manifest, indent=2, sort_keys=True).encode()
    (path / mb.MANIFEST_FILE).write_bytes(data)
    ptr = json.loads((models_dir / mb.POINTER_FILE).read_text())
    ptr["manifest_sha256"] = mb._sha256(data)
    (models_dir / mb.POINTER_FILE).write_text(json.dumps(ptr))


def _problems(models_dir):
    with pytest.raises(mb.BundleError) as exc:
        mb.load_bundle(models_dir, fx.PROP_LINES)
    return " | ".join(exc.value.problems)


class TestBundle:
    def test_round_trip_gives_identical_raw_probabilities(self, bundle):
        models_dir, path, _ = bundle
        entry, rows = trained_entry("passing_yards", "starter", seed=0)
        loaded = mb.load_bundle(models_dir, fx.PROP_LINES)
        X = fx.model_matrix(rows[rows["season"] == 2025], "passing_yards")
        served = xgb.XGBClassifier()
        served.load_model(bytearray(entry["data"]))
        a = mb.score_probabilities(loaded["models"]["passing_yards_starter"]["model"], X, "passing_yards")
        b = mb.score_probabilities(served, X, "passing_yards")
        assert (a == b).all()

    def test_manifest_records_provenance(self, bundle):
        _, path, data = bundle
        m = _manifest(path)
        assert m["scorer"] == mb.SCORER and m["feature_contract"] == fx.FEATURE_CONTRACT
        assert m["training_source"] == training_source(data)
        assert set(m["libraries"]) >= {"python", "xgboost", "numpy", "pandas", "scikit-learn"}
        assert set(m["code"]["files"]) == set(mb.CODE_FILES)
        e = m["models"]["passing_yards_good_qb"]
        assert (e["prop_type"], e["line_type"], e["line_value"]) == ("passing_yards", "good_qb", 225.0)
        assert e["features"] == NAMES
        assert e["sha256"] == mb._sha256((path / "passing_yards_good_qb.json").read_bytes())
        # build time is its own field, separate from the data windows
        assert "built_at" in m and "built_at" not in e["training"]

    def test_metrics_file_matches_the_manifest(self, bundle):
        _, path, _ = bundle
        metrics = json.loads((path / mb.METRICS_FILE).read_text())
        assert metrics == mb._metrics_document(_manifest(path))
        assert {m["model_name"] for m in metrics["models"]} == {"passing_yards_starter", "passing_yards_good_qb"}

    def test_missing_pointer(self, tmp_path):
        assert "no verified model bundle has been published" in _problems(tmp_path)

    @pytest.mark.parametrize("content", ["{", "[]", '{"kind": "x"}',
                                         '{"kind": "player_props_current_bundle", "bundle_id": "../x", '
                                         '"manifest_sha256": "' + "0" * 64 + '"}'])
    def test_malformed_pointer(self, bundle, content):
        models_dir, _, _ = bundle
        (models_dir / mb.POINTER_FILE).write_text(content)
        assert "pointer" in _problems(models_dir) or "malformed" in _problems(models_dir)

    def test_pointer_to_a_missing_bundle(self, bundle):
        models_dir, path, _ = bundle
        os.rename(path, path.with_name("20000101T000000Z-00000000"))
        assert "doesn't exist" in _problems(models_dir)

    def test_manifest_changed_after_publication(self, bundle):
        models_dir, path, _ = bundle
        (path / mb.MANIFEST_FILE).write_text((path / mb.MANIFEST_FILE).read_text() + " ")
        assert "doesn't match the hash recorded in the pointer" in _problems(models_dir)

    def test_missing_or_malformed_manifest(self, bundle):
        models_dir, path, _ = bundle
        (path / mb.MANIFEST_FILE).unlink()
        assert "manifest can't be read" in _problems(models_dir)

    def test_model_file_changed(self, bundle):
        models_dir, path, _ = bundle
        f = path / "passing_yards_starter.json"
        f.write_bytes(f.read_bytes().replace(b'"', b"'", 1))
        assert "passing_yards_starter.json doesn't match its manifest hash" in _problems(models_dir)

    def test_swapped_model_files(self, bundle):
        models_dir, path, _ = bundle
        a, b = path / "passing_yards_starter.json", path / "passing_yards_good_qb.json"
        da, db = a.read_bytes(), b.read_bytes()
        a.write_bytes(db)
        b.write_bytes(da)
        text = _problems(models_dir)
        assert "passing_yards_starter.json doesn't match" in text and "passing_yards_good_qb.json doesn't match" in text

    def test_swapped_identity_with_forged_hashes(self, bundle):
        # A forger swaps the entries too; the names no longer match their line.
        models_dir, path, _ = bundle
        m = _manifest(path)
        s, g = m["models"]["passing_yards_starter"], m["models"]["passing_yards_good_qb"]
        m["models"]["passing_yards_starter"], m["models"]["passing_yards_good_qb"] = (
            {**g, "file": "passing_yards_starter.json"}, {**s, "file": "passing_yards_good_qb.json"})
        _rewrite(models_dir, path, m)
        assert "name doesn't match its prop and line" in _problems(models_dir)

    @pytest.mark.parametrize("field,value,message", [
        ("prop_type", "rushing_yards", "name doesn't match"),
        ("prop_type", "kicking", "unknown prop_type"),
        ("line_type", "elite_qb", "name doesn't match"),
        ("line_type", "superstar", "unknown line_type"),
        ("line_value", 210.0, "line_value"),
        ("features", list(reversed(NAMES)), "features differ"),
        ("features", None, "features differ"),
        ("scorer", {**mb.SCORER, "version": 2}, "unsupported scorer"),
        ("target", {"stat": "passing_yards", "rule": "stat >= line"}, "target"),
        ("file", "../passing_yards_starter.json", "file must be"),
        ("metrics", None, "metrics must be an object"),
    ])
    def test_wrong_model_identity(self, bundle, field, value, message):
        models_dir, path, _ = bundle
        m = _manifest(path)
        m["models"]["passing_yards_starter"][field] = value
        _rewrite(models_dir, path, m)
        assert message in _problems(models_dir)

    @pytest.mark.parametrize("edit,message", [
        (lambda m: m.update(manifest_version=2), "not a version-1"),
        (lambda m: m.update(kind="other"), "not a version-1"),
        (lambda m: m.update(scorer={**mb.SCORER, "version": 2}), "unsupported scorer"),
        (lambda m: m.update(scorer={**mb.SCORER, "name": "ensemble"}), "unsupported scorer"),
        (lambda m: m.update(feature_contract={**fx.FEATURE_CONTRACT, "schema_version": "props-features-0"}),
         "feature contract"),
        (lambda m: m["training_source"].pop("stats_manifest_sha256"), "stats_manifest_sha256"),
        (lambda m: m["training_source"].update(feature_definition="stage0"), "feature_definition"),
        (lambda m: m["training_source"].update(last_game_date="2025-01-01"), "after the training source ends"),
        (lambda m: m["libraries"].update(xgboost="0.0.1"), "this environment has"),
        (lambda m: m.update(built_at="2999-01-01T00:00:00+00:00"), "built_at is in the future"),
        (lambda m: m.update(built_at="yesterday"), "built_at must be"),
        (lambda m: m["models"]["passing_yards_starter"]["metrics"].update(roc_auc=float("nan")),
         "non-finite"),
        (lambda m: m["models"]["passing_yards_starter"]["metrics"].update(scorer="ensemble"),
         "not computed with the bundle scorer"),
        (lambda m: m["models"]["passing_yards_starter"]["training"].update(last_date="2026-01-01"),
         "training rows reach into the evaluation window"),
        (lambda m: m.update(models={}), "non-empty"),
        (lambda m: m["models"]["passing_yards_starter"]["evaluation"].pop("positives"), "positives"),
        (lambda m: m["models"]["passing_yards_starter"]["training"].update(positives=0), "positives"),
        (lambda m: m["models"]["passing_yards_starter"]["evaluation"].update(positives=100), "positives"),
        (lambda m: m["models"]["passing_yards_starter"]["metrics"].pop("brier"), "metrics missing"),
        (lambda m: m["models"]["passing_yards_starter"]["metrics"].pop("log_loss"), "metrics missing"),
    ])
    def test_malformed_or_unsupported_manifest(self, bundle, edit, message):
        models_dir, path, _ = bundle
        m = _manifest(path)
        edit(m)
        _rewrite(models_dir, path, m)
        assert message in _problems(models_dir)

    def test_metrics_that_disagree_with_the_manifest(self, bundle):
        models_dir, path, _ = bundle
        metrics = json.loads((path / mb.METRICS_FILE).read_text())
        metrics["models"][0]["reliable"] = not metrics["models"][0]["reliable"]
        (path / mb.METRICS_FILE).write_text(json.dumps(metrics))
        assert "model_metrics.json doesn't match its manifest hash" in _problems(models_dir)

    def test_metrics_hash_forged_but_content_differs(self, bundle):
        models_dir, path, _ = bundle
        metrics = json.loads((path / mb.METRICS_FILE).read_text())
        metrics["models"][0]["accuracy"] = 0.99
        data = json.dumps(metrics).encode()
        (path / mb.METRICS_FILE).write_bytes(data)
        m = _manifest(path)
        m["metrics_file"]["sha256"] = mb._sha256(data)
        _rewrite(models_dir, path, m)
        assert "doesn't match the manifest's metrics" in _problems(models_dir)

    def test_model_trained_on_another_feature_order_inside_a_bundle(self, tmp_path):
        data = league(tmp_path)
        entry, _ = trained_entry("passing_yards", "starter")
        bad, _ = _model(names=NAMES[1:] + NAMES[:1])
        entry["data"] = bytes(bad.get_booster().save_raw("json"))
        with pytest.raises(mb.BundleError, match="feature names"):
            mb.publish_bundle(tmp_path / "m", {"passing_yards_starter": entry}, training_source(data),
                              fx.PROP_LINES)
        assert not (tmp_path / "m" / mb.POINTER_FILE).exists()


class TestPublication:
    def test_failed_validation_writes_nothing(self, tmp_path):
        data = league(tmp_path)
        entry, _ = trained_entry("passing_yards", "starter")
        entry["line_value"] = 1.0
        with pytest.raises(mb.BundleError, match="bundle not published"):
            mb.publish_bundle(tmp_path / "m", {"passing_yards_starter": entry}, training_source(data),
                              fx.PROP_LINES)
        assert not (tmp_path / "m").exists()

    def test_failed_pointer_switch_keeps_the_previous_bundle(self, bundle, monkeypatch):
        models_dir, old, data = bundle
        real = os.replace

        def fail_on_pointer(src, dst):
            if str(dst).endswith(mb.POINTER_FILE):
                raise OSError("disk full")
            return real(src, dst)
        monkeypatch.setattr(mb.os, "replace", fail_on_pointer)
        with pytest.raises(OSError, match="disk full"):
            publish_test_bundle(models_dir, data, seed=5)
        monkeypatch.setattr(mb.os, "replace", real)
        assert mb.load_bundle(models_dir, fx.PROP_LINES)["manifest"]["bundle_id"] == old.name
        bundles = sorted(p.name for p in (models_dir / mb.BUNDLES_DIR).iterdir())
        assert bundles == [old.name]                         # the unfinished one is removed
        assert not list(models_dir.glob(".*.tmp"))

    def test_failed_verification_keeps_the_previous_bundle(self, bundle, monkeypatch):
        models_dir, old, data = bundle
        monkeypatch.setattr(mb, "verify_bundle_dir",
                            lambda *a, **k: (_ for _ in ()).throw(mb.BundleError(["bad write"])))
        with pytest.raises(mb.BundleError, match="bad write"):
            publish_test_bundle(models_dir, data, seed=5)
        monkeypatch.undo()
        assert mb.load_bundle(models_dir, fx.PROP_LINES)["manifest"]["bundle_id"] == old.name
        assert [p.name for p in (models_dir / mb.BUNDLES_DIR).iterdir()] == [old.name]

    def test_failed_cleanup_reports_what_was_left(self, bundle, monkeypatch):
        models_dir, old, data = bundle
        monkeypatch.setattr(mb, "verify_bundle_dir",
                            lambda *a, **k: (_ for _ in ()).throw(mb.BundleError(["bad write"])))
        monkeypatch.setattr(mb.shutil, "rmtree", lambda p: (_ for _ in ()).throw(OSError("locked")))
        with pytest.raises(mb.BundleError, match="could not be cleaned up") as exc:
            publish_test_bundle(models_dir, data, seed=5)
        assert ".partial" in str(exc.value) and "previous bundle is still current" in str(exc.value)
        monkeypatch.undo()
        assert mb.load_bundle(models_dir, fx.PROP_LINES)["manifest"]["bundle_id"] == old.name

    def test_a_new_bundle_replaces_the_old_one_atomically(self, bundle):
        models_dir, old, data = bundle
        new = publish_test_bundle(models_dir, data, seed=9)
        assert new != old and old.exists()                   # old bundle kept, not overwritten
        assert mb.load_bundle(models_dir, fx.PROP_LINES)["manifest"]["bundle_id"] == new.name


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class TestPrediction:
    @pytest.fixture
    def setup(self, tmp_path, monkeypatch):
        from player_props import predict
        data = league(tmp_path)
        _prediction_files(data)
        _players(data)
        models_dir = tmp_path / "models"
        monkeypatch.setattr(predict, "DATA_DIR", data)
        monkeypatch.setattr(predict, "MODELS_DIR", models_dir)
        return predict, data, models_dir

    def _run(self, predict):
        return predict.generate_predictions(season=2026, week=3, freeze=True, skip_injuries=True,
                                            skip_weather=True, skip_market_odds=True)

    def test_no_bundle_blocks_before_any_write(self, setup, capsys):
        predict, data, _ = setup
        before = snapshot(data)
        assert self._run(predict) == 1
        assert snapshot(data) == before
        assert "No verified model bundle" in capsys.readouterr().out

    def test_tampered_bundle_blocks_before_any_write(self, setup):
        predict, data, models_dir = setup
        path = publish_test_bundle(models_dir, data)
        f = path / "passing_yards_starter.json"
        f.write_bytes(f.read_bytes() + b" ")
        before = snapshot(data)
        assert self._run(predict) == 1
        assert snapshot(data) == before

    def test_model_trained_on_the_target_weeks_games_is_refused(self, setup, capsys):
        predict, data, models_dir = setup
        path = publish_test_bundle(models_dir, data)
        m = _manifest(path)
        m["training_source"]["last_game_date"] = "2026-09-27"         # the target week's first game
        _rewrite(models_dir, path, m)
        before = snapshot(data)
        assert self._run(predict) == 1
        assert snapshot(data) == before
        assert "on or after this week's first game" in capsys.readouterr().out

    def test_scoring_error_blocks_before_any_write(self, setup, monkeypatch):
        predict, data, models_dir = setup
        publish_test_bundle(models_dir, data)
        monkeypatch.setattr(predict, "score_probabilities",
                            lambda *a, **k: (_ for _ in ()).throw(mb.ScoringError("bad output")))
        before = snapshot(data)
        with pytest.raises(mb.ScoringError):
            self._run(predict)
        assert snapshot(data) == before

    def test_outputs_separate_raw_and_adjusted_probability(self, setup):
        predict, data, models_dir = setup
        path = publish_test_bundle(models_dir, data)
        self._run(predict)
        out = pd.read_csv(data / "player_props_predictions.csv")
        assert set(out["model_name"]) == {"passing_yards_starter"}
        assert set(out["model_bundle"]) == {path.name}
        assert set(out["line_value"]) == {fx.PROP_LINES["passing_yards"]["starter"]}
        # Fixture QBs average ~22 yards vs a 200 line, so the low-average cap
        # applies: recorded only where it changed the raw value.
        capped = out["raw_model_prob_over"] > 0.25
        assert (out.loc[capped, "adjustments"] == "clamp_low_average_cap_0.25").all()
        assert (out.loc[capped, "prob_over"] == 0.25).all()
        assert (out.loc[~capped, "adjustments"] == "none").all()
        assert (out.loc[~capped, "prob_over"] == out.loc[~capped, "raw_model_prob_over"]).all()
        # raw values equal the shared scorer on the shared feature rows
        loaded = mb.load_bundle(models_dir, fx.PROP_LINES)["models"]["passing_yards_starter"]["model"]
        check = pv.check_inputs(2026, 3, data)
        games = pv.load_games(data)
        for r in out.itertuples():
            team = r.team
            game = gid(2026, 3, "KC", "DEN") if team in ("KC", "DEN") else gid(2026, 3, "BUF", "LV")
            target = pd.DataFrame([dict(player_id=f"{team}-QB", player_name=r.player_name, game_id=game,
                                        season=2026, week=3, team=team, opponent=r.opponent,
                                        game_date="2026-09-27")])
            X = fx.model_matrix(fx.inference_rows(check.feature_tables["passing"], target, "passing",
                                                  games), "passing_yards")
            assert r.raw_model_prob_over == pytest.approx(
                float(mb.score_probabilities(loaded, X, "passing_yards")[0]), abs=1e-15)

    def test_adjustment_is_recorded_when_it_changes_the_raw_value(self, setup, monkeypatch):
        predict, data, models_dir = setup
        publish_test_bundle(models_dir, data)
        monkeypatch.setattr(predict, "score_probabilities", lambda *a, **k: np.array([0.9]))
        self._run(predict)
        out = pd.read_csv(data / "player_props_predictions.csv")
        assert (out["raw_model_prob_over"] == 0.9).all() and (out["prob_over"] == 0.25).all()
        assert set(out["adjustments"]) == {"clamp_low_average_cap_0.25"}

    def test_no_fallback_to_another_tier(self, setup):
        # Only star_wr is published: elite_wr players get nothing, not star_wr's model.
        predict, data, models_dir = setup
        publish_test_bundle(models_dir, data, specs=(("receiving_yards", "star_wr"),))
        models = predict.load_models()
        assert list(models) == ["receiving_yards_star_wr"]
        assert "receiving_yards_elite_wr" not in models

    def test_production_models_remain_refused(self):
        from player_props import predict
        assert predict.load_models() == {}
        with pytest.raises(mb.BundleError, match="no verified model bundle has been published"):
            predict.load_model_bundle()


class TestOneDefinition:
    def test_one_set_of_lines(self):
        from player_props import models, predict
        assert predict.PROP_LINES is fx.PROP_LINES and models.PROP_LINES is fx.PROP_LINES

    def test_page_scores_no_unverified_model(self):
        text = (Path(pv.ROOT) / "pages" / "2_Player_Props.py").read_text(encoding="utf-8")
        assert "predict_proba" not in text and "load_model(" not in text


# ---------------------------------------------------------------------------
# Injury and weather adjustments are recorded (real helpers, synthetic data)
# ---------------------------------------------------------------------------

def _injuries(rows):
    from player_props.injuries import INJURY_COLUMNS
    return pd.DataFrame([{**dict.fromkeys(INJURY_COLUMNS, ""), **r} for r in rows])


class TestAdjustmentProvenance:
    """The prediction path with injuries.adjust_prediction_for_injury and
    weather.adjust_for_weather themselves. Only the ESPN and Open-Meteo fetches
    are replaced with synthetic data."""

    WEATHER = {"is_dome": False, "wind_mph": 20.0, "temp_f": 50.0, "precipitation_mm": 0.0,
               "humidity_pct": 40.0}

    @pytest.fixture
    def setup(self, tmp_path, monkeypatch):
        from player_props import predict
        data = league(tmp_path)
        _prediction_files(data)
        _players(data)
        sched = pd.read_csv(data / "nfl_schedule_2026.csv").assign(venue="Outdoor Field")
        sched.to_csv(data / "nfl_schedule_2026.csv", index=False)
        models_dir = tmp_path / "models"
        publish_test_bundle(models_dir, data)
        monkeypatch.setattr(predict, "DATA_DIR", data)
        monkeypatch.setattr(predict, "MODELS_DIR", models_dir)
        return predict, data

    def _run(self, predict, monkeypatch, injuries, raw=None, weather=False):
        monkeypatch.setattr(predict, "get_injury_report", lambda *a, **k: injuries)
        monkeypatch.setattr(predict, "get_weather_for_game", lambda *a, **k: dict(self.WEATHER))
        if raw is not None:
            monkeypatch.setattr(predict, "score_probabilities", lambda *a, **k: np.array([raw]))
        assert predict.generate_predictions(season=2026, week=3, freeze=False, skip_injuries=False,
                                            skip_weather=not weather, skip_market_odds=True) is None
        return pd.read_csv(predict.DATA_DIR / "player_props_predictions.csv",
                           float_precision="round_trip").set_index("player_name")

    def test_each_injury_rule_is_recorded(self, setup, monkeypatch):
        from player_props.injuries import adjust_prediction_for_injury
        predict, _ = setup
        out = self._run(predict, monkeypatch, _injuries([
            {"player_name": "KC QB", "status": "questionable", "injury_type": "Ankle"},
            {"player_name": "BUF QB", "status": "doubtful", "injury_type": "Knee"},
            {"player_name": "DEN QB", "status": "", "practice_participation": "limited"},
        ]), raw=0.9)
        for name, label, factor in [("KC.QB", "injury_questionable", 0.8),
                                    ("BUF.QB", "injury_doubtful", 0.7),
                                    ("DEN.QB", "injury_limited_practice", 0.9)]:
            r = out.loc[name]
            assert r["adjustments"] == f"clamp_low_average_cap_0.25;{label}"
            assert r["raw_model_prob_over"] == 0.9
            # the existing calculation: clamp to 0.25, then the helper's own factor
            expected = adjust_prediction_for_injury(
                {"prob_over": 0.25, "confidence": 0.75}, {"status": "questionable"} if factor == 0.8
                else {"status": "doubtful"} if factor == 0.7 else {"practice_participation": "limited"})
            assert r["prob_over"] == expected["prob_over"] == 0.25 * factor
        assert out.loc["LV.QB", "adjustments"] == "clamp_low_average_cap_0.25"   # no injury listed

    def test_injury_without_a_clamp_is_recorded(self, setup, monkeypatch):
        predict, _ = setup
        # raw 0.2 is below the 0.25 cap, so the injury is the only change
        out = self._run(predict, monkeypatch, _injuries([
            {"player_name": "KC QB", "status": "questionable", "injury_type": "Ankle"}]), raw=0.2)
        r = out.loc["KC.QB"]
        assert r["adjustments"] == "injury_questionable"
        assert r["raw_model_prob_over"] == 0.2 and r["prob_over"] == 0.2 * 0.8

    def test_status_rule_takes_precedence_over_practice(self, setup, monkeypatch):
        # The helper applies the questionable factor (0.8), not limited practice (0.9).
        predict, _ = setup
        out = self._run(predict, monkeypatch, _injuries([
            {"player_name": "KC QB", "status": "questionable", "practice_participation": "limited",
             "injury_type": "Ankle"}]), raw=0.2)
        r = out.loc["KC.QB"]
        assert r["adjustments"] == "injury_questionable" and r["prob_over"] == 0.2 * 0.8

    @pytest.mark.parametrize("info", [
        {"status": "probable"}, {"status": "active"}, {"status": "", "practice_participation": "full"},
        {"status": "", "practice_participation": ""},
    ])
    def test_no_op_injury_claims_no_adjustment(self, setup, monkeypatch, info):
        predict, _ = setup
        out = self._run(predict, monkeypatch, _injuries([{"player_name": "KC QB", **info}]), raw=0.2)
        r = out.loc["KC.QB"]
        assert r["adjustments"] == "none" and r["prob_over"] == r["raw_model_prob_over"] == 0.2

    @pytest.mark.parametrize("info", [{"status": "out"}, {"status": "ir"},
                                      {"status": "", "practice_participation": "out"}])
    def test_out_players_stay_excluded(self, setup, monkeypatch, info):
        predict, _ = setup
        out = self._run(predict, monkeypatch, _injuries([{"player_name": "KC QB", **info}]), raw=0.2)
        assert "KC.QB" not in out.index and {"BUF.QB", "DEN.QB", "LV.QB"} <= set(out.index)

    def test_clamp_injury_and_weather_are_all_recorded(self, setup, monkeypatch):
        from player_props.weather import adjust_for_weather
        predict, _ = setup
        out = self._run(predict, monkeypatch, _injuries([
            {"player_name": "KC QB", "status": "questionable", "injury_type": "Ankle"}]),
            raw=0.9, weather=True)
        r = out.loc["KC.QB"]
        assert r["adjustments"] == "clamp_low_average_cap_0.25;injury_questionable;weather"
        assert r["raw_model_prob_over"] == 0.9
        assert r["prob_over"] == adjust_for_weather(0.25 * 0.8, "passing_yards", self.WEATHER)
        other = out.loc["BUF.QB"]                        # no injury: clamp then weather
        assert other["adjustments"] == "clamp_low_average_cap_0.25;weather"
        assert other["prob_over"] == adjust_for_weather(0.25, "passing_yards", self.WEATHER)
        assert (out["raw_model_prob_over"] == 0.9).all()
