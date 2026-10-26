"""Stage 5B preparation: the isolated end-to-end Player Props rehearsal
(player_props/rehearsal.py). Synthetic league, injected downloads, no network;
everything is written under pytest temporary directories."""
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from player_props import features as fx
from player_props import model_bundle as mb
from player_props import rehearsal as rh
from player_props import rehearsal_fixtures as fixtures

ROOT = Path(rh.ROOT)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(directory):
    d = Path(directory)
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()} \
        if d.exists() else {}


def production_state():
    paths = []
    for base in (ROOT / "data_files", ROOT / "player_props" / "models"):
        for p in sorted(base.rglob("*")):
            if "bet_journal" in p.parts:
                continue
            paths.append((str(p.relative_to(ROOT)), p.stat().st_size if p.is_file() else -1,
                          int(p.stat().st_mtime) if p.is_file() else 0))
    return paths


def status(summary):
    return {s["stage"]: s["status"] for s in summary["stages"]}


def run(workspace, hook=None, **kw):
    return rh.run_rehearsal(rh.RehearsalConfig(workspace=workspace, **kw), after_stage=hook)


def run_dir(summary, workspace):
    return Path(workspace).resolve() / "runs" / summary["run_id"]


# ---------------------------------------------------------------------------
# One successful run, inspected by several tests
# ---------------------------------------------------------------------------

SUCCESS_RUN = {}


@pytest.fixture(scope="module")
def success(tmp_path_factory):
    before = production_state()
    ws = tmp_path_factory.mktemp("rehearsal") / "ws"
    summary = rh.run_rehearsal(rh.RehearsalConfig(workspace=ws),
                               after_stage=lambda name, r: SUCCESS_RUN.setdefault("run", r))
    return ws, summary, before


class TestSuccessfulRun:
    def test_every_stage_succeeds(self, success):
        ws, summary, _ = success
        assert summary["status"] == "succeeded"
        assert list(status(summary).values()) == ["ok"] * len(rh.STAGES)
        assert json.loads((run_dir(summary, ws) / "summary.json").read_text()) == summary

    def test_training_and_inference_provenance_stay_distinct(self, success):
        _, s, _ = success
        train, infer = s["coverage"]["training_source"], s["coverage"]["inference_source"]
        assert train["seasons"] == ["2022", "2023", "2024", "2025"]
        assert infer["seasons"] == ["2022", "2023", "2024", "2025", "2026"]
        assert train["sha256"] != infer["sha256"]
        assert s["bundle"]["trained_through"] == train["last_date"] < s["coverage"]["inputs"]["feature_cutoff"]
        for w in s["windows"].values():
            assert w["evaluation"]["seasons"] == [2025] and max(w["training"]["seasons"]) == 2024

    def test_production_training_rules_ran_unweakened(self, success):
        _, s, _ = success
        assert len(s["windows"]) >= 20
        assert all("imbalanced" in r or "rows" in r or "season" in r
                   for r in s["exclusions"]["skipped_models"].values())
        assert all(w["evaluation"]["rows"] >= 40 for w in s["windows"].values())

    def test_outputs_carry_identity_line_raw_and_adjustments(self, success):
        ws, s, _ = success
        out = pd.read_csv(run_dir(s, ws) / "runtime" / "data" / "player_props_predictions.csv",
                          float_precision="round_trip")
        assert set(out["model_bundle"]) == {s["bundle"]["bundle_id"]}
        assert {"model_name", "line_value", "raw_model_prob_over", "adjustments"} <= set(out.columns)
        for r in out.itertuples():
            prop = next(p for p in sorted(fx.PROP_LINES, key=len, reverse=True)
                        if r.model_name.startswith(p + "_"))
            assert r.line_value == fx.PROP_LINES[prop][r.model_name[len(prop) + 1:]]
        assert s["bundle"]["feature_contract"] == fx.FEATURE_CONTRACT
        v = s["outputs"]["verify_outputs"]
        assert v["predictions_verified"] == v["snapshot_rows_verified"] == len(out)
        assert v["models_metric_parity"] == len(s["windows"])
        # Both clamps occur in the synthetic league, so their rules are exercised.
        assert set(v["adjustments"]) == {"none", rh.LOW_LABEL, rh.STAR_LABEL}
        assert v["latest_sha256"] == sha(run_dir(s, ws) / "runtime" / "data" / "player_props_predictions.csv")

    def test_restored_runtime_is_separate_and_identical(self, success):
        ws, s, _ = success
        d = run_dir(s, ws)
        bid = s["bundle"]["bundle_id"]
        assert tree(d / "runtime" / "models" / "bundles" / bid) == tree(d / "train" / "models" / "bundles" / bid)
        assert mb.load_bundle(d / "runtime" / "models", fx.PROP_LINES)["manifest"]["bundle_id"] == bid

    def test_artifact_hashes_are_real(self, success):
        ws, s, _ = success
        d = run_dir(s, ws)
        assert s["artifacts"]["export/bundle.tar"] == sha(d / "export" / "bundle.tar")
        assert s["artifacts"]["runtime/data/player_props_predictions.csv"] == sha(
            d / "runtime" / "data" / "player_props_predictions.csv")
        assert len(s["artifacts"]) > 20

    def test_published_with_provenance(self, success):
        ws, s, _ = success
        pub = Path(ws).resolve() / "published"
        snap = pub / "player_props_predictions_week3_2026.csv"
        record = json.loads((pub / "snapshots.json").read_text())[snap.name]
        assert record == {"run_id": s["run_id"], "sha256": sha(snap), "bundle_id": s["bundle"]["bundle_id"]}
        assert s["outputs"]["publish"]["snapshot_status"] == "frozen"

    def test_no_production_writes(self, success):
        _, _, before = success
        assert production_state() == before

    def test_summary_claims_no_accuracy(self, success):
        _, s, _ = success
        text = json.dumps(s).lower()
        assert "no accuracy" in text and "edge is measured" in text
        assert set(s["exclusions"]["external_inputs_skipped"]) == {"injuries", "weather", "market_odds"}


# ---------------------------------------------------------------------------
# Failures stop downstream stages and leave evidence
# ---------------------------------------------------------------------------

def assert_failed_at(summary, stage, workspace, match=None):
    st = status(summary)
    assert summary["status"] == "failed" and st[stage] == "failed"
    later = rh.STAGES[rh.STAGES.index(stage) + 1:]
    assert all(st[s] == "not_run" for s in later)
    d = run_dir(summary, workspace)
    assert (d / "diagnostics" / "traceback.txt").exists()
    assert json.loads((d / "summary.json").read_text())["status"] == "failed"
    if match:
        err = next(s for s in summary["stages"] if s["stage"] == stage)["error"]
        assert match in json.dumps(err)


class TestFailures:
    def test_missing_source_season(self, tmp_path):
        s = run(tmp_path / "ws", drop_seasons=(2024,))
        assert_failed_at(s, "source_training", tmp_path / "ws", "season 2024: download failed")
        assert not (tmp_path / "ws" / "published").exists()

    def test_missing_source_game(self, tmp_path):
        game = fixtures.game_id(2026, 1, *fixtures.round_robin(1)[0])
        s = run(tmp_path / "ws", drop_games=frozenset({game}))
        assert_failed_at(s, "source_inference", tmp_path / "ws", "missing from the download")

    def test_training_window_leakage_is_refused(self, tmp_path):
        # Target week already played and included in the training source.
        s = run(tmp_path / "ws", train_seasons=[2023, 2024, 2025, 2026], played_target_weeks=3)
        assert_failed_at(s, "source_training", tmp_path / "ws", "not strictly before")

    def test_invalid_stats_provenance(self, tmp_path):
        def corrupt(name, r):
            if name == "aggregate_inference":
                f = r.paths["runtime_data"] / "player_passing_stats.csv"
                f.write_bytes(f.read_bytes() + b"\n")
        s = run(tmp_path / "ws", corrupt)
        assert_failed_at(s, "validate_inputs", tmp_path / "ws", "does not match its manifest hash")

    def test_incompatible_model_contract(self, tmp_path, monkeypatch):
        def change_contract(name, r):
            if name == "train":
                monkeypatch.setattr(mb, "FEATURE_CONTRACT",
                                    {**fx.FEATURE_CONTRACT, "schema_version": "props-features-99"})
        s = run(tmp_path / "ws", change_contract)
        assert_failed_at(s, "export", tmp_path / "ws", "feature contract")

    def test_corrupt_bundle(self, tmp_path):
        def corrupt(name, r):
            if name == "export":
                a = r.info["archive"]
                a.write_bytes(a.read_bytes()[: len(a.read_bytes()) // 3])
        s = run(tmp_path / "ws", corrupt)
        assert_failed_at(s, "restore", tmp_path / "ws")
        d = run_dir(s, tmp_path / "ws")
        assert not (d / "runtime" / "models" / mb.POINTER_FILE).exists()

    def test_swapped_archive_from_another_run_is_refused(self, tmp_path, success):
        # A valid, verifiable archive, but of a different bundle than this run trained.
        other = run_dir(success[1], success[0]) / "export" / "bundle.tar"

        def swap(name, r):
            if name == "export":
                shutil.copyfile(other, r.info["archive"])
        s = run(tmp_path / "ws", swap)
        assert_failed_at(s, "restore", tmp_path / "ws", "differs from the trained bundle")

    def test_tampered_raw_probability_is_caught(self, tmp_path):
        def tamper(name, r):
            if name == "predict":
                f = r.info["latest"]
                df = pd.read_csv(f, float_precision="round_trip")
                df.loc[0, "raw_model_prob_over"] = df.loc[0, "raw_model_prob_over"] + 1e-9
                df.to_csv(f, index=False)
        s = run(tmp_path / "ws", tamper)
        assert_failed_at(s, "verify_outputs", tmp_path / "ws", "raw_model_prob_over is")

    def test_metric_parity_is_checked(self, tmp_path, monkeypatch):
        real = rh.training_rows

        def fewer_rows(table, prop):
            rows, report = real(table, prop)
            return rows.iloc[: len(rows) // 2], report

        def swap(name, r):
            if name == "predict":
                monkeypatch.setattr(rh, "training_rows", fewer_rows)
        s = run(tmp_path / "ws", swap)
        assert_failed_at(s, "verify_outputs", tmp_path / "ws", "recomputed")

    def test_prediction_paths_are_restored(self, success):
        from player_props import predict
        assert predict.DATA_DIR == ROOT / "data_files"
        assert predict.MODELS_DIR == ROOT / "player_props" / "models"

    def test_network_is_blocked(self, tmp_path):
        def call_out(name, r):
            if name == "workspace":
                socket.create_connection(("example.com", 443))
        s = run(tmp_path / "ws", call_out)
        assert_failed_at(s, "workspace", tmp_path / "ws", "network access attempted during the rehearsal")

    def test_prediction_never_reads_the_training_folder(self, tmp_path):
        def remove_training_models(name, r):
            if name == "restore":
                shutil.rmtree(r.paths["train_models"])
        s = run(tmp_path / "ws", remove_training_models)
        assert s["status"] == "succeeded"


class TestPublishing:
    @pytest.fixture
    def published_ws(self, success, tmp_path):
        ws, s, _ = success
        copy = tmp_path / "ws"
        shutil.copytree(Path(ws).resolve() / "published", copy / "published")
        return copy

    def test_failure_preserves_published_predictions(self, published_ws):
        before = tree(published_ws / "published")

        def fail(name, r):
            if name == "verify_outputs":
                raise RuntimeError("injected failure after verification")
        s = run(published_ws, fail)
        assert_failed_at(s, "verify_outputs", published_ws, "injected failure")
        assert tree(published_ws / "published") == before

    def test_repeated_run_does_not_overwrite_a_different_snapshot(self, published_ws):
        before = tree(published_ws / "published")
        s = run(published_ws)
        assert s["status"] == "conflict" and status(s)["publish"] == "failed"
        err = next(x for x in s["stages"] if x["stage"] == "publish")["error"]["message"]
        assert "already frozen by" in err and "not overwritten" in err
        assert tree(published_ws / "published") == before

    def test_identical_snapshot_is_accepted(self, published_ws):
        pub = published_ws / "published"

        # The published copy already holds exactly this run's verified snapshot
        # (as after an earlier attempt of the same publication).
        def same(name, r):
            if name == "verify_outputs":
                shutil.copy2(r.info["snapshot"], pub / r.info["snapshot"].name)
        s = run(published_ws, same)
        assert s["status"] == "succeeded"
        assert s["outputs"]["publish"]["snapshot_status"] == "identical_existing"
        record = json.loads((pub / "snapshots.json").read_text())
        # the snapshot is still attributed to the run that froze it
        assert record["player_props_predictions_week3_2026.csv"]["run_id"] != s["run_id"]

    def test_failure_then_successful_retry(self, tmp_path):
        ws = tmp_path / "ws"

        def fail(name, r):
            if name == "export":
                raise RuntimeError("interrupted between export and restore")
        first = run(ws, fail)
        assert_failed_at(first, "export", ws, "interrupted")
        assert not (Path(ws).resolve() / "published").exists()
        second = run(ws)
        assert second["status"] == "succeeded" and second["run_id"] != first["run_id"]
        assert run_dir(first, ws).exists()                      # evidence kept
        assert json.loads((run_dir(first, ws) / "summary.json").read_text())["status"] == "failed"


# ---------------------------------------------------------------------------
# Output verification: the final fields, labels and both files
# ---------------------------------------------------------------------------

def edit_predictions(path, change):
    df = rh.read_predictions(path)
    change(df)
    df.to_csv(path, index=False)


def fabricate(df):
    # The reproduced case: a consistent-looking but invented final probability.
    df.loc[0, ["prob_over", "prob_under", "confidence"]] = [0.123, 0.877, 0.877]
    df.loc[0, "adjustments"] = "invented_adjustment"


@pytest.fixture(scope="module")
def verified(success):
    r = SUCCESS_RUN["run"]
    return r, rh.read_predictions(r.info["latest"]), rh._Expectations(r)


class TestOutputVerification:
    """Unit level: the checks run against the success run's real output, with
    one field changed in memory. End to end: the rehearsal stops before
    publishing and leaves published files untouched."""

    def problems(self, verified, df):
        r, _, expect = verified
        return rh.prediction_problems(df, "latest feed", r, expect)

    def row(self, df, label):
        return df.index[df["adjustments"] == label][0]

    def test_real_output_passes(self, verified):
        r, df, expect = verified
        assert self.problems(verified, df) == []
        assert rh.prediction_problems(rh.read_predictions(r.info["snapshot"]), "frozen snapshot",
                                      r, expect) == []

    def test_fabricated_probability_and_label(self, verified):
        df = verified[1].copy()
        fabricate(df)
        found = " ".join(self.problems(verified, df))
        assert "prob_over is 0.123" in found and "'invented_adjustment'" in found

    def test_altered_probability_alone(self, verified):
        df = verified[1].copy()
        i = self.row(df, "none")
        df.loc[i, "prob_over"] = df.loc[i, "prob_over"] + 0.01
        assert any("prob_over is" in p for p in self.problems(verified, df))

    @pytest.mark.parametrize("clamp", [rh.LOW_LABEL, rh.STAR_LABEL])
    @pytest.mark.parametrize("label", ["missing", "wrong", "duplicated", "unexpected_first",
                                       "unexpected_after", "on_unclamped_row"])
    def test_clamp_labels_must_be_exact(self, verified, clamp, label):
        df = verified[1].copy()
        other = rh.STAR_LABEL if clamp == rh.LOW_LABEL else rh.LOW_LABEL
        i = self.row(df, clamp)
        value = {"missing": "none", "wrong": other, "duplicated": f"{clamp};{clamp}",
                 "unexpected_first": f"weather;{clamp}", "unexpected_after": f"{clamp};weather",
                 "on_unclamped_row": clamp}[label]
        if label == "on_unclamped_row":
            i = self.row(df, "none")
        df.loc[i, "adjustments"] = value
        assert any(f"adjustments is {value!r}" in p for p in self.problems(verified, df))

    def test_label_order_is_exact(self):
        # Under these rules at most one clamp applies, so order is enforced by
        # comparing the exact joined string (also for the unexpected-label cases).
        assert rh.expected_post_processing(0.9, "elite_wr", 200.0, 50.0)["adjustments"] == "none"
        assert rh.expected_post_processing(0.6, "elite_wr", 200.0, 50.0)["adjustments"] == rh.STAR_LABEL
        assert rh.expected_post_processing(0.6, "good_wr", 10.0, 50.0)["adjustments"] == rh.LOW_LABEL
        assert rh.expected_post_processing(0.6, "good_wr", 200.0, 50.0)["adjustments"] == "none"

    @pytest.mark.parametrize("p,side,included", [
        (0.55, "OVER", True), (0.5499, "UNDER", False), (0.52, "UNDER", False), (0.5, "UNDER", False),
        # production's band edge is 1 - 0.55 == 0.44999999999999996, so 0.45 is excluded
        (0.45, "UNDER", False), (0.4499, "UNDER", True), (0.9, "OVER", True),
    ])
    def test_side_and_inclusion_rules(self, p, side, included):
        e = rh.expected_post_processing(p, "good_wr", 50.0, 50.0)
        assert (e["prob_over"], e["recommendation"], e["included"]) == (p, side, included)
        assert e["prob_under"] == 1 - p and e["confidence"] == max(p, 1 - p)

    def test_coherent_claim_of_another_tier(self, verified):
        # Every dependent field rewritten consistently for a tier the player is
        # not in: only reconstructing the tier from the stats catches it.
        r, df, _ = verified
        models = r.info["bundle"]["models"].values()
        df = df.astype(object)
        i = next(j for j in df.index if any(m["prop_type"] == df.loc[j, "prop_type"]
                                            and m["line_type"] != df.loc[j, "line_type"] for m in models))
        row = df.loc[i].to_dict()
        other = next(m["line_type"] for m in models
                     if m["prop_type"] == row["prop_type"] and m["line_type"] != row["line_type"])
        fake = rh._Expectations(r)
        fake.tier_of = lambda *a: other
        claimed = fake.for_row(row)
        for f in rh.EXPECTED_FIELDS:
            df.loc[i, f] = claimed[f]
        found = rh.prediction_problems(df, "latest feed", r, rh._Expectations(r))
        assert any(f"line_type is {other!r}" in p for p in found)

    @pytest.mark.parametrize("field,change", [
        ("prob_under", lambda r: r["prob_under"] + 1e-12),
        ("prob_under", lambda r: r["prob_over"]),
        ("confidence", lambda r: min(r["prob_over"], r["prob_under"])),
        ("confidence", lambda r: r["confidence"] - 1e-12),
        ("recommendation", lambda r: "UNDER" if r["recommendation"] == "OVER" else "OVER"),
    ])
    def test_complement_confidence_and_side(self, verified, field, change):
        df = verified[1].copy()
        for label in ("none", rh.LOW_LABEL, rh.STAR_LABEL):
            i = self.row(df, label)
            df.loc[i, field] = change(df.loc[i])
        found = [p for p in self.problems(verified, df) if f"{field} is" in p]
        assert len(found) == 3

    @pytest.mark.parametrize("field,value", [
        ("avg_L5", 999.0), ("line_type", "elite_wr"), ("model_name", "receiving_yards_elite_wr"),
        ("line_value", 1.5), ("model_reliable", "maybe"), ("model_bundle", "20260101T000000Z-other"),
        ("season", 2025), ("week", 4), ("weather_adjusted", True), ("weather_conditions", "Dome"),
        ("market_line_available", True), ("market_line", 55.5), ("display_name", "Nobody Known"),
    ])
    def test_identity_and_skipped_inputs(self, verified, field, value):
        df = verified[1].copy()
        i = df.index[df[field].astype(str) != str(value)][0]
        df[field] = df[field].astype(object)
        df.loc[i, field] = value
        assert any(f in p for p in self.problems(verified, df)
                   for f in (f"{field} is", "not this run's bundle", "is not the target",
                             f"{field} {value!r} is not"))

    def test_row_production_would_exclude(self, verified, monkeypatch):
        # A row whose final probability is inside the no-recommendation band
        # can't be made consistent by editing the CSV (the raw score is
        # recomputed), so the expectation itself is moved into the band.
        r, df, _ = verified
        real = rh.expected_post_processing
        monkeypatch.setattr(rh, "expected_post_processing",
                            lambda *a: {**real(*a), "included": False})
        found = rh.prediction_problems(df, "latest feed", r, rh._Expectations(r))
        assert len([p for p in found if "would not include it" in p]) == len(df)

    def test_duplicate_rows(self, verified):
        df = verified[1]
        assert any("duplicate predictions" in p
                   for p in self.problems(verified, pd.concat([df, df.iloc[[0]]], ignore_index=True)))

    @pytest.fixture
    def published_copy(self, success, tmp_path):
        ws, _, _ = success
        copy = tmp_path / "ws"
        shutil.copytree(Path(ws).resolve() / "published", copy / "published")
        return copy, tree(copy / "published")

    def test_reproduced_fabrication_fails_before_publication(self, published_copy):
        ws, before = published_copy

        def tamper(name, r):
            if name == "predict":
                for f in (r.info["latest"], r.info["snapshot"]):
                    edit_predictions(f, fabricate)
        s = run(ws, tamper)
        assert_failed_at(s, "verify_outputs", ws, "prob_over is 0.123")
        assert "invented_adjustment" in json.dumps(s["stages"])
        assert tree(ws / "published") == before

    def test_tampered_snapshot_with_correct_latest(self, published_copy):
        ws, before = published_copy

        def change(df):
            df.loc[0, "prob_over"] = 0.51

        def tamper(name, r):
            if name == "predict":
                edit_predictions(r.info["snapshot"], change)
        s = run(ws, tamper)
        assert_failed_at(s, "verify_outputs", ws, "frozen snapshot: ")
        err = json.dumps(next(x for x in s["stages"] if x["stage"] == "verify_outputs")["error"])
        assert "latest feed: " not in err
        assert "does not hold exactly the latest feed" in err
        assert tree(ws / "published") == before

    def test_valid_snapshot_from_another_run(self, published_copy, success):
        ws, before = published_copy
        other = run_dir(success[1], success[0]) / "runtime" / "data" / "player_props_predictions_week3_2026.csv"

        def swap(name, r):
            if name == "predict":
                shutil.copyfile(other, r.info["snapshot"])
        s = run(ws, swap)
        assert_failed_at(s, "verify_outputs", ws, "is not this run's bundle")
        assert tree(ws / "published") == before

    @pytest.mark.parametrize("which", ["latest", "snapshot"])
    def test_change_after_verification_is_not_published(self, published_copy, which):
        ws, before = published_copy

        def tamper(name, r):
            if name == "verify_outputs":
                edit_predictions(r.info[which], fabricate)
        s = run(ws, tamper)
        assert_failed_at(s, "publish", ws, "changed after verification")
        assert tree(ws / "published") == before


# ---------------------------------------------------------------------------
# Stable player identity: rows carry the player_id that scored them
# ---------------------------------------------------------------------------

class TestPlayerId:
    def problems(self, verified, df):
        r, _, _ = verified
        return rh.prediction_problems(df, "latest feed", r, rh._Expectations(r))

    def found(self, verified, df, text):
        return [p for p in self.problems(verified, df) if text in p]

    def test_rows_carry_the_scoring_id(self, verified):
        r, df, _ = verified
        assert df["player_id"].notna().all() and df["player_id"].map(type).eq(str).all()
        # Every ID is a synthetic player of the row's own team (fixtures: "<TEAM>-<ROLE>").
        assert (df["player_id"].str.split("-").str[0] == df["team"]).all()
        assert self.problems(verified, df) == []

    @pytest.mark.parametrize("value,text", [(None, "no player_id"), ("", "no player_id"),
                                            ("ZZ-NOPE", "unknown player_id 'ZZ-NOPE'")])
    def test_missing_or_unknown_id(self, verified, value, text):
        df = verified[1].copy()
        df.loc[0, "player_id"] = value
        assert self.found(verified, df, text)

    def test_substituted_id(self, verified):
        # Another real player's ID on this row: names no longer match that ID.
        df = verified[1].copy()
        i = 0
        other = df.loc[(df["team"] == df.loc[i, "team"]) & (df["player_id"] != df.loc[i, "player_id"]),
                       "player_id"].iloc[0]
        df.loc[i, "player_id"] = other
        assert self.found(verified, df, "is not")

    def test_id_from_another_team(self, verified):
        df = verified[1].copy()
        df.loc[0, "player_id"] = df.loc[df["team"] != df.loc[0, "team"], "player_id"].iloc[0]
        assert self.found(verified, df, f"is not a {df.loc[0, 'team']} candidate")

    @pytest.mark.parametrize("field,value,text", [
        ("display_name", "Someone Else", "display_name 'Someone Else' is not"),
        ("player_name", "S.Else", "player_name 'S.Else' is not"),
        ("position", "TE", "position 'TE' is not"),
    ])
    def test_id_inconsistent_with_the_row(self, verified, field, value, text):
        df = verified[1].copy()
        i = df.index[df[field] != value][0]
        df.loc[i, field] = value
        assert self.found(verified, df, text)

    def test_prop_not_eligible_for_the_ids_position(self, verified):
        df = verified[1].copy()
        i = df.index[df["position"] == "QB"][0]
        df.loc[i, "prop_type"] = "receptions"
        assert self.found(verified, df, "prop 'receptions' is not eligible")

    def test_team_changed_under_the_same_id(self, verified):
        df = verified[1].copy()
        df.loc[0, "team"] = df.loc[0, "opponent"]
        assert self.found(verified, df, "candidate")

    def test_no_player_id_column_fails_closed(self, verified):
        df = verified[1].drop(columns="player_id")
        assert self.problems(verified, df) == ["latest feed: no player_id column, so no row's player can be verified"]

    def test_same_name_different_ids_are_not_duplicates(self, verified):
        # Row a takes row b's names (same prop, different ID): identity checks may
        # object to a's names, but the two rows must not count as duplicates.
        df = verified[1].copy()
        same = df.groupby(["team", "prop_type"])["player_id"].transform("nunique") > 1
        b = df.index[same][0]
        a = df.index[(df["team"] == df.loc[b, "team"]) & (df["prop_type"] == df.loc[b, "prop_type"])
                     & (df["player_id"] != df.loc[b, "player_id"])][0]
        df.loc[a, ["player_name", "display_name"]] = df.loc[b, ["player_name", "display_name"]].values
        assert not self.found(verified, df, "duplicate predictions")

    def test_id_repeated_in_the_reference(self, verified):
        r, df, _ = verified
        e = rh._Expectations(r)
        e.repeated_ids = {df.loc[0, "player_id"]}
        assert any("several player-reference records" in p
                   for p in rh.prediction_problems(df, "latest feed", r, e))

    def test_duplicate_identity(self, verified):
        df = verified[1]
        dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        assert self.found(verified, dup, "duplicate predictions")

    @pytest.mark.parametrize("ids", [["00-0012345", "0012345", "KC-QB", "1e5"],
                                     # every ID numeric-looking: a plain read would make numbers
                                     ["0012345", "00987", "1e5", "42"]])
    def test_csv_round_trip_keeps_ids_exact(self, tmp_path, ids):
        path = tmp_path / "p.csv"
        pd.DataFrame({"player_id": ids, "x": range(len(ids))}).to_csv(path, index=False)
        assert rh.read_predictions(path)["player_id"].tolist() == ids

    @pytest.fixture
    def published_copy(self, success, tmp_path):
        ws, _, _ = success
        copy = tmp_path / "ws"
        shutil.copytree(Path(ws).resolve() / "published", copy / "published")
        return copy, tree(copy / "published")

    def test_snapshot_id_mismatch_blocks_publication(self, published_copy):
        ws, before = published_copy

        def swap(df):
            df.loc[[0, 1], "player_id"] = df.loc[[1, 0], "player_id"].values

        def tamper(name, r):
            if name == "predict":
                edit_predictions(r.info["snapshot"], swap)
        s = run(ws, tamper)
        assert_failed_at(s, "verify_outputs", ws, "player identities differ from the latest feed")
        assert tree(ws / "published") == before

    def test_substituted_id_in_both_files_blocks_publication(self, published_copy):
        ws, before = published_copy

        def substitute(df):
            df.loc[0, "player_id"] = df.loc[df["player_id"] != df.loc[0, "player_id"], "player_id"].iloc[0]

        def tamper(name, r):
            if name == "predict":
                for f in (r.info["latest"], r.info["snapshot"]):
                    edit_predictions(f, substitute)
        s = run(ws, tamper)
        assert_failed_at(s, "verify_outputs", ws, "is not")
        assert tree(ws / "published") == before


# ---------------------------------------------------------------------------
# Completeness: the expected rows, derived independently from the inputs
# ---------------------------------------------------------------------------

from player_props import rehearsal_expected as rx  # noqa: E402


@pytest.fixture(scope="module")
def expected(success):
    r = SUCCESS_RUN["run"]
    data = r.paths["runtime_data"]
    return rx.expected_rows(data, r.info["check"].feature_tables, r.info["bundle"]["models"],
                            fx_load_games(data), r.config.target_season, r.config.target_week,
                            rh.expected_post_processing,
                            pd.read_csv(data / "players.csv", dtype={"gsis_id": str}))


def fx_load_games(data):
    from player_props.provenance import load_games
    return load_games(data)


def key_of(row):
    return rx.identity(row["player_id"], row["team"], row["opponent"], row["prop_type"], row["line_value"])


class TestCompletenessUnit:
    """The comparison against the success run's real output, one change at a time."""

    def check(self, expected, df):
        return rx.completeness_problems(df, "latest feed", expected)

    def test_valid_output_matches_exactly(self, verified, expected, success):
        df = verified[1]
        problems, counts = self.check(expected, df)
        assert problems == [] and set(map(key_of, df.to_dict("records"))) == set(expected.rows)
        assert counts == {"checked": True, "observed": len(df), "distinct": len(df), "missing": 0,
                          "unexpected": 0, "duplicates": 0, "wrong_side": 0}
        s = success[1]["completeness"]
        assert s["status"] == "complete" and s["expected"] == len(df) == s["latest feed"]["observed"]
        assert s["frozen snapshot"]["missing"] == s["frozen snapshot"]["unexpected"] == 0

    def test_exclusions_are_accounted_for(self, expected):
        # 8 teams x (QB 2 + RB 3 + WR1 3 + WR2 3 + TE 3) props; each one either
        # expected or excluded under a named rule.
        assert len(expected.rows) + sum(expected.exclusions.values()) == 8 * 14
        assert set(expected.exclusions) <= {"confidence_band", "no_model", "no_history",
                                            "position_not_eligible", "not_in_player_reference",
                                            "latest_appearance_elsewhere"}

    def test_one_row_missing(self, verified, expected):
        problems, counts = self.check(expected, verified[1].drop(index=[5]))
        assert counts["missing"] == 1 and "1 expected prediction(s) missing" in problems[0]

    @pytest.mark.parametrize("what", ["player", "prop", "game"])
    def test_everything_for_a_player_prop_or_game_missing(self, verified, expected, what):
        df = verified[1]
        first = df.iloc[0]
        if what == "player":
            gone = df["player_id"] == first["player_id"]
        elif what == "prop":
            gone = df["prop_type"] == first["prop_type"]
        else:
            gone = df["team"].isin([first["team"], first["opponent"]])
        problems, counts = self.check(expected, df[~gone])
        assert counts["missing"] == int(gone.sum()) > 0 and "missing" in problems[0]

    def test_extra_row(self, verified, expected):
        df = verified[1]
        extra = df.iloc[[0]].assign(line_value=999.5)
        problems, counts = self.check(expected, pd.concat([df, extra], ignore_index=True))
        assert counts["unexpected"] == 1 and any("unexpected prediction" in p for p in problems)

    def test_duplicate_identity(self, verified, expected):
        df = verified[1]
        problems, counts = self.check(expected, pd.concat([df, df.iloc[[3]]], ignore_index=True))
        assert counts["duplicates"] == 1 and any("listed more than once" in p for p in problems)

    def test_wrong_home_away_side(self, verified, expected):
        df = verified[1].copy()
        df["is_home"] = df["is_home"].astype(object)
        df.loc[0, "is_home"] = not bool(df.loc[0, "is_home"])
        problems, counts = self.check(expected, df)
        assert counts["wrong_side"] == 1 and any("home/away side" in p for p in problems)

    def test_no_player_id_column(self, verified, expected):
        problems, _ = self.check(expected, verified[1].drop(columns="player_id"))
        assert problems == ["latest feed: no player_id column, so completeness can't be checked"]


class TestExpectedSetFailsClosed:
    """Ambiguous or invalid inputs raise instead of shrinking the expected set."""

    def table(self, rows):
        return pd.DataFrame(rows, columns=["player_id", "player_name", "team", "game_id", "game_date"])

    def test_tie_deciding_the_recent_games(self):
        t = self.table([("a", "A", "KC", f"g{i}", d) for i, d in
                        enumerate(["2026-09-01", "2026-09-08", "2026-09-15", "2026-09-15"])])
        with pytest.raises(rx.ExpectedSetError, match="most recent games are ambiguous"):
            rx._team_candidates({"receiving": t}, "KC", rx.Counter())

    def test_ambiguous_latest_team(self):
        t = self.table([("a", "A", "KC", "g1", "2026-09-15"), ("a", "A", "BUF", "g1", "2026-09-15")])
        with pytest.raises(rx.ExpectedSetError, match="latest appearance is ambiguous"):
            rx._team_candidates({"receiving": t}, "KC", rx.Counter())

    def test_three_most_recent_games_window(self):
        # "a" last played in the team's 3rd most recent game (a candidate);
        # "b" only in the 4th (not one).
        t = self.table([("x", "X", "KC", "g4", "2026-09-29"), ("x", "X", "KC", "g3", "2026-09-22"),
                        ("a", "A", "KC", "g2", "2026-09-15"), ("b", "B", "KC", "g1", "2026-09-08")])
        assert rx._team_candidates({"receiving": t}, "KC", rx.Counter()) == {"x": "X", "a": "A"}

    def test_latest_team_elsewhere_is_counted_not_silent(self):
        t = self.table([("a", "A", "KC", "g1", "2026-09-08"), ("a", "A", "BUF", "g2", "2026-09-15")])
        seen = rx.Counter()
        assert rx._team_candidates({"receiving": t}, "KC", seen) == {}
        assert seen == {"latest_appearance_elsewhere": 1}

    def test_roster_filter_is_refused(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PROP_ROSTER_FILTER", "1")
        with pytest.raises(rx.ExpectedSetError, match="PROP_ROSTER_FILTER"):
            rx.expected_rows(tmp_path, {}, {}, pd.DataFrame(), 2026, 3, None, pd.DataFrame())

    def test_missing_feature_table(self, tmp_path):
        with pytest.raises(rx.ExpectedSetError, match="feature table"):
            rx.expected_rows(tmp_path, {"passing": pd.DataFrame()}, {}, pd.DataFrame(), 2026, 3, None,
                             pd.DataFrame())

    def test_game_listed_twice(self, tmp_path):
        pd.DataFrame({"week": [3, 3], "date": ["2026-09-20"] * 2, "home_team": ["KC"] * 2,
                      "away_team": ["BUF"] * 2}).to_csv(tmp_path / "nfl_schedule_2026.csv", index=False)
        games = pd.DataFrame({"game_id": ["2026_03_BUF_KC"], "season": [2026], "week": [3],
                              "home_team": ["KC"], "away_team": ["BUF"], "gameday": ["2026-09-20"]})
        with pytest.raises(rx.ExpectedSetError, match="lists a week-3 game twice"):
            rx._week_games(tmp_path, games, 2026, 3)

    GAMES = pd.DataFrame({"game_id": ["2026_03_BUF_KC"], "season": [2026], "week": [3],
                          "home_team": ["KC"], "away_team": ["BUF"], "gameday": ["2026-09-20"]})

    def schedule(self, tmp_path, date):
        path = tmp_path / "nfl_schedule_2026.csv"
        path.write_text(f"week,date,home_team,away_team,venue,status\n3,{date},KC,BUF,Field,REG\n",
                        encoding="utf-8")

    def test_matching_date_is_accepted(self, tmp_path):
        self.schedule(tmp_path, "2026-09-20")
        assert [g["game_id"] for g in rx._week_games(tmp_path, self.GAMES, 2026, 3)] == ["2026_03_BUF_KC"]

    def test_wrong_date_for_matching_teams(self, tmp_path):
        self.schedule(tmp_path, "2026-09-21")
        with pytest.raises(rx.ExpectedSetError, match="BUF @ KC is dated 2026-09-21, but the schedule has "
                                                      "'2026-09-20' for 2026_03_BUF_KC"):
            rx._week_games(tmp_path, self.GAMES, 2026, 3)

    @pytest.mark.parametrize("date", ["", "2026-9-20", "20/09/2026", "2026-13-01", "2026-09-20T13:00", "soon"])
    def test_missing_or_malformed_date(self, tmp_path, date):
        self.schedule(tmp_path, date)
        with pytest.raises(rx.ExpectedSetError, match="has a missing or malformed date"):
            rx._week_games(tmp_path, self.GAMES, 2026, 3)

    def test_no_date_column(self, tmp_path):
        (tmp_path / "nfl_schedule_2026.csv").write_text("week,home_team,away_team\n3,KC,BUF\n", encoding="utf-8")
        with pytest.raises(rx.ExpectedSetError, match="has no date column"):
            rx._week_games(tmp_path, self.GAMES, 2026, 3)


class TestCompletenessStatus:
    ok = {"checked": True, "missing": 0, "unexpected": 0, "duplicates": 0, "wrong_side": 0}

    def test_complete(self):
        assert rx.completeness_status({"a": dict(self.ok), "b": dict(self.ok)}) == "complete"

    @pytest.mark.parametrize("field", ["missing", "unexpected", "duplicates", "wrong_side"])
    def test_any_failure_count_is_incomplete(self, field):
        assert rx.completeness_status({"a": dict(self.ok), "b": {**self.ok, field: 1}}) == "incomplete"

    def test_unchecked_file_is_never_complete(self):
        assert rx.completeness_status({"a": dict(self.ok), "b": {"checked": False, "observed": 9}}) == "not checked"
        assert rx.completeness_status({"a": {**self.ok, "checked": False}}) == "not checked"
        assert rx.completeness_status({}) == "not checked"


class TestCompletenessEndToEnd:
    """Each failure blocks publication and leaves published files byte-identical."""

    @pytest.fixture
    def published_copy(self, success, tmp_path):
        ws, _, _ = success
        copy = tmp_path / "ws"
        shutil.copytree(Path(ws).resolve() / "published", copy / "published")
        return copy, tree(copy / "published")

    def blocked(self, ws, before, s, text):
        assert_failed_at(s, "verify_outputs", ws, text)
        assert tree(ws / "published") == before
        return s["completeness"]

    def both(self, change):
        def tamper(name, r):
            if name == "predict":
                for f in (r.info["latest"], r.info["snapshot"]):
                    edit_predictions(f, change)
        return tamper

    def test_row_dropped_from_both_files(self, published_copy):
        ws, before = published_copy
        c = self.blocked(ws, before, run(ws, self.both(lambda df: df.drop(index=[7], inplace=True))),
                         "expected prediction(s) missing")
        assert c["status"] == "incomplete" and c["latest feed"]["missing"] == c["frozen snapshot"]["missing"] == 1
        assert c["expected"] == c["latest feed"]["observed"] + 1

    def test_all_rows_for_a_player_dropped_from_both(self, published_copy):
        ws, before = published_copy

        def drop_player(df):
            df.drop(index=df.index[df["player_id"] == df.loc[0, "player_id"]], inplace=True)
        c = self.blocked(ws, before, run(ws, self.both(drop_player)), "expected prediction(s) missing")
        assert c["latest feed"]["missing"] >= 1

    def test_extra_row_injected_into_both(self, published_copy):
        ws, before = published_copy

        def inject(df):
            df.loc[len(df)] = df.iloc[0].copy()
            df.loc[len(df) - 1, "line_value"] = 999.5
        c = self.blocked(ws, before, run(ws, self.both(inject)), "unexpected prediction")
        assert c["latest feed"]["unexpected"] == c["frozen snapshot"]["unexpected"] == 1

    def saved(self, ws, s):
        """The failed run's summary.json as written to disk."""
        return json.loads((run_dir(s, ws) / "summary.json").read_text())

    def test_wrong_side_marks_the_saved_summary_incomplete(self, published_copy):
        ws, before = published_copy

        def flip(df):
            df["is_home"] = df["is_home"].astype(object)
            df.loc[0, "is_home"] = not bool(df.loc[0, "is_home"])
        s = run(ws, self.both(flip))
        # The row checks also object (they look up the swapped game), so the
        # message varies; the saved summary must still say incomplete.
        self.blocked(ws, before, s, None)
        c = self.saved(ws, s)["completeness"]
        assert c["status"] == "incomplete"
        for f in ("latest feed", "frozen snapshot"):
            assert c[f]["wrong_side"] == 1 and c[f]["missing"] == c[f]["unexpected"] == c[f]["duplicates"] == 0

    def test_unchecked_files_are_never_complete_in_the_saved_summary(self, published_copy):
        ws, before = published_copy
        s = run(ws, self.both(lambda df: df.drop(columns="player_id", inplace=True)))
        self.blocked(ws, before, s, "completeness can't be checked")
        c = self.saved(ws, s)["completeness"]
        assert c["status"] == "not checked"
        assert c["latest feed"]["checked"] is False and c["frozen snapshot"]["checked"] is False

    def test_wrong_schedule_date_blocks_publication(self, published_copy):
        ws, before = published_copy

        def redate(name, r):
            if name == "predict":
                p = r.paths["runtime_data"] / f"nfl_schedule_{r.config.target_season}.csv"
                sched = pd.read_csv(p, dtype={"date": str})
                i = sched.index[sched["week"] == r.config.target_week][0]
                sched.loc[i, "date"] = "2026-12-25"
                sched.to_csv(p, index=False)
        s = run(ws, redate)
        self.blocked(ws, before, s, "is dated 2026-12-25, but the schedule has")
        c = self.saved(ws, s)["completeness"]
        assert c["status"] == "expected set not derived" and "2026-12-25" in c["problems"][0]

    def test_production_omits_a_candidate(self, published_copy, monkeypatch):
        # A slip in production's selection (one eligible player silently dropped).
        # The row checks see nothing wrong with the rows that exist; completeness does.
        from player_props import predict
        real = predict.candidate_players
        dropped = []

        def omit_one(all_stats, team, n_games=3, roster_ids=None):
            names, report = real(all_stats, team, n_games, roster_ids)
            if not dropped and names:
                dropped.append(sorted(names)[0])
                names = {k: v for k, v in names.items() if k != dropped[0]}
            return names, report
        monkeypatch.setattr(predict, "candidate_players", omit_one)
        ws, before = published_copy
        s = run(ws)
        c = self.blocked(ws, before, s, "expected prediction(s) missing")
        err = json.dumps(next(x for x in s["stages"] if x["stage"] == "verify_outputs")["error"])
        assert dropped[0] in err and c["latest feed"]["missing"] >= 1

    def test_production_tier_rule_changed(self, published_copy, monkeypatch):
        # A changed tier rule moves rows to other models and lines. The row
        # checks reuse production's tier function and agree; completeness
        # restates the rule and catches it.
        from player_props import predict
        monkeypatch.setattr(predict, "get_player_performance_tier",
                            lambda pid, prop, stats: predict.TIER_RULES[prop][2])
        ws, before = published_copy
        c = self.blocked(ws, before, run(ws), "expected prediction(s) missing")
        assert c["latest feed"]["missing"] > 0 and c["latest feed"]["unexpected"] > 0

    def test_production_confidence_band_changed(self, published_copy, monkeypatch):
        from player_props import predict
        monkeypatch.setattr(predict, "MIN_CONFIDENCE", 0.5)          # admits rows the band excludes
        ws, before = published_copy
        c = self.blocked(ws, before, run(ws), "unexpected prediction")
        assert c["latest feed"]["unexpected"] > 0

    def test_schedules_disagree(self, published_copy):
        # The prediction schedule loses a game the schedule lists (after
        # prediction, which refuses this itself): invalid input, so the check
        # fails rather than shrinking the expected set.
        ws, before = published_copy

        def drop_game(name, r):
            if name == "predict":
                p = r.paths["runtime_data"] / f"nfl_schedule_{r.config.target_season}.csv"
                sched = pd.read_csv(p)
                wk = sched[sched["week"] == r.config.target_week].index[0]
                sched.drop(index=[wk]).to_csv(p, index=False)
        s = run(ws, drop_game)
        c = self.blocked(ws, before, s, "completeness can't be checked")
        assert c["status"] == "expected set not derived" and "disagree" in c["problems"][0]


class TestHistoricalFeeds:
    """Files written before player_id stay readable as they are, and are never
    given an ID by name or treated as identity-verified."""

    def old_feed(self, verified, tmp_path):
        old = verified[1].drop(columns="player_id")
        path = tmp_path / "player_props_predictions_week3_2026.csv"
        old.to_csv(path, index=False)
        return path

    def test_readable_without_modification(self, verified, tmp_path):
        from player_props import backtest
        path = self.old_feed(verified, tmp_path)
        raw = path.read_bytes()
        df = rh.read_predictions(path)
        assert "player_id" not in df.columns and len(df) == len(verified[1])
        assert backtest.prediction_week_problem(pd.read_csv(path), 3, 2026) is None
        assert path.read_bytes() == raw

    def test_never_verified_by_name(self, verified, tmp_path):
        r = verified[0]
        df = rh.read_predictions(self.old_feed(verified, tmp_path))
        assert rh.prediction_problems(df, "historical", r, rh._Expectations(r)) == [
            "historical: no player_id column, so no row's player can be verified"]

    def test_grading_is_unchanged_by_the_new_column(self, verified):
        from player_props import backtest
        new = verified[1]
        actual = pd.DataFrame({"player_name": new["player_name"].unique()})
        for c in ("passing_yards", "rushing_yards", "receiving_yards", "passing_tds", "rushing_tds",
                  "receiving_tds", "receptions"):
            actual[c] = 50.0
        with_id = backtest.calculate_hit_rate(new, actual)
        without = backtest.calculate_hit_rate(new.drop(columns="player_id"), actual)
        assert with_id["total_predictions"] == without["total_predictions"] > 0
        assert with_id["overall_accuracy"] == without["overall_accuracy"]


# ---------------------------------------------------------------------------
# Path protection
# ---------------------------------------------------------------------------

def _junction(link, target):
    if os.name != "nt":
        pytest.skip("Windows junctions only")
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
    if r.returncode:
        pytest.skip(f"junction not permitted: {r.stderr!r}")


@pytest.fixture
def no_repo_mkdir(monkeypatch):
    """Make any directory creation inside the repository (or its parent) fail
    loudly, so a broken guard can't write there even while being tested."""
    real_mkdir = Path.mkdir
    guarded = [Path(os.path.realpath(ROOT)), Path(os.path.realpath(ROOT.parent))]

    def mkdir(self, *a, **k):
        target = Path(os.path.realpath(self))
        if any(target == g or target.is_relative_to(g) for g in guarded) \
                and not target.is_relative_to(Path(os.path.realpath(ROOT / ".pytest_cache"))):
            raise AssertionError(f"test tried to create {target} inside the repository")
        return real_mkdir(self, *a, **k)
    monkeypatch.setattr(Path, "mkdir", mkdir)


class TestPathProtection:
    @pytest.mark.parametrize("where", [lambda: ROOT / "data_files" / "rehearsal",
                                       lambda: ROOT / "player_props" / "models" / "x",
                                       lambda: ROOT / "rehearsal_ws", lambda: ROOT.parent, lambda: ROOT])
    def test_workspace_overlapping_the_repository_is_refused(self, where, no_repo_mkdir):
        target = where()
        existed = target.exists()
        with pytest.raises(rh.RehearsalError, match="overlaps protected location"):
            run(target)
        assert target.exists() == existed

    def test_protected_location_is_refused(self, tmp_path):
        prod = tmp_path / "prod"
        prod.mkdir()
        with pytest.raises(rh.RehearsalError, match="overlaps protected"):
            run(prod / "inside", protected=[prod])

    def test_symlinked_workspace_is_resolved(self, tmp_path):
        prod = tmp_path / "prod"
        prod.mkdir()
        link = tmp_path / "link"
        try:
            link.symlink_to(prod, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks not permitted here")
        with pytest.raises(rh.RehearsalError, match="overlaps protected"):
            run(link, protected=[prod])
        assert list(prod.iterdir()) == []

    def test_junction_workspace_is_resolved(self, tmp_path):
        prod = tmp_path / "prod"
        prod.mkdir()
        _junction(tmp_path / "jws", prod)
        with pytest.raises(rh.RehearsalError, match="overlaps protected"):
            run(tmp_path / "jws", protected=[prod])
        assert list(prod.iterdir()) == []

    def test_junction_inside_the_workspace_is_refused(self, tmp_path):
        prod = tmp_path / "prod"
        prod.mkdir()
        ws = tmp_path / "ws"
        ws.mkdir()
        _junction(ws / "runs", prod)
        with pytest.raises(rh.RehearsalError, match="outside the workspace|overlapping protected"):
            run(ws, protected=[prod])
        assert list(prod.iterdir()) == []

    def test_junction_leaving_the_workspace_is_refused(self, tmp_path):
        # The target is not protected: only the stay-inside-the-workspace rule applies.
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        ws = tmp_path / "ws"
        ws.mkdir()
        _junction(ws / "runs", elsewhere)
        with pytest.raises(rh.RehearsalError, match="outside the workspace"):
            run(ws)
        assert list(elsewhere.iterdir()) == []

    def test_cli_requires_a_workspace(self):
        with pytest.raises(SystemExit) as exc:
            rh.main([])
        assert exc.value.code == 2

    def test_directories_are_checked_before_creation(self, tmp_path, monkeypatch):
        # Defence in depth: even with the workspace check bypassed, nothing is
        # created under a protected location by the run itself.
        prod = tmp_path / "prod"
        prod.mkdir()
        monkeypatch.setattr(rh, "check_workspace", lambda ws, protected=None: Path(os.path.realpath(ws)))
        with pytest.raises(rh.RehearsalError, match="overlapping protected location"):
            run(prod / "inside", protected=[prod])
        assert not (prod / "inside" / "runs").exists()

    def test_cli_refuses_the_repository(self, capsys, no_repo_mkdir):
        assert rh.main(["--workspace", str(ROOT / "data_files" / "rehearsal")]) == 2
        assert json.loads(capsys.readouterr().out)["status"] == "refused"
        assert not (ROOT / "data_files" / "rehearsal").exists()


def test_fixture_schedule_is_a_valid_round_robin():
    for week in range(1, fixtures.WEEKS + 1):
        pairs = fixtures.round_robin(week)
        teams = [t for p in pairs for t in p]
        assert sorted(teams) == sorted(fixtures.TEAMS)
