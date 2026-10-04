"""pregame_snapshots.py: kickoff times, capture eligibility and statuses,
immutability/idempotency, provenance validation, read-only selection, and
the pipeline's run manifest (without breaking pipeline determinism).

Every write goes to pytest's tmp_path; data_files/ is never touched.
"""

import json
import os
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pregame_snapshots as ps
from test_team_features import ROOT, _FastClassifier, _load, _schedule

UTC = timezone.utc
T0 = datetime(2026, 10, 3, 3, 0, tzinfo=UTC)          # a nightly run, 03:00 UTC Saturday


def _write_inputs(data_dir, rows, probs, signals):
    """Tiny schedule + matching predictions CSV in data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)
    cols = ["game_id", "season", "week", "gameday", "gametime", "home_team", "away_team",
            "home_score", "away_score", "spread_line"]
    sched = pd.DataFrame(rows, columns=cols)
    sched.to_csv(data_dir / ps.SCHEDULE_NAME, sep="\t", index=False)
    preds = sched[["game_id", "season", "week", "home_team", "away_team"]].assign(
        spread_line=sched["spread_line"].fillna(0), home_score=sched["home_score"].fillna(0),
        away_score=sched["away_score"].fillna(0), prob_underdogCovered=probs,
        pred_spreadCovered_optimal=signals)
    preds.to_csv(data_dir / ps.PREDICTIONS_NAME, sep="\t", index=False)
    return sched, preds


def _manifest(data_dir, now=T0 - timedelta(minutes=5), monkeypatch=None, sha="abc123"):
    if monkeypatch is not None:
        monkeypatch.setenv("GITHUB_SHA", sha)
    code = data_dir / "code.py"
    code.write_text("# generating code\n")
    return ps.write_run_manifest(
        data_dir, schedule_sha256=ps.sha256_file(data_dir / ps.SCHEDULE_NAME), code_files=[code],
        repo_dir=data_dir, config={"model": "xgb", "n_estimators": 150, "missing": float("nan")},
        features=["spread_line", "homeTeamWinPct"],
        training_cutoff={"season": 2023, "week": 14, "gameday": "2023-12-10", "games": 1044},
        data_cutoff={"season": 2026, "week": 4, "gameday": "2026-10-01", "games": 1742},
        spread_threshold=0.5438, now=now)


def _reseal(doc):
    """Recompute the (unkeyed) checksum after a deliberate edit - as a forger could."""
    doc[ps.CHECKSUM_FIELD] = ps.payload_checksum(doc)
    return doc


ROWS = [
    # game_id, season, week, gameday, gametime, home, away, hs, as, spread_line
    ("2026_04_PIT_CLE", 2026, 4, "2026-10-01", "20:15", "CLE", "PIT", 27, 24, -2.5),   # completed
    ("2026_04_NYJ_CHI", 2026, 4, "2026-10-04", "13:00", "CHI", "NYJ", None, None, 3.5),  # predicted + signal
    ("2026_04_DAL_HOU", 2026, 4, "2026-10-04", "16:25", "HOU", "DAL", None, None, 2.5),  # predicted, no signal
    ("2026_04_ATL_NO", 2026, 4, "2026-10-05", "20:15", "NO", "ATL", None, None, None),   # no line
    ("2026_04_KC_LV", 2026, 4, "2026-10-04", "20:20", "LV", "KC", None, None, 0.0),      # genuine pick'em
    ("2026_04_TB_GB", 2026, 4, "2026-10-04", "09:30", "GB", "TB", None, None, -1.5),     # valid line, no prob
    ("2026_05_X_Y", 2026, 5, "2026-10-11", None, "Y", "X", None, None, 3.0),             # missing kickoff time
    ("2026_04_A_B", 2026, 4, "2026-10-02", "23:00", "B", "A", None, None, 7.0),          # kicked off before T0
]
# The no-line and pick'em rows deliberately carry a stale probability AND a signal in
# the predictions file: the snapshot must still record them as missing / no bet.
PROBS = [0.51, 0.58, 0.47, 0.50, 0.53, np.nan, 0.55, 0.60]
SIGNALS = [0, 1, 0, 1, 1, 0, 0, 1]


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data_files"
    _write_inputs(d, ROWS, PROBS, SIGNALS)
    _manifest(d)
    return d


# ------------------------------------------------------------ kickoff times --

class TestKickoff:
    @pytest.mark.parametrize("day,time,expected", [
        ("2026-10-04", "13:00", "2026-10-04T17:00:00+00:00"),   # EDT (UTC-4)
        ("2026-11-08", "13:00", "2026-11-08T18:00:00+00:00"),   # EST (UTC-5)
        ("2026-11-01", "13:00", "2026-11-01T18:00:00+00:00"),   # DST ends 02:00 that morning
        ("2026-10-31", "23:59", "2026-11-01T03:59:00+00:00"),   # last minute of EDT
        ("2026-03-08", "13:00", "2026-03-08T17:00:00+00:00"),   # DST began 02:00 that morning
        ("2026-10-04", "09:30", "2026-10-04T13:30:00+00:00"),   # London game, listed in ET
        ("2026-12-28", "20:15", "2026-12-29T01:15:00+00:00"),   # MNF crosses the UTC date
    ])
    def test_eastern_to_utc(self, day, time, expected):
        ko, reason = ps.kickoff_utc(day, time)
        assert reason is None and ko.isoformat() == expected

    @pytest.mark.parametrize("day,time,reason", [
        ("2026-11-01", "01:30", ps.AMBIGUOUS_KICKOFF),        # repeated hour
        ("2026-03-08", "02:30", ps.NONEXISTENT_KICKOFF),      # skipped hour
        (None, "13:00", ps.MISSING_KICKOFF), ("2026-10-04", None, ps.MISSING_KICKOFF),
        ("2026-10-04", "", ps.MISSING_KICKOFF), (float("nan"), "13:00", ps.MISSING_KICKOFF),
        ("2026-10-04", "25:61", ps.INVALID_KICKOFF), ("04/10/2026", "13:00", ps.INVALID_KICKOFF),
    ])
    def test_missing_ambiguous_or_invalid_kickoff_is_skipped_with_reason(self, day, time, reason):
        assert ps.kickoff_utc(day, time) == (None, reason)


# ---------------------------------------------------------- snapshot build --

def _build(data_dir, now=T0):
    manifest = ps.load_verified_manifest(data_dir)
    sched = pd.read_csv(data_dir / ps.SCHEDULE_NAME, sep="\t")
    preds = pd.read_csv(data_dir / ps.PREDICTIONS_NAME, sep="\t")
    return ps.build_snapshot(manifest, sched, preds, now)


class TestSnapshotContent:
    def test_statuses_signals_and_exclusions(self, data_dir):
        snap = _build(data_dir)
        g = {r["game_id"]: r for r in snap["games"]}
        assert set(g) == {"2026_04_NYJ_CHI", "2026_04_DAL_HOU", "2026_04_ATL_NO", "2026_04_KC_LV", "2026_04_TB_GB"}
        nyj = g["2026_04_NYJ_CHI"]
        assert (nyj["prediction_status"], nyj["bet_signal"], nyj["prob_underdog_covers"],
                nyj["underdog_team"], nyj["line_status"], nyj["spread_line"]) ==                ("predicted", True, 0.58, "NYJ", "valid", 3.5)
        # Valid prediction WITHOUT a bet is captured and distinguishable from no-line / pick'em.
        assert (g["2026_04_DAL_HOU"]["prediction_status"], g["2026_04_DAL_HOU"]["bet_signal"],
                g["2026_04_DAL_HOU"]["prob_underdog_covers"]) == ("predicted", False, 0.47)
        assert (g["2026_04_ATL_NO"]["prediction_status"], g["2026_04_ATL_NO"]["line_status"],
                g["2026_04_ATL_NO"]["spread_line"], g["2026_04_ATL_NO"]["prob_underdog_covers"],
                g["2026_04_ATL_NO"]["bet_signal"]) == ("no_line", "missing", None, None, False)
        assert (g["2026_04_KC_LV"]["prediction_status"], g["2026_04_KC_LV"]["line_status"],
                g["2026_04_KC_LV"]["spread_line"], g["2026_04_KC_LV"]["prob_underdog_covers"],
                g["2026_04_KC_LV"]["underdog_team"]) == ("pickem", "pickem", 0.0, None, None)
        assert not g["2026_04_KC_LV"]["bet_signal"]
        assert g["2026_04_TB_GB"]["prediction_status"] == "no_probability"
        assert g["2026_04_TB_GB"]["underdog_team"] == "GB"            # away favored -> home underdog
        assert g["2026_04_NYJ_CHI"]["kickoff_utc"] == "2026-10-04T17:00:00Z"
        skipped = {s["game_id"]: s["reason"] for s in snap["skipped"]}
        assert skipped == {"2026_05_X_Y": ps.MISSING_KICKOFF, "2026_04_A_B": ps.NOT_BEFORE_KICKOFF}
        assert snap["counts"]["completed_excluded"] == 1
        assert "2026_04_PIT_CLE" not in json.dumps(snap)              # completed game never appears

    def test_no_outcome_or_settlement_fields(self, data_dir):
        allowed = {"season", "week", "game_id", "home_team", "away_team", "kickoff_utc", "kickoff_source",
                   "spread_line", "line_status", "underdog_team", "prob_underdog_covers", "bet_signal",
                   "prediction_status"}
        snap = _build(data_dir)
        assert all(set(g) == allowed for g in snap["games"])
        text = json.dumps(snap).lower()
        for word in ("home_score", "away_score", "result", "profit", "covered\"", "settle"):
            assert word not in text

    def test_exact_kickoff_is_excluded_one_second_earlier_is_kept(self, data_dir):
        ko = datetime(2026, 10, 4, 17, 0, tzinfo=UTC)                  # NYJ@CHI kickoff
        at = {g["game_id"] for g in _build(data_dir, ko)["games"]}
        before = {g["game_id"] for g in _build(data_dir, ko - timedelta(seconds=1))["games"]}
        assert "2026_04_NYJ_CHI" not in at and "2026_04_NYJ_CHI" in before

    def test_capture_requires_aware_time_after_generation(self, data_dir):
        with pytest.raises(ValueError):
            _build(data_dir, datetime(2026, 10, 3, 3, 0))              # naive
        with pytest.raises(ps.ProvenanceError):
            _build(data_dir, T0 - timedelta(hours=1))                  # before the run generated it


# ------------------------------------------------------------- provenance --

class TestProvenance:
    def test_manifest_records_generating_run_not_capture_checkout(self, tmp_path, monkeypatch):
        d = tmp_path / "data_files"
        _write_inputs(d, ROWS, PROBS, SIGNALS)
        _manifest(d, monkeypatch=monkeypatch, sha="generating-run-sha")
        monkeypatch.setenv("GITHUB_SHA", "later-checkout-sha")      # capture runs on another checkout
        path, created, snap = ps.capture(d, now=T0)
        assert created and snap["run"]["code_revision"] == "generating-run-sha"
        assert snap["run"]["artifact"]["sha256"] == ps.sha256_file(d / ps.PREDICTIONS_NAME)
        assert ps._RUN_ID_RE.match(snap["run_id"]) and snap["run_id"] == snap["run"]["run_id"]
        assert snap["run"]["config"]["missing"] == "nan"               # JSON-safe config

    @pytest.mark.parametrize("victim", [ps.PREDICTIONS_NAME, ps.SCHEDULE_NAME])
    def test_artifact_or_schedule_changed_after_run_is_refused(self, data_dir, victim):
        with open(data_dir / victim, "a") as f:
            f.write("\n")
        with pytest.raises(ps.ProvenanceError, match="does not match the pipeline run"):
            ps.capture(data_dir, now=T0)
        assert not (data_dir / "pregame_snapshots").exists()

    def test_missing_manifest_is_refused(self, data_dir):
        (data_dir / ps.MANIFEST_NAME).unlink()
        with pytest.raises(ps.ProvenanceError, match="not found"):
            ps.capture(data_dir, now=T0)


# ------------------------------------------------- immutability / retries --

class TestImmutability:
    def test_same_run_retry_is_idempotent(self, data_dir):
        p1, created1, _ = ps.capture(data_dir, now=T0)
        before = p1.read_bytes()
        p2, created2, snap2 = ps.capture(data_dir, now=T0 + timedelta(days=2))   # after some kickoffs
        assert (created1, created2, p1) == (True, False, p2)
        assert p1.read_bytes() == before                                          # original kept
        assert snap2["captured_at"] == "2026-10-03T03:00:00Z"
        assert len(list((data_dir / "pregame_snapshots").iterdir())) == 1

    def test_later_run_never_replaces_earlier_capture(self, data_dir):
        p1, _, _ = ps.capture(data_dir, now=T0)
        first = p1.read_bytes()
        probs = list(PROBS)
        probs[1] = 0.62                                                           # nightly retrain moved it
        _write_inputs(data_dir, ROWS, probs, SIGNALS)
        _manifest(data_dir, now=T0 + timedelta(hours=23, minutes=55))
        p2, created, _ = ps.capture(data_dir, now=T0 + timedelta(days=1))
        assert created and p2 != p1 and p1.read_bytes() == first
        assert len(list((data_dir / "pregame_snapshots").glob("*.json"))) == 2

    @pytest.mark.parametrize("existing", ["other-artifact", "not-json"])
    def test_conflicting_reuse_of_run_id_is_refused(self, data_dir, existing):
        run_id = ps.load_verified_manifest(data_dir)["run_id"]
        snap_dir = data_dir / "pregame_snapshots"
        snap_dir.mkdir()
        target = snap_dir / f"{run_id}.json"
        target.write_text(json.dumps({"run_id": run_id, "run": {"artifact": {"sha256": "f" * 64}}})
                          if existing == "other-artifact" else "{broken")
        before = target.read_bytes()
        with pytest.raises(ps.SnapshotConflictError, match="refusing to overwrite"):
            ps.capture(data_dir, now=T0)
        assert target.read_bytes() == before

    def test_write_failure_leaves_no_partial_file(self, data_dir, monkeypatch):
        def boom(src, dst):
            raise OSError("disk full")
        monkeypatch.setattr(ps.os, "link", boom)
        with pytest.raises(OSError, match="disk full"):
            ps.capture(data_dir, now=T0)
        assert list((data_dir / "pregame_snapshots").iterdir()) == []            # no final, no temp

    def test_concurrent_writer_of_same_run_wins_race_and_we_noop(self, data_dir, monkeypatch):
        real_link = os.link

        def racing_link(src, dst):
            real_link(src, dst)                     # the "other" writer lands the same run first
            raise FileExistsError(dst)
        monkeypatch.setattr(ps.os, "link", racing_link)
        path, created, _ = ps.capture(data_dir, now=T0)
        assert not created and path.exists()
        assert len(list(path.parent.iterdir())) == 1                             # temp cleaned up


# --------------------------------------------------------------- selection --

def _two_runs(data_dir):
    ps.capture(data_dir, now=T0)
    probs = list(PROBS)
    probs[1], probs[2] = 0.62, 0.56
    _write_inputs(data_dir, ROWS, probs, [0, 1, 1, 1, 1, 0, 0, 1])
    _manifest(data_dir, now=T0 + timedelta(hours=20))
    ps.capture(data_dir, now=T0 + timedelta(hours=20))
    return data_dir / "pregame_snapshots"


class TestSelection:
    def test_earliest_and_latest(self, data_dir):
        snap_dir = _two_runs(data_dir)
        early = ps.select_captures(snap_dir, "earliest").set_index("game_id")
        late = ps.select_captures(snap_dir, "latest").set_index("game_id")
        assert early.loc["2026_04_NYJ_CHI", "prob_underdog_covers"] == 0.58
        assert late.loc["2026_04_NYJ_CHI", "prob_underdog_covers"] == 0.62
        assert late.loc["2026_04_DAL_HOU", "bet_signal"]
        assert early.loc["2026_04_ATL_NO", "prediction_status"] == "no_line"
        assert early.index.is_unique and set(early.index) == set(late.index)

    def test_status_and_before_filters(self, data_dir):
        snap_dir = _two_runs(data_dir)
        only_pred = ps.select_captures(snap_dir, "latest", statuses=["predicted"])
        assert set(only_pred["prediction_status"]) == {"predicted"}
        asof = ps.select_captures(snap_dir, "latest", before=T0 + timedelta(hours=1)).set_index("game_id")
        assert asof.loc["2026_04_NYJ_CHI", "prob_underdog_covers"] == 0.58      # second run not yet captured

    def test_ties_broken_by_run_id_regardless_of_file_order(self, data_dir, tmp_path):
        base = _build(data_dir)                                       # a valid, sealed snapshot
        snap_dir = tmp_path / "snaps"
        snap_dir.mkdir()
        prefix = base["run_id"][:16]
        for name, suffix, prob in (("zz.json", "aaaaaaaaaaaa", 0.51), ("aa.json", "bbbbbbbbbbbb", 0.52)):
            doc = json.loads(json.dumps(base))
            doc["run_id"] = doc["run"]["run_id"] = f"{prefix}-{suffix}"
            doc["games"][0]["prob_underdog_covers"] = prob if doc["games"][0]["prediction_status"] == "predicted"                 else doc["games"][0]["prob_underdog_covers"]
            (snap_dir / name).write_text(json.dumps(_reseal(doc)))
        assert ps.select_captures(snap_dir, "earliest").iloc[0]["run_id"].endswith("aaaaaaaaaaaa")
        assert ps.select_captures(snap_dir, "latest").iloc[0]["run_id"].endswith("bbbbbbbbbbbb")

    def test_capture_at_or_after_kickoff_is_rejected_not_selected(self, data_dir, tmp_path):
        doc = _build(data_dir)
        doc["captured_at"] = doc["games"][0]["kickoff_utc"]           # forged: captured AT kickoff
        snap_dir = tmp_path / "snaps"
        snap_dir.mkdir()
        (snap_dir / "x.json").write_text(json.dumps(_reseal(doc)))
        with pytest.raises(ps.SnapshotValidationError, match="not strictly after captured_at"):
            ps.select_captures(snap_dir, "latest")

    def test_selection_is_read_only_and_detects_tampering(self, data_dir):
        snap_dir = _two_runs(data_dir)
        state = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in snap_dir.iterdir()}
        ps.select_captures(snap_dir, "earliest")
        ps.select_captures(snap_dir, "latest", statuses=["predicted"])
        assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in snap_dir.iterdir()} == state
        victim = sorted(snap_dir.iterdir())[0]
        doc = json.loads(victim.read_text())
        doc["games"][0]["prob_underdog_covers"] = 0.99
        victim.write_text(json.dumps(doc))
        with pytest.raises(ps.SnapshotValidationError, match="changed after capture"):
            ps.select_captures(snap_dir)

    def test_cli_refuses_output_inside_snapshot_dir(self, data_dir):
        snap_dir = _two_runs(data_dir)
        with pytest.raises(SystemExit, match="inside the snapshot directory"):
            ps.main(["select", "--snapshot-dir", str(snap_dir), "--output", str(snap_dir / "x.csv")])
        out = data_dir / "picks.csv"
        assert ps.main(["select", "--snapshot-dir", str(snap_dir), "--output", str(out)]) == 0
        assert len(pd.read_csv(out)) == 5


# ---------------------------------------------------- pipeline integration --

def _run_pipeline(data_dir, name):
    gd = _load(name, "nfl-gather-data.py")
    gd.DATA_DIR = str(data_dir) + "/"
    gd.XGBClassifier = _FastClassifier
    gd._LGBM_AVAILABLE = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gd.main()


class TestPipelineManifest:
    def test_manifest_valid_and_determinism_intact(self, tmp_path):
        raw = _schedule(seasons=(2018, 2019, 2020, 2021), weeks=12, unplayed_from=(2021, 9))
        outputs = []
        for i in range(2):
            d = tmp_path / f"run{i}"
            d.mkdir()
            raw.to_csv(d / ps.SCHEDULE_NAME, sep="\t", index=False)
            _run_pipeline(d, f"nfl_gather_manifest_{i}")
            outputs.append(d)
        for f in ("nfl_games_historical_with_predictions.csv", "model_metrics.json", "best_features_spread.txt"):
            assert (outputs[0] / f).read_bytes() == (outputs[1] / f).read_bytes(), f   # determinism intact
        m = [ps.load_verified_manifest(d) for d in outputs]                            # hashes validate
        volatile = {"run_id", "generated_at", "ci_run"}
        assert {k: v for k, v in m[0].items() if k not in volatile} == \
               {k: v for k, v in m[1].items() if k not in volatile}
        assert m[0]["artifact"]["sha256"] == ps.sha256_file(outputs[0] / ps.PREDICTIONS_NAME)
        assert m[0]["training_cutoff"]["season"] <= m[0]["data_cutoff"]["season"]
        assert m[0]["features"] and len(m[0]["feature_set_id"]) == 16 and len(m[0]["config_id"]) == 16

    def test_failed_run_leaves_no_stale_manifest(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        (d / ps.MANIFEST_NAME).write_text("{}")                                       # from an older run
        _schedule(seasons=(2019, 2020), weeks=6).sample(frac=1, random_state=0).to_csv(
            d / ps.SCHEDULE_NAME, sep="\t", index=False)                               # unsorted -> fails
        with pytest.raises(ValueError):
            _run_pipeline(d, "nfl_gather_manifest_fail")
        assert not (d / ps.MANIFEST_NAME).exists()

    def test_capture_after_real_pipeline_run(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        raw = _schedule(seasons=(2018, 2019, 2020, 2021), weeks=12, unplayed_from=(2021, 9))
        # Move the synthetic calendar 12 years ahead so the unplayed games are in the future.
        raw["gameday"] = (pd.to_datetime(raw["gameday"]) + pd.DateOffset(years=12)).dt.strftime("%Y-%m-%d")
        raw.to_csv(d / ps.SCHEDULE_NAME, sep="	", index=False)
        _run_pipeline(d, "nfl_gather_manifest_capture")
        path, created, snap = ps.capture(d)                      # real clock, after the real run
        assert created and path.parent == d / "pregame_snapshots"
        unplayed = raw[raw["home_score"].isna()]
        assert {g["game_id"] for g in snap["games"]} == set(unplayed["game_id"])
        assert snap["counts"]["completed_excluded"] == int(raw["home_score"].notna().sum())
        by_status = pd.Series([g["prediction_status"] for g in snap["games"]]).value_counts().to_dict()
        assert by_status == {"predicted": int(unplayed["spread_line"].notna().sum()),
                             "no_line": int(unplayed["spread_line"].isna().sum())}
        preds = pd.read_csv(d / ps.PREDICTIONS_NAME, sep="	").set_index("game_id")
        for g in snap["games"]:
            if g["prediction_status"] == "predicted":
                assert g["prob_underdog_covers"] == pytest.approx(preds.loc[g["game_id"], "prob_underdogCovered"])
                assert g["bet_signal"] == bool(preds.loc[g["game_id"], "pred_spreadCovered_optimal"])


# ============================================================ review blockers ==

def _edit_manifest(data_dir, **changes):
    """Rewrite the manifest on disk with changes, KEEPING its run_id (a reused identity)."""
    path = data_dir / ps.MANIFEST_NAME
    m = json.loads(path.read_text())
    m.update(changes)
    if "config" in changes:
        m["config_id"] = ps._short_hash(m["config"])
    if "features" in changes:
        m["feature_set_id"] = ps._short_hash(m["features"])
    path.write_text(json.dumps(m))
    return m


class TestReadOnce:
    """Blocker 1: the hash always describes the exact bytes that were parsed."""

    def test_capture_parses_the_verified_bytes_not_a_reopened_file(self, data_dir, monkeypatch):
        real = ps.read_bytes_once
        calls = []

        def read_then_swap(path):
            data, digest = real(path)
            calls.append(Path(path).name)
            if Path(path).name == ps.PREDICTIONS_NAME:          # file changes right after the read
                Path(path).write_text(Path(path).read_text().replace("0.58", "0.99"))
            return data, digest
        monkeypatch.setattr(ps, "read_bytes_once", read_then_swap)
        _, created, snap = ps.capture(data_dir, now=T0)
        assert created and sorted(calls) == sorted([ps.PREDICTIONS_NAME, ps.SCHEDULE_NAME])   # once each
        nyj = next(g for g in snap["games"] if g["game_id"] == "2026_04_NYJ_CHI")
        assert nyj["prob_underdog_covers"] == 0.58                 # the verified bytes, not the swapped file

    def test_pipeline_reads_schedule_once_and_hashes_what_it_parsed(self, tmp_path, monkeypatch):
        d = tmp_path / "data"
        d.mkdir()
        raw = _schedule(seasons=(2018, 2019, 2020, 2021), weeks=12, unplayed_from=(2021, 9))
        raw.to_csv(d / ps.SCHEDULE_NAME, sep="\t", index=False)
        original = (d / ps.SCHEDULE_NAME).read_bytes()
        gd = _load("nfl_gather_read_once", "nfl-gather-data.py")
        gd.DATA_DIR = str(d) + "/"
        gd.XGBClassifier = _FastClassifier
        gd._LGBM_AVAILABLE = False
        reads = []
        real_read = gd.read_bytes_once

        def spy(path):
            out = real_read(path)
            reads.append(Path(path).name)
            Path(path).write_bytes(original + b"\n")                # schedule replaced after the read
            return out
        monkeypatch.setattr(gd, "read_bytes_once", spy)
        real_csv = gd.pd.read_csv

        def no_path_reads(src, *a, **kw):
            assert not str(src).endswith(ps.SCHEDULE_NAME), "schedule re-opened by path"
            return real_csv(src, *a, **kw)
        monkeypatch.setattr(gd.pd, "read_csv", no_path_reads)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gd.main()
        manifest = json.loads((d / ps.MANIFEST_NAME).read_text())
        assert reads == [ps.SCHEDULE_NAME]
        assert manifest["schedule"]["sha256"] == ps.sha256_bytes(original)   # what was parsed, not the swap


class TestRunIdentity:
    """Blocker 2: one identity per run, reused by retries; any provenance change conflicts."""

    def test_identity_is_unique_per_run_even_with_identical_inputs(self, tmp_path):
        d = tmp_path / "data_files"
        _write_inputs(d, ROWS, PROBS, SIGNALS)
        ids = {_manifest(d)["run_id"] for _ in range(5)}            # same time, same bytes
        assert len(ids) == 5 and all(ps._RUN_ID_RE.match(i) for i in ids)

    def test_retry_after_kickoff_reuses_manifest_and_preserves_original(self, data_dir):
        p1, created1, _ = ps.capture(data_dir, now=T0)
        before = p1.read_bytes()
        p2, created2, snap = ps.capture(data_dir, now=T0 + timedelta(days=30))   # every game kicked off
        assert (created1, created2, p1) == (True, False, p2)
        assert p1.read_bytes() == before and snap["counts"]["captured"] == 5
        assert len(list(p1.parent.glob("*.json"))) == 1

    @pytest.mark.parametrize("change", [
        {"code_revision": "a-different-commit"},
        {"code_sha256": "0" * 64},
        {"code_dirty": True},
        {"config": {"model": "xgb", "n_estimators": 300}},
        {"features": ["homeTeamWinPct"]},
        {"training_cutoff": {"season": 2024, "week": 1, "gameday": "2024-09-05", "games": 1100}},
        {"data_cutoff": {"season": 2026, "week": 3, "gameday": "2026-09-27", "games": 1700}},
        {"spread_threshold": 0.55},
    ], ids=lambda c: next(iter(c)))
    def test_changed_provenance_with_identical_prediction_bytes_conflicts(self, data_dir, change):
        path, _, _ = ps.capture(data_dir, now=T0)
        before = path.read_bytes()
        m = _edit_manifest(data_dir, **change)                      # same run_id, same artifact bytes
        assert m["artifact"]["sha256"] == ps.sha256_file(data_dir / ps.PREDICTIONS_NAME)
        with pytest.raises(ps.SnapshotConflictError, match="different provenance"):
            ps.capture(data_dir, now=T0 + timedelta(minutes=1))
        assert path.read_bytes() == before

    def test_changed_schedule_with_identical_prediction_bytes_conflicts(self, data_dir):
        path, _, _ = ps.capture(data_dir, now=T0)
        sched = data_dir / ps.SCHEDULE_NAME
        sched.write_bytes(sched.read_bytes() + b"\n")               # different schedule bytes, same parse
        _edit_manifest(data_dir, schedule={"path": ps.SCHEDULE_NAME, "sha256": ps.sha256_file(sched)})
        with pytest.raises(ps.SnapshotConflictError, match="schedule"):
            ps.capture(data_dir, now=T0)

    def test_concurrent_writers_produce_exactly_one_file(self, data_dir):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: ps.capture(data_dir, now=T0), range(8)))
        assert sum(created for _, created, _ in results) == 1
        assert len({str(p) for p, _, _ in results}) == 1
        assert len({json.dumps(s, sort_keys=True) for _, _, s in results}) == 1
        assert [p.name for p in (data_dir / "pregame_snapshots").iterdir()] == [results[0][0].name]


def _mutations():
    """(id, mutate(doc)) - each makes a sealed snapshot structurally invalid."""
    def game(i, **kw):
        def f(d):
            d["games"][i].update(kw)
        return f

    def on_status(status, **kw):
        def f(d):
            d["games"][next(i for i, g in enumerate(d["games"]) if g["prediction_status"] == status)].update(kw)
        return f

    on_pred = lambda **kw: on_status("predicted", **kw)            # noqa: E731

    return [
        ("schema_version", lambda d: d.update(schema_version=2)),
        ("kind", lambda d: d.update(kind="something_else")),
        ("missing_counts", lambda d: d.pop("counts")),
        ("extra_game_field", game(0, home_score=24)),
        ("season_type", game(0, season="2026")),
        ("week_range", game(0, week=23)),
        ("game_id_mismatch", game(0, game_id="2026_04_XXX_YYY")),
        ("duplicate_game", lambda d: d["games"].append(dict(d["games"][0]))),
        ("team_format", game(0, home_team="C HI")),
        ("kickoff_not_utc", game(0, kickoff_utc="2026-10-04T13:00:00-04:00")),
        ("captured_not_utc", lambda d: d.update(captured_at="2026-10-03T03:00:00+00:00")),
        ("captured_before_generated", lambda d: d.update(captured_at="2000-01-01T00:00:00Z")),
        ("unknown_status", on_pred(prediction_status="maybe")),
        ("line_status_mismatch", on_pred(line_status="missing")),
        ("missing_line_with_value", on_status("no_line", spread_line=3.5)),
        ("pickem_nonzero", on_status("pickem", spread_line=3.0)),
        ("wrong_underdog", on_pred(underdog_team="ZZZ")),
        ("prob_out_of_range", on_pred(prob_underdog_covers=1.5)),
        ("prob_not_float", on_pred(prob_underdog_covers=1)),
        ("prob_on_no_line", on_status("no_line", prob_underdog_covers=0.5)),
        ("signal_not_bool", on_pred(bet_signal="yes")),
        ("signal_on_no_probability", on_status("no_probability", bet_signal=True)),
        ("counts_mismatch", lambda d: d["counts"].update(captured=99)),
        ("run_id_mismatch", lambda d: d.update(run_id=d["run_id"][:-1] + ("0" if d["run_id"][-1] != "0" else "1"))),
        ("config_id_mismatch", lambda d: d["run"].update(config_id="0" * 16)),
        ("features_unsorted", lambda d: d["run"].update(features=list(reversed(d["run"]["features"])))),
        ("threshold_range", lambda d: d["run"].update(spread_threshold=1.5)),
        ("bad_skip_reason", lambda d: d["skipped"][0].update(reason="whatever")),
        ("timezone", lambda d: d.update(kickoff_timezone="UTC")),
    ]


class TestSchemaAndIntegrity:
    """Blocker 3: full-payload checksum + structural validation everywhere."""

    @pytest.mark.parametrize("name,mutate", _mutations(), ids=[m[0] for m in _mutations()])
    def test_malformed_snapshot_rejected_even_when_resealed(self, data_dir, name, mutate):
        doc = _build(data_dir)
        ps.validate_snapshot(doc)                                    # the original is valid
        mutate(doc)
        with pytest.raises(ps.SnapshotValidationError):
            ps.validate_snapshot(_reseal(doc))

    @pytest.mark.parametrize("mutate", [
        lambda d: d.update(captured_at="2026-10-03T02:59:59Z"),            # altered capture time
        lambda d: d["games"][0].update(kickoff_utc="2026-10-05T17:00:00Z"),  # altered kickoff
        lambda d: d["run"].update(code_revision="forged"),                   # altered provenance
        lambda d: d["counts"].update(completed_excluded=d["counts"]["completed_excluded"] + 1),
        lambda d: d["skipped"][0].update(gametime="13:00"),
    ], ids=["captured_at", "kickoff_utc", "provenance", "counts", "skipped"])
    def test_checksum_covers_the_whole_payload(self, data_dir, mutate):
        doc = _build(data_dir)
        mutate(doc)
        with pytest.raises(ps.SnapshotValidationError, match="changed after capture"):
            ps.validate_snapshot(doc)

    def test_existing_snapshot_is_validated_before_acceptance(self, data_dir):
        path, _, _ = ps.capture(data_dir, now=T0)
        doc = json.loads(path.read_text())
        doc["captured_at"] = "2026-10-03T02:59:00Z"                 # altered capture time, not resealed
        path.write_text(json.dumps(doc))
        tampered = path.read_bytes()
        with pytest.raises(ps.SnapshotConflictError, match="not a valid snapshot"):
            ps.capture(data_dir, now=T0)
        assert path.read_bytes() == tampered                         # still never overwritten
        with pytest.raises(ps.SnapshotValidationError, match="changed after capture"):
            ps.select_captures(path.parent)

    def test_malformed_file_blocks_selection(self, data_dir):
        path, _, _ = ps.capture(data_dir, now=T0)
        doc = _reseal({**json.loads(path.read_text()), "schema_version": 9})
        path.write_text(json.dumps(doc))
        with pytest.raises(ps.SnapshotValidationError, match="unsupported schema"):
            ps.select_captures(path.parent)

    def test_resealed_edit_passes_the_unkeyed_checksum(self, data_dir):
        # Documented limitation: a deliberate, internally consistent edit that is
        # resealed can't be told apart by an unkeyed checksum.
        doc = _build(data_dir)
        doc["captured_at"] = "2026-10-03T02:58:00Z"                 # still after generated_at
        ps.validate_snapshot(_reseal(doc))

    def test_invalid_manifest_is_a_provenance_error(self, data_dir):
        _edit_manifest(data_dir, config_id="f" * 16)               # inconsistent with config
        with pytest.raises(ps.ProvenanceError, match="invalid manifest"):
            ps.capture(data_dir, now=T0)


WORKFLOW = ROOT / ".github" / "workflows" / "nightly-update.yml"


@pytest.fixture(scope="module")
def steps():
    import yaml
    wf = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = wf["jobs"]["update-predictions"]["steps"]
    return {s["name"]: s for s in steps}, [s["name"] for s in steps]


class TestWorkflowFailureReporting:
    """Blocker 4: a failed capture can't pass silently or be reported as success."""

    def test_capture_runs_right_after_pipeline_and_reports_outcome(self, steps):
        by_name, order = steps
        cap = by_name["Capture pregame spread snapshot"]
        assert order[order.index("Run prediction pipeline") + 1] == cap["name"]
        assert cap["id"] == "pregame_capture"
        assert "set -o pipefail" in cap["run"] and "GITHUB_STEP_SUMMARY" in cap["run"]
        assert "continue-on-error is deliberate" in WORKFLOW.read_text(encoding="utf-8")

    def test_failed_capture_fails_the_run_after_publishing(self, steps):
        by_name, order = steps
        fail = by_name["Fail the run if the pregame capture failed"]
        assert order[-1] == fail["name"]
        assert "always()" in fail["if"] and "steps.pregame_capture.outcome == 'failure'" in fail["if"]
        assert "::error" in fail["run"] and "GITHUB_STEP_SUMMARY" in fail["run"] and "exit 1" in fail["run"]

    def test_summary_does_not_claim_success(self, steps):
        by_name, _ = steps
        run = by_name["Summary"]["run"]
        assert "completed successfully" not in run
        assert "steps.pregame_capture.outcome" in run
