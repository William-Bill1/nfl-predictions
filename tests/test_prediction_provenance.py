"""Home-page "Predictions last generated" caption (prediction_provenance.py).

Manifests are written by the pipeline's own pregame_snapshots.write_run_manifest
into temporary directories; nothing in data_files/ is read or written.
"""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

import pregame_snapshots as ps
import prediction_provenance as pp
from test_team_features import ROOT

UTC = timezone.utc
GEN = datetime(2026, 10, 21, 8, 12, 27, tzinfo=UTC)          # 04:12 EDT
CUT = {"season": 2026, "week": 6, "gameday": "2026-10-19", "games": 96}


def make_run(tmp_path, now=GEN, content="season\tweek\n2026\t7\n"):
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    (d / ps.PREDICTIONS_NAME).write_text(content, encoding="utf-8")
    m = ps.write_run_manifest(
        d, schedule_sha256="c" * 64, code_files=[], repo_dir=ROOT, config={"model": "xgb"},
        features=["spread_line"], training_cutoff=CUT,
        data_cutoff=CUT, spread_threshold=0.5438, now=now)
    return d, m


def rewrite_manifest(d, mutate):
    p = d / ps.MANIFEST_NAME
    doc = json.loads(p.read_text())
    mutate(doc)
    p.write_text(json.dumps(doc))


class TestVerified:
    def test_valid_matching_manifest(self, tmp_path):
        d, m = make_run(tmp_path)
        at, run_id, reason = pp.verified_generation_time(d)
        assert (at, run_id, reason) == (GEN, m["run_id"], "")
        assert pp.generation_caption(d) == (
            f"Predictions last generated: 2026-10-21 04:12 EDT (pipeline run {m['run_id']})")

    def test_file_modification_time_is_never_used(self, tmp_path):
        # Regression: a checkout/copy that resets the file's mtime used to change the caption.
        d, _ = make_run(tmp_path)
        before = pp.generation_caption(d)
        for f in (d / ps.PREDICTIONS_NAME, d / ps.MANIFEST_NAME):
            os.utime(f, (1_900_000_000, 1_900_000_000))           # 2030
        assert pp.generation_caption(d) == before and "2030" not in before
        assert "04:12 EDT" in before

    def test_not_the_current_time(self, tmp_path, monkeypatch):
        d, _ = make_run(tmp_path)

        class Frozen(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2031, 1, 1, tzinfo=tz or UTC)
        monkeypatch.setattr(pp, "datetime", Frozen, raising=False)
        assert "2026-10-21 04:12 EDT" in pp.generation_caption(d)


class TestUnavailable:
    def _assert_unavailable(self, d, why):
        at, run_id, reason = pp.verified_generation_time(d)
        assert at is None and run_id is None and why in reason
        cap = pp.generation_caption(d)
        assert cap.startswith("Predictions last generated: Generation time unavailable (")
        assert not re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", cap)  # no time shown at all
        return cap

    def test_missing_manifest(self, tmp_path):
        d, _ = make_run(tmp_path)
        (d / ps.MANIFEST_NAME).unlink()
        self._assert_unavailable(d, "no pipeline run manifest")

    @pytest.mark.parametrize("text", ["{not json", "[]", "", "null"])
    def test_malformed_manifest(self, tmp_path, text):
        d, _ = make_run(tmp_path)
        (d / ps.MANIFEST_NAME).write_text(text)
        self._assert_unavailable(d, "malformed")

    @pytest.mark.parametrize("mutate", [
        lambda m: m.pop("generated_at"),
        lambda m: m.update(generated_at="2026-10-21 08:12:27"),           # not UTC form
        lambda m: m.update(generated_at="2026-10-21T09:00:00Z"),          # disagrees with run_id
        lambda m: m.update(schema_version=2),
        lambda m: m.update(artifact="x"),
    ])
    def test_invalid_manifest_fields(self, tmp_path, mutate):
        d, _ = make_run(tmp_path)
        rewrite_manifest(d, mutate)
        self._assert_unavailable(d, "malformed")

    def test_manifest_for_a_different_predictions_file(self, tmp_path):
        # Regression: a manifest from another run must not lend its time to this file.
        d, _ = make_run(tmp_path)
        (d / ps.PREDICTIONS_NAME).write_text("season\tweek\n2026\t8\n", encoding="utf-8")
        self._assert_unavailable(d, "differs from the one the pipeline run recorded")

    def test_manifest_naming_another_artifact(self, tmp_path):
        d, _ = make_run(tmp_path)
        rewrite_manifest(d, lambda m: m["artifact"].update(path="other.csv"))
        self._assert_unavailable(d, "malformed")            # validate_manifest rejects it

    def test_missing_predictions_file(self, tmp_path):
        d, _ = make_run(tmp_path)
        (d / ps.PREDICTIONS_NAME).unlink()
        self._assert_unavailable(d, "can't be read")


class TestToronto:
    @pytest.mark.parametrize("utc,expected", [
        (datetime(2026, 10, 21, 8, 12, 27, tzinfo=UTC), "2026-10-21 04:12 EDT"),
        (datetime(2026, 11, 1, 5, 30, tzinfo=UTC), "2026-11-01 01:30 EDT"),   # before fall-back
        (datetime(2026, 11, 1, 6, 30, tzinfo=UTC), "2026-11-01 01:30 EST"),   # after: same wall time
        (datetime(2027, 1, 15, 9, 5, tzinfo=UTC), "2027-01-15 04:05 EST"),
        (datetime(2027, 3, 14, 6, 59, tzinfo=UTC), "2027-03-14 01:59 EST"),   # before spring-forward
        (datetime(2027, 3, 14, 7, 0, tzinfo=UTC), "2027-03-14 03:00 EDT"),    # after
        (datetime(2026, 10, 22, 3, 30, tzinfo=UTC), "2026-10-21 23:30 EDT"),  # previous Toronto day
    ])
    def test_utc_to_toronto(self, utc, expected):
        assert pp.toronto_text(utc) == expected

    @pytest.mark.parametrize("now,label", [(datetime(2026, 7, 1, 9, 0, tzinfo=UTC), "05:00 EDT"),
                                           (datetime(2027, 1, 15, 9, 0, tzinfo=UTC), "04:00 EST")])
    def test_caption_label_follows_dst(self, tmp_path, now, label):
        d, _ = make_run(tmp_path, now=now)
        assert label in pp.generation_caption(d)

    def test_independent_of_the_machine_timezone(self, tmp_path, monkeypatch):
        d, _ = make_run(tmp_path)
        monkeypatch.setenv("TZ", "UTC")
        if hasattr(__import__("time"), "tzset"):
            __import__("time").tzset()
        assert "2026-10-21 04:12 EDT" in pp.generation_caption(d)


def test_display_only_writes_nothing(tmp_path):
    d, _ = make_run(tmp_path)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in d.iterdir()}
    pp.generation_caption(d)
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in d.iterdir()} == before


def test_home_page_uses_verified_caption_not_mtime():
    src = (ROOT / "predictions.py").read_text(encoding="utf-8")
    block = src[src.index("st.header('NFL Game Outcome Predictor')"):
                src.index("# Honest, up-front reality check")]
    assert "generation_caption(DATA_DIR)" in block
    assert "getmtime" not in block and "st_mtime" not in block and "datetime.now" not in block
    assert "Generation time unavailable" in block
