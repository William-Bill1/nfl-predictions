"""Console encoding must not mask training/prediction diagnostics."""
import io
import gzip
import shutil
import os
from pathlib import Path
import subprocess
import sys

import pytest

from player_props.console import say


@pytest.mark.parametrize("encoding", ["cp1252", "ascii", "utf-8"])
def test_text_and_stream_preserved(monkeypatch, encoding):
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding=encoding, errors="strict", newline="\n")
    monkeypatch.setattr(sys, "stdout", stream)
    original = "Łukasz Żółć café 🎯"
    say(original)
    say()
    stream.flush()
    assert raw.getvalue().decode(encoding) == original.encode(encoding, "backslashreplace").decode(encoding) + "\n\n"
    assert stream.encoding == encoding
    assert stream.errors == "strict"
    assert original == "Łukasz Żółć café 🎯"


def test_stringio_preserves_unicode(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    say("Łukasz 🎯")
    assert stream.getvalue() == "Łukasz 🎯\n"


@pytest.mark.parametrize("error", [BrokenPipeError, OSError, ValueError])
def test_real_write_errors_propagate(monkeypatch, error):
    class Broken:
        encoding = "cp1252"
        def write(self, text):
            raise error("original write failure")
    monkeypatch.setattr(sys, "stdout", Broken())
    with pytest.raises(error, match="original write failure"):
        say("🎯")


@pytest.mark.parametrize("encoding", ["cp1252", "utf-8"])
@pytest.mark.parametrize("entry", ["models", "predict"])
def test_real_refusal_path(tmp_path, encoding, entry):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    for key in ("PYTHONUTF8", "PYTHONIOENCODING", "PYTHONPATH", "PYTHONHOME", "SSLKEYLOGFILE"):
        env.pop(key, None)
    env.update(PYTHONIOENCODING=encoding, PYTHONDONTWRITEBYTECODE="1")
    code = """
import sys
from pathlib import Path
from player_props import models, predict
p = Path(sys.argv[1])
def unexpected(*a, **k):
    raise AssertionError('training or network must not run')
models.train_prop_model = unexpected
models.training_inputs = lambda *a: (None, ['Łukasz: unverified inputs'])
predict.load_schedule = lambda *a, **k: None
if sys.argv[2] == 'models':
    result = models.train_all_models(p / 'data', p / 'models')
    assert result['published'] is False
else:
    result = predict.generate_predictions(2026, 3, data_dir=p / 'data', models_dir=p / 'models', skip_injuries=True, skip_weather=True, skip_market_odds=True)
    assert result is None
assert not list(p.iterdir())
"""
    result = subprocess.run([os.environ.get("CONSOLE_TEST_PYTHON", sys.executable), "-B", "-c", code, str(tmp_path), entry], cwd=root, env=env, capture_output=True)
    assert result.returncode == 0, result.stderr.decode(encoding, errors="replace")
    assert b"UnicodeEncodeError" not in result.stderr


@pytest.fixture(scope="module")
def successful_inputs(tmp_path_factory):
    """Small genuine stats manifest and model bundle, generated only from tests."""
    import pandas as pd
    import xgboost as xgb
    import test_player_props_stage2 as stage2
    from test_player_props_stage3 import _players
    from prop_bundle_helpers import publish_test_bundle

    root = tmp_path_factory.mktemp("console_synthetic")
    plays = stage2.pbp()
    plays["passer_player_name"] = plays["passer_player_name"].replace({"KC.QB": "Łukasz Żółć"})
    # The source fixture writer uses the OS locale; this case needs a UTF-8 source.
    def utf8_source(path, frame):
        with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, sep="\t", index=False)
        return path
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(stage2, "write_source", utf8_source)
        data = stage2.league(root, plays=plays)
    stage2._prediction_files(data)
    _players(data)
    players = pd.read_csv(data / "players.csv")
    players.loc[players.gsis_id == "KC-QB", ["short_name", "display_name"]] = "Łukasz Żółć"
    players.to_csv(data / "players.csv", index=False, encoding="utf-8")
    for path in data.glob("player_props_predictions*.csv"):
        path.unlink()  # Only generated test-fixture placeholders.
    models = root / "models"
    with xgb.config_context(nthread=1):
        publish_test_bundle(models, data)
    return data, models


def _files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in root.rglob("*") if p.is_file()}


def test_success_preserves_unicode_csv_and_frozen_snapshot(successful_inputs, tmp_path):
    data, models = successful_inputs
    before = _files(data), _files(models)
    root = Path(__file__).resolve().parents[1]
    code = r"""
import importlib.util
from pathlib import Path
import sys
spec = importlib.util.spec_from_file_location('console_socket_guard', 'tests/no_network_plugin.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
guard.pytest_configure(None)
from player_props import predict
predict.ROSTER_FILTER_ON = False
kwargs = dict(season=2026, week=3, data_dir=Path(sys.argv[1]), models_dir=Path(sys.argv[2]),
              skip_injuries=True, skip_weather=True, skip_market_odds=True)
assert predict.generate_predictions(**kwargs) is None
feed = Path(sys.argv[1]) / 'player_props_predictions.csv'
snap = Path(sys.argv[1]) / 'player_props_predictions_week3_2026.csv'
assert feed.read_bytes() == snap.read_bytes()
original = snap.read_bytes()
assert predict.generate_predictions(**kwargs) is None
assert snap.read_bytes() == original
"""
    outputs = []
    for encoding in ("cp1252", "utf-8"):
        target = tmp_path / encoding
        shutil.copytree(data, target)
        env = dict(os.environ)
        for key in ("PYTHONUTF8", "PYTHONIOENCODING", "PYTHONPATH", "PYTHONHOME", "SSLKEYLOGFILE"):
            env.pop(key, None)
        env.update(PYTHONIOENCODING=encoding, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
        result = subprocess.run([os.environ.get("CONSOLE_TEST_PYTHON", sys.executable), "-B", "-c",
                                 code, str(target), str(models)], cwd=root, env=env, capture_output=True,
                                timeout=120)
        assert result.returncode == 0, result.stderr.decode(encoding, errors="replace")
        console = result.stdout.decode(encoding)
        expected = "Łukasz Żółć".encode(encoding, "backslashreplace").decode(encoding)
        assert expected in console  # The real top-picks summary prints the name.
        feed = (target / "player_props_predictions.csv").read_bytes()
        assert "Łukasz Żółć" in feed.decode("utf-8")
        import pandas as pd
        from io import BytesIO
        frame = pd.read_csv(BytesIO(feed))
        assert len(frame) == 4
        assert {"raw_model_prob_over", "prob_over", "player_id"} <= set(frame.columns)
        outputs.append(feed)
    assert outputs[0] == outputs[1]
    assert (_files(data), _files(models)) == before
