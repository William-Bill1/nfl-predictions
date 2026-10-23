"""Test helpers: tiny synthetic prop models trained only inside tests, and
verified bundles published from them (player_props/model_bundle.py)."""
import numpy as np
import pandas as pd

from player_props import features as fx
from player_props import model_bundle as mb


def synthetic_rows(prop="passing_yards", n_train=200, n_eval=100, seed=0, line=None):
    """Training-style rows for one prop: two seasons (2024 train, 2025 evaluate),
    every feature in the contract, and a stat that rises with the L3 average so
    the model learns a real OVER direction."""
    rng = np.random.default_rng(seed)
    stat_col = fx.PROP_STAT[prop][1]
    line = fx.PROP_LINES[prop]["starter" if "starter" in fx.PROP_LINES[prop] else "anytime"] \
        if line is None else line
    n = n_train + n_eval
    rows = pd.DataFrame({c: rng.uniform(0.5, 2.0, n) * max(line, 1.0) for c in fx.rolling_columns(prop)})
    rows["is_home"] = rng.integers(0, 2, n)
    rows["days_rest"] = rng.integers(4, 15, n)
    lead = fx.rolling_columns(prop)[0]
    rows[stat_col] = rows[lead] + rng.normal(0, 0.3 * max(line, 1.0), n)
    rows["season"] = [2024] * n_train + [2025] * n_eval
    days = list(pd.date_range("2024-09-08", periods=n_train, freq="D").strftime("%Y-%m-%d")) + \
        list(pd.date_range("2025-09-07", periods=n_eval, freq="D").strftime("%Y-%m-%d"))
    rows["game_date"] = days
    rows["week"] = 1
    return rows


def trained_entry(prop="passing_yards", line_type=None, seed=0):
    """A bundle entry from the real training function on synthetic rows."""
    from player_props import models
    line_type = line_type or ("starter" if "starter" in fx.PROP_LINES[prop] else "anytime")
    line = fx.PROP_LINES[prop][line_type]
    rows = synthetic_rows(prop, seed=seed, line=line)
    entry, reason = models.train_prop_model(rows, fx.model_features(prop), prop, line_type, line)
    assert entry is not None, reason
    return entry, rows


def training_source(data_dir):
    from player_props import models
    inputs, problems = models.training_inputs(data_dir)
    assert inputs is not None, problems
    return inputs["source"]


def publish_test_bundle(models_dir, data_dir, specs=(("passing_yards", "starter"),), seed=0):
    """Publish a verified bundle of tiny synthetic models; returns its directory."""
    entries = {}
    for i, (prop, line_type) in enumerate(specs):
        entry, _ = trained_entry(prop, line_type, seed=seed + i)
        entries[f"{prop}_{line_type}"] = entry
    return mb.publish_bundle(models_dir, entries, training_source(data_dir), fx.PROP_LINES)
