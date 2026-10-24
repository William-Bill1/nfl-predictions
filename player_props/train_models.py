"""
Standalone CLI script to train all NFL player prop models.

Usage:
    python player_props/train_models.py

Workflow:
    1. Load play-by-play data (historical CSV from data_files/), recording its
       content hash and actual coverage
    2. Aggregate to game-level player stats (passing, rushing, receiving)
    3. Calculate exponentially-weighted rolling averages
    4. Add matchup features (is_home, days_rest) and save the validated stats
       with their provenance manifest
    5. Train one XGBoost classifier per prop line (no ensemble), evaluated with
       the scorer prediction uses - only from stats whose provenance manifest
       verifies
    6. Publish the models, metrics and manifest as one verified bundle under
       player_props/models/bundles/ and switch player_props/models/current_bundle.json
       (exit 1 when no bundle is published)
"""
from pathlib import Path
import sys

# Ensure project root is on path when running from any directory
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from player_props.aggregators import (
    build_player_stats,
    load_matchup_schedule,
    write_player_stats,
)
from player_props.models import train_all_models
from player_props.provenance import read_source, stats_manifest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / 'data_files'
PBP_FILE = DATA_DIR / 'nfl_play_by_play_historical.csv.gz'
PBP_FILE_UNCOMPRESSED = DATA_DIR / 'nfl_play_by_play_historical.csv'

ROLLING_WINDOWS = [3, 5, 10]

# Columns that must be present in the PBP data
REQUIRED_PBP_COLS = [
    'game_id', 'season', 'week', 'posteam', 'defteam', 'game_date',
    'passer_player_id', 'passer_player_name',
    'rusher_player_id', 'rusher_player_name',
    'receiver_player_id', 'receiver_player_name',
    'passing_yards', 'rushing_yards', 'receiving_yards',
    'pass_touchdown', 'rush_touchdown',
    'complete_pass', 'pass_attempt', 'interception',
]


def _load_pbp() -> tuple[pd.DataFrame, dict]:
    """Load play-by-play data from data_files/, with an observation of the
    exact bytes read (provenance.read_source)."""
    # The PBP file is tab-separated (see create-play-by-play.py / update_pbp_smart.py).
    if PBP_FILE.exists():
        print(f"Loading PBP from {PBP_FILE} ...")
        pbp, source = read_source(PBP_FILE)
    elif PBP_FILE_UNCOMPRESSED.exists():
        print(f"Loading PBP from {PBP_FILE_UNCOMPRESSED} ...")
        pbp, source = read_source(PBP_FILE_UNCOMPRESSED)
    else:
        raise FileNotFoundError(
            f"Play-by-play file not found. Expected one of:\n"
            f"  {PBP_FILE}\n"
            f"  {PBP_FILE_UNCOMPRESSED}\n"
            "Run 'python create-play-by-play.py' first."
        )

    # Keep only columns present in the file to avoid key errors
    cols_to_use = [c for c in REQUIRED_PBP_COLS if c in pbp.columns]
    missing = set(REQUIRED_PBP_COLS) - set(cols_to_use)
    if missing:
        print(f"Warning: {len(missing)} expected PBP columns are absent: {missing}")

    return pbp, source


def _aggregate_and_roll(pbp: pd.DataFrame, source: dict | None = None) -> dict:
    """
    Aggregate PBP to game-level stats with rolling averages and matchup
    features, then validate and atomically save them to data_files/.

    The features are added BEFORE saving (models.py trains from the saved
    CSVs), using aggregators.build_player_stats; a validation failure raises
    and leaves the existing CSVs untouched. With ``source`` (from _load_pbp)
    the provenance manifest is written in the same set.

    Returns a dict with keys 'passing', 'rushing', 'receiving', each containing
    a fully-featured DataFrame ready for model training.
    """
    tables, reports = build_player_stats(pbp, load_matchup_schedule(DATA_DIR), ROLLING_WINDOWS)
    manifest_for = stats_manifest(source, ROLLING_WINDOWS) if source is not None else None
    written = write_player_stats(tables, DATA_DIR, ROLLING_WINDOWS, manifest_for)
    for stat_type in tables:
        print(f"Saved {len(tables[stat_type]):,} {stat_type} records -> {written[stat_type]} "
              f"({reports[stat_type]['excluded_plays']} plays excluded, "
              f"{reports[stat_type]['name_merges']} name merges)")
    if 'manifest' in written:
        print(f"Saved provenance manifest -> {written['manifest']}")
    else:
        print("Warning: no provenance manifest written; these stats are unverified")
    return tables


def run(skip_aggregation: bool = False) -> dict:
    """
    Full player prop model training pipeline.

    Args:
        skip_aggregation: If True, load pre-built CSVs from data_files/ instead
                          of reprocessing PBP data (faster for iterating on models).
    """
    print("=" * 70)
    print("NFL Player Prop Model Training Pipeline")
    print("=" * 70)

    if skip_aggregation:
        print("Skipping PBP aggregation - loading pre-built stat CSVs ...")
        # The train_all_models function (in models.py) handles loading from CSV
    else:
        pbp, source = _load_pbp()
        stats = _aggregate_and_roll(pbp, source)
        print(f"\nAggregation complete:")
        for stat_type, df in stats.items():
            print(f"  {stat_type}: {len(df):,} player-game records")

    # Call the master training function from models.py, which verifies the
    # saved CSVs in data_files/, trains every prop model and publishes one bundle.
    print("\nStarting model training ...")
    result = train_all_models()

    if result['published']:
        print(f"\nTraining pipeline complete. Bundle: {result['bundle']}")
    else:
        print("\nTraining pipeline finished without publishing a model bundle.")
    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train NFL player prop models')
    parser.add_argument(
        '--skip-aggregation',
        action='store_true',
        help='Skip PBP aggregation and use pre-built stat CSVs (faster)',
    )
    args = parser.parse_args()
    sys.exit(0 if run(skip_aggregation=args.skip_aggregation)['published'] else 1)
