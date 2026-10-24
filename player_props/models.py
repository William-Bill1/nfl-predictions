"""
Player Props XGBoost Models
Train baseline models for predicting player prop over/under outcomes.

One XGBoost classifier per prop line, scored with model_bundle.score_probabilities
in evaluation and prediction alike, and published as one verified bundle
(model_bundle.publish_bundle). See docs/architecture.md ("Stage 4 model provenance").
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)
import xgboost as xgb
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from player_props.aggregators import read_stats_csv
from player_props.features import (
    FEATURE_CONTRACT, PROP_LINES, PROP_STAT, FeatureContractError, model_features, model_matrix,
    training_rows,
)
from player_props.model_bundle import SCORER, publish_bundle, score_probabilities, window

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path('data_files')
MODELS_DIR = Path('player_props/models')

# Training configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_TEST_ROWS = 40    # need at least this many held-out rows to report a metric

# A metric is only marked "reliable" if the model clears these out-of-time bars
# on the held-out season. TD props never clear them (see note below) so they are
# force-flagged unreliable regardless.
RELIABLE_MIN_AUC = 0.58
RELIABLE_MIN_ACC_EDGE = 0.01   # accuracy must beat the majority-class base rate by this
TD_NOTE = ("TD props are ~coin-flip out-of-time (all tiers collapse to the same "
           "0.5 line); kept for display only, not a betting signal.")


def temporal_holdout(df):
    """Return (train_df, test_df, holdout_season).

    Hold out the most recent season present so reported metrics are out-of-time
    instead of the optimistic random split. Falls back to a date-ordered last-20%
    cut if only one season is available.
    """
    if 'season' in df.columns and df['season'].nunique() >= 2:
        holdout = int(df['season'].max())
        return df[df['season'] < holdout], df[df['season'] >= holdout], holdout
    order = [c for c in ('season', 'week') if c in df.columns]
    if order:
        df = df.sort_values(order)
    cut = int(len(df) * (1 - TEST_SIZE))
    return df.iloc[:cut], df.iloc[cut:], None

# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_player_stats(stat_type='passing'):
    """Load aggregated player stats."""
    file_map = {
        'passing': 'player_passing_stats.csv',
        'rushing': 'player_rushing_stats.csv',
        'receiving': 'player_receiving_stats.csv'
    }
    
    file_path = DATA_DIR / file_map[stat_type]
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        print(f"   Run 'python player_props/aggregators.py' first")
        return None
    
    df = read_stats_csv(file_path)
    print(f"✅ Loaded {len(df):,} {stat_type} records")
    return df


def training_inputs(data_dir=None):
    """(inputs, problems). inputs = {'tables', 'source'} only when the stats
    files verify against their Stage 2 provenance manifest and use the
    current feature definition; otherwise (None, reasons).

    The tables are the exact bytes that were verified, so a bundle's
    training_source hashes describe the data its models were trained on. The
    training source is recorded on its own; it need not be the latest feature
    source used at prediction time."""
    from player_props.provenance import MANIFEST_FILE, _read_stats, _sha256, check_provenance
    data_dir = Path(data_dir or DATA_DIR)
    raw, tables, problems = _read_stats(data_dir)
    more, _, manifest = check_provenance(data_dir, raw, tables)
    problems += more
    if problems or manifest is None:
        return None, problems or ["no valid provenance manifest"]
    if manifest['aggregation']['feature_definition'] != FEATURE_CONTRACT['feature_definition']:
        return None, ["the stats use another feature definition"]
    stats = manifest['stats']
    source = {
        'stats_manifest_sha256': _sha256((data_dir / MANIFEST_FILE).read_bytes()),
        'stats_written_at': manifest['written_at'],
        'feature_definition': manifest['aggregation']['feature_definition'],
        'stats_files': {e['file']: e['sha256'] for e in stats.values()},
        'last_game_date': max(e['coverage']['last_date'] for e in stats.values()),
        'play_by_play': {k: manifest['source'][k] for k in ('file', 'sha256', 'observed_at')},
    }
    return {'tables': tables, 'source': source}, []


def training_feature_contract(data_dir=None):
    """The feature contract the stats being trained on meet, or None (Stage 3
    helper; training_inputs is what training uses)."""
    inputs, problems = training_inputs(data_dir)
    if inputs is None:
        print("⚠️  No feature contract: " + "; ".join(problems))
        return None
    return {**FEATURE_CONTRACT, 'stats_manifest_sha256': inputs['source']['stats_manifest_sha256']}


def create_prop_targets(df, stat_col, line_values):
    """
    Create binary targets for different prop lines.
    
    Args:
        df: DataFrame with player stats
        stat_col: Column name for the stat (e.g., 'passing_yards')
        line_values: Dict of prop lines to create targets for
        
    Returns:
        DataFrame with added target columns
    """
    for line_name, line_value in line_values.items():
        target_col = f'over_{line_name}'
        df[target_col] = (df[stat_col] > line_value).astype(int)
    
    return df


def prepare_training_features(df, stat_type='passing'):
    """
    Prepare features for model training.
    
    Features include:
    - Rolling averages (L3, L5, L10)
    - Rolling std dev for consistency
    - Team strength indicators
    - Home/away
    - Recent trends
    - Matchup features (opponent defense rank, home/away, days rest)
    """
    # Core stat column mapping
    stat_col_map = {
        'passing': 'passing_yards',
        'rushing': 'rushing_yards',
        'receiving': 'receiving_yards'
    }
    
    return _prepare(df, f'{stat_col_map[stat_type]}')


def _prepare(df, prop_type):
    """Rows and exact feature list for one prop type (features.MODEL_FEATURES).
    Rows without an earlier appearance are excluded and counted; any other
    missing or bad value raises (features.training_rows)."""
    rows, report = training_rows(df, prop_type)
    feature_cols = model_features(prop_type)
    print(f"📊 {prop_type}: {len(feature_cols)} features {feature_cols}; "
          f"{report['used']:,} rows, {report['excluded_no_history']:,} first appearances excluded")
    return rows.copy(), feature_cols


def prepare_td_training_features(df, stat_type='passing'):
    """
    Prepare features for TD model training.
    
    TD models use different features than yards models:
    - Focus on TD-related stats and volume metrics
    - Exclude yards (to avoid data leakage for TD predictions)
    - Include matchup features for better predictions
    """
    return _prepare(df, {'passing': 'passing_tds', 'rushing': 'rushing_tds',
                         'receiving': 'receiving_tds'}[stat_type])


def prepare_receptions_training_features(df):
    """
    Prepare features specifically for receptions model training.
    
    Receptions models use reception-focused features:
    - Rolling receptions averages (L3, L5, L10)
    - Rolling targets and TD features
    - Matchup features (opponent defense rank, home/away, days rest)
    """
    return _prepare(df, 'receptions')


def train_prop_model(df, features, prop_type, line_type, line_value):
    """
    Train one XGBoost classifier for "stat > line" and evaluate it out of time.

    The scorer evaluated is the one prediction serves
    (model_bundle.score_probabilities on this exact model): there is no
    ensemble. Rows must come from features.training_rows.

    Returns (bundle entry, None) or (None, reason it was skipped). The entry
    holds the serialised model ('data') and everything the bundle manifest
    records for it; nothing is written here (see train_all_models).
    """
    name = f'{prop_type}_{line_type}'
    if list(features) != model_features(prop_type):
        raise FeatureContractError(f"{name}: features {features} != {model_features(prop_type)}")
    stat_col = PROP_STAT[prop_type][1]
    X_all = model_matrix(df, prop_type)
    y_all = (df[stat_col] > line_value).astype(int)
    meta = df[['season', 'week', 'game_date']]
    if len(X_all) < 100:
        return None, f"only {len(X_all)} rows"

    class_dist = y_all.value_counts()
    print(f"\n{name}: over {class_dist.get(1, 0):,} / under {class_dist.get(0, 0):,}")
    if min(class_dist.get(0, 0), class_dist.get(1, 0)) / len(y_all) < 0.1:
        return None, "too imbalanced (under 10% of one class)"

    # Temporal hold-out: train on earlier seasons, evaluate on the latest one.
    frame = pd.concat([X_all, y_all.rename('_y'), meta], axis=1)
    train_df, test_df, holdout_season = temporal_holdout(frame)
    if holdout_season is None:
        return None, "needs at least two seasons for an out-of-time evaluation"
    X_train, y_train = train_df[features], train_df['_y']
    X_test, y_test = test_df[features], test_df['_y']
    if len(X_test) < MIN_TEST_ROWS or y_test.nunique() < 2 or y_train.nunique() < 2:
        return None, f"held-out season too small or one-class ({len(X_test)} rows)"

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Serialise, reload and score the reloaded model: the metrics describe the
    # exact bytes the bundle stores and prediction loads.
    data = bytes(model.get_booster().save_raw('json'))
    served = xgb.XGBClassifier()
    served.load_model(bytearray(data))
    proba = score_probabilities(served, X_test, prop_type)
    y_pred = (proba >= 0.5).astype(int)

    accuracy = float(accuracy_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, proba))
    base_rate = float(max(y_test.mean(), 1 - y_test.mean()))
    is_td = '_tds_' in name
    reliable = bool(not is_td and roc_auc >= RELIABLE_MIN_AUC
                    and accuracy >= base_rate + RELIABLE_MIN_ACC_EDGE)
    metrics = {
        'scorer': SCORER['name'],
        'base_rate': base_rate, 'accuracy': accuracy, 'roc_auc': roc_auc,
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        # Depend on every probability, not just ranks or the 0.5 threshold.
        'brier': float(brier_score_loss(y_test, proba)),
        'log_loss': float(log_loss(y_test, proba, labels=[0, 1])),
        'reliable': reliable,
        'note': TD_NOTE if is_td else ('' if reliable else 'No out-of-time edge over the base rate.'),
    }
    print(f"   holdout {holdout_season}: accuracy {accuracy:.3f} (base {base_rate:.3f}), "
          f"AUC {roc_auc:.3f}, reliable {reliable}")
    entry = {
        'data': data, 'prop_type': prop_type, 'line_type': line_type, 'line_value': line_value,
        'target': {'stat': stat_col, 'rule': 'stat > line'},
        'features': list(features), 'scorer': dict(SCORER),
        'training': window(train_df, y_train),
        'evaluation': {**window(test_df, y_test),
                       'protocol': (f"temporal hold-out: trained on seasons before {holdout_season}, "
                                    f"evaluated on season {holdout_season}; raw model probability "
                                    f"(no tier, clamp, injury or weather adjustment); "
                                    f"OVER when P >= 0.5")},
        'metrics': metrics,
    }
    return entry, None


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_all_models(data_dir=None, models_dir=None):
    """Train every prop model and publish them as one verified bundle.

    Training uses only stats whose Stage 2 manifest verifies
    (training_inputs); otherwise nothing is trained or written. Models are
    held in memory and published together by model_bundle.publish_bundle,
    which switches the current bundle only after the new one verifies from
    disk. Nothing is written to the flat player_props/models/*.json files.

    Returns {'published': bool, 'bundle': path or None, 'models': [...],
    'skipped': {name: reason}, 'problems': [...]}.
    """
    print("=" * 70)
    print("🏈 NFL Player Props Model Training (XGBoost, one bundle)")
    print("=" * 70)
    models_dir = Path(models_dir or MODELS_DIR)
    inputs, problems = training_inputs(data_dir)
    if inputs is None:
        print("⛔ Not training: the stats can't be verified - " + "; ".join(problems))
        return {'published': False, 'bundle': None, 'models': [], 'skipped': {}, 'problems': problems}

    entries, skipped = {}, {}
    for prop_type, lines in PROP_LINES.items():
        stat_type = PROP_STAT[prop_type][0]
        rows, features = _prepare(inputs['tables'][stat_type], prop_type)
        for line_type, line_value in lines.items():
            entry, reason = train_prop_model(rows, features, prop_type, line_type, line_value)
            name = f'{prop_type}_{line_type}'
            if entry is None:
                skipped[name] = reason
                print(f"   skipped {name}: {reason}")
            else:
                entries[name] = entry
    if not entries:
        problems = ["no model could be trained"]
        print("⛔ " + problems[0])
        return {'published': False, 'bundle': None, 'models': [], 'skipped': skipped, 'problems': problems}

    path = publish_bundle(models_dir, entries, inputs['source'], PROP_LINES)
    reliable = sum(e['metrics']['reliable'] for e in entries.values())
    print(f"\n✅ Published bundle {path.name}: {len(entries)} models "
          f"({reliable} clear the out-of-time bar), {len(skipped)} skipped")
    return {'published': True, 'bundle': path, 'models': sorted(entries), 'skipped': skipped,
            'problems': []}


if __name__ == '__main__':
    raise SystemExit(0 if train_all_models()['published'] else 1)
