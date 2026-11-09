"""player_props/aggregators.py console output under Windows redirected output.

With stdout redirected (a log file, a captured subprocess), Python on Windows
writes in the locale code page - cp1252 - and an emoji status line raises
UnicodeEncodeError. The Stage 5B rehearsal on native Windows Python 3.12
stopped in aggregate_training that way: build_player_stats printed a check
mark before any model was trained. Each status line in the module was an
emoji, so the first one is not the only failure.

Every case runs the aggregation in a child process whose stdout is a pipe
(so a file, not a console) and whose encoding is set only in that child's
environment: explicit cp1252, explicit UTF-8, and the platform default with no
override. The product code must work without any of them.

Synthetic fixtures only (rehearsal_fixtures.league), temporary directories,
no network: the child installs tests/no_network_plugin's socket guard.

The child interpreter is this one unless AGGREGATORS_ENCODING_TEST_PYTHON
names another (e.g. a Python 3.12 environment without pytest); only the child
runs under it.
"""
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
AGGREGATORS = ROOT / 'player_props' / 'aggregators.py'
CHILD_PYTHON = os.environ.get('AGGREGATORS_ENCODING_TEST_PYTHON') or sys.executable

# Not in cp1252 (Ł, ż, ł, ć), so printing it would fail there; written to the
# stats files it must survive byte-exactly as UTF-8.
NON_ASCII_NAME = 'Łukasz Żółć'
NON_ASCII_ID = 'KC-QB'

# Encoding set in the child's environment only. None: no override, so the
# child gets the platform default for a redirected stream (cp1252 on Windows
# with a Western locale, UTF-8 on Linux CI).
ENCODINGS = {'cp1252': 'cp1252', 'utf-8': 'utf-8', 'default': None}

CHILD = r'''
import gzip, hashlib, importlib.util, json, sys
from pathlib import Path

root, work, scenario = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
name, pid = sys.argv[4], sys.argv[5]
sys.path.insert(0, str(root))

spec = importlib.util.spec_from_file_location('no_network', root / 'tests' / 'no_network_plugin.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
guard.pytest_configure(None)
import socket
try:
    socket.getaddrinfo('example.com', 443)
except guard.NetworkBlocked:
    pass
else:
    raise SystemExit('socket guard self-test failed')

import pandas as pd
from player_props import aggregators as agg, rehearsal_fixtures as fixtures
from player_props.provenance import read_source, stats_manifest

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def inputs(drop_passers=False):
    plays_by_season, schedule = fixtures.league([2024, 2025], {})
    plays = pd.concat(plays_by_season.values(), ignore_index=True)
    for col in ('passer', 'rusher', 'receiver'):
        hit = plays[col + '_player_id'] == pid
        plays.loc[hit, col + '_player_name'] = name
    if drop_passers:
        plays['passer_player_id'] = None
        plays['passer_player_name'] = None
    src = work / 'source' / 'nfl_play_by_play_props.csv.gz'
    src.parent.mkdir(parents=True)
    src.write_bytes(gzip.compress(plays.to_csv(sep='\t', index=False).encode('utf-8'), mtime=0))
    out = work / 'stats'
    out.mkdir()
    schedule.to_csv(out / agg.SCHEDULE_FILE, sep='\t', index=False)
    pbp, observation = read_source(src)
    return pbp, observation, out

result = {'stdout_encoding': sys.stdout.encoding, 'stdout_errors': sys.stdout.errors,
          'utf8_mode': sys.flags.utf8_mode, 'python': sys.version.split()[0]}
if scenario == 'rehearsal_path':
    # player_props/rehearsal.py _aggregate, the stage that failed.
    pbp, observation, out = inputs()
    tables, reports = agg.build_player_stats(pbp, agg.load_matchup_schedule(out))
    written = agg.write_player_stats(tables, out, manifest_for=stats_manifest(observation))
elif scenario == 'aggregate_all_stats':
    pbp, observation, out = inputs()
    agg.aggregate_all_stats(pbp, save_to_dir=out, schedule=agg.load_matchup_schedule(out),
                            source=observation)
    reports = None
elif scenario == 'no_passing_plays':
    pbp, observation, out = inputs(drop_passers=True)
    agg.build_player_stats(pbp, agg.load_matchup_schedule(out))   # must raise
    raise SystemExit('build_player_stats accepted a source with no passing plays')
elif scenario in ('missing_rolling_column', 'missing_rolling_column_unicode'):
    df = pd.DataFrame({'player_id': ['a', 'a'], 'game_id': ['g1', 'g2'],
                       'game_date': ['2025-09-07', '2025-09-14'], 'passing_yards': [200.0, 250.0]})
    missing_column = 'not_a_stat' if scenario == 'missing_rolling_column' else 'missing_\u0141_stat'
    rolled = agg.calculate_rolling_averages(df, ['passing_yards', missing_column], [3])
    result['requested_missing_column'] = missing_column
    result['rolled'] = rolled.to_dict('list')
    out = None
    reports = None
else:
    raise SystemExit('unknown scenario ' + scenario)

if out is not None:
    result['files'] = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    manifest = json.loads((out / agg.MANIFEST_FILE).read_text(encoding='utf-8'))
    for volatile in ('written_at',):
        manifest.pop(volatile)
    manifest['source'].pop('observed_at')
    result['manifest'] = manifest
    passing = (out / agg.STATS_FILES['passing']).read_bytes().decode('utf-8')   # strict
    result['non_ascii_name_in_passing'] = name in passing
    result['reports'] = reports
(work / 'result.json').write_text(json.dumps(result, indent=1, sort_keys=True), encoding='utf-8')
'''


def child_env(encoding):
    env = {k: v for k, v in os.environ.items()
           if k not in ('PYTHONUTF8', 'PYTHONIOENCODING', 'PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP')}
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    if encoding is not None:
        env['PYTHONIOENCODING'] = encoding
    return env


def run_child(tmp_path, scenario, encoding):
    script = tmp_path / 'child.py'
    script.write_text(CHILD, encoding='utf-8')
    work = tmp_path / 'work'
    work.mkdir()
    proc = subprocess.run([CHILD_PYTHON, '-B', str(script), str(ROOT), str(work), scenario,
                           NON_ASCII_NAME, NON_ASCII_ID],
                          cwd=tmp_path, env=child_env(ENCODINGS[encoding]),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    stdout = proc.stdout.decode('utf-8', 'backslashreplace')
    stderr = proc.stderr.decode('utf-8', 'backslashreplace')
    result_file = work / 'result.json'
    result = json.loads(result_file.read_text(encoding='utf-8')) if result_file.exists() else None
    return proc.returncode, proc.stdout, stdout, stderr, result


def assert_clean(code, raw, stderr, result, encoding):
    assert 'UnicodeEncodeError' not in stderr, stderr
    assert code == 0, f"exit {code}\n{stderr}"
    assert 'Traceback' not in stderr, stderr
    assert result is not None
    raw.decode('ascii')                 # every status line is plain ASCII
    if ENCODINGS[encoding] is not None:
        assert result['stdout_encoding'].replace('_', '-').lower() == ENCODINGS[encoding]


@pytest.mark.parametrize('encoding', list(ENCODINGS))
def test_rehearsal_aggregation_path_logs_under_each_encoding(tmp_path, encoding):
    code, raw, stdout, stderr, result = run_child(tmp_path, 'rehearsal_path', encoding)
    assert_clean(code, raw, stderr, result, encoding)
    lines = stdout.splitlines()
    for stat_type in ('passing', 'rushing', 'receiving'):
        assert any(line.startswith('OK: Aggregated ') and line.endswith(f' player-game {stat_type} records')
                   for line in lines), stdout
    assert stdout.count('OK: Calculated exponentially-weighted rolling averages') == 3
    assert set(result['files']) == {'player_passing_stats.csv', 'player_rushing_stats.csv',
                                    'player_receiving_stats.csv', 'player_props_stats_manifest.json',
                                    'nfl_games_historical.csv'}
    assert result['non_ascii_name_in_passing'] is True


def test_rehearsal_aggregation_outputs_identical_across_encodings(tmp_path):
    runs = {}
    for encoding in ENCODINGS:
        sub = tmp_path / encoding
        sub.mkdir()
        code, raw, stdout, stderr, result = run_child(sub, 'rehearsal_path', encoding)
        assert_clean(code, raw, stderr, result, encoding)
        runs[encoding] = (stdout, result)
    # The manifest's bytes carry written_at / observed_at, so it is compared
    # parsed, with exactly those two removed (see CHILD); every other file byte-exactly.
    def data_files(result):
        return {k: v for k, v in result['files'].items() if k != 'player_props_stats_manifest.json'}

    (base_out, base), *others = runs.values()
    assert len(data_files(base)) == 4
    for stdout, result in others:
        assert stdout == base_out
        assert data_files(result) == data_files(base)
        assert result['manifest'] == base['manifest']
        assert result['reports'] == base['reports']


@pytest.mark.parametrize('encoding', list(ENCODINGS))
def test_aggregate_all_stats_logs_under_each_encoding(tmp_path, encoding):
    code, raw, stdout, stderr, result = run_child(tmp_path, 'aggregate_all_stats', encoding)
    assert_clean(code, raw, stderr, result, encoding)
    assert 'Aggregating Player Stats...' in stdout
    assert 'Saved player_props_stats_manifest.json' in stdout
    assert 'OK: All stats aggregated successfully!' in stdout
    assert result['non_ascii_name_in_passing'] is True


@pytest.mark.parametrize('encoding', list(ENCODINGS))
def test_no_passing_plays_reports_the_validation_error(tmp_path, encoding):
    """The warning must not replace the real error: the run still stops with
    FeatureValidationError and a non-zero exit, not UnicodeEncodeError."""
    code, raw, stdout, stderr, result = run_child(tmp_path, 'no_passing_plays', encoding)
    assert code == 1, f"exit {code}\n{stderr}"
    assert 'UnicodeEncodeError' not in stderr, stderr
    assert 'FeatureValidationError' in stderr and 'passing: no plays to aggregate' in stderr, stderr
    assert 'WARNING: No passing plays found in dataset' in stdout
    assert result is None
    raw.decode('ascii')


@pytest.mark.parametrize('encoding', list(ENCODINGS))
def test_missing_rolling_column_warns_and_continues(tmp_path, encoding):
    code, raw, stdout, stderr, result = run_child(tmp_path, 'missing_rolling_column', encoding)
    assert_clean(code, raw, stderr, result, encoding)
    assert "WARNING: Column 'not_a_stat' not found, skipping" in stdout
    assert result['rolled']['passing_yards_L3'][1] == 200.0
    assert 'not_a_stat_L3' not in result['rolled']


@pytest.mark.parametrize('encoding', list(ENCODINGS))
def test_unencodable_missing_column_warns_without_changing_data(tmp_path, encoding):
    """Exercise interpolated diagnostic text, not just Unicode CSV contents."""
    code, raw, stdout, stderr, result = run_child(tmp_path, 'missing_rolling_column_unicode', encoding)
    assert_clean(code, raw, stderr, result, encoding)
    assert "WARNING: Column 'missing_\\u0141_stat' not found, skipping" in stdout
    assert result['requested_missing_column'] == 'missing_Ł_stat'
    assert result['rolled']['passing_yards'] == [200.0, 250.0]
    assert result['rolled']['passing_yards_L3'][1] == 200.0
    assert 'missing_Ł_stat_L3' not in result['rolled']


@pytest.mark.parametrize('encoding', list(ENCODINGS))
def test_cli_missing_source_reports_and_exits_1(tmp_path, encoding):
    """python player_props/aggregators.py from a directory without the
    play-by-play: exit 1 with the reason, not an encoding traceback. (Its
    success path reads the repository's schedule, so it is not run here.)"""
    proc = subprocess.run([CHILD_PYTHON, '-B', str(AGGREGATORS)], cwd=tmp_path,
                          env=child_env(ENCODINGS[encoding]),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    stdout = proc.stdout.decode('utf-8', 'backslashreplace')
    stderr = proc.stderr.decode('utf-8', 'backslashreplace')
    assert proc.returncode == 1, f"exit {proc.returncode}\n{stderr}"
    assert 'Traceback' not in stderr, stderr
    assert 'NFL Player Stats Aggregator' in stdout
    assert 'ERROR: File not found: data_files' in stdout
    assert list(tmp_path.iterdir()) == []          # nothing written
    proc.stdout.decode('ascii')


def test_every_aggregator_print_literal_is_ascii():
    """Static guard for the lines the subprocess cases cannot reach (the CLI
    success path): no print() in the module carries a non-ASCII literal."""
    source = AGGREGATORS.read_text(encoding='utf-8')
    offenders = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'print':
            text = ast.get_source_segment(source, node)
            if not text.isascii():
                offenders.append(f"line {node.lineno}: {text}")
    assert not offenders, "\n".join(offenders)
