"""Managed/legacy page input regression tests. Synthetic temp workspaces only."""
from pathlib import Path
import ast
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from player_props import page_data as ui
from player_props import publication as pub, release_manifest as rm
from player_props import bundle_store as bs, model_bundle as mb, rehearsal as rh

ROOT = Path(__file__).resolve().parents[1]
PAGES = ['2_Player_Props.py', '3_Parlay_Builder.py']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree(path):
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob('*') if p.is_file()}


@pytest.fixture(scope='module')
def managed_seed(tmp_path_factory):
    base = tmp_path_factory.mktemp('managed-ui')
    result = rh.run_rehearsal(rh.RehearsalConfig(workspace=base / 'rehearsal', seed=7))
    assert result['status'] == 'succeeded'
    run = base / 'rehearsal' / 'runs' / result['run_id']
    package = base / 'package'
    report = rm.build_package(rm.inputs_from_run(run, run / 'runtime/data/players.csv'),
                              package, 'byte_copy', input_dirs=[run])
    assert report['ok']
    seed = base / 'app'
    data = seed / 'data_files'
    pub.publish(package, data, run / 'runtime/data/nfl_schedule_2026.csv', packaged=True)
    for n in ['players.csv', *ui.STATS.values(), 'nfl_schedule_2026.csv']:
        shutil.copyfile(run / 'runtime/data' / n, data / n)
    bs.import_bundle(seed / 'player_props/models', run / 'export/bundle.tar')
    manifest = json.loads((package / rm.MANIFEST_NAME).read_bytes())
    config = {'version': 1, 'package': str(package), 'release_id': manifest['release_id'],
              'manifest_sha256': sha(package / rm.MANIFEST_NAME),
              'publication_sha256': sha(data / pub.sidecar_name(2026, 3)), 'season': 2026, 'week': 3}
    (data / ui.CONFIG_FILE).write_text(json.dumps(config))
    return seed, package


@pytest.fixture
def managed(managed_seed, tmp_path):
    seed, package = managed_seed
    app = tmp_path / 'app'
    shutil.copytree(seed, app)
    return app, package


def test_managed_exact_values_and_no_writes(managed):
    app, package = managed
    before = tree(app)
    loaded = ui.load_page_inputs(app / 'data_files')
    expected = pd.read_csv(app / 'data_files' / ui.LATEST_FILE, dtype={'player_id': str},
                           float_precision='round_trip')
    pd.testing.assert_frame_equal(loaded.predictions, expected)
    assert loaded.managed and 'not release approval' in loaded.notice
    assert set(loaded.payloads) >= {'players.csv', *ui.STATS.values()}
    assert before == tree(app)


@pytest.mark.parametrize('filename', [ui.LATEST_FILE, pub.snapshot_name(2026, 3), pub.sidecar_name(2026, 3),
                                       'players.csv', *ui.STATS.values(), 'nfl_schedule_2026.csv'])
def test_missing_managed_input_never_falls_back(managed, filename):
    app, _ = managed
    (app / 'data_files' / filename).unlink()
    before = tree(app)
    with pytest.raises(ui.PageDataError):
        ui.load_page_inputs(app / 'data_files')
    assert tree(app) == before


@pytest.mark.parametrize('filename', [ui.LATEST_FILE, pub.snapshot_name(2026, 3), pub.sidecar_name(2026, 3),
                                       'players.csv', *ui.STATS.values(), 'nfl_schedule_2026.csv'])
def test_changed_managed_input_after_a_success_is_detected(managed, filename):
    app, _ = managed
    ui.load_page_inputs(app / 'data_files')
    (app / 'data_files' / filename).write_bytes(b'corrupt\n')
    with pytest.raises(ui.PageDataError):
        ui.load_page_inputs(app / 'data_files')


@pytest.mark.parametrize('change', ['missing', 'malformed', 'wrong_bundle', 'wrong_hash'])
def test_bad_pointer_is_refused(managed, change):
    app, _ = managed
    p = app / 'player_props/models' / mb.POINTER_FILE
    d = json.loads(p.read_bytes())
    if change == 'missing':
        p.unlink()
    else:
        if change == 'malformed':
            d = []
        elif change == 'wrong_bundle':
            d['bundle_id'] = '20990101T000000Z-00000000'
        else:
            d['manifest_sha256'] = '0' * 64
        p.write_text(json.dumps(d))
    with pytest.raises(ui.PageDataError):
        ui.load_page_inputs(app / 'data_files')


@pytest.mark.parametrize('change', ['corrupt_model', 'missing_model', 'extra_model', 'contract'])
def test_installed_bundle_must_be_exact_package_bundle(managed, change):
    app, _ = managed
    models = app / 'player_props/models'
    pointer = json.loads((models / mb.POINTER_FILE).read_bytes())
    active = models / mb.BUNDLES_DIR / pointer['bundle_id']
    manifest = json.loads((active / mb.MANIFEST_FILE).read_bytes())
    model = active / next(iter(manifest['models'].values()))['file']
    if change == 'corrupt_model':
        model.write_bytes(b'broken')
    elif change == 'missing_model':
        model.unlink()
    elif change == 'extra_model':
        (active / 'extra.json').write_text('{}')
    else:
        manifest['feature_contract'] = {'feature_definition': 'wrong'}
        (active / mb.MANIFEST_FILE).write_text(json.dumps(manifest))
        pointer['manifest_sha256'] = sha(active / mb.MANIFEST_FILE)
        (models / mb.POINTER_FILE).write_text(json.dumps(pointer))
    with pytest.raises(ui.PageDataError):
        ui.load_page_inputs(app / 'data_files')


@pytest.mark.parametrize('change', ['bad_json', 'list', 'unknown_field', 'boolean_week', 'wrong_release',
                                    'wrong_manifest', 'missing_package', 'wrong_target'])
def test_invalid_configuration_never_downgrades(managed, change):
    app, _ = managed
    path = app / 'data_files' / ui.CONFIG_FILE
    d = json.loads(path.read_bytes())
    if change == 'bad_json':
        path.write_text('{')
    else:
        if change == 'list': d = []
        elif change == 'unknown_field': d['extra'] = 1
        elif change == 'boolean_week': d['week'] = True
        elif change == 'wrong_release': d['release_id'] = '20990101T000000Z-00000000'
        elif change == 'wrong_manifest': d['manifest_sha256'] = '0' * 64
        elif change == 'missing_package': d['package'] = str(app / 'absent')
        elif change == 'wrong_target': d['week'] = 8
        path.write_text(json.dumps(d))
    with pytest.raises(ui.PageDataError):
        ui.load_page_inputs(app / 'data_files')


def test_inaccessible_config_does_not_downgrade(managed, monkeypatch):
    app, _ = managed
    original = ui._read
    def blocked(path):
        if path.name == ui.CONFIG_FILE: raise PermissionError('test denial')
        return original(path)
    monkeypatch.setattr(ui, '_read', blocked)
    with pytest.raises(ui.PageDataError, match='PermissionError'):
        ui.load_page_inputs(app / 'data_files')


def test_mutated_package_asset_is_not_hidden_by_cache(managed):
    app, package = managed
    private = app.parent / 'changed_package'
    shutil.copytree(package, private)
    config = app / 'data_files' / ui.CONFIG_FILE
    d = json.loads(config.read_bytes()); d['package'] = str(private); config.write_text(json.dumps(d))
    ui.load_page_inputs(app / 'data_files')
    (private / rm.ROLES['reference_runtime']).write_bytes(b'changed')
    with pytest.raises(ui.PageDataError):
        ui.load_page_inputs(app / 'data_files')


def test_config_can_be_deliberately_removed_to_restore_legacy_mode(managed):
    app, _ = managed
    ui.load_page_inputs(app / 'data_files')
    (app / 'data_files' / ui.CONFIG_FILE).unlink()
    # Legacy does not claim the remaining sidecar/bundle is verified.
    loaded = ui.load_page_inputs(app / 'data_files')
    assert not loaded.managed and 'not verified' in loaded.notice


def test_sidecar_bundle_identity_cannot_be_substituted(managed):
    app, _ = managed
    p = app / 'data_files' / pub.sidecar_name(2026, 3)
    d = json.loads(p.read_bytes()); d['bundle']['bundle_id'] = '20990101T000000Z-00000000';p.write_text(json.dumps(d))
    cfg = app / 'data_files' / ui.CONFIG_FILE
    c = json.loads(cfg.read_bytes()); c['publication_sha256'] = sha(p); cfg.write_text(json.dumps(c))
    with pytest.raises(ui.PageDataError, match='bundle'):
        ui.load_page_inputs(app / 'data_files')


@pytest.mark.parametrize('change', ['rows', 'expected_rows', 'completeness', 'reference', 'inputs', 'eligibility'])
def test_sidecar_claims_must_match_package(managed, change):
    app, _ = managed
    p = app / 'data_files' / pub.sidecar_name(2026, 3)
    d = json.loads(p.read_bytes())
    if change in ('rows', 'expected_rows'): d['verification'][change] = 1
    elif change == 'completeness': d['verification'][change] = 'not checked'
    else: d[change] = {}
    p.write_text(json.dumps(d))
    cfg = app / 'data_files' / ui.CONFIG_FILE
    c = json.loads(cfg.read_bytes()); c['publication_sha256'] = sha(p); cfg.write_text(json.dumps(c))
    with pytest.raises(ui.PageDataError):
        ui.load_page_inputs(app / 'data_files')


def test_captured_reference_and_stats_do_not_reread_changed_paths(managed, monkeypatch):
    app, _ = managed
    loaded = ui.load_page_inputs(app / 'data_files')
    old = loaded.payloads['players.csv']
    (app / 'data_files/players.csv').write_bytes(b'changed after capture')
    assert loaded.payloads['players.csv'] == old
    assert loaded.reference.usable
    with pytest.raises(ui.PageDataError): ui.load_page_inputs(app / 'data_files')


@pytest.mark.parametrize('change', ['empty', 'malformed', 'missing_columns', 'bad_number', 'bad_average', 'range'])
def test_legacy_csv_errors_have_readable_diagnostics(tmp_path, change):
    from test_player_props_reference_join import write_pages_fixture
    write_pages_fixture(tmp_path)
    path = tmp_path / 'data_files' / ui.LATEST_FILE
    if change == 'empty': path.write_bytes(b'')
    elif change == 'malformed': path.write_text('player_name,confidence\n"unterminated')
    elif change == 'missing_columns': path.write_text('player_name\nExample\n')
    else:
        d = pd.read_csv(path)
        if change == 'bad_number': d.loc[0, 'confidence'] = np.inf
        elif change == 'bad_average': d['avg_L3'] = 'garbage'
        else: d.loc[0, 'prob_over'] = 1.1
        d.to_csv(path, index=False)
    before = path.read_bytes()
    with pytest.raises(ui.PageDataError, match='CSV'):
        ui.load_page_inputs(tmp_path / 'data_files')
    assert path.read_bytes() == before


@pytest.mark.parametrize('schema', ['current', 'legacy', 'no_details'])
def test_legacy_namesake_behavior_preserved(tmp_path, schema):
    from test_player_props_reference_join import write_pages_fixture
    expected = write_pages_fixture(tmp_path, schema)
    loaded = ui.load_page_inputs(tmp_path / 'data_files')
    assert not loaded.managed
    assert loaded.predictions['player_name'].tolist() == expected['player_name'].tolist()
    np.testing.assert_array_equal(loaded.predictions['prob_over'], expected['prob_over'])


def render(page, cwd, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    monkeypatch.chdir(cwd)
    st.cache_data.clear()
    return AppTest.from_file(str(ROOT / 'pages' / page), default_timeout=90).run()


@pytest.mark.parametrize('page', PAGES)
def test_managed_pages_render_and_detect_changes_on_rerun(managed, monkeypatch, page):
    app, _ = managed
    at = render(page, app, monkeypatch)
    assert not at.exception and not at.error
    assert 'Do not use' in at.warning[0].value
    assert any('Managed release' in c.value for c in at.caption)
    expected = pd.read_csv(app / 'data_files' / ui.LATEST_FILE)
    assert len(at.dataframe) > 0 or page.startswith('3')
    if page.startswith('3'):
        assert at.multiselect[0].options
        at.multiselect[0].set_value(at.multiselect[0].options[:2]).run()
        assert not at.exception
    else:
        assert at.selectbox(key='player_search').options
        at.selectbox(key='player_search').set_value(at.selectbox(key='player_search').options[-1]).run()
        assert not at.exception
    p = app / 'data_files' / ui.LATEST_FILE
    p.write_bytes(p.read_bytes() + b'\n')
    at.run()  # Do NOT clear Streamlit caches: the changed bytes must be noticed.
    assert not at.exception and any('unavailable' in e.value for e in at.error)
    assert not at.dataframe and not at.multiselect
    # Restore the previous exact set; cached failure must not prevent recovery.
    # The snapshot is the original exact feed for this release.
    p.write_bytes((app / 'data_files' / pub.snapshot_name(2026, 3)).read_bytes())
    at.run()
    assert not at.exception and not at.error


@pytest.mark.parametrize('page', PAGES)
@pytest.mark.parametrize('change', ['missing', 'corrupt_model', 'reference', 'empty_csv', 'configuration'])
def test_pages_show_controlled_managed_failure(managed, monkeypatch, page, change):
    app, _ = managed
    if change == 'missing': (app / 'player_props/models' / mb.POINTER_FILE).unlink()
    elif change == 'corrupt_model':
        ptr = json.loads((app / 'player_props/models' / mb.POINTER_FILE).read_bytes())
        d = app / 'player_props/models' / mb.BUNDLES_DIR / ptr['bundle_id']
        m = json.loads((d / mb.MANIFEST_FILE).read_bytes()); (d / next(iter(m['models'].values()))['file']).write_bytes(b'broken')
    elif change == 'reference': (app / 'data_files/players.csv').write_bytes(b'')
    elif change == 'empty_csv': (app / 'data_files' / ui.LATEST_FILE).write_bytes(b'')
    else: (app / 'data_files' / ui.CONFIG_FILE).write_bytes(b'{')
    before = tree(app)
    at = render(page, app, monkeypatch)
    assert not at.exception
    assert 'Do not use' in at.warning[0].value
    assert any('unavailable' in e.value for e in at.error)
    assert not at.dataframe and not at.multiselect
    assert tree(app) == before


@pytest.mark.parametrize('page', PAGES)
def test_corrupt_legacy_csv_is_visible_without_exception(tmp_path, monkeypatch, page):
    from test_player_props_reference_join import write_pages_fixture
    write_pages_fixture(tmp_path)
    (tmp_path / 'data_files' / ui.LATEST_FILE).write_bytes(b'')
    at = render(page, tmp_path, monkeypatch)
    assert not at.exception and at.error
    assert 'Do not use' in at.warning[0].value
    assert not at.dataframe


def test_output_and_sidecar_resealing_cannot_bypass_fixed_config(managed):
    app, _ = managed
    data = app / 'data_files'
    latest = data / ui.LATEST_FILE
    frame = pd.read_csv(latest, dtype={'player_id': str}, float_precision='round_trip')
    frame.loc[0, 'prob_over'] = 0.123
    raw = frame.to_csv(index=False).encode()
    latest.write_bytes(raw)
    (data / pub.snapshot_name(2026, 3)).write_bytes(raw)
    side = data / pub.sidecar_name(2026, 3)
    doc = json.loads(side.read_bytes())
    for role in ('latest', 'snapshot'):
        doc['outputs'][role].update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    side.write_text(json.dumps(doc))
    with pytest.raises(ui.PageDataError, match='configured publication hash'):
        ui.load_page_inputs(data)


def test_s1_cache_result_must_name_the_bytes_that_were_captured(monkeypatch):
    monkeypatch.setattr(rm, 'verify_package', lambda p: {'ok': True, 'manifest_sha256': 'f' * 64})
    with pytest.raises(ui.PageDataError, match='different package bytes'):
        ui._verified_package.__wrapped__('unused', (('0' * 64,), ()), 'policy')
