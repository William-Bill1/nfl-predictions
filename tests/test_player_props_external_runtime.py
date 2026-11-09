"""External managed runtime and captured banners; isolated synthetic fixtures only."""
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
import json
import os
import shutil
import subprocess

import pandas as pd
import pytest

from player_props import page_data as ui, containment as c, season_schedule as ss
from player_props import model_bundle as mb, publication as pub
from player_props.game_dates import TORONTO
from test_player_props_page_data import managed_seed, managed, render, tree


@pytest.fixture
def external(managed_seed, tmp_path):
    seed, package = managed_seed
    app = tmp_path / 'application'
    runtime = tmp_path / 'runtime'
    shutil.copytree(seed, app)
    shutil.copytree(seed, runtime)
    (runtime / 'data_files' / ui.CONFIG_FILE).unlink()
    config = app / 'data_files' / ui.CONFIG_FILE
    cfg = json.loads(config.read_bytes())
    cfg.update(version=2, runtime_root=str(runtime))
    config.write_text(json.dumps(cfg))
    return app, runtime, package, cfg


def configure(app, cfg):
    (app / 'data_files' / ui.CONFIG_FILE).write_text(json.dumps(cfg))


def test_external_exact_values_read_only_and_no_legacy_reads(external, monkeypatch):
    app, runtime, package, _ = external
    before = (tree(app), tree(runtime), tree(package))
    original = ui._read
    def read(path):
        if path.is_relative_to(app / 'data_files') and path.name != ui.CONFIG_FILE:
            raise AssertionError('legacy data read')
        return original(path)
    monkeypatch.setattr(ui, '_read', read)
    loaded = ui.load_page_inputs(app / 'data_files')
    expected = pd.read_csv(runtime / 'data_files' / ui.LATEST_FILE,
                           dtype={'player_id': str}, float_precision='round_trip')
    pd.testing.assert_frame_equal(loaded.predictions, expected)
    assert loaded.managed and len(loaded.payloads) == 8
    assert before == (tree(app), tree(runtime), tree(package))


def test_legacy_writes_cannot_change_external_outputs_or_banners(external):
    app, runtime, _, _ = external
    loaded = ui.load_page_inputs(app / 'data_files')
    date_before = c.latest_stats_date(app / 'data_files', captured=loaded.payloads)
    banner = ss.remaining_games_banner(loaded.predictions, app / 'data_files', captured=loaded.payloads)
    before = tree(runtime)
    for name in [ui.LATEST_FILE, 'players.csv', *ui.STATS.values(), 'nfl_schedule_2026.csv']:
        (app / 'data_files' / name).write_bytes(b'broken legacy input')
    (app / 'player_props/models' / mb.POINTER_FILE).write_bytes(b'broken legacy pointer')
    again = ui.load_page_inputs(app / 'data_files')
    pd.testing.assert_frame_equal(loaded.predictions, again.predictions)
    assert c.latest_stats_date(app / 'data_files', captured=again.payloads) == date_before
    assert ss.remaining_games_banner(again.predictions, app / 'data_files', captured=again.payloads) == banner
    assert tree(runtime) == before


@pytest.mark.parametrize('name', [ui.LATEST_FILE, pub.snapshot_name(2026, 3),
    pub.sidecar_name(2026, 3), 'players.csv', *ui.STATS.values(), 'nfl_schedule_2026.csv'])
def test_runtime_corruption_refuses_without_fallback_or_writes(external, name):
    app, runtime, _, _ = external
    ui.load_page_inputs(app / 'data_files')
    (runtime / 'data_files' / name).write_bytes(b'bad')
    before = tree(app), tree(runtime)
    with pytest.raises(ui.PageDataError): ui.load_page_inputs(app / 'data_files')
    assert before == (tree(app), tree(runtime))


@pytest.mark.parametrize('change', ['pointer', 'model', 'extra'])
def test_external_models_are_used_and_checked(external, change):
    app, runtime, _, _ = external
    models = runtime / 'player_props/models'
    pointer = json.loads((models / mb.POINTER_FILE).read_bytes())
    active = models / mb.BUNDLES_DIR / pointer['bundle_id']
    if change == 'pointer': (models / mb.POINTER_FILE).write_bytes(b'{}')
    elif change == 'model':
        manifest = json.loads((active / mb.MANIFEST_FILE).read_bytes())
        (active / next(iter(manifest['models'].values()))['file']).write_bytes(b'broken')
    else: (active / 'unlisted.txt').write_bytes(b'extra')
    before = tree(app), tree(runtime)
    with pytest.raises(ui.PageDataError): ui.load_page_inputs(app / 'data_files')
    assert before == (tree(app), tree(runtime))


@pytest.mark.parametrize('value', [None, True, 12, [], {}, '', ' ', 'relative/runtime',
                                   ' runtime ', 'C:/runtime/../other'])
def test_invalid_roots_fail_before_package_read(layout, monkeypatch, value):
    app, runtime, _, cfg = layout
    cfg['runtime_root'] = value
    configure(app, cfg)
    from player_props import release_manifest as rm
    monkeypatch.setattr(rm, 'read_package', lambda *a: pytest.fail('package read before root refusal'))
    before = tree(app), tree(runtime)
    with pytest.raises(ui.PageDataError): ui.load_page_inputs(app / 'data_files')
    assert before == (tree(app), tree(runtime))


@pytest.mark.parametrize('where', ['app', 'app_child', 'app_parent', 'package',
                                   'package_child', 'package_parent', 'missing', 'file'])
def test_overlaps_missing_and_file_roots_refused(layout, monkeypatch, where):
    app, runtime, package, cfg = layout
    locations = {'app': app, 'app_child': app / 'data_files', 'app_parent': app.parent,
                 'package': package, 'package_child': package / 'stats', 'package_parent': package.parent,
                 'missing': app.parent / 'absent', 'file': app / 'data_files/players.csv'}
    cfg['runtime_root'] = str(locations[where]); configure(app, cfg)
    from player_props import release_manifest as rm
    monkeypatch.setattr(rm, 'read_package', lambda *a: pytest.fail('package read before overlap refusal'))
    before = tree(app), tree(runtime)
    with pytest.raises(ui.PageDataError): ui.load_page_inputs(app / 'data_files')
    assert before == (tree(app), tree(runtime))


@pytest.mark.parametrize('role', ['runtime', 'package'])
def test_git_checkout_ancestor_refused(layout, role, monkeypatch):
    app, runtime, package, _ = layout
    (runtime if role == 'runtime' else package).joinpath('.git').write_text('fake git metadata')
    from player_props import release_manifest as rm
    monkeypatch.setattr(rm, 'read_package', lambda *a: pytest.fail('package read before Git refusal'))
    try:
        with pytest.raises(ui.PageDataError, match='outside Git'): ui.load_page_inputs(app / 'data_files')
    finally:
        (runtime if role == 'runtime' else package).joinpath('.git').unlink()


def directory_link(link, target):
    if os.name == 'nt':
        result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)],
                                capture_output=True, text=True)
        if result.returncode: pytest.skip('junction creation unavailable: ' + result.stderr)
        assert link.is_junction()  # independent of the guard under test
    else:
        link.symlink_to(target, target_is_directory=True)
        assert link.is_symlink()


@pytest.mark.parametrize('where', ['root', 'data', 'models', 'inside_root'])
def test_directory_aliases_are_refused_even_inside_runtime(external, where):
    app, runtime, _, cfg = external
    if where == 'root':
        link = runtime.parent / 'alias'; directory_link(link, runtime)
        cfg['runtime_root'] = str(link); configure(app, cfg)
    else:
        path = runtime / ('data_files' if where in ('data', 'inside_root') else 'player_props/models')
        target = runtime / 'preserved' if where == 'inside_root' else runtime.parent / ('preserved-' + where)
        path.rename(target); directory_link(path, target)
    with pytest.raises(ui.PageDataError, match='links or reparse'): ui.load_page_inputs(app / 'data_files')


def test_hardlinked_runtime_file_is_refused(external):
    app, runtime, _, _ = external
    source = app / 'data_files' / ui.LATEST_FILE
    target = runtime / 'data_files' / ui.LATEST_FILE
    target.unlink(); os.link(source, target)
    with pytest.raises(ui.PageDataError, match='hard links'): ui.load_page_inputs(app / 'data_files')


def test_config_switch_and_deliberate_legacy_rollback(external):
    app, runtime, _, cfg = external
    original = ui.load_page_inputs(app / 'data_files')
    second = runtime.parent / 'second'; shutil.copytree(runtime, second)
    latest = runtime / 'data_files' / ui.LATEST_FILE
    good = latest.read_bytes(); latest.write_bytes(b'broken old runtime')
    cfg['runtime_root'] = str(second); configure(app, cfg)
    pd.testing.assert_frame_equal(original.predictions, ui.load_page_inputs(app / 'data_files').predictions)
    cfg['runtime_root'] = str(runtime); configure(app, cfg)
    with pytest.raises(ui.PageDataError): ui.load_page_inputs(app / 'data_files')
    latest.write_bytes(good)
    pd.testing.assert_frame_equal(original.predictions, ui.load_page_inputs(app / 'data_files').predictions)
    (app / 'data_files' / ui.CONFIG_FILE).unlink()
    legacy = ui.load_page_inputs(app / 'data_files')
    assert not legacy.managed and 'Legacy stored outputs' in legacy.notice
    assert latest.read_bytes() == good


def test_config_changed_after_capture_refused(external, monkeypatch):
    app, runtime, _, cfg = external
    original = ui._frame
    def change(*a, **k):
        frame = original(*a, **k)
        cfg['runtime_root'] = str(runtime.parent / 'other'); configure(app, cfg)
        return frame
    monkeypatch.setattr(ui, '_frame', change)
    with pytest.raises(ui.PageDataError, match='configuration changed'): ui.load_page_inputs(app / 'data_files')


def test_alias_introduced_after_s1_refused(external, monkeypatch):
    app, runtime, _, _ = external
    original = ui._verified_package
    def change(*a, **k):
        original(*a, **k)
        data = runtime / 'data_files'; preserved = runtime / 'preserved'
        data.rename(preserved); directory_link(data, preserved)
    monkeypatch.setattr(ui, '_verified_package', change)
    with pytest.raises(ui.PageDataError, match='links or reparse'): ui.load_page_inputs(app / 'data_files')


def test_captured_banners_never_open_a_disk_file(tmp_path, monkeypatch):
    captured = {n: b'game_date\n2026-01-04\n' for n in ui.STATS.values()}
    captured['nfl_schedule_2026.csv'] = b'week,date,home_team,away_team,venue,status\n9,2026-11-08,KC,BUF,Field,REG\n'
    predictions = pd.DataFrame({'season': [2026]})
    original = pd.read_csv
    def reader(source, *a, **k):
        assert isinstance(source, BytesIO), 'disk read from managed banner'
        return original(source, *a, **k)
    monkeypatch.setattr(pd, 'read_csv', reader)
    assert c.latest_stats_date(tmp_path, captured=captured) == date(2026, 1, 4)
    level, text = ss.remaining_games_banner(predictions, tmp_path,
        now=datetime(2026, 11, 8, tzinfo=TORONTO), captured=captured)
    assert level == 'info' and '**1 remaining' in text
    assert c.latest_stats_date(tmp_path, captured={}) is None
    assert ss.remaining_games_banner(predictions, tmp_path, captured={})[0] == 'warning'
    assert ss.remaining_games_banner(predictions, tmp_path, captured={'nfl_schedule_2026.csv': b''})[0] == 'warning'


@pytest.mark.parametrize('page', ['2_Player_Props.py', '3_Parlay_Builder.py'])
def test_external_pages_use_captured_banners_and_recheck(external, monkeypatch, page):
    app, runtime, _, _ = external
    raw = ui.load_page_inputs(app / 'data_files')
    wanted = c.latest_stats_date(app / 'data_files', captured=raw.payloads)
    for name in ui.STATS.values():
        (app / 'data_files' / name).write_bytes(b'game_date\n2050-01-01\n')
    (app / 'data_files/nfl_schedule_2026.csv').write_bytes(b'broken legacy schedule')
    at = render(page, app, monkeypatch)
    assert not at.exception and not at.error
    assert wanted.isoformat() in at.warning[0].value and '2050-01-01' not in at.warning[0].value
    if page.startswith('2'):
        assert not any('schedule' in msg.value.lower() for msg in at.warning[1:])
        assert at.selectbox(key='player_search').options
    else:
        at.multiselect[0].set_value(at.multiselect[0].options[:2]).run()
        assert not at.exception
    target = runtime / 'data_files' / ui.LATEST_FILE
    good = target.read_bytes(); target.write_bytes(b'bad runtime')
    at.run()
    assert not at.exception and at.error and not at.dataframe and not at.multiselect
    assert 'could not be checked' in at.warning[0].value
    target.write_bytes(good); at.run()
    assert not at.exception and not at.error

@pytest.fixture
def layout(tmp_path):
    app, runtime, package = (tmp_path / n for n in ['application', 'runtime', 'package'])
    (app / 'data_files').mkdir(parents=True)
    (app / 'data_files/players.csv').write_text('id\nplaceholder\n')
    runtime.mkdir(); package.mkdir()
    cfg = {'version': 2, 'runtime_root': str(runtime), 'package': str(package),
           'release_id': '20261108T120000Z-12345678', 'manifest_sha256': 'a' * 64,
           'publication_sha256': 'b' * 64, 'season': 2026, 'week': 9}
    configure(app, cfg)
    return app, runtime, package, cfg


@pytest.mark.parametrize('change', ['missing_root', 'extra_key', 'v1_with_root', 'unknown_version'])
def test_external_schema_is_explicit(layout, monkeypatch, change):
    app, runtime, _, cfg = layout
    if change == 'missing_root': del cfg['runtime_root']
    elif change == 'extra_key': cfg['another_root'] = str(runtime)
    elif change == 'v1_with_root': cfg['version'] = 1
    else: cfg['version'] = 3
    configure(app, cfg)
    from player_props import release_manifest as rm
    monkeypatch.setattr(rm, 'read_package', lambda *a: pytest.fail('package read before schema refusal'))
    before = tree(app), tree(runtime)
    with pytest.raises(ui.PageDataError): ui.load_page_inputs(app / 'data_files')
    assert before == (tree(app), tree(runtime))


def test_root_alias_refused_before_package_read(layout, monkeypatch):
    app, runtime, _, cfg = layout
    link = runtime.parent / 'alias'; directory_link(link, runtime)
    cfg['runtime_root'] = str(link); configure(app, cfg)
    from player_props import release_manifest as rm
    monkeypatch.setattr(rm, 'read_package', lambda *a: pytest.fail('package read through alias'))
    before = tree(runtime)
    with pytest.raises(ui.PageDataError, match='links or reparse'): ui.load_page_inputs(app / 'data_files')
    assert before == tree(runtime)


@pytest.mark.parametrize('page', ['2_Player_Props.py', '3_Parlay_Builder.py'])
def test_v1_banners_keep_captured_bytes_after_validation(managed, monkeypatch, page):
    app, _ = managed
    original = ui.load_page_inputs
    expected = original(app / 'data_files')
    wanted = c.latest_stats_date(app / 'data_files', captured=expected.payloads)
    def change_after_capture(data):
        result = original(data)
        for name in ui.STATS.values():
            (app / 'data_files' / name).write_bytes(b'game_date\n2050-01-01\n')
        (app / 'data_files/nfl_schedule_2026.csv').write_bytes(b'broken schedule after validation')
        return result
    monkeypatch.setattr(ui, 'load_page_inputs', change_after_capture)
    at = render(page, app, monkeypatch)
    assert not at.exception and not at.error
    assert wanted.isoformat() in at.warning[0].value and '2050-01-01' not in at.warning[0].value
    if page.startswith('2'):
        assert not any('schedule' in msg.value.lower() for msg in at.warning[1:])

def test_parent_traversal_to_existing_runtime_refused(layout, monkeypatch):
    app, runtime, _, cfg = layout
    cfg['runtime_root'] = str(runtime / '..' / runtime.name)
    configure(app, cfg)
    from player_props import release_manifest as rm
    monkeypatch.setattr(rm, 'read_package', lambda *a: pytest.fail('package read through traversal'))
    with pytest.raises(ui.PageDataError, match='parent traversal'):
        ui.load_page_inputs(app / 'data_files')


def test_linked_config_directory_refused_before_package_read(layout, monkeypatch):
    app, runtime, _, cfg = layout
    data = app / 'data_files'; saved = app / 'saved_data'
    data.rename(saved); directory_link(data, saved)
    from player_props import release_manifest as rm
    monkeypatch.setattr(rm, 'read_package', lambda *a: pytest.fail('package read through config alias'))
    before = tree(saved), tree(runtime)
    with pytest.raises(ui.PageDataError, match='links or reparse'):
        ui.load_page_inputs(data)
    assert before == (tree(saved), tree(runtime))


def test_config_alias_after_capture_refused(external, monkeypatch):
    app, runtime, _, _ = external
    original = ui._frame
    saved = app / 'saved_data'
    def change(*a, **k):
        frame = original(*a, **k)
        data = app / 'data_files'
        data.rename(saved); directory_link(data, saved)
        return frame
    monkeypatch.setattr(ui, '_frame', change)
    with pytest.raises(ui.PageDataError, match='links or reparse'):
        ui.load_page_inputs(app / 'data_files')
