"""Rehearsal runs that fail early still save summary.json with the stage's
own diagnostic, even when the runtime players reference was never written or
can't be read. Synthetic league, no network, nothing is trained: every failure
here happens in (or right after) the workspace stage, and training would fail
the test."""
import json
import socket
import sys
from pathlib import Path

import pytest

from player_props import rehearsal as rh

LATER = [s for s in rh.STAGES if s != 'workspace']


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("models were trained")
    monkeypatch.setattr('player_props.models.train_all_models', refuse)


def run(workspace, hook=None, **kw):
    return rh.run_rehearsal(rh.RehearsalConfig(workspace=workspace, **kw), after_stage=hook)


def run_dir(summary, workspace):
    return Path(workspace).resolve() / "runs" / summary["run_id"]


def fail_after_workspace(prepare=None):
    """after_stage hook: optionally touch the run, then fail the workspace stage."""
    def hook(name, r):
        if name == 'workspace':
            if prepare:
                prepare(r)
            raise rh.RehearsalError("injected workspace failure")
    return hook


def failed_early(summary, workspace, message):
    """The workspace stage failed with ``message``; every later stage is
    not_run; summary.json on disk is the returned summary."""
    assert summary['status'] == 'failed'
    stages = {s['stage']: s for s in summary['stages']}
    assert stages['workspace']['status'] == 'failed'
    assert message in stages['workspace']['error']['message']
    assert all(stages[s]['status'] == 'not_run' and 'error' not in stages[s] for s in LATER)
    saved = json.loads((run_dir(summary, workspace) / 'summary.json').read_text(encoding='utf-8'))
    assert saved['stages'] == summary['stages'] and saved['eligibility'] == summary['eligibility']
    assert summary['bundle'] is None and summary['completeness'] is None
    e = summary['eligibility']
    assert (e['policy'], e['field']) == ('position-group-v1', 'position_group')
    # Verification never ran: no reference was read, none is substituted.
    assert e['players_reference_sha256'] is None and e['players_reference_source'] is None
    assert e['verified'] is False and e['players_reference_unavailable']
    assert 'players_reference_current_sha256' not in e
    return e


class TestInvalidTrainingSeasons:
    @pytest.mark.parametrize('seasons, message', [
        ([2022, 2024, 2025], "must be consecutive"),
        ([2025], "at least two seasons"),
        ([], "train_seasons is empty"),
    ])
    def test_rejected_before_workspace_setup(self, tmp_path, seasons, message):
        s = run(tmp_path / 'ws', train_seasons=seasons)
        e = failed_early(s, tmp_path / 'ws', message)
        assert e['restated_policy'] == 'position-group-v1'
        assert e['players_reference_unavailable'] == "the run stopped before runtime/data was set up"
        rd = run_dir(s, tmp_path / 'ws')
        # No workspace directory was created; only the run's diagnostics and summary.
        assert sorted(p.name for p in rd.iterdir()) == ['diagnostics', 'summary.json']
        assert message in (rd / 'diagnostics' / 'traceback.txt').read_text(encoding='utf-8')


class TestDirectorySetupFailure:
    @staticmethod
    def block_runtime(monkeypatch, also_diagnostics=False):
        """A plain file where runtime/ should be, so creating runtime/data
        really fails (before run.paths['runtime_data'] is assigned)."""
        real = rh.safe_dir

        def safe_dir(path, *a, **k):
            if Path(path).parts[-2:] == ('runtime', 'data') and not Path(path).parent.exists():
                Path(path).parent.write_text('not a directory', encoding='utf-8')
            if also_diagnostics and Path(path).name == 'diagnostics':
                raise PermissionError("diagnostics can't be created")
            return real(path, *a, **k)
        monkeypatch.setattr(rh, 'safe_dir', safe_dir)

    def test_failure_before_runtime_data_is_assigned(self, tmp_path, monkeypatch):
        self.block_runtime(monkeypatch)
        s = run(tmp_path / 'ws')
        stages = {x['stage']: x for x in s['stages']}
        assert stages['workspace']['error']['type'] in ('FileExistsError', 'NotADirectoryError',
                                                        'FileNotFoundError', 'OSError')
        e = failed_early(s, tmp_path / 'ws', 'runtime')
        assert e['players_reference_unavailable'] == "the run stopped before runtime/data was set up"
        assert 'traceback_not_saved' not in stages['workspace']['error']

    def test_diagnostics_failure_keeps_the_original_error(self, tmp_path, monkeypatch):
        self.block_runtime(monkeypatch, also_diagnostics=True)
        s = run(tmp_path / 'ws')
        error = {x['stage']: x for x in s['stages']}['workspace']['error']
        assert error['type'] != 'PermissionError'
        assert "diagnostics can't be created" in error['traceback_not_saved']
        failed_early(s, tmp_path / 'ws', 'runtime')


class TestRuntimeReferenceDuringFailureReporting:
    def test_missing_reference_is_recorded_not_substituted(self, tmp_path):
        s = run(tmp_path / 'ws', fail_after_workspace())
        e = failed_early(s, tmp_path / 'ws', "injected workspace failure")
        assert e['players_reference_unavailable'] == "runtime/data/players.csv was never written"
        # The fixtures copy exists (and is hashed as an artifact), but isn't used as the reference.
        rd = run_dir(s, tmp_path / 'ws')
        assert (rd / 'fixtures' / 'players.csv').is_file()
        assert s['artifacts']['fixtures/players.csv'] == rh._sha256_file(rd / 'fixtures' / 'players.csv')
        assert s['artifacts']['fixtures/players.csv'] not in e.values()

    def test_reference_that_is_a_directory(self, tmp_path):
        def prepare(r):
            (r.paths['runtime_data'] / 'players.csv').mkdir()
        s = run(tmp_path / 'ws', fail_after_workspace(prepare))
        e = failed_early(s, tmp_path / 'ws', "injected workspace failure")
        assert e['players_reference_unavailable'].startswith("runtime/data/players.csv can't be read (")

    def test_unreadable_reference_and_artifacts(self, tmp_path, monkeypatch):
        real = rh._sha256_file

        def sha(path):
            if Path(path).name == 'players.csv':
                raise PermissionError(f"denied: {Path(path).name}")
            return real(path)
        monkeypatch.setattr(rh, '_sha256_file', sha)

        def prepare(r):
            (r.paths['runtime_data'] / 'players.csv').write_text('gsis_id\n', encoding='utf-8')
        s = run(tmp_path / 'ws', fail_after_workspace(prepare))
        e = failed_early(s, tmp_path / 'ws', "injected workspace failure")
        assert e['players_reference_unavailable'] == ("runtime/data/players.csv can't be read "
                                                      "(PermissionError: denied: players.csv)")
        assert set(s['artifacts_unreadable']) == {'fixtures/players.csv', 'runtime/data/players.csv'}
        assert not {'fixtures/players.csv', 'runtime/data/players.csv'} & set(s['artifacts'])

    def test_restated_policy_unavailable_keeps_production_policy(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, 'player_props.rehearsal_expected', None)
        s = run(tmp_path / 'ws', fail_after_workspace())
        e = failed_early(s, tmp_path / 'ws', "injected workspace failure")
        assert e['restated_policy'] is None and 'ModuleNotFoundError' in e['restated_policy_unavailable']
