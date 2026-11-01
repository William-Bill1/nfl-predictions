"""Verified publication (S2): publishing Player Props outputs only after
verifying them against an S1 release package. Synthetic rehearsal runs only;
everything is written under tmp_path."""
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from player_props import publication as pub
from player_props import rehearsal as rh
from player_props import release_manifest as rm

ROOT = Path(__file__).resolve().parents[1]
SEASON, WEEK = 2026, 3
SNAP = pub.snapshot_name(SEASON, WEEK)
SIDE = pub.sidecar_name(SEASON, WEEK)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_dir(ws: Path, seed: int) -> Path:
    summary = rh.run_rehearsal(rh.RehearsalConfig(workspace=ws, seed=seed))
    assert summary['status'] == 'succeeded', summary['stages']
    return ws / 'runs' / summary['run_id']


def package(run: Path, base: Path, name: str) -> Path:
    cand = base / f"{name}_candidate.csv"
    pd.read_csv(run / 'runtime/data/players.csv', dtype=str, keep_default_na=False).to_csv(
        cand, index=False, quoting=csv.QUOTE_ALL)
    out = base / name
    rm.build_package(rm.inputs_from_run(run, cand), out, 'reserialized', input_dirs=[run])
    return out


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    base = tmp_path_factory.mktemp("publication")
    a, b = run_dir(base / "wsA", 7), run_dir(base / "wsB", 11)
    e = {'base': base, 'runA': a, 'runB': b,
         'pkgA': package(a, base, 'pkgA'), 'pkgB': package(b, base, 'pkgB'),
         'pkgA2': package(a, base, 'pkgA2')}       # same run, another release
    for k, run in (('A', a), ('B', b)):
        data = run / 'runtime' / 'data'
        e[f'latest{k}'], e[f'snap{k}'] = data / pub.LATEST_NAME, data / SNAP
        e[f'sched{k}'] = data / f'nfl_schedule_{SEASON}.csv'
    return e


@pytest.fixture(scope="module")
def ctx(env, tmp_path_factory):
    return pub.prepare(env['pkgA'], env['schedA'].read_bytes(), tmp_path_factory.mktemp("ctx"))


@pytest.fixture
def outputs(env, tmp_path):
    """Copies of run A's outputs in their own directory (inputs for publish)."""
    d = tmp_path / 'outputs'
    d.mkdir()
    shutil.copyfile(env['latestA'], d / pub.LATEST_NAME)
    shutil.copyfile(env['snapA'], d / SNAP)
    return {'latest': d / pub.LATEST_NAME, 'snapshot': d / SNAP, 'dir': d}


def frame(data: bytes) -> pd.DataFrame:
    return rh.read_predictions(io.BytesIO(data))


def to_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')


def tampered(env, change) -> bytes:
    df = frame(env['latestA'].read_bytes())
    change(df)
    return to_bytes(df)


def check(ctx, latest: bytes, snapshot: bytes = None) -> dict:
    return pub.verify_outputs(ctx, latest, latest if snapshot is None else snapshot)


def has(items, text) -> bool:
    return any(text in i for i in items)


def tree(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): sha(p.read_bytes()) for p in sorted(d.rglob('*')) if p.is_file()} \
        if d.exists() else None


def do_publish(env, dest, outputs=None, package='pkgA', **kw):
    if outputs is None:
        return pub.publish(env[package], dest, env['schedA'], packaged=True, **kw)
    return pub.publish(env[package], dest, env['schedA'], latest=outputs['latest'], snapshot=outputs['snapshot'], **kw)


# ---------------------------------------------------------------------------

class TestPublication:
    def test_supplied_files_publish_with_sidecar(self, env, outputs, tmp_path):
        dest = tmp_path / 'dest'
        report = do_publish(env, dest, outputs)
        assert {k: v['status'] for k, v in report['files'].items()} == \
            {'latest': 'created', 'snapshot': 'created', 'sidecar': 'created'}
        assert sorted(p.name for p in dest.iterdir()) == sorted([pub.LATEST_NAME, SNAP, SIDE])
        assert (dest / pub.LATEST_NAME).read_bytes() == env['latestA'].read_bytes()
        assert (dest / SNAP).read_bytes() == env['snapA'].read_bytes()
        side = json.loads((dest / SIDE).read_text(encoding='utf-8'))
        man = (env['pkgA'] / rm.MANIFEST_NAME).read_bytes()
        assert side['release'] == {'release_id': json.loads(man)['release_id'], 'manifest_sha256': sha(man)}
        assert side['outputs']['snapshot']['sha256'] == sha(env['snapA'].read_bytes())
        assert side['inference_target'] == {'season': SEASON, 'week': WEEK}
        d = json.loads(man)['derived']
        assert side['bundle']['bundle_id'] == d['bundle']['bundle_id']
        assert side['reference']['runtime_sha256'] == d['identities']['reference_runtime']['sha256']
        assert side['eligibility']['players_reference_sha256'] == side['reference']['runtime_sha256']
        assert side['eligibility']['policy'] == 'position-group-v1'
        assert side['verification']['result'] == 'passed'
        assert side['verification']['rows'] == side['verification']['expected_rows']
        assert pub.verify_published(dest, SEASON, WEEK, env['pkgA'])['ok']

    def test_identical_repeat_is_safe(self, env, outputs, tmp_path):
        dest = tmp_path / 'dest'
        do_publish(env, dest, outputs)
        before = tree(dest)
        report = do_publish(env, dest, outputs)
        assert {v['status'] for v in report['files'].values()} == {'identical_existing'}
        assert tree(dest) == before

    def test_packaged_snapshot_is_published_and_marked(self, env, tmp_path):
        report = do_publish(env, tmp_path / 'dest')
        assert report['outputs_source'] == 'package_snapshot' and report['package_rehearsal_names_these_bytes']
        packaged = (env['pkgA'] / rm.ROLES['verified_predictions']).read_bytes()
        assert (tmp_path / 'dest' / SNAP).read_bytes() == packaged
        assert (tmp_path / 'dest' / pub.LATEST_NAME).read_bytes() == packaged

    def test_same_bytes_from_either_source_give_the_same_sidecar(self, env, outputs, tmp_path):
        do_publish(env, tmp_path / 'a')
        do_publish(env, tmp_path / 'b', outputs)
        assert tree(tmp_path / 'a') == tree(tmp_path / 'b')
        assert do_publish(env, tmp_path / 'a', outputs)['files']['sidecar']['status'] == 'identical_existing'

    def test_new_outputs_are_not_covered_by_package_evidence(self, env, tmp_path):
        """Outputs that verify but aren't the bytes the package's rehearsal
        verified (here: rows reordered) are marked as independent only."""
        df = frame(env['latestA'].read_bytes()).iloc[::-1]
        d = tmp_path / 'new'
        d.mkdir()
        (d / pub.LATEST_NAME).write_bytes(to_bytes(df))
        (d / SNAP).write_bytes(to_bytes(df))
        report = do_publish(env, tmp_path / 'dest', {'latest': d / pub.LATEST_NAME, 'snapshot': d / SNAP})
        assert report['package_rehearsal_names_these_bytes'] is False
        side = json.loads((tmp_path / 'dest' / SIDE).read_text(encoding='utf-8'))
        assert side['evidence']['package_rehearsal_names_these_bytes'] is False
        assert "does not cover these bytes" in side['evidence']['statement']

    def test_cli_publish_and_verify(self, env, outputs, tmp_path):
        """The review's failing case, with no UTF-8 forcing in the environment
        (Python's default pipe encoding, cp1252 on Windows)."""
        dest = tmp_path / 'dest'
        r = subprocess.run([sys.executable, 'player_props/publication.py', 'publish', '--package', str(env['pkgA']),
                            '--dest', str(dest), '--prediction-schedule', str(env['schedA']),
                            '--latest', str(outputs['latest']), '--snapshot', str(outputs['snapshot'])],
                           cwd=ROOT, capture_output=True, env=cli_env(None))
        assert r.returncode == 0, r.stdout.decode('ascii', 'replace') + r.stderr.decode('ascii', 'replace')
        assert json.loads(r.stdout.decode('ascii'))['state'] == 'committed'
        r = subprocess.run([sys.executable, 'player_props/publication.py', 'verify-published', '--dest', str(dest),
                            '--season', str(SEASON), '--week', str(WEEK), '--package', str(env['pkgA'])],
                           cwd=ROOT, capture_output=True, env=cli_env(None))
        assert r.returncode == 0 and json.loads(r.stdout.decode('ascii'))['ok'], r.stdout


class TestRows:
    """Row-level verification of captured bytes against the package (one prepared context)."""

    def test_round_trip_control(self, env, ctx):
        """The tampering helper itself leaves a valid file valid."""
        assert check(ctx, tampered(env, lambda df: None))['ok']

    def test_missing_row(self, env, ctx):
        r = check(ctx, tampered(env, lambda df: df.drop(index=3, inplace=True)))
        assert not r['ok'] and has(r['problems'], "1 expected prediction(s) missing")

    def test_extra_row(self, env, ctx):
        def change(df):
            extra = df.iloc[[0]].copy()
            extra['line_value'] = extra['line_value'] + 100
            df.loc[len(df)] = extra.iloc[0]
        r = check(ctx, tampered(env, change))
        assert not r['ok'] and has(r['problems'], "1 unexpected prediction(s)")

    def test_duplicate_row(self, env, ctx):
        r = check(ctx, tampered(env, lambda df: df.loc.__setitem__(len(df), df.iloc[0])))
        assert not r['ok'] and has(r['problems'], "listed more than once")

    def test_wrong_side(self, env, ctx):
        r = check(ctx, tampered(env, lambda df: df.loc.__setitem__((0, 'is_home'), not df.loc[0, 'is_home'])))
        assert not r['ok'] and has(r['problems'], "latest feed completeness: wrong_side 1")

    def test_wrong_identity(self, env, ctx):
        def change(df):
            other = df.loc[df['team'] != df.loc[0, 'team'], 'player_id'].iloc[0]
            df.loc[0, 'player_id'] = other
        r = check(ctx, tampered(env, change))
        assert not r['ok'] and (has(r['problems'], "is not a") or has(r['problems'], "missing"))

    def test_wrong_display_name(self, env, ctx):
        r = check(ctx, tampered(env, lambda df: df.loc.__setitem__((0, 'display_name'), 'Someone Else')))
        assert not r['ok'] and has(r['problems'], "display_name 'Someone Else' is not")

    def test_changed_probability(self, env, ctx):
        r = check(ctx, tampered(env, lambda df: df.loc.__setitem__((0, 'prob_over'), df.loc[0, 'prob_over'] + 1e-9)))
        assert not r['ok'] and has(r['problems'], "prob_over is")

    def test_changed_adjustment(self, env, ctx):
        r = check(ctx, tampered(env, lambda df: df.loc.__setitem__((0, 'adjustments'), 'clamp_star_floor_0.75')))
        assert not r['ok'] and has(r['problems'], "adjustments is")

    def test_wrong_bundle(self, env, ctx):
        r = check(ctx, tampered(env, lambda df: df.__setitem__('model_bundle', '20990101T000000Z-00000000')))
        assert not r['ok'] and has(r['problems'], "is not this run's bundle")

    def test_wrong_target_week(self, env, ctx):
        r = check(ctx, tampered(env, lambda df: df.__setitem__('week', WEEK + 1)))
        assert not r['ok'] and has(r['problems'], "is not the target")

    def test_wrong_eligibility_provenance(self, env, ctx):
        r = check(ctx, tampered(env, lambda df: df.loc.__setitem__((0, 'eligibility_group'), 'XX')))
        assert not r['ok'] and has(r['problems'], "eligibility_group 'XX' is not")

    def test_snapshot_differs_from_latest(self, env, ctx):
        latest = env['latestA'].read_bytes()
        snap = tampered(env, lambda df: df.loc.__setitem__((1, 'confidence'), df.loc[1, 'confidence'] + 1e-9))
        r = check(ctx, latest, snap)
        assert not r['ok'] and has(r['problems'], "does not hold exactly the latest feed")

    def test_other_runs_outputs(self, env, ctx):
        """Outputs from another release's run (different bundle) don't verify here."""
        r = check(ctx, env['latestB'].read_bytes(), env['snapB'].read_bytes())
        assert not r['ok'] and has(r['problems'], "is not this run's bundle")

    def test_reference_other_than_the_packages(self, env, ctx, tmp_path_factory):
        """If the reference the verifier reads isn't the package's runtime
        reference, verification fails."""
        c = pub.prepare(env['pkgA'], env['schedA'].read_bytes(), tmp_path_factory.mktemp("ctx_ref"))
        ref = c.runtime_data / 'players.csv'
        ref.write_bytes(ref.read_bytes() + b'\n')
        r = check(c, env['latestA'].read_bytes())
        assert not r['ok'] and has(r['problems'], "players reference other than the package's runtime reference")

    @pytest.mark.parametrize("data, text", [
        (b'', "is empty"),
        (b'\x00\x01\x02 not a csv \xff', "is not readable as a predictions CSV (UnicodeDecodeError"),
        (b'a,b\n1,2\n', "lacks required column(s) ['player_id', 'team', 'opponent'"),
        (b'player_id,team\n', "lacks required column(s)"),
        (b'"unterminated\n', "not readable as a predictions CSV"),
    ])
    def test_malformed_outputs(self, ctx, data, text):
        with pytest.raises(pub.PublicationError) as exc:
            check(ctx, data)
        assert has(exc.value.problems, text), exc.value.problems

    def test_header_only(self, env, ctx):
        header = env['latestA'].read_bytes().splitlines(keepends=True)[0]
        with pytest.raises(pub.PublicationError, match="has a header but no predictions"):
            check(ctx, header)


class TestPackageAndInputs:
    def test_wrong_release_package(self, env, outputs, tmp_path):
        """Run A's outputs presented with package B: B's schedule is required,
        and then the bundle and rows disagree."""
        with pytest.raises(pub.PublicationError) as exc:
            pub.publish(env['pkgB'], tmp_path / 'dest', env['schedB'], latest=outputs['latest'],
                        snapshot=outputs['snapshot'])
        assert has(exc.value.problems, "outputs not published") and has(exc.value.problems, "bundle")
        assert not (tmp_path / 'dest').exists() or not any((tmp_path / 'dest').iterdir())

    def test_invalid_package(self, env, outputs, tmp_path):
        bad = tmp_path / 'pkg'
        shutil.copytree(env['pkgA'], bad)
        p = bad / rm.ROLES['rehearsal_summary']
        p.write_bytes(p.read_bytes() + b' ')
        with pytest.raises(pub.PublicationError) as exc:
            pub.publish(bad, tmp_path / 'dest', env['schedA'], latest=outputs['latest'], snapshot=outputs['snapshot'])
        assert has(exc.value.problems, "the release package does not verify: evidence/summary.json: size")

    def test_prediction_schedule_not_the_packages(self, env, outputs, tmp_path):
        sched = tmp_path / 'sched.csv'
        sched.write_bytes(env['schedA'].read_bytes() + b'\n')
        with pytest.raises(pub.PublicationError, match="is not the one the package's verified rehearsal used"):
            pub.publish(env['pkgA'], tmp_path / 'dest', sched, latest=outputs['latest'], snapshot=outputs['snapshot'])

    def test_missing_prediction_schedule(self, env, outputs, tmp_path):
        with pytest.raises(pub.PublicationError, match="prediction schedule .* can't be read"):
            pub.publish(env['pkgA'], tmp_path / 'dest', tmp_path / 'nope.csv', latest=outputs['latest'],
                        snapshot=outputs['snapshot'])

    def test_snapshot_named_for_another_target(self, env, outputs, tmp_path):
        other = outputs['dir'] / pub.snapshot_name(SEASON, WEEK + 1)
        shutil.copyfile(outputs['snapshot'], other)
        with pytest.raises(pub.PublicationError, match="but the package's target is 2026 week 3"):
            pub.publish(env['pkgA'], tmp_path / 'dest', env['schedA'], latest=outputs['latest'], snapshot=other)

    def test_packaged_refused_when_the_rehearsal_latest_feed_differed(self, env, tmp_path):
        """The package holds only the snapshot; if the rehearsal's latest feed had
        other bytes, the snapshot can't be published as the latest feed. The
        package is resealed so it still passes S1."""
        import tempfile
        pkg = tmp_path / 'pkg'
        shutil.copytree(env['pkgA'], pkg)
        p = pkg / rm.ROLES['rehearsal_summary']
        summary = json.loads(p.read_text(encoding='utf-8'))
        summary['outputs']['verify_outputs']['latest_sha256'] = '0' * 64
        p.write_text(json.dumps(summary), encoding='utf-8')
        doc = json.loads((pkg / rm.MANIFEST_NAME).read_text(encoding='utf-8'))
        contents = {r: (pkg / rm.ROLES[r]).read_bytes() for r in rm.ROLES}
        doc['assets'] = [{'role': r, 'name': rm.ROLES[r], 'bytes': len(c), 'sha256': sha(c)} for r, c in contents.items()]
        with tempfile.TemporaryDirectory() as tmp:
            rm._write_snapshot(contents, Path(tmp))
            bm, bs, index, problems = rm._bundle(contents, Path(tmp))
        assert not problems
        doc['derived'] = rm.derive(contents, bm, bs, index)
        (pkg / rm.MANIFEST_NAME).write_text(json.dumps(doc, indent=2), encoding='utf-8')
        assert rm.verify_package(pkg)['ok']
        with pytest.raises(pub.PublicationError, match="its rehearsal's latest feed had other bytes"):
            pub.publish(pkg, tmp_path / 'dest', env['schedA'], packaged=True)
        assert not (tmp_path / 'dest').exists() or not any((tmp_path / 'dest').iterdir())

    def test_mode_arguments(self, env, outputs, tmp_path):
        with pytest.raises(pub.PublicationError, match="give either packaged=True"):
            pub.publish(env['pkgA'], tmp_path / 'd', env['schedA'], latest=outputs['latest'])
        with pytest.raises(pub.PublicationError, match="give either packaged=True"):
            pub.publish(env['pkgA'], tmp_path / 'd', env['schedA'], packaged=True, latest=outputs['latest'],
                        snapshot=outputs['snapshot'])


class TestCapturedBytes:
    def test_inputs_changed_after_verification_do_not_change_published_bytes(self, env, outputs, tmp_path):
        original = {k: outputs[k].read_bytes() for k in ('latest', 'snapshot')}

        def change_inputs():
            for k in ('latest', 'snapshot'):
                outputs[k].write_bytes(b'player_id\nreplaced after verification\n')
        do_publish(env, tmp_path / 'dest', outputs, after_verify=change_inputs)
        assert (tmp_path / 'dest' / pub.LATEST_NAME).read_bytes() == original['latest']
        assert (tmp_path / 'dest' / SNAP).read_bytes() == original['snapshot']
        assert outputs['latest'].read_bytes() != original['latest']          # the source really changed

    def test_inputs_removed_after_verification(self, env, outputs, tmp_path):
        original = outputs['snapshot'].read_bytes()
        do_publish(env, tmp_path / 'dest', outputs, after_verify=lambda: shutil.rmtree(outputs['dir']))
        assert (tmp_path / 'dest' / SNAP).read_bytes() == original

    def test_package_changed_after_verification_does_not_change_sidecar(self, env, outputs, tmp_path):
        pkg = tmp_path / 'pkg'
        shutil.copytree(env['pkgA'], pkg)
        manifest_sha = sha((pkg / rm.MANIFEST_NAME).read_bytes())
        do_publish(env, tmp_path / 'dest', outputs, package='pkgA',
                   after_verify=lambda: (pkg / rm.MANIFEST_NAME).write_text('{}', encoding='utf-8'))
        side = json.loads((tmp_path / 'dest' / SIDE).read_text(encoding='utf-8'))
        assert side['release']['manifest_sha256'] == manifest_sha


class TestConflicts:
    def test_conflicting_snapshot_is_never_replaced(self, env, outputs, tmp_path):
        dest = tmp_path / 'dest'
        dest.mkdir()
        (dest / SNAP).write_bytes(b'someone else froze this week\n')
        (dest / pub.LATEST_NAME).write_bytes(b'previous latest\n')
        before = tree(dest)
        with pytest.raises(pub.PublicationError, match=f"{SNAP} is already published with different content"):
            do_publish(env, dest, outputs)
        assert tree(dest) == before

    def test_conflicting_sidecar_from_another_release(self, env, outputs, tmp_path):
        """Same verified bytes, another release package: the frozen provenance
        is not silently rewritten."""
        dest = tmp_path / 'dest'
        do_publish(env, dest, outputs)
        before = tree(dest)
        with pytest.raises(pub.PublicationError, match=f"{SIDE} is already published with different content"):
            do_publish(env, dest, outputs, package='pkgA2')
        assert tree(dest) == before

    def test_lock_held_by_another_publication(self, env, outputs, tmp_path):
        dest = tmp_path / 'dest'
        dest.mkdir()
        lock = dest / f".{pub.LOCK_SLOT}.lock"
        lock.write_text(json.dumps({'id': 'other', 'pid': 1, 'host': 'elsewhere',
                                    'started_at': '2026-10-31T00:00:00Z'}), encoding='utf-8')
        with pytest.raises(pub.PublicationError, match="another prop publication holds lock file"):
            do_publish(env, dest, outputs)
        assert sorted(p.name for p in dest.iterdir()) == [lock.name]


class TestRollback:
    """Catchable failures at each replacement point restore the complete previous set."""

    def _prepared(self, tmp_path):
        dest = tmp_path / 'dest'
        dest.mkdir()
        (dest / pub.LATEST_NAME).write_bytes(b'previous latest feed\n')
        (dest / pub.snapshot_name(SEASON, WEEK - 1)).write_bytes(b'an earlier week, untouched\n')
        return dest

    @staticmethod
    def failing_replace(monkeypatch, fail_on: int, fail_restore: bool = False):
        real, calls = os.replace, {'move': 0}

        def replace(src, dst, *a, **k):
            s = str(src)
            if '.publish-' in s:
                calls['move'] += 1
                if calls['move'] == fail_on:
                    raise OSError(f"simulated failure moving into place #{fail_on}")
            elif s.endswith('.bak') and fail_restore:
                raise OSError("simulated failure restoring a backup")
            return real(src, dst, *a, **k)
        monkeypatch.setattr(os, 'replace', replace)

    @pytest.mark.parametrize("fail_on", [1, 2, 3])
    def test_failure_with_a_previous_latest(self, env, outputs, tmp_path, monkeypatch, fail_on):
        dest = self._prepared(tmp_path)
        before = tree(dest)
        self.failing_replace(monkeypatch, fail_on)
        with pytest.raises(pub.PublicationError, match="previous latest feed, snapshot and sidecar were restored") as exc:
            do_publish(env, dest, outputs)
        monkeypatch.undo()
        assert exc.value.state == 'rolled_back' and exc.value.retained_files == []
        assert tree(dest) == before                    # previous latest back; new snapshot/sidecar absent
        assert not list(dest.glob('.publish-*')) and not list(dest.glob('.*.bak'))

    @pytest.mark.parametrize("fail_on", [1, 2, 3])
    def test_failure_when_nothing_existed(self, env, outputs, tmp_path, monkeypatch, fail_on):
        dest = tmp_path / 'dest'
        self.failing_replace(monkeypatch, fail_on)
        with pytest.raises(pub.PublicationError, match="were restored") as exc:
            do_publish(env, dest, outputs)
        monkeypatch.undo()
        assert exc.value.state == 'rolled_back'
        assert sorted(p.name for p in dest.iterdir()) == []

    def test_backup_copy_failure(self, env, outputs, tmp_path, monkeypatch):
        dest = self._prepared(tmp_path)
        before = tree(dest)

        def copy2(*a, **k):
            raise OSError("simulated failure copying a backup")
        monkeypatch.setattr(pub.shutil, 'copy2', copy2)
        with pytest.raises(pub.PublicationError, match="nothing was replaced") as exc:
            do_publish(env, dest, outputs)
        monkeypatch.undo()
        assert exc.value.state == 'rolled_back'
        assert tree(dest) == before and not list(dest.glob('.*.bak'))

    def test_staging_write_failure(self, env, outputs, tmp_path, monkeypatch):
        dest = self._prepared(tmp_path)
        before = tree(dest)
        real = Path.write_bytes

        def write_bytes(self, data):
            if '.publish-' in str(self):
                raise OSError("simulated failure staging a file")
            return real(self, data)
        monkeypatch.setattr(Path, 'write_bytes', write_bytes)
        with pytest.raises(pub.PublicationError, match="nothing was replaced") as exc:
            do_publish(env, dest, outputs)
        monkeypatch.undo()
        assert exc.value.state == 'rolled_back' and tree(dest) == before

    def test_failed_rollback_keeps_and_names_recovery_files(self, env, outputs, tmp_path, monkeypatch):
        dest = self._prepared(tmp_path)
        self.failing_replace(monkeypatch, 3, fail_restore=True)
        with pytest.raises(pub.PublicationError) as exc:
            do_publish(env, dest, outputs)
        monkeypatch.undo()
        err = exc.value
        assert err.state == 'rollback_failed' and has(err.problems, "the rollback also failed")
        assert len(err.recovery_files) == 1 and has(err.problems, "recovery file kept")
        backup = Path(err.recovery_files[0])
        assert backup.parent == rm._real(dest) and backup.read_bytes() == b'previous latest feed\n'


class TestCommitBoundary:
    """After every file is in place the publication has committed: cleanup
    failures are warnings with retained paths, never a claimed rollback."""

    @staticmethod
    def three_existing(tmp_path):
        d = tmp_path / 'd'
        d.mkdir()
        old = {d / n: f"old {n}\n".encode() for n in ('a.csv', 'b.csv', 'c.json')}
        for p, data in old.items():
            p.write_bytes(data)
        return d, old, {p: f"new {p.name}\n".encode() for p in old}

    @staticmethod
    def failing_backup_unlink(monkeypatch, fail_at):
        """Deleting the backups: the ``fail_at`` deletions (1-based) raise."""
        real, calls = Path.unlink, {'n': 0}

        def unlink(self, *a, **k):
            if self.name.endswith('.bak'):
                calls['n'] += 1
                if calls['n'] in fail_at:
                    raise PermissionError(f"simulated: {self.name} is in use")
            return real(self, *a, **k)
        monkeypatch.setattr(Path, 'unlink', unlink)

    @pytest.mark.parametrize("fail_at", [{1}, {2}, {3}, {2, 3}, {1, 2, 3}],
                             ids=["first", "second-after-first-deleted", "third-after-two-deleted", "last-two", "all"])
    def test_backup_cleanup_failures(self, tmp_path, monkeypatch, fail_at):
        d, old, new = self.three_existing(tmp_path)
        self.failing_backup_unlink(monkeypatch, fail_at)
        outcome = pub.install_set(new, d / '.publish-x')
        monkeypatch.undo()
        assert outcome['committed'] is True
        assert {p: p.read_bytes() for p in new} == new                     # every new file installed
        baks = sorted(d.glob('.*.bak'))
        assert sorted(outcome['retained_files']) == sorted(str(b) for b in baks)
        assert len(baks) == len(fail_at) == len(outcome['cleanup_warnings'])
        # Each retained backup holds the previous content of its own file, and the
        # backups deleted before a failure are really gone.
        kept = {b.name.split('.')[1]: b.read_bytes() for b in baks}
        assert all(old[d / f"{stem}.{'json' if stem == 'c' else 'csv'}"] == data for stem, data in kept.items())
        assert not (d / '.publish-x').exists()

    def test_staging_cleanup_failure_after_commit(self, tmp_path, monkeypatch):
        d, old, new = self.three_existing(tmp_path)
        real = shutil.rmtree

        def rmtree(path, *a, **k):
            if '.publish-' in str(path):
                raise PermissionError("simulated: staging directory in use")
            return real(path, *a, **k)
        monkeypatch.setattr(pub.shutil, 'rmtree', rmtree)
        outcome = pub.install_set(new, d / '.publish-x')
        monkeypatch.undo()
        assert {p: p.read_bytes() for p in new} == new
        assert outcome['retained_files'] == [str(d / '.publish-x')] and (d / '.publish-x').is_dir()
        assert has(outcome['cleanup_warnings'], "staging directory was not deleted")

    def test_publication_reports_committed_with_cleanup_warnings(self, env, outputs, tmp_path, monkeypatch):
        """The review's reproduction: a .bak deletion fails after installation."""
        dest = tmp_path / 'dest'
        dest.mkdir()
        (dest / pub.LATEST_NAME).write_bytes(b'previous latest feed\n')
        self.failing_backup_unlink(monkeypatch, {1})
        report = do_publish(env, dest, outputs)
        monkeypatch.undo()
        assert report['published'] and report['state'] == 'committed_with_cleanup_warnings'
        assert (dest / pub.LATEST_NAME).read_bytes() == outputs['latest'].read_bytes()
        assert (dest / SNAP).read_bytes() == outputs['snapshot'].read_bytes() and (dest / SIDE).is_file()
        [bak] = dest.glob('.*.bak')
        assert report['retained_files'] == [str(bak)] and bak.read_bytes() == b'previous latest feed\n'
        assert has(report['cleanup_warnings'], "backup of the previous player_props_predictions.csv was not deleted")
        assert pub.verify_published(dest, SEASON, WEEK, env['pkgA'])['ok']

    def test_clean_publication_has_no_warnings(self, env, outputs, tmp_path):
        report = do_publish(env, tmp_path / 'dest', outputs)
        assert report['state'] == 'committed' and report['cleanup_warnings'] == report['retained_files'] == []

    def test_cli_exit_code_for_cleanup_warnings(self, env, outputs, tmp_path, monkeypatch, capsys):
        dest = tmp_path / 'dest'
        dest.mkdir()
        (dest / pub.LATEST_NAME).write_bytes(b'previous latest feed\n')
        self.failing_backup_unlink(monkeypatch, {1})
        code = pub.main(['publish', '--package', str(env['pkgA']), '--dest', str(dest), '--prediction-schedule',
                         str(env['schedA']), '--latest', str(outputs['latest']), '--snapshot', str(outputs['snapshot'])])
        monkeypatch.undo()
        out = json.loads(capsys.readouterr().out)
        assert code == 3 and out['state'] == 'committed_with_cleanup_warnings' and out['retained_files']


LOCK = f".{pub.LOCK_SLOT}.lock"


@pytest.fixture
def lock_unlink_fails(monkeypatch):
    """The real SlotLock.__exit__ runs, but deleting the lock file raises
    PermissionError, which it retries (window shortened here) and swallows."""
    import ontario_spreads
    from datetime import timedelta
    monkeypatch.setattr(ontario_spreads.SlotLock, 'UNLINK_RETRY', timedelta(0))
    real = Path.unlink

    def unlink(self, *a, **k):
        if self.name == LOCK:
            raise PermissionError(f"simulated: {self.name} is in use")
        return real(self, *a, **k)
    monkeypatch.setattr(Path, 'unlink', unlink)
    return monkeypatch


def rewrite_lock_during_install(monkeypatch, content):
    """While the lock is held, replace its file's content (another holder's
    token, or something unreadable) before installation proceeds."""
    real = pub.install_set

    def install_set(files, staging):
        (staging.parent / LOCK).write_text(content, encoding='utf-8')
        return real(files, staging)
    monkeypatch.setattr(pub, 'install_set', install_set)


class TestLockRelease:
    """A lock that SlotLock fails to delete (it swallows the error) is detected
    and reported; another holder's lock is never deleted."""

    def test_normal_release_is_warning_free(self, env, outputs, tmp_path):
        report = do_publish(env, tmp_path / 'dest', outputs)
        assert report['state'] == 'committed' and report['cleanup_warnings'] == report['retained_files'] == []
        assert not (tmp_path / 'dest' / LOCK).exists()

    def test_swallowed_unlink_failure_after_success(self, env, outputs, tmp_path, lock_unlink_fails):
        dest = tmp_path / 'dest'
        report = do_publish(env, dest, outputs)
        lock_unlink_fails.undo()
        lock = rm._real(dest) / LOCK
        assert lock.exists()                                       # SlotLock really left it behind
        assert report['published'] and report['state'] == 'committed_with_cleanup_warnings'
        assert report['retained_files'] == [str(lock)]
        assert has(report['cleanup_warnings'], "this publication's lock file")
        assert has(report['cleanup_warnings'], "blocks the next publication to this destination")
        assert not has(report['cleanup_warnings'], "still running")
        assert (dest / pub.LATEST_NAME).read_bytes() == outputs['latest'].read_bytes()
        assert (dest / SNAP).read_bytes() == outputs['snapshot'].read_bytes()
        assert pub.verify_published(dest, SEASON, WEEK, env['pkgA'])['ok']
        with pytest.raises(pub.PublicationError, match="holds lock file"):       # it does block the next one
            do_publish(env, dest, outputs)
        assert lock.exists()                                       # and is never removed by us

    def test_cli_exits_3(self, env, outputs, tmp_path, lock_unlink_fails, capsys):
        code = pub.main(['publish', '--package', str(env['pkgA']), '--dest', str(tmp_path / 'dest'),
                         '--prediction-schedule', str(env['schedA']), '--latest', str(outputs['latest']),
                         '--snapshot', str(outputs['snapshot'])])
        lock_unlink_fails.undo()
        out = json.loads(capsys.readouterr().out)
        assert code == 3 and out['state'] == 'committed_with_cleanup_warnings'
        assert out['retained_files'] == [str(rm._real(tmp_path / 'dest') / LOCK)]

    def test_failed_publication_keeps_its_error_and_state(self, env, outputs, tmp_path, lock_unlink_fails):
        dest = tmp_path / 'dest'
        dest.mkdir()
        (dest / SNAP).write_bytes(b'someone else froze this week\n')
        with pytest.raises(pub.PublicationError) as exc:
            do_publish(env, dest, outputs)
        lock_unlink_fails.undo()
        err = exc.value
        assert err.state == 'not_published'
        assert err.problems[0].startswith(f"{SNAP} is already published with different content")
        assert has(err.problems, "this publication's lock file") and err.retained_files == [str(rm._real(dest) / LOCK)]
        assert (dest / SNAP).read_bytes() == b'someone else froze this week\n'

    def test_rolled_back_publication_keeps_its_state(self, env, outputs, tmp_path, lock_unlink_fails):
        dest = tmp_path / 'dest'
        dest.mkdir()
        (dest / pub.LATEST_NAME).write_bytes(b'previous latest feed\n')
        TestRollback.failing_replace(lock_unlink_fails, 2)
        with pytest.raises(pub.PublicationError) as exc:
            do_publish(env, dest, outputs)
        lock_unlink_fails.undo()
        err = exc.value
        assert err.state == 'rolled_back' and has(err.problems, "were restored")
        assert err.retained_files == [str(rm._real(dest) / LOCK)]
        assert (dest / pub.LATEST_NAME).read_bytes() == b'previous latest feed\n'
        assert sorted(p.name for p in dest.iterdir()) == sorted([pub.LATEST_NAME, LOCK])

    def test_replacement_owner_lock_is_never_deleted(self, env, outputs, tmp_path, monkeypatch):
        other = json.dumps({'what': 'prop publication', 'id': 'other-run', 'pid': 1, 'host': 'elsewhere',
                            'started_at': '2026-10-31T00:00:00Z', 'token': 'f' * 16})
        rewrite_lock_during_install(monkeypatch, other)
        dest = tmp_path / 'dest'
        report = do_publish(env, dest, outputs)
        monkeypatch.undo()
        assert (dest / LOCK).read_text(encoding='utf-8') == other            # untouched
        assert report['state'] == 'committed_with_cleanup_warnings' and report['retained_files'] == []
        assert has(report['cleanup_warnings'], "holds another token, not this publication's; it was left untouched")
        assert not has(report['cleanup_warnings'], "still running")

    @pytest.mark.parametrize("content", ["not json at all", "[1, 2]", "", '{"id": "no token"}', '{"token": 7}'])
    def test_unreadable_or_malformed_lock_is_left_and_reported(self, env, outputs, tmp_path, monkeypatch, content):
        rewrite_lock_during_install(monkeypatch, content)
        dest = tmp_path / 'dest'
        report = do_publish(env, dest, outputs)
        monkeypatch.undo()
        assert (dest / LOCK).read_text(encoding='utf-8') == content          # not deleted
        assert report['state'] == 'committed_with_cleanup_warnings'
        assert report['retained_files'] == [str(rm._real(dest) / LOCK)]
        warning = " ".join(report['cleanup_warnings'])
        assert "can't be read or parsed" in warning and "whose it is can't be determined" in warning
        assert "still running" not in warning

    def test_unreadable_lock_file(self, env, outputs, tmp_path, lock_unlink_fails):
        """Deleting fails and reading the lock back fails too: kept and reported."""
        real = Path.read_text

        def read_text(self, *a, **k):
            if self.name == LOCK:
                raise PermissionError(f"simulated: can't read {self.name}")
            return real(self, *a, **k)
        lock_unlink_fails.setattr(Path, 'read_text', read_text)
        dest = tmp_path / 'dest'
        report = do_publish(env, dest, outputs)
        lock_unlink_fails.undo()
        assert (dest / LOCK).exists()
        assert report['state'] == 'committed_with_cleanup_warnings'
        assert report['retained_files'] == [str(rm._real(dest) / LOCK)]
        assert has(report['cleanup_warnings'], "can't be read or parsed (PermissionError")


class TestGameDates:
    """Every row's game_date is its schedule game's calendar day at UTC midnight."""

    def test_recorded_form(self, env):
        assert set(frame(env['latestA'].read_bytes())['game_date']) == {'2026-09-21 00:00:00+00:00'}

    def test_wrong_date_in_both_files_preserves_published_files(self, env, outputs, tmp_path):
        """The review's reproduction: game_date 2040-01-01 in both otherwise-valid files."""
        dest = tmp_path / 'dest'
        do_publish(env, dest, outputs)
        before = tree(dest)
        bad = tampered(env, lambda df: df.__setitem__('game_date', '2040-01-01 00:00:00+00:00'))
        d = tmp_path / 'bad'
        d.mkdir()
        (d / pub.LATEST_NAME).write_bytes(bad)
        (d / SNAP).write_bytes(bad)
        with pytest.raises(pub.PublicationError) as exc:
            do_publish(env, dest, {'latest': d / pub.LATEST_NAME, 'snapshot': d / SNAP})
        for label in ('latest feed', 'frozen snapshot'):
            assert has(exc.value.problems, f"{label}: 100 row(s) whose game_date is not their game's day")
        assert has(exc.value.problems, "2040-01-01 vs the schedule's 2026-09-21")
        assert exc.value.state == 'not_published' and tree(dest) == before

    def test_one_wrong_row(self, env, ctx):
        r = check(ctx, tampered(env, lambda df: df.loc.__setitem__((4, 'game_date'), '2026-09-22 00:00:00+00:00')))
        assert not r['ok'] and has(r['problems'], "latest feed: 1 row(s) whose game_date is not their game's day: row 6")

    @pytest.mark.parametrize("value", ['2026-09-21', '2026-09-21T00:00:00Z', '2026-09-20 20:00:00-04:00',
                                       '2026-09-21 13:00:00+00:00', '2026-02-30 00:00:00+00:00', 'soon'])
    def test_malformed_date(self, env, ctx, value):
        r = check(ctx, tampered(env, lambda df: df.loc.__setitem__((0, 'game_date'), value)))
        assert not r['ok'] and has(r['problems'], "1 row(s) whose game_date isn't a calendar date at UTC midnight")

    def test_missing_value(self, env, ctx):
        r = check(ctx, tampered(env, lambda df: df.loc.__setitem__((0, 'game_date'), None)))
        assert not r['ok'] and has(r['problems'], "latest feed: 1 row(s) without a game_date: row 2")

    def test_missing_column(self, env, ctx):
        with pytest.raises(pub.PublicationError, match=r"lacks required column\(s\) \['game_date'\]"):
            check(ctx, tampered(env, lambda df: df.drop(columns='game_date', inplace=True)))

    def test_only_the_snapshot_wrong(self, env, ctx):
        snap = tampered(env, lambda df: df.__setitem__('game_date', '2040-01-01 00:00:00+00:00'))
        r = check(ctx, env['latestA'].read_bytes(), snap)
        assert not r['ok'] and has(r['problems'], "frozen snapshot: 100 row(s) whose game_date is not")
        assert not has(r['problems'], "latest feed: 100 row(s)")

def directory_alias(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if sys.platform != 'win32':
        pytest.skip("directory symlinks are not available here")
    r = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


class TestDestinationGuards:
    # Nonexistent paths only: production directories are never read by these tests.
    @pytest.mark.parametrize("where", [ROOT / 'publish_out', ROOT / 'data_files' / 'publish_test_out',
                                       ROOT / 'player_props' / 'models' / 'publish_test_out'])
    def test_production_and_repository_refused(self, env, outputs, where):
        assert not where.exists()
        with pytest.raises(pub.PublicationError, match="overlaps protected location"):
            do_publish(env, where, outputs)
        assert not where.exists()

    def test_protected_path_refused(self, env, outputs, tmp_path):
        with pytest.raises(pub.PublicationError, match="overlaps protected location"):
            do_publish(env, tmp_path / 'p' / 'dest', outputs, protected=[tmp_path / 'p'])
        assert not (tmp_path / 'p').exists()

    @pytest.mark.parametrize("where", ["package", "inside_package", "contains_package"])
    def test_package_overlap(self, env, outputs, tmp_path, where):
        pkg = tmp_path / 'holder' / 'pkg'
        shutil.copytree(env['pkgA'], pkg)
        before = tree(tmp_path / 'holder')
        dest = {'package': pkg, 'inside_package': pkg / 'out', 'contains_package': tmp_path / 'holder'}[where]
        with pytest.raises(pub.PublicationError, match="overlaps the release package"):
            pub.publish(pkg, dest, env['schedA'], latest=outputs['latest'], snapshot=outputs['snapshot'])
        assert tree(tmp_path / 'holder') == before

    def test_destination_holding_the_inputs(self, env, outputs):
        before = tree(outputs['dir'])
        with pytest.raises(pub.PublicationError, match="overlaps input latest feed"):
            do_publish(env, outputs['dir'], outputs)
        assert tree(outputs['dir']) == before

    def test_destination_inside_the_run_that_produced_the_outputs(self, env, tmp_path):
        """Publishing back over the rehearsal's own runtime/data is refused."""
        data = env['runA'] / 'runtime' / 'data'
        before = tree(data)
        with pytest.raises(pub.PublicationError, match="overlaps input"):
            pub.publish(env['pkgA'], data, env['schedA'], latest=data / pub.LATEST_NAME, snapshot=data / SNAP)
        assert tree(data) == before

    @pytest.mark.parametrize("target", ["package", "outputs"])
    def test_resolved_alias(self, env, outputs, tmp_path, target):
        pkg = tmp_path / 'pkg'
        shutil.copytree(env['pkgA'], pkg)
        alias = tmp_path / 'alias'
        directory_alias(alias, pkg if target == 'package' else outputs['dir'])
        before = (tree(pkg), tree(outputs['dir']))
        with pytest.raises(pub.PublicationError, match="overlaps"):
            pub.publish(pkg, alias, env['schedA'], latest=outputs['latest'], snapshot=outputs['snapshot'])
        assert (tree(pkg), tree(outputs['dir'])) == before

    def test_guards_run_before_anything_is_read(self, env, tmp_path):
        """An overlapping destination is refused even when every input is missing."""
        missing = tmp_path / 'missing'
        with pytest.raises(pub.PublicationError, match="overlaps input"):
            pub.publish(env['pkgA'], missing, missing / 's.csv', latest=missing / pub.LATEST_NAME,
                        snapshot=missing / SNAP)
        assert not missing.exists()


@pytest.fixture(scope="module")
def later(env):
    """A second release whose target is the next week (2026 week 4), for a
    legitimate later publication into the same destination."""
    base = env['base']
    summary = rh.run_rehearsal(rh.RehearsalConfig(workspace=base / 'wsLater', seed=7, target_week=WEEK + 1,
                                                  played_target_weeks=WEEK))
    assert summary['status'] == 'succeeded', summary['stages']
    run = base / 'wsLater' / 'runs' / summary['run_id']
    return {'pkg': package(run, base, 'pkgLater'), 'sched': run / 'runtime/data' / f'nfl_schedule_{SEASON}.csv'}


@pytest.fixture
def published(env, outputs, tmp_path):
    dest = tmp_path / 'dest'
    do_publish(env, dest, outputs)
    return dest


def side_doc(dest):
    return json.loads((dest / SIDE).read_text(encoding='utf-8'))


class TestVerifyPublished:
    """Two explicit modes: 'current' verifies the complete current set;
    'historical' verifies only the frozen snapshot and its sidecar."""

    def test_current_set_verifies(self, env, published):
        r = pub.verify_published(published, SEASON, WEEK, env['pkgA'])
        assert r['ok'] and r['mode'] == 'current' and r['not_verified'] == []
        assert has(r['verified'], "the latest feed matches the sidecar: this is the current set")

    def test_historical_mode_states_what_it_did_not_verify(self, published):
        r = pub.verify_published(published, SEASON, WEEK, mode='historical')
        assert r['ok'] and r['mode'] == 'historical'
        assert has(r['not_verified'], "the current latest feed")
        assert not has(r['verified'], "latest feed")

    def test_deleted_latest_fails_current(self, published):
        (published / pub.LATEST_NAME).unlink()
        r = pub.verify_published(published, SEASON, WEEK)
        assert not r['ok'] and has(r['problems'], "the latest feed player_props_predictions.csv is missing")
        assert pub.verify_published(published, SEASON, WEEK, mode='historical')['ok']

    def test_corrupted_latest_fails_without_a_guess(self, published):
        (published / pub.LATEST_NAME).write_bytes(b'tampered\n')
        r = pub.verify_published(published, SEASON, WEEK)
        assert not r['ok'] and has(r['problems'], "doesn't match its sidecar")
        assert has(r['notes'], "nothing here accounts for the latest feed's current state")
        assert not has(r['notes'], "another publication") and not has(r['notes'], "matches the published set")

    def test_unreadable_latest_fails(self, published):
        (published / pub.LATEST_NAME).unlink()
        (published / pub.LATEST_NAME).mkdir()                 # a directory where the file should be
        r = pub.verify_published(published, SEASON, WEEK)
        assert not r['ok'] and has(r['problems'], "the latest feed player_props_predictions.csv can't be read")

    def test_changed_snapshot_fails_both_modes(self, published):
        (published / SNAP).write_bytes((published / SNAP).read_bytes() + b'\n')
        for mode in ('current', 'historical'):
            r = pub.verify_published(published, SEASON, WEEK, mode=mode)
            assert not r['ok'] and has(r['problems'], "the frozen snapshot")

    def test_legitimate_later_publication(self, env, later, published):
        """Week 4 is published into the same destination: week 4 is the current
        set; week 3 is not, but its snapshot still verifies historically, and
        the evidence for the replaced latest feed is named."""
        pub.publish(later['pkg'], published, later['sched'], packaged=True)
        assert pub.verify_published(published, SEASON, WEEK + 1, later['pkg'])['ok']
        current = pub.verify_published(published, SEASON, WEEK)
        assert not current['ok'] and has(current['problems'], "the latest feed player_props_predictions.csv doesn't match")
        later_side = pub.sidecar_name(SEASON, WEEK + 1)
        assert has(current['notes'], f"matches the published set for 2026 week 4 ({later_side}), so 2026 week 3 is not")
        hist = pub.verify_published(published, SEASON, WEEK, env['pkgA'], mode='historical')
        assert hist['ok'] and has(hist['not_verified'], "the current latest feed")
        assert has(hist['notes'], "the current latest feed matches the published set for 2026 week 4")

    def test_evidence_requires_the_other_snapshot_to_verify(self, env, later, published):
        """A later sidecar alone isn't evidence if its own snapshot no longer matches."""
        pub.publish(later['pkg'], published, later['sched'], packaged=True)
        later_snap = published / pub.snapshot_name(SEASON, WEEK + 1)
        later_snap.write_bytes(later_snap.read_bytes() + b'\n')
        r = pub.verify_published(published, SEASON, WEEK)
        assert not r['ok'] and has(r['notes'], "nothing here accounts for the latest feed's current state")

    def test_wrong_package(self, env, published):
        assert has(pub.verify_published(published, SEASON, WEEK, env['pkgB'])['problems'], "names another release")

    def test_missing_sidecar(self, published):
        (published / SIDE).unlink()
        for mode in ('current', 'historical'):
            r = pub.verify_published(published, SEASON, WEEK, mode=mode)
            assert not r['ok'] and has(r['problems'], f"the sidecar {SIDE} is missing")

    @pytest.mark.parametrize("text, message", [
        ('{"kind": ', "is not valid JSON: line 1, column 10"),
        ('[1, 2]', "the sidecar must be a JSON object"),
        ('\xff\xfe', "can't be read (UnicodeDecodeError"),
    ])
    def test_unparseable_sidecar(self, published, text, message):
        (published / SIDE).write_bytes(text.encode('latin-1'))
        r = pub.verify_published(published, SEASON, WEEK)
        assert not r['ok'] and has(r['problems'], message), r['problems']

    @pytest.mark.parametrize("change, message", [
        (lambda s: s.update(kind='other'), "is not a version-1 player_props_publication sidecar"),
        (lambda s: s.update(sidecar_version=2), "is not a version-1"),
        (lambda s: s.update(inference_target={'season': 2026, 'week': 9}), "inference_target is"),
        (lambda s: s['release'].pop('manifest_sha256'), "release must name a release_id and manifest_sha256"),
        (lambda s: s['verification'].update(result='failed'), "does not record a passed verification"),
        (lambda s: s.pop('outputs'), "outputs must be an object"),
        (lambda s: s['outputs']['snapshot'].update(sha256='abc'), "outputs.snapshot.sha256 must be"),
        (lambda s: s['outputs']['snapshot'].update(bytes='12'), "outputs.snapshot.bytes must be"),
        (lambda s: s['outputs']['snapshot'].update(file='other.csv'), "outputs.snapshot.file must be"),
        (lambda s: s['outputs']['latest'].update(sha256=None), "outputs.latest.sha256 must be"),
    ])
    def test_malformed_sidecar_fields(self, published, change, message):
        doc = side_doc(published)
        change(doc)
        (published / SIDE).write_text(json.dumps(doc), encoding='utf-8')
        r = pub.verify_published(published, SEASON, WEEK)
        assert not r['ok'] and has(r['problems'], message), r['problems']

    def test_historical_mode_does_not_rely_on_the_latest_entry(self, published):
        doc = side_doc(published)
        doc['outputs']['latest'] = None
        (published / SIDE).write_text(json.dumps(doc), encoding='utf-8')
        assert not pub.verify_published(published, SEASON, WEEK)['ok']
        assert pub.verify_published(published, SEASON, WEEK, mode='historical')['ok']

    def test_unknown_mode(self, published):
        with pytest.raises(ValueError, match="mode must be"):
            pub.verify_published(published, SEASON, WEEK, mode='latest')

    def test_cli_modes(self, published):
        def cli(*extra):
            r = subprocess.run([sys.executable, 'player_props/publication.py', 'verify-published', '--dest',
                                str(published), '--season', str(SEASON), '--week', str(WEEK), *extra],
                               cwd=ROOT, capture_output=True, text=True, encoding='utf-8', env=cli_env('utf-8'))
            return r.returncode, json.loads(r.stdout)
        (published / pub.LATEST_NAME).unlink()
        code, out = cli()
        assert code == 1 and out['mode'] == 'current' and not out['ok']
        code, out = cli('--historical')
        assert code == 0 and out['mode'] == 'historical' and out['not_verified']


def cli_env(encoding):
    """The caller's environment without any UTF-8 forcing, plus the given
    stdout encoding (None: Python's default for a pipe, e.g. cp1252 on Windows)."""
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONUTF8', 'PYTHONIOENCODING')}
    env['PYTHONUTF8'] = '0'
    if encoding:
        env['PYTHONIOENCODING'] = encoding
    return env


class TestCliEncoding:
    """The CLI must work, and print parseable JSON, whatever the console encoding."""

    def run(self, env, outputs, dest, encoding, snapshot=None):
        r = subprocess.run([sys.executable, 'player_props/publication.py', 'publish', '--package', str(env['pkgA']),
                            '--dest', str(dest), '--prediction-schedule', str(env['schedA']),
                            '--latest', str(outputs['latest']), '--snapshot', str(snapshot or outputs['snapshot'])],
                           cwd=ROOT, capture_output=True, env=cli_env(encoding))
        text = r.stdout.decode('ascii')                  # the JSON is ASCII-only
        return r.returncode, json.loads(text), r.stderr

    @pytest.mark.parametrize("encoding", ['cp1252', 'utf-8', None], ids=['cp1252', 'utf-8', 'default'])
    def test_publish(self, env, outputs, tmp_path, encoding):
        """The review's failure: '✅' printed by a reused helper on a cp1252 console."""
        code, out, stderr = self.run(env, outputs, tmp_path / 'dest', encoding)
        assert code == 0, (out, stderr)
        assert out['published'] and out['state'] == 'committed'
        assert (tmp_path / 'dest' / SNAP).read_bytes() == outputs['snapshot'].read_bytes()
        assert b'Traceback' not in stderr

    def test_refusal_on_cp1252_keeps_useful_diagnostics(self, env, outputs, tmp_path):
        bad = tmp_path / 'bad'
        bad.mkdir()
        (bad / SNAP).write_bytes(tampered(env, lambda df: df.__setitem__('game_date', '2040-01-01 00:00:00+00:00')))
        code, out, stderr = self.run(env, outputs, tmp_path / 'dest', 'cp1252', snapshot=bad / SNAP)
        assert code == 1 and out['published'] is False and out['state'] == 'not_published'
        assert has(out['problems'], "frozen snapshot: 100 row(s) whose game_date is not their game's day")
        assert isinstance(out['library_output_tail'], list) and out['library_output_tail']
        assert not (tmp_path / 'dest').exists() or not any((tmp_path / 'dest').iterdir())
