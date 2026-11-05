"""Release storage (offline slice): deterministic archive of a verified S1
package plus its prediction schedule, a strict storage descriptor, and a
verified restore into an isolated directory. Synthetic rehearsal runs only;
everything is written under tmp_path."""
import copy
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from player_props import publication as pub
from player_props import rehearsal as rh
from player_props import release_manifest as rm
from player_props import release_storage as rs

ROOT = Path(__file__).resolve().parents[1]
SEASON = 2026


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): sha(p.read_bytes()) for p in sorted(d.rglob('*')) if p.is_file()} \
        if d.exists() else None


# Tests that aim restores or packs at repository/production paths use names
# containing PROBE. If a guard were broken, such a test could leave a real
# directory there, and a fixed leftover would then fail later, unrelated runs.
PROBE = 'rs-probe'
PROBE_PARENTS = [ROOT, ROOT / 'data_files', ROOT / 'player_props' / 'models']


def _probes() -> list:
    return [q for d in PROBE_PARENTS if d.is_dir() for q in d.iterdir() if PROBE in q.name]


@pytest.fixture(autouse=True)
def no_probe_leaks():
    stale = _probes()
    assert not stale, f"leftover probe paths from an earlier run; remove them first: {stale}"
    yield
    leaked = _probes()
    for q in leaked:                                    # only names carrying PROBE are ever removed
        shutil.rmtree(q) if q.is_dir() and not q.is_symlink() else q.unlink()
    assert not leaked, f"a refused operation wrote into a protected location: {leaked}"


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
    base = tmp_path_factory.mktemp("storage")
    a, b = run_dir(base / "wsA", 7), run_dir(base / "wsB", 11)
    e = {'base': base, 'pkgA': package(a, base, 'pkgA'), 'pkgB': package(b, base, 'pkgB'),
         'schedA': a / 'runtime' / 'data' / f'nfl_schedule_{SEASON}.csv',
         'schedB': b / 'runtime' / 'data' / f'nfl_schedule_{SEASON}.csv'}
    # Both runs share one synthetic schedule, so the substitute is schedule A
    # with one venue changed: a real schedule-like file with other bytes.
    sched = e['schedA'].read_bytes()
    assert sched == e['schedB'].read_bytes() and b'Synthetic Field' in sched
    e['schedX'] = base / f'nfl_schedule_{SEASON}.csv'
    e['schedX'].write_bytes(sched.replace(b'Synthetic Field', b'Synthetic Park', 1))
    e['pkgA_tree'] = tree(e['pkgA'])
    e['packed'] = rs.pack(e['pkgA'], e['schedA'], base / 'packedA')
    e['archive'] = Path(e['packed']['archive']['path'])
    e['descriptor_path'] = Path(e['packed']['descriptor']['path'])
    e['descriptor'] = json.loads(e['descriptor_path'].read_text(encoding='utf-8'))
    return e


def write_json(path: Path, doc) -> Path:
    path.write_text(json.dumps(doc, indent=2), encoding='utf-8')
    return path


def entries_of(archive: Path) -> dict:
    with tarfile.open(archive, 'r:') as tar:
        return {m.name: tar.extractfile(m).read() for m in tar}


def craft(env, tmp: Path, files=None, extra=(), consistent=False, name='crafted'):
    """An archive from ``files`` (default: the real entries) plus ``extra``
    raw members [(TarInfo, bytes|None)], and a descriptor whose archive hash
    and size match it. With ``consistent``, the descriptor's entries (and
    manifest and schedule hashes) are recomputed from ``files`` too."""
    files = dict(entries_of(env['archive']) if files is None else files)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w', format=tarfile.GNU_FORMAT) as tar:
        for n in sorted(files):
            info = tarfile.TarInfo(n)
            info.size = len(files[n])
            tar.addfile(info, io.BytesIO(files[n]))
        for info, data in extra:
            tar.addfile(info, None if data is None else io.BytesIO(data))
    data = buf.getvalue()
    archive = tmp / f'{name}.tar'
    archive.write_bytes(data)
    doc = copy.deepcopy(env['descriptor'])
    doc['archive'].update(sha256=sha(data), bytes=len(data))
    if consistent:
        doc['entries'] = [{'name': n, 'sha256': sha(files[n]), 'bytes': len(files[n])} for n in sorted(files)]
        doc['manifest_sha256'] = sha(files[f'package/{rm.MANIFEST_NAME}'])
        s = files[doc['prediction_schedule']['entry']]
        doc['prediction_schedule'].update(sha256=sha(s), bytes=len(s))
    return archive, write_json(tmp / f'{name}.storage.json', doc), doc


# Descriptor fields that must be strings because they are used as lookup keys,
# compared with paths or formatted into them.
STRING_FIELDS = [('kind',), ('status',), ('statement',), ('release_id',), ('archive', 'file'),
                 ('archive', 'format'), ('entries', 0, 'name'), ('prediction_schedule', 'entry'),
                 ('prediction_schedule', 'summary_artifact')]
BAD_VALUES = [[], {}, None, 7, 1.5, True]


def type_name(v) -> str:
    return 'null' if v is None else type(v).__name__


def field_name(path) -> str:
    return ''.join(f'[{p}]' if isinstance(p, int) else (f'.{p}' if i else p) for i, p in enumerate(path))


def set_field(doc, path, value):
    o = doc
    for k in path[:-1]:
        o = o[k]
    o[path[-1]] = value
    return doc


def member(name, kind=tarfile.REGTYPE, data=b'x', link=''):
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = link
    info.size = len(data) if kind in (tarfile.REGTYPE, tarfile.AREGTYPE) else 0
    return info, (data if kind in (tarfile.REGTYPE, tarfile.AREGTYPE) else None)


def refused(fn, *a, **k) -> rs.RestoreError:
    with pytest.raises(rs.RestoreError) as e:
        fn(*a, **k)
    return e.value


def has(items, text) -> bool:
    return any(text in i for i in items)


def leftovers(parent: Path, name: str) -> list:
    return sorted(p.name for p in parent.iterdir() if p.name.startswith(f'.{name}.'))


# ---------------------------------------------------------------------------

class TestPack:
    def test_round_trip_restores_identical_bytes(self, env, tmp_path):
        dest = tmp_path / 'restored'
        report = rs.restore(env['archive'], env['descriptor_path'], dest)
        assert report['state'] == 'restored' and report['previous'] == 'none'
        assert report['approval'] == 'not_established' and report['s2_publication'] == 'not run'
        assert tree(dest / 'package') == env['pkgA_tree']
        assert (dest / rs.schedule_entry(SEASON)).read_bytes() == env['schedA'].read_bytes()
        assert sorted(p.name for p in dest.iterdir()) == ['package', 'restore_report.json', 'schedule']
        rep = json.loads((dest / rs.REPORT_NAME).read_text(encoding='utf-8'))
        assert rep['descriptor_sha256'] == sha(env['descriptor_path'].read_bytes())
        assert rep['manifest_sha256'] == sha((env['pkgA'] / rm.MANIFEST_NAME).read_bytes())
        assert has(rep['not_established'], 'approval') and has(rep['not_established'], 'S2 publication')
        assert leftovers(tmp_path, 'restored') == [] and not list(tmp_path.glob('.*.lock'))

    def test_archive_is_deterministic_and_package_unchanged(self, env, tmp_path):
        again = rs.pack(env['pkgA'], env['schedA'], tmp_path / 'again')
        assert Path(again['archive']['path']).read_bytes() == env['archive'].read_bytes()
        assert Path(again['descriptor']['path']).read_bytes() == env['descriptor_path'].read_bytes()
        assert tree(env['pkgA']) == env['pkgA_tree']
        assert sorted(p.name for p in (tmp_path / 'again').iterdir()) == sorted(
            [rs.archive_name(again['release_id']), rs.descriptor_name(again['release_id'])])
        with tarfile.open(env['archive'], 'r:') as tar:
            ms = tar.getmembers()
        assert [m.name for m in ms] == sorted(m.name for m in ms)
        assert all(m.isreg() and m.mtime == 0 and m.uid == 0 and m.mode == 0o644 for m in ms)

    def test_descriptor_records_identity_and_every_entry(self, env):
        d = env['descriptor']
        man = (env['pkgA'] / rm.MANIFEST_NAME).read_bytes()
        assert d['release_id'] == json.loads(man)['release_id'] and d['manifest_sha256'] == sha(man)
        assert d['archive']['sha256'] == sha(env['archive'].read_bytes())
        assert {e['name'] for e in d['entries']} == {f'package/{k}' for k in env['pkgA_tree']} | {
            rs.schedule_entry(SEASON)}
        assert d['prediction_schedule']['sha256'] == sha(env['schedA'].read_bytes())
        # No self-reference: the descriptor never records its own hash, and the
        # archive doesn't contain the descriptor.
        assert set(d) == rs.DESCRIPTOR_KEYS and 'descriptor_sha256' not in json.dumps(d)
        assert not any(n.endswith('.storage.json') for n in entries_of(env['archive']))

    def test_restored_package_is_usable_by_s2_only_when_asked(self, env, tmp_path):
        dest = tmp_path / 'restored'
        rs.restore(env['archive'], env['descriptor_path'], dest)
        assert not (tmp_path / 'published').exists()              # restoring published nothing
        out = pub.publish(dest / 'package', tmp_path / 'published', dest / rs.schedule_entry(SEASON), packaged=True)
        assert out['state'] == 'committed'

    def test_substituted_schedule_refused(self, env, tmp_path):
        e = refused(rs.pack, env['pkgA'], env['schedX'], tmp_path / 'out')
        assert has(e.problems, 'is not the one the package')
        assert not (tmp_path / 'out').exists() and tree(env['pkgA']) == env['pkgA_tree']

    def test_missing_schedule_refused(self, env, tmp_path):
        e = refused(rs.pack, env['pkgA'], tmp_path / 'nope.csv', tmp_path / 'out')
        assert has(e.problems, "can't be read") and not (tmp_path / 'out').exists()

    def test_unverifiable_package_refused(self, env, tmp_path):
        bad = tmp_path / 'pkg'
        shutil.copytree(env['pkgA'], bad)
        p = bad / 'evidence' / 'predictions_snapshot.csv'
        p.write_bytes(p.read_bytes() + b'\n')
        e = refused(rs.pack, bad, env['schedA'], tmp_path / 'out')
        assert has(e.problems, 'does not verify') and not (tmp_path / 'out').exists()

    @pytest.mark.parametrize('where', ['existing', 'inside_package', 'repo', 'data_files'])
    def test_pack_output_location_refused(self, env, tmp_path, where):
        out = {'existing': tmp_path, 'inside_package': env['pkgA'] / 'x',
               'repo': ROOT / 'rs-probe-out', 'data_files': ROOT / 'data_files' / 'rs-probe-x'}[where]
        refused(rs.pack, env['pkgA'], env['schedA'], out)
        assert tree(env['pkgA']) == env['pkgA_tree']
        assert not (ROOT / 'rs-probe-out').exists()


class TestDescriptor:
    def test_valid_descriptor_is_not_an_approval(self, env):
        v = rs.validate_descriptor(env['descriptor'])
        assert v['ok'] and v['approval'] == 'not_established' and 'does not approve' in v['note']
        assert env['descriptor']['status'] == 'candidate_storage_only'

    @pytest.mark.parametrize('field', ['approved', 'active', 'sequence', 'previous', 'approved_by'])
    def test_approval_or_chain_fields_refused(self, env, field):
        doc = copy.deepcopy(env['descriptor'])
        doc[field] = True
        assert has(rs.descriptor_problems(doc), 'unexpected')

    @pytest.mark.parametrize('change, text', [
        (lambda d: d.update(status='approved'), 'never records an approval'),
        (lambda d: d.update(statement='approved'), 'never records an approval'),
        (lambda d: d.update(manifest_sha256='0' * 64), 'manifest_sha256 is not the hash'),
        (lambda d: d.update(manifest_sha256='ABC'), '64-character'),
        (lambda d: d.update(release_id='latest'), 'release_id must look like'),
        (lambda d: d.update(release_id='20261030T151244Z-00000000'), 'archive.file must be'),
        (lambda d: d.update(descriptor_version=True), 'descriptor_version must be an integer, not bool'),
        (lambda d: d.update(descriptor_version=2), 'not a version-1'),
        (lambda d: d.update(kind='other'), 'not a version-1'),
        (lambda d: d['archive'].update(bytes=-1), 'non-negative integer'),
        (lambda d: d['archive'].update(bytes=1.5), 'non-negative integer'),
        (lambda d: d['archive'].update(format='tar.gz'), 'archive.format'),
        (lambda d: d['archive'].update(extra=1), 'exactly the fields'),
        (lambda d: d['entries'].reverse(), 'sorted'),
        (lambda d: d['entries'].append(dict(d['entries'][-1])), 'duplicate entry'),
        (lambda d: d['entries'][0].update(name='package/../x'), "'..'"),
        (lambda d: d['entries'][0].update(sha256='z' * 64), '64-character'),
        (lambda d: d.update(entries=[]), 'non-empty'),
        (lambda d: d['inference_target'].update(week=0), 'inference_target'),
        (lambda d: d['prediction_schedule'].update(sha256='0' * 64), "does not match its entry"),
        (lambda d: d['prediction_schedule'].update(entry='schedule/nfl_schedule_2025.csv'), 'prediction_schedule.entry'),
        (lambda d: d['prediction_schedule'].update(summary_artifact='x'), 'summary_artifact'),
        (lambda d: d.pop('prediction_schedule'), 'missing'),
    ])
    def test_malformed_descriptor_refused(self, env, change, text):
        doc = copy.deepcopy(env['descriptor'])
        change(doc)
        problems = rs.descriptor_problems(doc)
        assert has(problems, text), problems
        assert not rs.validate_descriptor(doc)['ok']

    @pytest.mark.parametrize('path', STRING_FIELDS, ids=lambda p: field_name(p))
    @pytest.mark.parametrize('value', BAD_VALUES, ids=lambda v: type_name(v))
    def test_wrong_type_for_string_field_is_a_diagnostic(self, env, path, value):
        """Wrong JSON types where a string is used as a key, compared with a
        path or formatted into one: a field-specific problem, no exception,
        and the document is not repaired or normalised."""
        doc = set_field(copy.deepcopy(env['descriptor']), path, value)
        snapshot = copy.deepcopy(doc)
        problems = rs.descriptor_problems(doc)
        assert has(problems, f"{field_name(path)} must be a string, not {type_name(value)}"), problems
        v = rs.validate_descriptor(doc)
        assert not v['ok'] and v['approval'] == 'not_established'
        assert doc == snapshot

    @pytest.mark.parametrize('value', [v for v in BAD_VALUES if type(v) is not int], ids=lambda v: type_name(v))
    def test_wrong_type_for_descriptor_version(self, env, value):
        doc = set_field(copy.deepcopy(env['descriptor']), ('descriptor_version',), value)
        assert has(rs.descriptor_problems(doc), f"descriptor_version must be an integer, not {type_name(value)}")
        doc['descriptor_version'] = 7                      # right type, wrong version
        assert has(rs.descriptor_problems(doc), 'not a version-1')

    @pytest.mark.parametrize('value', BAD_VALUES, ids=lambda v: type_name(v))
    def test_wrong_type_reaches_restore_error(self, env, tmp_path, value):
        """The reviewer's case (prediction_schedule.entry = [] or {}) and the
        other bad types: load_descriptor and restore raise RestoreError with
        the field named, before any destination, lock or staging exists."""
        desc = write_json(tmp_path / 'bad.storage.json',
                          set_field(copy.deepcopy(env['descriptor']), ('prediction_schedule', 'entry'), value))
        before = desc.read_bytes()
        text = f"prediction_schedule.entry must be a string, not {type_name(value)}"
        assert has(refused(rs.load_descriptor, desc).problems, text)
        e = refused(rs.restore, env['archive'], desc, tmp_path / 'dest')
        assert e.state == 'not_restored' and has(e.problems, text)
        assert sorted(p.name for p in tmp_path.iterdir()) == ['bad.storage.json'] and desc.read_bytes() == before

    def test_package_entries_come_from_s1_roles(self):
        assert rs.package_entries() == {f'package/{p}' for p in rm.ROLES.values()} | {f'package/{rm.MANIFEST_NAME}'}
        assert len(rs.package_entries()) == len(rm.ROLES) + 1

    def test_descriptor_must_list_exactly_the_s1_package_paths(self, env):
        doc = copy.deepcopy(env['descriptor'])
        doc['entries'].append({'name': 'package/extra.txt', 'sha256': '0' * 64, 'bytes': 1})
        doc['entries'].sort(key=lambda e: e['name'])
        assert has(rs.descriptor_problems(doc), "not S1 package paths: ['package/extra.txt']")
        doc = copy.deepcopy(env['descriptor'])
        doc['entries'] = [e for e in doc['entries'] if e['name'] != 'package/evidence/summary.json']
        assert has(rs.descriptor_problems(doc), "S1 package paths missing from entries: ['package/evidence/summary.json']")

    def test_extra_schedule_entry_refused(self, env):
        doc = copy.deepcopy(env['descriptor'])
        doc['entries'].append({'name': 'schedule/zz.csv', 'sha256': '0' * 64, 'bytes': 1})
        assert has(rs.descriptor_problems(doc), 'must hold exactly')

    def test_limits(self, env, monkeypatch):
        monkeypatch.setattr(rs, 'MAX_ENTRIES', 3)
        assert has(rs.descriptor_problems(env['descriptor']), 'at most 3')
        monkeypatch.setattr(rs, 'MAX_ENTRIES', 64)
        monkeypatch.setattr(rs, 'MAX_ENTRY_BYTES', 10)
        assert has(rs.descriptor_problems(env['descriptor']), 'entry limit')

    def test_duplicate_json_keys_and_bad_json_refused(self, env, tmp_path):
        text = env['descriptor_path'].read_text(encoding='utf-8')
        dup = tmp_path / 'dup.json'
        dup.write_text(text.replace('{\n', '{\n  "status": "candidate_storage_only",\n', 1), encoding='utf-8')
        assert has(refused(rs.load_descriptor, dup).problems, 'duplicate field')
        bad = tmp_path / 'bad.json'
        bad.write_bytes(b'\xff{')
        assert has(refused(rs.load_descriptor, bad).problems, 'not valid UTF-8 JSON')

    def test_validation_has_no_approval_side_effects(self, env, tmp_path):
        before = tree(env['base'] / 'packedA')
        rs.validate_descriptor(env['descriptor'])
        rs.inspect_archive(env['archive'], env['descriptor'])
        assert tree(env['base'] / 'packedA') == before
        assert not list(ROOT.glob('player_props/releases'))


class TestInspect:
    def test_real_archive_inspects(self, env):
        found = rs.inspect_archive(env['archive'], env['descriptor'])
        assert len(found['entries']) == len(env['descriptor']['entries'])

    def test_wrong_archive_hash_or_size(self, env, tmp_path):
        doc = copy.deepcopy(env['descriptor'])
        doc['archive']['sha256'] = '0' * 64
        assert has(refused(rs.inspect_archive, env['archive'], doc).problems, "doesn't match the descriptor")
        doc = copy.deepcopy(env['descriptor'])
        doc['archive']['bytes'] += 1
        assert has(refused(rs.inspect_archive, env['archive'], doc).problems, "doesn't match the descriptor")

    @pytest.mark.parametrize('name, text', [
        ('package/../evil', "'..'"),
        ('/abs/evil', 'absolute'),
        ('package\\evil', 'backslash'),
        ('package/C:evil', 'drive'),
        ('package/CON.csv', 'reserved'),
        ('package/evil.', 'alias'),
        ('package/café', 'ASCII'),
        ('package/a b', 'characters'),
        ('other/evil', 'outside'),
        ('package', 'outside'),
    ])
    def test_unsafe_names_refused(self, env, tmp_path, name, text):
        archive, _, doc = craft(env, tmp_path, extra=[member(name)])
        assert has(refused(rs.inspect_archive, archive, doc).problems, text)

    @pytest.mark.parametrize('kind, text', [
        (tarfile.SYMTYPE, 'not a regular file'), (tarfile.LNKTYPE, 'not a regular file'),
        (tarfile.DIRTYPE, ''),            # tarfile writes 'package/link/': refused by name or type
        (tarfile.FIFOTYPE, 'not a regular file'),
        (tarfile.CHRTYPE, 'not a regular file'),
    ])
    def test_non_regular_entries_refused(self, env, tmp_path, kind, text):
        archive, _, doc = craft(env, tmp_path, extra=[member('package/link', kind, link='../../evil')])
        assert has(refused(rs.inspect_archive, archive, doc).problems, text)

    def test_link_replacing_a_real_entry_refused(self, env, tmp_path):
        files = entries_of(env['archive'])
        target = 'package/evidence/summary.json'
        files.pop(target)
        archive, _, doc = craft(env, tmp_path, files=files, extra=[member(target, tarfile.SYMTYPE, link='/etc/passwd')])
        assert has(refused(rs.inspect_archive, archive, doc).problems, 'not a regular file')

    def test_case_collision_and_file_dir_clash(self, env, tmp_path):
        archive, _, doc = craft(env, tmp_path, extra=[member('package/EVIDENCE/summary.json')])
        assert has(refused(rs.inspect_archive, archive, doc).problems, 'case-insensitive')
        archive, _, doc = craft(env, tmp_path, extra=[member('package/evidence/summary.json/x')], name='clash')
        assert has(refused(rs.inspect_archive, archive, doc).problems, 'also a directory')

    def test_duplicate_unexpected_and_missing(self, env, tmp_path):
        files = entries_of(env['archive'])
        archive, _, doc = craft(env, tmp_path, extra=[member('package/evidence/summary.json', data=b'{}')])
        assert has(refused(rs.inspect_archive, archive, doc).problems, 'duplicate entry')
        archive, _, doc = craft(env, tmp_path, extra=[member('package/extra.csv')], name='extra')
        assert has(refused(rs.inspect_archive, archive, doc).problems, 'unexpected entry')
        files.pop('package/stats/inference/player_passing_stats.csv')
        archive, _, doc = craft(env, tmp_path, files=files, name='missing')
        assert has(refused(rs.inspect_archive, archive, doc).problems, 'missing from the archive')

    def test_changed_entry_refused(self, env, tmp_path):
        files = entries_of(env['archive'])
        files['package/evidence/summary.json'] += b' '
        archive, _, doc = craft(env, tmp_path, files=files)
        assert has(refused(rs.inspect_archive, archive, doc).problems, "doesn't match the descriptor")

    def test_size_and_count_limits(self, env, tmp_path, monkeypatch):
        monkeypatch.setattr(rs, 'MAX_ARCHIVE_BYTES', 1000)
        assert has(refused(rs.inspect_archive, env['archive'], env['descriptor']).problems, 'exceeds')
        monkeypatch.undo()
        archive, _, doc = craft(env, tmp_path, extra=[member(f'package/x{i}.csv') for i in range(70)])
        assert has(refused(rs.inspect_archive, archive, doc).problems, 'more than 64 entries')
        biggest = max(e['bytes'] for e in env['descriptor']['entries'])
        archive, _, doc = craft(env, tmp_path, extra=[member('package/big.csv', data=b'0' * (biggest + 1))],
                                name='big')
        monkeypatch.setattr(rs, 'MAX_ENTRY_BYTES', biggest)          # the descriptor's own entries still fit
        assert rs.descriptor_problems(doc) == []
        assert has(refused(rs.inspect_archive, archive, doc).problems, f'larger than {biggest}')

    def test_compressed_or_truncated_archive_refused(self, env, tmp_path):
        data = gzip.compress(env['archive'].read_bytes(), mtime=0)
        gz = tmp_path / 'a.tar'
        gz.write_bytes(data)
        doc = copy.deepcopy(env['descriptor'])
        doc['archive'].update(sha256=sha(data), bytes=len(data))
        assert has(refused(rs.inspect_archive, gz, doc).problems, 'uncompressed tar')
        cut = env['archive'].read_bytes()[:len(env['archive'].read_bytes()) // 2]
        tr = tmp_path / 'cut.tar'
        tr.write_bytes(cut)
        doc['archive'].update(sha256=sha(cut), bytes=len(cut))
        refused(rs.inspect_archive, tr, doc)


def _junction(link: Path, target: Path) -> bool:
    try:
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
        return True
    except (ImportError, OSError):
        pass
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        return False


class TestRestoreLocation:
    @pytest.mark.parametrize('where', ['repo', 'data_files', 'models', 'archive_dir', 'archive_file'])
    def test_overlap_refused(self, env, where):
        dest = {'repo': ROOT / 'rs-probe-restore', 'data_files': ROOT / 'data_files' / 'rs-probe-restore',
                'models': ROOT / 'player_props' / 'models' / 'rs-probe-restore',
                'archive_dir': env['archive'].parent, 'archive_file': env['archive']}[where]
        before = tree(env['archive'].parent)
        assert has(refused(rs.restore, env['archive'], env['descriptor_path'], dest).problems, 'overlaps')
        assert tree(env['archive'].parent) == before and not (ROOT / 'rs-probe-restore').exists()

    def test_link_or_junction_into_repo_refused(self, env, tmp_path):
        link = tmp_path / 'link'
        if not _junction(link, ROOT / 'data_files'):
            pytest.skip("neither junctions nor symlinks can be created here")
        e = refused(rs.restore, env['archive'], env['descriptor_path'], link / 'rs-probe-restored')
        assert has(e.problems, 'overlaps protected location')
        assert not (ROOT / 'data_files' / 'rs-probe-restored').exists()

    def test_link_to_inputs_refused(self, env, tmp_path):
        link = tmp_path / 'link'
        if not _junction(link, env['archive'].parent):
            pytest.skip("neither junctions nor symlinks can be created here")
        assert has(refused(rs.restore, env['archive'], env['descriptor_path'], link).problems, 'overlaps the')

    def test_existing_non_restore_destination_untouched(self, env, tmp_path):
        dest = tmp_path / 'dest'
        dest.mkdir()
        (dest / 'keep.txt').write_text('mine')
        assert has(refused(rs.restore, env['archive'], env['descriptor_path'], dest).problems, 'not a previous restore')
        assert tree(dest) == {'keep.txt': sha(b'mine')}

    def test_missing_parent_refused(self, env, tmp_path):
        assert has(refused(rs.restore, env['archive'], env['descriptor_path'], tmp_path / 'a' / 'b').problems,
                   'does not exist')

    def test_lock_held_refused(self, env, tmp_path):
        from ontario_spreads import SlotLock
        with SlotLock(tmp_path, rs.LOCK_PREFIX + 'dest', 'other', what='release restore'):
            refused(rs.restore, env['archive'], env['descriptor_path'], tmp_path / 'dest')
        assert not (tmp_path / 'dest').exists()


class TestRestoreFailures:
    @pytest.fixture
    def previous(self, env, tmp_path):
        dest = tmp_path / 'dest'
        rs.restore(env['archive'], env['descriptor_path'], dest)
        (dest / rs.REPORT_NAME).write_text(json.dumps({'kind': rs.REPORT_KIND, 'marker': 'previous'}),
                                           encoding='utf-8')
        return dest, tree(dest)

    def test_replacing_a_previous_restore(self, env, previous):
        dest, _ = previous
        report = rs.restore(env['archive'], env['descriptor_path'], dest)
        assert report['previous'] == 'replaced' and report['state'] == 'restored'
        assert json.loads((dest / rs.REPORT_NAME).read_text(encoding='utf-8'))['kind'] == rs.REPORT_KIND
        assert 'marker' not in (dest / rs.REPORT_NAME).read_text(encoding='utf-8')
        assert leftovers(dest.parent, 'dest') == []

    @pytest.mark.parametrize('stage', ['inspected', 'extracted', 'verified'])
    def test_failure_before_install_keeps_previous(self, env, previous, monkeypatch, stage):
        dest, before = previous

        def boom(s):
            if s == stage:
                raise OSError(f"injected at {s}")
        monkeypatch.setattr(rs, '_hook', boom)
        e = refused(rs.restore, env['archive'], env['descriptor_path'], dest)
        assert e.state == 'not_restored' and has(e.problems, 'injected') and e.retained_files == []
        assert tree(dest) == before and leftovers(dest.parent, 'dest') == []

    def test_failure_after_move_aside_rolls_back(self, env, previous, monkeypatch):
        dest, before = previous
        monkeypatch.setattr(rs, '_hook', lambda s: (_ for _ in ()).throw(OSError('injected')) if s == 'moved_aside'
                            else None)
        e = refused(rs.restore, env['archive'], env['descriptor_path'], dest)
        assert e.state == 'rolled_back' and e.recovery_files == []
        assert tree(dest) == before and leftovers(dest.parent, 'dest') == []

    def test_rollback_failure_reports_recovery_copy(self, env, previous, monkeypatch):
        dest, before = previous
        real_replace = os.replace

        def replace(a, b):
            if os.path.realpath(b) == os.path.realpath(dest):   # installing and moving back both fail
                raise OSError('injected replace failure')
            return real_replace(a, b)
        monkeypatch.setattr(rs.os, 'replace', replace)
        e = refused(rs.restore, env['archive'], env['descriptor_path'], dest)
        monkeypatch.undo()
        assert e.state == 'rollback_failed' and len(e.recovery_files) == 1
        kept = Path(e.recovery_files[0])
        assert not dest.exists() and tree(kept) == before
        assert has(e.problems, 'it is kept at')

    def test_cleanup_failure_after_commit_is_a_warning(self, env, previous, monkeypatch):
        dest, _ = previous
        real = rs._remove
        monkeypatch.setattr(rs, '_remove', lambda p: 'injected' if '.previous-' in str(p) else real(p))
        report = rs.restore(env['archive'], env['descriptor_path'], dest)
        assert report['state'] == 'restored_with_cleanup_warnings' and len(report['retained_files']) == 1
        assert Path(report['retained_files'][0]).exists()
        assert tree(dest / 'package') == env['pkgA_tree']

    def test_staging_left_behind_is_reported(self, env, previous, monkeypatch):
        dest, before = previous
        monkeypatch.setattr(rs, '_hook', lambda s: (_ for _ in ()).throw(OSError('x')) if s == 'extracted' else None)
        monkeypatch.setattr(rs, '_remove', lambda p: 'injected')
        e = refused(rs.restore, env['archive'], env['descriptor_path'], dest)
        assert e.state == 'not_restored' and len(e.retained_files) == 1 and '.restore-' in e.retained_files[0]
        assert tree(dest) == before

    def test_corrupted_write_is_caught_by_the_reread(self, env, previous, monkeypatch):
        """Bytes hashed while streaming are correct, but the file on disk is
        not: only re-reading what was written catches it."""
        dest, before = previous
        real_fdopen = os.fdopen

        class Flip:
            def __init__(self, f):
                self.f, self.first = f, True

            def write(self, block):
                if self.first and block:
                    block, self.first = bytes([block[0] ^ 1]) + block[1:], False
                return self.f.write(block)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return self.f.__exit__(*a)
        # Only the extractor's binary writes; the lock's own text writes are untouched.
        monkeypatch.setattr(rs.os, 'fdopen', lambda fd, mode='r', *a, **k: Flip(real_fdopen(fd, mode))
                            if mode == 'wb' else real_fdopen(fd, mode, *a, **k))
        e = refused(rs.restore, env['archive'], env['descriptor_path'], dest)
        monkeypatch.undo()
        assert has(e.problems, 'on re-read') and tree(dest) == before and leftovers(dest.parent, 'dest') == []

    def test_tampered_archive_keeps_previous(self, env, previous, tmp_path):
        dest, before = previous
        data = bytearray(env['archive'].read_bytes())
        data[2048] ^= 1
        bad = tmp_path / 'bad.tar'
        bad.write_bytes(bytes(data))
        e = refused(rs.restore, bad, env['descriptor_path'], dest)
        assert has(e.problems, "doesn't match the descriptor") and tree(dest) == before

    def test_consistent_descriptor_but_invalid_package_fails_s1(self, env, previous, tmp_path):
        """The descriptor matches the archive, but the package inside doesn't
        match its own manifest: S1 verification after extraction refuses."""
        dest, before = previous
        files = entries_of(env['archive'])
        files['package/evidence/predictions_snapshot.csv'] += b'\n'
        archive, desc, _ = craft(env, tmp_path, files=files, consistent=True)
        e = refused(rs.restore, archive, desc, dest)
        assert has(e.problems, 'restored package does not verify') and tree(dest) == before
        assert leftovers(dest.parent, 'dest') == []

    def test_consistent_descriptor_but_substituted_schedule_fails(self, env, previous, tmp_path):
        dest, before = previous
        files = entries_of(env['archive'])
        files[rs.schedule_entry(SEASON)] = env['schedX'].read_bytes()
        archive, desc, _ = craft(env, tmp_path, files=files, consistent=True)
        e = refused(rs.restore, archive, desc, dest)
        assert has(e.problems, 'prediction schedule') and tree(dest) == before

    @pytest.mark.parametrize('change', ['extra', 'missing'])
    def test_package_file_set_refused_before_staging(self, env, previous, tmp_path, monkeypatch, change):
        """An archive and descriptor that agree with each other (hashes and
        sizes recomputed) but hold an extra package file, or lack a required
        one: refused before the archive is opened, any lock or staging
        directory is created, or any file is written."""
        dest, before = previous
        files = entries_of(env['archive'])
        if change == 'extra':
            files['package/extra.txt'] = b'not an S1 package file'
            text = "not S1 package paths: ['package/extra.txt']"
        else:
            files.pop('package/evidence/summary.json')
            text = "S1 package paths missing from entries: ['package/evidence/summary.json']"
        inputs = tmp_path / 'inputs'
        inputs.mkdir()
        archive, desc, doc = craft(env, inputs, files=files, consistent=True)
        assert entries_of(archive) == files                       # a self-consistent pair
        assert {e['name']: (e['sha256'], e['bytes']) for e in doc['entries']} == {
            n: (sha(b), len(b)) for n, b in files.items()}
        calls = []
        real_open = rs.tarfile.open
        monkeypatch.setattr(rs, '_hook', lambda s: calls.append(('hook', s)))
        monkeypatch.setattr(rs, '_extract', lambda *a: calls.append(('extract',)))
        monkeypatch.setattr(rs.tarfile, 'open', lambda *a, **k: calls.append(('open',)) or real_open(*a, **k))
        inputs_before = tree(inputs)
        e = refused(rs.restore, archive, desc, dest)
        assert e.state == 'not_restored' and has(e.problems, text) and e.retained_files == []
        assert has(refused(rs.inspect_archive, archive, doc).problems, text)
        assert calls == []
        assert tree(dest) == before and leftovers(dest.parent, 'dest') == []
        assert not list(dest.parent.glob('.*.lock')) and tree(inputs) == inputs_before

    def test_other_release_in_archive_fails(self, env, previous, tmp_path):
        """A descriptor rewritten for another package's archive still can't
        pass: its release_id / archive name must match the other manifest."""
        dest, before = previous
        other = rs.pack(env['pkgB'], env['schedB'], tmp_path / 'packedB')
        doc = copy.deepcopy(env['descriptor'])
        data = Path(other['archive']['path']).read_bytes()
        doc['archive'].update(sha256=sha(data), bytes=len(data))
        desc = write_json(tmp_path / 'mixed.json', doc)
        e = refused(rs.restore, other['archive']['path'], desc, dest)
        assert has(e.problems, "doesn't match the descriptor") and tree(dest) == before


class TestCli:
    def _run(self, *args):
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONIOENCODING', 'PYTHONUTF8')}
        r = subprocess.run([sys.executable, str(ROOT / 'player_props' / 'release_storage.py'), *map(str, args)],
                           capture_output=True, cwd=ROOT, env=env)
        return r.returncode, json.loads(r.stdout.decode('ascii'))

    def test_inspect_and_restore_exit_codes(self, env, tmp_path):
        code, out = self._run('inspect', '--archive', env['archive'], '--descriptor', env['descriptor_path'])
        assert code == 0 and out['ok'] and out['approval'] == 'not_established'
        code, out = self._run('restore', '--archive', env['archive'], '--descriptor', env['descriptor_path'],
                              '--dest', tmp_path / 'r')
        assert code == 0 and out['state'] == 'restored'
        code, out = self._run('restore', '--archive', env['archive'], '--descriptor', env['descriptor_path'],
                              '--dest', ROOT / 'data_files' / 'rs-probe-r')
        assert code == 1 and out['state'] == 'not_restored'
        code, out = self._run('pack', '--package', env['pkgA'], '--prediction-schedule', env['schedX'],
                              '--out', tmp_path / 'p')
        assert code == 1 and not (tmp_path / 'p').exists()

    @pytest.mark.parametrize('value', BAD_VALUES, ids=lambda v: type_name(v))
    def test_malformed_descriptor_types_are_a_normal_refusal(self, env, tmp_path, value):
        """Exit 1 with the field named, never exit 2 from an unexpected exception."""
        desc = write_json(tmp_path / 'bad.storage.json',
                          set_field(copy.deepcopy(env['descriptor']), ('prediction_schedule', 'entry'), value))
        before = desc.read_bytes()
        text = f"prediction_schedule.entry must be a string, not {type_name(value)}"
        for args in (['inspect'], ['restore', '--dest', tmp_path / 'r']):
            code, out = self._run(*args[:1], '--archive', env['archive'], '--descriptor', desc, *args[1:])
            assert code == 1 and 'error' not in out and out['state'] == 'not_restored', out
            assert has(out['problems'], text)
        assert not (tmp_path / 'r').exists() and desc.read_bytes() == before
