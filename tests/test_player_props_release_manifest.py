"""Release manifest (S1): building and verifying a candidate release package
from synthetic rehearsal runs. Offline; everything is written under tmp_path."""
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from player_props import rehearsal as rh
from player_props import release_manifest as rm

ROOT = Path(__file__).resolve().parents[1]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_dir(ws: Path, seed: int) -> Path:
    summary = rh.run_rehearsal(rh.RehearsalConfig(workspace=ws, seed=seed))
    assert summary['status'] == 'succeeded', summary['stages']
    return ws / 'runs' / summary['run_id']


def reserialized(src: Path, dest: Path) -> Path:
    """Same text values, different bytes (every field quoted)."""
    pd.read_csv(src, dtype=str, keep_default_na=False).to_csv(dest, index=False, quoting=csv.QUOTE_ALL)
    return dest


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("release")
    a, b = run_dir(base / "wsA", 7), run_dir(base / "wsB", 11)
    return {'base': base, 'A': a, 'B': b,
            'candA': reserialized(a / 'runtime/data/players.csv', base / 'candA.csv'),
            'candB': reserialized(b / 'runtime/data/players.csv', base / 'candB.csv')}


@pytest.fixture(scope="module")
def built(runs):
    base = runs['base']
    out = {}
    for k in ('A', 'B'):
        out[k] = base / f"pkg{k}"
        out[f"report{k}"] = rm.build_package(rm.inputs_from_run(runs[k], runs[f"cand{k}"]), out[k],
                                             'reserialized', input_dirs=[runs[k]])
    return out


@pytest.fixture
def pkg(built, tmp_path):
    dest = tmp_path / "pkg"
    shutil.copytree(built['A'], dest)
    return dest


def manifest(p: Path) -> dict:
    return json.loads((p / rm.MANIFEST_NAME).read_text(encoding='utf-8'))


def write_manifest(p: Path, doc: dict) -> None:
    (p / rm.MANIFEST_NAME).write_text(json.dumps(doc, indent=2), encoding='utf-8')


def reseal(p: Path) -> None:
    """What a forger would do: make every asset hash and derived section agree
    with the (tampered) files, so only relationship checks can catch it."""
    doc = manifest(p)
    contents = {r: (p / rm.ROLES[r]).read_bytes() for r in rm.ROLES}
    doc['assets'] = [{'role': r, 'name': rm.ROLES[r], 'bytes': len(c), 'sha256': sha(c)}
                     for r, c in contents.items()]
    with tempfile.TemporaryDirectory() as tmp:
        rm._write_snapshot(contents, Path(tmp))
        bm, bs, index, problems = rm._bundle(contents, Path(tmp))
    assert not problems, problems
    doc['derived'] = rm.derive(contents, bm, bs, index)
    write_manifest(p, doc)


def swap(p: Path, other: Path, *roles: str) -> None:
    for r in roles:
        shutil.copyfile(other / rm.ROLES[r], p / rm.ROLES[r])


def refused(p: Path, *parent) -> list:
    with pytest.raises(rm.ReleaseError) as exc:
        rm.verify_package(p, *parent)
    return exc.value.problems


def problems(p: Path, *parent) -> list:
    report = rm.verify_package(p, *parent)
    assert not report['ok'] and not report['verified']
    return report['problems']


def has(items: list, text: str) -> bool:
    return any(text in i for i in items)


STATS = [f"training_stats_{k}" for k in ('passing', 'rushing', 'receiving')] + ['training_stats_manifest']
INFER_STATS = [f"inference_stats_{k}" for k in ('passing', 'rushing', 'receiving')] + ['inference_stats_manifest']


class TestValidPackage:
    def test_builds_and_verifies(self, built):
        report = built['reportA']
        assert report['ok'] and report['problems'] == []
        assert rm.verify_package(built['A'])['ok']
        assert len(report['verified']) >= 8 and report['declared'] == rm.CLAIMS
        assert "do not make this package" in report['immutability']

    def test_package_holds_exactly_the_manifest_and_listed_assets(self, built):
        files = sorted(p.relative_to(built['A']).as_posix() for p in built['A'].rglob('*') if p.is_file())
        assert files == sorted([rm.MANIFEST_NAME] + list(rm.ROLES.values()))
        doc = manifest(built['A'])
        assert [a['role'] for a in doc['assets']] == list(rm.ROLES)
        assert doc['external']['tag'] is None and "descriptive only" in doc['external']['note']

    def test_training_and_inference_inputs_are_distinct_and_linked(self, built):
        ids = manifest(built['A'])['derived']['identities']
        b = manifest(built['A'])['derived']['bundle']
        assert ids['training_source']['sha256'] != ids['inference_source']['sha256']
        assert ids['training_stats']['manifest_sha256'] != ids['inference_stats']['manifest_sha256']
        # Linked, not equal: the bundle names the training set, each stats set its own source.
        assert b['training_source']['stats_manifest_sha256'] == ids['training_stats']['manifest_sha256']
        assert b['training_source']['stats_files'] == ids['training_stats']['files']
        assert ids['training_stats']['source_sha256'] == ids['training_source']['sha256']
        assert ids['inference_stats']['source_sha256'] == ids['inference_source']['sha256']
        w = manifest(built['A'])['derived']['windows']
        assert w['trained_through'] < w['feature_cutoff']

    def test_candidate_and_runtime_reference_keep_separate_hashes(self, built):
        d = manifest(built['A'])['derived']
        cand, run = d['identities']['reference_candidate'], d['identities']['reference_runtime']
        assert cand['sha256'] != run['sha256']
        assert d['eligibility']['players_reference_sha256'] == run['sha256']
        assert manifest(built['A'])['reference']['derivation'] == 'reserialized'

    def test_byte_copy_reference(self, runs, tmp_path):
        inputs = rm.inputs_from_run(runs['A'], runs['A'] / 'runtime/data/players.csv')
        report = rm.build_package(inputs, tmp_path / 'copy', 'byte_copy', input_dirs=[runs['A']])
        assert report['ok']
        d = manifest(tmp_path / 'copy')['derived']['identities']
        assert d['reference_candidate'] == d['reference_runtime']

    def test_target_and_evidence_are_recorded(self, built, runs):
        d = manifest(built['A'])['derived']
        assert d['inference_target'] == {'season': 2026, 'week': 3}
        assert d['verification']['status'] == 'succeeded'
        assert d['verification']['run_id'] == runs['A'].name
        assert d['eligibility']['policy'] == 'position-group-v1'

    def test_cli_verify(self, built):
        r = subprocess.run([sys.executable, 'player_props/release_manifest.py', 'verify', '--package', str(built['A'])],
                           cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        assert r.returncode == 0, r.stdout + r.stderr
        assert json.loads(r.stdout)['ok'] is True


class TestAssetEntries:
    def test_missing_asset_file(self, pkg):
        (pkg / rm.ROLES['bundle_archive']).unlink()
        assert has(refused(pkg), "bundle/bundle.tar (bundle_archive) is listed but missing")

    def test_unlisted_file(self, pkg):
        (pkg / 'evidence' / 'extra.json').write_text('{}', encoding='utf-8')
        assert has(refused(pkg), "unlisted file(s) in the package: ['evidence/extra.json']")

    def test_duplicate_role(self, pkg):
        doc = manifest(pkg)
        doc['assets'].append(dict(doc['assets'][0]))
        write_manifest(pkg, doc)
        found = refused(pkg)
        assert has(found, "role training_source is listed more than once")
        assert has(found, "is listed more than once")

    def test_conflicting_entries_for_one_name(self, pkg):
        doc = manifest(pkg)
        doc['assets'][1] = {**doc['assets'][1], 'name': doc['assets'][0]['name']}
        write_manifest(pkg, doc)
        found = refused(pkg)
        assert has(found, "training_source_manifest must be stored at source/training/player_props_source_manifest.json")
        assert has(found, "name source/training/nfl_play_by_play_props.csv.gz is listed more than once")

    def test_unexpected_role(self, pkg):
        doc = manifest(pkg)
        doc['assets'].append({'role': 'extra_model', 'name': 'bundle/extra.json', 'bytes': 2, 'sha256': '0' * 64})
        write_manifest(pkg, doc)
        assert has(refused(pkg), "unexpected role 'extra_model'")

    def test_missing_role(self, pkg):
        doc = manifest(pkg)
        doc['assets'] = [a for a in doc['assets'] if a['role'] != 'rehearsal_summary']
        write_manifest(pkg, doc)
        assert has(refused(pkg), "missing asset role(s) ['rehearsal_summary']")

    @pytest.mark.parametrize("role", [["training_source"], {"role": "training_source"}, None, 7, True, 1.5],
                             ids=["list", "object", "null", "int", "bool", "float"])
    def test_non_string_role(self, pkg, role):
        """Review finding 3: an unhashable role raised TypeError at the ROLES lookup."""
        doc = manifest(pkg)
        doc['assets'][2]['role'] = role
        write_manifest(pkg, doc)
        found = refused(pkg)
        assert has(found, f"assets[2].role must be a string naming one of the 19 asset roles, "
                          f"not {type(role).__name__}")
        assert has(found, "missing asset role(s) ['training_schedule']")

    @pytest.mark.parametrize("name", [["stats", "x.csv"], {"path": "x"}, None, 3])
    def test_non_string_name(self, pkg, name):
        doc = manifest(pkg)
        doc['assets'][2]['name'] = name
        write_manifest(pkg, doc)
        assert has(refused(pkg), f"assets[2].name must be a string path, not {type(name).__name__}")

    def test_cli_reports_bad_role_without_a_traceback(self, pkg):
        doc = manifest(pkg)
        doc['assets'][0]['role'] = ['training_source']
        write_manifest(pkg, doc)
        r = subprocess.run([sys.executable, 'player_props/release_manifest.py', 'verify', '--package', str(pkg)],
                           cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        assert r.returncode == 1, r.stdout + r.stderr
        assert "assets[0].role must be a string" in r.stdout and "Traceback" not in r.stderr

    @pytest.mark.parametrize("name", ["../outside.json", "/abs/summary.json", "evidence\\summary.json",
                                      "C:summary.json", "evidence/./summary.json", ""])
    def test_unsafe_names(self, pkg, name):
        doc = manifest(pkg)
        i = [a['role'] for a in doc['assets']].index('rehearsal_summary')
        doc['assets'][i]['name'] = name
        write_manifest(pkg, doc)
        assert has(refused(pkg), "is not a safe relative path")

    def test_size_mismatch(self, pkg):
        doc = manifest(pkg)
        doc['assets'][0]['bytes'] += 1
        write_manifest(pkg, doc)
        assert has(refused(pkg), "source/training/nfl_play_by_play_props.csv.gz: size")

    def test_hash_mismatch(self, pkg):
        p = pkg / rm.ROLES['inference_stats_passing']
        data = bytearray(p.read_bytes())
        data[-2] = ord('9') if data[-2] != ord('9') else ord('8')
        p.write_bytes(bytes(data))
        assert has(refused(pkg), "stats/inference/player_passing_stats.csv: SHA-256 differs from the manifest")

    def test_linked_asset_is_refused(self, pkg, tmp_path):
        target = pkg / rm.ROLES['rehearsal_summary']
        outside = tmp_path / 'outside.json'
        shutil.copyfile(target, outside)
        target.unlink()
        try:
            os.symlink(outside, target)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are not available here")
        assert has(refused(pkg), "evidence/summary.json is or passes through a link")

    @pytest.mark.skipif(sys.platform != 'win32', reason="Windows junctions only")
    def test_junction_is_refused(self, pkg, tmp_path):
        outside = tmp_path / 'outside_evidence'
        shutil.move(str(pkg / 'evidence'), str(outside))
        r = subprocess.run(['cmd', '/c', 'mklink', '/J', str(pkg / 'evidence'), str(outside)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        found = refused(pkg)
        assert has(found, "evidence is a link or junction; refused")
        assert has(found, "evidence/summary.json (rehearsal_summary) is listed but missing")


class TestMalformedManifest:
    def test_invalid_json_names_line_and_column(self, pkg):
        (pkg / rm.MANIFEST_NAME).write_text('{\n  "kind": "player_props_release",\n  oops\n}', encoding='utf-8')
        found = refused(pkg)
        assert found[0].startswith("release_manifest.json is not valid JSON: line 3, column 3")

    def test_not_an_object(self, pkg):
        (pkg / rm.MANIFEST_NAME).write_text('[1, 2]', encoding='utf-8')
        assert refused(pkg) == ["release_manifest.json: release_manifest.json must be a JSON object"]

    def test_wrong_kind_or_version(self, pkg):
        doc = manifest(pkg)
        doc['manifest_version'] = 2
        write_manifest(pkg, doc)
        assert has(refused(pkg), "is not a version-1 player_props_release manifest")

    def test_unknown_and_missing_keys(self, pkg):
        doc = manifest(pkg)
        doc['verified'] = True
        del doc['claims']
        write_manifest(pkg, doc)
        found = refused(pkg)
        assert has(found, "unexpected top-level key(s) ['verified']")
        assert has(found, "missing top-level key(s) ['claims']")

    def test_field_level_diagnostics(self, pkg):
        doc = manifest(pkg)
        doc['assets'][3]['sha256'] = 'ABC'
        doc['release_id'] = 'latest'
        doc['reference']['derivation'] = 'normalised'
        doc['parent_release'] = {'release_id': 'x'}
        write_manifest(pkg, doc)
        found = refused(pkg)
        for text in ("assets[3]: sha256 must be a 64-character lowercase hex SHA-256",
                     "release_id must look like", "reference must be {derivation: one of",
                     "parent_release must be null or {release_id, manifest_sha256}"):
            assert has(found, text), (text, found)

    def test_malformed_evidence_json(self, pkg):
        (pkg / rm.ROLES['rehearsal_summary']).write_text('{"kind": ', encoding='utf-8')
        doc = manifest(pkg)
        data = (pkg / rm.ROLES['rehearsal_summary']).read_bytes()
        for a in doc['assets']:
            if a['role'] == 'rehearsal_summary':
                a.update(bytes=len(data), sha256=sha(data))
        write_manifest(pkg, doc)
        assert has(problems(pkg), "evidence/summary.json is not valid JSON: line 1, column 10")

    def test_edited_derived_section(self, pkg):
        doc = manifest(pkg)
        doc['derived']['windows']['trained_through'] = '2030-01-01'
        doc['derived']['eligibility']['policy'] = 'label-v0'
        write_manifest(pkg, doc)
        found = problems(pkg)
        assert has(found, "derived.windows doesn't match the assets at trained_through")
        assert has(found, "derived.eligibility doesn't match the assets at policy")


class TestRelationships:
    """Every asset hash is valid (resealed); only the relationships disagree."""

    def test_training_stats_from_another_run(self, pkg, built):
        swap(pkg, built['B'], *STATS)
        reseal(pkg)
        found = problems(pkg)
        assert has(found, "the bundle was trained on stats manifest")
        assert has(found, "the training stats were built from source")
        assert has(found, "the rehearsal summary records train/stats/player_props_stats_manifest.json")

    def test_inference_stats_from_another_run(self, pkg, built):
        swap(pkg, built['B'], *INFER_STATS)
        reseal(pkg)
        found = problems(pkg)
        assert has(found, "the inference stats were built from source")
        assert has(found, "the rehearsal summary records runtime/data/player_passing_stats.csv")

    def test_inference_source_from_another_run(self, pkg, built):
        swap(pkg, built['B'], 'inference_source', 'inference_source_manifest')
        reseal(pkg)
        found = problems(pkg)
        assert has(found, "the inference stats were built from source")
        assert has(found, "the rehearsal summary's inference_source differs")

    def test_bundle_from_another_run(self, pkg, built):
        swap(pkg, built['B'], 'bundle_archive')
        reseal(pkg)
        found = problems(pkg)
        assert has(found, "the bundle was trained on stats manifest")
        assert has(found, "names another bundle")

    def test_unrelated_verification_evidence(self, pkg, built):
        swap(pkg, built['B'], 'rehearsal_summary')
        reseal(pkg)
        found = problems(pkg)
        assert has(found, "the rehearsal summary records train/source/nfl_play_by_play_props.csv.gz")
        assert has(found, "names another bundle")
        assert has(found, "names another predictions snapshot")

    def test_verified_true_is_not_enough(self, pkg):
        """A summary that claims success and verification, but whose artifact
        list describes other bytes, is refused."""
        p = pkg / rm.ROLES['rehearsal_summary']
        doc = json.loads(p.read_text(encoding='utf-8'))
        assert doc['status'] == 'succeeded' and doc['eligibility']['verified'] is True
        doc['artifacts']['export/bundle.tar'] = '0' * 64
        p.write_text(json.dumps(doc), encoding='utf-8')
        reseal(pkg)
        assert has(problems(pkg), "the rehearsal summary records export/bundle.tar as 000000000000")

    def test_evidence_naming_the_candidate_reference(self, pkg):
        p = pkg / rm.ROLES['rehearsal_summary']
        doc = json.loads(p.read_text(encoding='utf-8'))
        doc['eligibility']['players_reference_sha256'] = sha((pkg / rm.ROLES['reference_candidate']).read_bytes())
        p.write_text(json.dumps(doc), encoding='utf-8')
        reseal(pkg)
        assert has(problems(pkg), "(the candidate reference), not the runtime reference")

    def test_unverified_eligibility_evidence(self, pkg):
        p = pkg / rm.ROLES['rehearsal_summary']
        doc = json.loads(p.read_text(encoding='utf-8'))
        doc['eligibility'].update(verified=False, players_reference_source='current file')
        p.write_text(json.dumps(doc), encoding='utf-8')
        reseal(pkg)
        assert has(problems(pkg), "was not recorded by a passing verification")

    def test_failed_rehearsal_evidence(self, pkg):
        p = pkg / rm.ROLES['rehearsal_summary']
        doc = json.loads(p.read_text(encoding='utf-8'))
        doc['status'] = 'failed'
        doc['stages'][-1]['status'] = 'not_run'
        p.write_text(json.dumps(doc), encoding='utf-8')
        reseal(pkg)
        found = problems(pkg)
        assert has(found, "the rehearsal's status is 'failed', not 'succeeded'")
        assert has(found, "stages[11]: stage publish has status 'not_run', not 'ok'")


    def test_runtime_reference_replaced(self, pkg):
        """The runtime reference differs from what verification read, even
        though its identity columns still match the candidate."""
        p = pkg / rm.ROLES['reference_runtime']
        p.write_bytes(p.read_bytes().replace(b'\n', b'\r\n'))
        reseal(pkg)
        found = problems(pkg)
        assert has(found, "the eligibility evidence names players reference")
        assert has(found, "the rehearsal summary records runtime/data/players.csv")

    def test_byte_copy_declared_but_reserialized(self, pkg):
        doc = manifest(pkg)
        doc['reference']['derivation'] = 'byte_copy'
        write_manifest(pkg, doc)
        assert has(problems(pkg), "declared a byte copy of the candidate, but their bytes differ")

    def test_reserialized_declared_but_identical(self, pkg):
        shutil.copyfile(pkg / rm.ROLES['reference_runtime'], pkg / rm.ROLES['reference_candidate'])
        reseal(pkg)
        assert has(problems(pkg), "declared reserialized, but it is byte-identical")

    def test_candidate_identity_change(self, pkg):
        p = pkg / rm.ROLES['reference_candidate']
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        df.loc[0, 'position_group'] = 'WR' if df.loc[0, 'position_group'] != 'WR' else 'RB'
        df.to_csv(p, index=False, quoting=csv.QUOTE_ALL)
        reseal(pkg)
        assert has(problems(pkg), "identity column position_group differs in 1 row(s)")

    def test_whitespace_is_not_normalised(self, pkg):
        p = pkg / rm.ROLES['reference_candidate']
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        df.loc[2, 'display_name'] = df.loc[2, 'display_name'] + ' '
        df.to_csv(p, index=False, quoting=csv.QUOTE_ALL)
        reseal(pkg)
        assert has(problems(pkg), "identity column display_name differs in 1 row(s)")

    def test_schedule_from_elsewhere(self, pkg):
        p = pkg / rm.ROLES['inference_schedule']
        p.write_bytes(p.read_bytes() + b'\n')
        reseal(pkg)
        assert has(problems(pkg), "the inference source manifest names schedule")

    def test_feature_cutoff_claim_disagrees(self, pkg):
        p = pkg / rm.ROLES['rehearsal_summary']
        doc = json.loads(p.read_text(encoding='utf-8'))
        doc['coverage']['inputs']['feature_cutoff'] = '2024-01-01'
        p.write_text(json.dumps(doc), encoding='utf-8')
        reseal(pkg)
        found = problems(pkg)
        assert has(found, "must end before the inference feature cutoff (2024-01-01)")
        assert has(found, "differs from the rehearsal's 2024-01-01")

    def test_windows_claim_disagrees(self, pkg):
        p = pkg / rm.ROLES['rehearsal_summary']
        doc = json.loads(p.read_text(encoding='utf-8'))
        first = sorted(doc['windows'])[0]
        doc['windows'][first]['training']['rows'] += 1
        p.write_text(json.dumps(doc), encoding='utf-8')
        reseal(pkg)
        assert has(problems(pkg), "training/evaluation windows differ from the bundle's")

    def test_target_week_is_rechecked_by_the_stage2_gate(self, pkg):
        """The evidence moves the verified snapshot to week 9; re-running the
        Stage 2 gate for that target fails (the synthetic season has 8 weeks)."""
        p = pkg / rm.ROLES['rehearsal_summary']
        doc = json.loads(p.read_text(encoding='utf-8'))
        old = 'runtime/data/player_props_predictions_week3_2026.csv'
        doc['artifacts']['runtime/data/player_props_predictions_week9_2026.csv'] = doc['artifacts'].pop(old)
        p.write_text(json.dumps(doc), encoding='utf-8')
        reseal(pkg)
        assert manifest(pkg)['derived']['inference_target'] == {'season': 2026, 'week': 9}
        assert has(problems(pkg), "inference inputs for 2026 week 9: schedule has no games for 2026 week 9")

    def test_predictions_snapshot_from_another_run(self, pkg, built):
        swap(pkg, built['B'], 'verified_predictions')
        reseal(pkg)
        found = problems(pkg)
        assert has(found, "exactly one is required")
        assert has(found, "names another predictions snapshot")


def edit_stages(pkg: Path, change) -> list:
    """Apply ``change`` to the summary's stage records, reseal every hash and
    derived section, and return the verification problems."""
    p = pkg / rm.ROLES['rehearsal_summary']
    doc = json.loads(p.read_text(encoding='utf-8'))
    assert doc['status'] == 'succeeded'           # only the stage records are wrong
    doc['stages'] = change(doc['stages'])
    p.write_text(json.dumps(doc), encoding='utf-8')
    reseal(pkg)
    return problems(pkg)


def _set(i, **fields):
    def change(stages):
        stages[i] = {**stages[i], **fields}
        return stages
    return change


def _replace(i, record):
    def change(stages):
        stages[i] = record
        return stages
    return change


class TestStageEvidence:
    """Exactly rehearsal.STAGES, in order, each once, each 'ok' (review finding 2)."""

    def test_complete_stages_are_accepted(self, built):
        stages = json.loads((built['A'] / rm.ROLES['rehearsal_summary']).read_text(encoding='utf-8'))['stages']
        assert [s['stage'] for s in stages] == rh.STAGES and rm.stage_problems(stages) == []

    def test_workspace_only_is_refused_after_resealing(self, pkg):
        """Previously accepted: status 'succeeded' and one record, workspace: ok."""
        found = edit_stages(pkg, lambda s: [{'stage': 'workspace', 'status': 'ok'}])
        assert manifest(pkg)['derived']['verification']['stages'] == [['workspace', 'ok']]   # resealed
        assert has(found, "has no record of stage(s) ['source_training', 'source_inference',")
        assert has(found, "all 12 supported stages must be recorded as 'ok'")

    @pytest.mark.parametrize("case, change, text", [
        ("missing", lambda s: [x for x in s if x['stage'] != 'train'], "has no record of stage(s) ['train']"),
        ("empty", lambda s: [], "has no record of stage(s) ['workspace',"),
        ("duplicate", lambda s: s + [dict(s[4])], "stages[12]: stage train is recorded more than once"),
        ("unknown", lambda s: s + [{'stage': 'deploy', 'status': 'ok'}], "stages[12]: unknown stage 'deploy'"),
        ("renamed", _set(0, stage='Workspace'), "stages[0]: unknown stage 'Workspace'"),
        ("not_a_list", lambda s: {'workspace': 'ok'}, "stages must be a list of {stage, status} records, not dict"),
        ("record_not_object", _replace(3, 'aggregate_training'), "stages[3] must be an object, not str"),
        ("record_null", _replace(3, None), "stages[3] must be an object, not NoneType"),
        ("status_missing", _replace(5, {'stage': 'export'}), "stages[5] must have a string stage and status"),
        ("stage_not_string", _set(5, stage=['export']), "stages[5] must have a string stage and status"),
        ("status_not_string", _set(5, status=True), "stages[5] must have a string stage and status"),
        ("extra_field", _set(6, verified=True), "stages[6] (restore) has unexpected field(s) ['verified']"),
        ("failed", _set(9, status='failed'), "stages[9]: stage predict has status 'failed', not 'ok'"),
        ("not_run", _set(10, status='not_run'), "stages[10]: stage verify_outputs has status 'not_run'"),
        ("ok_with_error", _set(8, error={'type': 'X'}), "stage validate_inputs is 'ok' but records an error"),
        ("out_of_order", lambda s: [s[1], s[0]] + s[2:], "records the stages out of order"),
    ])
    def test_incomplete_stage_evidence_is_refused(self, pkg, case, change, text):
        found = edit_stages(pkg, change)
        assert has(found, text), found
        assert not has(found, "Traceback")



class TestBuild:
    def test_mixed_inputs_are_not_packaged(self, runs, tmp_path):
        inputs = rm.inputs_from_run(runs['A'], runs['candA'])
        inputs['training_stats_manifest'] = runs['B'] / rm.RUN_PATHS['training_stats_manifest']
        with pytest.raises(rm.ReleaseError) as exc:
            rm.build_package(inputs, tmp_path / 'mixed', 'reserialized')
        assert has(exc.value.problems, "not packaged: ")
        assert not (tmp_path / 'mixed').exists()
        assert [p.name for p in tmp_path.iterdir()] == []

    def test_input_roles_must_match(self, runs, tmp_path):
        inputs = rm.inputs_from_run(runs['A'], runs['candA'])
        del inputs['rehearsal_summary']
        with pytest.raises(rm.ReleaseError, match="missing \\['rehearsal_summary'\\]"):
            rm.build_package(inputs, tmp_path / 'p', 'reserialized')

    @pytest.mark.parametrize("where", ["inside_run", "is_run", "holds_candidate"])
    def test_output_overlapping_inputs_is_refused(self, runs, tmp_path, where):
        inputs = rm.inputs_from_run(runs['A'], runs['candA'])
        out = {'inside_run': runs['A'] / 'pkg', 'is_run': runs['A'], 'holds_candidate': runs['candA'].parent}[where]
        before = sorted(p.name for p in runs['A'].iterdir())
        with pytest.raises(rm.ReleaseError, match="overlaps input"):
            rm.build_package(inputs, out, 'reserialized', input_dirs=[runs['A']])
        assert sorted(p.name for p in runs['A'].iterdir()) == before

    @pytest.mark.parametrize("target", ["directory", "file"])
    def test_output_overlapping_an_input_file_is_refused(self, runs, tmp_path, target):
        """No input directories given: each input file is still protected."""
        ref_dir = tmp_path / 'refs'
        ref_dir.mkdir()
        cand = ref_dir / 'cand.csv'
        shutil.copyfile(runs['candA'], cand)
        before = sha(cand.read_bytes())
        out = ref_dir if target == 'directory' else cand
        with pytest.raises(rm.ReleaseError, match="overlaps input reference_candidate"):
            rm.build_package(rm.inputs_from_run(runs['A'], cand), out, 'reserialized')
        assert sha(cand.read_bytes()) == before and sorted(p.name for p in ref_dir.iterdir()) == ['cand.csv']

    @pytest.mark.parametrize("where", [ROOT / 'release_pkg', ROOT / 'data_files' / 'release',
                                       ROOT / 'player_props' / 'models' / 'release'])
    def test_output_in_protected_paths_is_refused(self, runs, where):
        with pytest.raises(rm.ReleaseError, match="overlaps protected location"):
            rm.build_package(rm.inputs_from_run(runs['A'], runs['candA']), where, 'reserialized')
        assert not where.exists()

    def _snapshot(self, d: Path) -> dict:
        return {p.relative_to(d).as_posix(): sha(p.read_bytes()) for p in sorted(d.rglob('*')) if p.is_file()}

    def test_failed_verification_keeps_previous_package(self, runs, built, tmp_path):
        out = tmp_path / 'pkg'
        shutil.copytree(built['A'], out)
        before = self._snapshot(out)
        inputs = rm.inputs_from_run(runs['A'], runs['candA'])
        inputs['bundle_archive'] = runs['B'] / rm.RUN_PATHS['bundle_archive']
        with pytest.raises(rm.ReleaseError, match="not packaged"):
            rm.build_package(inputs, out, 'reserialized')
        assert self._snapshot(out) == before
        assert sorted(p.name for p in tmp_path.iterdir()) == ['pkg']

    def test_failed_write_keeps_previous_package(self, runs, built, tmp_path, monkeypatch):
        out = tmp_path / 'pkg'
        shutil.copytree(built['A'], out)
        before = self._snapshot(out)
        real_write, calls = Path.write_bytes, []

        def flaky(self, data):
            calls.append(self)
            if len(calls) == 5:
                raise OSError("disk full (simulated)")
            return real_write(self, data)
        monkeypatch.setattr(Path, 'write_bytes', flaky)
        with pytest.raises(OSError, match="disk full"):
            rm.build_package(rm.inputs_from_run(runs['A'], runs['candA']), out, 'reserialized')
        monkeypatch.undo()
        assert self._snapshot(out) == before
        assert sorted(p.name for p in tmp_path.iterdir()) == ['pkg']

    def test_failed_swap_restores_previous_package(self, runs, built, tmp_path, monkeypatch):
        out = tmp_path / 'pkg'
        shutil.copytree(built['A'], out)
        before = self._snapshot(out)
        real_rename = os.rename

        def failing(src, dst):
            if '.staging-' in str(src):
                raise OSError("rename refused (simulated)")
            return real_rename(src, dst)
        monkeypatch.setattr(rm.os, 'rename', failing)
        with pytest.raises(OSError, match="rename refused"):
            rm.build_package(rm.inputs_from_run(runs['A'], runs['candA']), out, 'reserialized')
        monkeypatch.undo()
        assert self._snapshot(out) == before
        assert sorted(p.name for p in tmp_path.iterdir()) == ['pkg']

    def test_rebuild_replaces_a_previous_package(self, runs, built, tmp_path):
        out = tmp_path / 'pkg'
        shutil.copytree(built['A'], out)
        old_id = manifest(out)['release_id']
        report = rm.build_package(rm.inputs_from_run(runs['A'], runs['candA']), out, 'reserialized')
        assert report['ok'] and manifest(out)['release_id'] != old_id
        assert sorted(p.name for p in tmp_path.iterdir()) == ['pkg']


@pytest.fixture(scope="module")
def child(runs, built, tmp_path_factory):
    out = tmp_path_factory.mktemp("child") / "pkg"
    rm.build_package(rm.inputs_from_run(runs['A'], runs['candA']), out, 'reserialized', parent=built['A'],
                     tag='props-candidate-1', pointer='not published')
    return out


class TestParentAndExternal:
    def test_parent_identity_recorded(self, child, built):
        doc = manifest(child)
        assert doc['parent_release'] == {'release_id': manifest(built['A'])['release_id'],
                                         'manifest_sha256': sha((built['A'] / rm.MANIFEST_NAME).read_bytes())}
        assert doc['external']['tag'] == 'props-candidate-1'

    def test_parent_checked_when_given(self, child, built):
        assert rm.verify_package(child, built['A'])['ok']

    def test_parent_declared_when_not_given(self, child):
        report = rm.verify_package(child)
        assert report['ok'] and has(report['declared'], "(not checked: no parent package given)")

    def test_wrong_parent(self, child, built):
        assert has(problems(child, built['B']), "the parent package is not the parent this manifest names")

    def test_parent_given_without_parent_release(self, built):
        assert has(problems(built['A'], built['B']), "this manifest names no parent release")


def tree(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): sha(p.read_bytes()) for p in sorted(d.rglob('*')) if p.is_file()}


def directory_alias(link: Path, target: Path) -> None:
    """A symlink to ``target``, or on Windows without symlink rights a junction."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if sys.platform != 'win32':
        pytest.skip("directory symlinks are not available here")
    r = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


class TestParentOverlap:
    """Review finding 1: the parent package is an input; the output must not
    be it, contain it or lie inside it, after resolving links."""

    @pytest.fixture
    def setup(self, runs, built, tmp_path):
        inputs = rm.inputs_from_run(runs['A'], runs['candA'])
        return {'inputs': inputs, 'input_hashes': {r: sha(Path(p).read_bytes()) for r, p in inputs.items()},
                'tmp': tmp_path}

    def refuse(self, setup, out, parent, text, input_dirs=None):
        tmp = setup['tmp']
        before = {p.relative_to(tmp).as_posix(): sha(p.read_bytes()) for p in sorted(tmp.rglob('*')) if p.is_file()}
        names = sorted(p.name for p in tmp.rglob('*'))
        with pytest.raises(rm.ReleaseError) as exc:
            rm.build_package(setup['inputs'], out, 'reserialized', parent=parent, input_dirs=input_dirs)
        assert has(exc.value.problems, text), exc.value.problems
        assert has(exc.value.problems, "the parent is an input and must not be replaced or written into")
        # Parent, previous output and everything else under tmp: byte-identical, nothing added.
        after = {p.relative_to(tmp).as_posix(): sha(p.read_bytes()) for p in sorted(tmp.rglob('*')) if p.is_file()}
        assert after == before and sorted(p.name for p in tmp.rglob('*')) == names
        assert {r: sha(Path(p).read_bytes()) for r, p in setup['inputs'].items()} == setup['input_hashes']
        return exc.value.problems

    def test_parent_is_the_output(self, setup, built):
        out = setup['tmp'] / 'pkg'
        shutil.copytree(built['A'], out)
        self.refuse(setup, out, out, "is the parent package")
        assert rm.verify_package(out)['ok']

    def test_output_inside_parent(self, setup, built):
        parent = setup['tmp'] / 'parent'
        shutil.copytree(built['A'], parent)
        self.refuse(setup, parent / 'child', parent, "is inside the parent package")
        assert not (parent / 'child').exists() and rm.verify_package(parent)['ok']

    def test_output_inside_parent_subdirectory(self, setup, built):
        parent = setup['tmp'] / 'parent'
        shutil.copytree(built['A'], parent)
        self.refuse(setup, parent / 'evidence' / 'next', parent, "is inside the parent package")

    def test_output_contains_parent(self, setup, built):
        """The previous output holds the parent; replacing it would delete the parent."""
        out = setup['tmp'] / 'releases'
        shutil.copytree(built['A'], out / 'parent')
        (out / 'notes.txt').write_text('previous output content', encoding='utf-8')
        self.refuse(setup, out, out / 'parent', "contains the parent package")

    @pytest.mark.parametrize("alias_of", ["parent", "output"])
    def test_resolved_alias(self, setup, built, alias_of):
        real = setup['tmp'] / 'pkg'
        shutil.copytree(built['A'], real)
        alias = setup['tmp'] / 'alias'
        directory_alias(alias, real)
        out, parent = (real, alias) if alias_of == 'parent' else (alias, real)
        self.refuse(setup, out, parent, "is the parent package")
        assert rm.verify_package(real)['ok']

    def test_alias_to_a_parent_ancestor(self, setup, built):
        holder = setup['tmp'] / 'holder'
        shutil.copytree(built['A'], holder / 'parent')
        alias = setup['tmp'] / 'alias'
        directory_alias(alias, holder)
        self.refuse(setup, alias / 'parent' / 'child', holder / 'parent', "is inside the parent package")

    def test_checked_before_the_parent_is_read(self, setup):
        """An overlapping parent that isn't even a package is refused for the
        overlap, so nothing about it is read or created first."""
        out = setup['tmp'] / 'missing'
        self.refuse(setup, out, out, "is the parent package")
        assert not out.exists()

    def test_cli_refuses_overlapping_parent(self, runs, built, tmp_path):
        out = tmp_path / 'pkg'
        shutil.copytree(built['A'], out)
        before = tree(out)
        r = subprocess.run([sys.executable, 'player_props/release_manifest.py', 'build', '--run-dir', str(runs['A']),
                            '--candidate-reference', str(runs['candA']), '--reference-derivation', 'reserialized',
                            '--out', str(out), '--parent', str(out)],
                           cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        assert r.returncode == 1 and "is the parent package" in r.stdout, r.stdout + r.stderr
        assert tree(out) == before

    def test_separate_parent_still_builds(self, child, built):
        """Sibling directories: a valid parent/child pair (the module-level child fixture)."""
        assert not rm._overlaps(rm._real(child), rm._real(built['A']))
        assert rm.verify_package(child, built['A'])['ok']
