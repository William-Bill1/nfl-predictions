"""Players-reference candidate builder (player_props/reference_builder.py).

Synthetic references only: small hand-built players files, a downloads
manifest with their hashes, and an optional weekly roster. No network."""
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from player_props import reference_builder as rb

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ['gsis_id', 'display_name', 'short_name', 'last_name', 'position', 'latest_team', 'status',
           'esb_id', 'smart_id', 'draft_year']


@pytest.fixture(autouse=True)
def blocked(monkeypatch):
    def refuse(*a, **k):
        raise OSError("network blocked in this test")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def rec(gsis, pos='WR', esb=None, smart=None, name=None, team='AAA', status='ACT', draft='2020'):
    name = name or f"Player {gsis}"
    return {'gsis_id': gsis, 'display_name': name, 'short_name': name[0] + '.' + name.split()[-1],
            'last_name': name.split()[-1], 'position': pos, 'latest_team': team, 'status': status,
            'esb_id': esb, 'smart_id': smart, 'draft_year': draft}


def write(path, rows, columns=COLUMNS):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def setup(tmp_path, existing, fresh, roster=None, fresh_columns=COLUMNS):
    """(existing path, downloads dir) with a manifest recording the fresh files."""
    ex = write(tmp_path / 'existing.csv', existing)
    dl = tmp_path / 'downloads'
    dl.mkdir()
    write(dl / 'players.csv', fresh, fresh_columns)
    if roster is not None:
        pd.DataFrame(roster).to_csv(dl / 'roster_weekly_2026.csv', index=False)
    files = {}
    for f in dl.iterdir():
        b = f.read_bytes()
        files[f.name] = {'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b), 'url': f'https://x/{f.name}'}
    (dl / 'manifest.json').write_text(json.dumps({'files': files}), encoding='utf-8')
    return ex, dl


def built(out):
    return pd.read_csv(out / 'players_candidate.csv', dtype=str, keep_default_na=False, na_values=[''])


def walk(out):
    return pd.read_csv(out / 'crosswalk.csv', dtype=str).set_index('old_id')


def tree(d):
    return {p.name: p.read_bytes() for p in sorted(Path(d).iterdir())}


class TestRetentionAndOrder:
    def test_historical_id_absent_from_source_is_retained(self, tmp_path):
        ex, dl = setup(tmp_path, [rec('00-0000001'), rec('00-0000002')], [rec('00-0000001')])
        r = rb.build(ex, dl, tmp_path / 'out')
        out = built(tmp_path / 'out')
        assert set(out['gsis_id']) == {'00-0000001', '00-0000002'}
        assert r['identities']['retained_historical'] == {'records': 1, 'people': 1, 'gsis_format': 1}
        assert r['identities']['retained_in_source']['records'] == 1

    def test_backstop_refuses_any_lost_id(self, tmp_path, monkeypatch):
        ex, dl = setup(tmp_path, [rec('00-0000001'), rec('00-0000002')], [rec('00-0000001')])
        monkeypatch.setattr(rb, '_assemble', lambda fresh, hist, cols: fresh[cols].reset_index(drop=True))
        with pytest.raises(rb.ReferenceBuildError, match=r"loses 1 existing ID\(s\): \['00-0000002'\]"):
            rb.build(ex, dl, tmp_path / 'out')
        assert not (tmp_path / 'out').exists()

    def test_added_ids_counted_separately(self, tmp_path):
        ex, dl = setup(tmp_path, [rec('00-0000001')], [rec('00-0000001'), rec('00-0000009')])
        r = rb.build(ex, dl, tmp_path / 'out')
        assert r['identities']['added'] == {'records': 1, 'people': 1, 'gsis_format': 1}

    def test_column_order_and_text_values_preserved(self, tmp_path):
        # Fresh columns in another order, plus one the reference doesn't have.
        cols = list(reversed(COLUMNS)) + ['extra']
        fresh = [dict(rec('00-0000001', draft='2023.0'), extra='x')]
        ex, dl = setup(tmp_path, [rec('00-0000001', draft='2023'), rec('00-0000002', draft='2014.0')],
                       fresh, fresh_columns=cols)
        r = rb.build(ex, dl, tmp_path / 'out')
        out = built(tmp_path / 'out')
        assert list(out.columns) == COLUMNS and r['fresh_columns_not_kept'] == ['extra']
        assert out.set_index('gsis_id').loc['00-0000002', 'draft_year'] == '2014.0'   # retained as text
        assert out.set_index('gsis_id').loc['00-0000001', 'draft_year'] == '2023.0'
        assert r['field_changes']['draft_year'] == {'meaningful': 0, 'format_only': 1, 'consumers': []}

    def test_ids_kept_as_text(self, tmp_path):
        ex, dl = setup(tmp_path, [rec('00-0012345', esb='0123', smart='007')],
                       [rec('00-0012345', esb='0123', smart='007')])
        rb.build(ex, dl, tmp_path / 'out')
        raw = (tmp_path / 'out' / 'players_candidate.csv').read_text(encoding='utf-8')
        assert '00-0012345' in raw and ',0123,007,' in raw


class TestCrosswalk:
    def test_agreed_crosswalk_corrects_the_placeholder(self, tmp_path):
        ex, dl = setup(tmp_path, [rec('ABC123456', esb='ABC123456', smart='s-1')],
                       [rec('00-0040001', esb='ABC123456', smart='s-1')])
        r = rb.build(ex, dl, tmp_path / 'out')
        out, w = built(tmp_path / 'out'), walk(tmp_path / 'out')
        assert list(out['gsis_id']) == ['00-0040001']
        assert w.loc['ABC123456', 'result'] == 'corrected' and w.loc['ABC123456', 'new_id'] == '00-0040001'
        assert r['identities']['corrected_placeholder'] == {'records': 1, 'people': 1, 'gsis_format': 1}

    @pytest.mark.parametrize('fresh, result', [
        ([rec('00-0040001', esb='ABC123456', smart='s-x'), rec('00-0040002', esb='x', smart='s-1')],
         'rejected_conflicting'),
        ([rec('00-0040001', esb='ABC123456', smart='s-1'), rec('00-0040002', esb='ABC123456', smart='s-9')],
         'rejected_ambiguous'),
        ([rec('00-0040001', esb='ABC123456', smart='s-9')], 'rejected_single_id_match'),
        ([rec('00-0040003', esb='nope', smart='nope')], 'no_match'),
    ])
    def test_unsafe_mapping_is_rejected_and_the_placeholder_retained(self, tmp_path, fresh, result):
        ex, dl = setup(tmp_path, [rec('ABC123456', esb='ABC123456', smart='s-1')], fresh)
        rb.build(ex, dl, tmp_path / 'out')
        assert walk(tmp_path / 'out').loc['ABC123456', 'result'] == result
        assert 'ABC123456' in set(built(tmp_path / 'out')['gsis_id'])

    def test_target_already_an_existing_id_is_rejected(self, tmp_path):
        ex, dl = setup(tmp_path, [rec('ABC123456', esb='ABC123456', smart='s-1'), rec('00-0040001')],
                       [rec('00-0040001', esb='ABC123456', smart='s-1')])
        rb.build(ex, dl, tmp_path / 'out')
        assert walk(tmp_path / 'out').loc['ABC123456', 'result'] == 'rejected_target_already_existing'
        assert {'ABC123456', '00-0040001'} <= set(built(tmp_path / 'out')['gsis_id'])

    # Validated inputs can't reach these two (a supported placeholder is its own
    # esb_id, so no fresh placeholder can match another one, and two existing
    # placeholders never share an esb_id). crosswalk() still guards them.
    def test_two_placeholders_claiming_one_target_are_rejected(self):
        existing = pd.DataFrame([rec('ABC123456', esb='E1', smart='s-1'), rec('DEF123456', esb='E1', smart='s-1')])
        w = rb.crosswalk(existing, pd.DataFrame([rec('00-0040001', esb='E1', smart='s-1')]))
        assert set(w['result']) == {'rejected_shared_target'} and w['new_id'].isna().all()

    def test_non_gsis_target_is_rejected(self):
        existing = pd.DataFrame([rec('ABC123456', esb='ABC123456', smart='s-1')])
        w = rb.crosswalk(existing, pd.DataFrame([rec('XYZ999999', esb='ABC123456', smart='s-1')]))
        assert list(w['result']) == ['rejected_target_not_gsis'] and w['new_id'].isna().all()

    def test_names_are_never_used_to_link(self, tmp_path):
        ex, dl = setup(tmp_path, [rec('ABC123456', esb='ABC123456', smart='s-1', name='Same Name')],
                       [rec('00-0040001', esb='DEF123456', smart='s-2', name='Same Name')])
        rb.build(ex, dl, tmp_path / 'out')
        assert walk(tmp_path / 'out').loc['ABC123456', 'result'] == 'no_match'
        assert len(built(tmp_path / 'out')) == 2


class TestSourceAliases:
    def test_alias_records_are_reported_not_merged(self, tmp_path):
        # The source lists one person twice: a placeholder and a GSIS record.
        fresh = [rec('00-0040793', esb='EKE080143', smart='s-e'), rec('EKE080143', esb='EKE080143', smart='s-e'),
                 rec('00-0000001', esb='E1', smart='s-1')]
        ex, dl = setup(tmp_path, [rec('EKE080143', esb='EKE080143', smart='s-e'), rec('00-0000001', esb='E1', smart='s-1')],
                       fresh)
        r = rb.build(ex, dl, tmp_path / 'out')
        assert len(built(tmp_path / 'out')) == 3
        a = r['identities']['source_aliases']
        assert a['records'] == 2 and a['people'] == 1 and a['examples'] == [['00-0040793', 'EKE080143']]
        assert r['identities']['total'] == {'records': 3, 'people': 2}

    @pytest.mark.parametrize('shared', ['esb_id', 'smart_id'])
    def test_either_crosswalk_id_links_aliases(self, tmp_path, shared):
        a = rec('00-0040001', esb='E1', smart='s-1')
        b = rec('00-0040002', esb='E1' if shared == 'esb_id' else 'E2', smart='s-1' if shared == 'smart_id' else 's-2')
        ex, dl = setup(tmp_path, [rec('00-0000001')], [rec('00-0000001'), a, b])
        a = rb.build(ex, dl, tmp_path / 'out')['identities']['source_aliases']
        assert (a['records'], a['people']) == (2, 1)


class TestValidation:
    @pytest.mark.parametrize('which', ['existing', 'fresh'])
    def test_duplicate_ids_refused(self, tmp_path, which):
        rows = [rec('00-0000001'), rec('00-0000001', pos='RB')]
        ex, dl = setup(tmp_path, rows if which == 'existing' else [rec('00-0000001')],
                       rows if which == 'fresh' else [rec('00-0000001')])
        with pytest.raises(rb.ReferenceBuildError, match="more than one record"):
            rb.build(ex, dl, tmp_path / 'out')

    def test_missing_position_refused(self, tmp_path):
        ex, dl = setup(tmp_path, [rec('00-0000001')], [rec('00-0000001'), rec('00-0000002', pos=None)])
        with pytest.raises(rb.ReferenceBuildError, match="missing or blank position"):
            rb.build(ex, dl, tmp_path / 'out')

    def test_missing_gsis_id_refused(self, tmp_path):
        ex, dl = setup(tmp_path, [rec('00-0000001')], [rec('00-0000001'), rec(None)])
        with pytest.raises(rb.ReferenceBuildError, match="missing or blank gsis_id"):
            rb.build(ex, dl, tmp_path / 'out')

    def test_missing_required_column_refused(self, tmp_path):
        cols = [c for c in COLUMNS if c != 'smart_id']
        ex, dl = setup(tmp_path, [rec('00-0000001')], [{k: v for k, v in rec('00-0000001').items() if k != 'smart_id'}],
                       fresh_columns=cols)
        with pytest.raises(rb.ReferenceBuildError, match="missing required column"):
            rb.build(ex, dl, tmp_path / 'out')

    def test_fresh_lacking_an_existing_column_refused(self, tmp_path):
        cols = [c for c in COLUMNS if c != 'draft_year']
        ex, dl = setup(tmp_path, [rec('00-0000001')], [{k: v for k, v in rec('00-0000001').items() if k != 'draft_year'}],
                       fresh_columns=cols)
        with pytest.raises(rb.ReferenceBuildError, match="lacks existing column"):
            rb.build(ex, dl, tmp_path / 'out')

    def test_download_differing_from_manifest_refused(self, tmp_path):
        ex, dl = setup(tmp_path, [rec('00-0000001')], [rec('00-0000001')])
        with open(dl / 'players.csv', 'a', encoding='utf-8') as f:
            f.write('\n')
        with pytest.raises(rb.ReferenceBuildError, match="differs from manifest.json"):
            rb.build(ex, dl, tmp_path / 'out')

    def test_output_inside_the_repository_refused(self, tmp_path, monkeypatch):
        ex, dl = setup(tmp_path, [rec('00-0000001')], [rec('00-0000001')])
        written = []
        # Never write into the repository, even if the guard were broken.
        monkeypatch.setattr(rb, '_write', lambda *a, **k: written.append(a))
        with pytest.raises(rb.ReferenceBuildError, match="overlaps protected"):
            rb.build(ex, dl, ROOT / 'data_files' / 'reference_candidate')
        assert written == []

    def test_output_inside_a_protected_location_refused(self, tmp_path):
        ex, dl = setup(tmp_path, [rec('00-0000001')], [rec('00-0000001')])
        with pytest.raises(rb.ReferenceBuildError, match="overlaps protected"):
            rb.build(ex, dl, tmp_path / 'prod' / 'out', protected=[tmp_path / 'prod'])
        assert not (tmp_path / 'prod').exists()

    def test_failure_leaves_the_previous_output_unchanged(self, tmp_path):
        ex, dl = setup(tmp_path, [rec('00-0000001')], [rec('00-0000001'), rec('00-0000002')])
        out = tmp_path / 'out'
        rb.build(ex, dl, out)
        before = tree(out)
        write(dl / 'players.csv', [rec('00-0000001'), rec('00-0000001')])   # now duplicate IDs
        b = (dl / 'players.csv').read_bytes()
        (dl / 'manifest.json').write_text(json.dumps({'files': {'players.csv': {
            'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}}}), encoding='utf-8')
        with pytest.raises(rb.ReferenceBuildError):
            rb.build(ex, dl, out)
        assert tree(out) == before
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith('.out.')] == []

    def test_write_failure_leaves_the_previous_output_unchanged(self, tmp_path, monkeypatch):
        ex, dl = setup(tmp_path, [rec('00-0000001')], [rec('00-0000001')])
        out = tmp_path / 'out'
        rb.build(ex, dl, out)
        before = tree(out)

        def boom(*a, **k):
            raise OSError("disk full")
        monkeypatch.setattr(rb, 'sha256_file', lambda p: (_ for _ in ()).throw(OSError("disk full"))
                            if '.staging-' in str(p) else hashlib.sha256(Path(p).read_bytes()).hexdigest())
        with pytest.raises(OSError, match="disk full"):
            rb.build(ex, dl, out)
        assert tree(out) == before
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith('.out.')] == []


class TestFieldChanges:
    def test_meaningful_and_format_only_separated(self, tmp_path):
        ex, dl = setup(tmp_path, [rec('00-0000001', pos='RB', name='Ann Lee', draft='2023.0')],
                       [rec('00-0000001', pos='FB', name='Annie Lee', draft='2023')])
        r = rb.build(ex, dl, tmp_path / 'out')
        fc = r['field_changes']
        assert fc['position']['meaningful'] == 1 and fc['display_name']['meaningful'] == 1
        assert fc['draft_year'] == {'meaningful': 0, 'format_only': 1, 'consumers': []}
        assert fc['position']['consumers'] and fc['display_name']['consumers']
        assert 'short_name' not in fc                                   # "A.Lee" either way
        assert r['eligibility_flips'] == [{'gsis_id': '00-0000001', 'old': 'RB', 'new': 'FB'}]

    def test_headshot_template_fill_is_format_only(self, tmp_path):
        cols = COLUMNS + ['headshot']
        url = 'https://static.www.nfl.com/image/private/{}/league/abc123'
        _, dl = setup(tmp_path, [], [dict(rec('00-0000001'), headshot=url.format('f_auto,q_auto')),
                                     dict(rec('00-0000002'), headshot=url.format('f_auto,q_auto').replace('abc', 'xyz'))],
                      fresh_columns=cols)
        ex = write(tmp_path / 'existing.csv', [dict(rec('00-0000001'), headshot=url.format('{formatInstructions}')),
                                               dict(rec('00-0000002'), headshot=url.format('{formatInstructions}'))], cols)
        r = rb.build(ex, dl, tmp_path / 'out')
        assert r['field_changes']['headshot'] == {'meaningful': 1, 'format_only': 1, 'consumers': []}

    def test_corrected_placeholder_changes_are_attributed(self, tmp_path):
        ex, dl = setup(tmp_path, [rec('ABC123456', esb='ABC123456', smart='s-1', team='AAA')],
                       [rec('00-0040001', esb='ABC123456', smart='s-1', team='BBB')])
        rb.build(ex, dl, tmp_path / 'out')
        ch = pd.read_csv(tmp_path / 'out' / 'field_changes.csv', dtype=str)
        row = ch[ch['field'] == 'latest_team'].iloc[0]
        assert (row['old_id'], row['gsis_id'], row['identity']) == ('ABC123456', '00-0040001', 'corrected_placeholder')


def roster_rows(week, rows):
    return [{'season': 2026, 'week': week, 'gsis_id': g, 'team': t, 'position': p, 'status': s,
             'depth_chart_position': p} for g, t, p, s in rows]


class TestRosterEvidence:
    def candidates(self, tmp_path, rows):
        return write(tmp_path / 'cands.csv', rows, ['player_id', 'team'])

    def test_traded_injured_reserve_and_absent_players_are_retained(self, tmp_path):
        ids = ['00-0000001', '00-0000002', '00-0000003', '00-0000004', '00-0000005']
        roster = roster_rows(7, [('00-0000001', 'BBB', 'WR', 'ACT'),      # traded AAA -> BBB
                                 ('00-0000002', 'AAA', 'WR', 'RES'),      # reserve
                                 ('00-0000003', 'AAA', 'WR', 'INA')])     # inactive / injured
        ex, dl = setup(tmp_path, [rec(i) for i in ids], [rec(i, status='CUT' if i == ids[4] else 'ACT')
                                                         for i in ids[:2] + ids[3:]], roster)
        c = self.candidates(tmp_path, [{'player_id': i, 'team': 'AAA'} for i in ids])
        r = rb.build(ex, dl, tmp_path / 'out', c, (2026, 7))
        assert set(built(tmp_path / 'out')['gsis_id']) == set(ids)   # nothing filtered
        cr = r['candidates']
        assert cr['in_reference'] == {'before': 5, 'after': 5} and cr['lost_coverage'] == []
        assert cr['roster_team_differs'] == ['00-0000001']
        assert cr['roster_status'] == {'not on roster that week': 2, 'ACT': 1, 'RES': 1, 'INA': 1}

    def test_only_the_named_week_is_evidence(self, tmp_path):
        roster = roster_rows(7, [('00-0000001', 'AAA', 'WR', 'ACT')]) + roster_rows(8, [('00-0000001', 'CCC', 'TE', 'RES')])
        ex, dl = setup(tmp_path, [rec('00-0000001')], [rec('00-0000001')], roster)
        c = self.candidates(tmp_path, [{'player_id': '00-0000001', 'team': 'AAA'}])
        r = rb.build(ex, dl, tmp_path / 'out', c, (2026, 7))
        assert r['roster']['later_weeks_present'] == [8] and r['roster']['rows_other_weeks_not_used'] == 1
        assert r['candidates']['roster_team_differs'] == [] and r['candidates']['roster_position_disagrees'] == []

    def test_no_rows_for_the_week_refused(self, tmp_path):
        ex, dl = setup(tmp_path, [rec('00-0000001')], [rec('00-0000001')],
                       roster_rows(6, [('00-0000001', 'AAA', 'WR', 'ACT')]))
        with pytest.raises(rb.ReferenceBuildError, match="no rows for 2026 week 7"):
            rb.build(ex, dl, tmp_path / 'out', None, (2026, 7))

    def test_eligibility_flip_reports_roster_disagreement_without_resolving(self, tmp_path):
        roster = roster_rows(7, [('00-0000001', 'AAA', 'RB', 'ACT')])
        ex, dl = setup(tmp_path, [rec('00-0000001', pos='RB')], [rec('00-0000001', pos='FB')], roster)
        c = self.candidates(tmp_path, [{'player_id': '00-0000001', 'team': 'AAA'}])
        r = rb.build(ex, dl, tmp_path / 'out', c, (2026, 7))
        cr = r['candidates']
        assert cr['eligibility_lost'] == ['00-0000001'] and cr['roster_position_disagrees'] == ['00-0000001']
        assert cr['candidate_eligibility_flips'][0]['roster_position'] == 'RB'
        assert built(tmp_path / 'out').loc[0, 'position'] == 'FB'     # source value kept, not overridden

    def test_eligible_count_is_explained(self, tmp_path):
        existing = [rec('00-0000001', pos='RB'), rec('00-0000002', pos='FB'), rec('00-0000003', pos='P')]
        fresh = [rec('00-0000001', pos='FB'), rec('00-0000002', pos='RB'), rec('00-0000003', pos='P'),
                 rec('00-0000004', pos='TE'), rec('00-0000005', pos='K')]
        ex, dl = setup(tmp_path, existing, fresh)
        c = self.candidates(tmp_path, [{'player_id': f'00-000000{i}', 'team': 'AAA'} for i in range(1, 6)])
        cr = rb.build(ex, dl, tmp_path / 'out', c)['candidates']
        assert cr['eligible']['before'] == 1 and cr['eligible']['after'] == 2
        assert cr['newly_covered'] == {'total': 2, 'by_identity': {'added': 2}, 'eligible': 1}
        assert cr['eligible']['explained'] == cr['eligible']['after']


def snapshot(root):
    """Every file under root (inputs and previous outputs), by relative path."""
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(Path(root).rglob('*')) if p.is_file()}


def link_dir(link, target):
    """A directory link: a symlink where permitted, else a Windows junction."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return 'symlink'
    except (OSError, NotImplementedError):
        pass
    if sys.platform == 'win32':
        r = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)], capture_output=True)
        if r.returncode == 0:
            return 'junction'
    pytest.skip("neither directory symlinks nor junctions can be created here")


class TestOutputLocation:
    """The output may not overwrite or delete any input, and a refusal changes nothing."""

    def prepared(self, tmp_path):
        work = tmp_path / 'work'
        work.mkdir()
        ex, dl = setup(work, [rec('00-0000001')], [rec('00-0000001'), rec('00-0000002')])
        cands = write(work / 'cands.csv', [{'player_id': '00-0000001', 'team': 'AAA'}], ['player_id', 'team'])
        rb.build(ex, dl, tmp_path / 'prev', cands)          # a previous output that must survive
        return ex, dl, cands

    def refused(self, tmp_path, out, match, cands=None, ex=None, dl=None):
        before = snapshot(tmp_path)
        with pytest.raises(rb.ReferenceBuildError, match=match):
            rb.build(ex, dl, out, cands)
        assert snapshot(tmp_path) == before
        assert not [p for p in tmp_path.rglob('*') if '.staging-' in p.name or '.previous-' in p.name]

    @pytest.mark.parametrize('where', ['equal', 'inside', 'ancestor'])
    def test_overlapping_the_downloads_directory_refused(self, tmp_path, where):
        ex, dl, cands = self.prepared(tmp_path)
        out = {'equal': dl, 'inside': dl / 'out', 'ancestor': tmp_path / 'work'}[where]
        self.refused(tmp_path, out, "overlaps the downloads directory", cands, ex, dl)

    def test_output_that_is_the_existing_reference_refused(self, tmp_path):
        ex, dl, cands = self.prepared(tmp_path)
        self.refused(tmp_path, ex, "would replace input", cands, ex, dl)

    def test_output_containing_the_existing_reference_refused(self, tmp_path):
        ex, dl, cands = self.prepared(tmp_path)
        moved = tmp_path / 'refdir' / 'existing.csv'
        moved.parent.mkdir()
        moved.write_bytes(ex.read_bytes())
        self.refused(tmp_path, tmp_path / 'refdir', "would replace input", cands, moved, dl)

    @pytest.mark.parametrize('where', ['equal', 'inside'])
    def test_output_that_would_remove_the_candidates_file_refused(self, tmp_path, where):
        ex, dl, cands = self.prepared(tmp_path)
        # The candidates file sits in the previous output, which a build would replace.
        kept = tmp_path / 'prev' / 'candidates.csv'
        target = kept if where == 'equal' else tmp_path / 'prev'
        self.refused(tmp_path, target, "would replace input", kept, ex, dl)

    def test_sibling_output_is_allowed(self, tmp_path):
        ex, dl, cands = self.prepared(tmp_path)
        rb.build(ex, dl, tmp_path / 'work' / 'out', cands)  # beside the inputs, overlapping none
        assert (tmp_path / 'work' / 'out' / 'players_candidate.csv').exists()

    @pytest.mark.parametrize('where', ['equal', 'inside'])
    def test_linked_alias_of_the_downloads_directory_refused(self, tmp_path, where):
        ex, dl, cands = self.prepared(tmp_path)
        alias = tmp_path / 'alias'
        link_dir(alias, dl)
        out = alias if where == 'equal' else alias / 'out'
        self.refused(tmp_path, out, "overlaps the downloads directory", cands, ex, dl)

    def test_linked_alias_containing_the_existing_reference_refused(self, tmp_path):
        ex, dl, cands = self.prepared(tmp_path)
        # The reference in its own directory; the output names that directory through a link.
        moved = tmp_path / 'refdir' / 'existing.csv'
        moved.parent.mkdir()
        moved.write_bytes(ex.read_bytes())
        alias = tmp_path / 'alias'
        link_dir(alias, tmp_path / 'refdir')
        self.refused(tmp_path, alias, "would replace input", cands, moved, dl)

    def test_input_named_through_a_link_still_compared_resolved(self, tmp_path):
        ex, dl, cands = self.prepared(tmp_path)
        moved = tmp_path / 'refdir' / 'existing.csv'
        moved.parent.mkdir()
        moved.write_bytes(ex.read_bytes())
        alias = tmp_path / 'alias'
        link_dir(alias, tmp_path / 'refdir')
        # The reference named through the link; the output names the real directory.
        self.refused(tmp_path, tmp_path / 'refdir', "would replace input", cands, alias / 'existing.csv', dl)

    def test_linked_alias_of_the_candidates_file_refused(self, tmp_path):
        ex, dl, cands = self.prepared(tmp_path)
        alias = tmp_path / 'alias'
        link_dir(alias, tmp_path / 'prev')
        kept = tmp_path / 'prev' / 'candidates.csv'
        self.refused(tmp_path, alias, "would replace input", kept, ex, dl)

    def test_downloads_reached_through_a_link_still_compared_resolved(self, tmp_path):
        ex, dl, cands = self.prepared(tmp_path)
        alias = tmp_path / 'alias'
        link_dir(alias, dl)
        # Downloads named through the alias; output is the real directory.
        self.refused(tmp_path, dl / 'out', "overlaps the downloads directory", cands, ex, alias)


class TestBlankValues:
    @pytest.mark.parametrize('field', ['gsis_id', 'position'])
    @pytest.mark.parametrize('value', ['', ' ', '   ', '\t'])
    @pytest.mark.parametrize('which', ['existing', 'fresh'])
    def test_blank_required_value_refused_before_output_replacement(self, tmp_path, field, value, which):
        ex, dl = setup(tmp_path, [rec('00-0000001')], [rec('00-0000001')])
        out = tmp_path / 'out'
        rb.build(ex, dl, out)
        before = snapshot(out)
        bad = rec('00-0000002')
        bad[field] = value
        if which == 'existing':
            write(ex, [rec('00-0000001'), bad])
        else:
            write(dl / 'players.csv', [rec('00-0000001'), bad])
            b = (dl / 'players.csv').read_bytes()
            (dl / 'manifest.json').write_text(json.dumps({'files': {'players.csv': {
                'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}}}), encoding='utf-8')
        label = 'existing reference' if which == 'existing' else 'fresh source'
        with pytest.raises(rb.ReferenceBuildError, match=rf"{label}: 1 record\(s\) with a missing or blank {field}"):
            rb.build(ex, dl, out)
        assert snapshot(out) == before
        assert not [p for p in tmp_path.iterdir() if '.staging-' in p.name or '.previous-' in p.name]

    def test_blank_value_in_the_assembled_candidate_refused(self, tmp_path, monkeypatch):
        ex, dl = setup(tmp_path, [rec('00-0000001')], [rec('00-0000001')])
        out = tmp_path / 'out'
        rb.build(ex, dl, out)
        before = snapshot(out)

        def assemble(fresh, hist, cols):
            c = fresh[cols].reset_index(drop=True)
            c.loc[0, 'position'] = '  '
            return c
        monkeypatch.setattr(rb, '_assemble', assemble)
        with pytest.raises(rb.ReferenceBuildError, match=r"candidate: 1 record\(s\) with a missing or blank position"):
            rb.build(ex, dl, out)
        assert snapshot(out) == before

    def test_two_blank_ids_are_not_reported_as_duplicates(self, tmp_path):
        bad = [rec('00-0000001'), dict(rec('00-0000008'), gsis_id=' '), dict(rec('00-0000009'), gsis_id=' ')]
        ex, dl = setup(tmp_path, [rec('00-0000001')], bad)
        with pytest.raises(rb.ReferenceBuildError) as e:
            rb.build(ex, dl, tmp_path / 'out')
        assert e.value.problems == ["fresh source: 2 record(s) with a missing or blank gsis_id"]
        assert not (tmp_path / 'out').exists()

    def test_nonidentity_text_kept_exactly(self, tmp_path):
        # Valid IDs and positions; other fields with odd spacing are kept as written.
        fresh = [rec('00-0000001'), dict(rec('00-0000002'), display_name=' Name  Two ', latest_team='AAA ')]
        hist = dict(rec('00-0000003'), short_name='  N.Three', draft_year='2014.0')
        ex, dl = setup(tmp_path, [rec('00-0000001'), hist], fresh)
        rb.build(ex, dl, tmp_path / 'out')
        out = built(tmp_path / 'out').set_index('gsis_id')
        assert out.loc['00-0000002', 'display_name'] == ' Name  Two '
        assert out.loc['00-0000002', 'latest_team'] == 'AAA '
        assert out.loc['00-0000003', 'short_name'] == '  N.Three'      # retained historical record
        assert out.loc['00-0000003', 'draft_year'] == '2014.0'


def manifest_for(dl):
    files = {}
    for f in dl.iterdir():
        if f.name != 'manifest.json':
            b = f.read_bytes()
            files[f.name] = {'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}
    (dl / 'manifest.json').write_text(json.dumps({'files': files}), encoding='utf-8')


class TestPaddedAndMalformed:
    """Padded or malformed required values are refused, never trimmed, and a
    refusal leaves every input and the previous output byte-identical."""

    def refused_after_good_build(self, tmp_path, existing, fresh, match):
        ex, dl = setup(tmp_path, [rec('00-0000001')], [rec('00-0000001')])
        out = tmp_path / 'out'
        rb.build(ex, dl, out)
        write(ex, existing)
        write(dl / 'players.csv', fresh)
        manifest_for(dl)
        before = snapshot(tmp_path)
        with pytest.raises(rb.ReferenceBuildError, match=match) as e:
            rb.build(ex, dl, out)
        assert snapshot(tmp_path) == before
        assert not [p for p in tmp_path.iterdir() if '.staging-' in p.name or '.previous-' in p.name]
        return e.value.problems

    @pytest.mark.parametrize('which', ['existing', 'fresh'])
    @pytest.mark.parametrize('pid', [' 00-0000002', '00-0000002 ', '\t00-0000002', '00-0000002\n'])
    def test_padded_gsis_id_refused(self, tmp_path, which, pid):
        rows = [rec('00-0000001'), dict(rec('00-0000002'), gsis_id=pid)]
        label = 'existing reference' if which == 'existing' else 'fresh source'
        problems = self.refused_after_good_build(
            tmp_path, rows if which == 'existing' else [rec('00-0000001')],
            rows if which == 'fresh' else [rec('00-0000001')], rf"{label}: 1 gsis_id\(s\) with surrounding whitespace")
        assert repr(pid) in problems[0]

    @pytest.mark.parametrize('which', ['existing', 'fresh'])
    @pytest.mark.parametrize('pos', [' RB ', ' RB', 'RB ', '\tWR'])
    def test_padded_position_refused(self, tmp_path, which, pos):
        rows = [rec('00-0000001'), rec('00-0000002', pos=pos)]
        label = 'existing reference' if which == 'existing' else 'fresh source'
        self.refused_after_good_build(
            tmp_path, rows if which == 'existing' else [rec('00-0000001')],
            rows if which == 'fresh' else [rec('00-0000001')], rf"{label}: 1 position\(s\) with surrounding whitespace")

    @pytest.mark.parametrize('which', ['existing', 'fresh'])
    def test_canonical_and_padded_spelling_together_refused(self, tmp_path, which):
        # Distinct strings, so not duplicates as written: refused for the padding.
        rows = [rec('00-0000001'), dict(rec('00-0000001', pos='RB'), gsis_id='00-0000001 ')]
        label = 'existing reference' if which == 'existing' else 'fresh source'
        problems = self.refused_after_good_build(
            tmp_path, rows if which == 'existing' else [rec('00-0000001')],
            rows if which == 'fresh' else [rec('00-0000001')], rf"{label}: 1 gsis_id\(s\) with surrounding whitespace")
        assert not any('more than one record' in p for p in problems)

    @pytest.mark.parametrize('pid', ['0-0000002', '00-000002', '00-00000002', '00_0000002', 'abc123456',
                                     'AB1234567', 'ABCD12345', 'ABC12345', 'ABC1234567', 'ABC123456X',
                                     'player-2', '00-0000002x'])
    @pytest.mark.parametrize('which', ['existing', 'fresh'])
    def test_malformed_id_refused(self, tmp_path, which, pid):
        rows = [rec('00-0000001'), dict(rec('00-0000002', esb=pid), gsis_id=pid)]
        label = 'existing reference' if which == 'existing' else 'fresh source'
        problems = self.refused_after_good_build(
            tmp_path, rows if which == 'existing' else [rec('00-0000001')],
            rows if which == 'fresh' else [rec('00-0000001')], rf"{label}: 1 gsis_id\(s\) neither GSIS")
        assert repr(pid) in problems[0]

    @pytest.mark.parametrize('which', ['existing', 'fresh'])
    @pytest.mark.parametrize('esb', ['XYZ999999', None])
    def test_placeholder_shape_without_matching_esb_id_refused(self, tmp_path, which, esb):
        rows = [rec('00-0000001'), rec('ABC123456', esb=esb)]
        label = 'existing reference' if which == 'existing' else 'fresh source'
        self.refused_after_good_build(
            tmp_path, rows if which == 'existing' else [rec('00-0000001')],
            rows if which == 'fresh' else [rec('00-0000001')], rf"{label}: 1 gsis_id\(s\) neither GSIS")

    def test_supported_placeholders_kept_in_existing_and_fresh(self, tmp_path):
        ex, dl = setup(tmp_path, [rec('00-0000001'), rec('ABC123456', esb='ABC123456', smart='s-a')],
                       [rec('00-0000001'), rec('DEF654321', esb='DEF654321', smart='s-d')])
        r = rb.build(ex, dl, tmp_path / 'out')
        assert set(built(tmp_path / 'out')['gsis_id']) == {'00-0000001', 'ABC123456', 'DEF654321'}
        assert r['identities']['retained_historical']['records'] == 1   # the existing placeholder
        assert r['identities']['added']['records'] == 1                 # the fresh one

    def test_id_kinds(self):
        df = pd.DataFrame([rec('00-0000001'), rec('ABC123456', esb='ABC123456'), rec('ABC123456', esb='ABC000000'),
                           rec('00-000001'), rec('abc123456', esb='abc123456')])
        assert rb.id_kind(df).tolist() == ['gsis', 'placeholder', 'malformed', 'malformed', 'malformed']

    def test_padded_value_in_the_assembled_candidate_refused(self, tmp_path, monkeypatch):
        ex, dl = setup(tmp_path, [rec('00-0000001')], [rec('00-0000001')])
        out = tmp_path / 'out'
        rb.build(ex, dl, out)
        before = snapshot(tmp_path)

        def assemble(fresh, hist, cols):
            c = fresh[cols].reset_index(drop=True)
            c.loc[0, 'position'] = ' WR '
            return c
        monkeypatch.setattr(rb, '_assemble', assemble)
        with pytest.raises(rb.ReferenceBuildError, match=r"candidate: 1 position\(s\) with surrounding whitespace"):
            rb.build(ex, dl, out)
        assert snapshot(tmp_path) == before
