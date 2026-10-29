"""Prop eligibility by position_group (policy position-group-v1).

Synthetic fixtures only (the rehearsal's synthetic league); no network, no
real data. Production's rule (predict.eligibility_group) and the rehearsal's
restatement (rehearsal_expected.restated_group) are separate implementations;
these tests check each and that they agree."""
import json
import shutil
import socket
from pathlib import Path

import pandas as pd
import pytest

from player_props import predict
from player_props import rehearsal as rh
from player_props import rehearsal_expected as rx
from player_props import rehearsal_fixtures as fixtures

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


# ---------------------------------------------------------------------------
# The rule, in production and restated
# ---------------------------------------------------------------------------

CASES = [
    # (position, position_group, eligible group or None, kind of problem or None)
    ('RB', 'RB', 'RB', None),
    ('FB', 'RB', 'RB', None),
    ('TE', 'TE', 'TE', None),
    ('WR', 'WR', 'WR', None),
    ('QB', 'QB', 'QB', None),
    ('K', 'SPEC', 'SPEC', None),                 # usable, not eligible
    ('DT', 'DL', 'DL', None),
    ('FB', None, None, 'no_group'),              # missing
    ('RB', '', None, 'no_group'),
    ('RB', '   ', None, 'no_group'),             # blank
    ('RB', ' RB', None, 'padded_group'),
    ('FB', 'RB ', None, 'padded_group'),
    ('RB ', 'RB', None, 'padded_label'),
    (None, 'RB', None, 'no_label'),
    ('TE', 'RB', None, 'clash'),                 # label/group contradiction
    ('FB', 'TE', None, 'clash'),
    ('RB', 'WR', None, 'clash'),
    ('DT', 'RB', None, 'eligible_other'),        # non-skill label with an eligible group
    ('K', 'QB', None, 'eligible_other'),
]
# What each implementation says for each kind of problem (written separately).
PROD_TEXT = {'no_group': 'missing position_group', 'padded_group': 'position_group',
             'padded_label': 'position ', 'no_label': 'missing position', 'clash': '(expected',
             'eligible_other': 'with eligible position_group'}
RESTATED_TEXT = {'no_group': 'no position_group', 'padded_group': 'position_group', 'padded_label': 'position ',
                 'no_label': 'no position', 'clash': ', not ', 'eligible_other': 'has the eligible position_group'}


class TestRule:
    @pytest.mark.parametrize('position, group, want, problem', CASES)
    def test_production_rule(self, position, group, want, problem):
        got, why = predict.eligibility_group(position, group)
        assert got == want and (why is None) == (problem is None)
        if problem:
            assert why.startswith(PROD_TEXT[problem]) if problem.startswith('padded') else PROD_TEXT[problem] in why, why
        if problem and problem.startswith('padded'):
            assert 'surrounding whitespace' in why

    @pytest.mark.parametrize('position, group, want, problem', CASES)
    def test_restated_rule_agrees(self, position, group, want, problem):
        got, why = rx.restated_group(position, group)
        assert got == want and (why is None) == (problem is None)
        if problem:
            assert why.startswith(RESTATED_TEXT[problem]) if problem.startswith('padded') else RESTATED_TEXT[problem] in why, why
        if problem and problem.startswith('padded'):
            assert why.endswith('is padded')

    def test_restatement_refuses_a_reference_without_the_group_column(self):
        players = pd.DataFrame([dict(gsis_id='00-0000001', position='RB')])
        tables = {s: pd.DataFrame() for s in ('passing', 'rushing', 'receiving')}
        with pytest.raises(rx.ExpectedSetError, match='no position_group'):
            rx.expected_rows(Path('.'), tables, {}, None, 2026, 7, None, players)

    def test_fb_and_rb_get_the_same_prop_families(self):
        fb, _ = predict.eligibility_group('FB', 'RB')
        rb, _ = predict.eligibility_group('RB', 'RB')
        assert predict.POSITION_MAP[fb] == predict.POSITION_MAP[rb] == [
            'rushing_yards', 'rushing_tds', 'receptions']

    def test_te_stays_distinct_from_fb(self):
        te, _ = predict.eligibility_group('TE', 'TE')
        fb, _ = predict.eligibility_group('FB', 'RB')
        assert te != fb and predict.POSITION_MAP[te] != predict.POSITION_MAP[fb]

    def test_policy_versions_match(self):
        assert predict.ELIGIBILITY_POLICY == rx.RESTATED_POLICY == 'position-group-v1'
        assert predict.ELIGIBILITY_FIELD == 'position_group'

    def test_restatement_does_not_import_the_production_rule(self):
        import ast
        src = (ROOT / 'player_props' / 'rehearsal_expected.py').read_text(encoding='utf-8')
        from_predict = [a.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)
                        and n.module == 'player_props.predict' for a in n.names]
        # Only the shared prop-family and tier constants; the group rule is restated.
        assert sorted(from_predict) == ['POSITION_MAP', 'TIER_RULES']


def write_players(tmp_path, rows, columns=None):
    df = pd.DataFrame(rows)
    if columns is not None:
        df = df[columns]
    path = tmp_path / 'players.csv'
    df.to_csv(path, index=False)
    return path


def player(pid, pos, group, name='Pat Player'):
    return dict(gsis_id=pid, short_name='P.Player', display_name=name, last_name='Player',
                position=pos, position_group=group)


class TestProductionLookup:
    @pytest.fixture
    def data_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(predict, 'DATA_DIR', tmp_path)
        return tmp_path

    def test_display_label_kept_and_group_selects(self, data_dir):
        write_players(data_dir, [player('00-0000001', 'FB', 'RB', 'Full Back'), player('00-0000002', 'RB', 'RB')])
        fb = predict.get_player_identity('00-0000001')
        assert (fb['position'], fb['group'], fb['display_name'], fb['problem']) == ('FB', 'RB', 'Full Back', None)
        assert predict.get_player_position('00-0000001') == ('FB', 'Full Back')     # label, not relabelled

    @pytest.mark.parametrize('group, fragment', [(None, 'missing'), (' RB', 'whitespace'), ('TE', "expected 'RB'")])
    def test_unusable_group_gets_no_group_and_says_why(self, data_dir, group, fragment):
        write_players(data_dir, [player('00-0000001', 'FB', group)])
        ident = predict.get_player_identity('00-0000001')
        assert ident['group'] is None and fragment in ident['problem']
        assert ident['position'] == 'FB'                               # never replaced by a group

    def test_reference_without_the_group_column_is_refused(self, data_dir):
        write_players(data_dir, [player('00-0000001', 'RB', 'RB')],
                      ['gsis_id', 'short_name', 'display_name', 'last_name', 'position'])
        with pytest.raises(ValueError, match=r"lacks column\(s\) \['position_group'\]"):
            predict.load_players()

    def test_repeated_or_unknown_ids(self, data_dir):
        write_players(data_dir, [player('00-0000001', 'RB', 'RB'), player('00-0000001', 'RB', 'RB')])
        assert predict.get_player_identity('00-0000001')['problem'] == 'several players-reference records'
        assert predict.get_player_identity('00-0000009')['problem'] == 'not in the players reference'


# ---------------------------------------------------------------------------
# End to end: the synthetic rehearsal under the policy
# ---------------------------------------------------------------------------

def tree(d):
    d = Path(d)
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob('*')) if p.is_file()} if d.exists() else {}


def relabelled_players(label='FB'):
    base = fixtures.players()
    return lambda: base.assign(position=base['position'].replace({'RB': label}))


def run(ws, hook=None, players=None):
    with pytest.MonkeyPatch.context() as mp:
        if players is not None:
            mp.setattr(fixtures, 'players', players)
        return rh.run_rehearsal(rh.RehearsalConfig(workspace=ws), after_stage=hook)


def feed(ws):
    return rh.read_predictions(Path(ws).resolve() / 'published' / 'player_props_predictions.csv')


def stage_status(s):
    return {x['stage']: x['status'] for x in s['stages']}


@pytest.fixture(scope='module')
def baseline(tmp_path_factory):
    ws = tmp_path_factory.mktemp('pg_base') / 'ws'
    return ws, run(ws)


@pytest.fixture(scope='module')
def fullbacks(tmp_path_factory):
    """The same league with every RB labelled FB (position_group still RB)."""
    ws = tmp_path_factory.mktemp('pg_fb') / 'ws'
    return ws, run(ws, players=relabelled_players('FB'))


COMPARE = ['player_id', 'team', 'opponent', 'prop_type', 'line_type', 'line_value', 'model_name',
           'raw_model_prob_over', 'prob_over', 'confidence', 'recommendation', 'adjustments', 'eligibility_group']


class TestEndToEnd:
    def test_baseline_succeeds_and_records_provenance(self, baseline):
        ws, s = baseline
        assert s['status'] == 'succeeded' and s['completeness']['status'] == 'complete'
        e = s['eligibility']
        ref = Path(ws).resolve() / 'runs' / s['run_id'] / 'runtime' / 'data' / 'players.csv'
        assert e['policy'] == e['restated_policy'] == 'position-group-v1' and e['field'] == 'position_group'
        assert e['players_reference_sha256'] == rh._sha256_file(ref)
        assert e['players_reference_source'] == 'verification' and e['verified'] is True
        assert e['players_reference_current_sha256'] == e['players_reference_sha256']
        assert e['players_reference_current_unavailable'] is None
        verify = next(x for x in s['stages'] if x['stage'] == 'verify_outputs')
        assert s['outputs']['verify_outputs']['players_reference_sha256'] == e['players_reference_sha256']
        assert verify['status'] == 'ok'
        saved = json.loads((Path(ws).resolve() / 'runs' / s['run_id'] / 'summary.json').read_text())
        assert saved['eligibility'] == e

    def test_fb_labels_select_the_same_rows_as_rb_labels(self, baseline, fullbacks):
        (ws_b, sb), (ws_f, sf) = baseline, fullbacks
        assert sf['status'] == 'succeeded' and sf['completeness']['status'] == 'complete'
        assert sf['completeness']['expected'] == sb['completeness']['expected']
        b, f = feed(ws_b), feed(ws_f)
        key = ['player_id', 'prop_type']
        b, f = b.sort_values(key).reset_index(drop=True), f.sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(b[COMPARE], f[COMPARE])
        rb_rows = b['position'] == 'RB'
        assert rb_rows.any()
        assert set(f.loc[rb_rows, 'position']) == {'FB'}               # display label kept, not relabelled
        assert set(f.loc[rb_rows, 'eligibility_group']) == {'RB'}
        assert (f.loc[~rb_rows, 'position'] == b.loc[~rb_rows, 'position']).all()

    def test_te_rows_keep_te_families(self, fullbacks):
        f = feed(fullbacks[0])
        te = f[f['position'] == 'TE']
        assert not te.empty and set(te['eligibility_group']) == {'TE'}
        assert set(te['prop_type']) <= set(predict.POSITION_MAP['TE'])
        fb = f[f['position'] == 'FB']
        assert set(fb['prop_type']) <= set(predict.POSITION_MAP['RB'])


def both(change):
    def tamper(name, r):
        if name == 'predict':
            for p in (r.info['latest'], r.info['snapshot']):
                df = rh.read_predictions(p)
                change(df)
                df.to_csv(p, index=False)
    return tamper


@pytest.fixture
def published_copy(fullbacks, tmp_path):
    ws, _ = fullbacks
    copy = tmp_path / 'ws'
    shutil.copytree(Path(ws).resolve() / 'published', copy / 'published')
    return copy, tree(copy / 'published')


def blocked(s, ws, before, text):
    st = stage_status(s)
    assert s['status'] == 'failed' and st['verify_outputs'] == 'failed' and st['publish'] == 'not_run'
    err = next(x for x in s['stages'] if x['stage'] == 'verify_outputs')['error']
    assert text in json.dumps(err), err
    assert tree(Path(ws) / 'published') == before


class TestVerificationUnderThePolicy:
    def test_omitted_fullback_row_is_caught(self, published_copy):
        ws, before = published_copy

        def drop_fb(df):
            df.drop(index=df.index[df['position'] == 'FB'][:1], inplace=True)
        s = run(ws, both(drop_fb), players=relabelled_players('FB'))
        blocked(s, ws, before, 'expected prediction(s) missing')
        assert s['completeness']['latest feed']['missing'] == 1

    def test_wrong_recorded_group_is_rejected(self, published_copy):
        ws, before = published_copy

        def regroup(df):
            df.loc[df.index[df['position'] == 'FB'][:1], 'eligibility_group'] = 'WR'
        s = run(ws, both(regroup), players=relabelled_players('FB'))
        blocked(s, ws, before, "eligibility_group 'WR' is not")

    def test_missing_recorded_group_is_rejected(self, published_copy):
        ws, before = published_copy
        s = run(ws, both(lambda df: df.drop(columns='eligibility_group', inplace=True)),
                players=relabelled_players('FB'))
        blocked(s, ws, before, 'no eligibility_group recorded')

    def test_relabelled_position_is_rejected(self, published_copy):
        ws, before = published_copy

        def relabel(df):
            df.loc[df['position'] == 'FB', 'position'] = 'RB'         # the label must stay the source's
        s = run(ws, both(relabel), players=relabelled_players('FB'))
        blocked(s, ws, before, "position 'RB' is not")

    def test_inconsistent_reference_group_fails_closed(self, published_copy):
        ws, before = published_copy
        base = fixtures.players()
        bad = lambda: base.assign(position_group=base['position_group'].where(base['position'] != 'TE', 'RB'))
        s = run(ws, players=bad)
        st = stage_status(s)
        assert s['status'] == 'failed' and st['verify_outputs'] == 'failed' and st['publish'] == 'not_run'
        assert s['completeness']['status'] == 'expected set not derived'
        assert any("position 'TE' has position_group 'RB'" in p for p in s['completeness']['problems'])
        assert tree(Path(ws) / 'published') == before

    def test_policy_version_mismatch_blocks_verification(self, published_copy, monkeypatch):
        ws, before = published_copy
        monkeypatch.setattr(rx, 'RESTATED_POLICY', 'position-group-v0')
        s = run(ws, players=relabelled_players('FB'))
        blocked(s, ws, before, "differs from the restated policy 'position-group-v0'")
        assert s['eligibility']['policy'] == 'position-group-v1'
        assert s['eligibility']['restated_policy'] == 'position-group-v0'
        # What verification read is recorded, marked as not verified.
        assert s['eligibility']['players_reference_source'] == 'verification'
        assert s['eligibility']['verified'] is False

    def test_reference_without_group_column_fails_closed(self, published_copy):
        ws, before = published_copy
        base = fixtures.players()
        s = run(ws, players=lambda: base.drop(columns='position_group'))
        assert s['status'] == 'failed'
        assert stage_status(s)['publish'] == 'not_run'
        assert 'position_group' in json.dumps([x.get('error') for x in s['stages']])
        assert tree(Path(ws) / 'published') == before


def runtime_reference(r):
    return r.paths['runtime_data'] / 'players.csv'


def make_unreadable(path):
    """A directory where the reference was: it exists but can't be read as a file."""
    path.unlink()
    path.mkdir()


class TestVerifiedProvenance:
    """The reference hash verification read is the one recorded, and the
    reference must still have it when publishing."""

    def test_reference_changed_after_verification_refuses_publication(self, published_copy):
        ws, before = published_copy

        def edit(name, r):
            if name == 'verify_outputs':
                with open(runtime_reference(r), 'a', encoding='utf-8') as f:
                    f.write('\n')
        s = run(ws, edit, players=relabelled_players('FB'))
        st = stage_status(s)
        assert s['status'] == 'failed' and st['verify_outputs'] == 'ok' and st['publish'] == 'failed'
        err = next(x for x in s['stages'] if x['stage'] == 'publish')['error']['message']
        assert 'the players reference changed after verification' in err and 'nothing was published' in err
        assert tree(Path(ws) / 'published') == before
        e = s['eligibility']
        assert e['verified'] is True and e['players_reference_source'] == 'verification'
        assert e['players_reference_sha256'] == s['outputs']['verify_outputs']['players_reference_sha256']
        assert e['players_reference_current_sha256'] not in (None, e['players_reference_sha256'])

    def test_reference_unreadable_before_publication_refuses(self, published_copy):
        ws, before = published_copy
        s = run(ws, lambda name, r: name == 'verify_outputs' and make_unreadable(runtime_reference(r)),
                players=relabelled_players('FB'))
        st = stage_status(s)
        assert st['verify_outputs'] == 'ok' and st['publish'] == 'failed'
        err = next(x for x in s['stages'] if x['stage'] == 'publish')['error']['message']
        assert 'the players reference became unreadable after verification' in err
        assert tree(Path(ws) / 'published') == before
        e = s['eligibility']
        assert e['players_reference_sha256'] == s['outputs']['verify_outputs']['players_reference_sha256']
        assert e['players_reference_current_sha256'] is None
        assert e['players_reference_current_unavailable'].startswith("runtime/data/players.csv can't be read (")

    def test_reference_changed_during_verification_is_refused(self, published_copy, monkeypatch):
        ws, before = published_copy
        real = rh._Expectations.__init__

        def init(self, r, *a, **k):
            with open(runtime_reference(r), 'a', encoding='utf-8') as f:
                f.write('\n')
            real(self, r, *a, **k)
        monkeypatch.setattr(rh._Expectations, '__init__', init)
        s = run(ws, players=relabelled_players('FB'))
        blocked(s, ws, before, 'the players reference changed during verification')
        e = s['eligibility']
        assert e['verified'] is False and e['players_reference_source'] == 'verification'
        assert e['players_reference_current_sha256'] != e['players_reference_sha256']

    def test_unreadable_after_publication_keeps_the_verified_hash(self, tmp_path):
        ws = tmp_path / 'ws'
        s = run(ws, lambda name, r: name == 'publish' and make_unreadable(runtime_reference(r)),
                players=relabelled_players('FB'))
        # Published before the reference went; the status isn't changed retroactively.
        assert s['status'] == 'succeeded' and stage_status(s)['publish'] == 'ok'
        assert (Path(ws).resolve() / 'published' / 'player_props_predictions.csv').is_file()
        e = s['eligibility']
        assert e['verified'] is True and e['players_reference_source'] == 'verification'
        assert e['players_reference_sha256'] == s['outputs']['verify_outputs']['players_reference_sha256']
        assert e['players_reference_unavailable'] is None
        assert e['players_reference_current_sha256'] is None
        assert e['players_reference_current_unavailable'].startswith("runtime/data/players.csv can't be read (")
        saved = json.loads((Path(ws).resolve() / 'runs' / s['run_id'] / 'summary.json').read_text())
        assert saved['eligibility'] == e


class TestHistoricalFiles:
    def test_files_without_eligibility_group_stay_readable(self, fullbacks, tmp_path):
        old = feed(fullbacks[0]).drop(columns='eligibility_group')
        path = tmp_path / 'player_props_predictions_week3_2026.csv'
        old.to_csv(path, index=False)
        raw = path.read_bytes()
        back = rh.read_predictions(path)
        assert len(back) == len(old) and 'eligibility_group' not in back.columns
        from player_props import reference_join as rj
        enriched = rj.enrich(back, None)
        assert len(enriched) == len(back) and enriched['position'].tolist() == back['position'].tolist()
        assert path.read_bytes() == raw                                # never rewritten
