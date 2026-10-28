"""Stable-ID player-reference joins in the Player Props and Parlay Builder pages
(player_props/reference_join.py). Synthetic fixtures only; data_files/ is never read."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from player_props import reference_join as rj

ROOT = Path(__file__).resolve().parents[1]

# Two players share "J.Smith"; D4 is on two records; T.Solo is unique.
REFERENCE = pd.DataFrame([
    dict(gsis_id='A1', short_name='J.Smith', display_name='John Smith', last_name='Smith', position='WR'),
    dict(gsis_id='B2', short_name='J.Smith', display_name='Jake Smith', last_name='Smith', position='RB'),
    dict(gsis_id='C3', short_name='T.Solo', display_name='Tom Solo', last_name='Solo', position='QB'),
    dict(gsis_id='D4', short_name='D.Dup', display_name='Dan Dup', last_name='Dup', position='TE'),
    dict(gsis_id='D4', short_name='D.Dupe', display_name='Dan Dupe', last_name='Dupe', position='WR'),
    dict(gsis_id='00-0012345', short_name='Z.Zero', display_name='Zed Zero', last_name='Zero', position='TE'),
    # Same display name AND position: only the ID tells them apart.
    dict(gsis_id='F6', short_name='M.Williams', display_name='Mike Williams', last_name='Williams', position='WR'),
    dict(gsis_id='G7', short_name='M.Williams', display_name='Mike Williams', last_name='Williams', position='WR'),
])


def ref(df=REFERENCE):
    return rj.build_reference(df)


class TestReference:
    def test_duplicate_ids_reported_and_kept(self):
        r = ref()
        assert r.duplicate_ids == {'D4'}
        assert len(r.records) == len(REFERENCE)                       # nothing silently dropped
        assert any("appear on more than one" in p and 'D4' in p for p in r.problems)
        assert 'D4' not in r.by_id.index and {'A1', 'B2', 'C3'} <= set(r.by_id.index)

    def test_unique_names_exclude_shared_names_and_duplicate_ids(self):
        names = ref().unique_names
        assert 'T.Solo' in names.index and 'J.Smith' not in names.index
        assert 'D.Dup' not in names.index                              # its ID is ambiguous

    def test_ids_read_as_text(self, tmp_path):
        p = tmp_path / 'players.csv'
        REFERENCE.to_csv(p, index=False)
        r = rj.load_reference(p)
        assert '00-0012345' in r.by_id.index
        num = tmp_path / 'num.csv'
        REFERENCE.assign(gsis_id=['0012'] + [str(20 + i) for i in range(len(REFERENCE) - 1)]).to_csv(num, index=False)
        assert '0012' in rj.load_reference(num).by_id.index            # leading zeros kept

    def test_missing_columns_reported_not_raised(self):
        r = rj.build_reference(REFERENCE.drop(columns=['position']))
        assert r.by_id.empty and "lacks column(s) ['position']" in r.problems[0]

    def test_unreadable_file_reported_not_raised(self, tmp_path):
        p = tmp_path / 'players.csv'
        p.write_bytes(b'')
        r = rj.load_reference(p)
        assert r is not None and "could not be read" in r.problems[0]

    def test_missing_file_is_none(self, tmp_path):
        assert rj.load_reference(tmp_path / 'absent.csv') is None

    def test_blank_ids_reported(self):
        r = rj.build_reference(pd.concat([REFERENCE, pd.DataFrame([dict(
            gsis_id=' ', short_name='B.Blank', display_name='Bo Blank', last_name='Blank', position='WR')])]))
        assert any("no gsis_id" in p for p in r.problems) and 'B.Blank' not in r.unique_names.index


def predictions(**extra):
    df = pd.DataFrame({
        'player_id': ['A1', 'B2', 'Z9', 'D4', 'C3'],
        'player_name': ['J.Smith', 'J.Smith', 'T.Solo', 'D.Dup', 'T.Solo'],
        'prob_over': [0.61, 0.38, 0.70, 0.66, 0.57],
        'team': ['AAA', 'AAA', 'BBB', 'CCC', 'DDD'],
    }, index=[40, 10, 30, 20, 50])
    return df.assign(**extra)


class TestEnrich:
    def test_namesakes_get_only_their_own_details(self):
        out = rj.enrich(predictions(), ref())
        assert out.loc[40, ['display_name', 'position']].tolist() == ['John Smith', 'WR']
        assert out.loc[10, ['display_name', 'position']].tolist() == ['Jake Smith', 'RB']
        assert out.loc[[40, 10], 'reference_match'].tolist() == ['id', 'id']

    def test_unknown_id_never_takes_a_namesakes_details(self):
        out = rj.enrich(predictions(), ref())
        # Z9 is named "T.Solo", the unique short_name of C3: still not enriched.
        assert out.loc[30, 'reference_match'] == 'unknown_id'
        assert out.loc[30, 'display_name'] == 'T.Solo' and pd.isna(out.loc[30, 'position'])
        assert out.loc[50, ['display_name', 'position']].tolist() == ['Tom Solo', 'QB']

    def test_duplicate_reference_id_not_selected(self):
        out = rj.enrich(predictions(), ref())
        assert out.loc[20, 'reference_match'] == 'duplicate_reference_id'
        assert out.loc[20, 'display_name'] == 'D.Dup' and pd.isna(out.loc[20, 'position'])

    def test_row_count_order_and_values_preserved(self):
        before = predictions()
        out = rj.enrich(before, ref())
        assert out.index.tolist() == before.index.tolist()
        pd.testing.assert_frame_equal(out[before.columns], before)
        assert before.columns.tolist() == ['player_id', 'player_name', 'prob_over', 'team']   # input untouched

    def test_stored_values_never_overwritten(self):
        df = predictions(display_name=['Stored A', None, '', 'Stored D', 'Stored C'],
                         position=['XX', None, None, None, None])
        out = rj.enrich(df, ref())
        assert out['display_name'].tolist() == ['Stored A', 'Jake Smith', 'T.Solo', 'Stored D', 'Stored C']
        assert out['position'].tolist()[:2] == ['XX', 'RB']

    def test_legacy_rows_use_a_unique_name_only(self):
        legacy = pd.DataFrame({'player_name': ['T.Solo', 'J.Smith', 'X.None', 'D.Dup', None],
                               'prob_over': [0.6, 0.6, 0.6, 0.6, 0.6]})
        out = rj.enrich(legacy, ref())
        # D.Dup is one record's name, but that record's ID is repeated: still not used.
        assert out['reference_match'].tolist() == ['name_unique', 'ambiguous_name', 'name_not_found',
                                                   'duplicate_reference_id', 'no_identity']
        assert out['display_name'].tolist()[:4] == ['Tom Solo', 'J.Smith', 'X.None', 'D.Dup']
        assert out.loc[0, 'position'] == 'QB' and out.loc[1:, 'position'].isna().all()
        assert len(out) == 5

    def test_blank_ids_are_legacy_rows(self):
        df = pd.DataFrame({'player_id': ['A1', None, ' ', np.nan], 'player_name': ['J.Smith', 'T.Solo', 'J.Smith', 'Q.Q']})
        out = rj.enrich(df, ref())
        assert out['reference_match'].tolist() == ['id', 'name_unique', 'ambiguous_name', 'name_not_found']

    def test_season_totals_cannot_multiply(self):
        # Namesakes on one team, and the reference has the shared short_name twice.
        totals = pd.DataFrame({'player_id': ['A1', 'B2'], 'player_name': ['J.Smith', 'J.Smith'],
                               'team': ['AAA', 'AAA'], 'receiving_yards': [500, 120]})
        out = rj.enrich(totals, ref(), fields=('position',))
        assert len(out) == 2 and out['position'].tolist() == ['WR', 'RB']
        assert out['receiving_yards'].tolist() == [500, 120]

    def test_no_reference(self):
        out = rj.enrich(predictions(), None)
        assert len(out) == 5 and out['display_name'].tolist() == predictions()['player_name'].tolist()
        assert out['position'].isna().all()

    def test_numeric_looking_ids_compared_as_text(self):
        df = pd.DataFrame({'player_id': ['00-0012345'], 'player_name': ['Z.Zero']})
        assert rj.enrich(df, ref())['position'].tolist() == ['TE']


class TestSummary:
    def test_notes_cover_problems_and_unmatched_rows(self):
        r = ref()
        out = rj.enrich(pd.concat([predictions(), pd.DataFrame({'player_name': ['T.Solo', 'J.Smith']})]), r)
        text = "; ".join(rj.enrichment_summary(out, r))
        assert "appear on more than one" in text
        assert "1 row(s) have a player ID that is not in the players reference" in text
        assert "1 row(s) have a player ID that appears more than once" in text
        assert "1 row(s) matched by a unique name only (no player ID in the file; inferred, not verified)" in text
        assert "1 row(s) have no player ID and a name shared by several players" in text

    def test_clean_input_has_no_notes(self):
        clean = rj.build_reference(REFERENCE[REFERENCE['gsis_id'] != 'D4'])
        df = pd.DataFrame({'player_id': ['A1', 'B2'], 'player_name': ['J.Smith', 'J.Smith']})
        assert rj.enrichment_summary(rj.enrich(df, clean), clean) == []


class TestPlayerRowsAndKeys:
    def test_player_rows_by_id(self):
        df = pd.DataFrame({'player_id': ['A1', 'B2', 'A1'], 'display_name': ['John Smith', 'John Smith', 'X']})
        assert rj.player_rows(df, ref(), 'A1').tolist() == [True, False, True]

    def test_player_rows_legacy_unique_display_name_only(self):
        df = pd.DataFrame({'display_name': ['Tom Solo', 'John Smith', 'Dan Dup', 'Nobody']})
        assert rj.player_rows(df, ref(), 'C3').tolist() == [True, False, False, False]
        assert rj.player_rows(df, ref(), 'A1').tolist() == [False, True, False, False]
        two = rj.build_reference(pd.concat([REFERENCE, REFERENCE.iloc[[0]].assign(gsis_id='A9')]))
        assert not rj.player_rows(df, two, 'A1').any()                 # display name now shared

    def test_unknown_id_row_never_matches_by_name(self):
        df = pd.DataFrame({'player_id': ['Z9'], 'display_name': ['Tom Solo']})
        assert not rj.player_rows(df, ref(), 'C3').any()

    def test_player_key_separates_namesakes(self):
        df = pd.DataFrame({'player_id': ['A1', 'B2', None], 'player_name': ['J.Smith', 'J.Smith', 'J.Smith']})
        assert rj.player_key(df).tolist() == ['A1', 'B2', 'name:J.Smith']
        assert rj.player_key(df.drop(columns='player_id')).tolist() == ['name:J.Smith'] * 3

    def test_unique_labels(self):
        assert rj.unique_labels(['J.Smith - WR', 'J.Smith - WR', 'Tom'], ['A1', 'B2', 'C3']) == [
            'J.Smith - WR [A1]', 'J.Smith - WR [B2]', 'Tom']
        assert rj.unique_labels(['X', 'X'], ['A1', 'A1']) == ['X', 'X (2)']


# ---------------------------------------------------------------------------
# The rendered pages (AppTest) with synthetic files in a temporary directory
# ---------------------------------------------------------------------------

PRED_COLUMNS = ['week', 'season', 'game_date', 'player_name', 'player_id', 'display_name', 'position', 'team',
                'opponent', 'is_home', 'prop_type', 'line_type', 'line_value', 'raw_model_prob_over', 'prob_over',
                'prob_under', 'recommendation', 'confidence', 'model_reliable', 'model_name', 'model_bundle',
                'avg_L3', 'avg_L5', 'avg_L10', 'trend', 'weather_adjusted', 'weather_conditions', 'market_line',
                'market_book', 'market_implied_prob', 'market_edge', 'market_line_available', 'adjustments']


def pred_row(pid, name, prop, line, p, team='AAA', opp='BBB'):
    return dict(week=7, season=2026, game_date='2026-10-25', player_name=name, player_id=pid, display_name=None,
                position=None, team=team, opponent=opp, is_home=1, prop_type=prop, line_type='starter',
                line_value=line, raw_model_prob_over=p, prob_over=p, prob_under=1 - p,
                recommendation='OVER' if p >= 0.5 else 'UNDER', confidence=max(p, 1 - p), model_reliable=True,
                model_name=f'{prop}_starter', model_bundle='synthetic', avg_L3=3.0, avg_L5=3.0, avg_L10=3.0,
                trend='➡️', weather_adjusted=False, weather_conditions='', market_line=None, market_book=None,
                market_implied_prob=None, market_edge=None, market_line_available=False, adjustments='none')


PREDS = [pred_row('A1', 'J.Smith', 'receptions', 4.5, 0.62), pred_row('B2', 'J.Smith', 'receptions', 4.5, 0.62),
         pred_row('Z9', 'T.Solo', 'receptions', 3.5, 0.70), pred_row('C3', 'T.Solo', 'passing_yards', 220.0, 0.64),
         pred_row('A1', 'J.Smith', 'receiving_yards', 45.0, 0.60)]


def write_pages_fixture(root: Path, schema='current'):
    data = root / 'data_files'
    data.mkdir()
    REFERENCE.to_csv(data / 'players.csv', index=False)
    preds = pd.DataFrame(PREDS, columns=PRED_COLUMNS)
    if schema == 'current':
        preds = rj.enrich(preds.drop(columns=['display_name', 'position']), ref())[PRED_COLUMNS]
    elif schema == 'no_details':            # player_id but no display_name/position (as written before enrichment)
        preds = preds.drop(columns=['display_name', 'position'])
    elif schema == 'legacy':                # before player_id existed
        preds = preds.drop(columns=['player_id', 'display_name', 'position'])
    preds.to_csv(data / 'player_props_predictions.csv', index=False)
    stats_cols = ['player_id', 'player_name', 'game_id', 'season', 'week', 'team', 'opponent', 'game_date']
    rec = pd.DataFrame([
        dict(player_id=pid, player_name='J.Smith', game_id=f'2025_0{w}_AAA_BBB', season=2025, week=w, team='AAA',
             opponent='BBB', game_date=f'2025-09-0{w}', receiving_yards=y, receptions=r, rec_tds=0, targets=8)
        for pid, ys in (('A1', [80, 70]), ('B2', [10, 15])) for w, y, r in zip((1, 2), ys, (6, 2))]
        + [dict(player_id=pid, player_name='M.Williams', game_id=f'2025_01_{tm}_ZZZ', season=2025, week=1, team=tm,
                opponent='ZZZ', game_date='2025-09-01', receiving_yards=y, receptions=3, rec_tds=0, targets=5)
           for pid, tm, y in (('F6', 'FFF', 40), ('G7', 'GGG', 30))])
    rec.to_csv(data / 'player_receiving_stats.csv', index=False)
    pd.DataFrame([dict(player_id='C3', player_name='T.Solo', game_id='2025_01_DDD_EEE', season=2025, week=1,
                       team='DDD', opponent='EEE', game_date='2025-09-01', passing_yards=250, pass_tds=2,
                       completions=20, attempts=30, interceptions=1)]).to_csv(data / 'player_passing_stats.csv',
                                                                              index=False)
    pd.DataFrame(columns=stats_cols + ['rushing_yards', 'rush_tds', 'rush_attempts']).to_csv(
        data / 'player_rushing_stats.csv', index=False)
    return preds


def _app(tmp_path, monkeypatch, page, schema='current'):
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    write_pages_fixture(tmp_path, schema)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(ROOT))
    st.cache_data.clear()
    at = AppTest.from_file(str(ROOT / 'pages' / page), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _warnings(at):
    return "\n".join(str(w.value) for w in at.warning)


def _table(at, first_columns):
    for df in at.dataframe:
        v = df.value
        if list(v.columns[:len(first_columns)]) == first_columns:
            return v.reset_index(drop=True)
    raise AssertionError(f"no table starting with {first_columns}")


class TestPlayerPropsPage:
    @pytest.mark.parametrize('schema', ['current', 'no_details', 'legacy'])
    def test_renders_for_every_schema(self, tmp_path, monkeypatch, schema):
        at = _app(tmp_path, monkeypatch, '2_Player_Props.py', schema)
        table = _table(at, ['Player', 'Pos', 'Team', 'Opp'])
        assert len(table) == len(PREDS)                               # nothing multiplied or dropped
        assert "appear on more than one" in _warnings(at)             # duplicate D4 surfaced

    def test_predictions_by_id_get_their_own_details(self, tmp_path, monkeypatch):
        at = _app(tmp_path, monkeypatch, '2_Player_Props.py', 'no_details')
        t = _table(at, ['Player', 'Pos', 'Team', 'Opp'])
        got = sorted(zip(t['Player'], t['Pos'].fillna('')))
        # Independent expectation: by ID; Z9 unknown keeps its own name, no position.
        assert got == sorted([('John Smith', 'WR'), ('Jake Smith', 'RB'), ('T.Solo', ''), ('Tom Solo', 'QB'),
                              ('John Smith', 'WR')])
        assert "have a player ID that is not in the players reference" in _warnings(at)

    def test_legacy_predictions_enrich_unique_names_only(self, tmp_path, monkeypatch):
        at = _app(tmp_path, monkeypatch, '2_Player_Props.py', 'legacy')
        t = _table(at, ['Player', 'Pos', 'Team', 'Opp'])
        got = sorted(zip(t['Player'], t['Pos'].fillna('')))
        # J.Smith is ambiguous (not enriched); T.Solo is unique, so both T.Solo rows get Tom Solo.
        assert got == sorted([('J.Smith', ''), ('J.Smith', ''), ('J.Smith', ''), ('Tom Solo', 'QB'),
                              ('Tom Solo', 'QB')])
        w = _warnings(at)
        assert "matched by a unique name only" in w and "inferred, not verified" in w
        assert "name shared by several players" in w

    def test_season_leaders_keep_namesakes_apart(self, tmp_path, monkeypatch):
        at = _app(tmp_path, monkeypatch, '2_Player_Props.py')
        t = _table(at, ['Player', 'Pos', 'Team', 'GP', 'Total Yds', 'Yds/G', 'Rec'])
        assert len(t) == 4                                            # one row per player ID, nothing multiplied
        t = t[t['Player'] == 'J.Smith']
        assert len(t) == 2                                            # namesakes on one team not merged
        assert sorted(zip(t['Pos'], t['Total Yds'], t['Rec'])) == [('RB', 25, 8), ('WR', 150, 8)]

    def test_player_search_lists_namesakes_separately(self, tmp_path, monkeypatch):
        at = _app(tmp_path, monkeypatch, '2_Player_Props.py')
        sb = at.selectbox(key='player_search')
        assert sorted(sb.options) == ['Jake Smith', 'John Smith', 'Mike Williams [F6]', 'Mike Williams [G7]',
                                      'Tom Solo']
        at.text_input(key='dk_player_search').input('Smith').run()
        assert not at.exception
        opts = at.selectbox(key='dk_player_select').options
        assert sorted(opts) == ['Jake Smith - RB', 'John Smith - WR']

    def test_identical_name_and_position_stay_separate(self, tmp_path, monkeypatch):
        at = _app(tmp_path, monkeypatch, '2_Player_Props.py')
        at.text_input(key='dk_player_search').input('Mike Williams').run()
        assert not at.exception
        sb = at.selectbox(key='dk_player_select')
        assert sorted(sb.options) == ['Mike Williams - WR [F6]', 'Mike Williams - WR [G7]']
        # Selecting each label leads to that player's own games (by ID).
        for pid, yards in (('F6', 40), ('G7', 30)):
            at.selectbox(key='dk_player_select').set_value(f'Mike Williams - WR [{pid}]').run()
            assert not at.exception
            assert any(str(k).startswith(f'dk_line_value_{pid}_') for k in at.session_state.filtered_state)
        tab = at.selectbox(key='player_search')
        tab.set_value('Mike Williams [G7]').run()
        assert not at.exception and at.selectbox(key='player_search').value == 'Mike Williams [G7]'

    def test_search_reports_a_duplicate_reference_id(self, tmp_path, monkeypatch):
        at = _app(tmp_path, monkeypatch, '2_Player_Props.py')
        at.text_input(key='dk_player_search').input('Dan Dup').run()
        assert not at.exception
        opts = at.selectbox(key='dk_player_select').options
        assert len(opts) == 2                                         # both records listed, none dropped
        assert "repeated or missing in the players reference" in _warnings(at)


class TestParlayBuilderPage:
    @pytest.mark.parametrize('schema', ['current', 'no_details', 'legacy'])
    def test_renders_for_every_schema(self, tmp_path, monkeypatch, schema):
        at = _app(tmp_path, monkeypatch, '3_Parlay_Builder.py', schema)
        opts = at.multiselect[0].options
        assert len(opts) == len(PREDS) and len(set(opts)) == len(opts)

    def test_namesakes_are_not_one_player(self, tmp_path, monkeypatch):
        at = _app(tmp_path, monkeypatch, '3_Parlay_Builder.py', 'no_details')
        opts = at.multiselect[0].options
        smiths = [o for o in opts if 'Receptions' in o and 'Smith' in o]
        assert sorted(smiths) == sorted(o for o in opts if o.startswith(('John Smith (AAA) - Receptions',
                                                                          'Jake Smith (AAA) - Receptions')))
        at.multiselect[0].set_value(smiths).run()
        assert not at.exception
        text = "\n".join(str(m.value) for m in at.markdown)
        assert "Multiple props for" not in text                       # two players, not one
        legs = _table(at, ['display_name', 'position'])
        assert sorted(zip(legs['display_name'], legs['position'])) == [('Jake Smith', 'RB'), ('John Smith', 'WR')]

    def test_same_player_two_props_still_warned(self, tmp_path, monkeypatch):
        at = _app(tmp_path, monkeypatch, '3_Parlay_Builder.py', 'no_details')
        opts = [o for o in at.multiselect[0].options if o.startswith('John Smith')]
        at.multiselect[0].set_value(opts).run()
        assert "Multiple props for John Smith" in "\n".join(str(m.value) for m in at.markdown)

    def test_identical_labels_map_to_their_own_rows(self, tmp_path, monkeypatch):
        # Legacy file: both J.Smith receptions rows have the same label text.
        at = _app(tmp_path, monkeypatch, '3_Parlay_Builder.py', 'legacy')
        opts = at.multiselect[0].options
        same = [o for o in opts if o.startswith('J.Smith (AAA) - Receptions')]
        assert len(same) == 2 and len(set(same)) == 2
        at.multiselect[0].set_value(same).run()
        assert not at.exception
        legs = _table(at, ['display_name', 'position'])
        assert len(legs) == 2


# ---------------------------------------------------------------------------
# The real page loaders, run as written (cache decorator stubbed out)
# ---------------------------------------------------------------------------

def page_loader(page):
    """load_player_props_predictions exactly as defined in the page source."""
    import ast
    src = (ROOT / 'pages' / page).read_text(encoding='utf-8')
    keep = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)
            and n.name in ('load_player_props_predictions', 'load_reference_cached')]
    assert len(keep) == 2

    class St:
        @staticmethod
        def cache_data(*a, **k):
            return a[0] if a and callable(a[0]) else (lambda f: f)

    ns = {'st': St, 'pd': pd, 'Path': Path, 'enrich': rj.enrich, 'load_reference': rj.load_reference}
    exec(compile(ast.Module(body=keep, type_ignores=[]), page, 'exec'), ns)
    return ns['load_player_props_predictions']


LOADER_PAGES = ['2_Player_Props.py', '3_Parlay_Builder.py']


def write_loader_fixture(root, preds, reference=REFERENCE):
    data = root / 'data_files'
    data.mkdir()
    if reference is not None:
        reference.to_csv(data / 'players.csv', index=False)
    pd.DataFrame(preds, columns=PRED_COLUMNS).to_csv(data / 'player_props_predictions.csv', index=False)
    return pd.read_csv(data / 'player_props_predictions.csv', dtype={'player_id': str})


class TestPageLoadersAlwaysClassify:
    """Identity is classified even when display_name and position are stored."""

    def stored(self):
        rows = [pred_row('A1', 'J.Smith', 'receptions', 4.5, 0.62), pred_row('Z9', 'T.Solo', 'receptions', 3.5, 0.70),
                pred_row('D4', 'D.Dup', 'receptions', 3.5, 0.66), pred_row('B2', 'J.Smith', 'rushing_yards', 30.0, 0.61)]
        # Stored values, deliberately not the reference's, must survive as written.
        for r, (d, p) in zip(rows, [('Stored John', 'WR'), ('Stored Solo', 'QB'), ('Stored Dan', 'TE'),
                                     ('Stored Jake', 'RB')]):
            r.update(display_name=d, position=p)
        return rows

    @pytest.mark.parametrize('page', LOADER_PAGES)
    def test_complete_stored_details_kept_and_classified(self, tmp_path, monkeypatch, page):
        written = write_loader_fixture(tmp_path, self.stored())
        monkeypatch.chdir(tmp_path)
        df = page_loader(page)()
        assert len(df) == len(written) and df['player_id'].tolist() == written['player_id'].tolist()
        assert df['display_name'].tolist() == ['Stored John', 'Stored Solo', 'Stored Dan', 'Stored Jake']
        assert df['position'].tolist() == ['WR', 'QB', 'TE', 'RB']
        assert df['reference_match'].tolist() == ['id', 'unknown_id', 'duplicate_reference_id', 'id']
        np.testing.assert_allclose(df['prob_over'], written['prob_over'], rtol=1e-6)
        notes = "; ".join(rj.enrichment_summary(df, rj.load_reference(tmp_path / 'data_files' / 'players.csv')))
        assert "1 row(s) have a player ID that is not in the players reference" in notes
        assert "1 row(s) have a player ID that appears more than once" in notes

    @pytest.mark.parametrize('page', LOADER_PAGES)
    def test_partially_blank_details_filled_only_from_a_permitted_match(self, tmp_path, monkeypatch, page):
        rows = self.stored()
        for r in rows:                      # blank every display name and position except the last row's
            r.update(display_name=None, position=None)
        rows[3].update(display_name='Stored Jake', position='RB')
        rows[0].update(position='XX')       # a stored position next to a blank display name
        written = write_loader_fixture(tmp_path, rows)
        monkeypatch.chdir(tmp_path)
        df = page_loader(page)()
        assert df['display_name'].tolist() == ['John Smith', 'T.Solo', 'D.Dup', 'Stored Jake']
        assert df['position'].tolist()[0] == 'XX' and df['position'].tolist()[3] == 'RB'
        assert pd.isna(df['position'].tolist()[1]) and pd.isna(df['position'].tolist()[2])   # unknown / repeated: no fallback
        assert df['player_id'].tolist() == written['player_id'].tolist() and len(df) == 4

    @pytest.mark.parametrize('page', LOADER_PAGES)
    def test_unavailable_reference_differs_from_an_unmatched_name(self, tmp_path, monkeypatch, page):
        rows = self.stored()
        write_loader_fixture(tmp_path, rows, reference=None)
        monkeypatch.chdir(tmp_path)
        df = page_loader(page)()
        assert set(df['reference_match']) == {'reference_unavailable'}
        assert df['display_name'].tolist() == ['Stored John', 'Stored Solo', 'Stored Dan', 'Stored Jake']
        notes = rj.enrichment_summary(df, None)
        assert notes[0] == "players reference not found; player identities can't be checked"
        assert "can't be checked because the players reference is unavailable" in notes[1]

    @pytest.mark.parametrize('page', LOADER_PAGES)
    def test_unreadable_reference_is_unavailable(self, tmp_path, monkeypatch, page):
        write_loader_fixture(tmp_path, self.stored(), reference=REFERENCE.drop(columns=['position']))
        monkeypatch.chdir(tmp_path)
        df = page_loader(page)()
        assert set(df['reference_match']) == {'reference_unavailable'}

    def test_legacy_unmatched_name_is_not_unavailable(self):
        out = rj.enrich(pd.DataFrame({'player_name': ['X.None']}), ref())
        assert out['reference_match'].tolist() == ['name_not_found']


class TestSeasonTotals:
    def stats(self):
        return pd.DataFrame({
            'player_id': ['A1', 'A1', 'B2', 'B2', None, None, ' '],
            'player_name': ['J.Smith', 'J.Smith', 'J.Smith', 'J.Smith', 'Q.Nobody', 'J.Smith', 'Q.Nobody'],
            'team': ['AAA'] * 7, 'receiving_yards': [80, 70, 10, 15, 33, 5, 7], 'game_id': list('abcdefg')})

    def test_ids_kept_apart_and_id_less_rows_never_merged_into_them(self):
        t = rj.season_totals(self.stats(), {'receiving_yards': 'sum', 'game_id': 'count'})
        got = {(r.player_id, r.player_name): (r.receiving_yards, r.game_id) for r in t.itertuples()}
        assert got == {('A1', 'J.Smith'): (150, 2), ('B2', 'J.Smith'): (25, 2),
                       (None, 'Q.Nobody'): (40, 2), (None, 'J.Smith'): (5, 1)}
        assert t['game_id'].sum() == len(self.stats())                 # no row lost

    def test_id_less_groups_are_classified_not_verified(self):
        t = rj.enrich(rj.season_totals(self.stats(), {'receiving_yards': 'sum'}), ref(), fields=('position',))
        m = dict(zip(zip(t['player_id'].fillna(''), t['player_name']), t['reference_match']))
        assert m == {('A1', 'J.Smith'): 'id', ('B2', 'J.Smith'): 'id', ('', 'Q.Nobody'): 'name_not_found',
                     ('', 'J.Smith'): 'ambiguous_name'}
        assert rj.status_label(t['reference_match']).tolist().count('') == 2

    def test_rows_with_a_missing_team_or_name_are_not_lost(self):
        s = pd.concat([self.stats(), pd.DataFrame({'player_id': ['A1', None], 'player_name': ['J.Smith', None],
                                                   'team': [None, 'AAA'], 'receiving_yards': [9, 4],
                                                   'game_id': ['h', 'i']})], ignore_index=True)
        t = rj.season_totals(s, {'receiving_yards': 'sum', 'game_id': 'count'})
        assert t['game_id'].sum() == len(s) and t['receiving_yards'].sum() == s['receiving_yards'].sum()

    def test_all_id_less_stats_still_grouped_by_name(self):
        s = self.stats().drop(columns='player_id')
        t = rj.season_totals(s, {'receiving_yards': 'sum'})
        assert sorted(zip(t['player_name'], t['receiving_yards'])) == [('J.Smith', 180), ('Q.Nobody', 40)]
        assert t['player_id'].isna().all()


class TestMixedIdSeasonLeadersPage:
    def test_rendered_leaders_keep_namesakes_and_id_less_rows_apart(self, tmp_path, monkeypatch):
        from streamlit.testing.v1 import AppTest
        import streamlit as st
        write_pages_fixture(tmp_path)
        rec = pd.read_csv(tmp_path / 'data_files' / 'player_receiving_stats.csv', dtype={'player_id': str})
        extra = pd.DataFrame([dict(player_id=None, player_name='Q.Nobody', game_id='2025_03_QQQ_RRR', season=2025,
                                   week=3, team='QQQ', opponent='RRR', game_date='2025-09-14', receiving_yards=33,
                                   receptions=2, rec_tds=0, targets=4)])
        pd.concat([rec, extra]).to_csv(tmp_path / 'data_files' / 'player_receiving_stats.csv', index=False)
        monkeypatch.chdir(tmp_path)
        st.cache_data.clear()
        at = AppTest.from_file(str(ROOT / 'pages' / '2_Player_Props.py'), default_timeout=60)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        t = _table(at, ['Player', 'Pos', 'Team', 'GP', 'Total Yds', 'Yds/G', 'Rec'])
        assert len(t) == 5                                            # 2 J.Smith + 2 M.Williams + Q.Nobody
        rows = sorted(zip(t['Player'], t['Pos'].fillna(''), t['Total Yds'], t['GP'], t['Identity']))
        assert rows == [('J.Smith', 'RB', 25, 2, ''), ('J.Smith', 'WR', 150, 2, ''),
                        ('M.Williams', 'WR', 30, 1, ''), ('M.Williams', 'WR', 40, 1, ''),
                        ('Q.Nobody', '', 33, 1, 'not in reference')]
        assert t['GP'].sum() == len(rec) + 1                          # every stats row counted once
