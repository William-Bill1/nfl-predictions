"""Coverage-first records: synthetic SQLite only, no network or production data."""
import hashlib
import json
import sqlite3
from pathlib import Path
import pytest
from scripts import football_query as query


def build(tmp_path, extra=()):
    path=tmp_path/'audit.sqlite3'
    c=sqlite3.connect(path)
    c.execute('CREATE TABLE query_coverage_metadata (assessment_id,source_sha256,evaluation_toronto_date,population_basis)')
    c.execute("INSERT INTO query_coverage_metadata VALUES (1,'synthetic-sha','2026-11-13','synthetic schedule')")
    c.execute('CREATE TABLE query_coverage_games (assessment_id,game_id,season,game_type,gameday,home_team,away_team,completion_status,completion_reason,home_score,away_score,scores_verified,roof_category,surface_category,roof_issue,source_url)')
    rows=[row('win'),row('turf',surface_category='artificial',home_score=7,away_score=20),
          row('future',gameday='2026-11-15',completion_status='excluded',home_score=None,away_score=None,scores_verified=0)]
    rows.extend(extra)
    for item in rows:c.execute('INSERT INTO query_coverage_games VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',tuple(item.values()))
    c.commit();c.close()
    return path


def row(gid,**changes):
    result=dict(assessment_id=1,game_id=gid,season=2026,game_type='REG',gameday='2026-11-01',
                home_team='AAA',away_team='BBB',completion_status='final',completion_reason='verified',
                home_score=20,away_score=7,scores_verified=1,roof_category='outdoors',
                surface_category='natural_grass',roof_issue=None,source_url='https://example.invalid/gamebook')
    result.update(changes)
    return result


def run(path,**kw):return query.query_record(path,1,'AAA',2026,**kw)


def test_complete_record_excludes_known_nonmatches_and_future_and_preserves_db(tmp_path):
    path=build(tmp_path);before=hashlib.sha256(path.read_bytes()).hexdigest();r=run(path)
    assert r['coverage_status']=='complete_for_snapshot'
    assert r['complete_record']['W']==1 and r['complete_record']['L']==0
    assert r['pending_games']==['future'] and r['excluded_by_known_conditions']==['turf']
    assert r['known_subset'] is None and r['blocking_games']==[]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before


@pytest.mark.parametrize('changes,field',[
    ({'roof_category':None},'roof'),
    ({'roof_category':'dome','roof_issue':'Sources disagree'},'roof'),
    ({'surface_category':None},'surface'),
    ({'surface_category':'unknown'},'surface'),
])
def test_unknown_or_conflicting_condition_blocks_complete_record(tmp_path,changes,field):
    r=run(build(tmp_path,[row('gap',**changes)]))
    assert r['coverage_status']=='incomplete' and r['complete_record'] is None
    assert r['known_subset']['W']==1
    assert r['blocking_games'][0]['game_id']=='gap' and field in r['blocking_games'][0]['fields']


def test_unknown_roof_does_not_block_when_known_surface_rules_game_out(tmp_path):
    r=run(build(tmp_path,[row('roof_conflict',surface_category='artificial',roof_issue='Conflicting roof',roof_category=None)]))
    assert r['coverage_status']=='complete_for_snapshot'
    assert 'roof_conflict' in r['excluded_by_known_conditions']


@pytest.mark.parametrize('day',['2026-11-12','',None])
def test_past_or_undated_game_without_final_evidence_blocks(tmp_path,day):
    r=run(build(tmp_path,[row('unresolved',gameday=day,completion_status='excluded',completion_reason='No final evidence')]))
    assert r['complete_record'] is None and r['coverage_status']=='incomplete'
    assert r['blocking_games'][0]['fields']==['scores/completion']


def test_scores_only_query_ignores_irrelevant_roof_and_surface_gaps(tmp_path):
    r=run(build(tmp_path,[row('roof',roof_category=None,roof_issue='Conflicting descriptions',surface_category=None)]),all_conditions=True)
    assert r['coverage_status']=='complete_for_snapshot'
    assert r['required_fields']==['verified scores']
    assert (r['complete_record']['W'],r['complete_record']['L'])==(2,1)


def test_unverified_score_blocks_even_scores_only_query(tmp_path):
    r=run(build(tmp_path,[row('score',scores_verified=0)]),all_conditions=True)
    assert r['complete_record'] is None and r['blocking_games'][0]['fields']==['scores']


def test_no_data_is_not_reported_as_complete_zero_record(tmp_path):
    r=query.query_record(build(tmp_path),1,'MISSING',2026)
    assert r['coverage_status']=='no_data' and r['complete_record'] is None


def test_duplicate_games_refused_instead_of_multiplying_counts(tmp_path):
    with pytest.raises(ValueError,match='Duplicate game IDs'):run(build(tmp_path,[row('win')]))


def test_cli_incomplete_has_nonzero_exit_and_no_complete_record(tmp_path,capsys):
    path=build(tmp_path,[row('gap',roof_category=None)])
    assert query.main(['--db',str(path),'--assessment-id','1','--team','AAA','--season','2026'])==3
    result=json.loads(capsys.readouterr().out)
    assert result['coverage_status']=='incomplete' and result['complete_record'] is None


def test_old_database_without_audit_metadata_never_falls_back(tmp_path,capsys):
    path=tmp_path/'raw.sqlite3';sqlite3.connect(path).close();before=path.read_bytes()
    assert query.main(['--db',str(path),'--assessment-id','1','--team','AAA','--season','2026'])==2
    result=json.loads(capsys.readouterr().out)
    assert result['coverage_status']=='not_checked' and result['complete_record'] is None
    assert path.read_bytes()==before


def test_cli_unreadable_database_is_not_checked(tmp_path, monkeypatch, capsys):
    def refused(*args, **kwargs):
        raise PermissionError("Database access denied")
    monkeypatch.setattr(query.repository, "open_read_only", refused)
    assert query.main(["--db", str(tmp_path/"unreadable.sqlite3"),
                       "--assessment-id", "1", "--team", "AAA", "--season", "2026"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["coverage_status"] == "not_checked"
    assert result["complete_record"] is None
    assert "access denied" in result["error"]
    assert not list(tmp_path.iterdir())
