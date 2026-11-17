import json
import sqlite3
from pathlib import Path
import pytest
from tests.test_football_epa import inputs
from scripts import football_epa as epa, football_weekly_scorecards as weekly


def prepare(inputs):
    source,audit,db,_,_=inputs
    with sqlite3.connect(audit) as c:
        c.execute('ALTER TABLE query_coverage_games ADD COLUMN home_score INTEGER')
        c.execute('ALTER TABLE query_coverage_games ADD COLUMN away_score INTEGER')
        c.execute('UPDATE query_coverage_games SET home_score=21,away_score=14')
        c.execute('CREATE TABLE game_observations(import_id,game_id,gameday,away_team,home_team,season,game_type,week,location)')
        c.execute('CREATE TABLE completion_assessments(assessment_id,import_id)')
        c.execute('INSERT INTO completion_assessments VALUES(1,1)')
        c.executemany("INSERT INTO game_observations VALUES(1,?,?,?,?,2026,'REG',?,'Home')",
                      [('g1','2026-09-01','BUF','KC',1),('g2','2026-09-08','KC','BUF',2),('future','2026-09-15','KC','BUF',3)])
        c.execute("INSERT INTO query_coverage_games VALUES(1,2026,'REG','pending',0,'future','2026-09-15','BUF','KC',NULL,NULL)")
    epa.build(source,audit,db,assessment_id=1,season=2026)
    return db,audit,db.parent/'weekly'


def test_batch_reuses_report_and_preserves_inputs_and_existing_output(inputs):
    db,audit,out=prepare(inputs);before=db.read_bytes(),audit.read_bytes()
    result=weekly.build_week(db,audit,out,season=2026,week=3,before='2026-09-09')
    assert len(result['games'])==1
    assert result['status']=='generated_with_coverage_gaps'
    assert result['games'][0]['coverage_notices']
    assert json.loads((out/'INDEX.json').read_text())==result
    assert (out/'future.md').exists() and (out/'future.json').exists()
    after={p.name:p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):weekly.build_week(db,audit,out,season=2026,week=3,before='2026-09-09')
    assert after=={p.name:p.read_bytes() for p in out.iterdir()}
    assert before==(db.read_bytes(),audit.read_bytes())


@pytest.mark.parametrize('kwargs',[
    {'game_ids':['missing']},{'game_ids':['future','future']},{'season':2025},
    {'week':1},{'before':'20260909'},{'week':19}])
def test_bad_selection_refused_before_output(inputs,kwargs):
    db,audit,out=prepare(inputs)
    args=dict(season=2026,week=3,before='2026-09-09');args.update(kwargs)
    with pytest.raises(ValueError):weekly.build_week(db,audit,out,**args)
    assert not out.exists()


def test_mismatched_context_refused(inputs):
    db,audit,out=prepare(inputs)
    with sqlite3.connect(audit) as c:c.execute("UPDATE game_observations SET location='Neutral'")
    with pytest.raises(ValueError,match='hash'):weekly.build_week(db,audit,out,season=2026,week=3,before='2026-09-09')
    assert not out.exists()


def test_write_failure_never_writes_completion_index(inputs,monkeypatch):
    db,audit,out=prepare(inputs);original=Path.open
    def fail(path,*args,**kwargs):
        if path==out/'future.md':raise OSError('simulated disk error')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',fail)
    with pytest.raises(OSError,match='disk error'):
        weekly.build_week(db,audit,out,season=2026,week=3,before='2026-09-09')
    assert out.exists() and not (out/'INDEX.json').exists()
