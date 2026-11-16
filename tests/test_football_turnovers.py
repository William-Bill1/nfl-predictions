import csv
import sqlite3
import pytest
from tests.test_football_epa import inputs
from scripts import football_epa as epa, football_turnovers as turnovers, football_scorecard as scorecard


def prepare(inputs, mutate=None):
    pbp,audit,db,_,_=inputs
    epa.build(pbp,audit,db,assessment_id=1,season=2026)
    rows=[]
    for gid in ('g1','g2'):
        for team,opponent,ints,fumbles in [('KC','BUF',1,1),('BUF','KC',0,1)]:
            rows.append(dict(game_id=gid,team=team,opponent_team=opponent,season='2026',season_type='REG',
                             passing_interceptions=str(ints),fumbles_lost_total=str(fumbles)))
    if mutate:mutate(rows)
    source=db.parent/'stats.csv'
    with source.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return source,db,db.parent/'turnovers.sqlite3'


def test_touchback_counted_without_recovery_field_and_cutoff(inputs):
    source,db,out=prepare(inputs)
    assert turnovers.build(source,db,out)['team_game_rows']==4
    before=out.read_bytes()
    result=turnovers.summary(out,db,'KC',['g1'])
    assert (result['giveaways'],result['takeaways'],result['differential'])==(2,1,-1)
    report=scorecard.matchup(db,['KC','BUF'],'2026-09-08',turnover_db=out)
    assert report['turnovers']['KC']['season']['games']==1
    assert '| Turnover differential (season total) | -1 | +1 | BUF |' in scorecard.markdown(report)
    assert out.read_bytes()==before


@pytest.mark.parametrize('problem',['missing','duplicate','identity','blank','negative','fraction'])
def test_invalid_source_refused(inputs,problem):
    def mutate(rows):
        if problem=='missing':rows.pop()
        elif problem=='duplicate':rows.append(rows[0].copy())
        elif problem=='identity':rows[0]['opponent_team']='NYJ'
        else:rows[0]['fumbles_lost_total']={'blank':'','negative':'-1','fraction':'1.5'}[problem]
    source,db,out=prepare(inputs,mutate)
    with pytest.raises(ValueError):turnovers.build(source,db,out)
    assert not out.exists()


def test_pinning_missing_selection_and_no_overwrite(inputs):
    source,db,out=prepare(inputs);turnovers.build(source,db,out)
    before=out.read_bytes()
    with pytest.raises(FileExistsError):turnovers.build(source,db,out)
    assert out.read_bytes()==before
    assert turnovers.summary(out,db,'KC',['missing'])['status']=='incomplete'
    with sqlite3.connect(db) as conn:conn.execute("INSERT INTO metadata VALUES('changed','yes')")
    with pytest.raises(ValueError,match='does not match'):turnovers.summary(out,db,'KC',['g1'])


def test_incomplete_turnover_selection_displays_unavailable_not_zero(inputs):
    source,db,out=prepare(inputs);turnovers.build(source,db,out)
    report=scorecard.matchup(db,['KC','BUF'],'2026-09-08',turnover_db=out)
    report['turnovers']['KC']['season']=turnovers.summary(out,db,'KC',['missing'])
    assert report['turnovers']['KC']['season']['differential'] is None
    text=scorecard.markdown(report)
    assert '| Turnover differential (season total) | Unavailable | +1 | Not assessed |' in text
    assert '| Turnover differential (season total) | 0 ' not in text
