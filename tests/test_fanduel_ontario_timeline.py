from datetime import datetime,timezone
import json
from pathlib import Path
import sqlite3
import pandas as pd
import pytest
import ontario_spreads as on
from scripts import fanduel_ontario_timeline as pilot
from scripts.football_repository import open_read_only


@pytest.fixture
def observations(tmp_path):
    schedule=tmp_path/'schedule.tsv'
    pd.DataFrame([{'game_id':'2026_11_NYJ_LAC','season':2026,'week':11,'gameday':'2026-11-22',
                   'gametime':'13:00','home_team':'LAC','away_team':'NYJ','home_score':None,'away_score':None}]).to_csv(schedule,sep='\t',index=False)
    directory=tmp_path/'manual';directory.mkdir()
    def add(time,spread,price,name,opponent_price=None,entered='2026-11-20T18:00:00Z'):
        doc=on.prepare_manual_quote(game_id='2026_11_NYJ_LAC',team='NYJ',handicap=spread,price=price,
                                    observed_at=time,opponent_price=opponent_price,now=datetime.fromisoformat(entered.replace('Z','+00:00')),
                                    schedule_path=schedule,repo_dir=tmp_path,note='SYNTHETIC TEST ONLY')
        (directory/(name+'.json')).write_text(json.dumps(doc))
        return doc
    return directory,schedule,add


def test_tuesday_friday_spread_and_key_margin(observations):
    directory,schedule,add=observations
    add('2026-11-17T17:00:00Z',3.5,-110,'early')
    add('2026-11-20T17:00:00Z',2.5,-105,'late')
    group=pilot.timeline(directory,schedule,'2026-11-20T19:00:00Z')['groups'][0]
    assert group['status']=='compared'
    assert group['spread_change']==-1
    assert group['crossed_margins']==[3]
    assert group['price_probability_change_pp'] is None
    assert group['observations'][0]['jurisdiction']=='CA-ON'


def test_price_only_change(observations):
    directory,schedule,add=observations
    add('2026-11-17T17:00:00Z',3.5,-105,'early')
    add('2026-11-20T17:00:00Z',3.5,-120,'late')
    group=pilot.timeline(directory,schedule,'2026-11-20T19:00:00Z')['groups'][0]
    assert group['spread_change']==0
    assert group['price_probability_change_pp']==pytest.approx(3.3259423503)


def test_conflicting_same_time_never_picks_endpoint(observations):
    directory,schedule,add=observations
    add('2026-11-17T17:00:00Z',3.5,-110,'a')
    add('2026-11-17T17:00:00Z',2.5,-110,'b')
    group=pilot.timeline(directory,schedule,'2026-11-20T19:00:00Z')['groups'][0]
    assert group['status']=='conflicting_observations'
    assert group['spread_change'] is None and len(group['observations'])==2


def test_entry_time_cutoff_and_no_fallback(observations):
    directory,schedule,add=observations
    add('2026-11-17T17:00:00Z',3.5,-110,'a',entered='2026-11-20T18:00:00Z')
    report=pilot.timeline(directory,schedule,'2026-11-18T19:00:00Z')
    assert report['groups']==[] and len(report['excluded'])==1
    with pytest.raises(ValueError):pilot.timeline(directory,schedule,'2026-11-18T19:00:00')


def test_us_quote_or_corruption_refused(observations):
    directory,schedule,add=observations
    doc=add('2026-11-17T17:00:00Z',3.5,-110,'a')
    doc['jurisdiction']='US';doc['book_key']='fanduel';doc=on._seal(doc)
    (directory/'a.json').write_text(json.dumps(doc))
    with pytest.raises(ValueError):pilot.timeline(directory,schedule,'2026-11-20T19:00:00Z')


def test_current_kickoff_change_refused(observations):
    directory,schedule,add=observations
    add('2026-11-17T17:00:00Z',3.5,-110,'a')
    df=pd.read_csv(schedule,sep='\t');df['gametime']='16:05';df.to_csv(schedule,sep='\t',index=False)
    with pytest.raises(ValueError,match='anchored'):pilot.timeline(directory,schedule,'2026-11-20T19:00:00Z')


def test_queryable_snapshot_and_no_overwrite(observations,monkeypatch,tmp_path):
    directory,schedule,add=observations
    add('2026-11-17T17:00:00Z',3.5,-110,'a')
    report=pilot.timeline(directory,schedule,'2026-11-20T19:00:00Z')
    monkeypatch.setattr(pilot,'ROOT',tmp_path)
    out=tmp_path/'reports/output';pilot.write_report(report,out)
    assert (out/'COMPLETE.json').exists()
    db=out/'quotes.sqlite3';before=db.read_bytes()
    conn=open_read_only(db,require_current_schema=False)
    try:
        assert conn.execute('SELECT jurisdiction,handicap,price FROM quotes').fetchone()==('CA-ON',3.5,-110)
        with pytest.raises(sqlite3.DatabaseError):conn.execute('DELETE FROM quotes')
    finally:conn.close()
    assert db.read_bytes()==before
    with pytest.raises(FileExistsError):pilot.write_report(report,out)
    with pytest.raises(ValueError):pilot.write_report(report,tmp_path/'data_files/output')
