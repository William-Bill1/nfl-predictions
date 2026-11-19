from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import pandas as pd
import pytest
import pregame_snapshots as ps
from tests.test_fanduel_ontario_timeline import observations
from tests.test_pregame_snapshots import _write_inputs, _manifest
from scripts import football_market_context as market


@pytest.fixture
def prepared(observations,tmp_path,monkeypatch):
    directory,schedule,add=observations
    data=tmp_path/'model'
    _write_inputs(data,[('2026_11_NYJ_LAC',2026,11,'2026-11-22','13:00','LAC','NYJ',None,None,3.5)],[.6],[1])
    _manifest(data,now=datetime(2026,11,17,15,55,tzinfo=timezone.utc),monkeypatch=monkeypatch)
    snapshot,_,_=ps.capture(data,now=datetime(2026,11,17,16,tzinfo=timezone.utc))
    def load(asof='2026-11-20T19:00:00Z'):
        return market.load(directory,schedule,asof,snapshot,season=2026,week=11)
    return directory,schedule,add,snapshot,load


def test_movement_missing_days_and_model_line(prepared):
    directory,schedule,add,snapshot,load=prepared
    add('2026-11-17T17:00:00Z',3.5,-115,'tue')
    add('2026-11-19T12:40:00Z',3,-110,'thu')
    # A different side observed Wednesday establishes a missing NYJ checkpoint.
    import ontario_spreads as on
    doc=on.prepare_manual_quote(game_id='2026_11_NYJ_LAC',team='LAC',handicap=-3.5,price=-110,
        observed_at='2026-11-18T17:00:00Z',now=datetime(2026,11,20,18,tzinfo=timezone.utc),schedule_path=schedule)
    (directory/'home.json').write_text(json.dumps(doc))
    before={str(p):p.read_bytes() for p in [schedule,snapshot,*directory.glob('*.json')]}
    context=market.for_game(load(),'2026_11_NYJ_LAC','NYJ','LAC',2026,11)
    rows=context['sides'][0]['rows']
    assert [r['day'] for r in rows]==['2026-11-17','2026-11-18','2026-11-19','2026-11-20']
    assert rows[1]['quote'] is None and rows[3]['quote'] is None
    assert rows[2]['change']['spread_change']==-.5 and rows[2]['change']['crossed_margins']==[3]
    assert rows[2]['change']['price_probability_change_pp'] is None
    assert rows[2]['model_line_matches'] is False
    text=market.markdown(context)
    assert 'LINE MISMATCH' in text and 'No observation' in text and '2026-11-19T07:40-05:00' in text
    assert before=={str(p):p.read_bytes() for p in [schedule,snapshot,*directory.glob('*.json')]}


def test_price_only_and_conflicts(prepared):
    directory,schedule,add,snapshot,load=prepared
    add('2026-11-17T17:00:00Z',3.5,-110,'a')
    add('2026-11-18T17:00:00Z',3.5,-105,'b')
    c=market.for_game(load(),'2026_11_NYJ_LAC','NYJ','LAC',2026,11)
    rows=[r for r in c['sides'][0]['rows'] if r['quote']]
    assert rows[1]['change']['spread_change']==0
    assert rows[1]['change']['price_probability_change_pp']<0
    add('2026-11-18T17:00:00Z',3,-105,'conflict')
    c=market.for_game(load(),'2026_11_NYJ_LAC','NYJ','LAC',2026,11)
    assert c['sides'][0]['conflicting_observations']
    assert all(r['change'] is None and r['model_line_matches'] is None for r in c['sides'][0]['rows'])
    assert 'CONFLICT' in market.markdown(c)


def test_no_quotes_and_asof(prepared):
    directory,schedule,add,snapshot,load=prepared
    add('2026-11-19T17:00:00Z',3,-110,'later')
    c=market.for_game(load('2026-11-18T17:00:00Z'),'2026_11_NYJ_LAC','NYJ','LAC',2026,11)
    assert all(r['quote'] is None for side in c['sides'] for r in side['rows'])
    with pytest.raises(ValueError,match='after market as-of'):
        load('2026-11-17T15:59:00Z')


def test_identity_and_integrity_refusals(prepared):
    directory,schedule,add,snapshot,load=prepared
    with pytest.raises(ValueError,match='identity differs'):
        market.for_game(load(),'2026_11_NYJ_LAC','LAC','NYJ',2026,11)
    doc=json.loads(snapshot.read_text());doc['run_id']='tampered';snapshot.write_text(json.dumps(doc))
    with pytest.raises(ValueError):load()


def test_repeated_identical_timestamp_is_not_a_change(prepared):
    directory,schedule,add,snapshot,load=prepared
    add('2026-11-17T17:00:00Z',3.5,-110,'a')
    add('2026-11-17T17:00:00Z',3.5,-110,'b')
    context=market.for_game(load(),'2026_11_NYJ_LAC','NYJ','LAC',2026,11)
    assert not context['sides'][0]['conflicting_observations']
    assert 'FanDuel Ontario' in market.markdown(context)
    with pytest.raises(ValueError,match='date differs'):
        market.for_game(load(),'2026_11_NYJ_LAC','NYJ','LAC',2026,11,gameday='2026-11-23')


# Week 11 games: (game_id, gameday, entry/production gametime, reviewed gametime, away, home, spread_line)
UTC=timezone.utc
WEEK11={'PIT':('2026_11_PIT_PHI','2026-11-22','16:25','13:00','PIT','PHI',-4.5),
        'JAX':('2026_11_JAX_NYG','2026-11-22','13:00','16:25','JAX','NYG',3.0),
        'THU':('2026_11_BUF_MIA','2026-11-19','20:15','20:15','BUF','MIA',-2.5),
        'SUN':('2026_11_NYJ_LAC','2026-11-22','13:00','13:00','NYJ','LAC',0.0)}


def _schedule(path,games,column):
    pd.DataFrame([{'game_id':g[0],'season':2026,'week':11,'gameday':g[1],'gametime':g[column],
                   'home_team':g[5],'away_team':g[4],'home_score':None,'away_score':None} for g in games]
                 ).to_csv(path,sep='\t',index=False)
    return path


@pytest.fixture
def week11(tmp_path,monkeypatch):
    """Quotes entered and snapshot captured against the production (pre-change) schedule,
    as the owner's tools do; the market schedule is the reviewed one unless overridden."""
    import ontario_spreads as on
    games=list(WEEK11.values())
    entry=_schedule(tmp_path/'entry.tsv',games,2)
    reviewed=_schedule(tmp_path/'reviewed.tsv',games,3)
    directory=tmp_path/'manual';directory.mkdir()
    def add(key,team,handicap,observed,name,price=-110,entered=None,schedule=None):
        g=WEEK11[key]
        doc=on.prepare_manual_quote(game_id=g[0],team=team,handicap=handicap,price=price,observed_at=observed,
            now=datetime.fromisoformat((entered or observed).replace('Z','+00:00')),schedule_path=schedule or entry,
            repo_dir=tmp_path,note='SYNTHETIC TEST ONLY')
        (directory/(name+'.json')).write_text(json.dumps(doc))
    def snapshot(at=datetime(2026,11,17,16,tzinfo=UTC)):
        data=tmp_path/('model_'+at.strftime('%d%H%M'))
        _write_inputs(data,[(g[0],2026,11,g[1],g[2],g[5],g[4],None,None,g[6]) for g in games],[.6]*len(games),[1]*len(games))
        _manifest(data,now=at.replace(minute=0)-timedelta(minutes=5),monkeypatch=monkeypatch)
        return ps.capture(data,now=at)[0]
    def card(key,asof,schedule=None,snap=None,**changes):
        g=WEEK11[key]
        loaded=market.load(directory,schedule or reviewed,asof,snap or snapshot(),season=2026,week=11)
        return market.for_game(loaded,g[0],g[4],g[5],2026,11,gameday=changes.get('gameday',g[1]))
    return add,card,entry,reviewed,snapshot


def _shown(context,side=0):
    return [r['quote']['observed_at'] for r in context['sides'][side]['rows'] if r['quote']]


@pytest.mark.parametrize('key',['PIT','JAX'])
def test_stale_market_schedule_refused_for_reviewed_games(week11,key):
    add,card,entry,_,_=week11
    add(key,WEEK11[key][4],WEEK11[key][6],'2026-11-17T17:00:00Z','tue')
    with pytest.raises(ValueError,match='pre-change kickoff'):
        card(key,'2026-11-20T19:00:00Z',schedule=entry)


def test_reviewed_schedule_drops_in_game_pit_and_keeps_afternoon_jax(week11):
    add,card,_,reviewed,_=week11
    add('PIT','PIT',-4.5,'2026-11-17T17:00:00Z','pit_tue')
    add('PIT','PIT',-6.5,'2026-11-22T18:30:00Z','pit_in_game')    # 13:30 ET, after the real 13:00 kickoff
    add('JAX','JAX',3,'2026-11-17T17:00:00Z','jax_tue')
    # 14:00 ET, before the real 16:25 kickoff; entry needs the reviewed schedule (the old one refuses it).
    add('JAX','JAX',2.5,'2026-11-22T19:00:00Z','jax_afternoon',schedule=reviewed)
    pit=card('PIT','2026-11-22T20:00:00Z')
    assert _shown(pit)==['2026-11-17T17:00:00Z']
    assert [r['day'] for r in pit['sides'][0]['rows']]==['2026-11-17','2026-11-22']
    jax=card('JAX','2026-11-22T20:00:00Z')
    assert _shown(jax)==['2026-11-17T17:00:00Z','2026-11-22T19:00:00Z']
    assert jax['sides'][0]['rows'][-1]['change']['spread_change']==-.5


def test_model_timing_uses_reviewed_kickoff(week11):
    _,card,_,_,snapshot=week11
    late=snapshot(datetime(2026,11,22,18,30,tzinfo=UTC))           # pregame only under the old 16:25 time
    with pytest.raises(ValueError,match='not pregame'):
        card('PIT','2026-11-22T19:00:00Z',snap=late)
    assert card('JAX','2026-11-22T19:00:00Z',snap=late)['model_captured_at']=='2026-11-22T18:30:00Z'


@pytest.mark.parametrize('column,value,match',[('gametime','20:20','review required'),
                                              ('gameday','2026-11-23','review required'),
                                              ('week',12,'identity differs')])
def test_unexpected_reviewed_game_changes_refused(week11,tmp_path,column,value,match):
    _,card,_,reviewed,_=week11
    df=pd.read_csv(reviewed,sep='\t');df.loc[df.game_id=='2026_11_PIT_PHI',column]=value
    changed=tmp_path/'changed.tsv';df.to_csv(changed,sep='\t',index=False)
    with pytest.raises(ValueError,match=match):
        card('PIT','2026-11-20T19:00:00Z',schedule=changed,gameday=value if column=='gameday' else None)


def test_thursday_card_stops_at_kickoff_date(week11):
    add,card,_,_,_=week11
    add('THU','BUF',-2.5,'2026-11-17T17:00:00Z','buf_tue')
    add('SUN','NYJ',0,'2026-11-18T17:00:00Z','nyj_wed')
    add('SUN','NYJ',0,'2026-11-21T17:00:00Z','nyj_sat')
    thu=card('THU','2026-11-22T12:00:00Z')
    # Wednesday is listed because another game was observed then; Thursday was not observed by anyone.
    assert [(r['day'],r['quote'] is None) for r in thu['sides'][0]['rows']]==[
        ('2026-11-17',False),('2026-11-18',True)]
    text=market.markdown(thu)
    table=[line for line in text.splitlines() if line.startswith(('| BUF','| MIA'))]
    assert len(table)==4 and not any('2026-11-2' in line for line in table)
    assert ('Dates shown are observed dates across this week plus the applicable as-of date'
            '—not a complete daily or planned-checkpoint calendar.') in text


def test_asof_after_sunday_kickoff_adds_no_later_dates(week11):
    add,card,_,_,_=week11
    add('SUN','NYJ',0,'2026-11-17T17:00:00Z','nyj_tue',entered='2026-11-17T17:05:00Z')
    rows=card('SUN','2026-11-24T12:00:00Z')['sides'][0]['rows']
    assert [(r['day'],r['quote'] is None) for r in rows]==[('2026-11-17',False)]


def test_pickem_model_line_is_zero_without_inventing_a_side(week11):
    add,card,_,_,_=week11
    add('SUN','NYJ',0,'2026-11-17T17:00:00Z','nyj_pk')
    add('SUN','LAC',1,'2026-11-18T17:00:00Z','lac_one')
    context=card('SUN','2026-11-20T19:00:00Z')
    nyj,lac=context['sides']
    assert nyj['model_handicap']==0 and lac['model_handicap']==0
    assert [r['model_line_matches'] for r in nyj['rows'] if r['quote']]==[True]
    assert [r['model_line_matches'] for r in lac['rows'] if r['quote']]==[False]
    assert set(nyj)=={'team','model_handicap','conflicting_observations','rows'}
    text=market.markdown(context)
    assert '| NYJ | 2026-11-17T12:00-05:00 | +0 | -110 | — | — | — | 0 |' in text
    assert '| LAC | 2026-11-18T12:00-05:00 | +1 | -110 | — | — | — | 0 — LINE MISMATCH' in text
    assert 'underdog' not in json.dumps(context).lower() and 'signal' not in json.dumps(context).lower()
