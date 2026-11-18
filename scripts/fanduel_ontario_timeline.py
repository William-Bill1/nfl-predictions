"""Read-only manual FanDuel Ontario quote history; no API calls or bet advice."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import ontario_spreads as on
import pregame_snapshots as ps


def compare_quotes(quotes):
    """All input quotes have the same game/team. Preserve conflicts, never pick one."""
    ordered=sorted(quotes,key=lambda q:(q['observed_at'],q['quote_id']))
    times={}
    for q in ordered:
        times.setdefault(q['observed_at'],set()).add((q['handicap'],q['price'],q['opponent_price']))
    conflicts=sorted(t for t,values in times.items() if len(values)>1)
    result={'observations':ordered,'status':'conflicting_observations' if conflicts else 'one_observation' if len(times)==1 else 'compared' if times else 'no_observations',
            'conflicting_times':conflicts,'spread_change':None,'price_probability_change_pp':None,'crossed_margins':[],
            'price_comparison_basis':'One-side implied probability (includes margin), comparable only at unchanged handicap; not fair probability or expected return.'}
    if len(times)<2 or conflicts:
        return result
    first,last=ordered[0],ordered[-1]
    result['spread_change']=last['handicap']-first['handicap']
    lo,hi=sorted((first['handicap'],last['handicap']))
    result['crossed_margins']=[v for v in (-7,-3,3,7) if lo<=v<=hi and lo!=hi]
    if first['handicap']==last['handicap']:
        prob=lambda odds:(-odds)/((-odds)+100) if odds<0 else 100/(odds+100)
        result['price_probability_change_pp']=100*(prob(last['price'])-prob(first['price']))
    result['first_observed_at']=first['observed_at']
    result['last_observed_pregame_at']=last['observed_at']
    return result


def timeline(manual_dir,schedule_path,as_of,*,reassess_kickoffs=False):
    if not Path(manual_dir).is_dir():
        raise ValueError('Manual quote directory does not exist')
    cutoff=on._parse_observed(as_of)
    schedule,schedule_hash=on.load_schedule(Path(schedule_path))
    groups={};excluded=[];identities={};reassessments=[]
    for path in sorted(Path(manual_dir).glob('*.json')):
        doc=json.loads(path.read_text(encoding='utf-8'));on.validate_manual(doc,str(path))
        if doc['quote_id'] in identities:
            if identities[doc['quote_id']]!=doc:raise ValueError('Conflicting duplicate quote ID')
            continue
        identities[doc['quote_id']]=doc
        if on.parse_utc(doc['observed_at'])>cutoff or on.parse_utc(doc['entered_at'])>cutoff:
            excluded.append({'quote_id':doc['quote_id'],'reason':'not observed and entered by as-of cutoff'});continue
        rows=schedule[schedule.game_id==doc['game_id']]
        if len(rows)!=1:raise ValueError(f"Game not unique in schedule: {doc['game_id']}")
        game=rows.iloc[0];kickoff,problem=ps.kickoff_utc(game['gameday'],game['gametime'])
        if kickoff is None or any(doc[k]!=game[k] for k in ('home_team','away_team')) or int(game['season'])!=doc['season'] or int(game['week'])!=doc['week']:
            raise ValueError(f"Quote identity/kickoff unverified: {doc['quote_id']}")
        changed=on.parse_utc(doc['kickoff_utc'])!=kickoff
        if changed and not reassess_kickoffs:
            raise ValueError(f"Quote not anchored to current pregame kickoff: {doc['quote_id']}")
        if changed:
            reassessments.append({'quote_id':doc['quote_id'],'recorded_kickoff_utc':doc['kickoff_utc'],
                                  'evaluated_kickoff_utc':on.iso(kickoff),
                                  'pregame':on.parse_utc(doc['observed_at'])<kickoff})
        if on.parse_utc(doc['observed_at'])>=kickoff:
            if not reassess_kickoffs:
                raise ValueError(f"Quote not anchored to current pregame kickoff: {doc['quote_id']}")
            excluded.append({'quote_id':doc['quote_id'],'reason':'observed at or after evaluated kickoff',
                             'evaluated_kickoff_utc':on.iso(kickoff)})
            continue
        groups.setdefault((doc['game_id'],doc['team']),[]).append(doc)
    report={'as_of':on.iso(cutoff),'schedule_sha256':schedule_hash,
            'book_key':'fanduel_on_manual','jurisdiction':'CA-ON','source':'manual',
            'groups':[{'game_id':gid,'team':team,**compare_quotes(quotes)} for (gid,team),quotes in sorted(groups.items())],
            'excluded':excluded,'limits':'First/last OBSERVED pregame quotes, not verified open/close. No missed-checkpoint inference, US fallback, injury causation or betting-edge claim.'}
    if reassess_kickoffs:
        report['kickoff_evaluation']={'basis':'supplied schedule; original quotes unchanged',
                                      'reassessments':reassessments}
    return report


def write_report(report,output):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'reports'):
        raise ValueError('Output must resolve below repository reports/')
    output.mkdir(parents=True,exist_ok=False)
    # Queryable snapshot; preserves original validated artifacts as JSON in each row.
    with closing(sqlite3.connect(output/'quotes.sqlite3')) as conn:
        conn.execute('CREATE TABLE quotes(quote_id TEXT PRIMARY KEY,game_id TEXT,team TEXT,observed_at TEXT,entered_at TEXT,handicap REAL,price INTEGER,opponent_price INTEGER,book_key TEXT,jurisdiction TEXT,raw_json TEXT)')
        for group in report['groups']:
            for q in group['observations']:
                conn.execute('INSERT INTO quotes VALUES(?,?,?,?,?,?,?,?,?,?,?)',tuple(q[k] for k in ('quote_id','game_id','team','observed_at','entered_at','handicap','price','opponent_price','book_key','jurisdiction'))+(json.dumps(q,sort_keys=True),))
        conn.commit()
    with (output/'timeline.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    # Completion marker last; no claim of atomic or durable storage.
    with (output/'COMPLETE.json').open('x',encoding='utf-8') as f:json.dump({'groups':len(report['groups']),'as_of':report['as_of']},f)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manual-dir',required=True)
    parser.add_argument('--schedule',required=True)
    parser.add_argument('--as-of',required=True)
    parser.add_argument('--reassess-kickoffs',action='store_true',help='Explicitly reassess unchanged historical quotes against the supplied schedule')
    parser.add_argument('--output',help='New folder below reports/; omit to print JSON')
    args=parser.parse_args()
    try:
        if not Path(args.manual_dir).is_dir():raise ValueError('Manual quote directory does not exist')
        report=timeline(args.manual_dir,args.schedule,args.as_of,reassess_kickoffs=args.reassess_kickoffs)
        if args.output:write_report(report,args.output)
        else:print(json.dumps(report,indent=2))
        return 0
    except (OSError,ValueError,sqlite3.Error) as exc:
        print(str(exc),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
