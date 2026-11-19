"""Generate a new research-only weekly scorecard folder from local snapshots."""
from contextlib import closing
from datetime import date
import argparse
import json
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import football_market_context as market
from scripts.football_epa import digest
from scripts.football_repository import open_read_only
from scripts.football_scorecard import matchup, markdown, _iso_date


def build_week(epa_db, context_db, output, *, season, week, before, turnover_db=None, game_ids=None,
               manual_quotes=None, market_schedule=None, market_as_of=None, model_snapshot=None):
    if not 1 <= week <= 18 or not _iso_date(before):
        raise ValueError("REG week must be 1-18; before must be YYYY-MM-DD")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / "reports"):
        raise ValueError("Repository output must be inside reports")
    sources = {"epa":Path(epa_db),"context":Path(context_db)}
    if turnover_db is not None:
        sources["turnovers"] = Path(turnover_db)
    hashes = {k:digest(p) for k,p in sources.items()}
    with closing(open_read_only(epa_db,require_current_schema=False)) as conn:
        meta = dict(conn.execute("SELECT key,value FROM metadata"))
    if int(meta["season"]) != season or hashes["context"] != meta["audit_sha256"]:
        raise ValueError("Season or context hash does not match EPA snapshot")
    with closing(open_read_only(context_db,require_current_schema=False)) as conn:
        rows = conn.execute("SELECT o.game_id,o.gameday,o.away_team,o.home_team FROM game_observations o "
                            "JOIN completion_assessments a ON a.import_id=o.import_id "
                            "WHERE a.assessment_id=? AND o.season=? AND o.game_type='REG' AND o.week=? "
                            "ORDER BY o.gameday,o.game_id",(int(meta["assessment_id"]),season,week)).fetchall()
    if not rows or len({r[0] for r in rows}) != len(rows):
        raise ValueError("No scheduled games or duplicate game identities")
    for gid,day,away,home in rows:
        if not re.fullmatch(r"[A-Za-z0-9_]+",gid) or not away or not home or away==home:
            raise ValueError("Invalid scheduled game identity")
        if not _iso_date(day):
            raise ValueError(f"Invalid scheduled date: {gid}")
    requested = set(game_ids) if game_ids else {r[0] for r in rows}
    if len(requested) != len(game_ids or requested) or requested - {r[0] for r in rows}:
        raise ValueError("Duplicate or unknown selected game IDs")
    selected = [r for r in rows if r[0] in requested]
    # Reserved names and case-insensitive uniqueness: a card must never occupy INDEX.json or README.md.
    planned = ["INDEX.json", "README.md"] + [r[0] + ext for r in selected for ext in (".md", ".json")]
    if len({name.casefold() for name in planned}) != len(planned):
        raise ValueError("Planned output filenames collide (reserved INDEX.json/README.md or game IDs differing only by case)")
    if any(day < before for _,day,_,_ in selected):
        raise ValueError("Selected week contains games before cutoff; choose explicit on/after-cutoff game IDs")
    options = (manual_quotes, market_schedule, market_as_of, model_snapshot)
    if any(v is not None for v in options) and not all(v is not None for v in options):
        raise ValueError("Manual quote history requires manual-quotes, market-schedule, market-as-of and model-snapshot together")
    loaded_market = market.load(*options, season=season, week=week) if all(v is not None for v in options) else None
    reports=[]
    for gid,day,away,home in selected:
        report=matchup(epa_db,[away,home],before,context_db,gid,turnover_db)
        if loaded_market is not None:
            report["market_context"] = market.for_game(loaded_market, gid, away, home, season, week, gameday=day)
        gaps=[]
        for team in (away,home):
            for window in ("season","last_four"):
                if report["comparisons"][team][window]["coverage"]["status"]!="covered":
                    gaps.append(f"{team} {window}: EPA coverage unavailable/incomplete")
                if report["opponent_records"][team][window]["status"]!="covered_for_snapshot":
                    gaps.append(f"{team} {window}: schedule context unavailable/incomplete")
                if turnover_db is not None and report["turnovers"][team][window]["status"]!="covered":
                    gaps.append(f"{team} {window}: turnovers unavailable/incomplete")
        reports.append((gid,day,report,gaps))
    if hashes != {k:digest(p) for k,p in sources.items()}:
        raise ValueError("Input changed during report generation")
    if loaded_market is not None and market.fingerprints(manual_quotes, market_schedule, model_snapshot) != loaded_market["hashes"]:
        raise ValueError("Market inputs changed during report generation")
    index={"season":season,"week":week,"before":before,"source_hashes":hashes,
           "status":"generated_with_coverage_gaps" if any(r[3] for r in reports) else "generated",
           "games":[{"game_id":gid,"gameday":day,"markdown":gid+".md","json":gid+".json","coverage_notices":gaps}
                    for gid,day,_,gaps in reports],
           "limits":"Research-only retrospective snapshots; no collection, prediction, price ranking or engine changes. Coverage relative to retained schedule. INDEX.json written last; absence means incomplete output."}
    if loaded_market is not None:
        index["market_context"] = {"as_of": loaded_market["history"]["as_of"], "source_hashes": loaded_market["hashes"]}
    output.mkdir(parents=True,exist_ok=False)
    for gid,_,report,_ in reports:
        with (output/(gid+".md")).open("x",encoding="utf-8") as f:f.write(markdown(report) + (market.markdown(report["market_context"]) if "market_context" in report else ""))
        with (output/(gid+".json")).open("x",encoding="utf-8") as f:json.dump(report,f,indent=2)
    text=[f"# {season} Week {week} research scorecards","",f"Exclusive game-date cutoff: {before}. Status: {index['status']}.",
          "These are matchup reports, not ranked picks or betting recommendations. No live injuries/weather are collected.",""]
    for game in index["games"]:
        text.append(f"- [{game['game_id']}]({game['markdown']}) ({game['gameday']})")
        text.extend("  - "+notice for notice in game["coverage_notices"])
    with (output/"README.md").open("x",encoding="utf-8") as f:f.write("\n".join(text)+"\n")
    with (output/"INDEX.json").open("x",encoding="utf-8") as f:json.dump(index,f,indent=2)
    return index


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("epa-db","context-db","output","before"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--turnover-db")
    for option in ("manual-quotes", "market-schedule", "market-as-of", "model-snapshot"):
        parser.add_argument("--" + option, help="Optional Ontario history inputs; supply all four")
    parser.add_argument("--season",type=int,required=True)
    parser.add_argument("--week",type=int,required=True)
    parser.add_argument("--game-id",action="append",help="Optional exact game IDs for selected picks; repeat as needed")
    args=parser.parse_args()
    try:
        result=build_week(args.epa_db,args.context_db,args.output,season=args.season,week=args.week,before=args.before,
                          turnover_db=args.turnover_db,game_ids=args.game_id, manual_quotes=args.manual_quotes,
                          market_schedule=args.market_schedule, market_as_of=args.market_as_of, model_snapshot=args.model_snapshot)
        print(json.dumps({"status":result["status"],"games":len(result["games"]),"output":args.output}))
        return 0
    except (OSError,ValueError,sqlite3.Error) as exc:
        print(str(exc),file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
