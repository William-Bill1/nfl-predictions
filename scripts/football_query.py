"""Read-only, coverage-first records from the isolated audited game snapshot.

This does not validate arbitrary SQL or modify/import a database. Required views
are created by the documented capture audit, not by application startup.
"""
from __future__ import annotations
import argparse
from datetime import date
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import football_repository as repository


def query_record(db_path, assessment_id: int, team: str, season: int, *,
                 all_conditions: bool = False, game_type: str = "REG") -> dict:
    """Return a complete record only when every relevant required field is resolved.

    Known nonmatching conditions can exclude a game even when another condition
    is unresolved. Future/same-day games are pending, not missing final results.
    Past or undated games without final evidence block a complete-so-far claim.
    """
    conn = repository.open_read_only(db_path, require_current_schema=False)
    try:
        meta = conn.execute("SELECT source_sha256,evaluation_toronto_date,population_basis "
                            "FROM query_coverage_metadata WHERE assessment_id=?", (assessment_id,)).fetchone()
        if meta is None:
            raise ValueError("No audited metadata for this assessment; no fallback to raw results")
        source_hash, cutoff, population = meta
        cutoff_day = date.fromisoformat(cutoff)
        cur = conn.execute("SELECT * FROM query_coverage_games WHERE assessment_id=? "
                           "AND season=? AND game_type=? AND (home_team=? OR away_team=?) "
                           "ORDER BY gameday,game_id", (assessment_id,season,game_type,team,team))
        columns = [x[0] for x in cur.description]
        rows = [dict(zip(columns,row)) for row in cur.fetchall()]
    finally:
        conn.close()
    ids = [row["game_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate game IDs in audited query scope; refusing multiplied results")
    included, excluded, pending, blockers = [], [], [], []
    for row in rows:
        gid = row["game_id"]
        if row["completion_status"] != "final":
            try:
                scheduled = date.fromisoformat(str(row["gameday"])[:10])
            except ValueError:
                scheduled = None
            if scheduled is not None and scheduled >= cutoff_day:
                pending.append(gid)
            else:
                blockers.append({"game_id":gid,"fields":["scores/completion"],
                                 "reason":row["completion_reason"] or "Past or undated game lacks final evidence"})
            continue
        if row["scores_verified"] != 1 or not all(isinstance(row[k],int) and row[k]>=0
                                                  for k in ("home_score","away_score")):
            blockers.append({"game_id":gid,"fields":["scores"],"reason":"Scores not verified against gamebook"})
            continue
        if not all_conditions:
            roof, surface = row["roof_category"], row["surface_category"]
            roof_known = roof in repository.ROOF_CATEGORIES and not row["roof_issue"]
            surface_known = surface in (repository.NATURAL_GRASS, repository.ARTIFICIAL)
            if (roof_known and roof != "outdoors") or (surface_known and surface != repository.NATURAL_GRASS):
                excluded.append(gid)
                continue
            unknown = ([] if roof_known else ["roof"]) + ([] if surface_known else ["surface"])
            if unknown:
                blockers.append({"game_id":gid,"fields":unknown,
                                 "reason":row["roof_issue"] or "Required condition missing or unverified"})
                continue
        mine,theirs = (row["home_score"],row["away_score"]) if row["home_team"]==team else (row["away_score"],row["home_score"])
        included.append({"game_id":gid,"outcome":"W" if mine>theirs else "L" if mine<theirs else "T",
                         "location":"Home" if row["home_team"]==team else "Away",
                         "source_url":row["source_url"]})
    w = sum(x["outcome"]=="W" for x in included)
    lost = sum(x["outcome"]=="L" for x in included)
    ties = sum(x["outcome"]=="T" for x in included)
    values = {"W":w,"L":lost,"T":ties,"games":len(included),
              "win_pct_excluding_ties":w/(w+lost) if w+lost else None,
              "included_games":included}
    status = "no_data" if not rows else "incomplete" if blockers else "complete_for_snapshot"
    return {"coverage_status":status,
            "coverage_message":("Required fields resolved within the captured schedule; not a guarantee against upstream omissions"
                                if status=="complete_for_snapshot" else "Cannot establish the complete record from this snapshot"),
            "required_fields":["verified scores"]+([] if all_conditions else ["roof","surface"]),
            "assessment_id":assessment_id,"evaluation_toronto_date":cutoff,"source_sha256":source_hash,
            "population_basis":population,"team":team,"season":season,"game_type":game_type,
            "conditions":"all" if all_conditions else "outdoors and natural grass",
            "games_in_scope":len(rows),"blocking_games":blockers,
            "excluded_by_known_conditions":excluded,"pending_games":pending,
            "complete_record":values if status=="complete_for_snapshot" else None,
            "known_subset":values if status=="incomplete" else None,
            "conditions_basis":"Retrospective gamebook evidence; no forecast or injury coverage claimed"}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db",required=True)
    parser.add_argument("--assessment-id",type=int,required=True)
    parser.add_argument("--team",required=True)
    parser.add_argument("--season",type=int,required=True)
    parser.add_argument("--game-type",default="REG")
    parser.add_argument("--all-conditions",action="store_true",help="Scores-only record; no roof/surface filtering")
    args=parser.parse_args(argv)
    try:
        result=query_record(args.db,args.assessment_id,args.team.strip().upper(),args.season,
                            all_conditions=args.all_conditions,game_type=args.game_type)
    except (sqlite3.DatabaseError,ValueError,OSError) as exc:
        print(json.dumps({"coverage_status":"not_checked","complete_record":None,"error":str(exc)}))
        return 2
    print(json.dumps(result,indent=2))
    return 0 if result["coverage_status"]=="complete_for_snapshot" else 3


if __name__=="__main__":
    raise SystemExit(main())
