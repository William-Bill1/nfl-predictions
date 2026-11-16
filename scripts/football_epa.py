"""Research-only EPA snapshots. Supplied CSV input; no engine or network wiring."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
from datetime import date

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.football_repository import open_read_only

VERSION = "epa-v1-dropbacks-designed-rush-no-kneel-spike"
REQUIRED = {"game_id", "play_id", "play_type", "posteam", "defteam", "epa", "qb_dropback",
            "rush_attempt", "qb_kneel", "qb_spike"}
SCHEMA = """
CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE games(game_id TEXT PRIMARY KEY,gameday TEXT NOT NULL,home_team TEXT NOT NULL,away_team TEXT NOT NULL);
CREATE TABLE plays(game_id TEXT NOT NULL,play_id TEXT NOT NULL,offense TEXT NOT NULL,
 defense TEXT NOT NULL,kind TEXT NOT NULL,epa REAL,PRIMARY KEY(game_id,play_id));
CREATE VIEW team_game_epa AS
 SELECT p.game_id,g.gameday,p.offense AS team,p.defense AS opponent,'offense' AS side,
 p.kind,COUNT(*) AS eligible_plays,COUNT(p.epa) AS measured_plays,
 SUM(p.epa) AS total_epa,AVG(p.epa) AS epa_per_play,
 AVG(CASE WHEN p.epa IS NULL THEN NULL WHEN p.epa>0 THEN 1.0 ELSE 0.0 END) AS success_rate
 FROM plays p JOIN games g USING(game_id) GROUP BY p.game_id,p.offense,p.kind
 UNION ALL
 SELECT p.game_id,g.gameday,p.defense,p.offense,'defense',p.kind,COUNT(*),COUNT(p.epa),
 SUM(p.epa),AVG(p.epa),AVG(CASE WHEN p.epa IS NULL THEN NULL WHEN p.epa>0 THEN 1.0 ELSE 0.0 END)
 FROM plays p JOIN games g USING(game_id) GROUP BY p.game_id,p.defense,p.kind;
"""


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def flag(value):
    if value in ("0", "0.0"):
        return False
    if value in ("1", "1.0"):
        return True
    raise ValueError(f"Missing or invalid play flag: {value!r}")


def build(pbp, audited_db, output, *, assessment_id, season):
    """Create a NEW database exclusively; refuse malformed data, retain missing EPA."""
    pbp, audited_db, output = map(Path, (pbp, audited_db, output))
    source_hash, audit_hash = digest(pbp), digest(audited_db)
    with open_read_only(audited_db, require_current_schema=False) as conn:
        games = conn.execute("SELECT game_id,gameday,home_team,away_team FROM query_coverage_games "
                             "WHERE assessment_id=? AND season=? AND game_type='REG' "
                             "AND completion_status='final' AND scores_verified=1",
                             (assessment_id, season)).fetchall()
    if not games or len({g[0] for g in games}) != len(games):
        raise ValueError("Empty or duplicate audited game population")
    population = {g[0]: g for g in games}
    for gid, day, home, away in games:
        if date.fromisoformat(day).isoformat() != day or not home or not away or home == away:
            raise ValueError(f"Invalid audited game identity/date: {gid}")
    # Build in memory: refusal leaves existing files and source databases untouched.
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO games VALUES(?,?,?,?)", games)
        opener = gzip.open if pbp.suffix == ".gz" else open
        seen = set()
        with opener(pbp, "rt", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not REQUIRED.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing columns: {sorted(REQUIRED-set(reader.fieldnames or []))}")
            for row in reader:
                gid = row["game_id"]
                if gid not in population:
                    continue
                pid = row["play_id"]
                if not pid or (gid, pid) in seen:
                    raise ValueError(f"Missing or duplicate play identity: {gid}/{pid}")
                seen.add((gid, pid))
                # Non-play rows often have no team or flags. They are not eligible.
                if row["play_type"] not in ("pass", "run"):
                    continue
                if {row["posteam"], row["defteam"]} != set(population[gid][2:]):
                    raise ValueError(f"Play teams disagree with audited game: {gid}/{pid}")
                if flag(row["qb_kneel"]) or flag(row["qb_spike"]):
                    continue
                dropback, rush = flag(row["qb_dropback"]), flag(row["rush_attempt"])
                if not (dropback or rush):
                    continue
                raw = row["epa"]
                value = None if raw in ("", "NA", "NaN", "nan") else float(raw)
                if value is not None and not math.isfinite(value):
                    raise ValueError(f"Nonfinite EPA: {gid}/{pid}")
                conn.execute("INSERT INTO plays VALUES(?,?,?,?,?,?)",
                             (gid, pid, row["posteam"], row["defteam"],
                              "dropback" if dropback else "designed_rush", value))
        metadata = {"definition_version": VERSION, "pbp_sha256": source_hash,
                    "audit_sha256": audit_hash, "assessment_id": str(assessment_id),
                    "season": str(season), "knowledge_basis": "retrospective current-source statistics; not historical receipt-time evidence"}
        conn.executemany("INSERT INTO metadata VALUES(?,?)", metadata.items())
        conn.commit()
        if digest(pbp) != source_hash or digest(audited_db) != audit_hash:
            raise ValueError("Input changed during build")
        # Exclusive creation; never replace a previous snapshot.
        with output.open("xb") as stream:
            stream.write(conn.serialize())
    finally:
        conn.close()
    return coverage(output)


def coverage(db, game_ids=None):
    with open_read_only(db, require_current_schema=False) as conn:
        scope = "" if game_ids is None else " WHERE g.game_id IN (" + (",".join("?" for _ in game_ids) or "NULL") + ")"
        params = [] if game_ids is None else list(game_ids)
        missing = conn.execute("SELECT g.game_id FROM games g LEFT JOIN plays p USING(game_id) " + scope +
                               " GROUP BY g.game_id HAVING COUNT(p.play_id)=0 ORDER BY g.game_id", params).fetchall()
        missing_sides = conn.execute("SELECT g.game_id FROM games g JOIN plays p USING(game_id) " + scope +
                                    " GROUP BY g.game_id HAVING COUNT(DISTINCT offense)!=2", params).fetchall()
        absent_epa = conn.execute("SELECT COUNT(*) FROM games g JOIN plays p USING(game_id) " + scope +
                                 (" AND " if scope else " WHERE ") + "epa IS NULL", params).fetchone()[0]
        count = conn.execute("SELECT COUNT(*) FROM games g" + scope, params).fetchone()[0]
        return {"expected_games": count,
                "games_without_eligible_plays": [r[0] for r in missing],
                "games_without_both_offenses": [r[0] for r in missing_sides],
                "eligible_plays_missing_epa": absent_epa,
                "status": "no_data" if not count else "incomplete" if missing or missing_sides or absent_epa else "covered",
                "limitation": "Coverage does not prove every source play is present or correctly valued"}


def team_summary(db, team, before, last=None):
    if date.fromisoformat(before).isoformat() != before:
        raise ValueError("before must be YYYY-MM-DD")
    if last is not None and last < 1:
        raise ValueError("last must be positive")
    with open_read_only(db, require_current_schema=False) as conn:
        if not conn.execute("SELECT 1 FROM games WHERE home_team=? OR away_team=? LIMIT 1", (team,team)).fetchone():
            raise ValueError(f"Unknown team in this snapshot: {team}")
        games = conn.execute("SELECT game_id FROM games WHERE gameday<? AND (home_team=? OR away_team=?) "
                             "ORDER BY gameday DESC,game_id DESC", (before, team, team)).fetchall()
        ids = [g[0] for g in games][:last]
        result = {"team": team, "before": before, "games": len(ids), "game_ids": ids, "metrics": []}
        for side in ("offense", "defense"):
            column = "offense" if side == "offense" else "defense"
            for kind in ("all", "dropback", "designed_rush"):
                params = [team, before]
                clause = ""
                if ids:
                    clause = " AND p.game_id IN (" + ",".join("?" for _ in ids) + ")"
                    params.extend(ids)
                else:
                    clause = " AND 0"
                if kind != "all":
                    clause += " AND kind=?"
                    params.append(kind)
                row = conn.execute(f"SELECT COUNT(*),COUNT(epa),AVG(epa),AVG(CASE WHEN epa IS NULL THEN NULL "
                                   f"WHEN epa>0 THEN 1.0 ELSE 0.0 END) FROM plays p JOIN games g USING(game_id) "
                                   f"WHERE {column}=? AND gameday<?" + clause, params).fetchone()
                result["metrics"].append(dict(zip(("eligible_plays", "measured_plays", "epa_per_play", "success_rate"), row),
                                             side=side, kind=kind))
        result["coverage"] = coverage(db, ids)
        offense, defense = result["metrics"][0]["epa_per_play"], result["metrics"][3]["epa_per_play"]
        result["net_epa_per_play"] = None if offense is None or defense is None else offense - defense
        if result["coverage"]["status"] != "covered":
            # Keep denominators visible, but do not turn partial source data into a rating.
            for metric in result["metrics"]:
                metric["epa_per_play"] = metric["success_rate"] = None
            result["net_epa_per_play"] = None
        result["basis"] = "Descriptive, unadjusted for opponent; no probability or point-spread conversion"
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    for key in ("pbp", "audited-db", "output"):
        build_parser.add_argument("--" + key, required=True)
    build_parser.add_argument("--assessment-id", type=int, required=True)
    build_parser.add_argument("--season", type=int, required=True)
    for name in ("coverage", "scorecard", "sql"):
        p = sub.add_parser(name)
        p.add_argument("--db", required=True)
        if name == "scorecard":
            p.add_argument("--teams", nargs=2, required=True)
            p.add_argument("--before", required=True, help="Exclusive game-date cutoff; not receipt-time proof")
        if name == "sql":
            p.add_argument("--query", required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            result = build(args.pbp, args.audited_db, args.output, assessment_id=args.assessment_id, season=args.season)
        elif args.command == "coverage":
            result = coverage(args.db)
        elif args.command == "scorecard":
            result = {team: {"season": team_summary(args.db, team, args.before),
                             "last_four": team_summary(args.db, team, args.before, 4)} for team in args.teams}
        else:
            with open_read_only(args.db, require_current_schema=False) as conn:
                cursor = conn.execute(args.query)
                result = [dict(zip([c[0] for c in cursor.description], row)) for row in cursor]
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
