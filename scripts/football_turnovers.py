"""Research turnover snapshot from supplied nflverse weekly team statistics."""
from contextlib import closing
import csv
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.football_epa import digest
from scripts.football_repository import open_read_only

VERSION = "turnovers-v1-opponent-giveaways"


def count(value):
    try:
        number = Decimal(value)
        if not number.is_finite() or number < 0 or number != number.to_integral():
            raise ValueError("Turnover counts must be nonnegative integers")
        return int(number)
    except (InvalidOperation, TypeError):
        raise ValueError("Missing or malformed turnover count") from None


def build(source, epa_db, output):
    source_hash, epa_hash = digest(source), digest(epa_db)
    with closing(open_read_only(epa_db, require_current_schema=False)) as conn:
        games = {r[0]:r[1:] for r in conn.execute("SELECT game_id,gameday,home_team,away_team FROM games")}
        meta = dict(conn.execute("SELECT key,value FROM metadata"))
    rows = {}
    required = {"game_id","team","opponent_team","season","season_type","passing_interceptions","fumbles_lost_total"}
    with Path(source).open(encoding="utf-8-sig",newline="") as stream:
        reader = csv.DictReader(stream)
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Missing required team-stat columns")
        for row in reader:
            gid = row["game_id"]
            if gid not in games:
                continue
            _,home,away = games[gid]
            if row["season"] != meta["season"] or row["season_type"] != "REG" or {row["team"],row["opponent_team"]} != {home,away}:
                raise ValueError(f"Team-stat identity mismatch: {gid}")
            key = gid,row["team"]
            if key in rows:
                raise ValueError(f"Duplicate team-game row: {key}")
            rows[key] = (count(row["passing_interceptions"]),count(row["fumbles_lost_total"]))
    expected = {(gid,team) for gid,(_,home,away) in games.items() for team in (home,away)}
    if set(rows) != expected:
        raise ValueError(f"Incomplete team statistics: {len(expected-set(rows))} team-game rows missing")
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.executescript("""
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE team_game_turnovers(game_id TEXT,team TEXT,opponent TEXT,gameday TEXT,
          interceptions_thrown INTEGER,fumbles_lost INTEGER,giveaways INTEGER,takeaways INTEGER,
          differential INTEGER,PRIMARY KEY(game_id,team));
        """)
        for gid,(day,home,away) in games.items():
            for team,opponent in ((home,away),(away,home)):
                ints,fumbles = rows[gid,team]
                giveaways,takeaways = ints+fumbles,sum(rows[gid,opponent])
                conn.execute("INSERT INTO team_game_turnovers VALUES(?,?,?,?,?,?,?,?,?)",
                             (gid,team,opponent,day,ints,fumbles,giveaways,takeaways,takeaways-giveaways))
        conn.executemany("INSERT INTO metadata VALUES(?,?)", {"definition_version":VERSION,"source_sha256":source_hash,
                         "epa_sha256":epa_hash,"basis":"takeaways = opponent interceptions thrown + opponent total fumbles lost; includes special teams and touchbacks"}.items())
        conn.commit()
        if digest(source)!=source_hash or digest(epa_db)!=epa_hash:
            raise ValueError("Input changed during build")
        with Path(output).open("xb") as stream:
            stream.write(conn.serialize())
    return {"games":len(games),"team_game_rows":len(rows),"definition_version":VERSION}


def summary(db, epa_db, team, game_ids):
    with closing(open_read_only(db,require_current_schema=False)) as conn:
        meta = dict(conn.execute("SELECT key,value FROM metadata"))
        if meta.get("definition_version") != VERSION or meta.get("epa_sha256") != digest(epa_db):
            raise ValueError("Turnover snapshot does not match this EPA snapshot/definition")
        rows=[]
        for gid in game_ids:
            found = conn.execute("SELECT giveaways,takeaways,differential FROM team_game_turnovers WHERE game_id=? AND team=?",(gid,team)).fetchall()
            if len(found)!=1:
                return {"status":"incomplete","games":len(game_ids),"giveaways":None,"takeaways":None,"differential":None}
            rows.append(found[0])
    return {"status":"covered" if rows else "no_data","games":len(rows),
            **{key:sum(r[i] for r in rows) if rows else None for i,key in enumerate(("giveaways","takeaways","differential"))},
            "provenance":meta}


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True)
    parser.add_argument("--epa-db",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    try:
        print(json.dumps(build(args.source,args.epa_db,args.output),indent=2))
    except (ValueError,OSError,sqlite3.Error) as exc:
        print(str(exc),file=sys.stderr)
        sys.exit(2)
