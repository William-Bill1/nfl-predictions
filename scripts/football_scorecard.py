"""Readable, read-only matchup report over an existing EPA research snapshot."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import date
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.football_epa import team_summary, digest
from scripts.football_repository import open_read_only


def _location(conn, assessment_id, game_id):
    """Source-reported location, never inferred from designated home team alone."""
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"game_observations", "completion_assessments"} <= names:
        return None
    rows = conn.execute("SELECT o.location FROM game_observations o JOIN completion_assessments a "
                        "ON a.import_id=o.import_id WHERE a.assessment_id=? AND o.game_id=?",
                        (assessment_id, game_id)).fetchall()
    return rows[0][0] if len(rows) == 1 and rows[0][0] in ("Home", "Neutral") else None


def _iso_date(value):
    """A missing or non-ISO date cannot be compared with the cutoff, so it is refused."""
    try:
        return date.fromisoformat(value).isoformat() == value
    except (TypeError, ValueError):
        return False


def _record(games):
    margins = [g["margin"] for g in games]
    return {"games": len(games), "wins": sum(m>0 for m in margins),
            "losses": sum(m<0 for m in margins), "ties": sum(m==0 for m in margins)}


def opponent_records(finals, team, game_ids):
    """Pooled opponent W-L-T excluding all games against subject; repeats intentional."""
    games = {}
    for gid,day,home,away,hs,aws in finals:
        if gid in games or not _iso_date(day) or home == away or any(not isinstance(s,int) or s<0 for s in (hs,aws)):
            raise ValueError(f"Invalid or duplicate audited opponent game: {gid}")
        games[gid] = (day,home,away,hs,aws)
    details=[]
    for gid in game_ids:
        if gid not in games or team not in games[gid][1:3]:
            raise ValueError(f"Selected game missing from opponent population: {gid}")
        day,home,away,_,_ = games[gid]
        opponent = away if team==home else home
        wins=losses=ties=0
        for _,other_home,other_away,hs,aws in games.values():
            if opponent not in (other_home,other_away) or team in (other_home,other_away):
                continue
            margin = hs-aws if opponent==other_home else aws-hs
            wins += margin>0
            losses += margin<0
            ties += margin==0
        details.append({"game_id":gid,"gameday":day,"opponent":opponent,
                        "wins":wins,"losses":losses,"ties":ties,"games":wins+losses+ties})
    w,l,t = (sum(d[k] for d in details) for k in ("wins","losses","ties"))
    missing=[d["game_id"] for d in details if not d["games"]]
    return {"status":"no_data" if not details else "incomplete" if missing else "covered_for_snapshot",
            "subject_games":len(details),"opponent_wins":w,"opponent_losses":l,"opponent_ties":t,
            "opponent_game_observations":w+l+t,
            "opponent_win_fraction":(w+.5*t)/(w+l+t) if w+l+t and not missing else None,
            "games_without_opponent_sample":missing,"details":details,
            "definition_version":"opponent-record-v1-exclude-subject-pooled",
            "basis":"Retained audited finals before cutoff; games missing from the snapshot are not counted, so not complete or live standings; exclude games against subject team; repeat opponents counted per meeting; ties half-win. Opponent records use the full season-to-cutoff even for last-four subject selection. Not opponent-adjusted EPA or a power rating."}


def matchup(db, teams, before, context_db=None, game_id=None, turnover_db=None):
    if len(teams) != 2 or teams[0] == teams[1]:
        raise ValueError("Provide two different teams")
    with closing(open_read_only(db, require_current_schema=False)) as conn:
        provenance = dict(conn.execute("SELECT key,value FROM metadata"))
        first, latest = conn.execute("SELECT MIN(gameday),MAX(gameday) FROM games").fetchone()
    report = {"teams": list(teams), "before": before,
            "snapshot_game_dates": {"first": first, "latest": latest},
            "provenance": provenance,
            "comparisons": {team: {"season": team_summary(db, team, before),
                                   "last_four": team_summary(db, team, before, 4)} for team in teams}}
    report["score_results"] = {}
    report["selected_game"] = None
    report["opponent_records"] = {}
    if game_id and context_db is None:
        raise ValueError("A selected game requires --context-db")
    if context_db is not None:
        if digest(context_db) != provenance["audit_sha256"]:
            raise ValueError("Context database does not match the EPA snapshot's pinned audit hash")
        with closing(open_read_only(context_db, require_current_schema=False)) as conn:
            assessment_id = int(provenance["assessment_id"])
            opponent_finals = conn.execute("SELECT game_id,gameday,home_team,away_team,home_score,away_score "
                                           "FROM query_coverage_games WHERE assessment_id=? AND season=? AND game_type='REG' "
                                           "AND completion_status='final' AND scores_verified=1 AND gameday<?",
                                           (assessment_id,int(provenance["season"]),before)).fetchall()
            if game_id:
                rows = conn.execute("SELECT home_team,away_team,gameday FROM query_coverage_games "
                                    "WHERE assessment_id=? AND game_id=?", (assessment_id,game_id)).fetchall()
                if (len(rows) != 1 or set(rows[0][:2]) != set(teams)
                        or not _iso_date(rows[0][2]) or rows[0][2] < before):
                    raise ValueError("Selected game must uniquely match both teams and be on/after the cutoff")
                home,away,day = rows[0]
                location = _location(conn,assessment_id,game_id)
                report["selected_game"] = {"game_id":game_id,"gameday":day,"home_team":home,"away_team":away,
                                           "location":location,"basis":"source-reported location; not independently verified home-field advantage"}
            for team in teams:
                report["opponent_records"][team] = {}
                for window, summary in report["comparisons"][team].items():
                    report["opponent_records"][team][window] = opponent_records(opponent_finals,team,summary["game_ids"])
                    margins = []
                    results = []
                    for gid in summary["game_ids"]:
                        rows = conn.execute("SELECT home_team,away_team,home_score,away_score FROM query_coverage_games "
                                            "WHERE assessment_id=? AND game_id=? AND completion_status='final' AND scores_verified=1",
                                            (int(provenance["assessment_id"]), gid)).fetchall()
                        if len(rows) != 1 or team not in rows[0][:2]:
                            raise ValueError(f"Missing or ambiguous verified score: {gid}")
                        home, away, hs, aws = rows[0]
                        if any(not isinstance(s, int) or s < 0 for s in (hs, aws)):
                            raise ValueError(f"Invalid verified score: {gid}")
                        margins.append((hs-aws) if team == home else (aws-hs))
                        location = _location(conn,assessment_id,gid)
                        role = "neutral" if location == "Neutral" else ("home" if team == home else "away") if location == "Home" else "unknown"
                        with closing(open_read_only(db, require_current_schema=False)) as epa_conn:
                            day = epa_conn.execute("SELECT gameday FROM games WHERE game_id=?",(gid,)).fetchone()[0]
                        results.append({"game_id":gid,"gameday":day,"opponent":away if team==home else home,
                                        "score_for":hs if team==home else aws,"score_against":aws if team==home else hs,
                                        "margin":margins[-1],"site":role})
                    # Explicit ordering makes streaks independent of database row order and bye weeks.
                    results.sort(key=lambda r:(r["gameday"],r["game_id"]),reverse=True)
                    streak = None
                    if results:
                        outcome = "W" if results[0]["margin"]>0 else "L" if results[0]["margin"]<0 else "T"
                        count = 1
                        if outcome != "T":
                            for result in results[1:]:
                                if ("W" if result["margin"]>0 else "L" if result["margin"]<0 else "T") != outcome:
                                    break
                                count += 1
                        streak = {"outcome":outcome,"games":count}
                    report["score_results"][f"{team}:{window}"] = {
                        "games": len(margins), "point_differential": sum(margins),
                        "point_differential_per_game": sum(margins)/len(margins) if margins else None,
                        "wins": sum(m>0 for m in margins), "losses": sum(m<0 for m in margins),
                        "ties": sum(m==0 for m in margins),
                        "through":results[0]["gameday"] if results else None,
                        "latest_result":results[0] if results else None,"streak":streak,
                        "site_records":{site:_record([r for r in results if r["site"]==site])
                                        for site in ("home","away","neutral","unknown")},
                        "location_basis":"source-reported Home/Neutral; unknown locations are excluded from named site splits",
                        "record_basis":"final, score-verified games in the retained audited snapshot only; games missing from it are not counted"}
        if digest(context_db) != provenance["audit_sha256"]:
            raise ValueError("Context database changed during score query")
    report["turnovers"] = {}
    if turnover_db is not None:
        from scripts.football_turnovers import summary as turnover_summary
        for team in teams:
            report["turnovers"][team] = {window:turnover_summary(turnover_db,db,team,s["game_ids"])
                                         for window,s in report["comparisons"][team].items()}
    return report


def markdown(report):
    a, b = report["teams"]
    lines = [f"# {a} vs {b}: EPA matchup scorecard", "",
             f"Game-date cutoff: **before {report['before']}** (exclusive).",
             f"Snapshot game dates: {report['snapshot_game_dates']['first']} to {report['snapshot_game_dates']['latest']}.",
             "", "Retrospective, opponent-unadjusted statistics. This cutoff does not prove what was known before a bet.",
             "No projected spread, win probability or betting recommendation is calculated.",
             "", "Higher offensive EPA/success is better; lower defensive EPA/success allowed is better.",
             "Success means EPA > 0. Sacks and scrambles count as dropbacks; two-point attempts are included.",
             ""]

    def value(v, percent=False):
        return "Unavailable" if v is None else f"{v:.1%}" if percent else f"{v:+.3f}"

    def count(v):
        return "Unavailable" if v is None else f"{v:+d}"

    def record(s):
        return "Unavailable" if not s else f"{s['wins']}-{s['losses']}-{s['ties']} ({s['games']} games)"

    lines += [f"## {report['provenance']['season']} regular-season records in the retained audited snapshot", "",
              "Only final, score-verified games in this snapshot are counted; a game missing from it is not counted.", "",
              f"| Record | {a} | {b} |", "|---|---|---|"]
    season_scores = [report["score_results"].get(f"{t}:season") for t in (a,b)]
    lines += [f"| Overall W-L-T | {record(season_scores[0])} | {record(season_scores[1])} |"]
    for site in ("home","away","neutral","unknown"):
        cells = [record(s['site_records'][site]) if s else 'Unavailable' for s in season_scores]
        lines += [f"| {site.title()} W-L-T | {cells[0]} | {cells[1]} |"]
    recent = [record(report['score_results'].get(f'{t}:last_four')) for t in (a,b)]
    lines += [f"| Last four W-L-T | {recent[0]} | {recent[1]} |"]
    for team,s in zip((a,b),season_scores):
        if not s or not s['latest_result']:
            lines += [f"{team}: latest result and streak unavailable."]
            continue
        latest,streak = s['latest_result'],s['streak']
        label = 'Tied last game' if streak['outcome']=='T' else ('Won' if streak['outcome']=='W' else 'Lost')+f" {streak['games']}"
        lines += ["",f"{team}: **{label}**, through {s['through']}. Latest: vs {latest['opponent']}, {latest['score_for']}-{latest['score_against']} ({latest['gameday']})."]
    selected = report.get('selected_game')
    if selected:
        for team,s in zip((a,b),season_scores):
            role = 'neutral' if selected['location']=='Neutral' else ('home' if team==selected['home_team'] else 'away') if selected['location']=='Home' else 'unknown'
            lines += ["",f"**Selected matchup: {team} {role} record: {record(s['site_records'][role]) if s else 'Unavailable'}.**"]
    lines += ["", "Site splits use source-reported locations, not a verified home-field advantage. Neutral games are separate; unknown locations are not assumed home/away.", "Streaks ignore bye weeks; a tie ends a winning or losing streak. All records use the selected audited games before the cutoff.", ""]

    def measure(team, window, side, kind, field):
        return next(m[field] for m in report["comparisons"][team][window]["metrics"]
                    if m["side"] == side and m["kind"] == kind)

    def edge(x, y, lower=False):
        if x is None or y is None:
            return "Not assessed"
        if abs(x-y) < 1e-12:
            return "Even"
        return a if (x<y if lower else x>y) else b

    lines += ["## Head-to-head measurements", "", f"| Category | {a} | {b} | Descriptive edge |",
              "|---|---:|---:|---|"]
    for label, side, field, lower in (("Offensive EPA/play", "offense", "epa_per_play", False),
                                      ("Defensive EPA allowed/play", "defense", "epa_per_play", True),
                                      ("Offensive success rate", "offense", "success_rate", False)):
        x,y = (measure(t,"season",side,"all",field) for t in (a,b))
        lines += [f"| {label} | {value(x, field=='success_rate')} | {value(y, field=='success_rate')} | {edge(x,y,lower)} |"]
    scores = [report["score_results"].get(f"{t}:season", {}) for t in (a,b)]
    x,y = (s.get("point_differential_per_game") for s in scores)
    sites = ["Not supplied", "Not supplied"]
    if selected:
        sites = [("Neutral" if selected["location"]=="Neutral" else
                  ("Home" if t==selected["home_team"] else "Away") if selected["location"]=="Home" else "Unknown")
                 for t in (a,b)]
    turnover_values = [report.get("turnovers",{}).get(t,{}).get("season",{}).get("differential") for t in (a,b)]
    schedule_values = [report.get("opponent_records",{}).get(t,{}).get("season",{}).get("opponent_win_fraction") for t in (a,b)]
    lines += [f"| Point differential/game | {value(x)} | {value(y)} | {edge(x,y)} |",
              f"| Schedule context: opponent win fraction | {value(schedule_values[0],True)} | {value(schedule_values[1],True)} | Not a team edge |",
              f"| Turnover differential (season total) | {count(turnover_values[0])} | {count(turnover_values[1])} | {edge(*turnover_values)} |",
              f"| Game site (no point adjustment) | {sites[0]} | {sites[1]} | Not assessed |"]
    x,y = (report["comparisons"][t]["last_four"]["net_epa_per_play"] for t in (a,b))
    lines += [f"| Recent form: last-four net EPA/play | {value(x)} | {value(y)} | {edge(x,y)} |",
              "| Injury impact | Insufficient data for numeric impact | Insufficient data for numeric impact | Not assessed |",
              "| Composite matchup advantage | Not calculated | Not calculated | Not assessed |", "",
              "No weights or combined total. Descriptive edges compare unrounded values; they are not betting edges.",
              "Turnover differential is takeaways minus giveaways; takeaways use the opponent's interceptions thrown plus total fumbles lost, including touchbacks and special teams. Unavailable means no matching complete turnover snapshot was supplied.",
              ""]
    lines += ["## Opponent-record schedule context", "",
              "Higher opponent win fraction describes a tougher past schedule by this measure, not a stronger team or a betting edge.",
              "Records exclude every game against the subject team. Repeat opponents count once per meeting; ties count as half a win. Pooled records weight opponents by their available game counts.",
              "Only the retained audited season-to-cutoff finals are used; later corrections may be present. Missing opponent samples withhold the percentage.",
              "Games missing from the snapshot (for example, finals not yet score-verified) are not counted, so these are not complete or live standings.", ""]
    for team in (a,b):
        for window in ("season","last_four"):
            context = report.get("opponent_records",{}).get(team,{}).get(window)
            if context is None:
                lines += [f"{team} {window}: unavailable; no audited context supplied."]
                continue
            lines += [f"{team} {window}: opponents {context['opponent_wins']}-{context['opponent_losses']}-{context['opponent_ties']}, "
                      f"{value(context['opponent_win_fraction'],True)} over {context['opponent_game_observations']} opponent-game observations "
                      f"from {context['subject_games']} subject games; {context['status']}."]
            if context['games_without_opponent_sample']:
                lines += [f"No opponent sample for: {', '.join(context['games_without_opponent_sample'])}."]
    lines += ["", "For the last-four selection, opponent records still use the full season before the cutoff.", ""]
    for offense, defense in ((a,b),(b,a)):
        for kind,label in (("dropback","Passing"),("designed_rush","Rushing")):
            x=measure(offense,"season","offense",kind,"epa_per_play")
            y=measure(defense,"season","defense",kind,"epa_per_play")
            lines += [f"{label} matchup: {offense} offense {value(x)} EPA/play versus {defense} defense {value(y)} EPA allowed/play."]
    lines += ["", "These matchup components are shown separately, without an invented composite score.", ""]

    for window, heading in (("season", "Season to date"), ("last_four", "Last four completed games (or fewer available)")):
        sa, sb = (report["comparisons"][t][window] for t in (a, b))
        lines += [f"## {heading}", "", f"| Measure | {a} | {b} |", "|---|---:|---:|",
                  f"| Games in selection | {sa['games']} | {sb['games']} |",
                  f"| Selection coverage | {sa['coverage']['status']} | {sb['coverage']['status']} |",
                  f"| Net EPA/play | {value(sa['net_epa_per_play'])} | {value(sb['net_epa_per_play'])} |"]
        for side, label in (("offense", "Offense"), ("defense", "Defense allowed")):
            for kind, kind_label in (("all", "all eligible plays"), ("dropback", "dropbacks"), ("designed_rush", "designed rushes")):
                ma, mb = (next(m for m in s["metrics"] if m["side"] == side and m["kind"] == kind) for s in (sa, sb))
                prefix = f"{label}: {kind_label}"
                lines += [f"| {prefix}, EPA/play | {value(ma['epa_per_play'])} | {value(mb['epa_per_play'])} |",
                          f"| {prefix}, success rate | {value(ma['success_rate'], True)} | {value(mb['success_rate'], True)} |",
                          f"| {prefix}, measured / eligible plays | {ma['measured_plays']} / {ma['eligible_plays']} | {mb['measured_plays']} / {mb['eligible_plays']} |"]
        for team, summary in ((a, sa), (b, sb)):
            coverage = summary["coverage"]
            lines += ["", f"{team} game IDs: " + (", ".join(summary["game_ids"]) or "None") + "."]
            if coverage["status"] != "covered":
                lines += [f"{team}: rates withheld. Games without plays: {coverage['games_without_eligible_plays']}; "
                          f"without both offenses: {coverage['games_without_both_offenses']}; "
                          f"eligible plays missing EPA: {coverage['eligible_plays_missing_epa']}."]
        lines += [""]
    lines += ["## Provenance and limits", "",
              "Coverage does not establish that every provider play is present. Raw ad hoc SQL does not automatically apply scorecard coverage checks.",
              "Injuries, forecasts, prices and prior meetings are not included in this EPA-only report.", ""]
    lines += [f"- {key}: `{val}`" for key, val in sorted(report["provenance"].items())]
    for team,windows in report.get("turnovers",{}).items():
        for window,summary in windows.items():
            lines += [f"- {team} {window} turnovers: {json.dumps(summary,sort_keys=True)}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--teams", nargs=2, required=True)
    parser.add_argument("--before", required=True)
    parser.add_argument("--context-db", help="Optional audited score database; must match pinned audit hash")
    parser.add_argument("--game-id", help="Optional selected matchup; highlights applicable site record")
    parser.add_argument("--turnover-db",help="Optional team-stat turnover snapshot pinned to this EPA database")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", help="New output file; existing files are never replaced")
    args = parser.parse_args()
    try:
        report = matchup(args.db, args.teams, args.before, args.context_db, args.game_id,args.turnover_db)
        content = markdown(report) if args.format == "markdown" else json.dumps(report, indent=2) + "\n"
        if args.output:
            with Path(args.output).open("x", encoding="utf-8") as stream:
                stream.write(content)
        else:
            print(content)
        return 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
