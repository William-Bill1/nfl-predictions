"""Optional manual Ontario history beside research cards; no requests or pricing model."""
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import kickoff_corrections as kc
import ontario_spreads as on
import pregame_snapshots as ps
from scripts import fanduel_ontario_timeline as timeline
from scripts.football_epa import digest

TORONTO = ZoneInfo("America/Toronto")


def fingerprints(manual_dir, schedule, snapshot):
    return {"schedule": digest(Path(schedule)), "model_snapshot": digest(Path(snapshot)),
            "quotes": {str(p): digest(p) for p in sorted(Path(manual_dir).glob("*.json"))}}


def load(manual_dir, schedule, as_of, snapshot, *, season, week):
    before = fingerprints(manual_dir, schedule, snapshot)
    history = timeline.timeline(manual_dir, schedule, as_of, reassess_kickoffs=True)
    model = json.loads(Path(snapshot).read_text(encoding="utf-8"))
    ps.validate_snapshot(model, str(snapshot))
    if on.parse_utc(model["captured_at"]) > on._parse_observed(as_of):
        raise ValueError("Model snapshot was captured after market as-of cutoff")
    frame, _ = on.load_schedule(Path(schedule))
    groups = {(g["game_id"], g["team"]): g for g in history["groups"]}
    days = {on.parse_utc(q["observed_at"]).astimezone(TORONTO).date().isoformat()
            for g in history["groups"] for q in g["observations"]
            if (q["season"], q["week"]) == (season, week)}
    days.add(on._parse_observed(as_of).astimezone(TORONTO).date().isoformat())
    if fingerprints(manual_dir, schedule, snapshot) != before:
        raise ValueError("Market inputs changed during read")
    return {"history": history, "model": model, "schedule": frame, "groups": groups,
            "days": sorted(days), "hashes": before}


def for_game(loaded, gid, away, home, season, week, gameday=None):
    rows = loaded["schedule"][loaded["schedule"].game_id == gid]
    if len(rows) != 1:
        raise ValueError("Market schedule game identity missing or duplicated: " + gid)
    game = rows.iloc[0]
    if (game.away_team, game.home_team, int(game.season), int(game.week)) != (away, home, season, week):
        raise ValueError("Market schedule identity differs from scorecard: " + gid)
    if gameday is not None and str(game.gameday) != gameday:
        raise ValueError("Market schedule date differs from scorecard: " + gid)
    # Reviewed kickoffs: the supplied schedule must already carry the reviewed time.
    # Unexpected values raise KickoffReviewError (a ValueError); stale times are refused.
    review = kc.resolve(gid, season=game.season, week=game.week, away_team=game.away_team,
                        home_team=game.home_team, gameday=game.gameday, gametime=game.gametime)
    if review is not None and review["status"] != kc.SOURCE_ALREADY_CORRECTED:
        raise ValueError(f"Market schedule lists the pre-change kickoff {review['source_gametime']} for {gid}; "
                         f"reviewed kickoff is {review['effective_gametime']} ({review['authority']})")
    kickoff = on.parse_utc(review["effective_kickoff_utc"]) if review else ps.kickoff_utc(game.gameday, game.gametime)[0]
    if kickoff is None or on.parse_utc(loaded["model"]["captured_at"]) >= kickoff:
        raise ValueError("Model snapshot is not pregame for market schedule: " + gid)
    models = [g for g in loaded["model"]["games"] if g["game_id"] == gid]
    model = models[0] if models else None
    if model and (model["away_team"], model["home_team"], model["season"], model["week"]) != (away, home, season, week):
        raise ValueError("Model snapshot identity differs from scorecard: " + gid)
    kickoff_day = kickoff.astimezone(TORONTO).date().isoformat()
    days = [day for day in loaded["days"] if day <= kickoff_day]
    sides = []
    for team in (away, home):
        group = loaded["groups"].get((gid, team))
        observations = group["observations"] if group else []
        status = model["line_status"] if model else None
        model_line = (model["spread_line"] * (1 if team == away else -1) if status == "valid"
                      else 0.0 if status == "pickem" else None)
        conflict = bool(group and group["conflicting_times"])
        day_rows = []
        previous = None
        for day in days:
            observed = [q for q in observations if on.parse_utc(q["observed_at"]).astimezone(TORONTO).date().isoformat() == day]
            if not observed:
                day_rows.append({"day": day, "quote": None, "change": None, "model_line_matches": None})
            for quote in observed:
                change = None if previous is None or conflict else timeline.compare_quotes([previous, quote])
                day_rows.append({"day": day, "quote": quote, "change": change,
                                 "model_line_matches": None if model_line is None or conflict else quote["handicap"] == model_line})
                previous = quote
        sides.append({"team": team, "model_handicap": model_line, "conflicting_observations": conflict, "rows": day_rows})
    return {"as_of": loaded["history"]["as_of"], "model_captured_at": loaded["model"]["captured_at"],
            "source_hashes": loaded["hashes"], "sides": sides,
            "limits": "Owner-observed FanDuel Ontario history, not live/opening/closing prices. No probability transfer across lines, causal explanation or betting-edge claim."}


def markdown(context):
    lines = ["", "## FanDuel Ontario observed spread history", "",
             f"As-of: {context['as_of']}. Model snapshot captured: {context['model_captured_at']}.",
             "Times below are Toronto local time with UTC offset. Dates shown are observed dates across this week plus the applicable as-of date—not a complete daily or planned-checkpoint calendar. "
             "Dates run to the game's Toronto kickoff date; a listed date without an observation for a team shows No observation, and earlier prices are not carried forward. "
             "Changes compare successive observations, not necessarily successive days.", "",
             "| Team | Observed Toronto time | Handicap | American odds | Points change | Price-only implied change (pp) | Margin touched/crossed | Model line |", 
             "|---|---|---:|---:|---:|---:|---|---|"]
    for side in context["sides"]:
        for row in side["rows"]:
            q, change = row["quote"], row["change"]
            if q is None:
                lines.append(f"| {side['team']} | {row['day']}: No observation | — | — | — | — | — | Not assessed |")
                continue
            time = on.parse_utc(q["observed_at"]).astimezone(TORONTO).isoformat(timespec="minutes")
            points = "—" if change is None or change["spread_change"] is None else f"{change['spread_change']:+g}"
            price = "—" if change is None or change['price_probability_change_pp'] is None else f"{change['price_probability_change_pp']:+.2f}"
            margins = "—" if change is None or not change['crossed_margins'] else ", ".join(f"{v:+g}" for v in change['crossed_margins'])
            label = ("Unavailable" if side['model_handicap'] is None else "0" if side['model_handicap'] == 0
                     else f"{side['model_handicap']:+g}")
            if row['model_line_matches'] is False: label += " — LINE MISMATCH; model probability not transferable"
            if side['conflicting_observations']: label += " — CONFLICT; comparisons withheld"
            lines.append(f"| {side['team']} | {time} | {q['handicap']:+g} | {q['price']:+d} | {points} | {price} | {margins} | {label} |")
    lines += ["", "Price-only implied changes include bookmaker margin; they are not fair probabilities or expected returns.", context['limits'], ""]
    return "\n".join(lines)
