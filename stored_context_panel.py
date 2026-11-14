"""Native Streamlit renderer for the read-only "Stored game context" panel.

Shown beside the weekly spread picks. Display only: it reads stored research
captures through stored_context.py (guarded read-only connections, rechecked on
every rerun, nothing cached) and never changes picks, probabilities or exports.
"""
from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
import streamlit as st

import stored_context as sc

GAME_KEY = "stored_context_game"


def _md(text) -> str:
    """Escape '$' so data text never renders as LaTeX."""
    return str(text).replace("$", "\\$")


def _when(dt) -> str:
    return f"{sc.toronto(dt)} ({dt:%Y-%m-%d %H:%M:%S} UTC)" if dt else "unknown"


def _game_options(games: pd.DataFrame) -> dict[str, str]:
    """game_id -> label built only from immutable identity fields."""
    out = {}
    for _, r in games.iterrows():
        gid = str(r["game_id"])
        if gid not in out:
            out[gid] = f"{sc._text(r.get('away_team'))} @ {sc._text(r.get('home_team'))} · " \
                       f"{sc._text(r.get('gameday'))[:10]} · {gid}"
    return out


def render(games: pd.DataFrame, *, environ=None, now: datetime | None = None) -> None:
    st.write("### Stored game context")
    st.caption("Read-only research captures for owner review. They are not model inputs and do not "
               "change the picks, probabilities or edges above.")
    cfg = sc.load_config(environ)
    if cfg["state"] == "not_configured":
        st.info(sc.NOT_CONFIGURED)
        return
    if cfg["state"] == "error":
        st.error(_md(cfg["message"]))
        return
    options = _game_options(games)
    if not options:
        st.info("No games to show context for.")
        return
    # One shared read of each database per render: every offered game once.
    result = sc.build_all(games, list(options), environ=environ, now=now)
    if result["state"] != "ok":
        st.error(_md(result["message"]))
        return
    _coverage(sc.coverage(result, list(options)), _span(games, options))
    game_id = st.selectbox("Game", list(options), format_func=lambda g: options.get(g, g), key=GAME_KEY)
    ctx = result["contexts"][str(game_id)]
    if ctx["state"] != "ok":
        st.error(_md(f"Context not shown: {ctx['message']}"))
        return
    g = ctx["game"]
    st.caption(_md(f"Cutoff {sc.toronto(ctx['cutoff'])} - the earlier of now and kickoff "
                   f"({ctx['cutoff_basis']}). Kickoff {sc.toronto(g['kickoff_utc'])}. Captures received "
                   "after the cutoff are excluded; source entry dates are not capture times."))
    _receipt_times(ctx)
    with st.container(border=True):
        _injuries(ctx["injuries"], g)
    with st.container(border=True):
        _weather(ctx["weather"], g)


# --------------------------------------------------------------------------
# Coverage across the selector and the selected game's receipt times
# --------------------------------------------------------------------------

COVERAGE_COLUMNS = {"usable": "Usable", "missing": "Missing (nothing usable by cutoff)",
                    "error": "Invalid or error", "not_configured": "Not configured"}


WEEKS = range(1, 23)    # nflverse schedule weeks: 1-18 regular season, 19-22 postseason


def _week(value) -> int | None:
    """A finite, whole-number week in WEEKS, else None (unknown). Never truncates."""
    try:
        w = float(sc._text(value))
    except ValueError:
        return None
    if not math.isfinite(w) or not w.is_integer() or int(w) not in WEEKS:
        return None
    return int(w)


def _span(games: pd.DataFrame, options: dict) -> str:
    """Describe the selector's actual population (it may cover several weeks).

    Display only: week metadata never affects which games are offered or their context.
    A game whose rows lack a valid week, or disagree on it, is counted as week unknown.
    """
    rows = games[games["game_id"].astype(str).isin(list(options))]
    per_game: dict[str, set] = {}
    for gid, w in zip(rows["game_id"].astype(str), rows["week"] if "week" in rows.columns else [None] * len(rows)):
        per_game.setdefault(gid, set()).add(_week(w))
    weeks = sorted({next(iter(s)) for s in per_game.values() if len(s) == 1 and None not in s})
    unknown = sum(1 for s in per_game.values() if len(s) != 1 or None in s)
    parts = []
    if weeks:
        parts.append(f"week {weeks[0]}" if len(weeks) == 1 else "weeks " + ", ".join(str(w) for w in weeks))
    if unknown:
        parts.append(f"week unknown for {unknown} game(s)")
    days = sorted(d for d in (sc._text(v)[:10] for v in rows.get("gameday", [])) if d)
    if days:
        parts.append(days[0] if days[0] == days[-1] else f"{days[0]} to {days[-1]}")
    return "; ".join(parts) or "dates unknown"


def _coverage(cov: dict, span: str) -> None:
    n = cov["total"]
    st.write(_md(f"**Coverage across the {n} game(s) in this selector** ({span})"))
    rows = [{"Context": label, **{COVERAGE_COLUMNS[s]: cov[kind][s] for s in sc.COVERAGE_STATES},
             "Games in selector": n}
            for kind, label in (("injury", "Injury reports"), ("weather", "Weather forecasts"))]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Usable means accepted by the exact game, team, kickoff, venue (weather) and pregame-cutoff "
               "checks. Missing means nothing usable was received by the cutoff; it does not mean healthy, "
               "indoor or neutral weather. Invalid or error means a game, capture or database failed a check "
               "or could not be read. Injury and weather are counted separately, each game once per type.")


def _receipt_line(section: dict, received: datetime | None, now: datetime) -> str:
    state = section["state"]
    if state == "ok":
        return (f"received {_when(received)}; {sc.receipt_age_hours(received, now):.1f} h since receipt")
    if state == "none":
        return "missing: no usable capture was received by the cutoff"
    if state == "not_configured":
        return "not configured"
    return "invalid or error: " + section.get("message", "unknown problem")


def _receipt_times(ctx: dict) -> None:
    now = ctx["now"]
    inj, wx = ctx["injuries"], ctx["weather"]
    st.write("**Selected game: capture receipt times**")
    st.markdown(_md(
        f"- Injury report: {_receipt_line(inj, (inj.get('latest') or {}).get('received_at_utc'), now)}\n"
        f"- Weather forecast: {_receipt_line(wx, (wx.get('capture') or {}).get('received_at_utc'), now)}"))
    st.caption(sc.AGE_NOTE)
    st.caption(sc.AVAILABILITY_NOTE)


# --------------------------------------------------------------------------
# Injuries
# --------------------------------------------------------------------------

_ENTRY_COLUMNS = {"player_name": "Player", "position": "Pos", "source_status": "Reported status",
                  "injury_type": "Injury", "injury_detail": "Detail", "short_comment": "Comment",
                  "source_return_date": "Return date (source)", "source_entry_date_raw": "Entry date (source, unverified)",
                  "identity": "Identity"}


def _identity(row: dict) -> str:
    return f"ESPN ID {row['player_id']}" if row.get("player_id") else "no source ID"


def _injuries(inj: dict, game: dict) -> None:
    st.write("**Injury reports (stored captures)**")
    st.caption(_md(sc.INJURY_SOURCE_NOTE))
    if inj["state"] == "not_configured":
        st.info("Injury database: " + sc.NOT_CONFIGURED)
        return
    if inj["state"] == "error":
        st.error(_md(inj["message"]))
        return
    for f in inj["newer_failed"]:
        st.warning(_md(f"A newer injury capture failed: {f['capture_id']} received {_when(f['received_at_utc'])}, "
                       f"outcome {f['outcome']} (scope: {f['scope']}). "
                       + ("The capture shown below is older and may not be current."
                          if inj["latest"] else "No usable capture is available.")))
    for n in inj["notices"]:
        st.warning(_md(n))
    if inj["excluded_received_after_cutoff"]:
        st.caption(f"{inj['excluded_received_after_cutoff']} capture(s) received after the cutoff are excluded.")
    if inj["latest"] is None:
        st.info(_md(inj["message"]))
        return
    latest = inj["latest"]
    st.caption(_md(f"Showing capture {latest['capture_id']}, received {_when(latest['received_at_utc'])} "
                   f"(slot {latest['slot_id']}, {latest['slot_status']}, outcome {latest['outcome']})."))
    for team in (game["away_team"], game["home_team"]):
        t = inj["teams"][team]
        st.write(_md(f"**{team}**: {t['entries']} retained entries (coverage: {t['coverage_state']})"))
        if len(t["entries_per_feed_team"]) == 1:
            st.caption(f"Every team in this capture has exactly {t['entries_per_feed_team'][0]} entries, "
                       "which suggests a per-team cap; older entries may be omitted.")
        if t["rows"]:
            df = pd.DataFrame([{**r, "identity": _identity(r)} for r in t["rows"]])
            st.dataframe(df[list(_ENTRY_COLUMNS)].rename(columns=_ENTRY_COLUMNS), hide_index=True)
    if inj["previous"] is None:
        st.caption("Only one eligible capture; no changes to show.")
        return
    prev = inj["previous"]
    st.write(_md(f"**Changes since {prev['capture_id']}** (received {_when(prev['received_at_utc'])})"))
    for team in (game["away_team"], game["home_team"]):
        _comparison(team, inj["comparisons"][team])


def _comparison(team: str, cmp: dict) -> None:
    if "refused" in cmp:
        st.warning(_md(f"{team}: comparison not available ({cmp['refused']})"))
        return
    c = cmp["counts"]
    st.write(_md(f"{team}: {c['newly_observed']} newly observed, {c['changed']} changed, "
                 f"{c['unchanged']} unchanged, {c['absent_from_later']} not listed in the later capture, "
                 f"{c['identity_unresolved']} with unresolved identity"))
    rows = []
    for e in cmp["newly_observed"]:
        rows.append({"Change": "newly observed", "Player": e["player_name"], "Identity": e["identity_basis"],
                     "Earlier status": None, "Later status": e["source_status"], "Fields changed": None})
    for e in cmp["changed"]:
        rows.append({"Change": "changed", "Player": e["player_name"], "Identity": e["identity_basis"],
                     "Earlier status": e["earlier_status"], "Later status": e["later_status"],
                     "Fields changed": ", ".join(e["changed_fields"])})
    for e in cmp["absent_from_later"]:
        rows.append({"Change": "not listed later (not a recovery)", "Player": e["player_name"],
                     "Identity": e["identity_basis"], "Earlier status": e["source_status"],
                     "Later status": None, "Fields changed": None})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    if c["absent_from_later"]:
        st.warning(_md(f"{team}: {c['absent_from_later']} entry(ies) are not listed in the later capture. "
                       f"This is not a recovery or clearance: {cmp['absent_from_later_meaning']}."))
    if cmp["identity"]["matched_by_inferred_name"]:
        st.caption(f"{team}: {cmp['identity']['matched_by_inferred_name']} match(es) are "
                   "inferred_name_match (no source ID on either side; exact unique name).")
    unresolved = cmp["identity_unresolved"]["earlier"] + cmp["identity_unresolved"]["later"]
    if unresolved:
        st.warning(_md(f"{team}: {len(unresolved)} entry(ies) with unresolved identity were not compared."))
        st.dataframe(pd.DataFrame([{"Player": u["player_name"], "Source ID": u["player_id"],
                                    "Reported status": u["source_status"], "Reason": u["reason"]}
                                   for u in unresolved]), hide_index=True)
    for n in cmp["evidence_notes"]:
        st.caption(_md(n))


# --------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------

def _weather(w: dict, game: dict) -> None:
    st.write("**Weather (captured model forecasts)**")
    st.caption(_md(sc.WEATHER_SOURCE_NOTE))
    if w["state"] == "not_configured":
        st.info("Weather database: " + sc.NOT_CONFIGURED)
        return
    if w["state"] == "error":
        st.error(_md(w["message"]))
        return
    for f in w["newer_failed"]:
        st.warning(_md(f"A newer forecast capture failed: {f['capture_id']} received {_when(f['received_at_utc'])}, "
                       f"outcome {f['outcome']} ({f['detail']}). "
                       + ("The forecast shown below is older." if w["capture"] else "")))
    if w["excluded_received_after_cutoff"]:
        st.caption(f"{w['excluded_received_after_cutoff']} capture(s) received after the cutoff are excluded.")
    cap = w["capture"]
    if cap is None:
        st.info(sc.WEATHER_MISSING_NOTE)
        st.caption(sc.ATTRIBUTION)
        return
    st.caption(_md(f"Capture {cap['capture_id']}, received {_when(cap['received_at_utc'])}; "
                   f"{cap['age_since_capture_hours']:.1f} h since capture. Forecast issue time: not available. "
                   f"Venue {cap['stadium']} ({cap['field_exposure']}, schedule roof {cap['schedule_roof']})."))
    if cap["missing_variables"]:
        st.warning("Not returned in this capture: " + ", ".join(cap["missing_variables"]))
    if cap["missing_requested_hours"]:
        st.warning("Requested hours not returned: " + ", ".join(cap["missing_requested_hours"]))
    units = cap["units"]
    label = {"temperature_2m": "Temperature", "wind_speed_10m": "Wind", "wind_gusts_10m": "Gusts (prev. hour max)",
             "precipitation": "Precipitation (prev. hour)", "precipitation_probability": "Precip. probability (prev. hour)"}
    rows = []
    for h in w["hours"]:
        valid = sc._parse(h["valid_time_utc"])
        row = {"Valid time (UTC)": h["valid_time_utc"], "Valid time (Toronto)": sc.toronto(valid),
               "Minutes from kickoff": round((valid - game["kickoff_utc"]).total_seconds() / 60)}
        for var, name in label.items():
            row[f"{name} [{units[var]}]" if var in units else f"{name} [not returned]"] = h[var]
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Empty cells are values the provider did not supply; nothing is filled in.")
    st.caption(sc.ATTRIBUTION)
