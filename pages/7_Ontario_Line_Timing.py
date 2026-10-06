"""Ontario Line Timing - Wednesday-noon vs Sunday-morning Ontario spread quotes.

Builds the Phase 2 comparison in memory from the validated capture artifacts
in data_files/ontario_spreads/ (written by ontario_spreads.py). It makes no
API calls and never shows sample, fixture or regenerated odds. Its only write
is the manual FanDuel Ontario entry form, which saves an observed quote
through ontario_spreads' own validation and persistence - never a bet,
recommendation or capture. See docs/ONTARIO_SPREAD_TRACKING.md.
"""

import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

import ontario_line_timing as olt  # noqa: E402
import ontario_spreads as on  # noqa: E402
from footer import add_betting_oracle_footer  # noqa: E402

ONT, MAN, US = olt.rpt.GROUP_ONTARIO, olt.rpt.GROUP_MANUAL, olt.rpt.GROUP_US_REFERENCE

st.title("Ontario line timing")
st.caption("Ontario sportsbook spreads and prices at Wednesday 12:00 vs Sunday 09:00 "
           "(America/Toronto). Describes how quotes moved.")


@st.cache_data(max_entries=8, show_spinner="Reading captures...")
def load_report(fingerprint: tuple, capture_dir: str, manual_dir: str) -> dict:
    """Cached on the content fingerprint of every source file, so any new,
    changed or deleted capture or manual quote produces a fresh report."""
    return olt.build(Path(capture_dir), Path(manual_dir))


@st.cache_data(max_entries=8)
def load_manual(fingerprint: tuple, manual_dir: str) -> list[dict]:
    return olt.load_manual_docs(Path(manual_dir))


@st.cache_data(max_entries=4)
def load_schedule(path: str, mtime_ns: int, size: int):
    return on.load_schedule(Path(path))[0]


capture_dir, manual_dir = on.CAPTURE_DIR, on.MANUAL_DIR
try:
    fingerprint = olt.source_fingerprint(capture_dir, manual_dir)
    report = load_report(fingerprint, str(capture_dir), str(manual_dir))
    manual_docs = load_manual(fingerprint, str(manual_dir))
except olt.IntegrityError as exc:
    st.error(
        "**Integrity error: a stored Ontario spread artifact failed validation.** "
        "Nothing is shown rather than partial or substituted data. Fix or restore "
        f"the file, then reload.\n\n`{exc}`",
        icon=":material/gpp_bad:",
    )
    st.stop()

now = olt.now_utc()

with st.expander("How to read this page", icon=":material/help:"):
    st.markdown(
        """
- **Two scheduled snapshots per NFL week:** Wednesday 12:00 and Sunday 09:00,
  America/Toronto. A Sunday quote is the line at 9 a.m., **not a closing line**.
- **Spreads** are each team's own handicap: `-3.5` = laying 3.5, `+3.5` = getting
  3.5, `PK` = pick'em. A larger signed number is better for that team.
- **Break-even** is the win rate needed at the American price, with pushes
  excluded (52.4% at -110, 40% at +150).
- **Outcome** compares the two quotes for the same team by **payoff dominance**:
  both bets are settled (win, push or loss) for every possible integer final
  margin. *Sunday dominates* means Sunday's bet pays at least as much for
  every margin and more for at least one (*Wednesday dominates* the reverse).
  *Equivalent payoff* means different quotes that pay the same for every
  margin (e.g. -100 vs +100).
- **Trade-offs** (better on some margins, worse on others, e.g. more points at a
  worse price) are **not ranked**: doing so would need outcome probabilities
  this page doesn't validate.
- Only fresh, valid quotes from both scheduled slots, observed before kickoff,
  are compared. Stale, missing or invalid quotes are shown with the reason.
- This page does **not** establish a best betting time, a predictive edge or
  any increase in ROI.
"""
    )


# ---- manual FanDuel Ontario entry -----------------------------------------
# Preview and Save are both submit buttons of one form, handled in click
# callbacks. Each reads the values actually submitted with that click, so Save
# can only write exactly what was previewed (any change -> preview again), and
# both re-validate against the current clock.



def _submitted_args() -> dict:
    """The manual quote described by the form's submitted values (raises
    ValidationError for missing values or an ambiguous/nonexistent Toronto time)."""
    s = st.session_state
    game = s.get("fd_game")
    if game is None:
        raise on.ValidationError("choose a game")
    if s.get("fd_spread") is None or s.get("fd_price") is None:
        raise on.ValidationError("enter both the spread and the American odds")
    observed = on.toronto_local_to_utc(s["fd_date"], s["fd_time"])
    return {"game_id": game["game_id"], "team": s["fd_team"], "handicap": float(s["fd_spread"]),
            "price": int(s["fd_price"]),
            "opponent_price": None if s.get("fd_opp_price") is None else int(s["fd_opp_price"]),
            "observed_at": on.iso(observed), "note": (s.get("fd_note") or "").strip() or None}


def _prepare(args: dict) -> dict:
    """Phase 1 validation at the current clock, plus the form's own rule that
    the game must not have kicked off yet."""
    now_ = olt.now_utc()
    doc = on.prepare_manual_quote(**args, now=now_, schedule_path=on.SCHEDULE_PATH)
    if on.parse_utc(doc["kickoff_utc"]) <= now_:
        raise on.ValidationError(
            f"{doc['away_team']} @ {doc['home_team']} kicked off at "
            f"{olt.toronto(doc['kickoff_utc'])}; the form records quotes for upcoming games "
            "only (use `ontario_spreads.py manual-quote` for a late entry of a pregame quote)")
    return doc


def _on_preview() -> None:
    st.session_state.pop("fd_pending", None)
    try:
        args = _submitted_args()
        doc = _prepare(args)
    except ValueError as exc:
        st.session_state["fd_result"] = ("error", f"Not valid: {exc}")
        return
    team = args["team"]
    other = doc["away_team"] if team == doc["home_team"] else doc["home_team"]
    st.session_state["fd_pending"] = {
        "token": uuid.uuid4().hex, "args": args,
        "duplicate": on.find_manual_duplicate(doc, on.MANUAL_DIR) is not None,
        "line": olt.bettor_line(team, doc["handicap"], doc["price"]),
        "opponent_line": None if doc["opponent_price"] is None else
        olt.bettor_line(other, doc["opponent_handicap"], doc["opponent_price"]),
        "observed": doc["observed_at"],
        "slot": olt.manual_slot_status(on.parse_utc(doc["observed_at"]), doc["kickoff_utc"]),
    }


def _on_save() -> None:
    """Writes the previewed quote exactly once, and only if the submitted
    values are still exactly the previewed ones."""
    pending = st.session_state.get("fd_pending")
    if not pending:
        st.session_state["fd_result"] = ("error", "Nothing to save: preview the quote first.")
        return
    saved_tokens = st.session_state.setdefault("fd_saved_tokens", set())
    if pending["token"] in saved_tokens:
        st.session_state.pop("fd_pending", None)
        st.session_state["fd_result"] = ("info", "This observation was already saved.")
        return
    try:
        args = _submitted_args()
    except ValueError as exc:
        args, error = None, exc
    if args != pending["args"]:
        st.session_state.pop("fd_pending", None)
        st.session_state["fd_result"] = (
            "error", "Not saved: the form changed after Preview. Check the values and "
                     "preview again." + (f" ({error})" if args is None else ""))
        return
    try:
        doc = _prepare(args)                              # re-validated now
        path, created, doc = on.save_manual_quote(doc, on.MANUAL_DIR)
    except (ValueError, on.CaptureError, OSError) as exc:
        st.session_state.pop("fd_pending", None)
        st.session_state["fd_result"] = ("error", f"Not saved: {exc}")
        return
    saved_tokens.add(pending["token"])
    st.session_state.pop("fd_pending", None)
    line = olt.bettor_line(doc["team"], doc["handicap"], doc["price"])
    st.session_state["fd_result"] = (
        ("success", f"Saved: {line}, observed {olt.toronto(doc['observed_at'])} "
                    f"(manual FanDuel Ontario observation, {path.name}).")
        if created else
        ("info", f"Already recorded: {line}, observed {olt.toronto(doc['observed_at'])} "
                 f"({path.name}). Nothing new was written."))
    # Show the manual view for the saved quote's week, with its filters cleared
    # so the new observation can't be hidden by an earlier selection.
    for key in (f"olt_books_{MAN}", f"olt_game_{MAN}", f"olt_team_{MAN}"):
        st.session_state.pop(key, None)
    st.session_state["olt_view"] = MAN
    st.session_state["olt_season"], st.session_state["olt_week"] = doc["season"], doc["week"]


def _on_discard() -> None:
    st.session_state.pop("fd_pending", None)


with st.expander("Record a FanDuel Ontario quote (manual observation)",
                 icon=":material/edit_note:",
                 expanded="fd_pending" in st.session_state or "fd_result" in st.session_state):
    st.caption("A **manually observed** FanDuel Ontario price, for line-timing comparisons. "
               "It is **not a placed bet**: saving it creates no recommendation and no bet "
               "record. FanDuel Ontario has no API feed, so these are kept separate from "
               "the automated Ontario sportsbooks and from the US FanDuel reference.")
    result = st.session_state.pop("fd_result", None)
    if result:
        getattr(st, result[0])(result[1])

    sched_path = on.SCHEDULE_PATH
    try:
        stat = sched_path.stat()
        games = olt.upcoming_games(load_schedule(str(sched_path), stat.st_mtime_ns, stat.st_size),
                                   now)
    except OSError:
        games = []
    if not games:
        st.info("No upcoming games with a usable kickoff in the next 14 days.")
    else:
        game = st.selectbox("Game", games, format_func=lambda g: g["label"], key="fd_game")
        pending = st.session_state.get("fd_pending")
        if pending and pending["args"]["game_id"] != game["game_id"]:
            st.session_state.pop("fd_pending", None)          # preview belongs to another game
            pending = None
        local_now = now.astimezone(on.TORONTO)
        with st.form("fd_form", border=False):
            st.selectbox("Team", [game["away_team"], game["home_team"]],
                         format_func=lambda t: f"{olt.team_name(t)} ({t})", key="fd_team")
            with st.container(horizontal=True):
                st.number_input("Spread (bettor-facing, signed)", min_value=-60.0,
                                max_value=60.0, step=0.5, value=None, format="%.1f",
                                placeholder="e.g. +4.5 or -3", key="fd_spread",
                                help="As FanDuel shows it for this team: -3.5 lays 3.5, "
                                     "+3.5 gets 3.5, 0 is pick'em.")
                st.number_input("American odds", step=1, value=None,
                                placeholder="e.g. -110 or +105", key="fd_price")
                st.number_input("Opponent's odds (optional)", step=1, value=None,
                                placeholder="e.g. -110", key="fd_opp_price")
            with st.container(horizontal=True):
                st.date_input("Observed on (Toronto)", value=local_now.date(), key="fd_date")
                st.time_input("Observed at (Toronto)",
                              value=local_now.time().replace(second=0, microsecond=0),
                              step=60, key="fd_time")
            st.text_input("Note (optional)", max_chars=200, key="fd_note")
            if pending:
                with st.container(border=True):
                    st.markdown(f"**Preview: {pending['line']}**")
                    if pending["opponent_line"]:
                        st.caption(f"Opponent: {pending['opponent_line']}")
                    st.caption(f"Manually observed FanDuel Ontario quote, not a placed bet. "
                               f"Observed {olt.toronto(pending['observed'])} "
                               f"({pending['observed']} UTC). Save writes exactly these values; "
                               "change anything and you'll need to preview again.")
                    slot_note = pending["slot"]
                    (st.info if slot_note["state"].startswith("intended_") else st.warning)(
                        slot_note["label"])
                    if pending["duplicate"]:
                        st.info("This exact observation is already recorded; saving would "
                                "write nothing new.")
            with st.container(horizontal=True):
                st.form_submit_button("Preview", icon=":material/visibility:", key="fd_preview",
                                      on_click=_on_preview)
                st.form_submit_button("Save observation", type="primary", icon=":material/save:",
                                      key="fd_save", on_click=_on_save,
                                      disabled=not pending or pending["duplicate"])
                st.form_submit_button("Discard preview", key="fd_discard", on_click=_on_discard,
                                      disabled=not pending)

if report["status"] == "no_observations_yet":
    st.info(
        "**No observations yet.** Ontario spread captures run on a schedule: "
        "**Wednesday at 12:00** and **Sunday at 09:00**, America/Toronto, during "
        "the season (the Wednesday window closes at 15:00, the Sunday window at "
        "11:00). This page will show them once the first capture has been "
        "committed. Collection is opt-in (`ODDS_API_KEY`); manual FanDuel Ontario "
        "quotes can be recorded above.",
        icon=":material/schedule:",
    )
    add_betting_oracle_footer()
    st.stop()


def _keep_valid(key: str, options: list) -> None:
    """Drop a remembered widget value that is no longer one of its options."""
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


# ---- source view (Ontario automated by default) --------------------------
st.session_state.setdefault("olt_view", ONT)
view = st.segmented_control("Quotes", options=list(olt.VIEWS), required=True,
                            format_func=olt.VIEWS.get, key="olt_view")
if view == US:
    st.warning(
        "**US reference quotes; not verified as available in Ontario.** These are "
        "the API's FanDuel US prices, shown only for comparison with the Ontario "
        "sportsbooks.",
        icon=":material/public:",
    )
elif view == MAN:
    st.info(
        "**Manual entries.** FanDuel Ontario quotes recorded by hand (with the form "
        "above or `ontario_spreads.py manual-quote`), not from an automated feed. "
        "Each is compared only if it was observed inside its game week's intended "
        "Wednesday or Sunday slot window.",
        icon=":material/edit_note:",
    )

# ---- week selection ------------------------------------------------------
weeks = olt.weeks_with_manual(report, manual_docs)
default = olt.default_week(report) or (max(weeks) if weeks else None)
seasons = sorted({s for s, _ in weeks}, reverse=True)
_keep_valid("olt_season", seasons)
st.session_state.setdefault("olt_season", default[0] if default else seasons[0])
with st.container(horizontal=True):
    season = st.selectbox("Season", seasons, key="olt_season")
    season_weeks = sorted({w for s, w in weeks if s == season}, reverse=True)
    _keep_valid("olt_week", season_weeks)
    st.session_state.setdefault(
        "olt_week", default[1] if default and default[0] == season else season_weeks[0])
    week = st.selectbox("Week", season_weeks, key="olt_week")

slots = olt.week_slots(report, season, week, now)
with st.container(horizontal=True):
    for day in ("wednesday", "sunday"):
        s = slots.get(day, {})
        with st.container(border=True):
            st.markdown(f"**{day.capitalize()} slot**")
            st.caption(s.get("label", "No slot for this week"))

if view == MAN:
    recorded = olt.manual_observation_rows(manual_docs, report, season, week)
    st.markdown("**Recorded manual observations this week**")
    if recorded.empty:
        st.caption("None recorded for this week.")
    else:
        st.dataframe(recorded, hide_index=True, key="olt_manual_recorded")

# ---- filters -------------------------------------------------------------
week_rows = olt.display_rows(report, view, season, week, now)
if week_rows.empty:
    st.info("No comparison rows from this source for the selected week.", icon=":material/info:")
    add_betting_oracle_footer()
    st.stop()

book_options = sorted(week_rows["Sportsbook"].unique())
games_in_week = week_rows[["_game_id", "Matchup"]].drop_duplicates().sort_values("Matchup")
with st.container(horizontal=True):
    picked_books = st.multiselect("Sportsbook", book_options, key=f"olt_books_{view}",
                                  placeholder="All sportsbooks")
    game_label = st.selectbox("Game", ["All games"] + games_in_week["Matchup"].tolist(),
                              key=f"olt_game_{view}")
    team_options = sorted(week_rows["Team"].unique()) if game_label == "All games" else \
        sorted(week_rows.loc[week_rows["Matchup"] == game_label, "Team"].unique())
    team_filter = st.selectbox("Team", ["All teams"] + team_options, key=f"olt_team_{view}")

book_keys = sorted(week_rows.loc[week_rows["Sportsbook"].isin(picked_books), "_book_key"].unique()) \
    if picked_books else None
game_id = None if game_label == "All games" else \
    games_in_week.loc[games_in_week["Matchup"] == game_label, "_game_id"].iloc[0]
rows = olt.display_rows(report, view, season, week, now, books=book_keys, game_id=game_id,
                        team=None if team_filter == "All teams" else team_filter)

if rows.empty:
    st.info("No rows match these filters. Clear a sportsbook, game or team filter "
            "to see more.", icon=":material/filter_alt_off:")
    add_betting_oracle_footer()
    st.stop()


def _n(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


# ---- summary -------------------------------------------------------------
summary = olt.summarize(rows)
with st.container(horizontal=True):
    st.metric("Compared sides", summary["compared_sides"], border=True,
              help="Each game/sportsbook pair has two sides (one per team).")
    st.metric("Sunday dominates", summary["outcomes"].get("Sunday dominates", 0), border=True)
    st.metric("Wednesday dominates", summary["outcomes"].get("Wednesday dominates", 0), border=True)
    st.metric("Trade-offs (not ranked)", summary["outcomes"].get("Trade-off (not ranked)", 0),
              border=True)
    st.metric("Not compared", summary["sides"] - summary["compared_sides"], border=True,
              help="Pending, missed, unmatched, missing or stale - see Status and Reasons.")
st.caption(f"{_n(summary['sides'], 'side')} across {_n(summary['pairs'], 'game/sportsbook pair')} "
           f"in {_n(summary['games'], 'game')}. Counts describe this selection only; no "
           "sportsbook is ranked from them.")

# ---- table ---------------------------------------------------------------
pct = st.column_config.NumberColumn(format="%.1f%%")
st.dataframe(
    rows.drop(columns=["_game_id", "_book_key"]),
    hide_index=True,
    key="olt_table",
    column_config={
        "Wed break-even": pct, "Sun break-even": pct,
        "Matchup": st.column_config.TextColumn(pinned=True),
        "Team": st.column_config.TextColumn(pinned=True),
    },
)
st.caption("Times are America/Toronto. Provider update = when the sportsbook's spread "
           "market last changed according to The Odds API; capture = when this project "
           "requested it.")

add_betting_oracle_footer()
