"""Ontario Line Timing - Wednesday-noon vs Sunday-morning Ontario spread quotes.

Read-only: builds the Phase 2 comparison in memory from the validated capture
artifacts in data_files/ontario_spreads/ (written by ontario_spreads.py). It
makes no API calls and never shows sample, fixture or regenerated odds.
See docs/ONTARIO_SPREAD_TRACKING.md.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

import ontario_line_timing as olt  # noqa: E402
import ontario_spreads as on  # noqa: E402
from footer import add_betting_oracle_footer  # noqa: E402

st.title("Ontario line timing")
st.caption("Ontario sportsbook spreads and prices at Wednesday 12:00 vs Sunday 09:00 "
           "(America/Toronto). Read-only; describes how quotes moved.")


@st.cache_data(max_entries=8, show_spinner="Reading captures...")
def load_report(fingerprint: tuple, capture_dir: str, manual_dir: str) -> dict:
    """Cached on the content fingerprint of every source file, so any new or
    changed capture produces a fresh report."""
    return olt.build(Path(capture_dir), Path(manual_dir))


capture_dir, manual_dir = on.CAPTURE_DIR, on.MANUAL_DIR
try:
    fingerprint = olt.source_fingerprint(capture_dir, manual_dir)
    report = load_report(fingerprint, str(capture_dir), str(manual_dir))
except olt.IntegrityError as exc:
    st.error(
        "**Integrity error: a stored Ontario spread artifact failed validation.** "
        "Nothing is shown rather than partial or substituted data. Fix or restore "
        f"the file, then reload.\n\n`{exc}`",
        icon=":material/gpp_bad:",
    )
    st.stop()

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

if report["status"] == "no_observations_yet":
    st.info(
        "**No observations yet.** Ontario spread captures run on a schedule: "
        "**Wednesday at 12:00** and **Sunday at 09:00**, America/Toronto, during "
        "the season (the Wednesday window closes at 15:00, the Sunday window at "
        "11:00). This page will show them once the first capture has been "
        "committed. Collection is opt-in (`ODDS_API_KEY`); manual FanDuel Ontario "
        "quotes can be added with `python ontario_spreads.py manual-quote`.",
        icon=":material/schedule:",
    )
    add_betting_oracle_footer()
    st.stop()

now = olt.now_utc()

# ---- source view (Ontario automated by default) --------------------------
view = st.segmented_control(
    "Quotes", options=list(olt.VIEWS), default=olt.rpt.GROUP_ONTARIO, required=True,
    format_func=olt.VIEWS.get, key="olt_view",
)
if view == olt.rpt.GROUP_US_REFERENCE:
    st.warning(
        "**US reference quotes; not verified as available in Ontario.** These are "
        "the API's FanDuel US prices, shown only for comparison with the Ontario "
        "sportsbooks.",
        icon=":material/public:",
    )
elif view == olt.rpt.GROUP_MANUAL:
    st.info(
        "**Manual entries.** FanDuel Ontario quotes typed in by hand with "
        "`ontario_spreads.py manual-quote`, not from an automated feed. Each is "
        "used only if it was observed inside the week's intended slot window.",
        icon=":material/edit_note:",
    )

# ---- week selection ------------------------------------------------------
weeks = olt.weeks_available(report)
default = olt.default_week(report)
seasons = sorted({s for s, _ in weeks}, reverse=True)
with st.container(horizontal=True):
    season = st.selectbox("Season", seasons,
                          index=seasons.index(default[0]) if default else 0, key="olt_season")
    season_weeks = sorted({w for s, w in weeks if s == season}, reverse=True)
    week_default = default[1] if default and default[0] == season else season_weeks[0]
    week = st.selectbox("Week", season_weeks, index=season_weeks.index(week_default),
                        key="olt_week")

slots = olt.week_slots(report, season, week, now)
with st.container(horizontal=True):
    for day in ("wednesday", "sunday"):
        s = slots.get(day, {})
        with st.container(border=True):
            st.markdown(f"**{day.capitalize()} slot**")
            st.caption(s.get("label", "No slot for this week"))

# ---- filters -------------------------------------------------------------
week_rows = olt.display_rows(report, view, season, week, now)
if week_rows.empty:
    st.info("No quotes from this source for the selected week.", icon=":material/info:")
    add_betting_oracle_footer()
    st.stop()

book_options = sorted(week_rows["Sportsbook"].unique())
games = week_rows[["_game_id", "Matchup"]].drop_duplicates().sort_values("Matchup")
with st.container(horizontal=True):
    picked_books = st.multiselect("Sportsbook", book_options, key=f"olt_books_{view}",
                                  placeholder="All sportsbooks")
    game_label = st.selectbox("Game", ["All games"] + games["Matchup"].tolist(),
                              key=f"olt_game_{view}")
    team_options = sorted(week_rows["Team"].unique()) if game_label == "All games" else \
        sorted(week_rows.loc[week_rows["Matchup"] == game_label, "Team"].unique())
    team = st.selectbox("Team", ["All teams"] + team_options, key=f"olt_team_{view}")

book_keys = sorted(week_rows.loc[week_rows["Sportsbook"].isin(picked_books), "_book_key"].unique()) \
    if picked_books else None
game_id = None if game_label == "All games" else \
    games.loc[games["Matchup"] == game_label, "_game_id"].iloc[0]
rows = olt.display_rows(report, view, season, week, now, books=book_keys, game_id=game_id,
                        team=None if team == "All teams" else team)

if rows.empty:
    st.info("No rows match these filters. Clear a sportsbook, game or team filter "
            "to see more.", icon=":material/filter_alt_off:")
    add_betting_oracle_footer()
    st.stop()

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
def _n(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


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
