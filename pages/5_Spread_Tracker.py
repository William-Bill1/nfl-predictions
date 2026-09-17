"""
Spread Line Tracker

Season-long comparison of real US + Canadian sportsbook game-spread lines
against nflverse's own spread_line (the number the spread model is trained
and evaluated against) - opt-in, populated weekly by spread_tracker.py +
scripts/spread_tracker_report.py (see docs/MARKET_SPREAD_TRACKER_PLAN.md).
Read-only: this page only displays the already-generated report/log, it
never calls The Odds API itself.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from footer import add_betting_oracle_footer

REPORT_PATH = Path('data_files/spread_tracker_report.json')
LOG_PATH = Path('data_files/spread_tracker_log.csv')

st.title("📊 Spread Line Tracker")
st.markdown(
    "Season-long comparison of real US + Canadian sportsbook game-spread "
    "lines against nflverse's own line - answering \"which sportsbook(s) "
    "consistently offer a different number than nflverse?\" from tracked "
    "data instead of one-off manual checks. Opt-in feature "
    "(`ODDS_API_KEY`); see `docs/MARKET_SPREAD_TRACKER_PLAN.md`."
)


@st.cache_data(ttl=3600)
def load_report():
    if not REPORT_PATH.exists():
        return None
    with open(REPORT_PATH) as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def load_log():
    if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
        return None
    return pd.read_csv(LOG_PATH)


def get_dataframe_height(df, row_height=35, header_height=38, padding=2, max_height=500):
    calculated_height = (len(df) * row_height) + header_height + padding
    return min(calculated_height, max_height) if max_height is not None else calculated_height


report = load_report()

if not report or not report.get("overall", {}).get("per_book"):
    st.info(
        "ℹ️ No spread-tracker data yet. This runs weekly via the "
        "`spread-tracker.yml` GitHub Action once `ODDS_API_KEY` is set - "
        "check back after it's run at least once, or run "
        "`python spread_tracker.py && python scripts/spread_tracker_report.py` "
        "locally."
    )
    add_betting_oracle_footer()
    st.stop()

overall = report["overall"]
st.caption(
    f"Season to date: {overall['n_weeks']} week(s), {overall['n_games']} games, "
    f"{overall['n_quotes']} book/game quotes tracked. Report generated "
    f"{report.get('generated_at', 'unknown')}."
)

# --- Week filter ------------------------------------------------------------
ALL_WEEKS = "All weeks (season to date)"
week_options = [ALL_WEEKS] + [w["week"] for w in report["by_week"]]
selected_week = st.sidebar.selectbox("Week", week_options, index=0)

if selected_week == ALL_WEEKS:
    per_book = overall["per_book"]
    scope_label = "season to date"
else:
    wk_entry = next(w for w in report["by_week"] if w["week"] == selected_week)
    per_book = wk_entry["per_book"]
    scope_label = f"Week {selected_week}"

# --- Per-book ranking ---------------------------------------------------
st.subheader(f"Which book is closest to nflverse's line? ({scope_label})")

book_rows = [
    {
        "Book": stats["book_title"],
        "Region": stats["region"].upper(),
        "Quotes": stats["n_with_comparison"],
        "Mean Deviation (pt)": stats["mean_deviation_pts"],
        "Mean |Deviation| (pt)": stats["mean_abs_deviation_pts"],
    }
    for stats in per_book.values()
    if stats["mean_abs_deviation_pts"] is not None
]

if book_rows:
    book_df = pd.DataFrame(book_rows).sort_values("Mean |Deviation| (pt)").reset_index(drop=True)
    st.dataframe(
        book_df,
        column_config={
            "Mean Deviation (pt)": st.column_config.NumberColumn(format="%+.2f"),
            "Mean |Deviation| (pt)": st.column_config.NumberColumn(format="%.2f"),
        },
        width='stretch',
        height=get_dataframe_height(book_df),
        hide_index=True,
    )

    closest = book_df.iloc[0]
    st.success(
        f"🏆 **{closest['Book']}** ({closest['Region']}) tracks nflverse's line most "
        f"closely {scope_label} - mean |deviation| of {closest['Mean |Deviation| (pt)']:.2f} "
        f"points across {closest['Quotes']} quotes."
    )

    try:
        import plotly.express as px
        fig = px.bar(
            book_df.sort_values("Mean |Deviation| (pt)", ascending=True),
            x="Mean |Deviation| (pt)", y="Book", color="Region", orientation="h",
            title=f"Mean absolute deviation from nflverse's line ({scope_label})",
        )
        fig.update_layout(yaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig, width='stretch')
    except ImportError:
        pass
else:
    st.info("No books had a matched nflverse line for this scope yet.")

# --- Best line per game ------------------------------------------------
st.subheader("Best available line per game")

best_lines = report.get("best_line_per_game", [])
if selected_week != ALL_WEEKS:
    best_lines = [r for r in best_lines if r["week"] == selected_week]

if best_lines:
    bl_df = pd.DataFrame(best_lines)
    bl_df["Matchup"] = bl_df["away_team"] + " @ " + bl_df["home_team"]
    display_bl = bl_df[[
        "week", "Matchup", "best_home_book_title", "best_home_point",
        "best_away_book_title", "best_away_point",
    ]].rename(columns={
        "week": "Week", "best_home_book_title": "Best Home Book", "best_home_point": "Home Points",
        "best_away_book_title": "Best Away Book", "best_away_point": "Away Points",
    })
    st.dataframe(
        display_bl,
        column_config={
            "Home Points": st.column_config.NumberColumn(format="%+.1f"),
            "Away Points": st.column_config.NumberColumn(format="%+.1f"),
        },
        width='stretch',
        height=get_dataframe_height(display_bl),
        hide_index=True,
    )
    st.caption(
        "\"Home Points\"/\"Away Points\" are the number of points that side's "
        "bettor gets from the best available book (favorite-negative, "
        "underdog-positive - the raw sportsbook convention, not nflverse's)."
    )
else:
    st.info("No games tracked for this scope yet.")

# --- Anomalies ------------------------------------------------------------
st.subheader("Anomalies (lines that diverge from the field)")

anomalies = report.get("anomalies", [])
if selected_week != ALL_WEEKS:
    anomalies = [a for a in anomalies if a["week"] == selected_week]

threshold = report.get("anomaly_threshold_pts", 1.5)

if anomalies:
    an_df = pd.DataFrame(anomalies)
    display_an = an_df[[
        "week", "game_id", "book_title", "region", "book_home_spread",
        "field_median_home_spread", "diff_from_field_median",
    ]].rename(columns={
        "week": "Week", "game_id": "Game", "book_title": "Book", "region": "Region",
        "book_home_spread": "Book's Home Spread", "field_median_home_spread": "Field Median",
        "diff_from_field_median": "Diff (pt)",
    })
    st.dataframe(
        display_an,
        column_config={
            "Book's Home Spread": st.column_config.NumberColumn(format="%+.1f"),
            "Field Median": st.column_config.NumberColumn(format="%+.1f"),
            "Diff (pt)": st.column_config.NumberColumn(format="%+.1f"),
        },
        width='stretch',
        height=get_dataframe_height(display_an),
        hide_index=True,
    )
    st.caption(
        f"Flagged when a book's normalized home spread is more than {threshold}pt "
        "off the field's median for that game/week - a generalized version of the "
        "PlayNow (Canadian provincial book) divergence first spotted by hand."
    )
else:
    st.info(f"No anomalies flagged for {scope_label} - every book stayed within "
             f"{threshold}pt of the field's median.")

# --- Raw data -------------------------------------------------------------
with st.expander("Raw tracked lines"):
    log_df = load_log()
    if log_df is not None:
        if selected_week != ALL_WEEKS:
            log_df = log_df[log_df["week"] == selected_week]
        st.dataframe(
            log_df, width='stretch',
            height=get_dataframe_height(log_df, max_height=500), hide_index=True,
        )
    else:
        st.info("No raw log found.")

add_betting_oracle_footer()
