"""
Value Finder

Two price-adjusted views on top of the spread tracker's raw data (see
pages/5_Spread_Tracker.py and docs/MARKET_SPREAD_TRACKER_PLAN.md), both
read-only - this page never calls The Odds API itself:

- Book vs Field: for one sportsbook, is each side priced favorably relative
  to the OTHER tracked books (scripts/spread_value_finder.py)?
- Model vs Books: for the spread model's own current picks, is each
  tracked book's actual line priced favorably relative to what OUR MODEL
  thinks (scripts/model_line_shop.py)? The model never sees individual book
  lines - this extrapolates its nflverse-line probability onto each book's
  specific number and labels the result "extrapolated," not the model's
  literal output.

Kept on its own page (not folded into Spread Tracker) since this is the
actively-iterated, prescriptive half of the feature - "what to bet" rather
than "what happened" - and isolating it means refining it can't destabilize
the already-verified tracker page.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

from footer import add_betting_oracle_footer
import spread_value_finder as svf
import model_line_shop as mls

LOG_PATH = Path('data_files/spread_tracker_log.csv')
PREDICTIONS_PATH = Path('data_files/nfl_games_historical_with_predictions.csv')

st.title("🔎 Value Finder")
st.markdown(
    "Price-adjusted edges on top of the tracked sportsbook spread lines - "
    "point divergence alone isn't the same question as \"is this side "
    "priced favorably,\" since a book can move the points and shade the "
    "price to compensate. See `docs/MARKET_SPREAD_TRACKER_PLAN.md`."
)


@st.cache_data(ttl=3600)
def load_log():
    if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
        return None
    return pd.read_csv(LOG_PATH)


@st.cache_data(ttl=3600)
def load_predictions():
    if not PREDICTIONS_PATH.exists():
        return None
    # Despite the .csv extension this file is TAB-separated - see
    # betting_log.py's own PREDICTIONS_PATH read.
    return pd.read_csv(PREDICTIONS_PATH, sep='\t')


def get_dataframe_height(df, row_height=35, header_height=38, padding=2, max_height=500):
    calculated_height = (len(df) * row_height) + header_height + padding
    return min(calculated_height, max_height) if max_height is not None else calculated_height


log_df = load_log()

if log_df is None or log_df.empty:
    st.info(
        "ℹ️ No spread-tracker data yet. This runs weekly via the "
        "`spread-tracker.yml` GitHub Action once `ODDS_API_KEY` is set - "
        "check back after it's run at least once, or run "
        "`python spread_tracker.py && python scripts/spread_tracker_report.py` "
        "locally."
    )
    add_betting_oracle_footer()
    st.stop()

CAVEAT = (
    "Fair-value uses a standard normal margin-of-victory approximation "
    f"(sigma={svf.MOV_SIGMA}), not a validated model; assumes the comparison "
    "baseline is close to true probability; does NOT account for "
    "parlay-specific pricing (real multi-leg odds usually carry more hold "
    "than multiplying single-leg no-vig prices would suggest); and is only "
    "as good as however many weeks are tracked so far - no track record of "
    "either signal holding up over a season."
)

tab_book, tab_model = st.tabs(["📖 Book vs Field", "🤖 Model vs Books"])

# ---------------------------------------------------------------------------
# Book vs Field
# ---------------------------------------------------------------------------
with tab_book:
    st.markdown(
        "For one sportsbook, is each side priced favorably relative to the "
        "**other tracked books** that same week?"
    )

    week_options = ["All weeks (season to date)"] + sorted(log_df["week"].unique().tolist())
    selected_week = st.selectbox("Week", week_options, index=0, key="book_week")
    week_filter = None if selected_week == "All weeks (season to date)" else selected_week

    book_options = (
        log_df[["book_key", "book_title"]].drop_duplicates().sort_values("book_title")
    )
    book_label_to_key = dict(zip(book_options["book_title"], book_options["book_key"]))
    selected_book_label = st.selectbox("Book", list(book_label_to_key.keys()), key="book_pick")
    selected_book_key = book_label_to_key[selected_book_label]

    edges = svf.compute_book_edges(log_df, selected_book_key, week=week_filter)

    if edges.empty:
        st.info(
            f"No priceable rows for {selected_book_label} in this scope - it may not have "
            f"quoted these games, or fewer than {svf.MIN_FIELD_BOOKS} other books did."
        )
    else:
        edges = edges.sort_values("edge_pts", ascending=False).reset_index(drop=True)
        best = edges.iloc[0]
        sign = "+" if best["line"] >= 0 else ""
        st.success(
            f"🏆 Best edge: **{best['team']} {sign}{best['line']:.1f} ({best['price']:+.0f})** "
            f"vs {best['opponent']} - fair {best['fair_prob']:.1%} / required "
            f"{best['required_prob']:.1%} → **{best['edge_pts']:+.1f}pt**"
        )

        display = edges.copy()
        display["Matchup"] = display["team"] + " vs " + display["opponent"]
        display["Line"] = display["line"].map(lambda v: f"{'+' if v >= 0 else ''}{v:.1f}")
        display["Price"] = display["price"].map(lambda v: f"{v:+.0f}")
        display["Fair %"] = display["fair_prob"].map("{:.1%}".format)
        display["Required %"] = display["required_prob"].map("{:.1%}".format)
        display["Edge (pt)"] = display["edge_pts"].map(lambda v: f"{v:+.1f}")
        display = display.rename(columns={"week": "Week"})[
            ["Week", "Matchup", "Line", "Price", "Fair %", "Required %", "Edge (pt)"]
        ]

        st.dataframe(
            display,
            width='stretch',
            height=get_dataframe_height(display),
            hide_index=True,
        )

    with st.expander("Caveats"):
        st.caption(CAVEAT)

# ---------------------------------------------------------------------------
# Model vs Books
# ---------------------------------------------------------------------------
with tab_model:
    predictions_df = load_predictions()

    if predictions_df is None or "prob_underdogCovered" not in predictions_df.columns:
        st.info("No predictions CSV found - run the prediction pipeline first.")
    else:
        st.markdown(
            "For the spread model's own picks, is each tracked book's actual line priced "
            "favorably relative to **what the model thinks**? The model only ever evaluates "
            "nflverse's own consensus line - the numbers below are the model's probability "
            "*extrapolated* onto each book's specific line, not the model's literal output."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            seasons = sorted(predictions_df["season"].dropna().unique().tolist())
            selected_season = st.selectbox("Season", seasons, index=len(seasons) - 1, key="model_season")
        with col2:
            weeks_avail = sorted(
                predictions_df.loc[predictions_df["season"] == selected_season, "week"].dropna().unique().tolist()
            )
            selected_model_week = st.selectbox("Week", weeks_avail, index=len(weeks_avail) - 1, key="model_week")
        with col3:
            picks_only = st.checkbox("Model picks only", value=True, key="picks_only",
                                      help="Only games where pred_spreadCovered_optimal == 1 "
                                           "(the model's actual +EV signal), not every tracked game.")

        candidates = predictions_df[
            (predictions_df["season"] == selected_season) & (predictions_df["week"] == selected_model_week)
        ]
        if picks_only:
            candidates = candidates[candidates.get("pred_spreadCovered_optimal", 0) == 1]
        game_ids = candidates["game_id"].dropna().unique().tolist()

        if not game_ids:
            note = " with a qualifying model signal" if picks_only else ""
            st.info(f"No games found for Week {selected_model_week}, {selected_season}{note}. "
                    "Try unchecking \"Model picks only\" or a different week.")
        else:
            any_edges = False
            for gid in game_ids:
                game_edges = mls.compute_model_edges_for_game(gid, predictions_df, log_df)
                if game_edges.empty:
                    continue
                any_edges = True
                game_edges = game_edges.sort_values("edge_pts", ascending=False).reset_index(drop=True)
                first = game_edges.iloc[0]
                st.subheader(
                    f"{first['underdog_team']} vs {first['favorite_team']} "
                    f"— model: +{first['nflverse_line']:.1f} → {first['model_prob_at_nflverse_line']:.1%} "
                    "(nflverse's line)"
                )

                display = game_edges.copy()
                display["Line"] = display["book_line"].map(lambda v: f"{'+' if v >= 0 else ''}{v:.1f}")
                display["Price"] = display["book_price"].map(lambda v: f"{v:+.0f}")
                display["Extrapolated %"] = display["extrapolated_prob"].map("{:.1%}".format)
                display["Required %"] = display["required_prob"].map("{:.1%}".format)
                display["Edge (pt)"] = display["edge_pts"].map(lambda v: f"{v:+.1f}")
                display = display.rename(columns={"book_title": "Book", "region": "Region"})[
                    ["Book", "Region", "Line", "Price", "Extrapolated %", "Required %", "Edge (pt)"]
                ]

                st.dataframe(
                    display,
                    width='stretch',
                    height=get_dataframe_height(display, max_height=350),
                    hide_index=True,
                )

            if not any_edges:
                st.info("No tracked-book quotes for these games yet - the spread tracker may not "
                        "have run for this week, or these games are pick'ems.")

    with st.expander("Caveats"):
        st.caption(CAVEAT)
        st.caption(
            "The underlying spread model is only ~break-even out-of-sample "
            "(see the Model Performance page / `model_metrics.json` -> "
            "`Spread_OOS_Test`) - this ranks books given the model's read, "
            "it does not validate that the model's read is correct."
        )

add_betting_oracle_footer()
