"""
Model Performance Dashboard

Displays prediction accuracy tracking, ROI analysis, and model calibration metrics.
Shows historical performance trends and helps validate model improvements.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from footer import add_betting_oracle_footer
from season_utils import completed_weeks, upcoming_or_current_season

try:
    from player_props.backtest import (
        run_weekly_accuracy_check,
        load_accuracy_history,
        load_accuracy_results_for_week,
        save_accuracy_results,
        calculate_hit_rate,
        calculate_roi,
        profitable_subset,
        collect_actual_results,
        week_results_status,
    )
except ImportError:
    st.error("❌ Could not import backtest module. Please ensure player_props/backtest.py exists.")
    st.stop()

# Set page to wide layout for better chart display

st.title("📈 Model Performance Dashboard")

st.markdown("""
Track prediction accuracy, ROI analysis, and model calibration over time.
Validate that our model improvements are actually working!
""")


def get_dataframe_height(df, row_height=35, header_height=38, padding=2, max_height=600):
    """
    Calculate the optimal height for a Streamlit dataframe based on number of rows.
    
    Args:
        df (pd.DataFrame): The dataframe to display
        row_height (int): Height per row in pixels. Default: 35
        header_height (int): Height of header row in pixels. Default: 38
        padding (int): Extra padding in pixels. Default: 2
        max_height (int): Maximum height cap in pixels. Default: 600 (None for no limit)
    
    Returns:
        int: Calculated height in pixels
    
    Example:
        height = get_dataframe_height(my_df)
        st.dataframe(my_df, height=height)
    """
    num_rows = len(df)
    calculated_height = (num_rows * row_height) + header_height + padding
    
    if max_height is not None:
        return min(calculated_height, max_height)
    return calculated_height

# Sidebar controls
st.sidebar.header("📊 Analysis Controls")

# Season selection - current season and two prior seasons
current_season = upcoming_or_current_season()
available_seasons = [current_season - 2, current_season - 1, current_season]
season_labels = {season: f"{season} Season" for season in available_seasons}

SCHEDULE_PATH = Path(__file__).resolve().parent.parent / "data_files" / "nfl_games_historical.csv"


@st.cache_data(show_spinner=False)
def load_schedule_results(path: str, mtime: float) -> pd.DataFrame:
    """Season/week/scores from the nightly schedule. ``path`` and ``mtime`` key
    the cache, so a refreshed file is re-read."""
    return pd.read_csv(path, sep="\t",
                       usecols=["season", "game_type", "week", "home_score", "away_score"])


try:
    schedule_results = load_schedule_results(str(SCHEDULE_PATH), SCHEDULE_PATH.stat().st_mtime)
except (OSError, ValueError) as e:
    schedule_results = pd.DataFrame(columns=["season", "game_type", "week", "home_score", "away_score"])
    st.sidebar.warning(f"⚠️ Could not read the schedule to find completed weeks: {e}")

weeks_by_season = {season: completed_weeks(schedule_results, season) for season in available_seasons}

# Default to the newest season that has completed games (the current season
# before its Week 1 has none), falling back to the current season.
seasons_with_results = [i for i, s in enumerate(available_seasons) if weeks_by_season[s][0]]
selected_season_idx = st.sidebar.selectbox(
    "Select Season to Analyze",
    options=range(len(available_seasons)),
    format_func=lambda i: season_labels[available_seasons[i]],
    index=seasons_with_results[-1] if seasons_with_results else len(available_seasons) - 1,
    help="Choose which NFL season to analyze. Only weeks with completed games are listed."
)
selected_season = available_seasons[selected_season_idx]

# Week selection - only regular-season weeks with completed games, defaulting
# to the latest fully completed week (never an unplayed week).
available_weeks, default_week = weeks_by_season[selected_season]
if available_weeks:
    selected_week = st.sidebar.selectbox(
        "Select Week to Analyze",
        options=available_weeks,
        index=available_weeks.index(default_week),
        help=f"Weeks of the {selected_season} season with completed games. Defaults to the latest week whose games have all finished."
    )
else:
    selected_week = None
    st.sidebar.info(f"ℹ️ No {selected_season} regular-season games have been completed yet, so there are no weeks to analyze. Choose an earlier season.")

# Analysis type
analysis_type = st.sidebar.radio(
    "Analysis Type",
    ["Current Week", "Historical Trends", "ROI Analysis"],
    help="Choose what type of analysis to display"
)

# Auto-run analysis button
if selected_week is not None and st.sidebar.button("🔄 Run Fresh Analysis", help="Re-run accuracy analysis for selected week and update cached results"):
    with st.spinner("Running fresh accuracy analysis..."):
        # First check if we can collect actual results
        test_df, error_msg = collect_actual_results(selected_week, selected_season)

        if test_df.empty and error_msg:
            st.sidebar.error(f"❌ **Data Collection Failed**: {error_msg}")
            st.sidebar.info("💡 **Troubleshooting Tips:**\n"
                           "- Check your internet connection\n"
                           "- The NFL data API may be temporarily unavailable\n"
                           "- Historical data may not be available for recent seasons\n"
                           "- Try selecting an earlier week with completed games")
        else:
            accuracy_results = run_weekly_accuracy_check(selected_week, selected_season)
            if accuracy_results and accuracy_results.get('final'):
                st.sidebar.success(f"✅ Fresh analysis complete for Week {selected_week} (results cached)")
                st.rerun()
            elif accuracy_results:
                st.sidebar.warning(f"⚠️ Week {selected_week} results are provisional and were not cached: "
                                   f"{accuracy_results.get('provisional_reason')}")
            else:
                st.sidebar.error("❌ Analysis failed - check data availability")


def display_current_week_analysis(week: int, season: int):
    """Display accuracy analysis for a specific week."""
    st.header(f"🎯 Week {week} Accuracy Analysis")

    # First check for cached results
    cached_results = load_accuracy_results_for_week(week, season)
    
    if cached_results:
        st.info(f"📋 **Using cached results** from previous analysis. Click '🔄 Run Fresh Analysis' in sidebar to recalculate.")
        accuracy_metrics = cached_results
    else:
        st.info("🔄 **Calculating fresh analysis** for this week...")
        
        # Load predictions and actual results
        predictions_file = f"data_files/player_props_predictions_week{week}.csv"
        if not Path(predictions_file).exists():
            predictions_file = "data_files/player_props_predictions.csv"

        if not Path(predictions_file).exists():
            st.error(f"❌ No predictions file found for Week {week}")
            return

        predictions_df = pd.read_csv(predictions_file)

        # Filter to high-confidence predictions
        high_conf_predictions = predictions_df[predictions_df['confidence'] >= 0.55]

        if high_conf_predictions.empty:
            st.warning(f"⚠️ No high-confidence predictions (≥55%) found for Week {week}")
            return

        # Collect actual results
        actuals_df, error_msg = collect_actual_results(week, season)

        if actuals_df.empty:
            if error_msg:
                st.error(f"❌ **Data Collection Failed**: {error_msg}")
                st.info("💡 **Troubleshooting Tips:**\n"
                       "- Check your internet connection\n"
                       "- The NFL data API may be temporarily unavailable\n"
                       "- Historical data may not be available for recent seasons\n"
                       "- Try selecting an earlier week with completed games")
            else:
                st.info(f"ℹ️ Actual results for Week {week} are not yet available. Games may still be in progress.")
            st.markdown("**Preview Analysis** (based on available data)")

            # Show prediction distribution
            fig = px.histogram(
                high_conf_predictions,
                x='confidence',
                nbins=20,
                title=f"Week {week} Prediction Confidence Distribution",
                labels={'confidence': 'Model Confidence', 'count': 'Number of Predictions'}
            )
            st.plotly_chart(fig, width='stretch')

            return

        # Calculate accuracy metrics
        accuracy_metrics = calculate_hit_rate(high_conf_predictions, actuals_df)

        if accuracy_metrics['total_predictions'] == 0:
            st.warning("⚠️ No matching predictions found with actual results")
            return

        # Cache only final results. Provisional ones (a game not final, or
        # play-by-play still missing a game) are shown but not saved, so they
        # can't be served later as the week's result.
        final, reason = week_results_status(actuals_df, week, season)
        if final:
            save_accuracy_results(accuracy_metrics, week, season=season)
        else:
            st.warning(f"⚠️ **Provisional results, not cached:** {reason}. "
                       "Figures may change once all of the week's data is available.")

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Overall Accuracy",
            f"{accuracy_metrics['overall_accuracy']:.1%}",
            help="Percentage of predictions that were correct"
        )

    with col2:
        st.metric(
            "Total Predictions",
            accuracy_metrics['total_predictions'],
            help="Number of predictions evaluated"
        )

    with col3:
        # Calculate average confidence
        avg_conf = accuracy_metrics['detailed_results']['confidence'].mean()
        st.metric(
            "Avg Confidence",
            f"{avg_conf:.1%}",
            help="Average model confidence for evaluated predictions"
        )

    with col4:
        # Calculate ROI
        roi_metrics = calculate_roi(accuracy_metrics['detailed_results'])
        st.metric(
            "Hypothetical ROI",
            f"{roi_metrics['roi']:.1f}%",
            delta=f"{roi_metrics['roi']:.1f}%" if roi_metrics['roi'] != 0 else None,
            help=f"ROI at -110 odds. Breakeven: {roi_metrics['breakeven_rate']:.1f}%"
        )

    # Accuracy by confidence tier
    st.subheader("🎯 Accuracy by Confidence Tier")

    if not accuracy_metrics['by_confidence_tier'].empty:
        # Create bar chart
        conf_df = accuracy_metrics['by_confidence_tier'].reset_index()
        conf_df.columns = ['Confidence Tier', 'Accuracy']

        fig = px.bar(
            conf_df,
            x='Confidence Tier',
            y='Accuracy',
            title="Prediction Accuracy by Confidence Level",
            labels={'Accuracy': 'Hit Rate'},
            color='Accuracy',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig, width='stretch')

        # Show as table too
        st.dataframe(
            conf_df.style.format({'Accuracy': '{:.1%}'}),
            width=400,
            height=get_dataframe_height(conf_df),
            hide_index=True
        )
    else:
        st.info("Not enough data for confidence tier analysis")

    # Accuracy by prop type
    st.subheader("🏈 Accuracy by Prop Type")

    if not accuracy_metrics['by_prop_type'].empty:
        prop_df = accuracy_metrics['by_prop_type'].reset_index()
        prop_df.columns = ['Prop Type', 'Accuracy']

        # Sort by accuracy
        prop_df = prop_df.sort_values('Accuracy', ascending=False)

        fig = px.bar(
            prop_df,
            x='Prop Type',
            y='Accuracy',
            title="Prediction Accuracy by Prop Type",
            labels={'Accuracy': 'Hit Rate'},
            color='Accuracy',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Not enough data for prop type analysis")

    # Detailed results table
    st.subheader("📋 Detailed Results")

    detailed_df = accuracy_metrics['detailed_results'][[
        'display_name', 'prop_type', 'line_value', 'predicted', 'actual', 'actual_value', 'hit', 'confidence'
    ]].copy()

    # Format columns
    detailed_df['prop_type'] = detailed_df['prop_type'].str.replace('_', ' ').str.title()
    detailed_df['confidence'] = detailed_df['confidence'].map('{:.1%}'.format)
    detailed_df['hit'] = detailed_df['hit'].map({True: '✅', False: '❌'})

    height = get_dataframe_height(detailed_df)

    st.dataframe(
        detailed_df,
        column_config={
            'display_name': st.column_config.TextColumn('Player', width='medium'),
            'prop_type': st.column_config.TextColumn('Prop Type', width='medium'),
            'line_value': st.column_config.NumberColumn('Line', width='small'),
            'predicted': st.column_config.TextColumn('Predicted', width='small'),
            'actual': st.column_config.TextColumn('Actual', width='small'),
            'actual_value': st.column_config.NumberColumn('Actual Value', width='small'),
            'hit': st.column_config.TextColumn('Hit', width='small'),
            'confidence': st.column_config.TextColumn('Confidence', width='small'),
        },
        width=1000,
        height=height,
        hide_index=True
    )


def display_historical_trends():
    """Display historical accuracy trends across multiple weeks."""
    st.header("📈 Historical Performance Trends")

    # Load historical accuracy data
    history_df = load_accuracy_history()

    if history_df.empty:
        st.info("ℹ️ No historical accuracy data found. Run some weekly analyses first!")
        st.markdown("""
        **To build historical data:**
        1. Select "Current Week" analysis
        2. Choose different weeks
        3. Click "🔄 Run Fresh Analysis" for each week
        4. Return here to see trends
        """)
        return

    # Display summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        avg_accuracy = history_df['overall_accuracy'].mean()
        st.metric("Average Accuracy", f"{avg_accuracy:.1%}")

    with col2:
        total_predictions = history_df['total_predictions'].sum()
        st.metric("Total Predictions", f"{total_predictions:,}")

    with col3:
        weeks_analyzed = len(history_df)
        st.metric("Weeks Analyzed", weeks_analyzed)

    # Accuracy trend over time
    st.subheader("📊 Accuracy Trend Over Time")

    fig = px.line(
        history_df,
        x='week',
        y='overall_accuracy',
        title="Weekly Prediction Accuracy Trend",
        labels={'week': 'NFL Week', 'overall_accuracy': 'Accuracy'},
        markers=True
    )
    fig.update_layout(yaxis_tickformat='.1%')
    fig.update_xaxes(tickmode='linear')
    st.plotly_chart(fig, width='stretch')

    # Volume trend
    st.subheader("📈 Prediction Volume Trend")

    fig = px.bar(
        history_df,
        x='week',
        y='total_predictions',
        title="Weekly Prediction Volume",
        labels={'week': 'NFL Week', 'total_predictions': 'Predictions'}
    )
    fig.update_xaxes(tickmode='linear')
    st.plotly_chart(fig, width='stretch')

    # Historical data table
    st.subheader("📋 Historical Results")

    display_df = history_df.copy()
    display_df['overall_accuracy'] = display_df['overall_accuracy'].map('{:.1%}'.format)
    display_df = display_df.sort_values('week', ascending=False)

    st.dataframe(
        display_df,
        column_config={
            'week': st.column_config.NumberColumn('Week', width='small'),
            'overall_accuracy': st.column_config.TextColumn('Accuracy', width='small'),
            'total_predictions': st.column_config.NumberColumn('Predictions', width='small'),
            'timestamp': st.column_config.TextColumn('Analysis Date', width='medium', help='When the accuracy analysis was run'),
        },
        width=600,
        hide_index=True
    )


def display_roi_analysis(week: int, season: int):
    """Display ROI analysis for different confidence thresholds."""
    st.header("💰 ROI Analysis")

    # Load predictions and actual results
    predictions_file = f"data_files/player_props_predictions_week{week}.csv"
    if not Path(predictions_file).exists():
        predictions_file = "data_files/player_props_predictions.csv"

    if not Path(predictions_file).exists():
        st.error(f"❌ No predictions file found for Week {week}")
        return

    predictions_df = pd.read_csv(predictions_file)
    actuals_df, error_msg = collect_actual_results(week, season)

    if actuals_df.empty:
        if error_msg:
            st.error(f"❌ **Data Collection Failed**: {error_msg}")
            st.info("💡 **Troubleshooting Tips:**\n"
                   "- Check your internet connection\n"
                   "- The NFL data API may be temporarily unavailable\n"
                   "- Historical data may not be available for recent seasons\n"
                   "- Try selecting an earlier week with completed games")
        else:
            st.info(f"ℹ️ Actual results for Week {week} are not yet available for ROI analysis.")
        return

    # Calculate accuracy metrics
    accuracy_metrics = calculate_hit_rate(predictions_df, actuals_df)

    if accuracy_metrics['total_predictions'] == 0:
        st.warning("⚠️ No matching predictions found with actual results")
        return

    final, reason = week_results_status(actuals_df, week, season)
    if not final:
        st.warning(f"⚠️ **Provisional results:** {reason}. "
                   "Figures may change once all of the week's data is available.")

    # ROI analysis for different confidence thresholds
    roi_table = profitable_subset(accuracy_metrics['detailed_results'])

    if roi_table.empty:
        st.warning("⚠️ Not enough data for ROI analysis")
        return

    st.subheader("💵 ROI by Confidence Threshold")

    # Format the ROI table for display
    display_roi = roi_table.copy()
    display_roi['hit_rate'] = display_roi['hit_rate'].map('{:.1%}'.format)
    display_roi['roi'] = display_roi['roi'].map('{:.1f}%'.format)
    display_roi['breakeven_rate'] = display_roi['breakeven_rate'].map('{:.1f}%'.format)

    st.dataframe(
        display_roi[['total_bets', 'hit_rate', 'roi', 'breakeven_rate']],
        column_config={
            'total_bets': st.column_config.NumberColumn('Bets', width='small'),
            'hit_rate': st.column_config.TextColumn('Hit Rate', width='small'),
            'roi': st.column_config.TextColumn('ROI', width='small'),
            'breakeven_rate': st.column_config.TextColumn('Breakeven', width='small'),
        },
        width='stretch',
    )

    # Find best performing threshold
    if not roi_table.empty:
        best_threshold = roi_table['roi'].idxmax()
        best_roi = roi_table.loc[best_threshold, 'roi']

        st.success(f"🎯 **Best Strategy**: Bet on {best_threshold:.0%}+ confidence props (ROI: {best_roi:.1f}%)")

    # ROI vs Confidence Threshold Chart
    st.subheader("📊 ROI vs Confidence Threshold")

    fig = px.line(
        roi_table.reset_index(),
        x='index',
        y='roi',
        title="ROI by Minimum Confidence Threshold",
        labels={'index': 'Minimum Confidence', 'roi': 'ROI (%)'},
        markers=True
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Breakeven")
    fig.update_layout(yaxis_tickformat='.1f')
    st.plotly_chart(fig, width='stretch')

    # Hit Rate vs Confidence Threshold
    st.subheader("🎯 Hit Rate vs Confidence Threshold")

    fig = px.line(
        roi_table.reset_index(),
        x='index',
        y='hit_rate',
        title="Hit Rate by Minimum Confidence Threshold",
        labels={'index': 'Minimum Confidence', 'hit_rate': 'Hit Rate'},
        markers=True
    )
    fig.update_layout(yaxis_tickformat='.1%')
    st.plotly_chart(fig, width='stretch')

    # Betting strategy recommendations
    st.subheader("🎲 Betting Strategy Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**✅ Profitable Thresholds:**")
        profitable = roi_table[roi_table['roi'] > 0]
        if not profitable.empty:
            for threshold, row in profitable.iterrows():
                st.write(f"• {threshold:.0%}+ confidence: {row['roi']:.1f}% ROI")
        else:
            st.write("• None found - model needs improvement")

    with col2:
        st.markdown("**📈 Improvement Opportunities:**")
        st.write("• Focus on prop types with low accuracy")
        st.write("• Investigate high-confidence misses")
        st.write("• Consider adjusting confidence thresholds")


# Main content based on analysis type
if analysis_type == "Historical Trends":
    display_historical_trends()

elif selected_week is None:
    st.info(f"ℹ️ The {selected_season} season has no completed regular-season games yet. "
            "Choose an earlier season, or use Historical Trends.")

elif analysis_type == "Current Week":
    display_current_week_analysis(selected_week, selected_season)

elif analysis_type == "ROI Analysis":
    display_roi_analysis(selected_week, selected_season)


# Add footer to the page
add_betting_oracle_footer()