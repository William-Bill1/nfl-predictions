import streamlit as st
import pandas as pd
from pathlib import Path

from footer import add_betting_oracle_footer
from player_props.reference_join import enrich, enrichment_summary, load_reference, player_key, unique_labels

st.title("🎲 Parlay Builder")

# Load predictions
@st.cache_data(ttl=3600)
def load_player_props_predictions():
    """Load player prop predictions with memory optimization."""
    file_path = Path('data_files/player_props_predictions.csv')
    if not file_path.exists():
        return None

    df = pd.read_csv(file_path, dtype={'player_id': str})

    # Memory optimization
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    # Parse game_date
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])

    # Classify every row's identity against the players reference (by stable
    # ID; see player_props/reference_join.py) and fill only blank display_name
    # / position cells from a permitted match. Stored values are kept.
    df = enrich(df, load_reference_cached())

    # opponent_def_rank was removed (leaky, constant); newer snapshots carry
    # model_reliable instead.
    if 'model_reliable' not in df.columns:
        df['model_reliable'] = True

    # Ensure injury_note column exists (for backward compatibility)
    if 'injury_note' not in df.columns:
        df['injury_note'] = None

    return df

@st.cache_data
def load_reference_cached():
    """The players reference (IDs as text, duplicate IDs flagged), or None."""
    return load_reference(Path('data_files/players.csv'))

def detect_correlations(props_df):
    """
    Detect correlated props that may not be independent.
    """
    warnings = []

    # Check for same-game correlations
    for (team, opponent), game_props in props_df.groupby(['team', 'opponent']):
        if len(game_props) > 1:
            # QB passing yards + team total OVERs = correlated
            qb_pass = game_props[game_props['prop_type'] == 'passing_yards']
            if not qb_pass.empty and qb_pass.iloc[0]['recommendation'] == 'OVER':
                warnings.append({
                    'warning': f"QB {qb_pass.iloc[0]['display_name']} passing OVER may correlate with other OVERs in this game ({team} vs {opponent})"
                })

    # Check for opposing players (zero-sum correlations)
    teams = props_df['team'].unique()
    opponents = props_df['opponent'].unique()
    if len(teams) == 2 and len(opponents) == 2 and set(teams) == set(opponents):
        # Opposing QBs, RBs, etc. in same game
        warnings.append({
            'warning': "Legs contain opposing teams - outcomes may be negatively correlated"
        })

    # Check for same player multiple props (by player ID where present, so two
    # players sharing a name aren't treated as one)
    keys = player_key(props_df)
    player_counts = keys.value_counts()
    for player, count in player_counts.items():
        if count > 1:
            player_props = props_df[keys == player]
            warnings.append({
                'warning': f"Multiple props for {player_props.iloc[0]['display_name']} - outcomes are highly correlated"
            })

    return warnings

def calculate_parlay_metrics(selected_props):
    """Calculate parlay probability and odds."""
    if len(selected_props) == 0:
        return None

    # Calculate combined probability (assuming independence)
    combined_prob = selected_props['confidence'].prod()

    # Calculate decimal odds
    decimal_odds = 1 / combined_prob

    # Convert to American odds
    if decimal_odds >= 2:
        american_odds = (decimal_odds - 1) * 100
        odds_str = f"+{american_odds:.0f}"
    else:
        american_odds = -100 / (decimal_odds - 1)
        odds_str = f"{american_odds:.0f}"

    # Calculate implied win probability
    implied_prob = 1 / decimal_odds

    return {
        'combined_prob': combined_prob,
        'decimal_odds': decimal_odds,
        'american_odds': american_odds,
        'odds_str': odds_str,
        'implied_prob': implied_prob,
        'num_legs': len(selected_props)
    }

# Load predictions
predictions_df = load_player_props_predictions()

# Containment notice (player_props/containment.py), before any prop output.
from player_props.containment import latest_stats_date, notice_markdown, single_season
st.warning(notice_markdown(latest_stats_date(Path('data_files')), single_season(predictions_df)))

if predictions_df is None or predictions_df.empty:
    st.error("❌ No player prop predictions found. Please generate predictions first.")
    st.stop()

_reference_notes = enrichment_summary(predictions_df, load_reference_cached())
if _reference_notes:
    st.warning("Player details: " + "; ".join(_reference_notes) + ".")

# Props with a model output of at least 55% (a filter, not validated confidence)
high_conf_predictions = predictions_df[predictions_df['confidence'] >= 0.55].copy()

if high_conf_predictions.empty:
    st.warning("⚠️ No props with a model output of 55% or more.")
    st.stop()

# Create options for multiselect
def create_prop_option(row):
    """Create a readable option string for each prop."""
    trend_emoji = row.get('trend', '➡️')
    return f"{row['display_name']} ({row['team']}) - {row['prop_type'].replace('_', ' ').title()} {row['recommendation']} {row['line_value']} {trend_emoji} ({row['confidence']:.1%})"

prop_options = []
prop_indices = []

for idx, row in high_conf_predictions.iterrows():
    option = create_prop_option(row)
    prop_options.append(option)
    prop_indices.append(idx)
# Two rows can share a label (e.g. namesakes on one team): make each unique.
prop_options = unique_labels(prop_options, player_key(high_conf_predictions.loc[prop_indices]))

# Multiselect for parlay legs
st.subheader("Build Your Parlay")
st.markdown("Select 2-6 props to see their combined model probability. These outputs are not "
            "validated for betting; see the notice above.")

selected_options = st.multiselect(
    "Select props to combine (2-6 legs recommended)",
    options=prop_options,
    max_selections=8,  # Allow up to 8 for flexibility
    help="Props with a model output of 55% or more. Correlation warnings appear for linked legs."
)

if len(selected_options) >= 2:
    # Get selected rows
    selected_indices = [prop_indices[prop_options.index(option)] for option in selected_options]
    selected_props = high_conf_predictions.loc[selected_indices].copy()

    # Calculate parlay metrics
    metrics = calculate_parlay_metrics(selected_props)

    # Detect correlations
    correlations = detect_correlations(selected_props)

    # Display parlay summary
    st.subheader("📊 Parlay Summary")

    # No "Edge": it compared the combined probability with odds derived from that
    # same probability, so it was always 0% and implied a value that doesn't exist.
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Combined model probability", f"{metrics['combined_prob']:.1%}")
    with col2:
        st.metric("Model-implied odds", metrics['odds_str'])
    with col3:
        st.metric("Legs", metrics['num_legs'])
    st.caption("The combined probability multiplies the legs' model outputs and assumes they are "
               "independent. It is not validated, and the odds are not a sportsbook price.")

    # Show correlation warnings
    if correlations:
        st.warning("⚠️ Correlation Warnings Detected")
        st.markdown("**These combinations may not be truly independent:**")
        for corr in correlations:
            st.write(f"• {corr['warning']}")
        st.info("💡 Correlated legs make the combined probability (which assumes independence) "
                "less meaningful.")

    # Parlay legs table
    st.subheader("🎯 Selected Legs")

    # Prepare display dataframe
    display_df = selected_props[[
        'display_name', 'position', 'team', 'opponent', 'prop_type',
        'recommendation', 'line_value', 'confidence', 'trend',
        'avg_L3', 'avg_L5', 'avg_L10'
    ]].copy()

    # Format columns
    display_df['prop_type'] = display_df['prop_type'].str.replace('_', ' ').str.title()
    display_df['confidence'] = display_df['confidence'].map('{:.1%}'.format)
    display_df['avg_L3'] = display_df['avg_L3'].round(1)
    display_df['avg_L5'] = display_df['avg_L5'].round(1)
    display_df['avg_L10'] = display_df['avg_L10'].round(1)

    st.dataframe(
        display_df,
        column_config={
            'display_name': st.column_config.TextColumn('Player', width='medium'),
            'position': st.column_config.TextColumn('Pos', width='small'),
            'team': st.column_config.TextColumn('Team', width='small'),
            'opponent': st.column_config.TextColumn('Opp', width='small'),
            'prop_type': st.column_config.TextColumn('Prop Type', width='medium'),
            'recommendation': st.column_config.TextColumn('Side', width='small'),
            'line_value': st.column_config.NumberColumn('Line', width='small'),
            'confidence': st.column_config.TextColumn('Model prob', width='small'),
            'trend': st.column_config.TextColumn('Trend', width='small'),
            'avg_L3': st.column_config.NumberColumn('L3 Avg', width='small'),
            'avg_L5': st.column_config.NumberColumn('L5 Avg', width='small'),
            'avg_L10': st.column_config.NumberColumn('L10 Avg', width='small'),
        },
        width='stretch',
        hide_index=True
    )

    # How to read the summary (no betting advice: the outputs are not validated)
    st.subheader("📖 Reading This Summary")
    st.markdown(f"- **Model-implied odds**: {metrics['odds_str']}, the odds that match the combined "
                "model probability. They are not a sportsbook price or a validated edge.")
    st.markdown("- Same-game, opposing-team and same-player legs are correlated, so the "
                "independence assumption overstates or understates the combined probability.")

    # Export option
    if st.button("📄 Export Parlay Summary"):
        # Create summary text
        summary = f"""
NFL Player Props Parlay - {len(selected_props)} Legs
NOT VALIDATED FOR BETTING: player prop outputs have unresolved data and training problems.

Combined model probability (assumes independent legs): {metrics['combined_prob']:.1%}
Model-implied odds: {metrics['odds_str']}

Selected Props:
"""
        for _, row in selected_props.iterrows():
            summary += f"- {row['display_name']} ({row['team']}): {row['prop_type']} {row['recommendation']} {row['line_value']} ({row['confidence']:.1%})\n"

        if correlations:
            summary += "\nCorrelation Warnings:\n"
            for corr in correlations:
                summary += f"- {corr['warning']}\n"

        st.download_button(
            label="📥 Download Summary",
            data=summary,
            file_name="nfl_parlay_summary.txt",
            mime="text/plain"
        )

elif len(selected_options) == 1:
    st.info("💡 Add at least one more prop to create a parlay!")

else:
    st.info("🎯 **How to Use This Page:**")
    st.markdown("""
    1. **Select 2-6 props** to see their combined model probability
    2. **Check the correlation warnings**: legs from the same game or player are linked
    3. **Treat every figure as an unvalidated model output** (see the notice above)
    """)

    # Highest model outputs, for inspection only
    st.subheader("📋 Highest Model Outputs (not validated)")
    top_picks = high_conf_predictions.nlargest(10, 'confidence')

    for _, row in top_picks.iterrows():
        trend_emoji = row.get('trend', '➡️')
        st.write(f"**{row['display_name']}** ({row['team']}) - {row['prop_type'].replace('_', ' ').title()} {row['recommendation']} {row['line_value']} {trend_emoji} ({row['confidence']:.1%})")

# Add footer to the page
add_betting_oracle_footer()