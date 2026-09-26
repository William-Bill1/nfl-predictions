"""
NFL Injury Data Fetcher
Pulls injury reports from ESPN's public JSON API for player prop adjustments.
"""
import re

import pandas as pd
import requests
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(__file__).parent.parent / 'data_files'
INJURIES_FILE = DATA_DIR / 'espn_injuries.csv'

# ESPN's public JSON injuries feed. The HTML page (espn.com/nfl/injuries) this
# module used to scrape stopped yielding parseable tables in 2026 - the scrape
# silently returned zero rows - so this reads the structured feed instead.
ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"

# No spoofed browser User-Agent: the old hard-coded Chrome 91 UA gets a 403
# from this endpoint, while requests' default UA is accepted.
HEADERS = {'Accept': 'application/json'}

INJURY_COLUMNS = ['player_name', 'position', 'injury_type', 'status',
                  'practice_participation', 'team', 'source']

# ============================================================================
# FETCH FUNCTIONS
# ============================================================================

def _parse_espn_injuries(payload):
    """ESPN injuries JSON -> one row per listed player (INJURY_COLUMNS).

    The feed has no practice-participation field, so that column is left
    blank; adjust_prediction_for_injury keys off `status` first anyway.
    """
    rows = []
    for team in (payload or {}).get('injuries', []) or []:
        team_name = team.get('displayName', 'Unknown')
        for item in team.get('injuries', []) or []:
            athlete = item.get('athlete') or {}
            player_name = (athlete.get('displayName') or '').strip()
            if not player_name:
                continue
            rows.append({
                'player_name': player_name,
                'position': (athlete.get('position') or {}).get('abbreviation', ''),
                'injury_type': (item.get('details') or {}).get('type', ''),
                'status': item.get('status', ''),
                'practice_participation': '',
                'team': team_name,
                'source': 'ESPN',
            })
    return pd.DataFrame(rows, columns=INJURY_COLUMNS)


def scrape_espn_injuries():
    """
    Fetch the current NFL injury report from ESPN's JSON feed.

    Returns:
        pd.DataFrame: DataFrame with injury data (empty on any failure)
    """
    print("🔍 Fetching ESPN injury report...")

    try:
        response = requests.get(ESPN_INJURIES_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        df = _parse_espn_injuries(response.json())

        if df.empty:
            print("⚠️  No injuries found - ESPN may have changed the feed format")
            return pd.DataFrame()

        print(f"✅ Fetched {len(df)} injuries from {df['team'].nunique()} teams")
        return df

    except requests.RequestException as e:
        print(f"❌ Request error: {e}")
        return pd.DataFrame()
    except ValueError as e:
        print(f"❌ Could not parse ESPN injuries JSON: {e}")
        return pd.DataFrame()


def clean_injury_data(df):
    """
    Clean and standardize injury data

    Args:
        df (pd.DataFrame): Raw injury data

    Returns:
        pd.DataFrame: Cleaned injury data
    """
    if df.empty:
        return df

    # Standardize status values
    status_mapping = {
        'questionable': 'Questionable',
        'probable': 'Probable',
        'doubtful': 'Doubtful',
        'out': 'Out',
        'injured reserve': 'IR',
        'pup': 'PUP',
        'nf-inj': 'NFI',
        'suspended': 'Suspended'
    }

    df['status'] = df['status'].str.lower().map(status_mapping).fillna(df['status'])

    # Standardize practice participation
    practice_mapping = {
        'full': 'Full',
        'limited': 'Limited',
        'dnp': 'DNP',
        'did not participate': 'DNP',
        'out': 'Out'
    }

    df['practice_participation'] = df['practice_participation'].str.lower().map(practice_mapping).fillna(df['practice_participation'])

    # Clean player names (remove extra spaces, standardize format)
    df['player_name'] = df['player_name'].str.strip()

    # Add timestamp
    df['scraped_at'] = pd.Timestamp.now()

    return df


def save_injuries_to_csv(df, filename=None):
    """
    Save injury data to CSV

    Args:
        df (pd.DataFrame): Injury data
        filename (str): Output filename (optional)
    """
    if filename is None:
        filename = INJURIES_FILE

    try:
        df.to_csv(filename, index=False)
        print(f"💾 Saved {len(df)} injuries to {filename}")
    except Exception as e:
        print(f"❌ Error saving to CSV: {e}")


def load_cached_injuries(max_age_hours=6):
    """
    Load cached injury data if it's recent enough

    Args:
        max_age_hours (int): Maximum age of cached data in hours

    Returns:
        pd.DataFrame or None: Cached injury data, or None if too old/no cache
    """
    if not INJURIES_FILE.exists():
        return None

    try:
        df = pd.read_csv(INJURIES_FILE)
        df['scraped_at'] = pd.to_datetime(df['scraped_at'])

        # Check if data is recent enough
        age_hours = (pd.Timestamp.now() - df['scraped_at'].max()).total_seconds() / 3600

        if age_hours <= max_age_hours:
            print(f"📋 Using cached injuries ({age_hours:.1f} hours old)")
            return df
        else:
            print(f"📋 Cached injuries too old ({age_hours:.1f} hours) - refreshing")
            return None

    except Exception as e:
        print(f"⚠️  Error loading cached injuries: {e}")
        return None


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def get_injury_report(use_cache=True, max_cache_age_hours=6):
    """
    Get current NFL injury report

    Args:
        use_cache (bool): Whether to use cached data if available
        max_cache_age_hours (int): Maximum age of cached data

    Returns:
        pd.DataFrame: Current injury data
    """
    # Try to load cached data first
    if use_cache:
        cached_data = load_cached_injuries(max_cache_age_hours)
        if cached_data is not None:
            return cached_data

    # Scrape fresh data
    raw_data = scrape_espn_injuries()

    if not raw_data.empty:
        # Clean the data
        clean_data = clean_injury_data(raw_data)

        # Save to cache
        save_injuries_to_csv(clean_data)

        return clean_data
    else:
        # Return cached data as fallback if scraping fails
        print("⚠️  Scraping failed, trying cached data as fallback...")
        cached_fallback = load_cached_injuries(max_age_hours=24)  # Allow older data as fallback
        if cached_fallback is not None:
            return cached_fallback

        return pd.DataFrame()


def _normalize_name(name):
    """Lowercase, drop generational suffixes and punctuation, collapse spaces -
    so "James Cook III" and "James Cook" (or "D'Andre Swift" / "DAndre Swift")
    compare equal."""
    name = re.sub(r"\b(Jr|Sr|II|III|IV)\.?(?=\s|$)", "", str(name), flags=re.IGNORECASE)
    name = re.sub(r"[^a-z0-9 ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


def find_player_injury(player_name, injuries_df):
    """
    Find injury information for a specific player

    Matches on the full name only - exact, then suffix/punctuation-normalized.
    There is deliberately no last-name fallback: against a league-wide feed a
    substring match on the last name hit the wrong player for 58 of 317
    prop players (e.g. "Tahj Brooks" -> "Jonathon Brooks" (IR), which would
    delete a healthy player's prediction; "Rasheen Ali" -> "Khalil Shakir",
    since "ali" is inside "khalil").

    Args:
        player_name (str): Player name to search for
        injuries_df (pd.DataFrame): Injury data

    Returns:
        dict or None: Injury info for the player
    """
    if injuries_df.empty or not isinstance(player_name, str) or not player_name:
        return None

    exact_match = injuries_df[injuries_df['player_name'] == player_name]
    if not exact_match.empty:
        return exact_match.iloc[0].to_dict()

    target = _normalize_name(player_name)
    if not target:
        return None
    normalized = injuries_df['player_name'].map(_normalize_name)
    matches = injuries_df[normalized == target]
    if not matches.empty:
        return matches.iloc[0].to_dict()

    return None


# ============================================================================
# PREDICTION ADJUSTMENT FUNCTIONS
# ============================================================================

def adjust_prediction_for_injury(prediction, injury_info):
    """
    Adjust a player prop prediction based on injury status

    Args:
        prediction (dict): Prediction dictionary
        injury_info (dict): Injury information

    Returns:
        dict or None: Adjusted prediction, or None if player is out
    """
    if not injury_info:
        return prediction

    # Ensure status and practice are strings
    status = str(injury_info.get('status', '')).lower() if injury_info.get('status') is not None else ''
    practice = str(injury_info.get('practice_participation', '')).lower() if injury_info.get('practice_participation') is not None else ''

    # Player is out - remove prediction entirely
    if status in ['out', 'ir', 'pup', 'nfi', 'suspended'] or practice == 'out':
        return None

    # Questionable players - reduce confidence
    if status == 'questionable':
        reduction_factor = 0.80  # Reduce confidence by 20%
        prediction['confidence'] *= reduction_factor
        prediction['prob_over'] *= reduction_factor
        prediction['prob_under'] = 1 - prediction['prob_over']

        # Add injury note
        injury_type = injury_info.get('injury_type', 'Unknown')
        prediction['injury_note'] = f"⚠️ {injury_type} injury (Questionable)"

    # Doubtful players - reduce confidence more
    elif status == 'doubtful':
        reduction_factor = 0.70  # Reduce confidence by 30%
        prediction['confidence'] *= reduction_factor
        prediction['prob_over'] *= reduction_factor
        prediction['prob_under'] = 1 - prediction['prob_over']

        injury_type = injury_info.get('injury_type', 'Unknown')
        prediction['injury_note'] = f"❌ {injury_type} injury (Doubtful)"

    # Limited practice - slight reduction
    elif practice == 'limited':
        reduction_factor = 0.90  # Reduce confidence by 10%
        prediction['confidence'] *= reduction_factor
        prediction['prob_over'] *= reduction_factor
        prediction['prob_under'] = 1 - prediction['prob_over']

        prediction['injury_note'] = "🟡 Limited practice participation"

    return prediction


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_injury_summary(injuries_df):
    """
    Print summary of injury data

    Args:
        injuries_df (pd.DataFrame): Injury data
    """
    if injuries_df.empty:
        print("📋 No injury data available")
        return

    print("📋 Injury Report Summary:")
    print(f"   Total injuries: {len(injuries_df)}")
    print(f"   Teams affected: {injuries_df['team'].nunique()}")
    print(f"   Status breakdown: {injuries_df['status'].value_counts().to_dict()}")
    print(f"   Practice participation: {injuries_df['practice_participation'].value_counts().to_dict()}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🏥 NFL Injury Report Scraper")
    print("=" * 60)

    # Get injury data
    injuries = get_injury_report()

    if not injuries.empty:
        print_injury_summary(injuries)

        # Show sample injuries
        print("\n📋 Sample Injuries:")
        sample = injuries.head(10)[['player_name', 'team', 'status', 'injury_type', 'practice_participation']]
        print(sample.to_string(index=False))

    else:
        print("❌ No injury data retrieved")

    print("=" * 60)