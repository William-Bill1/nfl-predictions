import nfl_data_py as nfl
import pandas as pd
from os import path

import os
import requests
import argparse

from season_utils import FIRST_SEASON, latest_pbp_season

DATA_DIR = 'data_files/'
# Newest season with play-by-play data (rolls over at September kickoff).
end_year = latest_pbp_season()

def main():
    parser = argparse.ArgumentParser(description='Download NFL play-by-play data')
    parser.add_argument('--current-season-only', action='store_true', 
                       help='Only download current season data (much faster for nightly updates)')
    args = parser.parse_args()
    
    if args.current_season_only:
        # Only download current season for nightly updates
        YEARS = [end_year]
        print(f"📅 Downloading only current season ({end_year}) data...")
    else:
        # Download all historical seasons
        YEARS = range(FIRST_SEASON, end_year + 1)
        print(f"📚 Downloading all seasons ({min(YEARS)}-{end_year}) data...")

    data = pd.DataFrame()

    for i in YEARS:
        try:
            print(f"Downloading {i} season data...")
            i_data = pd.read_csv('https://github.com/nflverse/nflverse-data/releases/download/pbp/' \
                           'play_by_play_' + str(i) + '.csv.gz',
                           compression= 'gzip', low_memory= False)
            
            data = pd.concat([data, i_data], ignore_index=True, sort=True)
            data.reset_index(drop=True, inplace=True)
            print(f"✓ {i} season: {len(i_data):,} plays")
        except Exception as e:
            print(f"⚠️  Warning: Could not download {i} season data - {e}")
            print(f"   Continuing with available data...")

    data.to_csv(path.join(DATA_DIR, 'nfl_play_by_play_historical.csv.gz'), compression='gzip', index=False, sep='\t')
    print(f"\n✅ Saved {len(data):,} plays ({min(YEARS)}-{end_year}) to nfl_play_by_play_historical.csv.gz")

if __name__ == '__main__':
    main()
