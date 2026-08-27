import nfl_data_py as nfl
import pandas as pd
from os import path

import os
import requests

from season_utils import season_range

DATA_DIR = 'data_files/'

# Schedules for every season from FIRST_SEASON through the upcoming one.
seasons = list(season_range())
games = nfl.import_schedules(seasons)

# Show a sample
print(games.tail(15))

# Save to CSV
out_path = "nfl_games_historical.csv"
games.to_csv(path.join(DATA_DIR, out_path), index=False, sep='\t')
print(f"Saved game summaries to {out_path}")
