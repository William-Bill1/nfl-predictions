"""get_recent_starters roster-id filtering (the opt-in PROP_ROSTER_FILTER path)."""

import pandas as pd

from player_props.predict import get_recent_starters


def _stats():
    # two players on ATL across the same two recent games
    return pd.DataFrame({
        'team': ['ATL', 'ATL', 'ATL', 'ATL'],
        'game_id': ['2025_18_ATL_x', '2025_18_ATL_x', '2025_17_ATL_y', '2025_17_ATL_y'],
        'game_date': ['2025-01-05', '2025-01-05', '2024-12-29', '2024-12-29'],
        'player_name': ['A.Active', 'B.Gone', 'A.Active', 'B.Gone'],
        'player_id': ['00-0000001', '00-0000002', '00-0000001', '00-0000002'],
    })


def test_no_roster_ids_returns_everyone():
    got = set(get_recent_starters(_stats(), 'ATL', roster_ids=None))
    assert got == {'A.Active', 'B.Gone'}


def test_roster_ids_drop_players_not_on_it():
    got = set(get_recent_starters(_stats(), 'ATL', roster_ids={'00-0000001'}))
    assert got == {'A.Active'}


def test_empty_roster_ids_drops_everyone():
    assert get_recent_starters(_stats(), 'ATL', roster_ids=frozenset()) == []
