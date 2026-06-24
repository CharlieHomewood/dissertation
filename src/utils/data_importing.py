from socceraction.data.statsbomb import StatsBombLoader
from statsbombpy import sb
from typing import Optional, Tuple
import requests
import pandas as pd

session = requests.Session()
SBL = StatsBombLoader()

def load_360(match_id: int) -> Optional[pd.DataFrame]:
    """
    Load StatsBomb 360 data for a given match from the public GitHub repository.

    The function attempts to download the 360 event tracking data for a match.
    If the data exists (HTTP 200), it is returned as a pandas DataFrame.
    If the data is not available (e.g., 404) or an error occurs, None is returned.

    Args:
        match_id (int): StatsBomb match ID.

    Returns:
        Optional[pd.DataFrame]:
            - DataFrame containing 360 tracking data if available
            - None if no data exists or the request fails
    """
    url = f"https://raw.githubusercontent.com/statsbomb/open-data/3bfbffe1de5750ebd47d770be0bb924a10cde54f/data/three-sixty/{match_id}.json"
    
    try:
        r = session.get(url, timeout=10)

        if r.status_code == 200:
            data = r.json()

            if data:
                return pd.DataFrame(data)

        elif r.status_code != 404:
            print(f"360 request failed ({r.status_code}) for match {match_id}")

    except requests.exceptions.RequestException as e:
        print(f"Request error for match {match_id}: {e}")

    return None

def process_match(match_id: int) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]]:
    """
    Download StatsBomb events, SoccerAction events, and StatsBomb 360 data for a match.

    Args:
        match_id (int): StatsBomb match ID.

    Returns:
        Tuple of:
            - sb_events (DataFrame)
            - sa_events (DataFrame)
            - three_sixty (DataFrame or None)

        Returns None if the match fails.
    """
    try:
        sb_events = sb.events(match_id=match_id)
        sa_events = SBL.events(game_id=match_id)[[
            "game_id", "event_id", "period_id",
            "type_id", "type_name", "play_pattern_id",
            "play_pattern_name", "position_id", "extra"
        ]]
        three_sixty = load_360(match_id)

        return sb_events, sa_events, three_sixty

    except Exception as e:
        print(f"Error with match {match_id}: {e}")
        return None