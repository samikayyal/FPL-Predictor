import os

import pandas as pd

from utils.general import get_data_path, time_function
from utils.get_ids import external_team_name_to_fpl_name, get_player_team, get_team_id


def get_fbref_player_merged_gws(season: str) -> pd.DataFrame:
    dir_list: list[str] = [
        dir for dir in os.listdir(get_data_path(season)) if dir.startswith("gws_")
    ]
