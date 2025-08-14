import os
import sys

import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils.general import get_data_path, time_function  # noqa: E402


@time_function
def get_past_season_ids(season: str) -> None:
    """
    Get the player and teams IDs for the specified season.
    I downloaded data from a repo into a separate directory

    Args:
        season (str): The season for which to retrieve player IDs.
    """

    # ================= Players
    ids_df: pd.DataFrame = pd.read_csv(
        f"C:/Users/kayya/Documents/FPL Data/data/{season}/player_idlist.csv"
    )
    players_raw: pd.DataFrame = pd.read_csv(
        f"C:/Users/kayya/Documents/FPL Data/data/{season}/players_raw.csv"
    )

    # full name field
    ids_df["full_name"] = ids_df["first_name"] + " " + ids_df["second_name"]

    # get web name from players raw into ids_df
    web_name_map = players_raw.set_index("id")["web_name"]
    ids_df["web_name"] = ids_df["id"].map(web_name_map)

    team_id_map = players_raw.set_index("id")["team"]
    ids_df["team"] = ids_df["id"].map(team_id_map)

    # reorder
    ids_df = ids_df[["id", "first_name", "second_name", "web_name", "team", "full_name"]]  # type: ignore

    ids_df.to_csv(get_data_path(season, "players_ids.csv"), index=False)

    # ================= Teams
    team_df = pd.read_csv(f"C:/Users/kayya/Documents/FPL Data/data/{season}/teams.csv")
    team_df = team_df[["id", "name", "short_name"]]

    team_df.to_csv(get_data_path(season, "teams.csv"), index=False)


if __name__ == "__main__":
    get_past_season_ids("2023-24")
