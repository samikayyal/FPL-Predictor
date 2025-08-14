import os

import pandas as pd

from utils.general import get_data_path, time_function
from utils.get_ids import external_team_name_to_fpl_name, get_player_team, get_team_id


def get_fbref_player_merged_gws(season: str) -> pd.DataFrame:
    dir_list: list[str] = [
        dir for dir in os.listdir(get_data_path(season)) if dir.startswith("gws_")
    ]

    # Get the list of gameweek numbers
    gw_numbers = [
        max(
            [
                int(filename[2:-4])
                for filename in os.listdir(get_data_path(season, dir))
                if filename.endswith(".csv")
            ]
        )
        for dir in dir_list
    ]

    # Ensure all gameweek numbers are the same
    if not all(n == gw_numbers[0] for n in gw_numbers):
        raise ValueError("Not all gameweek numbers are the same")

    # merge each stat type's gw i dataframes
    gw_dfs: dict[int, pd.DataFrame] = {}
    for dir in dir_list:
        for gw in range(1, gw_numbers[0] + 1):
            df = pd.read_csv(get_data_path(season, dir, f"gw{gw}.csv"))

            # drop useless columns
            columns_to_drop: list[str] = [
                "player_name",
                "player_number",
                "nation",
                "pos",
                "age",
            ]
            for col in columns_to_drop:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)

            if gw not in gw_dfs:
                gw_dfs[gw] = df
            else:
                # Dont include columns already in gw_dfs[gw]
                for col in df.columns:
                    if col in gw_dfs[gw].columns and col != "player_id":
                        df.drop(columns=[col], inplace=True)
                gw_dfs[gw] = pd.merge(gw_dfs[gw], df, on="player_id", how="outer")

    # concat all gameweek dataframes
    print(gw_dfs.keys())
    merged: pd.DataFrame = pd.concat(gw_dfs.values(), ignore_index=True)

    return merged


@time_function
def main():
    season = "2024-25"
    print(get_fbref_player_merged_gws(season).columns.tolist())


if __name__ == "__main__":
    main()
