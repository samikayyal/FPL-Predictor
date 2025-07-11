import os

import pandas as pd

from utils.constants import (  # noqa: F401
    LAST_PLAYED_GAMEWEEK,
    MANAGER_PLAYER_IDS,
    SEASON,
)
from utils.general import get_data_path, time_function
from utils.get_ids import external_team_name_to_fpl_name, get_player_team, get_team_id


def get_merged_stat_df(season: str = SEASON) -> pd.DataFrame:
    """
    Merges all CSV files in the stat_data directory into a single DataFrame,
    Which are scraped gw-by-gw from fpl-data.
    Excludes the 'web_name' column if present and checks for duplicate entries.
    Returns:
        pd.DataFrame: Merged DataFrame containing player statistics.
    """
    merged_stat_df = None
    for file in os.listdir(get_data_path(season, "stat_data")):
        if file.endswith(".csv"):
            stat_df = pd.read_csv(get_data_path(season, "stat_data", file))
            if "web_name" in stat_df.columns:
                stat_df.drop(columns=["web_name"], inplace=True)

            if merged_stat_df is None:
                merged_stat_df = stat_df
            else:
                if stat_df.duplicated(subset=["player_id", "gw"]).any():
                    print(f"Duplicate entries found in {file}.")
                    continue

                merged_stat_df = pd.merge(
                    merged_stat_df, stat_df, on=["player_id", "gw"], how="outer"
                )

    # Goalkeepers dont have data for any of these (obviously)
    return merged_stat_df


def get_merged_gws_df_official_fpl(season: str = SEASON) -> pd.DataFrame:
    """
    Merges all CSV files in the gws directory into a single DataFrame,
    Which are scraped gw-by-gw from the official fpl website.
    Returns:
        pd.DataFrame: Merged DataFrame containing player statistics.
    """
    gw_df_list = [
        pd.read_csv(get_data_path(season, "gws", gw_df))
        for gw_df in os.listdir(get_data_path(season, "gws"))
        if gw_df.endswith(".csv")
    ]

    final_df = pd.concat(gw_df_list, ignore_index=True)

    final_df.rename(columns={"gameweek": "gw"}, inplace=True)
    final_df.drop(columns=["fixture", "value"], inplace=True)

    final_df["was_home"] = final_df["was_home"].astype(bool)

    return final_df


def get_merged_passing_gws_fbref(season: str = SEASON) -> pd.DataFrame:
    """
    Merges all CSV files in the fbref/gws_passing directory into a single DataFrame,
    Which are scraped gw-by-gw from fbref.
    Returns:
        pd.DataFrame: Merged DataFrame containing player statistics.
    """
    passing_gw_df_list = [
        pd.read_csv(get_data_path(season, "fbref", "gws_passing", passing_gw_df))
        for passing_gw_df in os.listdir(get_data_path(season, "fbref", "gws_passing"))
        if passing_gw_df.endswith(".csv")
    ]

    merged_passing_gw_df = pd.concat(passing_gw_df_list, ignore_index=True)
    return merged_passing_gw_df[
        ["player_id", "gw", "key_passes", "total_pass_completion_percentage"]
    ]


def merge_all_gw_data(season: str = SEASON) -> pd.DataFrame:
    """
    Merges all gw-by-gw data from fbref, fpl-data, and official fpl into a single DataFrame.
    Returns:
        pd.DataFrame: Merged DataFrame containing player statistics.
    """
    print("Merging all gw data...")

    final_df = get_merged_gws_df_official_fpl(season).fillna(0)
    merged_stat_df = get_merged_stat_df(season).fillna(0)
    merged_passing_gw_df = get_merged_passing_gws_fbref(season).fillna(0)

    final_merged = pd.merge(
        final_df,
        merged_stat_df,
        on=["player_id", "gw"],
        how="left",
    )

    final_merged = pd.merge(
        final_merged,
        merged_passing_gw_df,
        on=["player_id", "gw"],
        how="left",
    )

    # Drop managers
    final_merged = final_merged[~final_merged["player_id"].isin(MANAGER_PLAYER_IDS)]

    final_merged["team_id"] = final_merged["player_id"].apply(
        get_player_team, args=(season,)
    )

    final_merged.fillna(0, inplace=True)

    return final_merged


def get_team_stats_df(season: str = SEASON, lag: int = 10) -> pd.DataFrame:
    """
    Get the average of stats last x gameweeks for each team using the same logic as get_last_x_players_gw.
    Returns:
        pd.DataFrame: DataFrame containing team statistics for the last x gameweeks.
    """
    print(f"Getting team stats with lag {lag}...")

    # Load and prepare all gameweek data
    gws_dfs: list[pd.DataFrame] = []
    for file in os.listdir(get_data_path(season, "team_gws")):
        if file.endswith(".csv"):
            df = pd.read_csv(get_data_path(season, "team_gws", file))
            # Drop useless columns
            df.drop(
                columns=[
                    "xP",
                    "P",
                    "P vs xP",
                    "C 1/3",
                    "C PA",
                    "TPA",
                    "T",
                    "xCS",
                    "CS",
                ],
                inplace=True,
            )
            # Get team ids
            df["team_id"] = (
                df["Team"]
                .apply(external_team_name_to_fpl_name)
                .apply(get_team_id, args=("name", SEASON))
                .astype(int)
            )
            df["vs_team_id"] = (
                df["vs Team"]
                .apply(external_team_name_to_fpl_name)
                .apply(get_team_id, args=("name", SEASON))
                .astype(int)
            )
            df.drop(columns=["Team", "vs Team"], inplace=True)

            df["gw"] = int(file.split(".")[0][2:])

            gws_dfs.append(df)

    if len(gws_dfs) == 0:
        print("No team gameweek data found.")
        return pd.DataFrame()

    # Concatenate all gameweek data
    all_gws_df = pd.concat(gws_dfs, ignore_index=True)
    all_gws_df.rename(
        columns={
            "Home?": "was_home",
            "Shots": "shots",
            "SoT": "shots_on_target",
            "SiB": "shots_in_box",
            "G": "goals",
            "CC": "chances_created",
            "A": "assists",
            "GC": "goals_conceded",
        },
        inplace=True,
    )

    if lag > max(all_gws_df["gw"]):
        print(
            f"Warning: Lag ({lag}) is greater than the number of gameweeks available ({max(all_gws_df['gw'])}). Using all available gameweeks."
        )
        lag = max(all_gws_df["gw"])

    # Remove Gameweek column as it's not needed for averaging
    all_gws_df = all_gws_df.drop(columns=["Gameweek"])

    data = []
    # Loop through each team and gameweeks that team played in
    for team_id in all_gws_df["team_id"].unique():
        team_df = all_gws_df[all_gws_df["team_id"] == team_id]

        played_gameweeks = team_df["gw"].unique().tolist()

        team_gw_dfs = []
        for gw in played_gameweeks:
            # Get the gameweeks the team played in before the current gameweek
            gws_before_current = sorted([g for g in played_gameweeks if g < gw])[-lag:]
            if not gws_before_current:
                continue

            # Get the data for the last x gameweeks
            last_x_df = team_df[team_df["gw"].isin(gws_before_current)].drop(
                columns=["gw"]
            )

            # Group by team_id and was_home, then average the stats for the last x gameweeks
            last_x_grouped = (
                last_x_df.groupby(["team_id", "was_home"]).mean().reset_index()
            )

            # Create vs_grouped for opponent stats
            vs_last_x_df = team_df[team_df["gw"].isin(gws_before_current)]
            vs_grouped = (
                vs_last_x_df.groupby(["vs_team_id", "was_home"]).mean().reset_index()
            )

            # Add opponent stats
            for idx, row in last_x_grouped.iterrows():
                matching_vs = vs_grouped[
                    (vs_grouped["vs_team_id"] == row["team_id"])
                    & (vs_grouped["was_home"] == row["was_home"])
                ]
                if not matching_vs.empty:
                    last_x_grouped.loc[idx, "shots_in_box_against"] = matching_vs[
                        "shots_in_box"
                    ].values[0]
                    last_x_grouped.loc[idx, "shots_against"] = matching_vs[
                        "shots"
                    ].values[0]
                    last_x_grouped.loc[idx, "shots_on_target_against"] = matching_vs[
                        "shots_on_target"
                    ].values[0]
                    last_x_grouped.loc[idx, "cc_against"] = matching_vs[
                        "chances_created"
                    ].values[0]
                else:
                    # Fill with 0 if no matching opponent data
                    last_x_grouped.loc[idx, "shots_in_box_against"] = 0
                    last_x_grouped.loc[idx, "shots_against"] = 0
                    last_x_grouped.loc[idx, "shots_on_target_against"] = 0
                    last_x_grouped.loc[idx, "cc_against"] = 0

            # Drop vs_team_id if it exists
            if "vs_team_id" in last_x_grouped.columns:
                last_x_grouped.drop(columns=["vs_team_id"], inplace=True)

            last_x_grouped.loc[:, "gw"] = gw

            team_gw_dfs.append(last_x_grouped)

        # If the team has no data for the last x gameweeks, skip
        if len(team_gw_dfs) == 0:
            print(f"Team {team_id} has no data for last {lag} gameweeks.")
            continue

        # Concatenate the data for the team across all gameweeks
        team_df_final = pd.concat(team_gw_dfs, ignore_index=True)
        data.append(team_df_final)

    if len(data) == 0:
        print(f"No data found for last {lag} gameweeks.")
        return pd.DataFrame()

    final_df = pd.concat(data, ignore_index=True)

    # Rename columns to include the lag suffix
    final_df.columns = [
        (col + f"_last_{lag}_team" if col not in ["team_id", "was_home", "gw"] else col)
        for col in final_df.columns
    ]

    return final_df


def get_players_season_data(season: str = SEASON) -> pd.DataFrame:

    df = pd.read_csv(get_data_path(season, "players_season_data.csv"))
    df.drop(columns=["web_name", "team_id"], inplace=True)

    # Remove managers
    df = df[df.element_type != 5]

    # One hot encoding for the position (element_type)
    df = pd.get_dummies(df, columns=["element_type"], prefix="pos")
    df.rename(
        columns={
            "pos_1": "is_GK",
            "pos_2": "is_DEF",
            "pos_3": "is_MID",
            "pos_4": "is_FWD",
        },
        inplace=True,
    )

    df.columns = [
        (
            col + "_season"
            if col not in ["player_id", "is_GK", "is_DEF", "is_MID", "is_FWD"]
            else col
        )
        for col in df.columns
    ]

    return df


def get_last_x_players_gw(merged_gw: pd.DataFrame, lag: int) -> pd.DataFrame:
    """
    Get the sum of stats last x gameweeks for each player in the merged DataFrame.
    Args:
        merged_gw (pd.DataFrame): Merged DataFrame containing player statistics. from merge_all_gw_data
        last_x_gameweeks (int): Number of gameweeks to consider.
    Returns:
        pd.DataFrame: DataFrame containing player statistics for the last x gameweeks.
    """
    print(f"Getting player stats with lag {lag}...")
    if lag > max(merged_gw["gw"]):
        print(
            f"Warning: last_x_gameweeks ({lag}) is greater than the number of gameweeks available ({len(merged_gw)}). Using all available gameweeks."
        )
        lag = max(merged_gw["gw"])

    # columns to sum and columns to keep
    id_cols = ["player_id", "gw", "opponent_team_id"]
    cols_to_drop = ["was_home", "team_id", "in_starting_xi"]
    stat_cols = [
        col for col in merged_gw.columns if col not in (id_cols + cols_to_drop)
    ]

    df = merged_gw[id_cols + stat_cols].copy()
    df.set_index(id_cols, inplace=True)  # Set index by player and gameweek
    df = df.sort_values(by=["player_id", "gw"])

    # Create rolling sums for each player
    # use level here because player_id is index
    rolling_sum = (
        df.groupby(level="player_id")[stat_cols]
        .rolling(window=lag, min_periods=1, closed="left")
        .sum()
    )
    rolling_sum.columns = [f"{col}_last_{lag}" for col in rolling_sum.columns]
    # Reset index to remove the extra player_id level
    rolling_sum = rolling_sum.reset_index(level=0, drop=True)

    # reset again to get player_id and gw back as columns
    rolling_sum = rolling_sum.reset_index()

    # Drop na values (first gameweek for each player will have NaN values)
    rolling_sum.dropna(inplace=True)

    return rolling_sum


@time_function  # 200 seconds
def main():
    merged_gw = merge_all_gw_data()
    final_df = merged_gw[
        ["player_id", "gw", "team_id", "opponent_team_id", "was_home", "total_points"]
    ].copy()

    # Get the last x gameweeks for each player
    print("----------- Getting last x gameweeks for players ----------")
    for lag in [3, 5, 10]:
        players_df = get_last_x_players_gw(merged_gw, lag)
        final_df = pd.merge(
            final_df,
            players_df,
            on=["player_id", "gw", "opponent_team_id"],
            how="left",
        )
        print(f"Added last {lag} gameweeks stats for players. Shape: {final_df.shape}")

    # Nulls are players that didn't play in the last x gameweeks and players with blank gameweeks
    # and players that joined mid-season
    # We can drop these rows as they are not useful
    final_df.dropna(inplace=True)
    final_df.to_csv("temp.csv", index=False)
    # # Merge with team stats
    print("----------- Getting last x gameweeks for teams ----------")
    for lag in [3, 5, 10]:
        team_df = get_team_stats_df(lag=lag)
        print("Team df nulls:", team_df.isnull().sum().sum())
        final_df = pd.merge(
            final_df,
            team_df,
            on=["team_id", "was_home", "gw"],
            how="left",
        )
        team_df.columns = [
            (col + "_opponent" if col not in ["team_id", "was_home", "gw"] else col)
            for col in team_df.columns
        ]
        final_df = pd.merge(
            final_df,
            team_df,
            left_on=["opponent_team_id", "was_home", "gw"],
            right_on=["team_id", "was_home", "gw"],
            how="left",
            suffixes=("", "_drop"),
        )
        final_df.drop(columns=["team_id_drop"], inplace=True)

        print(
            f"Added last {lag} gameweeks stats for teams. Shape: {final_df.shape}, Nulls: {final_df.isnull().sum().sum()}"
        )

    # For now drop na, i have to fix the data gathering to ensure that all teams have data for all gameweeks
    final_df.dropna(inplace=True)

    # Merge with players season data which includes positions
    print("----------- Merging with players season data ----------")
    players_season_df = get_players_season_data()
    final_df = pd.merge(
        final_df,
        players_season_df,
        on=["player_id"],
        how="left",
    )

    final_df.to_csv("final_df.csv", index=False)
    print(f"Final DataFrame shape: {final_df.shape}")
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
