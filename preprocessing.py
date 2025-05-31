import os

import pandas as pd

from utils.constants import LAST_PLAYED_GAMEWEEK, MANAGER_PLAYER_IDS, SEASON
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


def get_final_dfs_official_fpl(season: str = SEASON) -> pd.DataFrame:
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

    final_df_df = pd.concat(gw_df_list, ignore_index=True)

    final_df_df.rename(columns={"gameweek": "gw"}, inplace=True)
    final_df_df.drop(columns=["fixture", "value"], inplace=True)

    final_df_df["was_home"] = final_df_df["was_home"].astype(bool)

    return final_df_df


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

    final_df_df = get_final_dfs_official_fpl(season).fillna(0)
    merged_stat_df = get_merged_stat_df(season).fillna(0)
    merged_passing_gw_df = get_merged_passing_gws_fbref(season).fillna(0)

    final_merged = pd.merge(
        final_df_df,
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


def get_team_stats_df(season: str = SEASON, last_x_gameweeks: int = 10) -> pd.DataFrame:
    """
    Get the average of stats last x gameweeks for each team using the same logic as get_last_x_players_gw.
    Returns:
        pd.DataFrame: DataFrame containing team statistics for the last x gameweeks.
    """
    print(f"Getting team stats with lag {last_x_gameweeks}...")
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
    all_gws = pd.concat(gws_dfs, ignore_index=True)
    all_gws.rename(
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

    if last_x_gameweeks > max(all_gws["gw"]):
        print(
            f"Warning: last_x_gameweeks ({last_x_gameweeks}) is greater than the number of gameweeks available ({max(all_gws['gw'])}). Using all available gameweeks."
        )
        last_x_gameweeks = max(all_gws["gw"])

    # Remove Gameweek column as it's not needed for averaging
    all_gws = all_gws.drop(columns=["Gameweek"])

    data = []
    # Loop through each team and gameweeks that team played in
    for team_id in all_gws["team_id"].unique():
        team_df = all_gws[all_gws["team_id"] == team_id]

        played_gameweeks = team_df["gw"].unique().tolist()

        team_gw_dfs = []
        for gw in played_gameweeks:
            # Get the gameweeks the team played in before the current gameweek
            gws_before_current = sorted(
                [g for g in played_gameweeks if g < gw], reverse=True
            )[-last_x_gameweeks:]
            if len(gws_before_current) == 0:
                continue
            # If the team has played less than last_x_gameweeks prior to 'gw', use all available gameweeks
            elif len(gws_before_current) < last_x_gameweeks:
                gws_before_current = sorted(gws_before_current, reverse=True)

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
            print(f"Team {team_id} has no data for last {last_x_gameweeks} gameweeks.")
            continue

        # Concatenate the data for the team across all gameweeks
        team_df_final = pd.concat(team_gw_dfs, ignore_index=True)
        data.append(team_df_final)

    if len(data) == 0:
        print(f"No data found for last {last_x_gameweeks} gameweeks.")
        return pd.DataFrame()

    final_df = pd.concat(data, ignore_index=True)

    # Rename columns to include the lag suffix
    final_df.columns = [
        (
            col + f"_last_{last_x_gameweeks}_team"
            if col not in ["team_id", "was_home", "gw"]
            else col
        )
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


def get_last_x_players_gw(
    merged_gw: pd.DataFrame, last_x_gameweeks: int
) -> pd.DataFrame:
    """
    Get the sum of stats last x gameweeks for each player in the merged DataFrame.
    Args:
        merged_gw (pd.DataFrame): Merged DataFrame containing player statistics. from merge_all_gw_data
        last_x_gameweeks (int): Number of gameweeks to consider.
    Returns:
        pd.DataFrame: DataFrame containing player statistics for the last x gameweeks.
    """
    print(f"Getting player stats with lag {last_x_gameweeks}...")
    if last_x_gameweeks > max(merged_gw["gw"]):
        print(
            f"Warning: last_x_gameweeks ({last_x_gameweeks}) is greater than the number of gameweeks available ({len(merged_gw)}). Using all available gameweeks."
        )
        last_x_gameweeks = max(merged_gw["gw"])

    merged_gw = merged_gw.copy()
    merged_gw = merged_gw.drop(
        columns=[
            "was_home",
            "opponent_team_id",
            "team_id",
            "in_starting_xi",
        ]
    )

    data = []
    # Loop through each player and gameweeks that player played in
    for player_id in merged_gw["player_id"].unique():
        player_df = merged_gw[merged_gw["player_id"] == player_id]

        played_gameweeks = player_df["gw"].unique().tolist()

        player_gw_dfs = []
        for gw in played_gameweeks:
            # Get the 3 gameweeks the player played in before the current gameweek
            gws_before_current = sorted(
                [g for g in played_gameweeks if g < gw], reverse=True
            )[-last_x_gameweeks:]
            if len(gws_before_current) == 0:
                continue
            # If the player has played less than last_x_gameweeks prior to 'gw', use all available gameweeks
            elif len(gws_before_current) < last_x_gameweeks:
                gws_before_current = sorted(gws_before_current, reverse=True)

            # Get the data for the last x gameweeks
            last_x_df = player_df[player_df["gw"].isin(gws_before_current)].drop(
                columns=["gw"]
            )

            # Sum the stats for the last x gameweeks
            last_x_df = last_x_df.groupby("player_id").sum().reset_index()
            last_x_df.loc[:, "gw"] = gw

            player_gw_dfs.append(last_x_df)

        # If the player has no data for the last x gameweeks, skip
        if len(player_gw_dfs) == 0:
            print(
                f"Player {player_id} has no data for last {last_x_gameweeks} gameweeks."
            )
            continue

        # Concatenate the data for the player across all gameweeks
        player_df = pd.concat(player_gw_dfs, ignore_index=True)
        data.append(player_df)

    if len(data) == 0:
        print(f"No data found for last {last_x_gameweeks} gameweeks.")
        return pd.DataFrame()

    final_df = pd.concat(data, ignore_index=True)

    final_df.columns = [
        (
            col + f"_last_{last_x_gameweeks}"
            if col not in ["player_id", "gw", "team_id", "opponent_team_id"]
            else col
        )
        for col in final_df.columns
    ]

    # final_df.to_csv(f"last_{last_x_gameweeks}_players_gw.csv", index=False)
    return final_df


@time_function  # 200 seconds
def main():
    merged_gw = merge_all_gw_data()
    final_df = merged_gw[
        ["player_id", "gw", "team_id", "opponent_team_id", "was_home", "total_points"]
    ].copy()

    # Get the last x gameweeks for each player
    for lag in [3, 5, 10]:
        players_df = get_last_x_players_gw(merged_gw, lag)
        final_df = pd.merge(
            final_df,
            players_df,
            on=["player_id", "gw"],
            how="left",
        )

    # Nulls are players that didn't play in the last x gameweeks and players with blank gameweeks
    # and players that joined mid-season
    # We can drop these rows as they are not useful
    final_df.dropna(inplace=True)

    # # Merge with team stats
    for lag in [3, 5, 10]:
        team_df = get_team_stats_df(last_x_gameweeks=lag)
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

    # For now drop na, i have to fix the data gathering to ensure that all teams have data for all gameweeks
    final_df.dropna(inplace=True)

    # Merge with players season data which includes positions
    players_season_df = get_players_season_data()
    final_df = pd.merge(
        final_df,
        players_season_df,
        on=["player_id"],
        how="left",
    )

    final_df.to_csv("final_df.csv", index=False)


if __name__ == "__main__":
    main()
