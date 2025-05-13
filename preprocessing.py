import os

import pandas as pd

from utils.constants import SEASON
from utils.general import get_data_path, time_function
from utils.get_ids import external_team_name_to_fpl_name, get_player_team, get_team_id


# Merge stats from fpl-data into a single DataFrame
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

    return merged_stat_df


def get_merged_gws_official_fpl(season: str = SEASON) -> pd.DataFrame:
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

    merged_gw_df = pd.concat(gw_df_list, ignore_index=True)

    merged_gw_df.rename(columns={"gameweek": "gw"}, inplace=True)
    merged_gw_df.drop(columns=["fixture", "value"], inplace=True)

    merged_gw_df["was_home"] = merged_gw_df["was_home"].astype(bool)

    return merged_gw_df


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

    merged_stat_df = get_merged_stat_df(season)
    merged_gw_df = get_merged_gws_official_fpl(season)
    merged_passing_gw_df = get_merged_passing_gws_fbref(season)

    final_merged = pd.merge(
        merged_gw_df,
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
    final_merged["team_id"] = final_merged["player_id"].apply(
        get_player_team, args=(season,)
    )

    final_merged.fillna(0, inplace=True)

    return final_merged


def get_team_stats_df(season: str = SEASON, last_x_gameweeks: int = 10) -> pd.DataFrame:
    """
    Merges and averages the stats for each team in the merged DataFrame.
    Returns:
        pd.DataFrame: DataFrame containing team statistics.
    """
    print("Getting team stats...")
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

    if last_x_gameweeks > len(gws_dfs):
        print(
            f"Warning: last_x_gameweeks ({last_x_gameweeks}) is greater than the number of gameweeks available ({len(gws_dfs)}). Using all available gameweeks."
        )
        last_x_gameweeks = len(gws_dfs)

    last_x = pd.concat(gws_dfs[-last_x_gameweeks:], ignore_index=True)

    last_x.rename(
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
    last_x.drop(columns=["gw"], inplace=True)

    grouped = last_x.groupby(["team_id", "was_home"]).mean().reset_index()
    vs_grouped = last_x.groupby(["vs_team_id", "was_home"]).mean().reset_index()

    # Get the against stats to merge with the grouped stats
    grouped["shots_in_box_against"] = grouped.apply(
        lambda row: vs_grouped[
            (vs_grouped["vs_team_id"] == row["team_id"])
            & (vs_grouped["was_home"] == row["was_home"])
        ]["shots_in_box"].values[0],
        axis=1,
    )

    grouped["shots_against"] = grouped.apply(
        lambda row: vs_grouped[
            (vs_grouped["vs_team_id"] == row["team_id"])
            & (vs_grouped["was_home"] == row["was_home"])
        ]["shots"].values[0],
        axis=1,
    )
    grouped["shots_on_target_against"] = grouped.apply(
        lambda row: vs_grouped[
            (vs_grouped["vs_team_id"] == row["team_id"])
            & (vs_grouped["was_home"] == row["was_home"])
        ]["shots_on_target"].values[0],
        axis=1,
    )

    grouped["cc_against"] = grouped.apply(
        lambda row: vs_grouped[
            (vs_grouped["vs_team_id"] == row["team_id"])
            & (vs_grouped["was_home"] == row["was_home"])
        ]["chances_created"].values[0],
        axis=1,
    )

    return grouped


@time_function  # 200 seconds
def main():
    merged_gw = merge_all_gw_data(SEASON)

    team_stats = get_team_stats_df(SEASON, 10)
    team_stats = team_stats[
        ["team_id", "was_home"]
        + [col for col in team_stats.columns if col not in merged_gw.columns]
    ]

    print("Merging team stats with merged_gw")
    # Add team stats to merged_gw depending on home/away
    merged_gw = pd.merge(
        merged_gw,
        team_stats,
        on=["team_id", "was_home"],
        how="left",
    )

    # Add season data to merged_gw
    print("Merging season data with merged_gw")
    season_data = pd.read_csv(get_data_path(SEASON, "players_season_data.csv"))

    season_data.drop(columns=["team_id"], inplace=True)

    merged_gw = pd.merge(
        merged_gw,
        season_data,
        on=["player_id"],
        how="left",
        suffixes=("", "_season"),
    )

    merged_gw.to_csv("temp.csv", index=False)


if __name__ == "__main__":
    main()
