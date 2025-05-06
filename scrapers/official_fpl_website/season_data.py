import os
import sys

import pandas as pd
import requests
from unidecode import unidecode

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from utils.general import get_data_path, time_function  # noqa: E402


@time_function
def scrape_season_data(season: str):
    response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")

    data = response.json()
    players_data = data["elements"]

    # Full raw player data
    players_data_df = pd.DataFrame(players_data)
    players_data_df["first_name"] = players_data_df["first_name"].apply(unidecode)
    players_data_df["second_name"] = players_data_df["second_name"].apply(unidecode)
    players_data_df["web_name"] = players_data_df["web_name"].apply(unidecode)

    # Player IDs
    players_ids = pd.DataFrame(
        players_data_df[["id", "first_name", "second_name", "web_name", "team"]]
    )
    players_ids["full_name"] = (
        players_ids["first_name"] + " " + players_ids["second_name"]
    )
    players_ids.to_csv(get_data_path(season, "players_ids.csv"), index=False)

    # Team ids
    team_data = {}

    for team in data["teams"]:
        team_id = team["id"]
        team_name = team["name"]
        team_short_name = team["short_name"]
        team_data[team_id] = {"name": team_name, "short_name": team_short_name}

    teams_df = pd.DataFrame.from_dict(team_data, orient="index")
    teams_df.reset_index(inplace=True)
    teams_df.rename(columns={"index": "id"}, inplace=True)
    teams_df.to_csv(get_data_path(season, "teams_ids.csv"), index=False)

    # Get Players Season data
    players_data_df[players_data_df["id"] == 2].to_dict(orient="records")

    players_season_data = pd.DataFrame(
        players_data_df[
            [
                "id",
                "web_name",
                "team",
                "element_type",
                "points_per_game",
                "total_points",
                "minutes",
                "starts",
                "goals_scored",
                "assists",
                "yellow_cards",
                "clean_sheets",
                "goals_conceded",
                "saves",
                # per 90 (Not expected data)
                "goals_conceded_per_90",
                # Expected data
                "expected_goals",
                "expected_assists",
                "expected_goals_conceded",
                "expected_goal_involvements",
                # per 90
                "expected_goals_per_90",
                "expected_assists_per_90",
                "expected_goals_conceded_per_90",
                "expected_goal_involvements_per_90",
            ]
        ]
    )

    players_season_data.rename(
        columns={
            "id": "player_id",
            "team": "team_id",
            "expected_goals": "xG",
            "expected_assists": "xA",
            "expected_goals_conceded": "xGC",
            "expected_goal_involvements": "xGI",
            "expected_goals_per_90": "xG_per90",
            "expected_assists_per_90": "xA_per90",
            "expected_goals_conceded_per_90": "xGC_per90",
            "expected_goal_involvements_per_90": "xGI_per90",
        },
        inplace=True,
    )

    # Dtypes
    players_season_data["points_per_game"] = players_season_data[
        "points_per_game"
    ].astype(float)
    players_season_data["xG"] = players_season_data["xG"].astype(float)
    players_season_data["xA"] = players_season_data["xA"].astype(float)
    players_season_data["xGC"] = players_season_data["xGC"].astype(float)
    players_season_data["xGI"] = players_season_data["xGI"].astype(float)

    players_season_data.to_csv(
        get_data_path(season, "players_season_data.csv"), index=False
    )


if __name__ == "__main__":
    scrape_season_data("2024-25")
