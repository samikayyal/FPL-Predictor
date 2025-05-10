import os
import sys
import time

import pandas as pd
import requests

# Add the project root directory to the system path to import utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)


from utils.constants import SEASON  # noqa: E402
from utils.general import get_data_path, time_function  # noqa: E402
from utils.get_ids import get_player_name  # noqa: E402


@time_function # about 1476 seconds
def scrape_gw_data_official_fpl_website(season: str):
    # player idlist
    players_ids = sorted(
        pd.read_csv(get_data_path(season, "players_ids.csv"))["id"].tolist()
    )
    num_players = len(players_ids)

    gw_data: dict[int:list] = {gw_number: [] for gw_number in range(1, 39)}

    for player_id in players_ids:
        # Add a delay to avoid overwhelming the server
        time.sleep(0.7)

        response = requests.get(
            f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
        )
        if response.status_code != 200:
            print(
                f"Failed to fetch data for player {player_id}/{num_players}, status code: {response.status_code}"
            )
            continue

        data = response.json()
        print(
            f"Got data for player {player_id}/{num_players}, {get_player_name(player_id, season)}"
        )
        for game in data["history"]:
            gw_number = game["round"]
            player_id = game["element"]

            player_data = {
                "player_id": player_id,
                "minutes": game["minutes"],
                "in_starting_xi": game["starts"],
                "was_home": game["was_home"],
                "goals_scored": game["goals_scored"],
                "assists": game["assists"],
                "goals_conceded": game["goals_conceded"],
                "clean_sheets": game["clean_sheets"],
                "xG": game["expected_goals"],
                "xA": game["expected_assists"],
                "xGC": game["expected_goals_conceded"],
                "xGI": game["expected_goal_involvements"],
                # Need to add key passes and chances created
                # and others
                "total_points": game["total_points"],
                "gameweek": gw_number,
                "oppeonent_team_id": game["opponent_team"],
                "fixture": game["fixture"],  # I still don't know what this is
                "value": game["value"],  # Player price in that gameweek
            }

            gw_data[gw_number].append(player_data)

        # After each player, save the data to a CSV file in case of interruption
        for gw_number, players in gw_data.items():
            if players:
                df = pd.DataFrame(players)
                df.to_csv(get_data_path(season, f"gws/gw{gw_number}.csv"), index=False)


if __name__ == "__main__":
    scrape_gw_data_official_fpl_website(SEASON)
