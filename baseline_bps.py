import pandas as pd

from utils.constants import SEASON
from utils.general import get_data_path, time_function


@time_function  # 0.0669 seconds
def calculate_baseline_bps(season: str = SEASON):
    players_season_df = pd.read_csv(get_data_path(season, "players_season_data.csv"))

    goalkeepers = players_season_df[players_season_df["element_type"] == 1].copy()
    defenders = players_season_df[players_season_df["element_type"] == 2].copy()
    midfielders = players_season_df[players_season_df["element_type"] == 3].copy()
    forwards = players_season_df[players_season_df["element_type"] == 4].copy()

    # Goalkeepers
    goalkeepers.loc[:, "baseline_bps"] = (
        goalkeepers["bps"]
        - goalkeepers["goals_scored"] * 12
        - goalkeepers["assists"] * 9
        - goalkeepers["clean_sheets"] * 12
        - goalkeepers["penalties_saved"] * 9
        - goalkeepers["red_cards"] * -9
        - goalkeepers["yellow_cards"] * -3
        - goalkeepers["goals_conceded"] * -4
        - goalkeepers["own_goals"] * -6
    )

    # Defenders
    defenders.loc[:, "baseline_bps"] = (
        defenders["bps"]
        - defenders["goals_scored"] * 12
        - defenders["assists"] * 9
        - defenders["clean_sheets"] * 12
        - defenders["red_cards"] * -9
        - defenders["yellow_cards"] * -3
        - defenders["penalties_missed"] * -6
        - defenders["goals_conceded"] * -4
        - defenders["own_goals"] * -6
    )

    # Midfielders
    midfielders.loc[:, "baseline_bps"] = (
        midfielders["bps"]
        - midfielders["goals_scored"] * 18
        - midfielders["assists"] * 9
        - midfielders["red_cards"] * -9
        - midfielders["yellow_cards"] * -3
        - midfielders["penalties_missed"] * -6
        - midfielders["own_goals"] * -6
    )

    # Forwards
    forwards.loc[:, "baseline_bps"] = (
        forwards["bps"]
        - forwards["goals_scored"] * 24
        - forwards["assists"] * 9
        - forwards["yellow_cards"] * -3
        - forwards["red_cards"] * -9
        - forwards["penalties_missed"] * -6
        - forwards["own_goals"] * -6
    )

    # Combine all players
    players_season_df = pd.concat(
        [goalkeepers, defenders, midfielders, forwards], ignore_index=True
    ).sort_values(by="player_id")

    # Save the data
    players_season_df.to_csv(
        get_data_path(SEASON, "players_season_data.csv"), index=False
    )
    print("Baseline BPS data calculated and saved to players_season_data.csv")


if __name__ == "__main__":
    calculate_baseline_bps(SEASON)
