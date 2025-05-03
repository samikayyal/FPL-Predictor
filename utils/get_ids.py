import pandas as pd
from unidecode import unidecode

from utils.general import get_data_path


def get_player_id(player_name: str, name_type: str, season: str) -> int:
    """
    Get the player ID from the player name.
    Args:
        player_name (str): The player name.
        name_type (str): The type of name to search for. Must be one of ['web_name', 'first_name', 'second_name', 'full_name'].
        season (str): The season for which to get the player ID. must be in format like '2023-24'
    Returns:
        int: The player ID.
    """
    if name_type not in ["web_name", "first_name", "second_name", "full_name"]:
        raise ValueError(
            "name_type must be one of ['web_name', 'first_name', 'second_name', 'full_name']"
        )

    # Normalize the player name to handle special characters
    player_name = unidecode(player_name)

    filepath = get_data_path(season, "players_ids.csv")
    df = pd.read_csv(filepath)
    player_id = df.loc[df[name_type] == player_name, "id"]
    if player_id.empty:
        raise ValueError(
            f"Player {player_name} not found in the dataset for season {season}."
        )  # Added season context
    return player_id.values[0]


def get_player_name(player_id: int, season: str) -> str:
    """
    Get the player name from the player ID.
    Args:
        player_id (int): The player ID.
        season (str): The season for which to get the player name. must be in format like '2023-24'
    Returns:
        str: The player name as web_name.
    """
    filepath = get_data_path(season, "players_ids.csv")
    df = pd.read_csv(filepath)
    player_name = df.loc[df["id"] == player_id, "web_name"]
    if player_name.empty:
        raise ValueError(
            f"Player ID {player_id} not found in the dataset for season {season}."
        )  # Added season context
    return player_name.values[0]


def get_team_id(team_name: str, name_type: str, season: str) -> int:
    """
    Get the team ID from the team name.
    Args:
        team_name (str): The team name.
        name_type (str): The type of name to search for. Must be one of ['name', 'short_name'].
        season (str): The season for which to get the team ID. must be in format like '2023-24'
    Returns:
        int: The team ID.
    """
    if name_type not in ["name", "short_name"]:
        raise ValueError("name_type must be one of ['name', 'short_name']")

    filepath = get_data_path(season, "teams_ids.csv")
    df = pd.read_csv(filepath)

    team_id = df.loc[df[name_type] == team_name, "id"]
    if team_id.empty:
        raise ValueError(
            f"Team {team_name} not found in the dataset for season {season}."
        )  # Added season context
    return team_id.values[0]


def get_team_name(team_id: int, season: str) -> str:
    """
    Get the team name from the team ID.
    Args:
        team_id (int): The team ID.
        season (str): The season for which to get the team name. must be in format like '2023-24'
    Returns:
        str: The team name as full name.
    """
    filepath = get_data_path(season, "teams_ids.csv")
    df = pd.read_csv(filepath)
    team_name = df.loc[df["id"] == team_id, "name"]
    if team_name.empty:
        raise ValueError(
            f"Team ID {team_id} not found in the dataset for season {season}."
        )  # Added season context
    return team_name.values[0]
