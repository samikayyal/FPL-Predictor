import pandas as pd
from rapidfuzz import fuzz, process

from utils.constants import SEASON
from utils.general import get_data_path, normalize_name

# Preload CSVs for the current season to improve performance
PLAYERS_IDS_DF = pd.read_csv(get_data_path(SEASON, "players_ids.csv"))
TEAMS_IDS_DF = pd.read_csv(get_data_path(SEASON, "teams_ids.csv"))
FIXTURES_DF = pd.read_csv(get_data_path(SEASON, "fixtures.csv"))


def get_player_id(player_name: str, name_type: str, season: str) -> int | None:
    """
    Get the player ID from the player name.
    Args:
        player_name (str): The player name.
        name_type (str): The type of name to search for. Must be one of ['web_name', 'first_name', 'second_name', 'full_name'].
        season (str): The season for which to get the player ID. must be in format like '2023-24'
        fbref_name (bool): Whether the player name is in FBRef format.
    Returns:
        int: The player ID.
    """
    if name_type not in ["web_name", "first_name", "second_name", "full_name"]:
        raise ValueError(
            "name_type must be one of ['web_name', 'first_name', 'second_name', 'full_name']"
        )

    # Normalize the player name to handle special characters
    player_name = normalize_name(player_name)

    if season == SEASON:
        df = PLAYERS_IDS_DF
    else:
        df = pd.read_csv(get_data_path(season, "players_ids.csv"))
    player_id = df.loc[df[name_type] == player_name, "id"]
    if player_id.empty:
        # raise ValueError(
        #     f"Player {player_name} not found in the dataset for season {season}."
        # )
        return None
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
    if season == SEASON:
        df = PLAYERS_IDS_DF
    else:
        df = pd.read_csv(get_data_path(season, "players_ids.csv"))
    player_name = df.loc[df["id"] == player_id, "web_name"]
    if player_name.empty:
        raise ValueError(
            f"Player ID {player_id} not found in the dataset for season {season}."
        )  # Added season context
    return player_name.values[0]


def get_team_id(team_name: str, name_type: str, season: str) -> int | None:
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

    if season == SEASON:
        df = TEAMS_IDS_DF
    else:
        df = pd.read_csv(get_data_path(season, "teams_ids.csv"))

    team_id = df.loc[df[name_type] == team_name, "id"]
    if team_id.empty:
        # raise ValueError(
        #     f"Team {team_name} not found in the dataset for season {season}."
        # )
        return None
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
    if season == SEASON:
        df = TEAMS_IDS_DF
    else:
        df = pd.read_csv(get_data_path(season, "teams_ids.csv"))
    team_name = df.loc[df["id"] == team_id, "name"]
    if team_name.empty:
        raise ValueError(
            f"Team ID {team_id} not found in the dataset for season {season}."
        )  # Added season context
    return team_name.values[0]


def external_team_name_to_fpl_name(name: str) -> str:
    """
    Convert a team name from an external source to FPL format.
    Args:
        name (str): The team name in an external source.
    Returns:
        str: The team name in FPL format. If not found, returns the original name.
    """
    # Mapping of FBRef team names to FPL team names
    external_to_fpl_mapping = {
        "tottenham": "Spurs",
        "nott'ham forest": "Nott'm Forest",
        "newcastle utd": "Newcastle",
        "manchester utd": "Man Utd",
        "manchester city": "Man City",
        "leicester city": "Leicester",
        "ipswich town": "Ipswich",
        "wolverhampton wanderers": "Wolves",
        "nottingham forest": "Nott'm Forest",
        "manchester united": "Man Utd",
        "newcastle united": "Newcastle",
        # Add more mappings as needed
    }

    return external_to_fpl_mapping.get(
        name.lower().strip(), name
    )  # Return the original name if not found


def get_player_team(player_id: int, season: str) -> int:
    """
    Get the team name from the player ID.
    Args:
        player_id (int): The player ID.
        season (str): The season for which to get the team name. must be in format like '2023-24'
    Returns:
        int: The team ID for current season.
    """
    if season == SEASON:
        player_ids_df = PLAYERS_IDS_DF
    else:
        player_ids_df = pd.read_csv(get_data_path(season, "players_ids.csv"))

    team_id = player_ids_df[player_ids_df["id"] == player_id]

    if team_id.empty:
        raise ValueError(
            f"Player ID {player_id} not found in the dataset for season {season}."
        )
    return int(team_id["team"].values[0])


def get_match_gw(home_team_id: int, away_team_id: int, season: str) -> int:
    """
    Get the gameweek of a match.
    """
    if season == SEASON:
        fixtures = FIXTURES_DF
    else:
        fixtures = pd.read_csv(get_data_path(season, "fixtures.csv"))
    match_gw = fixtures[
        (fixtures["home_team_id"] == home_team_id)
        & (fixtures["away_team_id"] == away_team_id)
    ].gw.values[0]
    return int(match_gw)


def get_fbref_player_id(player_name: str, team_id: int, season: str) -> int | None:
    """
    Get the player ID from the player name.
    Args:
        player_name (str): The player name.
        team_id (int): The team ID.
        season (str): The season for which to get the player ID. must be in format like '2023-24'
    Returns:
        int: The player ID. If not found, returns None.
    """
    player_name = normalize_name(player_name)

    if season == SEASON:
        df = PLAYERS_IDS_DF
    else:
        df = pd.read_csv(get_data_path(season, "players_ids.csv"))
    team_df = df[df["team"] == team_id]
    if team_df.empty:
        print(f"No player found for {player_name} in {season}.")
        return None

    FUZZY_THRESHOLD = 60  # mess with this

    best_match = process.extractOne(
        player_name,
        team_df["full_name"],
        scorer=fuzz.token_sort_ratio,
        score_cutoff=FUZZY_THRESHOLD,
    )

    if best_match:
        return team_df[team_df["full_name"] == best_match[0]].id.values[0]
    else:
        # try web_name
        best_match = process.extractOne(
            player_name,
            team_df["web_name"],
            scorer=fuzz.token_sort_ratio,
            score_cutoff=FUZZY_THRESHOLD,
        )
        if best_match:
            return team_df[team_df["web_name"] == best_match[0]].id.values[0]
        else:
            # try for all teams (use with caution)
            FUZZY_THRESHOLD = 80  # Make this higher to be more strict
            best_match = process.extractOne(
                player_name,
                df["full_name"],
                scorer=fuzz.token_sort_ratio,
                score_cutoff=FUZZY_THRESHOLD,
            )
            if best_match:
                return df[df["full_name"] == best_match[0]].id.values[0]

    return None


def get_opponent_team_id(
    team_id: int,
    gw: int,
    season: str,
):
    if season == SEASON:
        fixtures = FIXTURES_DF
    else:
        fixtures = pd.read_csv(get_data_path(season, "fixtures.csv"))
    desired_gw = fixtures[fixtures.gw == gw]
    opponent_team_id = desired_gw[
        (desired_gw.home_team_id == team_id) | (desired_gw.away_team_id == team_id)
    ]
    if opponent_team_id.empty:
        return None  # Blank gameweek
    if opponent_team_id.home_team_id.values[0] == team_id:
        return opponent_team_id.away_team_id.values[0]

    return opponent_team_id.home_team_id.values[0]


def team_was_home(
    team_id: int,
    gw: int,
    season: str,
) -> bool:
    if season == SEASON:
        fixtures = FIXTURES_DF
    else:
        fixtures = pd.read_csv(get_data_path(season, "fixtures.csv"))
    desired_gw = fixtures[fixtures.gw == gw]

    desired_gw = desired_gw[
        (desired_gw.home_team_id == team_id) | (desired_gw.away_team_id == team_id)
    ]
    if desired_gw.empty:
        return None  # Blank gameweek
    if desired_gw.home_team_id.values[0] == team_id:
        return True

    return False
