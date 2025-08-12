import pandas as pd
import requests


def __load_manager_ids() -> list[int]:
    res = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
    if res.status_code == 200:
        data = res.json()
        df = pd.DataFrame(data["elements"])
        df = df[(df.mng_win > 0) | (df.mng_draw > 0) | (df.mng_loss > 0)]

        return df["id"].tolist()
    else:
        return []


SEASON: str = "2025-26"
LAST_SEASON: str = "2024-25"
LAST_PLAYED_GAMEWEEK: int = 38

# If the manager IDs are already loaded, use them; otherwise, load them from the API.
# Only for 2024-25
__MANAGER_PLAYER_IDS: list[int] = __load_manager_ids() if SEASON == "2024-25" else []
MANAGER_PLAYER_IDS = (
    (__MANAGER_PLAYER_IDS if __MANAGER_PLAYER_IDS else __load_manager_ids())
    if SEASON == "2024-25"
    else []
)


# column names for fbref season team tables to avoid duplicate names
# Since some are per 90 but are named the same as the normal one
FBREF_SQUAD_STANDARD_COLUMNS_NO_DUPLICATES = [
    "team",
    "number_of_players",
    "age",
    "possession",
    "matches_played",
    "starts",
    "minutes",
    "90s",
    "goals",
    "assists",
    "goals_plus_assists",
    "non_penalty_goals",
    "penalties_scored",
    "penalties_attempted",
    "yellow_cards",
    "red_cards",
    "xG",
    "npxG",
    "xAG",
    "npxG_plus_xAG",
    "progressive_carries",
    "progressive_passes",
    "goals/90",
    "assists/90",
    "goals_plus_assists/90",
    "non_penalty_goals/90",
    "goals_plus_assists_minus_penalties_scored/90",
    "xG/90",
    "xAG/90",
    "xG_plus_xAG/90",
    "npxG/90",
    "npxG_plus_xAG/90",
]

FBREF_SQUAD_SHOOTING_COLUMNS_NO_DUPLICATES = [
    "team",
    "number_of_players",
    "90s",
    "goals",
    "shots",
    "shots_on_target",
    "shots_on_target_percentage",
    "shots/90",
    "shots_on_target/90",
    "goals/shots",
    "goals/shots_on_target",
    "avg_shot_distance",
    "freekick_shots",
    "penalties_scored",
    "penalties_attempted",
    "xG",
    "npxG",
    "npxG/shot",
    "goals_minus_xG",
    "non_penalty_goals_minus_xG",
]

FBREF_SQUAD_GOAL_CREATION_COLUMNS_NO_DUPLICATES = [
    "team",
    "number_of_players",
    "90s",
    "shot_creating_actions",
    "shot_creating_actions/90",
    "passes_led_to_shot",
    "passes_dead_led_to_shot",
    "takeons_led_to_shot",
    "shots_led_to_shot",
    "fouls_drawn_led_to_shot",
    "defensive_actions_led_to_shot",
    "goal_creating_actions",
    "goal_creating_actions/90",
    "passes_led_to_goal",
    "passes_dead_led_to_goal",
    "takeons_led_to_goal",
    "shots_led_to_goal",
    "fouls_drawn_led_to_goal",
    "defensive_actions_led_to_goal",
]

FBREF_SQUAD_DEFENSE_COLUMNS_NO_DUPLICATES = [
    "team",
    "number_of_players",
    "90s",
    "players_tackled",
    "tackles_won",
    "tackles_in_defensive_3rd",
    "tackles_in_mid_3rd",
    "tackles_in_attacking_3rd",
    "dribblers_tackled",
    "dribblers_tackles_attempted",
    "dribblers_tackles_percentage",
    "dribblers_tackles_lost",
    "blocks",
    "shots_blocked",
    "passes_blocked",
    "interceptions",
    "tackles_plus_interceptions",
    "clearances",
    "errors",
]


FBREF_SQUAD_POSSESSION_COLUMNS_NO_DUPLICATES = [
    "team",
    "number_of_players",
    "possession",
    "90s",
    "touches",
    "touches_in_defensive_penalty_area",
    "touches_in_defensive_third",
    "touches_in_midfield_third",
    "touches_in_attacking_third",
    "touches_in_attacking_penalty_area",
    "live",
    "takeons_attempted",
    "takeons_successful",
    "takeons_successful_percentage",
    "tackled_during_takeon",
    "tackled_during_takeon_percentage",
    "carries",
    "total_distance_carried",
    "progressive_carry_distance",
    "progressive_carries",
    "carries_into_final_third",
    "carries_into_penalty_area",
    "miscontrols",
    "dispossessed",
    "passes_received",
    "progressive_passes_received",
]
