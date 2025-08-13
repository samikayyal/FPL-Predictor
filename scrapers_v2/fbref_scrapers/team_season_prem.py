import os
import sys
import time

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.chrome.service import Service

# Add the project root directory to the system path to import utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from utils.constants import (  # noqa: E402
    FBREF_SQUAD_DEFENSE_COLUMNS_NO_DUPLICATES,
    FBREF_SQUAD_GOAL_CREATION_COLUMNS_NO_DUPLICATES,
    FBREF_SQUAD_POSSESSION_COLUMNS_NO_DUPLICATES,
    FBREF_SQUAD_SHOOTING_COLUMNS_NO_DUPLICATES,
    FBREF_SQUAD_STANDARD_COLUMNS_NO_DUPLICATES,
    SEASON,
)
from utils.general import (  # noqa: E402
    get_data_path,
    get_fbref_table_data,
    time_function,
)


@time_function
def scrape_prem_teams_season_data(season: str) -> pd.DataFrame:
    # Get link depending on the season
    season_formatted = season.replace("-", "-20")
    if season == SEASON:
        link: str = "https://fbref.com/en/comps/9/Premier-League-Stats"
    else:
        link: str = (
            f"https://fbref.com/en/comps/9/{season_formatted}/{season_formatted}-Premier-League-Stats"
        )

    service = Service()
    options = ChromeOptions()
    options.add_argument("--headless")

    browser = webdriver.Chrome(service=service, options=options)
    browser.implicitly_wait(20)
    try:
        browser.get(link)
        fbref_data = pd.DataFrame()
        time.sleep(3)  # allow time to load

        soup = BeautifulSoup(browser.page_source, "lxml")

        # ================================== OVERALL TABLE ==================================
        overall_table = soup.find(
            "table", {"id": f"results{season_formatted}91_overall"}
        )
        # Find headers
        thead = overall_table.find("thead")
        headers = [th.text.strip() for th in thead.find_all("th")]
        tbody = overall_table.find("tbody")
        rows = tbody.find_all("tr")

        data = []
        for row in rows:
            league_pos = row.find("th").text.strip()
            cells = row.find_all("td")
            team_name: str = cells[0].find("a").text.strip()

            data.append(
                [league_pos, team_name] + [cell.text.strip() for cell in cells[1:]]
            )

        # Overrite the fbref data df, since its empty at first
        fbref_data = pd.DataFrame(columns=headers, data=data)
        fbref_data.rename(
            columns={
                "Squad": "team",
                "W": "wins",
                "D": "draws",
                "L": "losses",
                "GF": "goals_for",
                "GA": "goals_against",
                "GD": "goal_difference",
                "Pts": "points",
                "Pts/MP": "points_per_match",
            },
            inplace=True,
        )
        fbref_data: pd.DataFrame = fbref_data[  # type: ignore
            [
                "team",
                "wins",
                "draws",
                "losses",
                "goals_for",
                "goals_against",
                "goal_difference",
                "points",
                "points_per_match",
                "xGA",
                "xGD",
                "xGD/90",
            ]
        ]

        # ================================== STANDARD ==================================
        standard_table = get_fbref_table_data(
            soup,
            {"id": "stats_squads_standard_for"},
            # Rename columns because there are 'duplicate' column names
            new_columns=FBREF_SQUAD_STANDARD_COLUMNS_NO_DUPLICATES,
        )
        if not standard_table.empty:
            # What columns to keep - using renamed column names from constants
            standard_table = standard_table[
                [
                    "team",
                    "npxG",
                    "progressive_carries",
                    "progressive_passes",
                    "goals/90",
                    "xG/90",
                    "xAG/90",
                    "xG_plus_xAG/90",
                    "npxG/90",
                    "npxG_plus_xAG/90",
                ]
            ]
            fbref_data = pd.merge(fbref_data, standard_table, on=["team"], how="left")

        # ================================== SHOOTING ==================================
        shooting_table = get_fbref_table_data(
            soup,
            {"id": "stats_squads_shooting_for"},
            new_columns=FBREF_SQUAD_SHOOTING_COLUMNS_NO_DUPLICATES,
        )
        if not shooting_table.empty:
            # What columns to keep - using renamed column names from constants
            shooting_table = shooting_table[
                [
                    "team",
                    "shots",
                    "shots_on_target",
                    "shots_on_target_percentage",
                    "shots/90",
                    "shots_on_target/90",
                    "goals/shots",
                    "goals/shots_on_target",
                    "npxG/shot",
                    "goals_minus_xG",
                ]
            ]
            # Reset index to ensure proper merging
            shooting_data = shooting_table.drop(columns=["team"]).reset_index(drop=True)
            fbref_data = pd.concat(
                [fbref_data.reset_index(drop=True), shooting_data], axis=1
            )
        # ================================== GOAL CREATION ==================================
        goal_creation_table = get_fbref_table_data(
            soup,
            {"id": "stats_squads_gca_for"},
            new_columns=FBREF_SQUAD_GOAL_CREATION_COLUMNS_NO_DUPLICATES,
        )
        if not goal_creation_table.empty:
            # What columns to keep - using renamed column names from constants
            goal_creation_table = goal_creation_table[
                [
                    "team",
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
            ]
            # Reset index to ensure proper merging
            goal_creation_data = goal_creation_table.drop(columns=["team"]).reset_index(
                drop=True
            )
            fbref_data = pd.concat(
                [fbref_data.reset_index(drop=True), goal_creation_data], axis=1
            )
        # ================================== DEFENSE ==================================
        defense_table = get_fbref_table_data(
            soup,
            {"id": "stats_squads_defense_for"},
            new_columns=FBREF_SQUAD_DEFENSE_COLUMNS_NO_DUPLICATES,
        )
        if not defense_table.empty:
            # What columns to keep - using renamed column names from constants
            defense_table = defense_table[
                [
                    "team",
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
            ]
            # Reset index to ensure proper merging
            defense_data = defense_table.drop(columns=["team"]).reset_index(drop=True)
            fbref_data = pd.concat(
                [fbref_data.reset_index(drop=True), defense_data], axis=1
            )
        # ================================== POSSESSION ==================================

        possession_table = get_fbref_table_data(
            soup,
            {"id": "stats_squads_possession_for"},
            new_columns=FBREF_SQUAD_POSSESSION_COLUMNS_NO_DUPLICATES,
        )
        if not possession_table.empty:
            # What columns to keep - using renamed column names from constants
            # Exclude "possession" since it's already in overall table and "90s" may be duplicated
            possession_table = possession_table[
                [
                    "team",
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
                    "carries_into_final_third",
                    "carries_into_penalty_area",
                    "miscontrols",
                    "dispossessed",
                    "passes_received",
                    "progressive_passes_received",
                ]
            ]
            # Reset index to ensure proper merging
            possession_data = possession_table.drop(columns=["team"]).reset_index(
                drop=True
            )
            fbref_data = pd.concat(
                [fbref_data.reset_index(drop=True), possession_data], axis=1
            )

    except Exception as e:
        print(f"Error occurred while scraping team data: {e}")
    finally:
        browser.quit()

    return fbref_data


if __name__ == "__main__":
    season: str = "2024-25"
    df = scrape_prem_teams_season_data(season)
    df.to_csv(get_data_path(season, "teams_season.csv"))
