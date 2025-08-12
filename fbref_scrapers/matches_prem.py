import os
import sys
import time
from enum import IntEnum

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Add the project root directory to the system path to import utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils.constants import SEASON  # noqa: E402
from utils.general import get_new_data_path, time_function  # noqa: E402
from utils.get_ids import (  # noqa: E402
    external_team_name_to_fpl_name,
    get_fbref_player_id,
    get_match_gw,
    get_team_id,
)


class FbrefStatType(IntEnum):
    # Value is index in filter switcher
    SUMMARY = 1
    PASSING = 2
    POSSESSION = 5
    DEFENSE = 4


# Column names for fbref match data
FBREF_MATCH_DATA__COLUMNS: dict[FbrefStatType, list[str]] = {
    FbrefStatType.SUMMARY: [
        "goals",
        "assists",
        "penalty_kicks_made",
        "penalty_kicks_attempted",
        "shots",
        "shots_on_target",
        "yellow_cards",
        "red_cards",
        "touches",
        "tackles",
        "interceptions",
        "blocks",
        "xG",
        "npxG",
        "xAG",
        "shot_creating_actions",
        "goal_creating_actions",
        "total_completed_passes",
        "total_attempted_passes",
        "total_pass_completion_percentage",
        "progressive_passes",
        "carries",
        "progressive_carries",
        "attempted_takeons",
        "successful_takeons",
    ],
    FbrefStatType.PASSING: [
        "total_completed_passes",
        "total_attempted_passes",
        "total_pass_completion_percentage",
        "total_passing_distance",
        "progressive_passing_distance",
        "short_completed_passes",
        "short_attempted_passes",
        "short_pass_completion_percentage",
        "medium_completed_passes",
        "medium_attempted_passes",
        "medium_pass_completion_percentage",
        "long_completed_passes",
        "long_attempted_passes",
        "long_pass_completion_percentage",
        "assists",
        "xAG",
        "xA",
        "key_passes",
        "passes_into_final_third",
        "passes_into_penalty_area",
        "crosses_into_penalty_area",
        "progressive_passes",
    ],
    FbrefStatType.POSSESSION: [
        "touches",
        "touches_in_def_pen",
        "touches_in_def_3rd",
        "touches_in_mid_3rd",
        "touches_in_att_3rd",
        "touches_in_att_pen",
        "live_touches",
        "attempted_takeons",
        "successful_takeons",
        "successful_takeon_percent",
        "times_tackled_during_takeon",
        "tackled_during_takeon_percentage",
        "carries",
        "total_carry_distance",
        "progressive_carry_distance",
        "progressive_carries",
        "carries_into_final_third",
        "carries_into_penalty_area",
        "miscontrols",
        "dispossessions",
        "passes_received",
        "progressive_passes_received",
    ],
    FbrefStatType.DEFENSE: [
        "tackles",
        "tackles_won",
        "tackles_def_3rd",
        "tackles_mid_3rd",
        "tackles_att_3rd",
        "dribblers_tackled",
        "dribblers_challenged",
        "percentage_of_dribblers_tackled",
        "challenges_lost",
        "blocks",
        "shots_blocked",
        "passes_blocked",
        "interceptions",
        "players_tackled_plus_interceptions",
        "clearances",
        "errors",
    ],
}


def __scrape_individual_match_data(
    url: str,
    home_team_id: int,
    away_team_id: int,
    season: str,
) -> dict[FbrefStatType, pd.DataFrame]:
    """
    Scrape all stat types for a single match using one browser session.
    Returns a dict mapping FbrefStatType to DataFrame.
    """

    result_dfs = {}

    service = Service()
    options = ChromeOptions()
    options.add_argument("--headless")

    try:
        match_report_browser = webdriver.Chrome(service=service, options=options)
        match_report_browser.get(url)

        # wait for the page to load
        time.sleep(5)

        soup = BeautifulSoup(match_report_browser.page_source, "lxml")
        table_divs = soup.find_all("div", {"class": "switcher_content"})[:2]

        # Get squad IDs from the table divs
        squad_ids = []
        for table_div in table_divs:
            squad_id = table_div.get("id", "").split("_")[-1]
            squad_ids.append(squad_id)

        # Scrape all stat types in this browser session
        for stat_type in [
            FbrefStatType.SUMMARY,
            FbrefStatType.PASSING,
            FbrefStatType.DEFENSE,
            FbrefStatType.POSSESSION,
        ]:
            hdf = None
            adf = None

            data = {
                home_team_id: [],
                away_team_id: [],
            }  # {team_id: [row1, row2, row3, ...]}

            for i, squad_id in enumerate(squad_ids):
                stat_filter_switcher = match_report_browser.find_element(
                    By.XPATH,
                    f'//*[@id="all_player_stats_{squad_id}"]/div[7]/div[{stat_type.value}]/a',
                )
                if stat_type != FbrefStatType.SUMMARY:
                    # scroll down to the filter switcher
                    match_report_browser.execute_script(
                        "arguments[0].scrollIntoView(true);", stat_filter_switcher
                    )
                    time.sleep(1)  # Allow time for the scroll
                    stat_filter_switcher.click()
                    time.sleep(2)  # Allow time for the filter to load

                table_sel = match_report_browser.find_element(
                    By.ID, f"stats_{squad_id}_{stat_type.name.lower()}"
                )

                soup = BeautifulSoup(table_sel.get_attribute("innerHTML"), "lxml")

                # get the column names
                thead = soup.find("thead")
                colnames_header = thead.find_all("tr")[1]
                colnames = [
                    colname.text.strip() for colname in colnames_header.find_all("th")
                ]

                # get the table data
                tbody = soup.find("tbody")
                rows = tbody.find_all("tr")
                for row in rows:
                    cols = row.find_all("td")
                    row_data = [col.text.strip() for col in cols]
                    # Player name is in the first column which is the first td
                    row_data.insert(0, row.find("th").find("a").text.strip())

                    data[home_team_id if i == 0 else away_team_id].append(row_data)

                if i == 0:
                    hdf = pd.DataFrame(data[home_team_id], columns=colnames)
                    hdf["player_id"] = hdf["Player"].apply(
                        get_fbref_player_id, args=(home_team_id, season)
                    )
                    hdf["team_id"] = home_team_id
                    hdf["opponent_team_id"] = away_team_id
                    hdf["was_home"] = True
                else:
                    adf = pd.DataFrame(data[away_team_id], columns=colnames)
                    adf["player_id"] = adf["Player"].apply(
                        get_fbref_player_id, args=(away_team_id, season)
                    )
                    adf["team_id"] = away_team_id
                    adf["opponent_team_id"] = home_team_id
                    adf["was_home"] = False

            # Get stat-specific columns for this stat type
            stat_specific_columns = FBREF_MATCH_DATA__COLUMNS[stat_type]

            for df in [
                hdf,
                adf,
            ]:  # i do it now because columns get mixed up after concat
                df.columns = [
                    "player_name",
                    "player_number",
                    "nation",
                    "pos",
                    "age",
                    "minutes",
                    *stat_specific_columns,
                    "player_id",
                    "team_id",
                    "opponent_team_id",
                    "was_home",
                ]

            # merge the two dataframes
            df = pd.concat([hdf, adf])

            # Reorder
            df = df[
                ["player_id", "team_id", "opponent_team_id", "was_home"]
                + df.columns.tolist()[:-4]
            ]

            result_dfs[stat_type] = df

    finally:
        match_report_browser.quit()

    return result_dfs


@time_function
def scrape_prem_fixtures(season: str, gw_start: int, gw_end: int):
    # Get the link
    if season == SEASON:
        link: str = (
            "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
        )
    else:
        # format season from 2024-25 to 2024-2025
        season_formatted = season.replace("-", "-20")
        link: str = (
            f"https://fbref.com/en/comps/9/{season_formatted}/schedule/{season_formatted}-Premier-League-Scores-and-Fixtures"
        )

    service = Service()
    options = ChromeOptions()
    options.add_argument("--headless")

    browser = webdriver.Chrome(service=service, options=options)
    browser.implicitly_wait(20)
    try:
        browser.get(link)

        wait = WebDriverWait(browser, 20)
        table = wait.until(
            EC.presence_of_element_located((By.ID, "sched_2024-2025_9_1"))
        )

        soup = BeautifulSoup(table.get_attribute("innerHTML"), "lxml")
        rows = soup.find("tbody").find_all("tr")

        gws_data: dict[FbrefStatType, dict[int, pd.DataFrame]] = {}
        # Initialize stat type dictionaries
        for stat_type in [
            FbrefStatType.SUMMARY,
            FbrefStatType.PASSING,
            FbrefStatType.DEFENSE,
            FbrefStatType.POSSESSION,
        ]:
            gws_data[stat_type] = {}

        for row in rows:
            if row.get("class") == ["spacer", "partial_table", "result_all"]:
                print("Skipping spacer row")
                continue
            elif row.get("class") == ["thead"]:
                print("Skipping header row")
                continue

            cells = row.find_all("td")
            home_team = cells[3].text.strip()
            away_team = cells[7].text.strip()

            home_id = get_team_id(
                external_team_name_to_fpl_name(home_team), "name", season
            )
            away_id = get_team_id(
                external_team_name_to_fpl_name(away_team), "name", season
            )
            if home_id is None or away_id is None:
                print(f"Skipping {home_team} vs {away_team} due to missing team IDs")
                continue

            gw = get_match_gw(home_id, away_id, season)
            if gw < gw_start or gw > gw_end:
                continue

            try:
                # if it says match report then the game is played
                if cells[-2].text.strip() == "Match Report":
                    print(
                        f"****Scraping all stat types for {home_team} vs {away_team} in GW{gw}****"
                    )
                    match_data_dict = __scrape_individual_match_data(
                        f"https://fbref.com{cells[-2].find('a')['href']}",
                        home_id,
                        away_id,
                        season,
                    )

                    # Process each stat type from the returned dictionary
                    for stat_type, match_data in match_data_dict.items():
                        match_data["gw"] = gw

                        if gw not in gws_data[stat_type]:
                            gws_data[stat_type][gw] = match_data
                        else:
                            gws_data[stat_type][gw] = pd.concat(
                                [gws_data[stat_type][gw], match_data]
                            )
            except Exception as e:
                with open("error.txt", "a") as f:
                    f.write(f"{home_team} vs {away_team} in GW{gw}\n")
                    f.write(f"Error type: {type(e)}\n")
                    f.write(f"Error message: {e}\n")
                    f.write("\n")
    finally:
        browser.quit()

    # Save the data
    for stat_type, gws in gws_data.items():
        for gw, df in gws.items():
            os.makedirs(
                get_new_data_path(season, f"gws_{stat_type.name.lower()}"),
                exist_ok=True,
            )
            df.to_csv(
                get_new_data_path(
                    season,
                    f"gws_{stat_type.name.lower()}",
                    f"gw{gw}.csv",
                ),
                index=False,
            )
            print(f"Saved GW{gw} data for {stat_type.name}")


if __name__ == "__main__":
    scrape_prem_fixtures("2024-25", 1, 1)
