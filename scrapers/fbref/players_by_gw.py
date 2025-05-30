import os
import sys
import time

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Add the project root directory to the system path to import utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from utils.constants import LAST_PLAYED_GAMEWEEK, SEASON  # noqa: E402
from utils.general import get_data_path, time_function  # noqa: E402
from utils.get_ids import (  # noqa: E402
    external_team_name_to_fpl_name,
    get_fbref_player_id,
    get_match_gw,
    get_team_id,
)


def __scrape_individual_match_data(
    url: str, home_team_id: int, away_team_id: int, season: str, stat_type: str
) -> pd.DataFrame:
    hdf = None
    adf = None

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

        data = {
            home_team_id: [],
            away_team_id: [],
        }  # {team_id: [row1, row2, row3, ...]}

        for i, table_div in enumerate(table_divs):
            # get team id
            # Extract Squad ID from div ID (e.g., all_player_stats_19538871 -> 19538871)
            squad_id = table_div.get("id", "").split("_")[-1]
            if stat_type == "passing":
                # get the filter switcher
                filter_switcher_passing = match_report_browser.find_element(
                    By.XPATH, f'//*[@id="all_player_stats_{squad_id}"]/div[7]/div[2]/a'
                )
                # scroll down to the filter switcher
                match_report_browser.execute_script(
                    "arguments[0].scrollIntoView(true);", filter_switcher_passing
                )
                time.sleep(1)  # Allow time for the scroll
                filter_switcher_passing.click()

                time.sleep(2)  # Allow time for the filter to load
                table_sel = match_report_browser.find_element(
                    By.ID, f"stats_{squad_id}_passing"
                )
            elif stat_type == "summary":
                table_sel = match_report_browser.find_element(
                    By.ID, f"stats_{squad_id}_summary"
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
    finally:
        match_report_browser.quit()

    for df in [hdf, adf]:  # i do it now because columns get mixed up after concat
        df.columns = [
            "player_name",
            "player_number",
            "nation",
            "pos",
            "age",
            "minutes",
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
            "player_id",
            "team_id",
            "opponent_team_id",
            "was_home",
        ]

    # merge the two dataframes
    df = pd.concat([hdf, adf])
    df = df[
        ["player_id", "team_id", "opponent_team_id", "was_home"]
        + df.columns.tolist()[:-4]
    ]

    return df


@time_function
def scrape_match_data_players(
    stat_type: str, gw_start: int = 1, gw_end: int = LAST_PLAYED_GAMEWEEK
) -> None:
    """Scrape match data for players

    Args:
        stat_type (str): The type of stat to scrape. One of "passing", "summary" for now
        gw_start (int, optional): The start gameweek to scrape. Defaults to 1.
        gw_end (int, optional): The end gameweek to scrape. Defaults to LAST_PLAYED_GAMEWEEK.
    """

    if stat_type not in ["passing", "summary"]:
        raise ValueError(f"Invalid stat type: {stat_type}")

    season = SEASON

    service = Service()
    options = ChromeOptions()
    options.add_argument("--headless")

    browser = webdriver.Chrome(service=service, options=options)
    browser.implicitly_wait(20)
    try:
        browser.get(
            "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
        )

        wait = WebDriverWait(browser, 20)
        table = wait.until(
            EC.presence_of_element_located((By.ID, "sched_2024-2025_9_1"))
        )

        soup = BeautifulSoup(table.get_attribute("innerHTML"), "lxml")
        rows = soup.find("tbody").find_all("tr")

        gws_data = {}  # gw: pd.DataFrame
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

            gw = get_match_gw(home_id, away_id, season)
            if gw < gw_start or gw > gw_end:
                continue
            print(f"Processing {home_team} vs {away_team} in GW{gw}")

            try:
                # if it says match report then the game is played
                if cells[-2].text.strip() == "Match Report":
                    match_data = __scrape_individual_match_data(
                        f"https://fbref.com{cells[-2].find('a')['href']}",
                        home_id,
                        away_id,
                        season,
                        stat_type,
                    )
                    match_data["gw"] = gw

                    if gw not in gws_data:
                        gws_data[gw] = match_data
                    else:
                        gws_data[gw] = pd.concat([gws_data[gw], match_data])
            except Exception as e:
                with open("error.txt", "a") as f:
                    f.write(f"{home_team} vs {away_team} in GW{gw}\n")
                    f.write(f"Error type: {type(e)}\n")
                    f.write(f"Error message: {e}\n")
                    f.write("\n")

    finally:
        browser.quit()

    for gw, df in gws_data.items():
        if not os.path.exists(get_data_path(season, "fbref", "gws_passing")):
            os.makedirs(get_data_path(season, "fbref", "gws_passing"))
        df.to_csv(
            get_data_path(season, "fbref", "gws_passing", f"gw{gw}.csv"), index=False
        )
        print(f"Saved GW{gw} data")


if __name__ == "__main__":
    scrape_match_data_players(
        gw_start=34, gw_end=LAST_PLAYED_GAMEWEEK, stat_type="passing"
    )
