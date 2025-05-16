import os
import sys
import time

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Add the project root directory to the system path to import utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from utils.constants import LAST_PLAYED_GAMEWEEK, SEASON  # noqa: E402
from utils.general import (  # noqa: E402
    get_data_path,
    time_function,
)
from utils.get_ids import get_player_id  # noqa: E402


@time_function  # 4577 seconds
def scrape_fpl_data_website(season: str):
    service = Service()
    options = ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode (no GUI)
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    chrome = webdriver.Chrome(service=service, options=options)
    chrome.implicitly_wait(20)
    chrome.get("https://www.fpl-data.co.uk/statistics")

    column_names: list[str] = []

    try:
        # in gameweek by gameweek data section

        # Wait for the page to load
        time.sleep(7)
        for statistic in ["shots", "SoT", "SiB", "tpa", "npxG", "cc", "npxgi", "npg"]:
            print(f"\nGetting {statistic} data...")
            # clear statistic
            clear_button = chrome.find_element(
                By.XPATH, "/html/body/div/div/div[4]/div/div[13]/div/div/span[1]/span"
            )
            clear_button.click()

            # Statistic name input
            statistic_name = chrome.find_element(
                By.XPATH,
                "/html/body/div/div/div[4]/div/div[13]/div/div/div/div[2]/input",
            )
            statistic_name.send_keys(statistic)
            statistic_name.send_keys(Keys.ENTER)

            time.sleep(5)  # Wait for the stat table to load

            table = chrome.find_element(
                By.XPATH,
                "/html/body/div/div/div[4]/div/div[14]/div[2]/div/div[2]/div[2]/table",
            )

            soup = BeautifulSoup(table.get_attribute("innerHTML"), "lxml")
            rows = soup.find_all("tr")

            # Get table column names after Player
            column_names = soup.find_all("span", class_="column-header-name")
            column_names = [name.text.strip() for name in column_names]
            if "Total" in column_names:
                column_names.remove("Total")

            data = []
            page_num = 1
            max_pages = int(
                chrome.find_element(
                    By.XPATH, "/html/body/div/div/div[4]/div/div[14]/div[3]/div/div[2]"
                ).text
            )
            while page_num < max_pages:
                page_start_time = time.time()
                print(f"Page {page_num} of {max_pages}", end=" - ")
                time.sleep(8)  # just to be sure the page is loaded
                wait = WebDriverWait(chrome, 20)
                table_element = wait.until(
                    EC.presence_of_element_located(
                        (
                            By.XPATH,
                            "/html/body/div/div/div[4]/div/div[14]/div[2]/div/div[2]/div[2]/table",
                        )
                    )
                )

                soup = BeautifulSoup(table_element.get_attribute("innerHTML"), "lxml")
                rows = soup.find_all("tr")

                for row in rows[1:]:
                    cells = row.find_all("td")
                    row_data = [
                        cell.text.strip() for cell in cells[:-1]
                    ]  # Don't include total column
                    data.append(row_data)

                # next page button
                next_button = chrome.find_element(
                    By.XPATH,
                    "/html/body/div/div/div[4]/div/div[14]/div[3]/button[3]",
                )
                next_button.click()
                page_num += 1
                page_end_time = time.time()

                print(f"Took {page_end_time - page_start_time:.2f} seconds")

            df = pd.DataFrame(data, columns=["web_name"] + column_names)
            if "Total" in df.columns:
                df.drop(columns=["Total"], inplace=True)

            gw_columns = df.columns.drop(["web_name"]).to_list()
            df_melted = pd.melt(
                df,
                id_vars=["web_name"],
                value_vars=gw_columns,
                var_name="gw",
                value_name=statistic,
            )
            df_melted["gw"] = df_melted["gw"].str[2:].astype(int)
            # Calculate player_id for the entire melted dataframe once per statistic
            df_melted["player_id"] = df_melted["web_name"].apply(
                get_player_id, args=("web_name", season)
            )

            if df_melted.duplicated(subset=["player_id", "gw"]).any():
                print(
                    f"Warning: Duplicated entries found in {statistic} data for player_id and gw."
                )
                df_melted.drop_duplicates(
                    subset=["player_id", "gw"], inplace=True, keep="first"
                )

            if not os.path.exists(get_data_path(season, "stat_data")):
                os.makedirs(get_data_path(season, "stat_data"), exist_ok=True)
            df_melted.to_csv(
                get_data_path(season, f"stat_data/{statistic}.csv"), index=False
            )

            # Go back to the first page in the table
            back_button = chrome.find_element(
                By.XPATH,
                "/html/body/div/div/div[4]/div/div[14]/div[3]/button[1]",
            )
            back_button.click()
            time.sleep(10)

    except Exception as e:
        raise e

    finally:
        chrome.quit()


if __name__ == "__main__":
    scrape_fpl_data_website(SEASON)
