import os
import sys
import time

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Add the project root directory to the system path to import utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from utils.constants import LAST_PLAYED_GAMEWEEK  # noqa: E402
from utils.general import (  # noqa: E402
    get_data_path,
    time_function,
)
from utils.get_ids import get_player_id  # noqa: E402


@time_function
def scrape_fpl_data_website(season: str):
    service = Service()
    chrome = webdriver.Chrome(service=service)
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

            # Get table column names after Player
            if not column_names:
                for i in range(
                    2, LAST_PLAYED_GAMEWEEK + 3
                ):  # Starting from gw1 column to total column
                    column_name = chrome.find_element(
                        By.XPATH,
                        f"/html/body/div/div/div[4]/div/div[14]/div[2]/div/div[2]/div[2]/table/tbody/tr[1]/th[{i}]/div/span",
                    ).text
                    column_names.append(column_name)
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
                time.sleep(10)  # just to be sure the page is loaded
                wait = WebDriverWait(chrome, 20)
                table_element = wait.until(
                    EC.presence_of_element_located(
                        (
                            By.XPATH,
                            "/html/body/div/div/div[4]/div/div[14]/div[2]/div/div[2]/div[2]/table",
                        )
                    )
                )

                rows = table_element.find_elements(By.TAG_NAME, "tr")

                for i, row in enumerate(rows):
                    if i == 0:
                        continue
                    cells = row.find_elements(By.TAG_NAME, "td")
                    row_data = [
                        cell.text.strip()
                        for col_num, cell in enumerate(cells)
                        if col_num > 0
                    ]

                    player_name = chrome.find_element(
                        By.XPATH,
                        f"/html/body/div/div/div[4]/div/div[14]/div[2]/div/div[2]/div[1]/table/tbody/tr[{i+1}]/td[1]/div",
                    ).text
                    row_data.insert(
                        0, player_name
                    )  # Insert player name at the beginning of the row data

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

            # Merge into gw csv's
            print(f"  Merging {statistic} into gameweek files...")
            for gw_num in range(1, LAST_PLAYED_GAMEWEEK + 1):
                gw_file_path = get_data_path(season, f"gws/gw{gw_num}.csv")
                try:
                    gw_df = pd.read_csv(gw_file_path)

                    # Filter the melted data for the current gameweek
                    current_gw_stat_data = df_melted[df_melted["gw"] == gw_num]

                    # Select only necessary columns for merging
                    stat_to_merge = current_gw_stat_data[["player_id", statistic]]

                    # Perform the merge
                    merged_df = pd.merge(
                        gw_df, stat_to_merge, on="player_id", how="left"
                    )

                    # Overwrite the gameweek file
                    merged_df.to_csv(gw_file_path, index=False)
                    print(f"    Updated {statistic} for GW{gw_num}")
                except FileNotFoundError:
                    print(
                        f"    Warning: Gameweek file not found, skipping GW{gw_num}: {gw_file_path}"
                    )
                except Exception as e:
                    print(
                        f"    Warning: Error processing GW{gw_num} ({gw_file_path}): {e}"
                    )

            print(f"Data for {statistic} collected and merged into all gameweek files.")
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
