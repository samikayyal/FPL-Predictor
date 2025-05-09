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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)


from utils.constants import LAST_PLAYED_GAMEWEEK, SEASON  # noqa: E402
from utils.general import get_data_path, time_function  # noqa: E402


@time_function
def scrape_team_gw_data(
    start_gw: int = 1, end_gw: int = LAST_PLAYED_GAMEWEEK, season=SEASON
) -> None:
    try:
        if start_gw < 1 or end_gw > 38 or start_gw > end_gw:
            raise ValueError("Invalid gameweek range. Must be between 1 and 38.")

        # Create a list of gameweeks to scrape
        gameweeks = list(range(start_gw, end_gw + 1))

        # Initialize the Chrome WebDriver
        service = Service()
        options = ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        browser = webdriver.Chrome(service=service, options=options)

        browser.get("https://www.fpl-data.co.uk/statistics")

        # Team radio button
        wait = WebDriverWait(browser, 20)
        team_radio = wait.until(
            EC.presence_of_element_located(
                (By.XPATH, "/html/body/div/div/div[4]/div/div[1]/label[3]/input")
            )
        )
        team_radio.click()

        time.sleep(2)  # Wait for team data to load

        # Split Gameweeks button
        split_gameweeks_button = browser.find_element(
            By.XPATH, "/html/body/div/div/div[4]/div/div[2]/div/label[2]/input"
        )
        split_gameweeks_button.click()

        time.sleep(3)  # Wait for the split gameweeks data to load

        # Max Gameweek number
        max_gw_num = browser.find_element(
            By.XPATH, "/html/body/div/div/div[4]/div/div[8]/div[3]/div/div[2]"
        ).text

        if int(max_gw_num) < end_gw:
            raise ValueError(
                f"Data for gameweek {end_gw} is not available. Maximum gameweek is {max_gw_num}."
            )

        # Stat types to show input
        clear_stats = browser.find_element(
            By.XPATH, "/html/body/div/div/div[4]/div/div[7]/div/div/span[1]/span"
        )
        clear_stats.click()

        for stat_type in [
            "info",
            "shooting",
            "assisting",
            "defending",
            "points",
            "possession",
        ]:
            stat_input = browser.find_element(By.ID, "input-stat-types").find_element(
                By.TAG_NAME, "input"
            )
            stat_input.send_keys(stat_type)
            stat_input.send_keys(Keys.ENTER)
            time.sleep(1)

        time.sleep(2)  # Wait for the stat table to load

        # Sort by gameweek (ascending)
        sort_by_gw = browser.find_element(
            By.XPATH,
            '//*[@id="stats-data-table"]/div[2]/div/div[2]/div[2]/table/tbody/tr[2]/th[4]/div/div/span',
        )
        sort_by_gw.click()
        sort_by_gw.click()  # Click again to sort in ascending order
        time.sleep(2)  # Wait for the table to sort

        # Get column names
        column_names = []
        soup = BeautifulSoup(browser.page_source, "lxml")
        stats_table_div = soup.find("div", {"id": "stats-data-table"})
        header_spans = stats_table_div.find_all("span", {"class": "column-header-name"})
        for span in header_spans:
            name = span.get_text(strip=True)
            if name:
                # Check if the parent <th> has colspan > 1 (these are group headers like 'Info', 'Shooting')
                # We want to skip those group headers.
                parent_th = span.find_parent("th")
                if not parent_th or parent_th.get("colspan", "1") == "1":
                    column_names.append(name)

        column_names.insert(0, "Team")
        print("Column names:", column_names)

        # Scroll through gameweeks
        # I couldnt get the gw entry to work so I just scroll through the gameweeks
        # starts at 1 so
        if start_gw > 1:
            for _ in range(start_gw - 1):
                next_button = wait.until(
                    EC.element_to_be_clickable(
                        (
                            By.XPATH,
                            "//div[@id='stats-data-table']//button[@class='next-page']",
                        )
                    )
                )
                next_button.click()
                time.sleep(2)
                print("Scrolling to next gameweek...")

        for gw_num in gameweeks:
            data = []
            print(f"Getting data for gameweek {gw_num}...")

            # Extract the data
            stats_table_div = browser.find_element(By.ID, "stats-data-table")
            soup = BeautifulSoup(stats_table_div.get_attribute("innerHTML"), "lxml")
            rows = soup.find_all("tr")

            for row in rows[2:]:
                cells = row.find_all("td")
                row_data = [cell.text.strip() for cell in cells]
                if len(cells) == len(column_names) and row_data not in data:
                    data.append(row_data)

            # Convert the data to a DataFrame
            df = pd.DataFrame(data, columns=column_names)
            df.to_csv(get_data_path(season, f"team_gws/gw{gw_num}.csv"), index=False)

            if gw_num < end_gw:
                next_button = wait.until(
                    EC.element_to_be_clickable(
                        (
                            By.XPATH,
                            "//div[@id='stats-data-table']//button[@class='next-page']",
                        )
                    )
                )
                next_button.click()
                time.sleep(4)

    finally:
        browser.quit()


if __name__ == "__main__":
    scrape_team_gw_data(1, 35)
