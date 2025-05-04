import os
import sys
import time

import pandas as pd
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

# Add the project root directory to the system path to import utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
from utils.general import get_data_path, time_function  # noqa: E402


@time_function
def scrape_team_season_data() -> None:
    try:
        service = Service()
        options = ChromeOptions()
        options.add_argument("--headless")  # Run in headless mode (no GUI)
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        browser = webdriver.Chrome(service=service, options=options)
        browser.implicitly_wait(20)

        browser.get("https://fbref.com/en/comps/9/Premier-League-Stats")

        # Scroll down a bit to avoid cookies banner
        browser.execute_script("window.scrollTo(0, 200);")

        home_away_toggle = browser.find_elements(By.CLASS_NAME, "sr_preset")[
            1
        ]  # second element is the home/away toggle
        home_away_toggle.click()

        # Wait for the page to load
        time.sleep(2)

        header_row = browser.find_element(
            By.XPATH, '//*[@id="results2024-202591_home_away"]/thead/tr[2]'
        )
        colnames = [
            th.text
            for th in header_row.find_elements(By.TAG_NAME, "th")
            if th.text != ""
        ]
        print(colnames)

        tbody = browser.find_element(
            By.XPATH, '//*[@id="results2024-202591_home_away"]/tbody'
        )

        rows = tbody.find_elements(By.TAG_NAME, "tr")
        print(f"Number of rows: {len(rows)}")
        data = []
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) > 0:
                data.append([col.text for col in cols if col.text])

        print(data)
        df = pd.DataFrame(data[1:], columns=colnames[1:])
        df.columns = [
            "Squad",
            "home_matches_played",
            "home_wins",
            "home_draws",
            "home_losses",
            "home_goals_scored",
            "home_goals_conceded",
            "home_goal_difference",
            "home_pts",
            "home_pts/mp",
            "home_xG",
            "home_xGA",
            "home_xGD",
            "home_xGD/90",
            "away_matches_played",
            "away_wins",
            "away_draws",
            "away_losses",
            "away_goals_scored",
            "away_goals_conceded",
            "away_goal_difference",
            "away_pts",
            "away_pts/mp",
            "away_xG",
            "away_xGA",
            "away_xGD",
            "away_xGD/90",
        ]
        df.to_csv(
            get_data_path("2024-25", "home_away_team_season_data.csv"), index=False
        )

    except Exception as e:
        raise e

    finally:
        browser.quit()


if __name__ == "__main__":
    scrape_team_season_data()
