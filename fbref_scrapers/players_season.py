import os
import sys
import time

import pandas as pd
from bs4 import BeautifulSoup
from pyparsing import Any
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
from utils.general import get_data_path, time_function  # noqa: E402
from utils.get_ids import (  # noqa: E402
    get_fbref_player_id,
)

STAT_TYPES_URL: list[str] = [
    "gca",
    "stats",
    "keepers",
    "keepersadv",
    "shooting",
    "passing",
    "passing_types",
    "defense",
    "possession",
    "playingtime",
    "misc",
]


@time_function
def scrape_players_season_data(season: str) -> None:
    if season == SEASON:
        formatted_season = season
    else:
        formatted_season = season.replace("-", "-20")
    service = Service()
    options = ChromeOptions()
    options.add_argument("--headless")
    all_data = []
    for stat_type in STAT_TYPES_URL:
        print(f"****** Scraping {stat_type} data for season {formatted_season} ******")
        try:
            browser = webdriver.Chrome(service=service, options=options)
            # ============== goal creating actions ==================
            browser.get(
                f"https://fbref.com/en/comps/9/{formatted_season}/{stat_type}/{formatted_season}-Premier-League-Stats"
            )

            # wait for the page to load
            time.sleep(5)

            # Close any potential overlays or popups
            try:
                # Look for common popup/overlay elements and close them
                close_buttons = browser.find_elements(
                    By.CSS_SELECTOR, "[class*='close'], [class*='dismiss']"
                )
                for button in close_buttons:
                    if button.is_displayed():
                        button.click()
                        time.sleep(1)
            except Exception:
                pass  # Ignore if no overlays found

            # passing types has no checkbox
            if stat_type != "passing_types":
                checkbox_id = f"fs_check_stats_{stat_type}"
                if stat_type == "stats":
                    checkbox_id: str = "fs_check_stats_standard"
                elif stat_type == "keepers":
                    checkbox_id: str = "fs_check_stats_keeper"
                elif stat_type == "keepersadv":
                    checkbox_id: str = "fs_check_stats_keeper_adv"
                elif stat_type == "playingtime":
                    checkbox_id: str = "fs_check_stats_playing_time"

                hide_players_checkbox = WebDriverWait(browser, 10).until(
                    EC.element_to_be_clickable((By.ID, checkbox_id))
                )

                # scroll to the checkbox with more aggressive scrolling
                browser.execute_script(
                    "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center', inline: 'center'});",
                    hide_players_checkbox,
                )

                # Wait a bit more for scrolling to complete
                time.sleep(2)

                # Additional scroll up a bit to ensure it's not at the very bottom of viewport
                browser.execute_script("window.scrollBy(0, -100);")
                time.sleep(1)

                # Try to click using JavaScript if regular click fails
                try:
                    hide_players_checkbox.click()  # uncheck it to reveal all players
                except Exception as click_error:
                    print(
                        f"Regular click failed for {stat_type}, trying JavaScript click: {click_error}"
                    )
                    browser.execute_script(
                        "arguments[0].click();", hide_players_checkbox
                    )

                time.sleep(1)  # wait for the players to be revealed

            soup = BeautifulSoup(browser.page_source, "lxml")

            table_div_id = f"all_stats_{stat_type}"
            if stat_type == "stats":
                table_div_id = "all_stats_standard"
            elif stat_type == "keepers":
                table_div_id = "all_stats_keeper"
            elif stat_type == "keepersadv":
                table_div_id = "all_stats_keeper_adv"
            elif stat_type == "playingtime":
                table_div_id = "all_stats_playing_time"

            table_div = soup.find("div", {"id": table_div_id})
            table = table_div.find("table")
            rows = table.find_all("tr")

            # get actual rows
            rows = [
                row
                for row in rows
                if row.get("class") and "thead" not in row.get("class")
            ]

            for row in rows:
                cells = row.find_all("td")
                if not cells:
                    continue

                cells = [
                    cell
                    for cell in cells
                    if cell.get("data-stat")
                    not in [
                        "player",
                        "nationality",
                        "position",
                        "team",
                        "age",
                        "birth_year",
                        "matches",
                    ]
                ]
                player_name: str = cells[0].find("a").text.strip()

                player_id: int | None = get_fbref_player_id(cells[0])
                if player_id is None:
                    raise ValueError(f"Player ID not found for player {player_name}")

                player_data: dict[str, Any] = {
                    cell.get("data-stat"): cell.text.strip() for cell in cells
                }
                player_data["player_name"] = player_name
                player_data["player_id"] = player_id
                all_data.append(player_data)

                df = pd.DataFrame(all_data)
                df.to_csv(get_data_path(season, "players_season.csv"), index=False)
        except Exception as e:
            print(f"Error occurred while scraping {stat_type}: {e}")
            print("Error type: ", type(e).__name__)
            break

        finally:
            browser.quit()


if __name__ == "__main__":
    scrape_players_season_data("2024-25")
