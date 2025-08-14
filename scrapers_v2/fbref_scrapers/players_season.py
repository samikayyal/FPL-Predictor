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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from utils.constants import SEASON  # noqa: E402
from utils.general import get_data_path, time_function  # noqa: E402
from utils.get_ids import (  # noqa: E402
    external_team_name_to_fpl_name,
    get_fbref_player_id,
    get_team_id,
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


@time_function  # 300 seconds
def scrape_players_season_data(season: str) -> None:
    if season == SEASON:
        formatted_season = season
    else:
        formatted_season = season.replace("-", "-20")
    service = Service()
    options = ChromeOptions()
    options.add_argument("--headless")

    all_players_data: dict[int, dict[str, Any]] = {}
    for stat_type in STAT_TYPES_URL:
        browser = webdriver.Chrome(service=service, options=options)
        print(f"****** Scraping {stat_type} data for season {formatted_season} ******")
        try:
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

            print(f"\n\n\nFound {len(rows)} rows for {stat_type}")
            # get actual rows
            rows = [row for row in rows if "thead" not in row.get("class", [])]
            print(f"Found {len(rows)} actual rows for {stat_type}\n\n\n")

            for row in rows:
                cells = row.find_all("td")
                if not cells:
                    continue

                if stat_type == "playingtime":
                    games_played_cell = row.find("td", {"data-stat": "games"})
                    # TODO: Theres an issue here where if getting data for new season
                    # everyone will have less than 5 games
                    if games_played_cell and int(games_played_cell.text.strip()) < 5:
                        continue

                player_name_cell = cells[0].find("a")
                team_name_cell = cells[3].find("a")

                if not player_name_cell or not team_name_cell:
                    continue  # Skip header/separator rows that might not have been filtered out

                player_name = player_name_cell.text.strip()
                team_name = team_name_cell.text.strip()

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

                player_id: int | None = get_fbref_player_id(
                    player_name,
                    get_team_id(
                        external_team_name_to_fpl_name(team_name), "name", season
                    ),
                    season,
                )
                if player_id is None:
                    raise ValueError(f"Player ID not found for player {player_name}")

                player_data: dict[str, Any] = {
                    cell.get("data-stat"): cell.text.strip() for cell in cells
                }
                player_data["player_name"] = player_name

                if player_id not in all_players_data:
                    all_players_data[player_id] = player_data
                else:
                    all_players_data[player_id].update(player_data)

        except Exception as e:
            print(f"Error occurred while scraping {stat_type}: {e}")
            print("Error type: ", type(e).__name__)
            break

        finally:

            browser.quit()

    df = pd.DataFrame.from_dict(all_players_data, orient="index")

    # make playerid a column instead of index
    df.index.name = "player_id"
    df = df.reset_index()

    print("df shape:", df.shape)
    os.makedirs(get_data_path(season), exist_ok=True)
    df.to_csv(get_data_path(season, "players_season.csv"), index=False)


if __name__ == "__main__":
    scrape_players_season_data("2024-25")
