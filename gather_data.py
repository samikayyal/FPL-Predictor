import threading

from scrapers.fbref.players_by_gw import scrape_key_passes_data
from scrapers.fbref.teams_season_data import scrape_team_season_data
from scrapers.fpl_data_website.gw_data import scrape_fpl_data_website
from scrapers.fpl_data_website.team_gw_data import scrape_team_gw_data
from scrapers.official_fpl_website.fixtures import scrape_fixtures
from scrapers.official_fpl_website.gw_data import get_gw_data
from scrapers.official_fpl_website.season_data import scrape_season_data
from utils.constants import SEASON
from utils.general import time_function


@time_function
def main():
    scraping_funcs = [
        lambda: scrape_fixtures(),
        lambda: get_gw_data(SEASON),
        lambda: scrape_season_data(SEASON),
        lambda: scrape_key_passes_data(season=SEASON),
        lambda: scrape_team_season_data(),
        lambda: scrape_fpl_data_website(SEASON),
        lambda: scrape_team_gw_data(SEASON),
    ]

    threads = []
    for func in scraping_funcs:
        thread = threading.Thread(target=func)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()


# NOTE NOT TESTED YET
if __name__ == "__main__":
    main()
