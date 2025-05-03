from scrapers.FPL_website.gw_data import get_gw_data
from scrapers.FPL_website.season_data import scrape_season_data
from scrapers.Others.fpl_data_website import scrape_fpl_data_website
from utils.constants import SEASON
from utils.general import time_function


@time_function
def main():
    # Scrape season data
    # scrape_season_data(SEASON)

    # Scrape gameweek data
    get_gw_data(SEASON)

    # Scrape FPL data website
    # scrape_fpl_data_website(SEASON)


if __name__ == "__main__":
    main()
