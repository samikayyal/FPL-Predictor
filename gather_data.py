"""
FPL Data Gathering Pipeline

This module provides a comprehensive pipeline for gathering Fantasy Premier League (FPL) data
from multiple sources including:

1. Official FPL Website (priority scrapers - must run in order):
   - Player gameweek data
   - Season data
   - Fixtures data

2. FPL Data Website (independent scrapers):
   - Team gameweek data
   - Additional gameweek data

3. FBref (independent scrapers):
   - Team season data
   - Player match data (passing and summary stats)

Usage:
    python gather_data.py                          # Run full pipeline with defaults
    python gather_data.py --season 2023-24        # Specify different season
    python gather_data.py --start-gw 10 --end-gw 20  # Specify gameweek range
    python gather_data.py --skip-priority          # Skip priority scrapers
    python gather_data.py --skip-independent       # Skip independent scrapers

Requirements:
    - All packages from requirements.txt must be installed
    - Chrome browser for Selenium-based scrapers
    - Internet connection for API access
"""

import os
import sys
from typing import Optional

from utils.general import time_function

# Add the project root directory to the system path to import modules
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

# Import scrapers from official FPL website
from scrapers.fbref.players_by_gw import scrape_match_data_players  # noqa: E402

# Import scrapers from FBref
from scrapers.fbref.teams_season_data import scrape_team_season_data  # noqa: E402

# Import scrapers from FPL data website
from scrapers.fpl_data_website.gw_data import scrape_fpl_data_website  # noqa: E402
from scrapers.fpl_data_website.team_gw_data import scrape_team_gw_data  # noqa: E402
from scrapers.official_fpl_website.fixtures import scrape_fixtures  # noqa: E402
from scrapers.official_fpl_website.gw_data import (
    scrape_gw_data_official_fpl_website,  # noqa: E402
)
from scrapers.official_fpl_website.season_data import scrape_season_data  # noqa: E402

# Import constants
from utils.constants import LAST_PLAYED_GAMEWEEK, SEASON  # noqa: E402


def run_priority_scrapers(season: str = SEASON) -> None:
    """
    Run scrapers that need to be executed in order (dependencies).
    According to README.md:
    1. scrape_gw_data_official_fpl_website
    2. scrape_fpl_data_website
    """
    print("=" * 60)
    print("RUNNING PRIORITY SCRAPERS (Sequential Order)")
    print("=" * 60)

    print("\n1. Scraping GW data from Official FPL Website...")
    try:
        scrape_gw_data_official_fpl_website(season)
        print("✓ Successfully scraped GW data from Official FPL Website")
    except Exception as e:
        print(f"✗ Error scraping GW data from Official FPL Website: {e}")
        raise

    print("\n2. Scraping data from FPL Data Website...")
    try:
        scrape_fpl_data_website(season)
        print("✓ Successfully scraped data from FPL Data Website")
    except Exception as e:
        print(f"✗ Error scraping data from FPL Data Website: {e}")
        raise


def run_independent_scrapers(
    season: str = SEASON, start_gw: int = 1, end_gw: int = LAST_PLAYED_GAMEWEEK
) -> None:
    """
    Run scrapers that can be executed in any order (no dependencies).
    """
    print("\n" + "=" * 60)
    print("RUNNING INDEPENDENT SCRAPERS (Any Order)")
    print("=" * 60)

    # Official FPL Website scrapers
    print("\n3. Scraping fixtures from Official FPL Website...")
    try:
        scrape_fixtures()
        print("✓ Successfully scraped fixtures")
    except Exception as e:
        print(f"✗ Error scraping fixtures: {e}")

    print("\n4. Scraping season data from Official FPL Website...")
    try:
        scrape_season_data(season)
        print("✓ Successfully scraped season data")
    except Exception as e:
        print(f"✗ Error scraping season data: {e}")

    # FPL Data Website scrapers
    print("\n5. Scraping team GW data from FPL Data Website...")
    try:
        scrape_team_gw_data(start_gw, end_gw, season)
        print("✓ Successfully scraped team GW data")
    except Exception as e:
        print(f"✗ Error scraping team GW data: {e}")

    # FBref scrapers
    print("\n6. Scraping team season data from FBref...")
    try:
        scrape_team_season_data()
        print("✓ Successfully scraped team season data")
    except Exception as e:
        print(f"✗ Error scraping team season data: {e}")

    print("\n7. Scraping player passing data by GW from FBref...")
    try:
        scrape_match_data_players("passing", start_gw, end_gw)
        print("✓ Successfully scraped player passing data")
    except Exception as e:
        print(f"✗ Error scraping player passing data: {e}")

    print("\n8. Scraping player summary data by GW from FBref...")
    try:
        scrape_match_data_players("summary", start_gw, end_gw)
        print("✓ Successfully scraped player summary data")
    except Exception as e:
        print(f"✗ Error scraping player summary data: {e}")


@time_function
def main(
    season: Optional[str] = None,
    start_gw: int = 1,
    end_gw: Optional[int] = None,
    skip_priority: bool = False,
    skip_independent: bool = False,
) -> None:
    """
    Main function to run the complete data gathering pipeline.

    Args:
        season: Season to scrape data for (defaults to current season from constants)
        start_gw: Starting gameweek for range-based scrapers
        end_gw: Ending gameweek for range-based scrapers (defaults to last played GW)
        skip_priority: Skip priority scrapers (useful if they've already been run)
        skip_independent: Skip independent scrapers
    """
    # Set defaults
    if season is None:
        season = SEASON
    if end_gw is None:
        end_gw = LAST_PLAYED_GAMEWEEK

    print("FPL DATA GATHERING PIPELINE")
    print("=" * 60)
    print(f"Season: {season}")
    print(f"Gameweek Range: {start_gw} - {end_gw}")
    print("=" * 60)

    try:
        # Run priority scrapers (must be run in order)
        if not skip_priority:
            run_priority_scrapers(season)
        else:
            print("\nSkipping priority scrapers...")

        # Run independent scrapers (can be run in any order)
        if not skip_independent:
            run_independent_scrapers(season, start_gw, end_gw)
        else:
            print("\nSkipping independent scrapers...")

        print("\n" + "=" * 60)
        print("DATA GATHERING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FPL Data Gathering Pipeline")
    parser.add_argument(
        "--season", type=str, help=f"Season to scrape (default: {SEASON})"
    )
    parser.add_argument(
        "--start-gw", type=int, default=1, help="Starting gameweek (default: 1)"
    )
    parser.add_argument(
        "--end-gw", type=int, help=f"Ending gameweek (default: {LAST_PLAYED_GAMEWEEK})"
    )
    parser.add_argument(
        "--skip-priority", action="store_true", help="Skip priority scrapers"
    )
    parser.add_argument(
        "--skip-independent", action="store_true", help="Skip independent scrapers"
    )

    args = parser.parse_args()

    main(
        season=args.season,
        start_gw=args.start_gw,
        end_gw=args.end_gw,
        skip_priority=args.skip_priority,
        skip_independent=args.skip_independent,
    )
