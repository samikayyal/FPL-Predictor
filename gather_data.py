"""
FPL Data Gathering Pipeline

This module provides a comprehensive pipeline for gathering Fantasy Premier League (FPL) data
from multiple sources including:

1. Official FPL Website:
   - Player gameweek data
   - Season data
   - Fixtures data

2. FPL Data Website:
   - Team gameweek data
   - Additional gameweek data

3. FBref:
   - Team season data
   - Player match data (passing and summary stats)

4. Baseline BPS Calculation:
   - Calculates baseline BPS for all players

Usage:
    python gather_data.py                          # Run full pipeline with defaults
    python gather_data.py --season 2023-24        # Specify different season
    python gather_data.py --start-gw 10 --end-gw 20  # Specify gameweek range
    python gather_data.py --skip-scrapers          # Skip all data scraping steps

Requirements:
    - All packages from requirements.txt must be installed
    - Chrome browser for Selenium-based scrapers
    - Internet connection for API access
"""

import sys
import os
from typing import Optional

# Add the project root directory to the system path to import modules
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

# Import scrapers from official FPL website
from scrapers.official_fpl_website.gw_data import (
    scrape_gw_data_official_fpl_website,
)  # noqa: E402
from scrapers.official_fpl_website.season_data import scrape_season_data  # noqa: E402
from scrapers.official_fpl_website.fixtures import scrape_fixtures  # noqa: E402

# Import scrapers from FPL data website
from scrapers.fpl_data_website.gw_data import scrape_fpl_data_website  # noqa: E402
from scrapers.fpl_data_website.team_gw_data import scrape_team_gw_data  # noqa: E402

# Import scrapers from FBref
from scrapers.fbref.teams_season_data import scrape_team_season_data  # noqa: E402
from scrapers.fbref.players_by_gw import scrape_match_data_players  # noqa: E402

# Import baseline BPS calculation
from baseline_bps import calculate_baseline_bps  # noqa: E402

# Import constants
from utils.constants import SEASON, LAST_PLAYED_GAMEWEEK  # noqa: E402


def run_all_scrapers_and_calculations(
    season: str = SEASON, start_gw: int = 1, end_gw: int = LAST_PLAYED_GAMEWEEK
) -> None:
    """
    Run all scrapers and data processing steps.
    """
    print("\n" + "=" * 60)
    print("RUNNING ALL SCRAPERS AND CALCULATIONS")
    print("=" * 60)

    # Official FPL Website scrapers
    print("\n1. Scraping GW data from Official FPL Website...")
    try:
        scrape_gw_data_official_fpl_website(season)
        print("✓ Successfully scraped GW data from Official FPL Website")
    except Exception as e:
        print(f"✗ Error scraping GW data from Official FPL Website: {e}")
        # Decide if this should be a critical error that stops the pipeline
        # For now, we'll print the error and continue

    print("\n2. Scraping data from FPL Data Website...")
    try:
        scrape_fpl_data_website(season)
        print("✓ Successfully scraped data from FPL Data Website")
    except Exception as e:
        print(f"✗ Error scraping data from FPL Data Website: {e}")

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

    # Baseline BPS Calculation
    print("\n9. Calculating Baseline BPS...")
    try:
        calculate_baseline_bps(season)
        print(
            "✓ Successfully calculated Baseline BPS"
        )  # Message updated in baseline_bps.py
    except Exception as e:
        print(f"✗ Error calculating Baseline BPS: {e}")


def main(
    season: Optional[str] = None,
    start_gw: int = 1,
    end_gw: Optional[int] = None,
    skip_scrapers: bool = False,
) -> None:
    """
    Main function to run the complete data gathering pipeline.

    Args:
        season: Season to scrape data for (defaults to current season from constants)
        start_gw: Starting gameweek for range-based scrapers
        end_gw: Ending gameweek for range-based scrapers (defaults to last played GW)
        skip_scrapers: Skip all scraping and calculation steps
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
        if not skip_scrapers:
            run_all_scrapers_and_calculations(season, start_gw, end_gw)
        else:
            print("\nSkipping all scrapers and calculations...")

        print("\n" + "=" * 60)
        print("DATA GATHERING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        print("=" * 60)
        # raise # Optionally re-raise the exception if you want the script to exit with an error code


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
        "--skip-scrapers",
        action="store_true",
        help="Skip all scrapers and calculations",
    )

    args = parser.parse_args()

    main(
        season=args.season,
        start_gw=args.start_gw,
        end_gw=args.end_gw,
        skip_scrapers=args.skip_scrapers,
    )
