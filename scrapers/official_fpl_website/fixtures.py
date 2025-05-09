import os
import sys

import pandas as pd
import requests

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from utils.constants import SEASON  # noqa: E402
from utils.general import get_data_path  # noqa: E402


def scrape_fixtures() -> None:
    """
    Scrape the fixtures for the current season.
    """
    response = requests.get("https://fantasy.premierleague.com/api/fixtures")
    fixtures = pd.DataFrame(response.json())
    fixtures = fixtures[
        ["event", "team_h", "team_a", "team_h_score", "team_a_score", "stats"]
    ]
    fixtures.rename(
        columns={
            "event": "gw",
            "team_h": "home_team_id",
            "team_a": "away_team_id",
            "team_h_score": "home_team_score",
            "team_a_score": "away_team_score",
        },
        inplace=True,
    )

    fixtures.to_csv(get_data_path(SEASON, "fixtures.csv"), index=False)


if __name__ == "__main__":
    scrape_fixtures()
