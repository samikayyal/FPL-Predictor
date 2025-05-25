import pandas as pd
import requests


def __load_manager_ids() -> list[int]:
    res = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
    if res.status_code == 200:
        data = res.json()
        df = pd.DataFrame(data["elements"])
        df = df[(df.mng_win > 0) | (df.mng_draw > 0) | (df.mng_loss > 0)]

        return df["id"].tolist()
    else:
        return []


SEASON: str = "2024-25"
LAST_PLAYED_GAMEWEEK: int = 36

__MANAGER_PLAYER_IDS: list[int] = __load_manager_ids()
# If the manager IDs are already loaded, use them; otherwise, load them from the API.
MANAGER_PLAYER_IDS = (
    __MANAGER_PLAYER_IDS if __MANAGER_PLAYER_IDS else __load_manager_ids()
)
