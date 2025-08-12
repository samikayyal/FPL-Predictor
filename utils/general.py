import functools
import os
import re
import time

import pandas as pd
from bs4 import BeautifulSoup
from unidecode import unidecode


def get_pos_from_element_id(element_id: int) -> str:
    """
    Get the position of a player from their element id.

    Args:
        element_id (int): The element id of the player.

    Returns:
        str: The position of the player.
    """
    if element_id not in range(1, 6):
        raise ValueError("Element ID must be between 1 and 5.")

    positions = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD", 5: "MGR"}
    return positions[element_id]


def time_function(func):
    """
    A decorator that prints the execution time of the function it decorates.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # perf_counter is more precise than time.time()
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")

        return result

    return wrapper


# Get the directory where get_ids.py is located
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (assuming utils is one level down from root)
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))


def get_data_path(season: str, *args) -> str:
    """Constructs the absolute path to a new data file in the mydata directory."""
    return os.path.join(_PROJECT_ROOT, "data", season, *args)


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = unidecode(name)  # Remove accents (e.g., Ødegaard -> Odegaard)
    name = name.lower()  # Lowercase
    name = re.sub(
        r"[^\w\s-]", "", name
    )  # Remove special characters except hyphen and space
    name = name.strip()  # Remove leading/trailing whitespace
    return name


def get_fbref_table_data(soup: BeautifulSoup, find_by: dict, new_columns: list[str] | None = None) -> pd.DataFrame:
    """
    Extracts data from a table in the given BeautifulSoup object.
    Args:
        soup (BeautifulSoup): The BeautifulSoup object containing the HTML.
        find_by (dict): A dictionary to find the table (e.g., {"id": "stats_squads_standard_for"}).
        new_columns (list[str] | None): Optional list of new column names to set for the DataFrame.
    """

    table = soup.find("table", find_by)
    if table is None:
        print("No table found")
        return pd.DataFrame()  # Return empty DataFrame if no table found

    # Extract headers
    thead = table.find("thead")
    rows_in_thead = thead.find_all("tr")
    if not rows_in_thead:
        return pd.DataFrame()  # Return empty DataFrame if no header rows found

    for row in rows_in_thead:
        if row.get("class") and "over_header" not in row.get("class"):
            continue
        headers_row = row

    headers = [th.text.strip() for th in headers_row.find_all("th")]
    # Extract rows
    rows = []
    for row in table.find("tbody").find_all("tr"):
        cells = row.find_all("td")
        cells_text = [cell.text.strip() for cell in cells]
        # if the squad name is weird
        if row.find("th"):
            cells_text.insert(0, row.find("th").find("a").text.strip())

        rows.append(cells_text)

    df = pd.DataFrame(rows, columns=headers)
    if new_columns:
        df.columns = new_columns
    
    return df
