import functools
import os
import re
import time

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
    """Constructs the absolute path to a data file in the mydata directory."""
    return os.path.join(_PROJECT_ROOT, "mydata", season, *args)


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
