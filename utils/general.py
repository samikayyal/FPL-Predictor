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
