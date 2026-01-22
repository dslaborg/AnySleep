import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that handles numpy types.

    Converts numpy arrays to lists and formats numpy floats with 4 decimal
    places for compact, readable output.
    """

    def default(self, obj):
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        # format numpy floats with only 4 decimals
        if isinstance(obj, np.floating):
            return f"{obj:.4f}"
        return str(obj)


def remove_empty_values(dictionary):
    """
    Recursively remove empty values from a dictionary.

    Removes keys with None values or empty dicts/lists to produce
    cleaner JSON output.

    Args:
        dictionary (dict): Dictionary to clean in-place.

    Returns:
        dict: The same dictionary with empty values removed.
    """
    for key, value in list(dictionary.items()):
        if isinstance(value, dict):
            remove_empty_values(value)
        if value is None or (isinstance(value, (dict, list)) and len(value) == 0):
            dictionary.pop(key)
    return dictionary
