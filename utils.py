import pandas as pd
import json
from typing import Any, Union, Optional, Dict, List
from pathlib import Path
import numpy as np


def load_csv(file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Load a CSV file into a pandas DataFrame.

    :param file_path: Path to the CSV file.
    :type file_path: Union[str, Path]
    :returns: Data from the CSV file if successful, otherwise None if an error occurs.
    :rtype: Optional[pd.DataFrame]
    """

    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
    except pd.errors.ParserError:
        print(f"Error: Could not parse '{file_path}'.")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for NumPy data types.
    This encoder extends the standard :class:`json.JSONEncoder` to handle
    NumPy-specific objects that are not JSON serializable by default.

    Supported conversions:
        - np.integer → int
        - np.floating → float
        - np.ndarray → list
        - np.bool_ / bool → bool
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def store_json(data: Any, path: Union[str, Path]) -> None:
    """Store data as a JSON file.

    :param data: The data to be serialized into JSON.
    :type data: Any
    :param path: The file path where the JSON will be stored.
    :type path: Union[str, Path]
    """

    with open(path, "w") as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON file into a Python dictionary.

    :param path: Path to the JSON file.
    :type path: str | Path
    :returns: Parsed JSON content.
    :rtype: dict
    :raises FileNotFoundError: If the file does not exist.
    :raises json.JSONDecodeError: If the file is not a valid JSON.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    return data


def store_df_to_csv(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Save a pandas DataFrame to CSV.

    :param df: DataFrame to save.
    :type df: pd.DataFrame
    :param path: File path where the CSV will be written.
    :type path: Union[str, Path]
    """

    df.to_csv(path, index=False, header=True)


def is_between(x: float, low: float = 0, high: float = 1) -> bool:
    """Check if a value lies within a closed interval [low, high].

    :param x: The value to check.
    :type x: float
    :param low: Lower bound of the interval. Defaults to 0.
    :type low: float, optional
    :param high: Upper bound of the interval. Defaults to 1.
    :type high: float, optional
    :returns: True if `x` is between `low` and `high`, False otherwise.
    :rtype: bool
    """
    return low <= x <= high


def concat_dfs(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate a list of DataFrames along columns (axis=1).

    :param df_list: List of DataFrames to concatenate.
    :type df_list: List[pd.DataFrame]
    :returns: A single DataFrame with columns from all input DataFrames.
    :rtype: pd.DataFrame
    """

    return pd.concat(df_list, axis=1)
