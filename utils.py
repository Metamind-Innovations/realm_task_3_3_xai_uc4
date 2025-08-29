import pandas as pd
import json
from typing import Any, Union, Optional, Dict, List
from pathlib import Path


def load_csv(file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Load a CSV file into a pandas DataFrame.

    Args:
        file_path (Union[str, Path]): Path to the CSV file.

    Returns:
        Optional[pd.DataFrame]: Data from the CSV file if successful,
        otherwise None if an error occurs.

    Raises:
        None: Errors are caught and printed instead of raised.
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


def store_json(data: Any, path: Union[str, Path]) -> None:
    """Store data as a JSON file.

    Args:
        data (Any): The data to be serialized into JSON.
        path (Union[str, Path]): The file path where the JSON will be stored.

    Returns:
        None
    """

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a JSON file into a Python dictionary.

    Args:
        path (str | Path): Path to the JSON file.

    Returns:
        dict: Parsed JSON content.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    return data


def store_df_to_csv(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    """
    Save a pandas DataFrame to CSV.

    Args:
        df (pd.DataFrame): DataFrame to save.
        path (Union[str, Path]): File path where the CSV will be written.
    """

    df.to_csv(path, index=False, header=True)


def is_between(x: float, low: float = 0, high: float = 1) -> bool:
    """
    Check if a value lies within a closed interval [low, high].

    Args:
        x (float): The value to check.
        low (float, optional): Lower bound of the interval. Defaults to 0.
        high (float, optional): Upper bound of the interval. Defaults to 1.

    Returns:
        bool: True if `x` is between `low` and `high`, False otherwise.
    """
    return low <= x <= high


def concat_dfs(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate a list of DataFrames along columns (axis=1).

    Args:
        df_list (List[pd.DataFrame]): List of DataFrames to concatenate.

    Returns:
        pd.DataFrame: A single DataFrame with columns from all input DataFrames.
    """

    return pd.concat(df_list, axis=1)
