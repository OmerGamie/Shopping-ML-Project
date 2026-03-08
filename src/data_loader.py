"""
data_loader.py

Utility functions for loading and validating datasets used in the ML pipeline.

This module centralizes dataset loading logic so that other modules
(train.py, preprocess.py, evaluate.py) do not need to know where the data lives.
"""

from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    """
    Return the root directory of the project.

    This ensures file paths work regardless of where the script is executed.

    Returns
    -------
    Path
        Absolute path to the project root directory.
    """
    return Path(__file__).resolve().parents[1]


def get_raw_data_path(filename: str) -> Path:
    """
    Construct the path to a raw dataset file.

    Parameters
    ----------
    filename : str
        Name of the dataset file (e.g., "shopping_behavior.csv")

    Returns
    -------
    Path
        Full path to the dataset inside data/raw
    """
    project_root = get_project_root()
    return project_root / "data" / "raw" / filename


def load_raw_data(filename: str) -> pd.DataFrame:
    """
    Load a dataset from the raw data directory.

    Parameters
    ----------
    filename : str
        Name of the dataset file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If dataset does not exist.
    """

    data_path = get_raw_data_path(filename)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Run scripts/download_data.py first."
        )

    df = pd.read_csv(data_path)

    return df


def basic_data_check(df: pd.DataFrame) -> None:
    """
    Perform basic sanity checks on the dataset.

    Prints:
    - dataset shape
    - column types
    - missing values
    """

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])


if __name__ == "__main__":
    """
    Allows the module to be run directly for quick dataset inspection.
    """

    dataset_name = "online vs store shopping dataset.csv"

    df = load_raw_data(dataset_name)

    basic_data_check(df)

    print("\nFirst rows of dataset:")
    print(df.head())
    
    