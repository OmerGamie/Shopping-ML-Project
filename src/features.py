"""
features.py

Feature engineering utilities for the Shopping ML Project.

This module creates new features from the original dataset to improve
machine learning model performance.

Feature engineering is performed AFTER loading the dataset and BEFORE
training models.

Typical responsibilities of this module:

- Creating behavioral features
- Creating ratio features
- Creating aggregated features
- Improving signal for ML models

These functions are reusable across notebooks, training scripts,
and inference pipelines.
"""

from typing import Tuple
import pandas as pd


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create behavioral features describing user shopping habits.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataset containing raw shopping behavior features.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with new behavioral features added.
    """

    # Ratio of online orders vs total shopping activity
    df["online_order_ratio"] = df["monthly_online_orders"] / (
        df["monthly_online_orders"] + df["monthly_store_visits"] + 1
    )

    # Ratio of online spending vs total spending
    df["online_spend_ratio"] = df["avg_online_spend"] / (
        df["avg_online_spend"] + df["avg_store_spend"] + 1
    )

    # Average spend per online order
    df["avg_spend_per_online_order"] = df["avg_online_spend"] / (
        df["monthly_online_orders"] + 1
    )

    return df


def add_digital_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features representing the user's digital engagement level.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing digital behavior variables.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with digital engagement features added.
    """

    # Combined digital usage score
    df["digital_engagement_score"] = (
        df["daily_internet_hours"]
        + df["social_media_hours"]
        + df["tech_savvy_score"]
    )

    # Internet usage per year of smartphone experience
    df["internet_usage_per_year"] = df["daily_internet_hours"] / (
        df["smartphone_usage_years"] + 1
    )

    return df


def add_purchase_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features describing purchasing psychology.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing purchase behavior metrics.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with additional purchase behavior features.
    """

    # Sensitivity to promotions
    df["promotion_sensitivity"] = (
        df["discount_sensitivity"] + df["free_return_importance"]
    ) / 2

    # Convenience preference score
    df["convenience_score"] = (
        df["delivery_fee_sensitivity"]
        + df["product_availability_online"]
        + df["time_pressure_level"]
    ) / 3

    # Offline preference indicator
    df["offline_shopping_preference_score"] = (
        df["need_touch_feel_score"] + df["brand_loyalty_score"]
    ) / 2

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to the dataset.

    This function orchestrates all feature engineering functions.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset before feature engineering.

    Returns
    -------
    df : pd.DataFrame
        Dataset enriched with engineered features.
    """

    df = add_behavioral_features(df)
    df = add_digital_engagement_features(df)
    df = add_purchase_behavior_features(df)

    return df


if __name__ == "__main__":
    """
    Allows this module to be run independently for testing.

    Example:
    python src/features.py
    """

    from src.data_loader import load_raw_data

    dataset_name = "online vs store shopping dataset.csv"

    df = load_raw_data(dataset_name)

    df = engineer_features(df)

    print("New feature columns created:")
    print(df.columns)