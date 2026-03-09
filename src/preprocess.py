"""
preprocess.py

Preprocessing utilities for the Shopping ML Project.

This module handles:

- Categorical encoding
- Numerical feature scaling
- Feature-target separation
- Preprocessing pipeline creation for machine learning models

The goal is to centralize all preprocessing steps so that
train.py, evaluate.py, and inference.py can reuse them.
"""

from typing import Tuple, List
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from src.features import engineer_features

def separate_features_target(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The complete dataset.
    target_column : str
        Name of the target column (e.g., 'shopping_preference').

    Returns
    -------
    X : pd.DataFrame
        Features (all columns except target).
    y : pd.Series
        Target values.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def get_feature_types(
    X: pd.DataFrame
) -> Tuple[List[str], List[str]]:
    """
    Identify categorical and numerical feature columns.

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataset.

    Returns
    -------
    numerical_features : List[str]
        Names of numerical columns.
    categorical_features : List[str]
        Names of categorical columns.
    """
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'string']).columns.tolist()

    return numerical_features, categorical_features


def build_preprocessing_pipeline(
    numerical_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Build a preprocessing pipeline that:

    - Scales numerical features with StandardScaler
    - Encodes categorical features with OneHotEncoder

    Parameters
    ----------
    numerical_features : List[str]
        Names of numerical columns.
    categorical_features : List[str]
        Names of categorical columns.

    Returns
    -------
    preprocessor : ColumnTransformer
        Preprocessing pipeline that can be applied to training and test data.
    """

    # Pipeline for numerical features: scaling
    numeric_transformer = Pipeline(
        steps=[
            ('scaler', StandardScaler())
        ]
    )

    # Pipeline for categorical features: one-hot encoding
    categorical_transformer = Pipeline(
        steps=[
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ]
    )

    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def split_data(
    X: pd.DataFrame, y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into training and test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataset
    y : pd.Series
        Target variable
    test_size : float
        Fraction of data to hold out for testing
    random_state : int
        Seed for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def preprocess_dataset(
    df: pd.DataFrame, target_column: str
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Full preprocessing pipeline for the dataset:

    Steps:
    1. Feature engineering
    2. Separate features and target
    3. Detect feature types
    4. Train/test split
    5. Fit preprocessing pipeline
    6. Transform features

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset
    target_column : str
        Name of the target column

    Returns
    -------
    preprocessor : Pipeline
        Fitted preprocessing pipeline
    X_train, X_test : pd.DataFrame
        Train and test features
    y_train, y_test : pd.Series
        Train and test target values
    """
    
    df = df.copy()
    
    # Step 1: Apply feature engineering
    df = engineer_features(df)
    
    # Step 2: Separate features and target
    X, y = separate_features_target(df, target_column)

    # Step 3: Identify feature types
    numerical_features, categorical_features = get_feature_types(X)

    # Step 4: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 5: Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(numerical_features, categorical_features)

    # Step 6: Fit on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Step 7: Transform test data
    X_test_processed = preprocessor.transform(X_test)
    

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )


if __name__ == "__main__":
    """
    Allows testing the preprocessing module independently.
    """
    from src.data_loader import load_raw_data

    # Load raw data
    dataset_name = "online vs store shopping dataset.csv"
    df = load_raw_data(dataset_name)

    # Preprocess dataset
    preprocessor, X_train, X_test, y_train, y_test = preprocess_dataset(
        df, target_column="shopping_preference"
    )

    print("Training Features Shape:", X_train.shape)
    print("Test Features Shape:", X_test.shape)
    print("Training Target Distribution:\n", y_train.value_counts())