"""
evaluate.py

Model evaluation script for the Shopping ML Project.

This module loads a previously trained machine learning model
and evaluates its performance on the test dataset.

Evaluation includes:
- Standard classification metrics
- Detailed classification report
- Confusion matrix visualization

This script ensures that model evaluation is separated
from training logic, which is a best practice in ML pipelines.
"""

import pandas as pd

from src.data_loader import load_raw_data
from src.preprocess import preprocess_dataset
from src.utils import (
    load_model,
    print_classification_metrics,
    plot_confusion_matrix
)


# --------------------------------------------------
# Evaluation Pipeline
# --------------------------------------------------

def evaluate_model(model_path: str, dataset_name: str) -> None:
    """
    Evaluate a trained model on the dataset.

    Parameters
    ----------
    model_path : str
        Path to the saved trained model (.pkl file).

    dataset_name : str
        Name of the dataset located in data/raw/.

    Steps
    -----
    1. Load dataset
    2. Run preprocessing pipeline
    3. Load trained model
    4. Generate predictions
    5. Print evaluation metrics
    6. Plot confusion matrix
    """

    print("\nLoading dataset...")
    df = load_raw_data(dataset_name)

    print("Dataset loaded successfully.")

    print("\nRunning preprocessing pipeline...")

    X_train, X_test, y_train, y_test, preprocessor = preprocess_dataset(
        df,
        target_column="shopping_preference"
    )

    print("Preprocessing completed.")

    print("\nLoading trained model...")

    model = load_model(model_path)

    print("\nGenerating predictions...")

    predictions = model.predict(X_test)

    print("\nEvaluation Results")
    print("------------------")

    print_classification_metrics(y_test, predictions)

    print("\nGenerating confusion matrix...")

    plot_confusion_matrix(
        y_test,
        predictions,
        title="Shopping Preference Prediction Confusion Matrix"
    )


# --------------------------------------------------
# Run Evaluation
# --------------------------------------------------

def main():
    """
    Main evaluation entry point.
    """

    model_path = "models/best_shopping_model.pkl"

    dataset_name = "online vs store shopping dataset.csv"

    evaluate_model(
        model_path=model_path,
        dataset_name=dataset_name
    )


# --------------------------------------------------
# Script Execution
# --------------------------------------------------

if __name__ == "__main__":
    """
    Allows the module to be executed directly.

    Example:
    python -m src.evaluate
    """

    main()