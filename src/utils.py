"""
utils.py

Utility functions for the Shopping ML Project.

This module provides helper functions that are reused across
different parts of the machine learning pipeline, including:

- Saving trained models
- Loading saved models
- Printing classification metrics
- Plotting confusion matrices
- Creating directories if they do not exist

These utilities help keep training and evaluation scripts
clean and focused on core ML logic.
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# Directory Utilities

def ensure_directory(path: str) -> None:
    """
    Ensure that a directory exists. If it does not exist,
    it will be created.

    Parameters
    ----------
    path : str
        Path to the directory that should exist.

    Example
    -------
    ensure_directory("models/")
    """
    Path(path).mkdir(parents=True, exist_ok=True)


# Model Persistence


def save_model(model, model_name: str, directory: str = "models") -> None:
    """
    Save a trained machine learning model to disk.

    Parameters
    ----------
    model : object
        Trained model object.
    model_name : str
        Name for the saved model file.
    directory : str
        Directory where the model should be saved.

    Example
    -------
    save_model(model, "random_forest_model")
    """

    ensure_directory(directory)

    model_path = Path(directory) / f"{model_name}.pkl"

    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")


def load_model(model_path: str):
    """
    Load a previously saved machine learning model.

    Parameters
    ----------
    model_path : str
        Path to the saved model file.

    Returns
    -------
    model : object
        Loaded machine learning model.

    Example
    -------
    model = load_model("models/random_forest_model.pkl")
    """

    model = joblib.load(model_path)

    print(f"Model loaded from: {model_path}")

    return model


# Evaluation Metrics


def print_classification_metrics(y_true, y_pred) -> None:
    """
    Print standard classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.

    Metrics printed:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Full classification report
    """

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print("\nModel Performance Metrics")
    print("-------------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report")
    print("---------------------")
    print(classification_report(y_true, y_pred))


# Confusion Matrix Visualization

def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix") -> None:
    """
    Plot a confusion matrix using seaborn heatmap.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    title : str
        Title for the confusion matrix plot.

    Example
    -------
    plot_confusion_matrix(y_test, predictions)
    """

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    plt.tight_layout()

    plt.show()


# Testing the module

if __name__ == "__main__":
    """
    Allows testing this module independently.

    Example:
    python -m src.utils
    """

    print("Utils module loaded successfully.")