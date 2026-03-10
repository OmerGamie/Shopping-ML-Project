"""
train.py

Model training script for the Shopping ML Project.

This script performs the following steps:

1. Load dataset
2. Apply feature engineering
3. Preprocess data
4. Train multiple machine learning models
5. Evaluate model performance
6. Compare models
7. Save the best performing model

This file represents the core training pipeline of the project.
"""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import f1_score

from src.data_loader import load_raw_data
from src.preprocess import preprocess_dataset
from src.utils import save_model, print_classification_metrics


# --------------------------------------------------
# Model Registry
# --------------------------------------------------

def get_models():
    """
    Define machine learning models to train.

    Returns
    -------
    dict
        Dictionary mapping model names to model objects.
    """

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),

        "Support Vector Machine": SVC(
            class_weight="balanced"
        ),

        "K Nearest Neighbors": KNeighborsClassifier(),

        "Naive Bayes": GaussianNB()
    }

    return models


# --------------------------------------------------
# Model Training
# --------------------------------------------------

def train_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple machine learning models.

    Parameters
    ----------
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like
        Training labels
    y_test : array-like
        Test labels

    Returns
    -------
    results_df : pd.DataFrame
        Table comparing model performance
    best_model : object
        Best performing trained model
    best_model_name : str
        Name of best model
    """

    models = get_models()

    results = []

    best_f1 = 0
    best_model = None
    best_model_name = None

    for name, model in models.items():

        print(f"\nTraining: {name}")
        print("-" * 40)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted"
        )

        results.append({
            "Model": name,
            "F1 Score": f1
        })

        print_classification_metrics(y_test, predictions)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(results).sort_values(
        by="F1 Score",
        ascending=False
    )

    return results_df, best_model, best_model_name


# --------------------------------------------------
# Training Pipeline
# --------------------------------------------------

def main():
    """
    Main training pipeline.
    """

    print("\nLoading dataset...")
    df = load_raw_data(
        "online vs store shopping dataset.csv"
    )

    print("Dataset loaded successfully.")

    print("\nPreprocessing dataset...")

    X_train, X_test, y_train, y_test, preprocessor = preprocess_dataset(
        df,
        target_column="shopping_preference"
    )

    print("Preprocessing completed.")

    print("\nTraining models...")

    results_df, best_model, best_model_name = train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\nModel Comparison")
    print("----------------")
    print(results_df)

    print(f"\nBest Model: {best_model_name}")

    print("\nSaving best model...")

    save_model(best_model, "best_shopping_model")

    print("\nTraining pipeline completed successfully.")


# --------------------------------------------------
# Run Training
# --------------------------------------------------

if __name__ == "__main__":
    """
    Run training when executing the module.

    Example:
    python -m src.train
    """

    main()