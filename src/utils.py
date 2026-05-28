import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


def create_folders():
    """
    Create required project folders
    """
    folders = [
        "models",
        "outputs",
        "logs"
    ]

    for folder in folders:
        os.makedirs(
            folder,
            exist_ok=True
        )

    print("Folders Created")


def save_model(
    model,
    model_path
):
    """
    Save trained model
    """
    joblib.dump(
        model,
        model_path
    )

    print(
        f"Model Saved -> {model_path}"
    )


def load_model(
    model_path
):
    """
    Load saved model
    """
    model = joblib.load(
        model_path
    )

    print(
        f"Model Loaded -> {model_path}"
    )

    return model


def evaluate_model(
    y_true,
    y_pred
):
    """
    Calculate evaluation metrics
    """

    r2 = r2_score(
        y_true,
        y_pred
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = mse ** 0.5

    metrics = {
        "R2 Score": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }

    print("\nModel Evaluation")
    print("--------------------")

    for key, value in metrics.items():
        print(
            f"{key}: {value}"
        )

    return metrics


def save_submission(
    index_col,
    predictions,
    output_path="outputs/submission.csv"
):
    """
    Save submission CSV
    """

    predictions = np.clip(predictions, 0, 1)

    submission = pd.DataFrame({
        "Index": index_col,
        "demand": predictions
    })

    submission.to_csv(
        output_path,
        index=False
    )

    print(
        f"Submission Saved -> {output_path}"
    )

    return submission


def print_separator():
    """
    Print console separator
    """
    print(
        "=" * 50
    )


if __name__ == "__main__":

    print_separator()

    create_folders()

    print(
        "utils.py Ready"
    )

    print_separator()