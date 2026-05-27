import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from train_model import train_model
from predict import predict_demand


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

    print("Folders Created Successfully")


def main():
    """
    Full ML Pipeline
    """

    print("=" * 50)
    print("TRAFFIC DEMAND PREDICTION PIPELINE")
    print("=" * 50)

    # Create folders
    create_folders()

    # Step 1: Train Model
    print("\nSTEP 1 : TRAIN MODEL")
    print("-" * 50)

    model, submission = train_model()

    print(
        "\nTraining Completed Successfully"
    )

    # Step 2: Predict Using Saved Model
    print("\nSTEP 2 : PREDICT TEST DATA")
    print("-" * 50)

    MODEL_PATH = "models/catboost_model.pkl"
    ENCODER_PATH = "models/encoders.pkl"
    TEST_PATH = "data/test.csv"

    predictions = predict_demand(
        model_path=MODEL_PATH,
        encoder_path=ENCODER_PATH,
        test_path=TEST_PATH,
        output_path="outputs/submission.csv"
    )

    print(
        "\nPrediction Completed Successfully"
    )

    print("\nFinal Submission Preview")
    print("-" * 50)
    print(predictions.head())

    print("\nPipeline Finished Successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()