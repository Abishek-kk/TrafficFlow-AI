import os
import pandas as pd
import joblib

from feature_engineering import (
    preprocess_test,
    MODEL_FEATURES
)


def load_model(model_path):
    """
    Load trained model.
    """
    model = joblib.load(model_path)
    print('Model Loaded Successfully')
    return model


def load_encoders(encoder_path):
    """
    Load saved encoders.
    """
    encoders = joblib.load(encoder_path)
    print('Encoders Loaded Successfully')
    return encoders


def predict_demand(
    model_path,
    encoder_path,
    test_path,
    output_path='outputs/submission.csv'
):
    """
    Predict traffic demand using the trained model and saved encoders.
    """
    model = load_model(model_path)
    encoders = load_encoders(encoder_path)

    test = pd.read_csv(test_path)
    test = preprocess_test(test, encoders)

    index_col = test['Index']
    predictions = model.predict(test[MODEL_FEATURES])

    submission = pd.DataFrame({
        'Index': index_col,
        'demand': predictions
    })

    os.makedirs('outputs', exist_ok=True)
    submission.to_csv(output_path, index=False)

    print('Prediction Completed')
    print(f'Submission Saved -> {output_path}')

    return submission


if __name__ == '__main__':
    MODEL_PATH = 'models/catboost_model.pkl'
    ENCODER_PATH = 'models/encoders.pkl'
    TEST_PATH = 'data/test.csv'

    submission = predict_demand(
        MODEL_PATH,
        ENCODER_PATH,
        TEST_PATH
    )
    print(submission.head())