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
    model_path=None,
    encoder_path='models/encoders.pkl',
    test_path='data/test.csv',
    output_path='outputs/submission.csv'
):
    """
    Predict traffic demand using the ensemble of trained models.
    Loads CatBoost, XGBoost, and LightGBM models with their weights.
    """
    encoders = load_encoders(encoder_path)

    test = pd.read_csv(test_path)
    test = preprocess_test(test, encoders)

    index_col = test['Index']
    test_X = test[MODEL_FEATURES]

    # Load ensemble weights
    weights_path = 'models/ensemble_weights.pkl'
    if os.path.exists(weights_path):
        weights = joblib.load(weights_path)
        print(f'Ensemble weights: {weights}')

        # Load all models
        cat_model = joblib.load('models/catboost_model.pkl')
        lgb_model = joblib.load('models/lightgbm_model.pkl')

        pred_cat = cat_model.predict(test_X)
        pred_lgb = lgb_model.predict(test_X)

        predictions = (
            weights['catboost'] * pred_cat +
            weights['lightgbm'] * pred_lgb
        )
        print('Ensemble prediction completed (CatBoost + LightGBM)')
    else:
        # Fallback: single CatBoost model
        model = joblib.load(model_path or 'models/catboost_model.pkl')
        predictions = model.predict(test_X)
        print('Single model prediction completed (fallback)')

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
    submission = predict_demand()
    print(submission.head())