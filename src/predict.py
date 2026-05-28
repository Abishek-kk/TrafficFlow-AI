import os
import numpy as np
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
    Loads CatBoost, LightGBM, and XGBoost models with their weights.
    """
    encoders = load_encoders(encoder_path)

    # Load config to check if log target was used
    config_path = 'models/config.pkl'
    use_log_target = False
    if os.path.exists(config_path):
        config = joblib.load(config_path)
        use_log_target = config.get('use_log_target', False)
        print(f'Log target: {use_log_target}')

    test = pd.read_csv(test_path)
    test = preprocess_test(test, encoders)

    index_col = test['Index']
    test_X = test[MODEL_FEATURES]

    # Load ensemble weights if available
    weights_path = 'models/ensemble_weights.pkl'
    xgb_path = 'models/xgboost_model.pkl'
    cat_path = 'models/catboost_model.pkl'
    lgb_path = 'models/lightgbm_model.pkl'

    if os.path.exists(weights_path):
        weights = joblib.load(weights_path)
        print(f'Ensemble weights: {weights}')

        # Load CatBoost and LightGBM models if they exist
        cat_model = joblib.load(cat_path)
        pred_cat = cat_model.predict(test_X)
        if use_log_target:
            pred_cat = np.expm1(pred_cat)

        lgb_model = None
        pred_lgb = 0
        if os.path.exists(lgb_path) and 'lightgbm' in weights:
            lgb_model = joblib.load(lgb_path)
            pred_lgb = lgb_model.predict(test_X)
            if use_log_target:
                pred_lgb = np.expm1(pred_lgb)

        pred_xgb = 0
        xgb_model = None
        if os.path.exists(xgb_path) and 'xgboost' in weights:
            xgb_model = joblib.load(xgb_path)
            pred_xgb = xgb_model.predict(test_X)
            if use_log_target:
                pred_xgb = np.expm1(pred_xgb)

        predictions = weights['catboost'] * pred_cat
        if lgb_model is not None:
            predictions += weights['lightgbm'] * pred_lgb
        if xgb_model is not None:
            predictions += weights['xgboost'] * pred_xgb

        if xgb_model is not None:
            print('Ensemble prediction completed (CatBoost + LightGBM + XGBoost)')
        elif lgb_model is not None:
            print('Ensemble prediction completed (CatBoost + LightGBM)')
        else:
            print('Ensemble prediction completed (CatBoost only)')
    else:
        # Fallback: use CatBoost + XGBoost if both saved, otherwise CatBoost alone
        cat_model = joblib.load(model_path or cat_path)
        pred_cat = cat_model.predict(test_X)
        if use_log_target:
            pred_cat = np.expm1(pred_cat)

        if os.path.exists(xgb_path):
            xgb_model = joblib.load(xgb_path)
            pred_xgb = xgb_model.predict(test_X)
            if use_log_target:
                pred_xgb = np.expm1(pred_xgb)
            predictions = 0.5 * pred_cat + 0.5 * pred_xgb
            print('Fallback ensemble prediction completed (CatBoost + XGBoost)')
        else:
            predictions = pred_cat
            print('Single model prediction completed (fallback CatBoost only)')

    # Clip predictions to valid range
    predictions = np.clip(predictions, 0, 1)

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