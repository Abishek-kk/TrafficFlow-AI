import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

from feature_engineering import (
    load_data,
    preprocess_train_test,
    MODEL_FEATURES
)


def train_model():
    """
    Train the traffic demand model and generate test predictions.
    """
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    TRAIN_PATH = 'data/train.csv'
    TEST_PATH = 'data/test.csv'

    print('Loading Data...')
    train, test = load_data(TRAIN_PATH, TEST_PATH)

    print('Preprocessing Data...')
    train, test, encoders = preprocess_train_test(train, test)

    X = train[MODEL_FEATURES]
    y = train['demand']

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print('Training CatBoost Model...')
    model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.04,
        depth=9,
        loss_function='RMSE',
        eval_metric='R2',
        random_seed=42,
        verbose=200
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    val_pred = model.predict(X_val)
    r2 = r2_score(y_val, val_pred)
    print(f'Validation R2 Score: {r2}')

    print('Training on Full Dataset...')
    model.fit(X, y)

    MODEL_PATH = 'models/catboost_model.pkl'
    joblib.dump(model, MODEL_PATH)
    print(f'Model Saved -> {MODEL_PATH}')

    ENCODER_PATH = 'models/encoders.pkl'
    joblib.dump(encoders, ENCODER_PATH)
    print(f'Encoders Saved -> {ENCODER_PATH}')

    print('Predicting Test Data...')
    test_pred = model.predict(test[MODEL_FEATURES])

    submission = pd.DataFrame({
        'Index': test['Index'],
        'demand': test_pred
    })

    SUBMISSION_PATH = 'outputs/submission.csv'
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f'Submission Saved -> {SUBMISSION_PATH}')

    return model, submission


if __name__ == '__main__':
    train_model()