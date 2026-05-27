import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping as lgbm_early_stopping, log_evaluation as lgbm_log_evaluation

from feature_engineering import (
    load_data,
    preprocess_train_test,
    MODEL_FEATURES
)


def get_catboost_model(iterations=3000):
    """Create a CatBoost regressor with tuned hyperparameters."""
    return CatBoostRegressor(
        iterations=iterations,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3,
        bagging_temperature=0.2,
        random_strength=0.5,
        border_count=254,
        loss_function='RMSE',
        eval_metric='R2',
        random_seed=42,
        verbose=0
    )


def get_lgbm_model(n_estimators=3000):
    """Create a LightGBM regressor with tuned hyperparameters."""
    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.02,
        num_leaves=128,
        max_depth=8,
        reg_lambda=3,
        reg_alpha=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42,
        verbose=-1
    )


def train_model():
    """
    Train an ensemble of CatBoost and LightGBM
    using K-Fold CV, then retrain on full data with weighted predictions.
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

    print(f'Feature count: {len(MODEL_FEATURES)}')
    print(f'Training samples: {len(X)}')

    # =========================================================
    # Phase 1: K-Fold CV to evaluate models and find best iters
    # =========================================================
    N_SPLITS = 5
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    oof_cat = np.zeros(len(X))
    oof_lgb = np.zeros(len(X))

    cat_best_iters = []
    lgb_best_iters = []

    fold_scores = {'catboost': [], 'lightgbm': []}

    print('\n' + '=' * 60)
    print('PHASE 1: K-Fold Cross Validation')
    print('=' * 60)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f'\n--- Fold {fold_idx + 1}/{N_SPLITS} ---')

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # CatBoost
        cat_model = get_catboost_model(iterations=3000)
        cat_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=200
        )
        cat_pred = cat_model.predict(X_val)
        oof_cat[val_idx] = cat_pred
        cat_r2 = r2_score(y_val, cat_pred)
        cat_best_iters.append(cat_model.get_best_iteration())
        fold_scores['catboost'].append(cat_r2)
        print(f'  CatBoost  R2: {cat_r2:.6f}  (best_iter: {cat_model.get_best_iteration()})')

        # LightGBM
        lgb_model = get_lgbm_model(n_estimators=3000)
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgbm_early_stopping(200, verbose=False),
                lgbm_log_evaluation(0)
            ]
        )
        lgb_pred = lgb_model.predict(X_val)
        oof_lgb[val_idx] = lgb_pred
        lgb_r2 = r2_score(y_val, lgb_pred)
        lgb_best_iters.append(lgb_model.best_iteration_)
        fold_scores['lightgbm'].append(lgb_r2)
        print(f'  LightGBM  R2: {lgb_r2:.6f}  (best_iter: {lgb_model.best_iteration_})')

    # Print CV summary
    print('\n' + '=' * 60)
    print('CV SUMMARY')
    print('=' * 60)

    mean_cat = np.mean(fold_scores['catboost'])
    mean_lgb = np.mean(fold_scores['lightgbm'])

    print(f'CatBoost  Mean R2: {mean_cat:.6f}  (per-fold: {[f"{s:.4f}" for s in fold_scores["catboost"]]})')
    print(f'LightGBM  Mean R2: {mean_lgb:.6f}  (per-fold: {[f"{s:.4f}" for s in fold_scores["lightgbm"]]})')

    # Compute OOF R2 for each model
    oof_r2_cat = r2_score(y, oof_cat)
    oof_r2_lgb = r2_score(y, oof_lgb)

    print(f'\nOOF CatBoost  R2: {oof_r2_cat:.6f}')
    print(f'OOF LightGBM  R2: {oof_r2_lgb:.6f}')

    # Compute ensemble weights proportional to OOF R2
    total_r2 = oof_r2_cat + oof_r2_lgb
    w_cat = oof_r2_cat / total_r2
    w_lgb = oof_r2_lgb / total_r2

    weights = {'catboost': w_cat, 'lightgbm': w_lgb}
    print(f'\nEnsemble Weights: CatBoost={w_cat:.4f}, LightGBM={w_lgb:.4f}')

    # OOF ensemble R2
    oof_ensemble = w_cat * oof_cat + w_lgb * oof_lgb
    oof_ens_r2 = r2_score(y, oof_ensemble)
    print(f'OOF Ensemble  R2: {oof_ens_r2:.6f}')

    # =========================================================
    # Phase 2: Retrain on full data
    # =========================================================
    print('\n' + '=' * 60)
    print('PHASE 2: Training on Full Dataset')
    print('=' * 60)

    # Use median of best iterations (with a small buffer)
    cat_iters = int(np.median(cat_best_iters) * 1.1)
    lgb_iters = int(np.median(lgb_best_iters) * 1.1)

    print(f'CatBoost iters: {cat_iters}')
    print(f'LightGBM iters: {lgb_iters}')

    # Train CatBoost on full data
    print('\nTraining CatBoost on full data...')
    final_cat = get_catboost_model(iterations=cat_iters)
    final_cat.fit(X, y)

    # Train LightGBM on full data
    print('Training LightGBM on full data...')
    final_lgb = get_lgbm_model(n_estimators=lgb_iters)
    final_lgb.fit(X, y)

    # Save models
    joblib.dump(final_cat, 'models/catboost_model.pkl')
    joblib.dump(final_lgb, 'models/lightgbm_model.pkl')
    joblib.dump(weights, 'models/ensemble_weights.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')

    print('\nAll models saved to models/')

    # =========================================================
    # Phase 3: Generate test predictions
    # =========================================================
    print('\n' + '=' * 60)
    print('PHASE 3: Generating Test Predictions')
    print('=' * 60)

    test_X = test[MODEL_FEATURES]

    pred_cat = final_cat.predict(test_X)
    pred_lgb = final_lgb.predict(test_X)

    test_pred = w_cat * pred_cat + w_lgb * pred_lgb

    submission = pd.DataFrame({
        'Index': test['Index'],
        'demand': test_pred
    })

    SUBMISSION_PATH = 'outputs/submission.csv'
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f'Submission Saved -> {SUBMISSION_PATH}')

    print('\n' + '=' * 60)
    print(f'FINAL OOF ENSEMBLE R2: {oof_ens_r2:.6f}')
    print(f'TARGET: >= 0.9400')
    status = 'PASSED' if oof_ens_r2 >= 0.94 else 'BELOW TARGET'
    print(f'STATUS: {status}')
    print('=' * 60)

    return final_cat, submission


if __name__ == '__main__':
    train_model()