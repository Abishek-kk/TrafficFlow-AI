import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping as lgbm_early_stopping, log_evaluation as lgbm_log_evaluation
from xgboost import XGBRegressor

from feature_engineering import (
    load_data,
    preprocess_train_test,
    MODEL_FEATURES
)

# Whether to log-transform the target
USE_LOG_TARGET = True


def get_catboost_model(iterations=5000):
    """Create a CatBoost regressor with tuned hyperparameters."""
    return CatBoostRegressor(
        iterations=iterations,
        learning_rate=0.04,
        depth=8,
        l2_leaf_reg=3,
        bagging_temperature=0.5,
        min_data_in_leaf=20,
        random_strength=0.5,
        border_count=254,
        loss_function='RMSE',
        eval_metric='R2',
        random_seed=42,
        verbose=0
    )


def get_lgbm_model(n_estimators=5000):
    """Create a LightGBM regressor with tuned hyperparameters."""
    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.015,
        num_leaves=255,
        max_depth=10,
        reg_lambda=2,
        reg_alpha=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        random_state=42,
        verbose=-1
    )


def get_xgb_model(n_estimators=5000):
    """Create an XGBoost regressor with tuned hyperparameters."""
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.015,
        max_depth=9,
        reg_lambda=2,
        reg_alpha=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        tree_method='hist',
        random_state=42,
        verbosity=0
    )


def train_model():
    """
    Train an ensemble of CatBoost, LightGBM, and XGBoost
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
    y_raw = train['demand']

    # Log-transform target if enabled
    if USE_LOG_TARGET:
        y = np.log1p(y_raw)
        print('Using log1p target transform')
    else:
        y = y_raw

    print(f'Feature count: {len(MODEL_FEATURES)}')
    print(f'Training samples: {len(X)}')

    # =========================================================
    # Phase 1: K-Fold CV to evaluate models and find best iters
    # =========================================================
    N_SPLITS = 5
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    oof_cat = np.zeros(len(X))
    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))

    cat_best_iters = []
    lgb_best_iters = []
    xgb_best_iters = []

    fold_scores = {'catboost': [], 'lightgbm': [], 'xgboost': []}

    print('\n' + '=' * 60)
    print('PHASE 1: K-Fold Cross Validation')
    print('=' * 60)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f'\n--- Fold {fold_idx + 1}/{N_SPLITS} ---')

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        y_val_raw = y_raw.iloc[val_idx]

        # CatBoost
        cat_model = get_catboost_model(iterations=5000)
        cat_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=300
        )
        cat_pred = cat_model.predict(X_val)
        oof_cat[val_idx] = cat_pred
        if USE_LOG_TARGET:
            cat_pred_raw = np.expm1(cat_pred)
        else:
            cat_pred_raw = cat_pred
        cat_r2 = r2_score(y_val_raw, cat_pred_raw)
        cat_best_iters.append(cat_model.get_best_iteration())
        fold_scores['catboost'].append(cat_r2)
        print(f'  CatBoost  R2: {cat_r2:.6f}  (best_iter: {cat_model.get_best_iteration()})')

        # LightGBM
        lgb_model = get_lgbm_model(n_estimators=5000)
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgbm_early_stopping(300, verbose=False),
                lgbm_log_evaluation(0)
            ]
        )
        lgb_pred = lgb_model.predict(X_val)
        oof_lgb[val_idx] = lgb_pred
        if USE_LOG_TARGET:
            lgb_pred_raw = np.expm1(lgb_pred)
        else:
            lgb_pred_raw = lgb_pred
        lgb_r2 = r2_score(y_val_raw, lgb_pred_raw)
        lgb_best_iters.append(lgb_model.best_iteration_)
        fold_scores['lightgbm'].append(lgb_r2)
        print(f'  LightGBM  R2: {lgb_r2:.6f}  (best_iter: {lgb_model.best_iteration_})')

        # XGBoost
        xgb_model = get_xgb_model(n_estimators=5000)
        xgb_model.set_params(early_stopping_rounds=300)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        xgb_pred = xgb_model.predict(X_val)
        oof_xgb[val_idx] = xgb_pred
        if USE_LOG_TARGET:
            xgb_pred_raw = np.expm1(xgb_pred)
        else:
            xgb_pred_raw = xgb_pred
        xgb_r2 = r2_score(y_val_raw, xgb_pred_raw)
        xgb_iter = xgb_model.best_iteration
        xgb_best_iters.append(xgb_iter)
        fold_scores['xgboost'].append(xgb_r2)
        print(f'  XGBoost   R2: {xgb_r2:.6f}  (best_iter: {xgb_iter})')

    # Print CV summary
    print('\n' + '=' * 60)
    print('CV SUMMARY')
    print('=' * 60)

    mean_cat = np.mean(fold_scores['catboost'])
    mean_lgb = np.mean(fold_scores['lightgbm'])
    mean_xgb = np.mean(fold_scores['xgboost'])

    print(f'CatBoost  Mean R2: {mean_cat:.6f}  (per-fold: {[f"{s:.4f}" for s in fold_scores["catboost"]]})')
    print(f'LightGBM  Mean R2: {mean_lgb:.6f}  (per-fold: {[f"{s:.4f}" for s in fold_scores["lightgbm"]]})')
    print(f'XGBoost   Mean R2: {mean_xgb:.6f}  (per-fold: {[f"{s:.4f}" for s in fold_scores["xgboost"]]})')

    # Compute OOF R2 for each model (in raw space)
    if USE_LOG_TARGET:
        oof_cat_raw = np.expm1(oof_cat)
        oof_lgb_raw = np.expm1(oof_lgb)
        oof_xgb_raw = np.expm1(oof_xgb)
    else:
        oof_cat_raw = oof_cat
        oof_lgb_raw = oof_lgb
        oof_xgb_raw = oof_xgb

    oof_r2_cat = r2_score(y_raw, oof_cat_raw)
    oof_r2_lgb = r2_score(y_raw, oof_lgb_raw)
    oof_r2_xgb = r2_score(y_raw, oof_xgb_raw)

    print(f'\nOOF CatBoost  R2: {oof_r2_cat:.6f}')
    print(f'OOF LightGBM  R2: {oof_r2_lgb:.6f}')
    print(f'OOF XGBoost   R2: {oof_r2_xgb:.6f}')

    # Optimize ensemble weights using grid search on OOF predictions
    print('\nOptimizing ensemble weights...')
    best_r2 = -1
    best_weights = (1/3, 1/3, 1/3)

    for w1 in np.arange(0.0, 1.01, 0.05):
        for w2 in np.arange(0.0, 1.01 - w1, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 < 0:
                continue
            blend = w1 * oof_cat_raw + w2 * oof_lgb_raw + w3 * oof_xgb_raw
            score = r2_score(y_raw, blend)
            if score > best_r2:
                best_r2 = score
                best_weights = (w1, w2, w3)

    w_cat, w_lgb, w_xgb = best_weights
    weights = {'catboost': w_cat, 'lightgbm': w_lgb, 'xgboost': w_xgb}
    print(f'Optimal Weights: CatBoost={w_cat:.2f}, LightGBM={w_lgb:.2f}, XGBoost={w_xgb:.2f}')
    print(f'OOF Ensemble R2: {best_r2:.6f}')

    # =========================================================
    # Phase 2: Retrain on full data
    # =========================================================
    print('\n' + '=' * 60)
    print('PHASE 2: Training on Full Dataset')
    print('=' * 60)

    # Use median of best iterations found during CV.
    # For final CatBoost retrain we set iterations explicitly to this value
    # and disable `use_best_model` to ensure the model is trained on full data.
    cat_iters = int(np.median(cat_best_iters))
    lgb_iters = int(np.median(lgb_best_iters) * 1.1)
    xgb_iters = int(np.median(xgb_best_iters) * 1.1)

    print(f'CatBoost iters: {cat_iters}')
    print(f'LightGBM iters: {lgb_iters}')
    print(f'XGBoost  iters: {xgb_iters}')

    # Train CatBoost on full data — explicitly set the best CV iteration
    # and use a slightly deeper model for the final fit.
    print('\nTraining CatBoost on full data...')
    best_iter = int(np.median(cat_best_iters))
    final_cat = CatBoostRegressor(
        iterations=best_iter,
        learning_rate=0.04,
        depth=9,
        loss_function='RMSE',
        random_seed=42,
        verbose=200
    )
    final_cat.fit(X, y, use_best_model=False)

    # Train LightGBM on full data
    print('Training LightGBM on full data...')
    final_lgb = get_lgbm_model(n_estimators=lgb_iters)
    final_lgb.fit(X, y)

    # Train XGBoost on full data
    print('Training XGBoost on full data...')
    final_xgb = get_xgb_model(n_estimators=xgb_iters)
    final_xgb.fit(X, y)

    # Save models
    joblib.dump(final_cat, 'models/catboost_model.pkl')
    joblib.dump(final_lgb, 'models/lightgbm_model.pkl')
    joblib.dump(final_xgb, 'models/xgboost_model.pkl')
    joblib.dump(weights, 'models/ensemble_weights.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')
    joblib.dump({'use_log_target': USE_LOG_TARGET}, 'models/config.pkl')

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
    pred_xgb = final_xgb.predict(test_X)

    if USE_LOG_TARGET:
        pred_cat = np.expm1(pred_cat)
        pred_lgb = np.expm1(pred_lgb)
        pred_xgb = np.expm1(pred_xgb)

    test_pred = w_cat * pred_cat + w_lgb * pred_lgb + w_xgb * pred_xgb

    # Clip predictions to valid range
    test_pred = np.clip(test_pred, 0, 1)

    submission = pd.DataFrame({
        'Index': test['Index'],
        'demand': test_pred
    })

    SUBMISSION_PATH = 'outputs/submission.csv'
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f'Submission Saved -> {SUBMISSION_PATH}')
    print(f'Submission shape: {submission.shape}')
    print(f'Demand stats: mean={test_pred.mean():.6f}, std={test_pred.std():.6f}, min={test_pred.min():.6f}, max={test_pred.max():.6f}')

    print('\n' + '=' * 60)
    print(f'FINAL OOF ENSEMBLE R2: {best_r2:.6f}')
    print(f'SCORE (max(0, 100*R2)): {max(0, 100*best_r2):.2f}')
    print(f'TARGET: >= 95.00')
    status = 'PASSED' if best_r2 >= 0.95 else 'BELOW TARGET'
    print(f'STATUS: {status}')
    print('=' * 60)

    return final_cat, submission


if __name__ == '__main__':
    train_model()