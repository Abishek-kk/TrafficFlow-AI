import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import sys
sys.path.insert(0, 'src')
from feature_engineering import preprocess_train_test, MODEL_FEATURES

# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Let's write a custom preprocessing that does NOT leak 'day' or 'demand' stats
# preprocess_train_test uses out-of-fold target encoding which is correct, but let's check
train_processed, test_processed, encoders = preprocess_train_test(train_df, test_df)

# Sort by day and time_of_day
train_processed['time_val'] = train_processed['day'] * 1440 + train_processed['time_of_day']
train_processed = train_processed.sort_values('time_val').reset_index(drop=True)

# Define temporal split: train on first 80%, validate on last 20%
split_idx = int(len(train_processed) * 0.8)
train_fold = train_processed.iloc[:split_idx]
val_fold = train_processed.iloc[split_idx:]

print(f"Train size: {len(train_fold)}, Val size: {len(val_fold)}")
print(f"Train day range: {train_fold['day'].min()} to {train_fold['day'].max()}")
print(f"Val day range: {val_fold['day'].min()} to {val_fold['day'].max()}")

X_train = train_fold[MODEL_FEATURES]
y_train = train_fold['demand']
X_val = val_fold[MODEL_FEATURES]
y_val = val_fold['demand']

# Evaluate CatBoost
cat = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=8, random_seed=42, verbose=200)
cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
pred_val_cat = cat.predict(X_val)
r2_cat = r2_score(y_val, pred_val_cat)
print(f"CatBoost Validation R2 (Temporal): {r2_cat:.6f}")

# Evaluate LightGBM
lgb = LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=128, max_depth=8, random_state=42, verbose=-1)
lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[
    __import__('lightgbm').early_stopping(100, verbose=False)
])
pred_val_lgb = lgb.predict(X_val)
r2_lgb = r2_score(y_val, pred_val_lgb)
print(f"LightGBM Validation R2 (Temporal): {r2_lgb:.6f}")
