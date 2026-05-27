import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from src.feature_engineering import load_data, preprocess_train_test, MODEL_FEATURES

train, test = load_data('data/train.csv','data/test.csv')
train, test, encoders = preprocess_train_test(train, test)
X = train[MODEL_FEATURES]
y = train['demand']
print('FEATURES', len(MODEL_FEATURES), MODEL_FEATURES)
print('train shape', X.shape)

kf = KFold(n_splits=3, shuffle=True, random_state=42)
for name, cols in [('current', MODEL_FEATURES), ('extended', MODEL_FEATURES + ['hour','day','time_of_day'])]:
    Xc = train[cols]
    scores=[]
    for train_idx,val_idx in kf.split(Xc):
        X_train,X_val = Xc.iloc[train_idx], Xc.iloc[val_idx]
        y_train,y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=8, loss_function='RMSE', verbose=False, random_seed=42)
        model.fit(X_train,y_train, eval_set=(X_val,y_val), use_best_model=True, early_stopping_rounds=20)
        scores.append(r2_score(y_val, model.predict(X_val)))
    print(name, 'mean r2', np.mean(scores), scores)
