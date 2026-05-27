import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from src.feature_engineering import load_data, preprocess_train_test, MODEL_FEATURES

train, test = load_data('data/train.csv','data/test.csv')
train, test, encoders = preprocess_train_test(train, test)
train['time_of_day'] = train['hour']*60 + train['minute']
features = MODEL_FEATURES + ['geo_prefix_te','weather_hour_te','landmarks_te','large_veh_te','hour','day','time_of_day']
print('features len', len(features))
print('extra features', ['geo_prefix_te','weather_hour_te','landmarks_te','large_veh_te','hour','day','time_of_day'])

kf = KFold(n_splits=3, shuffle=True, random_state=42)
scores=[]
for train_idx,val_idx in kf.split(train):
    X_train = train.iloc[train_idx][features]
    X_val = train.iloc[val_idx][features]
    y_train = train.iloc[train_idx]['demand']
    y_val = train.iloc[val_idx]['demand']
    model = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=8, loss_function='RMSE', verbose=False, random_seed=42)
    model.fit(X_train,y_train, eval_set=(X_val,y_val), use_best_model=True, early_stopping_rounds=20)
    scores.append(r2_score(y_val, model.predict(X_val)))
print('extended mean r2', np.mean(scores), scores)
