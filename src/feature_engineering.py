import pandas as pd
import numpy as np
try:
    import geohash
except ImportError:
    import geohash2 as geohash
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

LABEL_COLS = [
    'geohash',
    'geo_prefix4',
    'geo_prefix5',
    'geo_prefix6',
    'RoadType',
    'LargeVehicles',
    'Landmarks',
    'Weather'
]

# Target encoding groups — NO day-based groupings (test is unseen day)
TARGET_GROUPS = {
    # Single-feature target encodings
    'geo_te': ['geohash'],
    'geo_prefix4_te': ['geo_prefix4'],
    'geo_prefix5_te': ['geo_prefix5'],
    'geo_prefix6_te': ['geo_prefix6'],
    'hour_te': ['hour'],
    'quarter_te': ['quarter'],
    'time_block_te': ['time_block'],
    'weather_te': ['Weather'],
    'road_te': ['RoadType'],
    'lanes_te': ['NumberofLanes'],
    'landmarks_te': ['Landmarks'],
    'large_veh_te': ['LargeVehicles'],

    # 2-way interactions
    'geo_hour_te': ['geohash', 'hour'],
    'geo_quarter_te': ['geohash', 'quarter'],
    'geo_time_block_te': ['geohash', 'time_block'],
    'geo_weather_te': ['geohash', 'Weather'],
    'geo_road_te': ['geohash', 'RoadType'],
    'geo_lanes_te': ['geohash', 'NumberofLanes'],
    'geo_landmarks_te': ['geohash', 'Landmarks'],
    'geo_large_veh_te': ['geohash', 'LargeVehicles'],

    'weather_hour_te': ['Weather', 'hour'],
    'road_hour_te': ['RoadType', 'hour'],
    'lanes_hour_te': ['NumberofLanes', 'hour'],
    'landmarks_hour_te': ['Landmarks', 'hour'],
    'large_veh_hour_te': ['LargeVehicles', 'hour'],

    'geo_prefix4_hour_te': ['geo_prefix4', 'hour'],
    'geo_prefix5_hour_te': ['geo_prefix5', 'hour'],
    'geo_prefix6_hour_te': ['geo_prefix6', 'hour'],
    'prefix_hour_te': ['geo_prefix4', 'hour'],
    'geo_prefix_time_te': ['geo_prefix4', 'time_block'],
    'geo_prefix5_weather_te': ['geo_prefix5', 'Weather'],
    'geo_prefix4_weather_te': ['geo_prefix4', 'Weather'],

    # New 2-way interactions
    'geo_prefix5_road_te': ['geo_prefix5', 'RoadType'],
    'geo_prefix4_lanes_te': ['geo_prefix4', 'NumberofLanes'],
    'lanes_weather_te': ['NumberofLanes', 'Weather'],
    'road_weather_te': ['RoadType', 'Weather'],
    'landmarks_weather_te': ['Landmarks', 'Weather'],
    'large_veh_weather_te': ['LargeVehicles', 'Weather'],
    'geo_temp_bin_te': ['geohash', 'temp_bin'],
    'hour_temp_bin_te': ['hour', 'temp_bin'],

    # 3-way interactions
    'weather_road_hour_te': ['Weather', 'RoadType', 'hour'],
    'geo_weather_hour_te': ['geohash', 'Weather', 'hour'],
    'geo_road_hour_te': ['geohash', 'RoadType', 'hour'],
    'geo_lanes_hour_te': ['geohash', 'NumberofLanes', 'hour'],
    'geo_landmarks_hour_te': ['geohash', 'Landmarks', 'hour'],
}

MODEL_FEATURES = [
    # Target encoding features
    'geo_te',
    'geo_prefix4_te',
    'geo_prefix5_te',
    'geo_prefix6_te',
    'hour_te',
    'quarter_te',
    'time_block_te',
    'weather_te',
    'road_te',
    'lanes_te',
    'landmarks_te',
    'large_veh_te',

    'geo_hour_te',
    'geo_quarter_te',
    'geo_time_block_te',
    'geo_weather_te',
    'geo_road_te',
    'geo_lanes_te',
    'geo_landmarks_te',
    'geo_large_veh_te',

    'weather_hour_te',
    'road_hour_te',
    'lanes_hour_te',
    'landmarks_hour_te',
    'large_veh_hour_te',

    'geo_prefix4_hour_te',
    'geo_prefix5_hour_te',
    'geo_prefix6_hour_te',
    'prefix_hour_te',
    'geo_prefix_time_te',
    'geo_prefix5_weather_te',
    'geo_prefix4_weather_te',

    'geo_prefix5_road_te',
    'geo_prefix4_lanes_te',
    'lanes_weather_te',
    'road_weather_te',
    'landmarks_weather_te',
    'large_veh_weather_te',
    'geo_temp_bin_te',
    'hour_temp_bin_te',

    'weather_road_hour_te',
    'geo_weather_hour_te',
    'geo_road_hour_te',
    'geo_lanes_hour_te',
    'geo_landmarks_hour_te',

    # Label encoded features
    'geohash_le',
    'geo_prefix4_le',
    'geo_prefix5_le',
    'geo_prefix6_le',
    'RoadType_le',
    'LargeVehicles_le',
    'Landmarks_le',
    'Weather_le',

    # Time features
    'hour',
    'minute',
    'time_of_day',
    'quarter',
    'time_block',
    'is_morning',
    'is_evening',
    'is_peak',
    'peak_hour',
    'is_weekend',
    'is_night',
    'is_midday',
    'sin_time',
    'cos_time',
    'sin_minute',
    'cos_minute',
    'sin_hour',
    'cos_hour',

    # Geohash coordinates
    'geohash_lat',
    'geohash_lon',

    # Numeric features
    'NumberofLanes',
    'Temperature',
    'day',
    'temp_bin',
    'lanes_x_hour',

    # Count features
    'geo_count',
    'geo_prefix4_count',
    'geo_prefix5_count',
    'geo_prefix6_count',
    'geo_hour_count',
    'geo_prefix4_hour_count',
    'geo_weather_count',
    'geo_road_count',

    # Statistical aggregations
    'geo_demand_std',
    'geo_demand_median',
    'geo_demand_min',
    'geo_demand_max',
    'geo_demand_range',
    'geo_demand_q25',
    'geo_demand_q75',
    'geo_demand_iqr',
    'geo_demand_skew',
    'geo_demand_cv',

    'geo_prefix4_demand_std',
    'geo_prefix4_demand_median',
    'geo_prefix4_demand_range',
    'geo_prefix4_demand_q25',
    'geo_prefix4_demand_q75',

    'geo_prefix5_demand_std',
    'geo_prefix5_demand_median',

    'hour_demand_std',
    'hour_demand_median',

    'geo_hour_demand_mean',
    'geo_hour_demand_std',
    'geo_hour_demand_median',

    'weather_hour_demand_mean',
    'weather_hour_demand_median',

    'geo_weather_demand_mean',
    'geo_weather_demand_std',
]

# Lag and rolling features will be added for modeling
LAG_FEATURES = [
    'lag_1',
    'lag_4',
    'lag_96',
    'roll_mean_4',
    'roll_std_4',
    'roll_mean_8',
    'roll_std_8',
]

# expose in MODEL_FEATURES
MODEL_FEATURES += LAG_FEATURES


def load_data(train_path, test_path):
    """
    Load train and test datasets
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print("Train Shape:", train.shape)
    print("Test Shape:", test.shape)

    return train, test


def fill_missing_values(df, stats=None):
    """
    Fill missing values for numeric and categorical columns.

    If `stats` is provided, use train-derived imputation statistics for
    `Temperature` and `Weather` based on `(geohash, hour)` groups.
    """
    df = df.copy()

    if stats is not None and 'geohash' in df.columns and 'hour' in df.columns:
        idx = pd.MultiIndex.from_arrays([df['geohash'], df['hour']])
        temp_map = pd.Series(idx.map(stats['temp_median']), index=df.index)
        df['Temperature'] = df['Temperature'].fillna(temp_map.astype(float))
        df['Temperature'] = df['Temperature'].fillna(stats['temp_global_median'])

        weather_map = pd.Series(idx.map(stats['weather_mode']), index=df.index)
        df['Weather'] = df['Weather'].fillna(weather_map)
        if pd.isna(stats['weather_global_mode']):
            df['Weather'] = df['Weather'].fillna('Unknown')
        else:
            df['Weather'] = df['Weather'].fillna(stats['weather_global_mode'])

    for col in df.columns:
        if col in ['Temperature', 'Weather']:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def build_imputation_stats(df):
    """
    Build train-derived imputation statistics for Temperature and Weather.
    """
    stats = {}
    stats['temp_median'] = df.groupby(['geohash', 'hour'])['Temperature'].median()
    stats['temp_global_median'] = df['Temperature'].median()

    stats['weather_mode'] = df.groupby(['geohash', 'hour'])['Weather'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
    )
    weather_mode = df['Weather'].mode()
    stats['weather_global_mode'] = weather_mode.iloc[0] if len(weather_mode) > 0 else np.nan

    return stats


def create_time_features(df):
    """
    Create time-based features from the timestamp column.
    """
    df = df.copy()

    df['timestamp'] = pd.to_datetime(
        df['timestamp'],
        format='%H:%M',
        errors='coerce'
    )

    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['time_of_day'] = df['hour'] * 60 + df['minute']
    df['quarter'] = (df['time_of_day'] // 15).astype(int)
    df['time_block'] = (df['time_of_day'] // 60).astype(int)

    # Cyclical encoding for time_of_day
    df['sin_time'] = np.sin(
        2 * np.pi * df['time_of_day'] / 1440
    )
    df['cos_time'] = np.cos(
        2 * np.pi * df['time_of_day'] / 1440
    )

    # Cyclical encoding for minute
    df['sin_minute'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['cos_minute'] = np.cos(2 * np.pi * df['minute'] / 60)

    # Cyclical encoding for hour
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Boolean time flags
    df['is_morning'] = df['hour'].between(6, 11).astype(int)
    df['is_evening'] = df['hour'].between(16, 20).astype(int)
    df['is_peak'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df['peak_hour'] = df['is_peak']
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_midday'] = df['hour'].between(10, 14).astype(int)
    df['is_weekend'] = (df['day'].astype(int) % 7).isin([5, 6]).astype(int)

    # Interaction feature
    df['lanes_x_hour'] = df['NumberofLanes'] * df['hour']

    return df


def create_geo_features(df):
    """
    Add geohash-based location features.
    """
    df = df.copy()
    df['geo_prefix4'] = df['geohash'].astype(str).str[:4]
    df['geo_prefix5'] = df['geohash'].astype(str).str[:5]
    df['geo_prefix6'] = df['geohash'].astype(str).str[:6]

    def decode_center(code):
        if pd.isna(code) or str(code).strip() == '':
            return (np.nan, np.nan)
        lat, lon = geohash.decode(str(code))
        return (float(lat), float(lon))

    coords = df['geohash'].apply(decode_center)
    df['geohash_lat'] = coords.apply(lambda x: x[0]).astype(float)
    df['geohash_lon'] = coords.apply(lambda x: x[1]).astype(float)
    return df


def create_temp_features(df):
    """
    Create temperature-based features.
    """
    df = df.copy()
    # Temperature bins (5 bins)
    df['temp_bin'] = pd.cut(
        df['Temperature'],
        bins=[-20, 5, 15, 25, 35, 55],
        labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(2).astype(int)
    return df


def build_count_maps(df):
    """
    Build mapping dictionaries for count-based group features.
    """
    return {
        'geo_count': df['geohash'].value_counts().to_dict(),
        'geo_prefix4_count': df['geo_prefix4'].value_counts().to_dict(),
        'geo_prefix5_count': df['geo_prefix5'].value_counts().to_dict(),
        'geo_prefix6_count': df['geo_prefix6'].value_counts().to_dict(),
        'geo_hour_count': df.groupby(['geohash', 'hour']).size().to_dict(),
        'geo_prefix4_hour_count': df.groupby(['geo_prefix4', 'hour']).size().to_dict(),
        'geo_weather_count': df.groupby(['geohash', 'Weather']).size().to_dict(),
        'geo_road_count': df.groupby(['geohash', 'RoadType']).size().to_dict(),
    }


def apply_count_maps(df, count_maps):
    """
    Apply count-based mappings to new data.
    """
    df = df.copy()
    df['geo_count'] = df['geohash'].map(count_maps['geo_count']).fillna(0).astype(int)
    df['geo_prefix4_count'] = df['geo_prefix4'].map(count_maps['geo_prefix4_count']).fillna(0).astype(int)
    df['geo_prefix5_count'] = df['geo_prefix5'].map(count_maps['geo_prefix5_count']).fillna(0).astype(int)
    df['geo_prefix6_count'] = df['geo_prefix6'].map(count_maps['geo_prefix6_count']).fillna(0).astype(int)
    df['geo_hour_count'] = (
        df[['geohash', 'hour']]
        .apply(tuple, axis=1)
        .map(count_maps['geo_hour_count'])
        .fillna(0)
        .astype(int)
    )
    df['geo_prefix4_hour_count'] = (
        df[['geo_prefix4', 'hour']]
        .apply(tuple, axis=1)
        .map(count_maps['geo_prefix4_hour_count'])
        .fillna(0)
        .astype(int)
    )
    df['geo_weather_count'] = (
        df[['geohash', 'Weather']]
        .apply(tuple, axis=1)
        .map(count_maps['geo_weather_count'])
        .fillna(0)
        .astype(int)
    )
    df['geo_road_count'] = (
        df[['geohash', 'RoadType']]
        .apply(tuple, axis=1)
        .map(count_maps['geo_road_count'])
        .fillna(0)
        .astype(int)
    )
    return df


def fit_label_encoders(train, test=None):
    """
    Fit label encoders on training data, optionally including test data to avoid unseen categories.
    """
    encoders = {}

    for col in LABEL_COLS:
        le = LabelEncoder()
        if test is not None:
            combined = pd.concat([
                train[col].astype(str),
                test[col].astype(str)
            ], axis=0)
            le.fit(combined)
        else:
            le.fit(train[col].astype(str))

        encoders[col] = le

    return encoders


def transform_label_column(series, le):
    """
    Transform a column using a fitted LabelEncoder and handle unseen labels.
    """
    mapping = {
        label: idx
        for idx, label in enumerate(le.classes_)
    }

    return series.astype(str).map(mapping).fillna(-1).astype(int)


def apply_label_encoders(df, encoders):
    """
    Apply label encoding to the dataset.
    """
    df = df.copy()

    for col, le in encoders.items():
        df[col + '_le'] = transform_label_column(df[col], le)

    return df


def build_target_encoding_map(df, target_col, keys):
    """
    Build a target encoding mapping for a group key combination.
    """
    mapping = {}
    grouped = df.groupby(keys)[target_col].mean()

    for key, value in grouped.items():
        if not isinstance(key, tuple):
            key = (key,)
        mapping[key] = value

    return mapping


def oof_target_encoding(df, target_col, keys, n_splits=5):
    """
    Create an out-of-fold target encoding feature for training.
    """
    oof = np.zeros(len(df))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    target_mean = df[target_col].mean()

    for train_idx, val_idx in kf.split(df):
        tr = df.iloc[train_idx]
        val = df.iloc[val_idx]
        mapping = build_target_encoding_map(tr, target_col, keys)
        keyvals = val[keys].apply(tuple, axis=1)
        oof[val_idx] = keyvals.map(mapping).fillna(target_mean)

    return oof


def add_target_encoding_features(df, target_col='demand'):
    """
    Add target encoding features to the training dataset.
    """
    df = df.copy()
    encoders = {}

    for name, keys in TARGET_GROUPS.items():
        df[name] = oof_target_encoding(
            df,
            target_col,
            keys
        )
        encoders[name] = {
            'keys': keys,
            'mapping': build_target_encoding_map(
                df,
                target_col,
                keys
            )
        }

    return df, encoders


def apply_target_encoding_features(df, target_encoders, fallback):
    """
    Apply saved target encoding mappings to new data.
    """
    df = df.copy()

    for name, meta in target_encoders.items():
        keys = meta['keys']
        mapping = meta['mapping']
        keyvals = df[keys].apply(tuple, axis=1)
        df[name] = keyvals.map(mapping).fillna(fallback)

    return df


def build_stat_maps(df, target_col='demand'):
    """
    Build statistical aggregation maps from training data.
    Returns mappings for std, median, min, max, quantiles, skew, cv per group.
    """
    stat_maps = {}

    # Per-geohash stats
    geo_group = df.groupby('geohash')[target_col]
    stat_maps['geo_demand_std'] = geo_group.std().fillna(0).to_dict()
    stat_maps['geo_demand_median'] = geo_group.median().to_dict()
    stat_maps['geo_demand_min'] = geo_group.min().to_dict()
    stat_maps['geo_demand_max'] = geo_group.max().to_dict()
    stat_maps['geo_demand_q25'] = geo_group.quantile(0.25).to_dict()
    stat_maps['geo_demand_q75'] = geo_group.quantile(0.75).to_dict()
    stat_maps['geo_demand_skew'] = geo_group.skew().fillna(0).to_dict()
    geo_mean = geo_group.mean()
    geo_std = geo_group.std().fillna(0)
    stat_maps['geo_demand_cv'] = (geo_std / (geo_mean + 1e-8)).to_dict()

    # Per-geo_prefix4 stats
    gp4_group = df.groupby('geo_prefix4')[target_col]
    stat_maps['geo_prefix4_demand_std'] = gp4_group.std().fillna(0).to_dict()
    stat_maps['geo_prefix4_demand_median'] = gp4_group.median().to_dict()
    stat_maps['geo_prefix4_demand_q25'] = gp4_group.quantile(0.25).to_dict()
    stat_maps['geo_prefix4_demand_q75'] = gp4_group.quantile(0.75).to_dict()
    stat_maps['geo_prefix4_demand_min'] = gp4_group.min().to_dict()
    stat_maps['geo_prefix4_demand_max'] = gp4_group.max().to_dict()

    # Per-geo_prefix5 stats
    gp5_group = df.groupby('geo_prefix5')[target_col]
    stat_maps['geo_prefix5_demand_std'] = gp5_group.std().fillna(0).to_dict()
    stat_maps['geo_prefix5_demand_median'] = gp5_group.median().to_dict()

    # Per-hour stats
    hour_group = df.groupby('hour')[target_col]
    stat_maps['hour_demand_std'] = hour_group.std().fillna(0).to_dict()
    stat_maps['hour_demand_median'] = hour_group.median().to_dict()

    # Per-geohash-hour stats
    geo_hour_group = df.groupby(['geohash', 'hour'])[target_col]
    stat_maps['geo_hour_demand_mean'] = geo_hour_group.mean().to_dict()
    stat_maps['geo_hour_demand_std'] = geo_hour_group.std().fillna(0).to_dict()
    stat_maps['geo_hour_demand_median'] = geo_hour_group.median().to_dict()

    # Per-weather-hour stats
    weather_hour_group = df.groupby(['Weather', 'hour'])[target_col]
    stat_maps['weather_hour_demand_mean'] = weather_hour_group.mean().to_dict()
    stat_maps['weather_hour_demand_median'] = weather_hour_group.median().to_dict()

    # Per-geohash-weather stats
    geo_weather_group = df.groupby(['geohash', 'Weather'])[target_col]
    stat_maps['geo_weather_demand_mean'] = geo_weather_group.mean().to_dict()
    stat_maps['geo_weather_demand_std'] = geo_weather_group.std().fillna(0).to_dict()

    return stat_maps


def add_lag_and_rolling(df, demand_col='demand', lags=(1, 4, 96), roll_windows=(4, 8)):
    """
    Add lag features and rolling mean/std per `geohash`.

    - `lag_N`: demand shifted by N timesteps (15-min steps)
    - `roll_mean_W` / `roll_std_W`: rolling statistics over previous W timesteps (exclude current)

    Assumes `day` and `quarter` columns exist (quarter in 0..95 per day).
    """
    df = df.copy()

    # create a monotonic time index: day * 96 + quarter (96 15-min steps per day)
    if 'quarter' not in df.columns or 'day' not in df.columns:
        raise ValueError('DataFrame must contain `day` and `quarter` columns before adding lags')

    df['time_index'] = df['day'].astype(int) * 96 + df['quarter'].astype(int)

    # sort so shifts and rollings are aligned
    df = df.sort_values(['geohash', 'time_index']).reset_index(drop=True)

    # compute simple lags
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('geohash')[demand_col].shift(lag)

    # compute rolling stats over previous steps (exclude current step)
    for w in roll_windows:
        # shift by 1 so rolling window looks at previous W values
        df[f'roll_mean_{w}'] = (
            df.groupby('geohash')[demand_col]
            .apply(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        df[f'roll_std_{w}'] = (
            df.groupby('geohash')[demand_col]
            .apply(lambda x: x.shift(1).rolling(window=w, min_periods=1).std())
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

    # Fill NA lags/rolls with the overall demand mean to avoid missing values downstream
    fill_value = df[demand_col].mean()
    for col in [f'lag_{l}' for l in lags] + [f'roll_mean_{w}' for w in roll_windows] + [f'roll_std_{w}' for w in roll_windows]:
        df[col] = df[col].fillna(fill_value)

    # drop helper column
    df = df.drop(columns=['time_index'])
    return df


def apply_stat_maps(df, stat_maps):
    """
    Apply statistical aggregation maps to a dataframe.
    """
    df = df.copy()

    # Per-geohash
    df['geo_demand_std'] = df['geohash'].map(stat_maps['geo_demand_std']).fillna(0)
    df['geo_demand_median'] = df['geohash'].map(stat_maps['geo_demand_median']).fillna(0)
    df['geo_demand_min'] = df['geohash'].map(stat_maps['geo_demand_min']).fillna(0)
    df['geo_demand_max'] = df['geohash'].map(stat_maps['geo_demand_max']).fillna(0)
    df['geo_demand_range'] = df['geo_demand_max'] - df['geo_demand_min']
    df['geo_demand_q25'] = df['geohash'].map(stat_maps['geo_demand_q25']).fillna(0)
    df['geo_demand_q75'] = df['geohash'].map(stat_maps['geo_demand_q75']).fillna(0)
    df['geo_demand_iqr'] = df['geo_demand_q75'] - df['geo_demand_q25']
    df['geo_demand_skew'] = df['geohash'].map(stat_maps['geo_demand_skew']).fillna(0)
    df['geo_demand_cv'] = df['geohash'].map(stat_maps['geo_demand_cv']).fillna(0)

    # Per-geo_prefix4
    df['geo_prefix4_demand_std'] = df['geo_prefix4'].map(stat_maps['geo_prefix4_demand_std']).fillna(0)
    df['geo_prefix4_demand_median'] = df['geo_prefix4'].map(stat_maps['geo_prefix4_demand_median']).fillna(0)
    df['geo_prefix4_demand_q25'] = df['geo_prefix4'].map(stat_maps['geo_prefix4_demand_q25']).fillna(0)
    df['geo_prefix4_demand_q75'] = df['geo_prefix4'].map(stat_maps['geo_prefix4_demand_q75']).fillna(0)
    gp4_min = df['geo_prefix4'].map(stat_maps['geo_prefix4_demand_min']).fillna(0)
    gp4_max = df['geo_prefix4'].map(stat_maps['geo_prefix4_demand_max']).fillna(0)
    df['geo_prefix4_demand_range'] = gp4_max - gp4_min

    # Per-geo_prefix5
    df['geo_prefix5_demand_std'] = df['geo_prefix5'].map(stat_maps['geo_prefix5_demand_std']).fillna(0)
    df['geo_prefix5_demand_median'] = df['geo_prefix5'].map(stat_maps['geo_prefix5_demand_median']).fillna(0)

    # Per-hour
    df['hour_demand_std'] = df['hour'].map(stat_maps['hour_demand_std']).fillna(0)
    df['hour_demand_median'] = df['hour'].map(stat_maps['hour_demand_median']).fillna(0)

    # Per-geohash-hour
    geo_hour_keys = df[['geohash', 'hour']].apply(tuple, axis=1)
    df['geo_hour_demand_mean'] = geo_hour_keys.map(stat_maps['geo_hour_demand_mean']).fillna(0)
    df['geo_hour_demand_std'] = geo_hour_keys.map(stat_maps['geo_hour_demand_std']).fillna(0)
    df['geo_hour_demand_median'] = geo_hour_keys.map(stat_maps['geo_hour_demand_median']).fillna(0)

    # Per-weather-hour
    wh_keys = df[['Weather', 'hour']].apply(tuple, axis=1)
    df['weather_hour_demand_mean'] = wh_keys.map(stat_maps['weather_hour_demand_mean']).fillna(0)
    df['weather_hour_demand_median'] = wh_keys.map(stat_maps['weather_hour_demand_median']).fillna(0)

    # Per-geohash-weather
    gw_keys = df[['geohash', 'Weather']].apply(tuple, axis=1)
    df['geo_weather_demand_mean'] = gw_keys.map(stat_maps['geo_weather_demand_mean']).fillna(0)
    df['geo_weather_demand_std'] = gw_keys.map(stat_maps['geo_weather_demand_std']).fillna(0)

    return df


def preprocess_train_test(train, test):
    """
    Preprocess training and test datasets with matching feature engineering.
    """
    train = create_time_features(train)
    test = create_time_features(test)

    impute_stats = build_imputation_stats(train)
    train = fill_missing_values(train, stats=impute_stats)
    test = fill_missing_values(test, stats=impute_stats)

    train = create_geo_features(train)
    test = create_geo_features(test)

    train = create_temp_features(train)
    test = create_temp_features(test)

    # Add lag and rolling features — compute on combined chronological data so
    # test rows can inherit past demand from training where available.
    train['_is_train'] = 1
    test['_is_train'] = 0
    combined = pd.concat([train, test], ignore_index=True, sort=False)
    combined = add_lag_and_rolling(combined, demand_col='demand')
    # split back
    train = combined[combined['_is_train'] == 1].drop(columns=['_is_train']).reset_index(drop=True)
    test = combined[combined['_is_train'] == 0].drop(columns=['_is_train']).reset_index(drop=True)

    # Save minimal train snapshot to allow computing lags for future test sets
    train_for_lags = train[['geohash', 'day', 'quarter', 'demand']].copy()

    count_maps = build_count_maps(train)
    train = apply_count_maps(train, count_maps)
    test = apply_count_maps(test, count_maps)

    label_encoders = fit_label_encoders(train, test)
    train = apply_label_encoders(train, label_encoders)
    test = apply_label_encoders(test, label_encoders)

    train, target_encoders = add_target_encoding_features(train)
    target_mean = train['demand'].mean()
    test = apply_target_encoding_features(
        test,
        target_encoders,
        target_mean
    )

    # Build and apply statistical aggregation features
    stat_maps = build_stat_maps(train, 'demand')
    train = apply_stat_maps(train, stat_maps)
    test = apply_stat_maps(test, stat_maps)

    encoders = {
        'label_encoders': label_encoders,
        'target_encoders': target_encoders,
        'target_mean': target_mean,
        'count_maps': count_maps,
        'stat_maps': stat_maps,
        'impute_stats': impute_stats
    }

    # include train snapshot for lagging when preprocessing unseen test data
    encoders['train_for_lags'] = train_for_lags

    return train, test, encoders


def preprocess_test(test, encoders):
    """
    Preprocess test data using saved encoders.
    """
    test = create_time_features(test)
    test = fill_missing_values(test, stats=encoders.get('impute_stats'))
    test = create_geo_features(test)
    test = create_temp_features(test)
    test = apply_count_maps(test, encoders['count_maps'])
    test = apply_label_encoders(test, encoders['label_encoders'])
    test = apply_target_encoding_features(
        test,
        encoders['target_encoders'],
        encoders['target_mean']
    )
    test = apply_stat_maps(test, encoders['stat_maps'])
    # If we have a train snapshot, build lags by concatenating train snapshot + test
    if 'train_for_lags' in encoders:
        train_snap = encoders['train_for_lags'].copy()
        train_snap['_is_train'] = 1
        test_copy = test.copy()
        test_copy['_is_train'] = 0
        combined = pd.concat([train_snap, test_copy], ignore_index=True, sort=False)
        combined = add_lag_and_rolling(combined, demand_col='demand')
        # extract test portion
        test = combined[combined['_is_train'] == 0].drop(columns=['_is_train']).reset_index(drop=True)

        # ensure label/stat/target encoded columns still present (they should be)
        # fill any remaining NaNs in lag features
        for col in LAG_FEATURES:
            if col in test.columns:
                test[col] = test[col].fillna(0)

    return test


if __name__ == '__main__':
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'

    train, test = load_data(train_path, test_path)
    train, test, encoders = preprocess_train_test(train, test)
    print('Shape after preprocess:', train.shape, test.shape)
    print('Number of model features:', len(MODEL_FEATURES))
    print(train.head())
