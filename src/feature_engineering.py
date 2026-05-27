import pandas as pd
import numpy as np
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

# We MUST remove all target encodings that group by 'day'.
# This is because the test set is day 49 (which has no daytime history in the train set).
# Day-based target encodings will result in serious feature mismatch / leakage.
TARGET_GROUPS = {
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
    
    # Interactions (without day)
    'geo_hour_te': ['geohash', 'hour'],
    'geo_quarter_te': ['geohash', 'quarter'],
    'geo_time_block_te': ['geohash', 'time_block'],
    'geo_weather_te': ['geohash', 'Weather'],
    'geo_road_te': ['geohash', 'RoadType'],
    'geo_lanes_te': ['geohash', 'NumberofLanes'],
    
    'weather_hour_te': ['Weather', 'hour'],
    'road_hour_te': ['RoadType', 'hour'],
    'lanes_hour_te': ['NumberofLanes', 'hour'],
    
    'geo_prefix4_hour_te': ['geo_prefix4', 'hour'],
    'geo_prefix5_hour_te': ['geo_prefix5', 'hour'],
    'geo_prefix6_hour_te': ['geo_prefix6', 'hour'],
    'geo_prefix_time_te': ['geo_prefix4', 'time_block'],
    'geo_prefix5_weather_te': ['geo_prefix5', 'Weather'],
    'geo_prefix4_weather_te': ['geo_prefix4', 'Weather'],
    
    'weather_road_hour_te': ['Weather', 'RoadType', 'hour'],
    'landmarks_hour_te': ['Landmarks', 'hour'],
    'large_veh_hour_te': ['LargeVehicles', 'hour'],
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
    
    'weather_hour_te',
    'road_hour_te',
    'lanes_hour_te',
    
    'geo_prefix4_hour_te',
    'geo_prefix5_hour_te',
    'geo_prefix6_hour_te',
    'geo_prefix_time_te',
    'geo_prefix5_weather_te',
    'geo_prefix4_weather_te',
    
    'weather_road_hour_te',
    'landmarks_hour_te',
    'large_veh_hour_te',
    
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
    'is_night',
    'sin_time',
    'cos_time',
    'sin_minute',
    'cos_minute',
    'sin_hour',
    'cos_hour',
    
    # Numeric features
    'NumberofLanes',
    'Temperature',
    
    # Count features (no day-based counts)
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
    'geo_prefix4_demand_std',
    'geo_prefix4_demand_median',
    'hour_demand_std',
    'hour_demand_median',
    'geo_demand_range',
    'geo_prefix4_demand_range',
]


def load_data(train_path, test_path):
    """
    Load train and test datasets
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print("Train Shape:", train.shape)
    print("Test Shape:", test.shape)

    return train, test


def fill_missing_values(df):
    """
    Fill missing values for numeric and categorical columns.
    """
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


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
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

    return df


def create_geo_features(df):
    """
    Add geohash-based location features.
    """
    df = df.copy()
    df['geo_prefix4'] = df['geohash'].astype(str).str[:4]
    df['geo_prefix5'] = df['geohash'].astype(str).str[:5]
    df['geo_prefix6'] = df['geohash'].astype(str).str[:6]
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
    Returns mappings for std, median, min, max per group.
    """
    stat_maps = {}

    # Per-geohash stats
    geo_group = df.groupby('geohash')[target_col]
    stat_maps['geo_demand_std'] = geo_group.std().fillna(0).to_dict()
    stat_maps['geo_demand_median'] = geo_group.median().to_dict()
    stat_maps['geo_demand_min'] = geo_group.min().to_dict()
    stat_maps['geo_demand_max'] = geo_group.max().to_dict()

    # Per-geo_prefix4 stats
    gp4_group = df.groupby('geo_prefix4')[target_col]
    stat_maps['geo_prefix4_demand_std'] = gp4_group.std().fillna(0).to_dict()
    stat_maps['geo_prefix4_demand_median'] = gp4_group.median().to_dict()

    # Per-hour stats
    hour_group = df.groupby('hour')[target_col]
    stat_maps['hour_demand_std'] = hour_group.std().fillna(0).to_dict()
    stat_maps['hour_demand_median'] = hour_group.median().to_dict()

    return stat_maps


def apply_stat_maps(df, stat_maps):
    """
    Apply statistical aggregation maps to a dataframe.
    """
    df = df.copy()

    df['geo_demand_std'] = df['geohash'].map(stat_maps['geo_demand_std']).fillna(0)
    df['geo_demand_median'] = df['geohash'].map(stat_maps['geo_demand_median']).fillna(0)
    df['geo_demand_min'] = df['geohash'].map(stat_maps['geo_demand_min']).fillna(0)
    df['geo_demand_max'] = df['geohash'].map(stat_maps['geo_demand_max']).fillna(0)
    df['geo_demand_range'] = df['geo_demand_max'] - df['geo_demand_min']

    df['geo_prefix4_demand_std'] = df['geo_prefix4'].map(stat_maps['geo_prefix4_demand_std']).fillna(0)
    df['geo_prefix4_demand_median'] = df['geo_prefix4'].map(stat_maps['geo_prefix4_demand_median']).fillna(0)

    df['hour_demand_std'] = df['hour'].map(stat_maps['hour_demand_std']).fillna(0)
    df['hour_demand_median'] = df['hour'].map(stat_maps['hour_demand_median']).fillna(0)

    # Compute geo_prefix4_demand_range properly as max - min
    gp4_max = {k: 0 for k in stat_maps['geo_prefix4_demand_std']}
    gp4_min = {k: 1 for k in stat_maps['geo_prefix4_demand_std']}
    # Approximate range from median and std
    for k in stat_maps['geo_prefix4_demand_std']:
        med = stat_maps['geo_prefix4_demand_median'].get(k, 0)
        std = stat_maps['geo_prefix4_demand_std'].get(k, 0)
        gp4_max[k] = med + 2 * std
        gp4_min[k] = max(0, med - 2 * std)
    df['geo_prefix4_demand_range'] = df['geo_prefix4'].map(
        {k: gp4_max[k] - gp4_min[k] for k in gp4_max}
    ).fillna(0)

    return df


def preprocess_train_test(train, test):
    """
    Preprocess training and test datasets with matching feature engineering.
    """
    train = fill_missing_values(train)
    test = fill_missing_values(test)

    train = create_time_features(train)
    test = create_time_features(test)

    train = create_geo_features(train)
    test = create_geo_features(test)

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
        'stat_maps': stat_maps
    }

    return train, test, encoders


def preprocess_test(test, encoders):
    """
    Preprocess test data using saved encoders.
    """
    test = fill_missing_values(test)
    test = create_time_features(test)
    test = create_geo_features(test)
    test = apply_count_maps(test, encoders['count_maps'])
    test = apply_label_encoders(test, encoders['label_encoders'])
    test = apply_target_encoding_features(
        test,
        encoders['target_encoders'],
        encoders['target_mean']
    )
    test = apply_stat_maps(test, encoders['stat_maps'])
    return test


if __name__ == '__main__':
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'

    train, test = load_data(train_path, test_path)
    train, test, encoders = preprocess_train_test(train, test)
    print('Shape after preprocess:', train.shape, test.shape)
    print('Number of model features:', len(MODEL_FEATURES))
    print(train.head())
