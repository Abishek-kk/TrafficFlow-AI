import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

LABEL_COLS = [
    'geohash',
    'geo_prefix',
    'RoadType',
    'LargeVehicles',
    'Landmarks',
    'Weather'
]

TARGET_GROUPS = {
    'geo_te': ['geohash'],
    'geo_prefix_te': ['geo_prefix'],
    'hour_te': ['hour'],
    'day_te': ['day'],
    'weather_te': ['Weather'],
    'road_te': ['RoadType'],
    'geo_hour_te': ['geohash', 'hour'],
    'geo_time_te': ['geohash', 'time_of_day'],
    'geo_day_te': ['geohash', 'day'],
    'weather_hour_te': ['Weather', 'hour'],
    'landmarks_te': ['Landmarks'],
    'large_veh_te': ['LargeVehicles'],
}

MODEL_FEATURES = [
    'geo_te',
    'geo_prefix_te',
    'hour_te',
    'day_te',
    'weather_te',
    'road_te',
    'geo_hour_te',
    'geo_time_te',
    'geo_day_te',
    'weather_hour_te',
    'landmarks_te',
    'large_veh_te',
    'geohash_le',
    'geo_prefix_le',
    'RoadType_le',
    'LargeVehicles_le',
    'Landmarks_le',
    'Weather_le',
    'hour',
    'day',
    'time_of_day',
    'sin_time',
    'cos_time',
    'NumberofLanes',
    'Temperature',
    'geo_count',
    'geo_hour_count',
    'geo_day_count'
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
    df['sin_time'] = np.sin(
        2 * np.pi * df['time_of_day'] / 1440
    )
    df['cos_time'] = np.cos(
        2 * np.pi * df['time_of_day'] / 1440
    )

    return df


def create_geo_features(df):
    """
    Add geohash-based location features.
    """
    df = df.copy()
    df['geo_prefix'] = df['geohash'].astype(str).str[:4]
    return df


def build_count_maps(df):
    """
    Build mapping dictionaries for count-based group features.
    """
    return {
        'geo_count': df['geohash'].value_counts().to_dict(),
        'geo_hour_count': df.groupby(['geohash', 'hour']).size().to_dict(),
        'geo_day_count': df.groupby(['geohash', 'day']).size().to_dict()
    }


def apply_count_maps(df, count_maps):
    """
    Apply count-based mappings to new data.
    """
    df = df.copy()
    df['geo_count'] = df['geohash'].map(count_maps['geo_count']).fillna(0).astype(int)
    df['geo_hour_count'] = (
        df[['geohash', 'hour']]
        .apply(tuple, axis=1)
        .map(count_maps['geo_hour_count'])
        .fillna(0)
        .astype(int)
    )
    df['geo_day_count'] = (
        df[['geohash', 'day']]
        .apply(tuple, axis=1)
        .map(count_maps['geo_day_count'])
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

    encoders = {
        'label_encoders': label_encoders,
        'target_encoders': target_encoders,
        'target_mean': target_mean,
        'count_maps': count_maps
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
    return test


if __name__ == '__main__':
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'

    train, test = load_data(train_path, test_path)
    train, test, encoders = preprocess_train_test(train, test)
    print('Shape after preprocess:', train.shape, test.shape)
    print(train.head())
