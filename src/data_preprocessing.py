
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(train_path, test_path):
    """
    Load train and test datasets
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print("Train Shape:", train.shape)
    print("Test Shape:", test.shape)

    return train, test


def preprocess_data(train, test):
    """
    Complete preprocessing pipeline
    """

    train = train.copy()
    test = test.copy()

    # Missing Value Handling
    for col in train.columns:
        if train[col].dtype == 'object':
            train[col] = train[col].fillna(train[col].mode()[0])

    for col in test.columns:
        if test[col].dtype == 'object':
            test[col] = test[col].fillna(test[col].mode()[0])

    for col in train.select_dtypes(include=np.number).columns:
        train[col] = train[col].fillna(train[col].median())

    for col in test.select_dtypes(include=np.number).columns:
        test[col] = test[col].fillna(test[col].median())

    # Remove Duplicates
    train.drop_duplicates(inplace=True)

    # Timestamp Feature Engineering
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])

    for df in [train, test]:
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['peak_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        df.drop('timestamp', axis=1, inplace=True)

    # Label Encoding
    cat_cols = train.select_dtypes(include='object').columns
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()

        combined = pd.concat(
            [train[col], test[col]],
            axis=0
        ).astype(str)

        le.fit(combined)

        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

        encoders[col] = le

    print("Preprocessing Completed")

    return train, test, encoders


def split_features_target(train):
    """
    Split features and target
    """
    X = train.drop('demand', axis=1)
    y = train['demand']
    return X, y


if __name__ == "__main__":
    train_path = "data/train.csv"
    test_path = "data/test.csv"

    train, test = load_data(train_path, test_path)
    train, test, encoders = preprocess_data(train, test)
    X, y = split_features_target(train)

    print("Feature Shape:", X.shape)
    print("Target Shape:", y.shape)
