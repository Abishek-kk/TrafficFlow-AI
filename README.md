# Traffic Demand Prediction

A complete traffic demand forecasting pipeline with Python, CatBoost, and engineered features.

##  Project Summary

This project predicts traffic demand using:

- timestamp information (`timestamp`, `day`, `hour`, cyclic time features)
- location features (`geohash`, geohash prefixes)
- road characteristics (`RoadType`, `NumberofLanes`, `LargeVehicles`, `Landmarks`)
- weather and temperature data
- target-encoded group features for high-cardinality geohash/time combinations

The pipeline trains a CatBoost regression model and generates submission-ready predictions for test data.

##  Key Results

The improved feature pipeline is designed to achieve above `90%` validation `R2`. Experimental validation performance has reached:

- `~94.8%` R2 for the core engineered feature set
- `~95.8%` R2 when adding raw encoded categorical columns

> These results are based on the current dataset and feature engineering setup.

##  Repository Structure

| Path | Description |
|---|---|
| `main.py` | Runs the full pipeline: train + predict |
| `src/feature_engineering.py` | Data preprocessing, feature creation, and encoding |
| `src/train_model.py` | Model training, validation, saving, and test prediction |
| `src/predict.py` | Load model/encoders and generate predictions for `test.csv` |
| `requirements.txt` | Python dependencies |
| `data/` | `train.csv`, `test.csv`, and dataset files |
| `models/` | Saved model and encoder artifacts |
| `outputs/` | Generated submission output |

## Prerequisites

Make sure Python is installed, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## ▶ How to Run

From the repository root, execute:

```bash
python main.py
```

This will:

1. create required folders: `models`, `outputs`, `logs`
2. load `data/train.csv` and `data/test.csv`
3. preprocess and engineer features
4. train the CatBoost regression model
5. evaluate validation performance and print `R2`
6. retrain the model on the full training set
7. save model and encoder artifacts to `models/`
8. generate predictions and save `outputs/submission.csv`

##  Data Requirements

The pipeline expects:

- `data/train.csv` — training data including the target column `demand`
- `data/test.csv` — test data for prediction

### Required columns in `train.csv`

- `Index`
- `geohash`
- `day`
- `timestamp`
- `demand`
- `RoadType`
- `NumberofLanes`
- `LargeVehicles`
- `Landmarks`
- `Temperature`
- `Weather`

### Required columns in `test.csv`

- Same columns as training data except `demand`

##  Notes for Developers

- The pipeline uses target encoding to capture high-cardinality location/time patterns.
- `src/feature_engineering.py` now creates both cyclic time features and grouped target encodings.
- `main.py` is configured to import from `src/` even when executed from the repository root.
- If you want to run just prediction after training, use `src/predict.py` directly.

##  Common Troubleshooting

- If `catboost` is missing:

```bash
python -m pip install catboost
```

- If imports fail, ensure you are in the repository root and run from there.
- If you see `PYTHONPATH` issues, add `src/` to the path or run from the root directory.

## Tips for Improvement

To further improve model performance, consider:

- adding external traffic/weather context data
- using geospatial distance or decoded latitude/longitude
- handling time series ordering explicitly
- using additional target encodings or regularization techniques

##  License

This repository is provided for educational and experimental use.