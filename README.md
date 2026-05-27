# Traffic Demand Prediction

A complete traffic demand forecasting pipeline with Python, CatBoost, LightGBM, and advanced engineered features.

##  Project Summary

This project predicts traffic demand using:

- **Timestamp & Cyclical Features**: `timestamp`, `day`, `hour`, `minute`, and cyclical sine/cosine time transformations.
- **Location Features**: `geohash`, geohash prefixes (lengths 4, 5, and 6), and count metrics.
- **Road Characteristics**: `RoadType`, `NumberofLanes`, `LargeVehicles`, and `Landmarks`.
- **Weather & Temperature**: Real-time weather categories and temperature values.
- **Advanced Target-Encoded Interactions**: Out-of-fold target encodings for complex high-cardinality interactions (e.g., location × weather, location × road, location × day/hour).
- **Statistical Aggregations**: Standard deviation, median, and range values of historical demand grouped by location and time.

The pipeline trains a robust, weighted ensemble of **CatBoost** and **LightGBM** models using 5-Fold Cross-Validation.

##  Key Results

Through extensive feature engineering and ensembling, the model's accuracy (R² Score) has been significantly improved:

- **Original Performance**: `~89.68%` R²
- **New Ensemble Performance**: **`~96.2%` R²** (Validation Cross-Validation Score)

##  Repository Structure

| Path | Description |
|---|---|
| `main.py` | Runs the full pipeline: training the ensemble + predicting test data |
| `src/feature_engineering.py` | Preprocessing, interaction target encodings, count maps, and statistical aggregations |
| `src/train_model.py` | 5-Fold CV training, out-of-fold weighting, full-dataset retraining, and artifact saving |
| `src/predict.py` | Load ensemble models/weights/encoders and predict traffic demand for `test.csv` |
| `requirements.txt` | Python dependencies (pandas, numpy, scikit-learn, catboost, lightgbm, joblib) |
| `data/` | Data folder containing `train.csv` and `test.csv` |
| `models/` | Saved models (`catboost_model.pkl`, `lightgbm_model.pkl`), weights, and encoder artifacts |
| `outputs/` | Generated submission predictions (`submission.csv`) |

## 🛠️ Prerequisites

Make sure Python is installed, then install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

##  How to Run

To run the entire pipeline (train the models and generate predictions):

```bash
python main.py
```

This will automatically:
1. Create the required directories (`models`, `outputs`, `logs`).
2. Preprocess the datasets and generate engineered interaction features.
3. Perform 5-Fold Cross Validation for both CatBoost and LightGBM models.
4. Calculate optimal weights for each model based on out-of-fold validation scores.
5. Retrain the final models on the entire dataset using the best iterations.
6. Save the trained models, encoders, and ensemble weights to the `models/` folder.
7. Generate and save the final predictions to `outputs/submission.csv`.

##  Data Requirements

The pipeline expects two files in the `data/` directory:

- `data/train.csv` — training data with target column `demand`
- `data/test.csv` — test data for prediction

### Required columns:
`Index`, `geohash`, `day`, `timestamp`, `RoadType`, `NumberofLanes`, `LargeVehicles`, `Landmarks`, `Temperature`, `Weather`

##  Notes for Developers

- **Target Encoding**: All target encodings are computed out-of-fold (OOF) during training to prevent data leakage.
- **Ensemble Inference**: If you only want to generate predictions using already trained models, you can run `src/predict.py` directly.
- **Error Handling**: Missing values in numeric columns are filled with median values, and categorical columns are filled with mode values.
