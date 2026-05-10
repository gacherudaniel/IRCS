"""
Offline training script for the IRCS room-state classifier.

Usage
-----
    python3 -m ml.train                          # uses default data path
    python3 -m ml.train --csv data/sensor_log.csv

The script:
1. Loads labelled sensor data from a CSV file (or the SQLite database).
2. Splits into train/test sets.
3. Trains a Random Forest classifier.
4. Fits a StandardScaler on the training split.
5. Evaluates on the test set and prints a classification report.
6. Saves the model and scaler to disk (config.MODEL_PATH / SCALER_PATH).
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble       import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler, LabelEncoder
from sklearn.metrics        import classification_report
import joblib

# Allow running as a script from the ircs/ root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import MODEL_PATH, SCALER_PATH, DB_PATH, LABEL_MAP
from ml.feature_extractor import FEATURE_KEYS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_COL  = "label"
TEST_SIZE   = 0.20
RANDOM_SEED = 42


def load_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def load_from_db() -> pd.DataFrame:
    import sqlite3
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT temperature, humidity, pressure, air_quality, ldr, "
        "distance, occupancy, label FROM sensor_log WHERE label IS NOT NULL",
        conn,
    )
    conn.close()
    logger.info("Loaded %d rows from database %s", len(df), DB_PATH)
    return df


def train(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURE_KEYS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    X = df[FEATURE_KEYS].values.astype(np.float32)
    y_raw = df[TARGET_COL].values

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(list(LABEL_MAP.values()))
    y = label_encoder.transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    report = classification_report(
        y_test, y_pred, target_names=list(LABEL_MAP.values())
    )
    logger.info("Classification report:\n%s", report)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf,    MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Model saved  → %s", MODEL_PATH)
    logger.info("Scaler saved → %s", SCALER_PATH)

    importances = clf.feature_importances_
    for name, imp in sorted(zip(FEATURE_KEYS, importances), key=lambda x: -x[1]):
        logger.info("  Feature %-20s importance=%.4f", name, imp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IRCS room classifier.")
    parser.add_argument("--csv", default=None, help="Path to labelled CSV file.")
    args = parser.parse_args()

    if args.csv:
        df = load_from_csv(args.csv)
    else:
        df = load_from_db()

    train(df)


if __name__ == "__main__":
    main()
