"""
Offline training script for the IRCS room-state classifier.

Usage
-----
    python3 -m ml.train                          # uses SQLite database
    python3 -m ml.train --csv data/sensor_log.csv
    python3 -m ml.train --csv data/sensor_log.csv --benchmark

The script:
1. Loads labelled sensor data from a CSV file or the SQLite database.
2. Trains a Random Forest with stratified 5-fold cross-validation.
3. Fits a StandardScaler on the full training set.
4. Reports per-fold and mean accuracy, plus a final classification report.
5. Saves the model and scaler to disk (config.MODEL_PATH / SCALER_PATH).
6. Optionally benchmarks single-sample inference latency on the Pi.
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.metrics         import classification_report
from sklearn.pipeline        import Pipeline
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import MODEL_PATH, SCALER_PATH, DB_PATH, LABEL_MAP
from ml.feature_extractor import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_COL  = "label"
N_FOLDS     = 5
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
    cols = ", ".join(FEATURE_NAMES + [TARGET_COL])
    df   = pd.read_sql_query(
        f"SELECT {cols} FROM sensor_log WHERE label IS NOT NULL", conn
    )
    conn.close()
    logger.info("Loaded %d rows from %s", len(df), DB_PATH)
    return df


def train(df: pd.DataFrame, benchmark: bool = False) -> None:
    missing = [c for c in FEATURE_NAMES + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y_labels = df[TARGET_COL].values

    # Encode string labels to integers matching LABEL_MAP
    inv_label_map = {v: k for k, v in LABEL_MAP.items()}
    try:
        y = np.array([inv_label_map[lbl] for lbl in y_labels], dtype=int)
    except KeyError as e:
        raise ValueError(f"Unknown label in dataset: {e}. Expected: {list(LABEL_MAP.values())}")

    logger.info("Class distribution: %s",
                {LABEL_MAP[k]: int(np.sum(y == k)) for k in LABEL_MAP})

    # ── Scaler (fit on all data – we use CV for generalisation estimate) ─────
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Classifier ────────────────────────────────────────────────────────────
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    # ── Stratified 5-fold cross-validation ───────────────────────────────────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring="f1_macro", n_jobs=-1)
    logger.info("5-fold CV macro-F1: %.4f ± %.4f", fold_scores.mean(), fold_scores.std())

    # ── Final fit on full dataset ─────────────────────────────────────────────
    clf.fit(X_scaled, y)

    # Classification report on full training set (as a sanity check)
    y_pred  = clf.predict(X_scaled)
    report  = classification_report(y, y_pred, target_names=list(LABEL_MAP.values()))
    logger.info("Training-set classification report:\n%s", report)

    # ── Feature importance ────────────────────────────────────────────────────
    for name, imp in sorted(zip(FEATURE_NAMES, clf.feature_importances_),
                            key=lambda x: -x[1]):
        logger.info("  %-22s importance=%.4f", name, imp)

    # ── Inference latency benchmark ───────────────────────────────────────────
    if benchmark:
        sample = X_scaled[:1]
        runs   = 1000
        t0 = time.perf_counter()
        for _ in range(runs):
            clf.predict_proba(sample)
        elapsed_ms = (time.perf_counter() - t0) / runs * 1000
        logger.info("Inference latency: %.3f ms per sample (mean of %d runs)",
                    elapsed_ms, runs)

    # ── Persist artefacts ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH),  exist_ok=True)
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(clf,    MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Model  saved → %s", MODEL_PATH)
    logger.info("Scaler saved → %s", SCALER_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IRCS room-state classifier.")
    parser.add_argument("--csv",       default=None,        help="Path to labelled CSV.")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark inference latency.")
    args = parser.parse_args()

    df = load_from_csv(args.csv) if args.csv else load_from_db()
    train(df, benchmark=args.benchmark)


if __name__ == "__main__":
    main()

