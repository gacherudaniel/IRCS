"""
Preprocessing pipeline for IRCS collected sensor data.

Reads one or more raw CSV files produced by data/collect.py (or
data/generate_synthetic_data.py) and outputs a single clean CSV
ready to be consumed by ml/train.py.

Steps performed
---------------
1. Load and merge all input files
2. Validate required columns are present
3. Drop rows with any missing values and report count
4. Clip each feature to its physically plausible range
5. Remove outliers using the IQR method (per-class, per-feature)
6. Report final class distribution
7. Write clean CSV

Usage
-----
    # Single file
    python3 data/preprocess.py --input data/collected.csv

    # Merge multiple collection sessions + synthetic data
    python3 data/preprocess.py \\
        --input data/session_sleeping.csv \\
                data/session_active.csv \\
                data/synthetic.csv \\
        --out   data/training_data.csv

    # Adjust outlier aggressiveness (default IQR factor = 2.5)
    python3 data/preprocess.py --input data/collected.csv --iqr-factor 3.0
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml.feature_extractor import FEATURE_NAMES

TARGET_COL  = "label"
VALID_LABELS = {"ROOM_EMPTY", "ACTIVE_AWAKE", "RESTING", "SLEEPING"}

# ── Physical plausible ranges per feature ────────────────────────────────────
# Values outside these hard limits are physically impossible and are clipped.
FEATURE_BOUNDS: dict[str, tuple] = {
    "temperature":    (-10.0,  50.0),
    "humidity":       (  0.0, 100.0),
    "pressure":       (900.0, 1100.0),
    "co2_ppm":        (300.0, 5000.0),
    "lux":            (  0.0, 100000.0),
    "distance":       (  2.0,  500.0),
    "posture":        ( -1.0,    2.0),
    "flow_score":     (  0.0,    1.0),
    "ultrasonic_var": (  0.0, 50000.0),
    "lux_rate":       (-5000.0, 5000.0),
    "hour_sin":       ( -1.0,    1.0),
    "hour_cos":       ( -1.0,    1.0),
    "dow_sin":        ( -1.0,    1.0),
    "dow_cos":        ( -1.0,    1.0),
}

# Features to skip during IQR outlier removal (they are bounded by design)
IQR_SKIP = {"posture", "flow_score", "hour_sin", "hour_cos", "dow_sin", "dow_cos"}


def load(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not os.path.isfile(path):
            print(f"[preprocess] WARNING: file not found – {path}")
            continue
        df = pd.read_csv(path)
        print(f"  Loaded {len(df):>6} rows from {path}")
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No valid input files found.")
    merged = pd.concat(frames, ignore_index=True)
    print(f"  Total after merge: {len(merged)} rows\n")
    return merged


def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = set(FEATURE_NAMES) | {TARGET_COL}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    # Drop unknown labels
    before = len(df)
    df = df[df[TARGET_COL].isin(VALID_LABELS)].copy()
    dropped = before - len(df)
    if dropped:
        print(f"[preprocess] Dropped {dropped} rows with unrecognised labels.")
    return df


def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=FEATURE_NAMES + [TARGET_COL])
    dropped = before - len(df)
    print(f"[preprocess] Dropped {dropped} rows with missing values. ({len(df)} remain)")
    return df


def clip_ranges(df: pd.DataFrame) -> pd.DataFrame:
    for feat, (lo, hi) in FEATURE_BOUNDS.items():
        if feat in df.columns:
            clipped = ((df[feat] < lo) | (df[feat] > hi)).sum()
            if clipped:
                print(f"[preprocess] Clipping {clipped} out-of-range values in '{feat}'")
            df[feat] = df[feat].clip(lower=lo, upper=hi)
    # posture must be integer
    df["posture"] = df["posture"].round().astype(int)
    return df


def remove_outliers(df: pd.DataFrame, iqr_factor: float) -> pd.DataFrame:
    """
    Per-class IQR outlier removal.  For each label group and each numeric
    feature (excluding bounded/categorical ones), rows beyond
    [Q1 - factor*IQR, Q3 + factor*IQR] are removed.
    """
    before = len(df)
    keep   = pd.Series(True, index=df.index)

    numeric_feats = [f for f in FEATURE_NAMES if f not in IQR_SKIP]

    for label in df[TARGET_COL].unique():
        mask  = df[TARGET_COL] == label
        group = df.loc[mask, numeric_feats]

        q1  = group.quantile(0.25)
        q3  = group.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr

        in_range = ((group >= lower) & (group <= upper)).all(axis=1)
        keep.loc[mask & ~in_range] = False

    df = df[keep].copy()
    print(f"[preprocess] IQR outlier removal (factor={iqr_factor}): "
          f"removed {before - len(df)} rows. ({len(df)} remain)")
    return df


def report_distribution(df: pd.DataFrame) -> None:
    total = len(df)
    print("\n[preprocess] Final class distribution:")
    for label in sorted(VALID_LABELS):
        count = (df[TARGET_COL] == label).sum()
        pct   = count / total * 100 if total else 0
        bar   = "█" * int(pct / 2)
        print(f"  {label:<14} {count:>5} rows  ({pct:5.1f}%)  {bar}")
    print(f"  {'TOTAL':<14} {total:>5} rows\n")
    if total < 300:
        print("[preprocess] WARNING: fewer than 300 rows – consider collecting more data.")
    for label in VALID_LABELS:
        count = (df[TARGET_COL] == label).sum()
        if count < 50:
            print(f"[preprocess] WARNING: only {count} rows for '{label}' – "
                  f"aim for ≥ 50 per class.")


def preprocess(input_paths: list[str], out_path: str, iqr_factor: float) -> None:
    print("=" * 60)
    print("[preprocess] Loading data…")
    df = load(input_paths)

    print("[preprocess] Validating columns…")
    df = validate_columns(df)

    print("[preprocess] Dropping null rows…")
    df = drop_nulls(df)

    print("[preprocess] Clipping physical ranges…")
    df = clip_ranges(df)

    print("[preprocess] Removing outliers…")
    df = remove_outliers(df, iqr_factor)

    report_distribution(df)

    # Keep only the columns train.py needs
    out_cols = FEATURE_NAMES + [TARGET_COL]
    if "timestamp" in df.columns:
        out_cols = ["timestamp"] + out_cols

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df[out_cols].to_csv(out_path, index=False)
    print(f"[preprocess] Clean dataset written to {out_path}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess IRCS collected sensor data for model training."
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="One or more raw CSV files to merge and clean.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "training_data.csv"),
        help="Output path for cleaned CSV (default: data/training_data.csv).",
    )
    parser.add_argument(
        "--iqr-factor", type=float, default=2.5,
        help="IQR multiplier for outlier removal (default: 2.5). "
             "Higher = keep more data; lower = stricter.",
    )
    args = parser.parse_args()
    preprocess(args.input, args.out, args.iqr_factor)


if __name__ == "__main__":
    main()
