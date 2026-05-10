"""
Generate synthetic sensor data for IRCS development and ML training.

Usage
-----
    python3 data/generate_synthetic_data.py               # 2000 rows → data/synthetic.csv
    python3 data/generate_synthetic_data.py --rows 5000 --out data/custom.csv
    python3 data/generate_synthetic_data.py --seed 99
"""

import argparse
import os
import random
import csv
from datetime import datetime, timedelta, timezone

# ── Room-state profiles ────────────────────────────────────────────────────────
# Each profile describes mean ± half-range for each sensor dimension.
# Format: (mean, half_range)
PROFILES = {
    "empty": {
        "temperature": (21.0, 1.5),
        "humidity":    (45.0, 5.0),
        "pressure":    (1013.0, 5.0),
        "altitude":    (100.0, 2.0),
        "air_quality": (3000,  800),   # ADS1115 raw (0-32767); low = clean air
        "ldr":         (500.0, 200.0), # lux; lights off → very low
        "distance":    (350.0, 50.0),  # cm; large open space
        "occupancy":   (0.05,  0),     # rarely flagged
    },
    "occupied": {
        "temperature": (23.0, 2.0),
        "humidity":    (52.0, 7.0),
        "pressure":    (1013.0, 5.0),
        "altitude":    (100.0, 2.0),
        "air_quality": (14000, 3000),
        "ldr":         (8000.0, 1500.0),  # lux; lights on
        "distance":    (120.0, 40.0),
        "occupancy":   (0.85,  0),
    },
    "high_activity": {
        "temperature": (26.5, 2.5),
        "humidity":    (62.0, 8.0),
        "pressure":    (1013.0, 5.0),
        "altitude":    (100.0, 2.0),
        "air_quality": (24000, 5000),
        "ldr":         (12000.0, 2000.0),
        "distance":    (80.0,  30.0),
        "occupancy":   (0.97,  0),
    },
}

LABEL_WEIGHTS = {"empty": 0.35, "occupied": 0.45, "high_activity": 0.20}

FIELDNAMES = [
    "timestamp", "temperature", "humidity", "pressure", "altitude",
    "air_quality", "ldr", "distance", "occupancy", "label",
]


def sample_value(mean: float, half_range: float, is_int: bool = False):
    """Uniform sample centred on mean with given half-range."""
    if half_range == 0:
        # Binary probability
        val = 1 if random.random() < mean else 0
        return val
    val = mean + random.uniform(-half_range, half_range)
    return int(round(val)) if is_int else round(val, 2)


def generate_row(label: str, timestamp: datetime) -> dict:
    profile = PROFILES[label]
    row = {"timestamp": timestamp.isoformat(), "label": label}
    for field, (mean, hr) in profile.items():
        is_int = field in ("air_quality", "occupancy")
        row[field] = sample_value(mean, hr, is_int=is_int)
    # Clamp values to physically sensible bounds
    row["humidity"]    = max(0.0,   min(100.0, row["humidity"]))
    row["temperature"] = max(-10.0, min(50.0,  row["temperature"]))
    row["air_quality"] = max(0,     min(32767,  row["air_quality"]))
    row["ldr"]         = max(0.0,   min(200000.0, row["ldr"]))
    row["distance"]    = max(2.0,   min(500.0,  row["distance"]))
    return row


def generate(n_rows: int, out_path: str, seed: int) -> None:
    random.seed(seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    labels  = list(LABEL_WEIGHTS.keys())
    weights = list(LABEL_WEIGHTS.values())
    start   = datetime(2025, 1, 1, tzinfo=timezone.utc)
    delta   = timedelta(seconds=5)

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
        for i in range(n_rows):
            label = random.choices(labels, weights=weights, k=1)[0]
            ts    = start + delta * i
            writer.writerow(generate_row(label, ts))

    print(f"Generated {n_rows} rows → {out_path}")
    _print_summary(out_path)


def _print_summary(path: str) -> None:
    import collections
    counts: dict = collections.Counter()
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts[row["label"]] += 1
    total = sum(counts.values())
    print("\nLabel distribution:")
    for label, count in counts.most_common():
        print(f"  {label:<16} {count:>5}  ({100*count/total:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic IRCS sensor data.")
    parser.add_argument("--rows", type=int, default=2000, help="Number of rows to generate.")
    parser.add_argument("--out",  default=os.path.join(os.path.dirname(__file__), "synthetic.csv"),
                        help="Output CSV file path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    generate(args.rows, args.out, args.seed)
