"""
Generate synthetic labelled sensor data for IRCS development and ML training.

Produces 14-feature rows matching the feature vector consumed by train.py.
Use this to bootstrap the classifier before you have enough real collected data.

Usage
-----
    python3 data/generate_synthetic_data.py               # 2000 rows → data/synthetic.csv
    python3 data/generate_synthetic_data.py --rows 5000 --out data/custom.csv
    python3 data/generate_synthetic_data.py --seed 42
"""

import argparse
import csv
import math
import os
import random
from datetime import datetime, timedelta

# ── 4-class physiologically-grounded profiles ─────────────────────────────────
# Each feature: (mean, half_range).  Time features are generated from a
# realistic hour distribution per state rather than stored as a profile value.
PROFILES = {
    "ROOM_EMPTY": {
        "temperature":    (19.0, 1.5),
        "humidity":       (48.0, 6.0),
        "pressure":       (1013.0, 4.0),
        "co2_ppm":        (420,   60),    # near-background CO2
        "lux":            (5.0,   4.0),   # lights off
        "distance":       (430.0, 50.0),  # no one detected
        "posture":        (-1,    0),     # unknown
        "flow_score":     (0.02,  0.02),  # no motion
        "ultrasonic_var": (2.0,   2.0),
        "lux_rate":       (0.0,   0.02),
    },
    "ACTIVE_AWAKE": {
        "temperature":    (22.5, 2.0),
        "humidity":       (53.0, 7.0),
        "pressure":       (1013.0, 4.0),
        "co2_ppm":        (750,  200),
        "lux":            (320.0, 80.0),  # lights on, daytime
        "distance":       (90.0,  40.0),
        "posture":        (0,     0),     # UPRIGHT
        "flow_score":     (0.55,  0.25),
        "ultrasonic_var": (120.0, 80.0),
        "lux_rate":       (0.0,   0.5),
    },
    "RESTING": {
        "temperature":    (20.5, 1.5),
        "humidity":       (56.0, 6.0),
        "pressure":       (1013.0, 4.0),
        "co2_ppm":        (620,  150),
        "lux":            (75.0, 30.0),   # dim reading light
        "distance":       (110.0, 30.0),
        "posture":        (1,     0),     # RECLINED
        "flow_score":     (0.08,  0.06),
        "ultrasonic_var": (15.0,  10.0),
        "lux_rate":       (0.0,   0.1),
    },
    "SLEEPING": {
        "temperature":    (18.5, 1.0),
        "humidity":       (61.0, 5.0),
        "pressure":       (1013.0, 4.0),
        "co2_ppm":        (540,  100),
        "lux":            (3.0,  2.5),    # near-darkness
        "distance":       (100.0, 20.0),
        "posture":        (2,     0),     # HORIZONTAL
        "flow_score":     (0.02,  0.02),
        "ultrasonic_var": (3.0,   3.0),
        "lux_rate":       (0.0,   0.01),
    },
}

# Realistic hour-of-day distributions per state (mean_hour, std_hours)
_HOUR_DIST = {
    "ROOM_EMPTY":   (14.0, 5.0),   # spread across day
    "ACTIVE_AWAKE": (11.0, 3.0),   # morning–afternoon
    "RESTING":      (15.0, 3.0),   # afternoon
    "SLEEPING":     (2.0,  2.5),   # middle of night
}

LABEL_WEIGHTS = {
    "ROOM_EMPTY":   0.25,
    "ACTIVE_AWAKE": 0.30,
    "RESTING":      0.25,
    "SLEEPING":     0.20,
}

FIELDNAMES = [
    "timestamp",
    "temperature", "humidity", "pressure", "co2_ppm", "lux",
    "distance", "posture", "flow_score",
    "ultrasonic_var", "lux_rate",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "label",
]


def _cyclic(hour: float, dow: int) -> tuple:
    h = (hour / 24) * 2 * math.pi
    d = (dow  / 7)  * 2 * math.pi
    return round(math.sin(h), 6), round(math.cos(h), 6), \
           round(math.sin(d), 6), round(math.cos(d), 6)


def generate_row(label: str, timestamp: datetime) -> dict:
    profile = PROFILES[label]
    row = {"timestamp": timestamp.isoformat(), "label": label}

    for field, (mean, hr) in profile.items():
        if hr == 0:
            row[field] = mean
        else:
            row[field] = round(mean + random.uniform(-hr, hr), 4)

    # Clamp to physically plausible ranges
    row["temperature"]    = max(-10.0, min(50.0,   row["temperature"]))
    row["humidity"]       = max(0.0,   min(100.0,  row["humidity"]))
    row["pressure"]       = max(900.0, min(1100.0, row["pressure"]))
    row["co2_ppm"]        = max(300,   min(5000,   int(row["co2_ppm"])))
    row["lux"]            = max(0.0,   min(100000, row["lux"]))
    row["distance"]       = max(2.0,   min(500.0,  row["distance"]))
    row["posture"]        = int(row["posture"])
    row["flow_score"]     = max(0.0,   min(1.0,    row["flow_score"]))
    row["ultrasonic_var"] = max(0.0,               row["ultrasonic_var"])

    # Realistic circadian time
    mean_h, std_h = _HOUR_DIST[label]
    hour = max(0.0, min(23.99, random.gauss(mean_h, std_h))) if std_h else mean_h
    # Wrap night hours (SLEEPING can go negative → wrap to late night)
    hour = hour % 24
    dow  = random.randint(0, 6)
    row["hour_sin"], row["hour_cos"], row["dow_sin"], row["dow_cos"] = _cyclic(hour, dow)

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic IRCS sensor data.")
    parser.add_argument("--rows",  type=int, default=2000, help="Number of rows to generate.")
    parser.add_argument("--out",   default=os.path.join(os.path.dirname(__file__), "synthetic.csv"))
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    labels      = list(LABEL_WEIGHTS.keys())
    weights     = list(LABEL_WEIGHTS.values())
    start_time  = datetime(2026, 1, 1, 0, 0, 0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for i in range(args.rows):
            label     = random.choices(labels, weights=weights, k=1)[0]
            timestamp = start_time + timedelta(seconds=10 * i)
            writer.writerow(generate_row(label, timestamp))

    counts = {l: 0 for l in labels}
    print(f"Written {args.rows} rows to {args.out}")
    # Re-read to count
    with open(args.out) as f:
        for row in csv.DictReader(f):
            counts[row["label"]] += 1
    for lbl, cnt in counts.items():
        print(f"  {lbl:<14}: {cnt:5d}  ({cnt/args.rows*100:.1f}%)")


if __name__ == "__main__":
    main()


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
