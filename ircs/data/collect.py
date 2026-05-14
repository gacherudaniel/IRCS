"""
Real sensor data collection script for IRCS model training.

Run a separate session for each context state, e.g.:

    # Collect 10 minutes of SLEEPING data
    python3 data/collect.py --label SLEEPING --duration 600

    # Collect 15 minutes of ACTIVE_AWAKE data
    python3 data/collect.py --label ACTIVE_AWAKE --duration 900

    # Append to an existing file instead of creating a new one
    python3 data/collect.py --label RESTING --duration 600 --out data/collected.csv

Valid labels: ROOM_EMPTY, ACTIVE_AWAKE, RESTING, SLEEPING

Each row written contains all 10 model features plus the timestamp and label.
The rolling-window feature (lux_rate) stabilises after the first 30 seconds,
so the first few rows may be less representative.

Recommended collection protocol
---------------------------------
State           | Min duration  | Suggested sessions
ROOM_EMPTY      | 10 min        | Various times of day, lights off + on
ACTIVE_AWAKE    | 15 min        | Occupant walking, conversing, exercises
RESTING         | 10 min        | Occupant sitting/reading in chair
SLEEPING        | 20 min        | Occupant lying down, lights off

Aim for ≥ 300 rows per class before training (≈ 50 min total collection).
"""

import argparse
import csv
import os
import signal
import sys
import time
from datetime import datetime

# Allow running from the ircs/ root directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sensors.bmp280_sensor import BMP280Sensor
from sensors.adc           import ADCSensor
from sensors.air_quality   import AirQualitySensor
from sensors.ldr           import LDRSensor
from sensors.camera        import CameraSensor
from ml.feature_extractor  import FeatureExtractor, FEATURE_NAMES
from config                import SENSOR_POLL_INTERVAL

VALID_LABELS = {"ROOM_EMPTY", "ACTIVE_AWAKE", "RESTING", "SLEEPING"}

FIELDNAMES = ["timestamp"] + FEATURE_NAMES + ["label"]

_running = True


def _handle_sigint(sig, frame):
    global _running
    print("\n[collect] Interrupted – flushing and closing file.")
    _running = False


def _init_sensors():
    return {
        "bmp280":      BMP280Sensor(),
        "adc":         ADCSensor(),
        "air_quality": AirQualitySensor(),
        "ldr":         LDRSensor(),
        "camera":      CameraSensor(),
    }


_last_humidity: float = float("nan")


def _read_sensors(sensors: dict) -> dict:
    """Read all sensors and return a raw reading dict."""
    cv_data = sensors["camera"].analyse()
    flow_score = cv_data.get("flow_score", 0.0)

    return {
        "temperature": sensors["bmp280"].read_temperature(),
        "pressure":    sensors["bmp280"].read_pressure(),
        "co2_ppm":     sensors["air_quality"].read_ppm(),
        "lux":         sensors["ldr"].read_lux(),
        "flow_score":  flow_score,
    }


def collect(label: str, duration: int, out_path: str) -> None:
    signal.signal(signal.SIGINT, _handle_sigint)

    file_exists = os.path.isfile(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print(f"[collect] Initialising sensors…")
    sensors   = _init_sensors()
    extractor = FeatureExtractor()

    print(f"[collect] Recording label='{label}' for {duration}s → {out_path}")
    print(f"[collect] Press Ctrl+C to stop early.\n")

    # Warm-up: let rolling window fill before writing data
    WARMUP_CYCLES = max(1, 30 // SENSOR_POLL_INTERVAL)
    print(f"[collect] Warming up rolling window ({WARMUP_CYCLES} cycles)…")

    end_time   = time.monotonic() + duration
    rows_written = 0

    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        cycle = 0
        while _running and time.monotonic() < end_time:
            cycle_start = time.monotonic()

            try:
                reading  = _read_sensors(sensors)
                now      = datetime.now()
                features = extractor.extract(reading, now)

                if cycle >= WARMUP_CYCLES:
                    row = {"timestamp": now.isoformat(), "label": label}
                    for i, name in enumerate(FEATURE_NAMES):
                        row[name] = round(float(features[0, i]), 6)
                    writer.writerow(row)
                    f.flush()
                    rows_written += 1

                    remaining = int(end_time - time.monotonic())
                    print(
                        f"  [{rows_written:>4} rows | {remaining:>4}s left] "
                        f"T={reading['temperature']:.1f}°C  "
                        f"CO2={reading['co2_ppm']}ppm  "
                        f"lux={reading['lux']:.0f}  "
                        f"flow={reading['flow_score']:.2f}",
                        end="\r",
                    )
                else:
                    print(f"  Warming up… cycle {cycle+1}/{WARMUP_CYCLES}", end="\r")

            except Exception as exc:
                print(f"\n[collect] Sensor error (skipping cycle): {exc}")

            cycle += 1
            elapsed = time.monotonic() - cycle_start
            sleep_for = max(0.0, SENSOR_POLL_INTERVAL - elapsed)
            time.sleep(sleep_for)

    print(f"\n[collect] Done. {rows_written} rows written to {out_path}")

    # Clean up camera resources
    if hasattr(sensors["camera"], "release"):
        sensors["camera"].release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect labelled IRCS sensor data for model training."
    )
    parser.add_argument(
        "--label",
        required=True,
        choices=sorted(VALID_LABELS),
        help="Ground-truth context state label for this session.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=600,
        help="Recording duration in seconds (default: 600 = 10 min).",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "collected.csv"),
        help="Output CSV path (appended if it already exists).",
    )
    args = parser.parse_args()

    collect(args.label, args.duration, args.out)


if __name__ == "__main__":
    main()
