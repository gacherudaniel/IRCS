"""
Populate ircs.db with 2 weeks of realistic synthetic sensor data.

Run from the ircs/ directory:
    python data/populate_db.py

Generates one row every 10 minutes from 2025-04-30 00:00 UTC
through 2025-05-13 23:50 UTC  (14 days × 144 rows/day = 2016 rows).
"""

import os
import random
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH  # noqa: E402  (must follow sys.path insert)

SEED = 42
random.seed(SEED)

# ── Time range ────────────────────────────────────────────────────────────────
START   = datetime(2025, 4, 30, 0, 0, 0, tzinfo=timezone.utc)
DAYS    = 14
STEP    = timedelta(minutes=10)
TOTAL   = DAYS * 24 * 6   # 144 samples/day

LABELS  = ["SLEEPING", "RESTING", "ACTIVE_AWAKE", "ROOM_EMPTY"]

# ── Day-of-week absence probability (0 = always home, 1 = always out) ────────
DOW_OUT_PROB = {0: 0.15, 1: 0.40, 2: 0.40, 3: 0.40,
                4: 0.40, 5: 0.25, 6: 0.15}   # Mon-Sun


def _gauss(mu: float, sigma: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, random.gauss(mu, sigma)))


def _label(hour: int, minute: int, is_out: bool) -> str:
    if is_out:
        return "ROOM_EMPTY"
    if 0 <= hour < 7 or (hour == 22 and minute >= 30) or hour == 23:
        return "SLEEPING"
    if 7 <= hour < 9:
        return "ACTIVE_AWAKE"
    if 9 <= hour < 12:
        return random.choice(["ACTIVE_AWAKE", "ACTIVE_AWAKE", "RESTING"])
    if 12 <= hour < 13:
        return random.choice(["RESTING", "ACTIVE_AWAKE"])
    if 13 <= hour < 18:
        return random.choice(["RESTING", "ACTIVE_AWAKE", "RESTING"])
    if 18 <= hour < 21:
        return random.choice(["ACTIVE_AWAKE", "RESTING"])
    # 21-22:30
    return random.choice(["RESTING", "RESTING", "SLEEPING"])


def _readings(ts: datetime, label: str) -> dict:
    hour = ts.hour

    # Temperature: warmest midday, cooler overnight
    temp_base = 22.0 + 3.5 * max(0.0, 1.0 - abs(hour - 14) / 10.0)
    temperature = _gauss(temp_base, 0.4, 18.0, 30.0)

    # Pressure: gentle diurnal drift
    pressure = _gauss(850.0 + 0.8 * (1.0 - abs(hour - 12) / 12.0), 0.3, 845.0, 856.0)
    altitude  = _gauss(1455.0, 2.0, 1440.0, 1470.0)

    # Air quality: CO2 rises with activity, drops when empty/sleeping
    aq_map = {
        "SLEEPING":     _gauss(560, 40, 420, 680),
        "RESTING":      _gauss(680, 60, 500, 850),
        "ACTIVE_AWAKE": _gauss(760, 80, 550, 980),
        "ROOM_EMPTY":   _gauss(440, 30, 400, 510),
    }
    air_quality = int(aq_map[label])

    # Light: dark at night, bright during day/activity
    if label == "SLEEPING":
        lux = _gauss(2.0, 1.5, 0.1, 10.0)
    elif label == "ROOM_EMPTY":
        # curtains may be open or closed
        lux = _gauss(200, 400, 0.1, 1800) if 8 <= hour <= 17 else _gauss(1.0, 0.5, 0.1, 5.0)
    elif label == "ACTIVE_AWAKE":
        lux = _gauss(600, 200, 50, 3000) if hour >= 7 else _gauss(5, 3, 0.5, 20)
    else:  # RESTING
        lux = _gauss(150, 80, 5, 600)
    lux = max(0.1, lux)

    # Flow score: proportional to activity
    flow_map = {
        "SLEEPING":     _gauss(0.02, 0.02, 0.0, 0.08),
        "RESTING":      _gauss(0.12, 0.05, 0.0, 0.25),
        "ACTIVE_AWAKE": _gauss(0.45, 0.15, 0.10, 0.90),
        "ROOM_EMPTY":   _gauss(0.01, 0.01, 0.0, 0.04),
    }
    flow_score = round(max(0.0, min(1.0, flow_map[label])), 4)

    explanation = {
        "SLEEPING":     "Low light and minimal motion suggest the occupant is asleep.",
        "RESTING":      "Moderate light and low motion indicate a resting state.",
        "ACTIVE_AWAKE": "Elevated CO\u2082 and high motion confirm active occupancy.",
        "ROOM_EMPTY":   "No motion detected and stable CO\u2082 indicate an unoccupied room.",
    }[label]

    return {
        "temperature": round(temperature, 2),
        "pressure":    round(pressure,    2),
        "altitude":    round(altitude,    1),
        "air_quality": air_quality,
        "ldr":         round(lux,         2),
        "flow_score":  flow_score,
        "label":       label,
        "explanation": explanation,
    }


_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS sensor_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT    NOT NULL,
    temperature  REAL,
    pressure     REAL,
    altitude     REAL,
    air_quality  INTEGER,
    ldr          REAL,
    flow_score   REAL,
    label        TEXT,
    explanation  TEXT
);
"""

_INSERT_SQL = """
INSERT INTO sensor_log
    (timestamp, temperature, pressure, altitude,
     air_quality, ldr, flow_score, label, explanation)
VALUES
    (:timestamp, :temperature, :pressure, :altitude,
     :air_quality, :ldr, :flow_score, :label, :explanation);
"""


def main() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(_CREATE_SQL)
    conn.commit()

    rows = []
    ts   = START
    # Decide per-day whether occupant is out (midday absence)
    for i in range(TOTAL):
        dow = ts.weekday()
        # Morning slots: decide absence for the midday block (10:00-16:00)
        # We do it per-slot based on hour window to keep it simple
        is_out = (10 <= ts.hour < 16) and (random.random() < DOW_OUT_PROB[dow])

        label = _label(ts.hour, ts.minute, is_out)
        r     = _readings(ts, label)
        r["timestamp"] = ts.isoformat()
        rows.append(r)
        ts += STEP

    conn.executemany(_INSERT_SQL, rows)
    conn.commit()
    conn.close()
    print(f"Inserted {len(rows)} rows into {DB_PATH}")


if __name__ == "__main__":
    main()
