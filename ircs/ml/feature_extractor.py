"""
Feature extractor – builds the 14-feature input vector for the IRCS
Random Forest classifier from a 30-second rolling window of sensor readings.

Feature vector (in order):
  0  temperature        – DHT11 °C
  1  humidity           – DHT11 % RH
  2  pressure           – BMP280 hPa
  3  co2_ppm            – MQ-135 ppm equivalent
  4  lux                – LDR lux
  5  distance           – HC-SR04 cm
  6  posture            – MediaPipe pose: 0=UPRIGHT, 1=RECLINED, 2=HORIZONTAL, -1=unknown
  7  flow_score         – Farneback optical flow magnitude (0.0–1.0)
  8  ultrasonic_var     – variance of distance readings over rolling window (stillness proxy)
  9  lux_rate           – lux rate-of-change over rolling window (lux/s)
 10  hour_sin           – sin encoding of hour-of-day (circadian feature)
 11  hour_cos           – cos encoding of hour-of-day
 12  dow_sin            – sin encoding of day-of-week
 13  dow_cos            – cos encoding of day-of-week
"""

import logging
import math
import os
from collections import deque
from datetime import datetime
from typing import Deque

import numpy as np

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

from config import SCALER_PATH, ROLLING_WINDOW_SECONDS, SENSOR_POLL_INTERVAL

logger = logging.getLogger(__name__)

# Maximum number of readings stored in the rolling window
_WINDOW_SIZE = max(1, ROLLING_WINDOW_SECONDS // SENSOR_POLL_INTERVAL)

FEATURE_NAMES = [
    "temperature", "humidity", "pressure", "co2_ppm", "lux",
    "distance", "posture", "flow_score",
    "ultrasonic_var", "lux_rate",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


def _time_encoding(now: datetime) -> tuple[float, float, float, float]:
    """Cyclic encoding of hour-of-day and day-of-week."""
    hour_rad = (now.hour + now.minute / 60) / 24 * 2 * math.pi
    dow_rad  = now.weekday() / 7 * 2 * math.pi
    return (
        math.sin(hour_rad), math.cos(hour_rad),
        math.sin(dow_rad),  math.cos(dow_rad),
    )


class FeatureExtractor:
    """
    Maintains a rolling window of raw readings and exposes `extract()`,
    which returns the full 14-feature numpy array ready for the classifier.
    """

    def __init__(self) -> None:
        self._window: Deque[dict] = deque(maxlen=_WINDOW_SIZE)
        self._scaler = None
        if _JOBLIB_AVAILABLE and os.path.exists(SCALER_PATH):
            self._scaler = joblib.load(SCALER_PATH)
            logger.info("Scaler loaded from %s", SCALER_PATH)
        else:
            logger.warning("No scaler found at %s – features will be unscaled.", SCALER_PATH)

    def update(self, reading: dict) -> None:
        """Push a new sensor reading into the rolling window."""
        self._window.append(reading)

    def extract(self, reading: dict, now: datetime | None = None) -> np.ndarray:
        """
        Build and return a (1, 14) float32 feature array.

        Parameters
        ----------
        reading : latest sensor reading dict (also pushed into the window)
        now     : timestamp for circadian features; defaults to datetime.now()
        """
        self.update(reading)
        if now is None:
            now = datetime.now()

        # ── Instantaneous features ────────────────────────────────────────────
        temp       = float(reading.get("temperature", 22.0))
        humidity   = float(reading.get("humidity",    50.0))
        pressure   = float(reading.get("pressure",    1013.0))
        co2_ppm    = float(reading.get("co2_ppm",     400.0))
        lux        = float(reading.get("lux",         500.0))
        distance   = float(reading.get("distance",    400.0))
        posture    = float(reading.get("posture",     -1))
        flow_score = float(reading.get("flow_score",  0.0))

        # ── Window-derived features ───────────────────────────────────────────
        distances = [r.get("distance", distance) for r in self._window]
        lux_vals  = [r.get("lux", lux)           for r in self._window]

        ultrasonic_var = float(np.var(distances)) if len(distances) > 1 else 0.0

        if len(lux_vals) >= 2:
            elapsed    = max(len(lux_vals) - 1, 1) * SENSOR_POLL_INTERVAL
            lux_rate   = (lux_vals[-1] - lux_vals[0]) / elapsed   # lux/s
        else:
            lux_rate = 0.0

        # ── Circadian features ────────────────────────────────────────────────
        hour_sin, hour_cos, dow_sin, dow_cos = _time_encoding(now)

        vector = np.array([
            temp, humidity, pressure, co2_ppm, lux,
            distance, posture, flow_score,
            ultrasonic_var, lux_rate,
            hour_sin, hour_cos, dow_sin, dow_cos,
        ], dtype=np.float32).reshape(1, -1)

        if self._scaler is not None:
            vector = self._scaler.transform(vector)

        return vector

    @staticmethod
    def feature_names() -> list[str]:
        return list(FEATURE_NAMES)

