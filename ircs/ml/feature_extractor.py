"""
Feature extractor – transforms raw sensor readings into a normalised
numpy feature vector ready for the ML classifier.
"""

import os
import logging
import numpy as np

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

from config import SCALER_PATH

logger = logging.getLogger(__name__)

# Ordered list of keys that form the feature vector.
# Must match the column order used during training (see ml/train.py).
FEATURE_KEYS = [
    "temperature",
    "humidity",
    "pressure",
    "air_quality",
    "ldr",
    "distance",
    "occupancy",  # bool cast to int
]


class FeatureExtractor:
    """
    Converts a sensor-reading dict into a scaled numpy array.

    If a fitted scaler artefact exists on disk it is loaded automatically;
    otherwise feature values are passed through unscaled (useful during
    initial data collection before training).
    """

    def __init__(self) -> None:
        self._scaler = None
        if _JOBLIB_AVAILABLE and os.path.exists(SCALER_PATH):
            self._scaler = joblib.load(SCALER_PATH)
            logger.info("Scaler loaded from %s", SCALER_PATH)
        else:
            logger.warning("No scaler found at %s – features will be unscaled.", SCALER_PATH)

    def extract(self, reading: dict) -> np.ndarray:
        """
        Parameters
        ----------
        reading : dict
            Raw sensor values keyed as defined in FEATURE_KEYS.

        Returns
        -------
        np.ndarray of shape (1, n_features), dtype float32
        """
        vector = np.array(
            [float(int(reading[k]) if k == "occupancy" else reading[k])
             for k in FEATURE_KEYS],
            dtype=np.float32,
        ).reshape(1, -1)

        if self._scaler is not None:
            vector = self._scaler.transform(vector)

        return vector

    @staticmethod
    def feature_names() -> list[str]:
        return list(FEATURE_KEYS)
