"""
Room-state classifier – wraps a persisted scikit-learn Random Forest and
exposes a predict() method used by the main control loop.

Four context states:
  0 ROOM_EMPTY   – no occupant detected
  1 ACTIVE_AWAKE – occupant is moving / awake
  2 RESTING      – occupant is still but awake (sitting/reclined)
  3 SLEEPING     – occupant is horizontal and still

Confidence gate: if predict_proba() < SAFETY_CONFIDENCE_FLOOR the classifier
falls back to a CV-based heuristic (posture + flow_score from the feature
vector) rather than acting on an uncertain ML prediction.
"""

import os
import logging

import numpy as np

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

from config import MODEL_PATH, LABEL_MAP, SAFETY_CONFIDENCE_FLOOR

logger = logging.getLogger(__name__)

# Feature indices within the 10-feature vector (see feature_extractor.py)
_IDX_LUX        = 3
_IDX_FLOW_SCORE = 4
_IDX_CO2        = 2


def _cv_fallback(features: np.ndarray) -> int:
    """
    Rule-based heuristic using lux, flow_score and CO2 when ML confidence
    is insufficient.
    """
    flow_score = float(features[0, _IDX_FLOW_SCORE])
    lux        = float(features[0, _IDX_LUX])
    co2        = float(features[0, _IDX_CO2])

    if lux < 10 and flow_score < 0.05:
        return 3  # SLEEPING – dark and still
    if flow_score > 0.25:
        return 1  # ACTIVE_AWAKE – significant motion
    if co2 > 700 and lux < 100:
        return 2  # RESTING – someone present but still
    if lux < 5 and co2 < 500:
        return 0  # ROOM_EMPTY – very dark, low CO2
    return 2  # default RESTING


class RoomClassifier:
    """
    Loads a pre-trained sklearn Random Forest estimator from disk.
    Falls back to `_cv_fallback()` when confidence is below threshold
    or when no model file exists.
    """

    def __init__(self) -> None:
        self._model = None
        if _JOBLIB_AVAILABLE and os.path.exists(MODEL_PATH):
            self._model = joblib.load(MODEL_PATH)
            logger.info("Classifier model loaded from %s", MODEL_PATH)
        else:
            logger.warning(
                "No model found at %s – using CV-based fallback.", MODEL_PATH
            )

    def predict(self, features: np.ndarray) -> tuple[int, float]:
        """
        Parameters
        ----------
        features : np.ndarray of shape (1, 10)

        Returns
        -------
        (class_id: int, confidence: float)
        """
        if self._model is None:
            return _cv_fallback(features), 0.0

        proba      = self._model.predict_proba(features)[0]
        class_id   = int(np.argmax(proba))
        confidence = float(proba[class_id])

        if confidence < SAFETY_CONFIDENCE_FLOOR:
            fallback_id = _cv_fallback(features)
            logger.debug(
                "Low confidence (%.2f < %.2f) – CV fallback: %s → %s",
                confidence, SAFETY_CONFIDENCE_FLOOR,
                LABEL_MAP.get(class_id), LABEL_MAP.get(fallback_id),
            )
            return fallback_id, confidence

        return class_id, confidence

    def label_name(self, class_id: int) -> str:
        return LABEL_MAP.get(class_id, "UNKNOWN")

