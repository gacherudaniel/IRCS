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

# Posture encoding indices within the 14-feature vector (see feature_extractor.py)
_IDX_DISTANCE   = 5
_IDX_POSTURE    = 6
_IDX_FLOW_SCORE = 7


def _cv_fallback(features: np.ndarray) -> int:
    """
    Rule-based heuristic using posture and optical flow when ML confidence
    is insufficient.  Uses raw (unscaled) feature indices — called before
    the scaler is applied, so values may be scaled; only relative magnitudes
    and sign are used.
    """
    distance   = float(features[0, _IDX_DISTANCE])
    posture    = int(round(float(features[0, _IDX_POSTURE])))
    flow_score = float(features[0, _IDX_FLOW_SCORE])

    if distance > 400:              # no one in range
        return 0  # ROOM_EMPTY
    if flow_score > 0.25:
        return 1  # ACTIVE_AWAKE – significant motion
    if posture == 2:                # HORIZONTAL
        return 3  # SLEEPING
    if posture == 1:                # RECLINED
        return 2  # RESTING
    return 1  # default ACTIVE_AWAKE when upright and still


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
        features : np.ndarray of shape (1, 14)

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

