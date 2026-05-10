"""
Room-state classifier – wraps a persisted scikit-learn model and exposes
a predict() method used by the main control loop.
"""

import os
import logging

import numpy as np

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

from config import MODEL_PATH, LABEL_MAP

logger = logging.getLogger(__name__)


class RoomClassifier:
    """
    Loads a pre-trained sklearn estimator from disk.

    Falls back to a rule-based heuristic when no model file is found,
    so the system can operate before training has been performed.
    """

    def __init__(self) -> None:
        self._model = None
        if _JOBLIB_AVAILABLE and os.path.exists(MODEL_PATH):
            self._model = joblib.load(MODEL_PATH)
            logger.info("Classifier model loaded from %s", MODEL_PATH)
        else:
            logger.warning(
                "No model found at %s – using rule-based fallback.", MODEL_PATH
            )

    def predict(self, features: np.ndarray) -> int:
        """
        Parameters
        ----------
        features : np.ndarray of shape (1, n_features)

        Returns
        -------
        int – class index (see config.LABEL_MAP)
        """
        if self._model is not None:
            return int(self._model.predict(features)[0])
        # Rule-based fallback: occupancy flag (index 6) drives the decision
        occupancy = int(features[0, 6])
        return 1 if occupancy else 0

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return class probabilities if the model supports it."""
        if self._model is not None and hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(features)[0]
        # Return a dummy probability vector
        n_classes = len(LABEL_MAP)
        proba = np.zeros(n_classes)
        proba[self.predict(features)] = 1.0
        return proba

    def label_name(self, class_id: int) -> str:
        """Map class index to human-readable label."""
        return LABEL_MAP.get(class_id, "unknown")
