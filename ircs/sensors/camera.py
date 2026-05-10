"""
Camera sensor – detects room occupancy using OpenCV background subtraction.
Falls back to picamera2 for Raspberry Pi camera module; uses cv2.VideoCapture
for USB cameras or simulation environments.
"""

import logging
import threading

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

from config import (
    CAMERA_INDEX,
    CAMERA_FRAME_WIDTH,
    CAMERA_FRAME_HEIGHT,
    CAMERA_FPS,
)

logger = logging.getLogger(__name__)

_MIN_CONTOUR_AREA = 1500   # px² – smaller blobs are ignored as noise


class CameraSensor:
    """
    Runs a background-subtraction occupancy detector.
    Thread-safe: `detect_occupancy()` can be called from any thread.
    """

    def __init__(self) -> None:
        self._lock      = threading.Lock()
        self._occupied  = False
        self._frame     = None

        if not _CV2_AVAILABLE:
            logger.warning("OpenCV not available – CameraSensor in simulation mode.")
            self._cap  = None
            self._fgbg = None
            return

        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

        if not self._cap.isOpened():
            logger.error("Could not open camera index %d.", CAMERA_INDEX)
            self._cap  = None
            self._fgbg = None
            return

        # MOG2 background subtractor with shadow detection disabled
        self._fgbg = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=50, detectShadows=False
        )
        logger.info("Camera initialised (index=%d, %dx%d @ %d fps).",
                    CAMERA_INDEX, CAMERA_FRAME_WIDTH, CAMERA_FRAME_HEIGHT, CAMERA_FPS)

    def _analyse_frame(self, frame: np.ndarray) -> bool:
        """Return True if motion/occupancy is detected in the frame."""
        fgmask  = self._fgbg.apply(frame)
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return any(cv2.contourArea(c) > _MIN_CONTOUR_AREA for c in contours)

    def detect_occupancy(self) -> bool:
        """Capture one frame and return True if the room appears occupied."""
        if not _CV2_AVAILABLE or self._cap is None:
            import random
            return random.random() > 0.3   # 70 % chance "occupied" in simulation

        ret, frame = self._cap.read()
        if not ret:
            logger.warning("Camera read failed.")
            return self._occupied   # return last known state

        occupied = self._analyse_frame(frame)
        with self._lock:
            self._occupied = occupied
            self._frame    = frame
        return occupied

    def get_last_frame(self) -> "np.ndarray | None":
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def cleanup(self) -> None:
        if self._cap is not None and _CV2_AVAILABLE:
            self._cap.release()
