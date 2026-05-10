"""
Camera sensor – MediaPipe Pose Landmarker + optical flow for the IRCS.

Provides two outputs consumed by the feature extractor:
  - posture   : int  – 0=UPRIGHT, 1=RECLINED, 2=HORIZONTAL  (-1=unknown)
  - flow_score: float – normalised frame-to-frame optical flow magnitude
                        (0.0 = completely still, 1.0 = maximum motion)

The camera pipeline is designed to be gated externally: it is only called
when the ultrasonic sensor confirms occupancy (distance ≤ PRESENCE_DISTANCE_CM).
"""

import logging
import threading
from enum import IntEnum

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

from config import (
    CAMERA_INDEX,
    CAMERA_FRAME_WIDTH,
    CAMERA_FRAME_HEIGHT,
    CAMERA_FPS,
)

logger = logging.getLogger(__name__)


class Posture(IntEnum):
    UNKNOWN    = -1
    UPRIGHT    = 0
    RECLINED   = 1
    HORIZONTAL = 2


# Thresholds for vertical displacement of landmark centroid
_HIP_SHOULDER_RATIO_UPRIGHT    = 0.25   # shoulder y much higher than hip y (normalised)
_HIP_SHOULDER_RATIO_HORIZONTAL = 0.08   # very small vertical separation → lying flat

# Optical flow normalisation: max expected mean-magnitude per frame
_FLOW_MAX_MAGNITUDE = 15.0


class CameraSensor:
    """
    Runs MediaPipe Pose Landmarker for posture classification and
    dense optical flow (Farneback) for a motion/stillness score.

    Thread-safe: `analyse()` may be called from the sensor loop thread.
    """

    def __init__(self) -> None:
        self._lock        = threading.Lock()
        self._prev_gray   = None
        self._last_posture     = Posture.UNKNOWN
        self._last_flow_score  = 0.0

        if not _CV2_AVAILABLE:
            logger.warning("OpenCV not available – CameraSensor in simulation mode.")
            self._cap  = None
            self._pose = None
            return

        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

        if not self._cap.isOpened():
            logger.error("Could not open camera index %d.", CAMERA_INDEX)
            self._cap  = None
            self._pose = None
            return

        if _MP_AVAILABLE:
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,          # balanced speed/accuracy on Pi 4B
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("MediaPipe Pose initialised.")
        else:
            logger.warning("mediapipe not installed – posture classification disabled.")
            self._pose = None

    # ── Posture classification ────────────────────────────────────────────────

    def _classify_posture(self, landmarks) -> Posture:
        """
        Use shoulder and hip y-coordinates (normalised 0-1, top=0) to
        infer body orientation.
        """
        try:
            lm = mp.solutions.pose.PoseLandmark
            l_shoulder = landmarks[lm.LEFT_SHOULDER.value].y
            r_shoulder = landmarks[lm.RIGHT_SHOULDER.value].y
            l_hip      = landmarks[lm.LEFT_HIP.value].y
            r_hip      = landmarks[lm.RIGHT_HIP.value].y

            shoulder_y = (l_shoulder + r_shoulder) / 2
            hip_y      = (l_hip + r_hip) / 2
            delta      = abs(hip_y - shoulder_y)   # 0 when lying flat

            if delta > _HIP_SHOULDER_RATIO_UPRIGHT:
                return Posture.UPRIGHT
            elif delta > _HIP_SHOULDER_RATIO_HORIZONTAL:
                return Posture.RECLINED
            else:
                return Posture.HORIZONTAL
        except Exception:
            return Posture.UNKNOWN

    # ── Optical flow score ────────────────────────────────────────────────────

    def _compute_flow_score(self, gray: np.ndarray) -> float:
        """
        Compute normalised Farneback optical flow magnitude between the
        previous and current greyscale frames.  Returns 0.0 if no previous
        frame is stored yet.
        """
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray,
            None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0,
        )
        self._prev_gray = gray
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mean_mag  = float(np.mean(magnitude))
        return min(1.0, mean_mag / _FLOW_MAX_MAGNITUDE)

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(self) -> dict:
        """
        Capture one frame, run pose estimation and optical flow.

        Returns
        -------
        dict with keys:
          "posture"    : int  (Posture enum value; -1 = unknown)
          "flow_score" : float (0.0 – 1.0)
        """
        if not _CV2_AVAILABLE or self._cap is None:
            import random
            return {
                "posture":    int(random.choice([Posture.UPRIGHT, Posture.RECLINED,
                                                 Posture.HORIZONTAL])),
                "flow_score": round(random.uniform(0.0, 0.5), 3),
            }

        ret, frame = self._cap.read()
        if not ret:
            logger.warning("Camera read failed.")
            return {"posture": int(self._last_posture),
                    "flow_score": self._last_flow_score}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow_score = self._compute_flow_score(gray)

        posture = Posture.UNKNOWN
        if _MP_AVAILABLE and self._pose is not None:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._pose.process(rgb)
            if results.pose_landmarks:
                posture = self._classify_posture(results.pose_landmarks.landmark)

        with self._lock:
            self._last_posture    = posture
            self._last_flow_score = flow_score

        return {"posture": int(posture), "flow_score": round(flow_score, 3)}

    def cleanup(self) -> None:
        if _CV2_AVAILABLE and self._cap is not None:
            self._cap.release()
        if _MP_AVAILABLE and self._pose is not None:
            self._pose.close()

