"""
IRCS Demo – Live Camera with MediaPipe Pose + Optical Flow Overlay
===================================================================
Run this script (outside the main sensor loop) to get a real-time
window that visualises exactly what the camera sensor sees during a
showcase:

  • Skeleton overlay  – MediaPipe Pose landmarks and connections
  • Optical-flow arrows – dense Farneback vectors sampled on a grid
  • HUD panel (right side bar):
      - Current posture state (UPRIGHT / RECLINED / HORIZONTAL / UNKNOWN)
      - Flow score bar  (0.0 = still → 1.0 = max motion)
      - Live FPS counter

Usage
-----
  cd ircs/
  python demo_camera.py [--camera 0] [--width 640] [--height 480] [--fps 15]
                        [--no-flow] [--no-pose] [--flow-step 16]

Press  Q  or  Esc  to quit.
"""

import argparse
import time
import sys

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("OpenCV (cv2) is required.  Install with:  pip install opencv-python")

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    print("[WARN] mediapipe not installed – pose skeleton disabled.")
    _MP_AVAILABLE = False


# ── Posture constants (mirrors sensors/camera.py) ─────────────────────────────
_HIP_SHOULDER_RATIO_UPRIGHT    = 0.25
_HIP_SHOULDER_RATIO_HORIZONTAL = 0.08
_FLOW_MAX_MAGNITUDE            = 15.0

POSTURE_LABELS = {-1: "UNKNOWN", 0: "UPRIGHT", 1: "RECLINED", 2: "HORIZONTAL"}
POSTURE_COLORS = {  # BGR
    -1: (180, 180, 180),
    0:  (80,  220,  80),   # green  – upright
    1:  (50,  180, 255),   # amber  – reclined
    2:  (60,   60, 255),   # red    – horizontal
}

# HUD dimensions
HUD_W = 220       # width of the right-hand info panel (pixels)
HUD_BG = (30, 30, 30)  # dark background


# ── Utility helpers ───────────────────────────────────────────────────────────

def _classify_posture(landmarks) -> int:
    try:
        lm = mp.solutions.pose.PoseLandmark
        l_sh = landmarks[lm.LEFT_SHOULDER.value].y
        r_sh = landmarks[lm.RIGHT_SHOULDER.value].y
        l_hi = landmarks[lm.LEFT_HIP.value].y
        r_hi = landmarks[lm.RIGHT_HIP.value].y
        shoulder_y = (l_sh + r_sh) / 2
        hip_y      = (l_hi + r_hi) / 2
        delta = abs(hip_y - shoulder_y)
        if delta > _HIP_SHOULDER_RATIO_UPRIGHT:
            return 0   # UPRIGHT
        elif delta > _HIP_SHOULDER_RATIO_HORIZONTAL:
            return 1   # RECLINED
        else:
            return 2   # HORIZONTAL
    except Exception:
        return -1  # UNKNOWN


def _compute_flow(prev_gray: np.ndarray, curr_gray: np.ndarray):
    """Return (flow_score, flow_xy) where flow_xy is the raw 2-channel array."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0,
    )
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    score = float(min(1.0, np.mean(magnitude) / _FLOW_MAX_MAGNITUDE))
    return score, flow


def _draw_flow_arrows(canvas: np.ndarray, flow: np.ndarray, step: int) -> None:
    """Sample the dense flow field on a regular grid and draw arrows."""
    h, w = canvas.shape[:2]
    y_coords = range(step // 2, h, step)
    x_coords = range(step // 2, w, step)
    for y in y_coords:
        for x in x_coords:
            dx, dy = flow[y, x]
            mag = (dx ** 2 + dy ** 2) ** 0.5
            if mag < 0.5:           # suppress near-zero noise
                continue
            end_x = int(x + dx * 3)
            end_y = int(y + dy * 3)
            # colour: blue → red as magnitude increases
            intensity = min(255, int(mag / _FLOW_MAX_MAGNITUDE * 255 * 6))
            colour = (255 - intensity, 60, intensity)
            cv2.arrowedLine(canvas, (x, y), (end_x, end_y),
                            colour, 1, tipLength=0.4)


def _build_hud(height: int, posture: int, flow_score: float, fps: float) -> np.ndarray:
    """Build the right-hand info panel as a numpy array."""
    panel = np.full((height, HUD_W, 3), HUD_BG, dtype=np.uint8)

    # ── Title ────────────────────────────────────────────────────────────────
    cv2.putText(panel, "IRCS", (10, 34),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, "Camera Demo", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.line(panel, (10, 68), (HUD_W - 10, 68), (80, 80, 80), 1)

    # ── Posture state ────────────────────────────────────────────────────────
    cv2.putText(panel, "POSTURE", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)
    label  = POSTURE_LABELS[posture]
    colour = POSTURE_COLORS[posture]
    cv2.putText(panel, label, (10, 130),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, colour, 2, cv2.LINE_AA)

    # ── Flow score bar ───────────────────────────────────────────────────────
    cv2.putText(panel, "FLOW SCORE", (10, 168),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)
    bar_x, bar_y, bar_h = 10, 178, 18
    bar_max_w = HUD_W - 20
    filled_w  = int(flow_score * bar_max_w)

    # background track
    cv2.rectangle(panel, (bar_x, bar_y),
                  (bar_x + bar_max_w, bar_y + bar_h), (70, 70, 70), -1)
    # filled portion – colour shifts green → red with score
    bar_r = int(flow_score * 255)
    bar_g = int((1 - flow_score) * 200)
    if filled_w > 0:
        cv2.rectangle(panel, (bar_x, bar_y),
                      (bar_x + filled_w, bar_y + bar_h),
                      (20, bar_g, bar_r), -1)
    cv2.putText(panel, f"{flow_score:.2f}", (bar_x + bar_max_w + 4 - 50, bar_y + bar_h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    # ── Flow state label ─────────────────────────────────────────────────────
    if flow_score < 0.05:
        flow_label, f_colour = "STILL",    (80, 220, 80)
    elif flow_score < 0.25:
        flow_label, f_colour = "LOW",      (50, 200, 160)
    elif flow_score < 0.55:
        flow_label, f_colour = "MODERATE", (50, 180, 255)
    else:
        flow_label, f_colour = "HIGH",     (60, 60, 255)
    cv2.putText(panel, flow_label, (10, 222),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, f_colour, 1, cv2.LINE_AA)

    cv2.line(panel, (10, 240), (HUD_W - 10, 240), (80, 80, 80), 1)

    # ── FPS ──────────────────────────────────────────────────────────────────
    cv2.putText(panel, f"FPS  {fps:5.1f}", (10, 268),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1, cv2.LINE_AA)

    # ── Key hint ─────────────────────────────────────────────────────────────
    cv2.putText(panel, "[ Q / Esc ] quit", (10, height - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1, cv2.LINE_AA)

    return panel


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IRCS live camera demo")
    parser.add_argument("--camera",    type=int,   default=0,   help="Camera device index")
    parser.add_argument("--width",     type=int,   default=640, help="Capture width (px)")
    parser.add_argument("--height",    type=int,   default=480, help="Capture height (px)")
    parser.add_argument("--fps",       type=int,   default=15,  help="Capture FPS")
    parser.add_argument("--no-flow",   action="store_true",     help="Disable flow arrows overlay")
    parser.add_argument("--no-pose",   action="store_true",     help="Disable skeleton overlay")
    parser.add_argument("--flow-step", type=int,   default=16,  help="Arrow grid spacing (px)")
    args = parser.parse_args()

    # ── Camera init ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS,          args.fps)

    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera index {args.camera}")

    # ── MediaPipe Pose init ───────────────────────────────────────────────────
    pose_proc = None
    mp_drawing = None
    mp_drawing_styles = None
    if _MP_AVAILABLE and not args.no_pose:
        pose_proc = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        mp_drawing        = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

    # ── State ─────────────────────────────────────────────────────────────────
    prev_gray   = None
    posture     = -1
    flow_score  = 0.0
    flow_field  = None
    fps_display = 0.0
    t_prev      = time.perf_counter()

    window_name = "IRCS – Camera Demo  (Q / Esc to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("[INFO] Camera demo running – press Q or Esc in the window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame capture failed – retrying…")
            continue

        # ── FPS ───────────────────────────────────────────────────────────────
        t_now       = time.perf_counter()
        fps_display = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev      = t_now

        # ── Optical flow ──────────────────────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow_score, flow_field = _compute_flow(prev_gray, gray)
        prev_gray = gray

        # ── MediaPipe Pose ────────────────────────────────────────────────────
        if pose_proc is not None:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_proc.process(rgb)
            if results.pose_landmarks:
                posture = _classify_posture(results.pose_landmarks.landmark)
                if not args.no_pose:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
            else:
                posture = -1   # no person detected

        # ── Flow arrow overlay ────────────────────────────────────────────────
        if not args.no_flow and flow_field is not None:
            _draw_flow_arrows(frame, flow_field, step=args.flow_step)

        # ── Compose final display ─────────────────────────────────────────────
        h = frame.shape[0]
        hud = _build_hud(h, posture, flow_score, fps_display)
        display = np.concatenate([frame, hud], axis=1)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):   # Q or Esc
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if pose_proc is not None:
        pose_proc.close()
    cv2.destroyAllWindows()
    print("[INFO] Demo closed.")


if __name__ == "__main__":
    main()
