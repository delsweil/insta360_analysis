# ── Kalman Ball Tracker ───────────────────────────────────────────
# Drop-in replacement for BallState.
# State vector: [x, y, vx, vy] in pixel coordinates.
# Predicts ball position when detection is lost, giving the camera
# a physically plausible target to follow rather than falling back
# to player centroid immediately.
#
# To apply: paste this class into autopan_infer.py, replacing the
# BallState dataclass and updating choose_target as shown below.

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

OUT_FPS = 29.97  # already defined in autopan_infer.py

# Kalman tuning constants — add near other constants
KALMAN_PROCESS_NOISE  = 50.0   # how much we expect ball to accelerate (px/frame²)
KALMAN_MEASURE_NOISE  = 15.0   # pixel uncertainty in detection position
KALMAN_MAX_PREDICT    = 45     # max frames to trust prediction (~1.5s at 30fps)
KALMAN_REINIT_DIST    = 200    # px — if detection jumps this far, reinitialise


class KalmanBallTracker:
    """
    Constant-velocity Kalman filter for ball tracking.

    State: [x, y, vx, vy]
    Observation: [x, y]

    Usage:
        tracker = KalmanBallTracker()

        # Each detection frame:
        tracker.update(cx, cy)
        pos = tracker.predicted_pos()  # smoothed position

        # Each non-detection frame:
        tracker.predict_only()
        pos = tracker.predicted_pos()  # extrapolated position

        # Check if prediction is still trustworthy:
        if tracker.is_valid():
            tx, ty = tracker.predicted_pos()
    """

    def __init__(self):
        self._initialised = False
        self.frames_since_detection = 999
        self.frames_tracked = 0
        self.last_conf = 0.0

        # State vector [x, y, vx, vy]
        self._x = np.zeros((4, 1), dtype=np.float64)

        # State covariance
        self._P = np.eye(4, dtype=np.float64) * 500.0

        # State transition matrix (constant velocity)
        self._F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Observation matrix (we observe x, y only)
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process noise covariance
        q = KALMAN_PROCESS_NOISE
        self._Q = np.array([
            [q/4, 0,   q/2, 0  ],
            [0,   q/4, 0,   q/2],
            [q/2, 0,   q,   0  ],
            [0,   q/2, 0,   q  ],
        ], dtype=np.float64)

        # Measurement noise covariance
        r = KALMAN_MEASURE_NOISE ** 2
        self._R = np.array([
            [r, 0],
            [0, r],
        ], dtype=np.float64)

    def _init_state(self, cx: float, cy: float):
        self._x = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64) * 500.0
        self._initialised = True
        self.frames_tracked = 1
        self.frames_since_detection = 0

    def update(self, cx: float, cy: float, conf: float = 1.0):
        """Call when ball is detected at (cx, cy)."""
        if not self._initialised:
            self._init_state(cx, cy)
            self.last_conf = conf
            return

        # Check for teleport — reinitialise if ball jumps too far
        pred_x, pred_y = float(self._x[0]), float(self._x[1])
        dist = ((cx - pred_x)**2 + (cy - pred_y)**2) ** 0.5
        if dist > KALMAN_REINIT_DIST and self.frames_since_detection == 0:
            # Consecutive detection but huge jump — likely false positive, ignore
            return
        if dist > KALMAN_REINIT_DIST * 2:
            # Extremely large jump after gap — reinitialise
            self._init_state(cx, cy)
            self.last_conf = conf
            return

        # Predict step
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

        # Update step
        z = np.array([[cx], [cy]], dtype=np.float64)
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (np.eye(4) - K @ self._H) @ self._P

        self.frames_since_detection = 0
        self.frames_tracked += 1
        self.last_conf = conf

    def predict_only(self):
        """Call each frame when ball is NOT detected — advances the prediction."""
        if not self._initialised:
            self.frames_since_detection += 1
            return
        # Predict step only (no measurement update)
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        self.frames_since_detection += 1

        # Damp velocity over time so prediction doesn't run away
        decay = 0.92 ** self.frames_since_detection
        self._x[2] *= decay
        self._x[3] *= decay

    def predicted_pos(self) -> Optional[Tuple[float, float]]:
        """Returns predicted (x, y) or None if not initialised."""
        if not self._initialised:
            return None
        return float(self._x[0]), float(self._x[1])

    def velocity(self) -> Tuple[float, float]:
        """Returns (vx, vy) in pixels per frame."""
        if not self._initialised:
            return 0.0, 0.0
        return float(self._x[2]), float(self._x[3])

    def speed(self) -> float:
        """Returns speed in pixels per frame."""
        vx, vy = self.velocity()
        return (vx**2 + vy**2) ** 0.5

    def is_valid(self) -> bool:
        """True if prediction is still within trustworthy window."""
        return self._initialised and self.frames_since_detection < KALMAN_MAX_PREDICT

    def is_fresh(self) -> bool:
        """True if ball was detected very recently (within short fallback)."""
        return self._initialised and self.frames_since_detection < BALL_SHORT_FALLBACK

    def reset(self):
        self.__init__()


# ── Updated choose_target using KalmanBallTracker ────────────────
#
# Replace the existing choose_target function with this version.
# Also replace:
#   ball_state = BallState()
# with:
#   ball_state = KalmanBallTracker()
#
# The key change: instead of falling back to player centroid when
# ball is lost, we follow the Kalman prediction for up to
# KALMAN_MAX_PREDICT frames. This means when the ball is kicked
# left and disappears, we keep panning left following the predicted
# trajectory rather than snapping to wherever the players are.

def choose_target_kalman(
    players,
    ball,           # (cx, cy, conf) or None
    tracker,        # KalmanBallTracker instance
) -> tuple:
    """Choose pan target using Kalman ball tracker."""

    # Update tracker
    if ball is not None:
        tracker.update(ball[0], ball[1], ball[2])
    else:
        tracker.predict_only()

    # 1. Fresh high-confidence ball detection — trust immediately
    if ball is not None and ball[2] >= 0.50:
        return ball[0], ball[1], 'ball_highconf'

    # 2. Fresh detection (lower conf but recently confirmed)
    if tracker.is_fresh() and tracker.predicted_pos() is not None:
        px, py = tracker.predicted_pos()
        return px, py, 'ball'

    # 3. Kalman prediction still valid — follow predicted trajectory
    if tracker.is_valid() and tracker.predicted_pos() is not None:
        px, py = tracker.predicted_pos()
        # Blend prediction with player centroid, weighted by prediction age
        # Fresh prediction: 90% kalman, 10% players
        # Old prediction: fades to 30% kalman, 70% players
        age_ratio = tracker.frames_since_detection / KALMAN_MAX_PREDICT
        kalman_weight = 0.9 * (1.0 - age_ratio) + 0.3 * age_ratio

        if players and len(players) >= 4:
            xs = np.array([p[0] for p in players])
            ys = np.array([p[1] for p in players])
            med_x, med_y = float(np.median(xs)), float(np.median(ys))
            sigma = OUT_W * 0.25
            dists_sq = (xs - med_x)**2 + (ys - med_y)**2
            weights = np.exp(-dists_sq / (2 * sigma**2))
            weights /= weights.sum()
            pcx = float(np.sum(weights * xs))
            pcy = float(np.sum(weights * ys))
            tx = kalman_weight * px + (1 - kalman_weight) * pcx
            ty = kalman_weight * py + (1 - kalman_weight) * pcy
            return tx, ty, 'kalman_blend'
        return px, py, 'kalman'

    # 4. No valid prediction — fall back to weighted player centroid
    if len(players) >= 4:
        xs = np.array([p[0] for p in players])
        ys = np.array([p[1] for p in players])
        med_x, med_y = float(np.median(xs)), float(np.median(ys))
        sigma = OUT_W * 0.25
        dists_sq = (xs - med_x)**2 + (ys - med_y)**2
        weights = np.exp(-dists_sq / (2 * sigma**2))
        weights /= weights.sum()
        cx = float(np.sum(weights * xs))
        cy = float(np.sum(weights * ys))
        return cx, cy, 'players'
    elif len(players) >= 2:
        xs = np.array([p[0] for p in players])
        ys = np.array([p[1] for p in players])
        med_x, med_y = float(np.median(xs)), float(np.median(ys))
        sigma = OUT_W * 0.25
        dists_sq = (xs - med_x)**2 + (ys - med_y)**2
        weights = np.exp(-dists_sq / (2 * sigma**2))
        weights /= weights.sum()
        cx = float(np.sum(weights * xs))
        cy = float(np.sum(weights * ys))
        tx = 0.2 * cx + 0.8 * (OUT_W / 2)
        ty = 0.2 * cy + 0.8 * (OUT_H / 2)
        return tx, ty, 'few_players'
    elif len(players) == 1:
        tx = 0.05 * players[0][0] + 0.95 * (OUT_W / 2)
        ty = 0.05 * players[0][1] + 0.95 * (OUT_H / 2)
        return tx, ty, 'single_player'

    return OUT_W / 2, OUT_H / 2, 'centre'
