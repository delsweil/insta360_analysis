#!/usr/bin/env python3
"""
autopan_infer.py — Autopan inference using direct PD control (v5 approach).

Instead of predicting absolute pan values with ML, this script:
1. Projects the current view
2. Detects players and ball in that view
3. Computes error between target and frame centre
4. Moves camera proportionally (PD control with velocity damping)
5. Applies exponential smoothing on target position

This is more robust than absolute pan prediction — it only needs to know
"is the target left or right of centre".

Usage:
    python autopan_infer.py \
        --insv  /path/to/VID_xxx_10_001.insv \
        --calib calibration/pitch.json \
        --output /tmp/out.mp4 \
        [--players models/yolo11n.pt] \
        [--ball    models/ball_v2.pt] \
        [--segments 5] [--seg-duration 30] \
        [--device cpu|mps|cuda] \
        [--debug]
"""

from __future__ import annotations
import argparse, json, subprocess, time, warnings
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import cv2
import pickle
from sklearn.cluster import DBSCAN
import numpy as np
from py360convert import e2p
from scipy.interpolate import PchipInterpolator

# ── Output ────────────────────────────────────────────────────────
OUT_W, OUT_H = 1280, 720
OUT_FPS      = 29.97

# ── Detection ─────────────────────────────────────────────────────
IMGSZ_PLAYERS  = 960
IMGSZ_BALL     = 640
CONF_PLAYERS   = 0.20
CONF_BALL      = 0.40
DETECT_EVERY   = 5      # detect every N frames (~0.17s at 30fps)

# ── Control ───────────────────────────────────────────────────────
YAW_GAIN       = 0.30   # how aggressively to pan toward target
PITCH_GAIN     = 0.22
MAX_YAW_STEP   = 1.6    # degrees per frame max
MAX_PITCH_STEP = 1.2
VEL_ALPHA      = 0.15   # velocity smoothing (lower = more damping)
TARGET_ALPHA   = 0.12   # target EMA smoothing (lower = smoother)
DEADBAND_X     = 0.08   # ignore errors smaller than this fraction of frame
DEADBAND_Y     = 0.06
MAX_YAW_DEV    = 55.0   # max pan deviation from centre
MAX_PITCH_DEV  =  5.0   # pitch stays negative (always looking down)

# ── Ball gating ───────────────────────────────────────────────────
BALL_CONFIRM_FRAMES  = 3    # frames needed to trust ball
BALL_MAX_JUMP_PX     = 150  # max pixels between detections to trust immediately
BALL_SHORT_FALLBACK  = 10   # frames before falling back to players
BALL_LONG_FALLBACK   = 90   # frames before falling back to centre

# ── Kalman ball tracker ──────────────────────────────────────────────
KALMAN_PROCESS_NOISE  = 50.0   # expected ball acceleration (px/frame²)
KALMAN_MEASURE_NOISE  = 15.0   # pixel uncertainty in detection
KALMAN_MAX_PREDICT    = 45     # max frames to trust prediction (~1.5s)
KALMAN_REINIT_DIST    = 200    # px — jump distance to trigger reinit

# ── DBSCAN player clustering ─────────────────────────────────────────
DBSCAN_EPS_DEG        = 8.0    # cluster radius in equirect degrees
DBSCAN_MIN_SAMPLES    = 2      # minimum players to form a cluster
DBSCAN_MIN_SIZE       = 3      # min cluster size to use as target
HOLD_THRESHOLD        = 0.65   # fraction of half-frame — only move when cluster in outer 35%

# ── Pitch polygon ─────────────────────────────────────────────────
PITCH_COORDS_NORM = [
    (0.00,0.00),(0.17,0.00),(0.33,0.00),(0.50,0.00),
    (0.67,0.00),(0.83,0.00),(1.00,0.00),
    (1.00,0.25),(1.00,0.50),(1.00,0.75),
    (0.83,1.00),(0.50,1.00),(0.17,1.00),(0.00,1.00),
    (0.00,0.75),(0.00,0.50),
]


# ── Calibration ───────────────────────────────────────────────────

def derive_tilt_fov(calib_path: str) -> Tuple[float, float]:
    """Derive e2p tilt and FOV from pitch polygon."""
    with open(calib_path) as f:
        d = json.load(f)
    raw = d.get('pitch_polygon') or d.get('pixel_polygon') or d.get('auto_polygon')
    if isinstance(raw[0], dict):
        poly = np.array([[p['x']*2880, p['y']*1440] for p in raw], dtype=np.float32)
    else:
        poly = np.array(raw, dtype=np.float32)
    tilt_deg  = (0.5 - poly[:,1] / 1440) * 180
    far_tilt  = float(np.mean(tilt_deg[:7]))
    near_tilt = float(np.mean(tilt_deg[7:]))
    mid_tilt  = (far_tilt + near_tilt) / 2
    e2p_tilt  = mid_tilt * 1.20
    e2p_fov   = float(np.clip(abs(near_tilt - far_tilt) * 1.85, 100, 130))
    print(f"  Calibration: far={far_tilt:.1f}° near={near_tilt:.1f}°")
    print(f"  e2p_tilt={e2p_tilt:.1f}° e2p_fov={e2p_fov:.1f}°")
    return e2p_tilt, e2p_fov


def build_pitch_mask(calib_path: str, yaw: float, pitch: float,
                     fov: float, h: int, w: int) -> Optional[np.ndarray]:
    """Project pitch polygon into current perspective view as binary mask."""
    with open(calib_path) as f:
        d = json.load(f)
    raw = d.get('pitch_polygon') or d.get('pixel_polygon') or d.get('auto_polygon')
    if isinstance(raw[0], dict):
        poly_eq = np.array([[p['x']*2880, p['y']*1440] for p in raw], dtype=np.float32)
    else:
        poly_eq = np.array(raw, dtype=np.float32)
    EW, EH = 2880, 1440

    # Build equirect mask
    mask_eq = np.zeros((EH, EW), dtype=np.uint8)
    pts = poly_eq.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask_eq, [pts], 255)

    # Project to perspective
    mask_rgb = np.stack([mask_eq]*3, axis=-1)
    proj = e2p(mask_rgb, fov_deg=fov, u_deg=-yaw, v_deg=pitch,
               out_hw=(h, w), mode='bilinear')
    return (proj[:,:,0] > 127).astype(np.uint8)


# ── FFmpeg ────────────────────────────────────────────────────────

def get_duration(insv_path: str) -> float:
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', insv_path],
        capture_output=True, text=True)
    txt = r.stdout.strip()
    if not txt:
        print("WARNING: could not determine duration, assuming 1800s")
        return 1800.0
    return float(txt)


def open_stream(insv_path: str, start_s: float, duration_s: float):
    cmd = [
        'ffmpeg', '-ss', str(max(0, start_s)),
        '-i', insv_path,
        '-t', str(duration_s + 2),
        '-vf', 'rotate=PI/2*3,v360=fisheye:equirect:ih_fov=200:iv_fov=200,scale=2880:1440',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-an', 'pipe:1',
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def read_frame(proc, w=2880, h=1440) -> Optional[np.ndarray]:
    nbytes = w * h * 3
    chunks, remaining = [], nbytes
    while remaining > 0:
        chunk = proc.stdout.read(min(65536, remaining))
        if not chunk: break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b''.join(chunks)
    if len(data) < nbytes: return None
    return np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3)).copy()


def open_writer(output_path: str, fps: float):
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{OUT_W}x{OUT_H}', '-r', str(fps),
        '-i', 'pipe:0',
        '-c:v', 'h264_videotoolbox', '-b:v', '8000k',
        '-pix_fmt', 'yuv420p', output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


# ── Detection ─────────────────────────────────────────────────────

def detect_players(frame: np.ndarray, model, device: str,
                   mask: Optional[np.ndarray] = None) -> List[Tuple[float,float]]:
    """Returns list of (cx, cy) foot positions within pitch mask."""
    names = model.names
    res = model(frame, imgsz=IMGSZ_PLAYERS, conf=CONF_PLAYERS,
                device=device, verbose=False)[0]
    centroids = []
    for b in res.boxes:
        if names.get(int(b.cls[0]), '') != 'person': continue
        x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
        cx, cy = (x1+x2)/2, (y1+y2)/2
        foot_y = y2
        if mask is not None:
            xi = int(np.clip(cx, 0, mask.shape[1]-1))
            yi = int(np.clip(foot_y, 0, mask.shape[0]-1))
            if not mask[yi, xi]: continue
        centroids.append((cx, foot_y))
    return centroids


def detect_ball(frame: np.ndarray, model, device: str,
                mask: Optional[np.ndarray] = None,
                last_trusted_pos: Optional[Tuple[float,float]] = None,
                mask_eroded: Optional[np.ndarray] = None) -> Optional[Tuple[float,float,float]]:
    """Returns (cx, cy, conf) of best ball detection within pitch mask, or None."""
    res = model(frame, imgsz=IMGSZ_BALL, conf=CONF_BALL,
                device=device, verbose=False)[0]
    best = None
    for b in res.boxes:
        x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
        area = (x2-x1)*(y2-y1)
        if area > OUT_W * OUT_H * 0.02: continue
        cx, cy = (x1+x2)/2, (y1+y2)/2
        conf = float(b.conf[0])
        # Use eroded mask for ball to reject edge detections
        _bmask = mask_eroded if mask_eroded is not None else mask
        if _bmask is not None:
            xi = int(np.clip(cx, 0, _bmask.shape[1]-1))
            yi = int(np.clip(cy, 0, _bmask.shape[0]-1))
            if not _bmask[yi, xi]: continue
        # Spatial consistency: penalise detections far from last trusted position
        if last_trusted_pos is not None:
            dx = cx - last_trusted_pos[0]
            dy = cy - last_trusted_pos[1]
            dist = (dx*dx + dy*dy) ** 0.5
            if dist > BALL_MAX_JUMP_PX:
                conf *= 0.3  # heavily discount jumped detections
        if best is None or conf > best[2]:
            best = (cx, cy, conf)
    return best


# ── Control ───────────────────────────────────────────────────────

@dataclass
class CameraState:
    yaw:   float = 0.0
    pitch: float = 0.0
    yaw_vel:   float = 0.0
    pitch_vel: float = 0.0


@dataclass
class BallState:
    trusted:      bool  = False
    confirm:      int   = 0
    frames_since: int   = 999
    last_pos:     Optional[Tuple[float,float]] = None


class KalmanBallTracker:
    """Constant-velocity Kalman filter for ball tracking.
    State: [x, y, vx, vy]. Predicts ball position when detection is lost."""

    def __init__(self):
        self._initialised = False
        self.frames_since_detection = 999
        self.frames_tracked = 0
        self.last_conf = 0.0
        # Compatibility with existing BallState references
        self.last_pos = None
        self.trusted = False
        self.frames_since = 999
        self._x = np.zeros((4, 1), dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64) * 500.0
        self._F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
        self._H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)
        q = KALMAN_PROCESS_NOISE
        self._Q = np.array([[q/4,0,q/2,0],[0,q/4,0,q/2],[q/2,0,q,0],[0,q/2,0,q]], dtype=np.float64)
        r = KALMAN_MEASURE_NOISE ** 2
        self._R = np.array([[r,0],[0,r]], dtype=np.float64)

    def _init_state(self, cx, cy):
        self._x = np.array([[cx],[cy],[0.0],[0.0]], dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64) * 500.0
        self._initialised = True
        self.frames_tracked = 1
        self.frames_since_detection = 0
        self.frames_since = 0
        self.last_pos = (cx, cy)
        self.trusted = True

    def update(self, cx, cy, conf=1.0):
        if not self._initialised:
            self._init_state(cx, cy)
            self.last_conf = conf
            return
        pred_x, pred_y = float(self._x[0,0]), float(self._x[1,0])
        dist = ((cx-pred_x)**2 + (cy-pred_y)**2)**0.5
        if dist > KALMAN_REINIT_DIST and self.frames_since_detection == 0:
            return  # consecutive jump — likely false positive
        if dist > KALMAN_REINIT_DIST * 2:
            self._init_state(cx, cy)  # reinitialise on extreme jump
            self.last_conf = conf
            return
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        z = np.array([[cx],[cy]], dtype=np.float64)
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (np.eye(4) - K @ self._H) @ self._P
        self.frames_since_detection = 0
        self.frames_since = 0
        self.frames_tracked += 1
        self.last_conf = conf
        self.last_pos = (float(self._x[0,0]), float(self._x[1,0]))
        self.trusted = True

    def predict_only(self):
        if not self._initialised:
            self.frames_since_detection += 1
            self.frames_since += 1
            return
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        self.frames_since_detection += 1
        self.frames_since += 1
        decay = 0.92 ** min(self.frames_since_detection, 30)
        self._x[2] *= decay
        self._x[3] *= decay
        if self._initialised:
            self.last_pos = (float(self._x[0,0]), float(self._x[1,0]))

    def predicted_pos(self):
        if not self._initialised: return None
        return float(self._x[0,0]), float(self._x[1,0])

    def velocity(self):
        if not self._initialised: return 0.0, 0.0
        return float(self._x[2,0]), float(self._x[3,0])

    def is_valid(self):
        return self._initialised and self.frames_since_detection < KALMAN_MAX_PREDICT

    def is_fresh(self):
        return self._initialised and self.frames_since_detection < BALL_SHORT_FALLBACK

    def reset(self):
        self.__init__()


@dataclass
class TargetState:
    sm_tx: float = OUT_W / 2
    sm_ty: float = OUT_H / 2


def pixel_to_lon(px: float, yaw_deg: float,
                 w: int = OUT_W, fov_deg: float = 100.0) -> float:
    """Convert perspective frame pixel x to equirectangular longitude."""
    nx = (px - w / 2) / (w / 2)
    import math
    half_fov = math.radians(fov_deg / 2)
    angle_x = math.degrees(math.atan(nx * math.tan(half_fov)))
    return angle_x + yaw_deg


def find_action_cluster(players: List[Tuple[float, float]],
                        cam_yaw: float,
                        e2p_fov: float) -> Optional[Tuple[float, float]]:
    """Find the densest player cluster in equirectangular space.
    Returns (cluster_cx_px, cluster_cy_px) in perspective frame coords,
    or None if no clear cluster found."""
    if len(players) < DBSCAN_MIN_SAMPLES:
        return None

    # Convert player x positions to equirect longitudes
    lons = np.array([pixel_to_lon(p[0], cam_yaw, OUT_W, e2p_fov)
                     for p in players])
    heights = np.array([p[1] for p in players])  # foot_y as proxy

    # DBSCAN clustering on longitudes
    X = lons.reshape(-1, 1)
    db = DBSCAN(eps=DBSCAN_EPS_DEG, min_samples=DBSCAN_MIN_SAMPLES).fit(X)
    labels = db.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters == 0:
        return None

    # Score each cluster: size * mean_player_height
    best_score = -1
    best_cx = None
    best_cy = None

    for label in set(labels):
        if label == -1:
            continue
        mask = labels == label
        if mask.sum() < DBSCAN_MIN_SIZE:
            continue
        # Use mean pixel position of cluster players as target
        cluster_players = [players[i] for i in range(len(players)) if mask[i]]
        cx = float(np.mean([p[0] for p in cluster_players]))
        cy = float(np.mean([p[1] for p in cluster_players]))
        # Score by size / spread (tight dense cluster wins over loose large cluster)
        cluster_lons = lons[mask]
        spread = float(np.std(cluster_lons)) + 1.0  # avoid div by zero
        score = mask.sum() / spread
        if score > best_score:
            best_score = score
            best_cx = cx
            best_cy = cy

    if best_cx is None:
        return None
    return best_cx, best_cy


def choose_target(
    players: List[Tuple[float,float]],
    ball: Optional[Tuple[float,float,float]],
    ball_state,  # KalmanBallTracker
    cam_yaw: float = 0.0,
    e2p_fov: float = 100.0,
    cluster_selector=None,
    cluster_selector_features=None,
) -> Tuple[float, float, str]:
    """Choose pan target using Kalman ball tracker."""

    # Update Kalman tracker
    if ball is not None:
        ball_state.update(ball[0], ball[1], ball[2])
    else:
        ball_state.predict_only()

    # 1. Fresh high-confidence detection — trust immediately
    if ball is not None and ball[2] >= 0.50:
        return ball[0], ball[1], 'ball_highconf'

    # 2. Fresh detection (recently confirmed)
    if ball_state.is_fresh() and ball_state.predicted_pos() is not None:
        px, py = ball_state.predicted_pos()
        return px, py, 'ball'

    # 3. Kalman prediction still valid — follow predicted trajectory
    if ball_state.is_valid() and ball_state.predicted_pos() is not None:
        px, py = ball_state.predicted_pos()
        age_ratio = ball_state.frames_since_detection / KALMAN_MAX_PREDICT
        kalman_weight = max(0.3, 0.9 * (1.0 - age_ratio))
        # Blend Kalman prediction with DBSCAN cluster (or centroid)
        cluster = find_action_cluster(players, cam_yaw, e2p_fov, cluster_selector, cluster_selector_features) if len(players) >= DBSCAN_MIN_SAMPLES else None
        if cluster is not None:
            tx = kalman_weight * px + (1-kalman_weight) * cluster[0]
            ty = kalman_weight * py + (1-kalman_weight) * cluster[1]
            return tx, ty, 'kalman_blend'
        elif len(players) >= 4:
            xs = np.array([p[0] for p in players])
            ys = np.array([p[1] for p in players])
            med_x, med_y = float(np.median(xs)), float(np.median(ys))
            sigma = OUT_W * 0.25
            w = np.exp(-((xs-med_x)**2+(ys-med_y)**2)/(2*sigma**2))
            w /= w.sum()
            pcx, pcy = float(np.sum(w*xs)), float(np.sum(w*ys))
            tx = kalman_weight * px + (1-kalman_weight) * pcx
            ty = kalman_weight * py + (1-kalman_weight) * pcy
            return tx, ty, 'kalman_blend'
        return px, py, 'kalman'

    # 4. No valid ball prediction — edge-trigger hold logic
    # Only move when the action cluster is near the frame edge,
    # otherwise hold current position. This mimics human editing behaviour.
    if len(players) >= DBSCAN_MIN_SAMPLES:
        cluster = find_action_cluster(players, cam_yaw, e2p_fov, cluster_selector, cluster_selector_features)
        if cluster is not None:
            cx, cy = cluster
            # Normalised distance from frame centre (0=centre, 1=edge)
            edge_dist = abs(cx - OUT_W / 2) / (OUT_W / 2)
            if edge_dist < HOLD_THRESHOLD:
                # Action comfortably in frame — hold current position
                return OUT_W / 2, OUT_H / 2, 'hold'
            else:
                # Action near edge — move to recentre
                return cx, cy, 'players'

    # Few players — very weak pull toward centroid
    if len(players) >= 2:
        xs = np.array([p[0] for p in players])
        ys = np.array([p[1] for p in players])
        med_x, med_y = float(np.median(xs)), float(np.median(ys))
        sigma = OUT_W * 0.25
        dists_sq = (xs-med_x)**2 + (ys-med_y)**2
        weights = np.exp(-dists_sq/(2*sigma**2))
        weights /= weights.sum()
        cx = float(np.sum(weights*xs))
        cy = float(np.sum(weights*ys))
        edge_dist = abs(cx - OUT_W / 2) / (OUT_W / 2)
        if edge_dist > HOLD_THRESHOLD:
            tx = 0.2 * cx + 0.8 * (OUT_W/2)
            ty = 0.2 * cy + 0.8 * (OUT_H/2)
            return tx, ty, 'few_players'
        else:
            return OUT_W / 2, OUT_H / 2, 'hold'
    elif len(players) == 1:
        tx = 0.05 * players[0][0] + 0.95 * (OUT_W/2)
        ty = 0.05 * players[0][1] + 0.95 * (OUT_H/2)
        return tx, ty, 'single_player'

    return OUT_W / 2, OUT_H / 2, 'centre'


def update_camera(cam: CameraState, target_x: float, target_y: float,
                  sm_target: TargetState, e2p_tilt: float = -20.0,
                  mode: str = 'players') -> None:
    """PD control: move camera toward smoothed target."""

    # Hold mode — target is current camera position (don't move)
    if mode == 'hold':
        # Convert cam.yaw back to pixel x to set target at current view centre
        target_x = OUT_W / 2
        target_y = OUT_H / 2
        # Damp velocity to stop any existing movement
        cam.yaw_vel *= 0.5
        cam.pitch_vel *= 0.5
        return

    # Smooth target with EMA — react faster when ball is detected
    # Players-only mode uses slow alpha to avoid overreacting to cluster noise
    if 'ball' in mode:
        base_alpha = TARGET_ALPHA * 3
        error_norm = abs(sm_target.sm_tx - OUT_W/2) / (OUT_W/2)
        alpha = min(0.8, base_alpha * (1.0 + 5.0 * error_norm))
    else:
        alpha = TARGET_ALPHA * 0.5  # slow response for player-only modes
    sm_target.sm_tx += alpha * (target_x - sm_target.sm_tx)
    sm_target.sm_ty += alpha * (target_y - sm_target.sm_ty)

    # Compute normalised error (-0.5 to +0.5)
    err_x = (sm_target.sm_tx - OUT_W/2) / OUT_W
    err_y = (sm_target.sm_ty - OUT_H/2) / OUT_H

    # Deadband
    if abs(err_x) < DEADBAND_X: err_x = 0.0
    if abs(err_y) < DEADBAND_Y: err_y = 0.0

    # Update velocity with damping
    cam.yaw_vel   = cam.yaw_vel   * (1 - VEL_ALPHA) + err_x * YAW_GAIN
    cam.pitch_vel = cam.pitch_vel * (1 - VEL_ALPHA) + err_y * PITCH_GAIN

    # Clamp step — allow larger steps when far from target (recovery mode)
    target_err = abs(sm_target.sm_tx - OUT_W/2) / (OUT_W/2)
    recovery_scale = 1.0 + 2.0 * max(0, target_err - 0.3)  # up to 3x when very off
    max_yaw   = MAX_YAW_STEP   * recovery_scale
    max_pitch = MAX_PITCH_STEP * recovery_scale
    cam.yaw_vel   = float(np.clip(cam.yaw_vel,   -max_yaw,   max_yaw))
    cam.pitch_vel = float(np.clip(cam.pitch_vel, -max_pitch, max_pitch))

    # Update yaw/pitch
    cam.yaw   += cam.yaw_vel
    cam.pitch += cam.pitch_vel

    # Clamp to max deviation
    cam.yaw   = float(np.clip(cam.yaw,   -MAX_YAW_DEV,   MAX_YAW_DEV))
    # Pitch must always be negative — camera always looks down at pitch
    cam.pitch = float(np.clip(cam.pitch, e2p_tilt - MAX_PITCH_DEV, e2p_tilt + 5.0))


# ── Segment processing ────────────────────────────────────────────

def process_segment(insv_path: str, start_s: float, duration_s: float,
                    calib_path: str, e2p_tilt: float, e2p_fov: float,
                    player_model, ball_model, device: str,
                    writer, debug: bool = False,
                    yaw_init: float = 0.0,
                    csv_writer=None,
                    cluster_selector=None,
                    cluster_selector_features=None) -> int:

    cam        = CameraState(yaw=yaw_init, pitch=e2p_tilt)
    ball_state = KalmanBallTracker()
    sm_target  = TargetState(sm_tx=OUT_W/2, sm_ty=OUT_H/2)

    proc = open_stream(insv_path, start_s, duration_s)
    frame_idx  = 0
    ball_count = 0
    mode_counts = {'ball_highconf':0, 'ball':0, 'kalman':0, 'kalman_blend':0, 'blend':0, 'last_ball':0, 'players':0, 'few_players':0, 'single_player':0, 'hold':0, 'centre':0}
    mask_eroded = None
    last_players, last_ball = [], None
    t0 = time.time()

    while True:
        eq = read_frame(proc)
        if eq is None: break
        t = start_s + frame_idx / OUT_FPS
        if t > start_s + duration_s: break

        rgb = cv2.cvtColor(eq, cv2.COLOR_BGR2RGB)

        # ── Project current view ───────────────────────────────
        persp_rgb = e2p(rgb, fov_deg=e2p_fov,
                        u_deg=-cam.yaw, v_deg=cam.pitch,
                        out_hw=(OUT_H, OUT_W), mode='bilinear')
        persp = cv2.cvtColor(persp_rgb, cv2.COLOR_RGB2BGR)

        # ── Pitch mask in current view ─────────────────────────
        if frame_idx % (DETECT_EVERY * 5) == 0 or frame_idx == 0:
            # Mask always tracks cam.yaw exactly — no clamping
            # GK trap is prevented by weighted centroid, not mask manipulation
            mask = build_pitch_mask(calib_path, cam.yaw, cam.pitch,
                                    e2p_fov, OUT_H, OUT_W)
            # Eroded mask for ball detection — rejects near-edge false positives
            if mask is not None:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
                mask_eroded = cv2.erode(mask, kernel)
            else:
                mask_eroded = None

        # ── Detection ──────────────────────────────────────────
        if frame_idx % DETECT_EVERY == 0:
            last_players = detect_players(persp, player_model, device, mask)
            if ball_model is not None:
                # Use Kalman predicted position as spatial reference
                _last_trusted = ball_state.predicted_pos() if ball_state.is_valid() else None
                last_ball = detect_ball(persp, ball_model, device, mask, _last_trusted, mask_eroded)
            else:
                last_ball = None
            if last_ball: ball_count += 1

        # ── Choose target + update camera ──────────────────────
        tx, ty, mode = choose_target(last_players, last_ball, ball_state, cam.yaw, e2p_fov, cluster_selector, cluster_selector_features)
        mode_counts[mode] += 1
        update_camera(cam, tx, ty, sm_target, e2p_tilt=e2p_tilt, mode=mode)

        # ── Debug overlay ──────────────────────────────────────
        if debug:
            # Draw players
            for (cx, cy) in last_players:
                cv2.circle(persp, (int(cx), int(cy)), 8, (0,255,0), 2)
            # Draw ball
            if last_ball:
                cv2.circle(persp, (int(last_ball[0]), int(last_ball[1])),
                           12, (0,0,255), 3)
                cv2.putText(persp, f"{last_ball[2]:.2f}",
                            (int(last_ball[0])+14, int(last_ball[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            # Draw smoothed target
            cv2.circle(persp, (int(sm_target.sm_tx), int(sm_target.sm_ty)),
                       10, (0,255,255), 2)
            # Draw pitch mask contour — mask already tracks cam.yaw exactly
            if mask is not None:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(persp, contours, -1, (255,100,0), 1)
            # HUD
            cv2.putText(persp,
                        f"pan={cam.yaw:+.1f}° pitch={cam.pitch:.1f}° "
                        f"mode={mode} t={t-start_s:.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        else:
            cv2.putText(persp, f"pan={cam.yaw:+.1f}° t={t-start_s:.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        writer.stdin.write(persp.tobytes())
        if csv_writer is not None:
            _ball_conf = f"{last_ball[2]:.3f}" if last_ball else ""
            csv_writer.writerow([f"{t:.4f}", f"{cam.yaw:.4f}", mode, _ball_conf])
        frame_idx += 1

    proc.stdout.close()
    proc.wait()

    elapsed = time.time() - t0
    det_windows = frame_idx // DETECT_EVERY
    print(f"  {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)  "
          f"ball={ball_count}/{det_windows}  modes={mode_counts}")
    return frame_idx


# ── Main ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--insv',         required=True)
    p.add_argument('--calib',        required=True)
    p.add_argument('--output',       default='/tmp/infer_out.mp4')
    p.add_argument('--players',      default='models/yolo11s.pt')
    p.add_argument('--ball',         default='models/ball_v2.pt')  # yolov8s elevated
    p.add_argument('--segments',     type=int,   default=5)
    p.add_argument('--seg-duration', type=float, default=30.0)
    p.add_argument('--device',       default=None)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--start-times',  type=str,   default=None,
                   help='Comma-separated start times in seconds, overrides random selection')
    p.add_argument('--debug',        action='store_true')
    p.add_argument('--yaw-gain',     type=float, default=YAW_GAIN)
    p.add_argument('--target-alpha', type=float, default=TARGET_ALPHA)
    p.add_argument(
        "--yaw-inits",
        type=str, default=None, metavar="DEGREES",
        help="Comma-separated yaw_init values per segment, overrides warm-start",
    )
    p.add_argument(
        "--log-csv",
        type=str, default=None, metavar="PATH",
        help="Write per-frame pan log to this CSV file",
    )
    args = p.parse_args()

    # Allow tuning from command line
    yaw_gain     = args.yaw_gain
    target_alpha = args.target_alpha

    # Device
    if args.device is None:
        try:
            import torch
            device = ('mps' if torch.backends.mps.is_available() else
                      'cuda' if torch.cuda.is_available() else 'cpu')
        except ImportError:
            device = 'cpu'
    else:
        device = args.device
    print(f"Device: {device}")

    # Detectors
    from ultralytics import YOLO
    player_model = YOLO(args.players)
    ball_model   = YOLO(args.ball) if args.ball else None
    print("Detectors loaded")

    # Load cluster selector if available
    cluster_selector = None
    cluster_selector_features = None
    selector_path = pathlib.Path("models/cluster_selector.pkl")
    if selector_path.exists():
        with open(selector_path, "rb") as f:
            cluster_selector, cluster_selector_features = pickle.load(f)
        print("Cluster selector loaded")

    # Calibration
    print("Calibration...")
    e2p_tilt, e2p_fov = derive_tilt_fov(args.calib)

    # Duration + segments
    duration = get_duration(args.insv)
    print(f"Clip: {duration:.0f}s")
    if args.start_times:
        starts = [float(t) for t in args.start_times.split(',')]
    else:
        rng = np.random.default_rng(args.seed)
        max_start = max(10, duration - args.seg_duration - 5)
        starts = sorted(rng.uniform(10, max_start, args.segments).tolist())
    print(f"\nSegments:")
    for i, s in enumerate(starts):
        print(f"  {i+1}: {s:.0f}s – {s+args.seg_duration:.0f}s")

    # Process
    import csv as _csv
    _csv_file = None
    _csv_writer = None
    if args.log_csv:
        _csv_file = open(args.log_csv, "w", newline="")
        _csv_writer = _csv.writer(_csv_file)
        _csv_writer.writerow(["timestamp_s", "predicted_pan_deg", "mode", "ball_conf"])
    writer = open_writer(args.output, OUT_FPS)
    t0 = time.time()
    total = 0
    for i, start_s in enumerate(starts):
        print(f"\n[{i+1}/{args.segments}] t={start_s:.0f}s")
        # Warm-start yaw — use explicit value if provided, else detect from players
        yaw_init = 0.0
        if args.yaw_inits:
            explicit = [float(v) for v in args.yaw_inits.split(",")]
            if i < len(explicit):
                yaw_init = explicit[i]
                print(f"  Explicit yaw_init={yaw_init:.1f}°")
        else:
            warm_proc = open_stream(args.insv, start_s, 2)
            first_eq = read_frame(warm_proc)
            warm_proc.stdout.close(); warm_proc.wait()
            if first_eq is not None:
                rgb0 = cv2.cvtColor(first_eq, cv2.COLOR_BGR2RGB)
                p0 = e2p(rgb0, fov_deg=e2p_fov, u_deg=0, v_deg=e2p_tilt,
                         out_hw=(OUT_H, OUT_W), mode='bilinear')
                p0_bgr = cv2.cvtColor(p0, cv2.COLOR_RGB2BGR)
                players0 = detect_players(p0_bgr, player_model, device)
                if players0:
                    cx0 = float(np.mean([p[0] for p in players0]))
                    err_x = (cx0 - OUT_W/2) / OUT_W
                    yaw_init = err_x * e2p_fov * 0.5
                    print(f"  Warm start: {len(players0)} players → yaw_init={yaw_init:.1f}°")
        total += process_segment(
            args.insv, start_s, args.seg_duration,
            args.calib, e2p_tilt, e2p_fov,
            player_model, ball_model, device,
            writer, debug=args.debug,
            yaw_init=yaw_init,
            csv_writer=_csv_writer,
            cluster_selector=cluster_selector,
            cluster_selector_features=cluster_selector_features,
        )

    writer.stdin.close()
    if _csv_file is not None:
        _csv_file.close()
        print(f"[CSV] Pan log written to {args.log_csv}")
    writer.wait()
    elapsed = time.time() - t0
    print(f"\nDone: {total} frames in {elapsed:.1f}s ({total/elapsed:.1f} fps)")
    print(f"Output: {args.output}")


if __name__ == '__main__':
    main()
