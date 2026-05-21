#!/usr/bin/env python3
"""
autopan_infer.py — Autopan inference using PD control with Kalman ball tracking.

Pipeline:
1. Projects equirectangular frame to perspective view at current pan/tilt
2. Detects players (yolo11s) and ball (ball_v4) in perspective view
3. Kalman filter tracks ball position and predicts trajectory when lost
4. DBSCAN clustering finds densest player group as fallback target
5. Edge-trigger hold logic: only move camera when action near frame edge
6. PD control moves camera toward smoothed target

Usage:
    python autopan_infer.py \
        --insv  /path/to/VID_xxx_10_001.insv \
        --calib calibration/pitch.json \
        --output /tmp/out.mp4 \
        [--players models/yolo11s.pt] \
        [--ball    models/ball_v4.pt] \
        [--segments 5] [--seg-duration 30] \
        [--start-times 250,450,650] \
        [--device cpu|mps|cuda] \
        [--debug] \
        [--log-csv /tmp/pan_log.csv] \
        [--yaw-inits "-33.6,-16.4,+5.8"]
"""

from __future__ import annotations
import argparse, csv, json, math, pickle, pathlib, subprocess, time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import cv2
import numpy as np
from py360convert import e2p
from sklearn.cluster import DBSCAN


# ── Output ────────────────────────────────────────────────────────
OUT_W, OUT_H = 1280, 720
OUT_FPS      = 29.97

# ── Detection ─────────────────────────────────────────────────────
IMGSZ_PLAYERS  = 960
IMGSZ_BALL     = 640
CONF_PLAYERS   = 0.20
CONF_BALL      = 0.40
DETECT_EVERY   = 5       # run detection every N frames (~0.17s at 30fps)

# ── Camera control ────────────────────────────────────────────────
YAW_GAIN       = 0.30    # pan aggression toward target
PITCH_GAIN     = 0.22
MAX_YAW_STEP   = 1.6     # degrees per frame maximum
MAX_PITCH_STEP = 1.2
VEL_ALPHA      = 0.15    # velocity damping (lower = more damping)
TARGET_ALPHA   = 0.12    # target EMA smoothing
DEADBAND_X     = 0.08    # ignore errors < this fraction of frame width
DEADBAND_Y     = 0.06
MAX_YAW_DEV    = 55.0    # hard pan limit from centre
MAX_PITCH_DEV  =  5.0    # pitch deviation limit

# ── Kalman ball tracker ───────────────────────────────────────────
BALL_SHORT_FALLBACK   = 10     # frames before Kalman loses "fresh" status
BALL_MAX_JUMP_PX      = 150    # max px jump to trust without penalty
KALMAN_PROCESS_NOISE  = 50.0   # expected ball acceleration (px/frame²)
KALMAN_MEASURE_NOISE  = 15.0   # detection position uncertainty (px)
KALMAN_MAX_PREDICT    = 45     # max frames to trust Kalman prediction (~1.5s)
KALMAN_REINIT_DIST    = 200    # px — jump threshold to reinitialise filter

# ── DBSCAN player clustering ──────────────────────────────────────
DBSCAN_EPS_DEG    = 8.0   # cluster radius in equirect degrees
DBSCAN_MIN_SAMPLES = 2    # minimum players to form a cluster
DBSCAN_MIN_SIZE    = 3    # minimum cluster size to use as pan target
HOLD_THRESHOLD     = 0.80 # only move when cluster beyond this fraction of half-frame


# ── Calibration ───────────────────────────────────────────────────

def derive_tilt_fov(calib_path: str) -> Tuple[float, float]:
    """Derive perspective tilt and FOV from pitch polygon calibration."""
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
    e2p_tilt  = ((far_tilt + near_tilt) / 2) * 1.20
    e2p_fov   = float(np.clip(abs(near_tilt - far_tilt) * 1.85, 100, 130))
    print(f"  Calibration: far={far_tilt:.1f}° near={near_tilt:.1f}°")
    print(f"  e2p_tilt={e2p_tilt:.1f}° e2p_fov={e2p_fov:.1f}°")
    return e2p_tilt, e2p_fov


def build_pitch_mask(calib_path: str, yaw: float, pitch: float,
                     fov: float, h: int, w: int) -> Optional[np.ndarray]:
    """Project pitch polygon into perspective view as binary mask."""
    with open(calib_path) as f:
        d = json.load(f)
    raw = d.get('pitch_polygon') or d.get('pixel_polygon') or d.get('auto_polygon')
    if isinstance(raw[0], dict):
        poly_eq = np.array([[p['x']*2880, p['y']*1440] for p in raw], dtype=np.float32)
    else:
        poly_eq = np.array(raw, dtype=np.float32)

    mask_eq = np.zeros((1440, 2880), dtype=np.uint8)
    cv2.fillPoly(mask_eq, [poly_eq.astype(np.int32).reshape((-1, 1, 2))], 255)
    mask_rgb = np.stack([mask_eq]*3, axis=-1)
    proj = e2p(mask_rgb, fov_deg=fov, u_deg=-yaw, v_deg=pitch,
               out_hw=(h, w), mode='bilinear')
    return (proj[:,:,0] > 127).astype(np.uint8)


# ── FFmpeg I/O ────────────────────────────────────────────────────

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
    """Open FFmpeg pipe reading equirectangular frames."""
    cmd = [
        'ffmpeg', '-ss', str(max(0, start_s)), '-i', insv_path,
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
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{OUT_W}x{OUT_H}', '-r', str(fps), '-i', 'pipe:0',
        '-c:v', 'h264_videotoolbox', '-b:v', '8000k',
        '-pix_fmt', 'yuv420p', output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


# ── Detection ─────────────────────────────────────────────────────

def detect_players(frame: np.ndarray, model, device: str,
                   mask: Optional[np.ndarray] = None) -> List[Tuple[float, float]]:
    """Returns list of (cx, foot_y) for players within pitch mask."""
    names = model.names
    res = model(frame, imgsz=IMGSZ_PLAYERS, conf=CONF_PLAYERS,
                device=device, verbose=False)[0]
    centroids = []
    for b in res.boxes:
        if names.get(int(b.cls[0]), '') != 'person': continue
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        cx, foot_y = (x1+x2)/2, y2
        if mask is not None:
            xi = int(np.clip(cx, 0, mask.shape[1]-1))
            yi = int(np.clip(foot_y, 0, mask.shape[0]-1))
            if not mask[yi, xi]: continue
        centroids.append((cx, foot_y))
    return centroids


def detect_ball(frame: np.ndarray, model, device: str,
                mask: Optional[np.ndarray] = None,
                last_trusted_pos: Optional[Tuple[float, float]] = None,
                mask_eroded: Optional[np.ndarray] = None) -> Optional[Tuple[float, float, float]]:
    """Returns (cx, cy, conf) of best ball detection, or None."""
    res = model(frame, imgsz=IMGSZ_BALL, conf=CONF_BALL,
                device=device, verbose=False)[0]
    best = None
    for b in res.boxes:
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        if (x2-x1)*(y2-y1) > OUT_W * OUT_H * 0.02: continue  # too large
        cx, cy = (x1+x2)/2, (y1+y2)/2
        conf = float(b.conf[0])
        # Use eroded mask to reject near-edge detections
        bmask = mask_eroded if mask_eroded is not None else mask
        if bmask is not None:
            xi = int(np.clip(cx, 0, bmask.shape[1]-1))
            yi = int(np.clip(cy, 0, bmask.shape[0]-1))
            if not bmask[yi, xi]: continue
        # Penalise detections far from last trusted position
        if last_trusted_pos is not None:
            dist = math.hypot(cx - last_trusted_pos[0], cy - last_trusted_pos[1])
            if dist > BALL_MAX_JUMP_PX:
                conf *= 0.3
        if best is None or conf > best[2]:
            best = (cx, cy, conf)
    return best


# ── Camera state and Kalman tracker ──────────────────────────────

@dataclass
class CameraState:
    yaw:       float = 0.0
    pitch:     float = 0.0
    yaw_vel:   float = 0.0
    pitch_vel: float = 0.0


@dataclass
class TargetState:
    sm_tx: float = OUT_W / 2
    sm_ty: float = OUT_H / 2


class KalmanBallTracker:
    """Constant-velocity Kalman filter: state = [x, y, vx, vy].
    Predicts ball position when detection is lost, enabling the camera
    to follow ball trajectory even during occlusion."""

    def __init__(self):
        self._initialised = False
        self.frames_since_detection = 999
        self.frames_tracked = 0
        self.last_conf = 0.0
        self.last_pos: Optional[Tuple[float, float]] = None
        self.trusted = False
        self.frames_since = 999
        self._x = np.zeros((4, 1), dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64) * 500.0
        self._F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
        self._H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)
        q = KALMAN_PROCESS_NOISE
        self._Q = np.array([[q/4,0,q/2,0],[0,q/4,0,q/2],
                             [q/2,0,q,0],[0,q/2,0,q]], dtype=np.float64)
        r = KALMAN_MEASURE_NOISE ** 2
        self._R = np.array([[r,0],[0,r]], dtype=np.float64)

    def _init_state(self, cx: float, cy: float):
        self._x = np.array([[cx],[cy],[0.0],[0.0]], dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64) * 500.0
        self._initialised = True
        self.frames_tracked = 1
        self.frames_since_detection = 0
        self.frames_since = 0
        self.last_pos = (cx, cy)
        self.trusted = True

    def update(self, cx: float, cy: float, conf: float = 1.0):
        """Call when ball is detected at (cx, cy)."""
        if not self._initialised:
            self._init_state(cx, cy)
            self.last_conf = conf
            return
        pred_x, pred_y = float(self._x[0,0]), float(self._x[1,0])
        dist = math.hypot(cx - pred_x, cy - pred_y)
        if dist > KALMAN_REINIT_DIST and self.frames_since_detection == 0:
            return  # consecutive teleport — likely false positive, ignore
        if dist > KALMAN_REINIT_DIST * 2:
            self._init_state(cx, cy)  # extreme jump — reinitialise
            self.last_conf = conf
            return
        # Predict
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        # Update
        z = np.array([[cx],[cy]], dtype=np.float64)
        y_innov = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y_innov
        self._P = (np.eye(4) - K @ self._H) @ self._P
        self.frames_since_detection = 0
        self.frames_since = 0
        self.frames_tracked += 1
        self.last_conf = conf
        self.last_pos = (float(self._x[0,0]), float(self._x[1,0]))
        self.trusted = True

    def predict_only(self):
        """Call when ball is NOT detected — advances prediction."""
        if not self._initialised:
            self.frames_since_detection += 1
            self.frames_since += 1
            return
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        self.frames_since_detection += 1
        self.frames_since += 1
        # Decay velocity so prediction doesn't run away indefinitely
        decay = 0.92 ** min(self.frames_since_detection, 30)
        self._x[2] *= decay
        self._x[3] *= decay
        self.last_pos = (float(self._x[0,0]), float(self._x[1,0]))

    def predicted_pos(self) -> Optional[Tuple[float, float]]:
        if not self._initialised: return None
        return float(self._x[0,0]), float(self._x[1,0])

    def is_valid(self) -> bool:
        """True if prediction is within the trustworthy window."""
        return self._initialised and self.frames_since_detection < KALMAN_MAX_PREDICT

    def is_fresh(self) -> bool:
        """True if ball was detected very recently."""
        return self._initialised and self.frames_since_detection < BALL_SHORT_FALLBACK


# ── Clustering and target selection ──────────────────────────────

def pixel_to_lon(px: float, cam_yaw: float,
                 w: int = OUT_W, fov_deg: float = 100.0) -> float:
    """Convert perspective frame pixel x to equirectangular longitude (degrees)."""
    nx = (px - w / 2) / (w / 2)
    angle_x = math.degrees(math.atan(nx * math.tan(math.radians(fov_deg / 2))))
    return angle_x + cam_yaw


def find_action_cluster(players: List[Tuple[float, float]],
                        cam_yaw: float,
                        e2p_fov: float) -> Optional[Tuple[float, float]]:
    """Find the tightest/densest player cluster using DBSCAN.

    Clusters players in equirectangular longitude space.
    Scores clusters by size/spread — a tight cluster of 5 beats a
    loose cluster of 8, reducing goalkeeper trap susceptibility.

    Returns (cx_px, cy_px) of best cluster in perspective frame, or None.
    """
    if len(players) < DBSCAN_MIN_SAMPLES:
        return None

    lons = np.array([pixel_to_lon(p[0], cam_yaw, OUT_W, e2p_fov) for p in players])
    db = DBSCAN(eps=DBSCAN_EPS_DEG, min_samples=DBSCAN_MIN_SAMPLES).fit(
        lons.reshape(-1, 1))
    labels = db.labels_

    if len(set(labels)) - (1 if -1 in labels else 0) == 0:
        return None

    best_score, best_cx, best_cy = -1, None, None
    for label in set(labels):
        if label == -1: continue
        mask = labels == label
        if mask.sum() < DBSCAN_MIN_SIZE: continue
        cluster_px = [players[i] for i in range(len(players)) if mask[i]]
        cx = float(np.mean([p[0] for p in cluster_px]))
        cy = float(np.mean([p[1] for p in cluster_px]))
        spread = float(np.std(lons[mask])) + 1.0
        score = mask.sum() / spread  # tight dense cluster wins
        if score > best_score:
            best_score, best_cx, best_cy = score, cx, cy

    return (best_cx, best_cy) if best_cx is not None else None


def choose_target(
    players: List[Tuple[float, float]],
    ball: Optional[Tuple[float, float, float]],
    tracker: KalmanBallTracker,
    cam_yaw: float = 0.0,
    e2p_fov: float = 100.0,
) -> Tuple[float, float, str]:
    """Choose pan target. Priority order:
    1. ball_highconf — fresh high-conf detection near Kalman prediction
    2. ball          — Kalman smoothed position (fresh)
    3. kalman_blend  — Kalman prediction blended with player cluster
    4. kalman        — pure Kalman prediction
    5. players       — DBSCAN cluster (only if near frame edge)
    6. few_players   — weak weighted centroid (only if near frame edge)
    7. single_player — almost ignore, tiny pull toward player
    8. hold          — stay put (action comfortably in frame)
    9. centre        — no players visible, drift to centre
    """

    # Update Kalman
    if ball is not None:
        tracker.update(ball[0], ball[1], ball[2])
    else:
        tracker.predict_only()

    # 1. Fresh high-confidence ball — spatial gate prevents false positives
    #    from hijacking the camera when Kalman already has a good trajectory
    if ball is not None and ball[2] >= 0.50:
        if tracker.is_valid() and tracker.predicted_pos() is not None:
            kpx, kpy = tracker.predicted_pos()
            if math.hypot(ball[0]-kpx, ball[1]-kpy) <= KALMAN_REINIT_DIST:
                return ball[0], ball[1], 'ball_highconf'
            # else: far from prediction, treat as unconfirmed, fall through
        else:
            return ball[0], ball[1], 'ball_highconf'

    # 2. Fresh Kalman position (ball seen very recently)
    if tracker.is_fresh() and tracker.predicted_pos() is not None:
        px, py = tracker.predicted_pos()
        return px, py, 'ball'

    # 3+4. Kalman prediction still valid — blend with player cluster
    if tracker.is_valid() and tracker.predicted_pos() is not None:
        px, py = tracker.predicted_pos()
        age_ratio = tracker.frames_since_detection / KALMAN_MAX_PREDICT
        kalman_weight = max(0.85, 0.9 * (1.0 - age_ratio))

        cluster = find_action_cluster(players, cam_yaw, e2p_fov) \
            if len(players) >= DBSCAN_MIN_SAMPLES else None

        if cluster is not None:
            tx = kalman_weight * px + (1 - kalman_weight) * cluster[0]
            ty = kalman_weight * py + (1 - kalman_weight) * cluster[1]
            return tx, ty, 'kalman_blend'
        elif len(players) >= 4:
            xs = np.array([p[0] for p in players])
            ys = np.array([p[1] for p in players])
            med_x, med_y = float(np.median(xs)), float(np.median(ys))
            sigma = OUT_W * 0.25
            w = np.exp(-((xs-med_x)**2 + (ys-med_y)**2) / (2*sigma**2))
            w /= w.sum()
            tx = kalman_weight * px + (1-kalman_weight) * float(np.sum(w*xs))
            ty = kalman_weight * py + (1-kalman_weight) * float(np.sum(w*ys))
            return tx, ty, 'kalman_blend'
        return px, py, 'kalman'

    # 5. No ball info — edge-trigger hold: only move if cluster near frame edge
    if len(players) >= DBSCAN_MIN_SAMPLES:
        cluster = find_action_cluster(players, cam_yaw, e2p_fov)
        if cluster is not None:
            edge_dist = abs(cluster[0] - OUT_W / 2) / (OUT_W / 2)
            if edge_dist >= HOLD_THRESHOLD:
                return cluster[0], cluster[1], 'players'
            else:
                return OUT_W / 2, OUT_H / 2, 'hold'

    # 6. Few players — weak pull only if near edge
    if len(players) >= 2:
        xs = np.array([p[0] for p in players])
        ys = np.array([p[1] for p in players])
        med_x, med_y = float(np.median(xs)), float(np.median(ys))
        sigma = OUT_W * 0.25
        dists_sq = (xs-med_x)**2 + (ys-med_y)**2
        weights = np.exp(-dists_sq / (2*sigma**2))
        weights /= weights.sum()
        cx = float(np.sum(weights*xs))
        cy = float(np.sum(weights*ys))
        edge_dist = abs(cx - OUT_W / 2) / (OUT_W / 2)
        if edge_dist >= HOLD_THRESHOLD:
            tx = 0.2 * cx + 0.8 * (OUT_W/2)
            ty = 0.2 * cy + 0.8 * (OUT_H/2)
            return tx, ty, 'few_players'
        else:
            return OUT_W / 2, OUT_H / 2, 'hold'

    # 7. Single player — almost ignore
    if len(players) == 1:
        tx = 0.05 * players[0][0] + 0.95 * (OUT_W/2)
        ty = 0.05 * players[0][1] + 0.95 * (OUT_H/2)
        return tx, ty, 'single_player'

    # 8+9. No players
    return OUT_W / 2, OUT_H / 2, 'centre'


def update_camera(cam: CameraState, target_x: float, target_y: float,
                  sm_target: TargetState, e2p_tilt: float,
                  mode: str) -> None:
    """PD control: move camera toward smoothed target position."""

    # Hold — damp velocity, don't move
    if mode == 'hold':
        cam.yaw_vel   *= 0.5
        cam.pitch_vel *= 0.5
        return

    # EMA smoothing on target — faster for ball, slower for players
    if 'ball' in mode or 'kalman' in mode:
        base_alpha = TARGET_ALPHA * 3
        error_norm = abs(sm_target.sm_tx - OUT_W/2) / (OUT_W/2)
        alpha = min(0.8, base_alpha * (1.0 + 5.0 * error_norm))
    else:
        alpha = TARGET_ALPHA * 0.5  # slow for player-only to reduce noise sensitivity
    sm_target.sm_tx += alpha * (target_x - sm_target.sm_tx)
    sm_target.sm_ty += alpha * (target_y - sm_target.sm_ty)

    # Normalised error
    err_x = (sm_target.sm_tx - OUT_W/2) / OUT_W
    err_y = (sm_target.sm_ty - OUT_H/2) / OUT_H

    # Deadband
    if abs(err_x) < DEADBAND_X: err_x = 0.0
    if abs(err_y) < DEADBAND_Y: err_y = 0.0

    # Velocity update with damping
    cam.yaw_vel   = cam.yaw_vel   * (1 - VEL_ALPHA) + err_x * YAW_GAIN
    cam.pitch_vel = cam.pitch_vel * (1 - VEL_ALPHA) + err_y * PITCH_GAIN

    # Clamp step size — scale up recovery when far off target
    target_err    = abs(sm_target.sm_tx - OUT_W/2) / (OUT_W/2)
    recovery      = 1.0 + 2.0 * max(0, target_err - 0.3)
    cam.yaw_vel   = float(np.clip(cam.yaw_vel,   -MAX_YAW_STEP*recovery,   MAX_YAW_STEP*recovery))
    cam.pitch_vel = float(np.clip(cam.pitch_vel, -MAX_PITCH_STEP*recovery, MAX_PITCH_STEP*recovery))

    cam.yaw   += cam.yaw_vel
    cam.pitch += cam.pitch_vel

    cam.yaw   = float(np.clip(cam.yaw,   -MAX_YAW_DEV, MAX_YAW_DEV))
    cam.pitch = float(np.clip(cam.pitch,
                               e2p_tilt - MAX_PITCH_DEV, e2p_tilt + 5.0))


# ── Warm start ────────────────────────────────────────────────────

def scan_warm_start(insv_path: str, start_s: float,
                    e2p_tilt: float, e2p_fov: float,
                    player_model, ball_model, device: str,
                    calib_path: str) -> float:
    """Scan multiple pan angles to find the best initial yaw.

    Tries 9 candidate yaw angles. For each:
    - Checks for high-confidence ball detection near frame centre
    - Otherwise scores player cluster by size × centredness

    Returns best yaw_init in degrees.
    """
    SCAN_YAWS = [-40, -30, -20, -10, 0, 10, 20, 30, 40]

    warm_proc = open_stream(insv_path, start_s, 0.5)
    warm_eq   = read_frame(warm_proc)
    warm_proc.stdout.close(); warm_proc.wait()

    if warm_eq is None:
        print("  Warm start: frame read failed, defaulting to 0°")
        return 0.0

    warm_rgb = cv2.cvtColor(warm_eq, cv2.COLOR_BGR2RGB)
    cluster_candidates = []

    for scan_yaw in SCAN_YAWS:
        pv_rgb = e2p(warm_rgb, fov_deg=e2p_fov,
                     u_deg=-scan_yaw, v_deg=e2p_tilt,
                     out_hw=(OUT_H, OUT_W), mode='bilinear')
        pv_bgr = cv2.cvtColor(pv_rgb, cv2.COLOR_RGB2BGR)

        # Ball detection — only trust if near frame centre
        if ball_model is not None:
            ball_det = detect_ball(pv_bgr, ball_model, device)
            if ball_det is not None and ball_det[2] >= 0.60:
                edge = abs(ball_det[0] - OUT_W/2) / (OUT_W/2)
                if edge < 0.4:  # tighter — only use ball near frame centre
                    nx = (ball_det[0] - OUT_W/2) / (OUT_W/2)
                    angle_x = math.degrees(math.atan(nx * math.tan(math.radians(e2p_fov/2))))
                    yaw_init = scan_yaw + angle_x
                    print(f"  Warm start: ball at yaw={scan_yaw:+d}° "
                          f"conf={ball_det[2]:.2f} → yaw_init={yaw_init:.1f}°")
                    return yaw_init

        # Player cluster candidate
        players = detect_players(pv_bgr, player_model, device)
        if len(players) >= 4:
            cluster = find_action_cluster(players, scan_yaw, e2p_fov)
            if cluster is not None:
                cx = cluster[0]
                centredness = 1.0 - abs(cx - OUT_W/2) / (OUT_W/2)
                score = len(players) * centredness
                nx = (cx - OUT_W/2) / (OUT_W/2)
                pred_yaw = scan_yaw + math.degrees(
                    math.atan(nx * math.tan(math.radians(e2p_fov/2))))
                cluster_candidates.append((score, pred_yaw, scan_yaw, len(players)))

    if cluster_candidates:
        cluster_candidates.sort(reverse=True)
        best = cluster_candidates[0]
        print(f"  Warm start: best cluster at yaw={best[2]:+d}° "
              f"score={best[0]:.1f} players={best[3]} → yaw_init={best[1]:.1f}°")
        return best[1]

    print("  Warm start: no cluster found, defaulting to 0°")
    return 0.0


# ── Segment processing ────────────────────────────────────────────

def process_segment(insv_path: str, start_s: float, duration_s: float,
                    calib_path: str, e2p_tilt: float, e2p_fov: float,
                    player_model, ball_model, device: str,
                    writer, debug: bool = False,
                    yaw_init: float = 0.0,
                    csv_writer=None) -> int:

    cam       = CameraState(yaw=yaw_init, pitch=e2p_tilt)
    tracker   = KalmanBallTracker()
    sm_target = TargetState(sm_tx=OUT_W/2, sm_ty=OUT_H/2)

    proc = open_stream(insv_path, start_s, duration_s)
    frame_idx  = 0
    ball_count = 0
    mode_counts = {m: 0 for m in [
        'ball_highconf', 'ball', 'kalman', 'kalman_blend',
        'players', 'few_players', 'single_player', 'hold', 'centre']}
    mask = mask_eroded = None
    last_players, last_ball = [], None
    t0 = time.time()

    while True:
        eq = read_frame(proc)
        if eq is None: break
        t = start_s + frame_idx / OUT_FPS
        if t > start_s + duration_s: break

        rgb = cv2.cvtColor(eq, cv2.COLOR_BGR2RGB)

        # Project to perspective at current pan/tilt
        persp_rgb = e2p(rgb, fov_deg=e2p_fov,
                        u_deg=-cam.yaw, v_deg=cam.pitch,
                        out_hw=(OUT_H, OUT_W), mode='bilinear')
        persp = cv2.cvtColor(persp_rgb, cv2.COLOR_RGB2BGR)

        # Rebuild pitch mask periodically
        if frame_idx % (DETECT_EVERY * 5) == 0 or frame_idx == 0:
            mask = build_pitch_mask(calib_path, cam.yaw, cam.pitch,
                                    e2p_fov, OUT_H, OUT_W)
            if mask is not None:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
                mask_eroded = cv2.erode(mask, kernel)
            else:
                mask_eroded = None

        # Detection every DETECT_EVERY frames
        if frame_idx % DETECT_EVERY == 0:
            last_players = detect_players(persp, player_model, device, mask)
            if ball_model is not None:
                last_trusted = tracker.predicted_pos() if tracker.is_valid() else None
                last_ball = detect_ball(persp, ball_model, device,
                                        mask, last_trusted, mask_eroded)
            else:
                last_ball = None
            if last_ball: ball_count += 1

        # Choose target and update camera
        tx, ty, mode = choose_target(
            last_players, last_ball, tracker, cam.yaw, e2p_fov)
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        update_camera(cam, tx, ty, sm_target, e2p_tilt=e2p_tilt, mode=mode)

        # Debug overlay
        if debug:
            for cx, cy in last_players:
                cv2.circle(persp, (int(cx), int(cy)), 8, (0,255,0), 2)
            if last_ball:
                cv2.circle(persp, (int(last_ball[0]), int(last_ball[1])),
                           12, (0,0,255), 3)
                cv2.putText(persp, f"{last_ball[2]:.2f}",
                            (int(last_ball[0])+14, int(last_ball[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.circle(persp, (int(sm_target.sm_tx), int(sm_target.sm_ty)),
                       10, (0,255,255), 2)
            if mask is not None:
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(persp, contours, -1, (255,100,0), 1)
            cv2.putText(persp,
                        f"pan={cam.yaw:+.1f}° pitch={cam.pitch:.1f}° "
                        f"mode={mode} t={t-start_s:.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        else:
            cv2.putText(persp, f"pan={cam.yaw:+.1f}° t={t-start_s:.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        writer.stdin.write(persp.tobytes())
        if csv_writer is not None:
            ball_conf = f"{last_ball[2]:.3f}" if last_ball else ""
            csv_writer.writerow([f"{t:.4f}", f"{cam.yaw:.4f}", mode, ball_conf])
        frame_idx += 1

    proc.stdout.close()
    proc.wait()

    elapsed = time.time() - t0
    det_windows = frame_idx // DETECT_EVERY
    print(f"  {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)  "
          f"ball={ball_count}/{det_windows}  modes={mode_counts}")
    return frame_idx


# ── Entry point ───────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--insv',         required=True,  help='Input .insv file')
    p.add_argument('--calib',        required=True,  help='Calibration JSON')
    p.add_argument('--output',       default='/tmp/infer_out.mp4')
    p.add_argument('--players',      default='models/yolo11s.pt')
    p.add_argument('--ball',         default='models/ball_v4.pt')
    p.add_argument('--segments',     type=int,   default=5)
    p.add_argument('--seg-duration', type=float, default=30.0)
    p.add_argument('--device',       default=None)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--start-times',  type=str,   default=None,
                   help='Comma-separated start times (seconds); overrides random')
    p.add_argument('--yaw-inits',    type=str,   default=None, metavar='DEGREES',
                   help='Comma-separated yaw_init per segment; overrides scan warm-start')
    p.add_argument('--debug',        action='store_true')
    p.add_argument('--log-csv',      type=str,   default=None, metavar='PATH',
                   help='Write per-frame pan log to CSV')
    args = p.parse_args()

    # Device selection
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

    # Load models
    from ultralytics import YOLO
    player_model = YOLO(args.players)
    ball_model   = YOLO(args.ball) if args.ball else None
    print("Detectors loaded")

    # Calibration
    print("Calibration...")
    e2p_tilt, e2p_fov = derive_tilt_fov(args.calib)

    # Segment start times
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

    # CSV output
    _csv_file = _csv_writer = None
    if args.log_csv:
        _csv_file = open(args.log_csv, "w", newline="")
        _csv_writer = csv.writer(_csv_file)
        _csv_writer.writerow(["timestamp_s", "predicted_pan_deg", "mode", "ball_conf"])

    writer = open_writer(args.output, OUT_FPS)
    t0 = time.time()
    total = 0

    for i, start_s in enumerate(starts):
        print(f"\n[{i+1}/{len(starts)}] t={start_s:.0f}s")

        # Warm start
        if args.yaw_inits:
            explicit = [float(v) for v in args.yaw_inits.split(",")]
            yaw_init = explicit[i] if i < len(explicit) else 0.0
            print(f"  Explicit yaw_init={yaw_init:.1f}°")
        else:
            yaw_init = scan_warm_start(
                args.insv, start_s, e2p_tilt, e2p_fov,
                player_model, ball_model, device, args.calib)

        total += process_segment(
            args.insv, start_s, args.seg_duration,
            args.calib, e2p_tilt, e2p_fov,
            player_model, ball_model, device,
            writer, debug=args.debug,
            yaw_init=yaw_init,
            csv_writer=_csv_writer,
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
