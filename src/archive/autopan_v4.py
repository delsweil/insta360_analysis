import os
import time
import math
import json
import cv2
import numpy as np
from ultralytics import YOLO
from py360convert import e2p

# =========================
# CONFIG
# =========================

INPUT_PATH = "/Users/davidelsweiler/Desktop/test_run.mp4"          # 360° equirectangular video
OUTPUT_PATH = "data/processed/autopan_short_test_v4.mp4"

# Perspective output
OUT_W, OUT_H = 1280, 720
FOV_DEG = 90

# Initial camera direction (use what worked for you)
YAW_DEG_INITIAL = 0.0
PITCH_DEG_INITIAL = 0.0

# Models (LOCAL FILES)
PLAYER_MODEL_PATH = "models/yolo11n.pt"   # or yolo11m.pt
BALL_MODEL_PATH   = "models/ball.pt"

# YOLO parameters
IMG_SIZE_PLAYERS = 960
IMG_SIZE_BALL    = 640
CONF_PLAYERS     = 0.35
CONF_BALL        = 0.20

# Motion control (slow & smooth)
YAW_GAIN       = 0.30
PITCH_GAIN     = 0.22
MAX_YAW_STEP   = 1.6
MAX_PITCH_STEP = 1.2

# Exponential smoothing of target point (lower = smoother)
TARGET_SMOOTH_ALPHA = 0.12

# Ball gating
BALL_CONFIRM_FRAMES = 3
BALL_MISS_SHORT_FALLBACK = 10     # keep using last ball briefly
BALL_MISS_LONG_FALLBACK  = 90     # after this, recenter/home drift

# Hard corridor around initial view (prevents “stuck off pitch”)
CENTERING_FORCE = 0.01          # was ~0.03
MAX_YAW_DEVIATION = 40.0        # was ~30
MAX_PITCH_DEVIATION = 14.0      # was ~12

DEADBAND_X = 0.08   # 8% of frame width
DEADBAND_Y = 0.06

# Pitch polygon calibration file (created by calibrate_pitch.py)
CALIB_PATH = "calibration/pitch.json"

# Optional: draw debug overlays
DRAW_DEBUG = True

# --- Camera motion state ---
yaw_vel = 0.0
pitch_vel = 0.0

VEL_ALPHA = 0.15   # inertia (0.1–0.2 works well)


# =========================
# Helpers
# =========================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def soft_step(angle, delta, max_step):
    return angle + clamp(delta, -max_step, max_step)

def clamp_around_center(angle, center, max_dev):
    return clamp(angle, center - max_dev, center + max_dev)

def load_pitch_polygon(path):
    if not os.path.exists(path):
        print(f"[WARN] No pitch calibration found at {path}. "
              f"AutoPan will run without pitch filtering (less stable).")
        return None
    with open(path, "r") as f:
        data = json.load(f)
    poly = data.get("pitch_polygon", None)
    if not poly or len(poly) < 3:
        print(f"[WARN] pitch.json exists but no valid polygon. Ignoring.")
        return None
    poly_np = np.array(poly, dtype=np.int32)
    print(f"[OK] Loaded pitch polygon with {len(poly)} points from {path}")
    return poly_np

def inside_polygon(poly_np, x, y):
    # True if inside or on edge
    return cv2.pointPolygonTest(poly_np, (float(x), float(y)), False) >= 0


# =========================
# Sanity checks
# =========================

print("CWD:", os.getcwd())
for p in [PLAYER_MODEL_PATH, BALL_MODEL_PATH]:
    print("Model exists?", p, os.path.exists(p))
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing model file: {p}")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


# =========================
# Load calibration + models
# =========================

pitch_poly = load_pitch_polygon(CALIB_PATH)

print("Loading models...")
player_model = YOLO(PLAYER_MODEL_PATH)
ball_model   = YOLO(BALL_MODEL_PATH)
print("Models loaded.")

# =========================
# Video IO
# =========================

cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {INPUT_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Input: {in_w}x{in_h} @ {fps:.2f} fps, {n_frames} frames")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (OUT_W, OUT_H))

# =========================
# AutoPan state
# =========================

yaw = YAW_DEG_INITIAL
pitch = PITCH_DEG_INITIAL

yaw_center = YAW_DEG_INITIAL
pitch_center = PITCH_DEG_INITIAL

sm_tx = OUT_W / 2
sm_ty = OUT_H / 2

last_ball = None
ball_confirm = 0
ball_trusted = False
frames_since_ball = 999999

t0 = time.time()
idx = 0

# =========================
# Main loop
# =========================

while True:
    ok, frame360 = cap.read()
    if not ok:
        break

    # Project to perspective
    persp = e2p(frame360, fov_deg=FOV_DEG, u_deg=yaw, v_deg=pitch, out_hw=(OUT_H, OUT_W))

    # Detect players
    res_p = player_model(persp, imgsz=IMG_SIZE_PLAYERS, conf=CONF_PLAYERS, verbose=False)[0]
    names_p = player_model.names

    players_for_centroid = []
    player_boxes = []

    for b in res_p.boxes:
        cls_id = int(b.cls[0])
        cls_name = names_p.get(cls_id, str(cls_id))
        if cls_name != "person":
            continue

        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cx = (x1 + x2) // 2
        foot_y = y2

        # Pitch filtering: ignore people outside pitch polygon
        if pitch_poly is not None:
            if not inside_polygon(pitch_poly, cx, foot_y):
                continue

        player_boxes.append((x1, y1, x2, y2))
        players_for_centroid.append((cx, (y1 + y2) // 2))

    # Detect ball
    res_b = ball_model(persp, imgsz=IMG_SIZE_BALL, conf=CONF_BALL, verbose=False)[0]
    ball_candidates = []
    for b in res_b.boxes:
        conf = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        area = max(1, (x2 - x1) * (y2 - y1))

        # sanity: reject absurdly large "ball"
        if area > (OUT_W * OUT_H * 0.02):
            continue

        # pitch filter for ball as well
        if pitch_poly is not None:
            if not inside_polygon(pitch_poly, cx, cy):
                continue

        ball_candidates.append((conf, area, cx, cy, x1, y1, x2, y2))

    ball_pos = None
    used_ball = False

    if ball_candidates:
        # highest confidence, then largest
        ball_candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
        conf, area, cx, cy, x1, y1, x2, y2 = ball_candidates[0]
        ball_pos = (cx, cy)

        ball_confirm += 1
        if ball_confirm >= BALL_CONFIRM_FRAMES:
            ball_trusted = True

        last_ball = ball_pos
        frames_since_ball = 0
        used_ball = ball_trusted
    else:
        frames_since_ball += 1
        ball_confirm = 0  # must reconfirm

    # Choose target
    mode = "Center"
    tx, ty = OUT_W / 2, OUT_H / 2

    if used_ball and ball_pos is not None:
        tx, ty = ball_pos
        mode = "Ball"
    elif last_ball is not None and frames_since_ball < BALL_MISS_SHORT_FALLBACK:
        tx, ty = last_ball
        mode = "Last ball"
    elif players_for_centroid and frames_since_ball < BALL_MISS_LONG_FALLBACK:
        sx = sum(p[0] for p in players_for_centroid)
        sy = sum(p[1] for p in players_for_centroid)
        tx = sx / len(players_for_centroid)
        ty = sy / len(players_for_centroid)
        mode = "Players"
    else:
        mode = "Home drift"
        tx, ty = OUT_W / 2, OUT_H / 2
        yaw_vel *= 0.9
        pitch_vel *= 0.9

    # Smooth the target point
    sm_tx = (1 - TARGET_SMOOTH_ALPHA) * sm_tx + TARGET_SMOOTH_ALPHA * tx
    sm_ty = (1 - TARGET_SMOOTH_ALPHA) * sm_ty + TARGET_SMOOTH_ALPHA * ty

    # Compute yaw/pitch deltas from target point
    tx_norm = sm_tx / OUT_W
    ty_norm = sm_ty / OUT_H

    err_x = tx_norm - 0.5
    err_y = ty_norm - 0.5

    if abs(err_x) < DEADBAND_X:
        err_x = 0.0
    if abs(err_y) < DEADBAND_Y:
        err_y = 0.0

    # --- Desired angular velocity ---
    desired_yaw_vel = YAW_GAIN * err_x * FOV_DEG
    desired_pitch_vel = PITCH_GAIN * err_y * FOV_DEG * -1.0

    # --- Velocity smoothing (inertia) ---
    yaw_vel = (1.0 - VEL_ALPHA) * yaw_vel + VEL_ALPHA * desired_yaw_vel
    pitch_vel = (1.0 - VEL_ALPHA) * pitch_vel + VEL_ALPHA * desired_pitch_vel

    # Apply limited per-frame motion using SMOOTHED VELOCITY
    yaw   = soft_step(yaw,   yaw_vel,   MAX_YAW_STEP)
    pitch = soft_step(pitch, pitch_vel, MAX_PITCH_STEP)


    # Gentle centering force (prevents long drift)
    #yaw += (yaw_center - yaw) * CENTERING_FORCE
    #pitch += (pitch_center - pitch) * CENTERING_FORCE

    # Hard corridor limits (prevents “stuck off pitch”)
    yaw = clamp_around_center(yaw, yaw_center, MAX_YAW_DEVIATION)
    pitch = clamp_around_center(pitch, pitch_center, MAX_PITCH_DEVIATION)

    # Draw debug overlays
    if DRAW_DEBUG:
        if pitch_poly is not None:
            cv2.polylines(persp, [pitch_poly], isClosed=True, color=(255, 255, 0), thickness=2)

        for (x1, y1, x2, y2) in player_boxes:
            cv2.rectangle(persp, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if ball_pos is not None:
            col = (0, 255, 255) if used_ball else (0, 140, 255)
            cv2.circle(persp, ball_pos, 8, col, 2)

        cv2.circle(persp, (int(sm_tx), int(sm_ty)), 5, (255, 255, 255), 2)

        cv2.putText(persp, f"Frame {idx}  yaw={yaw:.1f} pitch={pitch:.1f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(persp, f"Mode: {mode}  ball_miss={frames_since_ball}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    out.write(persp)
    idx += 1

    if idx % 100 == 0:
        elapsed = time.time() - t0
        eff = idx / elapsed if elapsed > 0 else 0
        print(f"Processed {idx}/{n_frames} frames ({eff:.2f} fps)")

cap.release()
out.release()

elapsed = time.time() - t0
eff = idx / elapsed if elapsed > 0 else 0
print("\n--- AutoPan v3 complete ---")
print(f"Frames: {idx}")
print(f"Time: {elapsed:.2f}s  Effective FPS: {eff:.2f}")
print(f"Output: {OUTPUT_PATH}")
