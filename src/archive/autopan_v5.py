import os
import time
import json
import math
import cv2
import numpy as np
from ultralytics import YOLO
from py360convert import e2p

# =========================
# CONFIG
# =========================
INPUT_PATH  = "/Users/davidelsweiler/Desktop/test_run.mp4"
OUTPUT_PATH = "data/processed/autopan_short_test_v5.mp4"

OUT_W, OUT_H = 1280, 720
FOV_DEG = 90

YAW_DEG_INITIAL = 0.0
PITCH_DEG_INITIAL = 0.0

PLAYER_MODEL_PATH = "models/yolo11n.pt"
BALL_MODEL_PATH   = "models/ball.pt"

IMG_SIZE_PLAYERS = 960
IMG_SIZE_BALL    = 640
CONF_PLAYERS     = 0.35
CONF_BALL        = 0.20

YAW_GAIN       = 0.30
PITCH_GAIN     = 0.22
MAX_YAW_STEP   = 1.6
MAX_PITCH_STEP = 1.2

TARGET_SMOOTH_ALPHA = 0.12

BALL_CONFIRM_FRAMES = 3
BALL_MISS_SHORT_FALLBACK = 10
BALL_MISS_LONG_FALLBACK  = 90

MAX_YAW_DEVIATION = 40.0
MAX_PITCH_DEVIATION = 14.0

DEADBAND_X = 0.08
DEADBAND_Y = 0.06

VEL_ALPHA = 0.15

# NEW: 360 polygon calibration
CALIB_360_PATH = "calibration/pitch_360.json"

DRAW_DEBUG = True


# =========================
# Helpers
# =========================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def soft_step(angle, delta, max_step):
    return angle + clamp(delta, -max_step, max_step)

def clamp_around_center(angle, center, max_dev):
    return clamp(angle, center - max_dev, center + max_dev)

def load_pitch_poly_360(path):
    if not os.path.exists(path):
        print(f"[WARN] Missing 360 pitch calibration: {path}")
        return None

    with open(path, "r") as f:
        data = json.load(f)

    poly = data.get("pitch_polygon_equirect", None)
    size = data.get("equirect_size", None)
    if not poly or len(poly) < 3 or not size:
        print("[WARN] Invalid 360 calibration JSON.")
        return None

    W, H = size
    poly_np = np.array(poly, dtype=np.float32)
    print(f"[OK] Loaded 360 pitch polygon: {len(poly)} points, equirect {W}x{H}")
    return (poly_np, W, H)

def equirect_xy_to_dir(x, y, W, H):
    """Map equirect pixel -> unit direction vector in world coords."""
    lon = (x / W - 0.5) * 2.0 * math.pi
    lat = (0.5 - y / H) * math.pi

    vx = math.cos(lat) * math.sin(lon)
    vy = math.sin(lat)
    vz = math.cos(lat) * math.cos(lon)
    return np.array([vx, vy, vz], dtype=np.float32)

def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float32)

def rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]], dtype=np.float32)

def project_dir_to_persp(v_cam, out_w, out_h, fov_deg):
    """Project camera-space direction vector -> perspective pixel, or None if behind camera."""
    x, y, z = float(v_cam[0]), float(v_cam[1]), float(v_cam[2])
    if z <= 1e-6:
        return None

    f = 0.5 * out_w / math.tan(math.radians(fov_deg) * 0.5)
    u = f * (x / z) + out_w * 0.5
    v = f * (-y / z) + out_h * 0.5
    return (u, v)

def project_poly_360_to_persp(poly360, W360, H360, yaw_deg, pitch_deg, out_w, out_h, fov_deg, samples_per_edge=50):
    # World->camera rotation: try this first (matches e2p in most setups)
    #R = rot_x(-pitch_deg) @ rot_y(+yaw_deg)
    #R = rot_y(+yaw_deg) @ rot_x(-pitch_deg)
    R = rot_x(-pitch_deg) @ rot_y(+yaw_deg)

    # Densify edges
    dense = []
    n = len(poly360)
    for i in range(n):
        x1, y1 = poly360[i]
        x2, y2 = poly360[(i + 1) % n]

        # handle seam wrap (equirect x wraps around)
        dx = x2 - x1
        if abs(dx) > W360 / 2:
            # go the shorter way across the seam
            if dx > 0:
                x2 -= W360
            else:
                x2 += W360

        for t in np.linspace(0, 1, samples_per_edge, endpoint=False):
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            # wrap back into [0, W)
            x = x % W360
            dense.append((x, y))

    pts = []
    for (x, y) in dense:
        v_world = equirect_xy_to_dir(x, y, W360, H360)
        v_cam = R @ v_world
        uv = project_dir_to_persp(v_cam, out_w, out_h, fov_deg)
        if uv is None:
            continue
        u, v = uv
        pts.append([u, v])

    if len(pts) < 3:
        return None

    pts_np = np.array(pts, dtype=np.float32)

    # Make a stable polygon from the projected boundary samples
    hull = cv2.convexHull(pts_np).astype(np.int32)
    return hull


def inside_polygon(poly_np, x, y):
    return cv2.pointPolygonTest(poly_np, (float(x), float(y)), False) >= 0


# =========================
# Setup
# =========================
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

for p in [PLAYER_MODEL_PATH, BALL_MODEL_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing model file: {p}")

pitch360 = load_pitch_poly_360(CALIB_360_PATH)

print("Loading models...")
player_model = YOLO(PLAYER_MODEL_PATH)
ball_model   = YOLO(BALL_MODEL_PATH)
print("Models loaded.")

cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {INPUT_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Input: {in_w}x{in_h} @ {fps:.2f} fps, {n_frames} frames")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (OUT_W, OUT_H))

yaw = YAW_DEG_INITIAL
pitch = PITCH_DEG_INITIAL
yaw_center = YAW_DEG_INITIAL
pitch_center = PITCH_DEG_INITIAL

sm_tx = OUT_W / 2
sm_ty = OUT_H / 2

yaw_vel = 0.0
pitch_vel = 0.0

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

    persp = e2p(frame360, fov_deg=FOV_DEG, u_deg=yaw, v_deg=pitch, out_hw=(OUT_H, OUT_W))

    # Project 360 polygon into this view (for steering only)
    pitch_poly = None
    if pitch360 is not None:
        poly360, W360, H360 = pitch360
        pitch_poly = project_poly_360_to_persp(poly360, W360, H360, yaw, pitch, OUT_W, OUT_H, FOV_DEG)

    # --- Detect players (inclusive) ---
    res_p = player_model(persp, imgsz=IMG_SIZE_PLAYERS, conf=CONF_PLAYERS, verbose=False)[0]
    names_p = player_model.names

    player_boxes = []             # [(x1,y1,x2,y2,in_pitch)]
    players_for_centroid = []     # only in_pitch contribute to steering

    for b in res_p.boxes:
        cls_id = int(b.cls[0])
        cls_name = names_p.get(cls_id, str(cls_id))
        if cls_name != "person":
            continue

        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cx = (x1 + x2) // 2
        foot_y = y2

        in_pitch = True
        if pitch_poly is not None:
            in_pitch = inside_polygon(pitch_poly, cx, foot_y)

        player_boxes.append((x1, y1, x2, y2, in_pitch))
        if in_pitch:
            players_for_centroid.append((cx, (y1 + y2) // 2))

    # --- Detect ball (still filter with pitch_poly to avoid stands) ---
    res_b = ball_model(persp, imgsz=IMG_SIZE_BALL, conf=CONF_BALL, verbose=False)[0]
    ball_candidates = []
    for b in res_b.boxes:
        conf = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        area = max(1, (x2 - x1) * (y2 - y1))
        if area > (OUT_W * OUT_H * 0.02):
            continue
        if pitch_poly is not None and not inside_polygon(pitch_poly, cx, cy):
            continue
        ball_candidates.append((conf, area, cx, cy))

    ball_pos = None
    used_ball = False

    if ball_candidates:
        ball_candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
        conf, area, cx, cy = ball_candidates[0]
        ball_pos = (cx, cy)
        ball_confirm += 1
        if ball_confirm >= BALL_CONFIRM_FRAMES:
            ball_trusted = True
        last_ball = ball_pos
        frames_since_ball = 0
        used_ball = ball_trusted
    else:
        frames_since_ball += 1
        ball_confirm = 0

    # --- Choose target ---
    mode = "Center"
    tx, ty = OUT_W / 2, OUT_H / 2

    if used_ball and ball_pos is not None:
        tx, ty = ball_pos
        mode = "Ball"
    elif last_ball is not None and frames_since_ball < BALL_MISS_SHORT_FALLBACK:
        tx, ty = last_ball
        mode = "Last ball"
    elif players_for_centroid and frames_since_ball < BALL_MISS_LONG_FALLBACK:
        # guard: don’t steer from players if too few in-pitch players visible
        if len(players_for_centroid) >= 4:
            sx = sum(p[0] for p in players_for_centroid)
            sy = sum(p[1] for p in players_for_centroid)
            tx = sx / len(players_for_centroid)
            ty = sy / len(players_for_centroid)
            mode = "Players"
        else:
            mode = "Home drift"
            tx, ty = OUT_W / 2, OUT_H / 2
    else:
        mode = "Home drift"
        tx, ty = OUT_W / 2, OUT_H / 2

    # Smooth target point
    sm_tx = (1 - TARGET_SMOOTH_ALPHA) * sm_tx + TARGET_SMOOTH_ALPHA * tx
    sm_ty = (1 - TARGET_SMOOTH_ALPHA) * sm_ty + TARGET_SMOOTH_ALPHA * ty

    tx_norm = sm_tx / OUT_W
    ty_norm = sm_ty / OUT_H
    err_x = tx_norm - 0.5
    err_y = ty_norm - 0.5

    if abs(err_x) < DEADBAND_X:
        err_x = 0.0
    if abs(err_y) < DEADBAND_Y:
        err_y = 0.0

    desired_yaw_vel = YAW_GAIN * err_x * FOV_DEG
    desired_pitch_vel = PITCH_GAIN * err_y * FOV_DEG * -1.0

    yaw_vel = (1.0 - VEL_ALPHA) * yaw_vel + VEL_ALPHA * desired_yaw_vel
    pitch_vel = (1.0 - VEL_ALPHA) * pitch_vel + VEL_ALPHA * desired_pitch_vel

    if mode == "Home drift":
        yaw_vel *= 0.9
        pitch_vel *= 0.9

    yaw   = soft_step(yaw,   yaw_vel,   MAX_YAW_STEP)
    pitch = soft_step(pitch, pitch_vel, MAX_PITCH_STEP)

    yaw = clamp_around_center(yaw, yaw_center, MAX_YAW_DEVIATION)
    pitch = clamp_around_center(pitch, pitch_center, MAX_PITCH_DEVIATION)

    # Debug overlays
    if DRAW_DEBUG:
        if pitch_poly is not None:
            cv2.polylines(persp, [pitch_poly], True, (255, 255, 0), 2)

        for (x1, y1, x2, y2, in_pitch) in player_boxes:
            color = (0, 255, 0) if in_pitch else (0, 0, 255)
            cv2.rectangle(persp, (x1, y1), (x2, y2), color, 2)

        if ball_pos is not None:
            col = (0, 255, 255) if used_ball else (0, 140, 255)
            cv2.circle(persp, ball_pos, 8, col, 2)

        cv2.circle(persp, (int(sm_tx), int(sm_ty)), 5, (255, 255, 255), 2)

        cv2.putText(persp, f"Frame {idx}  yaw={yaw:.1f} pitch={pitch:.1f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(persp, f"Mode: {mode}  ball_miss={frames_since_ball}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    out.write(persp)
    idx += 1

    if idx % 100 == 0:
        elapsed = time.time() - t0
        print(f"Processed {idx}/{n_frames} frames ({idx/elapsed:.2f} fps)")

cap.release()
out.release()

elapsed = time.time() - t0
print("\n--- AutoPan v5 complete ---")
print(f"Frames: {idx}")
print(f"Time: {elapsed:.2f}s  Effective FPS: {idx/elapsed:.2f}")
print(f"Output: {OUTPUT_PATH}")
