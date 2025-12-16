import cv2
import time
import math
import os
from ultralytics import YOLO
from py360convert import e2p

# ------------- CONFIG -------------

INPUT_PATH = "/Users/davidelsweiler/Desktop/test_run.mp4"          # 360° equirectangular video
OUTPUT_PATH = "data/processed/autopan_short_test_v2.mp4"

# Virtual camera field of view
FOV_DEG = 90

# Starting orientation (roughly center of pitch)
YAW_DEG_INITIAL = 0.0
PITCH_DEG_INITIAL = 0.0

# Perspective output resolution
OUT_WIDTH = 1280
OUT_HEIGHT = 720

# YOLO settings (local files!)
PLAYER_MODEL_PATH = "models/yolo11n.pt"   # generic COCO model (person)
BALL_MODEL_PATH = "models/ball.pt"     # our trained ball model


IMG_SIZE_PLAYERS = 960
IMG_SIZE_BALL    = 640
CONF_PLAYERS     = 0.35
CONF_BALL        = 0.20   # slightly lower to get more hits; we add our own filtering

# AutoPan control parameters (smoother than v1)
YAW_GAIN        = 0.4
PITCH_GAIN      = 0.3
MAX_YAW_STEP    = 2.5     # deg per frame
MAX_PITCH_STEP  = 2.0
PITCH_MIN       = -30.0
PITCH_MAX       = 10.0

# Smoothing of target point (0=no smoothing, 1=frozen)
TARGET_SMOOTH_ALPHA = 0.2  # EMA weight for new target

# Ball trust / fallback behaviour
BALL_CONFIRM_FRAMES        = 3     # need this many consecutive hits to "trust" ball
BALL_MISS_FALLBACK_FRAMES  = 10    # short fallback with last ball pos
BALL_LONG_MISS_FRAMES      = 90    # after this many frames, drift back to home

# Only use players in this vertical band for centroid (to avoid coaches in stands)
PITCH_BAND_Y_MIN_FRAC = 0.20      # as fraction of OUT_HEIGHT
PITCH_BAND_Y_MAX_FRAC = 0.95


# ------------- UTIL -------------

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def soft_update_angle(angle, delta, max_step):
    """Apply delta but limit the amount of change per frame."""
    delta = clamp(delta, -max_step, max_step)
    return angle + delta


# ------------- SANITY CHECKS -------------

print("CWD:", os.getcwd())
print("Player model path:", PLAYER_MODEL_PATH, "exists:", os.path.exists(PLAYER_MODEL_PATH))
print("Ball model path:",   BALL_MODEL_PATH,   "exists:", os.path.exists(BALL_MODEL_PATH))

if not os.path.exists(PLAYER_MODEL_PATH):
    raise FileNotFoundError(f"Player model not found at {PLAYER_MODEL_PATH}")
if not os.path.exists(BALL_MODEL_PATH):
    raise FileNotFoundError(f"Ball model not found at {BALL_MODEL_PATH}")


# ------------- LOAD MODELS -------------

print("Loading models...")
player_model = YOLO(PLAYER_MODEL_PATH)
ball_model   = YOLO(BALL_MODEL_PATH)
print("Models loaded.")


# ------------- OPEN VIDEO -------------

cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video file: {INPUT_PATH}")

fps        = cap.get(cv2.CAP_PROP_FPS)
in_width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
in_height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Input: {in_width}x{in_height} @ {fps:.2f} fps, {frame_total} frames")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (OUT_WIDTH, OUT_HEIGHT))


# ------------- AUTOPAN STATE -------------

yaw_deg   = YAW_DEG_INITIAL
pitch_deg = PITCH_DEG_INITIAL

frame_idx = 0
last_ball_pos = None         # (x, y)
frames_since_ball = 0

ball_confirm_count = 0       # for initial ball confirmation
ball_trusted = False

# smoothed target point
smoothed_tx = OUT_WIDTH  / 2
smoothed_ty = OUT_HEIGHT / 2

start_time = time.time()


# ------------- MAIN LOOP -------------

while True:
    ret, frame_360 = cap.read()
    if not ret:
        break

    # STEP 1: Project 360° frame to current virtual camera view
    persp = e2p(
        frame_360,
        fov_deg=FOV_DEG,
        u_deg=yaw_deg,
        v_deg=pitch_deg,
        out_hw=(OUT_HEIGHT, OUT_WIDTH)
    )

    # STEP 2: Detect players
    results_players = player_model(
        persp,
        imgsz=IMG_SIZE_PLAYERS,
        conf=CONF_PLAYERS,
        verbose=False
    )[0]

    player_boxes   = []
    player_centers = []
    names_players  = player_model.names

    pitch_y_min = int(PITCH_BAND_Y_MIN_FRAC * OUT_HEIGHT)
    pitch_y_max = int(PITCH_BAND_Y_MAX_FRAC * OUT_HEIGHT)

    for box in results_players.boxes:
        cls_id = int(box.cls[0])
        cls_name = names_players[cls_id]
        if cls_name != "person":
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        player_boxes.append((x1, y1, x2, y2))

        # only use players in central vertical band for centroid
        if pitch_y_min <= cy <= pitch_y_max:
            player_centers.append((cx, cy))

    # STEP 3: Detect ball
    results_ball = ball_model(
        persp,
        imgsz=IMG_SIZE_BALL,
        conf=CONF_BALL,
        verbose=False
    )[0]

    ball_candidates = []
    for box in results_ball.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        area = (x2 - x1) * (y2 - y1)

        # Simple sanity filter: discard absurdly large "balls"
        if area > (OUT_WIDTH * OUT_HEIGHT * 0.02):
            continue

        ball_candidates.append((x1, y1, x2, y2, cx, cy, conf))

    ball_pos = None
    ball_used_this_frame = False

    if ball_candidates:
        # choose highest-confidence, then largest
        ball_candidates.sort(key=lambda b: (b[6], (b[2]-b[0])*(b[3]-b[1])), reverse=True)
        x1, y1, x2, y2, cx, cy, conf = ball_candidates[0]
        ball_pos = (cx, cy)

        # Confirmation logic: require several consecutive frames before trusting ball
        ball_confirm_count += 1
        if ball_confirm_count >= BALL_CONFIRM_FRAMES:
            ball_trusted = True

        last_ball_pos = ball_pos
        frames_since_ball = 0
        ball_used_this_frame = ball_trusted
    else:
        frames_since_ball += 1
        ball_confirm_count = 0  # lost ball; must reconfirm later

    # STEP 4: Decide target point
    raw_target_x = OUT_WIDTH // 2
    raw_target_y = OUT_HEIGHT // 2
    mode_text = "Center"

    if ball_used_this_frame:
        raw_target_x, raw_target_y = ball_pos
        mode_text = "Ball"
    elif last_ball_pos is not None and frames_since_ball < BALL_MISS_FALLBACK_FRAMES:
        raw_target_x, raw_target_y = last_ball_pos
        mode_text = "Last ball"
    elif player_centers and frames_since_ball < BALL_LONG_MISS_FRAMES:
        # Use centroid of on-pitch players
        sx = sum(p[0] for p in player_centers)
        sy = sum(p[1] for p in player_centers)
        raw_target_x = sx / len(player_centers)
        raw_target_y = sy / len(player_centers)
        mode_text = "Player centroid"
    else:
        # Long time without ball: drift back towards home orientation
        mode_text = "Home drift"
        # we don't change target_x/y; instead we gently pull yaw to initial
        yaw_delta_home = (YAW_DEG_INITIAL - yaw_deg) * 0.02
        yaw_deg = soft_update_angle(yaw_deg, yaw_delta_home, MAX_YAW_STEP)

    # STEP 5: Smooth target point (EMA)
    smoothed_tx = (1 - TARGET_SMOOTH_ALPHA) * smoothed_tx + TARGET_SMOOTH_ALPHA * raw_target_x
    smoothed_ty = (1 - TARGET_SMOOTH_ALPHA) * smoothed_ty + TARGET_SMOOTH_ALPHA * raw_target_y

    # STEP 6: Update yaw/pitch based on smoothed target
    tx_norm = smoothed_tx / OUT_WIDTH
    ty_norm = smoothed_ty / OUT_HEIGHT

    delta_yaw   = YAW_GAIN   * (tx_norm - 0.5) * FOV_DEG
    delta_pitch = PITCH_GAIN * (ty_norm - 0.5) * FOV_DEG * -1.0

    yaw_deg   = soft_update_angle(yaw_deg,   delta_yaw,   MAX_YAW_STEP)
    pitch_deg = soft_update_angle(pitch_deg, delta_pitch, MAX_PITCH_STEP)
    pitch_deg = clamp(pitch_deg, PITCH_MIN, PITCH_MAX)
    yaw_deg   = yaw_deg % 360.0

    # STEP 7: Draw overlays for debugging
    # players
    for (x1, y1, x2, y2) in player_boxes:
        cv2.rectangle(persp, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ball
    if ball_pos is not None:
        color = (0, 255, 255) if ball_used_this_frame else (0, 140, 255)
        cv2.circle(persp, ball_pos, 8, color, 2)
        cv2.putText(
            persp, "BALL",
            (ball_pos[0] + 10, ball_pos[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            color, 2, cv2.LINE_AA
        )

    # target
    cv2.circle(persp, (int(smoothed_tx), int(smoothed_ty)), 5, (255, 255, 255), 2)

    cv2.putText(
        persp,
        f"Frame {frame_idx}  Yaw {yaw_deg:.1f}  Pitch {pitch_deg:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 255, 255), 2, cv2.LINE_AA
    )

    cv2.putText(
        persp,
        f"Mode: {mode_text}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (0, 255, 255), 2, cv2.LINE_AA
    )

    out.write(persp)
    frame_idx += 1

    if frame_idx % 100 == 0:
        elapsed = time.time() - start_time
        eff_fps = frame_idx / elapsed if elapsed > 0 else 0
        print(f"Processed {frame_idx}/{frame_total} frames ({eff_fps:.2f} fps)")


# ------------- CLEANUP -------------

cap.release()
out.release()

elapsed = time.time() - start_time
eff_fps = frame_idx / elapsed if elapsed > 0 else 0
print("\n--- AutoPan v2 complete ---")
print(f"Frames processed: {frame_idx}")
print(f"Total time: {elapsed:.2f} seconds")
print(f"Effective FPS: {eff_fps:.2f}")
print(f"Output: {OUTPUT_PATH}")
