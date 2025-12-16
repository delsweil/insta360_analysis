import os
import cv2
import time
import math
from ultralytics import YOLO
from py360convert import e2p

# ------------- CONFIG -------------

INPUT_PATH = "/Users/davidelsweiler/Desktop/test_run.mp4"          # 360° equirectangular video
OUTPUT_PATH = "autopan_output.mp4"

# Virtual camera field of view
FOV_DEG = 90

# Starting orientation (tweak these until the pitch is nicely framed)
YAW_DEG_INITIAL = 0.0     # horizontal angle in degrees
PITCH_DEG_INITIAL = 0.0   # vertical angle in degrees

# Perspective output resolution
OUT_WIDTH = 1280
OUT_HEIGHT = 720

# YOLO settings
PLAYER_MODEL_PATH = "models/yolo11n.pt"   # generic COCO model (person)
BALL_MODEL_PATH = "models/ball.pt"     # our trained ball model

IMG_SIZE_PLAYERS = 960
IMG_SIZE_BALL = 640
CONF_PLAYERS = 0.35
CONF_BALL = 0.25    # can tune later

# AutoPan control parameters
YAW_GAIN = 0.7          # how strongly we react to horizontal offset
PITCH_GAIN = 0.4        # how strongly we react to vertical offset
MAX_YAW_STEP = 4.0      # max yaw change per frame (deg)
MAX_PITCH_STEP = 3.0    # max pitch change per frame (deg)
PITCH_MIN = -30.0       # clamp pitch to sensible range
PITCH_MAX = 10.0

BALL_MISS_FALLBACK_FRAMES = 10   # after this many misses, use player centroid


# ------------- LOAD MODELS -------------
print("Current working dir:", os.getcwd())
print("Looking for player model at:", PLAYER_MODEL_PATH, "exists:", os.path.exists(PLAYER_MODEL_PATH))
print("Looking for ball model at:",   BALL_MODEL_PATH,   "exists:", os.path.exists(BALL_MODEL_PATH))


print("Loading models...")
player_model = YOLO(PLAYER_MODEL_PATH)
ball_model = YOLO(BALL_MODEL_PATH)
print("Models loaded.")


# ------------- OPEN VIDEO -------------

cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video file: {INPUT_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
in_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
in_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Input: {in_width}x{in_height} @ {fps:.2f} fps, {frame_total} frames")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (OUT_WIDTH, OUT_HEIGHT))


# ------------- AUTOPAN STATE -------------

yaw_deg = YAW_DEG_INITIAL
pitch_deg = PITCH_DEG_INITIAL

frame_idx = 0
last_ball_pos = None      # (x, y) in perspective frame
frames_since_ball = 0

start_time = time.time()


# ------------- HELPER FUNCTIONS -------------

def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def soft_update_angle(angle, delta, max_step):
    """Apply delta but limit the amount of change per frame."""
    delta = clamp(delta, -max_step, max_step)
    return angle + delta


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

    player_boxes = []
    player_centers = []
    names_players = player_model.names

    for box in results_players.boxes:
        cls_id = int(box.cls[0])
        cls_name = names_players[cls_id]
        if cls_name != "person":
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        player_boxes.append((x1, y1, x2, y2))
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
        # ball model should have a single class 'ball'
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        ball_candidates.append((x1, y1, x2, y2, cx, cy))

    # Choose ball detection (if any)
    ball_pos = None
    if ball_candidates:
        # if multiple, pick the largest box (likely the closest ball)
        best = max(ball_candidates, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        x1, y1, x2, y2, cx, cy = best
        ball_pos = (cx, cy)
        last_ball_pos = ball_pos
        frames_since_ball = 0
    else:
        frames_since_ball += 1

    # STEP 4: Decide camera target point
    target_x = OUT_WIDTH // 2
    target_y = OUT_HEIGHT // 2

    if ball_pos is not None:
        # Use detected ball
        target_x, target_y = ball_pos
    elif last_ball_pos is not None and frames_since_ball < BALL_MISS_FALLBACK_FRAMES:
        # For a short time after losing the ball, keep aiming where it last was
        target_x, target_y = last_ball_pos
    elif player_centers:
        # Fall back to player centroid
        sx = sum(p[0] for p in player_centers)
        sy = sum(p[1] for p in player_centers)
        target_x = sx / len(player_centers)
        target_y = sy / len(player_centers)
    # else: keep default center

    # STEP 5: Update yaw/pitch based on target point (servo control)
    # Normalize target position to [0,1]
    tx_norm = target_x / OUT_WIDTH
    ty_norm = target_y / OUT_HEIGHT

    # Compute desired adjustments
    # Positive tx_norm>0.5 means ball on right → yaw should increase
    delta_yaw = YAW_GAIN * (tx_norm - 0.5) * FOV_DEG
    delta_pitch = PITCH_GAIN * (ty_norm - 0.5) * FOV_DEG * -1.0
    # (invert pitch: ball lower on screen -> look down (negative))

    yaw_deg = soft_update_angle(yaw_deg, delta_yaw, MAX_YAW_STEP)
    pitch_deg = soft_update_angle(pitch_deg, delta_pitch, MAX_PITCH_STEP)
    pitch_deg = clamp(pitch_deg, PITCH_MIN, PITCH_MAX)

    # Keep yaw in [0,360) for sanity (not strictly necessary)
    yaw_deg = yaw_deg % 360.0

    # STEP 6: Draw overlays for debugging
    # Players
    for (x1, y1, x2, y2) in player_boxes:
        cv2.rectangle(persp, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Ball
    if ball_pos is not None:
        cv2.circle(persp, ball_pos, 8, (0, 255, 255), 2)
        cv2.putText(
            persp, "BALL",
            (ball_pos[0] + 10, ball_pos[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 255), 2, cv2.LINE_AA
        )

    # Target point (where camera is trying to center)
    cv2.circle(persp, (int(target_x), int(target_y)), 6, (255, 255, 255), 2)

    # Debug text
    cv2.putText(
        persp,
        f"Frame {frame_idx}  Yaw {yaw_deg:.1f}  Pitch {pitch_deg:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (255, 255, 255), 2, cv2.LINE_AA
    )

    if ball_pos is not None:
        cv2.putText(
            persp,
            "Ball detected",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 255), 2, cv2.LINE_AA
        )
    elif last_ball_pos is not None and frames_since_ball < BALL_MISS_FALLBACK_FRAMES:
        cv2.putText(
            persp,
            "Using last ball pos",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (200, 200, 0), 2, cv2.LINE_AA
        )
    elif player_centers:
        cv2.putText(
            persp,
            "Using player centroid",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 200, 0), 2, cv2.LINE_AA
        )
    else:
        cv2.putText(
            persp,
            "No guidance (center)",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 255), 2, cv2.LINE_AA
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
print("\n--- AutoPan complete ---")
print(f"Frames processed: {frame_idx}")
print(f"Total time: {elapsed:.2f} seconds")
print(f"Effective FPS: {eff_fps:.2f}")
print(f"Output: {OUTPUT_PATH}")
