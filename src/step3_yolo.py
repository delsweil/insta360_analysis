

import cv2
import time
from ultralytics import YOLO
from py360convert import e2p


# ---------------- CONFIG -----------------

INPUT_PATH = "/Users/davidelsweiler/Desktop/test_run.mp4"  # your 1-minute clip
OUTPUT_PATH = "output_yolo_clean.mp4"

# Perspective view settings
FOV_DEG = 90
YAW_DEG = 0
PITCH_DEG = 0

OUT_WIDTH = 1280
OUT_HEIGHT = 720

# YOLO speed/accuracy settings
IMG_SIZE = 960        # YOLO resizing
CONF_THRESH = 0.15    # lower for better ball detection


# ---------------- LOAD YOLO -----------------

model = YOLO("yolo11n.pt")   # fast + decent accuracy
print("Loaded YOLO model.")


# ---------------- OPEN VIDEO -----------------

cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video file: {INPUT_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
in_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
in_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Input video: {in_width}x{in_height}, {fps:.2f} fps")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (OUT_WIDTH, OUT_HEIGHT))


# ---------------- PROCESS LOOP -----------------

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Convert 360° → perspective
    persp = e2p(
        frame,
        fov_deg=FOV_DEG,
        u_deg=YAW_DEG,
        v_deg=PITCH_DEG,
        out_hw=(OUT_HEIGHT, OUT_WIDTH)
    )

    # 2. YOLO inference
    results = model(
        persp,
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        verbose=False
    )[0]

    names = model.names
    ball_count = 0

    # 3. Draw only players + ball
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]

        # Only keep person + sports ball
        if cls_name not in ("person", "sports ball"):
            continue

        # Box coords
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Set label + color
        if cls_name == "sports ball":
            label = "BALL"
            color = (0, 255, 255)
            ball_count += 1
        else:
            label = "PLAYER"
            color = (0, 255, 0)

        # Draw
        cv2.rectangle(persp, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            persp,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA
        )

    # Overlays
    cv2.putText(
        persp,
        f"Frame {frame_count}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
        cv2.LINE_AA
    )
    cv2.putText(
        persp,
        f"Balls: {ball_count}",
        (30, OUT_HEIGHT - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    out.write(persp)
    frame_count += 1


# ---------------- END & TIMING -----------------

end_time = time.time()
elapsed = end_time - start_time
fps_effective = frame_count / elapsed if elapsed > 0 else 0

cap.release()
out.release()

print("\n--- DONE ---")
print(f"Frames processed: {frame_count}")
print(f"Total time: {elapsed:.2f} seconds")
print(f"Effective FPS: {fps_effective:.2f}")
print(f"Output written to: {OUTPUT_PATH}")
