import cv2
from py360convert import e2p
import os

INPUT_PATH = "/Users/davidelsweiler/Desktop/test_run.mp4"  # your 1-minute clip
OUT_DIR = "dataset_frames"             # folder to save images

FOV_DEG = 90
YAW_DEG = 0      # use the yaw/pitch you liked in your last script
PITCH_DEG = 0

OUT_WIDTH = 1280
OUT_HEIGHT = 720

FRAME_STRIDE = 5   # save every 5th frame (tune as needed)

os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open {INPUT_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {total} frames at {fps:.2f} fps")

frame_idx = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % FRAME_STRIDE == 0:
        persp = e2p(
            frame,
            fov_deg=FOV_DEG,
            u_deg=YAW_DEG,
            v_deg=PITCH_DEG,
            out_hw=(OUT_HEIGHT, OUT_WIDTH)
        )

        out_path = os.path.join(OUT_DIR, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(out_path, persp)
        saved += 1

    frame_idx += 1

cap.release()
print(f"Saved {saved} frames into {OUT_DIR}")
