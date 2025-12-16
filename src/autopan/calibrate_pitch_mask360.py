import os
import json
import cv2
import numpy as np

# =========================
# CONFIG
# =========================
VIDEO_PATH = "/Users/davidelsweiler/Desktop/test_run.mp4"  # 360 equirect export
FRAME_INDEX = 0
OUT_MASK_PATH = "calibration/pitch_mask_360.png"
OUT_META_PATH = "calibration/pitch_mask_360.json"

os.makedirs("calibration", exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
ok, frame = cap.read()
cap.release()
if not ok:
    raise RuntimeError(f"Could not read frame {FRAME_INDEX}")

H, W = frame.shape[:2]
points = []

win = "Pitch mask calibration (360 equirect)"
help_text = "Left-click: add | Backspace: undo | S: save | Q/Esc: quit"

def redraw():
    vis = frame.copy()
    if len(points) >= 2:
        cv2.polylines(vis, [np.array(points, np.int32)], False, (0, 255, 255), 2)
    for (x, y) in points:
        cv2.circle(vis, (x, y), 4, (0, 255, 255), -1)
    cv2.putText(vis, help_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imshow(win, vis)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        redraw()

cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(win, on_mouse)
redraw()

while True:
    key = cv2.waitKey(0) & 0xFF

    if key in (27, ord('q')):  # ESC or q
        break

    if key == 8:  # backspace
        if points:
            points.pop()
            redraw()

    if key == ord('s'):
        if len(points) < 3:
            print("[WARN] Need at least 3 points.")
            continue

        mask = np.zeros((H, W), dtype=np.uint8)
        poly = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)

        cv2.imwrite(OUT_MASK_PATH, mask)

        meta = {
            "video": VIDEO_PATH,
            "frame_index": FRAME_INDEX,
            "w": W,
            "h": H,
            "polygon_points": points
        }
        with open(OUT_META_PATH, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[OK] Saved mask: {OUT_MASK_PATH}")
        print(f"[OK] Saved meta: {OUT_META_PATH}")
        break

cv2.destroyAllWindows()
