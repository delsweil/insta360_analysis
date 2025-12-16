import os
import json
import cv2
import numpy as np
from py360convert import e2p

# =========================
# CONFIG – adjust these
# =========================

INPUT_PATH = "/Users/davidelsweiler/Desktop/test_run.mp4"

# Choose which frame to calibrate on
FRAME_INDEX = 0  # set e.g. 0, 100, 300, ...

# The SAME projection settings you use in AutoPan
OUT_W, OUT_H = 1280, 720
FOV_DEG = 90
YAW_DEG = 0.0
PITCH_DEG = 0.0

# Output calibration file
OUT_DIR = "calibration"
OUT_JSON = os.path.join(OUT_DIR, "pitch.json")

# If you want homography points too (for future 2D mapping),
# set this True and click 4 points after finishing polygon.
CAPTURE_HOMOGRAPHY = False

# If CAPTURE_HOMOGRAPHY is True, you will be asked to click 4 points.
# You must ALSO provide their real-world coordinates in meters here,
# in the SAME order you click them (e.g., TL, TR, BR, BL).
#
# IMPORTANT: These do NOT have to be the pitch corners.
# They can be any 4 known field landmarks (e.g., penalty box corners).
WORLD_POINTS_METERS = [
    [0.0,   0.0],
    [16.5,  0.0],
    [16.5, 40.3],
    [0.0,  40.3],
]

# =========================
# UI state
# =========================

pitch_points = []
homo_points = []
mode = "polygon"  # polygon -> homography

def draw_ui(img):
    out = img.copy()

    # draw polygon points/lines
    for p in pitch_points:
        cv2.circle(out, p, 5, (0, 255, 255), -1)
    if len(pitch_points) >= 2:
        cv2.polylines(out, [np.array(pitch_points, dtype=np.int32)], False, (0, 255, 255), 2)

    # draw closed polygon if finished
    if mode != "polygon" and len(pitch_points) >= 3:
        cv2.polylines(out, [np.array(pitch_points, dtype=np.int32)], True, (0, 255, 255), 2)

    # draw homography points
    for p in homo_points:
        cv2.circle(out, p, 6, (255, 0, 255), -1)

    text1 = "Left-click: add point | Backspace: undo | Enter: finish polygon | S: save | Q: quit"
    cv2.putText(out, text1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    if mode == "polygon":
        text2 = f"MODE: Pitch polygon  ({len(pitch_points)} points)"
    else:
        text2 = f"MODE: Homography points ({len(homo_points)}/4)"
    cv2.putText(out, text2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    return out

def mouse_cb(event, x, y, flags, param):
    global pitch_points, homo_points, mode
    if event == cv2.EVENT_LBUTTONDOWN:
        if mode == "polygon":
            pitch_points.append((x, y))
        else:
            if len(homo_points) < 4:
                homo_points.append((x, y))

def load_frame(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index}")
    return frame

# =========================
# Main
# =========================

os.makedirs(OUT_DIR, exist_ok=True)

frame360 = load_frame(INPUT_PATH, FRAME_INDEX)

persp = e2p(frame360, fov_deg=FOV_DEG, u_deg=YAW_DEG, v_deg=PITCH_DEG, out_hw=(OUT_H, OUT_W))

win = "Pitch calibration (click points)"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, OUT_W, OUT_H)
cv2.setMouseCallback(win, mouse_cb)

print("\n--- Calibration instructions ---")
print("1) Click around the playable pitch boundary (polygon).")
print("2) Press ENTER to close/finish polygon.")
if CAPTURE_HOMOGRAPHY:
    print("3) Then click 4 points for homography (any known rectangle on the field).")
    print("   Real-world coords used:", WORLD_POINTS_METERS)
print("Press S to save, Q to quit.\n")

while True:
    ui = draw_ui(persp)
    cv2.imshow(win, ui)
    key = cv2.waitKey(10) & 0xFF

    # Undo
    if key in [8, 127]:  # backspace/delete
        if mode == "polygon" and pitch_points:
            pitch_points.pop()
        elif mode != "polygon" and homo_points:
            homo_points.pop()

    # Finish polygon
    if key == 13:  # Enter
        if mode == "polygon":
            if len(pitch_points) >= 3:
                if CAPTURE_HOMOGRAPHY:
                    mode = "homography"
                    print("Polygon finished. Now click 4 homography points.")
                else:
                    mode = "done"
                    print("Polygon finished (no homography). Press S to save.")
            else:
                print("Need at least 3 points for polygon.")

    # Save
    if key in [ord('s'), ord('S')]:
        if len(pitch_points) < 3:
            print("Cannot save: need at least 3 polygon points.")
            continue

        data = {
            "video": INPUT_PATH,
            "frame_index": FRAME_INDEX,
            "projection": {"fov_deg": FOV_DEG, "yaw_deg": YAW_DEG, "pitch_deg": PITCH_DEG,
                           "out_w": OUT_W, "out_h": OUT_H},
            "pitch_polygon": pitch_points,
        }

        if CAPTURE_HOMOGRAPHY:
            if len(homo_points) != 4:
                print("Homography enabled but you haven't clicked 4 points yet.")
                print("You can still save polygon-only by setting CAPTURE_HOMOGRAPHY=False.")
                continue
            data["homography"] = {
                "image_points": homo_points,
                "world_points_meters": WORLD_POINTS_METERS,
            }

        with open(OUT_JSON, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[OK] Saved calibration to {OUT_JSON}")

    # Quit
    if key in [ord('q'), ord('Q'), 27]:
        break

cv2.destroyAllWindows()
