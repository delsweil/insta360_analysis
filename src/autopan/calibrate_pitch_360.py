# src/autopan/calibrate_pitch_360.py
import os
import json
import cv2
import numpy as np


# ---- EDIT THESE ----
VIDEO_PATH = "/Users/davidelsweiler/Desktop/test_run.mp4"
OUT_JSON = "calibration/pitch.json"
FRAME_INDEX = 0

# display scaling so clicking is easier on screen
DISPLAY_W = 1600  # set lower if too big for your monitor


points = []
display_img = None
scale = 1.0


def on_mouse(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        # scale back to original equirect coords
        ox = int(round(x / scale))
        oy = int(round(y / scale))
        points.append([ox, oy])
        print(f"Added point: {ox}, {oy}  (total={len(points)})")


def main():
    global display_img, scale

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {in_w}x{in_h}, frames={n}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {FRAME_INDEX}")

    # scale for display
    if in_w > DISPLAY_W:
        scale = DISPLAY_W / float(in_w)
        disp_h = int(round(in_h * scale))
        display_img = cv2.resize(frame, (DISPLAY_W, disp_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        display_img = frame.copy()

    win = "Calibrate pitch in 360 (equirect)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    print("\nInstructions:")
    print("- Left click: add polygon points around the playable pitch boundary (in 360 frame)")
    print("- Backspace: undo last point")
    print("- Enter: finish polygon")
    print("- S: save to calibration/pitch.json")
    print("- Q or ESC: quit without saving\n")

    while True:
        vis = display_img.copy()

        # draw points/lines (scaled for display)
        if points:
            pts_disp = np.array([[int(p[0]*scale), int(p[1]*scale)] for p in points], dtype=np.int32)
            for p in pts_disp:
                cv2.circle(vis, tuple(p), 5, (0, 255, 255), -1)
            cv2.polylines(vis, [pts_disp], isClosed=False, color=(0, 255, 255), thickness=2)

        cv2.putText(vis, f"Points: {len(points)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.imshow(win, vis)

        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord('q')):  # ESC or q
            print("Quit without saving.")
            break
        elif k == 8:  # backspace
            if points:
                points.pop()
                print(f"Undo. points={len(points)}")
        elif k == 13:  # enter
            print("Finish polygon (press S to save).")
        elif k in (ord('s'), ord('S')):
            if len(points) < 3:
                print("Need at least 3 points to save.")
                continue
            os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
            payload = {
                "video": VIDEO_PATH,
                "frame_index": FRAME_INDEX,
                "equirect": {"in_w": in_w, "in_h": in_h},
                "pitch_polygon_360": points
            }
            with open(OUT_JSON, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved: {OUT_JSON}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
