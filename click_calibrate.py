#!/usr/bin/env python3
"""
Click-to-calibrate tool for Steady Cam footage.
Click the pitch boundary points on a frame, saves mask as JSON.

Usage:
    python3 click_calibrate.py --video path/to/video.mp4 --output calib/mygame.json
    python3 click_calibrate.py --video path/to/video.mp4  # saves next to video
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


INSTRUCTIONS = [
    "Click the FAR touchline (left to right)",
    "Then click the NEAR touchline (left to right)",
    "Press ENTER to confirm, R to reset, Q to quit",
]

COLORS = {
    'far':  (255, 100,   0),   # orange
    'near': (  0, 200, 255),   # cyan
    'fill': ( 50, 180,  50),   # green
}


def extract_frame(video_path: str, t: float = 10.0) -> np.ndarray:
    cmd = [
        'ffmpeg', '-ss', str(t), '-i', video_path,
        '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-loglevel', 'error', 'pipe:1'
    ]
    proc = subprocess.run(cmd, capture_output=True)
    arr = np.frombuffer(proc.stdout, dtype=np.uint8).copy()
    # Try landscape 2560x1440 first, then portrait
    if arr.size == 2560 * 1440 * 3:
        return arr.reshape((1440, 2560, 3))
    elif arr.size == 1440 * 2560 * 3:
        img = arr.reshape((2560, 1440, 3))
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:
        raise ValueError(f"Unexpected frame size: {arr.size}")


def build_mask(far_pts, near_pts, h, w):
    """Build a binary mask from far and near touchline points."""
    if len(far_pts) < 2 or len(near_pts) < 2:
        return None
    poly = np.array(far_pts + near_pts[::-1], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def draw_state(base_img, far_pts, near_pts, hover=None):
    img = base_img.copy()
    h, w = img.shape[:2]

    # Draw instructions
    for i, line in enumerate(INSTRUCTIONS):
        cv2.putText(img, line, (20, 35 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, line, (20, 35 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    # Draw mask overlay
    mask = build_mask(far_pts, near_pts, h, w)
    if mask is not None:
        overlay = img.copy()
        overlay[mask > 0] = (overlay[mask > 0] * 0.5 +
                              np.array(COLORS['fill']) * 0.5).astype(np.uint8)
        img = overlay

    # Draw far points
    for i, pt in enumerate(far_pts):
        cv2.circle(img, pt, 8, COLORS['far'], -1)
        cv2.circle(img, pt, 8, (255, 255, 255), 2)
        if i > 0:
            cv2.line(img, far_pts[i-1], pt, COLORS['far'], 2)

    # Draw near points
    for i, pt in enumerate(near_pts):
        cv2.circle(img, pt, 8, COLORS['near'], -1)
        cv2.circle(img, pt, 8, (255, 255, 255), 2)
        if i > 0:
            cv2.line(img, near_pts[i-1], pt, COLORS['near'], 2)

    # Labels
    if far_pts:
        cv2.putText(img, f"FAR ({len(far_pts)} pts)", far_pts[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['far'], 2)
    if near_pts:
        cv2.putText(img, f"NEAR ({len(near_pts)} pts)", near_pts[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['near'], 2)

    # Hover crosshair
    if hover:
        cv2.line(img, (hover[0]-20, hover[1]), (hover[0]+20, hover[1]),
                 (255, 255, 0), 1)
        cv2.line(img, (hover[0], hover[1]-20), (hover[0], hover[1]+20),
                 (255, 255, 0), 1)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--time', type=float, default=10.0,
                        help='Timestamp to extract calibration frame (default: 10s)')
    args = parser.parse_args()

    video_path = args.video
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(video_path).with_suffix('.calib.json')

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Extracting frame at t={args.time}s...")
    frame = extract_frame(video_path, args.time)
    h, w = frame.shape[:2]
    print(f"Frame: {w}x{h}")

    # Scale for display if too large
    scale = min(1.0, 1400 / w, 800 / h)
    disp_w = int(w * scale)
    disp_h = int(h * scale)

    far_pts = []
    near_pts = []
    phase = 'far'  # 'far' then 'near'
    hover = None

    def on_mouse(event, x, y, flags, param):
        nonlocal hover, phase
        # Scale back to original coords
        ox, oy = int(x / scale), int(y / scale)
        hover = (ox, oy)

        if event == cv2.EVENT_LBUTTONDOWN:
            if phase == 'far':
                far_pts.append((ox, oy))
                print(f"  Far point {len(far_pts)}: ({ox}, {oy})")
                if len(far_pts) >= 2:
                    print("  → Now click NEAR touchline points")
                    phase = 'near'  # auto-advance after 2+ far points on right-click
            elif phase == 'near':
                near_pts.append((ox, oy))
                print(f"  Near point {len(near_pts)}: ({ox}, {oy})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if phase == 'far' and len(far_pts) >= 2:
                phase = 'near'
                print("  → Switching to NEAR touchline")
            elif phase == 'near' and near_pts:
                near_pts.pop()
            elif phase == 'far' and far_pts:
                far_pts.pop()

    cv2.namedWindow('Calibrate', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Calibrate', disp_w, disp_h)
    cv2.setMouseCallback('Calibrate', on_mouse)

    print("\nInstructions:")
    print("  LEFT CLICK  — add point (far touchline first, then near)")
    print("  RIGHT CLICK — advance to near touchline / undo last point")
    print("  ENTER       — save and exit")
    print("  R           — reset all points")
    print("  Q           — quit without saving\n")

    while True:
        disp = draw_state(frame, far_pts, near_pts, hover)
        disp_small = cv2.resize(disp, (disp_w, disp_h))
        cv2.imshow('Calibrate', disp_small)

        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            print("Quit without saving.")
            break

        elif key == ord('r'):
            far_pts.clear()
            near_pts.clear()
            phase = 'far'
            print("Reset.")

        elif key in (13, 10):  # Enter
            if len(far_pts) < 2 or len(near_pts) < 2:
                print("Need at least 2 far and 2 near points. Keep clicking.")
                continue

            calib = {
                'video': str(video_path),
                'frame_width': w,
                'frame_height': h,
                'far_pts': far_pts,
                'near_pts': near_pts,
            }
            with open(out_path, 'w') as f:
                json.dump(calib, f, indent=2)
            print(f"\nSaved calibration to {out_path}")
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
