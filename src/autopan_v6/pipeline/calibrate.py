# pipeline/calibrate.py
"""
Interactive per-segment calibration tool.

Works on the VID _10_ (pitch-facing) file.
Extracts the same crop used by detection (rotate 90° CCW, bottom 2/3)
and lets you draw the pitch boundary polygon on it.

Two-step process:
  Step 1: Draw pitch boundary polygon (click around the pitch perimeter)
  Step 2: Confirm yaw boundaries (leftmost and rightmost polygon points
          define the yaw range automatically)

Controls:
  Left click  : add polygon point
  Right click : remove last point
  S           : save calibration
  R           : reset all points
  Q / Esc     : quit without saving

Output JSON saved to calibration/<segment_key>.json
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .decode import extract_pitch_crop, fisheye_cx_to_yaw
from .probe import probe_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def segment_key_from_path(path: str) -> str:
    name = Path(path).name
    m = re.match(
        r'^(?:VID|LRV)_(\d{8})_(\d{6})_\d{2}_(\d{3})\.insv$',
        name, re.IGNORECASE
    )
    if m:
        return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
    # For .mp4 test clips, use stem
    return Path(path).stem


def calib_path_for_segment(segment_key: str, calib_dir: str = "calibration") -> str:
    return os.path.join(calib_dir, f"{segment_key}.json")


def load_calibration(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_calibration_for_segment(
    segment_key: str,
    calib_dir: str = "calibration",
) -> Optional[dict]:
    return load_calibration(calib_path_for_segment(segment_key, calib_dir))


def find_best_calibration(
    segment_key: str,
    calib_dir: str = "calibration",
) -> Optional[dict]:
    """Try exact segment match first, then generic pitch.json."""
    calib = load_calibration_for_segment(segment_key, calib_dir)
    if calib:
        return calib
    return load_calibration(os.path.join(calib_dir, "pitch.json"))


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_calibration_frame(path: str, frame_index: int) -> np.ndarray:
    """
    Extract a single frame and apply the same crop as detection.
    Works on VID _10_ files (2880x2880) or LRV files (736x368).
    """
    info = probe_file(path, role='lrv')
    w, h = info.width, info.height
    expected = w * h * 3

    cmd = [
        "ffmpeg",
        "-i", path,
        "-vf", f"select=eq(n\\,{frame_index})",
        "-vsync", "0",
        "-vframes", "1",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if len(result.stdout) < expected:
        raise RuntimeError(
            f"Could not extract frame {frame_index} from {os.path.basename(path)}.\n"
            f"Got {len(result.stdout)} bytes, expected {expected}.\n"
            f"Try a different --frame value."
        )
    raw = np.frombuffer(result.stdout[:expected], dtype=np.uint8).reshape((h, w, 3)).copy()

    # Apply the same crop as detection
    if w == h:
        # Square frame = VID file (2880x2880) — use VID crop
        return extract_pitch_crop(raw)
    else:
        # Non-square = LRV or MP4 — just return as-is for now
        return raw


# ---------------------------------------------------------------------------
# Calibration UI
# ---------------------------------------------------------------------------

class CalibrationUI:

    def __init__(
        self,
        crop: np.ndarray,
        segment_key: str,
        source_file: str,
        frame_index: int,
        display_w: int = 1200,
    ):
        self.crop = crop           # the pitch crop (same view YOLO sees)
        self.segment_key = segment_key
        self.source_file = source_file
        self.frame_index = frame_index

        self.crop_h, self.crop_w = crop.shape[:2]

        # Scale for display
        self.scale = display_w / self.crop_w
        self.disp_w = display_w
        self.disp_h = int(self.crop_h * self.scale)

        self.points: List[Tuple[int, int]] = []  # in crop pixel coords

        self.window = (
            f"Calibrate {segment_key}  |  "
            f"Left click: add point  |  Right click: remove  |  "
            f"S: save  |  R: reset  |  Q: quit"
        )

    def _crop_to_disp(self, cx: int, cy: int) -> Tuple[int, int]:
        return int(cx * self.scale), int(cy * self.scale)

    def _disp_to_crop(self, dx: int, dy: int) -> Tuple[int, int]:
        return int(dx / self.scale), int(dy / self.scale)

    def _draw(self) -> np.ndarray:
        vis = cv2.resize(self.crop, (self.disp_w, self.disp_h))

        if len(self.points) >= 2:
            pts_disp = np.array(
                [self._crop_to_disp(x, y) for x, y in self.points],
                dtype=np.int32
            )
            cv2.polylines(vis, [pts_disp], isClosed=True,
                          color=(0, 255, 255), thickness=2)

        for (cx, cy) in self.points:
            dx, dy = self._crop_to_disp(cx, cy)
            cv2.circle(vis, (dx, dy), 5, (0, 255, 255), -1)
            cv2.circle(vis, (dx, dy), 6, (0, 0, 0), 1)

        # Show computed yaw range if we have points
        if len(self.points) >= 2:
            xs = [p[0] for p in self.points]
            # Approximate yaw mapping: left edge=yaw_left, right edge=yaw_right
            # We use a placeholder ±45° here — actual yaw saved uses calibrated values
            x_min, x_max = min(xs), max(xs)
            txt = (f"Points: {len(self.points)}  |  "
                   f"Left x={x_min}  Right x={x_max}  "
                   f"Width={x_max - x_min}px")
            cv2.putText(vis, txt, (10, self.disp_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, txt, (10, self.disp_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 1, cv2.LINE_AA)

        instructions = [
            f"Draw pitch boundary polygon | Points: {len(self.points)} | S=save  R=reset  Q=quit",
        ]
        for i, txt in enumerate(instructions):
            cv2.putText(vis, txt, (10, 28 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, txt, (10, 28 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 1, cv2.LINE_AA)

        return vis

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cx, cy = self._disp_to_crop(x, y)
            self.points.append((cx, cy))
            print(f"  Added ({cx}, {cy})  [{len(self.points)} points]")
            self._refresh()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                removed = self.points.pop()
                print(f"  Removed {removed}  [{len(self.points)} points]")
                self._refresh()

    def _refresh(self):
        cv2.imshow(self.window, self._draw())

    def run(self) -> Optional[dict]:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, self.disp_w, self.disp_h)
        cv2.setMouseCallback(self.window, self.on_mouse)
        self._refresh()

        print("\nCalibration tool open.")
        print("  Click around the pitch boundary (touchlines + goal lines).")
        print("  Include the far touchline, both goal lines, and near touchline.")
        print("  The leftmost and rightmost points define the yaw range.")
        print("  Press S to save when done.\n")

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                cv2.destroyAllWindows()
                print("Quit without saving.")
                return None
            elif key in (ord('r'), ord('R')):
                self.points = []
                print("Reset.")
                self._refresh()
            elif key in (ord('s'), ord('S')):
                if len(self.points) < 3:
                    print("  Need at least 3 points to define the pitch boundary.")
                    continue
                result = self._build_result()
                cv2.destroyAllWindows()
                return result

    def _build_result(self) -> dict:
        xs = [p[0] for p in self.points]
        x_left  = min(xs)
        x_right = max(xs)
        x_centre = (x_left + x_right) / 2.0

        # Yaw mapping: linear from pixel x to yaw degrees
        # The full crop width maps to the full fisheye horizontal extent
        # We use ±90° as the fisheye half-angle (200° total FOV / 2 = 100°,
        # but effective horizontal range after crop is ~180°)
        # These are stored as pixel fractions — detect.py uses
        # fisheye_cx_to_yaw() with the calibrated yaw_left/right to convert.
        # We store pixel coords here and let the user set yaw via probe output.
        half_fov = 90.0
        yaw_left   = ((x_left   / self.crop_w) - 0.5) * 2 * half_fov
        yaw_right  = ((x_right  / self.crop_w) - 0.5) * 2 * half_fov
        yaw_centre = ((x_centre / self.crop_w) - 0.5) * 2 * half_fov

        return {
            "segment_key": self.segment_key,
            "source_file": os.path.basename(self.source_file),
            "frame_index": self.frame_index,
            "crop_w": self.crop_w,
            "crop_h": self.crop_h,
            "pitch_polygon": [[x, y] for x, y in self.points],
            "pitch_yaw_left_deg":   round(yaw_left,   2),
            "pitch_yaw_right_deg":  round(yaw_right,  2),
            "pitch_yaw_centre_deg": round(yaw_centre, 2),
        }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def calibrate(
    video_path: str,
    output_dir: str = "calibration",
    frame_index: int = 3600,
    display_w: int = 1200,
) -> Optional[dict]:
    """
    Run interactive calibration on a VID _10_ file.

    Args:
        video_path : path to VID _10_ .insv file
        output_dir : directory to save calibration JSON
        frame_index: frame to use (default 3600 = 2min at 30fps)
        display_w  : UI window width in pixels
    """
    key = segment_key_from_path(video_path)
    print(f"Calibrating: {key}")
    print(f"Extracting frame {frame_index} from {os.path.basename(video_path)}...")

    try:
        crop = extract_calibration_frame(video_path, frame_index)
    except Exception as e:
        print(f"[ERROR] {e}")
        return None

    print(f"Crop size: {crop.shape[1]}x{crop.shape[0]}")

    ui = CalibrationUI(
        crop=crop,
        segment_key=key,
        source_file=video_path,
        frame_index=frame_index,
        display_w=display_w,
    )

    result = ui.run()
    if result is None:
        return None

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{key}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nCalibration saved: {out_path}")
    print(f"  Polygon points : {len(result['pitch_polygon'])}")
    print(f"  Yaw range      : {result['pitch_yaw_left_deg']}° to "
          f"{result['pitch_yaw_right_deg']}°")
    print(f"  Yaw centre     : {result['pitch_yaw_centre_deg']}°")

    return result
