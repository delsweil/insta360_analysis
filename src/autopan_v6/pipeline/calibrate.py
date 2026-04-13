# pipeline/calibrate.py
"""
Interactive pitch boundary calibration tool.

Replaces calibrate_pitch_360.py and calibrate_pitch_mask360.py with a
single tool that:
  1. Works directly on .insv fisheye frames (no stitching required)
  2. Lets you draw the pitch boundary on the raw fisheye frame
  3. Converts boundary points to yaw angles for use in the pipeline
  4. Saves a JSON config that probe.py and detect.py consume

Usage:
    python -m pipeline.calibrate path/to/game_segment.insv

Controls:
    Left click  : add boundary point
    Right click : remove last point
    S           : save calibration
    R           : reset points
    Q / Esc     : quit

Output JSON:
    {
      "camera_model": "ONE X2",
      "source_file": "...",
      "frame_index": 0,
      "frame_w": 5760,
      "frame_h": 2880,
      "boundary_points_px": [[x, y], ...],
      "pitch_yaw_left_deg": -38.5,
      "pitch_yaw_right_deg": 41.2,
      "pitch_yaw_centre_deg": 1.35,
      "notes": "..."
    }
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .probe import probe_file, VideoMeta as VideoInfo
from .lens_models import LensModel
from .decode import fisheye_pixel_to_yaw, equirect_pixel_to_yaw


# ---------------------------------------------------------------------------
# Frame extraction (single frame, no YOLO needed)
# ---------------------------------------------------------------------------

def extract_frame(path: str, frame_index: int = 0) -> np.ndarray:
    """
    Extract a single frame from a video file using FFmpeg.
    Works on .insv and .mp4.
    """
    cmd = [
        "ffmpeg",
        "-hwaccel", "videotoolbox",
        "-i", path,
        "-vf", f"select=eq(n\\,{frame_index})",
        "-vsync", "0",
        "-vframes", "1",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0 or len(result.stdout) == 0:
        raise RuntimeError(f"Could not extract frame {frame_index} from {path}")

    # We need to know the dimensions to parse the rawvideo output
    # Run ffprobe to get them
    from .probe import probe_file
    info = probe_file(path, role="lrv")
    w, h = info.width, info.height
    expected = w * h * 3
    if len(result.stdout) < expected:
        raise RuntimeError(
            f"Frame extraction incomplete: got {len(result.stdout)} bytes, "
            f"expected {expected} ({w}x{h})"
        )
    frame = np.frombuffer(result.stdout[:expected], dtype=np.uint8).reshape((h, w, 3))
    return frame.copy()


# ---------------------------------------------------------------------------
# Calibration UI state
# ---------------------------------------------------------------------------

class CalibrationUI:
    def __init__(
        self,
        frame: np.ndarray,
        info: VideoInfo,
        display_width: int = 1600,
    ):
        self.orig_frame = frame
        self.info = info
        self.display_width = display_width

        h, w = frame.shape[:2]
        self.src_w = w
        self.src_h = h

        # Scale for display
        if w > display_width:
            self.scale = display_width / float(w)
        else:
            self.scale = 1.0

        dw = int(w * self.scale)
        dh = int(h * self.scale)
        self.display_frame = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
        self.display_w = dw
        self.display_h = dh

        # Points in original pixel coordinates
        self.points: List[Tuple[int, int]] = []
        self.window_name = "Calibrate pitch boundary | S=save  R=reset  Q=quit"

    def _orig_to_display(self, x: int, y: int) -> Tuple[int, int]:
        return int(x * self.scale), int(y * self.scale)

    def _display_to_orig(self, x: int, y: int) -> Tuple[int, int]:
        return int(x / self.scale), int(y / self.scale)

    def _point_to_yaw(self, x: int, y: int) -> Optional[float]:
        """Convert a pixel point to yaw angle."""
        if False:  # yaw from fisheye model disabled, using equirect mapping
            return fisheye_pixel_to_yaw(
                float(x), float(y),
                self.src_w, self.src_h,
                self.info.model,
            )
        else:
            return equirect_pixel_to_yaw(float(x), self.src_w)

    def _draw(self) -> np.ndarray:
        vis = self.display_frame.copy()
        pts_disp = [self._orig_to_display(x, y) for x, y in self.points]

        # Draw points
        for i, (px, py) in enumerate(pts_disp):
            cv2.circle(vis, (px, py), 6, (0, 255, 255), -1)
            cv2.circle(vis, (px, py), 7, (0, 0, 0), 1)

            # Show yaw angle next to point
            ox, oy = self.points[i]
            yaw = self._point_to_yaw(ox, oy)
            if yaw is not None:
                label = f"{yaw:.1f}°"
                cv2.putText(vis, label, (px + 10, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw polygon outline
        if len(pts_disp) >= 2:
            cv2.polylines(vis, [np.array(pts_disp, dtype=np.int32)],
                          isClosed=False, color=(0, 255, 255), thickness=2)

        # Show computed yaw range
        if len(self.points) >= 2:
            yaws = [self._point_to_yaw(x, y) for x, y in self.points]
            yaws = [y for y in yaws if y is not None]
            if yaws:
                yaw_min = min(yaws)
                yaw_max = max(yaws)
                cv2.putText(vis,
                    f"Pitch: {yaw_min:.1f}° to {yaw_max:.1f}°  (range: {yaw_max-yaw_min:.1f}°)",
                    (20, self.display_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Instructions
        instructions = [
            "Left click: add point | Right click: remove last | S: save | R: reset | Q: quit",
            f"Points: {len(self.points)}  |  Model: {'LRV 736x368'}",
        ]
        for i, text in enumerate(instructions):
            cv2.putText(vis, text, (20, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, text, (20, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        return vis

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ox, oy = self._display_to_orig(x, y)
            self.points.append((ox, oy))
            yaw = self._point_to_yaw(ox, oy)
            print(f"  Added ({ox}, {oy})  yaw={yaw:.1f}°" if yaw else f"  Added ({ox}, {oy})  [outside fisheye]")
            self._refresh()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                removed = self.points.pop()
                print(f"  Removed {removed}")
                self._refresh()

    def _refresh(self):
        cv2.imshow(self.window_name, self._draw())

    def run(self) -> Optional[dict]:
        """
        Run the calibration UI.
        Returns calibration dict if saved, None if quit.
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_w, self.display_h)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        self._refresh()

        print("\nCalibration tool open.")
        print("  Click the left and right boundaries of the pitch (and any other key points).")
        print("  At minimum, click the leftmost and rightmost points you want to include.")

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("Quit without saving.")
                cv2.destroyAllWindows()
                return None
            elif key in (ord('r'), ord('R')):
                self.points = []
                print("Reset.")
                self._refresh()
            elif key in (ord('s'), ord('S')):
                if len(self.points) < 2:
                    print("[WARN] Need at least 2 points to define left/right boundaries.")
                    continue
                result = self._build_result()
                cv2.destroyAllWindows()
                return result

    def _build_result(self) -> dict:
        """Build the calibration result dict."""
        yaws = [self._point_to_yaw(x, y) for x, y in self.points]
        yaws = [y for y in yaws if y is not None]

        yaw_left  = min(yaws) if yaws else -45.0
        yaw_right = max(yaws) if yaws else  45.0
        yaw_centre = 0.5 * (yaw_left + yaw_right)

        return {
            "camera_model": "ONE X2",
            "source_file": self.info.path,
            "frame_w": self.src_w,
            "frame_h": self.src_h,
            "boundary_points_px": list(self.points),
            "pitch_yaw_left_deg": round(yaw_left, 2),
            "pitch_yaw_right_deg": round(yaw_right, 2),
            "pitch_yaw_centre_deg": round(yaw_centre, 2),
        }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def calibrate(
    video_path: str,
    output_json: str = "calibration/pitch.json",
    frame_index: int = 30,    # use frame 30 (avoid black first frame)
    model_hint: Optional[str] = None,
    display_width: int = 1600,
) -> Optional[dict]:
    """
    Run interactive calibration on a video file.

    Args:
        video_path  : path to .insv or .mp4
        output_json : where to save the calibration JSON
        frame_index : which frame to use for calibration
        model_hint  : camera model name override
        display_width: max display width in pixels

    Returns:
        Calibration dict if saved, None if cancelled.
    """
    print(f"Probing: {video_path}")
    info = probe_file(video_path, role="lrv")
    print(f"  {info.width}x{info.height} {info.codec} {info.duration_sec/60:.1f}min")

    print(f"\nExtracting frame {frame_index}...")
    try:
        frame = extract_frame(video_path, frame_index)
    except Exception as e:
        print(f"[ERROR] {e}")
        print("Try a different frame_index (e.g. --frame 0)")
        return None

    ui = CalibrationUI(frame, info, display_width=display_width)
    result = ui.run()

    if result is None:
        return None

    # Save
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nCalibration saved: {output_json}")
    print(f"  Pitch yaw range : {result['pitch_yaw_left_deg']}° to {result['pitch_yaw_right_deg']}°")
    print(f"  Pitch centre    : {result['pitch_yaw_centre_deg']}°")

    return result


def load_calibration(path: str) -> Optional[dict]:
    """Load a saved calibration JSON."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate pitch boundary")
    parser.add_argument("video", help="Path to .insv or .mp4 file")
    parser.add_argument("--output", default="calibration/pitch.json",
                        help="Output JSON path")
    parser.add_argument("--frame", type=int, default=30,
                        help="Frame index to use for calibration")
    parser.add_argument("--model", default=None,
                        help="Camera model (e.g. 'ONE X2')")
    args = parser.parse_args()

    calibrate(
        video_path=args.video,
        output_json=args.output,
        frame_index=args.frame,
        model_hint=args.model,
    )
