# pipeline/decode.py
"""
Frame extraction for Insta360 X2 VID files.

Confirmed X2 geometry (from testing):
  VID_..._10_...insv : pitch-facing lens, 2880x2880, H.264
  VID_..._00_...insv : rear-facing lens (car park side) — not used for detection

Each 2880x2880 frame is stored rotated 90° CCW.
After rotating 90° CCW:
  - Full frame is 2880 wide x 2880 tall
  - Taking bottom 2/3 (frame[h//3:]) gives the pitch fisheye view
    with full pitch visible including both goals and far touchline

Detection pipeline per sampled frame:
  1. Read raw 2880x2880 frame via FFmpeg pipe
  2. Rotate 90° CCW
  3. Crop bottom 2/3 -> pitch fisheye (~2880x1920)
  4. YOLO runs on this directly at imgsz=1280
  5. Detections filtered by pitch polygon mask
  6. Surviving detections mapped to yaw angles
"""

from __future__ import annotations

import subprocess
from typing import Generator, Optional, Tuple

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# FFmpeg frame reader
# ---------------------------------------------------------------------------

class FrameReader:
    """
    Reads frames from a video file via FFmpeg rawvideo pipe.
    Yields (frame_index, frame_bgr) for every Nth frame.
    """

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        sample_every_n: int = 8,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        use_hwaccel: bool = True,
    ):
        self.path = path
        self.width = width
        self.height = height
        self.sample_every_n = sample_every_n
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.use_hwaccel = use_hwaccel
        self.frame_bytes = width * height * 3
        self._proc = None

    def _build_cmd(self) -> list:
        cmd = ["ffmpeg"]
        if self.use_hwaccel:
            cmd += ["-hwaccel", "videotoolbox"]
        if self.start_sec > 0:
            cmd += ["-ss", str(self.start_sec)]
        cmd += ["-i", self.path]
        if self.end_sec is not None:
            cmd += ["-t", str(self.end_sec - self.start_sec)]
        filters = [
            f"select=not(mod(n\\,{self.sample_every_n}))",
            "setpts=N/FRAME_RATE/TB",
            "format=bgr24",
        ]
        cmd += [
            "-vf", ",".join(filters),
            "-vsync", "0",
            "-an",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:1",
        ]
        return cmd

    def __enter__(self):
        cmd = self._build_cmd()
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.frame_bytes * 2,
        )
        return self

    def __exit__(self, *args):
        if self._proc:
            self._proc.stdout.close()
            self._proc.terminate()
            self._proc.wait()
            self._proc = None

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yields (original_frame_index, frame_bgr)."""
        sample_idx = 0
        while True:
            raw = self._proc.stdout.read(self.frame_bytes)
            if len(raw) < self.frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            ).copy()
            orig_idx = sample_idx * self.sample_every_n
            yield orig_idx, frame
            sample_idx += 1


# ---------------------------------------------------------------------------
# X2 frame extraction
# ---------------------------------------------------------------------------

def extract_pitch_crop(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Extract the pitch-facing fisheye crop from an X2 VID _10_ frame.

    Steps:
      1. Rotate 90° CCW (frames are stored sideways)
      2. Take bottom 2/3 (removes sky, shows full pitch with both goals)

    Args:
        frame_bgr: raw 2880x2880 frame from VID _10_ file

    Returns:
        ~2880x1920 fisheye crop showing the full pitch
    """
    rotated = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h = rotated.shape[0]
    return rotated[h // 3:, :]


# ---------------------------------------------------------------------------
# Coordinate mapping
# ---------------------------------------------------------------------------

def fisheye_cx_to_yaw(
    cx: float,
    crop_w: int,
    yaw_left: float,
    yaw_right: float,
) -> float:
    """
    Convert an x pixel coordinate in the fisheye crop to a yaw angle.

    Uses the calibrated left/right yaw boundaries to define the mapping.
    Linear interpolation: x=0 -> yaw_left, x=crop_w -> yaw_right.

    Args:
        cx       : x pixel coordinate (centre of detection box)
        crop_w   : width of the fisheye crop in pixels
        yaw_left : yaw angle at left edge of crop (degrees)
        yaw_right: yaw angle at right edge of crop (degrees)

    Returns:
        yaw angle in degrees
    """
    norm = cx / crop_w   # 0.0 to 1.0
    return yaw_left + norm * (yaw_right - yaw_left)


# ---------------------------------------------------------------------------
# Pitch polygon mask
# ---------------------------------------------------------------------------

def build_pitch_mask(
    polygon_points: list,
    crop_h: int,
    crop_w: int,
) -> np.ndarray:
    """
    Build a binary mask from pitch boundary polygon points.

    Args:
        polygon_points: list of [x, y] points in crop pixel coordinates
        crop_h, crop_w: dimensions of the fisheye crop

    Returns:
        uint8 mask, 255 inside pitch, 0 outside
    """
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    if len(polygon_points) >= 3:
        pts = np.array(polygon_points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def point_in_mask(mask: np.ndarray, x: float, y: float) -> bool:
    """Check if a point falls inside the pitch mask."""
    if mask is None:
        return True
    h, w = mask.shape[:2]
    xi = int(np.clip(round(x), 0, w - 1))
    yi = int(np.clip(round(y), 0, h - 1))
    return bool(mask[yi, xi])


# ---------------------------------------------------------------------------
# LRV helpers (kept for probe compatibility)
# ---------------------------------------------------------------------------

def extract_pitch_lens(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Extract pitch lens from LRV frame (736x368).
    Kept for backwards compatibility with calibration tool.
    LRV: rotate 90° CCW, take top half.
    """
    rotated = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h = rotated.shape[0]
    return rotated[:h // 2, :]


def equirect_pixel_to_yaw(px: float, frame_w: int) -> float:
    """x pixel in equirect -> yaw degrees."""
    return (px / frame_w - 0.5) * 360.0
