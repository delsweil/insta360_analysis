# src/autopan/world.py
import json
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple

import cv2
import numpy as np


@dataclass
class Pitch360:
    """
    Pitch polygon represented in 360/equirectangular pixel coords.
    poly_360: list of (x,y) points in the input equirect frame coordinate system.
    """
    poly_360: np.ndarray  # shape (N,2), int32
    in_w: int
    in_h: int

    def build_mask360(self) -> np.ndarray:
        """Binary mask in equirect space (uint8 0/255)."""
        mask = np.zeros((self.in_h, self.in_w), dtype=np.uint8)
        cv2.fillPoly(mask, [self.poly_360.astype(np.int32)], 255)
        return mask


def _as_int32_poly(points: List[List[float]]) -> Optional[np.ndarray]:
    if not points or len(points) < 3:
        return None
    arr = np.array(points, dtype=np.float32)
    arr = np.round(arr).astype(np.int32)
    return arr


def load_pitch_polygon(calib_path: str) -> Optional[Pitch360]:
    """
    Loads pitch calibration.

    Preferred (new):
      {
        "video": "...",
        "frame_index": 0,
        "equirect": {"in_w": 3840, "in_h": 1920},
        "pitch_polygon_360": [[x,y], ...]
      }

    Legacy (old):
      { "pitch_polygon": [[x,y], ...] }  # these were perspective coords (NOT usable for panning)
    """
    if not calib_path or not os.path.exists(calib_path):
        print(f"[WARN] No calibration found at {calib_path}")
        return None

    with open(calib_path, "r") as f:
        data = json.load(f)

    poly360 = _as_int32_poly(data.get("pitch_polygon_360"))
    eq = data.get("equirect", {})
    in_w = int(eq.get("in_w", 0))
    in_h = int(eq.get("in_h", 0))

    if poly360 is not None and in_w > 0 and in_h > 0:
        print(f"[OK] Loaded 360 pitch polygon with {len(poly360)} points ({in_w}x{in_h})")
        return Pitch360(poly_360=poly360, in_w=in_w, in_h=in_h)

    # Legacy fallback: warn loudly
    legacy = _as_int32_poly(data.get("pitch_polygon"))
    if legacy is not None:
        print(
            "[WARN] pitch.json contains 'pitch_polygon' (legacy perspective coords). "
            "This will NOT track yaw/pitch correctly. Recalibrate with calibrate_pitch_360.py."
        )
        return None

    print("[WARN] pitch.json exists but contains no usable polygon.")
    return None
