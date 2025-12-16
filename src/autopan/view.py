# src/autopan/view.py
from dataclasses import dataclass
import numpy as np
from py360convert import e2p


@dataclass
class CameraState:
    yaw: float
    pitch: float
    yaw_center: float
    pitch_center: float


def project_e2p(frame360: np.ndarray, yaw_deg: float, pitch_deg: float, fov_deg: float, out_h: int, out_w: int) -> np.ndarray:
    """
    Projects equirectangular -> perspective using py360convert.e2p.
    frame360 can be HxWx3 uint8.
    """
    return e2p(frame360, fov_deg=fov_deg, u_deg=yaw_deg, v_deg=pitch_deg, out_hw=(out_h, out_w))


def project_mask_e2p(mask360: np.ndarray, yaw_deg: float, pitch_deg: float, fov_deg: float, out_h: int, out_w: int) -> np.ndarray:
    """
    Projects a binary/grayscale 360 mask into the perspective view.

    py360convert prefers 3-channel, so we safely expand:
      mask360: HxW uint8 (0/255)
      returns: HxW uint8-ish (0..255)
    """
    if mask360.ndim != 2:
        raise ValueError("mask360 must be a 2D array (HxW).")

    # Expand to 3-channel so e2p behaves consistently.
    m3 = np.repeat(mask360[:, :, None], 3, axis=2)
    p3 = e2p(m3, fov_deg=fov_deg, u_deg=yaw_deg, v_deg=pitch_deg, out_hw=(out_h, out_w))

    # Convert back to single channel
    if p3.ndim == 3:
        return p3[:, :, 0]
    return p3
