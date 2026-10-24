#!/usr/bin/env python3
"""
lens.py — the real geometry of the Insta360 X2 single-lens recording (2560x1440, "150 degree" mode).

Why this exists
---------------
The pipeline unwraps the raw frame with FFmpeg's  v360=fisheye:equirect:ih_fov=200 .  That model is wrong
for this lens (it is not a 200 degree equidistant fisheye): straight pitch lines come out curved, the near
touchline of a flat view bows by about 4 % of its length, and the unwrapped sphere is stretched sideways.
Fitting the lens to straight pitch lines (a touchline and the halfway line, 100+ points) gives a
*stereographic* fisheye,  r = 2 f tan(theta / 2),  f = 850.65 px, about 148 degrees across the 2560 px width.
The straight lines then stay straight to 0.03 % (measured), where the old model gave 4.3 %.

What it does
------------
Keyframes, detections and the board are still made in the old (legacy) unwrapped coordinates, and stay valid.
`convert_view` translates one legacy camera view (yaw, pitch, h_fov) into the view of the *same part of the
picture* in the true lens model, so the render can read the raw frame in ONE v360 pass
(v360=sg:flat, no intermediate equirect: also one resampling less):

    legacy view -> its rays -> raw pixels (legacy model) -> rays (true lens) -> yaw, pitch, h_fov

and shrinks the view when it would reach past what the lens captured (the pale blue patch).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

RAW_W, RAW_H = 2560, 1440
LEGACY_IH_FOV = 200.0
DEFAULT_IV_FOV = 112.0

# fitted on a Laufamholz frame (see README: "Lens model")
F_PX = 850.65
CX, CY = RAW_W / 2.0, RAW_H / 2.0

LENSES = ("legacy", "x2_150")


def sg_fovs() -> Tuple[float, float]:
    """ih_fov / iv_fov for FFmpeg's v360 input=sg that reproduce the fitted lens."""
    ih = 4.0 * math.degrees(math.atan((RAW_W / 2.0) / F_PX / 2.0))
    iv = 4.0 * math.degrees(math.atan((RAW_H / 2.0) / F_PX / 2.0))
    return ih, iv


# ── directions: x right, y up, z forward (the optical axis) ──────────────────

def _basis(yaw_deg: float, pitch_deg: float):
    lon, lat = math.radians(yaw_deg), math.radians(pitch_deg)
    f = np.array([math.cos(lat) * math.sin(lon), math.sin(lat), math.cos(lat) * math.cos(lon)])
    r = np.array([math.cos(lon), 0.0, -math.sin(lon)])
    return f, r, np.cross(f, r)


def legacy_dir_to_raw(v: np.ndarray, iv_fov: float = DEFAULT_IV_FOV) -> Tuple[np.ndarray, np.ndarray]:
    """Raw pixel that FFmpeg's fisheye unwrap (ih_fov=200, iv_fov=iv_fov) shows in direction v."""
    v = np.asarray(v, float)
    th = np.degrees(np.arccos(np.clip(v[..., 2], -1.0, 1.0)))
    phi = np.arctan2(v[..., 1], v[..., 0])
    xn = th * np.cos(phi) / (LEGACY_IH_FOV / 2.0)
    yn = -th * np.sin(phi) / (iv_fov / 2.0)
    return (xn + 1.0) / 2.0 * RAW_W - 0.5, (yn + 1.0) / 2.0 * RAW_H - 0.5


def true_dir_to_raw(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(v, float)
    th = np.arccos(np.clip(v[..., 2], -1.0, 1.0))
    r = 2.0 * F_PX * np.tan(th / 2.0)
    phi = np.arctan2(v[..., 1], v[..., 0])
    return CX - 0.5 + r * np.cos(phi), CY - 0.5 - r * np.sin(phi)


def raw_to_true_dir(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    x = np.asarray(px, float) + 0.5 - CX
    y = CY - (np.asarray(py, float) + 0.5)
    th = 2.0 * np.arctan(np.hypot(x, y) / (2.0 * F_PX))
    phi = np.arctan2(y, x)
    return np.stack([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)], axis=-1)


def _inside(px, py, margin: float) -> bool:
    return bool(np.all((px >= margin) & (px <= RAW_W - 1 - margin) & (py >= margin) & (py <= RAW_H - 1 - margin)))


def _samples(tx: float, ty: float, n: int = 7) -> np.ndarray:
    """(gx, gy) points on the image plane: the centre and n points along each side (corners included)."""
    t = np.linspace(-1.0, 1.0, n)
    top = [(x * tx, ty) for x in t]
    bot = [(x * tx, -ty) for x in t]
    left = [(-tx, y * ty) for y in t[1:-1]]
    right = [(tx, y * ty) for y in t[1:-1]]
    return np.array([[0.0, 0.0]] + top + bot + left + right, float)


def _rays(yaw, pitch, tx, ty):
    f, r, u = _basis(yaw, pitch)
    g = _samples(tx, ty)
    return f[None, :] + g[:, 0:1] * r[None, :] + g[:, 1:2] * u[None, :]


def _unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def _lonlat(v):
    return (math.degrees(math.atan2(v[0], v[2])), math.degrees(math.atan2(v[1], math.hypot(v[0], v[2]))))


def convert_view(yaw: float, pitch: float, h_fov: float, aspect: float, iv_fov: float = DEFAULT_IV_FOV,
                 margin_px: float = 4.0, min_scale: float = 0.55) -> Dict[str, float]:
    """The view (yaw, pitch, h_fov) of the legacy unwrapping, as a view of the true lens.
    `aspect` = out_h / out_w (square pixels). The result has the same centre and the same horizontal reach
    on the picture; `scale` < 1 says the view had to be narrowed to stay inside the lens image."""
    tx = math.tan(math.radians(h_fov) / 2.0)
    ty = tx * aspect
    f, r, u = _basis(yaw, pitch)
    key = _unit(np.stack([f, f - tx * r, f + tx * r]))             # centre, left middle, right middle
    px, py = legacy_dir_to_raw(key, iv_fov)
    d = raw_to_true_dir(np.clip(px, 0, RAW_W - 1), np.clip(py, 0, RAW_H - 1))
    t_yaw, t_pitch = _lonlat(d[0])
    hf = math.degrees(math.acos(float(np.clip(d[1] @ d[2], -1, 1))))
    scale = 1.0
    while True:
        tx2 = math.tan(math.radians(hf * scale) / 2.0)
        qx, qy = true_dir_to_raw(_unit(_rays(t_yaw, t_pitch, tx2, tx2 * aspect)))
        if _inside(qx, qy, margin_px) or scale <= min_scale:
            break
        scale *= 0.985
    hfs = hf * scale
    vfs = math.degrees(2.0 * math.atan(math.tan(math.radians(hfs) / 2.0) * aspect))
    return {"yaw": t_yaw, "pitch": t_pitch, "h_fov": hfs, "v_fov": vfs, "scale": scale}
