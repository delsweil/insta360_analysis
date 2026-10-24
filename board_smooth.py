#!/usr/bin/env python3
"""board_smooth.py — Kalman smoothing of the player positions on the tactical board.

A position on the board comes from one detection: the box height and the foot position say how far away someone is, and
both are noisy (a box that is a few pixels too tall puts the player metres closer; far away it matters most). So a
player standing still jitters on the board, and a running one zig-zags.

board_clean.link_tracks already chains detections of consecutive moments into tracks. Along each track the player is
modelled as a point moving with a nearly constant velocity (constant velocity + random acceleration), each axis on its
own. A Kalman filter runs forward, a Rauch-Tung-Striebel smoother backward, so every position is estimated from the
moments before *and* after it. Details that matter on a football pitch:

  * the measurement noise grows with the distance from the camera (0.4 m + 4 % of it): far positions are trusted less
  * a detection far away from where the track was heading (a wrong box, two players merged) is down-weighted (innovation
    gate) instead of dragging the track
  * short tracks (< 3 detections) and the moments themselves are left as they are: nothing is added or removed, only
    the x and y of existing detections move, and by at most a few metres

Pure functions on arrays, no I/O. The camera path and the statistics are not touched.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

SIGMA_ACC = 3.0           # m/s^2: how much a player's velocity may change per second (random acceleration)
R0, R_PER_M = 0.4, 0.04   # measurement sigma in metres: R0 + R_PER_M * distance from the camera
GATE = 4.0                # an innovation beyond this many sigma is treated as a bad measurement
GATE_INFLATE = 25.0       # ... its variance is multiplied by this


def smooth_axis(z: np.ndarray, dt: np.ndarray, sig_r: np.ndarray, sigma_acc: float = SIGMA_ACC,
                gate: float = GATE) -> Tuple[np.ndarray, int]:
    """RTS-smoothed positions of one axis. z: n measurements, dt: n-1 time steps (s), sig_r: n measurement sigmas.
    Returns (smoothed positions, number of measurements that were gated)."""
    n = len(z)
    xs = np.zeros((n, 2))             # filtered state (pos, vel)
    Ps = np.zeros((n, 2, 2))
    xp = np.zeros((n, 2))             # predicted
    Pp = np.zeros((n, 2, 2))
    Fs = np.zeros((n, 2, 2))
    x = np.array([z[0], 0.0])
    P = np.array([[sig_r[0] ** 2, 0.0], [0.0, 4.0]])      # velocity unknown: sigma 2 m/s
    gated = 0
    q = sigma_acc ** 2
    for k in range(n):
        if k > 0:
            h = float(dt[k - 1])
            F = np.array([[1.0, h], [0.0, 1.0]])
            Q = q * np.array([[h ** 4 / 4, h ** 3 / 2], [h ** 3 / 2, h ** 2]])
            x = F @ x
            P = F @ P @ F.T + Q
            Fs[k] = F
        xp[k], Pp[k] = x, P
        r = sig_r[k] ** 2
        s = P[0, 0] + r
        if k > 0 and abs(z[k] - x[0]) > gate * np.sqrt(s):
            r *= GATE_INFLATE
            s = P[0, 0] + r
            gated += 1
        kg = P[:, 0] / s
        x = x + kg * (z[k] - x[0])
        P = P - np.outer(kg, P[0])
        xs[k], Ps[k] = x, P
    out = xs.copy()
    for k in range(n - 2, -1, -1):
        C = Ps[k] @ Fs[k + 1].T @ np.linalg.inv(Pp[k + 1])
        out[k] = xs[k] + C @ (out[k + 1] - xp[k + 1])
    return out[:, 0], gated


def smooth(frame: np.ndarray, vt: np.ndarray, x: np.ndarray, y: np.ndarray, track: np.ndarray,
           sigma_acc: float = SIGMA_ACC, min_len: int = 3) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Smoothed copies of x, y (metres, y = away from the camera) of detections chained into `track`s
    (board_clean.link_tracks). frame: moment index of each detection, vt: time of each moment."""
    nx, ny = x.astype(float).copy(), y.astype(float).copy()
    stats = {"smoothed": 0, "gated": 0, "moved_m": 0.0}
    if len(x) == 0:
        return nx, ny, stats
    order = np.lexsort((frame, track))
    ts = track[order]
    cuts = np.flatnonzero(np.diff(ts)) + 1
    moved = 0.0
    for idx in np.split(order, cuts):
        if len(idx) < min_len or track[idx[0]] < 0:
            continue
        t = vt[frame[idx]]
        dt = np.maximum(np.diff(t), 1e-3)
        sig = R0 + R_PER_M * np.maximum(y[idx], 0.0)
        sx, g1 = smooth_axis(x[idx], dt, sig, sigma_acc)
        sy, g2 = smooth_axis(y[idx], dt, sig, sigma_acc)
        moved += float(np.hypot(sx - x[idx], sy - y[idx]).sum())
        nx[idx], ny[idx] = sx, sy
        stats["smoothed"] += len(idx)
        stats["gated"] += g1 + g2
    stats["moved_m"] = moved
    return nx, ny, stats
