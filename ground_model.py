#!/usr/bin/env python3
"""ground_model.py — where does a player stand: from the height of his box, from where his feet are, or from both?

The board has so far placed a player by the height of his box (a person is 1.7 m: tall box = near). A pixel of error is 4-7 %
of the height of a far player, so far dots scatter by 25-35 %: some land beyond the far touchline. The place of the feet
in the picture is a second ruler: on flat ground a foot at depression angle a below the horizon stands H / tan(a) from the
camera, and the angle is measured to a fraction of a degree. Near the camera the box is the better ruler (the foot angle
hardly changes with distance there), far away the feet are. Both are combined here, each weighted by its own precision.

One camera height per video is fitted on the clear near and middle players (c = box height / tan(depression)); it is not
needed to know the real height. If the two rulers disagree by much more than their combined error (legs hidden by a wall,
a player on a bank), the box height is kept.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

K_ANGLE = 0.75
MIN_DEPRESSION = 3.0          # degrees below the horizon: closer to the horizon the foot ruler is too steep to use
ANGLE_ERR = 0.30              # degrees: error of the foot position in the picture
SIGMA_BOX = 0.06              # relative error of a box height from the build of the player and the pose
SIGMA_PIX = 1.5               # px: error of the box edges
SIGMA_FLAT = 0.03             # relative error of the flat-ground model
GATE = 2.0                    # disagreement (in sigmas) above which the foot position is not believed (5 % of good ones fail it: harmless)


def model_h(lon, lat, centre, c: float, delta: float = 0.0):
    """Box height a standing person would have at this foot position: c * tan(depression) / cos(angle off the view axis)."""
    dep = np.radians(np.maximum(-np.asarray(lat, float) + delta, 0.5))
    off = np.radians(K_ANGLE * (np.asarray(lon, float) - np.asarray(centre, float)))
    return c * np.tan(dep) / np.maximum(np.cos(off), 0.3)


def fit(lon, lat, h, centre, min_h: float = 22.0, deltas=(0.0,)) -> Optional[Tuple[float, float, float, int]]:
    """-> (c, delta, typical log error, number of players used), or None when there is too little to fit."""
    lon, lat, h, centre = (np.asarray(v, float) for v in (lon, lat, h, centre))
    ok = (h >= min_h) & (lat < -MIN_DEPRESSION)
    if ok.sum() < 200 or (lat[ok].max() - lat[ok].min()) < 3.0:
        return None
    best = None
    for d in deltas:
        base = model_h(lon[ok], lat[ok], centre[ok], 1.0, d)
        c = float(np.median(h[ok] / base))
        res = float(np.median(np.abs(np.log(h[ok] / (c * base)))))
        if best is None or res < best[2] - 1e-9:
            best = (c, float(d), res, int(ok.sum()))
    return best


def fuse(lon, lat, h, centre, c: float, delta: float = 0.0):
    """Box height to use for the position: box and foot ruler combined in log space. -> (h_used, state) where state is
    0 = box height kept (near the horizon or the rulers disagree), 1 = combined."""
    lon, lat, h, centre = (np.atleast_1d(np.asarray(v, float)) for v in (lon, lat, h, centre))
    hf = model_h(lon, lat, centre, c, delta)
    dep = np.radians(np.clip(-lat + delta, MIN_DEPRESSION, 85.0))
    s_f = np.sqrt((np.radians(ANGLE_ERR) / (np.sin(dep) * np.cos(dep))) ** 2 + SIGMA_FLAT ** 2)
    s_b = np.sqrt(SIGMA_BOX ** 2 + (SIGMA_PIX / np.maximum(h, 4.0)) ** 2)
    ok = (-lat + delta >= MIN_DEPRESSION) & (np.abs(np.log(np.maximum(h, 1.0) / hf)) < GATE * np.sqrt(s_b ** 2 + s_f ** 2))
    w_b = (1 / s_b ** 2) / (1 / s_b ** 2 + 1 / s_f ** 2)
    used = np.where(ok, np.exp(w_b * np.log(np.maximum(h, 1.0)) + (1 - w_b) * np.log(hf)), h)
    return used, ok.astype(int)
