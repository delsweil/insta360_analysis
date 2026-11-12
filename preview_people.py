#!/usr/bin/env python3
"""preview_people.py - where do people stand in the picture? A quick look before the long detection run.

The coach draws the pitch outline on the unwrapped picture. To see which people the line keeps (green) and leaves out
(red: coaches, substitutes, spectators on the touchline) it needs players in the picture. The full detection run comes
later, so this looks at about a dozen moments spread over the video and writes where the feet were.

The unwrapped picture does not move with the virtual camera, so feet from different moments sit on the same picture.

  python preview_people.py --video game.insv --start 0 --duration 1700 --out pitch0.preview.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, List, Optional

import numpy as np

import extract_detections as X
import kf_core as K
import scene_check as S

HERE = os.path.dirname(os.path.abspath(__file__))
IV_FOV = 112.0
TILT = -22.0                  # the same as the full detection run (pipeline.DETECT_TILT)
MIN_CONF = 0.5                # the same as Job.feet()
MIN_HEIGHT = 18.0


def sample_times(start: float, duration: float, n: int) -> List[float]:
    """n moments spread over the video, leaving out the first and last tenth (warm-up, walking off)."""
    if n <= 1 or duration <= 0:
        return [start + duration * 0.5]
    return [start + duration * (0.1 + 0.8 * k / (n - 1)) for k in range(n)]


def feet_pixels(players: List[dict]) -> List[List[int]]:
    """[[x, y], ...] in the 2880x1440 unwrapped picture, for the players that are sure enough."""
    out = []
    for p in players:
        if p["conf"] < MIN_CONF or p["h"] < MIN_HEIGHT:
            continue
        x, y = K.lonlat_to_xy(p["lon"], p["lat"], X.EQ_W, X.EQ_H)
        out.append([int(round(float(x))), int(round(float(y)))])
    return out


def run(video: str, start: float, duration: float, n: int, raw: bool, model, device: str,
        grab: Optional[Callable] = None, progress: Optional[Callable[[int, int], None]] = None) -> dict:
    views = X.make_views(-80.0, 80.0, TILT, 40.0, legacy_fov=False)
    grab = grab or (lambda t: S.grab_frame(video, t, IV_FOV if raw else None, raw))
    times, feet = [], []
    ts = sample_times(start, duration, n)
    for k, t in enumerate(ts):
        fr = grab(t)
        if fr is not None:
            pl = S.detect_scene(fr, views, model, device, S.SQUARE_FY, 0.75, X.CONF_PLAYERS, 98.0)
            feet += feet_pixels(pl)
            times.append(round(t, 2))
        if progress:
            progress(k + 1, len(ts))
    return {"times": times, "feet": feet}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--raw", type=int, default=1, help="1 = Insta360 single-lens file, 0 = already unwrapped")
    ap.add_argument("--players", default=os.path.join(HERE, "models", "yolo11s.pt"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from ultralytics import YOLO
    model = YOLO(a.players)
    device = X.pick_device(a.device)
    res = run(a.video, a.start, a.duration, a.n, bool(a.raw), model, device,
              progress=lambda k, n: print(f"frame {k}/{n}", flush=True))
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh)
    os.replace(tmp, a.out)
    print(f"{len(res['feet'])} people seen in {len(res['times'])} pictures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
