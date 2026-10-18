#!/usr/bin/env python3
"""pitch_filter.py — mark the players who stand outside the pitch (coaches, substitutes, spectators).

Uses a pitch outline (calibration/pitch.json from calibrate_auto.py, in the pixels of the unwrapped
2880x1440 frame) and tests the feet of every detected player against it. Players outside get the
"on pitch" flag False (the fifth value); every later step (camera path, heat map, statistics) leaves
them out. The detections are rewritten, so no new extraction is needed. Colours (sixth value) stay.

  # 1) one frame, unwrapped exactly like the video, then the outline
  ffmpeg -ss 600 -i VID.mp4 -vf "v360=fisheye:equirect:ih_fov=200:iv_fov=112,scale=2880:1440" -frames:v 1 -update 1 frame.png
  python calibrate_auto.py --input frame.png --output pitch034.json --vis calib_vis.jpg
  # 2) look at calib_vis.jpg; if the outline is right:
  python pitch_filter.py --detections h034.detections.jsonl --calib pitch034.json --out h034.detections.pitch.jsonl

  --margin 25   how far outside the outline (in frame pixels, about 3 degrees) a foot may be and still count
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

import kf_core as K
import extract_detections as X


def pitch_mask(calib: str, margin: float) -> np.ndarray:
    import cv2
    m = X.polygon_mask(X.load_polygon(calib))
    if margin > 0:
        k = int(round(margin)) * 2 + 1
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m


def foot_on_mask(mask: np.ndarray, lon: float, lat: float) -> bool:
    x, y = K.lonlat_to_xy(lon, lat, X.EQ_W, X.EQ_H)
    xi = int(np.clip(round(float(x)), 0, X.EQ_W - 1))
    yi = int(np.clip(round(float(y)), 0, X.EQ_H - 1))
    return bool(mask[yi, xi])


def apply(src: str, dst: str, calib: str, margin: float = 25.0) -> tuple:
    mask = pitch_mask(calib, margin)
    n_all = n_off = 0
    with open(src) as fi, open(dst, "w") as fo:
        header = json.loads(fi.readline())
        header["pitch_filter"] = {"calib": calib, "margin_px": margin}
        fo.write(json.dumps(header) + "\n")
        for line in fi:
            if not line.strip():
                continue
            r = json.loads(line)
            for p in r["players"]:
                on = foot_on_mask(mask, p[0], p[1])
                while len(p) < 5:
                    p.append(None)
                p[4] = on
                n_all += 1
                n_off += 0 if on else 1
            fo.write(json.dumps(r) + "\n")
    return n_all, n_off


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detections", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--out", required=True, help="new detections file (the input is never changed)")
    p.add_argument("--margin", type=float, default=25.0)
    a = p.parse_args(argv)
    if a.out == a.detections:
        raise SystemExit("--out must differ from --detections")
    n, off = apply(a.detections, a.out, a.calib, a.margin)
    print(f"{n} detections, {off} outside the pitch ({100 * off / max(n, 1):.1f}%) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
