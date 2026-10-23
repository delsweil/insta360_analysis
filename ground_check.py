#!/usr/bin/env python3
"""ground_check.py — is the distance of far players too big? (diagnosis only, changes nothing)

The board places a player by the height of his box: a tall box is near, a short one far. At 14-24 px a pixel of error is
4-7 % of the height. Where his FEET are in the picture is a steadier ruler: a foot at depression angle a below the horizon
stands H / tan(a) from the camera, H = height of the camera. This fits H (and a small horizon offset) on the clear near and
middle players of one video and then shows

  * how well one camera height explains the box heights in each distance band,
  * where the far players would stand if they were placed by foot position,
  * where your pitch outline falls under that model (the real size of the pitch in the board's metres).

  python ground_check.py --detections v0.detections.jsonl --calib pitch0.manual.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

import auto_keyframes as AK
import ground_model as GM
import scene_check as S

K_ANGLE = 0.75


def load(det_path: str, min_conf: float, max_lon: float = 98.0):
    hdr, frames = AK.load_detections(det_path, max_lon=max_lon)
    views = np.array(hdr["view_lons"], float)
    fy = S.view_fy(hdr)
    rows = []
    for r in frames:
        for p in r["players"]:
            if p[3] >= min_conf:
                rows.append((p[0], p[1], p[2], p[3]))
    a = np.array(rows, float).reshape(-1, 4)
    centre = views[np.argmin(np.abs(views[None, :] - a[:, 0:1]), axis=1)] if len(a) else np.zeros(0)
    return a, centre, fy


def fit(a, centre, min_h: float = 22.0, deltas=(0.0,)):
    f = GM.fit(a[:, 0], a[:, 1], a[:, 2], centre, min_h, deltas)
    if f is None:
        raise SystemExit("too few clear players to fit one camera height")
    return (f[0], f[1], f[2]), f[3]


model_h = GM.model_h


def place(lon, h, centre, fy):
    return S.ground_position(float(lon), float(h), float(centre), fy, K_ANGLE)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--calib", default=None, help="pitch outline file")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--fit-horizon", action="store_true",
                    help="also fit a horizon offset (-1.5..1.5 deg); weakly determined, so off unless you suspect a tilted camera")
    a = ap.parse_args(argv)

    det, centre, fy = load(a.detections, a.min_conf)
    if len(det) < 50:
        print("too few confident players to fit")
        return 1
    (c, delta, res), n = fit(det, centre, deltas=np.arange(-1.5, 1.51, 0.25) if a.fit_horizon else (0.0,))
    print(f"fitted on {n} players with boxes >= 22 px: c = {c:.0f}, horizon offset = {delta:+.2f} deg, typical error {100 * (math.exp(res) - 1):.0f} %")
    lon, lat, h = det[:, 0], det[:, 1], det[:, 2]
    ruler = np.array([place(l, hh, cc, fy) for l, hh, cc in zip(lon, h, centre)])
    he = model_h(lon, lat, centre, c, delta)
    feet = np.array([place(l, hh, cc, fy) for l, hh, cc in zip(lon, he, centre)])
    ok = lat < -3.0
    ok = lat < -3.0
    print("\nplayers by distance from the FOOT position (the steadier ruler), y in metres:")
    print(f"{'band':>10} {'players':>8} {'foot y':>8} {'ruler y':>8} {'ruler/foot':>11} {'16-84 %':>14}")
    for name, lo, hi in (("near <30", 0, 30), ("mid 30-55", 30, 55), ("far 55-70", 55, 70), ("far >70", 70, 1e9)):
        m = ok & (feet[:, 1] >= lo) & (feet[:, 1] < hi)
        if m.sum() >= 5:
            r = ruler[m, 1] / feet[m, 1]
            print(f"{name:>10} {int(m.sum()):>8} {np.median(feet[m, 1]):8.1f} {np.median(ruler[m, 1]):8.1f} {np.median(r):11.2f} "
                  f"{np.percentile(r, 16):6.2f}-{np.percentile(r, 84):.2f}")
    if a.calib:
        import extract_detections as X
        import kf_core as K
        poly = X.load_polygon(a.calib)
        pts = []
        skipped = 0
        for px, py in poly:
            lo, la = K.xy_to_lonlat(float(px), float(py), X.EQ_W, X.EQ_H)
            if not (abs(lo) <= 98.0 and la < -3.0):
                skipped += 1
                print(f"  skipped: lon {float(lo):.1f} lat {float(la):.1f} (outside the lens range or above 3 degrees below the horizon)")
            if abs(lo) <= 98.0 and la < -3.0:
                pts.append((float(lo), float(la)))
        print(f"\nthe outline has {len(poly)} points; (lon, lat) of each, then where it falls in the board's metres:")
        views = sorted({round(float(v)) for v in np.unique(centre)})
        out = []
        for lo, la in pts:
            cen = float(min(views, key=lambda v: abs(v - lo))) if views else 0.0
            hh = float(model_h(lo, la, cen, c, delta))
            out.append(place(lo, hh, cen, fy))
            print(f"  lon {lo:7.1f}  lat {la:6.1f}  ->  x {out[-1][0]:6.1f}  y {out[-1][1]:6.1f}")
        if out:
            o = np.array(out)
            print(f"\nyour outline under the foot-position model: x {o[:, 0].min():.1f} .. {o[:, 0].max():.1f} m, "
                  f"y {o[:, 1].min():.1f} .. {o[:, 1].max():.1f} m  (width of the pitch seen from here: {o[:, 1].max() - o[:, 1].min():.1f} m)")
            far = o[:, 1].max()
            print(f"players beyond that far edge: by the box-height ruler {100 * np.mean(ruler[ok, 1] > far):.1f} %, "
                  f"by foot position {100 * np.mean(feet[ok, 1] > far):.1f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
