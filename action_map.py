#!/usr/bin/env python3
"""action_map.py - where was the action? A picture of one game from its player detections.

    python action_map.py --detections g030.detections.jsonl --out g030.action.png
    python action_map.py --detections g030.detections.jsonl g031.detections.jsonl --out game.action.png

Panels:
  * top left    where players were, in the camera's own angles (longitude along, foot latitude up/down)
  * top right   the action over time: mean longitude of the players (what the camera follows), the band
                most of them spread over, and how many players were seen
  * middle      a detection check: how many players were found, and how sure the detector was, by
                longitude and by how far away the player is. If these fall away at the edges or far
                side, the heat maps under-count those places.
  * bottom rows the same heat map for each block of --block minutes

Two choices keep the picture honest:
  * Each frame carries the same weight (its players share one unit), so frames where the detector
    happened to find many people do not outweigh frames where it found few.
  * The axes are angles, not metres. Turning foot angle into metres needs the camera's true horizon
    and height, which we don't have (the first version guessed and was far too small). Pass
    --metres --cam-height H for the old approximate ground view.

What it cannot fix: players the detector never finds (small, far, partly hidden, at the seams between
views) are missing from the data. The check panel shows where that probably happens.
"""
import argparse
import math
import sys
from typing import List, Tuple

import numpy as np

import auto_keyframes as AK


def ground_xy(lon_deg, lat_deg, cam_h: float, min_depr: float = 1.0):
    """Player feet (longitude, latitude in degrees) -> (along touchline x, distance from camera y) metres."""
    lon = np.radians(np.asarray(lon_deg, float))
    depr = np.maximum(-np.asarray(lat_deg, float), min_depr)          # angle below the horizon
    r = cam_h / np.tan(np.radians(depr))
    return r * np.sin(lon), r * np.cos(lon)


def collect(frames: List[dict], min_conf: float, cam_h: float, on_pitch_only: bool, metres: bool):
    """One row per detected player: time, x, y, weight (1/players-in-frame), confidence, box height."""
    t, x, y, w, c, hh = [], [], [], [], [], []
    for r in frames:
        P = [p for p in r["players"] if p[3] >= min_conf and (p[4] or not on_pitch_only)]
        if metres:
            P = [p for p in P if p[1] < -1.0]
        for p in P:
            if metres:
                gx, gy = ground_xy(p[0], p[1], cam_h)
            else:
                gx, gy = p[0], p[1]
            t.append(r["t"]); x.append(float(gx)); y.append(float(gy)); w.append(1.0 / len(P))
            c.append(p[3]); hh.append(p[2])
    return tuple(np.array(v) for v in (t, x, y, w, c, hh))


def blur(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return a
    k = np.arange(-int(3 * sigma), int(3 * sigma) + 1)
    k = np.exp(-0.5 * (k / sigma) ** 2); k /= k.sum()
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, a)


def density(x, y, w, xr, yr, bins=(90, 40), sigma=1.6):
    """share of all player-time in each cell (sums to 1 before blurring)."""
    w = w / max(float(np.sum(w)), 1e-12)
    H, _, _ = np.histogram2d(x, y, bins=bins, range=[xr, yr], weights=w)
    return blur(H, sigma).T                                           # rows = depth, cols = along


def per_frame_centre(frames: List[dict], min_conf: float, cam_h: float, metres: bool):
    """-> t, mean x, 10th/90th percentile x, player count per frame (NaN where nobody)."""
    T, M, LO, HI, N = [], [], [], [], []
    for r in frames:
        P = [p for p in r["players"] if p[3] >= min_conf and (p[1] < -1.0 or not metres)]
        T.append(r["t"]); N.append(len(P))
        if len(P) >= 2:
            gx = np.array([p[0] for p in P])
            if metres:
                gx, _ = ground_xy([p[0] for p in P], [p[1] for p in P], cam_h)
            wts = np.array([p[3] for p in P])
            M.append(float(np.average(gx, weights=wts))); LO.append(float(np.percentile(gx, 10)))
            HI.append(float(np.percentile(gx, 90)))
        else:
            M.append(np.nan); LO.append(np.nan); HI.append(np.nan)
    return np.array(T), np.array(M), np.array(LO), np.array(HI), np.array(N)


def smooth_nan(v: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(v, np.nan)
    for i in range(len(v)):
        s = v[max(0, i - n):i + n + 1]
        s = s[~np.isnan(s)]
        if len(s):
            out[i] = s.mean()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detections", nargs="+", required=True, help="one or more detections files (played in order)")
    ap.add_argument("--out", required=True, help="PNG to write")
    ap.add_argument("--cam-height", type=float, default=1.2, help="camera height above the ground, metres")
    ap.add_argument("--block", type=float, default=5.0, help="minutes per small map")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--metres", action="store_true", help="approximate ground metres instead of camera angles")
    ap.add_argument("--on-pitch-only", action="store_true", help="ignore people outside the pitch mask (needs --calib at extraction)")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # join the files on one clock (each file's own times, one after the other)
    frames, offset = [], 0.0
    for path in a.detections:
        hdr, fr = AK.load_detections(path)
        t0 = fr[0]["t"] if fr else 0.0
        for r in fr:
            frames.append({**r, "t": r["t"] - t0 + offset})
        offset += (fr[-1]["t"] - t0 + 1.0) if fr else 0.0
    if not frames:
        raise SystemExit("no frames in the detections")

    t, x, y, w, conf, bh = collect(frames, a.min_conf, a.cam_height, a.on_pitch_only, a.metres)
    if len(t) < 50:
        raise SystemExit("too few player detections to draw a map")
    pad = 3 if a.metres else 4
    xr = (float(np.percentile(x, 0.5)) - pad, float(np.percentile(x, 99.5)) + pad)
    yr = (float(np.percentile(y, 0.5)) - pad, float(np.percentile(y, 99.5)) + pad)
    if a.metres:
        yr = (max(0.0, yr[0]), yr[1])

    total = frames[-1]["t"]
    nb = max(1, int(math.ceil(total / (a.block * 60))))
    cols = min(nb, 4)
    rows_small = int(math.ceil(nb / cols))

    fig = plt.figure(figsize=(15, 7.2 + 2.6 * rows_small), constrained_layout=True)
    gs = fig.add_gridspec(2 + rows_small, cols * 2, height_ratios=[2.3, 1.15] + [1] * rows_small)
    ink, soft = "#1c1f26", "#8a8f9c"
    fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.96))       # leave room for the footnote

    # whole-game map
    ax = fig.add_subplot(gs[0, :cols])
    D = density(x, y, w, xr, yr)
    ax.imshow(D, origin="lower", extent=[xr[0], xr[1], yr[0], yr[1]], aspect=("equal" if a.metres else "auto"), cmap="magma_r")
    ax.set_title("Where players were found (whole game, each frame weighted equally)", loc="left", fontsize=12, color=ink)
    if a.metres:
        ax.set_xlabel("along the touchline (m, approx.)   ← left          right →")
        ax.set_ylabel("distance from camera (m, approx.)")
        ax.text(0.01, 0.97, "far touchline", transform=ax.transAxes, va="top", color=soft, fontsize=8)
        ax.text(0.01, 0.03, "camera side", transform=ax.transAxes, va="bottom", color=soft, fontsize=8)
    else:
        ax.set_xlabel("longitude (°)   ← left          right →")
        ax.set_ylabel("foot latitude (°)   far side ↑   ↓ near side")


    # timeline
    ax2 = fig.add_subplot(gs[0, cols:])
    T, M, LO, HI, N = per_frame_centre(frames, a.min_conf, a.cam_height, a.metres)
    n_s = max(1, int(round(6 * len(T) / max(total, 1))))                  # ~6 s smoothing
    Mm, LOm, HIm = smooth_nan(M, n_s), smooth_nan(LO, n_s), smooth_nan(HI, n_s)
    mins = T / 60.0
    ax2.fill_between(mins, LOm, HIm, color="#c9d3ea", lw=0, label="middle 80% of players")
    ax2.plot(mins, Mm, color="#2b4a9b", lw=1.1, label="action centre (what the camera follows)")
    ax2.axhline(np.nanmean(Mm), color=soft, lw=0.6, ls=":")
    ax2.set_xlim(0, mins[-1])
    ax2.set_xlabel("minutes"); ax2.set_ylabel("along the touchline (m, approx.)" if a.metres else "mean longitude (°)")
    ax2.set_title("The action over time", loc="left", fontsize=12, color=ink)
    ax2.legend(loc="upper right", fontsize=8, frameon=False)
    ax3 = ax2.twinx()
    ax3.plot(mins, smooth_nan(N.astype(float), n_s), color="#c0452f", lw=0.7, alpha=0.7)
    ax3.set_ylabel("players seen", color="#c0452f"); ax3.tick_params(axis="y", colors="#c0452f")
    ax3.set_ylim(0, max(30, np.nanmax(N) + 2))

    # detection check
    axd = fig.add_subplot(gs[1, :cols])
    nbin = 24
    edges = np.linspace(xr[0], xr[1], nbin + 1)
    which = np.clip(np.digitize(x, edges) - 1, 0, nbin - 1)
    nfr = max(len(frames), 1)
    cnt = np.array([(which == i).sum() for i in range(nbin)]) / nfr
    mc = np.array([conf[which == i].mean() if (which == i).sum() > 5 else np.nan for i in range(nbin)])
    mid = (edges[:-1] + edges[1:]) / 2
    axd.bar(mid, cnt, width=(edges[1] - edges[0]) * 0.9, color="#c9d3ea")
    axd.set_ylabel("players found per frame")
    axd.set_xlabel("longitude (°)" if not a.metres else "along the touchline (m)")
    axd.set_title("Detection check: by position along the pitch", loc="left", fontsize=11, color=ink)
    axd2 = axd.twinx(); axd2.plot(mid, mc, color="#c0452f", marker="o", ms=3, lw=1)
    axd2.set_ylabel("mean detector confidence", color="#c0452f"); axd2.set_ylim(0.3, 1.0)
    axd2.tick_params(axis="y", colors="#c0452f")

    axe = fig.add_subplot(gs[1, cols:])
    ye = np.linspace(yr[0], yr[1], nbin + 1)
    wy = np.clip(np.digitize(y, ye) - 1, 0, nbin - 1)
    cy = np.array([(wy == i).sum() for i in range(nbin)]) / nfr
    my = np.array([conf[wy == i].mean() if (wy == i).sum() > 5 else np.nan for i in range(nbin)])
    bh_y = np.array([bh[wy == i].mean() if (wy == i).sum() > 5 else np.nan for i in range(nbin)])
    ymid = (ye[:-1] + ye[1:]) / 2
    axe.bar(ymid, cy, width=(ye[1] - ye[0]) * 0.9, color="#c9d3ea")
    axe.set_ylabel("players found per frame")
    axe.set_xlabel("foot latitude (°): near side ← → far side" if not a.metres else "distance from camera (m)")
    axe.set_title("Detection check: by distance (player size on screen)", loc="left", fontsize=11, color=ink)
    axe2 = axe.twinx(); axe2.plot(ymid, my, color="#c0452f", marker="o", ms=3, lw=1)
    axe2.set_ylabel("mean detector confidence", color="#c0452f"); axe2.set_ylim(0.3, 1.0)
    axe2.tick_params(axis="y", colors="#c0452f")

    # blocks
    vmax = float(D.max())
    for i in range(nb):
        r_, c_ = divmod(i, cols)
        axb = fig.add_subplot(gs[2 + r_, c_ * 2:(c_ + 1) * 2])
        lo, hi = i * a.block * 60, (i + 1) * a.block * 60
        s = (t >= lo) & (t < hi)
        if s.sum() > 20:
            Db = density(x[s], y[s], w[s], xr, yr)
            axb.imshow(Db, origin="lower", extent=[xr[0], xr[1], yr[0], yr[1]], aspect=("equal" if a.metres else "auto"),
                       cmap="magma_r", vmin=0, vmax=max(float(Db.max()), 1e-9))
            axb.axvline(float(np.average(x[s], weights=w[s])), color="#2b4a9b", lw=1.0)
        axb.set_title(f"{int(lo // 60)}–{int(min(hi, total) // 60)} min", loc="left", fontsize=9, color=ink)
        axb.set_xticks([]); axb.set_yticks([])
    fig.suptitle(a.title or "Where the action was", x=0.01, ha="left", fontsize=14, color=ink)
    fig.text(0.01, 0.002, ("Approximate ground positions, camera height %s m; " % a.cam_height if a.metres else "Camera angles, not metres. ")
             + "Counts only players the detector found. Blue line = mean position in that block.", fontsize=8, color=soft, ha="left", va="bottom")
    fig.savefig(a.out, dpi=130)
    print(f"{len(frames)} frames, {len(t)} player detections, {total/60:.1f} min  ->  {a.out}")


if __name__ == "__main__":
    main()
