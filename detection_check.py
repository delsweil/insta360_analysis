#!/usr/bin/env python3
"""detection_check.py - how far can the player detections be trusted? No ground truth needed.

    python detection_check.py --detections g030.detections.jsonl [--players-on-pitch 16]

Three checks, all from the detections themselves:

 1. STATIC SPOTS   places where a "player" sits in nearly the same spot for much of the game (a coach,
                   a spectator, a post, a bag, a reflection, or something beyond the lens edge).
                   These pull the action centre and the tilt.
 2. FLICKER        a player found in the frame before and the frame after, but not in between, was
                   missed. The share of such gaps (by longitude and by distance) is a LOWER bound on
                   the miss rate there: players the detector never finds are invisible to this check.
 3. HEAD-COUNT     with --players-on-pitch N (players actually on the pitch, both teams): the average
                   found per frame as a share of N, by block of minutes. A rough overall recall.

It also lists the most useful 12 timestamps to spot-check by eye (frames with the biggest gap between
the people found and the people expected), so you can count players by hand on those.
"""
import argparse
import sys
from typing import List

import numpy as np

import auto_keyframes as AK

R = 3.0          # degrees: "the same player" in the next frame
CELL = 2.0       # degrees: grid for static spots


def arrays(frames: List[dict], min_conf: float):
    out = []
    for r in frames:
        P = [p for p in r["players"] if p[3] >= min_conf]
        out.append(np.array([[p[0], p[1], p[3]] for p in P], float).reshape(-1, 3))
    return out


def static_spots(frames, A, frac=0.20, step_deg=1.0):
    """cells occupied in > frac of the frames by a detection that hardly moves."""
    n = len(frames)
    occ = {}
    for i, a in enumerate(A):
        seen = {(int(np.floor(x / CELL)), int(np.floor(y / CELL))) for x, y, _ in a}
        for c in seen:
            occ.setdefault(c, []).append(i)
    spots = []
    for c, idx in occ.items():
        if len(idx) / n < frac:
            continue
        pts = np.array([[x, y] for i in idx for x, y, _ in A[i]
                        if int(np.floor(x / CELL)) == c[0] and int(np.floor(y / CELL)) == c[1]])
        if len(pts) and pts.std(axis=0).max() < step_deg * 1.2:
            spots.append((len(idx) / n, float(pts[:, 0].mean()), float(pts[:, 1].mean()), float(np.mean([A[i][:, 2].mean() for i in idx]))))
    spots.sort(reverse=True)
    merged = []                                            # neighbouring cells are one spot: re-measure it as a whole
    for share, lon, lat, c in spots:
        if any(np.hypot(lon - m[1], lat - m[2]) < 3.0 for m in merged):
            continue
        hit = sum(1 for a in A if len(a) and np.hypot(a[:, 0] - lon, a[:, 1] - lat).min() < 2.5)
        merged.append((hit / n, lon, lat, c))
    merged.sort(reverse=True)
    return merged


def flicker(A, step=1):
    """-> per-detection records (lon, lat, missed) for players present at i-1 and i+1 but maybe not at i."""
    rec = []
    for i in range(1, len(A) - 1):
        prev, cur, nxt = A[i - 1], A[i], A[i + 1]
        if not len(prev) or not len(nxt):
            continue
        for p in prev:
            d_n = np.hypot(nxt[:, 0] - p[0], nxt[:, 1] - p[1])
            if not len(d_n) or d_n.min() > 2 * R:                # nobody near it two frames later: not the same player
                continue
            q = nxt[int(d_n.argmin())]
            mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
            hit = len(cur) and np.hypot(cur[:, 0] - mid[0], cur[:, 1] - mid[1]).min() <= R
            rec.append((mid[0], mid[1], 0 if hit else 1))
    return np.array(rec)


def table(vals, miss, edges, label):
    print(f"\n  miss rate by {label}   (gaps = players found before and after but not between)")
    print(f"  {'range':>14}  {'cases':>7}  {'missed':>7}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (vals >= lo) & (vals < hi)
        if s.sum() >= 30:
            print(f"  {lo:6.0f}…{hi:<6.0f}  {int(s.sum()):7d}  {100 * miss[s].mean():6.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--players-on-pitch", type=int, default=None, help="players on the pitch, both teams (no subs, no referee)")
    ap.add_argument("--block", type=float, default=5.0, help="minutes per head-count line")
    a = ap.parse_args()

    hdr, frames = AK.load_detections(a.detections)
    A = arrays(frames, a.min_conf)
    T = np.array([r["t"] for r in frames])
    N = np.array([len(x) for x in A])
    print(f"{len(frames)} frames, {(T[-1] - T[0]) / 60:.1f} min, {N.mean():.1f} players found per frame on average")

    print("\n1) STATIC SPOTS (a 'player' that stays put for more than 20% of the game)")
    sp = static_spots(frames, A)
    if not sp:
        print("  none found")
    for share, lon, lat, c in sp[:8]:
        extra = "  <- beyond the lens edge (about +/-100 deg), almost certainly not a player" if abs(lon) > 100 else ""
        print(f"  longitude {lon:7.1f}  latitude {lat:6.1f}   present {100 * share:4.0f}% of frames, confidence {c:.2f}{extra}")
    if sp:
        print("  -> exclude these before steering the camera: they act as a permanent 'player' there.")

    print("\n2) FLICKER (missed in between frames)")
    rec = flicker(A)
    if len(rec) < 200:
        print("  too few cases")
    else:
        print(f"  overall: {100 * rec[:, 2].mean():.1f}% of players seen before and after were missed in between "
              f"({len(rec)} cases) - the true miss rate is higher")
        table(rec[:, 0], rec[:, 2], np.arange(-110, 111, 20), "longitude")
        lats = np.percentile(rec[:, 1], [0, 20, 40, 60, 80, 100]) if len(rec) else []
        table(rec[:, 1], rec[:, 2], np.round(lats, 0), "foot latitude (near side = more negative)")

    if a.players_on_pitch:
        print(f"\n3) HEAD-COUNT against {a.players_on_pitch} players on the pitch")
        step = a.block * 60
        t = T[0]
        while t < T[-1]:
            s = (T >= t) & (T < t + step)
            if s.sum() > 20:
                print(f"  {int((t - T[0]) // 60):3d}-{int(min(t + step - T[0], T[-1] - T[0]) // 60):<3d} min   "
                      f"found {N[s].mean():5.1f}   = {100 * N[s].mean() / a.players_on_pitch:4.0f}% of expected")
            t += step

    print("\nSPOT-CHECK these moments by eye (fewest players found relative to the typical count)")
    ref = a.players_on_pitch or float(np.percentile(N, 90))
    k = 3                                                  # smooth over about a second
    sm = np.convolve(N, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    order = np.argsort(sm)
    picks, taken = [], []
    for i in order:
        if all(abs(T[i] - u) > 60 for u in taken) and T[i] - T[0] > 5:
            picks.append(i); taken.append(T[i])
        if len(picks) == 12:
            break
    for i in sorted(picks, key=lambda j: T[j]):
        print(f"  video time {int(T[i] // 60):3d}:{int(T[i] % 60):02d}   found {sm[i]:4.1f} (typical {ref:.0f})")
    print("  Pause the ORIGINAL video there, count the players you can see, and compare. A dozen of these "
          "gives a direct recall figure.")


if __name__ == "__main__":
    main()
