#!/usr/bin/env python3
"""duplicates_check.py — are some players counted twice?  (read-only; needs colour_refit.py and track_stitch.py in the same folder)

The team counts sometimes exceed 11 (Herren: team A in 21-35% of the moments). The pitch outline removed nothing, so the extra
players are not people beside the pitch. One more explanation: the detector sees the same player twice, for example in two
overlapping camera views, or as two boxes on one body.

For each part it prints
  near <=0.5 m / <=1 m / <=2 m   share of detections that have another detection in the same moment that close (any team)
  other view                     of the <=1 m pairs, the share whose two detections come from different camera views (overlap)
  A / B before, after            team counts per moment before and after dropping the weaker one of every pair closer than 1 m
  >11 before, after              share of moments in which a team has more than 11 players

Two real players are seldom within 1 m of each other, so a large share at <=0.5 m, or a drop in the '>11' share after dropping
duplicates, points to double counting.

  cd <your insta360_analysis folder>
  python3 duplicates_check.py "/Users/davidelsweiler/FootballAutopan/TSV_Johannes_83_Herren_1."
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

try:
    import team_labels as TL
    import track_stitch as TS
    import colour_refit as CR
except ImportError:
    sys.exit("Run this from your insta360_analysis folder, with track_stitch.py and colour_refit.py in it.")


def analyse(P, radius=1.0):
    fr, x, y, lab, conf, view = P["fr"], P["x"], P["y"], P["lab"], P["conf"], P["view"]
    n = len(fr); nf = len(P["vt"])
    bounds = np.searchsorted(fr, np.arange(nf + 1))
    near = {0.5: 0, 1.0: 0, 2.0: 0}
    pairs = diff_view = 0
    drop = np.zeros(n, bool)
    for f in range(nf):
        idx = np.arange(bounds[f], bounds[f + 1])
        if len(idx) < 2:
            continue
        d = np.hypot(x[idx][:, None] - x[idx][None], y[idx][:, None] - y[idx][None])
        np.fill_diagonal(d, 1e9)
        for r in near:
            near[r] += int((d.min(axis=1) <= r).sum())
        iu = np.argwhere(np.triu(d <= radius, 1))
        for a, b in iu:
            pairs += 1
            diff_view += int(view[idx[a]] != view[idx[b]])
        order = idx[np.argsort(-conf[idx], kind="stable")]              # strongest first; a weaker one within `radius` of a kept one is a duplicate
        kept = []
        for i in order:
            if kept and (np.hypot(x[kept] - x[i], y[kept] - y[i]) <= radius).any():
                drop[i] = True
            else:
                kept.append(i)
    return near, pairs, diff_view, drop


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--det"); ap.add_argument("--kf"); ap.add_argument("--labels"); ap.add_argument("--calib")
    ap.add_argument("--assign", nargs="+", metavar="GROUP=WHO")
    ap.add_argument("--radius", type=float, default=1.0, help="metres under which two detections count as one player (default 1)")
    a = ap.parse_args()
    lab_path = a.labels or next(iter(sorted(glob.glob(os.path.join(a.folder, "*team_labels.json")))), None)
    if not lab_path:
        sys.exit("No team labels file found (name it with --labels).")
    clusters = TL.Labelled(lab_path)
    for item in a.assign or []:
        g, who = item.split("="); g = int(g); who = who.upper()
        if who in ("A", "B", "OTHER"):
            clusters.group_team[g] = {"A": 0, "B": 1, "OTHER": -1}[who]; clusters.group_role[g] = -1
        elif who in ("REF", "GKA", "GKB"):
            clusters.group_team[g] = -1; clusters.group_role[g] = {"REF": 3, "GKA": 4, "GKB": 5}[who]; clusters.has_roles = True
    dets = [(a.det, a.kf)] if a.det else TS.find_parts(a.folder)
    if not dets:
        sys.exit("No detections file found (*det*.jsonl); name a part with --det / --kf.")
    cals = TS.find_outlines(a.folder, dets, a.calib)
    print(f"\n{'part':>4s} | {'<=0.5 m':>8s} {'<=1 m':>7s} {'<=2 m':>7s} {'other view':>11s} | {'A before':>8s} {'B before':>8s} {'>11':>5s} | {'A after':>7s} {'B after':>7s} {'>11':>5s} | {'dropped':>8s}")
    for i, (d, k_) in enumerate(dets):
        try:
            P = CR.points_ex(d, k_, clusters, cals[i])
        except (TypeError, ValueError, KeyError, OSError) as e:
            print(f"  ! outline not used ({type(e).__name__}: {e})")
            P = CR.points_ex(d, k_, clusters, None)
        near, pairs, dv, drop = analyse(P, a.radius)
        n = len(P["fr"])
        c0, _ = CR.counts(P, P["lab"])
        keep = ~drop
        P2 = dict(P); 
        fr2 = P["fr"][keep]
        cnt = np.zeros((len(P["vt"]), 2))
        for s_ in (0, 1):
            sel = keep & (TS.side_of(P["lab"]) == s_)
            np.add.at(cnt[:, s_], P["fr"][sel], 1)
        o0 = 100 * float(((c0[:, 0] > 11) | (c0[:, 1] > 11)).mean()); o1 = 100 * float(((cnt[:, 0] > 11) | (cnt[:, 1] > 11)).mean())
        print(f"{i:4d} | {100 * near[0.5] / n:7.1f}% {100 * near[1.0] / n:6.1f}% {100 * near[2.0] / n:6.1f}% {100 * dv / max(pairs, 1):10.0f}% | "
              f"{c0[:, 0].mean():8.1f} {c0[:, 1].mean():8.1f} {o0:4.0f}% | {cnt[:, 0].mean():7.1f} {cnt[:, 1].mean():7.1f} {o1:4.0f}% | {100 * drop.mean():7.1f}%")
    print("\nHow to read it: two real players are seldom within 1 m. A share of a few percent at <=0.5 m, many pairs from different\n"
          "camera views, and a clear drop of the '>11' share after dropping duplicates, mean players are counted twice.")


if __name__ == "__main__":
    main()
