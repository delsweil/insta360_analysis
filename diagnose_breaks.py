#!/usr/bin/env python3
"""diagnose_breaks.py — why do the pieces end?  (read-only; needs track_stitch.py in the same folder)

After track_stitch's linking and stitching a player is still split into ~35 pieces per 10 minutes. This sorts every piece
end into one cause, by where on the pitch it happens (near / middle / far third from the camera):

  short      the piece lasts under 3 s: usually a false detection or a fragment, not a player who was followed
  edge       the player was within 4 m of the edge of the area where players are seen (left the view or the pitch)
  crowd      another player was within 3 m at the end (two players merged or swapped)
  contested  a continuation exists within reach, but the stitching gave it to another piece
  wide       the only continuation is further away than the gate, within twice the gate
  vanished   no piece starts within reach in the next 8 s: the detector lost him

  cd <your insta360_analysis folder>
  python3 diagnose_breaks.py /Users/davidelsweiler/FootballAutopan/TSV_Buch_A-Jugend/ --assign 6=B 5=REF

Takes the same options as track_stitch.py. It changes no file.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

try:
    import track_stitch as TS
except ImportError:
    sys.exit("Put diagnose_breaks.py in the same folder as track_stitch.py and run it from your insta360_analysis folder.")

CAUSES = ["short", "edge", "crowd", "contested", "wide", "vanished"]
CROWD_M = 3.0
EDGE_M = 4.0
SHORT_S = 3.0


def diagnose(fr, vt, x, y, lab, tid, max_gap=8.0, g0=2.5, gpm=0.06, gper_s=0.9):
    P = TS._pieces(fr, vt, x, y, lab, tid)
    ids = sorted(P, key=lambda k: P[k]["s"])
    n = len(ids)
    s_t = np.array([P[k]["s"] for k in ids]); e_t = np.array([P[k]["e"] for k in ids])
    s_p = np.array([P[k]["p0"] for k in ids]); e_p = np.array([P[k]["p1"] for k in ids])
    e_v = np.array([P[k]["v1"] for k in ids])
    gap = s_t[None, :] - e_t[:, None]
    ok = (gap > 0.1) & (gap <= max_gap)
    ex = np.minimum(gap, 2.5) * 0.7
    px = e_p[:, 0][:, None] + e_v[:, 0][:, None] * ex
    py = e_p[:, 1][:, None] + e_v[:, 1][:, None] * ex
    d = np.hypot(px - s_p[:, 0][None], py - s_p[:, 1][None])
    gate = g0 + gpm * 0.5 * (e_p[:, 1][:, None] + s_p[:, 1][None]) + gper_s * np.maximum(gap, 0)
    inside = ok & (d <= gate)
    wide = ok & (d <= 2 * gate)
    x_lo, x_hi = np.percentile(x, [1, 99]); y_lo, y_hi = np.percentile(y, [1, 99])
    y33, y66 = np.percentile(y, [33, 66])
    bounds = np.searchsorted(fr, np.arange(len(vt) + 1))
    part_end = float(vt[-1])
    out = []                                              # (third, cause)
    edges = []                                            # which edge an 'edge' end is at
    edge_info = []                                        # (edge name, end moment, x, y) for the off-pitch test
    for a, k in enumerate(ids):
        p = P[k]
        third = 0 if p["p1"][1] <= y33 else (2 if p["p1"][1] >= y66 else 1)
        if p["e"] >= part_end - 1.5:
            continue
        if p["e"] - p["s"] < SHORT_S:
            out.append((third, "short")); continue
        ex_, ey_ = p["p1"]
        if ex_ < x_lo + EDGE_M or ex_ > x_hi - EDGE_M or ey_ < y_lo + EDGE_M or ey_ > y_hi - EDGE_M:
            out.append((third, "edge"))
            dist = {"left end (x min)": ex_ - x_lo, "right end (x max)": x_hi - ex_, "near touchline (y min)": ey_ - y_lo, "far touchline (y max)": y_hi - ey_}
            edges.append(min(dist, key=dist.get))
            sdp = TS.side_of(lab[tid == k]); who = "player" if (sdp < 2).mean() >= 0.5 else "other/ref"
            edge_info.append((edges[-1], int(np.searchsorted(vt, p["e"])), ex_, ey_, who)); continue
        f_end = int(np.searchsorted(vt, p["e"]))
        near_other = False
        for f in (f_end - 1, f_end):
            if f < 0 or f >= len(vt):
                continue
            sl = slice(bounds[f], bounds[f + 1])
            m = tid[sl] != k
            if m.any() and (np.hypot(x[sl][m] - ex_, y[sl][m] - ey_) <= CROWD_M).any():
                near_other = True
        if near_other:
            out.append((third, "crowd")); continue
        if inside[a].any():
            out.append((third, "contested"))
        elif wide[a].any():
            out.append((third, "wide"))
        else:
            out.append((third, "vanished"))
    return out, edges, (x_lo, x_hi, y_lo, y_hi), edge_info


def off_points(det, kf, clusters, cal, vt, fr, x, y, tol=0.7):
    """positions (by moment) of the detections that the on-pitch flag removes: the points found when they are kept, minus those already used"""
    import auto_keyframes as AK
    orig = AK.load_detections
    def keep_all(path, max_lon=None):
        import json
        with open(path) as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
        frames = rows[1:]
        if max_lon is not None:
            for r in frames:
                r["players"] = [p for p in r["players"] if abs(p[0]) <= max_lon]
        return rows[0], frames
    AK.load_detections = keep_all
    try:
        vt2, fr2, x2, y2, lab2 = TS.part_points(det, kf, clusters, cal)
    finally:
        AK.load_detections = orig
    # moments are the same in both runs (same file); match by moment index through time
    idx = np.searchsorted(vt, vt2)
    idx = np.clip(idx, 0, len(vt) - 1)
    by_f = {}
    for f, px, py in zip(fr, x, y):
        by_f.setdefault(int(f), []).append((px, py))
    out = {}
    for f2, px, py in zip(fr2, x2, y2):
        f = int(np.argmin(np.abs(vt - vt2[f2])))
        if not any(np.hypot(px - qx, py - qy) <= tol for qx, qy in by_f.get(f, ())):
            out.setdefault(f, []).append((px, py))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--det"); ap.add_argument("--kf"); ap.add_argument("--labels"); ap.add_argument("--calib")
    ap.add_argument("--assign", nargs="+", metavar="GROUP=WHO")
    ap.add_argument("--edge-test", action="store_true", help="for the edge ends, check whether the player carries on as a detection that the on-pitch test removed (reads each part twice)")
    a = ap.parse_args()
    import team_labels as TL
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
    if a.det:
        dets = [(a.det, a.kf)]
    else:
        dets = TS.find_parts(a.folder)
    if not dets:
        sys.exit("No detections file found (*det*.jsonl); name a part with --det / --kf.")
    print("parts:", ", ".join(f"{os.path.basename(d)} (camera path: {os.path.basename(k) if k else 'none'})" for d, k in dets))
    cals = TS.find_outlines(a.folder, dets, a.calib)
    print("pitch outlines:", ", ".join(os.path.basename(c) if c else "none" for c in cals), "" if all(cals) else "  (people beside the pitch are not filtered where none)")
    allv, alle, ext = [], [], []
    edge_hits = {}
    for (d, k), cal in zip(dets, cals):
        print(f"reading {os.path.basename(d)} ...", flush=True)
        vt, fr, x, y, lab = TS.part_points(d, k, clusters, cal)
        tid = TS.stitch(fr, vt, x, y, lab, TS.link(fr, vt, x, y, lab))
        o, e, ex_, info = diagnose(fr, vt, x, y, lab, tid)
        allv += o; alle += e; ext.append(ex_)
        if a.edge_test:
            offs = off_points(d, k, clusters, cal, vt, fr, x, y)
            for name, f_end, px_, py_, who in info:
                hit = any(np.hypot(ox - px_, oy - py_) <= 3.0 for ff in range(f_end + 1, f_end + 7) for (ox, oy) in offs.get(ff, ()))
                edge_hits.setdefault(name, [0, 0, 0]); edge_hits[name][0] += 1; edge_hits[name][1] += int(hit); edge_hits[name][2] += int(who == "other/ref")
    tot = len(allv)
    print(f"\nWHY PIECES END ({tot} piece ends, all parts; third = where the player was, from the camera)")
    print(f"{'cause':10s} {'all':>7s} {'near':>7s} {'middle':>7s} {'far':>7s}")
    thirds = [sum(1 for t, _ in allv if t == i) for i in range(3)]
    for c in CAUSES:
        row = [sum(1 for t, cc in allv if cc == c and t == i) for i in range(3)]
        print(f"{c:10s} {100 * sum(row) / max(tot, 1):6.0f}% " + " ".join(f"{100 * r / max(th, 1):6.0f}%" for r, th in zip(row, thirds)))
    print(f"{'(ends)':10s} {tot:7d} " + " ".join(f"{th:7d}" for th in thirds))
    if alle:
        print(f"\nWHERE THE 'EDGE' ENDS ARE ({len(alle)}; area where players are seen: x {np.mean([e[0] for e in ext]):.0f}..{np.mean([e[1] for e in ext]):.0f} m, "
              f"y {np.mean([e[2] for e in ext]):.0f}..{np.mean([e[3] for e in ext]):.0f} m, 1st-99th percentile)")
        for k in sorted(set(alle), key=lambda k: -alle.count(k)):
            print(f"  {k:26s} {alle.count(k):5d}  {100 * alle.count(k) / len(alle):4.0f}%")
        print("  Touchline ends of a player walking along the line point to the on-pitch test; ends at the left / right end point to the goal areas or the camera view.")
    if edge_hits:
        print("\nDO THE EDGE ENDS CARRY ON OFF THE PITCH? (a detection the on-pitch test removed, within 3 m, in the next 2 s)")
        for k_, (n_, h_, o_) in sorted(edge_hits.items(), key=lambda kv: -kv[1][0]):
            print(f"  {k_:26s} {n_:5d} ends, {h_:5d} carry on off-pitch ({100 * h_ / max(n_, 1):.0f}%), {100 * o_ / max(n_, 1):.0f}% of the pieces are labelled other / referee, not a team")
        print("  A high share means the on-pitch test cuts players who walk along the line: keep them near the line (hysteresis) instead.")
    print("\nHow to read it: 'vanished' = detector loss (needs more detections per second or a better detector); 'crowd' and "
          "'contested' = ambiguity (a better linker, appearance); 'short' = junk pieces (filter or merge); 'edge' = leaves the view.")


if __name__ == "__main__":
    main()
