#!/usr/bin/env python3
"""board_probe.py — why is a player (not) on the board at a given moment?

Prints every detection of one moment of a video with the reasons the board keeps or drops it: confidence, the old outline
rule (a fixed angle outside the outline) and the board's rule (metres outside the outline), team, and the position on the board.

  python board_probe.py --detections v0.detections.jsonl --calib pitch0.json --team-labels teams.team_labels.json \
      --kf v0.auto.keyframes.json --at 1602          # 26:42 in the video of this part

  --at is the time in the finished video of this part (seconds); with --source-time it is the time in the raw footage.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

import auto_keyframes as AK
import scene_check as S


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detections", required=True, help="the RAW file v*.detections.jsonl (with every detection)")
    ap.add_argument("--calib", default=None, help="pitch outline (pitch0.json or pitch0.manual.json)")
    ap.add_argument("--team-labels", default=None)
    ap.add_argument("--kf", default=None, help="camera path of this video (tells where the finished video starts)")
    ap.add_argument("--at", type=float, required=True)
    ap.add_argument("--source-time", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--min-height", type=float, default=14.0)
    ap.add_argument("--margin-m", type=float, default=1.5)
    a = ap.parse_args(argv)

    with open(a.detections) as fh:
        hdr = json.loads(fh.readline())
        rows = [json.loads(l) for l in fh if l.strip()]
    t_in = float(hdr.get("start", 0.0))
    if a.kf and not a.source_time:
        import kf_core as K
        meta = K.load_meta(a.kf)
        if meta.get("trim_in") is not None:
            t_in = float(meta["trim_in"])
    t = a.at if a.source_time else a.at + t_in
    fr = min(rows, key=lambda r: abs(r["t"] - t))
    print(f"moment at source time {fr['t']:.2f} s (asked {t:.2f} s); {len(fr['players'])} detections in the file")

    fy = S.view_fy(hdr)
    views = np.array(hdr["view_lons"], float)
    tester = None
    if a.calib:
        import pitch_filter as PF
        tester = PF.FootTester(a.calib, margin_m=a.margin_m, fy=fy)
    cl = None
    if a.team_labels:
        import team_labels as TL
        cl = TL.Labelled(a.team_labels)

    print(f"{'lon':>7} {'foot':>6} {'box h':>6} {'conf':>5} {'x m':>6} {'y m':>6} {'outside':>8} {'allowed':>8}  team       verdict")
    for p in sorted(fr["players"], key=lambda q: -q[2]):
        centre = float(views[np.argmin(np.abs(views - p[0]))])
        x, y = S.ground_position(p[0], p[2], centre, fy, 0.75)
        out_px = allow = None
        why = []
        if abs(p[0]) > 98:
            why.append("beyond the lens range")
        if p[3] < a.min_conf:
            why.append(f"confidence < {a.min_conf}")
        if tester is not None:
            out_px, allow = tester.distance_px(p[0], p[1]), tester.allowed_px(p[2])
            if out_px > allow:
                why.append("outside the outline")
        team = "-"
        if cl is not None and len(p) > 5 and p[5] is not None and p[2] >= a.min_height:
            tm = int(cl.team_of_rescue(np.array([p[5]], float), 1.5)[0])
            team = cl.names[tm] if tm >= 0 else "unknown"
        elif cl is not None:
            team = "too small" if len(p) > 5 and p[5] is not None else "no colour"
        old = ""
        if tester is not None and out_px is not None and out_px > 25:
            old = "  (the old fixed-angle rule drops it too)" if out_px > allow else ""
        o = "-" if out_px is None else f"{out_px:.0f}px"
        al = "-" if allow is None else f"{allow:.0f}px"
        print(f"{p[0]:7.1f} {p[1]:6.1f} {p[2]:6.0f} {p[3]:5.2f} {x:6.1f} {y:6.1f} {o:>8} {al:>8}  {team:<10} "
              f"{'DROPPED: ' + ', '.join(why) if why else 'shown'}{old}")
    if cl is not None:
        import json as _j
        with open(a.team_labels) as fh:
            m = _j.load(fh)
        print("\ncolour groups: no, share, team, mean colour (B,G,R)")
        for j in range(len(cl.centres)):
            tm = int(cl.group_team[j])
            print(f"  g{j}: {m['sizes'][j]:>6}  {cl.names[tm] if tm >= 0 else 'other':<10} "
                  f"{[round(v) for v in m['mean_bgr'][j]]}")
        print("\nthe largest boxes: shirt colour (B,G,R), nearest groups (distance)")
        for p in sorted(fr["players"], key=lambda q: -q[2])[:5]:
            if len(p) > 5 and p[5] is not None:
                f = S.colour_features(np.array([p[5]], float))
                d = np.sqrt(((f[:, None, :] - cl.centres[None]) ** 2).sum(-1))[0]
                o = np.argsort(d)[:3]
                print(f"  box h {p[2]:.0f} at lon {p[0]:.1f}: colour {[round(v) for v in p[5]]}  " +
                      "  ".join(f"g{j} {d[j]:.1f}" for j in o))
    return 0


if __name__ == "__main__":
    sys.exit(main())
