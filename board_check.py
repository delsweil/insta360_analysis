#!/usr/bin/env python3
"""board_check.py — how many players does the detector find, and where is it blind?

Reads a detections file (v*.detections.jsonl) and prints, per band of distance from the camera, how many players are
found per moment at several confidence thresholds. With --team-labels it also shows how the found players split into the
two teams and 'unknown' (shirt colour not recognised), per band. It changes nothing: use it to choose the thresholds of
the tactical board (board_data.py --min-conf / --min-height) before touching anything else.

  python board_check.py --detections v0.detections.jsonl [--team-labels teams.team_labels.json]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import auto_keyframes as AK
import scene_check as S

BANDS = [("near (< 30 m)", 0, 30), ("middle (30-55 m)", 30, 55), ("far (> 55 m)", 55, 1e9)]


def scan(path: str, max_lon: float = 98.0, k_angle: float = 0.75):
    hdr, frames = AK.load_detections(path, max_lon=max_lon)
    fy = S.SQUARE_FY if hdr.get("square_views") else S.LEGACY_FY
    views = np.array(hdr["view_lons"], float)
    rows = []                                    # moment, distance, conf, box height, b, g, r (-1 = none)
    for i, r in enumerate(frames):
        for p in r["players"]:
            centre = float(views[np.argmin(np.abs(views - p[0]))])
            _, y = S.ground_position(p[0], p[2], centre, fy, k_angle)
            col = p[5] if len(p) > 5 and p[5] is not None else (-1, -1, -1)
            rows.append((i, y, p[3], p[2], *col))
    return len(frames), np.array(rows, float).reshape(-1, 7)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--team-labels", default=None)
    ap.add_argument("--min-height", type=float, default=14.0)
    a = ap.parse_args(argv)
    n, rows = scan(a.detections)
    if not len(rows):
        raise SystemExit("no players in this file")
    print(f"{n} moments, {len(rows)} detections (confidence >= {rows[:, 2].min():.2f} in the file)\n")
    print("players found per moment (average), by distance band and confidence threshold:")
    thr = [0.2, 0.3, 0.4, 0.5]
    print(f"  {'':18}" + "".join(f"{'>= ' + str(t):>9}" for t in thr))
    for name, lo, hi in BANDS:
        sel = (rows[:, 1] >= lo) & (rows[:, 1] < hi)
        print(f"  {name:18}" + "".join(f"{((rows[:, 2] >= t) & sel).sum() / n:9.2f}" for t in thr))
    print(f"  {'all':18}" + "".join(f"{(rows[:, 2] >= t).sum() / n:9.2f}" for t in thr))
    print("\n(a real game has about 22 players + referee: if the far band is far below the near band, "
          "the detector is blind there, not the thresholds)")
    if a.team_labels:
        import team_labels as TL
        cl = TL.Labelled(a.team_labels)
        ok = (rows[:, 4] >= 0) & (rows[:, 3] >= a.min_height) & (rows[:, 2] >= 0.3)
        team = np.full(len(rows), -2)                    # -2 = box too small to trust the colour
        team[ok] = cl.team_of(rows[ok, 4:7])
        print(f"\nshirt colour of the detections with confidence >= 0.3 (share per band; unknown = no team, e.g. referee):")
        print(f"  {'':18}{cl.names[0][:10]:>11}{cl.names[1][:10]:>11}{'unknown':>11}{'too small':>11}")
        for name, lo, hi in BANDS:
            sel = (rows[:, 1] >= lo) & (rows[:, 1] < hi) & (rows[:, 2] >= 0.3)
            m = max(1, sel.sum())
            print(f"  {name:18}" + "".join(f"{(sel & (team == v)).sum() / m * 100:10.0f}%" for v in (0, 1, -1, -2)))
        # which colour group takes the players? (a group that is neither team swallows the shirts seen small and washed out)
        import json
        with open(a.team_labels) as fh:
            meta = json.load(fh)
        grp = np.full(len(rows), -1)
        grp[ok] = cl.groups_of(rows[ok, 4:7])
        print("\nnearest colour group of each detection (confidence >= 0.3, colour trusted), share per band:")
        print(f"  {'group':>5} {'colour':>8} {'belongs to':>11}" + "".join(f"{n.split(' (')[0]:>9}" for n, _, _ in BANDS))
        for j in range(len(cl.centres)):
            mb = meta.get("mean_bgr", [[0, 0, 0]] * len(cl.centres))[j]
            hexc = "#%02x%02x%02x" % (int(mb[2]), int(mb[1]), int(mb[0]))
            who = {0: cl.names[0][:10], 1: cl.names[1][:10], -1: "other"}[int(cl.group_team[j])]
            cells = ""
            for _, lo, hi in BANDS:
                sel = (rows[:, 1] >= lo) & (rows[:, 1] < hi) & (rows[:, 2] >= 0.3) & ok
                cells += f"{(sel & (grp == j)).sum() / max(1, sel.sum()) * 100:8.0f}%"
            print(f"  {j:>5} {hexc:>8} {who:>11}{cells}")
        print("\nif a close-by colour may still count for a team (board_data --rescue-ratio), unknown shrinks to:")
        print(f"  {'ratio':>6}" + "".join(f"{n.split(' (')[0]:>9}" for n, _, _ in BANDS))
        for ratio in (1.0, 1.5, 2.0):
            t2 = np.full(len(rows), -2)
            t2[ok] = cl.team_of_rescue(rows[ok, 4:7], ratio)
            cells = ""
            for _, lo, hi in BANDS:
                sel = (rows[:, 1] >= lo) & (rows[:, 1] < hi) & (rows[:, 2] >= 0.3) & ok
                cells += f"{(sel & (t2 == -1)).sum() / max(1, sel.sum()) * 100:8.0f}%"
            print(f"  {ratio:>6.1f}{cells}   (unknown)")
        print("  ... and the two teams stay about level in each band when the rescue is right")
    return 0


if __name__ == "__main__":
    sys.exit(main())
