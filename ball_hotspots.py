#!/usr/bin/env python3
"""
ball_hotspots.py — how many ball candidates sit in the same place frame after frame?

A real ball moves; a line mark, the penalty spot, a bag or a bright patch of grass does not.
Reads a detections file (v0.detections.jsonl from the job folder), bins every ball candidate
into 1x1 degree cells, and reports the cells that fire in many frames. With --out it writes a
copy of the file without the candidates in those cells.

    python ball_hotspots.py /path/to/job/v0.detections.jsonl [--min-frames 12] [--out cleaned.jsonl]

A cell is a hotspot when it holds a candidate in at least --min-frames frames
(default: 4 % of the frames, at least 12). Neighbouring cells are merged (a ball can sit on a cell edge).
"""
import argparse
import json
from collections import defaultdict


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("detections")
    p.add_argument("--min-frames", type=int, default=None)
    p.add_argument("--cell", type=float, default=1.0, help="cell size in degrees")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    lines = [json.loads(l) for l in open(a.detections)]
    head, recs = lines[0], [r for r in lines[1:] if "t" in r]
    n_frames = len(recs)
    need = a.min_frames or max(12, int(0.04 * n_frames))

    def cell(lon, lat):
        return int(lon // a.cell), int(lat // a.cell)

    frames_in = defaultdict(set)                               # cell -> frame indices with a candidate
    for k, r in enumerate(recs):
        for b in r.get("balls", []):
            cx, cy = cell(b[0], b[1])
            for dx in (-1, 0, 1):                              # count a candidate in the neighbours too
                for dy in (-1, 0, 1):
                    frames_in[(cx + dx, cy + dy)].add(k)
    hot = {c for c, f in frames_in.items() if len(f) >= need}
    # the cell itself (not its neighbours) is what gets removed: keep only cells that are hotspots on their own
    own = defaultdict(set)
    for k, r in enumerate(recs):
        for b in r.get("balls", []):
            own[cell(b[0], b[1])].add(k)
    hot = {c for c in hot if len(own.get(c, ())) >= need // 2}

    total = sum(len(r.get("balls", [])) for r in recs)
    gone = sum(1 for r in recs for b in r.get("balls", []) if cell(b[0], b[1]) in hot)
    frames_before = sum(1 for r in recs if r.get("balls"))
    frames_after = sum(1 for r in recs if any(cell(b[0], b[1]) not in hot for b in r.get("balls", [])))
    print(f"{n_frames} frames, {total} candidates, {frames_before} frames with one ({100 * frames_before / max(1, n_frames):.0f} %)")
    print(f"hotspot rule: a 1-degree cell with a candidate in >= {need} frames")
    for c in sorted(hot, key=lambda c: -len(own[c]))[:15]:
        print(f"  lon {c[0] * a.cell:7.1f}  lat {c[1] * a.cell:6.1f}   {len(own[c]):5d} frames")
    print(f"{len(hot)} hotspot cells hold {gone} of {total} candidates ({100 * gone / max(1, total):.0f} %)")
    print(f"frames with a candidate left: {frames_after} ({100 * frames_after / max(1, n_frames):.0f} %)")
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(json.dumps(head) + "\n")
            for r in recs:
                r = dict(r)
                r["balls"] = [b for b in r.get("balls", []) if cell(b[0], b[1]) not in hot]
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
