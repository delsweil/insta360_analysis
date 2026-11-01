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


def hotspot_cells(recs, cell: float = 1.0, need: int = None, bin_s: float = 30.0, spread: float = 0.4):
    """-> (set of (ix, iy) cells, need). A cell is a hotspot when a ball candidate sits in it in at least `need` frames
    (default 4 % of the frames, at least 12) AND it keeps doing so through the half: in at least `spread` (40 %) of the
    `bin_s`-second windows. A mark, a spare ball or a bag is there all game; a real ball resting on the centre spot or a
    corner is there for a few kick-offs and set pieces, in a handful of windows, and must stay."""
    need = need or max(12, int(0.04 * len(recs)))
    ts = [r.get("t", k / 3.0) for k, r in enumerate(recs)]
    t0 = min(ts) if ts else 0.0
    n_bins = max(1, int((max(ts) - t0) // bin_s) + 1) if ts else 1
    min_bins = max(2, int(spread * n_bins))
    key = lambda lon, lat: (int(lon // cell), int(lat // cell))
    near, own = defaultdict(set), defaultdict(set)
    for k, r in enumerate(recs):
        for b in r.get("balls", []):
            cx, cy = key(b[0], b[1])
            own[(cx, cy)].add(k)
            for dx in (-1, 0, 1):                              # a ball on a cell edge counts for the neighbours too
                for dy in (-1, 0, 1):
                    near[(cx + dx, cy + dy)].add(k)
    bins = lambda frames: {int((ts[k] - t0) // bin_s) for k in frames}
    hot = {c for c, f in near.items()
           if len(f) >= need and len(own.get(c, ())) >= need // 2 and len(bins(own.get(c, ()))) >= min_bins}
    return hot, need


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
    hot, need = hotspot_cells(recs, a.cell, a.min_frames)

    def cell(lon, lat):
        return int(lon // a.cell), int(lat // a.cell)

    own = defaultdict(set)
    for k, r in enumerate(recs):
        for bb in r.get("balls", []):
            own[cell(bb[0], bb[1])].add(k)

    total = sum(len(r.get("balls", [])) for r in recs)
    gone = sum(1 for r in recs for b in r.get("balls", []) if cell(b[0], b[1]) in hot)
    frames_before = sum(1 for r in recs if r.get("balls"))
    frames_after = sum(1 for r in recs if any(cell(b[0], b[1]) not in hot for b in r.get("balls", [])))
    print(f"{n_frames} frames, {total} candidates, {frames_before} frames with one ({100 * frames_before / max(1, n_frames):.0f} %)")
    print(f"hotspot rule: a 1-degree cell with a candidate in >= {need} frames, spread over the half (not just a few kick-offs)")
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
