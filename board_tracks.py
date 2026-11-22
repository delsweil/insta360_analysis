#!/usr/bin/env python3
"""board_tracks.py — pieces of players and candidate links, for the "This is me" feature of the web player.

The board file (`<game>_board.json`) has every player at every moment but no identities. This step follows the players from one
moment to the next (motion model, in metres) and writes `<game>_tracks.json` next to the board:

  pieces   one per stretch that was followed without a break: [id, team, first moment, last moment]
  fp       for every moment of the board file, the piece id of each player in it, in the same order as the board's players
  links    for every piece end (and start) the pieces that could be its continuation (predecessor), with the chance of each

The web player lets a player tap himself on the board at any moment; the tracker then follows his piece, and asks a question
wherever it is not sure which piece comes next. Nothing in this file names a person.

  cd <your insta360_analysis folder>
  python3 board_tracks.py /path/to/game_board.json            # writes /path/to/game_tracks.json
  python3 board_tracks.py game_board.json --check              # also prints how many questions a followed player would get

Upload `<game>_tracks.json` next to the board file in the Supabase storage bucket `boards` as `<game id>.tracks.json`
(the web player offers "Upload tracks" to the coach when the board exists but the tracks file does not).

Needs track_stitch.py (the linker) in the same folder.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import track_stitch as TS
except ImportError:
    sys.exit("Put board_tracks.py in your insta360_analysis folder, next to track_stitch.py.")

FORMAT = "football-autopan-tracks"
VERSION = 1
TRACKER = "link-2: motion model in metres, colour-aware, with stitching"
PART_GAP_S = 10.0          # no moments for this long = a new part: pieces are never joined across it
WINDOW_S = 10.0           # how long after a piece ends a continuation may start
SURE = 0.80                # best candidate at least this likely, and close enough = the tracker links by itself
SIDE = {0: 0, 4: 0, 1: 1, 5: 1}


def load(path: str):
    raw = json.loads(Path(path).read_text())
    if raw.get("format") != "football-autopan-board":
        sys.exit("That is not a board file (<game>_board.json from the analysis app).")
    frames = raw["frames"]
    t = np.array([float(f[0]) for f in frames])
    fr, x, y, team, k_in = [], [], [], [], []
    for i, f in enumerate(frames):
        for k in range(3, len(f) - 2, 3):
            fr.append(i); x.append(f[k] / 10.0); y.append(f[k + 1] / 10.0); team.append(int(f[k + 2])); k_in.append((k - 3) // 3)
    return raw, t, np.array(fr, int), np.array(x), np.array(y), np.array(team, int), np.array(k_in, int)


def build(t, fr, x, y, team):
    """-> piece id per player (renumbered by start), pieces [(id, side, f0, f1)]"""
    order = np.argsort(t, kind="stable")                  # the board may list the parts out of order; linking needs time order
    rank = np.empty(len(t), int); rank[order] = np.arange(len(t))
    vt = t[order]
    f2 = rank[fr]
    o = np.lexsort((np.arange(len(f2)), f2))
    tid_s = np.full(len(fr), -1, int)
    t0_ = TS.link(f2[o], vt, x[o], y[o], team[o])
    tid_s[o] = TS.stitch(f2[o], vt, x[o], y[o], team[o], t0_)       # joins pieces across gaps where one continuation is clear
    # renumber by first appearance in time
    first = {}
    for idx in o:
        k = int(tid_s[idx])
        if k not in first:
            first[k] = len(first)
    pid = np.array([first[int(k)] for k in tid_s], int)
    pieces = []
    for k in range(len(first)):
        m = np.flatnonzero(pid == k)
        fs = f2[m]
        votes = np.bincount([SIDE.get(int(v), 2) if v != 3 else 3 for v in team[m]], minlength=4)
        side = int(votes.argmax())
        pieces.append((k, side, int(fr[m][np.argmin(fs)]), int(fr[m][np.argmax(fs)])))
    return pid, pieces


def candidates(pieces, t, fr, x, y, pid, window=WINDOW_S):
    """for every piece: the pieces that could continue it (next) and could have come before it (prev)"""
    n = len(pieces)
    s_t = np.array([t[p[2]] for p in pieces]); e_t = np.array([t[p[3]] for p in pieces])
    first_idx, last_idx = {}, {}
    for i in np.argsort(fr, kind="stable"):
        k = int(pid[i])
        if k not in first_idx:
            first_idx[k] = i
        last_idx[k] = i
    p0 = np.array([[x[first_idx[k]], y[first_idx[k]]] for k in range(n)]); p1 = np.array([[x[last_idx[k]], y[last_idx[k]]] for k in range(n)])
    def vel(k, at_end):
        idx = np.flatnonzero(pid == k); idx = idx[np.argsort(fr[idx])]
        if len(idx) < 2:
            return 0.0, 0.0
        a, b = (idx[-1 - min(3, len(idx) - 1)], idx[-1]) if at_end else (idx[0], idx[min(3, len(idx) - 1)])
        dt = t[fr[b]] - t[fr[a]]
        return ((x[b] - x[a]) / dt, (y[b] - y[a]) / dt) if dt > 0 else (0.0, 0.0)
    v1 = [vel(k, True) for k in range(n)]; v0 = [vel(k, False) for k in range(n)]
    side = np.array([p[1] for p in pieces])
    out_next, out_prev = {}, {}
    for a in range(n):
        if side[a] > 1:
            continue
        cand = []
        for b in range(n):
            if b == a or side[b] != side[a]:
                continue
            gap = s_t[b] - e_t[a]
            if gap < -0.4 or gap > window:
                continue
            gap = max(gap, 0.0)
            pred = (p1[a][0] + v1[a][0] * gap * 0.8, p1[a][1] + v1[a][1] * gap * 0.8)
            d = math.hypot(p0[b][0] - pred[0], p0[b][1] - pred[1])
            if d <= 4 + 7 * gap:
                cand.append([b, d, gap, math.exp(-d / (2 + 1.5 * gap))])
        if cand:
            tot = sum(c[3] for c in cand)
            cand.sort(key=lambda c: -c[3])
            out_next[a] = [[c[0], round(c[3] / tot, 3), round(c[1], 1), round(c[2], 1)] for c in cand[:4]]
    for a, lst in out_next.items():                                # the predecessors are the same pairs seen from the other end
        for b, p, d, gap in lst:
            out_prev.setdefault(b, []).append([a, p, d, gap])
    for b in out_prev:                                            # normalise the chance among the possible predecessors
        lst = out_prev[b]
        tot = sum(math.exp(-c[2] / (2 + 1.5 * c[3])) for c in lst)
        for c in lst:
            c[1] = round(math.exp(-c[2] / (2 + 1.5 * c[3])) / tot, 3)
        lst.sort(key=lambda c: -c[1]); out_prev[b] = lst[:4]
    return out_next, out_prev


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("board")
    ap.add_argument("--out")
    ap.add_argument("--check", action="store_true", help="print how long pieces are and how many links are sure")
    a = ap.parse_args()
    raw, t, fr, x, y, team, k_in = load(a.board)
    pid, pieces = build(t, fr, x, y, team)
    nxt, prv = candidates(pieces, t, fr, x, y, pid)
    # piece id of every player of every moment, in the order of the board's players
    fp = [[] for _ in range(len(t))]
    for i in np.lexsort((k_in, fr)):
        fp[int(fr[i])].append(int(pid[i]))
    def link_rec(d):
        rec = {}
        for k, lst in d.items():
            top = lst[0]
            sure = bool(top[1] >= SURE and top[2] <= 3 + 2 * top[3])
            rec[str(k)] = {"c": lst, "s": 1 if sure else 0}
        return rec
    out = {"format": FORMAT, "version": VERSION, "tracker": TRACKER, "board_title": raw.get("title", ""),
           "frames": len(t), "pieces": [list(p) for p in pieces], "fp": fp, "next": link_rec(nxt), "prev": link_rec(prv)}
    path = Path(a.out) if a.out else Path(a.board).with_name(Path(a.board).name.replace("_board.json", "_tracks.json").replace(".json", "") + ("" if a.board.endswith("_board.json") else "_tracks") + ".json")
    if a.out is None and a.board.endswith("_board.json"):
        path = Path(a.board[: -len("_board.json")] + "_tracks.json")
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB): {len(pieces)} pieces, {len(t)} moments")
    if a.check:
        dur = np.array([t[p[3]] - t[p[2]] for p in pieces if p[1] < 2])
        sure = sum(r["s"] for r in out["next"].values()); tot = len(out["next"])
        print(f"  team pieces: median {np.median(dur):.1f} s, {len(dur)} pieces; links sure {sure} of {tot} ({100 * sure / max(tot, 1):.0f}%)")


if __name__ == "__main__":
    main()
