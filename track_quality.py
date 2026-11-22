#!/usr/bin/env python3
"""track_quality.py — how well can players be followed through a game?  (read-only; changes nothing)

Answers, for one game, the questions behind the "My highlights" idea:

  1. How long are the pieces a player can be followed without a break?          (track lengths)
  2. When a piece ends, can the tracker tell where the player went on?          (links: sure / unsure / none)
  3. If a player claimed himself, how many questions would he get, and how much of the game would he be followed?
  4. How often is the ball seen at all?                                         (the other half of "near the ball")

It reads the board file the analysis app writes (`<game>_board.json`, see board_data.py): positions of every player
about 3 times a second, no player ids. Pieces ("tracks") are rebuilt with the same board_clean.link_tracks the board uses.

  cd <your insta360_analysis folder>
  python track_quality.py ~/path/to/game_board.json
  python track_quality.py game_board.json --out report.json --walks 300

Read the numbers with two cautions:
  * The board file is already cleaned: board_data fills short holes and smooths positions. Real breaks in the detections
    are therefore at least as many as shown here. The report is a best case.
  * Positions are approximate (box height as a ruler), most of all on the far side of the pitch.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import board_clean as BC                      # the repo's own track linker; run this script from the repo folder
except ImportError:
    sys.exit("Run this from your insta360_analysis folder (board_clean.py must be importable).")

PART_GAP_S = 10.0          # no moments for this long = a new part (half): tracks are never linked across it
WINDOW_S = 6.0             # how long after a track ends a continuation may start
SURE = 0.90                # best candidate at least this likely = the tracker links by itself (same as the prototype)


# ───────────── reading the board file ─────────────

def load_board(path: str):
    raw = json.loads(Path(path).read_text())
    if raw.get("format") != "football-autopan-board":
        sys.exit("That is not a board file (<game>_board.json from the analysis app).")
    frames = raw["frames"]
    ball = raw.get("ball")
    t, fr, xs, ys, tm = [], [], [], [], []
    for i, f in enumerate(frames):
        t.append(float(f[0]))
        for k in range(3, len(f) - 2, 3):
            team = int(f[k + 2])
            team = {0: 0, 4: 0, 1: 1, 5: 1}.get(team, -1)     # A / A's keeper, B / B's keeper; others are left out
            if team < 0:
                continue
            fr.append(i); xs.append(f[k] / 10.0); ys.append(f[k + 1] / 10.0); tm.append(team)   # decimetres -> metres
    order = np.argsort(np.array(t), kind="stable")           # moments in time order
    rank = np.empty(len(t), int); rank[order] = np.arange(len(t))
    vt = np.array(t)[order]
    frame = rank[np.array(fr, int)]
    o2 = np.lexsort((np.arange(len(frame)), frame))
    ball_s = None
    if ball is not None:
        ball_s = [ball[i] for i in order]
    return (vt, frame[o2], np.array(xs)[o2], np.array(ys)[o2], np.array(tm)[o2], ball_s)


def part_ids(vt: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(np.diff(vt) > PART_GAP_S)]).astype(int)


# ───────────── tracks ─────────────

class Track:
    __slots__ = ("id", "team", "s", "e", "n", "p0", "p1", "v0", "v1", "part", "conf")


def build_tracks(vt, frame, x, y, team):
    tid = BC.link_tracks(frame, x, y, vt)
    parts = part_ids(vt)
    tracks = []
    for k in np.unique(tid[tid >= 0]):
        idx = np.flatnonzero(tid == k)
        idx = idx[np.argsort(frame[idx])]
        if len(idx) == 0:
            continue
        T = Track(); T.id = int(k)
        T.team = int(round(float(team[idx].mean())))           # majority team
        T.conf = float(max((team[idx] == T.team).mean(), 1 - (team[idx] == T.team).mean()))
        tt = vt[frame[idx]]
        T.s, T.e, T.n = float(tt[0]), float(tt[-1]), len(idx)
        T.p0, T.p1 = (float(x[idx[0]]), float(y[idx[0]])), (float(x[idx[-1]]), float(y[idx[-1]]))
        m = min(len(idx) - 1, 3)                                # speed from the first / last few moments
        def vel(a, b):
            dt = tt[b] - tt[a]
            return (0.0, 0.0) if dt <= 0 else ((x[idx[b]] - x[idx[a]]) / dt, (y[idx[b]] - y[idx[a]]) / dt)
        T.v0 = vel(0, m); T.v1 = vel(len(idx) - 1 - m, len(idx) - 1)
        T.part = int(parts[frame[idx[0]]])
        tracks.append(T)
    tracks.sort(key=lambda t: t.s)
    return tracks


def candidates(T: Track, tracks, direction: int, part_end: dict, part_start: dict):
    """Pieces that could be the continuation (direction +1: after T ends) or the predecessor (-1: before T starts).
    Same rule as the prototype: same team, starts / ends inside a short window, within reach of where the player
    should be. Returns (list of (track, distance, prob), status) where status is 'edge' at the start / end of a part."""
    if direction > 0:
        at, pos, v = T.e, T.p1, T.v1
        if at >= part_end[T.part] - 1.5:
            return [], "edge"
    else:
        at, pos, v = T.s, T.p0, T.v0
        if at <= part_start[T.part] + 1.5:
            return [], "edge"
    out = []
    for h in tracks:
        if h is T or h.team != T.team or h.part != T.part:
            continue
        t_h = h.s if direction > 0 else h.e
        gap = (t_h - at) * direction                          # seconds between the two pieces
        if gap < -0.4 or gap > WINDOW_S:
            continue
        gap = max(gap, 0.0)
        pred = (pos[0] + v[0] * gap * 0.8 * direction, pos[1] + v[1] * gap * 0.8 * direction)
        q = h.p0 if direction > 0 else h.p1
        d = math.hypot(q[0] - pred[0], q[1] - pred[1])
        if d <= 4 + 7 * gap:
            out.append([h, d, gap])
    for c in out:
        c.append(math.exp(-c[1] / (2 + 1.5 * c[2])))
    s = sum(c[3] for c in out) or 1.0
    res = sorted(((c[0], c[1], c[3] / s) for c in out), key=lambda c: -c[2])
    return res, ("none" if not res else ("sure" if res[0][2] >= SURE else "unsure"))


# ───────────── the report ─────────────

def pct(a, b):
    return 100.0 * a / b if b else 0.0


def main():
    global WINDOW_S
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("board", help="<game>_board.json")
    ap.add_argument("--out", help="also write the numbers as JSON")
    ap.add_argument("--walks", type=int, default=200, help="how many players to simulate following (default 200)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--window", type=float, default=WINDOW_S, help="how long after a piece ends a continuation may start (s, default 6)")
    ap.add_argument("--min-track", type=float, default=10.0, help="a player is picked from pieces at least this long (s)")
    a = ap.parse_args()
    WINDOW_S = a.window

    vt, frame, x, y, team, ball = load_board(a.board)
    parts = part_ids(vt)
    part_start = {p: float(vt[parts == p][0]) for p in np.unique(parts)}
    part_end = {p: float(vt[parts == p][-1]) for p in np.unique(parts)}
    play_s = sum(part_end[p] - part_start[p] for p in part_start)
    nm = len(vt)
    print(f"\n{Path(a.board).name}: {nm} moments in {len(part_start)} part(s), {play_s/60:.1f} min, {len(frame)} player positions (teams A/B only)")

    rep: dict = {"file": Path(a.board).name, "minutes": round(play_s / 60, 1), "parts": len(part_start)}

    # 1. coverage: how many players are seen
    cnt = np.zeros((nm, 2), int)
    np.add.at(cnt, (frame, team), 1)
    far = y > np.percentile(y, 66)
    cf = np.zeros(nm, int); np.add.at(cf, frame[far], 1)
    cn = np.zeros(nm, int); np.add.at(cn, frame[y < np.percentile(y, 33)], 1)
    print("\n1. WHO IS SEEN")
    for k, nme in enumerate("AB"):
        print(f"   team {nme}: {cnt[:, k].mean():.1f} players per moment on average, {pct((cnt[:, k] >= 10).sum(), nm):.0f}% of moments with 10 or 11")
    print(f"   per moment, nearest third of the pitch: {cn.mean():.1f} players · farthest third: {cf.mean():.1f} players")
    rep["coverage"] = {"A": round(float(cnt[:, 0].mean()), 1), "B": round(float(cnt[:, 1].mean()), 1),
                       "near_third": round(float(cn.mean()), 1), "far_third": round(float(cf.mean()), 1)}

    # 2. pieces
    tracks = build_tracks(vt, frame, x, y, team)
    dur = np.array([t.e - t.s for t in tracks])
    tot = float(dur.sum()) or 1.0
    mixed = sum(1 for t in tracks if t.conf < 0.9 and t.n >= 6)
    print(f"\n2. HOW LONG CAN A PLAYER BE FOLLOWED WITHOUT A BREAK")
    print(f"   {len(tracks)} pieces · median {np.median(dur):.0f} s · 90th percentile {np.percentile(dur, 90):.0f} s")
    for lim in (10, 30, 60, 120):
        print(f"   {pct(dur[dur >= lim].sum(), tot):.0f}% of followed time lies in pieces of at least {lim} s")
    print(f"   pieces with mixed teams (a swap between teams inside one piece): {mixed}")
    per = len(tracks) / max(1e-9, 22 * play_s / 600.0)
    print(f"   about {per:.1f} pieces per player per 10 minutes")
    rep["pieces"] = {"count": len(tracks), "median_s": round(float(np.median(dur)), 1), "p90_s": round(float(np.percentile(dur, 90)), 1),
                     "per_player_per_10min": round(per, 1), "mixed_team": mixed,
                     "share_in_pieces_ge_30s": round(pct(dur[dur >= 30].sum(), tot), 1)}

    # 3. links
    stat = {"sure": 0, "unsure": 0, "none": 0, "edge": 0}
    ncand = []
    for T in tracks:
        cands, st = candidates(T, tracks, +1, part_end, part_start)
        stat[st] += 1
        if st in ("sure", "unsure"):
            ncand.append(len(cands))
    ends = stat["sure"] + stat["unsure"] + stat["none"]
    print("\n3. WHEN A PIECE ENDS, CAN THE TRACKER TELL WHERE THE PLAYER WENT")
    print(f"   {ends} piece ends inside the game")
    print(f"   links itself, sure (best guess ≥ {int(SURE*100)}%):  {stat['sure']:5d}  {pct(stat['sure'], ends):4.0f}%")
    print(f"   needs a question (several possible):   {stat['unsure']:5d}  {pct(stat['unsure'], ends):4.0f}%")
    print(f"   no continuation found (lost for good): {stat['none']:5d}  {pct(stat['none'], ends):4.0f}%")
    if ncand:
        print(f"   when there are candidates: {np.mean(ncand):.1f} on average")
    rep["links"] = {"ends": ends, **{k: stat[k] for k in ("sure", "unsure", "none")}}

    # 4. following a player
    rng = np.random.default_rng(a.seed)
    pool = [t for t in tracks if t.e - t.s >= a.min_track]
    if not pool:
        print("\n4. no pieces long enough to start from; try --min-track 5"); pool = tracks
    byid = {t.id: t for t in tracks}
    cache: dict = {}
    def cand(t, d):
        k = (t.id, d)
        if k not in cache:
            cache[k] = candidates(t, tracks, d, part_end, part_start)
        return cache[k]
    cover, qs, steps, stops = [], [], [], {"none": 0, "edge": 0, "loop": 0}
    for _ in range(a.walks):
        s = pool[int(rng.integers(len(pool)))]
        seen = {s.id}; followed = s.e - s.s; q = 0; n = 0
        for d in (+1, -1):
            cur = s
            for _step in range(400):
                cs, st = cand(cur, d)
                if st in ("edge", "none"):
                    stops[st] += 1; break
                top = cs[0][0]
                if st == "unsure":
                    q += 1
                if top.id in seen:
                    stops["loop"] += 1; break
                seen.add(top.id); followed += top.e - top.s; n += 1; cur = top
        part_len = part_end[s.part] - part_start[s.part]
        cover.append(min(1.0, followed / max(play_s / len(part_start), 1.0)))
        qs.append(q); steps.append(n)
    cover, qs = np.array(cover), np.array(qs)
    minutes = cover * (play_s / len(part_start)) / 60
    print("\n4. IF A PLAYER CLICKED HIMSELF, FOLLOWING THE TRACKER'S BEST GUESS EACH TIME")
    print(f"   ({a.walks} simulated players, starting from pieces of at least {a.min_track:.0f} s)")
    print(f"   followed time: median {np.median(minutes):.1f} min ({np.median(cover)*100:.0f}% of one half) · 10th–90th percentile {np.percentile(minutes,10):.1f}–{np.percentile(minutes,90):.1f} min")
    print(f"   questions asked: median {np.median(qs):.0f} · 90th percentile {np.percentile(qs, 90):.0f} · most {qs.max()}")
    qpm = qs / np.maximum(minutes, 0.1)
    print(f"   that is about {np.median(qpm):.1f} questions per followed minute (median)")
    print(f"   chains stopped because: lost for good {stops['none']}, reached the end of a half {stops['edge']}, looped {stops['loop']}")
    print("   (the best guess is taken at every question, so the chain may follow a wrong player: this counts the effort, not the accuracy)")
    rep["follow"] = {"walks": a.walks, "median_minutes": round(float(np.median(minutes)), 2), "median_questions": float(np.median(qs)),
                     "p90_questions": float(np.percentile(qs, 90)), "median_cover_of_half": round(float(np.median(cover)), 2)}

    # 5. ball
    if ball is not None:
        seen_b = np.array([b is not None for b in ball])
        conf = np.array([(b[2] if b is not None and len(b) > 2 else 0) for b in ball], float)
        # longest stretch without a ball
        run = best = 0
        for s_ in seen_b:
            run = 0 if s_ else run + 1; best = max(best, run)
        dt = float(np.median(np.diff(vt))) if nm > 1 else 0.33
        w = max(1, int(round(10.0 / max(dt, 1e-3))))
        blocks = [seen_b[i:i + w] for i in range(0, nm, w)]
        empty = sum(1 for b_ in blocks if not b_.any())
        print("\n5. THE BALL")
        print(f"   seen in {pct(seen_b.sum(), nm):.0f}% of moments · confidence ≥ 50% in {pct((conf >= 50).sum(), nm):.0f}%")
        print(f"   longest stretch without a ball: {best * dt:.0f} s · {pct(empty, len(blocks)):.0f}% of the 10-second windows have no ball at all")
        rep["ball"] = {"seen_pct": round(pct(seen_b.sum(), nm), 1), "longest_gap_s": round(best * dt, 1), "empty_10s_windows_pct": round(pct(empty, len(blocks)), 1)}
    else:
        print("\n5. THE BALL: this board file has no ball data (export it with the ball switched on).")

    print("\nHow to read it: a good game has long pieces (section 2), most links 'sure' (3), few questions per player (4) "
          "and a ball seen most of the time (5). Remember this file is already cleaned, so real breaks are at least as many.\n")
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=2)); print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
