#!/usr/bin/env python3
"""board_ball.py — the ball on the tactical board: one position per moment, or none.

The detector returns every ball-like thing in every moment (boots, line marks, a spare ball, ...), 1.5-2 per frame on average,
and the right one is often not the strongest. Two things tell them apart over time: the ball does not stay where it is
(board_hotspots), and it does not jump across the pitch between two moments. `choose_track` picks, from all candidates of
all moments, the path with the most evidence that follows those two rules; moments with no believable candidate get none.

Positions are metres on the ground, camera at the origin like the players (scene_check.ground_position), found from where the
ball is in the picture and the camera-height model fitted on the players (ground_model.py). A ball in the air is drawn
further away than it is: the position is only as good as 'the ball is on the ground'.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

MIN_CONF = 0.25          # the extraction's own threshold (pipeline.ball_args)
BASE = 0.20              # a candidate is worth conf - BASE: a weak one needs the company of its neighbours to survive
START = 0.20             # cost of starting a track (an isolated hit must be strong)
SPEED_OK_M = 12.0        # metres per moment (3/s) the ball may travel without penalty: 36 m/s, a hard shot
JUMP_COST = 0.05         # per metre beyond that
JUMP_MAX = 1.5           # a jump is never worse than this: the ball can really be lost and found elsewhere
BALL_DEPRESSION = 3.0    # degrees below the horizon: nearer the horizon the foot ruler is not usable


def project(lon: float, lat: float, centre_lon: float, c: float, delta: float, fy: float, k: float):
    """Ground position (x, y) in metres of a ball seen at (lon, lat); None when too near the horizon to tell."""
    import ground_model as GM
    import scene_check as S
    if -lat + delta < BALL_DEPRESSION:
        return None
    h = float(GM.model_h(lon, lat, centre_lon, c, delta))
    return S.ground_position(lon, h, centre_lon, fy, k)


def choose_track(cands: Sequence[Sequence[Tuple[float, float, float]]], base: float = BASE, start: float = START
                 ) -> List[Optional[int]]:
    """cands[i] = [(x, y, conf), ...] of moment i (metres). -> for each moment the index of the chosen candidate or None.

    Viterbi over 'candidate j of this moment, or no ball': a candidate adds conf - base; going from candidate to candidate
    costs a penalty for travelling more than SPEED_OK_M; starting from 'no ball' costs `start`."""
    n = len(cands)
    out: List[Optional[int]] = [None] * n
    if n == 0:
        return out
    NEG = -1e18
    score = [np.array([0.0] + [c[2] - base for c in cs]) for cs in cands]     # state 0 = no ball
    best = [np.full(len(s), NEG) for s in score]
    back = [np.zeros(len(s), int) for s in score]
    best[0] = score[0].copy()
    for i in range(1, n):
        prev = best[i - 1]
        pc = cands[i - 1]
        for s in range(len(score[i])):
            if s == 0:                                         # no ball now: continue from the best previous state
                j = int(np.argmax(prev))
                best[i][0] = prev[j]
                back[i][0] = j
                continue
            x, y, _ = cands[i][s - 1]
            bv, bj = prev[0] - start, 0                        # starting a new track
            for j in range(1, len(prev)):
                px, py, _ = pc[j - 1]
                d = float(np.hypot(x - px, y - py))
                v = prev[j] - min(JUMP_MAX, JUMP_COST * max(0.0, d - SPEED_OK_M))
                if v > bv:
                    bv, bj = v, j
            best[i][s] = bv + score[i][s]
            back[i][s] = bj
    s = int(np.argmax(best[-1]))
    for i in range(n - 1, -1, -1):
        out[i] = None if s == 0 else s - 1
        s = int(back[i][s])
    return out


def track(frames: Sequence[dict], views: np.ndarray, gm, fy: float, k: float, tester=None, min_conf: float = MIN_CONF,
          pitch_margin_px: float = 100.0, stats: Optional[dict] = None) -> List[Optional[Tuple[float, float, float]]]:
    """Per moment of a detections file: (x, y, conf) of the ball in metres, or None.
    Candidates in a fixed spot (a mark, a spare ball) and far outside the outline are dropped before the choice."""
    import ball_hotspots as BH
    if gm is None:
        return [None] * len(frames)
    hot, _ = BH.hotspot_cells([r for r in frames if "t" in r])
    c0, d0 = float(gm[0]), float(gm[1])
    cands: List[list] = []
    n_in = n_kept = 0
    for r in frames:
        cs = []
        for b in r.get("balls", []):
            n_in += 1
            lon, lat, conf = float(b[0]), float(b[1]), float(b[2])
            if conf < min_conf or (int(lon // 1.0), int(lat // 1.0)) in hot:
                continue
            if tester is not None and tester.distance_px(lon, lat) > pitch_margin_px:
                continue
            centre = float(views[int(np.argmin(np.abs(views - lon)))])
            p = project(lon, lat, centre, c0, d0, fy, k)
            if p is None:
                continue
            cs.append((float(p[0]), float(p[1]), conf))
        cands.append(cs)
        n_kept += len(cs)
    pick = choose_track(cands)
    out = [None if j is None else cands[i][j] for i, j in enumerate(pick)]
    if stats is not None:
        stats["ball_candidates"] = stats.get("ball_candidates", 0) + n_in
        stats["ball_usable"] = stats.get("ball_usable", 0) + n_kept
        stats["ball_moments"] = stats.get("ball_moments", 0) + sum(1 for o in out if o is not None)
        stats["ball_total"] = stats.get("ball_total", 0) + len(out)
    return out
