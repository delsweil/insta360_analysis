#!/usr/bin/env python3
"""board_clean.py — make the team labels of the tactical board agree with what football allows.

A shirt colour seen from 60 m under floodlights is a noisy witness. Three facts about the game are better witnesses:

  0. Roles: a colour group can be the referee (3) or a goalkeeper (4, resolved to the keeper of A or B by board_data from where
     he stands). They are voted along the track like a team and never count towards the 11 of a team.
  1. A player does not change team. Seen at 3 moments a second, a player moves less than a few metres between two
     moments, so the detections of consecutive moments can be chained into short tracks. The team of a detection is the
     weighted vote of its track within +-2.5 s (a sure colour counts more than an uncertain one). Small far-away boxes
     with no readable colour take the team of the track they belong to.
  2. People standing outside the pitch outline are not players: a team colour there turns into 'other' (referee,
     substitute, spectator).
  3. A team has at most 11 players on the pitch. When 12 or more are labelled, the least certain ones (and those
     furthest outside the outline) change to the other team if that one has room and the colour was nearly as close to it,
     otherwise to 'other'.

Pure functions on arrays, no I/O, so they can be tested without footage. The camera path and the statistics are not
touched: this runs only on the data written for the board.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

OTHER = 2
REFEREE = 3
GK = 4                                                                # either goalkeeper; board_data tells them apart by position


def link_tracks(frame: np.ndarray, x: np.ndarray, y: np.ndarray, vt: np.ndarray, gate_m: float = 3.0,
                gate_per_m: float = 0.06, max_gap_s: float = 0.8) -> np.ndarray:
    """Track id per detection (-1 never left): detections of consecutive moments within a gate (metres, wider far away,
    where positions are less sure) are matched one-to-one by the shortest total distance."""
    from scipy.optimize import linear_sum_assignment
    n = len(frame)
    track = np.full(n, -1, int)
    if n == 0:
        return track
    nf = len(vt)
    bounds = np.searchsorted(frame, np.arange(nf + 1))
    nxt = 0
    prev = np.zeros(0, int)
    prev_f = -1
    for f in range(nf):
        cur = np.arange(bounds[f], bounds[f + 1])
        if len(cur) == 0:
            continue
        used = np.zeros(len(cur), bool)
        if len(prev) and prev_f >= 0 and vt[f] - vt[prev_f] <= max_gap_s:
            d = np.hypot(x[prev][:, None] - x[cur][None], y[prev][:, None] - y[cur][None])
            gate = gate_m + gate_per_m * 0.5 * (y[prev][:, None] + y[cur][None])
            cost = np.where(d <= gate, d, 1e6)
            r, c = linear_sum_assignment(cost)
            for a, b in zip(r, c):
                if cost[a, b] < 1e5:
                    track[cur[b]] = track[prev[a]]
                    used[b] = True
        for k in np.where(~used)[0]:
            track[cur[k]] = nxt
            nxt += 1
        prev, prev_f = cur, f
    return track


def vote_in_tracks(track: np.ndarray, frame: np.ndarray, vt: np.ndarray, team: np.ndarray, weight: np.ndarray,
                   window_s: float = 2.5, min_share: float = 0.6, min_weight: float = 0.4) -> np.ndarray:
    """Team of every detection = weighted vote of its track's detections within +-window_s (needs min_weight of evidence
    and min_share of it for one class; otherwise the detection keeps its own label)."""
    new = team.copy()
    ncls = max(3, int(team.max()) + 1) if len(team) else 3           # 0 A, 1 B, 2 other, 3 referee, 4 goalkeeper
    t_of = vt[frame]
    order = np.argsort(track, kind="stable")
    tr = track[order]
    starts = np.r_[0, np.where(np.diff(tr) != 0)[0] + 1, len(tr)]
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a < 2:
            continue
        idx = order[a:b]
        idx = idx[np.argsort(t_of[idx], kind="stable")]
        t = t_of[idx]
        cs = np.zeros((ncls, len(idx) + 1))
        for c in range(ncls):
            cs[c, 1:] = np.cumsum(np.where(team[idx] == c, weight[idx], 0.0))
        lo = np.searchsorted(t, t - window_s, side="left")
        hi = np.searchsorted(t, t + window_s, side="right")
        s = cs[:, hi] - cs[:, lo]                                   # (3, n) votes in the window
        tot = s.sum(axis=0)
        win = s.argmax(axis=0)
        share = s[win, np.arange(len(idx))] / np.maximum(tot, 1e-9)
        ok = (tot >= min_weight) & (share >= min_share)
        new[idx[ok]] = win[ok]
    return new


def cap_teams(frame: np.ndarray, team: np.ndarray, weight: np.ndarray, margin: np.ndarray, out_m: np.ndarray,
              max_team: int = 11, swap_margin: float = 0.5) -> Tuple[np.ndarray, int, int]:
    """At most max_team of each team per moment. -> (team, moved to the other team, moved to 'other')."""
    new = team.copy()
    moved_team = moved_other = 0
    nf = int(frame.max()) + 1 if len(frame) else 0
    bounds = np.searchsorted(frame, np.arange(nf + 1))
    for f in range(nf):
        sl = np.arange(bounds[f], bounds[f + 1])
        if len(sl) <= max_team:
            continue
        for t in (0, 1):
            mem = sl[new[sl] == t]
            if len(mem) <= max_team:
                continue
            score = weight[mem] * (1.0 - np.minimum(out_m[mem], 2.0) / 2.0)
            for j in mem[np.argsort(score, kind="stable")][: len(mem) - max_team]:
                other = 1 - t
                if margin[j] < swap_margin and (new[sl] == other).sum() < max_team and out_m[j] <= 0.0:
                    new[j] = other
                    moved_team += 1
                else:
                    new[j] = OTHER
                    moved_other += 1
    return new, moved_team, moved_other


def clean(frame: np.ndarray, vt: np.ndarray, x: np.ndarray, y: np.ndarray, team: np.ndarray, weight: np.ndarray,
          margin: np.ndarray, out_m: np.ndarray, max_team: int = 11, team_margin_m: float = 0.5,
          window_s: float = 2.5) -> Tuple[np.ndarray, Dict[str, int]]:
    """frame (sorted) / vt: moment of each detection and time of each moment; x, y: metres; team 0/1/2; weight: how much the
    detection's own label counts as a vote (0 = no readable colour); margin: 0 = colour equally near both teams, 1 = sure;
    out_m: metres outside the pitch outline (0 inside)."""
    stats = {"relabelled": 0, "recovered": 0, "outside": 0, "to_other_team": 0, "to_other": 0}
    if len(frame) == 0:
        return team, stats
    outside = out_m > team_margin_m
    w = np.where(outside, 0.0, weight)                                # someone outside the pitch is no witness
    track = link_tracks(frame, x, y, vt)
    voted = vote_in_tracks(track, frame, vt, team, w, window_s)
    stats["relabelled"] = int(((voted != team) & (weight > 0)).sum())
    stats["recovered"] = int(((voted != team) & (weight <= 0) & (voted < 2)).sum())
    new = voted
    gone = outside & ((new < 2) | (new == GK))                         # a referee may be on the touchline; a keeper-coloured
                                                                    # person outside the pitch is no keeper
    new[gone] = OTHER
    stats["outside"] = int(gone.sum())
    new, stats["to_other_team"], stats["to_other"] = cap_teams(frame, new, weight, margin, out_m, max_team)
    return new, stats
