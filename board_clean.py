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
                gate_per_m: float = 0.06, max_gap_s: float = 2.5, speed_ms: float = 3.0) -> np.ndarray:
    """Track id per detection (-1 never left). A track continues with a detection when that is within a gate (metres,
    wider far away, where positions are less sure) of the track's last position; the gate also grows with the time the
    track was not seen (`speed_ms` metres per second beyond one moment), so a player missed in a few moments is picked
    up again instead of becoming a stranger. Detections and tracks are matched one-to-one by the shortest distance."""
    from scipy.optimize import linear_sum_assignment
    n = len(frame)
    track = np.full(n, -1, int)
    if n == 0:
        return track
    nf = len(vt)
    bounds = np.searchsorted(frame, np.arange(nf + 1))
    nxt = 0
    active: dict = {}                                    # track id -> (x, y, time) of its last detection
    for f in range(nf):
        cur = np.arange(bounds[f], bounds[f + 1])
        if len(cur) == 0:
            continue
        now = float(vt[f])
        for t in [t for t, (_, _, tt) in active.items() if now - tt > max_gap_s]:
            del active[t]
        used = np.zeros(len(cur), bool)
        if active:
            ids = list(active)
            ax = np.array([active[t][0] for t in ids])
            ay = np.array([active[t][1] for t in ids])
            age = np.array([now - active[t][2] for t in ids])
            d = np.hypot(ax[:, None] - x[cur][None], ay[:, None] - y[cur][None])
            gate = (gate_m + gate_per_m * 0.5 * (ay[:, None] + y[cur][None])
                    + speed_ms * np.maximum(age[:, None] - 0.45, 0.0))
            cost = np.where(d <= gate, d + 0.5 * age[:, None], 1e6)
            r, cc = linear_sum_assignment(cost)
            for a, b in zip(r, cc):
                if cost[a, b] < 1e5:
                    track[cur[b]] = ids[a]
                    used[b] = True
        for k in np.where(~used)[0]:
            track[cur[k]] = nxt
            nxt += 1
        for k in range(len(cur)):                        # the tracks move on to where they were last seen
            active[int(track[cur[k]])] = (float(x[cur[k]]), float(y[cur[k]]), now)
    return track


def fill_gaps(frame: np.ndarray, vt: np.ndarray, x: np.ndarray, y: np.ndarray, team: np.ndarray, track: np.ndarray,
              max_gap_s: float = 1.5):
    """Players the detector missed for a moment or two are still on the pitch: where a track has a hole of at most
    `max_gap_s` seconds, with the same label before and after it, the missing moments get a position on the straight line
    between the two ends. -> (frame, x, y, team) of the added detections (the originals are not touched)."""
    nf, nx, ny, nt = [], [], [], []
    for t in np.unique(track[track >= 0]):
        idx = np.flatnonzero(track == t)
        idx = idx[np.argsort(frame[idx])]
        for a, b in zip(idx[:-1], idx[1:]):
            f0, f1 = int(frame[a]), int(frame[b])
            if f1 - f0 < 2 or team[a] != team[b] or vt[f1] - vt[f0] > max_gap_s:
                continue
            for f in range(f0 + 1, f1):
                w = (vt[f] - vt[f0]) / max(vt[f1] - vt[f0], 1e-9)
                nf.append(f)
                nx.append(float(x[a] + w * (x[b] - x[a])))
                ny.append(float(y[a] + w * (y[b] - y[a])))
                nt.append(int(team[a]))
    return np.array(nf, int), np.array(nx), np.array(ny), np.array(nt, int)


def confirm_low(track: np.ndarray, conf: np.ndarray, min_conf: float, min_len: int = 6, sure: float = 0.45,
                typical: float = 0.26) -> np.ndarray:
    """Which detections may stay: all with a confidence of `min_conf` or more, and the weaker ones only when they belong to
    a steady track. A far, small, dark-shirted keeper is found in nearly every moment but with a confidence that flickers
    between 0.2 and 0.45: cutting at 0.3 makes him blink in and out. A weak detection stays when its track has at least
    `min_len` detections and either one of them is `sure` or the typical (median) confidence is `typical`;
    isolated weak detections (the usual false ones) go."""
    keep = conf >= min_conf
    if keep.all() or len(track) == 0:
        return keep
    ok_track = {}
    for t in np.unique(track[track >= 0]):
        c = conf[track == t]
        ok_track[t] = len(c) >= min_len and (c.max() >= sure or float(np.median(c)) >= typical)
    for i in np.flatnonzero(~keep):
        keep[i] = track[i] >= 0 and ok_track.get(track[i], False)
    return keep


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
