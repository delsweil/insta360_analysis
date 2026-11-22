#!/usr/bin/env python3
"""track_stitch.py — can better linking and a whole-track team vote make players easier to follow?  (read-only)

It takes the SAME detections the board is built from and tracks the players twice, then compares the two:

  today  board_clean.link_tracks (nearest-distance matching, no motion model) and a team vote within +-2.5 s of a track
  new    1. link with a motion model (position predicted from the speed), in metres, and a small penalty when a
            near-side shirt colour contradicts the track
         2. stitch: pieces that end and start close to where the player should be are joined over gaps of up to 8 s
         3. the team label of a detection is the vote of its track within +-15 s (not just 2.5 s); far-side colours count
            less than near-side ones, because a 28 px shirt is a weak witness

It changes no file. Nothing here uses the ball, names or footage beyond what the board already uses.

  cd <your insta360_analysis folder>
  python track_stitch.py /Users/davidelsweiler/FootballAutopan/TSV_Buch_A-Jugend/ --assign 6=B 5=REF

It finds v0.onpitch.detections.jsonl ... with their v0.auto.keyframes.json and teams.team_labels.json in the folder
(or name one part: --det X.jsonl --kf Y.json). Parts are tracked one by one, since their clocks overlap.

What to read in the report:
  * pieces per player per 10 min and the median piece length: fewer, longer pieces are better
  * "near-side contradictions": pieces whose confident near-side shirts say both A and B. That is a likely swap of two
    players (colour on the near side is reliable). It should fall.
  * team counts per moment: team A should no longer sink from the first to the last part
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

SIDE = {0: 0, 4: 0, 1: 1, 5: 1}                  # board team codes -> side: 0 = A (and keeper A), 1 = B (and keeper B), 2 = other
PITCH_W = 68.0


def side_of(lab):
    return np.array([SIDE.get(int(v), 2) for v in lab], int)


def reliability(y):
    """how much a shirt colour is worth at this distance from the camera (y in metres, small = near): 1 near, 0.4 far"""
    return 1.0 - 0.6 * np.clip(np.asarray(y, float) / PITCH_W, 0.0, 1.0)


# ───────────── 1. linking with a motion model ─────────────

def link(fr, vt, x, y, lab, colour_cost=4.0, g0=2.0, gpm=0.05, max_gap=3.0, vmax=9.0):
    """track id per detection. fr: moment index (sorted), vt: time of each moment."""
    from scipy.optimize import linear_sum_assignment
    n = len(fr)
    tid = np.full(n, -1, int)
    if n == 0:
        return tid
    sd = side_of(lab)
    rel = reliability(y)
    bounds = np.searchsorted(fr, np.arange(len(vt) + 1))
    act: dict = {}                                    # id -> [x, y, vx, vy, t, wA, wB]
    nxt = 0
    for f in range(len(vt)):
        cur = np.arange(bounds[f], bounds[f + 1])
        if len(cur) == 0:
            continue
        now = float(vt[f])
        for k in [k for k, s in act.items() if now - s[4] > max_gap]:
            del act[k]
        used = np.zeros(len(cur), bool)
        if act:
            ids = list(act)
            S = np.array([act[k] for k in ids])
            dt = now - S[:, 4]
            ex = np.minimum(dt, 2.0) * 0.7
            px = S[:, 0] + S[:, 2] * ex
            py = S[:, 1] + S[:, 3] * ex
            d = np.hypot(px[:, None] - x[cur][None], py[:, None] - y[cur][None])
            gate = g0 + gpm * 0.5 * (S[:, 1][:, None] + y[cur][None]) + 2.0 * np.maximum(dt[:, None] - 0.45, 0.0)
            cost = d + 0.3 * dt[:, None]
            if colour_cost > 0:
                tot = S[:, 5] + S[:, 6]
                tsd = np.where(S[:, 5] >= S[:, 6], 0, 1)
                share = np.maximum(S[:, 5], S[:, 6]) / np.maximum(tot, 1e-9)
                firm = (tot >= 2.0) & (share >= 0.75)
                clash = firm[:, None] & (sd[cur][None] < 2) & (sd[cur][None] != tsd[:, None])
                cost = cost + clash * colour_cost * rel[cur][None]
            cost = np.where(d <= gate, cost, 1e6)
            r, c = linear_sum_assignment(cost)
            for a, b in zip(r, c):
                if cost[a, b] < 1e5:
                    tid[cur[b]] = ids[a]
                    used[b] = True
        for k in np.where(~used)[0]:
            tid[cur[k]] = nxt
            act[nxt] = [float(x[cur[k]]), float(y[cur[k]]), 0.0, 0.0, now, 0.0, 0.0]
            nxt += 1
        for k in np.where(used)[0]:                   # tracks that were extended move on and learn their speed and colour
            i = cur[k]; s = act[int(tid[i])]
            dt = now - s[4]
            if 0 < dt <= 1.5:
                vx = (x[i] - s[0]) / dt; vy = (y[i] - s[1]) / dt
                sp = math.hypot(vx, vy)
                if sp > vmax:
                    vx *= vmax / sp; vy *= vmax / sp
                s[2] = 0.5 * s[2] + 0.5 * vx if s[2] or s[3] else vx
                s[3] = 0.5 * s[3] + 0.5 * vy if s[2] or s[3] else vy
            s[0], s[1], s[4] = float(x[i]), float(y[i]), now
            if sd[i] == 0: s[5] += rel[i]
            elif sd[i] == 1: s[6] += rel[i]
        for k in np.where(~used)[0]:
            i = cur[k]; s = act[int(tid[i])]
            if sd[i] == 0: s[5] += rel[i]
            elif sd[i] == 1: s[6] += rel[i]
    return tid


# ───────────── 2. stitching pieces over gaps ─────────────

def _pieces(fr, vt, x, y, lab, tid):
    sd = side_of(lab); rel = reliability(y)
    out = {}
    order = np.lexsort((fr, tid))
    t_sorted = tid[order]
    cuts = np.r_[0, np.flatnonzero(np.diff(t_sorted)) + 1, len(order)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        idx = order[a:b]
        if tid[idx[0]] < 0:
            continue
        tt = vt[fr[idx]]
        m = min(3, len(idx) - 1)
        def vel(i0, i1):
            dt = tt[i1] - tt[i0]
            return (0.0, 0.0) if dt <= 0 else ((x[idx[i1]] - x[idx[i0]]) / dt, (y[idx[i1]] - y[idx[i0]]) / dt)
        wA = float(rel[idx][sd[idx] == 0].sum()); wB = float(rel[idx][sd[idx] == 1].sum())
        out[int(tid[idx[0]])] = dict(s=float(tt[0]), e=float(tt[-1]), p0=(float(x[idx[0]]), float(y[idx[0]])),
                                     p1=(float(x[idx[-1]]), float(y[idx[-1]])), v0=vel(0, m), v1=vel(len(idx) - 1 - m, len(idx) - 1),
                                     wA=wA, wB=wB)
    return out


def stitch(fr, vt, x, y, lab, tid, rounds=2, max_gap=8.0, colour_cost=4.0, g0=2.5, gpm=0.06, gper_s=0.9, vmax=9.0,
           gate_scale=1.0, hard_clash=False, stats=None):
    from scipy.optimize import linear_sum_assignment
    tid = tid.copy()
    for _ in range(rounds):
        P = _pieces(fr, vt, x, y, lab, tid)
        ids = sorted(P, key=lambda k: P[k]["s"])
        if len(ids) < 2:
            break
        e_t = np.array([P[k]["e"] for k in ids]); s_t = np.array([P[k]["s"] for k in ids])
        e_p = np.array([P[k]["p1"] for k in ids]); s_p = np.array([P[k]["p0"] for k in ids])
        e_v = np.array([P[k]["v1"] for k in ids])
        wA = np.array([P[k]["wA"] for k in ids]); wB = np.array([P[k]["wB"] for k in ids])
        gap = s_t[None, :] - e_t[:, None]                          # end of a -> start of b
        ok = (gap > 0.1) & (gap <= max_gap)
        ex = np.minimum(gap, 2.5) * 0.7
        px = e_p[:, 0][:, None] + e_v[:, 0][:, None] * ex
        py = e_p[:, 1][:, None] + e_v[:, 1][:, None] * ex
        d = np.hypot(px - s_p[:, 0][None], py - s_p[:, 1][None])
        gate = gate_scale * (g0 + gpm * 0.5 * (e_p[:, 1][:, None] + s_p[:, 1][None]) + gper_s * np.maximum(gap, 0))
        cost = d + 0.4 * np.maximum(gap, 0)
        if colour_cost > 0:
            sa = np.where(wA >= wB, 0, 1); sh = np.maximum(wA, wB) / np.maximum(wA + wB, 1e-9)
            firm = ((wA + wB) >= 2.0) & (sh >= 0.75)
            clash = firm[:, None] & firm[None, :] & (sa[:, None] != sa[None, :])
            cost = cost + clash * colour_cost
            if hard_clash:
                ok = ok & ~clash
        cost = np.where(ok & (d <= gate), cost, 1e6)
        r, c = linear_sum_assignment(cost)
        remap = {}
        reach = cost < 1e5
        long_ = np.array([P[k]["e"] - P[k]["s"] >= 3.0 for k in ids])       # fragments under 3 s do not count as rivals
        n_end = (reach & long_[None, :]).sum(axis=1)
        n_start = (reach & long_[:, None]).sum(axis=0)
        for a, b in zip(r, c):
            if cost[a, b] < 1e5:
                remap[ids[b]] = ids[a]
                if stats is not None:                      # a join is 'risky' when another piece could just as well have been the continuation
                    stats["joins"] = stats.get("joins", 0) + 1
                    stats["risky"] = stats.get("risky", 0) + int(n_end[a] - int(long_[b]) > 0 or n_start[b] - int(long_[a]) > 0)
        if not remap:
            break
        def root(k):
            while k in remap:
                k = remap[k]
            return k
        lut = {k: root(k) for k in ids}
        m = tid >= 0
        tid[m] = np.array([lut.get(int(v), int(v)) for v in tid[m]])
    return tid


# ───────────── 3. one team per track ─────────────

def vote_window(fr, vt, y, lab, tid, window_s=15.0, min_w=1.0, min_share=0.55):
    """Team of every detection = vote of its track within +-window_s, far shirts counting less. A window, not the whole
    track, so a stitched track that has swapped two players does not carry one label for both."""
    import board_clean as BC
    sd = side_of(lab); rel = reliability(y)
    w = rel * np.where(sd == 2, 0.6, 1.0)
    vs = BC.vote_in_tracks(tid, fr, vt, sd.copy(), w, window_s, min_share, min_w)
    new = lab.copy()
    chg = vs != sd
    keeper_ok = np.isin(lab, (4, 5)) & (vs == sd)
    new[chg & (vs == 0)] = 0
    new[chg & (vs == 1)] = 1
    new[chg & (vs == 2)] = 2
    return new


def vote_whole(fr, y, lab, tid, min_w=1.0, min_share=0.55):
    """-> new labels. Every detection takes its track's team (A / B / other) when the vote is clear; else keeps its own."""
    new = lab.copy()
    sd = side_of(lab); rel = reliability(y)
    keeper = np.isin(lab, (4, 5)); ref = lab == 3
    order = np.argsort(tid, kind="stable")
    t_sorted = tid[order]
    cuts = np.r_[0, np.flatnonzero(np.diff(t_sorted)) + 1, len(order)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        idx = order[a:b]
        if tid[idx[0]] < 0:
            continue
        w = np.array([rel[idx][sd[idx] == s].sum() * (0.6 if s == 2 else 1.0) for s in (0, 1, 2)])
        tot = w.sum()
        if tot < min_w:
            continue
        win = int(w.argmax())
        if w[win] / tot < min_share:
            continue
        if win < 2:
            kfrac = keeper[idx][sd[idx] == win].mean() if (sd[idx] == win).any() else 0.0
            new[idx] = (4 + win) if kfrac >= 0.5 else win
        else:
            new[idx] = 3 if ref[idx].sum() >= 0.5 * (sd[idx] == 2).sum() and ref[idx].any() else 2
    return new


# ───────────── numbers ─────────────

def piece_stats(fr, vt, y, lab, tid, min_len=0.0):
    sd = side_of(lab)
    near = y < np.percentile(y, 33)
    L, contra, mixed = [], 0, 0
    order = np.lexsort((fr, tid))
    t_sorted = tid[order]
    cuts = np.r_[0, np.flatnonzero(np.diff(t_sorted)) + 1, len(order)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        idx = order[a:b]
        if tid[idx[0]] < 0:
            continue
        tt = vt[fr[idx]]
        if tt[-1] - tt[0] < min_len:
            continue
        L.append(tt[-1] - tt[0])
        na = int(((sd[idx] == 0) & near[idx]).sum()); nb = int(((sd[idx] == 1) & near[idx]).sum())
        if na >= 3 and nb >= 3:
            contra += 1
        a_, b_ = int((sd[idx] == 0).sum()), int((sd[idx] == 1).sum())
        if a_ + b_ >= 6 and max(a_, b_) / (a_ + b_) < 0.9:
            mixed += 1
    return np.array(L), contra, mixed


def counts(fr, nmoments, y, lab):
    cnt = np.zeros((nmoments, 2))
    cn = np.zeros((nmoments, 2)); cf = np.zeros((nmoments, 2))
    lo, hi = np.percentile(y, 33), np.percentile(y, 66)
    for s in (0, 1):
        sel = side_of(lab) == s
        np.add.at(cnt[:, s], fr[sel], 1)
        np.add.at(cn[:, s], fr[sel & (y <= lo)], 1)
        np.add.at(cf[:, s], fr[sel & (y >= hi)], 1)
    return cnt, cn, cf


def report(name, parts, min_piece=0.0):
    """parts: list of dicts(fr, vt, x, y, lab_raw, tid, lab_final)"""
    print(f"\n── {name}")
    print(f"{'part':>5s} {'tracks':>7s} {'per player/10min':>17s} {'median s':>9s} {'p90 s':>6s} {'≥30 s share':>12s} {'contradictions':>15s} {'mixed':>6s} "
          f"{'A':>5s} {'B':>5s} {'A far':>6s} {'B far':>6s} {'relabelled':>11s}")
    tot = dict(tracks=0, contra=0, mixed=0, minutes=0.0)
    for i, p in enumerate(parts):
        L, contra, mixed = piece_stats(p["fr"], p["vt"], p["y"], p["lab_raw"], p["tid"], min_piece)
        mins = (p["vt"][-1] - p["vt"][0]) / 60.0
        pp = len(L) / max(1e-9, 22 * mins / 10.0)
        cnt, cn, cf = counts(p["fr"], len(p["vt"]), p["y"], p["lab_final"])
        ch = 100.0 * float((side_of(p["lab_final"]) != side_of(p["lab_raw"])).mean())
        share30 = 100.0 * L[L >= 30].sum() / max(L.sum(), 1e-9)
        print(f"{i:5d} {len(L):7d} {pp:17.1f} {np.median(L):9.1f} {np.percentile(L, 90):6.0f} {share30:11.0f}% {contra:15d} {mixed:6d} "
              f"{cnt[:, 0].mean():5.1f} {cnt[:, 1].mean():5.1f} {cf[:, 0].mean():6.1f} {cf[:, 1].mean():6.1f} {ch:10.1f}%")
        tot["tracks"] += len(L); tot["contra"] += contra; tot["mixed"] += mixed; tot["minutes"] += mins
    print(f"      total: {tot['tracks']} pieces, {tot['contra']} near-side contradictions, {tot['mixed']} mixed-label pieces, "
          f"{tot['contra'] / max(tot['minutes'], 1e-9) * 10:.1f} contradictions per 10 min")
    return tot


def run_part(fr, vt, x, y, lab, vote_s=15.0, stitch_kw=None, baseline=True):
    """baseline and new on one part"""
    import board_clean as BC
    if baseline:
        base_tid = BC.link_tracks(fr, x, y, vt)
        w = np.where(np.isin(lab, (0, 1, 4, 5)), 1.0, 0.5)
        base_lab = BC.vote_in_tracks(base_tid, fr, vt, lab.copy(), w, 2.5)
    else:
        base_tid, base_lab = None, None
    new_tid = link(fr, vt, x, y, lab)
    new_tid = stitch(fr, vt, x, y, lab, new_tid, **(stitch_kw or {}))
    new_lab = vote_whole(fr, y, lab, new_tid) if vote_s <= 0 else vote_window(fr, vt, y, lab, new_tid, vote_s)
    P = dict(fr=fr, vt=vt, x=x, y=y, lab_raw=lab)
    return dict(P, tid=base_tid, lab_final=base_lab), dict(P, tid=new_tid, lab_final=new_lab)


# ───────────── reading the real detections ─────────────

def _is_outline(path):
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return False
    return isinstance(d, dict) and any(d.get(k) for k in ("pitch_polygon", "pixel_polygon", "auto_polygon"))


def find_outlines(folder, dets, override=None):
    """one pitch outline per part: pitch0.manual.json for the part v0 and so on (any file whose name has 'pitch' and the part number
    and that holds a polygon); else a single outline for the whole game; else a *calib*.json that holds one; else None."""
    import re
    out = []
    single = sorted(p for p in glob.glob(os.path.join(folder, "*pitch*.json")) if _is_outline(p))
    for d, _ in dets:
        if override:
            out.append(override); continue
        m = re.match(r"^\D*?(\d+)", os.path.basename(d))
        n = m.group(1) if m else None
        pick = None
        if n is not None:
            names = [os.path.join(folder, f"pitch{n}.manual.json"), os.path.join(folder, f"pitch{n}.json")]
            names += sorted(glob.glob(os.path.join(folder, f"*pitch*{n}*.json")))
            pick = next((p for p in names if os.path.exists(p) and _is_outline(p)), None)
        if pick is None and len(single) == 1:
            pick = single[0]
        if pick is None:
            pick = find_outline(folder)
        out.append(pick)
    return out


def find_outline(folder):
    """a *calib*.json that really holds the pitch outline (pitch_polygon / pixel_polygon / auto_polygon); other calibration files are skipped"""
    for p in sorted(glob.glob(os.path.join(folder, "*calib*.json"))):
        try:
            d = json.loads(Path(p).read_text())
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and any(d.get(k) for k in ("pitch_polygon", "pixel_polygon", "auto_polygon")):
            return p
    return None


def find_parts(folder):
    """-> [(detections file, keyframes file or None)] for the parts of a game folder, in order.
    Prefers *onpitch* detections (the ones the board is built from); skips sample and meta files."""
    hits = {p for pat in ("*det*.jsonl",) for p in glob.glob(os.path.join(folder, pat))}
    hits = [h for h in hits if "sample" not in os.path.basename(h) and ".meta." not in os.path.basename(h)]
    on = [h for h in hits if "onpitch" in os.path.basename(h)]
    hits = sorted(on or hits)
    out = []
    for d in hits:
        stem = os.path.basename(d)
        for cut in (".onpitch.detections.jsonl", ".detections.jsonl", ".det.jsonl", ".jsonl"):
            if stem.endswith(cut):
                stem = stem[: -len(cut)]; break
        cands = [os.path.join(folder, stem + e) for e in (".auto.keyframes.json", ".kf.json", ".keyframes.json")]
        cands += sorted(glob.glob(os.path.join(folder, stem + "*keyframes*.json"))) + sorted(glob.glob(os.path.join(folder, stem + "*.kf.json")))
        out.append((d, next((c for c in cands if os.path.exists(c)), None)))
    return out



def part_points(det, kf, clusters, calib):
    import board_data as BD
    try:
        return _part_points(BD, det, kf, clusters, calib)
    except (TypeError, ValueError, KeyError, OSError) as e:
        if not calib:
            raise
        print(f"  ! pitch outline {os.path.basename(calib)} could not be used ({type(e).__name__}); continuing without it")
        return _part_points(BD, det, kf, clusters, None)


def _part_points(BD, det, kf, clusters, calib):
    fr_ = BD.part_frames(det, kf, clusters, 0.0, 98.0, BD.MIN_CONF, BD.MIN_HEIGHT, 0.75, BD.RESCUE, calib, BD.MARGIN_M, False,
                         BD.MAX_TEAM, BD.TEAM_MARGIN_M, None, True, False, None, None, BD.LOW_CONF, False, False)
    vt = np.array([f["vt"] for f in fr_], float)
    order = np.argsort(vt, kind="stable")
    rank = np.empty(len(vt), int); rank[order] = np.arange(len(vt))
    vt = vt[order]
    F, X, Y, L = [], [], [], []
    for i, f in enumerate(fr_):
        for (px, py, t) in f["players"]:
            F.append(rank[i]); X.append(px); Y.append(py); L.append(t)
    F = np.array(F, int); o = np.lexsort((np.arange(len(F)), F))
    return vt, F[o], np.array(X)[o], np.array(Y)[o], np.array(L, int)[o]


def sweep(pts, vote_s):
    """same points, several settings of the stitching: one line each"""
    configs = [("today (baseline)", None, 0.0), ("new, gate x1", dict(gate_scale=1.0), 0.0), ("new, gate x1, colour block", dict(gate_scale=1.0, hard_clash=True), 0.0),
               ("new, gate x1.5, colour block", dict(gate_scale=1.5, hard_clash=True), 0.0),
               ("new, gate x2, colour block", dict(gate_scale=2.0, hard_clash=True), 0.0),
               ("new, gate x1.5, colour block, pieces >= 3 s", dict(gate_scale=1.5, hard_clash=True), 3.0)]
    print(f"\n{'setting':46s} {'pieces':>7s} {'/player/10min':>14s} {'median s':>9s} {'>=30 s':>7s} {'contrad/10min':>14s} {'risky joins':>12s} {'A far (last)':>13s} {'B far (last)':>13s}")
    for name, kw, minp in configs:
        pieces = contra = 0; mins = 0.0; meds = []; share = []; last = None; st = {}
        for (vt, fr, x, y, lab) in pts:
            b, n = run_part(fr, vt, x, y, lab, vote_s, dict(kw or {}, stats=st), baseline=kw is None)
            p = b if kw is None else n
            L, c, m = piece_stats(fr, vt, y, lab, p["tid"], minp)
            pieces += len(L); contra += c; mins += (vt[-1] - vt[0]) / 60.0
            meds.append(np.median(L)); share.append(100.0 * L[L >= 30].sum() / max(L.sum(), 1e-9)); last = p
        cnt, cn, cf = counts(last["fr"], len(last["vt"]), last["y"], last["lab_final"])
        print(f"{name:46s} {pieces:7d} {pieces / max(1e-9, 22 * mins / 10.0):14.1f} {np.mean(meds):9.1f} {np.mean(share):6.0f}% "
              f"{contra / max(mins, 1e-9) * 10:14.1f} {('%d%%' % round(100 * st['risky'] / max(st['joins'], 1))) if st else '-':>12s} {cf[:, 0].mean():13.1f} {cf[:, 1].mean():13.1f}")
    print("\nRead it: fewer pieces per player is better only if the joins stay trustworthy. 'contradictions' only sees a join between "
          "players of DIFFERENT teams; it cannot see two team-mates swapped. 'risky joins' = share of joins where another piece of at least 3 s could "
          "just as well have been the continuation: those are the ones that may carry the wrong player. Rising risk means rising doubt.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--det"); ap.add_argument("--kf"); ap.add_argument("--labels"); ap.add_argument("--calib")
    ap.add_argument("--assign", nargs="+", metavar="GROUP=WHO", help="same as player_census.py: try colour groups as assigned, e.g. 6=B 5=REF")
    ap.add_argument("--out", help="write the numbers as JSON")
    ap.add_argument("--gate-scale", type=float, default=1.0, help="widen (>1) or narrow (<1) the stitching gate; default 1")
    ap.add_argument("--hard-clash", action="store_true", help="never stitch two pieces whose reliable shirts say different teams")
    ap.add_argument("--min-piece", type=float, default=0.0, help="leave pieces shorter than this many seconds out of the piece statistics (default 0)")
    ap.add_argument("--sweep", action="store_true", help="try several gate widths / colour blocking / minimum piece lengths on the same points and print one line each")
    ap.add_argument("--vote-window", type=float, default=15.0, help="seconds either side for the team vote (default 15; 0 = whole track)")
    a = ap.parse_args()
    try:
        import team_labels as TL
    except ImportError:
        sys.exit("Run this from your insta360_analysis folder.")
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
        else:
            sys.exit(f"--assign: {item}?")
    if a.det:
        dets = [(a.det, a.kf)]
    else:
        dets = find_parts(a.folder)
    if not dets:
        sys.exit("No detections file found (*det*.jsonl); name a part with --det / --kf.")
    print("parts:", ", ".join(f"{os.path.basename(d)} (camera path: {os.path.basename(k) if k else 'none'})" for d, k in dets))
    cals = find_outlines(a.folder, dets, a.calib)
    print("pitch outlines:", ", ".join(os.path.basename(c) if c else "none" for c in cals), "" if all(cals) else "  (people beside the pitch are not filtered where none)")
    pts = []
    for (d, k), cal in zip(dets, cals):
        print(f"reading {os.path.basename(d)} ...", flush=True)
        pts.append(part_points(d, k, clusters, cal))
    if a.sweep:
        sweep(pts, a.vote_window)
        return
    kw = dict(gate_scale=a.gate_scale, hard_clash=a.hard_clash)
    base_parts, new_parts = [], []
    for (vt, fr, x, y, lab) in pts:
        b, n = run_part(fr, vt, x, y, lab, a.vote_window, kw)
        base_parts.append(b); new_parts.append(n)
    tb = report("TODAY  (board_clean.link_tracks, team vote within ±2.5 s)", base_parts, a.min_piece)
    tn = report(f"NEW    (motion model + stitching over gaps up to 8 s, gate x{a.gate_scale:g}{', colour block' if a.hard_clash else ''} + team vote within ±{a.vote_window:g} s)", new_parts, a.min_piece)
    print("\nHow to read it: 'per player/10min' = pieces a player is split into (fewer is better; today ≈ 49). "
          "'contradictions' = pieces whose reliable near-side shirts say both A and B (a probable swap). "
          "'A / B far' = team counts on the far third; A should not sink and B should not swell from part to part.")
    if a.out:
        Path(a.out).write_text(json.dumps({"today": tb, "new": tn}, indent=2)); print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
