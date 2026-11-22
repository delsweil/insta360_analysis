#!/usr/bin/env python3
"""colour_refit.py — re-learn the team colours of each part from the players that can be read, then relabel the rest.  (read-only)

Why: the colour groups of a game are fitted once, but the light and the distance change. A shirt at 28 px in the evening
does not look like the same shirt at 60 px at kick-off, so far-side players drift from one team to the other (Eibach: team B
14.6 players per moment). This script tests whether re-learning the colours per part and per distance fixes it.

How, for each part (nothing is written; the labels file is not touched):
  1. SEEDS     near-side players (the nearest third of the pitch, a readable box) whose colour clearly belongs to team A or B.
  2. TRACKS    the position-only tracks of track_stitch.py. A track that holds enough seeds of one team (>= 3 and 80%) passes
               that team to every detection on it, including the far ones.
  3. MODEL     for each window of time (default 5 min) and distance third, the labelled colours of that window form a small
               nearest-neighbour model of "what team A / team B look like here, now".
  4. RELABEL   each detection with a readable colour takes the model's team when its neighbours agree (>= 70%) and its colour
               is as near as the training colours are; otherwise it keeps the label it had. Referee and keepers are untouched.

  cd <your insta360_analysis folder>
  python3 colour_refit.py /Users/davidelsweiler/FootballAutopan/TSV_Buch_A-Jugend/ --assign 6=B 5=REF

Needs track_stitch.py in the same folder. Reads the same files as track_stitch.py. The test needs no ground truth: team counts
should settle at 9-11 for each team and stop drifting from part to part ("A" and "B" per moment, before and after).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

try:
    import board_data as BD
    import board_clean as BC
    import scene_check as S
    import team_labels as TL
    import track_stitch as TS
except ImportError:
    sys.exit("Run this from your insta360_analysis folder, with track_stitch.py in it.")


# ───────────── the points with their colours (same kept detections and labels as board_data.part_frames, unsmoothed) ─────────────

def points_ex(det_path, kf_path, clusters, calib=None, min_conf=BD.MIN_CONF, low_conf=BD.LOW_CONF, min_height=BD.MIN_HEIGHT,
              k_angle=0.75, rescue=BD.RESCUE, margin_m=BD.MARGIN_M):
    """-> dict(vt, fr, x, y, bgr (n,3; nan = no readable colour), h, conf, lab, margin)  in the order of track_stitch.part_points"""
    AK, S_, GM = BD.AK, BD.S, BD.GM
    hdr, frames = AK.load_detections(det_path, max_lon=98.0)
    fy = S_.view_fy(hdr)
    tester = None
    if calib:
        import pitch_filter as PF
        tester = PF.FootTester(calib, margin_m=margin_m, fy=fy, k=k_angle)
    views = np.array(hdr["view_lons"], float)
    t_in = float(hdr.get("start", 0.0))
    if kf_path:
        import kf_core as K
        try:
            meta = K.load_meta(kf_path)
            if meta.get("trim_in") is not None:
                t_in = float(meta["trim_in"])
        except (OSError, ValueError, SystemExit):
            pass
    kept, centres = [], []
    n_out = n_all = 0
    for i, r in enumerate(frames):
        for p in r["players"]:
            if p[3] < min(min_conf, low_conf):
                continue
            n_all += 1
            if tester is not None and not tester(p[0], p[1], p[2]):
                n_out += 1
                continue
            kept.append((i, p))
            centres.append(float(views[np.argmin(np.abs(views - p[0]))]))
    h_use = np.array([p[2] for _, p in kept], float)
    gm = None
    if kept:
        arr = np.array([[p[0], p[1], p[2]] for r in frames for p in r["players"] if p[3] >= 0.5], float).reshape(-1, 3)
        if len(arr):
            cen_all = views[np.argmin(np.abs(views[None, :] - arr[:, 0:1]), axis=1)]
            f = GM.fit(arr[:, 0], arr[:, 1], arr[:, 2], cen_all)
            if f is not None and f[2] < 0.15:
                gm = f
    if gm is not None:
        lo_ = np.array([p[0] for _, p in kept]); la_ = np.array([p[1] for _, p in kept])
        h_use, _ = GM.fuse(lo_, la_, h_use, np.array(centres), gm[0], gm[1])
    X, Y, BGR, H, C, OWN = [], [], [], [], [], []
    for k, (i, p) in enumerate(kept):
        x, y = S_.ground_position(p[0], float(h_use[k]), centres[k], fy, k_angle)
        has = len(p) > 5 and p[5] is not None and p[2] >= min_height
        X.append(x); Y.append(y); H.append(float(p[2])); C.append(float(p[3])); OWN.append(i)
        BGR.append(p[5] if has else (np.nan, np.nan, np.nan))
    X, Y, H, C, OWN = map(np.array, (X, Y, H, C, OWN)); BGR = np.array(BGR, float).reshape(-1, 3)
    if len(X) and low_conf < min_conf:
        vt0 = np.array([float(r["t"]) for r in frames])
        trk0 = BC.link_tracks(OWN, X, Y, vt0)
        keep = BC.confirm_low(trk0, C, min_conf)
        X, Y, H, C, OWN, BGR = X[keep], Y[keep], H[keep], C[keep], OWN[keep], BGR[keep]
        centres = list(np.array(centres)[keep])
    n = len(X)
    team = np.full(n, 2, int); margin = np.ones(n); gk_label = np.full(n, -1)
    ok = np.isfinite(BGR[:, 0]) if n else np.zeros(0, bool)
    if n and ok.any():
        fn = getattr(clusters, "team_of_rescue", None)
        t = fn(BGR[ok], rescue) if fn is not None else clusters.team_of(BGR[ok])
        team[ok] = np.where(t >= 0, t, 2)
        dist = getattr(clusters, "team_distances", None)
        if dist is not None:
            d = dist(BGR[ok]); dd = np.where(np.isfinite(d[:, :2]), d[:, :2], 1e3)
            margin[ok] = np.abs(dd[:, 0] - dd[:, 1]) / np.maximum(dd[:, 0] + dd[:, 1], 1e-6)
        if getattr(clusters, "has_roles", False):
            rl = clusters.role_of(BGR[ok])
            gk_label[ok] = np.where(rl >= 4, rl, -1)
            team[ok] = np.where(rl == 3, 3, np.where(rl >= 4, 4, team[ok]))
        if (team >= 4).any():
            team = np.where(team == 4, np.where(gk_label >= 4, gk_label, 2), team)
    vt = np.array([float(r["t"]) for r in frames], float) - t_in
    order = np.argsort(vt, kind="stable")
    rank = np.empty(len(vt), int); rank[order] = np.arange(len(vt))
    vt = vt[order]
    fr = rank[OWN]
    o = np.lexsort((np.arange(n), fr))
    return dict(vt=vt, fr=fr[o], x=X[o], y=Y[o], bgr=BGR[o], h=H[o], conf=C[o], lab=team[o], margin=margin[o], outline=(tester is not None, n_out, n_all), view=np.array(centres, float)[o])


# ───────────── the refit ─────────────

def refit_part(P, window_s=300.0, bands=3, k=15, min_seed_votes=3, seed_share=0.8, near_frac=1 / 3, seed_margin=0.4,
               min_agree=0.7, min_cell=60, min_class=10, rounds=2):
    """-> (new labels, info). P as returned by points_ex."""
    vt, fr, x, y, bgr, lab, margin = P["vt"], P["fr"], P["x"], P["y"], P["bgr"], P["lab"], P["margin"]
    n = len(fr)
    new = lab.copy()
    info = dict(seeds=0, tracked=0, train=0, cells=0, cells_skipped=0, changed=0, to_a=0, to_b=0, a_to_b=0, b_to_a=0, loo_ok=0, loo_n=0)
    if n == 0:
        return new, info
    readable = np.isfinite(bgr[:, 0])
    tid = TS.stitch(fr, vt, x, y, lab, TS.link(fr, vt, x, y, lab))
    near_cut = np.percentile(y, 100 * near_frac)
    seed = readable & np.isin(lab, (0, 1)) & (margin >= seed_margin) & (y <= near_cut)
    info["seeds"] = int(seed.sum())
    ttm = {}
    order = np.argsort(tid, kind="stable"); ts_ = tid[order]
    cuts = np.r_[0, np.flatnonzero(np.diff(ts_)) + 1, n]
    for a, b in zip(cuts[:-1], cuts[1:]):
        idx = order[a:b]
        if tid[idx[0]] < 0:
            continue
        wa = int(((lab[idx] == 0) & seed[idx]).sum()); wb = int(((lab[idx] == 1) & seed[idx]).sum())
        if wa + wb >= min_seed_votes and max(wa, wb) / (wa + wb) >= seed_share:
            ttm[int(tid[idx[0]])] = 0 if wa >= wb else 1
    tteam = np.full(n, -1, int)
    for t_, c in ttm.items():
        tteam[tid == t_] = c
    train = readable & (tteam >= 0) & np.isin(lab, (0, 1, 2))
    info["tracked"] = int(((tteam >= 0)).sum()); info["train"] = int(train.sum())
    tteam0 = tteam.copy()
    feats = np.full((n, 3), np.nan)
    feats[readable] = S.colour_features(bgr[readable])
    t_det = vt[fr]
    win = np.floor((t_det - vt[0]) / window_s).astype(int)
    edges = np.percentile(y, np.linspace(0, 100, bands + 1)[1:-1])
    band = np.searchsorted(edges, y)
    from scipy.spatial import cKDTree

    pred_all = np.full(n, -1, int)                                  # the model's team per detection in this round, -1 = not sure / not asked
    def run_cell(sel_train, sel_query, allow_loo=True):
        Ti = np.flatnonzero(sel_train)
        if len(Ti) < min_cell:
            return False
        cls = tteam[Ti]
        if (cls == 0).sum() < min_class or (cls == 1).sum() < min_class:
            return False
        tree = cKDTree(feats[Ti])
        kk = min(k + 1, len(Ti))
        dq, iq = tree.query(feats[Ti], k=kk)                       # leave-one-out on the training points: the first hit is the point itself
        loo_d = dq[:, 1:].mean(axis=1)
        D = 1.5 * float(np.percentile(loo_d, 95))
        pa_tr = (cls[iq[:, 1:]] == 0).mean(axis=1)
        pred_tr = np.where(pa_tr >= 0.5, 0, 1)
        near_seed = seed[Ti]
        info["loo_n"] += int(near_seed.sum()); info["loo_ok"] += int((pred_tr[near_seed] == cls[near_seed]).sum())
        Qi = np.flatnonzero(sel_query)
        if len(Qi) == 0:
            return True
        dq, iq = tree.query(feats[Qi], k=min(k, len(Ti)))
        dq = dq.reshape(len(Qi), -1); iq = iq.reshape(len(Qi), -1)
        pa = (cls[iq] == 0).mean(axis=1)
        md = dq.mean(axis=1)
        pred = np.where(pa >= 0.5, 0, 1)
        sure = (np.maximum(pa, 1 - pa) >= min_agree) & (md <= D)
        pred_all[Qi[sure]] = pred[sure]
        return True

    queryable = readable & np.isin(lab, (0, 1, 2))
    for rd in range(max(1, rounds)):
        pred_all[:] = -1
        info["cells"] = 0; info["cells_skipped"] = 0; info["loo_ok"] = 0; info["loo_n"] = 0
        for w in np.unique(win):
            in_w = win == w
            for bnd in range(bands):
                info["cells"] += 1
                ok = run_cell(train & in_w & (band == bnd), queryable & in_w & (band == bnd))
                if not ok:                                          # too little in this cell: the whole window at this distance
                    ok = run_cell(train & in_w, queryable & in_w & (band == bnd))
                if not ok:
                    info["cells_skipped"] += 1
        if rd + 1 < rounds:                                         # round 2: what the model was sure about joins the training set
            tteam = np.where(tteam0 >= 0, tteam0, pred_all)
            train = readable & (tteam >= 0) & np.isin(lab, (0, 1, 2))
    chg = (pred_all >= 0) & (pred_all != lab)
    info["a_to_b"] = int((chg & (lab == 0)).sum()); info["b_to_a"] = int((chg & (lab == 1)).sum())
    info["to_a"] = int((chg & (lab == 2) & (pred_all == 0)).sum()); info["to_b"] = int((chg & (lab == 2) & (pred_all == 1)).sum())
    new[chg] = pred_all[chg]
    info["changed"] = int((new != lab).sum())
    info["tid"] = tid
    return new, info


# ───────────── report ─────────────

def outline_note(P):
    used, n_out, n_all = P.get("outline", (False, 0, 0))
    if not used:
        return "NOT USED (no outline, or it could not be read)"
    return f"used, removed {n_out} of {n_all} detections ({100 * n_out / max(n_all, 1):.2f}%)"


def counts(P, lab):
    n = len(P["vt"]); fr = P["fr"]; y = P["y"]
    far = y >= np.percentile(y, 66)
    cnt = np.zeros((n, 2)); cf = np.zeros((n, 2))
    for s in (0, 1):
        sel = TS.side_of(lab) == s
        np.add.at(cnt[:, s], fr[sel], 1)
        np.add.at(cf[:, s], fr[sel & far], 1)
    return cnt, cf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--det"); ap.add_argument("--kf"); ap.add_argument("--labels"); ap.add_argument("--calib")
    ap.add_argument("--assign", nargs="+", metavar="GROUP=WHO")
    ap.add_argument("--window", type=float, default=300.0, help="seconds per colour model (default 300)")
    ap.add_argument("--k", type=int, default=15, help="neighbours per vote (default 15)")
    ap.add_argument("--rounds", type=int, default=2, help="1 = only the track-labelled players train the model; 2 (default) = what the model was sure about joins them")
    a = ap.parse_args()
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
    dets = [(a.det, a.kf)] if a.det else TS.find_parts(a.folder)
    if not dets:
        sys.exit("No detections file found (*det*.jsonl); name a part with --det / --kf.")
    print("parts:", ", ".join(os.path.basename(d) for d, _ in dets))
    rows = []
    print(f"\n{'part':>4s} | {'A before':>8s} {'B before':>8s} {'A far':>6s} {'B far':>6s} {'>11':>5s} | {'A after':>7s} {'B after':>7s} {'A far':>6s} {'B far':>6s} {'>11':>5s} | "
          f"{'relabelled':>10s} {'other->team':>11s} {'A->B':>5s} {'B->A':>5s} | {'check':>6s}")
    cals = TS.find_outlines(a.folder, dets, a.calib)
    print("pitch outlines:", ", ".join(os.path.basename(c) if c else "none" for c in cals), "" if all(cals) else "  (people beside the pitch are not filtered where none)")
    for i, (d, k_) in enumerate(dets):
        try:
            P = points_ex(d, k_, clusters, cals[i])
        except (TypeError, ValueError, KeyError, OSError) as e:
            if not cals[i]:
                raise
            print(f"  ! pitch outline could not be used ({type(e).__name__}); continuing without it")
            P = points_ex(d, k_, clusters, None)
        print(f"     outline: {outline_note(P)}")
        new, info = refit_part(P, a.window, k=a.k, rounds=a.rounds)
        c0, f0 = counts(P, P["lab"]); c1, f1 = counts(P, new)
        over0 = 100 * float(((c0[:, 0] > 11) | (c0[:, 1] > 11)).mean()); over1 = 100 * float(((c1[:, 0] > 11) | (c1[:, 1] > 11)).mean())
        chk = 100 * info["loo_ok"] / max(info["loo_n"], 1)
        print(f"{i:4d} | {c0[:, 0].mean():8.1f} {c0[:, 1].mean():8.1f} {f0[:, 0].mean():6.1f} {f0[:, 1].mean():6.1f} {over0:4.0f}% | "
              f"{c1[:, 0].mean():7.1f} {c1[:, 1].mean():7.1f} {f1[:, 0].mean():6.1f} {f1[:, 1].mean():6.1f} {over1:4.0f}% | "
              f"{100 * info['changed'] / max(len(new), 1):9.1f}% {info['to_a'] + info['to_b']:11d} {info['a_to_b']:5d} {info['b_to_a']:5d} | {chk:5.0f}%")
        rows.append((c0, c1))
        if info["cells_skipped"]:
            print(f"     ({info['cells_skipped']} of {info['cells']} colour models skipped: too few labelled players in that window and distance)")
    if len(rows) > 1:
        m0 = [r[0][:, 0].mean() - r[0][:, 1].mean() for r in rows]; m1 = [r[1][:, 0].mean() - r[1][:, 1].mean() for r in rows]
        print(f"\nA minus B per moment across the parts: before {', '.join(f'{v:+.1f}' for v in m0)}   after {', '.join(f'{v:+.1f}' for v in m1)}")
        print(f"spread of that difference across parts (std): before {np.std(m0):.2f}, after {np.std(m1):.2f}")
    print("\nHow to read it:\n"
          "  A and B per moment: each should end near 9-11 (the board sees about 90% of the players) and not drift from part to part.\n"
          "  '>11'  = share of moments in which a team has more than 11 players: impossible, so a measure of wrong labels.\n"
          "  'other->team' = detections that were 'other' and are now a team (shirts that had fallen out of the colour groups).\n"
          "  'A->B' / 'B->A' = detections moved between the teams.\n"
          "  'check' = of the near-side seed players, how many the model gives back their own team when left out of it (should be ~100%).\n"
          "  If the counts do not settle, far-side colour cannot separate the two teams in that game.")


if __name__ == "__main__":
    main()
