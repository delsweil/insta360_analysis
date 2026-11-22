#!/usr/bin/env python3
"""colour_tracks.py — decide the team of a whole track from its average shirt colour.  (read-only)

One far-side detection is a weak witness of the team (colour error 15-28%). A track of 12 s holds some 30 of them, and if
most of their noise is random from moment to moment, the average of a track is a much better witness. This tests that.

For each part:
  1. TRACKS    position-only tracks (track_stitch.py, never joining two pieces whose reliable shirts disagree).
  2. AVERAGE   the median shirt colour along each track (only detections with a readable colour).
  3. CLUSTER   in every window of time (default 5 min) and distance third, the track averages are split into two colour
               groups (team A / team B) by k-means. Tracks that hold >= 3 clear NEAR-side shirts of one team are SEEDS: they
               keep that team and give the groups their starting colours, so the groups cannot swap names or drift into
               'light shirts / dark shirts'. Everything else follows the nearest group, but only when it is clearly nearer
               to one than to the other (ratio >= 1.3); an unlike colour (referee, coaches, kit of someone else) stays 'other'.
  4. RELABEL   every detection on a sure track takes the track's team. Referee, keepers and detections on no track keep their label.

Nothing is written. The test needs no ground truth: team counts should settle at 9-11 each and stop drifting from part to part.

  cd <your insta360_analysis folder>
  python3 colour_tracks.py /Users/davidelsweiler/FootballAutopan/TSV_Buch_A-Jugend/ --assign 6=B 5=REF

Needs track_stitch.py, colour_refit.py and colour_separation.py in the same folder.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

try:
    import scene_check as S
    import team_labels as TL
    import track_stitch as TS
    import colour_refit as CR
    import colour_separation as CS
except ImportError:
    sys.exit("Run this from your insta360_analysis folder, with track_stitch.py, colour_refit.py and colour_separation.py in it.")


def track_means(P, tid, feats, readable, seed, min_read=6):
    """-> list of dicts per track: idx, mean colour (median), n readable, mean y, mid time, seed team, old side"""
    vt, fr, y, lab = P["vt"], P["fr"], P["y"], P["lab"]
    out = []
    order = np.argsort(tid, kind="stable"); ts_ = tid[order]
    cuts = np.r_[0, np.flatnonzero(np.diff(ts_)) + 1, len(order)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        idx = order[a:b]
        if tid[idx[0]] < 0:
            continue
        rd = idx[readable[idx]]
        if len(rd) < min_read:
            continue
        wa = int(((lab[rd] == 0) & seed[rd]).sum()); wb = int(((lab[rd] == 1) & seed[rd]).sum())
        st = (0 if wa >= wb else 1) if (wa + wb >= 3 and max(wa, wb) / (wa + wb) >= 0.8) else -1
        cnt = np.array([(lab[idx] == c).sum() for c in range(6)], float)
        out.append(dict(idx=idx, f=np.median(feats[rd], axis=0), n=len(rd), y=float(np.mean(y[idx])), t=float(np.mean(vt[fr[idx]])),
                        seed=st, old=int(cnt.argmax())))
    return out


def kmeans_fixed(X, w, seed, cA, cB, iters=12):
    """two-means with fixed seed assignments. seed: 0/1/-1. -> (centres (2,3), assignment, distance to A, to B)"""
    c = np.array([cA, cB], float)
    for _ in range(iters):
        d = np.sqrt(((X[:, None, :] - c[None]) ** 2).sum(-1))
        asg = np.where(seed >= 0, seed, d.argmin(axis=1))
        new = c.copy()
        for k in (0, 1):
            m = asg == k
            if m.sum() >= 3:
                new[k] = (X[m] * w[m, None]).sum(0) / w[m].sum()
        if np.allclose(new, c, atol=1e-3):
            break
        c = new
    d = np.sqrt(((X[:, None, :] - c[None]) ** 2).sum(-1))
    return c, np.where(seed >= 0, seed, d.argmin(axis=1)), d[:, 0], d[:, 1]


def relabel_part(P, window_s=300.0, bands=3, ratio=1.3, min_seed=5):
    vt, fr, x, y, bgr, lab, margin = P["vt"], P["fr"], P["x"], P["y"], P["bgr"], P["lab"], P["margin"]
    n = len(fr)
    new = lab.copy()
    info = dict(tracks=0, sure=0, cells=0, skipped=0, det_changed=0, other_to_team=0, a_to_b=0, b_to_a=0, seed_ok=0, seed_n=0, tid=None, tm=None)
    if n == 0:
        return new, info
    readable = np.isfinite(bgr[:, 0])
    feats = np.full((n, 3), np.nan)
    feats[readable] = S.colour_features(bgr[readable])
    tid = TS.stitch(fr, vt, x, y, lab, TS.link(fr, vt, x, y, lab), hard_clash=True)
    seed = readable & np.isin(lab, (0, 1)) & (margin >= 0.4) & (y <= np.percentile(y, 100 / 3))
    tm = track_means(P, tid, feats, readable, seed)
    info["tracks"] = len(tm)
    if not tm:
        return new, info
    F = np.array([t["f"] for t in tm]); W = np.sqrt([t["n"] for t in tm]); SEED = np.array([t["seed"] for t in tm])
    Y = np.array([t["y"] for t in tm]); T = np.array([t["t"] for t in tm]); OLD = np.array([t["old"] for t in tm])
    side_old = np.where(np.isin(OLD, (0, 4)), 0, np.where(np.isin(OLD, (1, 5)), 1, np.where(OLD == 3, 3, 2)))
    cand = np.isin(OLD, (0, 1, 2)) | (SEED >= 0)                    # referee / keeper tracks keep their role
    edges = np.percentile(y, np.linspace(0, 100, bands + 1)[1:-1])
    band = np.searchsorted(edges, Y)
    win = np.floor((T - vt[0]) / window_s).astype(int)
    t_label = np.full(len(tm), -1, int)

    def seeds_mean(sel):
        a = sel & (SEED == 0); b = sel & (SEED == 1)
        if a.sum() >= min_seed and b.sum() >= min_seed:
            return (F[a] * W[a, None]).sum(0) / W[a].sum(), (F[b] * W[b, None]).sum(0) / W[b].sum()
        return None

    near_all = seeds_mean(np.ones(len(tm), bool))
    for w_ in np.unique(win):
        for b_ in range(bands):
            info["cells"] += 1
            sel = (win == w_) & (band == b_) & cand
            if sel.sum() < 12:
                info["skipped"] += 1; continue
            init = seeds_mean(sel) or seeds_mean((win == w_) & cand) or near_all
            if init is None:
                info["skipped"] += 1; continue
            ii = np.flatnonzero(sel)
            c, asg, dA, dB = kmeans_fixed(F[ii], W[ii], SEED[ii], init[0], init[1])
            near_d = np.where(asg == 0, dA, dB)
            typical = float(np.median(near_d[SEED[ii] < 0])) if (SEED[ii] < 0).any() else float(np.median(near_d))
            r = np.maximum(dA, dB) / np.maximum(np.minimum(dA, dB), 1e-6)
            for j, t_ in enumerate(ii):
                if SEED[t_] >= 0:
                    t_label[t_] = SEED[t_]
                    info["seed_n"] += 1; info["seed_ok"] += int(np.argmin([dA[j], dB[j]]) == SEED[t_])
                elif r[j] >= ratio and (side_old[t_] < 2 or near_d[j] <= 2.0 * typical):
                    t_label[t_] = asg[j]
    for j, t_ in enumerate(tm):
        if t_label[j] < 0:
            continue
        info["sure"] += 1
        idx = t_["idx"]
        keep = np.isin(lab[idx], (0, 1, 2))                         # keepers / referee on the track stay
        for q in idx[keep]:
            if new[q] != t_label[j]:
                if lab[q] == 2: info["other_to_team"] += 1
                elif lab[q] == 0: info["a_to_b"] += 1
                else: info["b_to_a"] += 1
                new[q] = t_label[j]
    info["det_changed"] = int((new != lab).sum())
    info["tid"] = tid; info["tm"] = (F, SEED, Y, t_label)
    return new, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--det"); ap.add_argument("--kf"); ap.add_argument("--labels"); ap.add_argument("--calib")
    ap.add_argument("--assign", nargs="+", metavar="GROUP=WHO")
    ap.add_argument("--window", type=float, default=300.0, help="seconds per colour group (default 300)")
    ap.add_argument("--ratio", type=float, default=1.3, help="how much nearer the nearer group must be (default 1.3)")
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
    print(f"\n{'part':>4s} | {'A before':>8s} {'B before':>8s} {'A far':>6s} {'B far':>6s} {'>11':>5s} | {'A after':>7s} {'B after':>7s} {'A far':>6s} {'B far':>6s} {'>11':>5s} | "
          f"{'tracks':>6s} {'sure':>6s} {'relabelled':>10s} {'other->team':>11s} {'A->B':>5s} {'B->A':>5s} | {'seeds ok':>8s}")
    rows = []
    cals = TS.find_outlines(a.folder, dets, a.calib)
    print("pitch outlines:", ", ".join(os.path.basename(c) if c else "none" for c in cals), "" if all(cals) else "  (people beside the pitch are not filtered where none)")
    for i, (d, k_) in enumerate(dets):
        try:
            P = CR.points_ex(d, k_, clusters, cals[i])
        except (TypeError, ValueError, KeyError, OSError) as e:
            print(f"  ! pitch outline {os.path.basename(cals[i] or '')} could not be used ({type(e).__name__}: {e}); continuing without it")
            P = CR.points_ex(d, k_, clusters, None)
        print(f"     outline: {CR.outline_note(P)}")
        new, info = relabel_part(P, a.window, ratio=a.ratio)
        c0, f0 = CR.counts(P, P["lab"]); c1, f1 = CR.counts(P, new)
        over0 = 100 * float(((c0[:, 0] > 11) | (c0[:, 1] > 11)).mean()); over1 = 100 * float(((c1[:, 0] > 11) | (c1[:, 1] > 11)).mean())
        print(f"{i:4d} | {c0[:, 0].mean():8.1f} {c0[:, 1].mean():8.1f} {f0[:, 0].mean():6.1f} {f0[:, 1].mean():6.1f} {over0:4.0f}% | "
              f"{c1[:, 0].mean():7.1f} {c1[:, 1].mean():7.1f} {f1[:, 0].mean():6.1f} {f1[:, 1].mean():6.1f} {over1:4.0f}% | "
              f"{info['tracks']:6d} {100 * info['sure'] / max(info['tracks'], 1):5.0f}% {100 * info['det_changed'] / max(len(new), 1):9.1f}% {info['other_to_team']:11d} "
              f"{info['a_to_b']:5d} {info['b_to_a']:5d} | {100 * info['seed_ok'] / max(info['seed_n'], 1):7.0f}%")
        if info["skipped"]:
            print(f"     ({info['skipped']} of {info['cells']} colour groups skipped: too few tracks in that window and distance third)")
        rows.append((c0, c1))
        if info["tm"] is not None:                                  # separation of the track averages themselves, where a reference exists
            F, SEED, Y, _ = info["tm"]
            edges = np.percentile(P["y"], [33.3, 66.6]); bd = np.searchsorted(edges, Y)
            parts_ = []
            for b_, nm in enumerate(("near", "middle", "far")):
                r = CS.separation(F[(SEED == 0) & (bd == b_)], F[(SEED == 1) & (bd == b_)], min_n=10)
                parts_.append(f"{nm} d'={r[0]:.1f}" if r else f"{nm} -")
            print("     track averages, seed tracks:", "  ".join(parts_))
    if len(rows) > 1:
        m0 = [r[0][:, 0].mean() - r[0][:, 1].mean() for r in rows]; m1 = [r[1][:, 0].mean() - r[1][:, 1].mean() for r in rows]
        print(f"\nA minus B per moment across the parts: before {', '.join(f'{v:+.1f}' for v in m0)}   after {', '.join(f'{v:+.1f}' for v in m1)}")
        print(f"spread of that difference across parts (std): before {np.std(m0):.2f}, after {np.std(m1):.2f}")
    print("\nHow to read it:\n"
          "  A and B per moment should end near 9-11 and stop drifting from part to part; '>11' (impossible) should fall.\n"
          "  'sure' = share of tracks given a team; 'other->team' = detections that were 'other' and are now a team; 'A->B'/'B->A' = moved between teams.\n"
          "  'seeds ok' = seed tracks that the final colour groups give back their own team: should be ~100%.\n"
          "  'track averages, seed tracks' = d' of the track-averaged colours of the near-side reference tracks. Compare with the\n"
          "  per-detection values of colour_separation.py (near about 3): a clear rise means averaging helps.\n"
          "  If the counts do not settle, the far-side colours of the two teams are not separable even when averaged: that is\n"
          "  a limit of light and detail, which is where the X5 would have to help.")


if __name__ == "__main__":
    main()
