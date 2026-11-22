#!/usr/bin/env python3
"""colour_separation.py — can shirt colour tell the two teams apart at each distance and time?  (read-only)

The colour refit did not fix the team counts. Before building more on colour, this measures whether the colour of team A and
team B is separable at all on the far side and late in a game, and how that depends on the size of the box.

Reference teams: players whose track holds enough clear NEAR-side shirts (the nearest third of the pitch, at least 3 of one
team, 80% agreement). Every detection on such a track within 8 s of such a shirt, near or far, counts as a member of that team. This is a stand-in for
the truth, not the truth: a track that swapped two players is wrong for them.

For each part, early/late half of the part, and distance third it prints:
  nA / nB    members of each team found there
  d'         distance between the two teams' mean colours, in units of their spread (0 = same, 2 = clearly apart, 4 = easy)
  error      the share of these members a straight-line colour classifier gets wrong (train = test, so slightly optimistic)
  A / B move how far each team's colour has moved from the near side of the first half (in colour units; compare with d')
  old wrong  how often the board's fixed colour groups disagree with the reference, among detections they call A or B

and for all parts together the same separation by BOX HEIGHT, which is where the X5 matters: an X5 box is about 1.5 times as
high as an X2 box of the same player (flat 16:9 mode), so a 28 px far player on the X2 is about 42 px on the X5.

  cd <your insta360_analysis folder>
  python3 colour_separation.py /Users/davidelsweiler/FootballAutopan/TSV_Buch_A-Jugend/ --assign 6=B 5=REF

Needs track_stitch.py and colour_refit.py in the same folder.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np

try:
    import scene_check as S
    import team_labels as TL
    import track_stitch as TS
    import colour_refit as CR
except ImportError:
    sys.exit("Run this from your insta360_analysis folder, with track_stitch.py and colour_refit.py in it.")

H_BINS = [14, 20, 26, 32, 40, 52, 1e9]


def reference_labels(P, near_frac=1 / 3, seed_margin=0.4, min_votes=3, share=0.8, reach_s=8.0):
    """team 0/1 of each detection from its track's near-side seeds, -1 where unknown"""
    vt, fr, x, y, bgr, lab, margin = P["vt"], P["fr"], P["x"], P["y"], P["bgr"], P["lab"], P["margin"]
    n = len(fr)
    ref = np.full(n, -1, int)
    if n == 0:
        return ref
    readable = np.isfinite(bgr[:, 0])
    tid = TS.stitch(fr, vt, x, y, lab, TS.link(fr, vt, x, y, lab))
    seed = readable & np.isin(lab, (0, 1)) & (margin >= seed_margin) & (y <= np.percentile(y, 100 * near_frac))
    order = np.argsort(tid, kind="stable"); ts_ = tid[order]
    cuts = np.r_[0, np.flatnonzero(np.diff(ts_)) + 1, n]
    for a, b in zip(cuts[:-1], cuts[1:]):
        idx = order[a:b]
        if tid[idx[0]] < 0:
            continue
        wa = int(((lab[idx] == 0) & seed[idx]).sum()); wb = int(((lab[idx] == 1) & seed[idx]).sum())
        if wa + wb >= min_votes and max(wa, wb) / (wa + wb) >= share:
            team = 0 if wa >= wb else 1
            idx = idx[np.argsort(vt[fr[idx]], kind="stable")]
            tt = vt[fr[idx]]
            st = tt[seed[idx] & (lab[idx] == team)]               # only detections within reach_s of a seed of the team count:
            if len(st) == 0:                                      # a long track may have swapped players, a short stretch hardly
                continue
            j = np.searchsorted(st, tt)
            d_lo = np.abs(tt - st[np.clip(j - 1, 0, len(st) - 1)])
            d_hi = np.abs(tt - st[np.clip(j, 0, len(st) - 1)])
            near_ok = np.minimum(d_lo, d_hi) <= reach_s
            ref[idx[near_ok]] = team
    return ref


def separation(fa, fb, min_n=30):
    """-> (d', error share) of two colour clouds, or None if too few"""
    if len(fa) < min_n or len(fb) < min_n:
        return None
    ma, mb = fa.mean(0), fb.mean(0)
    cov = (np.cov(fa.T) * (len(fa) - 1) + np.cov(fb.T) * (len(fb) - 1)) / (len(fa) + len(fb) - 2) + 1e-6 * np.eye(3)
    w = np.linalg.solve(cov, ma - mb)
    d = math.sqrt(max(float((ma - mb) @ w), 0.0))
    sa, sb = fa @ w, fb @ w
    thr = 0.5 * (sa.mean() + sb.mean())
    err = 0.5 * float((sa < thr).mean()) + 0.5 * float((sb >= thr).mean())      # balanced error
    return d, err


def table_for_part(P):
    ref = reference_labels(P)
    bgr, y, vt, fr, lab = P["bgr"], P["y"], P["vt"], P["fr"], P["lab"]
    readable = np.isfinite(bgr[:, 0])
    feats = np.full((len(fr), 3), np.nan)
    feats[readable] = S.colour_features(bgr[readable])
    t = vt[fr]
    half = (t >= np.median(vt)).astype(int)
    edges = np.percentile(y, [33.3, 66.6])
    band = np.searchsorted(edges, y)
    use = readable & (ref >= 0)
    base = {}
    for c in (0, 1):
        sel = use & (half == 0) & (band == 0) & (ref == c)
        base[c] = feats[sel].mean(0) if sel.sum() >= 30 else None
    rows = []
    for hf in (0, 1):
        for b in (0, 1, 2):
            sel = use & (half == hf) & (band == b)
            fa, fb = feats[sel & (ref == 0)], feats[sel & (ref == 1)]
            r = separation(fa, fb)
            mv = [(float(np.linalg.norm(f.mean(0) - base[c])) if (len(f) >= 30 and base[c] is not None) else None) for c, f in ((0, fa), (1, fb))]
            ab = sel & np.isin(lab, (0, 1))
            wrong = float((lab[ab] != ref[ab]).mean()) if ab.sum() >= 30 else None
            rows.append((hf, b, len(fa), len(fb), r, mv, wrong))
    return rows, (feats, ref, use, P["h"])


def fmt(v, f="{:.1f}"):
    return "-" if v is None else f.format(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--det"); ap.add_argument("--kf"); ap.add_argument("--labels"); ap.add_argument("--calib")
    ap.add_argument("--assign", nargs="+", metavar="GROUP=WHO")
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
    print(f"\n{'part':>4s} {'half':>5s} {'third':>6s} {'nA':>6s} {'nB':>6s} {'d prime':>8s} {'error':>7s} {'A move':>7s} {'B move':>7s} {'old wrong':>10s}")
    pooled = []
    cals = TS.find_outlines(a.folder, dets, a.calib)
    print("pitch outlines:", ", ".join(os.path.basename(c) if c else "none" for c in cals), "" if all(cals) else "  (people beside the pitch are not filtered where none)")
    for i, (d, k_) in enumerate(dets):
        try:
            P = CR.points_ex(d, k_, clusters, cals[i])
        except (TypeError, ValueError, KeyError, OSError) as e:
            print(f"  ! pitch outline {os.path.basename(cals[i] or '')} could not be used ({type(e).__name__}: {e}); continuing without it")
            P = CR.points_ex(d, k_, clusters, None)
        print(f"     outline: {CR.outline_note(P)}")
        rows, pool = table_for_part(P)
        pooled.append(pool)
        for hf, b, na, nb, r, mv, wrong in rows:
            print(f"{i:4d} {('early', 'late')[hf]:>5s} {('near', 'middle', 'far')[b]:>6s} {na:6d} {nb:6d} {fmt(r[0] if r else None):>8s} "
                  f"{fmt(100 * r[1] if r else None, '{:.0f}%'):>7s} {fmt(mv[0]):>7s} {fmt(mv[1]):>7s} {fmt(100 * wrong if wrong is not None else None, '{:.0f}%'):>10s}")
    print("\nSEPARATION BY BOX HEIGHT (all parts together; box height in the pixels of the detection image)")
    print(f"{'box px':>9s} {'nA':>7s} {'nB':>7s} {'d prime':>8s} {'error':>7s}")
    feats = np.concatenate([p[0] for p in pooled]); ref = np.concatenate([p[1] for p in pooled]); use = np.concatenate([p[2] for p in pooled])
    h = np.concatenate([p[3] for p in pooled])
    res = []
    for lo, hi in zip(H_BINS[:-1], H_BINS[1:]):
        sel = use & (h >= lo) & (h < hi)
        r = separation(feats[sel & (ref == 0)], feats[sel & (ref == 1)])
        res.append((lo, hi, r))
        lab_ = f"{lo:.0f}-{hi:.0f}" if hi < 1e8 else f"{lo:.0f}+"
        print(f"{lab_:>9s} {int((sel & (ref == 0)).sum()):7d} {int((sel & (ref == 1)).sum()):7d} {fmt(r[0] if r else None):>8s} {fmt(100 * r[1] if r else None, '{:.0f}%'):>7s}")
    print("\nHow to read it:\n"
          "  d prime near 3-4 and error under 5%: colour separates the teams there. Under about 1.5 or error over 15%: it does not.\n"
          "  'A move' / 'B move' compared with d prime: a team's colour that has moved by more than the gap between the teams has crossed over.\n"
          "  'old wrong' = how often the board's colour groups disagree with the track reference where they say A or B.\n"
          "  By box height: if d prime rises steeply with height, the extra detail of the X5 should help; if it is flat, the\n"
          "  limit is light and shirt colour, not pixels. This is read from X2 footage and assumes that the height effect is\n"
          "  about detail and not distance or light, which it cannot separate completely.")


if __name__ == "__main__":
    main()
