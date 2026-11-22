#!/usr/bin/env python3
"""player_census.py — where do players disappear between the detector and the board?  (read-only)

The board shows about 8.7 players of team A and 10.3 of team B per moment, although 11 + 11 are on the pitch.
This script re-runs board_data.part_frames on the same detections several times, switching one step on at a time,
and counts the players per moment after each step:

  A  everything the detector found (confidence >= 0.2), no pitch outline, no clean-up, no smoothing
  B  only what the board shows by confidence (>= 0.3)
  C  + the pitch outline (people beside the pitch are left out)            (only if a --calib file is given)
  D  + clean-up: team label voted along the track, at most 11 a side
  E  + smoothing and filling short holes = the board as it is

Between two steps the number of a team's players drops where that step removes them. A team that is already low
in step A is a colour problem (shirts read as 'other'); one that drops at C or D is the outline or the 11-a-side cap.

  cd <your insta360_analysis folder>
  python player_census.py /Users/davidelsweiler/FootballAutopan/TSV_Buch_A-Jugend/

It looks in the folder for the detections (*.jsonl with 'det' in the name), the team labels (*team_labels.json), the
camera path (*.kf.json / *keyframes*.json) and a pitch outline (*calib*.json). If it finds several of one kind, or the
wrong one, name them:  --det X.jsonl --labels Y.json --kf Z.json --calib C.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import board_data as BD
    import team_labels as TL
except ImportError:
    sys.exit("Run this from your insta360_analysis folder (board_data.py must be importable).")

NAMES = {0: "team A", 1: "team B", 2: "other / unknown", 3: "referee", 4: "keeper A", 5: "keeper B"}


def find(folder: str, patterns, what: str, given):
    if given:
        return given
    hits = {p for pat in patterns for p in glob.glob(os.path.join(folder, pat))}
    hits = sorted((h for h in hits if ".meta." not in os.path.basename(h) and "sample" not in os.path.basename(h)),
                  key=lambda h: (-os.path.getsize(h), h))          # the biggest file is the whole game, not a sample
    if not hits:
        return None
    if len(hits) > 1:
        print(f"  several {what} files found, using {os.path.basename(hits[0])} (others: "
              f"{', '.join(os.path.basename(h) for h in hits[1:4])}). Use the option to choose.")
    return hits[0]


def census(frames, y_cut):
    """mean players per moment by team code, whole pitch and by thirds of the distance from the camera; share of full teams"""
    n = len(frames)
    cnt = np.zeros((n, 6)); near = np.zeros((n, 3)); far = np.zeros((n, 3))   # near/far: team A, team B, other
    for i, f in enumerate(frames):
        for (x, y, t) in f["players"]:
            if 0 <= t <= 5:
                cnt[i, t] += 1
            k = t if t in (0, 1) else 2
            if y <= y_cut[0]:
                near[i, k] += 1
            elif y >= y_cut[1]:
                far[i, k] += 1
    return cnt, near, far


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--det"); ap.add_argument("--labels"); ap.add_argument("--kf"); ap.add_argument("--calib")
    ap.add_argument("--board", help="the existing <game>_board.json, to print what its tidy-up reported")
    ap.add_argument("--assign", nargs="+", metavar="GROUP=WHO",
                    help="try colour groups as if assigned differently, without touching the labels file: e.g. --assign 6=B 5=A 11=REF 2=GKB "
                         "(WHO = A, B, OTHER, REF, GKA or GKB)")
    a = ap.parse_args()

    det = find(a.folder, ["*det*.jsonl"], "detections", a.det)
    lab = find(a.folder, ["*team_labels.json"], "team labels", a.labels)
    kf = find(a.folder, ["*.kf.json", "*keyframes*.json"], "keyframes", a.kf)
    cal = a.calib
    if not cal:                                           # only a calibration file that holds the pitch outline counts
        for p_ in sorted(glob.glob(os.path.join(a.folder, "*calib*.json"))):
            try:
                d_ = json.loads(Path(p_).read_text())
            except (OSError, ValueError):
                continue
            if isinstance(d_, dict) and any(d_.get(k) for k in ("pitch_polygon", "pixel_polygon", "auto_polygon")):
                cal = p_; break
    brd = find(a.folder, ["*_board.json"], "board", a.board)
    print("using:")
    for k, v in (("detections", det), ("team labels", lab), ("keyframes", kf), ("pitch outline", cal), ("board", brd)):
        print(f"  {k:14s} {v or '-'}")
    if not det or not lab:
        sys.exit("Need at least the detections and the team labels file (name them with --det / --labels).")

    if brd:
        try:
            st = json.loads(Path(brd).read_text()).get("tidy")
            if st:
                print("\nwhat the board export itself reported (tidy):")
                for k in ("players", "relabelled", "recovered", "outside", "to_other_team", "to_other", "weak_kept", "weak_dropped",
                          "filled", "keepers", "referees", "gated", "smoothed"):
                    if k in st:
                        print(f"  {k:14s} {st[k]}")
        except (OSError, ValueError):
            pass

    clusters = TL.Labelled(lab)
    for item in a.assign or []:
        try:
            g, who = item.split("="); g = int(g); who = who.upper()
            if who in ("A", "B", "OTHER"):
                clusters.group_team[g] = {"A": 0, "B": 1, "OTHER": -1}[who]; clusters.group_role[g] = -1
            elif who in ("REF", "GKA", "GKB"):
                clusters.group_team[g] = -1; clusters.group_role[g] = {"REF": 3, "GKA": 4, "GKB": 5}[who]; clusters.has_roles = True
            else:
                raise ValueError(who)
            print(f"  trying group {g} as {who}")
        except (ValueError, IndexError):
            sys.exit(f"--assign: '{item}' is not GROUP=WHO with WHO one of A, B, OTHER, REF, GKA, GKB and GROUP a number that exists")
    import auto_keyframes as AK
    hdr, raw = AK.load_detections(det, max_lon=98.0)
    nraw = len(raw)
    if nraw == 0:
        sys.exit("The detections file has no moments.")
    print(f"\n{nraw} moments in the detections file")
    for c in (0.2, 0.3, 0.5):
        m = np.mean([sum(1 for p in r["players"] if p[3] >= c) for r in raw])
        print(f"  people detected per moment at confidence >= {c}: {m:.1f}")
    readable = np.mean([sum(1 for p in r["players"] if p[3] >= 0.3 and len(p) > 5 and p[5] is not None and p[2] >= BD.MIN_HEIGHT) for r in raw])
    print(f"  of those at >= 0.3, with a readable shirt colour (box >= {BD.MIN_HEIGHT:.0f} px high): {readable:.1f}")

    def run(label, **kw):
        base = dict(min_conf=BD.MIN_CONF, min_height=BD.MIN_HEIGHT, k_angle=0.75, rescue=BD.RESCUE, calib=None, margin_m=BD.MARGIN_M,
                    tidy=False, max_team=BD.MAX_TEAM, team_margin_m=BD.TEAM_MARGIN_M, stats=None, feet=True, smooth=False,
                    a_attacks_right=None, box=None, low_conf=BD.MIN_CONF, fill=False, ball=False)
        base.update(kw)
        fr = BD.part_frames(det, kf, clusters, 0.0, 98.0, base.pop("min_conf"), base.pop("min_height"), base.pop("k_angle"),
                            base.pop("rescue"), base.pop("calib"), base.pop("margin_m"), base.pop("tidy"), base.pop("max_team"),
                            base.pop("team_margin_m"), base.pop("stats"), base.pop("feet"), base.pop("smooth"),
                            base.pop("a_attacks_right"), base.pop("box"), base.pop("low_conf"), base.pop("fill"), base.pop("ball"))
        return label, fr

    steps = [run("A  all detected (>= 0.2)", min_conf=0.2, low_conf=0.2)]
    steps.append(run("B  confidence >= 0.3", min_conf=0.3, low_conf=0.3))
    if cal:
        steps.append(run("C  + pitch outline", calib=cal))
    steps.append(run("D  + team vote, 11 a side", calib=cal, tidy=True))
    steps.append(run("E  = the board (smoothed, holes filled)", calib=cal, tidy=True, smooth=True, fill=True, low_conf=BD.LOW_CONF))

    ys = np.array([p[1] for f in steps[0][1] for p in f["players"]], float)
    y_cut = (float(np.percentile(ys, 33)), float(np.percentile(ys, 66))) if len(ys) else (0.0, 1e9)

    print(f"\nPLAYERS PER MOMENT AFTER EACH STEP (mean; '10+' = share of moments with 10 or 11 of that team)")
    print(f"{'step':42s} {'A':>5s} {'B':>5s} {'other':>6s} {'ref':>4s} {'gk':>4s} {'A 10+':>6s} {'B 10+':>6s}")
    rows = []
    for label, fr in steps:
        cnt, near, far = census(fr, y_cut)
        a_, b_, o_ = cnt[:, 0] + cnt[:, 4], cnt[:, 1] + cnt[:, 5], cnt[:, 2]
        print(f"{label:42s} {a_.mean():5.1f} {b_.mean():5.1f} {o_.mean():6.1f} {cnt[:, 3].mean():4.1f} {(cnt[:, 4] + cnt[:, 5]).mean():4.1f} "
              f"{100 * (a_ >= 10).mean():5.0f}% {100 * (b_ >= 10).mean():5.0f}%")
        rows.append((label, a_.mean(), b_.mean(), o_.mean(), near, far))
    print("\nNEAR vs FAR THIRD OF THE PITCH (players per moment: team A / team B / other)")
    for label, _, _, _, near, far in rows:
        print(f"{label:42s} near {near[:, 0].mean():4.1f} / {near[:, 1].mean():4.1f} / {near[:, 2].mean():4.1f}   far {far[:, 0].mean():4.1f} / {far[:, 1].mean():4.1f} / {far[:, 2].mean():4.1f}")

    if hasattr(clusters, "groups_of") and hasattr(clusters, "group_team"):
        pts = [p for r in raw for p in r["players"] if p[3] >= 0.3 and len(p) > 5 and p[5] is not None and p[2] >= BD.MIN_HEIGHT]
        if pts:
            g = clusters.groups_of(np.array([p[5] for p in pts], float))
            hh = np.array([p[2] for p in pts], float)
            role = getattr(clusters, "group_role", np.full(len(clusters.group_team), -1))
            print("\nCOLOUR GROUPS: who ends up in which group (detections >= 0.3 with a readable colour)")
            print(f"{'group':>5s}  {'assigned to':22s} {'per moment':>10s} {'share':>6s} {'median box px':>14s}")
            for j in np.argsort(-np.bincount(g, minlength=len(clusters.group_team))):
                n = int((g == j).sum())
                if n == 0:
                    continue
                who = {3: "referee", 4: "keeper A", 5: "keeper B"}.get(int(role[j])) or {0: "team A", 1: "team B"}.get(int(clusters.group_team[j]), "OTHER (not counted)")
                print(f"{j:5d}  {who:22s} {n / nraw:10.2f} {100 * n / len(pts):5.0f}% {np.median(hh[g == j]):14.0f}")
            print("  A group marked OTHER that holds several players per moment is probably a shirt of one team (in shade, far away,"
                  " or in a different light): give it to that team with team_labels.py assign, or compare it on the contact sheet.")

    A0, B0 = rows[0][1], rows[0][2]
    print("\nHow to read it:")
    print(f"  Step A is the ceiling: it shows what colour labelling alone gives of the people the detector found.")
    print(f"  If 'other' is large in step A, shirts are read as no team (colour groups / lighting), not lost by the detector.")
    print(f"  A drop between B and C is the pitch outline, between C and D the track vote or the 11-a-side cap.")
    if A0 < B0 - 1.0:
        print(f"  Team A is already {B0 - A0:.1f} players lower than B in step A: look at its colour groups first (team_labels.py).")
    print()


if __name__ == "__main__":
    main()
