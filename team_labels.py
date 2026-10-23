#!/usr/bin/env python3
"""team_labels.py — tell the program by eye which shirts belong to which team (floodlights, odd light).

Colours are unreliable under floodlights, and the same shirt looks different in a bright and a dark part of
the pitch. So instead of trusting colour names, you label what you see:

  1) make a contact sheet:   python team_labels.py make --video V.mp4 --detections g030.detections.jsonl --out g030
        -> g030.teams_sheet.png   one row per colour group, with swatch, size and a dozen real players of it
           g030.team_labels.json  the colour groups (and an empty assignment)
  2) look at the sheet and say which groups are which:
        python team_labels.py assign --labels g030.team_labels.json --a 0 3 --b 1 5 \\
               --name-a "Yellow" --name-b "Dark" --colour-a "#f2d04a" --colour-b "#3d7bd9"
        (groups you do not name are 'other': referee, goalkeepers, coaches, odd light)
  3) use it:   python team_stats.py --detections ... --team-labels g030.team_labels.json ...

Several groups per team are fine and useful: one team usually shows up as a bright group and a dark group
(light and shade); give both to it. The colour groups come from k-means on the shirt colours of players seen in
about 40 frames spread over the sample; --k (default 8) is the number of groups.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import numpy as np

import kf_core as K
import extract_detections as X
import scene_check as S
import football_style as FS

COLOURS = {"white": "#ececec", "blue": "#3d7bd9", "red": "#d9392b", "yellow": "#f2d04a", "green": "#3fb865",
           "orange": "#f08a2c", "black": "#8e8e93", "dark": "#8e8e93", "purple": "#9b59b6", "grey": "#b8b8bd"}


def hex_colour(v: str) -> str:
    v = (v or "").strip()
    return COLOURS.get(v.lower(), v)


def crop_of(frame: np.ndarray, outline, pad: float = 0.12, size=(48, 96)) -> np.ndarray:
    """The player cut out of the unwrapped frame, from the quadrilateral of the detection box."""
    import cv2
    xs = [p[0] for p in outline]
    ys = [p[1] for p in outline]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    x0, x1 = max(0, int(x0 - pad * w)), min(frame.shape[1], int(x1 + pad * w) + 1)
    y0, y1 = max(0, int(y0 - pad * h)), min(frame.shape[0], int(y1 + pad * h) + 1)
    c = frame[y0:y1, x0:x1]
    if c.size == 0:
        return np.zeros((size[1], size[0], 3), np.uint8)
    return cv2.resize(c, size, interpolation=cv2.INTER_AREA)


def build_groups(colours: np.ndarray, k: int, seed: int):
    """k-means on the shirt colours -> centres, labels, sizes (largest group first)."""
    feats = S.colour_features(colours)
    centres, lab = S.kmeans(feats, k, seed)
    order = np.argsort(-np.bincount(lab, minlength=len(centres)))
    centres = centres[order]
    lab = np.argmin(((feats[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
    return centres, lab, np.bincount(lab, minlength=len(centres))


def strip(g, per_row: int = 14) -> np.ndarray:
    """One row of real player crops of a colour group (BGR picture)."""
    row = np.full((96, 56 * per_row, 3), 38, np.uint8)
    for i, c in enumerate(g["crops"][:per_row]):
        row[:, 56 * i: 56 * i + 48] = c
    return row


def suggest(centres: np.ndarray, sizes: np.ndarray, min_share: float = 4.0):
    """A first guess for the coach to confirm: the biggest group is team A, the big group farthest from it is
    team B, and every other sizeable group joins whichever is nearer if it is clearly nearer. Rest: other."""
    n = len(centres)
    share = 100.0 * np.asarray(sizes, float) / max(1.0, float(np.sum(sizes)))
    if n < 2:
        return {"A": list(range(n)), "B": []}
    d = np.linalg.norm(centres - centres[0], axis=1)
    cand = [j for j in range(1, n) if share[j] >= 8.0] or list(range(1, n))
    b0 = max(cand, key=lambda j: d[j])
    A, B = [0], [b0]
    ab = float(np.linalg.norm(centres[0] - centres[b0]))
    for j in range(1, n):
        if j == b0 or share[j] < min_share:
            continue
        da, db = float(np.linalg.norm(centres[j] - centres[0])), float(np.linalg.norm(centres[j] - centres[b0]))
        if min(da, db) < 0.6 * ab and max(da, db) > 1.3 * min(da, db):
            (A if da < db else B).append(j)
    return {"A": sorted(A), "B": sorted(B)}


def quality(meta: dict) -> dict:
    """How well the confirmed groups cover the sampled players: shares of team A, team B and 'other'."""
    sizes = np.array(meta["sizes"], float)
    tot = max(1.0, sizes.sum())
    a = float(sizes[meta["assign"]["A"]].sum() / tot) if meta["assign"]["A"] else 0.0
    b = float(sizes[meta["assign"]["B"]].sum() / tot) if meta["assign"]["B"] else 0.0
    warn = []
    if not meta["assign"]["A"] or not meta["assign"]["B"]:
        warn.append("not_assigned")
    else:
        if min(a, b) < 0.5 * max(a, b):
            warn.append("unbalanced")
        if 1.0 - a - b > 0.35:
            warn.append("many_other")
    return {"a": a, "b": b, "other": max(0.0, 1.0 - a - b), "warnings": warn}


def sheet(groups, out: str, title: str, per_row: int = 14):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(15, 1.9 * n + 0.9), facecolor=FS.BG)
    axes = np.atleast_1d(axes)
    fig.suptitle(title, color=FS.TEXT, fontsize=14, x=0.01, ha="left")
    for ax, g in zip(axes, groups):
        ax.set_facecolor(FS.PANEL)
        ax.axis("off")
        row = strip(g, per_row)
        ax.imshow(row[:, :, ::-1], extent=[3.0, 3.0 + per_row * 1.0, 0, 1.0], aspect="auto")
        ax.add_patch(plt.Rectangle((0.0, 0.1), 0.8, 0.8, color=np.array(g["mean_bgr"][::-1]) / 255.0, ec=FS.TEXT, lw=1))
        ax.text(1.05, 0.62, f"group {g['index']}", color=FS.TEXT, fontsize=15, fontweight="bold", va="center")
        ax.text(1.05, 0.25, f"{g['size']} players ({g['share']:.0f} %)", color=FS.MUTED, fontsize=10, va="center")
        ax.set_xlim(0, 3.0 + per_row)
        ax.set_ylim(0, 1)
    fig.savefig(out, dpi=100, bbox_inches="tight", facecolor=FS.BG)
    plt.close(fig)


def make(a, model=None, grab=None) -> int:
    header, frames = S.read_detections(a.detections)
    iv_fov = a.iv_fov if a.iv_fov is not None else header.get("iv_fov")
    raw = header.get("raw_input", True)
    square = bool(header.get("square_views", iv_fov not in (None, K.DEFAULT_IV_FOV)))
    fy = S.SQUARE_FY if square else S.LEGACY_FY
    tilt = a.tilt if a.tilt is not None else float(header.get("tilt", -20.0))
    lo, hi = (min(header["view_lons"]) - 20, max(header["view_lons"]) + 20)
    views = X.make_views(lo, hi, tilt, 40.0, legacy_fov=not square)
    if model is None:
        from ultralytics import YOLO
        model = YOLO(a.players)
    device = X.pick_device(a.device)
    grab = grab or (lambda t: S.grab_frame(a.video, t, iv_fov, raw))
    ts = S.fit_times(frames, a.frames, a.seed)
    print(f"{len(ts)} frames, views {len(views)} tilt {tilt:+.1f} square={square}")
    cols, crops, sizes_px = [], [], []
    for i, t in enumerate(ts):
        fr = grab(t)
        if fr is None:
            continue
        for p in S.detect_scene(fr, views, model, device, fy, 0.75, X.CONF_PLAYERS, a.max_lon):
            if p["conf"] >= 0.5 and p["h"] >= a.min_height and not np.isnan(p["bgr"]).any():
                cols.append(p["bgr"])
                crops.append(crop_of(fr, p["outline"]))
        print(f"  frame {i + 1}/{len(ts)}  {len(cols)} players so far", end="\r", flush=True)
    print()
    if len(cols) < 3 * a.k:
        print("too few players found")
        return 1
    cols = np.array(cols)
    centres, lab, sizes = build_groups(cols, a.k, a.seed)
    feats = S.colour_features(cols)
    groups = []
    for j in range(len(centres)):
        idx = np.where(lab == j)[0]
        d = np.linalg.norm(feats[idx] - centres[j], axis=1)
        take = idx[np.argsort(d)][: a.per_group * 2: 2][: a.per_group]          # typical members, every second one
        groups.append({"index": j, "size": int(sizes[j]), "share": 100.0 * sizes[j] / len(cols),
                       "mean_bgr": cols[idx].mean(0).tolist(), "crops": [crops[i] for i in take]})
    out_png = a.out + ".teams_sheet.png"
    sheet(groups, out_png, f"{os.path.basename(a.video)}: colour groups of the shirts (which are the two teams?)")
    import cv2
    gdir = a.out + ".groups"
    os.makedirs(gdir, exist_ok=True)
    for g in groups:
        cv2.imwrite(os.path.join(gdir, f"g{g['index']}.png"), strip(g, a.per_group))
    meta = {"note": "centres are Lab features (lightness x0.4) of the shirt colour; edit with team_labels.py assign",
            "centres": centres.tolist(), "sizes": sizes.tolist(), "mean_bgr": [g["mean_bgr"] for g in groups],
            "suggest": suggest(centres, sizes), "strips": os.path.basename(gdir),
            "assign": {"A": [], "B": []}, "names": {"A": "team A", "B": "team B"},
            "colours": {"A": FS.TEAM_B, "B": FS.TEAM_A}}
    with open(a.out + ".team_labels.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"wrote {out_png} and {a.out}.team_labels.json -> look at the sheet, then run 'assign'")
    return 0


def assign(a) -> int:
    with open(a.labels) as fh:
        meta = json.load(fh)
    n = len(meta["centres"])
    for key, groups in (("A", a.a), ("B", a.b)):
        if groups is not None:
            bad = [g for g in groups if not 0 <= g < n]
            if bad:
                raise SystemExit(f"no such group(s): {bad} (there are {n}: 0..{n - 1})")
            meta["assign"][key] = sorted(set(groups))
    if set(meta["assign"]["A"]) & set(meta["assign"]["B"]):
        raise SystemExit("a group cannot belong to both teams")
    for key, name, colour in (("A", a.name_a, a.colour_a), ("B", a.name_b, a.colour_b)):
        if name:
            meta["names"][key] = name
        if colour:
            meta["colours"][key] = hex_colour(colour)
    with open(a.labels, "w") as fh:
        json.dump(meta, fh, indent=1)
    other = [j for j in range(n) if j not in meta["assign"]["A"] + meta["assign"]["B"]]
    print(f"team A '{meta['names']['A']}': groups {meta['assign']['A']}   team B '{meta['names']['B']}': "
          f"groups {meta['assign']['B']}   other: {other}")
    return 0


class Labelled:
    """Drop-in for scene_check.Clusters in team_stats: nearest labelled colour group -> team 0 (A), 1 (B) or -1."""

    def __init__(self, path: str):
        with open(path) as fh:
            m = json.load(fh)
        if not m["assign"]["A"] or not m["assign"]["B"]:
            raise SystemExit(f"{path}: assign the groups to the teams first (team_labels.py assign)")
        self.centres = np.array(m["centres"], float)
        self.group_team = np.full(len(self.centres), -1)
        for j in m["assign"]["A"]:
            self.group_team[j] = 0
        for j in m["assign"]["B"]:
            self.group_team[j] = 1
        sizes = np.array(m["sizes"], float)
        mb = np.array(m["mean_bgr"], float)
        self.sizes = np.array([sizes[self.group_team == t].sum() for t in (0, 1)]).astype(int)
        self.mean_bgr = np.array([np.average(mb[self.group_team == t], axis=0, weights=sizes[self.group_team == t])
                                  for t in (0, 1)])
        self.names = [m["names"]["A"], m["names"]["B"]]
        self.colours = [m["colours"]["A"], m["colours"]["B"]]

    def team_of(self, bgr: np.ndarray) -> np.ndarray:
        f = S.colour_features(bgr)
        g = np.argmin(((f[:, None, :] - self.centres[None]) ** 2).sum(-1), axis=1)
        return self.group_team[g]

    def groups_of(self, bgr: np.ndarray) -> np.ndarray:
        """Index of the nearest colour group of each colour."""
        f = S.colour_features(bgr)
        return np.argmin(((f[:, None, :] - self.centres[None]) ** 2).sum(-1), axis=1)

    def team_distances(self, bgr: np.ndarray) -> np.ndarray:
        """(n, 3): distance of each colour to the nearest group of team A, of team B and of any other group (inf = none)."""
        f = S.colour_features(bgr)
        d = np.sqrt(((f[:, None, :] - self.centres[None]) ** 2).sum(-1))
        out = np.full((len(d), 3), np.inf)
        for c in (0, 1, 2):
            sel = (self.group_team == c) if c < 2 else (self.group_team < 0)
            if sel.any():
                out[:, c] = d[:, sel].min(axis=1)
        return out

    def team_of_rescue(self, bgr: np.ndarray, ratio: float = 1.5) -> np.ndarray:
        """Like team_of, but a colour whose nearest group is an 'other' one (referee, goalkeeper, or a shirt seen small and
        washed out by the grass) still gets a team when a team group is almost as near: distance to the nearest team group
        <= ratio x distance to the nearest group. A clearly different colour (referee) stays -1. ratio <= 1 = team_of."""
        f = S.colour_features(bgr)
        d = np.sqrt(((f[:, None, :] - self.centres[None]) ** 2).sum(-1))
        g = np.argmin(d, axis=1)
        team = self.group_team[g].copy()
        if ratio <= 1.0 or not (self.group_team >= 0).any():
            return team
        tg = np.where(self.group_team >= 0)[0]
        best = tg[np.argmin(d[:, tg], axis=1)]
        near_d = d[np.arange(len(d)), g]
        team_d = d[np.arange(len(d)), best]
        rescue = (team < 0) & (team_d <= ratio * np.maximum(near_d, 1e-6))
        team[rescue] = self.group_team[best[rescue]]
        return team


def main(argv=None, model=None, grab=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="make the contact sheet of colour groups")
    m.add_argument("--video", required=True)
    m.add_argument("--detections", required=True)
    m.add_argument("--out", required=True, help="prefix: <out>.teams_sheet.png and <out>.team_labels.json")
    m.add_argument("--k", type=int, default=8, help="number of colour groups")
    m.add_argument("--frames", type=int, default=40)
    m.add_argument("--per-group", type=int, default=14)
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--min-height", type=float, default=26.0, help="smallest player (px high) used for the colour groups; far players blur into the grass")
    m.add_argument("--max-lon", type=float, default=98.0)
    m.add_argument("--iv-fov", type=float, default=None)
    m.add_argument("--tilt", type=float, default=None)
    m.add_argument("--players", default="models/yolo11s.pt")
    m.add_argument("--device", default=None)
    s = sub.add_parser("assign", help="say which groups are team A and team B")
    s.add_argument("--labels", required=True)
    s.add_argument("--a", type=int, nargs="+", default=None, help="groups of team A")
    s.add_argument("--b", type=int, nargs="+", default=None, help="groups of team B")
    s.add_argument("--name-a", default=None)
    s.add_argument("--name-b", default=None)
    s.add_argument("--colour-a", default=None, help="chart colour: a name (white, blue, red, ...) or #rrggbb")
    s.add_argument("--colour-b", default=None)
    a = p.parse_args(argv)
    return make(a, model, grab) if a.cmd == "make" else assign(a)


if __name__ == "__main__":
    sys.exit(main())
