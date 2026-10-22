#!/usr/bin/env python3
"""scene_check.py — sanity check for the colour clustering and the position estimate.

Takes a few random scenes of a game and shows, for each one
  left : the unwrapped frame with every detected player outlined in the colour of its cluster
         (numbered), and
  right: the estimated position of each player on a top-down map (camera at the origin).
Above them: what the clustering found (cluster colours, sizes).

How the clusters are made: the colour of each player's torso (median of the non-grass pixels in the
middle of the box) is measured on ~40 frames spread over the game, converted to Lab, and grouped with
k-means. The two biggest clusters are the two teams; the smaller ones are the referee, goalkeepers
and whatever else. The same cluster centres are then used to label the scenes.

Position: distance = 1.7 m * focal length / box height (a person is about 1.7 m), sideways from the
longitude. The angles of the unwrapped video are "virtual" (the real field of view is smaller), so
real angle = --angle-factor * virtual angle (default 0.75) and the distance is divided by the same
factor. Check the map against what you know: pitch size, goalkeepers near the goal lines.
Treat the numbers as approximate; the point is whether the picture is plausible.

  python scene_check.py --video VID_..._034.mp4 --detections h034.detections.jsonl \\
      --n 4 --seed 1 --out h034.scenes.png

  --times 300 900 1500   choose the moments yourself (seconds in the video)
  --k 4                  number of colour clusters (2 teams + 2 others)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from typing import Callable, List, Optional, Sequence

import numpy as np

import kf_core as K
import extract_detections as X
import pitch_filter as PF

PERSON_M = 1.7
SQUARE_FY = 640.0 / math.tan(math.radians(50.0))      # focal length of the square-pixel 1280x720 views (~537 px)
LEGACY_FY = 302.0                                     # the anamorphic views of the original unwrapping


def view_fy(header: dict) -> float:
    """Focal length (px) of the detection views of a detections file: from the view field of view when the file has square-pixel
    views (100 degrees -> SQUARE_FY, as before; narrower views -> longer), else the legacy value."""
    if not header.get("square_views"):
        return LEGACY_FY
    fov = float(header.get("view_fov") or 100.0)
    return 640.0 / math.tan(math.radians(fov / 2.0))
PALETTE = ["#e6194b", "#4363d8", "#ffe119", "#f58231", "#911eb4", "#3cb44b", "#bbbbbb"]


# ── colour ───────────────────────────────────────────────────────────────────

torso_colour = X.torso_colour                                  # lives in extract_detections (used there too)


def bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    """(n,3) BGR 0..255 -> (n,3) Lab (L 0..100)."""
    import cv2
    a = np.asarray(bgr, np.float32).reshape(-1, 1, 3) / 255.0
    return cv2.cvtColor(a, cv2.COLOR_BGR2LAB).reshape(-1, 3)


def colour_features(bgr: np.ndarray, l_weight: float = 0.4) -> np.ndarray:
    """Lab with the lightness turned down: shade and exposure differ more across the pitch than hue."""
    lab = bgr_to_lab(bgr)
    lab[:, 0] *= l_weight
    return lab


def kmeans(x: np.ndarray, k: int, seed: int = 0, restarts: int = 8, iters: int = 60):
    """Small k-means with k-means++ start. -> (centres (k,d), labels (n,)). Best of `restarts`."""
    x = np.asarray(x, float)
    k = max(1, min(k, len(x)))
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        c = [x[rng.integers(len(x))]]
        for _ in range(k - 1):
            d2 = np.min(((x[:, None, :] - np.array(c)[None]) ** 2).sum(-1), axis=1)
            tot = d2.sum()
            c.append(x[rng.choice(len(x), p=d2 / tot)] if tot > 0 else x[rng.integers(len(x))])
        c = np.array(c)
        for _ in range(iters):
            lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
            new = np.array([x[lab == j].mean(0) if np.any(lab == j) else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        inertia = float(((x - c[lab]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, c, lab)
    return best[1], best[2]


class Clusters:
    """Colour clusters ordered by size: 0 and 1 are the teams, the rest are 'other'.

    k-means with k=4 can cut one team in two (light and shadow, near and far). So when more than two
    clusters are big (>= team_share of all players), those big ones are pooled and re-cut into exactly
    two; the small clusters stay as they are (referee, goalkeepers, ...)."""

    def __init__(self, bgr: np.ndarray, k: int = 4, seed: int = 0, team_share: float = 0.10):
        feats = colour_features(bgr)
        centres, lab = kmeans(feats, k, seed)
        sizes = np.bincount(lab, minlength=len(centres))
        big = np.where(sizes >= team_share * len(feats))[0]
        if len(big) > 2:
            two, _ = kmeans(feats[np.isin(lab, big)], 2, seed)
            centres = np.vstack([two] + [centres[j] for j in range(len(centres)) if j not in big])
        lab = np.argmin(((feats[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
        order = np.argsort(-np.bincount(lab, minlength=len(centres)))
        self.centres = centres[order]
        rank = np.argsort(order)
        remap = rank[lab]
        self.sizes = np.bincount(remap, minlength=len(centres))
        # the mean shirt colour of each cluster, for the swatches
        self.mean_bgr = np.array([np.asarray(bgr, float)[remap == j].mean(0) if np.any(remap == j) else [128] * 3
                                  for j in range(len(centres))])

    def assign(self, bgr: np.ndarray):
        """-> (labels, ambiguity): ambiguity = distance to the best / distance to the second best (1 = coin toss)."""
        f = colour_features(bgr)
        d = np.sqrt(((f[:, None, :] - self.centres[None]) ** 2).sum(-1))
        o = np.argsort(d, axis=1)
        best = d[np.arange(len(d)), o[:, 0]]
        second = d[np.arange(len(d)), o[:, 1]] if d.shape[1] > 1 else best + 1
        return o[:, 0], best / np.maximum(second, 1e-6)

    def name(self, j: int) -> str:
        return "team A" if j == 0 else "team B" if j == 1 else f"other {j - 1}"


# ── position ─────────────────────────────────────────────────────────────────

def ground_position(lon: float, box_h: float, centre_lon: float, fy: float, k: float = 0.75):
    """Top-down (x sideways, y away from the camera) in metres, camera at the origin.

    box_h in view pixels; lon and centre_lon in virtual degrees; k = real angle / virtual angle."""
    z = PERSON_M * fy / (k * max(box_h, 4.0))                      # distance along the view axis
    off = math.radians(k * (lon - centre_lon))
    d = z / max(math.cos(off), 0.3)                                # distance from the camera
    a = math.radians(k * lon)
    return d * math.sin(a), d * math.cos(a)


# ── scenes ───────────────────────────────────────────────────────────────────

def read_detections(path: str):
    with open(path) as fh:
        header = json.loads(fh.readline())
        frames = [json.loads(line) for line in fh if line.strip()]
    return header, frames


def pick_scenes(frames: List[dict], n: int, seed: int, min_players: int = 14, min_gap: float = 90.0) -> List[float]:
    """n random moments with plenty of players found, at least min_gap seconds apart."""
    good = [fr["t"] for fr in frames if sum(1 for p in fr["players"] if p[3] >= 0.5) >= min_players]
    if not good:
        good = [fr["t"] for fr in frames]
    rng = np.random.default_rng(seed)
    rng.shuffle(good)
    picked: List[float] = []
    for t in good:
        if all(abs(t - u) >= min_gap for u in picked):
            picked.append(float(t))
        if len(picked) == n:
            break
    for t in good:                                                # too few far-apart moments: relax
        if len(picked) >= n:
            break
        if t not in picked:
            picked.append(float(t))
    return sorted(picked)


def fit_times(frames: List[dict], n: int, seed: int) -> List[float]:
    ts = [fr["t"] for fr in frames]
    if len(ts) <= n:
        return ts
    rng = np.random.default_rng(seed + 1000)
    idx = np.linspace(0, len(ts) - 1, n)
    return [ts[int(min(len(ts) - 1, max(0, round(i + rng.uniform(-0.4, 0.4) * len(ts) / n))))] for i in idx]


def grab_frame(video: str, t: float, iv_fov: Optional[float], raw: bool) -> Optional[np.ndarray]:
    vf = (K.raw_to_equirect(iv_fov) if raw else f"scale={X.EQ_W}:{X.EQ_H}")
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1", "-vf", vf,
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "pipe:1"]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    n = X.EQ_W * X.EQ_H * 3
    if len(out) < n:
        return None
    return np.frombuffer(out[:n], np.uint8).reshape(X.EQ_H, X.EQ_W, 3)


def detect_boxes(img, model, device, conf):
    res = model(img, imgsz=X.IMGSZ_PLAYERS, conf=conf, device=device, verbose=False)[0]
    out = []
    for b in res.boxes:
        if model.names.get(int(b.cls[0]), "") != "person":
            continue
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        out.append((x1, y1, x2, y2, float(b.conf[0])))
    return out


def detect_scene(eq_frame, views, model, device, fy: float, k: float, conf: float, max_lon: float) -> List[dict]:
    """Every player in one frame: lon/lat of the feet, torso colour, map position, outline in the eq frame."""
    players = []
    for vi, v in enumerate(views):
        img = v.sample(eq_frame)
        for x1, y1, x2, y2, c in detect_boxes(img, model, device, conf):
            cx, foot = (x1 + x2) / 2, y2
            lon, lat = v.lonlat(cx, foot)
            if X.nearest_view(views, lon) != vi or abs(lon) > max_lon:
                continue
            h = y2 - y1
            xi = lambda px, py: (float(v.map_x[int(np.clip(py, 0, v.h - 1)), int(np.clip(px, 0, v.w - 1))]),   # noqa: E731
                                 float(v.map_y[int(np.clip(py, 0, v.h - 1)), int(np.clip(px, 0, v.w - 1))]))
            players.append({
                "lon": lon, "lat": lat, "h": h, "conf": c,
                "bgr": torso_colour(img, (x1, y1, x2, y2)),
                "pos": ground_position(lon, h, v.centre_lon, fy, k),
                "outline": [xi(x1, y1), xi(x2, y1), xi(x2, y2), xi(x1, y2)],
            })
    return players


# ── drawing ──────────────────────────────────────────────────────────────────

def draw(scenes, clusters: Clusters, views, title: str, out: str, min_conf: float, show_all_conf: bool = False,
         outline=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle

    xs = np.concatenate([v.map_x.ravel() for v in views])
    ys = np.concatenate([v.map_y.ravel() for v in views])
    x0, x1 = int(max(0, xs.min())), int(min(X.EQ_W, xs.max()))
    y0, y1 = int(max(0, ys.min())), int(min(X.EQ_H, ys.max()))
    n = len(scenes)
    fig = plt.figure(figsize=(17, 2.2 + 4.6 * n))
    gs = fig.add_gridspec(n + 1, 2, width_ratios=[2.3, 1], height_ratios=[0.5] + [1] * n, hspace=0.18, wspace=0.07)

    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=14)
    total = max(1, int(clusters.sizes.sum()))
    for j in range(len(clusters.centres)):
        col = clusters.mean_bgr[j][::-1] / 255.0
        ax.add_patch(plt.Rectangle((j * 0.2, 0.15), 0.04, 0.6, color=col, ec="k", transform=ax.transAxes))
        ax.add_patch(plt.Rectangle((j * 0.2 + 0.045, 0.15), 0.012, 0.6, color=PALETTE[j % len(PALETTE)],
                                   transform=ax.transAxes))
        ax.text(j * 0.2 + 0.065, 0.45, f"{clusters.name(j)}\n{clusters.sizes[j]} players "
                f"({100 * clusters.sizes[j] / total:.0f}%)", transform=ax.transAxes, fontsize=10, va="center")
    ax.text(0.0, -0.05, "swatch = measured shirt colour, narrow bar = outline colour used below; "
            "hollow marker = ambiguous (nearly as close to a second cluster)", transform=ax.transAxes, fontsize=8)

    reach = 10.0
    for r, sc in enumerate(scenes):
        crop = sc["frame"][y0:y1, x0:x1][:, :, ::-1]
        a = fig.add_subplot(gs[r + 1, 0])
        a.imshow(crop)
        if outline is not None:
            a.add_patch(Polygon([(px - x0, py - y0) for px, py in outline], closed=True, fill=False,
                                ec="#00e5ff", lw=1.2, ls=":"))
        for p in sc.get("off", []):                           # outside the pitch: grey, not on the map
            a.add_patch(Polygon([(px - x0, py - y0) for px, py in p["outline"]], closed=True, fill=False,
                                ec="#888888", lw=1.4, ls=(0, (2, 2))))
        m = fig.add_subplot(gs[r + 1, 1])
        for i, p in enumerate(sc["players"]):
            if p["conf"] < min_conf and not show_all_conf:
                continue
            col = PALETTE[p["cluster"] % len(PALETTE)]
            poly = [(px - x0, py - y0) for px, py in p["outline"]]
            a.add_patch(Polygon(poly, closed=True, fill=False, ec=col, lw=1.8, ls="--" if p["ambig"] else "-"))
            a.text(poly[0][0], poly[0][1] - 3, str(i + 1), color="w", fontsize=8, ha="left", va="bottom",
                   bbox=dict(boxstyle="round,pad=0.1", fc=col, ec="none", alpha=0.85))
            px, py = p["pos"]
            reach = max(reach, abs(px), py)
            m.scatter([px], [py], s=130, c="none" if p["ambig"] else col, edgecolors="k", linewidths=1.4, zorder=3)
            m.scatter([px], [py], s=18, c=[p["bgr"][::-1] / 255.0], zorder=4)
            m.text(px + 0.8, py + 0.8, str(i + 1), fontsize=8, zorder=5)
        a.set_axis_off()
        a.set_title(f"scene {r + 1}:  t = {K.fmt_time(sc['t'])}   {sc['count']} players outlined, "
                    f"{sc['n_team']} in the two teams", loc="left", fontsize=11)
        sc["map_ax"] = m

    lim = math.ceil(reach / 10.0) * 10 + 5
    for sc in scenes:
        m = sc["map_ax"]
        m.scatter([0], [0], marker="^", s=120, c="k", zorder=6)
        for rr in range(10, int(lim * 1.5) + 1, 10):
            m.add_patch(Circle((0, 0), rr, fill=False, ec="#cccccc", lw=0.6, zorder=1))
            if rr <= lim:
                m.text(0.6, rr, f"{rr} m", fontsize=7, color="#888888")
        m.set_xlim(-lim, lim)
        m.set_ylim(-3, lim)
        m.set_aspect("equal")
        m.grid(alpha=0.2)
        m.set_title("estimated positions, seen from above (camera = triangle)", fontsize=9, loc="left")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def run(a, model=None, grab: Optional[Callable] = None) -> int:
    header, frames = read_detections(a.detections)
    if not frames:
        print("no frames in the detections file")
        return 1
    video = a.video
    iv_fov = a.iv_fov if a.iv_fov is not None else header.get("iv_fov")
    raw = header.get("raw_input", True)
    square = bool(header.get("square_views", iv_fov not in (None, K.DEFAULT_IV_FOV)))
    fy = SQUARE_FY if square else LEGACY_FY
    tilt = a.tilt if a.tilt is not None else float(header.get("tilt", -20.0))
    lo, hi = a.lon_range if a.lon_range else (min(header["view_lons"]) - 20, max(header["view_lons"]) + 20)
    views = X.make_views(lo, hi, tilt, a.view_spacing, legacy_fov=not square)
    if model is None:
        from ultralytics import YOLO
        model = YOLO(a.players)
    device = X.pick_device(a.device)
    grab = grab or (lambda t: grab_frame(video, t, iv_fov, raw))

    scene_ts = [float(t) for t in a.times] if a.times else pick_scenes(frames, a.n, a.seed)
    fit_ts = [t for t in fit_times(frames, a.fit_frames, a.seed) if all(abs(t - s) > 1e-6 for s in scene_ts)]
    print(f"scenes at {[K.fmt_time(t) for t in scene_ts]};  colour fit on {len(fit_ts)} more frames;  "
          f"views {len(views)} tilt {tilt:+.1f}  square={square}  k={a.angle_factor}")

    pitch = PF.pitch_mask(a.calib, a.margin) if a.calib else None
    scenes, fit_colours = [], []
    for kind, ts in (("scene", scene_ts), ("fit", fit_ts)):
        for i, t in enumerate(ts):
            fr = grab(t)
            if fr is None:
                print(f"  could not read a frame at {t:.1f}s")
                continue
            pl = [p for p in detect_scene(fr, views, model, device, fy, a.angle_factor, a.conf, a.max_lon)
                  if p["conf"] >= a.min_conf and p["h"] >= a.min_height and not np.isnan(p["bgr"]).any()]
            for p in pl:
                p["off"] = bool(pitch is not None and not PF.foot_on_mask(pitch, p["lon"], p["lat"]))
            fit_colours += [p["bgr"] for p in pl if not p["off"]]
            if kind == "scene":
                scenes.append({"t": t, "frame": fr, "players": pl})
            print(f"  {kind} {i + 1}/{len(ts)}  t={t:7.1f}s  {len(pl)} players", end="\r", flush=True)
    print()
    if len(fit_colours) < 8:
        print("too few players found to cluster")
        return 1

    clusters = Clusters(np.array(fit_colours), a.k, a.seed)
    for sc in scenes:
        keep = [p for p in sc["players"] if not p["off"]]
        sc["off"] = [p for p in sc["players"] if p["off"]]
        sc["players"] = keep
        if keep:
            lab, amb = clusters.assign(np.array([p["bgr"] for p in keep]))
            for p, l, am in zip(keep, lab, amb):
                p["cluster"], p["ambig"] = int(l), bool(am > a.ambiguous)
        sc["count"] = len(keep)
        sc["n_team"] = sum(1 for p in keep if p["cluster"] < 2)
        xs = [p["pos"][0] for p in keep]
        ys = [p["pos"][1] for p in keep]
        if keep:
            print(f"  t={K.fmt_time(sc['t'])}: {sc['count']} players  ({sc['n_team']} in teams A/B), "
                  f"x {min(xs):.0f}..{max(xs):.0f} m, distance {min(ys):.0f}..{max(ys):.0f} m, "
                  f"clusters " + ", ".join(f"{clusters.name(j)} {sum(1 for p in keep if p['cluster'] == j)}"
                                           for j in range(len(clusters.centres))))
    print("colour clusters (all fit players): " + ", ".join(
        f"{clusters.name(j)} {clusters.sizes[j]}" for j in range(len(clusters.centres))))
    out = a.out or os.path.splitext(a.detections)[0] + ".scenes.png"
    outline = X.load_polygon(a.calib) if a.calib else None
    draw(scenes, clusters, views, a.title or os.path.basename(a.video), out, a.min_conf, outline=outline)
    if outline is not None:
        print("outside the pitch outline (left out): " + ", ".join(
            f"{K.fmt_time(sc['t'])}: {len(sc['off'])}" for sc in scenes))
    print(f"wrote {out}")
    return 0


def main(argv=None, model=None, grab=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--detections", required=True, help="the .detections.jsonl of that video (times, unwrapping, views)")
    p.add_argument("--out", default=None)
    p.add_argument("--title", default=None)
    p.add_argument("--n", type=int, default=4, help="number of random scenes")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--times", type=float, nargs="+", default=None, help="video times (s) instead of random scenes")
    p.add_argument("--k", type=int, default=4, help="colour clusters")
    p.add_argument("--fit-frames", type=int, default=40, help="extra frames used to learn the colours")
    p.add_argument("--angle-factor", type=float, default=0.75, help="real angle / virtual angle of the unwrapped video")
    p.add_argument("--ambiguous", type=float, default=0.8, help="distance ratio above which a label is called ambiguous")
    p.add_argument("--min-conf", type=float, default=0.4)
    p.add_argument("--conf", type=float, default=X.CONF_PLAYERS)
    p.add_argument("--min-height", type=float, default=18.0, help="smallest box (view px) whose colour is trusted")
    p.add_argument("--max-lon", type=float, default=98.0)
    p.add_argument("--calib", default=None, help="pitch outline (pitch.json): players outside it are left out")
    p.add_argument("--margin", type=float, default=25.0, help="pixels outside the outline that still count as on the pitch")
    p.add_argument("--iv-fov", type=float, default=None)
    p.add_argument("--tilt", type=float, default=None)
    p.add_argument("--lon-range", type=float, nargs=2, default=None)
    p.add_argument("--view-spacing", type=float, default=40.0)
    p.add_argument("--players", default="models/yolo11s.pt")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    return run(a, model, grab)


if __name__ == "__main__":
    sys.exit(main())
