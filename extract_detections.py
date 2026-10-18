#!/usr/bin/env python3
"""
extract_detections.py — run the player and ball detectors on a clip, independent of where
the virtual camera points, and write every detection as a longitude/latitude on the sphere.

Why: autopan_infer.py detects inside the view the camera is *currently* showing, so what it
sees depends on what it already decided. To learn "where should the camera point" from your
annotations, the detections must not depend on the camera. Here the equirect frame is
sampled through a few fixed, overlapping perspective views; each detection is converted to
(lon, lat) and kept by the view whose centre is nearest, so nothing is counted twice.

    python extract_detections.py --video cut5min.equirect.mp4 \
        --keyframes cut5min.equirect.keyframes.json \
        [--calib calibration/pitch.json] [--fps 3] [--device mps]

--video  an equirect (2:1) video, or the raw Insta360 MP4 (converted on the fly with the
         unchanged pipeline filter). The detectors were tuned on 2880x1440 frames; a smaller
         proxy is scaled up, which works but loses small-ball detections, so for the best
         ball recall make the proxy with `make_proxy.py --width 2880`.
--keyframes  only the stretch you annotated (first to last keyframe, +/- --margin s) is
         processed. Without it, the whole video (or --start/--duration, or the IN/OUT marks).

Output (<video>.detections.jsonl): a header line, then one line per sampled frame:
    {"t": 12.333, "players": [[lon, lat, box_height_px, conf, on_pitch], ...],
                  "balls":   [[lon, lat, conf], ...]}
Conventions are kf_core's: lon in degrees, + = right on the equirect; pan_deg = -lon_centre.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

import kf_core as K

# detector settings are shared with autopan_infer.py so both stay in sync
try:
    import autopan_infer as AP
    IMGSZ_PLAYERS, IMGSZ_BALL = AP.IMGSZ_PLAYERS, AP.IMGSZ_BALL
    CONF_PLAYERS, CONF_BALL = AP.CONF_PLAYERS, AP.CONF_BALL
except Exception:                                     # keep the module importable on its own
    IMGSZ_PLAYERS, IMGSZ_BALL, CONF_PLAYERS, CONF_BALL = 960, 1280, 0.20, 0.40

EQ_W, EQ_H = 2880, 1440
VIEW_W, VIEW_H = 1280, 720
VIEW_FOV = 100.0                                      # as in autopan_infer (e2p scalar fov)
RAW_TO_EQUIRECT = "v360=fisheye:equirect:ih_fov=200:iv_fov=200,scale=2880:1440"   # unchanged


# ── views ────────────────────────────────────────────────────────────────────

class View:
    """A fixed perspective view of the equirect frame. Geometry is exactly py360convert's e2p
    (scalar fov, i.e. the vertical fov equals the horizontal one) so the detector models see
    the same kind of picture they were trained on."""

    def __init__(self, centre_lon: float, tilt: float, fov: float = VIEW_FOV,
                 w: int = VIEW_W, h: int = VIEW_H, eq_w: int = EQ_W, eq_h: int = EQ_H,
                 legacy_fov: bool = True):
        # legacy_fov=True: the vertical field equals the horizontal one (anamorphic pixels). That is what
        # the original iv_fov=200 unwrapping needs (its vertical pixels per degree are 0.56 of the horizontal
        # ones, so the view is 1:1 with the source). With a square-pixel unwrapping (iv_fov ~112) the same
        # views would squash every player to about half height, so those use square pixels instead.
        self.centre_lon, self.tilt, self.fov, self.w, self.h = centre_lon, tilt, fov, w, h
        pan = K.PAN_SIGN * centre_lon
        self.map_x, self.map_y = K.build_remap(pan, tilt, fov, eq_w, eq_h, w, h, legacy_fov=legacy_fov)
        # longitude / latitude of every pixel of the view
        rays = K.view_rays(pan, tilt, fov, w, h, legacy_fov=legacy_fov)
        self.lon, self.lat = K.rays_to_lonlat(rays, centre_lon)
        self.lon = np.asarray(self.lon, np.float32)
        self.lat = np.asarray(self.lat, np.float32)

    def sample(self, eq_frame: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.remap(eq_frame, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def sample_mask(self, eq_mask: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.remap(eq_mask, self.map_x, self.map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)

    def lonlat(self, px: float, py: float) -> Tuple[float, float]:
        xi = int(np.clip(round(px), 0, self.w - 1))
        yi = int(np.clip(round(py), 0, self.h - 1))
        return float(self.lon[yi, xi]), float(self.lat[yi, xi])


def make_views(lon_min: float, lon_max: float, tilt: float, spacing: float = 40.0,
               fov: float = VIEW_FOV, legacy_fov: bool = True) -> List[View]:
    n = max(1, int(np.ceil((lon_max - lon_min) / spacing)))
    centres = np.linspace(lon_min + (lon_max - lon_min) / (2 * n),
                          lon_max - (lon_max - lon_min) / (2 * n), n)
    return [View(float(c), tilt, fov, legacy_fov=legacy_fov) for c in centres]


def nearest_view(views: Sequence[View], lon: float) -> int:
    d = [abs(float(K.wrap180(lon - v.centre_lon))) for v in views]
    return int(np.argmin(d))


# ── calibration (pitch polygon) ──────────────────────────────────────────────

def load_polygon(calib_path: str) -> np.ndarray:
    with open(calib_path) as f:
        d = json.load(f)
    raw = d.get("pitch_polygon") or d.get("pixel_polygon") or d.get("auto_polygon")
    if isinstance(raw[0], dict):
        return np.array([[p["x"] * EQ_W, p["y"] * EQ_H] for p in raw], dtype=np.float32)
    return np.array(raw, dtype=np.float32)


def polygon_mask(poly: np.ndarray) -> np.ndarray:
    import cv2
    m = np.zeros((EQ_H, EQ_W), np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32).reshape(-1, 1, 2)], 255)
    return m


def polygon_lon_range(poly: np.ndarray, margin: float = 12.0) -> Tuple[float, float]:
    lon, _ = K.xy_to_lonlat(poly[:, 0], poly[:, 1], EQ_W, EQ_H)
    return float(lon.min() - margin), float(lon.max() + margin)


def torso_colour(img: np.ndarray, box: Sequence[float], min_pixels: int = 12) -> np.ndarray:
    """Median BGR of the shirt: the middle of the box, upper half, with grass-green pixels left out.
    `img` is a BGR image, `box` = x1, y1, x2, y2 in its pixels. Returns float BGR (3,)."""
    import cv2
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    xa, xb = int(max(0, x1 + 0.28 * bw)), int(min(w, x1 + 0.72 * bw + 1))
    ya, yb = int(max(0, y1 + 0.18 * bh)), int(min(h, y1 + 0.52 * bh + 1))
    if xb <= xa or yb <= ya:
        return np.array([np.nan] * 3)
    patch = img[ya:yb, xa:xb].reshape(-1, 3)
    hsv = cv2.cvtColor(patch.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    grass = (hsv[:, 0] >= 35) & (hsv[:, 0] <= 85) & (hsv[:, 1] > 70) & (hsv[:, 2] > 40)
    keep = patch[~grass]
    if len(keep) < min_pixels:
        keep = patch
    return np.median(keep.astype(np.float32), axis=0)


# ── detection ────────────────────────────────────────────────────────────────

def detect_players(img, model, device, mask=None, conf=CONF_PLAYERS, colours=False):
    """-> [(cx, foot_y, box_h, conf, on_pitch)] in view pixels, class 'person' only.
    With colours=True each tuple gets a sixth element: the shirt colour [b, g, r] (see torso_colour)."""
    res = model(img, imgsz=IMGSZ_PLAYERS, conf=conf, device=device, verbose=False)[0]
    names = model.names
    out = []
    for b in res.boxes:
        if names.get(int(b.cls[0]), "") != "person":
            continue
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        cx, foot = (x1 + x2) / 2, y2
        on = None
        if mask is not None:
            on = bool(mask[int(np.clip(foot, 0, mask.shape[0] - 1)), int(np.clip(cx, 0, mask.shape[1] - 1))])
        row = (cx, foot, y2 - y1, float(b.conf[0]), on)
        if colours:
            row += ([int(round(v)) for v in torso_colour(img, (x1, y1, x2, y2))],)
        out.append(row)
    return out


def ball_class_ids(model, override: Optional[str] = None):
    """Which classes of a detector count as 'ball'. A single-class model (a dedicated ball
    detector): all of them. A multi-class model (e.g. an all-purpose COCO model, which would
    otherwise report every person as a ball): only classes whose name contains 'ball'
    (COCO: 'sports ball'). `override` = comma-separated names or ids."""
    names = dict(model.names)
    if override:
        want = [w.strip().lower() for w in override.split(",") if w.strip()]
        return {i for i, n in names.items() if str(i) in want or str(n).lower() in want}
    if len(names) <= 1:
        return None                                   # None = accept every class
    ids = {i for i, n in names.items() if "ball" in str(n).lower()}
    if not ids:
        raise SystemExit(f"the ball model has classes {sorted(names.values())} but none looks like a ball; "
                         f"pick one with --ball-classes")
    return ids


def detect_balls(img, model, device, conf=CONF_BALL, max_area_frac=0.02, class_ids=None):
    """-> [(cx, cy, conf)] — every candidate, not just the best one."""
    res = model(img, imgsz=IMGSZ_BALL, conf=conf, device=device, verbose=False)[0]
    out = []
    for b in res.boxes:
        if class_ids is not None and int(b.cls[0]) not in class_ids:
            continue
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        if (x2 - x1) * (y2 - y1) > VIEW_W * VIEW_H * max_area_frac:
            continue
        out.append(((x1 + x2) / 2, (y1 + y2) / 2, float(b.conf[0])))
    return out


def process_frame(eq_frame, t, views, player_model, ball_model, device, eq_mask=None,
                  ball_conf=CONF_BALL, player_conf=CONF_PLAYERS, ball_classes=None, colours=False) -> dict:
    players, balls = [], []
    for vi, v in enumerate(views):
        img = v.sample(eq_frame)
        vmask = v.sample_mask(eq_mask) if eq_mask is not None else None
        for cx, foot, h, c, on, *extra in detect_players(img, player_model, device, vmask, player_conf, colours):
            lon, lat = v.lonlat(cx, foot)
            if nearest_view(views, lon) != vi:            # the neighbouring view owns this one
                continue
            players.append([round(lon, 3), round(lat, 3), round(h, 1), round(c, 3), on] + extra)   # extra = [shirt b,g,r]
        if ball_model is not None:
            for cx, cy, c in detect_balls(img, ball_model, device, ball_conf, class_ids=ball_classes):
                lon, lat = v.lonlat(cx, cy)
                if nearest_view(views, lon) != vi:
                    continue
                balls.append([round(lon, 3), round(lat, 3), round(c, 3)])
    return {"t": round(float(t), 3), "players": players, "balls": balls}


# ── video input ──────────────────────────────────────────────────────────────

def open_frames(video: str, start: float, duration: Optional[float], fps: float, raw: bool,
                iv_fov: Optional[float] = None):
    vf = (K.raw_to_equirect(iv_fov) + "," if raw else "") + \
         (f"scale={EQ_W}:{EQ_H}," if not raw else "") + f"fps={fps}"
    cmd = ["ffmpeg", "-v", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", video]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "pipe:1"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def read_frame(proc) -> Optional[np.ndarray]:
    n = EQ_W * EQ_H * 3
    buf = proc.stdout.read(n)
    while buf is not None and 0 < len(buf) < n:
        more = proc.stdout.read(n - len(buf))
        if not more:
            break
        buf += more
    if not buf or len(buf) < n:
        return None
    return np.frombuffer(buf, np.uint8).reshape(EQ_H, EQ_W, 3)


def pick_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    try:
        import torch
        return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None, player_model=None, ball_model=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--out", default=None, help="default: <video>.detections.jsonl")
    p.add_argument("--keyframes", default=None, help="process only the annotated stretch")
    p.add_argument("--margin", type=float, default=4.0, help="seconds around the annotated stretch")
    p.add_argument("--calib", default=None, help="pitch.json: flags players as on/off the pitch and sets the view range")
    p.add_argument("--tilt", type=float, default=None, help="tilt of the detection views (default: from calib, else the saved tilt, else -15)")
    p.add_argument("--lon-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                   help="longitudes to cover, e.g. --lon-range -80 80 (default: from calib, else -80 80)")
    p.add_argument("--view-spacing", type=float, default=40.0)
    p.add_argument("--fps", type=float, default=3.0, help="detections per second (default 3)")
    p.add_argument("--iv-fov", type=float, default=None,
                   help="vertical field of the fisheye model that unwraps the raw video (default 200, the original; "
                        "112 gives equal pixels per degree for a 2560x1440 frame). Written to the output header.")
    p.add_argument("--colours", action="store_true",
                   help="also store each player's shirt colour [b,g,r] as a sixth value (for the team analysis)")
    p.add_argument("--start", type=float, default=None)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--players", default="models/yolo11s.pt")
    p.add_argument("--ball", default="models/ball_v4.pt")
    p.add_argument("--no-ball", action="store_true")
    p.add_argument("--ball-conf", type=float, default=CONF_BALL)
    p.add_argument("--ball-classes", default=None,
                   help="class names/ids that count as the ball (default: all classes of a one-class model, "
                        "else classes with 'ball' in the name)")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)

    info = K.probe_video(a.video)
    raw = not K.is_equirect_size(info["width"], info["height"])
    out = a.out or os.path.splitext(a.video)[0] + ".detections.jsonl"

    # which stretch of the video
    start, duration = a.start, a.duration
    meta = K.load_meta(a.keyframes) if a.keyframes else {}
    if start is None and duration is None:
        # the game IN / OUT marks set in annotate.py ([ and ]) win; otherwise the span of the keyframes
        if meta.get("trim_in") is not None:
            start = float(meta["trim_in"])
        if meta.get("trim_out") is not None:
            duration = float(meta["trim_out"]) - (start or 0.0)
    if a.keyframes and start is None and duration is None and os.path.exists(a.keyframes):
        kfs = K.KeyframeSet.load(a.keyframes)
        if len(kfs):
            start = max(0.0, kfs.items[0].t - a.margin)
            duration = kfs.items[-1].t + a.margin - start
    start = start or 0.0
    if duration is None:
        duration = max(0.0, info["duration"] - start)

    iv_fov = a.iv_fov if a.iv_fov is not None else meta.get("iv_fov")
    iv_fov = K.DEFAULT_IV_FOV if iv_fov is None else float(iv_fov)

    poly = load_polygon(a.calib) if a.calib else None
    eq_mask = polygon_mask(poly) if poly is not None else None
    if a.tilt is not None:
        tilt = a.tilt
    elif a.calib:
        tilt = K.tilt_from_calib(a.calib)
    else:
        tilt = float(meta.get("tilt_deg", -15.0))
    if a.lon_range:
        lo, hi = a.lon_range
    elif poly is not None:
        lo, hi = polygon_lon_range(poly)
    else:
        lo, hi = -80.0, 80.0
    square_views = iv_fov != K.DEFAULT_IV_FOV           # a non-default unwrapping is meant to have square pixels
    views = make_views(lo, hi, tilt, a.view_spacing, legacy_fov=not square_views)

    device = pick_device(a.device)
    if player_model is None:
        from ultralytics import YOLO
        player_model = YOLO(a.players)
        ball_model = None if a.no_ball else YOLO(a.ball)
    ball_ids = ball_class_ids(ball_model, a.ball_classes) if ball_model is not None else None
    if ball_model is not None:
        shown = "all" if ball_ids is None else sorted(str(ball_model.names[i]) for i in ball_ids)
        print(f"ball model {a.ball}: classes {dict(ball_model.names)} -> counting {shown}")
    print(f"{a.video}: {'raw -> equirect' if raw else 'equirect'}  {K.fmt_time(start)} + {duration:.0f}s  "
          f"@ {a.fps:g}/s  device={device}")
    print(f"views    : {len(views)} at lon {[round(v.centre_lon) for v in views]}  tilt {tilt:+.1f}  fov {VIEW_FOV:g}")

    header = {"type": "header", "video": os.path.basename(a.video), "start": start, "duration": duration,
              "video_duration": round(float(info["duration"]), 3),
              "fps": a.fps, "tilt": tilt, "view_lons": [v.centre_lon for v in views], "view_fov": VIEW_FOV,
              "calib": a.calib, "players_model": a.players, "ball_model": None if ball_model is None else a.ball,
              "ball_conf": a.ball_conf, "ball_classes": None if ball_ids is None else sorted(ball_ids), "player_conf": CONF_PLAYERS, "raw_input": raw}
    if raw and iv_fov != K.DEFAULT_IV_FOV:               # only written when it differs: older files stay byte-compatible
        header["iv_fov"] = iv_fov
        header["square_views"] = square_views
        print(f"unwrap   : iv_fov={iv_fov:g} (default {K.DEFAULT_IV_FOV:g}); the keyframes made from this file carry it")
    if a.colours:
        header["colours"] = True
    n_total = max(1, int(duration * a.fps))
    proc = open_frames(a.video, start, duration, a.fps, raw, iv_fov)
    t0, n, n_ball = time.time(), 0, 0
    try:
        with open(out, "w") as fh:
            fh.write(json.dumps(header) + "\n")
            while True:
                frame = read_frame(proc)
                if frame is None:
                    break
                rec = process_frame(frame, start + n / a.fps, views, player_model, ball_model, device,
                                    eq_mask, a.ball_conf, ball_classes=ball_ids, colours=a.colours)
                fh.write(json.dumps(rec) + "\n")
                n += 1
                n_ball += 1 if rec["balls"] else 0
                if n % 20 == 0:
                    el = time.time() - t0
                    print(f"  {n}/{n_total} frames  ({el / n:.2f}s each, ~{(n_total - n) * el / n / 60:.1f} min left)",
                          end="\r", flush=True)
    finally:
        proc.stdout.close()
        proc.terminate()
    print(f"\nwrote {out}: {n} frames, ball seen in {n_ball} ({100 * n_ball / max(n, 1):.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
