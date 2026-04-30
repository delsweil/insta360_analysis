#!/usr/bin/env python3
"""
build_dataset.py — Extract training data for the autopan model.

For each annotated game clip:
1. Streams equirect frames from INSV via FFmpeg
2. At each sample timestamp, projects TWO views:
   a. Studio-pan view  — projected at the ground-truth Studio pan angle
      Used for player/ball detection (best quality, action-centred)
   b. Wide fixed view  — fov=120, yaw=0, tilt=-15 (for near-side features)
3. Maps detections to 2D pitch coordinates via homography
4. Computes pitch-space features (centroid, spread, cluster)
5. Interpolates Studio keyframe values as labels
6. Writes one CSV row per sample

Output CSV has two sets of features:
  - studio_* : features from studio-pan detection view
  - wide_*   : features from fixed wide detection view
This lets us train and compare two model variants.

Usage:
    python build_dataset.py \
        --clips  clips.json \
        --output dataset.csv \
        --players models/yolo11n.pt \
        --ball    models/ball.pt \
        [--device mps|cpu|cuda]
        [--interval 0.5]
        [--append]

clips.json format:
    [
      {
        "insv":    "/path/to/VID_xxx_10_010.insv",
        "project": "/path/to/VID_xxx_00_010.insv.insprj",
        "calib":   "/path/to/pitch_010.json"
      }
    ]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from py360convert import e2p
from scipy.interpolate import PchipInterpolator

# ── Detection view parameters ─────────────────────────────────────
WIDE_FOV   = 120.0
WIDE_YAW   =   0.0
WIDE_TILT  = -15.0
DETECT_W   = 1280
DETECT_H   =  720

STUDIO_FOV_SCALE  = 1.85
STUDIO_TILT_SCALE = 1.20

# ── Detection config ──────────────────────────────────────────────
IMGSZ        = 640
CONF_PLAYERS = 0.15
CONF_BALL    = 0.35

# ── Pitch polygon coordinate assignments (16 points) ─────────────
PITCH_COORDS_NORM = [
    (0.00, 0.00), (0.17, 0.00), (0.33, 0.00), (0.50, 0.00),
    (0.67, 0.00), (0.83, 0.00), (1.00, 0.00),
    (1.00, 0.25), (1.00, 0.50), (1.00, 0.75),
    (0.83, 1.00), (0.50, 1.00), (0.17, 1.00), (0.00, 1.00),
    (0.00, 0.75), (0.00, 0.50),
]


# ──────────────────────────────────────────────────────────────────
# Keyframe parsing + interpolation
# ──────────────────────────────────────────────────────────────────

@dataclass
class Keyframe:
    time_s: float
    pan_deg: float
    tilt_deg: float
    fov_deg: float


def parse_keyframes(insprj_path: str) -> List[Keyframe]:
    tree = ET.parse(insprj_path)
    kfs = []
    for kf in tree.getroot().iter('keyframe'):
        kfs.append(Keyframe(
            time_s   = float(kf.get('time', 0)) / 1000.0,
            pan_deg  = math.degrees(float(kf.get('pan',  0))),
            tilt_deg = math.degrees(float(kf.get('tilt', 0))),
            fov_deg  = math.degrees(float(kf.get('fov',  0))),
        ))
    kfs.sort(key=lambda k: k.time_s)
    return kfs


class StudioCurve:
    def __init__(self, keyframes: List[Keyframe]):
        times = np.array([k.time_s   for k in keyframes])
        pans  = np.array([k.pan_deg  for k in keyframes])
        tilts = np.array([k.tilt_deg for k in keyframes])
        fovs  = np.array([k.fov_deg  for k in keyframes])
        self.cs_pan  = PchipInterpolator(times, pans)
        self.cs_tilt = PchipInterpolator(times, tilts)
        self.cs_fov  = PchipInterpolator(times, fovs)
        self.t_min   = float(times[0])
        self.t_max   = float(times[-1])
        self.kf_times = times

    def at(self, t: float) -> Tuple[float, float, float]:
        t = float(np.clip(t, self.t_min, self.t_max))
        return (float(self.cs_pan(t)),
                float(self.cs_tilt(t)),
                float(self.cs_fov(t)))

    def is_keyframe(self, t: float, tol: float = 0.1) -> bool:
        return bool(np.any(np.abs(self.kf_times - t) < tol))

    def time_since_last_kf(self, t: float) -> float:
        past = self.kf_times[self.kf_times <= t]
        return float(t - past[-1]) if len(past) > 0 else 999.0


# ──────────────────────────────────────────────────────────────────
# Homography: detection frame → pitch coordinates
# ──────────────────────────────────────────────────────────────────

def build_homography(
    calib_path: str,
    fov: float,
    yaw: float,
    tilt: float,
    frame_w: int,
    frame_h: int,
) -> Optional[np.ndarray]:
    with open(calib_path) as f:
        poly_eq = np.array(json.load(f)['pitch_polygon'], dtype=np.float32)

    EW, EH = 2880, 1440
    src_pts, dst_pts = [], []

    for i, (px, py) in enumerate(poly_eq):
        eq = np.zeros((EH, EW, 3), dtype=np.uint8)
        cv2.circle(eq, (int(px), int(py)), 8, (255, 255, 255), -1)
        rgb = cv2.cvtColor(eq, cv2.COLOR_BGR2RGB)
        proj = e2p(rgb, fov_deg=fov, u_deg=yaw, v_deg=tilt,
                   out_hw=(frame_h, frame_w), mode='bilinear')
        gray = cv2.cvtColor(cv2.cvtColor(proj, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        locs = np.where(gray > 128)
        if len(locs[0]) > 0:
            src_pts.append([float(np.mean(locs[1])), float(np.mean(locs[0]))])
            px_n, py_n = PITCH_COORDS_NORM[i]
            dst_pts.append([px_n * frame_w, py_n * frame_h])

    if len(src_pts) < 4:
        print(f"  WARNING: only {len(src_pts)} polygon points visible")
        return None

    H, mask = cv2.findHomography(
        np.array(src_pts, dtype=np.float32),
        np.array(dst_pts, dtype=np.float32),
        cv2.RANSAC, 5.0,
    )
    inliers = int(mask.sum()) if mask is not None else 0
    print(f"  Homography: {len(src_pts)} pts, {inliers} inliers")
    return H


def map_to_pitch(fx, fy, H, fw, fh):
    pt = np.array([[[fx, fy]]], dtype=np.float32)
    m  = cv2.perspectiveTransform(pt, H)[0][0]
    return float(m[0] / fw), float(m[1] / fh)


def pixel_to_eq_pan(px, py, frame_w, frame_h, centre_pan, centre_tilt, fov_h):
    """Convert pixel position in perspective frame to equirect pan angle."""
    aspect = frame_w / frame_h
    fov_v  = np.degrees(2 * np.arctan(np.tan(np.radians(fov_h/2)) / aspect))
    ndx = (px - frame_w/2) / (frame_w/2)
    ndy = (py - frame_h/2) / (frame_h/2)
    dpan  = np.degrees(np.arctan(ndx * np.tan(np.radians(fov_h/2))))
    dtilt = np.degrees(np.arctan(ndy * np.tan(np.radians(fov_v/2))))
    return float(centre_pan + dpan), float(centre_tilt - dtilt)


# ──────────────────────────────────────────────────────────────────
# Detection
# ──────────────────────────────────────────────────────────────────

class Detector:
    def __init__(self, player_path: str, ball_path: str, device: str = 'cpu'):
        from ultralytics import YOLO
        print(f"  Loading models on {device}...")
        self.player_model  = YOLO(player_path)
        self.ball_model    = YOLO(ball_path)
        self.device        = device
        self._names        = getattr(self.player_model, 'names', {})

    def detect_players(self, frame: np.ndarray) -> List[dict]:
        res = self.player_model(frame, imgsz=IMGSZ, conf=CONF_PLAYERS,
                                device=self.device, verbose=False)[0]
        out = []
        for b in res.boxes:
            if self._names.get(int(b.cls[0]), '') != 'person':
                continue
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            out.append({'foot_x': (x1+x2)/2, 'foot_y': y2,
                        'conf': float(b.conf[0])})
        return out

    def detect_ball(self, frame: np.ndarray) -> Optional[dict]:
        res = self.ball_model(frame, imgsz=IMGSZ, conf=CONF_BALL,
                              device=self.device, verbose=False)[0]
        balls = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            area = (x2-x1)*(y2-y1)
            if area <= DETECT_W * DETECT_H * 0.02:
                balls.append({'foot_x': (x1+x2)/2, 'foot_y': y2,
                              'conf': float(b.conf[0]), 'area': area})
        if not balls:
            return None
        return max(balls, key=lambda b: (b['conf'], -b['area']))


# ──────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────

def compute_features(players, ball, H, fw, fh, prefix,
                     centre_pan=0.0, centre_tilt=0.0, fov_h=120.0):
    pitch_players = []
    eq_pans  = []
    eq_tilts = []
    if H is not None:
        for p in players:
            px, py = map_to_pitch(p['foot_x'], p['foot_y'], H, fw, fh)
            if -0.1 <= px <= 1.1 and -0.1 <= py <= 1.1:
                pitch_players.append((float(np.clip(px, 0, 1)),
                                      float(np.clip(py, 0, 1))))
            ep, et = pixel_to_eq_pan(p['foot_x'], p['foot_y'], fw, fh,
                                      centre_pan, centre_tilt, fov_h)
            eq_pans.append(ep)
            eq_tilts.append(et)

    if pitch_players:
        xs = [p[0] for p in pitch_players]
        ys = [p[1] for p in pitch_players]
        cx   = float(np.mean(xs))
        cy   = float(np.mean(ys))
        sx   = float(np.std(xs))  if len(xs) > 1 else 0.0
        sy   = float(np.std(ys))  if len(ys) > 1 else 0.0
        plf  = float(sum(1 for x in xs if x < 0.5) / len(xs))
    else:
        cx = cy = 0.5
        sx = sy = 0.0
        plf = 0.5

    # Equirect-space action centroid (key feature)
    if eq_pans:
        action_pan      = float(np.mean(eq_pans))
        action_pan_std  = float(np.std(eq_pans)) if len(eq_pans) > 1 else 0.0
        action_tilt     = float(np.mean(eq_tilts))
        action_tilt_std = float(np.std(eq_tilts)) if len(eq_tilts) > 1 else 0.0
    else:
        action_pan      = centre_pan
        action_pan_std  = 0.0
        action_tilt     = centre_tilt
        action_tilt_std = 0.0

    bx = by = bc = 0.0
    bd = 0
    ball_eq_pan = centre_pan
    if ball is not None and H is not None:
        bpx, bpy = map_to_pitch(ball['foot_x'], ball['foot_y'], H, fw, fh)
        if -0.1 <= bpx <= 1.1 and -0.1 <= bpy <= 1.1:
            bx, by, bc, bd = (float(np.clip(bpx, 0, 1)),
                               float(np.clip(bpy, 0, 1)),
                               float(ball['conf']), 1)
        bep, _ = pixel_to_eq_pan(ball['foot_x'], ball['foot_y'], fw, fh,
                                  centre_pan, centre_tilt, fov_h)
        ball_eq_pan = bep

    return {
        f'{prefix}_player_count':      len(eq_pans),
        f'{prefix}_action_pan':        action_pan,
        f'{prefix}_action_pan_offset': action_pan - centre_pan,
        f'{prefix}_action_pan_std':    action_pan_std,
        f'{prefix}_action_tilt':       action_tilt,
        f'{prefix}_action_tilt_offset': action_tilt - centre_tilt,
        f'{prefix}_action_tilt_std':   action_tilt_std,
        f'{prefix}_centroid_x':        cx,
        f'{prefix}_centroid_y':        cy,
        f'{prefix}_spread_x':          sx,
        f'{prefix}_spread_y':          sy,
        f'{prefix}_players_left_frac': plf,
        f'{prefix}_ball_eq_pan':       ball_eq_pan,
        f'{prefix}_ball_x':            bx,
        f'{prefix}_ball_y':            by,
        f'{prefix}_ball_conf':         bc,
        f'{prefix}_ball_detected':     bd,
    }


# ──────────────────────────────────────────────────────────────────
# FFmpeg stream
# ──────────────────────────────────────────────────────────────────

def open_stream(insv_path, start_s, duration_s):
    cmd = [
        'ffmpeg',
        '-ss', str(max(0, start_s - 1)),
        '-i', insv_path,
        '-ss', '1',
        '-t', str(duration_s + 1),
        '-vf', 'rotate=PI/2*3,v360=fisheye:equirect:ih_fov=200:iv_fov=200,scale=2880:1440',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-an', 'pipe:1',
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def read_frame(proc, w=2880, h=1440):
    data = proc.stdout.read(w * h * 3)
    if len(data) < w * h * 3:
        return None
    return np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3)).copy()


# ──────────────────────────────────────────────────────────────────
# Per-clip processing
# ──────────────────────────────────────────────────────────────────

def process_clip(insv, project, calib, detector, writer, interval_s=0.5, verbose=True):
    if verbose:
        print(f"\n{'='*60}")
        print(f"Clip: {Path(insv).name}")

    keyframes = parse_keyframes(project)
    curve     = StudioCurve(keyframes)
    t_start, t_end = curve.t_min, curve.t_max
    duration = t_end - t_start

    if verbose:
        print(f"  {len(keyframes)} keyframes, {t_start:.1f}s–{t_end:.1f}s "
              f"({duration/60:.1f} min)")

    print("  Building wide homography...")
    H_wide = build_homography(calib, WIDE_FOV, WIDE_YAW, WIDE_TILT, DETECT_W, DETECT_H)
    # Use wide homography for studio view too (same pitch, different viewing angle)
    # This is an approximation — good enough for pitch-relative features
    H_studio = H_wide

    fps   = 29.97
    stride = max(1, int(fps * interval_s))
    proc   = open_stream(insv, t_start, duration)

    frame_idx = rows = 0
    pan_prev = pan_velocity = 0.0
    t0 = time.time()

    while True:
        eq = read_frame(proc)
        if eq is None:
            break
        t = t_start + frame_idx / fps
        if t > t_end:
            break

        if frame_idx % stride == 0:
            pan, tilt, fov = curve.at(t)
            rgb = cv2.cvtColor(eq, cv2.COLOR_BGR2RGB)

            # Studio-pan view only (wide view dropped for speed)
            wf = compute_features([], None, None, DETECT_W, DETECT_H, 'wide')
            studio_bgr = cv2.cvtColor(
                e2p(rgb, fov_deg=fov*STUDIO_FOV_SCALE,
                    u_deg=-pan, v_deg=tilt*STUDIO_TILT_SCALE,
                    out_hw=(DETECT_H, DETECT_W), mode='bilinear'),
                cv2.COLOR_RGB2BGR)
            sp = detector.detect_players(studio_bgr)
            sb = detector.detect_ball(studio_bgr)
            sf = compute_features(sp, sb, H_studio, DETECT_W, DETECT_H, 'studio')

            pan_velocity = pan - pan_prev

            row = {
                'clip':          Path(insv).stem,
                'timestamp_s':   round(t, 3),
                'is_keyframe':   int(curve.is_keyframe(t)),
                'time_since_kf': round(min(curve.time_since_last_kf(t), 30.0), 2),
                'pan_prev':      round(pan_prev, 4),
                'pan_velocity':  round(pan_velocity, 4),
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in wf.items()},
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in sf.items()},
                'target_pan':    round(pan,  4),
                'target_tilt':   round(tilt, 4),
                'target_fov':    round(fov,  4),
            }
            writer.writerow(row)
            rows += 1
            pan_prev = pan

            if verbose and rows % 100 == 0:
                elapsed = time.time() - t0
                rate    = rows / elapsed
                eta     = max(0, (duration / interval_s - rows) / rate / 60)
                print(f"  t={t:.0f}s  rows={rows}  "
                      f"wide={wf['wide_player_count']}p  "
                      f"studio={sf['studio_player_count']}p  "
                      f"pan={pan:.1f}°  eta={eta:.1f}min")

        frame_idx += 1

    proc.stdout.close()
    proc.wait()
    elapsed = time.time() - t0
    if verbose:
        print(f"  Done: {rows} rows in {elapsed/60:.1f}min "
              f"({rows/elapsed:.1f} rows/s)")
    return rows


# ──────────────────────────────────────────────────────────────────
# Schema + main
# ──────────────────────────────────────────────────────────────────

FIELDNAMES = [
    'clip', 'timestamp_s', 'is_keyframe', 'time_since_kf',
    'pan_prev', 'pan_velocity',
    'wide_player_count', 'wide_action_pan', 'wide_action_pan_offset',
    'wide_action_pan_std', 'wide_action_tilt', 'wide_action_tilt_offset',
    'wide_action_tilt_std', 'wide_centroid_x', 'wide_centroid_y',
    'wide_spread_x', 'wide_spread_y', 'wide_players_left_frac',
    'wide_ball_eq_pan', 'wide_ball_x', 'wide_ball_y',
    'wide_ball_conf', 'wide_ball_detected',
    'studio_player_count', 'studio_action_pan', 'studio_action_pan_offset',
    'studio_action_pan_std', 'studio_action_tilt', 'studio_action_tilt_offset',
    'studio_action_tilt_std', 'studio_centroid_x', 'studio_centroid_y',
    'studio_spread_x', 'studio_spread_y', 'studio_players_left_frac',
    'studio_ball_eq_pan', 'studio_ball_x', 'studio_ball_y',
    'studio_ball_conf', 'studio_ball_detected',
    'target_pan', 'target_tilt', 'target_fov',
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clips',    required=True)
    p.add_argument('--output',   default='dataset.csv')
    p.add_argument('--players',  default='models/yolo11n.pt')
    p.add_argument('--ball',     default='models/ball.pt')
    p.add_argument('--device',   default=None)
    p.add_argument('--interval', type=float, default=0.5)
    p.add_argument('--append',   action='store_true')
    args = p.parse_args()

    if args.device is None:
        try:
            import torch
            device = ('mps' if torch.backends.mps.is_available() else
                      'cuda' if torch.cuda.is_available() else 'cpu')
        except ImportError:
            device = 'cpu'
    else:
        device = args.device
    print(f"Device: {device}")

    with open(args.clips) as f:
        clips = json.load(f)
    print(f"Clips: {len(clips)}")

    detector = Detector(args.players, args.ball, device=device)

    mode = 'a' if args.append else 'w'
    with open(args.output, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if not args.append:
            writer.writeheader()
        total = 0
        for clip in clips:
            insv, project, calib = clip['insv'], clip['project'], clip['calib']
            missing = [k for k in ('insv','project','calib')
                       if not Path(clip[k]).exists()]
            if missing:
                print(f"SKIP {Path(insv).name} — missing: {missing}")
                continue
            total += process_clip(insv, project, calib, detector, writer,
                                  interval_s=args.interval)

    print(f"\nTotal rows: {total} → {args.output}")


if __name__ == '__main__':
    main()
