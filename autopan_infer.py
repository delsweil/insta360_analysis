#!/usr/bin/env python3
"""
autopan_infer.py — Apply trained autopan model to unannotated footage.

Extracts 5 random 30-second segments from a game clip, runs the
trained pan MLP, smooths the output, and renders perspective video.

Usage:
    python autopan_infer.py \
        --insv  /path/to/VID_xxx_10_001.insv \
        --model models/autopan_model.pkl \
        --calib calibration/pitch_game2.json \
        --players models/yolo11n.pt \
        --output /tmp/infer_test.mp4 \
        [--segments 5] \
        [--seg-duration 30] \
        [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from py360convert import e2p
from scipy.interpolate import PchipInterpolator

# ── Output parameters ─────────────────────────────────────────────
OUT_W   = 1280
OUT_H    = 720
OUT_FPS  = 29.97
# Tilt and FOV derived from calibration polygon at runtime

# ── Detection parameters ──────────────────────────────────────────
DETECT_FOV   = 120.0
DETECT_YAW   =   0.0
DETECT_TILT  = -15.0
DETECT_W     = 1280
DETECT_H     =  720
IMGSZ        =  640
CONF_PLAYERS = 0.15

# ── Pitch polygon coordinate assignments ──────────────────────────
PITCH_COORDS_NORM = [
    (0.00,0.00),(0.17,0.00),(0.33,0.00),(0.50,0.00),
    (0.67,0.00),(0.83,0.00),(1.00,0.00),
    (1.00,0.25),(1.00,0.50),(1.00,0.75),
    (0.83,1.00),(0.50,1.00),(0.17,1.00),(0.00,1.00),
    (0.00,0.75),(0.00,0.50),
]


def derive_tilt_fov_from_calib(calib_path):
    """Estimate sensible default tilt and FOV from pitch polygon."""
    with open(calib_path) as f:
        poly = np.array(json.load(f)['pitch_polygon'], dtype=np.float32)
    EW, EH = 2880, 1440
    tilt_deg = (0.5 - poly[:,1] / EH) * 180
    far_tilt  = float(np.mean(tilt_deg[:7]))
    near_tilt = float(np.mean(tilt_deg[7:]))
    mid_tilt  = (far_tilt + near_tilt) / 2
    # Convert to e2p tilt using our calibrated scale
    e2p_tilt = mid_tilt * 1.20
    # FOV: span from far to near touchline + margin
    tilt_span = abs(near_tilt - far_tilt)
    e2p_fov   = min(130, max(80, tilt_span * 1.85))
    print(f"  Calibration: far_tilt={far_tilt:.1f}° near_tilt={near_tilt:.1f}°")
    print(f"  Derived: e2p_tilt={e2p_tilt:.1f}° e2p_fov={e2p_fov:.1f}°")
    return e2p_tilt, e2p_fov


def build_homography(calib_path, fov, yaw, tilt, fw, fh):
    with open(calib_path) as f:
        poly_eq = np.array(json.load(f)['pitch_polygon'], dtype=np.float32)
    EW, EH = 2880, 1440
    src, dst = [], []
    for i, (px, py) in enumerate(poly_eq):
        eq = np.zeros((EH, EW, 3), dtype=np.uint8)
        cv2.circle(eq, (int(px), int(py)), 8, (255,255,255), -1)
        rgb = cv2.cvtColor(eq, cv2.COLOR_BGR2RGB)
        proj = e2p(rgb, fov_deg=fov, u_deg=yaw, v_deg=tilt,
                   out_hw=(fh, fw), mode='bilinear')
        gray = cv2.cvtColor(cv2.cvtColor(proj, cv2.COLOR_RGB2BGR),
                            cv2.COLOR_BGR2GRAY)
        locs = np.where(gray > 128)
        if len(locs[0]) > 0:
            src.append([float(np.mean(locs[1])), float(np.mean(locs[0]))])
            px_n, py_n = PITCH_COORDS_NORM[i]
            dst.append([px_n * fw, py_n * fh])
    if len(src) < 4:
        return None
    H, _ = cv2.findHomography(np.array(src, dtype=np.float32),
                               np.array(dst, dtype=np.float32), cv2.RANSAC, 5.0)
    return H


def pixel_to_eq_pan(px, py, fw, fh, centre_pan, centre_tilt, fov_h):
    aspect = fw / fh
    fov_v  = np.degrees(2 * np.arctan(np.tan(np.radians(fov_h/2)) / aspect))
    ndx = (px - fw/2) / (fw/2)
    ndy = (py - fh/2) / (fh/2)
    dpan  = np.degrees(np.arctan(ndx * np.tan(np.radians(fov_h/2))))
    dtilt = np.degrees(np.arctan(ndy * np.tan(np.radians(fov_v/2))))
    return float(centre_pan + dpan), float(centre_tilt - dtilt)


def map_to_pitch(fx, fy, H, fw, fh):
    pt = np.array([[[fx, fy]]], dtype=np.float32)
    m  = cv2.perspectiveTransform(pt, H)[0][0]
    return float(m[0]/fw), float(m[1]/fh)


def get_clip_duration(insv_path):
    result = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', insv_path],
        capture_output=True, text=True)
    txt = result.stdout.strip()
    if not txt:
        # fallback: try stderr
        result2 = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'csv=p=0', insv_path],
            capture_output=True, text=True)
        txt = result2.stdout.strip()
    if not txt:
        # last resort: assume 30 minutes
        print("WARNING: could not determine duration, assuming 1800s")
        return 1800.0
    return float(txt)


def open_stream(insv_path, start_s, duration_s):
    cmd = [
        'ffmpeg',
        '-ss', str(max(0, start_s)),
        '-i', insv_path,
        '-t', str(duration_s + 2),
        '-vf', 'rotate=PI/2*3,v360=fisheye:equirect:ih_fov=200:iv_fov=200,scale=2880:1440',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-an', 'pipe:1',
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def read_frame(proc, w=2880, h=1440):
    nbytes = w * h * 3
    chunks = []
    remaining = nbytes
    while remaining > 0:
        chunk = proc.stdout.read(min(65536, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b''.join(chunks)
    if len(data) < nbytes:
        return None
    return np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3)).copy()


def open_writer(output_path, fps):
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{OUT_W}x{OUT_H}', '-r', str(fps),
        '-i', 'pipe:0',
        '-c:v', 'h264_videotoolbox', '-b:v', '8000k',
        '-pix_fmt', 'yuv420p', output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def smooth_pan_curve(times, pans, smooth_s=1.0):
    """Smooth pan predictions using PchipInterpolator on keypoints."""
    if len(times) < 2:
        return times, pans
    # Subsample to one point per smooth_s seconds for interpolation
    interp = PchipInterpolator(times, pans)
    times_dense = np.linspace(times[0], times[-1], len(times))
    return times_dense, interp(times_dense)


def detect_players(frame, player_model, device):
    results = player_model(frame, imgsz=IMGSZ, conf=CONF_PLAYERS,
                           device=device, verbose=False)[0]
    names = player_model.names
    players = []
    for b in results.boxes:
        if names.get(int(b.cls[0]), '') != 'person':
            continue
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        players.append({'foot_x': (x1+x2)/2, 'foot_y': y2, 'conf': float(b.conf[0])})
    return players


def run_segment(insv_path, start_s, duration_s, model_data, H,
                player_model, device, pan_init=0.0, tilt_init=-19.0,
                fov_init=111.0):
    """
    Run inference on one segment. Returns list of (t, pan, tilt, fov) tuples.
    """
    mlp     = model_data['pan_model']
    scaler  = model_data['pan_scaler']
    features = model_data['pan_features']

    fps = OUT_FPS
    proc = open_stream(insv_path, start_s, duration_s)

    tilt_cur      = tilt_init
    fov_cur       = fov_init
    time_since_kf = 0.0

    # Warm start: detect players on first frame to initialise pan
    first_eq = None
    warm_proc = open_stream(insv_path, start_s, 2)
    first_eq = read_frame(warm_proc)
    warm_proc.stdout.close()
    warm_proc.wait()

    if first_eq is not None and H is not None:
        rgb0 = cv2.cvtColor(first_eq, cv2.COLOR_BGR2RGB)
        det0 = cv2.cvtColor(
            e2p(rgb0, fov_deg=DETECT_FOV, u_deg=0, v_deg=DETECT_TILT,
                out_hw=(DETECT_H, DETECT_W), mode='bilinear'),
            cv2.COLOR_RGB2BGR)
        players0 = detect_players(det0, player_model, device)
        if players0:
            eq_pans0 = [pixel_to_eq_pan(p['foot_x'], p['foot_y'],
                                         DETECT_W, DETECT_H, 0, DETECT_TILT,
                                         DETECT_FOV)[0] for p in players0]
            pan_init = float(np.mean(eq_pans0))
            print(f"  Warm start: {len(players0)} players detected, "
                  f"initial pan={pan_init:.1f}°")
        else:
            print(f"  Warm start: no players detected, using pan=0°")

    pan_prev     = pan_init
    pan_velocity = 0.0

    predictions = []  # (t, pan, tilt, fov)
    frame_idx   = 0

    while True:
        eq = read_frame(proc)
        if eq is None:
            break
        t = start_s + frame_idx / fps
        if t > start_s + duration_s:
            break

        rgb = cv2.cvtColor(eq, cv2.COLOR_BGR2RGB)

        # ── Detection ──────────────────────────────────────────
        # Use current pan for detection view
        det_frame = cv2.cvtColor(
            e2p(rgb, fov_deg=fov_cur * (1/1.85) * DETECT_FOV/120 * 120,
                u_deg=-pan_prev, v_deg=DETECT_TILT,
                out_hw=(DETECT_H, DETECT_W), mode='bilinear'),
            cv2.COLOR_RGB2BGR)

        players = detect_players(det_frame, player_model, device)

        # ── Feature extraction ─────────────────────────────────
        eq_pans = []
        if H is not None and players:
            for p in players:
                ep, _ = pixel_to_eq_pan(p['foot_x'], p['foot_y'],
                                         DETECT_W, DETECT_H,
                                         pan_prev, DETECT_TILT, DETECT_FOV)
                eq_pans.append(ep)

        if eq_pans:
            action_pan        = float(np.mean(eq_pans))
            action_pan_offset = action_pan - pan_prev
            action_pan_std    = float(np.std(eq_pans)) if len(eq_pans) > 1 else 0.0
        else:
            action_pan        = pan_prev
            action_pan_offset = 0.0
            action_pan_std    = 0.0

        # Centroid_y in frame space
        if players:
            centroid_y = float(np.mean([p['foot_y'] for p in players])) / DETECT_H
            spread_x   = float(np.std([(p['foot_x']) for p in players]) / DETECT_W) \
                         if len(players) > 1 else 0.0
        else:
            centroid_y = 0.5
            spread_x   = 0.0

        # Build feature vector
        feat_dict = {
            'pan_prev':                  pan_prev,
            'pan_velocity':              pan_velocity,
            'studio_action_pan_offset':  action_pan_offset,
            'studio_action_pan_std':     action_pan_std,
            'studio_player_count':       len(players),
            'studio_centroid_y':         centroid_y,
            'studio_spread_x':           spread_x,
            'time_since_kf':             min(time_since_kf, 30.0),
        }
        X = np.array([[feat_dict[f] for f in features]])
        X_s = scaler.transform(X)

        # ── Predict pan ────────────────────────────────────────
        pan_pred = float(mlp.predict(X_s)[0])
        pan_pred = float(np.clip(pan_pred, -80, 80))

        predictions.append((t, pan_pred, tilt_cur, fov_cur))

        pan_velocity = pan_pred - pan_prev
        pan_prev     = pan_pred
        time_since_kf += 1.0 / fps
        frame_idx    += 1

    proc.stdout.close()
    proc.wait()
    return predictions


def render_segment(insv_path, start_s, duration_s, predictions, writer):
    """Render one segment using smoothed predictions."""
    if not predictions:
        return

    # Smooth pan with PchipInterpolator
    times = np.array([p[0] for p in predictions])
    pans  = np.array([p[1] for p in predictions])

    # Smooth over ~2 second window by downsampling then interpolating
    fps = OUT_FPS
    stride = max(1, int(fps * 2.0))
    kf_idx = list(range(0, len(times), stride))
    if kf_idx[-1] != len(times) - 1:
        kf_idx.append(len(times) - 1)

    kf_times = times[kf_idx]
    kf_pans  = pans[kf_idx]
    smoother = PchipInterpolator(kf_times, kf_pans)
    pans_smooth = smoother(times)

    tilt = predictions[0][2]
    fov  = predictions[0][3]

    # Stream equirect and render
    proc = open_stream(insv_path, start_s, duration_s)

    for i, (t, _, _, _) in enumerate(predictions):
        eq = read_frame(proc)
        if eq is None:
            break

        pan_s = float(pans_smooth[i])
        e2p_pan  = -pan_s
        e2p_tilt = tilt   # already in e2p space
        e2p_fov  = fov    # already in e2p space

        rgb  = cv2.cvtColor(eq, cv2.COLOR_BGR2RGB)
        persp = e2p(rgb, fov_deg=e2p_fov, u_deg=e2p_pan, v_deg=e2p_tilt,
                    out_hw=(OUT_H, OUT_W), mode='bilinear')
        out = cv2.cvtColor(persp, cv2.COLOR_RGB2BGR)

        # Burn pan value onto frame for debugging
        cv2.putText(out, f"pan={pan_s:+.1f}° t={t-start_s:.1f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        writer.stdin.write(out.tobytes())

    proc.stdout.close()
    proc.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--insv',     required=True)
    parser.add_argument('--model',    default='models/autopan_model.pkl')
    parser.add_argument('--calib',    required=True)
    parser.add_argument('--players',  default='models/yolo11n.pt')
    parser.add_argument('--output',   default='/tmp/infer_test.mp4')
    parser.add_argument('--segments', type=int,   default=5)
    parser.add_argument('--seg-duration', type=float, default=30.0)
    parser.add_argument('--device',   default=None)
    parser.add_argument('--seed',     type=int,   default=42)
    args = parser.parse_args()

    # Device
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

    # Load model
    with open(args.model, 'rb') as f:
        model_data = pickle.load(f)
    print(f"Model loaded: {model_data.get('training_rows',0)} training rows")

    # Load detector
    from ultralytics import YOLO
    player_model = YOLO(args.players)
    print("Player model loaded")

    # Derive tilt/fov from calibration
    print("Deriving tilt/FOV from calibration...")
    tilt_default, fov_default = derive_tilt_fov_from_calib(args.calib)

    # Build homography
    print("Building homography...")
    H = build_homography(args.calib, DETECT_FOV, DETECT_YAW, DETECT_TILT,
                         DETECT_W, DETECT_H)
    print(f"Homography: {'OK' if H is not None else 'FAILED'}")

    # Get clip duration and pick random segments
    duration = get_clip_duration(args.insv)
    print(f"Clip duration: {duration:.1f}s")

    rng = np.random.default_rng(args.seed)
    max_start = duration - args.seg_duration - 5
    if max_start < 0:
        print("Clip too short!")
        return

    starts = sorted(rng.uniform(10, max_start, args.segments).tolist())
    print(f"\nSelected {args.segments} segments:")
    for i, s in enumerate(starts):
        print(f"  {i+1}: t={s:.1f}s – {s+args.seg_duration:.1f}s")

    # Open video writer
    writer = open_writer(args.output, OUT_FPS)

    # Process each segment
    for i, start_s in enumerate(starts):
        print(f"\n[{i+1}/{args.segments}] Segment at t={start_s:.1f}s")
        t0 = time.time()

        # Inference pass
        predictions = run_segment(
            args.insv, start_s, args.seg_duration,
            model_data, H, player_model, device,
            tilt_init=tilt_default, fov_init=fov_default,
        )
        print(f"  Inference: {len(predictions)} frames in {time.time()-t0:.1f}s")

        # Render pass
        t1 = time.time()
        render_segment(args.insv, start_s, args.seg_duration, predictions, writer)
        print(f"  Render:    {time.time()-t1:.1f}s")

    writer.stdin.close()
    writer.wait()
    print(f"\nDone. Output: {args.output}")


if __name__ == '__main__':
    main()
