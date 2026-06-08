#!/usr/bin/env python3
"""
Steady Cam autopan inference.
Reads a wide-angle MP4, detects ball/players, outputs a panning 16:9 crop.

Usage:
    python3 steadycam_infer.py \
        --video path/to/game.mp4 \
        --calib path/to/game.calib.json \
        --output path/to/output.mp4 \
        --ball models/ball_v5_yolo11s_1280_candidate.pt \
        --players models/yolo11s.pt
"""

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False
    AutoDetectionModel = None
    get_sliced_prediction = None

# ── Output ────────────────────────────────────────────────────────────────────
OUT_W, OUT_H   = 1280, 720
OUT_FPS        = 25.0

# ── Detection ─────────────────────────────────────────────────────────────────
DETECT_EVERY   = 5       # run detection every N frames
CONF_BALL      = 0.40
CONF_PLAYER    = 0.40
IMGSZ_BALL     = 1280
IMGSZ_PLAYER   = 640

# ── Pan control ───────────────────────────────────────────────────────────────
PAN_GAIN       = 0.08    # how fast camera pans toward target
PAN_MAX_STEP   = 20      # max pixels per frame
TARGET_ALPHA   = 0.15    # target smoothing
HOLD_THRESHOLD = 0.75    # only move when target beyond this fraction of half-frame

# ── Zoom control ──────────────────────────────────────────────────────────────
CROP_NEAR      = 1280    # crop width for near side (no zoom)
CROP_FAR       = 820     # crop width for far side (zoom in ~1.56x)
ZOOM_ALPHA     = 0.04    # zoom smoothing (slow transitions)

# ── Ball tracking ─────────────────────────────────────────────────────────────
BALL_MAX_JUMP  = 200     # max pixel jump between detections


@dataclass
class CamState:
    x: float = 0.0          # current crop centre x
    crop_w: float = CROP_NEAR  # current crop width (for zoom)


@dataclass
class TargetState:
    tx: float = 0.0          # smoothed target x
    crop_w: float = CROP_NEAR


def load_calib(calib_path: str):
    with open(calib_path) as f:
        return json.load(f)


def build_mask(calib: dict, h: int, w: int) -> np.ndarray:
    far_pts  = [tuple(p) for p in calib['far_pts']]
    near_pts = [tuple(p) for p in calib['near_pts']]
    poly = np.array(far_pts + near_pts[::-1], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def detect_players(frame, model, mask=None):
    res = model(frame, imgsz=IMGSZ_PLAYER, conf=CONF_PLAYER,
                classes=[0], verbose=False)[0]
    players = []
    for b in res.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        cx, cy = (x1+x2)//2, (y1+y2)//2
        h_px = y2 - y1
        if mask is not None and not mask[min(cy, mask.shape[0]-1),
                                         min(cx, mask.shape[1]-1)]:
            continue
        players.append((cx, cy, h_px))
    return players


def detect_ball(frame, model, mask=None, last_pos=None, sahi_model=None, sahi_slice=640):
    if sahi_model is not None:
        cv2.imwrite('/tmp/_sc_sahi.jpg', frame)
        result = get_sliced_prediction(
            '/tmp/_sc_sahi.jpg', sahi_model,
            slice_height=sahi_slice, slice_width=sahi_slice,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2,
            verbose=0)
        best = None
        for pred in result.object_prediction_list:
            bbox = pred.bbox
            cx = (bbox.minx + bbox.maxx) / 2
            cy = (bbox.miny + bbox.maxy) / 2
            conf = pred.score.value
            if mask is not None:
                xi = int(np.clip(cx, 0, mask.shape[1]-1))
                yi = int(np.clip(cy, 0, mask.shape[0]-1))
                if not mask[yi, xi]:
                    continue
            if last_pos is not None:
                dist = math.hypot(cx - last_pos[0], cy - last_pos[1])
                if dist > BALL_MAX_JUMP:
                    conf *= 0.3
            if best is None or conf > best[2]:
                best = (cx, cy, conf)
        return best if (best and best[2] >= CONF_BALL) else None
    else:
        res = model(frame, imgsz=IMGSZ_BALL, conf=CONF_BALL, verbose=False)[0]
        best = None
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            cx, cy = (x1+x2)//2, (y1+y2)//2
            conf = float(b.conf[0])
            if mask is not None:
                xi = int(np.clip(cx, 0, mask.shape[1]-1))
                yi = int(np.clip(cy, 0, mask.shape[0]-1))
                if not mask[yi, xi]:
                    continue
            if last_pos is not None:
                dist = math.hypot(cx - last_pos[0], cy - last_pos[1])
                if dist > BALL_MAX_JUMP:
                    conf *= 0.3
            if best is None or conf > best[2]:
                best = (cx, cy, conf)
        return best if best else None


def player_cluster_x(players: list) -> Optional[float]:
    """Return weighted mean x of the densest player cluster."""
    if not players:
        return None
    if len(players) == 1:
        return float(players[0][0])
    xs = np.array([p[0] for p in players], dtype=float)
    return float(np.mean(xs))


def target_crop_w(players: list, ball: Optional[tuple],
                  frame_h: int, mask_far_y: float) -> float:
    """
    Compute target crop width based on action y-position.
    Higher y (near side) → wider crop.
    Lower y (far side) → narrower crop (zoom in).
    """
    ys = []
    if ball:
        ys.append(ball[1])
    ys += [p[1] for p in players[:6]]
    if not ys:
        return CROP_NEAR
    mean_y = float(np.mean(ys))
    # Normalise: mask_far_y = top of pitch, frame_h = bottom
    t = np.clip((mean_y - mask_far_y) / (frame_h - mask_far_y), 0, 1)
    return CROP_FAR + t * (CROP_NEAR - CROP_FAR)


def process_video(video_path: str, calib_path: str,
                  output_path: str,
                  player_model, ball_model,
                  sahi_model=None, sahi_slice=640,
                  debug=False, log_csv=None):

    calib = load_calib(calib_path)
    fw = calib['frame_width']
    fh = calib['frame_height']

    # Build pitch mask
    mask = build_mask(calib, fh, fw)
    # Far side y = topmost point of mask
    ys = [p[1] for p in calib['far_pts']]
    mask_far_y = float(min(ys))

    # Also open top third of mask for aerial balls
    ball_mask = mask.copy()
    ball_mask[:fh//3, :] = 255

    # FFmpeg reader
    cmd_in = [
        'ffmpeg', '-i', video_path,
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-loglevel', 'error', 'pipe:1'
    ]
    reader = subprocess.Popen(cmd_in, stdout=subprocess.PIPE)

    # FFmpeg writer
    cmd_out = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{OUT_W}x{OUT_H}', '-r', str(OUT_FPS),
        '-i', 'pipe:0',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-loglevel', 'error', output_path
    ]
    writer = subprocess.Popen(cmd_out, stdin=subprocess.PIPE)

    csv_file = open(log_csv, 'w') if log_csv else None
    if csv_file:
        csv_file.write('frame,cam_x,target_x,crop_w,ball_x,ball_y,ball_conf,n_players,mode\n')

    cam   = CamState(x=fw/2, crop_w=CROP_NEAR)
    tgt   = TargetState(tx=fw/2, crop_w=CROP_NEAR)
    last_ball = None
    last_players = []
    frame_idx = 0
    ball_count = 0
    mode_counts = {}

    bytes_per_frame = fw * fh * 3

    print(f"Processing {Path(video_path).name} → {Path(output_path).name}")

    while True:
        raw = reader.stdout.read(bytes_per_frame)
        if len(raw) < bytes_per_frame:
            break

        arr = np.frombuffer(raw, dtype=np.uint8).copy()
        frame = arr.reshape((fh, fw, 3))

        # Detection every N frames
        if frame_idx % DETECT_EVERY == 0:
            last_players = detect_players(frame, player_model, mask)
            if ball_model is not None:
                last_ball = detect_ball(
                    frame, ball_model, ball_mask,
                    last_pos=(last_ball[0], last_ball[1]) if last_ball else None,
                    sahi_model=sahi_model, sahi_slice=sahi_slice)
                if last_ball:
                    ball_count += 1

        # Choose target x
        mode = 'hold'
        if last_ball and last_ball[2] >= 0.55:
            raw_tx = last_ball[0]
            mode = 'ball_highconf'
        elif last_ball:
            raw_tx = last_ball[0]
            mode = 'ball'
        elif last_players:
            raw_tx = player_cluster_x(last_players) or cam.x
            mode = 'players'
        else:
            raw_tx = cam.x
            mode = 'hold'

        mode_counts[mode] = mode_counts.get(mode, 0) + 1

        # Smooth target
        tgt.tx = tgt.tx * (1 - TARGET_ALPHA) + raw_tx * TARGET_ALPHA

        # Compute adaptive crop width
        target_cw = target_crop_w(last_players, last_ball, fh, mask_far_y)
        tgt.crop_w = tgt.crop_w * (1 - ZOOM_ALPHA) + target_cw * ZOOM_ALPHA

        # Hold threshold — only move if target meaningfully off-centre
        half_cw = tgt.crop_w / 2
        offset = tgt.tx - cam.x
        if abs(offset) > half_cw * HOLD_THRESHOLD:
            step = np.clip(offset * PAN_GAIN, -PAN_MAX_STEP, PAN_MAX_STEP)
            cam.x += step

        # Clamp camera x so crop stays within frame
        half_out = tgt.crop_w / 2
        cam.x = float(np.clip(cam.x, half_out, fw - half_out))
        cam.crop_w = tgt.crop_w

        # Extract crop
        x1 = int(cam.x - cam.crop_w / 2)
        x2 = int(cam.x + cam.crop_w / 2)
        y1 = 0
        y2 = fh
        crop = frame[y1:y2, x1:x2]
        output = cv2.resize(crop, (OUT_W, OUT_H))

        # Debug overlay
        if debug:
            # Draw mask boundary
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (255, 100, 0), 2)
            # Draw players
            for px, py, ph in last_players:
                cv2.circle(frame, (px, py), 8, (0, 255, 0), 2)
            # Draw ball
            if last_ball:
                bx, by, bc = last_ball
                col = (0, 255, 255) if bc >= 0.55 else (0, 165, 255)
                cv2.circle(frame, (int(bx), int(by)), 12, col, 3)
                cv2.putText(frame, f"{bc:.2f}", (int(bx)+15, int(by)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            # Draw crop window
            cv2.rectangle(frame, (x1, y1+10), (x2, y2-10), (0, 0, 255), 3)
            cv2.putText(frame, f"x={cam.x:.0f} cw={cam.crop_w:.0f} {mode}",
                        (30, fh-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 0), 2)
            # Show scaled debug frame
            debug_small = cv2.resize(frame, (OUT_W, OUT_H))
            cv2.imshow('Debug', debug_small)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        writer.stdin.write(output.tobytes())

        if csv_file:
            bx = f"{last_ball[0]:.1f}" if last_ball else ""
            by = f"{last_ball[1]:.1f}" if last_ball else ""
            bc = f"{last_ball[2]:.3f}" if last_ball else ""
            csv_file.write(f"{frame_idx},{cam.x:.1f},{tgt.tx:.1f},"
                           f"{cam.crop_w:.1f},{bx},{by},{bc},"
                           f"{len(last_players)},{mode}\n")

        frame_idx += 1
        if frame_idx % 500 == 0:
            fps_est = frame_idx / max(1, frame_idx)
            print(f"  {frame_idx} frames  ball={ball_count}  modes={mode_counts}")

    reader.stdout.close()
    reader.wait()
    writer.stdin.close()
    writer.wait()
    if csv_file:
        csv_file.close()
    if debug:
        cv2.destroyAllWindows()

    print(f"\nDone: {frame_idx} frames")
    print(f"Ball detections: {ball_count}/{frame_idx // DETECT_EVERY}")
    print(f"Modes: {mode_counts}")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video',   required=True)
    parser.add_argument('--calib',   required=True)
    parser.add_argument('--output',  required=True)
    parser.add_argument('--ball',    default='models/ball_v5_yolo11s_1280_candidate.pt')
    parser.add_argument('--players', default='models/yolo11s.pt')
    parser.add_argument('--device',  default='mps')
    parser.add_argument('--sahi',    action='store_true')
    parser.add_argument('--sahi-slice', type=int, default=640)
    parser.add_argument('--debug',   action='store_true')
    parser.add_argument('--log-csv', default=None, metavar='PATH')
    args = parser.parse_args()

    player_model = YOLO(args.players)
    ball_model   = YOLO(args.ball) if args.ball else None

    sahi_model = None
    if args.sahi and args.ball and SAHI_AVAILABLE:
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path=args.ball,
            confidence_threshold=CONF_BALL,
            device=args.device)
        print(f"SAHI loaded (slice={args.sahi_slice})")
    elif args.sahi and not SAHI_AVAILABLE:
        print("WARNING: SAHI not available")

    process_video(
        args.video, args.calib, args.output,
        player_model, ball_model,
        sahi_model=sahi_model,
        sahi_slice=args.sahi_slice,
        debug=args.debug,
        log_csv=args.log_csv)


if __name__ == '__main__':
    main()
