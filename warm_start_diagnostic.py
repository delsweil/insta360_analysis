#!/usr/bin/env python3
"""
warm_start_diagnostic.py
Extracts and annotates warm start frames for each scan yaw angle.
Shows player detections, cluster, and ball detection at each candidate yaw.

Usage:
    python3 warm_start_diagnostic.py \
        --insv "/Volumes/Sickis disk/DCIM/Camera01/VID_20241028_160117_10_004.insv" \
        --calib calibration/pitch_asn_a.json \
        --players models/yolo11s.pt \
        --ball models/ball_v4.pt \
        --start-times 250,450,650,850,1000 \
        --output-dir /tmp/warm_start_diag \
        --device mps
"""

import argparse
import cv2
import numpy as np
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

SCAN_YAWS = [-40, -20, 0, 20, 40]
OUT_W, OUT_H = 1280, 720


def read_frame(insv_path, t):
    cmd = [
        'ffmpeg', '-ss', str(t), '-i', insv_path,
        '-vf', 'rotate=PI/2*3,v360=fisheye:equirect:ih_fov=200:iv_fov=200,scale=2880:1440',
        '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-loglevel', 'error', 'pipe:1'
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if not proc.stdout: return None
    arr = np.frombuffer(proc.stdout, dtype=np.uint8)
    if arr.size != 2880 * 1440 * 3: return None
    return arr.reshape((1440, 2880, 3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--insv', required=True)
    parser.add_argument('--calib', required=True)
    parser.add_argument('--players', default='models/yolo11s.pt')
    parser.add_argument('--ball', default=None)
    parser.add_argument('--start-times', default='250,450,650,850,1000')
    parser.add_argument('--output-dir', default='/tmp/warm_start_diag')
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    from ultralytics import YOLO
    from py360convert import e2p
    from autopan_infer import (derive_tilt_fov, detect_players, detect_ball,
                                find_action_cluster, build_pitch_mask)

    player_model = YOLO(args.players)
    ball_model = YOLO(args.ball) if args.ball else None
    e2p_tilt, e2p_fov = derive_tilt_fov(args.calib)

    # Try loading cluster selector
    import pickle, pathlib as _pl
    cluster_selector = cluster_selector_features = None
    sel_path = _pl.Path("models/cluster_selector.pkl")
    if sel_path.exists():
        with open(sel_path, 'rb') as f:
            cluster_selector, cluster_selector_features = pickle.load(f)

    starts = [float(t) for t in args.start_times.split(',')]

    for t0 in starts:
        print(f"\nProcessing t={t0:.0f}s...")
        eq = read_frame(args.insv, t0)
        if eq is None:
            print(f"  Failed to read frame")
            continue

        rgb = cv2.cvtColor(eq, cv2.COLOR_BGR2RGB)

        # One image per scan yaw
        frames = []
        for scan_yaw in SCAN_YAWS:
            from py360convert import e2p as py_e2p
            pv_rgb = py_e2p(rgb, fov_deg=e2p_fov,
                            u_deg=-scan_yaw, v_deg=e2p_tilt,
                            out_hw=(OUT_H, OUT_W), mode='bilinear')
            pv = cv2.cvtColor(pv_rgb, cv2.COLOR_RGB2BGR)

            # Build mask
            mask = build_pitch_mask(args.calib, scan_yaw, e2p_tilt, e2p_fov, OUT_H, OUT_W)

            # Detect players
            players = detect_players(pv, player_model, args.device, mask)

            # Detect ball
            ball = None
            if ball_model is not None:
                ball = detect_ball(pv, ball_model, args.device, mask)

            # Find cluster
            cluster = None
            if len(players) >= 2:
                cluster = find_action_cluster(players, scan_yaw, e2p_fov,
                                              cluster_selector, cluster_selector_features)

            # Draw annotations
            vis = pv.copy()

            # Pitch mask contour
            if mask is not None:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, (255, 100, 0), 1)

            # Players
            for cx, cy in players:
                cv2.circle(vis, (int(cx), int(cy)), 8, (0, 255, 0), 2)

            # Ball
            if ball is not None:
                col = (0, 0, 255) if ball[2] >= 0.50 else (0, 165, 255)
                cv2.circle(vis, (int(ball[0]), int(ball[1])), 12, col, 3)
                cv2.putText(vis, f"{ball[2]:.2f}",
                            (int(ball[0]) + 14, int(ball[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            # Cluster
            if cluster is not None:
                cv2.circle(vis, (int(cluster[0]), int(cluster[1])), 14, (0, 255, 255), 3)

            # Score
            score = 0
            if cluster is not None:
                centredness = 1.0 - abs(cluster[0] - OUT_W/2) / (OUT_W/2)
                score = len(players) * centredness

            # HUD
            ball_str = f"BALL conf={ball[2]:.2f}" if (ball and ball[2] >= 0.50) else ""
            cv2.putText(vis,
                        f"yaw={scan_yaw:+d}° players={len(players)} score={score:.1f} {ball_str}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255) if ball_str else (255, 255, 255), 2)

            frames.append(vis)
            print(f"  yaw={scan_yaw:+d}°: {len(players)} players "
                  f"cluster={'yes' if cluster else 'no'} "
                  f"ball={'yes' if ball and ball[2]>=0.50 else 'no'} "
                  f"score={score:.1f}")

        # Stitch into contact sheet (2 rows of 3 and 2)
        if frames:
            row1 = np.hstack(frames[:3])
            row2_frames = frames[3:]
            # Pad row2 to same width
            pad_w = row1.shape[1] - sum(f.shape[1] for f in row2_frames)
            pad = np.zeros((OUT_H, pad_w, 3), dtype=np.uint8)
            row2 = np.hstack(row2_frames + [pad])
            sheet = np.vstack([row1, row2])

            # Add timestamp header
            header = np.zeros((40, sheet.shape[1], 3), dtype=np.uint8)
            cv2.putText(header, f"t={t0:.0f}s — scan yaws: {SCAN_YAWS}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            sheet = np.vstack([header, sheet])

            out_path = out_dir / f"t{int(t0):04d}_warmstart.jpg"
            cv2.imwrite(str(out_path), sheet)
            print(f"  Saved: {out_path}")

    print(f"\nDone. Open: open {args.output_dir}")


if __name__ == '__main__':
    main()
