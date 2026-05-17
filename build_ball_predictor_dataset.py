#!/usr/bin/env python3
"""
build_ball_predictor_dataset.py
--------------------------------
Builds a dataset for predicting ball position from player features.

For each annotated Insta360 image (with ball label):
1. Parses filename to extract clip, timestamp, yaw
2. Runs yolo11s.pt to get player bboxes (position + size)
3. Loads ground truth ball position from label file
4. Converts positions to equirectangular angular coordinates
5. Computes derived features (clustering, size gradient, etc.)
6. Saves one row per image to CSV

Usage:
    python3 build_ball_predictor_dataset.py \
        --dataset ~/ball_dataset_v5 \
        --players models/yolo11s.pt \
        --output ball_predictor_dataset.csv \
        [--device mps]
"""

import argparse
import csv
import math
import re
from pathlib import Path

import cv2
import numpy as np

OUT_W, OUT_H = 1280, 720
E2P_FOV = 100.0  # degrees — matches autopan_infer.py default


def parse_filename(filename: str):
    """
    Parse VID_20241028_143616_10_008_t314_yaw+23_det.jpg
    Returns (clip_name, timestamp_s, yaw_deg) or None
    """
    # Remove Roboflow hash suffix: _jpg.rf.abc123.jpg -> extract base
    name = Path(filename).stem
    # Remove rf hash: name_jpg.rf.hash -> name
    name = re.sub(r'_jpg\.rf\.[a-f0-9]+$', '', name)

    # Match pattern
    m = re.match(
        r'(VID_\d+_\d+_\d+_\d+)_t(\d+)_yaw([+-]?\d+)_(det|neg)',
        name
    )
    if not m:
        return None
    clip = m.group(1)
    timestamp_s = float(m.group(2))
    yaw_deg = float(m.group(3))
    return clip, timestamp_s, yaw_deg


def pixel_to_equirect_angle(px: float, py: float,
                             yaw_deg: float, fov_deg: float,
                             w: int = OUT_W, h: int = OUT_H):
    """
    Convert pixel (px, py) in perspective frame to equirectangular
    (longitude_deg, latitude_deg), accounting for camera yaw.

    Returns (lon_deg, lat_deg) where lon=0 is pitch centre.
    """
    # Normalise to [-1, 1]
    nx = (px - w / 2) / (w / 2)
    ny = (py - h / 2) / (h / 2)

    # Angular offset from frame centre
    half_fov = math.radians(fov_deg / 2)
    angle_x = math.atan(nx * math.tan(half_fov))
    angle_y = math.atan(ny * math.tan(half_fov * h / w))

    # Add camera yaw to get absolute longitude
    lon_deg = math.degrees(angle_x) + yaw_deg
    lat_deg = math.degrees(angle_y)

    return lon_deg, lat_deg


def bbox_angular_size(x1: float, y1: float, x2: float, y2: float,
                      fov_deg: float, w: int = OUT_W, h: int = OUT_H):
    """Approximate angular width of a bounding box in degrees."""
    half_fov = math.radians(fov_deg / 2)
    nx1 = (x1 - w / 2) / (w / 2)
    nx2 = (x2 - w / 2) / (w / 2)
    a1 = math.degrees(math.atan(nx1 * math.tan(half_fov)))
    a2 = math.degrees(math.atan(nx2 * math.tan(half_fov)))
    return abs(a2 - a1)


def compute_features(players: list, ball_lon: float, ball_lat: float,
                     yaw_deg: float):
    """
    Compute features from player detections.
    players: list of (cx, cy, x1, y1, x2, y2, bbox_h_px) in pixel coords
    Returns dict of features.
    """
    if not players:
        return None

    # Convert players to equirectangular coordinates
    player_lons = []
    player_lats = []
    player_heights = []  # bbox height in pixels = depth proxy

    for cx, cy, x1, y1, x2, y2, bbox_h in players:
        lon, lat = pixel_to_equirect_angle(cx, cy, yaw_deg, E2P_FOV)
        player_lons.append(lon)
        player_lats.append(lat)
        player_heights.append(bbox_h)

    lons = np.array(player_lons)
    lats = np.array(player_lats)
    heights = np.array(player_heights)
    n = len(players)

    # Basic stats
    mean_lon = float(np.mean(lons))
    std_lon = float(np.std(lons))
    mean_lat = float(np.mean(lats))
    std_lat = float(np.std(lats))

    # Size-weighted centroid (larger = nearer = more important)
    w = heights / heights.sum()
    weighted_lon = float(np.sum(w * lons))
    weighted_lat = float(np.sum(w * lats))

    # Size stats (depth proxy)
    mean_height = float(np.mean(heights))
    max_height = float(np.max(heights))
    std_height = float(np.std(heights))

    # Size gradient: do large players cluster on one side?
    left_mask = lons < mean_lon
    right_mask = ~left_mask
    left_mean_h = float(np.mean(heights[left_mask])) if left_mask.any() else 0
    right_mean_h = float(np.mean(heights[right_mask])) if right_mask.any() else 0
    size_gradient = left_mean_h - right_mean_h  # +ve = near side is left

    # Largest player (nearest to camera)
    largest_idx = int(np.argmax(heights))
    largest_lon = float(lons[largest_idx])
    largest_lat = float(lats[largest_idx])

    # Nearest player to ball (in lon space)
    dists_to_ball = np.abs(lons - ball_lon)
    nearest_idx = int(np.argmin(dists_to_ball))
    nearest_lon = float(lons[nearest_idx])
    nearest_dist = float(dists_to_ball[nearest_idx])
    nearest_height = float(heights[nearest_idx])

    # Player density in thirds
    lon_min, lon_max = float(lons.min()), float(lons.max())
    lon_range = lon_max - lon_min if lon_max > lon_min else 1.0
    left_third = (lons < lon_min + lon_range / 3).sum()
    mid_third = ((lons >= lon_min + lon_range / 3) &
                 (lons < lon_min + 2 * lon_range / 3)).sum()
    right_third = (lons >= lon_min + 2 * lon_range / 3).sum()

    # Cluster tightness: std of top-5 players by size
    top5_idx = np.argsort(heights)[-5:]
    top5_lon_std = float(np.std(lons[top5_idx])) if len(top5_idx) > 1 else 0

    # Players left fraction
    players_left_frac = float((lons < 0).sum() / n)  # left of pitch centre

    return {
        'n_players': n,
        'mean_lon': mean_lon,
        'std_lon': std_lon,
        'mean_lat': mean_lat,
        'std_lat': std_lat,
        'weighted_lon': weighted_lon,
        'weighted_lat': weighted_lat,
        'mean_height_px': mean_height,
        'max_height_px': max_height,
        'std_height_px': std_height,
        'size_gradient': size_gradient,
        'largest_player_lon': largest_lon,
        'largest_player_lat': largest_lat,
        'nearest_player_lon': nearest_lon,
        'nearest_player_dist': nearest_dist,
        'nearest_player_height': nearest_height,
        'players_left_frac': players_left_frac,
        'left_third_count': int(left_third),
        'mid_third_count': int(mid_third),
        'right_third_count': int(right_third),
        'top5_cluster_std': top5_lon_std,
        'yaw_deg': yaw_deg,
        # Target
        'ball_lon': ball_lon,
        'ball_lat': ball_lat,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default=str(Path.home() / 'ball_dataset_v5'),
                        help='Path to YOLOv8 dataset directory')
    parser.add_argument('--players', default='models/yolo11s.pt',
                        help='Player detector model path')
    parser.add_argument('--output', default='ball_predictor_dataset.csv',
                        help='Output CSV path')
    parser.add_argument('--device', default='mps',
                        help='Device: mps, cpu, cuda')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Player detection confidence threshold')
    args = parser.parse_args()

    from ultralytics import YOLO
    player_model = YOLO(args.players)

    base = Path(args.dataset)
    splits = ['train', 'valid', 'test']

    rows = []
    skipped_no_parse = 0
    skipped_no_ball = 0
    skipped_no_players = 0

    for split in splits:
        img_dir = base / split / 'images'
        lbl_dir = base / split / 'labels'

        # Only Insta360 images
        imgs = sorted(img_dir.glob('VID_*.jpg'))
        print(f"\n{split}: {len(imgs)} Insta360 images")

        for img_path in imgs:
            # Parse filename
            parsed = parse_filename(img_path.name)
            if parsed is None:
                skipped_no_parse += 1
                continue
            clip, timestamp_s, yaw_deg = parsed

            # Load ball label
            lbl_path = lbl_dir / (img_path.stem + '.txt')
            if not lbl_path.exists() or lbl_path.stat().st_size == 0:
                skipped_no_ball += 1
                continue

            # Parse ball position (YOLO format: class cx cy w h normalised)
            ball_line = lbl_path.read_text().strip().split('\n')[0]
            parts = ball_line.split()
            if len(parts) < 5:
                skipped_no_ball += 1
                continue
            ball_cx_norm = float(parts[1])
            ball_cy_norm = float(parts[2])
            ball_cx_px = ball_cx_norm * OUT_W
            ball_cy_px = ball_cy_norm * OUT_H
            ball_lon, ball_lat = pixel_to_equirect_angle(
                ball_cx_px, ball_cy_px, yaw_deg, E2P_FOV)

            # Run player detection
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            res = player_model(img, imgsz=640, conf=args.conf,
                               device=args.device, verbose=False)[0]

            players = []
            for b in res.boxes:
                if res.names.get(int(b.cls[0]), '') != 'person':
                    continue
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                bbox_h = y2 - y1
                players.append((cx, cy, x1, y1, x2, y2, bbox_h))

            if not players:
                skipped_no_players += 1
                continue

            # Compute features
            feats = compute_features(players, ball_lon, ball_lat, yaw_deg)
            if feats is None:
                continue

            feats['clip'] = clip
            feats['timestamp_s'] = timestamp_s
            feats['split'] = split
            rows.append(feats)

        print(f"  Rows so far: {len(rows)}")

    print(f"\nTotal rows: {len(rows)}")
    print(f"Skipped — no parse: {skipped_no_parse}")
    print(f"Skipped — no ball:  {skipped_no_ball}")
    print(f"Skipped — no players: {skipped_no_players}")

    if not rows:
        print("No rows! Check dataset path and filenames.")
        return

    # Write CSV
    fieldnames = list(rows[0].keys())
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {args.output}")

    # Quick stats
    import pandas as pd
    df = pd.read_csv(args.output)
    print(f"\nDataset summary:")
    print(f"  Rows: {len(df)}")
    print(f"  Clips: {df['clip'].nunique()}")
    print(f"  Ball lon range: {df['ball_lon'].min():.1f}° to {df['ball_lon'].max():.1f}°")
    print(f"  Mean players per frame: {df['n_players'].mean():.1f}")
    print(f"\nFeature correlations with ball_lon:")
    corr_cols = ['mean_lon', 'weighted_lon', 'largest_player_lon',
                 'nearest_player_lon', 'size_gradient', 'players_left_frac']
    for col in corr_cols:
        r = df[col].corr(df['ball_lon'])
        print(f"  {col:<30} r={r:.3f}")


if __name__ == '__main__':
    main()
