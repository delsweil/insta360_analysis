#!/usr/bin/env python3
"""
cluster_ball_predictor.py
--------------------------
Rebuilds ball predictor dataset with raw player positions,
then applies DBSCAN clustering to find the tightest player cluster
as a proxy for ball location.

Usage:
    python3 cluster_ball_predictor.py \
        --dataset ~/ball_dataset_v5 \
        --players models/yolo11s.pt \
        --output ball_cluster_dataset.csv \
        [--device mps]
"""

import argparse
import csv
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

OUT_W, OUT_H = 1280, 720
E2P_FOV = 100.0


def parse_filename(filename: str):
    name = Path(filename).stem
    name = re.sub(r'_jpg\.rf\.[a-f0-9]+$', '', name)
    m = re.match(
        r'(VID_\d+_\d+_\d+_\d+)_t(\d+)_yaw([+-]?\d+)_(det|neg)',
        name
    )
    if not m:
        return None
    return m.group(1), float(m.group(2)), float(m.group(3))


def pixel_to_lon(px, yaw_deg, w=OUT_W, fov_deg=E2P_FOV):
    nx = (px - w / 2) / (w / 2)
    half_fov = math.radians(fov_deg / 2)
    angle_x = math.degrees(math.atan(nx * math.tan(half_fov)))
    return angle_x + yaw_deg


def find_best_cluster(lons, heights, eps_deg=15.0, min_samples=2):
    """
    Run DBSCAN on player longitudes weighted by height (size).
    Returns (cluster_lon, cluster_size, cluster_mean_height, n_clusters)
    Uses the densest/largest cluster as ball location proxy.
    """
    if len(lons) < min_samples:
        return float(np.mean(lons)), 1, float(np.mean(heights)), 0

    X = lons.reshape(-1, 1)
    db = DBSCAN(eps=eps_deg, min_samples=min_samples).fit(X)
    labels = db.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters == 0:
        return float(np.mean(lons)), 1, float(np.mean(heights)), 0

    # Score each cluster by: size * mean_height (density * nearness)
    best_score = -1
    best_lon = float(np.mean(lons))
    best_size = 1
    best_height = float(np.mean(heights))

    for label in set(labels):
        if label == -1:
            continue
        mask = labels == label
        cluster_lons = lons[mask]
        cluster_heights = heights[mask]
        size = mask.sum()
        mean_h = float(np.mean(cluster_heights))
        # Score: number of players * average size
        score = size * mean_h
        if score > best_score:
            best_score = score
            best_lon = float(np.mean(cluster_lons))
            best_size = int(size)
            best_height = mean_h

    return best_lon, best_size, best_height, n_clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=str(Path.home() / 'ball_dataset_v5'))
    parser.add_argument('--players', default='models/yolo11s.pt')
    parser.add_argument('--output', default='ball_cluster_dataset.csv')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()

    from ultralytics import YOLO
    player_model = YOLO(args.players)

    base = Path(args.dataset)
    rows = []
    skipped = 0

    for split in ['train', 'valid', 'test']:
        img_dir = base / split / 'images'
        lbl_dir = base / split / 'labels'
        imgs = sorted(img_dir.glob('VID_*.jpg'))
        print(f"\n{split}: {len(imgs)} Insta360 images")

        for img_path in imgs:
            parsed = parse_filename(img_path.name)
            if parsed is None:
                skipped += 1
                continue
            clip, timestamp_s, yaw_deg = parsed

            # Load ball label
            lbl_path = lbl_dir / (img_path.stem + '.txt')
            if not lbl_path.exists() or lbl_path.stat().st_size == 0:
                continue
            parts = lbl_path.read_text().strip().split('\n')[0].split()
            if len(parts) < 5:
                continue
            ball_lon = pixel_to_lon(float(parts[1]) * OUT_W, yaw_deg)

            # Player detection
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            res = player_model(img, imgsz=640, conf=args.conf,
                               device=args.device, verbose=False)[0]

            lons, heights, lats = [], [], []
            for b in res.boxes:
                if res.names.get(int(b.cls[0]), '') != 'person':
                    continue
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                cx = (x1 + x2) / 2
                lon = pixel_to_lon(cx, yaw_deg)
                lons.append(lon)
                heights.append(y2 - y1)
                lats.append((y1 + y2) / 2)

            if not lons:
                continue

            lons_arr = np.array(lons)
            heights_arr = np.array(heights)
            n = len(lons)

            # Simple features
            mean_lon = float(np.mean(lons_arr))
            w = heights_arr / heights_arr.sum()
            weighted_lon = float(np.sum(w * lons_arr))

            # DBSCAN clustering with different eps values
            cluster_results = {}
            for eps in [8, 12, 15, 20, 25]:
                clon, csize, cheight, nclust = find_best_cluster(
                    lons_arr, heights_arr, eps_deg=eps)
                cluster_results[f'cluster_lon_eps{eps}'] = clon
                cluster_results[f'cluster_size_eps{eps}'] = csize
                cluster_results[f'cluster_height_eps{eps}'] = cheight
                cluster_results[f'n_clusters_eps{eps}'] = nclust

            # Height-weighted DBSCAN: duplicate players proportional to height
            # This makes larger (nearer) players count more in clustering
            height_norm = (heights_arr / heights_arr.max() * 3).astype(int) + 1
            lons_weighted = np.repeat(lons_arr, height_norm)
            heights_weighted = np.repeat(heights_arr, height_norm)
            clon_hw, csize_hw, cheight_hw, nclust_hw = find_best_cluster(
                lons_weighted, heights_weighted, eps_deg=15.0)

            row = {
                'clip': clip,
                'timestamp_s': timestamp_s,
                'yaw_deg': yaw_deg,
                'split': split,
                'n_players': n,
                'mean_lon': mean_lon,
                'weighted_lon': weighted_lon,
                'height_weighted_cluster_lon': clon_hw,
                'height_weighted_cluster_size': csize_hw,
                'ball_lon': ball_lon,
            }
            row.update(cluster_results)
            rows.append(row)

        print(f"  Rows so far: {len(rows)}")

    print(f"\nTotal rows: {len(rows)}")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

    # Evaluate each approach
    print(f"\n{'='*55}")
    print(f"{'Method':<35} {'RMSE':>8} {'<10°':>6} {'<20°':>6}")
    print(f"{'-'*55}")

    def eval_pred(pred_col):
        err = (df[pred_col] - df['ball_lon']).abs()
        rmse = float(np.sqrt(np.mean((df[pred_col] - df['ball_lon'])**2)))
        pct10 = float((err < 10).mean() * 100)
        pct20 = float((err < 20).mean() * 100)
        return rmse, pct10, pct20

    methods = ['mean_lon', 'weighted_lon', 'height_weighted_cluster_lon']
    for eps in [8, 12, 15, 20, 25]:
        methods.append(f'cluster_lon_eps{eps}')

    for method in methods:
        if method in df.columns:
            rmse, p10, p20 = eval_pred(method)
            print(f"{method:<35} {rmse:>8.2f} {p10:>5.1f}% {p20:>5.1f}%")

    print(f"{'='*55}")
    print(f"\nBaseline (predict mean ball_lon={df['ball_lon'].mean():.1f}°):")
    baseline = np.sqrt(np.mean((df['ball_lon'] - df['ball_lon'].mean())**2))
    print(f"  RMSE: {baseline:.2f}°")


if __name__ == '__main__':
    main()
