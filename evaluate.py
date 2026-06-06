#!/usr/bin/env python3
"""
evaluate.py — Run autopan inference with a named approach and save standardised results.

Usage:
    python3 evaluate.py --approach track
    python3 evaluate.py --approach reanchor_triggered
    python3 evaluate.py --approach hold_only
    python3 evaluate.py --approach reanchor_fixed --reanchor-interval 25

Results saved to: results/<approach>_<clip>.csv
Summary saved to: results/<approach>_summary.json
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import xml.etree.ElementTree as ET

# ── Clip registry ─────────────────────────────────────────────────
CLIPS = {
    '001': {
        'insv':   '/Volumes/Sickis disk/DCIM/Camera01/VID_20241028_144020_10_001.insv',
        'insprj': 'VID_20241028_144020_00_001.insv.insprj',
        'calib':  'calibration/pitch_VID_20241028_144020_10_001.insv.json',
    },
    '002': {
        'insv':   '/Volumes/Sickis disk/DCIM/Camera01/VID_20241028_150939_10_002.insv',
        'insprj': 'VID_20241028_150939_00_002.insv.insprj',
        'calib':  'calibration/pitch_VID_20241028_150939_10_002.insv.json',
    },
    '003': {
        'insv':   '/Volumes/Sickis disk/DCIM/Camera01/VID_20241028_153158_10_003.insv',
        'insprj': 'VID_20241028_153158_00_003.insv.insprj',
        'calib':  'calibration/pitch_VID_20241028_153158_10_003.insv.json',
    },
    '004': {
        'insv':   '/Volumes/Sickis disk/DCIM/Camera01/VID_20241028_160117_10_004.insv',
        'insprj': 'VID_20241028_160117_00_004.insv.insprj',
        'calib':  'calibration/pitch_VID_20241028_160117_10_004.insv.json',
    },
    '005': {
        'insv':   '/Volumes/Sickis disk/DCIM/Camera01/VID_20241028_174053_10_005.insv',
        'insprj': 'VID_20241028_174053_00_005.insv.insprj',
        'calib':  'calibration/pitch_VID_20241028_174053_10_005.insv.json',
    },
}

PLAYERS  = 'models/yolo11s.pt'
BALL     = 'models/ball_v5_yolo11s_1280_candidate.pt'
SEG_DUR  = 45
N_SEGS   = 6
SEED     = 99


def load_gt(insprj_path: str):
    tree = ET.parse(insprj_path)
    kfs  = tree.getroot().findall(".//recording/keyframes/keyframe")
    pairs = sorted((float(k.attrib['time'])/1000,
                    math.degrees(float(k.attrib['pan']))) for k in kfs)
    times = [p[0] for p in pairs]
    pans  = [p[1] for p in pairs]
    return PchipInterpolator(times, pans), min(times), max(times)


def evaluate_csv(csv_path: str, insprj_path: str, seg_dur: float):
    """Return per-segment and overall RMSE/MAE."""
    gt_interp, gt_min, gt_max = load_gt(insprj_path)
    df = pd.read_csv(csv_path)

    # Auto-detect segment starts
    ts    = df['timestamp_s'].values
    diffs = np.diff(ts)
    gaps  = np.where(diffs > seg_dur * 0.5)[0]
    starts = [float(ts[0])] + [float(ts[i+1]) for i in gaps]

    segments = []
    for t0 in starts:
        seg = df[(df['timestamp_s'] >= t0) & (df['timestamp_s'] < t0 + seg_dur)]
        valid_ts = seg['timestamp_s'].values
        valid_ts = valid_ts[(valid_ts >= gt_min) & (valid_ts <= gt_max)]
        if len(valid_ts) < 50:
            continue
        gt_vals   = gt_interp(valid_ts)
        pred_vals = seg[seg['timestamp_s'].isin(valid_ts)]['predicted_pan_deg'].values
        if len(pred_vals) != len(gt_vals):
            mask = np.isin(seg['timestamp_s'].values, valid_ts)
            pred_vals = seg['predicted_pan_deg'].values[mask]
        errs = pred_vals - gt_vals
        segments.append({
            't0': t0,
            'rmse': float(np.sqrt(np.mean(errs**2))),
            'mae':  float(np.mean(np.abs(errs))),
            'n':    len(errs),
        })

    if not segments:
        return segments, float('nan'), float('nan')

    all_errs = []
    for seg in segments:
        all_errs.extend([seg['rmse']] * seg['n'])
    overall_rmse = float(np.sqrt(np.mean(
        [(s['rmse']**2) for s in segments])))
    overall_mae  = float(np.mean([s['mae'] for s in segments]))
    return segments, overall_rmse, overall_mae


def build_cmd(approach: str, clip: dict, csv_path: str,
              mp4_path: str, extra_args: list) -> list:
    """Build autopan_infer.py command for a given approach."""
    base = [
        'python3', 'autopan_infer.py',
        '--insv',        clip['insv'],
        '--calib',       clip['calib'],
        '--players',     PLAYERS,
        '--ball',        BALL,
        '--segments',    str(N_SEGS),
        '--seg-duration', str(SEG_DUR),
        '--seed',        str(SEED),
        '--log-csv',     csv_path,
        '--output',      mp4_path,
        '--device',      'mps',
    ]
    if 'reanchor' in approach:
        base += ['--mode', 'reanchor']
    if 'sahi' in approach:
        base += ['--sahi']

    if approach == 'track':
        base += ['--mode', 'track']
    elif approach == 'reanchor_triggered':
        base += ['--mode', 'reanchor']
    elif approach.startswith('reanchor_fixed'):
        interval = approach.split('_')[-1]
        base += ['--mode', 'reanchor_fixed',
                 '--reanchor-interval', interval]
    elif approach == 'hold_only':
        base += ['--mode', 'hold_only']

    base += extra_args
    return base


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--approach', required=True,
                   help='track | reanchor_triggered | reanchor_fixed_25 | hold_only')
    p.add_argument('--clips', default='001,002,003,004,005',
                   help='Comma-separated clip IDs to evaluate')
    p.add_argument('--skip-inference', action='store_true',
                   help='Skip inference, just re-evaluate existing CSVs')
    p.add_argument('--results-dir', default='results')
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    clip_ids = args.clips.split(',')
    summary  = {'approach': args.approach, 'clips': {}}

    for clip_id in clip_ids:
        if clip_id not in CLIPS:
            print(f"Unknown clip: {clip_id}")
            continue

        clip    = CLIPS[clip_id]
        csv_path = str(results_dir / f"{args.approach}_{clip_id}.csv")
        mp4_path = str(results_dir / f"{args.approach}_{clip_id}.mp4")

        if not Path(clip['insv']).exists():
            print(f"SKIP {clip_id} — insv not found")
            continue
        if not Path(clip['calib']).exists():
            print(f"SKIP {clip_id} — calib not found")
            continue

        print(f"\n{'='*50}")
        print(f"Clip {clip_id} | approach={args.approach}")
        print(f"{'='*50}")

        if not args.skip_inference:
            cmd = build_cmd(args.approach, clip, csv_path, mp4_path, [])
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"INFERENCE FAILED for clip {clip_id}")
                continue

        if not Path(csv_path).exists():
            print(f"No CSV found: {csv_path}")
            continue

        segments, overall_rmse, overall_mae = evaluate_csv(
            csv_path, clip['insprj'], SEG_DUR)

        print(f"\nResults for clip {clip_id}:")
        print(f"  {'Segment':>10} {'RMSE':>8} {'MAE':>8} {'N':>6}")
        print(f"  {'-'*35}")
        for s in segments:
            print(f"  t={s['t0']:>6.0f}s   {s['rmse']:>6.1f}°  {s['mae']:>6.1f}°  {s['n']:>6}")
        print(f"  {'OVERALL':>10} {overall_rmse:>6.1f}°  {overall_mae:>6.1f}°")

        summary['clips'][clip_id] = {
            'segments': segments,
            'overall_rmse': overall_rmse,
            'overall_mae':  overall_mae,
        }

    # Save summary
    summary_path = results_dir / f"{args.approach}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    # Print overall across all clips
    all_rmse = [v['overall_rmse'] for v in summary['clips'].values()
                if not math.isnan(v['overall_rmse'])]
    if all_rmse:
        print(f"\nOverall mean RMSE ({args.approach}): {np.mean(all_rmse):.1f}°")


if __name__ == '__main__':
    main()
