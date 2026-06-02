#!/usr/bin/env python3
"""
evaluate.py — Run autopan inference with a named approach and save standardised results.

Usage:
    python3 evaluate.py --approach track
    python3 evaluate.py --approach track_v2
    python3 evaluate.py --approach reanchor_triggered
    python3 evaluate.py --approach reanchor_ball_v5
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

ROOT = Path(__file__).resolve().parent

# ── Clip registry ─────────────────────────────────────────────────
DEFAULT_CLIPS = {
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
BALL     = 'models/ball_v4.pt'
BALL_V2  = 'models/ball_v5.pt'
SEG_DUR  = 45
N_SEGS   = 6
SEED     = 99


def _best_device() -> str:
    """Auto-detect the best available compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    return 'cpu'


def _repo_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    candidate = ROOT / path
    if candidate.exists() or value.startswith('models/'):
        return str(candidate)
    return value


def preferred_track_v2_ball_model() -> str:
    """Use the same preserved-candidate policy as production inference."""
    try:
        from autopan_infer import preferred_ball_model

        return preferred_ball_model()
    except Exception:
        return BALL_V2 if (ROOT / BALL_V2).exists() else BALL


def load_gt(insprj_path: str):
    tree = ET.parse(insprj_path)
    kfs  = tree.getroot().findall(".//recording/keyframes/keyframe")
    pairs = sorted((float(k.attrib['time'])/1000,
                    math.degrees(float(k.attrib['pan']))) for k in kfs)
    times = [p[0] for p in pairs]
    pans  = unwrap_longitudes_for_interp([p[1] for p in pairs])
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
        errs = ((pred_vals - gt_vals + 180.0) % 360.0) - 180.0
        segments.append({
            't0': t0,
            'rmse': float(np.sqrt(np.mean(errs**2))),
            'mae':  float(np.mean(np.abs(errs))),
            'n':    len(errs),
        })

    if not segments:
        return segments, float('nan'), float('nan')

    total_n = max(1, sum(int(s['n']) for s in segments))
    overall_rmse = float(np.sqrt(sum((float(s['rmse']) ** 2) * int(s['n']) for s in segments) / total_n))
    overall_mae  = float(sum(float(s['mae']) * int(s['n']) for s in segments) / total_n)
    return segments, overall_rmse, overall_mae


def load_ball_groundtruth(path: str) -> pd.DataFrame:
    """Load ball GT from CSV or JSON.

    Supported columns/keys:
      timestamp_s or time_s; lon/lat; x_m/y_m or pitch_x_m/pitch_y_m.
    """
    p = Path(path)
    if p.suffix.lower() == '.csv':
        df = pd.read_csv(p)
    elif p.suffix.lower() == '.jsonl':
        rows = [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
        df = pd.DataFrame(rows)
    else:
        raw = json.loads(p.read_text(encoding='utf-8'))
        if isinstance(raw, dict):
            raw = raw.get('frames') or raw.get('annotations') or raw.get('points') or []
        df = pd.DataFrame(raw)
    if 'time_s' in df.columns and 'timestamp_s' not in df.columns:
        df = df.rename(columns={'time_s': 'timestamp_s'})
    if 'pitch_x_m' in df.columns and 'x_m' not in df.columns:
        df = df.rename(columns={'pitch_x_m': 'x_m'})
    if 'pitch_y_m' in df.columns and 'y_m' not in df.columns:
        df = df.rename(columns={'pitch_y_m': 'y_m'})
    if 'timestamp_s' not in df.columns:
        raise ValueError(f"Ball GT needs timestamp_s/time_s: {path}")
    return df.sort_values('timestamp_s').reset_index(drop=True)


def angular_error_deg(pred_lon, pred_lat, gt_lon, gt_lat) -> np.ndarray:
    dlon = ((pred_lon - gt_lon + 180.0) % 360.0) - 180.0
    return np.sqrt(dlon ** 2 + (pred_lat - gt_lat) ** 2)


def unwrap_longitudes_for_interp(lon_values) -> np.ndarray:
    """Unwrap lon degrees before interpolation so tracks crossing +/-180 stay local."""
    vals = np.asarray(lon_values, dtype=np.float64)
    if vals.size == 0:
        return vals
    return np.rad2deg(np.unwrap(np.deg2rad(vals)))


def evaluate_ball_metrics(
    csv_path: str,
    fps: float = 29.97,
    ball_gt_path: str | None = None,
    calib_path: str | None = None,
) -> dict:
    """Compute ball metrics from autopan_infer per-frame logs."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return {
            'ball_detection_rate': float('nan'),
            'ball_mode_rate': float('nan'),
            'longest_ball_track_s': 0.0,
            'median_ball_track_s': 0.0,
            'n_ball_track_runs': 0,
        }
    ball_conf = pd.to_numeric(df.get('ball_conf', pd.Series([], dtype=float)), errors='coerce')
    has_det = ball_conf.notna()
    ball_modes = {
        'ball_highconf', 'ball', 'kalman', 'kalman_blend',
        'equirect_ball', 'equirect_kalman',
        'game_state', 'game_state_velocity', 'game_state_formation',
    }
    is_ball_mode = df['mode'].isin(ball_modes) if 'mode' in df else has_det
    runs = []
    cur = 0
    for val in is_ball_mode.tolist():
        if val:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    run_seconds = [r / fps for r in runs]
    metrics = {
        'ball_detection_rate': float(has_det.mean()),
        'ball_mode_rate': float(is_ball_mode.mean()),
        'longest_ball_track_s': float(max(run_seconds) if run_seconds else 0.0),
        'median_ball_track_s': float(np.median(run_seconds) if run_seconds else 0.0),
        'n_ball_track_runs': int(len(runs)),
    }
    if not ball_gt_path:
        return metrics

    gt = load_ball_groundtruth(ball_gt_path)
    pred = df.copy()
    pred_lon_col = 'tracker_lon' if 'tracker_lon' in pred.columns else 'ball_lon'
    pred_lat_col = 'tracker_lat' if 'tracker_lat' in pred.columns else 'ball_lat'
    if pred_lon_col not in pred.columns or pred_lat_col not in pred.columns:
        metrics['ball_gt_error'] = 'pan CSV has no ball_lon/lat or tracker_lon/lat columns'
        return metrics

    pred[pred_lon_col] = pd.to_numeric(pred[pred_lon_col], errors='coerce')
    pred[pred_lat_col] = pd.to_numeric(pred[pred_lat_col], errors='coerce')
    pred = pred.dropna(subset=[pred_lon_col, pred_lat_col])
    if pred.empty:
        metrics.update({
            'ball_gt_recall_deg3': 0.0,
            'ball_gt_rmse_deg': float('nan'),
            'ball_gt_n': int(len(gt)),
        })
        return metrics

    pred_lons_unwrapped = unwrap_longitudes_for_interp(pred[pred_lon_col].values)
    interp_lon = np.interp(gt['timestamp_s'].values, pred['timestamp_s'].values, pred_lons_unwrapped)
    interp_lat = np.interp(gt['timestamp_s'].values, pred['timestamp_s'].values, pred[pred_lat_col].values)

    if {'lon', 'lat'}.issubset(gt.columns):
        errs = angular_error_deg(interp_lon, interp_lat, gt['lon'].values, gt['lat'].values)
        metrics.update({
            'ball_gt_recall_deg3': float(np.mean(errs <= 3.0)),
            'ball_gt_rmse_deg': float(np.sqrt(np.mean(errs ** 2))),
            'ball_gt_mae_deg': float(np.mean(np.abs(errs))),
            'ball_gt_n': int(len(errs)),
        })

    if {'x_m', 'y_m'}.issubset(gt.columns) and calib_path:
        try:
            from pitch_model import PitchHomographyModel
            model = PitchHomographyModel.from_calibration(calib_path)
            pred_xy = np.array([
                [model.lon_lat_to_pitch(lon, lat).x_m, model.lon_lat_to_pitch(lon, lat).y_m]
                for lon, lat in zip(interp_lon, interp_lat)
            ], dtype=np.float64)
            gt_xy = gt[['x_m', 'y_m']].to_numpy(dtype=np.float64)
            dists = np.linalg.norm(pred_xy - gt_xy, axis=1)
            metrics.update({
                'ball_gt_recall_m3': float(np.mean(dists <= 3.0)),
                'ball_gt_rmse_m': float(np.sqrt(np.mean(dists ** 2))),
                'ball_gt_mae_m': float(np.mean(dists)),
            })

            speeds = np.linalg.norm(np.diff(gt_xy, axis=0), axis=1) / np.maximum(1e-6, np.diff(gt['timestamp_s'].values))
            pass_idxs = np.where(speeds * 3.6 >= 45.0)[0]
            latencies = []
            for idx in pass_idxs:
                for j in range(idx + 1, len(dists)):
                    if dists[j] <= 5.0:
                        latencies.append((gt['timestamp_s'].iloc[j] - gt['timestamp_s'].iloc[idx]) * fps)
                        break
            if latencies:
                metrics['switching_latency_frames_median'] = float(np.median(latencies))
        except Exception as exc:
            metrics['ball_gt_pitch_error'] = str(exc)

    return metrics


def build_cmd(approach: str, clip: dict, csv_path: str,
              mp4_path: str, extra_args: list, args) -> list:
    """Build autopan_infer.py command for a given approach."""
    is_track_v2 = approach == 'track_v2'
    is_reanchor_ball_v5 = approach == 'reanchor_ball_v5'
    ball_model = args.ball
    if (is_track_v2 or is_reanchor_ball_v5) and args.ball == BALL:
        ball_model = preferred_track_v2_ball_model()
    scan_every = args.scan_every
    if (is_track_v2 or is_reanchor_ball_v5) and scan_every <= 0:
        scan_every = 15
    ball_sahi = args.ball_sahi or is_track_v2
    field_opt = args.field_opt or is_track_v2

    base = [
        sys.executable, str(ROOT / 'autopan_infer.py'),
        '--insv',        clip['insv'],
        '--calib',       clip['calib'],
        '--players',     _repo_path(args.players),
        '--ball',        ball_model,
        '--segments',    str(args.segments),
        '--seg-duration', str(args.seg_duration),
        '--seed',        str(args.seed),
        '--log-csv',     csv_path,
        '--output',      mp4_path,
        '--device',      args.device or _best_device(),
        '--player-detect-every', str(args.player_detect_every),
        '--ball-detect-every', str(args.ball_detect_every),
        '--scan-every', str(scan_every),
    ]
    if ball_sahi:
        base += ['--ball-sahi', '--ball-sahi-every', str(args.ball_sahi_every)]
    if field_opt:
        base += ['--field-opt']

    if approach in ('track', 'track_v2'):
        base += ['--mode', 'track']
    elif approach in ('reanchor_triggered', 'reanchor_ball_v5'):
        base += ['--mode', 'reanchor']
    elif approach.startswith('reanchor_fixed'):
        interval = approach.split('_')[-1]
        if not interval.replace('.', '', 1).isdigit():
            interval = '25'
        base += ['--mode', 'reanchor_fixed',
                 '--reanchor-interval', interval]
    elif approach == 'hold_only':
        base += ['--mode', 'hold_only']
    else:
        raise ValueError(f"Unknown approach: {approach}")

    base += extra_args
    return base


def load_clips(path: str | None) -> dict:
    if not path:
        return {clip_id: normalize_clip(clip_id, clip) for clip_id, clip in DEFAULT_CLIPS.items()}
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return {clip_id: normalize_clip(str(clip_id), clip) for clip_id, clip in raw.items()}
    clips = {}
    for i, item in enumerate(raw, start=1):
        clip_id = str(item.get('id') or item.get('clip') or f'{i:03d}')
        clips[clip_id] = normalize_clip(clip_id, item)
    return clips


def normalize_clip(clip_id: str, clip: dict) -> dict:
    """Normalize historical clip registry variants.

    Older project files use `project` for the Insta360 Studio .insprj path and
    often contain absolute macOS calibration paths. Keep those files usable
    from this checkout when matching local basenames exist.
    """
    c = dict(clip)
    if 'insprj' not in c and 'project' in c:
        c['insprj'] = c['project']
    for key, local_dir in (('calib', ROOT / 'calibration'), ('insprj', ROOT)):
        value = c.get(key)
        if not value:
            continue
        p = Path(value)
        if p.exists():
            c[key] = str(p)
            continue
        candidate = local_dir / p.name
        if candidate.exists():
            c[key] = str(candidate)
    if 'insv' not in c:
        raise ValueError(f"Clip {clip_id} is missing insv")
    if 'insprj' not in c:
        raise ValueError(f"Clip {clip_id} is missing insprj/project")
    if 'calib' not in c:
        raise ValueError(f"Clip {clip_id} is missing calib")
    return c


def remap_clip_roots(clips: dict, clip_root: str | None) -> dict:
    if not clip_root:
        return clips
    root = Path(clip_root)
    out = {}
    for clip_id, clip in clips.items():
        c = dict(clip)
        c['insv'] = str(root / Path(c['insv']).name)
        out[clip_id] = c
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--approach', required=True,
                   help='track | track_v2 | reanchor_triggered | reanchor_ball_v5 | reanchor_fixed_25 | hold_only')
    p.add_argument('--clips', default='001,002,003,004,005',
                   help='Comma-separated clip IDs to evaluate')
    p.add_argument('--skip-inference', action='store_true',
                   help='Skip inference, just re-evaluate existing CSVs')
    p.add_argument('--clips-file', type=str, default=None,
                   help='JSON file with clip definitions (overrides built-in registry). '
                        'Format: {"001": {"insv": "...", "insprj": "...", "calib": "..."}}')
    p.add_argument('--clip-root', type=str, default=None,
                   help='Optional folder containing INSV files; remaps each clip by basename')
    p.add_argument('--ball-metrics', action='store_true',
                   help='Include ball detection/tracking proxy metrics from pan CSV')
    p.add_argument('--ball-groundtruth-dir', default=None,
                   help='Optional folder with per-clip ball GT files named <clip_id>.csv/json')
    p.add_argument('--players', default=PLAYERS)
    p.add_argument('--ball', default=BALL)
    p.add_argument('--device', default=None)
    p.add_argument('--segments', type=int, default=N_SEGS)
    p.add_argument('--seg-duration', type=float, default=SEG_DUR)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--player-detect-every', type=int, default=3)
    p.add_argument('--ball-detect-every', type=int, default=2)
    p.add_argument('--ball-sahi', action='store_true')
    p.add_argument('--ball-sahi-every', type=int, default=6)
    p.add_argument('--scan-every', type=int, default=0)
    p.add_argument('--field-opt', action='store_true')
    p.add_argument('--results-dir', default='results')
    args = p.parse_args()

    CLIPS = remap_clip_roots(load_clips(args.clips_file), args.clip_root)
    if args.clips_file:
        print(f"Loaded {len(CLIPS)} clips from {args.clips_file}")
    if args.approach == 'track_v2':
        if args.ball == BALL:
            print(f"track_v2 preset: using {preferred_track_v2_ball_model()}")
        else:
            print(f"track_v2 preset: using override {args.ball}")
        print("track_v2 preset: SAHI on, scan_every=15 unless overridden, field optimizer on")
    elif args.approach == 'reanchor_ball_v5':
        if args.ball == BALL:
            print(f"reanchor_ball_v5 preset: using {preferred_track_v2_ball_model()}")
        else:
            print(f"reanchor_ball_v5 preset: using override {args.ball}")
        print("reanchor_ball_v5 preset: reanchor mode, scan_every=15 unless overridden, no continuous Kalman target")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    clip_ids = args.clips.split(',')
    summary  = {'approach': args.approach, 'clips': {}}

    for clip_id in clip_ids:
        if clip_id not in CLIPS:
            print(f"Unknown clip: {clip_id}")
            continue

        clip    = CLIPS[clip_id]
        csv_path = str(results_dir / f"{args.approach}_{clip_id}.csv")
        mp4_path = str(results_dir / f"{args.approach}_{clip_id}.mp4")

        if not args.skip_inference and not Path(clip['insv']).exists():
            print(f"SKIP {clip_id} — insv not found")
            continue
        if not Path(clip['calib']).exists():
            print(f"SKIP {clip_id} — calib not found")
            continue
        if not Path(clip['insprj']).exists():
            print(f"SKIP {clip_id} — insprj not found")
            continue

        print(f"\n{'='*50}")
        print(f"Clip {clip_id} | approach={args.approach}")
        print(f"{'='*50}")

        if not args.skip_inference:
            cmd = build_cmd(args.approach, clip, csv_path, mp4_path, [], args)
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"INFERENCE FAILED for clip {clip_id}")
                continue

        if not Path(csv_path).exists():
            print(f"No CSV found: {csv_path}")
            continue

        segments, overall_rmse, overall_mae = evaluate_csv(
            csv_path, clip['insprj'], args.seg_duration)

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
        if args.ball_metrics:
            gt_path = clip.get('ball_gt')
            if not gt_path and args.ball_groundtruth_dir:
                root = Path(args.ball_groundtruth_dir)
                for suffix in ('.csv', '.json', '.jsonl'):
                    candidate = root / f"{clip_id}{suffix}"
                    if candidate.exists():
                        gt_path = str(candidate)
                        break
            ball_metrics = evaluate_ball_metrics(csv_path, ball_gt_path=gt_path, calib_path=clip.get('calib'))
            summary['clips'][clip_id]['ball_metrics'] = ball_metrics
            print("  Ball metrics:")
            print(f"    detection_rate={ball_metrics['ball_detection_rate']:.1%}")
            print(f"    ball_mode_rate={ball_metrics['ball_mode_rate']:.1%}")
            print(f"    longest_track={ball_metrics['longest_ball_track_s']:.1f}s")

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
