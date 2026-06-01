#!/usr/bin/env python3
"""
extract_frames.py
-----------------
Extract candidate frames from .insv game footage for ball detector training.
Runs the current ball detector on each frame, saves both detections and
random negatives, then uploads to Roboflow for labeling.

Usage:
    python3 extract_frames.py \
        --output-dir ~/ball_training_frames \
        [--frames-per-clip 200] \
        [--skip-start 240] \
        [--upload] \
        [--dry-run]

Requires:
    ROBOFLOW_API_KEY environment variable set
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

# ── Clip list ────────────────────────────────────────────────────────────────

CLIPS = [
    "/Users/davidelsweiler/footage/20241028/VID_20241028_140657_10_007.insv",
    "/Users/davidelsweiler/footage/20241028/VID_20241028_143616_10_008.insv",
    # "/Users/davidelsweiler/footage/20241028/VID_20241028_150534_10_009.insv",  # wrong lens
    "/Volumes/Sickis disk/DCIM/Camera01/VID_20241028_144020_10_001.insv",
    "/Volumes/Sickis disk/DCIM/Camera01/VID_20241028_150939_10_002.insv",
    "/Volumes/Sickis disk/DCIM/Camera01/VID_20241028_153158_10_003.insv",
    "/Volumes/Sickis disk/DCIM/Camera01/VID_20241028_160117_10_004.insv",
    "/Volumes/Sickis disk/DCIM/Camera01/VID_20241028_174053_10_005.insv",
    "/Volumes/Sickis disk/DCIM/Camera01/VID_20241028_180829_10_006.insv",
    "/Volumes/Sickis disk/DCIM/Camera01/VID_20241028_184414_10_007.insv",
    "/Volumes/Untitled/DCIM/Camera01/VID_20241028_124047_10_010.insv",
    "/Volumes/Untitled/DCIM/Camera01/VID_20241028_131005_10_011.insv",
    "/Volumes/Untitled/DCIM/Camera01/VID_20241028_132551_10_012.insv",
    "/Volumes/Untitled/DCIM/Camera01/VID_20241028_133508_10_013.insv",
    "/Volumes/Untitled/DCIM/Camera01/VID_20241028_140425_10_014.insv",
    # "/Volumes/Untitled/DCIM/Camera01/VID_20250628_151959_10_001.insv",  # training session not game
    # "/Volumes/Untitled/DCIM/Camera01/VID_20250628_154917_10_002.insv",  # training session not game
    "/Volumes/Untitled/DCIM/Camera01/VID_20250629_112549_10_003.insv",
    "/Volumes/Untitled/DCIM/Camera01/VID_20250629_120951_10_004.insv",
    "/Volumes/Untitled/DCIM/Camera01/VID_20250629_121009_10_005.insv",
    "/Volumes/Untitled/DCIM/Camera01/VID_20250629_123709_10_006.insv",
    "/Volumes/Untitled/DCIM/Camera01/VID_20250629_130627_10_007.insv",
    "/Volumes/Untitled/DCIM/Camera01/VID_20250629_131742_10_008.insv",
    "/Volumes/Untitled/DCIM/Camera01/VID_20250629_134123_10_009.insv",
]

# ── FFmpeg helpers ────────────────────────────────────────────────────────────

OUT_W, OUT_H = 1280, 720
E2P_FOV = 110.0   # default FOV for extraction (no calibration needed here)
E2P_TILT = -20.0  # default tilt


def get_duration(path: str) -> float:
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', path],
        capture_output=True, text=True)
    txt = r.stdout.strip()
    return float(txt) if txt else 1800.0


def extract_frame(insv_path: str, t: float) -> np.ndarray | None:
    """Extract a single equirectangular frame at time t, return as BGR numpy array."""
    cmd = [
        'ffmpeg', '-ss', str(t), '-i', insv_path,
        '-vf', 'rotate=PI/2*3,v360=fisheye:equirect:ih_fov=200:iv_fov=200,scale=2880:1440',
        '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-loglevel', 'error', 'pipe:1'
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or len(proc.stdout) == 0:
        return None
    arr = np.frombuffer(proc.stdout, dtype=np.uint8)
    if arr.size != 2880 * 1440 * 3:
        return None
    return arr.reshape((1440, 2880, 3))


def equirect_to_persp(eq_bgr: np.ndarray, yaw: float = 0.0,
                      pitch: float = E2P_TILT,
                      fov: float = E2P_FOV) -> np.ndarray:
    """Simple equirect crop to perspective view at given yaw (degrees)."""
    try:
        from py360convert import e2p
        rgb = cv2.cvtColor(eq_bgr, cv2.COLOR_BGR2RGB)
        persp = e2p(rgb, fov_deg=fov, u_deg=-yaw, v_deg=pitch,
                    out_hw=(OUT_H, OUT_W), mode='bilinear')
        return cv2.cvtColor(persp, cv2.COLOR_RGB2BGR)
    except Exception:
        # Fallback: simple equirect crop
        h, w = eq_bgr.shape[:2]
        cx = int(w * (0.5 + yaw / 360.0)) % w
        half_w = OUT_W // 2
        x1 = (cx - half_w) % w
        if x1 + OUT_W <= w:
            crop = eq_bgr[h//4:h//4+OUT_H, x1:x1+OUT_W]
        else:
            left = eq_bgr[h//4:h//4+OUT_H, x1:]
            right = eq_bgr[h//4:h//4+OUT_H, :OUT_W - left.shape[1]]
            crop = np.concatenate([left, right], axis=1)
        return cv2.resize(crop, (OUT_W, OUT_H))


# ── Ground-truth sampling ─────────────────────────────────────────────────────

def load_studio_pan_curve(insprj_path: str):
    tree = ET.parse(insprj_path)
    pairs = []
    for kf in tree.getroot().iter('keyframe'):
        if 'time' not in kf.attrib or 'pan' not in kf.attrib:
            continue
        pairs.append((float(kf.attrib['time']) / 1000.0,
                      -float(np.degrees(float(kf.attrib['pan'])))))
    pairs = sorted(set(pairs))
    if len(pairs) < 2:
        return None, pairs
    times = np.array([p[0] for p in pairs], dtype=np.float64)
    pans = np.array([p[1] for p in pairs], dtype=np.float64)
    return PchipInterpolator(times, pans, extrapolate=False), pairs


def resolve_project_path(entry: dict) -> str | None:
    for key in ('insprj', 'project', 'studio_project'):
        value = entry.get(key)
        if value and Path(value).exists():
            return value
    return None


def sample_random_times(dur: float, frames_per_clip: int,
                        skip_start: float, skip_end: float) -> list[tuple[float, float | None, str]]:
    t_start = skip_start
    t_end = dur - skip_end
    usable = t_end - t_start
    if usable <= 0:
        return []
    timestamps = []
    interval = usable / max(1, frames_per_clip)
    for i in range(frames_per_clip):
        t = t_start + i * interval + random.uniform(0, interval * 0.8)
        timestamps.append((min(t, t_end - 1), None, 'random'))
    return timestamps


def sample_insprj_times(
    insprj_path: str,
    frames_per_clip: int,
    dur: float,
    window_s: float,
    samples_per_keyframe: int,
    skip_start: float,
    skip_end: float,
) -> list[tuple[float, float | None, str]]:
    curve, pairs = load_studio_pan_curve(insprj_path)
    if not pairs:
        return []
    t_min = max(skip_start, 0.0)
    t_max = max(t_min, dur - skip_end)
    candidates: list[tuple[float, float | None, str]] = []
    for t0, pan in pairs:
        if t0 < t_min or t0 > t_max:
            continue
        offsets = [0.0]
        if samples_per_keyframe > 1:
            for _ in range(samples_per_keyframe - 1):
                offsets.append(random.uniform(-window_s, window_s))
        for off in offsets:
            t = min(max(t0 + off, t_min), max(t_min, t_max - 1))
            yaw = float(curve(t)) if curve is not None and not np.isnan(curve(t)) else pan
            candidates.append((t, yaw, 'insprj'))
    if len(candidates) > frames_per_clip:
        candidates = random.sample(candidates, frames_per_clip)
    candidates.sort(key=lambda x: x[0])
    return candidates


# ── Detection ─────────────────────────────────────────────────────────────────

def load_detector(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)


def detect_ball_raw(frame_bgr: np.ndarray, model, conf_thresh: float = 0.25,
                    device: str = 'cpu'):
    """Returns list of (cx, cy, conf, x1, y1, x2, y2) for all detections."""
    res = model(frame_bgr, imgsz=640, conf=conf_thresh,
                device=device, verbose=False)[0]
    detections = []
    for b in res.boxes:
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        conf = float(b.conf[0])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        detections.append((cx, cy, conf, x1, y1, x2, y2))
    return detections


# ── Frame saving ─────────────────────────────────────────────────────────────

def save_frame(frame_bgr: np.ndarray, out_path: Path,
               detections: list, draw: bool = True):
    """Save frame, optionally drawing detection boxes."""
    if draw and detections:
        vis = frame_bgr.copy()
        for cx, cy, conf, x1, y1, x2, y2 in detections:
            col = (0, 255, 0) if conf >= 0.5 else (0, 165, 255)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            cv2.putText(vis, f"{conf:.2f}", (int(x1), int(y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.imwrite(str(out_path), vis)
    else:
        cv2.imwrite(str(out_path), frame_bgr)


# ── Roboflow upload ───────────────────────────────────────────────────────────

def upload_to_roboflow(image_paths: list[Path], workspace: str,
                       project: str, api_key: str, batch_name: str):
    """Upload images to Roboflow project for labeling."""
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(1).download  # just to get project handle

    print(f"\nUploading {len(image_paths)} images to {workspace}/{project}...")
    ok, fail = 0, 0
    for i, img_path in enumerate(image_paths):
        try:
            proj.upload(str(img_path), batch_name=batch_name,
                        tag_names=[batch_name], num_retry_uploads=2)
            ok += 1
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(image_paths)} uploaded")
        except Exception as e:
            print(f"  FAIL {img_path.name}: {e}")
            fail += 1
        time.sleep(0.3)  # rate limit
    print(f"Upload complete: {ok} ok, {fail} failed")


def load_clip_entries(path: str | None, clip_root: str | None = None) -> list[dict]:
    if path is None:
        entries = [{'insv': c} for c in CLIPS]
    else:
        with open(path) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            entries = []
            for clip_id, value in raw.items():
                if isinstance(value, dict):
                    item = dict(value)
                    item.setdefault('id', clip_id)
                    entries.append(item)
                else:
                    entries.append({'id': clip_id, 'insv': value})
        else:
            entries = [dict(v) if isinstance(v, dict) else {'insv': v} for v in raw]
    if clip_root:
        root = Path(clip_root)
        for entry in entries:
            entry['insv'] = str(root / Path(entry['insv']).name)
    return entries


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', default=os.path.expanduser('~/ball_training_frames'),
                        help='Directory to save extracted frames')
    parser.add_argument('--frames-per-clip', type=int, default=200,
                        help='Frames to extract per clip (default: 200)')
    parser.add_argument('--skip-start', type=float, default=240.0,
                        help='Skip this many seconds at start of each clip (default: 240)')
    parser.add_argument('--skip-end', type=float, default=30.0,
                        help='Skip this many seconds at end of each clip (default: 30)')
    parser.add_argument('--ball-model', default='models/ball_v2.pt',
                        help='Ball detector model path')
    parser.add_argument('--device', default='cpu',
                        help='Inference device for active-learning pre-detections')
    parser.add_argument('--clips-file', default=None,
                        help='JSON list/dict of clips; entries can be strings or objects with insv')
    parser.add_argument('--clip-root', default=None,
                        help='Optional folder containing INSV files; remaps clips by basename')
    parser.add_argument('--use-insprj', action='store_true',
                        help='Sample timestamps/yaws from .insprj Studio keyframes when available')
    parser.add_argument('--groundtruth-window', type=float, default=2.0,
                        help='Seconds of jitter around each .insprj keyframe')
    parser.add_argument('--samples-per-keyframe', type=int, default=3,
                        help='Frames to sample around each .insprj keyframe')
    parser.add_argument('--upload', action='store_true',
                        help='Upload frames to Roboflow after extraction')
    parser.add_argument('--workspace', default='insta360analysis-ayiob',
                        help='Roboflow workspace slug')
    parser.add_argument('--project', default='ball_detector_merged',
                        help='Roboflow project slug')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be done without extracting')
    parser.add_argument('--yaw-range', type=float, default=30.0,
                        help='Random yaw range ±degrees for perspective extraction (default: 30)')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detected_dir = out_dir / 'detected'    # frames where ball was detected
    negative_dir = out_dir / 'negatives'   # frames without ball detection
    detected_dir.mkdir(exist_ok=True)
    negative_dir.mkdir(exist_ok=True)

    configured_entries = load_clip_entries(args.clips_file, args.clip_root)
    # Filter to clips that exist
    entries = [e for e in configured_entries if Path(e['insv']).exists()]
    missing = [e['insv'] for e in configured_entries if not Path(e['insv']).exists()]
    if missing:
        print(f"Skipping {len(missing)} missing clips:")
        for m in missing:
            print(f"  {m}")
    print(f"\nProcessing {len(entries)} clips, {args.frames_per_clip} frames each")
    print(f"Output: {out_dir}")

    if args.dry_run:
        for entry in entries:
            c = entry['insv']
            dur = get_duration(c)
            usable = max(0, dur - args.skip_start - args.skip_end)
            project = resolve_project_path(entry)
            print(f"  {Path(c).name}  duration={dur:.0f}s  usable={usable:.0f}s  insprj={bool(project)}")
        return

    # Load detector
    print(f"\nLoading ball detector: {args.ball_model}")
    model = load_detector(args.ball_model)

    all_saved = []
    total_detected = 0
    total_negative = 0

    for clip_idx, entry in enumerate(entries):
        clip_path = entry['insv']
        clip_name = Path(clip_path).stem
        print(f"\n[{clip_idx+1}/{len(entries)}] {clip_name}")

        dur = get_duration(clip_path)
        random_samples = sample_random_times(dur, args.frames_per_clip, args.skip_start, args.skip_end)
        if not random_samples:
            print("  Skipping — no usable footage")
            continue
        project = resolve_project_path(entry)
        if args.use_insprj and project:
            timestamps = sample_insprj_times(
                project,
                args.frames_per_clip,
                dur,
                args.groundtruth_window,
                args.samples_per_keyframe,
                args.skip_start,
                args.skip_end,
            )
            if not timestamps:
                timestamps = random_samples
        else:
            timestamps = random_samples

        clip_detected = 0
        clip_negative = 0

        for i, (t, yaw_hint, source) in enumerate(timestamps):
            yaw = (yaw_hint if yaw_hint is not None else 0.0) + random.uniform(-args.yaw_range, args.yaw_range)

            # Extract equirectangular frame
            eq = extract_frame(clip_path, t)
            if eq is None:
                print(f"  t={t:.1f}s: extraction failed, skipping")
                continue

            # Convert to perspective
            persp = equirect_to_persp(eq, yaw=yaw)

            # Run detector
            detections = detect_ball_raw(persp, model, device=args.device)
            high_conf = [d for d in detections if d[2] >= 0.50]

            # Save frame
            tag = f"{clip_name}_{source}_t{t:.1f}_yaw{yaw:+.0f}".replace('.', 'p')
            if high_conf:
                fname = detected_dir / f"{tag}_det.jpg"
                save_frame(persp, fname, high_conf, draw=False)
                all_saved.append(fname)
                clip_detected += 1
                total_detected += 1
            else:
                # Save every 3rd negative to avoid too many empty frames
                if i % 3 == 0:
                    fname = negative_dir / f"{tag}_neg.jpg"
                    save_frame(persp, fname, [], draw=False)
                    all_saved.append(fname)
                    clip_negative += 1
                    total_negative += 1

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{n} frames — {clip_detected} detected, {clip_negative} negatives")

        print(f"  Done: {clip_detected} detected, {clip_negative} negatives saved")

    print(f"\n{'='*55}")
    print(f"Total frames saved: {len(all_saved)}")
    print(f"  With detections: {total_detected}")
    print(f"  Negatives:       {total_negative}")
    print(f"Output dir: {out_dir}")
    print(f"\nNext step: review frames in {detected_dir}")
    print(f"Delete any that are clearly wrong before uploading.")

    if args.upload:
        api_key = os.environ.get('ROBOFLOW_API_KEY')
        if not api_key:
            print("\nERROR: ROBOFLOW_API_KEY not set — skipping upload")
            return
        batch = f"extract_{time.strftime('%Y%m%d_%H%M')}"
        # Upload detected frames (need labeling) and negatives separately
        print(f"\nUploading detected frames (batch: {batch}_detected)...")
        upload_to_roboflow(
            [p for p in all_saved if 'detected' in str(p)],
            args.workspace, args.project, api_key,
            batch_name=f"{batch}_detected"
        )
        print(f"\nUploading negatives (batch: {batch}_negatives)...")
        upload_to_roboflow(
            [p for p in all_saved if 'negatives' in str(p)],
            args.workspace, args.project, api_key,
            batch_name=f"{batch}_negatives"
        )


if __name__ == '__main__':
    main()
