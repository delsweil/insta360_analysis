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
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Clip list ────────────────────────────────────────────────────────────────

CLIPS = [
    # Oct 2024 mens game — new pitch, long balls, not yet annotated
    "/Volumes/Sickis disk/DCIM/VID_20241028_143608_10_001.insv",
    "/Volumes/Sickis disk/DCIM/VID_20241028_150214_10_002.insv",
    "/Volumes/Sickis disk/DCIM/VID_20241028_154008_10_003.insv",
    "/Volumes/Sickis disk/DCIM/VID_20241028_160641_10_004.insv",
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


def equirect_to_persp(eq_bgr: np.ndarray, yaw: float = 0.0) -> np.ndarray:
    """Simple equirect crop to perspective view at given yaw (degrees)."""
    try:
        from equilib import equi2pers
        import torch
        rgb = cv2.cvtColor(eq_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        out = equi2pers(t.unsqueeze(0),
                        rots=[{'roll': 0, 'pitch': np.radians(E2P_TILT), 'yaw': np.radians(yaw)}],
                        h_out=OUT_H, w_out=OUT_W, fov_x=E2P_FOV, mode='bilinear')
        out_np = (out.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
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


# ── Detection ─────────────────────────────────────────────────────────────────

def load_detector(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)


def detect_ball_raw(frame_bgr: np.ndarray, model, conf_thresh: float = 0.25):
    """Returns list of (cx, cy, conf, x1, y1, x2, y2) for all detections."""
    res = model(frame_bgr, imgsz=640, conf=conf_thresh,
                device='cpu', verbose=False)[0]
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

    # Filter to clips that exist
    clips = [c for c in CLIPS if Path(c).exists()]
    missing = [c for c in CLIPS if not Path(c).exists()]
    if missing:
        print(f"Skipping {len(missing)} missing clips:")
        for m in missing:
            print(f"  {m}")
    print(f"\nProcessing {len(clips)} clips, {args.frames_per_clip} frames each")
    print(f"Output: {out_dir}")

    if args.dry_run:
        for c in clips:
            dur = get_duration(c)
            usable = max(0, dur - args.skip_start - args.skip_end)
            print(f"  {Path(c).name}  duration={dur:.0f}s  usable={usable:.0f}s")
        return

    # Load detector
    print(f"\nLoading ball detector: {args.ball_model}")
    model = load_detector(args.ball_model)

    all_saved = []
    total_detected = 0
    total_negative = 0

    for clip_idx, clip_path in enumerate(clips):
        clip_name = Path(clip_path).stem
        print(f"\n[{clip_idx+1}/{len(clips)}] {clip_name}")

        dur = get_duration(clip_path)
        t_start = args.skip_start
        t_end = dur - args.skip_end
        usable = t_end - t_start

        if usable < 30:
            print(f"  Skipping — only {usable:.0f}s of usable footage")
            continue

        # Sample timestamps evenly across usable range with jitter
        n = args.frames_per_clip
        timestamps = []
        interval = usable / n
        for i in range(n):
            t = t_start + i * interval + random.uniform(0, interval * 0.8)
            timestamps.append(min(t, t_end - 1))

        clip_detected = 0
        clip_negative = 0

        for i, t in enumerate(timestamps):
            # Random yaw for variety
            yaw = random.uniform(-args.yaw_range, args.yaw_range)

            # Extract equirectangular frame
            eq = extract_frame(clip_path, t)
            if eq is None:
                print(f"  t={t:.1f}s: extraction failed, skipping")
                continue

            # Convert to perspective
            persp = equirect_to_persp(eq, yaw=yaw)

            # Run detector
            detections = detect_ball_raw(persp, model)
            high_conf = [d for d in detections if d[2] >= 0.50]

            # Save frame
            tag = f"{clip_name}_t{t:.0f}_yaw{yaw:+.0f}"
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
