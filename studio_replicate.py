#!/usr/bin/env python3
import argparse
import os
import platform
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from py360convert import e2p
from scipy.interpolate import PchipInterpolator


_ENCODER_CACHE = None


def pick_encoder():
    global _ENCODER_CACHE
    if _ENCODER_CACHE is not None:
        return _ENCODER_CACHE
    try:
        r = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'],
                           capture_output=True, text=True, timeout=5)
        available = set(r.stdout.split())
    except Exception:
        available = set()
    candidates = []
    if platform.system() == 'Darwin' and 'h264_videotoolbox' in available:
        candidates.append('h264_videotoolbox')
    try:
        has_nvidia = subprocess.run(['nvidia-smi'], capture_output=True, timeout=2).returncode == 0
    except Exception:
        has_nvidia = False
    if has_nvidia and 'h264_nvenc' in available:
        candidates.append('h264_nvenc')
    candidates.append('libx264')
    _ENCODER_CACHE = next((c for c in candidates if c == 'libx264' or c in available), 'libx264')
    return _ENCODER_CACHE


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
            time_s   = int(kf.get('time', 0)) / 1000,
            pan_deg  = float(np.degrees(float(kf.get('pan',  0)))),
            tilt_deg = float(np.degrees(float(kf.get('tilt', 0)))),
            fov_deg  = float(np.degrees(float(kf.get('fov',  0)))),
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
        self.t_min = times[0]
        self.t_max = times[-1]

    def at(self, t: float):
        t = float(np.clip(t, self.t_min, self.t_max))
        return float(self.cs_pan(t)), float(self.cs_tilt(t)), float(self.cs_fov(t))


def project_frame(frame, pan_deg, tilt_deg, fov_deg, out_h, out_w, rotate_ccw=True):
    if rotate_ccw:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    persp = e2p(rgb, fov_deg=fov_deg, u_deg=pan_deg, v_deg=tilt_deg,
                out_hw=(out_h, out_w), mode='bilinear')
    return cv2.cvtColor(persp, cv2.COLOR_RGB2BGR)


def open_writer(output_path, fps, out_w, out_h):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{out_w}x{out_h}', '-r', str(fps),
        '-i', 'pipe:0',
        '-c:v', pick_encoder(), '-b:v', '8000k',
        '-pix_fmt', 'yuv420p', output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def run(args):
    keyframes = parse_keyframes(args.project)
    print(f'Loaded {len(keyframes)} keyframes')
    print(f'Time range: {keyframes[0].time_s:.1f}s to {keyframes[-1].time_s:.1f}s')

    curve = StudioCurve(keyframes)

    cap = cv2.VideoCapture(args.insv)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open: {args.insv}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_t = args.offset if args.offset is not None else keyframes[0].time_s
    start_frame = int(start_t * fps)
    if start_frame >= total_frames:
        start_frame = 0
        print(f'Pre-stitched clip detected — starting from frame 0')
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out_w, out_h = 1280, 720
    max_frames = int(args.duration * fps)
    writer = open_writer(args.output, fps, out_w, out_h)

    t0 = time.time()
    idx = 0

    print(f'Rendering {args.duration}s from t={start_t:.1f}s...')
    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        t = start_t + idx / fps
        pan, tilt, fov = curve.at(t)
        # Convert Studio params to e2p params
        pan  = -pan        # Studio pan is inverted
        tilt = tilt * 1.20  # Empirical tilt conversion
        fov  = fov  * 1.85  # Empirical FOV conversion
        persp = project_frame(frame, pan, tilt, fov, out_h, out_w,
                              rotate_ccw=not args.no_rotate)
        writer.stdin.write(persp.tobytes())

        idx += 1
        if idx % 100 == 0:
            dt = time.time() - t0
            print(f'  [{100*idx/max_frames:5.1f}%] frame {idx:4d} '
                  f'{idx/dt:5.1f}fps  pan={pan:+.1f}° tilt={tilt:+.1f}° fov={fov:.1f}°')

    cap.release()
    writer.stdin.close()
    writer.wait()
    dt = time.time() - t0
    print(f'\nDone. {idx} frames in {dt:.1f}s ({idx/dt:.1f} fps)')
    print(f'Output: {args.output}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--insv',      required=True)
    p.add_argument('--project',   required=True)
    p.add_argument('--output',    default='/tmp/replicated.mp4')
    p.add_argument('--duration',  type=float, default=60)
    p.add_argument('--no-rotate', action='store_true')
    p.add_argument('--offset',   type=float, default=None, help='Project timeline start time in seconds')
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
