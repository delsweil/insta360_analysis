#!/usr/bin/env python3
"""
view_render.py — the fast renderer for raw X2 footage.

FFmpeg's v360 filter rebuilds its whole remap table every time the camera moves, i.e. on every frame of a
panning shot: that costs ~0.5 s per 1080p frame and is why a 1080p render ran at ~4x real time. Here the table is
built on a coarse grid (the projection is smooth: 1/16 of the points, error well under 0.3 px, the same trick as
kf_core.build_remap) and blown up with a bilinear resize, then cv2.remap cuts the view out of the raw frame in one
pass (fisheye -> flat view with the fitted lens, lens.py). Three threads overlap decoding, remapping and encoding.

    ffmpeg (decode, raw BGR) -> [this module: maps + remap] -> ffmpeg (denoise/sharpen, encode, audio)

Used by render_keyframes.py (--engine opencv|auto) for lens x2_150, raw input and the flat projection.
"""

from __future__ import annotations

import math
import queue
import subprocess
import sys
import threading
from typing import Dict, List

import numpy as np

import lens as L

STEP = 4        # grid spacing (output pixels) of the coarse map


def have_cv2() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except Exception:
        return False


def make_maps(yaw: float, pitch: float, h_fov: float, out_w: int, out_h: int, step: int = STEP):
    """(map_x, map_y) float32 for cv2.remap: where each output pixel reads the raw frame."""
    import cv2
    tx = math.tan(math.radians(h_fov) / 2.0)
    ty = tx * out_h / out_w
    # cv2.resize puts coarse sample k at big pixel k*step + (step-1)/2, so sample k is computed for the output pixel
    # (k-1)*step + (step-1)/2 and the result is cropped from `step` on (one spare cell on every side)
    nx, ny = out_w // step + 3, out_h // step + 3
    cx = (np.arange(nx) - 1) * step + (step - 1) / 2.0
    cy = (np.arange(ny) - 1) * step + (step - 1) / 2.0
    gx = ((2.0 * cx + 1.0) / out_w - 1.0) * tx                              # pixel centres, like v360
    gy = (1.0 - (2.0 * cy + 1.0) / out_h) * ty
    f, r, u = L._basis(yaw, pitch)
    rays = f[None, None, :] + gx[None, :, None] * r[None, None, :] + gy[:, None, None] * u[None, None, :]
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    px, py = L.true_dir_to_raw(rays)
    out = []
    for m in (px, py):
        big = cv2.resize(m.astype(np.float32), (nx * step, ny * step), interpolation=cv2.INTER_LINEAR)
        out.append(np.ascontiguousarray(big[step:step + out_h, step:step + out_w]))
    return out[0], out[1]


def build_commands(a, info: dict, n_frames: int, enhance_filters: List[str], encoder_args: List[str]):
    fps = f"{info['fps']:.6f}"
    reader = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if a.start > 0:
        reader += ["-ss", f"{a.start:.6f}"]
    reader += ["-i", a.input, "-frames:v", str(n_frames), "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    vf = ",".join(enhance_filters + ["setsar=1"])
    writer = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-stats", "-y",
              "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{a.width}x{a.height}", "-framerate", fps, "-i", "pipe:0"]
    if a.audio:
        if a.start > 0:
            writer += ["-ss", f"{a.start:.6f}"]
        writer += ["-t", f"{n_frames / info['fps']:.6f}", "-i", a.input,
                   "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "128k", "-shortest"]
    else:
        writer += ["-an"]
    writer += ["-vf", vf] + encoder_args + [a.output]
    return reader, writer


def interp_flag(name: str):
    import cv2
    return {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR}.get(
        name, cv2.INTER_LANCZOS4 if name == "lanczos" else cv2.INTER_CUBIC)


def run(a, info: dict, params: List[Dict[str, float]], n_frames: int, enhance_filters: List[str],
        encoder_args: List[str]) -> int:
    import cv2
    reader_cmd, writer_cmd = build_commands(a, info, n_frames, enhance_filters, encoder_args)
    in_w, in_h = L.RAW_W, L.RAW_H
    frame_bytes = in_w * in_h * 3
    flag = interp_flag(a.interp_filter)
    if a.interp_filter == "lanczos":
        print("note     : lanczos is slow in this renderer; cubic looks the same", file=sys.stderr)

    rd = subprocess.Popen(reader_cmd, stdout=subprocess.PIPE)
    wr = subprocess.Popen(writer_cmd, stdin=subprocess.PIPE)
    q_in: "queue.Queue" = queue.Queue(maxsize=6)
    q_out: "queue.Queue" = queue.Queue(maxsize=6)
    errors: List[str] = []

    def read_loop():
        try:
            while True:
                buf = rd.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                q_in.put(np.frombuffer(buf, np.uint8).reshape(in_h, in_w, 3))
        except Exception as e:  # pragma: no cover
            errors.append(f"reader: {e}")
        finally:
            q_in.put(None)

    def write_loop():
        try:
            while True:
                fr = q_out.get()
                if fr is None:
                    break
                wr.stdin.write(fr.tobytes())
        except Exception as e:
            errors.append(f"writer: {e}")
            while q_out.get() is not None:      # keep the producer from blocking
                pass

    tr = threading.Thread(target=read_loop, daemon=True)
    tw = threading.Thread(target=write_loop, daemon=True)
    tr.start()
    tw.start()
    last_key, maps = None, None
    i = 0
    try:
        while True:
            fr = q_in.get()
            if fr is None:
                break
            p = params[min(i, len(params) - 1)]
            key = (round(p["yaw"], 4), round(p["pitch"], 4), round(p["h_fov"], 4))
            if key != last_key:
                maps = make_maps(p["yaw"], p["pitch"], p["h_fov"], a.width, a.height)
                last_key = key
            q_out.put(cv2.remap(fr, maps[0], maps[1], flag, borderMode=cv2.BORDER_REPLICATE))
            i += 1
            if i >= n_frames:
                break
    finally:
        q_out.put(None)
        tw.join()
        try:
            wr.stdin.close()
        except Exception:
            pass
        rd.kill()
        rc = wr.wait()
        rd.wait()
    if errors:
        print("error: " + "; ".join(errors), file=sys.stderr)
        return 1
    if i == 0:
        print("error: no frames could be read from the input", file=sys.stderr)
        return 1
    return rc
