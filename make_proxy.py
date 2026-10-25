#!/usr/bin/env python3
"""
make_proxy.py — convert a raw Insta360 MP4 to an equirectangular video for annotating.

    python make_proxy.py VID_xxx.mp4                  # -> VID_xxx.equirect.mp4 (1920x960)
    python make_proxy.py VID_xxx.mp4 --width 2880     # full 2880x1440
    python make_proxy.py VID_xxx.mp4 --iv-fov 112     # the geometry the app's pipeline uses (keyframes made by it)
    python make_proxy.py VID_xxx.mp4 --start 600 --duration 120   # just a test clip

Uses the unchanged pipeline filter
    v360=fisheye:equirect:ih_fov=200:iv_fov=200,scale=<W>:<W/2>
(no -noautorotate / rotate= : FFmpeg applies the file's Display Matrix itself).

Made for annotation, so it is encoded with a short GOP (keyframe every 10 frames) and
constant frame rate: seeking, frame-stepping backwards and timeline scrubbing in
annotate.py are then instant and frame-accurate. The timeline is unchanged (t=0 is
t=0), so keyframes annotated on the proxy render correctly from the raw file.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import kf_core as K


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("output", nargs="?", default=None)
    p.add_argument("--width", type=int, default=1920,
                   help="equirect width, height = width/2 (default 1920; 2880 = full size)")
    p.add_argument("--crf", type=int, default=21)
    p.add_argument("--gop", type=int, default=10, help="keyframe interval in frames (default 10)")
    p.add_argument("--iv-fov", type=float, default=K.DEFAULT_IV_FOV,
                   help="vertical field of the fisheye model: use the same value as the detections/keyframes of the game "
                        "(the pipeline uses 112; the default 200 is the old circular model)")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--encoder", default="libx264", choices=["libx264", "h264_videotoolbox"])
    a = p.parse_args(argv)

    if a.width % 2:
        p.error("--width must be even")
    info = K.probe_video(a.input)
    if info["fps"] <= 0:
        print("error: could not read frame rate", file=sys.stderr)
        return 2
    out = a.output or os.path.splitext(a.input)[0] + ".equirect.mp4"
    h = a.width // 2
    vf = f"v360=fisheye:equirect:ih_fov=200:iv_fov={a.iv_fov:g},scale={a.width}:{h},setsar=1"

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-stats", "-y"]
    if a.start > 0:
        cmd += ["-ss", f"{a.start:.3f}"]
    cmd += ["-i", a.input]
    if a.duration:
        cmd += ["-t", f"{a.duration:.3f}"]
    cmd += ["-vf", vf, "-an", "-r", f"{info['fps']:.6f}"]
    if a.encoder == "libx264":
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(a.crf),
                "-g", str(a.gop), "-bf", "0", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "h264_videotoolbox", "-b:v", "20000k", "-g", str(a.gop), "-bf", "0",
                "-pix_fmt", "yuv420p"]
    cmd += ["-movflags", "+faststart", out]

    print(f"{a.input} ({info['width']}x{info['height']} @ {info['fps']:.3f} fps, {info['duration']:.0f}s)"
          f"\n  -> {out} ({a.width}x{h})")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"error: ffmpeg exited with code {rc}", file=sys.stderr)
        return rc
    print(f"\nwrote {out}\nnext: python annotate.py {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
