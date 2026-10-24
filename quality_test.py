#!/usr/bin/env python3
"""
quality_test.py — one short clip rendered in several quality variants, side by side on disk.

    python3 quality_test.py --input VID.mp4 --keyframes v0.auto.keyframes.json --start 600 --duration 10 --out qtest

Writes qtest/1_old_720p.mp4 (what the pipeline made before), 2_lens_720p.mp4 (fitted lens only),
3_lens_1080p_enhance.mp4 (1920x1080 + light denoise/sharpen) and 4_lens_1440p_enhance.mp4 (2560x1440 + lanczos),
all with the same encoder so the differences come from the picture. The seconds each variant took are printed at the end:
the render time of a whole half is about that ratio times the length of the half. Play them one after the other
(or upload 1 and 4 to YouTube as unlisted and compare there: YouTube's own compression is part of the result).
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = [
    ("1_old_720p",              ["--lens", "legacy", "--width", "1280", "--height", "720"]),
    ("2_lens_720p",             ["--lens", "x2_150", "--width", "1280", "--height", "720"]),
    ("3_lens_1080p_enhance",    ["--lens", "x2_150", "--width", "1920", "--height", "1080", "--enhance", "light"]),
    ("3f_lens_1080p_ffmpeg",    ["--lens", "x2_150", "--engine", "ffmpeg", "--width", "1920", "--height", "1080", "--enhance", "light"]),
    ("4_lens_1440p_enhance",    ["--lens", "x2_150", "--width", "2560", "--height", "1440", "--interp-filter", "lanczos", "--enhance", "light"]),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--keyframes", required=True)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--out", default="qtest")
    p.add_argument("--encoder", default=None, help="default: h264_videotoolbox on a Mac, else libx264")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    enc = ["--encoder", "libx264", "--crf", "18"]
    if (a.encoder or ("h264_videotoolbox" if sys.platform == "darwin" else "")) == "h264_videotoolbox":
        enc = ["--encoder", "h264_videotoolbox", "--bitrate", "18000k"]
    times = []
    for name, extra in VARIANTS:
        out = os.path.join(a.out, name + ".mp4")
        cmd = [sys.executable, os.path.join(HERE, "render_keyframes.py"), "--input", a.input, "--keyframes", a.keyframes,
               "--output", out, "--start", str(a.start), "--duration", str(a.duration),
               *enc, *extra]
        print("\n==", name)
        t0 = time.time()
        if subprocess.run(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode != 0:
            print("failed:", name)
            return 1
        times.append((name, time.time() - t0))
    print("\nseconds to render %.0f s of video:" % a.duration)
    for name, t in times:
        print(f"  {name:24s} {t:6.1f} s   ({t / a.duration:.1f}x real time)")
    print(f"\ndone: {a.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
