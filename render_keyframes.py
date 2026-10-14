#!/usr/bin/env python3
"""
render_keyframes.py — render the final broadcast-style video from a keyframe JSON.

The whole render is ONE FFmpeg process: the pan/zoom path is interpolated between
your keyframes here, written out as a per-frame `sendcmd` script, and FFmpeg's
v360 filter (equirect -> flat) re-aims the virtual camera on every frame.

    python render_keyframes.py \
        --input  VID_xxx.mp4 \
        --keyframes VID_xxx.keyframes.json \
        --output game.mp4

--input may be
    * the raw Insta360 MP4 (portrait 1440x2560): converted on the fly with the
      unchanged pipeline filter
          v360=fisheye:equirect:ih_fov=200:iv_fov=200,scale=2880:1440
      (no -noautorotate / rotate= : FFmpeg applies the Display Matrix itself), or
    * an already-equirect video (2:1), e.g. the proxy from make_proxy.py.
  Detected from the aspect ratio; override with --source raw|equirect.
  Keyframe times are seconds from the start of the video, identical for both.

Tilt (vertical aim) is a project default that individual keyframes may override with their
own optional tilt_deg (set in annotate.py with T/B, or by auto_keyframes.py --adaptive-tilt).
The project default is taken, in order, from --tilt, --calib
(same formula as autopan_infer.derive_tilt_fov), the <keyframes>.meta.json written by
annotate.py, else 0.

Interpolation (--interp): pchip (default; smooth, never overshoots a keyframe),
smoothstep (ease in/out between every pair — stops at each keyframe), linear.
Before the first / after the last keyframe the camera holds still.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import kf_core as K

# The pipeline's fisheye -> equirect step. Do not change.
RAW_TO_EQUIRECT = "v360=fisheye:equirect:ih_fov=200:iv_fov=200,scale=2880:1440"

ENCODERS = {
    "libx264":            ["-c:v", "libx264", "-preset", "medium", "-crf", "{crf}", "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart"],
    "h264_videotoolbox":  ["-c:v", "h264_videotoolbox", "-b:v", "{bitrate}", "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart"],
    "libx265":            ["-c:v", "libx265", "-preset", "medium", "-crf", "{crf}", "-pix_fmt", "yuv420p",
                           "-tag:v", "hvc1", "-movflags", "+faststart"],
    "ffv1":               ["-c:v", "ffv1"],     # lossless; use a .mkv output
}


def ff_escape(path: str) -> str:
    """Escape a path for use as a filter option value."""
    return path.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def build_sendcmd(path: K.Path, tilt: float, n_frames: int, fps: float, start: float,
                  out_w: int, out_h: int, legacy_fov: bool, target: str = "v360@cam",
                  projection: str = "flat"):
    """Per-frame v360 commands. Returns (script_text, n_command_lines, first_params).

    Each command fires half a frame *before* its frame's timestamp so float rounding in
    the timestamps can never push it onto the neighbouring frame. Frames whose camera
    didn't change since the previous one get no command at all (holds are free)."""
    idx = np.arange(n_frames, dtype=np.float64)
    t_abs = start + idx / fps
    pan, fov = path.at(t_abs)
    # per-keyframe tilt only when some keyframe sets one; otherwise the constant `tilt`, exactly as before
    tilts = path.tilt_at(t_abs) if getattr(path, "has_tilt", False) else np.full(n_frames, float(tilt))
    names = ("yaw", "pitch", "h_fov", "v_fov")
    lines, last, first = [], None, None
    for i in range(n_frames):
        p = K.to_v360_params(float(pan[i]), float(tilts[i]), float(fov[i]), out_w, out_h, legacy_fov, projection)
        vals = [p[n] for n in names]
        if first is None:                    # frame 0 uses the filter's initial options
            first, last = p, list(vals)
            continue
        changed = [(n, v) for n, v, l in zip(names, vals, last) if abs(v - l) >= 1e-4]
        if not changed:
            continue
        for n, v in changed:
            last[names.index(n)] = v
        t_cmd = max(0.0, i / fps - 0.5 / fps)
        lines.append(f"{t_cmd:.6f} " + ", ".join(f"{target} {n} {v:.5f}" for n, v in changed) + ";")
    return "\n".join(lines) + ("\n" if lines else ""), len(lines), first


def build_command(a, info: dict, source: str, cmd_file: str, n_frames: int, first: dict) -> list:
    # reset_rot=1 is essential: without it v360 treats runtime yaw/pitch commands as
    # *incremental* rotations that accumulate frame after frame.
    flat = (f"v360@cam=input=equirect:output={a.projection}:w={a.width}:h={a.height}:"
            f"interp={a.interp_filter}:reset_rot=1:"
            f"yaw={first['yaw']:.5f}:pitch={first['pitch']:.5f}:"
            f"h_fov={first['h_fov']:.5f}:v_fov={first['v_fov']:.5f}")
    parts = []
    if source == "raw":
        parts.append(RAW_TO_EQUIRECT)
    if cmd_file:                              # a static camera has no commands (sendcmd rejects an empty file)
        parts.append(f"sendcmd=f='{ff_escape(cmd_file)}'")
    parts.append(flat)
    parts.append("setsar=1")                 # the conversion's scale step leaves a non-square pixel aspect; players would stretch the video
    vf = ",".join(parts)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-stats", "-y"]
    if a.start > 0:
        cmd += ["-ss", f"{a.start:.6f}"]
    cmd += ["-i", a.input]
    cmd += ["-frames:v", str(n_frames)]
    cmd += ["-vf", vf]
    if a.audio:
        cmd += ["-map", "0:v:0", "-map", "0:a:0?", "-c:a", "aac", "-b:a", "128k"]
    else:
        cmd += ["-an"]
    enc = [x.format(crf=a.crf, bitrate=a.bitrate) for x in ENCODERS[a.encoder]]
    cmd += enc + [a.output]
    return cmd


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="raw Insta360 MP4 or equirect (2:1) video")
    p.add_argument("--keyframes", required=True, help="keyframe JSON from annotate.py")
    p.add_argument("--output", required=True)
    p.add_argument("--source", choices=["auto", "raw", "equirect"], default="auto")
    p.add_argument("--tilt", type=float, default=None, help="vertical aim in degrees (+ = up)")
    p.add_argument("--calib", default=None, help="pitch.json; derives tilt like autopan_infer.py")
    p.add_argument("--interp", choices=K.Path.METHODS, default=None,
                   help="default: the method annotate.py used (meta.json), else pchip")
    p.add_argument("--width", type=int, default=K.OUT_W)
    p.add_argument("--height", type=int, default=K.OUT_H)
    p.add_argument("--legacy-fov", action="store_true", default=None,
                   help="vertical FOV == horizontal FOV like py360convert's scalar fov (anamorphic)")
    p.add_argument("--projection", choices=K.PROJECTIONS, default=None,
                   help="flat (rectilinear) or cylindrical; default: the one annotate.py used (meta.json)")
    p.add_argument("--interp-filter", default="cubic",
                   choices=["nearest", "linear", "cubic", "lanczos", "spline16", "gaussian"],
                   help="v360 resampling filter (default cubic)")
    p.add_argument("--encoder", choices=list(ENCODERS), default="libx264")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--bitrate", default="8000k", help="for h264_videotoolbox")
    p.add_argument("--audio", action="store_true", help="keep (re-encode) the first audio track")
    p.add_argument("--start", type=float, default=None,
                   help="start time in seconds (default: the IN mark set in annotate.py, else 0)")
    p.add_argument("--duration", type=float, default=None,
                   help="seconds to render (default: up to the OUT mark set in annotate.py, else the end)")
    p.add_argument("--no-trim", action="store_true", help="ignore the IN/OUT marks and render everything")
    p.add_argument("--dry-run", action="store_true", help="print the ffmpeg command, don't run it")
    p.add_argument("--keep-cmds", default=None, metavar="PATH", help="also save the sendcmd script here")
    a = p.parse_args(argv)

    meta = K.load_meta(a.keyframes)
    tilt = a.tilt
    if tilt is None and a.calib:
        tilt = K.tilt_from_calib(a.calib)
    if tilt is None:
        tilt = float(meta.get("tilt_deg", 0.0))
    if a.legacy_fov is None:
        a.legacy_fov = bool(meta.get("legacy_fov", False))
    if a.projection is None:
        a.projection = meta.get("projection", "flat" if meta else "cylindrical")
        if a.projection not in K.PROJECTIONS:
            a.projection = "flat"
    if a.interp is None:
        a.interp = meta.get("interp", "pchip")
        if a.interp not in K.Path.METHODS:
            a.interp = "pchip"

    # the game window: explicit --start/--duration win, else the marks from annotate.py
    t_in, t_out = meta.get("trim_in"), meta.get("trim_out")
    use_marks = not a.no_trim and a.start is None and a.duration is None
    if a.start is None:
        a.start = float(t_in) if (use_marks and t_in is not None) else 0.0
    if a.duration is None and use_marks and t_out is not None:
        a.duration = max(0.0, float(t_out) - a.start)

    kfs = K.KeyframeSet.load(a.keyframes)
    if len(kfs) == 0:
        print("error: keyframe file is empty", file=sys.stderr)
        return 2
    path = K.Path(kfs.items, a.interp, base_tilt=tilt)

    info = K.probe_video(a.input)
    if info["fps"] <= 0:
        print("error: could not read the frame rate from the input", file=sys.stderr)
        return 2
    source = a.source
    if source == "auto":
        source = "equirect" if K.is_equirect_size(info["width"], info["height"]) else "raw"

    total = info["duration"]
    dur = a.duration if a.duration is not None else max(0.0, total - a.start)
    n_frames = max(1, int(round(dur * info["fps"])))
    last_kf = kfs.items[-1].t
    print(f"input    : {a.input}  ({info['width']}x{info['height']} @ {info['fps']:.3f} fps, "
          f"{total:.1f}s, source={source})")
    print(f"keyframes: {len(kfs)}  (first {kfs.items[0].t:.2f}s, last {last_kf:.2f}s)  interp={a.interp}")
    if a.start > 0 or a.duration is not None:
        end = a.start + a.duration if a.duration is not None else total
        print(f"game     : {K.fmt_time(a.start)} - {K.fmt_time(end)}  ({end - a.start:.1f}s rendered)")
    if path.has_tilt:
        tl = path.tilt_at(np.linspace(0, max(total, 1e-6), 2000))
        print(f"tilt     : per-keyframe, {tl.min():+.1f}° .. {tl.max():+.1f}°  (project default {tilt:+.2f}°)")
    print(f"camera   : tilt={tilt:+.2f}°  out={a.width}x{a.height}  "
          f"{a.projection}, {'legacy (anamorphic) fov' if a.legacy_fov else 'square-pixel fov'}")
    if total and last_kf > total + 1:
        print(f"warning  : last keyframe ({last_kf:.1f}s) is beyond the video ({total:.1f}s)", file=sys.stderr)

    text, n_cmds, first = build_sendcmd(path, tilt, n_frames, info["fps"], a.start,
                                        a.width, a.height, a.legacy_fov,
                                        projection=a.projection)
    tmpdir = tempfile.mkdtemp(prefix="kfrender_")
    cmd_file = None
    if n_cmds:
        cmd_file = os.path.join(tmpdir, "cmds.txt")
        with open(cmd_file, "w") as f:
            f.write(text)
    if a.keep_cmds:
        with open(a.keep_cmds, "w") as f:
            f.write(text)
    print(f"sendcmd  : {n_cmds} camera updates over {n_frames} frames"
          + ("  (static camera)" if not n_cmds else ""))

    cmd = build_command(a, info, source, cmd_file, n_frames, first)
    if a.dry_run:
        print("\n" + " ".join(shlex.quote(c) for c in cmd))
        shutil.rmtree(tmpdir, ignore_errors=True)
        return 0
    try:
        rc = subprocess.run(cmd).returncode
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    if rc != 0:
        print(f"error: ffmpeg exited with code {rc}", file=sys.stderr)
        return rc
    print(f"\nwrote {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
