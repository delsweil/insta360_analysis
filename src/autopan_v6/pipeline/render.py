# pipeline/render.py
"""
Render phase: takes the dense per-frame yaw curve and drives FFmpeg to
produce the final autopanned MP4.

Two rendering strategies depending on input:

1. .insv input (dual fisheye):
   FFmpeg pipeline:
     - Decode .insv (HEVC, VideoToolbox hw accel on Mac)
     - v360 filter: dfisheye -> rectilinear, per-frame yaw via sendcmd
     - Encode: h264 (VideoToolbox hw on Mac, or libx264 software)

2. Equirect .mp4 input:
   FFmpeg pipeline:
     - Decode equirect
     - v360 filter: equirect -> rectilinear, per-frame yaw
     - Encode: h264

The per-frame yaw is driven via FFmpeg's 'sendcmd' mechanism: we write
a commands file that sets the v360 yaw parameter at each frame timestamp.

Output is a standard h264/AAC MP4 suitable for YouTube upload.

Performance on 2019 Intel MBP:
  - VideoToolbox h264 encode: ~60-120 fps (hardware)
  - v360 filter: ~30-60 fps (CPU, unavoidably - it's a pixel remap)
  - Net: typically 3-6x realtime for a 90-min game = 15-30 min render
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .probe import VideoMeta as VideoInfo, GameInfo
from .lens_models import LensModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RenderConfig:
    # Output dimensions
    out_w: int = 1280
    out_h: int = 720

    # Field of view of the output rectilinear view (degrees)
    fov_deg: float = 90.0

    # Pitch angle of output camera (degrees, 0 = horizon)
    pitch_deg: float = 0.0

    # Encoder: 'h264_videotoolbox' (Mac hw), 'libx264' (sw), 'h264_nvenc' (Nvidia)
    # 'auto' = detect best available
    encoder: str = "auto"

    # Output quality
    # For videotoolbox: bitrate in kbps (6000 = good for 1280x720)
    videotoolbox_bitrate_kbps: int = 6000
    # For libx264: CRF (18=high quality, 23=default, 28=smaller file)
    libx264_crf: int = 20

    # Audio: copy from source (True) or strip (False)
    copy_audio: bool = True

    # YouTube-optimised: add faststart flag for progressive download
    youtube_optimise: bool = True

    # Intermediate concat: if True, segments are concatenated before
    # v360 processing (single FFmpeg pass). If False, each segment is
    # processed separately and then concatenated.
    # Single pass is cleaner but requires holding a concat demuxer file.
    single_pass: bool = True

    # Temporary file directory
    tmp_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Encoder detection
# ---------------------------------------------------------------------------

def detect_encoder() -> str:
    """
    Detect the best available h264 encoder.
    Returns encoder name string for FFmpeg.
    """
    # Try VideoToolbox (Mac hardware)
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10
        )
        if "h264_videotoolbox" in result.stdout:
            return "h264_videotoolbox"
        if "h264_nvenc" in result.stdout:
            return "h264_nvenc"
    except Exception:
        pass
    return "libx264"


# ---------------------------------------------------------------------------
# Sendcmd file generation
# ---------------------------------------------------------------------------

def write_sendcmd_file(
    yaw_curve: np.ndarray,
    pitch_curve: np.ndarray,
    fps: float,
    path: str,
    frame_offset: int = 0,
) -> None:
    """
    Write an FFmpeg sendcmd file that sets v360 yaw/pitch per frame.

    FFmpeg sendcmd format:
        <timestamp> [enter|leave] <filter>@<instance> <cmd> <arg>;

    We use the 'enter' trigger at each frame timestamp.

    Args:
        yaw_curve    : per-frame yaw angles (degrees)
        pitch_curve  : per-frame pitch angles (degrees)
        fps          : video fps
        path         : output file path
        frame_offset : start frame index (for multi-segment)
    """
    n = len(yaw_curve)
    with open(path, "w") as f:
        for i in range(n):
            t = (i + frame_offset) / fps
            y = float(yaw_curve[i])
            p = float(pitch_curve[i])
            # Set yaw
            f.write(f"{t:.6f} [enter] v360@panner yaw {y:.4f};\n")
            # Set pitch (v_flip in FFmpeg v360 = pitch)
            f.write(f"{t:.6f} [enter] v360@panner pitch {p:.4f};\n")


def write_sendcmd_file_multi(
    yaw_curve: np.ndarray,
    pitch_curve: np.ndarray,
    fps: float,
    path: str,
) -> None:
    """
    Write sendcmd for a full concatenated game (handles segment boundaries).
    Same as write_sendcmd_file but for the full game sequence.
    """
    write_sendcmd_file(yaw_curve, pitch_curve, fps, path, frame_offset=0)


# ---------------------------------------------------------------------------
# Concat demuxer file
# ---------------------------------------------------------------------------

def write_concat_file(paths: List[str], out_path: str) -> None:
    """Write an FFmpeg concat demuxer file."""
    with open(out_path, "w") as f:
        f.write("ffconcat version 1.0\n")
        for p in paths:
            # Escape single quotes in paths
            escaped = p.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")


# ---------------------------------------------------------------------------
# Build FFmpeg command
# ---------------------------------------------------------------------------

def _encoder_flags(encoder: str, cfg: RenderConfig) -> List[str]:
    """Return encoder-specific FFmpeg flags."""
    if encoder == "h264_videotoolbox":
        return [
            "-c:v", "h264_videotoolbox",
            "-b:v", f"{cfg.videotoolbox_bitrate_kbps}k",
            "-profile:v", "high",
            "-level", "4.1",
        ]
    elif encoder == "h264_nvenc":
        return [
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-cq", "20",
            "-profile:v", "high",
        ]
    else:  # libx264
        return [
            "-c:v", "libx264",
            "-crf", str(cfg.libx264_crf),
            "-preset", "faster",
            "-profile:v", "high",
            "-level", "4.1",
        ]


def build_render_cmd(
    input_paths: List[str],
    sendcmd_path: str,
    output_path: str,
    info: VideoInfo,
    cfg: RenderConfig,
    encoder: str,
    concat_file: Optional[str] = None,
) -> List[str]:
    """
    Build the FFmpeg command for the render phase.

    The filtergraph:
      [input] -> v360 (stitch + pan via sendcmd) -> scale -> [output]

    For .insv (dual fisheye):
      v360=dfisheye:rectilinear:ih_fov=210:iv_fov=210:yaw=0:pitch=0:roll=0

    For equirect:
      v360=e:rectilinear:yaw=0:pitch=0:roll=0
    """
    cmd = ["ffmpeg", "-hide_banner", "-y"]

    # Hardware decode
    cmd += ["-hwaccel", "videotoolbox"]

    # Input
    if concat_file:
        cmd += ["-f", "concat", "-safe", "0", "-i", concat_file]
    elif len(input_paths) == 1:
        cmd += ["-i", input_paths[0]]
    else:
        # Shouldn't reach here if concat_file is used for multi-input
        cmd += ["-i", input_paths[0]]

    # Determine v360 input format
    if info.is_insv and info.model:
        m = info.model
        v360_input = "dfisheye"
        fov_flags = f":ih_fov={m.lens_fov_deg:.1f}:iv_fov={m.lens_fov_deg:.1f}"
    else:
        v360_input = "e"        # equirectangular
        fov_flags = ""

    # Build filtergraph
    # sendcmd reads from file and drives named filter instance 'panner'
    vf_parts = [
        f"sendcmd=f={sendcmd_path}",
        (
            f"v360={v360_input}:rectilinear"
            f"{fov_flags}"
            f":w={cfg.out_w}:h={cfg.out_h}"
            f":v_fov={cfg.fov_deg:.1f}:h_fov={cfg.fov_deg:.1f}"
            f":yaw=0:pitch={cfg.pitch_deg:.1f}:roll=0"
            f"[panner]"        # named instance for sendcmd targeting
        ),
    ]

    # Note: sendcmd must come before the filter it controls
    # Full filtergraph: sendcmd,v360@panner
    filtergraph = (
        f"sendcmd=f='{sendcmd_path}',"
        f"v360={v360_input}:rectilinear"
        f"{fov_flags}"
        f":w={cfg.out_w}:h={cfg.out_h}"
        f":v_fov={cfg.fov_deg:.1f}:h_fov={cfg.fov_deg:.1f}"
        f":yaw=0:pitch={cfg.pitch_deg:.1f}:roll=0"
        f"@panner"
    )

    cmd += ["-vf", filtergraph]

    # Encoder
    cmd += _encoder_flags(encoder, cfg)

    # Audio
    if cfg.copy_audio:
        cmd += ["-c:a", "aac", "-b:a", "128k"]
    else:
        cmd += ["-an"]

    # YouTube optimisation
    if cfg.youtube_optimise:
        cmd += ["-movflags", "+faststart"]

    # Pixel format (required for h264)
    cmd += ["-pix_fmt", "yuv420p"]

    cmd.append(output_path)
    return cmd


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_game(
    game: GameInfo,
    yaw_curve: np.ndarray,
    pitch_curve: np.ndarray,
    output_path: str,
    cfg: RenderConfig,
) -> None:
    """
    Render the full game to a panned MP4.

    Args:
        game        : GameInfo (list of segments)
        yaw_curve   : dense per-frame yaw angles (full game)
        pitch_curve : dense per-frame pitch angles (full game)
        output_path : path for output MP4
        cfg         : RenderConfig
    """
    import time

    # Resolve encoder
    encoder = cfg.encoder
    if encoder == "auto":
        encoder = detect_encoder()
    print(f"\nRender: encoder={encoder}  output={output_path}")

    fps = game.segments[0].fps
    total_frames = game.total_frames
    info = game.segments[0]  # use first segment for format detection

    assert len(yaw_curve) == total_frames, (
        f"yaw_curve length {len(yaw_curve)} != total_frames {total_frames}"
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    tmp = cfg.tmp_dir or tempfile.mkdtemp(prefix="autopan_")
    os.makedirs(tmp, exist_ok=True)

    # Write sendcmd file
    sendcmd_path = os.path.join(tmp, "sendcmd.txt")
    print(f"Writing sendcmd file ({total_frames} entries)...")
    write_sendcmd_file_multi(yaw_curve, pitch_curve, fps, sendcmd_path)
    print(f"  -> {sendcmd_path}  ({os.path.getsize(sendcmd_path)//1024} KB)")

    # Write concat file if multiple segments
    input_paths = [s.path for s in game.segments]
    concat_file = None
    if len(input_paths) > 1:
        concat_file = os.path.join(tmp, "concat.txt")
        write_concat_file(input_paths, concat_file)
        print(f"Concat file: {concat_file}  ({len(input_paths)} segments)")

    # Build FFmpeg command
    cmd = build_render_cmd(
        input_paths=input_paths,
        sendcmd_path=sendcmd_path,
        output_path=output_path,
        info=info,
        cfg=cfg,
        encoder=encoder,
        concat_file=concat_file,
    )

    print(f"\nFFmpeg command:")
    print("  " + " ".join(cmd))
    print(f"\nRendering {game.total_duration_min:.1f} min of footage...")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"\n[ERROR] FFmpeg failed:")
        print(result.stderr[-3000:])  # last 3000 chars of stderr
        raise RuntimeError("FFmpeg render failed")

    elapsed = time.time() - t0
    speed = game.total_duration_min * 60 / elapsed
    size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\nRender complete:")
    print(f"  Output  : {output_path}")
    print(f"  Size    : {size_mb:.0f} MB")
    print(f"  Time    : {elapsed/60:.1f} min  ({speed:.1f}x realtime)")


# ---------------------------------------------------------------------------
# Segment-by-segment render (fallback if single-pass fails)
# ---------------------------------------------------------------------------

def render_segments_concat(
    game: GameInfo,
    yaw_curve: np.ndarray,
    pitch_curve: np.ndarray,
    output_path: str,
    cfg: RenderConfig,
) -> None:
    """
    Alternative: render each segment separately, then concatenate.
    Slower but more robust for edge cases (different resolutions, etc.)
    """
    import time

    encoder = cfg.encoder
    if encoder == "auto":
        encoder = detect_encoder()

    tmp = cfg.tmp_dir or tempfile.mkdtemp(prefix="autopan_")
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    rendered_parts = []
    frame_offset = 0

    for i, seg in enumerate(game.segments):
        seg_yaw = yaw_curve[frame_offset: frame_offset + seg.n_frames]
        seg_pitch = pitch_curve[frame_offset: frame_offset + seg.n_frames]

        sendcmd_path = os.path.join(tmp, f"sendcmd_{i:02d}.txt")
        write_sendcmd_file(seg_yaw, seg_pitch, seg.fps, sendcmd_path)

        part_path = os.path.join(tmp, f"part_{i:02d}.mp4")
        cmd = build_render_cmd(
            input_paths=[seg.path],
            sendcmd_path=sendcmd_path,
            output_path=part_path,
            info=seg,
            cfg=cfg,
            encoder=encoder,
        )

        print(f"\nRendering segment {i+1}/{len(game.segments)}: {os.path.basename(seg.path)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr[-2000:])
            raise RuntimeError(f"FFmpeg failed on segment {i+1}")

        rendered_parts.append(part_path)
        frame_offset += seg.n_frames

    # Concatenate parts
    if len(rendered_parts) == 1:
        import shutil
        shutil.move(rendered_parts[0], output_path)
    else:
        concat_file = os.path.join(tmp, "final_concat.txt")
        write_concat_file(rendered_parts, concat_file)
        concat_cmd = [
            "ffmpeg", "-hide_banner", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c", "copy",
            output_path,
        ]
        print(f"\nConcatenating {len(rendered_parts)} parts...")
        subprocess.run(concat_cmd, check=True)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nOutput: {output_path}  ({size_mb:.0f} MB)")
