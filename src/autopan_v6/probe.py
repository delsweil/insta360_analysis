# pipeline/probe.py
"""
Probes .insv (and .mp4 equirect) files using ffprobe.
Detects resolution, fps, codec, duration, and identifies the lens model.

This runs first in the pipeline so everything downstream knows exactly
what it's working with.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .lens_models import LensModel, detect_model_from_resolution, get_model, list_models


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    duration_sec: float
    n_frames: int
    codec: str
    is_insv: bool
    model: Optional[LensModel]       # None if equirect input
    file_size_mb: float

    @property
    def is_equirect(self) -> bool:
        return not self.is_insv

    @property
    def duration_min(self) -> float:
        return self.duration_sec / 60.0

    def summary(self) -> str:
        lines = [
            f"  File     : {os.path.basename(self.path)}",
            f"  Size     : {self.file_size_mb:.0f} MB",
            f"  Codec    : {self.codec}",
            f"  Dims     : {self.width}x{self.height}",
            f"  FPS      : {self.fps:.2f}",
            f"  Duration : {self.duration_min:.1f} min  ({self.n_frames} frames)",
        ]
        if self.model:
            lines.append(f"  Model    : {self.model.name}")
        else:
            lines.append(f"  Format   : equirectangular (no lens model needed)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ffprobe wrapper
# ---------------------------------------------------------------------------

def _run_ffprobe(path: str) -> dict:
    """Run ffprobe and return parsed JSON. Raises RuntimeError on failure."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffprobe not found. Please install FFmpeg:\n"
            "  Mac:   brew install ffmpeg\n"
            "  Linux: sudo apt install ffmpeg\n"
            "  Win:   https://ffmpeg.org/download.html"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffprobe timed out on: {path}")

    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed on {path}:\n{result.stderr}"
        )

    return json.loads(result.stdout)


def _parse_fps(fps_str: str) -> float:
    """Parse ffprobe fps strings like '30000/1001' or '60/1'."""
    if "/" in fps_str:
        num, den = fps_str.split("/")
        return float(num) / float(den)
    return float(fps_str)


# ---------------------------------------------------------------------------
# Main probe function
# ---------------------------------------------------------------------------

def probe_file(
    path: str,
    model_hint: Optional[str] = None,
) -> VideoInfo:
    """
    Probe a single video file (.insv or .mp4 equirect).

    Args:
        path        : path to the file
        model_hint  : optional camera model name (e.g. "ONE X2")
                      if None, we try to detect from resolution

    Returns:
        VideoInfo dataclass with all relevant metadata
    """
    path = str(Path(path).resolve())
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    data = _run_ffprobe(path)
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    # Find the video stream
    video_stream = None
    for s in streams:
        if s.get("codec_type") == "video":
            video_stream = s
            break

    if video_stream is None:
        raise RuntimeError(f"No video stream found in: {path}")

    width  = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    codec  = video_stream.get("codec_name", "unknown")

    # FPS: prefer r_frame_rate, fall back to avg_frame_rate
    fps_raw = video_stream.get("r_frame_rate") or video_stream.get("avg_frame_rate", "30/1")
    fps = _parse_fps(fps_raw)
    # Clamp obviously wrong values (some containers report 90000/1)
    if fps > 240:
        fps = _parse_fps(video_stream.get("avg_frame_rate", "30/1"))

    # Duration: prefer stream duration, fall back to format duration
    duration_str = video_stream.get("duration") or fmt.get("duration", "0")
    try:
        duration_sec = float(duration_str)
    except (ValueError, TypeError):
        duration_sec = 0.0

    n_frames_str = video_stream.get("nb_frames", "0")
    try:
        n_frames = int(n_frames_str)
    except (ValueError, TypeError):
        n_frames = 0

    # Estimate from duration if nb_frames missing (common with HEVC)
    if n_frames == 0 and duration_sec > 0 and fps > 0:
        n_frames = int(round(duration_sec * fps))

    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    is_insv = path.lower().endswith(".insv")

    # Identify lens model
    model = None
    if is_insv:
        if model_hint:
            model = get_model(model_hint)
            if model is None:
                print(f"[WARN] Unknown model hint '{model_hint}'. Trying auto-detect.")
        if model is None:
            model = detect_model_from_resolution(width, height, fps)
        if model is None:
            print(
                f"[WARN] Could not identify camera model for {width}x{height}@{fps:.0f}fps.\n"
                f"       Supported models:"
            )
            list_models()
            print(
                f"       You can specify the model with --model 'ONE X2'\n"
                f"       Proceeding without lens model - some features disabled."
            )

    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=fps,
        duration_sec=duration_sec,
        n_frames=n_frames,
        codec=codec,
        is_insv=is_insv,
        model=model,
        file_size_mb=file_size_mb,
    )


# ---------------------------------------------------------------------------
# Multi-file probe (for a full game split into segments)
# ---------------------------------------------------------------------------

@dataclass
class GameInfo:
    segments: List[VideoInfo]
    total_duration_min: float
    total_frames: int
    total_size_mb: float
    consistent: bool          # True if all segments have same dims/fps/model
    warnings: List[str]

    def summary(self) -> str:
        lines = [
            f"\nGame: {len(self.segments)} segment(s)",
            f"  Total duration : {self.total_duration_min:.1f} min",
            f"  Total frames   : {self.total_frames:,}",
            f"  Total size     : {self.total_size_mb:.0f} MB",
        ]
        if not self.consistent:
            lines.append(f"  [WARN] Segments are not consistent!")
        for w in self.warnings:
            lines.append(f"  [WARN] {w}")
        lines.append("")
        for i, seg in enumerate(self.segments):
            lines.append(f"  Segment {i+1}:")
            lines.append(seg.summary())
        return "\n".join(lines)


def probe_game(
    paths: List[str],
    model_hint: Optional[str] = None,
    sort: bool = True,
) -> GameInfo:
    """
    Probe a list of .insv files representing a full game.

    Args:
        paths      : list of file paths
        model_hint : camera model name, applied to all segments
        sort       : sort by filename (Insta360 names files chronologically)

    Returns:
        GameInfo with all segment metadata and consistency checks
    """
    if not paths:
        raise ValueError("No files provided")

    if sort:
        paths = sorted(paths)

    segments = []
    warnings = []

    for p in paths:
        try:
            info = probe_file(p, model_hint=model_hint)
            segments.append(info)
        except Exception as e:
            warnings.append(f"Could not probe {os.path.basename(p)}: {e}")

    if not segments:
        raise RuntimeError("No files could be probed successfully.")

    # Consistency check
    ref = segments[0]
    consistent = True
    for seg in segments[1:]:
        if seg.width != ref.width or seg.height != ref.height:
            warnings.append(
                f"{os.path.basename(seg.path)}: resolution {seg.width}x{seg.height} "
                f"differs from first segment {ref.width}x{ref.height}"
            )
            consistent = False
        if abs(seg.fps - ref.fps) > 0.5:
            warnings.append(
                f"{os.path.basename(seg.path)}: fps {seg.fps:.2f} "
                f"differs from first segment {ref.fps:.2f}"
            )
            consistent = False

    total_dur = sum(s.duration_sec for s in segments) / 60.0
    total_frames = sum(s.n_frames for s in segments)
    total_size = sum(s.file_size_mb for s in segments)

    return GameInfo(
        segments=segments,
        total_duration_min=total_dur,
        total_frames=total_frames,
        total_size_mb=total_size,
        consistent=consistent,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Utility: find .insv files in a directory
# ---------------------------------------------------------------------------

def find_insv_files(directory: str) -> List[str]:
    """Find all .insv files in a directory, sorted by name."""
    d = Path(directory)
    if not d.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    files = sorted(d.glob("*.insv")) + sorted(d.glob("*.INSV"))
    return [str(f) for f in files]


# ---------------------------------------------------------------------------
# Estimate processing time
# ---------------------------------------------------------------------------

def estimate_processing_time(
    game: GameInfo,
    sample_every_n: int = 5,
    cpu_only: bool = True,
) -> str:
    """
    Rough estimate of detection + render time.
    """
    total_frames = game.total_frames
    detection_frames = total_frames // sample_every_n

    # Detection fps estimates
    if cpu_only:
        det_fps = 10.0   # YOLOv8n on modern CPU, 960px input
    else:
        det_fps = 60.0   # MPS / CUDA

    det_sec = detection_frames / det_fps
    render_sec = game.total_duration_min * 60 / 8.0  # FFmpeg ~8x realtime on CPU

    total_sec = det_sec + render_sec
    total_min = total_sec / 60.0

    lines = [
        f"\nEstimated processing time (sample_every={sample_every_n}):",
        f"  Detection frames : {detection_frames:,} of {total_frames:,}",
        f"  Detection time   : {det_sec/60:.0f} min  ({'CPU' if cpu_only else 'GPU'})",
        f"  Render time      : {render_sec/60:.0f} min  (FFmpeg)",
        f"  Total            : ~{total_min:.0f} min",
    ]
    if cpu_only:
        lines.append(
            f"\n  Tip: increase sample_every_n to {sample_every_n*2} to halve detection time"
            f"\n       with negligible impact on pan quality."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.probe <file_or_dir> [--model 'ONE X2']")
        sys.exit(1)

    model_hint = None
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        model_hint = sys.argv[idx + 1]

    target = sys.argv[1]
    if os.path.isdir(target):
        files = find_insv_files(target)
        if not files:
            print(f"No .insv files found in {target}")
            sys.exit(1)
        game = probe_game(files, model_hint=model_hint)
        print(game.summary())
        print(estimate_processing_time(game, sample_every_n=5, cpu_only=True))
    else:
        info = probe_file(target, model_hint=model_hint)
        print(info.summary())
