# pipeline/probe.py
"""
Probes Insta360 .insv files, detects the file structure, and groups them
into segments for processing.

Insta360 X2 file naming convention:
  VID_YYYYMMDD_HHMMSS_00_SSS.insv  — front lens (2880x2880 fisheye)
  VID_YYYYMMDD_HHMMSS_10_SSS.insv  — rear lens  (2880x2880 fisheye)
  LRV_YYYYMMDD_HHMMSS_11_SSS.insv  — proxy (736x368, both lenses side by side)

SSS is the segment number (001, 002, ...) — segments are chunks of a recording.
The pipeline uses LRV files for detection (fast, small) and VID pairs for render.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# File role classification
# ---------------------------------------------------------------------------

def classify_insv(filename: str) -> Optional[str]:
    """
    Returns 'front', 'rear', 'lrv', or None if not a recognised .insv file.
    Based on the NN field in VID/LRV_YYYYMMDD_HHMMSS_NN_SSS.insv
    """
    name = Path(filename).name
    m = re.match(r'^(VID|LRV)_\d{8}_\d{6}_(\d{2})_\d{3}\.insv$', name, re.IGNORECASE)
    if not m:
        return None
    nn = m.group(2)
    if nn == '00':
        return 'front'
    if nn == '10':
        return 'rear'
    if nn == '11':
        return 'lrv'
    return None


def segment_key(filename: str) -> Optional[str]:
    """
    Extract the YYYYMMDD_HHMMSS_SSS key that groups related files into a segment.
    e.g. VID_20250701_174151_00_002.insv -> '20250701_174151_002'
    """
    name = Path(filename).name
    m = re.match(r'^(?:VID|LRV)_(\d{8})_(\d{6})_\d{2}_(\d{3})\.insv$', name, re.IGNORECASE)
    if not m:
        return None
    return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SegmentFiles:
    """All files belonging to one recording segment."""
    key: str               # e.g. '20250701_174151_002'
    front: Optional[str] = None   # VID _00_ path
    rear: Optional[str] = None    # VID _10_ path
    lrv: Optional[str] = None     # LRV _11_ path

    @property
    def has_lrv(self) -> bool:
        return self.lrv is not None

    @property
    def has_video(self) -> bool:
        return self.front is not None and self.rear is not None

    @property
    def detection_file(self) -> Optional[str]:
        """Best file to use for detection — LRV preferred, fall back to front."""
        return self.lrv or self.front

    def size_mb(self) -> float:
        total = 0
        for p in [self.front, self.rear, self.lrv]:
            if p and os.path.exists(p):
                total += os.path.getsize(p)
        return total / (1024 * 1024)


@dataclass
class VideoMeta:
    """Metadata for a single video file from ffprobe."""
    path: str
    width: int
    height: int
    fps: float
    duration_sec: float
    n_frames: int
    codec: str
    file_size_mb: float
    role: str   # 'front' | 'rear' | 'lrv'


@dataclass
class GameSegment:
    """One recording segment with metadata."""
    files: SegmentFiles
    lrv_meta: Optional[VideoMeta] = None
    front_meta: Optional[VideoMeta] = None

    @property
    def duration_sec(self) -> float:
        m = self.lrv_meta or self.front_meta
        return m.duration_sec if m else 0.0

    @property
    def fps(self) -> float:
        m = self.lrv_meta or self.front_meta
        return m.fps if m else 30.0

    @property
    def n_frames(self) -> int:
        m = self.lrv_meta or self.front_meta
        return m.n_frames if m else 0


@dataclass
class GameInfo:
    segments: List[GameSegment]
    total_duration_min: float
    total_frames: int
    total_size_mb: float
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\nGame: {len(self.segments)} segment(s)",
            f"  Total duration : {self.total_duration_min:.1f} min",
            f"  Total frames   : {self.total_frames:,}",
            f"  Total size     : {self.total_size_mb / 1024:.1f} GB",
        ]
        for w in self.warnings:
            lines.append(f"  [WARN] {w}")
        lines.append("")
        for i, seg in enumerate(self.segments):
            lrv = seg.files.lrv
            front = seg.files.front
            dur = seg.duration_sec
            fps = seg.fps
            lines.append(
                f"  Segment {i+1:2d}  [{seg.files.key}]"
                f"  {dur/60:.1f} min  {fps:.0f}fps"
                f"  LRV={'yes' if lrv else 'NO '}"
                f"  VID={'yes' if seg.files.has_video else 'NO'}"
            )
            if lrv and seg.lrv_meta:
                m = seg.lrv_meta
                lines.append(f"           LRV: {m.width}x{m.height}  {m.codec}  "
                              f"{os.path.basename(lrv)}")
            if front and seg.front_meta:
                m = seg.front_meta
                lines.append(f"           VID: {m.width}x{m.height}  {m.codec}  "
                              f"{os.path.basename(front)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ffprobe wrapper
# ---------------------------------------------------------------------------

def _run_ffprobe(path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        raise RuntimeError(
            "ffprobe not found. Install FFmpeg:\n"
            "  brew install ffmpeg"
        )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}:\n{result.stderr}")
    return json.loads(result.stdout)


def _parse_fps(s: str) -> float:
    if "/" in s:
        n, d = s.split("/")
        return float(n) / float(d)
    return float(s)


def probe_file(path: str, role: str) -> VideoMeta:
    data = _run_ffprobe(path)
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video:
        raise RuntimeError(f"No video stream in {path}")

    width = int(video.get("width", 0))
    height = int(video.get("height", 0))
    codec = video.get("codec_name", "unknown")
    fps = _parse_fps(video.get("r_frame_rate") or video.get("avg_frame_rate", "30/1"))
    if fps > 240:
        fps = _parse_fps(video.get("avg_frame_rate", "30/1"))

    dur_str = video.get("duration") or fmt.get("duration", "0")
    try:
        duration_sec = float(dur_str)
    except (ValueError, TypeError):
        duration_sec = 0.0

    try:
        n_frames = int(video.get("nb_frames", 0))
    except (ValueError, TypeError):
        n_frames = 0
    if n_frames == 0 and duration_sec > 0:
        n_frames = int(round(duration_sec * fps))

    size_mb = os.path.getsize(path) / (1024 * 1024)

    return VideoMeta(
        path=path, width=width, height=height, fps=fps,
        duration_sec=duration_sec, n_frames=n_frames,
        codec=codec, file_size_mb=size_mb, role=role,
    )


# ---------------------------------------------------------------------------
# Game probing
# ---------------------------------------------------------------------------

def find_and_group_segments(directory: str) -> List[SegmentFiles]:
    """
    Scan a directory for .insv files and group them into segments
    by their date/time/segment-number key.
    """
    d = Path(directory)
    all_files = sorted(d.glob("*.insv")) + sorted(d.glob("*.INSV"))

    groups: Dict[str, SegmentFiles] = {}
    unrecognised = []

    for f in all_files:
        key = segment_key(f.name)
        role = classify_insv(f.name)
        if key is None or role is None:
            unrecognised.append(f.name)
            continue
        if key not in groups:
            groups[key] = SegmentFiles(key=key)
        seg = groups[key]
        if role == 'front':
            seg.front = str(f)
        elif role == 'rear':
            seg.rear = str(f)
        elif role == 'lrv':
            seg.lrv = str(f)

    # Sort segments chronologically by key
    sorted_segs = [groups[k] for k in sorted(groups.keys())]
    return sorted_segs, unrecognised


def probe_game(directory: str) -> GameInfo:
    """
    Probe all segments in a directory and return a GameInfo.
    """
    seg_files, unrecognised = find_and_group_segments(directory)
    warnings = []

    if unrecognised:
        warnings.append(f"Unrecognised files skipped: {', '.join(unrecognised[:5])}")

    if not seg_files:
        raise RuntimeError(f"No recognisable Insta360 segments found in {directory}")

    segments = []
    for sf in seg_files:
        lrv_meta = None
        front_meta = None

        if sf.lrv:
            try:
                lrv_meta = probe_file(sf.lrv, role='lrv')
            except Exception as e:
                warnings.append(f"Could not probe LRV {os.path.basename(sf.lrv)}: {e}")

        if sf.front:
            try:
                front_meta = probe_file(sf.front, role='front')
            except Exception as e:
                warnings.append(f"Could not probe VID {os.path.basename(sf.front)}: {e}")

        if not sf.has_lrv:
            warnings.append(
                f"Segment {sf.key}: no LRV file found — "
                f"detection will use full-res VID (slower)"
            )
        if not sf.has_video:
            warnings.append(
                f"Segment {sf.key}: missing front or rear VID — "
                f"cannot render this segment"
            )

        segments.append(GameSegment(files=sf, lrv_meta=lrv_meta, front_meta=front_meta))

    total_dur = sum(s.duration_sec for s in segments) / 60.0
    total_frames = sum(s.n_frames for s in segments)
    total_size = sum(
        os.path.getsize(p)
        for sf in seg_files
        for p in [sf.front, sf.rear, sf.lrv]
        if p and os.path.exists(p)
    ) / (1024 * 1024)

    return GameInfo(
        segments=segments,
        total_duration_min=total_dur,
        total_frames=total_frames,
        total_size_mb=total_size,
        warnings=warnings,
    )


def estimate_processing_time(game: GameInfo, sample_every_n: int = 8) -> str:
    total_frames = game.total_frames
    det_frames = total_frames // sample_every_n

    # LRV is 736x368 H.264 on CPU — very fast, ~20-30fps detection throughput
    det_fps = 20.0
    det_sec = det_frames / det_fps
    # FFmpeg render: v360 stitch is CPU-bound, roughly 2-4x realtime
    render_sec = game.total_duration_min * 60 / 3.0

    total_min = (det_sec + render_sec) / 60.0

    return (
        f"\nEstimated processing time (sample_every={sample_every_n}):\n"
        f"  Detection : {det_sec/60:.0f} min  "
        f"({det_frames:,} frames from LRV files)\n"
        f"  Render    : {render_sec/60:.0f} min  (FFmpeg v360 stitch)\n"
        f"  Total     : ~{total_min:.0f} min"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.probe <directory>")
        sys.exit(1)
    game = probe_game(sys.argv[1])
    print(game.summary())
    print(estimate_processing_time(game))
