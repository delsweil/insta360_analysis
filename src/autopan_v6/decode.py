# pipeline/decode.py
"""
Efficient frame extraction from .insv (or equirect .mp4) files.

Key design decisions:
  - FFmpeg subprocess pipe: leverages VideoToolbox HEVC hw decode on Intel Mac
  - Only decodes every Nth frame (sample_every_n) - massive CPU saving
  - For .insv: extracts both fisheye half-frames for detection
  - For equirect: extracts the pitch band only
  - Frames are yielded as numpy arrays - no temp files on disk
  - Uses rawvideo pipe for zero-overhead frame transfer

On a 2019 Intel MBP with VideoToolbox:
  HEVC decode: ~120-200fps (hardware)
  Frame transfer via pipe: negligible
  Net cost per sampled frame: ~1-3ms
"""

from __future__ import annotations

import subprocess
from typing import Generator, Iterator, Optional, Tuple

import numpy as np

from .probe import VideoInfo
from .lens_models import LensModel


# ---------------------------------------------------------------------------
# Core: FFmpeg pipe reader
# ---------------------------------------------------------------------------

class FFmpegFrameReader:
    """
    Reads frames from a video file via FFmpeg rawvideo pipe.

    Uses select_frames filter to only decode needed frames,
    minimising unnecessary HEVC decode work.
    """

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        fps: float,
        sample_every_n: int = 5,
        scale_w: Optional[int] = None,
        scale_h: Optional[int] = None,
        crop: Optional[Tuple[int, int, int, int]] = None,  # x, y, w, h
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        use_hwaccel: bool = True,
    ):
        self.path = path
        self.src_width = width
        self.src_height = height
        self.fps = fps
        self.sample_every_n = sample_every_n
        self.scale_w = scale_w
        self.scale_h = scale_h
        self.crop = crop
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.use_hwaccel = use_hwaccel

        # Compute output frame dimensions
        if crop:
            cx, cy, cw, ch = crop
            out_w, out_h = cw, ch
        else:
            out_w, out_h = width, height

        if scale_w and scale_h:
            out_w, out_h = scale_w, scale_h
        elif scale_w:
            out_h = int(out_h * scale_w / out_w)
            out_w = scale_w
        elif scale_h:
            out_w = int(out_w * scale_h / out_h)
            out_h = scale_h

        self.out_width = out_w
        self.out_height = out_h
        self.frame_bytes = out_w * out_h * 3  # BGR24

        self._proc: Optional[subprocess.Popen] = None

    def _build_ffmpeg_cmd(self) -> list:
        cmd = ["ffmpeg"]

        # Hardware decode (VideoToolbox on Mac for HEVC)
        if self.use_hwaccel:
            cmd += ["-hwaccel", "videotoolbox"]

        # Seek before input for speed (keyframe-accurate)
        if self.start_sec > 0:
            cmd += ["-ss", str(self.start_sec)]

        cmd += ["-i", self.path]

        # Duration limit
        if self.end_sec is not None:
            dur = self.end_sec - self.start_sec
            cmd += ["-t", str(dur)]

        # Build filtergraph
        filters = []

        # Frame selection: select every Nth frame
        # select=not(mod(n\,N)) selects frames 0, N, 2N, ...
        filters.append(f"select=not(mod(n\\,{self.sample_every_n}))")
        filters.append("setpts=N/FRAME_RATE/TB")  # fix pts after select

        # Crop if requested
        if self.crop:
            cx, cy, cw, ch = self.crop
            filters.append(f"crop={cw}:{ch}:{cx}:{cy}")

        # Scale if requested
        if self.scale_w or self.scale_h:
            sw = self.scale_w or -2
            sh = self.scale_h or -2
            filters.append(f"scale={sw}:{sh}")

        # Output format: BGR24 for direct numpy compatibility
        filters.append("format=bgr24")

        cmd += ["-vf", ",".join(filters)]
        cmd += [
            "-vsync", "0",        # keep select filter's frame selection intact
            "-an",                # no audio
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:1",
        ]

        return cmd

    def open(self) -> None:
        cmd = self._build_ffmpeg_cmd()
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.frame_bytes * 4,  # buffer a few frames
        )

    def close(self) -> None:
        if self._proc:
            self._proc.stdout.close()
            self._proc.terminate()
            self._proc.wait()
            self._proc = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Yields (original_frame_index, frame_bgr) tuples.
        original_frame_index is the frame number in the source video.
        """
        if self._proc is None:
            self.open()

        sample_idx = 0
        while True:
            raw = self._proc.stdout.read(self.frame_bytes)
            if len(raw) < self.frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.out_height, self.out_width, 3)
            ).copy()  # copy so buffer can be reused
            orig_idx = sample_idx * self.sample_every_n
            yield orig_idx, frame
            sample_idx += 1


# ---------------------------------------------------------------------------
# Higher-level: extract frames suitable for detection
# ---------------------------------------------------------------------------

def make_detection_reader(
    info: VideoInfo,
    sample_every_n: int = 5,
    target_width: int = 1920,
) -> FFmpegFrameReader:
    """
    Build an FFmpegFrameReader optimised for detection.

    For .insv (dual fisheye):
      - Reads the full frame (both lenses side by side)
      - Scales to target_width for faster YOLO inference
      - Pitch band crop applied after scale

    For equirect .mp4:
      - Reads full width, crops to pitch band vertically
      - Scales to target_width

    Args:
        info          : VideoInfo from probe_file()
        sample_every_n: sample 1 in N frames
        target_width  : width to scale to before detection
                        960 or 1280 are good values for YOLOv8n
    """
    w, h = info.width, info.height

    if info.is_insv and info.model is not None:
        m = info.model
        # For fisheye, we work on the full frame (both lenses)
        # Pitch band crop: top/bottom fractions of the frame height
        y1 = int(m.pitch_band_top * h)
        y2 = int(m.pitch_band_bot * h)
        band_h = y2 - y1

        # Scale: keep aspect ratio from target_width
        scale_h = int(band_h * target_width / w)
        # Round to even (required by some codecs)
        scale_h = scale_h - (scale_h % 2)

        return FFmpegFrameReader(
            path=info.path,
            width=w,
            height=h,
            fps=info.fps,
            sample_every_n=sample_every_n,
            crop=(0, y1, w, band_h),
            scale_w=target_width,
            scale_h=scale_h,
            use_hwaccel=True,
        )

    else:
        # Equirect: crop pitch band (middle 70% vertically is usually enough)
        y1 = int(0.15 * h)
        y2 = int(0.85 * h)
        band_h = y2 - y1
        scale_h = int(band_h * target_width / w)
        scale_h = scale_h - (scale_h % 2)

        return FFmpegFrameReader(
            path=info.path,
            width=w,
            height=h,
            fps=info.fps,
            sample_every_n=sample_every_n,
            crop=(0, y1, w, band_h),
            scale_w=target_width,
            scale_h=scale_h,
            use_hwaccel=True,
        )


# ---------------------------------------------------------------------------
# Coordinate mapping: detection pixel -> yaw angle
# ---------------------------------------------------------------------------

def fisheye_pixel_to_yaw(
    px: float,
    py: float,
    frame_w: int,
    frame_h: int,
    model: LensModel,
    crop_y1: int = 0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Optional[float]:
    """
    Convert a detection pixel coordinate (in the scaled/cropped detection frame)
    back to a yaw angle in degrees.

    For dual fisheye, the two lenses cover:
      Left lens  (lens0): yaw  90° to 270° (rear half)  -- camera-dependent
      Right lens (lens1): yaw -90° to  90° (front half)

    In practice for a pitch-centre-mounted camera we care about the
    horizontal sweep across the pitch. We use a simplified equidistant
    fisheye model: angle from lens centre is proportional to distance
    from lens centre in pixels.

    Returns yaw in degrees [-180, 180], or None if point is outside
    valid fisheye circle.
    """
    # Undo scale
    orig_px = px * scale_x
    orig_py = (py + crop_y1) * scale_y  # undo crop offset

    # Determine which lens the point falls in
    mid_x = frame_w * 0.5
    lens_radius_px = model.lens_radius_frac * frame_h

    if orig_px < mid_x:
        # Left lens (lens0)
        cx = model.lens0_cx_frac * frame_w
        cy = model.lens0_cy_frac * frame_h
        lens_yaw_offset = 180.0   # left lens points backward
    else:
        # Right lens (lens1)
        cx = model.lens1_cx_frac * frame_w
        cy = model.lens1_cy_frac * frame_h
        lens_yaw_offset = 0.0     # right lens points forward

    dx = orig_px - cx
    dy = orig_py - cy
    r = np.sqrt(dx * dx + dy * dy)

    # Outside fisheye circle?
    if r > lens_radius_px * 1.05:  # small tolerance
        return None

    # Equidistant: angle from optical axis = r / lens_radius * (fov/2)
    half_fov = model.lens_fov_deg / 2.0
    theta = (r / lens_radius_px) * half_fov   # degrees from optical axis

    # Horizontal angle (azimuth within the lens)
    if r < 1e-3:
        phi = 0.0
    else:
        phi = np.degrees(np.arctan2(dx, -dy))  # 0=up, increases clockwise

    # Combine to global yaw
    # The horizontal component of phi gives us pan direction
    yaw = lens_yaw_offset + phi
    # Normalise to [-180, 180]
    yaw = ((yaw + 180) % 360) - 180

    return float(yaw)


def equirect_pixel_to_yaw(px: float, frame_w: int) -> float:
    """
    Convert x pixel coordinate in an equirectangular frame to yaw degrees.
    Linear mapping: x=0 -> -180°, x=W -> +180°
    """
    return (px / frame_w - 0.5) * 360.0


# ---------------------------------------------------------------------------
# Utility: count frames quickly without decoding
# ---------------------------------------------------------------------------

def fast_frame_count(path: str, fps: float, duration_sec: float) -> int:
    """Estimate frame count from duration and fps (avoids full decode)."""
    return int(round(fps * duration_sec))
