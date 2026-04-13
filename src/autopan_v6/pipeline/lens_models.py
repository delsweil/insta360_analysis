# pipeline/lens_models.py
"""
Lens parameter table for Insta360 cameras.

Each entry defines the dual-fisheye geometry needed to:
  1. Decode the raw .insv frame layout
  2. Map fisheye pixel coordinates -> yaw angles for detection
  3. Drive FFmpeg v360 for the final equirect stitch

Parameters per model:
  frame_w, frame_h : raw .insv frame dimensions (both lenses side by side)
  lens_fov_deg     : field of view of each fisheye lens (degrees)
  projection       : fisheye projection type for FFmpeg v360
                     'equidistant' | 'equisolid' | 'orthographic' | 'stereographic'
  lens0_cx_frac    : centre of left lens as fraction of frame width
  lens0_cy_frac    : centre of left lens as fraction of frame height
  lens1_cx_frac    : centre of right lens as fraction of frame width
  lens1_cy_frac    : centre of right lens as fraction of frame height
  lens_radius_frac : radius of each fisheye circle as fraction of frame height
  pitch_band_y_frac: (top, bottom) of pitch region as fraction of frame height
                     used to restrict detection to the relevant vertical band
  ffmpeg_input_fmt : v360 input format string for FFmpeg

Sources: Gyroflow lens database, community measurements, Insta360 SDK docs.
All parameters verified against known outputs where possible.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class LensModel:
    name: str
    # Raw frame dimensions from camera
    frame_w: int
    frame_h: int
    fps_options: Tuple[int, ...]       # supported fps values
    # Fisheye geometry
    lens_fov_deg: float                # full FOV per lens
    projection: str                    # FFmpeg v360 projection type
    # Lens centres as fractions of frame dimensions
    lens0_cx_frac: float               # left lens centre x
    lens0_cy_frac: float               # left lens centre y
    lens1_cx_frac: float               # right lens centre x
    lens1_cy_frac: float               # right lens centre y
    lens_radius_frac: float            # fisheye radius as fraction of frame_h
    # Detection band (fraction of frame_h) - where the pitch typically appears
    pitch_band_top: float = 0.15
    pitch_band_bot: float = 0.85
    # FFmpeg stitch parameters
    ffmpeg_input_fmt: str = "dfisheye"
    # Output equirect resolution for processing (not final render)
    proc_equirect_w: int = 1920
    proc_equirect_h: int = 960
    # Notes
    notes: str = ""


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

LENS_MODELS = {

    # ------------------------------------------------------------------
    # Insta360 ONE X2 (2020)
    # Dual fisheye, ~210° FOV per lens, equidistant projection
    # Raw frame: 5760x2880 @ 30fps or 3840x1920 @ 60fps
    # ------------------------------------------------------------------
    "ONE X2": LensModel(
        name="ONE X2",
        frame_w=5760,
        frame_h=2880,
        fps_options=(30, 50),
        lens_fov_deg=210.0,
        projection="equidistant",
        lens0_cx_frac=0.25,
        lens0_cy_frac=0.50,
        lens1_cx_frac=0.75,
        lens1_cy_frac=0.50,
        lens_radius_frac=0.4861,       # ~1400px radius in 2880h frame
        pitch_band_top=0.10,
        pitch_band_bot=0.90,
        ffmpeg_input_fmt="dfisheye",
        proc_equirect_w=1920,
        proc_equirect_h=960,
        notes="5.7K/30fps primary mode. Also records 4K/60fps (3840x1920)."
    ),

    # 4K/60fps variant of the X2 - same lens, different resolution
    "ONE X2 4K": LensModel(
        name="ONE X2 4K",
        frame_w=3840,
        frame_h=1920,
        fps_options=(60,),
        lens_fov_deg=210.0,
        projection="equidistant",
        lens0_cx_frac=0.25,
        lens0_cy_frac=0.50,
        lens1_cx_frac=0.75,
        lens1_cy_frac=0.50,
        lens_radius_frac=0.4861,
        pitch_band_top=0.10,
        pitch_band_bot=0.90,
        ffmpeg_input_fmt="dfisheye",
        proc_equirect_w=1920,
        proc_equirect_h=960,
        notes="4K/60fps mode of ONE X2."
    ),

    # ------------------------------------------------------------------
    # Insta360 X3 (2022)
    # 72MP sensor, 5.7K/30fps or 4K/60fps
    # Slightly wider lens than X2
    # ------------------------------------------------------------------
    "X3": LensModel(
        name="X3",
        frame_w=5760,
        frame_h=2880,
        fps_options=(30, 50),
        lens_fov_deg=210.0,
        projection="equidistant",
        lens0_cx_frac=0.25,
        lens0_cy_frac=0.50,
        lens1_cx_frac=0.75,
        lens1_cy_frac=0.50,
        lens_radius_frac=0.4900,
        pitch_band_top=0.10,
        pitch_band_bot=0.90,
        ffmpeg_input_fmt="dfisheye",
        proc_equirect_w=1920,
        proc_equirect_h=960,
        notes="X3 lens geometry very close to X2. Parameters from Gyroflow DB."
    ),

    # ------------------------------------------------------------------
    # Insta360 X4 (2024)
    # 8K/30fps, 5.7K/60fps
    # ------------------------------------------------------------------
    "X4": LensModel(
        name="X4",
        frame_w=8064,
        frame_h=4032,
        fps_options=(30,),
        lens_fov_deg=210.0,
        projection="equidistant",
        lens0_cx_frac=0.25,
        lens0_cy_frac=0.50,
        lens1_cx_frac=0.75,
        lens1_cy_frac=0.50,
        lens_radius_frac=0.4861,
        pitch_band_top=0.10,
        pitch_band_bot=0.90,
        ffmpeg_input_fmt="dfisheye",
        proc_equirect_w=2560,
        proc_equirect_h=1280,
        notes="8K model. Use 5.7K/60fps mode for sport."
    ),

    # X4 5.7K/60fps mode
    "X4 5.7K": LensModel(
        name="X4 5.7K",
        frame_w=5760,
        frame_h=2880,
        fps_options=(60,),
        lens_fov_deg=210.0,
        projection="equidistant",
        lens0_cx_frac=0.25,
        lens0_cy_frac=0.50,
        lens1_cx_frac=0.75,
        lens1_cy_frac=0.50,
        lens_radius_frac=0.4861,
        pitch_band_top=0.10,
        pitch_band_bot=0.90,
        ffmpeg_input_fmt="dfisheye",
        proc_equirect_w=1920,
        proc_equirect_h=960,
        notes="X4 in 5.7K/60fps sport mode."
    ),

    # ------------------------------------------------------------------
    # Insta360 ONE RS 1-Inch 360 Edition (2022)
    # Leica co-engineered dual fisheye
    # ------------------------------------------------------------------
    "ONE RS 360": LensModel(
        name="ONE RS 360",
        frame_w=6080,
        frame_h=3040,
        fps_options=(30,),
        lens_fov_deg=210.0,
        projection="equidistant",
        lens0_cx_frac=0.25,
        lens0_cy_frac=0.50,
        lens1_cx_frac=0.75,
        lens1_cy_frac=0.50,
        lens_radius_frac=0.4800,
        pitch_band_top=0.10,
        pitch_band_bot=0.90,
        ffmpeg_input_fmt="dfisheye",
        proc_equirect_w=1920,
        proc_equirect_h=960,
        notes="1-Inch 360 lens module for ONE RS."
    ),

    # ------------------------------------------------------------------
    # Insta360 ONE X (original, 2018)
    # ------------------------------------------------------------------
    "ONE X": LensModel(
        name="ONE X",
        frame_w=5760,
        frame_h=2880,
        fps_options=(30, 50),
        lens_fov_deg=210.0,
        projection="equidistant",
        lens0_cx_frac=0.25,
        lens0_cy_frac=0.50,
        lens1_cx_frac=0.75,
        lens1_cy_frac=0.50,
        lens_radius_frac=0.4750,
        pitch_band_top=0.10,
        pitch_band_bot=0.90,
        ffmpeg_input_fmt="dfisheye",
        proc_equirect_w=1920,
        proc_equirect_h=960,
        notes="Original ONE X. Slightly smaller radius than X2."
    ),
}

# Aliases for common name variants users might type
ALIASES = {
    "x2":         "ONE X2",
    "onex2":      "ONE X2",
    "one x2":     "ONE X2",
    "insta360 x2":"ONE X2",
    "x3":         "X3",
    "insta360 x3":"X3",
    "x4":         "X4",
    "insta360 x4":"X4",
    "one rs":     "ONE RS 360",
    "one rs 360": "ONE RS 360",
    "onex":       "ONE X",
    "one x":      "ONE X",
}


def get_model(name: str) -> Optional[LensModel]:
    """
    Look up a lens model by name, case-insensitive, with alias resolution.
    Returns None if not found.
    """
    key = name.strip().lower()
    # Try alias table first
    canonical = ALIASES.get(key)
    if canonical:
        return LENS_MODELS.get(canonical)
    # Try direct lookup (case-insensitive)
    for k, v in LENS_MODELS.items():
        if k.lower() == key:
            return v
    return None


def detect_model_from_resolution(w: int, h: int, fps: float) -> Optional[LensModel]:
    """
    Given raw frame dimensions and fps, guess the most likely model.
    Used as fallback when user hasn't specified the model.
    """
    candidates = []
    for model in LENS_MODELS.values():
        if model.frame_w == w and model.frame_h == h:
            if round(fps) in model.fps_options:
                candidates.append(model)
            else:
                # Resolution matches but fps doesn't — lower priority
                candidates.append(model)

    if not candidates:
        return None
    # Prefer exact fps match
    exact = [c for c in candidates if round(fps) in c.fps_options]
    return exact[0] if exact else candidates[0]


def list_models() -> None:
    """Print a summary of all supported models."""
    print("\nSupported Insta360 models:")
    print("-" * 60)
    for name, m in LENS_MODELS.items():
        print(f"  {name:<20} {m.frame_w}x{m.frame_h}  "
              f"{'/'.join(str(f) for f in m.fps_options)}fps  "
              f"FOV={m.lens_fov_deg}°")
    print()


if __name__ == "__main__":
    list_models()
