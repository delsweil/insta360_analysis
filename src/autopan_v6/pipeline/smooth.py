# pipeline/smooth.py
"""
Smoothing phase: converts a sparse yaw schedule (one entry per N frames)
into a dense, smooth per-frame yaw curve.

Why this matters:
  Raw detections are noisy - the ball flickers, players move, detections
  drop out. Directly applying raw yaw angles produces unwatchable jitter.

  The v5 approach used per-frame EMA smoothing (target_alpha=0.12) which
  works but has two problems:
    1. It lags behind fast action (the pan is always "catching up")
    2. It can't look ahead - a ball that moves right then immediately left
       causes unnecessary panning

  The v6 approach:
    1. Collect ALL detections for a segment first (the sparse schedule)
    2. Fill gaps with interpolation
    3. Apply Savitzky-Golay filter over the whole sequence
       - Smooths noise while preserving genuine fast motion
       - Can be tuned: wider window = smoother but more lag
    4. Apply velocity clamping to prevent jarring camera jumps
    5. Clamp to pitch boundary yaw range

The result feels more like a human camera operator: smooth, anticipatory,
and not constantly chasing small ball movements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from .detect import ScheduleEntry


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SmoothConfig:
    # Savitzky-Golay parameters
    # window_length: must be odd, in frames. Larger = smoother.
    #   At 30fps, window=61 = ~2 seconds of smoothing
    #   At 60fps, window=121 = ~2 seconds of smoothing
    sg_window_sec: float = 2.0      # window in seconds (converted to frames)
    sg_polyorder: int = 2            # polynomial order (2=quadratic is good)

    # Velocity clamping: max degrees per frame the camera can pan
    # At 30fps: 1.5 deg/frame = 45 deg/sec (reasonable for football)
    max_deg_per_frame: float = 1.5

    # Gap filling: how to interpolate between sparse detections
    # 'linear' | 'cubic' | 'previous'
    interp_method: str = "cubic"

    # After smoothing, apply a final light EMA to remove any remaining
    # high-frequency noise from the SG filter boundaries
    final_ema_alpha: float = 0.0    # 0.0 = disabled, 0.05 = light

    # Yaw centre (degrees): where the camera rests when nothing detected
    yaw_centre: float = 0.0

    # Hard clamp range (degrees from centre)
    yaw_max_dev: float = 50.0


# ---------------------------------------------------------------------------
# Core smoothing
# ---------------------------------------------------------------------------

def smooth_schedule(
    entries: List[ScheduleEntry],
    total_frames: int,
    fps: float,
    cfg: SmoothConfig,
) -> np.ndarray:
    """
    Convert a sparse list of ScheduleEntry into a dense per-frame yaw array.

    Args:
        entries      : sparse detection schedule (one entry per N frames)
        total_frames : total number of frames in the video
        fps          : video fps (used for window_length calculation)
        cfg          : SmoothConfig

    Returns:
        yaw_curve: np.ndarray of shape (total_frames,), dtype float32
                   Per-frame yaw angles in degrees.
    """
    if not entries:
        return np.full(total_frames, cfg.yaw_centre, dtype=np.float32)

    # --- Step 1: Extract sparse (frame_idx, yaw) pairs ---
    frame_idxs = np.array([e.frame_idx for e in entries], dtype=np.float64)
    yaws = np.array([e.target_yaw for e in entries], dtype=np.float64)

    # Ensure sorted
    order = np.argsort(frame_idxs)
    frame_idxs = frame_idxs[order]
    yaws = yaws[order]

    # --- Step 2: Unwrap yaw to remove discontinuities at ±180° ---
    # (Shouldn't happen often for football since yaw stays in a small range,
    #  but important for robustness)
    yaws_unwrapped = _unwrap_yaw(yaws)

    # --- Step 3: Interpolate to dense frame indices ---
    all_frames = np.arange(total_frames, dtype=np.float64)

    # Clamp query range to detected range (extrapolate beyond with nearest)
    if cfg.interp_method == "previous":
        # Step function: hold last known value
        dense = np.interp(all_frames, frame_idxs, yaws_unwrapped)
    else:
        try:
            interp_fn = interp1d(
                frame_idxs,
                yaws_unwrapped,
                kind=cfg.interp_method,
                bounds_error=False,
                fill_value=(yaws_unwrapped[0], yaws_unwrapped[-1]),
            )
            dense = interp_fn(all_frames)
        except Exception:
            # Fall back to linear if cubic fails (too few points)
            dense = np.interp(all_frames, frame_idxs, yaws_unwrapped)

    # --- Step 4: Savitzky-Golay smoothing ---
    window_frames = int(cfg.sg_window_sec * fps)
    # Must be odd and at least sg_polyorder+2
    window_frames = max(window_frames, cfg.sg_polyorder + 2)
    if window_frames % 2 == 0:
        window_frames += 1
    # Can't exceed total frames
    window_frames = min(window_frames, total_frames if total_frames % 2 == 1 else total_frames - 1)

    if len(dense) >= window_frames:
        smoothed = savgol_filter(
            dense,
            window_length=window_frames,
            polyorder=cfg.sg_polyorder,
            mode="nearest",   # 'nearest' avoids ringing at boundaries
        )
    else:
        smoothed = dense.copy()

    # --- Step 5: Optional final EMA ---
    if cfg.final_ema_alpha > 0:
        smoothed = _apply_ema(smoothed, cfg.final_ema_alpha)

    # --- Step 6: Velocity clamping ---
    smoothed = _clamp_velocity(smoothed, cfg.max_deg_per_frame)

    # --- Step 7: Hard yaw clamp ---
    lo = cfg.yaw_centre - cfg.yaw_max_dev
    hi = cfg.yaw_centre + cfg.yaw_max_dev
    smoothed = np.clip(smoothed, lo, hi)

    # Re-wrap to [-180, 180]
    smoothed = ((smoothed + 180) % 360) - 180

    return smoothed.astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap_yaw(yaws: np.ndarray) -> np.ndarray:
    """
    Unwrap yaw angles to remove jumps > 180° between consecutive samples.
    Similar to np.unwrap but for degrees.
    """
    unwrapped = yaws.copy()
    for i in range(1, len(unwrapped)):
        diff = unwrapped[i] - unwrapped[i-1]
        if diff > 180:
            unwrapped[i:] -= 360
        elif diff < -180:
            unwrapped[i:] += 360
    return unwrapped


def _apply_ema(arr: np.ndarray, alpha: float) -> np.ndarray:
    """Forward EMA pass."""
    out = arr.copy()
    for i in range(1, len(out)):
        out[i] = (1 - alpha) * out[i-1] + alpha * out[i]
    return out


def _clamp_velocity(yaw_curve: np.ndarray, max_deg_per_frame: float) -> np.ndarray:
    """
    Clamp frame-to-frame yaw change to max_deg_per_frame.
    This prevents the smoothed curve from having sharp jumps
    at the boundaries of the SG filter or at mode transitions.
    """
    out = yaw_curve.copy()
    for i in range(1, len(out)):
        delta = out[i] - out[i-1]
        if abs(delta) > max_deg_per_frame:
            out[i] = out[i-1] + np.sign(delta) * max_deg_per_frame
    return out


# ---------------------------------------------------------------------------
# Dense yaw curve -> per-frame pitch angle
# ---------------------------------------------------------------------------

def compute_pitch_curve(
    n_frames: int,
    pitch_centre: float = 0.0,
    pitch_amplitude: float = 0.0,
) -> np.ndarray:
    """
    Compute a per-frame pitch angle array.

    For now this is static (constant pitch), but could be extended to
    track vertical position of the action.

    Args:
        n_frames        : total frames
        pitch_centre    : centre pitch angle (degrees, 0 = horizon)
        pitch_amplitude : unused for now

    Returns:
        pitch_curve: np.ndarray of shape (n_frames,), dtype float32
    """
    return np.full(n_frames, pitch_centre, dtype=np.float32)


# ---------------------------------------------------------------------------
# Save / load dense curve
# ---------------------------------------------------------------------------

def save_dense_curve(yaw_curve: np.ndarray, pitch_curve: np.ndarray, path: str) -> None:
    """Save dense yaw+pitch curves as a .npz file."""
    np.savez_compressed(path, yaw=yaw_curve, pitch=pitch_curve)
    print(f"Dense curve saved: {path}  ({len(yaw_curve)} frames)")


def load_dense_curve(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load dense yaw+pitch curves from .npz file."""
    data = np.load(path)
    return data["yaw"], data["pitch"]


def save_curve_csv(yaw_curve: np.ndarray, pitch_curve: np.ndarray, path: str) -> None:
    """
    Save dense curve as CSV for inspection / FFmpeg sendcmd.
    Format: frame_idx, yaw_deg, pitch_deg
    """
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "yaw_deg", "pitch_deg"])
        for i, (y, p) in enumerate(zip(yaw_curve, pitch_curve)):
            writer.writerow([i, f"{y:.4f}", f"{p:.4f}"])
    print(f"Dense curve CSV saved: {path}")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def curve_stats(yaw_curve: np.ndarray, fps: float) -> str:
    """Return a human-readable summary of the yaw curve."""
    delta = np.diff(yaw_curve)
    lines = [
        f"  Yaw range  : {yaw_curve.min():.1f}° to {yaw_curve.max():.1f}°",
        f"  Yaw mean   : {yaw_curve.mean():.1f}°  std={yaw_curve.std():.1f}°",
        f"  Max speed  : {np.abs(delta).max():.2f}°/frame  "
        f"({np.abs(delta).max() * fps:.1f}°/sec)",
        f"  Mean speed : {np.abs(delta).mean():.3f}°/frame  "
        f"({np.abs(delta).mean() * fps:.1f}°/sec)",
    ]
    return "\n".join(lines)
