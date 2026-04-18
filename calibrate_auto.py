#!/usr/bin/env python3
"""
calibrate_auto.py
-----------------
Automatic pitch calibration for the ASN Pfeil Phönix Nürnberg analysis pipeline.

Replaces the manual click-based calibration tool. Given a single equirect frame
(2880x1440, produced by the ffmpeg fisheye→equirect conversion), automatically
detects the pitch boundary and writes calibration/pitch.json.

Algorithm:
  1. Black mask  — remove camera body occlusion → reliable ROI
  2. Far touchline Y — variance-minimum-in-bright-zone in the centre column band
  3. Pitch colour sampling — adaptive HSV sampling from multiple interior points
  4. Pitch corner X — HSV pitch mask scan (robust to shadow and surface type)
  5. Near touchline Y — local contrast scan down the halfway line
  6. Ellipse construction — pitch boundary modelled as two half-ellipses
  7. ROI clamping — all polygon points clamped within camera ROI
  8. Polygon output — saved as pitch.json

Tested on:
  - Natural grass, evening shadow (Nürnberg, July 2025)
  - Artificial turf, daylight (October 2024)

Usage:
    # Extract a frame first:
    ffmpeg -ss 00:04:00 -i /path/to/game.insv \\
        -vf "rotate=PI/2*3,v360=fisheye:equirect:ih_fov=200:iv_fov=200,scale=2880:1440" \\
        -frames:v 1 -update 1 /tmp/equirect_frame.png

    # Run auto-calibration:
    python3 calibrate_auto.py --input /tmp/equirect_frame.png --output calibration/pitch.json

    # With visualisation:
    python3 calibrate_auto.py --input /tmp/equirect_frame.png --output calibration/pitch.json --vis /tmp/calib_vis.jpg

    # If polygon is consistently too small:
    python3 calibrate_auto.py --input /tmp/equirect_frame.png --output calibration/pitch.json --vis /tmp/calib_vis.jpg --scale 1.02
"""

import argparse
import json
import os

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────
# Stage 1: Black mask
# ──────────────────────────────────────────────────────────────────

def compute_roi_mask(gray: np.ndarray) -> np.ndarray:
    """Return binary mask of non-black (non-camera-body) pixels."""
    return (~(gray < 20)).astype(np.uint8) * 255


# ──────────────────────────────────────────────────────────────────
# Stage 2: Far touchline Y detection
# ──────────────────────────────────────────────────────────────────

def detect_far_touchline_y(
    gray: np.ndarray,
    roi: np.ndarray,
    x_frac_left: float = 0.35,
    x_frac_right: float = 0.65,
    search_top: int = 550,
    search_bot: int = 950,
    smooth_k: int = 10,
    mean_frac: float = 0.70,
) -> int:
    """
    Find the far touchline Y using variance-minimum-in-bright-zone.

    The far touchline sits at the boundary between the high-variance
    stand/tree zone above and the low-variance grass below. Within the
    bright zone (above 70% of peak row-mean brightness), it appears as
    a local variance minimum — a narrow uniform stripe.
    """
    h, w = gray.shape
    cx_left  = int(w * x_frac_left)
    cx_right = int(w * x_frac_right)

    ys, vars_, means = [], [], []
    for y in range(search_top, min(search_bot, h)):
        strip = gray[y, cx_left:cx_right]
        if roi[y, (cx_left + cx_right) // 2] == 0:
            ys.append(y); vars_.append(0.0); means.append(0.0)
            continue
        ys.append(y)
        vars_.append(float(np.var(strip)))
        means.append(float(np.mean(strip)))

    ys    = np.array(ys)
    vars_ = np.array(vars_)
    means = np.array(means)
    k     = np.ones(smooth_k) / smooth_k
    vars_s  = np.convolve(vars_,  k, mode='same')
    means_s = np.convolve(means, k, mode='same')

    mean_threshold = np.max(means_s) * mean_frac
    search_mask = means_s > mean_threshold
    if search_mask.sum() < 5:
        return int(ys[len(ys) // 3])

    return int(ys[search_mask][np.argmin(vars_s[search_mask])])


# ──────────────────────────────────────────────────────────────────
# Stage 3: Pitch colour sampling
# ──────────────────────────────────────────────────────────────────

def sample_pitch_colour(
    hsv: np.ndarray,
    roi: np.ndarray,
    far_tl_y: int,
    patch_size: int = 20,
) -> tuple[int, int, int]:
    """
    Adaptively sample the pitch surface colour from multiple interior points.

    Samples from 6 points spread across the pitch interior avoiding the
    centre shadow band. Uses 75th percentile of V to avoid dark shadow
    pixels pulling the estimate too low.

    Returns: (hue, saturation, value) HSV of pitch surface.
    """
    h, w = hsv.shape[:2]
    depth = h - far_tl_y

    candidates = [
        (int(w * 0.35), int(far_tl_y + depth * 0.30)),
        (int(w * 0.65), int(far_tl_y + depth * 0.30)),
        (int(w * 0.30), int(far_tl_y + depth * 0.40)),
        (int(w * 0.70), int(far_tl_y + depth * 0.40)),
        (int(w * 0.40), int(far_tl_y + depth * 0.50)),
        (int(w * 0.60), int(far_tl_y + depth * 0.50)),
    ]

    best_v, best_hsv = -1, (40, 150, 80)
    for sx, sy in candidates:
        sy = min(sy, h - patch_size - 1)
        if roi[sy, sx] == 0:
            continue
        patch = hsv[sy - patch_size:sy + patch_size,
                    sx - patch_size:sx + patch_size]
        v  = int(np.percentile(patch[:, :, 2], 75))
        hh = int(np.median(patch[:, :, 0]))
        s  = int(np.median(patch[:, :, 1]))
        if v > best_v:
            best_v   = v
            best_hsv = (hh, s, v)

    return best_hsv


# ──────────────────────────────────────────────────────────────────
# Stage 4: Pitch corner X detection (HSV-based)
# ──────────────────────────────────────────────────────────────────

def detect_pitch_corners_x(
    hsv: np.ndarray,
    roi: np.ndarray,
    far_tl_y: int,
    pitch_hsv: tuple[int, int, int],
    scan_depth_frac: float = 0.25,
    h_tol: int = 20,
    s_tol: int = 80,
) -> tuple[int, int]:
    """
    Find pitch left/right X extents using HSV colour matching.

    Builds a pitch colour mask and finds the leftmost/rightmost pitch
    pixels at a scan Y below the far touchline. Robust to:
    - Natural grass in shadow (V is low but H/S are stable)
    - Artificial turf in daylight (bright, uniform colour)
    - Red running tracks (different H, excluded by hue tolerance)

    Falls back to ROI edges if pitch extends to the camera body.
    """
    h_img, w = hsv.shape[:2]
    pitch_h, pitch_s, _ = pitch_hsv

    # Build pitch colour mask — no V constraint to handle deep shadow
    pitch_mask = cv2.inRange(hsv,
        (max(0,   pitch_h - h_tol), max(0,   pitch_s - s_tol), 0),
        (min(179, pitch_h + h_tol), 255,                        255))
    pitch_mask = cv2.bitwise_and(pitch_mask, roi)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    pitch_mask = cv2.morphologyEx(pitch_mask, cv2.MORPH_CLOSE, k)
    pitch_mask = cv2.morphologyEx(pitch_mask, cv2.MORPH_OPEN,  k)

    # Scan at Y well below far touchline where pitch mask is reliable
    scan_y = int(far_tl_y + (h_img - far_tl_y) * scan_depth_frac)
    row_p  = pitch_mask[scan_y, :]
    row_r  = roi[scan_y, :]

    p_cols = np.where(row_p > 0)[0]
    r_cols = np.where(row_r > 0)[0]

    if len(p_cols) > 10:
        return int(p_cols.min()), int(p_cols.max())
    elif len(r_cols) > 0:
        # Fallback: pitch extends to camera body edges
        return int(r_cols.min()), int(r_cols.max())
    else:
        return int(w * 0.25), int(w * 0.75)


# ──────────────────────────────────────────────────────────────────
# Stage 5: Near touchline Y detection
# ──────────────────────────────────────────────────────────────────

def detect_near_touchline_y(
    gray: np.ndarray,
    far_tl_y: int,
    centre_x: int,
    band_w: int = 40,
    surround_offset: int = 60,
    contrast_threshold: float = 8.0,
) -> int:
    """
    Find near touchline Y via halfway line local contrast.

    The halfway line is a vertical white stripe at x ≈ w//2. Even in
    shadow and on artificial turf it is brighter than adjacent grass.
    The lowest row where centre brightness exceeds surroundings is the
    near touchline intersection with the halfway line.
    """
    h = gray.shape[0]
    last_positive_y = far_tl_y

    for y in range(far_tl_y, h - 10):
        c = float(np.mean(gray[y, centre_x - band_w:centre_x + band_w]))
        l = float(np.mean(gray[y, centre_x - band_w - surround_offset:centre_x - band_w]))
        r = float(np.mean(gray[y, centre_x + band_w:centre_x + band_w + surround_offset]))
        if c - (l + r) / 2.0 > contrast_threshold:
            last_positive_y = y

    return last_positive_y


# ──────────────────────────────────────────────────────────────────
# Stage 6 + 7: Build ellipse polygon with ROI clamping
# ──────────────────────────────────────────────────────────────────

def build_pitch_polygon(
    far_tl_y: int,
    pitch_left: int,
    pitch_right: int,
    near_tl_y: int,
    roi: np.ndarray,
    n_top: int = 7,
    n_bottom: int = 11,
    semi_b_top_ratio: float = 0.068,
    scale: float = 1.0,
) -> list[list[int]]:
    """
    Build pitch boundary polygon from the four detected anchor values.

    Models the pitch as two half-ellipses sharing the same semi-major axis:
      - Top arc (far touchline): slight downward droop at corners due to
        projection geometry of a straight line viewed at an angle
      - Bottom arc (near touchline): pronounced curve

    All points are clamped to stay within the camera ROI.
    """
    h, w = roi.shape[:2]
    ellipse_cx = (pitch_left + pitch_right) // 2
    semi_a     = (pitch_right - pitch_left) / 2 * scale
    semi_b     = (near_tl_y - far_tl_y) * scale
    semi_b_top = semi_a * semi_b_top_ratio

    # Top arc: left corner → centre → right corner
    # y = far_tl_y + semi_b_top * cos²(θ)
    top_thetas = np.linspace(np.pi, 0, n_top)
    top_pts = [
        [int(ellipse_cx + semi_a * np.cos(t)),
         int(far_tl_y   + semi_b_top * (np.cos(t) ** 2))]
        for t in top_thetas
    ]

    # Bottom arc: right corner → near touchline bottom → left corner
    # y = far_tl_y + semi_b * sin(θ)
    bot_thetas = np.linspace(0, np.pi, n_bottom)
    bot_pts = [
        [int(ellipse_cx + semi_a * np.cos(t)),
         int(far_tl_y   + semi_b   * np.sin(t))]
        for t in bot_thetas
    ]

    # Combine: top arc + bottom arc interior (skip shared corner points)
    full_poly = top_pts + bot_pts[1:-1]

    # Clamp all points to ROI — no point should land in the black camera body
    clamped = []
    for px, py in full_poly:
        py_c = int(np.clip(py, 0, h - 1))
        px_c = int(np.clip(px, 0, w - 1))

        if roi[py_c, px_c] == 0:
            # Scan inward toward frame centre to find ROI edge
            direction = 1 if px_c < w // 2 else -1
            for dx in range(0, w // 2, 2):
                nx = px_c + direction * dx
                if 0 <= nx < w and roi[py_c, nx] > 0:
                    px_c = nx
                    break

        clamped.append([px_c, py_c])

    return clamped


# ──────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────

def visualise(
    img: np.ndarray,
    poly: list[list[int]],
    far_tl_y: int,
    near_tl_y: int,
    pitch_left: int,
    pitch_right: int,
    output_path: str,
    manual_gt: list[list[int]] | None = None,
) -> None:
    diag = img.copy()
    h, w = img.shape[:2]

    pts = np.array(poly, dtype=np.int32)
    cv2.polylines(diag, [pts], True, (0, 255, 0), 3)
    for i, (px, py) in enumerate(poly):
        cv2.circle(diag, (px, py), 6, (0, 255, 0), -1)
        cv2.putText(diag, str(i + 1), (px + 8, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    cv2.line(diag, (0, far_tl_y),  (w, far_tl_y),  (255, 255, 0), 1)
    cv2.line(diag, (0, near_tl_y), (w, near_tl_y), (0, 165, 255), 1)
    cv2.line(diag, (pitch_left,  0), (pitch_left,  h), (255, 165, 0), 1)
    cv2.line(diag, (pitch_right, 0), (pitch_right, h), (255, 165, 0), 1)

    if manual_gt:
        cv2.polylines(diag, [np.array(manual_gt, dtype=np.int32)], True, (0, 0, 255), 2)

    legend = "Green=auto  Yellow=far TL  Orange=near TL"
    if manual_gt:
        legend += "  Red=manual GT"
    cv2.putText(diag, legend, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imwrite(output_path, diag)
    print(f"Visualisation saved to {output_path}")


# ──────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────

def auto_calibrate(
    input_path: str,
    output_path: str,
    vis_path: str | None = None,
    scale: float = 1.0,
    verbose: bool = True,
) -> list[list[int]]:
    """
    Run full auto-calibration pipeline.
    Returns the pitch polygon as a list of [x, y] points.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {input_path}")
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if verbose:
        print(f"Input: {input_path}  ({w}x{h})")

    # Stage 1: ROI
    roi = compute_roi_mask(gray)
    if verbose:
        black_pct = 100 * (gray < 20).sum() / (h * w)
        print(f"Stage 1: Black mask — {black_pct:.1f}% of frame is camera body")

    # Stage 2: Far touchline Y
    far_tl_y = detect_far_touchline_y(gray, roi)
    if verbose:
        print(f"Stage 2: Far touchline Y = {far_tl_y}")

    # Stage 3: Pitch colour
    pitch_hsv = sample_pitch_colour(hsv, roi, far_tl_y)
    if verbose:
        print(f"Stage 3: Pitch colour HSV = {pitch_hsv}")

    # Stage 4: Pitch corners X
    pitch_left, pitch_right = detect_pitch_corners_x(hsv, roi, far_tl_y, pitch_hsv)
    ellipse_cx = (pitch_left + pitch_right) // 2
    if verbose:
        print(f"Stage 4: Pitch corners X = {pitch_left} (left), {pitch_right} (right)")
        print(f"         Ellipse centre X = {ellipse_cx}  (frame centre = {w//2})")

    # Stage 5: Near touchline Y
    near_tl_y = detect_near_touchline_y(gray, far_tl_y, centre_x=w // 2)
    semi_b = near_tl_y - far_tl_y
    semi_a = (pitch_right - pitch_left) / 2
    if verbose:
        print(f"Stage 5: Near touchline Y = {near_tl_y}  (depth = {semi_b}px)")
        print(f"         Ellipse ratio b/a = {semi_b/semi_a:.3f}")

    # Stage 6+7: Build polygon with ROI clamping
    poly = build_pitch_polygon(
        far_tl_y=far_tl_y,
        pitch_left=pitch_left,
        pitch_right=pitch_right,
        near_tl_y=near_tl_y,
        roi=roi,
        scale=scale,
    )
    if verbose:
        print(f"Stage 6: Polygon — {len(poly)} points (ROI-clamped)")

    # Save JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    calib = {"pitch_polygon": poly}
    with open(output_path, 'w') as f:
        json.dump(calib, f, indent=2)
    if verbose:
        print(f"Saved: {output_path}")

    if vis_path:
        visualise(img, poly, far_tl_y, near_tl_y, pitch_left, pitch_right, vis_path)

    return poly


def parse_args():
    p = argparse.ArgumentParser(
        description="Auto pitch calibration for Insta360 equirect frames")
    p.add_argument("--input",  required=True,
                   help="Equirect frame (2880x1440 PNG/JPG)")
    p.add_argument("--output", default="calibration/pitch.json",
                   help="Output JSON path")
    p.add_argument("--vis",    default=None,
                   help="Optional visualisation output path")
    p.add_argument("--scale",  type=float, default=1.0,
                   help="Scale factor for ellipse axes (increase if polygon too small)")
    p.add_argument("--quiet",  action="store_true",
                   help="Suppress progress output")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    auto_calibrate(
        input_path=args.input,
        output_path=args.output,
        vis_path=args.vis,
        scale=args.scale,
        verbose=not args.quiet,
    )
    print("Done.")
