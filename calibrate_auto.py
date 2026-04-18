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
  3. Pitch corner X — brightness scan outward from centre at far touchline Y
  4. Near touchline Y — local contrast scan down the halfway line (vertical white stripe)
  5. Ellipse construction — pitch boundary modelled as two half-ellipses sharing
     the same semi-major axis, with different semi-minor axes for top/bottom arcs
  6. Polygon output — sampled from the ellipse, saved as pitch.json

Usage:
    # Extract a frame first:
    ffmpeg -ss 00:04:00 -i /path/to/game.insv \\
        -vf "rotate=PI/2*3,v360=fisheye:equirect:ih_fov=200:iv_fov=200,scale=2880:1440" \\
        -frames:v 1 -update 1 /tmp/equirect_frame.png

    # Run auto-calibration:
    python3 calibrate_auto.py --input /tmp/equirect_frame.png --output calibration/pitch.json

    # Optional: visualise result
    python3 calibrate_auto.py --input /tmp/equirect_frame.png --output calibration/pitch.json --vis /tmp/calib_vis.jpg
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────
# Stage 1: Black mask
# ──────────────────────────────────────────────────────────────────

def compute_roi_mask(gray: np.ndarray) -> np.ndarray:
    """Return binary mask of non-black (non-camera-body) pixels."""
    black_mask = gray < 20
    roi = (~black_mask).astype(np.uint8) * 255
    return roi


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
    Find the far touchline Y position using variance-minimum-in-bright-zone.

    The far touchline sits at the boundary between the high-variance stand/tree
    zone above and the low-variance grass below. Within the bright zone (above
    70% of peak row-mean brightness), the far touchline appears as a local
    variance minimum — it's a narrow uniform stripe.

    Returns: Y coordinate in equirect frame.
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

    k = np.ones(smooth_k) / smooth_k
    vars_s  = np.convolve(vars_,  k, mode='same')
    means_s = np.convolve(means, k, mode='same')

    mean_threshold = np.max(means_s) * mean_frac
    search_mask = means_s > mean_threshold
    if search_mask.sum() < 5:
        # Fallback: use rough horizon
        return int(ys[len(ys) // 3])

    search_ys   = ys[search_mask]
    search_vars = vars_s[search_mask]
    return int(search_ys[np.argmin(search_vars)])


# ──────────────────────────────────────────────────────────────────
# Stage 3: Pitch corner X detection
# ──────────────────────────────────────────────────────────────────

def detect_pitch_corners_x(
    gray: np.ndarray,
    roi: np.ndarray,
    far_tl_y: int,
    band_rows: int = 20,
    smooth_k: int = 30,
    brightness_margin: float = 15.0,
) -> tuple[int, int]:
    """
    Find the left and right X extents of the pitch at the far touchline Y.

    Scans horizontally: the pitch interior is dark shadowed grass, the
    surrounds (stands, track, etc.) are brighter. Scanning outward from
    centre, we find where brightness rises above the interior level.

    Returns: (pitch_left_x, pitch_right_x)
    """
    h, w = gray.shape

    # Average over a band of rows for robustness
    y1 = max(0, far_tl_y - band_rows)
    y2 = min(h, far_tl_y + band_rows)
    row_band   = np.mean(gray[y1:y2, :].astype(float), axis=0)
    row_smooth = np.convolve(row_band, np.ones(smooth_k) / smooth_k, mode='same')
    row_roi    = roi[far_tl_y, :]

    cx = w // 2
    interior_brightness = float(np.mean(row_smooth[int(w*0.40):int(w*0.60)]))
    threshold = interior_brightness + brightness_margin

    # ROI bounds
    roi_cols  = np.where(row_roi > 0)[0]
    roi_left  = int(roi_cols.min()) if len(roi_cols) else 0
    roi_right = int(roi_cols.max()) if len(roi_cols) else w - 1

    # Left: scan outward from centre leftward
    pitch_left = roi_left
    for x in range(cx, roi_left, -1):
        if row_roi[x] == 0:
            pitch_left = x + smooth_k
            break
        if row_smooth[x] > threshold:
            pitch_left = x
            break

    # Right: scan outward from centre rightward
    pitch_right = roi_right
    for x in range(cx, roi_right):
        if row_roi[x] == 0:
            pitch_right = x - smooth_k
            break
        if row_smooth[x] > threshold:
            pitch_right = x
            break

    return int(pitch_left), int(pitch_right)


# ──────────────────────────────────────────────────────────────────
# Stage 4: Near touchline Y detection
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
    Find near touchline Y by scanning the halfway line's local contrast.

    The halfway line is a vertical white stripe at x ≈ w//2. Even in shadow
    it is brighter than the grass immediately to its left and right. We scan
    downward from the far touchline, tracking where the centre band is
    brighter than its surroundings. The lowest such row is the near touchline.

    Returns: Y coordinate of near touchline.
    """
    h = gray.shape[0]
    last_positive_y = far_tl_y

    for y in range(far_tl_y, h - 10):
        centre_val = float(np.mean(gray[y, centre_x - band_w:centre_x + band_w]))
        left_val   = float(np.mean(gray[y, centre_x - band_w - surround_offset:centre_x - band_w]))
        right_val  = float(np.mean(gray[y, centre_x + band_w:centre_x + band_w + surround_offset]))
        surround   = (left_val + right_val) / 2.0
        contrast   = centre_val - surround

        if contrast > contrast_threshold:
            last_positive_y = y

    return last_positive_y


# ──────────────────────────────────────────────────────────────────
# Stage 5 + 6: Build ellipse polygon
# ──────────────────────────────────────────────────────────────────

def build_pitch_polygon(
    far_tl_y: int,
    pitch_left: int,
    pitch_right: int,
    near_tl_y: int,
    n_top: int = 7,
    n_bottom: int = 11,
    semi_b_top_ratio: float = 0.068,
    scale: float = 1.0,
) -> list[list[int]]:
    """
    Build pitch boundary polygon from the four detected anchor values.

    The pitch boundary is modelled as two half-ellipses:
      - Top arc (far touchline): low semi-minor axis, curves slightly downward
        at corners (projection of a straight line at distance)
      - Bottom arc (near touchline): large semi-minor axis, pronounced curve

    Both arcs share the same semi-major axis (half the pitch width).

    Args:
        far_tl_y:          Far touchline Y at centre (lowest point of top arc)
        pitch_left:        Left pitch edge X
        pitch_right:       Right pitch edge X
        near_tl_y:         Near touchline Y at centre (lowest point of bottom arc)
        n_top:             Number of points on top arc (incl. corners)
        n_bottom:          Number of points on bottom arc (excl. shared corners)
        semi_b_top_ratio:  Ratio of top semi-minor to semi-major (controls corner droop)
        scale:             Overall scale factor applied to semi axes

    Returns: List of [x, y] polygon points (closed, clockwise from top-left).
    """
    ellipse_cx = (pitch_left + pitch_right) // 2
    semi_a     = (pitch_right - pitch_left) / 2 * scale
    semi_b     = (near_tl_y - far_tl_y) * scale
    semi_b_top = semi_a * semi_b_top_ratio

    # Top arc: left corner → far touchline → right corner
    # y = far_tl_y + semi_b_top * cos²(θ)
    # At θ=π/2 (centre): y = far_tl_y          (minimum, highest point)
    # At θ=0,π (corners): y = far_tl_y + semi_b_top (droop at edges)
    top_thetas = np.linspace(np.pi, 0, n_top)
    top_pts = [
        [int(ellipse_cx + semi_a * np.cos(t)),
         int(far_tl_y   + semi_b_top * (np.cos(t) ** 2))]
        for t in top_thetas
    ]

    # Bottom arc: right corner → near touchline → left corner
    # y = far_tl_y + semi_b * sin(θ)
    # At θ=0,π (corners): y = far_tl_y  (matches top arc corner Y before droop)
    # At θ=π/2 (centre):  y = far_tl_y + semi_b (deepest point)
    bot_thetas = np.linspace(0, np.pi, n_bottom)
    bot_pts = [
        [int(ellipse_cx + semi_a * np.cos(t)),
         int(far_tl_y   + semi_b   * np.sin(t))]
        for t in bot_thetas
    ]

    # Combine: top arc (left→right) + bottom arc interior (right→left)
    # Skip first and last bot_pts (they duplicate the top arc corners in X)
    full_poly = top_pts + bot_pts[1:-1]
    return [[int(p[0]), int(p[1])] for p in full_poly]


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

    # Pitch polygon
    pts = np.array(poly, dtype=np.int32)
    cv2.polylines(diag, [pts], True, (0, 255, 0), 3)
    for i, (px, py) in enumerate(poly):
        cv2.circle(diag, (px, py), 6, (0, 255, 0), -1)
        cv2.putText(diag, str(i + 1), (px + 8, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # Reference lines
    cv2.line(diag, (0, far_tl_y),  (w, far_tl_y),  (255, 255, 0), 1)
    cv2.line(diag, (0, near_tl_y), (w, near_tl_y), (0, 165, 255), 1)
    cv2.line(diag, (pitch_left,  0), (pitch_left,  h), (255, 165, 0), 1)
    cv2.line(diag, (pitch_right, 0), (pitch_right, h), (255, 165, 0), 1)

    # Optional ground truth
    if manual_gt:
        gt_pts = np.array(manual_gt, dtype=np.int32)
        cv2.polylines(diag, [gt_pts], True, (0, 0, 255), 2)

    legend = "Green=auto  Yellow=far TL  Orange=near TL"
    if manual_gt:
        legend += "  Red=manual GT"
    cv2.putText(diag, legend, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imwrite(output_path, diag)
    print(f"Visualisation saved to {output_path}")


# ──────────────────────────────────────────────────────────────────
# Main
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
    # Load
    img = cv2.imread(input_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {input_path}")
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

    # Stage 3: Pitch corners X
    pitch_left, pitch_right = detect_pitch_corners_x(gray, roi, far_tl_y)
    ellipse_cx = (pitch_left + pitch_right) // 2
    if verbose:
        print(f"Stage 3: Pitch corners X = {pitch_left} (left), {pitch_right} (right)")
        print(f"         Ellipse centre X = {ellipse_cx}  (frame centre = {w//2})")

    # Stage 4: Near touchline Y
    near_tl_y = detect_near_touchline_y(gray, far_tl_y, centre_x=w // 2)
    semi_b = near_tl_y - far_tl_y
    semi_a = (pitch_right - pitch_left) / 2
    if verbose:
        print(f"Stage 4: Near touchline Y = {near_tl_y}  (depth = {semi_b}px)")
        print(f"         Ellipse ratio b/a = {semi_b/semi_a:.3f}")

    # Stage 5+6: Build polygon
    poly = build_pitch_polygon(
        far_tl_y=far_tl_y,
        pitch_left=pitch_left,
        pitch_right=pitch_right,
        near_tl_y=near_tl_y,
        scale=scale,
    )
    if verbose:
        print(f"Stage 5: Polygon — {len(poly)} points")

    # Save JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    calib = {"pitch_polygon": poly}
    with open(output_path, 'w') as f:
        json.dump(calib, f, indent=2)
    if verbose:
        print(f"Saved: {output_path}")

    # Visualise
    if vis_path:
        visualise(img, poly, far_tl_y, near_tl_y, pitch_left, pitch_right, vis_path)

    return poly


def parse_args():
    p = argparse.ArgumentParser(description="Auto pitch calibration for Insta360 equirect frames")
    p.add_argument("--input",  required=True, help="Equirect frame (2880x1440 PNG/JPG)")
    p.add_argument("--output", default="calibration/pitch.json", help="Output JSON path")
    p.add_argument("--vis",    default=None,  help="Optional visualisation output path")
    p.add_argument("--scale",  type=float, default=1.0,
                   help="Scale factor for ellipse axes (default 1.0, increase slightly if polygon too small)")
    p.add_argument("--quiet",  action="store_true", help="Suppress progress output")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    poly = auto_calibrate(
        input_path=args.input,
        output_path=args.output,
        vis_path=args.vis,
        scale=args.scale,
        verbose=not args.quiet,
    )
    print("Done.")
