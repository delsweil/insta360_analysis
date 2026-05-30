#!/usr/bin/env python3
"""
compare_pan.py
--------------
Compare Autopan predicted pan curve against Insta360 Studio ground truth.

Usage:
    python compare_pan.py \
        --insprj /path/to/VID_20241028_160117_00_004.insv.insprj \
        --predicted pan_log.csv \
        [--start-times 100,250,450,650,850] \
        [--seg-duration 15] \
        [--out comparison.png]

The predicted CSV must have columns: timestamp_s, predicted_pan_deg
(produced by autopan_infer.py --log-csv pan_log.csv)
"""

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import PchipInterpolator


# ---------------------------------------------------------------------------
# Ground truth parsing
# ---------------------------------------------------------------------------

def load_ground_truth(insprj_path: str) -> PchipInterpolator:
    """Parse keyframes from .insprj XML and return a PchipInterpolator
    that maps time_s -> pan_deg over the full clip."""
    tree = ET.parse(insprj_path)
    root = tree.getroot()

    keyframes = root.findall(".//recording/keyframes/keyframe")
    if not keyframes:
        raise ValueError(f"No keyframes found in {insprj_path}")

    times_s = []
    pans_deg = []
    for kf in keyframes:
        time_ms = float(kf.attrib["time"])
        pan_rad = float(kf.attrib["pan"])
        times_s.append(time_ms / 1000.0)
        pans_deg.append(math.degrees(pan_rad))

    # Sort by time (should already be sorted, but be safe)
    pairs = sorted(zip(times_s, pans_deg))
    times_s, pans_deg = zip(*pairs)

    interp = PchipInterpolator(times_s, pans_deg, extrapolate=False)
    print(f"[GT] Loaded {len(times_s)} keyframes, "
          f"clip span {times_s[0]:.1f}s – {times_s[-1]:.1f}s")
    print(f"[GT] Pan range: {min(pans_deg):.1f}° – {max(pans_deg):.1f}°")
    return interp, times_s[0], times_s[-1]


# ---------------------------------------------------------------------------
# Predicted CSV loading
# ---------------------------------------------------------------------------

def load_predicted(csv_path: str) -> pd.DataFrame:
    """Load the per-frame pan log written by autopan_infer.py --log-csv."""
    df = pd.read_csv(csv_path)
    required = {"timestamp_s", "predicted_pan_deg"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns {required}, got {list(df.columns)}")
    df = df.sort_values("timestamp_s").reset_index(drop=True)
    print(f"[Pred] Loaded {len(df)} frames, "
          f"time span {df.timestamp_s.min():.1f}s – {df.timestamp_s.max():.1f}s")
    print(f"[Pred] Pan range: {df.predicted_pan_deg.min():.1f}° – "
          f"{df.predicted_pan_deg.max():.1f}°")
    return df


# ---------------------------------------------------------------------------
# Segment helpers
# ---------------------------------------------------------------------------

def parse_start_times(s: str):
    """'100,250,450' -> [100.0, 250.0, 450.0]"""
    return [float(x.strip()) for x in s.split(",")]


def get_segment_mask(df: pd.DataFrame, start_times, seg_duration: float):
    """Return a boolean mask selecting only rows within any segment window."""
    mask = pd.Series(False, index=df.index)
    for t0 in start_times:
        t1 = t0 + seg_duration
        mask |= (df.timestamp_s >= t0) & (df.timestamp_s < t1)
    return mask


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(errors ** 2)))


def compute_mae(errors: np.ndarray) -> float:
    return float(np.mean(np.abs(errors)))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

SEGMENT_COLOURS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"
]


def plot_comparison(df_pred: pd.DataFrame,
                    gt_interp: PchipInterpolator,
                    start_times,
                    seg_duration: float,
                    out_path: str):

    # Sample GT on a fine grid over the predicted time range
    t_min = df_pred.timestamp_s.min()
    t_max = df_pred.timestamp_s.max()
    t_fine = np.linspace(t_min, t_max, 4000)
    gt_fine = gt_interp(t_fine)

    # GT sampled at exactly the predicted timestamps
    gt_at_pred = gt_interp(df_pred.timestamp_s.values)
    errors = df_pred.predicted_pan_deg.values - gt_at_pred
    valid = ~np.isnan(gt_at_pred)

    overall_rmse = compute_rmse(errors[valid])
    overall_mae = compute_mae(errors[valid])

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Autopan: Predicted vs Ground Truth Pan", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[3, 1.5, 1.2], hspace=0.45)

    ax_pan = fig.add_subplot(gs[0])
    ax_err = fig.add_subplot(gs[1], sharex=ax_pan)
    ax_bar = fig.add_subplot(gs[2])

    # ---- Pan curves ----
    ax_pan.plot(t_fine, gt_fine, color="steelblue", lw=1.5,
                label="Ground truth (Studio)", zorder=2)
    ax_pan.plot(df_pred.timestamp_s, df_pred.predicted_pan_deg,
                color="tomato", lw=1.2, alpha=0.85,
                label="Autopan predicted", zorder=3)

    # Shade each segment
    for i, t0 in enumerate(start_times):
        col = SEGMENT_COLOURS[i % len(SEGMENT_COLOURS)]
        ax_pan.axvspan(t0, t0 + seg_duration, alpha=0.08, color=col, zorder=1)
        ax_err.axvspan(t0, t0 + seg_duration, alpha=0.08, color=col, zorder=1)
        ax_pan.axvline(t0, color=col, lw=0.8, ls="--", alpha=0.6)
        ax_err.axvline(t0, color=col, lw=0.8, ls="--", alpha=0.6)
        ax_pan.text(t0 + 0.5, ax_pan.get_ylim()[0] if True else -65,
                    f"t={int(t0)}s", fontsize=7, color=col, va="bottom")

    ax_pan.set_ylabel("Pan (degrees)")
    ax_pan.legend(loc="upper right", fontsize=9)
    ax_pan.grid(True, alpha=0.3)
    ax_pan.set_title(f"Pan curves  |  RMSE={overall_rmse:.2f}°  MAE={overall_mae:.2f}°",
                     fontsize=10)

    # ---- Error curve ----
    ax_err.axhline(0, color="gray", lw=0.8)
    ax_err.fill_between(df_pred.timestamp_s[valid],
                        errors[valid], 0,
                        where=errors[valid] > 0,
                        color="tomato", alpha=0.4, label="Pred > GT")
    ax_err.fill_between(df_pred.timestamp_s[valid],
                        errors[valid], 0,
                        where=errors[valid] < 0,
                        color="steelblue", alpha=0.4, label="Pred < GT")
    ax_err.set_ylabel("Error (°)")
    ax_err.set_xlabel("Clip time (s)")
    ax_err.legend(loc="upper right", fontsize=8)
    ax_err.grid(True, alpha=0.3)
    ax_err.set_title("Prediction error (predicted − ground truth)", fontsize=10)

    # ---- Per-segment bar chart ----
    seg_rmses = []
    seg_maes = []
    seg_labels = []
    for t0 in start_times:
        mask = (df_pred.timestamp_s >= t0) & (df_pred.timestamp_s < t0 + seg_duration)
        seg_gt = gt_interp(df_pred.timestamp_s[mask].values)
        seg_err = df_pred.predicted_pan_deg[mask].values - seg_gt
        seg_valid = ~np.isnan(seg_gt)
        if seg_valid.sum() > 0:
            seg_rmses.append(compute_rmse(seg_err[seg_valid]))
            seg_maes.append(compute_mae(seg_err[seg_valid]))
        else:
            seg_rmses.append(float("nan"))
            seg_maes.append(float("nan"))
        seg_labels.append(f"t={int(t0)}s")

    x = np.arange(len(start_times))
    w = 0.35
    bars1 = ax_bar.bar(x - w/2, seg_rmses, w, label="RMSE", color="tomato", alpha=0.8)
    bars2 = ax_bar.bar(x + w/2, seg_maes, w, label="MAE", color="steelblue", alpha=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(seg_labels)
    ax_bar.set_ylabel("Error (°)")
    ax_bar.set_title("Per-segment error", fontsize=10)
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, alpha=0.3, axis="y")

    # Label bars
    for bar in bars1:
        h = bar.get_height()
        if not math.isnan(h):
            ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                        f"{h:.1f}°", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        if not math.isnan(h):
            ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                        f"{h:.1f}°", ha="center", va="bottom", fontsize=7)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[Plot] Saved to {out_path}")

    # Print summary table
    print("\n" + "="*55)
    print(f"{'Segment':<12} {'RMSE (°)':>10} {'MAE (°)':>10}")
    print("-"*55)
    for label, rmse, mae in zip(seg_labels, seg_rmses, seg_maes):
        print(f"{label:<12} {rmse:>10.2f} {mae:>10.2f}")
    print("-"*55)
    print(f"{'OVERALL':<12} {overall_rmse:>10.2f} {overall_mae:>10.2f}")
    print("="*55)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--insprj", required=True,
                        help="Path to .insv.insprj file")
    parser.add_argument("--predicted", required=True,
                        help="CSV produced by autopan_infer.py --log-csv")
    parser.add_argument("--start-times", default=None,
                        help="Comma-separated segment start times (auto-detected from CSV if not provided)")
    parser.add_argument("--seg-duration", type=float, default=15.0,
                        help="Duration of each segment in seconds (default: 15)")
    parser.add_argument("--out", default="pan_comparison.png",
                        help="Output plot filename (default: pan_comparison.png)")
    args = parser.parse_args()

    gt_interp, gt_t0, gt_t1 = load_ground_truth(args.insprj)
    df_pred = load_predicted(args.predicted)

    if args.start_times:
        start_times = parse_start_times(args.start_times)
    else:
        import numpy as _np
        ts = df_pred['timestamp_s'].values
        diffs = _np.diff(ts)
        gap_indices = _np.where(diffs > args.seg_duration * 0.5)[0]
        start_times = [float(ts[0])] + [float(ts[i+1]) for i in gap_indices]
        print(f'[Auto] Detected {len(start_times)} segments at: '
              f"{', '.join(f'{t:.0f}s' for t in start_times)}")

    # Warn if predicted timestamps fall outside GT range
    pred_t0 = df_pred.timestamp_s.min()
    pred_t1 = df_pred.timestamp_s.max()
    if pred_t0 < gt_t0 or pred_t1 > gt_t1:
        print(f"[WARN] Predicted range {pred_t0:.1f}–{pred_t1:.1f}s "
              f"extends beyond GT keyframe range {gt_t0:.1f}–{gt_t1:.1f}s. "
              f"Extrapolated GT values will be NaN.")

    plot_comparison(df_pred, gt_interp, start_times, args.seg_duration, args.out)


if __name__ == "__main__":
    main()
