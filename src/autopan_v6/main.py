#!/usr/bin/env python3
# main.py
"""
autopan v6 - Insta360 football autopan pipeline

Usage examples:

  # Full pipeline (detect + smooth + render):
  python main.py run path/to/game_folder/ --model "ONE X2" --output output/game.mp4

  # Step-by-step:
  python main.py detect path/to/game_folder/ --output data/yaw_schedule.csv
  python main.py smooth data/yaw_schedule.csv --fps 30 --frames 162000 --output data/yaw_curve.npz
  python main.py render path/to/game_folder/ data/yaw_curve.npz --output output/game.mp4

  # Calibrate pitch boundary:
  python main.py calibrate path/to/segment.insv

  # Probe files (no processing):
  python main.py probe path/to/game_folder/

  # List supported camera models:
  python main.py models
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_probe(args):
    from pipeline.probe import probe_game, find_insv_files, estimate_processing_time

    target = args.target
    if os.path.isdir(target):
        files = find_insv_files(target)
        if not files:
            # Also look for .mp4
            files = sorted(Path(target).glob("*.mp4")) + sorted(Path(target).glob("*.MP4"))
            files = [str(f) for f in files]
        if not files:
            print(f"No .insv or .mp4 files found in {target}")
            sys.exit(1)
    else:
        files = [target]

    game = probe_game(files, model_hint=args.model)
    print(game.summary())
    print(estimate_processing_time(
        game,
        sample_every_n=args.sample_every,
        cpu_only=True,
    ))


def cmd_calibrate(args):
    from pipeline.calibrate import calibrate
    calibrate(
        video_path=args.video,
        output_json=args.output,
        frame_index=args.frame,
        model_hint=args.model,
    )


def cmd_detect(args):
    from pipeline.probe import probe_game, find_insv_files
    from pipeline.detect import DetectionConfig, run_detection
    from pipeline.calibrate import load_calibration

    # Gather files
    target = args.target
    if os.path.isdir(target):
        files = find_insv_files(target)
        if not files:
            files = [str(f) for f in sorted(Path(target).glob("*.mp4"))]
    else:
        files = [target]

    if not files:
        print(f"No video files found at: {target}")
        sys.exit(1)

    game = probe_game(files, model_hint=args.model)
    print(game.summary())

    # Load calibration if available
    calib = load_calibration(args.calibration)
    pitch_yaw_left  = calib["pitch_yaw_left_deg"]  if calib else None
    pitch_yaw_right = calib["pitch_yaw_right_deg"] if calib else None
    yaw_centre      = calib["pitch_yaw_centre_deg"] if calib else 0.0

    cfg = DetectionConfig(
        sample_every_n=args.sample_every,
        detection_width=args.detection_width,
        player_model_path=args.player_model,
        ball_model_path=args.ball_model,
        pitch_yaw_left=pitch_yaw_left,
        pitch_yaw_right=pitch_yaw_right,
    )

    output_csv = args.output or "data/yaw_schedule.csv"
    run_detection(game, cfg, output_csv)


def cmd_smooth(args):
    from pipeline.detect import load_schedule_csv
    from pipeline.smooth import (
        SmoothConfig, smooth_schedule, compute_pitch_curve,
        save_dense_curve, curve_stats
    )

    entries = load_schedule_csv(args.schedule_csv)
    print(f"Loaded {len(entries)} schedule entries from {args.schedule_csv}")

    cfg = SmoothConfig(
        sg_window_sec=args.window_sec,
        max_deg_per_frame=args.max_speed,
        yaw_centre=args.yaw_centre,
        yaw_max_dev=args.yaw_max_dev,
    )

    print(f"Smoothing: window={args.window_sec}s  max_speed={args.max_speed}°/frame")
    yaw_curve = smooth_schedule(entries, args.total_frames, args.fps, cfg)
    pitch_curve = compute_pitch_curve(args.total_frames, pitch_centre=args.pitch_centre)

    print(curve_stats(yaw_curve, args.fps))

    output = args.output or "data/yaw_curve.npz"
    save_dense_curve(yaw_curve, pitch_curve, output)

    # Also save CSV for inspection
    csv_path = output.replace(".npz", "_preview.csv")
    from pipeline.smooth import save_curve_csv
    save_curve_csv(yaw_curve, pitch_curve, csv_path)


def cmd_render(args):
    from pipeline.probe import probe_game, find_insv_files
    from pipeline.smooth import load_dense_curve
    from pipeline.render import RenderConfig, render_game, render_segments_concat

    # Gather files
    target = args.target
    if os.path.isdir(target):
        files = find_insv_files(target)
        if not files:
            files = [str(f) for f in sorted(Path(target).glob("*.mp4"))]
    else:
        files = [target]

    game = probe_game(files, model_hint=args.model)
    yaw_curve, pitch_curve = load_dense_curve(args.curve_npz)

    cfg = RenderConfig(
        out_w=args.width,
        out_h=args.height,
        fov_deg=args.fov,
        encoder=args.encoder,
        copy_audio=not args.no_audio,
        youtube_optimise=True,
    )

    output = args.output or "output/game.mp4"
    os.makedirs(os.path.dirname(output), exist_ok=True)

    if args.segments:
        render_segments_concat(game, yaw_curve, pitch_curve, output, cfg)
    else:
        render_game(game, yaw_curve, pitch_curve, output, cfg)


def cmd_run(args):
    """Full pipeline in one command."""
    from pipeline.probe import probe_game, find_insv_files, estimate_processing_time
    from pipeline.detect import DetectionConfig, run_detection
    from pipeline.smooth import (
        SmoothConfig, smooth_schedule, compute_pitch_curve, save_dense_curve
    )
    from pipeline.render import RenderConfig, render_game
    from pipeline.calibrate import load_calibration
    import time

    # Gather files
    target = args.target
    if os.path.isdir(target):
        files = find_insv_files(target)
        if not files:
            files = [str(f) for f in sorted(Path(target).glob("*.mp4"))]
    else:
        files = [target]

    if not files:
        print(f"No video files found at: {target}")
        sys.exit(1)

    t_total = time.time()

    game = probe_game(files, model_hint=args.model)
    print(game.summary())
    print(estimate_processing_time(game, sample_every_n=args.sample_every))

    # Load calibration
    calib = load_calibration(args.calibration)
    if calib:
        print(f"\nCalibration loaded: {args.calibration}")
        print(f"  Pitch: {calib['pitch_yaw_left_deg']}° to {calib['pitch_yaw_right_deg']}°")
    else:
        print(f"\n[INFO] No calibration file found at {args.calibration}")
        print("       Run 'python main.py calibrate <video>' first for best results.")

    # --- Phase 1: Detection ---
    print("\n" + "="*60)
    print("PHASE 1: Detection")
    print("="*60)

    det_cfg = DetectionConfig(
        sample_every_n=args.sample_every,
        detection_width=args.detection_width,
        player_model_path=args.player_model,
        ball_model_path=args.ball_model,
        pitch_yaw_left=calib["pitch_yaw_left_deg"] if calib else None,
        pitch_yaw_right=calib["pitch_yaw_right_deg"] if calib else None,
    )

    os.makedirs("data", exist_ok=True)
    schedule_csv = args.schedule_csv or "data/yaw_schedule.csv"
    entries = run_detection(game, det_cfg, schedule_csv)

    # --- Phase 2: Smoothing ---
    print("\n" + "="*60)
    print("PHASE 2: Smoothing")
    print("="*60)

    fps = game.segments[0].fps
    total_frames = game.total_frames
    yaw_centre = calib["pitch_yaw_centre_deg"] if calib else 0.0

    smooth_cfg = SmoothConfig(
        sg_window_sec=2.0,
        max_deg_per_frame=1.5,
        yaw_centre=yaw_centre,
        yaw_max_dev=args.yaw_max_dev,
    )

    yaw_curve = smooth_schedule(entries, total_frames, fps, smooth_cfg)
    pitch_curve = compute_pitch_curve(total_frames, pitch_centre=0.0)

    curve_path = "data/yaw_curve.npz"
    save_dense_curve(yaw_curve, pitch_curve, curve_path)

    # --- Phase 3: Render ---
    print("\n" + "="*60)
    print("PHASE 3: Render")
    print("="*60)

    render_cfg = RenderConfig(
        out_w=args.width,
        out_h=args.height,
        fov_deg=args.fov,
        encoder=args.encoder,
        copy_audio=not args.no_audio,
        youtube_optimise=True,
    )

    output = args.output or "output/game.mp4"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    render_game(game, yaw_curve, pitch_curve, output, render_cfg)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"DONE  total time: {elapsed/60:.1f} min")
    print(f"Output: {output}")
    print(f"{'='*60}")


def cmd_models(args):
    from pipeline.lens_models import list_models
    list_models()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="autopan v6 - Insta360 football autopan pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared arguments
    def add_model_arg(p):
        p.add_argument("--model", default=None,
                       help="Camera model, e.g. 'ONE X2'. Auto-detected if omitted.")

    def add_detection_args(p):
        p.add_argument("--sample-every", type=int, default=8,
                       help="Sample 1 in N frames for detection (default: 8)")
        p.add_argument("--detection-width", type=int, default=1280,
                       help="Width to scale frames to before YOLO inference (default: 1280)")
        p.add_argument("--player-model", default="models/yolo11n.pt",
                       help="Path to player YOLO model")
        p.add_argument("--ball-model", default="models/ball.pt",
                       help="Path to ball YOLO model")
        p.add_argument("--calibration", default="calibration/pitch.json",
                       help="Path to pitch calibration JSON")

    def add_render_args(p):
        p.add_argument("--width", type=int, default=1280, help="Output width (default: 1280)")
        p.add_argument("--height", type=int, default=720, help="Output height (default: 720)")
        p.add_argument("--fov", type=float, default=90.0,
                       help="Output field of view in degrees (default: 90)")
        p.add_argument("--encoder", default="auto",
                       choices=["auto", "h264_videotoolbox", "libx264", "h264_nvenc"],
                       help="Video encoder (default: auto-detect)")
        p.add_argument("--no-audio", action="store_true", help="Strip audio from output")

    # --- probe ---
    p_probe = sub.add_parser("probe", help="Inspect video files")
    p_probe.add_argument("target", help="File or directory")
    p_probe.add_argument("--sample-every", type=int, default=8)
    add_model_arg(p_probe)
    p_probe.set_defaults(func=cmd_probe)

    # --- calibrate ---
    p_cal = sub.add_parser("calibrate", help="Interactive pitch boundary calibration")
    p_cal.add_argument("video", help="Path to .insv or .mp4 file")
    p_cal.add_argument("--output", default="calibration/pitch.json")
    p_cal.add_argument("--frame", type=int, default=30,
                       help="Frame index for calibration (default: 30)")
    add_model_arg(p_cal)
    p_cal.set_defaults(func=cmd_calibrate)

    # --- detect ---
    p_det = sub.add_parser("detect", help="Run detection phase only")
    p_det.add_argument("target", help="Directory of .insv files or single file")
    p_det.add_argument("--output", default=None, help="Output CSV path")
    add_model_arg(p_det)
    add_detection_args(p_det)
    p_det.set_defaults(func=cmd_detect)

    # --- smooth ---
    p_sm = sub.add_parser("smooth", help="Smooth a yaw schedule CSV")
    p_sm.add_argument("schedule_csv", help="Path to yaw_schedule.csv from detect phase")
    p_sm.add_argument("--fps", type=float, required=True)
    p_sm.add_argument("--total-frames", type=int, required=True, dest="total_frames")
    p_sm.add_argument("--output", default=None)
    p_sm.add_argument("--window-sec", type=float, default=2.0, dest="window_sec")
    p_sm.add_argument("--max-speed", type=float, default=1.5, dest="max_speed",
                       help="Max degrees per frame (default: 1.5)")
    p_sm.add_argument("--yaw-centre", type=float, default=0.0, dest="yaw_centre")
    p_sm.add_argument("--yaw-max-dev", type=float, default=50.0, dest="yaw_max_dev")
    p_sm.add_argument("--pitch-centre", type=float, default=0.0, dest="pitch_centre")
    p_sm.set_defaults(func=cmd_smooth)

    # --- render ---
    p_ren = sub.add_parser("render", help="Render phase only")
    p_ren.add_argument("target", help="Directory of .insv files or single file")
    p_ren.add_argument("curve_npz", help="Path to yaw_curve.npz from smooth phase")
    p_ren.add_argument("--output", default=None)
    p_ren.add_argument("--segments", action="store_true",
                        help="Render segments separately then concat (more robust)")
    add_model_arg(p_ren)
    add_render_args(p_ren)
    p_ren.set_defaults(func=cmd_render)

    # --- run (full pipeline) ---
    p_run = sub.add_parser("run", help="Full pipeline: detect + smooth + render")
    p_run.add_argument("target", help="Directory of .insv files or single file")
    p_run.add_argument("--output", default=None, help="Output MP4 path")
    p_run.add_argument("--schedule-csv", default=None, dest="schedule_csv",
                        help="Path to save intermediate yaw_schedule.csv")
    p_run.add_argument("--yaw-max-dev", type=float, default=50.0, dest="yaw_max_dev")
    add_model_arg(p_run)
    add_detection_args(p_run)
    add_render_args(p_run)
    p_run.set_defaults(func=cmd_run)

    # --- models ---
    p_mod = sub.add_parser("models", help="List supported camera models")
    p_mod.set_defaults(func=cmd_models)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
