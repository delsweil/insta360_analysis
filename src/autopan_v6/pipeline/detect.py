# pipeline/detect.py
"""
Detection phase: runs YOLO on sampled frames and produces a yaw schedule.

This is the CPU-heavy phase. Key optimisations vs v5:
  1. Works entirely in equirect/fisheye pixel space - no py360convert per frame
  2. Only processes every Nth frame (configurable)
  3. Detects on half-res frames (1920px wide instead of 3840)
  4. Ball and player detection on same frame in one pass where possible
  5. Pitch mask applied as a simple pixel crop, not a projected polygon
  6. Outputs a sparse CSV of (frame_index, target_yaw_deg) pairs
     which smooth.py then interpolates to a dense per-frame curve

Output CSV format:
  frame_idx, target_yaw, mode, n_players, ball_conf
  0, 12.3, ball, 8, 0.87
  5, 11.1, ball, 9, 0.91
  ...
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .probe import VideoMeta as VideoInfo, GameInfo
from .decode import make_detection_reader, fisheye_pixel_to_yaw, equirect_pixel_to_yaw
from .lens_models import LensModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DetectionConfig:
    # Frame sampling
    sample_every_n: int = 8
    # Detection input width (height scales proportionally)
    detection_width: int = 1280

    # YOLO model paths
    player_model_path: str = "models/yolo11n.pt"
    ball_model_path: str = "models/ball.pt"

    # YOLO inference sizes
    player_imgsz: int = 960
    ball_imgsz: int = 640

    # Confidence thresholds
    player_conf: float = 0.35
    ball_conf: float = 0.20

    # Ball gating
    ball_confirm_frames: int = 3     # must see ball N times before trusting
    ball_miss_short: int = 10        # frames before falling back from last_ball
    ball_miss_long: int = 90         # frames before falling back to players

    # Target selection weights (for player centroid)
    # If True, weight players by confidence (prefer high-conf detections)
    weight_by_conf: bool = True

    # Pan limits (degrees from centre)
    max_yaw_deg: float = 50.0

    # Pitch boundary (yaw angles defining left/right edges of pitch)
    # Set from calibration; None = no limit
    pitch_yaw_left: Optional[float] = None
    pitch_yaw_right: Optional[float] = None

    # Progress reporting interval (frames)
    report_every: int = 50


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def foot_y(self) -> float:
        return self.y2

    @property
    def area(self) -> float:
        return max(0.0, (self.x2 - self.x1) * (self.y2 - self.y1))


def _parse_yolo_results(results, cls_filter: Optional[str] = None, names: dict = None) -> List[Detection]:
    """Parse Ultralytics YOLO result into Detection list."""
    dets = []
    for b in results.boxes:
        cls_id = int(b.cls[0])
        if cls_filter and names:
            if names.get(cls_id, "") != cls_filter:
                continue
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        cf = float(b.conf[0])
        dets.append(Detection(x1, y1, x2, y2, cf))
    return dets


@dataclass
class BallState:
    last_yaw: Optional[float] = None
    confirm_count: int = 0
    trusted: bool = False
    frames_since_seen: int = 999999


@dataclass
class ScheduleEntry:
    frame_idx: int
    target_yaw: float
    mode: str               # 'ball' | 'last_ball' | 'players' | 'centre'
    n_players: int = 0
    ball_conf: float = 0.0


# ---------------------------------------------------------------------------
# Coordinate converter
# ---------------------------------------------------------------------------

class CoordConverter:
    """
    Converts detection pixel coordinates in the (scaled, cropped) detection
    frame back to yaw angles.
    """

    def __init__(
        self,
        src_info: VideoInfo,
        det_width: int,
        det_height: int,
        crop_y1: int,
        crop_y2: int,
    ):
        self.src_info = src_info
        self.det_width = det_width
        self.det_height = det_height
        self.crop_y1 = crop_y1
        self.crop_y2 = crop_y2

        # Scale factors: det_px -> original_px
        src_crop_h = crop_y2 - crop_y1
        self.scale_x = src_info.width / det_width
        self.scale_y = src_info.height / (det_height * src_info.height / src_crop_h)

    def to_yaw(self, px: float, py: float) -> Optional[float]:
        if False:  # fisheye model disabled, using equirect mapping
            return fisheye_pixel_to_yaw(
                px, py,
                self.src_info.width,
                self.src_info.height,
                self.src_info.width,
                crop_y1=self.crop_y1,
                scale_x=self.src_info.width / self.det_width,
                scale_y=self.src_info.height / (self.det_height + self.crop_y1),
            )
        else:
            # Equirect: linear mapping based on x position in full-width frame
            orig_x = px * (self.src_info.width / self.det_width)
            return equirect_pixel_to_yaw(orig_x, self.src_info.width)


# ---------------------------------------------------------------------------
# Main detection runner
# ---------------------------------------------------------------------------

class DetectionRunner:
    """
    Runs detection on a single video segment and produces yaw schedule entries.
    """

    def __init__(self, cfg: DetectionConfig):
        self.cfg = cfg
        self._player_model = None
        self._ball_model = None
        self._player_names = {}
        self._ball_names = {}

    def _load_models(self) -> None:
        """Lazy-load YOLO models (expensive, done once)."""
        if self._player_model is not None:
            return

        from ultralytics import YOLO

        if not os.path.exists(self.cfg.player_model_path):
            raise FileNotFoundError(
                f"Player model not found: {self.cfg.player_model_path}\n"
                f"Download with: pip install ultralytics && "
                f"python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\""
            )
        if not os.path.exists(self.cfg.ball_model_path):
            raise FileNotFoundError(
                f"Ball model not found: {self.cfg.ball_model_path}"
            )

        print(f"Loading player model: {self.cfg.player_model_path}")
        self._player_model = YOLO(self.cfg.player_model_path)
        self._player_names = getattr(self._player_model, "names", {})

        print(f"Loading ball model: {self.cfg.ball_model_path}")
        self._ball_model = YOLO(self.cfg.ball_model_path)
        self._ball_names = getattr(self._ball_model, "names", {})
        print("Models loaded.")

    def _detect_frame(
        self,
        frame_bgr: np.ndarray,
    ) -> Tuple[List[Detection], List[Detection]]:
        """Run player + ball detection on one frame. Returns (players, balls)."""

        # Players
        p_results = self._player_model(
            frame_bgr,
            imgsz=self.cfg.player_imgsz,
            conf=self.cfg.player_conf,
            verbose=False,
        )[0]
        players = _parse_yolo_results(p_results, cls_filter="person", names=self._player_names)

        # Ball
        b_results = self._ball_model(
            frame_bgr,
            imgsz=self.cfg.ball_imgsz,
            conf=self.cfg.ball_conf,
            verbose=False,
        )[0]
        balls = _parse_yolo_results(b_results)

        return players, balls

    def _pick_ball(self, balls: List[Detection], frame_w: int, frame_h: int) -> Optional[Detection]:
        """Pick best ball detection. Filter out implausibly large blobs."""
        max_area = frame_w * frame_h * 0.015
        candidates = [b for b in balls if b.area <= max_area]
        if not candidates:
            return None
        return max(candidates, key=lambda b: (b.conf, -b.area))

    def _players_to_yaw(
        self,
        players: List[Detection],
        converter: CoordConverter,
    ) -> Optional[float]:
        """Compute centroid yaw from player detections."""
        yaws = []
        weights = []
        for p in players:
            yaw = converter.to_yaw(p.cx, p.foot_y)
            if yaw is not None:
                yaws.append(yaw)
                weights.append(p.conf if self.cfg.weight_by_conf else 1.0)

        if not yaws:
            return None

        # Weighted mean (handle wrap-around via sin/cos)
        yaw_rad = np.radians(yaws)
        w = np.array(weights)
        mean_sin = np.average(np.sin(yaw_rad), weights=w)
        mean_cos = np.average(np.cos(yaw_rad), weights=w)
        return float(np.degrees(np.arctan2(mean_sin, mean_cos)))

    def run_segment(
        self,
        info: VideoInfo,
        frame_offset: int = 0,   # frame index of this segment's start in the full game
    ) -> List[ScheduleEntry]:
        """
        Run detection on one video segment.

        Args:
            info         : VideoInfo for this segment
            frame_offset : cumulative frame index offset (for multi-segment games)

        Returns:
            List of ScheduleEntry, one per sampled frame
        """
        self._load_models()

        reader = make_detection_reader(
            info,
            sample_every_n=self.cfg.sample_every_n,
            target_width=self.cfg.detection_width,
        )

        # Converter for yaw mapping
        crop_y1 = int(0.15 * info.height)
        crop_y2 = int(0.85 * info.height)
        converter = CoordConverter(
            info,
            det_width=reader.out_width,
            det_height=reader.out_height,
            crop_y1=crop_y1,
            crop_y2=crop_y2,
        )

        ball_state = BallState()
        schedule: List[ScheduleEntry] = []

        t0 = time.time()
        n_processed = 0

        print(f"\nDetecting: {os.path.basename(info.path)}")
        print(f"  Detection frame size: {reader.out_width}x{reader.out_height}")
        print(f"  Sampling 1 in {self.cfg.sample_every_n} frames "
              f"({info.n_frames // self.cfg.sample_every_n} inference calls)")

        with reader:
            for local_idx, frame_bgr in reader.frames():
                global_idx = frame_offset + local_idx

                players, balls = self._detect_frame(frame_bgr)

                # --- Ball ---
                best_ball = self._pick_ball(balls, reader.out_width, reader.out_height)
                ball_yaw = None
                ball_conf = 0.0

                if best_ball is not None:
                    yaw = converter.to_yaw(best_ball.cx, best_ball.cy)
                    if yaw is not None:
                        ball_yaw = yaw
                        ball_conf = best_ball.conf
                        ball_state.confirm_count += 1
                        if ball_state.confirm_count >= self.cfg.ball_confirm_frames:
                            ball_state.trusted = True
                        ball_state.last_yaw = ball_yaw
                        ball_state.frames_since_seen = 0
                else:
                    ball_state.frames_since_seen += self.cfg.sample_every_n
                    ball_state.confirm_count = 0

                # --- Target selection ---
                target_yaw: float
                mode: str

                if ball_yaw is not None and ball_state.trusted:
                    target_yaw = ball_yaw
                    mode = "ball"
                elif (ball_state.last_yaw is not None and
                      ball_state.frames_since_seen < self.cfg.ball_miss_short):
                    target_yaw = ball_state.last_yaw
                    mode = "last_ball"
                else:
                    player_yaw = self._players_to_yaw(players, converter)
                    if player_yaw is not None and ball_state.frames_since_seen < self.cfg.ball_miss_long:
                        target_yaw = player_yaw
                        mode = "players"
                    else:
                        target_yaw = 0.0
                        mode = "centre"

                # Clamp to pitch bounds if calibrated
                if self.cfg.pitch_yaw_left is not None and self.cfg.pitch_yaw_right is not None:
                    target_yaw = max(self.cfg.pitch_yaw_left,
                                     min(self.cfg.pitch_yaw_right, target_yaw))

                # Clamp to max deviation
                target_yaw = max(-self.cfg.max_yaw_deg,
                                 min(self.cfg.max_yaw_deg, target_yaw))

                schedule.append(ScheduleEntry(
                    frame_idx=global_idx,
                    target_yaw=target_yaw,
                    mode=mode,
                    n_players=len(players),
                    ball_conf=ball_conf,
                ))

                n_processed += 1
                if n_processed % self.cfg.report_every == 0:
                    elapsed = time.time() - t0
                    fps_det = n_processed / elapsed
                    pct = 100 * local_idx / max(1, info.n_frames)
                    print(f"  [{pct:5.1f}%] frame {global_idx:6d}  "
                          f"det_fps={fps_det:5.1f}  mode={mode:<10}  "
                          f"players={len(players):2d}  ball_conf={ball_conf:.2f}")

        elapsed = time.time() - t0
        print(f"  Done: {n_processed} frames in {elapsed:.1f}s "
              f"({n_processed/elapsed:.1f} det_fps)")

        return schedule


# ---------------------------------------------------------------------------
# Multi-segment runner
# ---------------------------------------------------------------------------

def run_detection(
    game: GameInfo,
    cfg: DetectionConfig,
    output_csv: str,
) -> List[ScheduleEntry]:
    """
    Run detection across all segments of a game.

    Args:
        game       : GameInfo from probe_game()
        cfg        : DetectionConfig
        output_csv : path to write the yaw schedule CSV

    Returns:
        Full list of ScheduleEntry for the game
    """
    runner = DetectionRunner(cfg)
    all_entries: List[ScheduleEntry] = []
    frame_offset = 0

    for seg in game.segments:
        entries = runner.run_segment(seg, frame_offset=frame_offset)
        all_entries.extend(entries)
        frame_offset += seg.n_frames

    # Write CSV
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    _write_schedule_csv(all_entries, output_csv)
    print(f"\nYaw schedule saved: {output_csv}  ({len(all_entries)} entries)")

    return all_entries


def _write_schedule_csv(entries: List[ScheduleEntry], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "target_yaw", "mode", "n_players", "ball_conf"])
        for e in entries:
            writer.writerow([
                e.frame_idx,
                f"{e.target_yaw:.4f}",
                e.mode,
                e.n_players,
                f"{e.ball_conf:.3f}",
            ])


def load_schedule_csv(path: str) -> List[ScheduleEntry]:
    """Load a previously saved yaw schedule CSV."""
    entries = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(ScheduleEntry(
                frame_idx=int(row["frame_idx"]),
                target_yaw=float(row["target_yaw"]),
                mode=row["mode"],
                n_players=int(row["n_players"]),
                ball_conf=float(row["ball_conf"]),
            ))
    return entries
