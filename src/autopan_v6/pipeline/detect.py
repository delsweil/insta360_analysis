# pipeline/detect.py
"""
Detection phase using Insta360 X2 VID _10_ (pitch-facing) files.

Pipeline per sampled frame:
  1. Read raw 2880x2880 frame from VID _10_ file
  2. extract_pitch_crop() -> ~2880x1920 fisheye showing full pitch
  3. YOLO detects persons and ball
  4. Pitch polygon mask filters out coaches/spectators/cars
  5. Surviving detections mapped to yaw angles via calibrated boundaries
  6. Target yaw selected: ball > last_ball > player centroid > pitch centre

Output: sparse CSV of (frame_idx, target_yaw, mode, n_players, ball_conf)
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cv2

from .decode import (
    FrameReader, extract_pitch_crop,
    fisheye_cx_to_yaw, build_pitch_mask, point_in_mask,
)
from .probe import VideoMeta


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DetectionConfig:
    # Frame sampling
    sample_every_n: int = 8

    # YOLO models
    player_model_path: str = "models/yolo11n.pt"
    ball_model_path: str = "models/ball.pt"

    # YOLO inference
    player_imgsz: int = 1280
    ball_imgsz: int = 1280
    player_conf: float = 0.15
    ball_conf: float = 0.10

    # Ball gating
    ball_confirm_frames: int = 3
    ball_miss_short: int = 10
    ball_miss_long: int = 90

    # Calibrated pitch bounds (from calibration JSON)
    pitch_yaw_left: float = -30.0
    pitch_yaw_right: float = 30.0
    pitch_yaw_centre: float = 0.0

    # Pitch polygon mask (list of [x,y] in crop pixel coords)
    pitch_polygon: Optional[list] = None

    # Progress reporting
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
        return float(self.y2)

    @property
    def area(self) -> float:
        return max(0.0, (self.x2 - self.x1) * (self.y2 - self.y1))


def _parse_detections(
    results,
    cls_filter: Optional[str],
    names: dict,
) -> List[Detection]:
    dets = []
    for b in results.boxes:
        cls_id = int(b.cls[0])
        if cls_filter and names.get(cls_id, "") != cls_filter:
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
    mode: str
    n_players: int = 0
    ball_conf: float = 0.0


# ---------------------------------------------------------------------------
# Detection runner
# ---------------------------------------------------------------------------

class DetectionRunner:

    def __init__(self, cfg: DetectionConfig):
        self.cfg = cfg
        self._player_model = None
        self._ball_model = None
        self._player_names = {}
        self._ball_names = {}

    def _load_models(self) -> None:
        if self._player_model is not None:
            return
        from ultralytics import YOLO

        for p in [self.cfg.player_model_path, self.cfg.ball_model_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Model not found: {p}")

        print(f"Loading player model: {self.cfg.player_model_path}")
        self._player_model = YOLO(self.cfg.player_model_path)
        self._player_names = getattr(self._player_model, "names", {})

        print(f"Loading ball model: {self.cfg.ball_model_path}")
        self._ball_model = YOLO(self.cfg.ball_model_path)
        self._ball_names = getattr(self._ball_model, "names", {})
        print("Models loaded.")

    def _detect(
        self,
        frame_bgr: np.ndarray,
    ) -> Tuple[List[Detection], List[Detection]]:
        p_res = self._player_model(
            frame_bgr,
            imgsz=self.cfg.player_imgsz,
            conf=self.cfg.player_conf,
            verbose=False,
        )[0]
        players = _parse_detections(
            p_res, cls_filter="person", names=self._player_names
        )

        b_res = self._ball_model(
            frame_bgr,
            imgsz=self.cfg.ball_imgsz,
            conf=self.cfg.ball_conf,
            verbose=False,
        )[0]
        balls = _parse_detections(b_res, cls_filter=None, names=self._ball_names)

        return players, balls

    def _filter_by_mask(
        self,
        detections: List[Detection],
        mask: Optional[np.ndarray],
    ) -> List[Detection]:
        """Keep only detections whose foot point is inside the pitch mask."""
        if mask is None:
            return detections
        return [
            d for d in detections
            if point_in_mask(mask, d.cx, d.foot_y)
        ]

    def _pick_ball(
        self,
        balls: List[Detection],
        crop_w: int,
        crop_h: int,
    ) -> Optional[Detection]:
        max_area = crop_w * crop_h * 0.015
        candidates = [b for b in balls if b.area <= max_area]
        if not candidates:
            return None
        return max(candidates, key=lambda b: (b.conf, -b.area))

    def _detections_to_yaw(
        self,
        detections: List[Detection],
        crop_w: int,
    ) -> Optional[float]:
        """Weighted centroid yaw from detections."""
        if not detections:
            return None
        yaws = [
            fisheye_cx_to_yaw(
                d.cx, crop_w,
                self.cfg.pitch_yaw_left,
                self.cfg.pitch_yaw_right,
            )
            for d in detections
        ]
        weights = np.array([d.conf for d in detections])
        yaw_rad = np.radians(yaws)
        mean_sin = np.average(np.sin(yaw_rad), weights=weights)
        mean_cos = np.average(np.cos(yaw_rad), weights=weights)
        return float(np.degrees(np.arctan2(mean_sin, mean_cos)))

    def run_segment(
        self,
        info: VideoMeta,
        frame_offset: int = 0,
    ) -> List[ScheduleEntry]:
        """
        Run detection on one VID _10_ file.

        Args:
            info        : VideoMeta from probe_file() for the VID _10_ file
            frame_offset: cumulative frame offset for multi-segment games
        """
        self._load_models()

        reader = FrameReader(
            path=info.path,
            width=info.width,
            height=info.height,
            sample_every_n=self.cfg.sample_every_n,
            use_hwaccel=True,
        )

        # Build pitch mask if polygon provided
        # We don't know crop dims until first frame, build lazily
        mask = None
        crop_w = crop_h = None

        ball_state = BallState()
        schedule: List[ScheduleEntry] = []
        t0 = time.time()
        n_processed = 0

        print(f"\nDetecting: {os.path.basename(info.path)}")
        print(f"  Sampling 1 in {self.cfg.sample_every_n} frames "
              f"(~{info.n_frames // self.cfg.sample_every_n} inference calls)")
        print(f"  Yaw range: {self.cfg.pitch_yaw_left:.1f}° to "
              f"{self.cfg.pitch_yaw_right:.1f}°  "
              f"centre={self.cfg.pitch_yaw_centre:.1f}°")

        with reader:
            for local_idx, raw_frame in reader.frames():
                global_idx = frame_offset + local_idx

                # Extract pitch crop
                crop = extract_pitch_crop(raw_frame)
                crop = cv2.resize(crop, (1440, 960))
                ch, cw = crop.shape[:2]

                # Build mask on first frame
                if mask is None and self.cfg.pitch_polygon:
                    mask = build_pitch_mask(
                        self.cfg.pitch_polygon, ch, cw
                    )
                    crop_w, crop_h = cw, ch
                    print(f"  Pitch mask: {cw}x{ch}  "
                          f"{len(self.cfg.pitch_polygon)} polygon points")
                elif crop_w is None:
                    crop_w, crop_h = cw, ch

                # Detect
                players_raw, balls_raw = self._detect(crop)

                # Filter by pitch polygon
                players = self._filter_by_mask(players_raw, mask)
                balls   = self._filter_by_mask(balls_raw, mask)

                # Ball selection
                best_ball = self._pick_ball(balls, cw, ch)
                ball_yaw  = None
                ball_conf = 0.0

                if best_ball is not None:
                    ball_yaw = fisheye_cx_to_yaw(
                        best_ball.cx, cw,
                        self.cfg.pitch_yaw_left,
                        self.cfg.pitch_yaw_right,
                    )
                    ball_conf = best_ball.conf
                    ball_state.confirm_count += 1
                    if ball_state.confirm_count >= self.cfg.ball_confirm_frames:
                        ball_state.trusted = True
                    ball_state.last_yaw = ball_yaw
                    ball_state.frames_since_seen = 0
                else:
                    ball_state.frames_since_seen += self.cfg.sample_every_n
                    ball_state.confirm_count = 0

                # Target selection
                if ball_yaw is not None and ball_state.trusted:
                    target_yaw = ball_yaw
                    mode = "ball"
                elif (ball_state.last_yaw is not None and
                      ball_state.frames_since_seen < self.cfg.ball_miss_short):
                    target_yaw = ball_state.last_yaw
                    mode = "last_ball"
                else:
                    player_yaw = self._detections_to_yaw(players, cw)
                    if (player_yaw is not None and
                            ball_state.frames_since_seen < self.cfg.ball_miss_long):
                        target_yaw = player_yaw
                        mode = "players"
                    else:
                        target_yaw = self.cfg.pitch_yaw_centre
                        mode = "centre"

                # Clamp to pitch bounds
                target_yaw = max(
                    self.cfg.pitch_yaw_left,
                    min(self.cfg.pitch_yaw_right, target_yaw)
                )

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
                    pct = 100 * local_idx / max(1, info.n_frames)
                    print(f"  [{pct:5.1f}%] frame {global_idx:6d}  "
                          f"det_fps={n_processed/elapsed:5.1f}  "
                          f"mode={mode:<10}  "
                          f"players={len(players):2d}  "
                          f"ball={ball_conf:.2f}")

        elapsed = time.time() - t0
        print(f"  Done: {n_processed} frames in {elapsed:.1f}s "
              f"({n_processed/elapsed:.1f} det_fps)")
        return schedule


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _write_schedule_csv(entries: List[ScheduleEntry], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
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
