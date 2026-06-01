#!/usr/bin/env python3
"""Lightweight game-state heuristics for autopan target recovery.

This module is intentionally conservative. It does not try to infer complete
football semantics; it provides a physically plausible fallback target when the
ball tracker is stale by combining last ball velocity with player formation
movement and simple pitch-area events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import math

import numpy as np

try:
    from pitch_model import PitchHomographyModel, PitchPoint
except Exception:  # pragma: no cover - optional when cv2 is unavailable
    PitchHomographyModel = None
    PitchPoint = None


def wrap_lon(lon_deg: float) -> float:
    return ((float(lon_deg) + 180.0) % 360.0) - 180.0


def shortest_lon_delta(a_deg: float, b_deg: float) -> float:
    return wrap_lon(float(a_deg) - float(b_deg))


def pixel_to_lon(px: float, cam_yaw: float, width: int, fov_deg: float) -> float:
    nx = (float(px) - width / 2.0) / (width / 2.0)
    angle_x = math.degrees(math.atan(nx * math.tan(math.radians(fov_deg / 2.0))))
    return wrap_lon(angle_x + float(cam_yaw))


@dataclass
class GameStateConfig:
    max_ball_prediction_s: float = 4.0
    formation_bias_deg: float = 10.0
    formation_memory_s: float = 6.0
    goal_zone_x_m: float = 10.0
    corner_zone_m: float = 8.0
    touchline_zone_m: float = 4.0


@dataclass
class GameStatePrediction:
    lon: float
    lat: float
    reason: str
    confidence: float


class GameStatePredictor:
    def __init__(self, cfg: GameStateConfig | None = None, pitch_model=None):
        self.cfg = cfg or GameStateConfig()
        self.pitch_model = pitch_model
        self.last_ball: Optional[Tuple[float, float, float]] = None
        self.prev_ball: Optional[Tuple[float, float, float]] = None
        self.last_formation: Optional[Tuple[float, float]] = None
        self.prev_formation: Optional[Tuple[float, float]] = None
        self.last_event: Optional[str] = None

    def update_players(
        self,
        players: Sequence[Tuple[float, float]],
        cam_yaw: float,
        fov_deg: float,
        timestamp_s: float,
        width: int = 1280,
    ) -> None:
        if len(players) < 2:
            return
        lons = np.array([pixel_to_lon(p[0], cam_yaw, width, fov_deg) for p in players], dtype=np.float64)
        # Use circular mean around current yaw to avoid seam issues.
        unwrapped = np.array([float(cam_yaw) + shortest_lon_delta(lon, cam_yaw) for lon in lons], dtype=np.float64)
        center_lon = wrap_lon(float(np.median(unwrapped)))
        self.prev_formation = self.last_formation
        self.last_formation = (timestamp_s, center_lon)

    def update_ball(self, lon: float, lat: float, timestamp_s: float) -> None:
        self.prev_ball = self.last_ball
        self.last_ball = (timestamp_s, wrap_lon(lon), float(lat))
        self.last_event = self._classify_event(lon, lat)

    def _classify_event(self, lon: float, lat: float) -> Optional[str]:
        if self.pitch_model is None or not hasattr(self.pitch_model, "lon_lat_to_pitch"):
            return None
        try:
            pt = self.pitch_model.lon_lat_to_pitch(lon, lat)
        except Exception:
            return None
        length = float(getattr(self.pitch_model, "pitch_length_m", 105.0))
        width = float(getattr(self.pitch_model, "pitch_width_m", 68.0))
        if pt.x_m < self.cfg.goal_zone_x_m:
            return "left_goal_area"
        if pt.x_m > length - self.cfg.goal_zone_x_m:
            return "right_goal_area"
        near_goal_line = pt.x_m < self.cfg.corner_zone_m or pt.x_m > length - self.cfg.corner_zone_m
        near_touch_line = pt.y_m < self.cfg.corner_zone_m or pt.y_m > width - self.cfg.corner_zone_m
        if near_goal_line and near_touch_line:
            return "corner_area"
        if pt.y_m < self.cfg.touchline_zone_m or pt.y_m > width - self.cfg.touchline_zone_m:
            return "throw_in_area"
        return None

    def _ball_velocity(self) -> Optional[Tuple[float, float]]:
        if self.last_ball is None or self.prev_ball is None:
            return None
        t1, lon1, lat1 = self.prev_ball
        t2, lon2, lat2 = self.last_ball
        dt = t2 - t1
        if dt <= 0:
            return None
        return shortest_lon_delta(lon2, lon1) / dt, (lat2 - lat1) / dt

    def _formation_direction(self, timestamp_s: float) -> float:
        if self.last_formation is None or self.prev_formation is None:
            return 0.0
        t1, lon1 = self.prev_formation
        t2, lon2 = self.last_formation
        if timestamp_s - t2 > self.cfg.formation_memory_s or t2 <= t1:
            return 0.0
        delta = shortest_lon_delta(lon2, lon1)
        if abs(delta) < 0.2:
            return 0.0
        return 1.0 if delta > 0 else -1.0

    def predict(self, timestamp_s: float) -> Optional[GameStatePrediction]:
        if self.last_ball is not None:
            t_ball, lon, lat = self.last_ball
            age = timestamp_s - t_ball
            if 0 <= age <= self.cfg.max_ball_prediction_s:
                vel = self._ball_velocity()
                pred_lon, pred_lat = lon, lat
                reason = "last_ball"
                confidence = max(0.25, 1.0 - age / self.cfg.max_ball_prediction_s)
                if vel is not None:
                    vx, vy = vel
                    pred_lon = wrap_lon(lon + vx * age)
                    pred_lat = max(-85.0, min(85.0, lat + vy * age))
                    reason = "ball_velocity"
                direction = self._formation_direction(timestamp_s)
                if direction:
                    pred_lon = wrap_lon(pred_lon + direction * self.cfg.formation_bias_deg * min(1.0, age / 2.0))
                    reason += "_formation"
                if self.last_event:
                    reason += f"_{self.last_event}"
                    confidence = max(confidence, 0.35)
                return GameStatePrediction(pred_lon, pred_lat, reason, confidence)

        if self.last_formation is not None:
            t_form, lon = self.last_formation
            age = timestamp_s - t_form
            if 0 <= age <= self.cfg.formation_memory_s:
                direction = self._formation_direction(timestamp_s)
                pred_lon = wrap_lon(lon + direction * self.cfg.formation_bias_deg)
                return GameStatePrediction(pred_lon, 0.0, "formation_shift", 0.25)
        return None
