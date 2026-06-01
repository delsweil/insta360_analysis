#!/usr/bin/env python3
"""Pitch homography helpers for physical plausibility checks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class PitchPoint:
    x_m: float
    y_m: float


@dataclass
class PitchHomographyModel:
    image_to_pitch_h: np.ndarray
    pitch_to_image_h: np.ndarray
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0

    @classmethod
    def from_calibration(
        cls,
        calib_path: str | Path,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ) -> "PitchHomographyModel":
        pts = load_calibration_polygon(calib_path)
        corners = polygon_to_quad(pts)
        dst = np.array([
            [0.0, 0.0],
            [pitch_length_m, 0.0],
            [pitch_length_m, pitch_width_m],
            [0.0, pitch_width_m],
        ], dtype=np.float32)
        h_img_to_pitch = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        h_pitch_to_img = cv2.getPerspectiveTransform(dst, corners.astype(np.float32))
        return cls(h_img_to_pitch, h_pitch_to_img, pitch_length_m, pitch_width_m)

    def image_to_pitch(self, x: float, y: float) -> PitchPoint:
        pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.image_to_pitch_h)[0, 0]
        return PitchPoint(float(out[0]), float(out[1]))

    def pitch_to_image(self, x_m: float, y_m: float) -> Tuple[float, float]:
        pt = np.array([[[float(x_m), float(y_m)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.pitch_to_image_h)[0, 0]
        return float(out[0]), float(out[1])

    def contains_pitch_point(self, point: PitchPoint, margin_m: float = 3.0) -> bool:
        return (
            -margin_m <= point.x_m <= self.pitch_length_m + margin_m
            and -margin_m <= point.y_m <= self.pitch_width_m + margin_m
        )

    def plausible_motion(
        self,
        prev: PitchPoint,
        cur: PitchPoint,
        dt_s: float,
        max_speed_kmh: float = 150.0,
    ) -> bool:
        if dt_s <= 0:
            return False
        dist_m = math.hypot(cur.x_m - prev.x_m, cur.y_m - prev.y_m)
        speed_kmh = (dist_m / dt_s) * 3.6
        return speed_kmh <= max_speed_kmh

    def lon_lat_to_pitch(
        self,
        lon_deg: float,
        lat_deg: float,
        eq_width: int = 2880,
        eq_height: int = 1440,
    ) -> PitchPoint:
        x, y = lon_lat_to_equirect_pixel(lon_deg, lat_deg, eq_width, eq_height)
        return self.image_to_pitch(x, y)


@dataclass
class PitchAwareBallGate:
    model: PitchHomographyModel
    max_speed_kmh: float = 150.0
    margin_m: float = 6.0
    last_point: Optional[PitchPoint] = None
    last_time_s: Optional[float] = None

    def check(self, lon_deg: float, lat_deg: float, timestamp_s: Optional[float]) -> bool:
        point = self.model.lon_lat_to_pitch(lon_deg, lat_deg)
        if not self.model.contains_pitch_point(point, margin_m=self.margin_m):
            return False
        if self.last_point is not None and self.last_time_s is not None and timestamp_s is not None:
            dt_s = float(timestamp_s) - float(self.last_time_s)
            if dt_s > 0 and not self.model.plausible_motion(
                self.last_point, point, dt_s, max_speed_kmh=self.max_speed_kmh
            ):
                return False
            if dt_s <= 0:
                return True
        self.last_point = point
        self.last_time_s = timestamp_s
        return True


def lon_lat_to_equirect_pixel(
    lon_deg: float,
    lat_deg: float,
    eq_width: int = 2880,
    eq_height: int = 1440,
) -> Tuple[float, float]:
    lon = ((float(lon_deg) + 180.0) % 360.0) - 180.0
    lat = max(-89.0, min(89.0, float(lat_deg)))
    x = (lon + 180.0) / 360.0 * float(eq_width)
    y = (90.0 - lat) / 180.0 * float(eq_height)
    return x, y


def load_calibration_polygon(calib_path: str | Path) -> np.ndarray:
    with open(calib_path, encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("pixel_polygon") or data.get("pitch_polygon") or data.get("auto_polygon")
    if not raw:
        raise ValueError(f"No calibration polygon in {calib_path}")
    if isinstance(raw[0], dict):
        source = data.get("source_frame") or {}
        w = float(source.get("width", 2880))
        h = float(source.get("height", 1440))
        pts = np.array([[float(p["x"]) * w, float(p["y"]) * h] for p in raw], dtype=np.float32)
    else:
        pts = np.array(raw, dtype=np.float32)
    if len(pts) < 4:
        raise ValueError("Need at least four polygon points for homography")
    return pts


def polygon_to_quad(points: np.ndarray) -> np.ndarray:
    """Approximate a pitch polygon as TL, TR, BR, BL image corners."""
    pts = np.asarray(points, dtype=np.float32)
    y_mid = float(np.median(pts[:, 1]))
    top = pts[pts[:, 1] <= y_mid]
    bottom = pts[pts[:, 1] > y_mid]
    if len(top) < 2 or len(bottom) < 2:
        ordered = order_quad_by_sum_diff(pts)
        return ordered
    tl = top[np.argmin(top[:, 0])]
    tr = top[np.argmax(top[:, 0])]
    bl = bottom[np.argmin(bottom[:, 0])]
    br = bottom[np.argmax(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def order_quad_by_sum_diff(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    if len(hull) > 4:
        rect = cv2.minAreaRect(hull)
        hull = cv2.boxPoints(rect).astype(np.float32)
    s = hull.sum(axis=1)
    d = np.diff(hull, axis=1).reshape(-1)
    tl = hull[np.argmin(s)]
    br = hull[np.argmax(s)]
    tr = hull[np.argmin(d)]
    bl = hull[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)
