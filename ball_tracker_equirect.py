#!/usr/bin/env python3
"""Equirectangular ball tracking utilities.

The tracker operates in longitude/latitude degrees instead of perspective
pixels, so the state survives camera pan changes. It uses a compact IMM-style
filter with stationary, constant-velocity, and constant-acceleration models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple
import math

import numpy as np


def wrap_lon(lon_deg: float) -> float:
    """Wrap longitude to [-180, 180)."""
    return ((float(lon_deg) + 180.0) % 360.0) - 180.0


def shortest_lon_delta(a_deg: float, b_deg: float) -> float:
    """Smallest signed delta a-b in degrees."""
    return wrap_lon(float(a_deg) - float(b_deg))


def unwrap_lon_near(lon_deg: float, reference_deg: float) -> float:
    """Return lon equivalent nearest to reference."""
    return float(reference_deg) + shortest_lon_delta(lon_deg, reference_deg)


@dataclass
class BallMeasurement:
    lon: float
    lat: float
    conf: float = 1.0
    source: str = "view"
    timestamp_s: Optional[float] = None


@dataclass
class TrackerSnapshot:
    lon: float
    lat: float
    vx: float
    vy: float
    conf: float
    frames_since_update: int
    model_probs: Tuple[float, float, float]


@dataclass
class IMMConfig:
    process_noise_stationary: float = 0.02
    process_noise_cv: float = 0.10
    process_noise_ca: float = 0.35
    measurement_noise_deg: float = 0.65
    max_missed_frames: int = 90
    reinit_distance_deg: float = 35.0
    transition: np.ndarray = field(default_factory=lambda: np.array([
        [0.92, 0.06, 0.02],
        [0.05, 0.88, 0.07],
        [0.03, 0.12, 0.85],
    ], dtype=np.float64))


class EquirectIMMTracker:
    """Single-hypothesis IMM tracker in equirectangular degrees."""

    def __init__(self, cfg: IMMConfig | None = None):
        self.cfg = cfg or IMMConfig()
        self.initialized = False
        self.frames_since_update = 999
        self.last_conf = 0.0
        self.model_probs = np.array([0.45, 0.40, 0.15], dtype=np.float64)
        self._xs = [np.zeros((6, 1), dtype=np.float64) for _ in range(3)]
        self._ps = [np.eye(6, dtype=np.float64) * 50.0 for _ in range(3)]
        self._x = np.zeros((6, 1), dtype=np.float64)
        self._p = np.eye(6, dtype=np.float64) * 50.0

    def reset(self) -> None:
        self.__init__(self.cfg)

    def _init_state(self, lon: float, lat: float, conf: float) -> None:
        x = np.array([[float(lon)], [float(lat)], [0.0], [0.0], [0.0], [0.0]], dtype=np.float64)
        p = np.diag([2.0, 2.0, 20.0, 20.0, 50.0, 50.0]).astype(np.float64)
        self._xs = [x.copy() for _ in range(3)]
        self._ps = [p.copy() for _ in range(3)]
        self._x = x.copy()
        self._p = p.copy()
        self.model_probs = np.array([0.55, 0.35, 0.10], dtype=np.float64)
        self.initialized = True
        self.frames_since_update = 0
        self.last_conf = float(conf)

    def _matrices(self, dt: float) -> List[Tuple[np.ndarray, np.ndarray]]:
        dt = max(1e-3, float(dt))
        dt2 = dt * dt

        f_stationary = np.eye(6, dtype=np.float64)
        f_stationary[2, 2] = 0.15
        f_stationary[3, 3] = 0.15
        f_stationary[4, 4] = 0.0
        f_stationary[5, 5] = 0.0

        f_cv = np.eye(6, dtype=np.float64)
        f_cv[0, 2] = dt
        f_cv[1, 3] = dt
        f_cv[4, 4] = 0.20
        f_cv[5, 5] = 0.20

        f_ca = np.eye(6, dtype=np.float64)
        f_ca[0, 2] = dt
        f_ca[1, 3] = dt
        f_ca[0, 4] = 0.5 * dt2
        f_ca[1, 5] = 0.5 * dt2
        f_ca[2, 4] = dt
        f_ca[3, 5] = dt

        qs = [
            self.cfg.process_noise_stationary,
            self.cfg.process_noise_cv,
            self.cfg.process_noise_ca,
        ]
        out = []
        for f, q in [(f_stationary, qs[0]), (f_cv, qs[1]), (f_ca, qs[2])]:
            qmat = np.diag([q, q, q * 4, q * 4, q * 12, q * 12]).astype(np.float64)
            out.append((f, qmat))
        return out

    def _combine_models(self) -> None:
        probs = self.model_probs / max(1e-12, float(self.model_probs.sum()))
        x = sum(float(probs[i]) * self._xs[i] for i in range(3))
        p = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            dx = self._xs[i] - x
            p += float(probs[i]) * (self._ps[i] + dx @ dx.T)
        self._x = x
        self._p = p

    def predict(self, dt_frames: float = 1.0) -> None:
        if not self.initialized:
            self.frames_since_update += 1
            return

        mixed_probs = self.cfg.transition.T @ self.model_probs
        self.model_probs = mixed_probs / max(1e-12, float(mixed_probs.sum()))

        for i, (f, q) in enumerate(self._matrices(dt_frames)):
            self._xs[i] = f @ self._xs[i]
            self._ps[i] = f @ self._ps[i] @ f.T + q

        self.frames_since_update += 1
        self._combine_models()

    def update(self, measurement: BallMeasurement) -> None:
        conf = float(np.clip(measurement.conf, 0.05, 1.0))
        lat = float(np.clip(measurement.lat, -89.0, 89.0))

        if not self.initialized:
            self._init_state(wrap_lon(measurement.lon), lat, conf)
            return

        lon = unwrap_lon_near(measurement.lon, float(self._x[0, 0]))
        dist = math.hypot(shortest_lon_delta(lon, float(self._x[0, 0])), lat - float(self._x[1, 0]))
        if self.frames_since_update <= 2 and dist > self.cfg.reinit_distance_deg:
            return
        if dist > self.cfg.reinit_distance_deg * 2.0:
            self._init_state(wrap_lon(measurement.lon), lat, conf)
            return

        h = np.zeros((2, 6), dtype=np.float64)
        h[0, 0] = 1.0
        h[1, 1] = 1.0
        noise = self.cfg.measurement_noise_deg / max(0.15, conf)
        r = np.eye(2, dtype=np.float64) * (noise * noise)
        z = np.array([[lon], [lat]], dtype=np.float64)

        likelihoods = np.zeros(3, dtype=np.float64)
        ident = np.eye(6, dtype=np.float64)
        for i in range(3):
            y = z - h @ self._xs[i]
            y[0, 0] = shortest_lon_delta(float(z[0, 0]), float((h @ self._xs[i])[0, 0]))
            s = h @ self._ps[i] @ h.T + r
            try:
                s_inv = np.linalg.inv(s)
                det_s = max(1e-12, float(np.linalg.det(s)))
            except np.linalg.LinAlgError:
                s_inv = np.linalg.pinv(s)
                det_s = 1e-12
            k = self._ps[i] @ h.T @ s_inv
            self._xs[i] = self._xs[i] + k @ y
            self._ps[i] = (ident - k @ h) @ self._ps[i]
            maha = float(y.T @ s_inv @ y)
            likelihoods[i] = math.exp(-0.5 * min(maha, 80.0)) / math.sqrt(det_s)

        self.model_probs = self.model_probs * np.maximum(likelihoods, 1e-12)
        self.model_probs = self.model_probs / max(1e-12, float(self.model_probs.sum()))
        self.frames_since_update = 0
        self.last_conf = conf
        self._combine_models()

    def update_many(self, measurements: Sequence[BallMeasurement]) -> None:
        if not measurements:
            return
        best = max(measurements, key=lambda m: float(m.conf))
        self.update(best)

    def is_valid(self) -> bool:
        return self.initialized and self.frames_since_update <= self.cfg.max_missed_frames

    def is_fresh(self, max_age_frames: int = 10) -> bool:
        return self.initialized and self.frames_since_update <= max_age_frames

    def snapshot(self) -> Optional[TrackerSnapshot]:
        if not self.initialized:
            return None
        return TrackerSnapshot(
            lon=wrap_lon(float(self._x[0, 0])),
            lat=float(np.clip(self._x[1, 0], -89.0, 89.0)),
            vx=float(self._x[2, 0]),
            vy=float(self._x[3, 0]),
            conf=float(self.last_conf),
            frames_since_update=int(self.frames_since_update),
            model_probs=tuple(float(v) for v in self.model_probs),
        )

    def position(self) -> Optional[Tuple[float, float]]:
        snap = self.snapshot()
        if snap is None:
            return None
        return snap.lon, snap.lat

class MultiHypothesisEquirectTracker:
    """Maintain up to three equirect ball hypotheses for temporary confusion."""

    def __init__(self, cfg: IMMConfig | None = None, max_hypotheses: int = 3):
        self.cfg = cfg or IMMConfig()
        self.max_hypotheses = max(1, int(max_hypotheses))
        self.hypotheses: List[EquirectIMMTracker] = []

    def predict(self, dt_frames: float = 1.0) -> None:
        for hyp in self.hypotheses:
            hyp.predict(dt_frames)
        self._prune()

    def update(self, measurements: Iterable[BallMeasurement]) -> None:
        measurements = sorted(list(measurements), key=lambda m: float(m.conf), reverse=True)
        if not measurements:
            return

        assigned = set()
        for hyp in self.hypotheses:
            snap = hyp.snapshot()
            if snap is None:
                continue
            best_i = None
            best_dist = float("inf")
            for i, m in enumerate(measurements):
                if i in assigned:
                    continue
                dist = math.hypot(shortest_lon_delta(m.lon, snap.lon), m.lat - snap.lat)
                if dist < best_dist:
                    best_i = i
                    best_dist = dist
            if best_i is not None and best_dist < self.cfg.reinit_distance_deg:
                hyp.update(measurements[best_i])
                assigned.add(best_i)

        for i, m in enumerate(measurements):
            if i in assigned:
                continue
            hyp = EquirectIMMTracker(self.cfg)
            hyp.update(m)
            self.hypotheses.append(hyp)

        self._prune()

    def _prune(self) -> None:
        valid = [h for h in self.hypotheses if h.is_valid()]

        def score(h: EquirectIMMTracker) -> float:
            return float(h.last_conf) - 0.02 * float(h.frames_since_update)

        valid.sort(key=score, reverse=True)
        self.hypotheses = valid[: self.max_hypotheses]

    @property
    def primary(self) -> Optional[EquirectIMMTracker]:
        self._prune()
        return self.hypotheses[0] if self.hypotheses else None

    def is_valid(self) -> bool:
        primary = self.primary
        return primary is not None and primary.is_valid()

    def is_fresh(self, max_age_frames: int = 10) -> bool:
        primary = self.primary
        return primary is not None and primary.is_fresh(max_age_frames)

    def snapshot(self) -> Optional[TrackerSnapshot]:
        primary = self.primary
        return primary.snapshot() if primary is not None else None

    def position(self) -> Optional[Tuple[float, float]]:
        snap = self.snapshot()
        if snap is None:
            return None
        return snap.lon, snap.lat
