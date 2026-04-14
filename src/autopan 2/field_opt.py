from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import cv2
from py360convert import e2p

@dataclass
class FieldOptConfig:
    # Evaluation resolution (keep small for speed)
    eval_w: int = 320
    eval_h: int = 180

    # Only run optimization if below these
    min_coverage: float = 0.18
    max_center_offset: float = 0.22  # fraction of width/height

    # Search grid (degrees)
    yaw_offsets: Tuple[float, ...] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    pitch_offsets: Tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)

    # How strongly we prefer pitch centered vs just present
    center_weight: float = 0.55  # 0..1
    coverage_weight: float = 0.45

class FieldViewOptimizer:
    def __init__(self, mask360: np.ndarray, cfg: FieldOptConfig):
        if mask360 is None:
            raise ValueError("mask360 is None")
        self.mask360 = mask360
        self.cfg = cfg

        # precompute a smaller 360 mask for faster e2p calls (huge speed win)
        self.mask360_small = cv2.resize(
            self.mask360,
            (self.mask360.shape[1] // 2, self.mask360.shape[0] // 2),
            interpolation=cv2.INTER_NEAREST
        )

    def _project_mask(self, yaw: float, pitch: float, fov_deg: float) -> np.ndarray:
        # project small 360 mask to small perspective mask
        mp = e2p(
            self.mask360_small,
            fov_deg=fov_deg,
            u_deg=yaw,
            v_deg=pitch,
            out_hw=(self.cfg.eval_h, self.cfg.eval_w),
        )
        # e2p may output float or uint8; normalize to uint8 and threshold
        if mp.dtype != np.uint8:
            mp = np.clip(mp, 0, 255).astype(np.uint8)
        return mp

    def evaluate(self, yaw: float, pitch: float, fov_deg: float) -> Tuple[float, float, float, Optional[np.ndarray]]:
        """
        Returns: (score, coverage, center_offset, mask_persp_small)
        center_offset is normalized (0 = perfect center, ~1 = far).
        """
        mp = self._project_mask(yaw, pitch, fov_deg)
        binm = (mp > 127).astype(np.uint8)

        area = float(binm.sum())
        coverage = area / float(self.cfg.eval_w * self.cfg.eval_h)

        if area < 50:  # essentially no pitch visible
            return 0.0, coverage, 1.0, binm

        ys, xs = np.where(binm > 0)
        cx = float(xs.mean())
        cy = float(ys.mean())

        dx = abs(cx - self.cfg.eval_w * 0.5) / (self.cfg.eval_w * 0.5)
        dy = abs(cy - self.cfg.eval_h * 0.5) / (self.cfg.eval_h * 0.5)
        center_offset = 0.5 * (dx + dy)

        # score: prefer coverage + centeredness
        # (1 - center_offset) is better when centered
        score = (
            self.cfg.coverage_weight * coverage
            + self.cfg.center_weight * max(0.0, 1.0 - center_offset)
        )

        return score, coverage, center_offset, binm

    def should_optimize(self, coverage: float, center_offset: float) -> bool:
        return (coverage < self.cfg.min_coverage) or (center_offset > self.cfg.max_center_offset)

    def optimize(self, yaw: float, pitch: float, fov_deg: float) -> Tuple[float, float]:
        """
        Local search around current yaw/pitch to maximize field view score.
        """
        best_yaw, best_pitch = yaw, pitch
        best_score, cov, off, _ = self.evaluate(yaw, pitch, fov_deg)

        # Only optimize when needed
        if not self.should_optimize(cov, off):
            return yaw, pitch

        for dyaw in self.cfg.yaw_offsets:
            for dp in self.cfg.pitch_offsets:
                y2 = yaw + dyaw
                p2 = pitch + dp
                s2, _, _, _ = self.evaluate(y2, p2, fov_deg)
                if s2 > best_score:
                    best_score = s2
                    best_yaw, best_pitch = y2, p2

        return best_yaw, best_pitch
