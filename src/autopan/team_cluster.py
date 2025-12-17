# src/autopan/team_cluster.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import cv2


@dataclass
class TeamClusterConfig:
    # Collect features for N frames, then k-means once
    bootstrap_frames: int = 250  # ~8 seconds at 30fps
    k: int = 2  # team0/team1 only (ref is not supported reliably yet)

    # Online center update
    ema_beta: float = 0.02
    update_dist_thresh: float = 2.5  # only update if close enough

    # Torso ROI (fractions of bbox)
    torso_y1: float = 0.15
    torso_y2: float = 0.55
    torso_xpad: float = 0.20

    # Pixel quality gates
    min_non_grass_frac: float = 0.18
    min_roi_px: int = 20 * 20
    # Make small/far players usable: normalize ROI size before feature extraction.
    roi_resize: int = 48  # 0 disables resizing
    # When ROIs get tiny, hard pixel-count gates become too strict. Keep this low.
    min_kept_px: int = 20

    # Grass mask in HSV
    grass_h_lo: int = 35
    grass_h_hi: int = 95
    grass_s_min: int = 40
    grass_v_min: int = 40

    # Feature settings
    hsv_h_bins: int = 18
    hsv_s_bins: int = 6
    hsv_v_bins: int = 6


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, v)))


def _torso_crop(img: np.ndarray, b, cfg: TeamClusterConfig) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)

    x1 = _clamp_int(x1, 0, w - 1)
    x2 = _clamp_int(x2, 0, w - 1)
    y1 = _clamp_int(y1, 0, h - 1)
    y2 = _clamp_int(y2, 0, h - 1)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    ry1 = y1 + int(cfg.torso_y1 * bh)
    ry2 = y1 + int(cfg.torso_y2 * bh)
    xpad = int(cfg.torso_xpad * bw)
    rx1 = x1 + xpad
    rx2 = x2 - xpad

    rx1 = _clamp_int(rx1, 0, w - 1)
    rx2 = _clamp_int(rx2, 0, w - 1)
    ry1 = _clamp_int(ry1, 0, h - 1)
    ry2 = _clamp_int(ry2, 0, h - 1)

    if rx2 <= rx1 or ry2 <= ry1:
        return None
    return img[ry1:ry2, rx1:rx2]


def _mask_grass_hsv(roi_hsv: np.ndarray, cfg: TeamClusterConfig) -> np.ndarray:
    """Return mask of *non-grass* pixels given an HSV ROI."""
    H, S, V = roi_hsv[:, :, 0], roi_hsv[:, :, 1], roi_hsv[:, :, 2]
    grass = (
        (H >= cfg.grass_h_lo) & (H <= cfg.grass_h_hi) &
        (S >= cfg.grass_s_min) &
        (V >= cfg.grass_v_min)
    )
    return ~grass  # keep non-grass


def _maybe_resize(roi: np.ndarray, cfg: TeamClusterConfig) -> np.ndarray:
    if cfg.roi_resize and cfg.roi_resize > 0:
        return cv2.resize(roi, (cfg.roi_resize, cfg.roi_resize), interpolation=cv2.INTER_LINEAR)
    return roi


def _feature_from_rois(
    roi_lab: np.ndarray,
    roi_hsv: np.ndarray,
    cfg: TeamClusterConfig,
) -> Optional[np.ndarray]:
    """Extract a normalized kit feature vector from LAB+HSV torso ROIs."""

    if roi_hsv.size < 3 * cfg.min_roi_px:
        return None

    keep_mask = _mask_grass_hsv(roi_hsv, cfg)

    keep_frac = float(keep_mask.mean())
    if keep_frac < cfg.min_non_grass_frac:
        return None

    lab_kept = roi_lab[keep_mask]
    if lab_kept.shape[0] < cfg.min_kept_px:
        return None

    lab_mean = lab_kept.mean(axis=0)
    lab_std = lab_kept.std(axis=0)

    hsv_kept = roi_hsv[keep_mask]
    if hsv_kept.shape[0] < cfg.min_kept_px:
        return None

    h = hsv_kept[:, 0].astype(np.float32)
    s = hsv_kept[:, 1].astype(np.float32)
    v = hsv_kept[:, 2].astype(np.float32)

    h_hist, _ = np.histogram(h, bins=cfg.hsv_h_bins, range=(0, 180), density=True)
    s_hist, _ = np.histogram(s, bins=cfg.hsv_s_bins, range=(0, 256), density=True)
    v_hist, _ = np.histogram(v, bins=cfg.hsv_v_bins, range=(0, 256), density=True)

    feat = np.concatenate([lab_mean, lab_std, h_hist, s_hist, v_hist]).astype(np.float32)
    feat = feat / (np.linalg.norm(feat) + 1e-8)
    return feat


def extract_kit_feature(
    frame_bgr: np.ndarray,
    b,
    cfg: TeamClusterConfig,
    frame_lab: Optional[np.ndarray] = None,
    frame_hsv: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Extract a kit feature for one box.

    Speed wins:
      - pass precomputed frame_lab/frame_hsv (computed once per frame)
      - avoid per-box cvtColor calls

    Robustness wins:
      - resize torso ROI to a fixed size for far/small players
    """

    if frame_lab is None:
        frame_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    if frame_hsv is None:
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    roi_lab = _torso_crop(frame_lab, b, cfg)
    roi_hsv = _torso_crop(frame_hsv, b, cfg)
    if roi_lab is None or roi_hsv is None:
        return None

    roi_lab = _maybe_resize(roi_lab, cfg)
    roi_hsv = _maybe_resize(roi_hsv, cfg)
    return _feature_from_rois(roi_lab, roi_hsv, cfg)


def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n, d = X.shape
    C = np.empty((k, d), dtype=X.dtype)

    i0 = rng.integers(0, n)
    C[0] = X[i0]

    dist2 = np.full(n, np.inf, dtype=np.float32)
    for i in range(1, k):
        d2 = np.sum((X - C[i - 1]) ** 2, axis=1)
        dist2 = np.minimum(dist2, d2)
        probs = dist2 / (dist2.sum() + 1e-12)
        idx = rng.choice(n, p=probs)
        C[i] = X[idx]

    return C


def _run_kmeans(X: np.ndarray, k: int, iters: int = 35, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    C = _kmeans_pp_init(X, k, rng)

    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)

        newC = C.copy()
        for j in range(k):
            sel = X[labels == j]
            if len(sel) > 0:
                newC[j] = sel.mean(axis=0)

        newC = newC / (np.linalg.norm(newC, axis=1, keepdims=True) + 1e-8)

        if np.max(np.abs(newC - C)) < 1e-4:
            C = newC
            break
        C = newC

    return C, labels


class TeamClusterer:
    """
    Unsupervised kit clustering: bootstrap k=2 then online assignment + slow updates.
    Outputs labels: 'team0'/'team1'
    """
    def __init__(self, cfg: TeamClusterConfig = TeamClusterConfig()):
        self.cfg = cfg
        self._boot_frames = 0
        self._boot_feats: List[np.ndarray] = []

        self.centers: Optional[np.ndarray] = None  # (k, d)
        self.counts = np.zeros(cfg.k, dtype=np.int64)

    @property
    def ready(self) -> bool:
        return self.centers is not None

    def _label_from_cluster(self, j: int) -> str:
        # deterministic: cluster 0->team0, 1->team1
        return f"team{j}"

    def update_with_boxes(self, frame_bgr: np.ndarray, boxes: List) -> None:
        """
        Collect bootstrap features. Call once per frame.
        """
        if self.ready:
            return

        self._boot_frames += 1

        # Precompute color spaces once per frame (big speed win)
        frame_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        feats = []
        for b in boxes:
            f = extract_kit_feature(frame_bgr, b, self.cfg, frame_lab=frame_lab, frame_hsv=frame_hsv)
            if f is not None:
                feats.append(f)
        if feats:
            self._boot_feats.extend(feats)

        # bootstrap condition
        if self._boot_frames >= self.cfg.bootstrap_frames and len(self._boot_feats) >= 60:
            X = np.stack(self._boot_feats, axis=0)
            C, labels = _run_kmeans(X, self.cfg.k, iters=35, seed=0)
            self.centers = C

            self.counts[:] = 0
            for j in labels:
                self.counts[int(j)] += 1

            self._boot_feats = []
            print(f"[TeamCluster] READY k_teams={self.cfg.k} counts={self.counts.tolist()}")

    def classify_box_with_margin(self, frame_bgr: np.ndarray, b):
        """
        Returns (label, margin, best_dist, second_dist)
        margin = second_best_dist - best_dist (higher => more confident)
        """
        if not self.ready or self.centers is None:
            return None, None, None, None

        frame_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        f = extract_kit_feature(frame_bgr, b, self.cfg, frame_lab=frame_lab, frame_hsv=frame_hsv)
        if f is None:
            return None, None, None, None

        d2 = ((self.centers - f[None, :]) ** 2).sum(axis=1)
        order = np.argsort(d2)
        j0 = int(order[0])
        j1 = int(order[1]) if len(order) > 1 else j0

        best = float(np.sqrt(d2[j0]))
        second = float(np.sqrt(d2[j1]))
        margin = float(second - best)

        # book-keeping
        self.counts[j0] += 1

        # optional slow centroid update (only if close enough)
        dist = best
        if dist < self.cfg.update_dist_thresh:
            beta = self.cfg.ema_beta
            self.centers[j0] = (1.0 - beta) * self.centers[j0] + beta * f
            self.centers[j0] = self.centers[j0] / (np.linalg.norm(self.centers[j0]) + 1e-8)

        return self._label_from_cluster(j0), margin, best, second

    def classify_boxes_with_margins(self, frame_bgr: np.ndarray, boxes: List):
        if not boxes:
            return []
        # One conversion per frame instead of per box
        frame_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        out = []
        for b in boxes:
            if not self.ready or self.centers is None:
                out.append((None, None, None, None))
                continue
            f = extract_kit_feature(frame_bgr, b, self.cfg, frame_lab=frame_lab, frame_hsv=frame_hsv)
            if f is None:
                out.append((None, None, None, None))
                continue

            d2 = ((self.centers - f[None, :]) ** 2).sum(axis=1)
            order = np.argsort(d2)
            j0 = int(order[0])
            j1 = int(order[1]) if len(order) > 1 else j0

            best = float(np.sqrt(d2[j0]))
            second = float(np.sqrt(d2[j1]))
            margin = float(second - best)

            self.counts[j0] += 1

            if best < self.cfg.update_dist_thresh:
                beta = self.cfg.ema_beta
                self.centers[j0] = (1.0 - beta) * self.centers[j0] + beta * f
                self.centers[j0] = self.centers[j0] / (np.linalg.norm(self.centers[j0]) + 1e-8)

            out.append((self._label_from_cluster(j0), margin, best, second))

        return out
