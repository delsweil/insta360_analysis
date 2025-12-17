# src/autopan/team_track.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import Counter
import numpy as np


@dataclass
class TrackConfig:
    iou_match_thresh: float = 0.30
    max_age_frames: int = 30          # drop track if unseen for this many frames
    history_len: int = 15             # label history length per track
    min_votes: int = 6                # need at least this many labels before we trust
    majority_frac: float = 0.60       # require >= this share of votes


@dataclass
class Track:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    labels: List[str]
    last_seen: int

    def update_box(self, b) -> None:
        self.x1, self.y1, self.x2, self.y2 = float(b.x1), float(b.y1), float(b.x2), float(b.y2)

    def xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


def _iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1e-9, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-9, (bx2 - bx1) * (by2 - by1))
    return float(inter / (area_a + area_b - inter + 1e-9))


class TeamLabelTracker:
    """
    Ultra-lightweight IoU matcher that stabilizes per-box team labels.

    Input per frame:
      - boxes: list of DetBox-like objects (x1,y1,x2,y2)
      - raw_labels: list[Optional[str]] aligned with boxes (None=unknown)

    Output per frame:
      - stable_labels: list[str] aligned with boxes, values in {"team0","team1","unknown"}
      - track_ids: list[int] aligned with boxes
    """
    def __init__(self, cfg: TrackConfig = TrackConfig()):
        self.cfg = cfg
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    def _stable_label(self, tr: Track) -> str:
        if len(tr.labels) < self.cfg.min_votes:
            return "unknown"
        c = Counter(tr.labels)
        lab, cnt = c.most_common(1)[0]
        if cnt / max(1, len(tr.labels)) < self.cfg.majority_frac:
            return "unknown"
        return lab

    def update(
        self,
        frame_idx: int,
        boxes: List,
        raw_labels: List[Optional[str]],
    ) -> Tuple[List[str], List[int]]:
        assert len(boxes) == len(raw_labels)

        # --- match boxes to existing tracks by greedy IoU ---
        used_tracks = set()
        matches: Dict[int, int] = {}  # box_idx -> track_id

        for i, b in enumerate(boxes):
            b_xyxy = (float(b.x1), float(b.y1), float(b.x2), float(b.y2))
            best_iou = 0.0
            best_tid = None
            for tid, tr in self.tracks.items():
                if tid in used_tracks:
                    continue
                iou = _iou_xyxy(tr.xyxy(), b_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None and best_iou >= self.cfg.iou_match_thresh:
                matches[i] = best_tid
                used_tracks.add(best_tid)

        # --- update / create tracks ---
        track_ids: List[int] = []
        stable_labels: List[str] = []

        for i, b in enumerate(boxes):
            lab = raw_labels[i]
            if i in matches:
                tid = matches[i]
                tr = self.tracks[tid]
                tr.update_box(b)
                tr.last_seen = frame_idx
                if lab is not None:
                    tr.labels.append(lab)
                    tr.labels = tr.labels[-self.cfg.history_len:]
            else:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = Track(
                    track_id=tid,
                    x1=float(b.x1), y1=float(b.y1), x2=float(b.x2), y2=float(b.y2),
                    labels=[lab] if lab is not None else [],
                    last_seen=frame_idx,
                )

            tr = self.tracks[tid]
            track_ids.append(tid)
            stable_labels.append(self._stable_label(tr))

        # --- prune old tracks ---
        dead = []
        for tid, tr in self.tracks.items():
            if frame_idx - tr.last_seen > self.cfg.max_age_frames:
                dead.append(tid)
        for tid in dead:
            del self.tracks[tid]

        return stable_labels, track_ids
