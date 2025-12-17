# src/autopan/tracking.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import itertools
import numpy as np


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))

    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    age: int = 0
    missed: int = 0
    team_votes: list = field(default_factory=list)

    def update(self, bbox):
        self.bbox = bbox
        self.age += 1
        self.missed = 0

    def mark_missed(self):
        self.missed += 1
        self.age += 1


class TrackManager:
    def __init__(
        self,
        iou_thresh: float = 0.3,
        max_missed: int = 10,
        vote_window: int = 15,
        vote_ratio: float = 0.7,
    ):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.vote_window = vote_window
        self.vote_ratio = vote_ratio

        self._next_id = itertools.count(1)
        self.tracks: Dict[int, Track] = {}

    def update(self, boxes: List[Tuple[int, int, int, int]]):
        """
        boxes: list of (x1,y1,x2,y2)
        returns: list of track_ids aligned with boxes
        """
        assigned_tracks = {}
        used_tracks = set()

        for i, box in enumerate(boxes):
            best_iou = 0.0
            best_tid = None

            for tid, tr in self.tracks.items():
                if tid in used_tracks:
                    continue
                s = iou(box, tr.bbox)
                if s > best_iou:
                    best_iou = s
                    best_tid = tid

            if best_tid is not None and best_iou >= self.iou_thresh:
                assigned_tracks[i] = best_tid
                used_tracks.add(best_tid)
                self.tracks[best_tid].update(box)
            else:
                tid = next(self._next_id)
                self.tracks[tid] = Track(tid, box)
                assigned_tracks[i] = tid

        # mark missed tracks
        for tid, tr in list(self.tracks.items()):
            if tid not in used_tracks and tid not in assigned_tracks.values():
                tr.mark_missed()
                if tr.missed > self.max_missed:
                    del self.tracks[tid]

        return [assigned_tracks[i] for i in range(len(boxes))]

    def vote(self, track_id: int, label: str | None):
        """
        Add a team vote for a track (label can be None/unknown).
        """
        tr = self.tracks.get(track_id)
        if tr is None or label is None:
            return

        tr.team_votes.append(label)
        if len(tr.team_votes) > self.vote_window:
            tr.team_votes.pop(0)

    def stable_label(self, track_id: int) -> str | None:
        tr = self.tracks.get(track_id)
        if tr is None or not tr.team_votes:
            return None

        counts = {}
        for v in tr.team_votes:
            counts[v] = counts.get(v, 0) + 1

        label, n = max(counts.items(), key=lambda x: x[1])
        if n / len(tr.team_votes) >= self.vote_ratio:
            return label
        return None
