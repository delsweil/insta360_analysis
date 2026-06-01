#!/usr/bin/env python3
"""Team classification wrapper around jersey colour clustering and IoU tracks."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autopan.team_cluster import TeamClusterConfig, TeamClusterer
from autopan.tracking import TrackManager


@dataclass
class TeamAssignment:
    track_id: int
    label: Optional[str]
    raw_label: Optional[str]
    margin: Optional[float]
    best_dist: Optional[float]
    second_dist: Optional[float]


class TeamClassifier:
    """Stable team labels for player boxes in perspective frames."""

    def __init__(
        self,
        cluster_cfg: TeamClusterConfig | None = None,
        iou_thresh: float = 0.30,
        max_missed: int = 10,
        vote_window: int = 15,
        vote_ratio: float = 0.65,
        min_votes: int = 4,
    ):
        self.clusterer = TeamClusterer(cluster_cfg or TeamClusterConfig())
        self.tracker = TrackManager(
            iou_thresh=iou_thresh,
            max_missed=max_missed,
            vote_window=vote_window,
            vote_ratio=vote_ratio,
            min_votes=min_votes,
        )

    @property
    def ready(self) -> bool:
        return self.clusterer.ready

    def update(self, frame_bgr: np.ndarray, boxes: Sequence) -> List[TeamAssignment]:
        if not boxes:
            return []

        if not self.clusterer.ready:
            self.clusterer.update_with_boxes(frame_bgr, list(boxes))

        bboxes = [(int(b.x1), int(b.y1), int(b.x2), int(b.y2)) for b in boxes]
        track_ids = self.tracker.update(bboxes)

        if not self.clusterer.ready:
            return [
                TeamAssignment(tid, None, None, None, None, None)
                for tid in track_ids
            ]

        raw = self.clusterer.classify_boxes_with_margins(frame_bgr, list(boxes))
        assignments: List[TeamAssignment] = []
        for tid, (label, margin, best, second) in zip(track_ids, raw):
            if label is not None:
                self.tracker.vote(tid, label)
            assignments.append(
                TeamAssignment(
                    track_id=tid,
                    label=self.tracker.stable_label(tid),
                    raw_label=label,
                    margin=margin,
                    best_dist=best,
                    second_dist=second,
                )
            )
        return assignments
