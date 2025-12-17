# src/autopan/perception.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cv2
from ultralytics import YOLO


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class DetBox:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 0.0
    cls_id: int = -1

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

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)


# ----------------------------
# Helpers: IoU + NMS + tiling
# ----------------------------

def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: (4,) float
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1])))
    area_b = max(0.0, (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])))
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)


def nms_boxes_xyxy(
    boxes: List[DetBox],
    iou_thresh: float = 0.55,
) -> List[DetBox]:
    """
    Plain IoU NMS. Good generic default.
    """
    if not boxes:
        return []

    boxes_sorted = sorted(boxes, key=lambda b: float(b.conf), reverse=True)
    kept: List[DetBox] = []
    kept_xyxy: List[np.ndarray] = []

    for b in boxes_sorted:
        bb = np.array([b.x1, b.y1, b.x2, b.y2], dtype=np.float32)
        ok = True
        for kb in kept_xyxy:
            if _iou_xyxy(bb, kb) >= iou_thresh:
                ok = False
                break
        if ok:
            kept.append(b)
            kept_xyxy.append(bb)

    return kept


def nms_people_xyxy(
    boxes: List[DetBox],
    iou_thresh: float = 0.55,
    center_dist_frac: float = 0.35,
) -> List[DetBox]:
    """
    Player-friendly NMS: suppress if either
      - IoU is high, OR
      - centers are very close relative to player height (handles loose vs tight boxes)

    This is useful when pass1 and pass2 produce slightly shifted boxes that
    don't reach IoU threshold but are clearly the same player.
    """
    if not boxes:
        return []

    boxes_sorted = sorted(boxes, key=lambda b: float(b.conf), reverse=True)
    kept: List[DetBox] = []

    for b in boxes_sorted:
        bb = np.array([b.x1, b.y1, b.x2, b.y2], dtype=np.float32)
        bc = np.array([b.cx, b.cy], dtype=np.float32)
        bh = max(1.0, b.h)

        keep = True
        for kb in kept:
            kb_arr = np.array([kb.x1, kb.y1, kb.x2, kb.y2], dtype=np.float32)
            iou = _iou_xyxy(bb, kb_arr)
            if iou >= iou_thresh:
                keep = False
                break

            kc = np.array([kb.cx, kb.cy], dtype=np.float32)
            kh = max(1.0, kb.h)
            dist = float(np.linalg.norm(bc - kc))
            thresh = float(center_dist_frac) * float(min(bh, kh))
            if dist <= thresh:
                keep = False
                break

        if keep:
            kept.append(b)

    return kept


def _tile_grid(w: int, h: int, tile_w: int, tile_h: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    """
    Returns list of (x1,y1,x2,y2) tiles covering the image.
    overlap in [0..0.5]
    """
    tile_w = max(64, min(tile_w, w))
    tile_h = max(64, min(tile_h, h))
    ox = int(tile_w * overlap)
    oy = int(tile_h * overlap)
    step_x = max(32, tile_w - ox)
    step_y = max(32, tile_h - oy)

    tiles = []
    y = 0
    while True:
        x = 0
        y2 = min(h, y + tile_h)
        y1 = max(0, y2 - tile_h)
        while True:
            x2 = min(w, x + tile_w)
            x1 = max(0, x2 - tile_w)
            tiles.append((x1, y1, x2, y2))
            if x2 >= w:
                break
            x += step_x
        if y2 >= h:
            break
        y += step_y
    return tiles


# ----------------------------
# Optional: y-adaptive filtering
# ----------------------------

def filter_by_y_adaptive_area(
    boxes: List[DetBox],
    img_w: int,
    img_h: int,
    far_min_area_frac: float = 0.00008,
    near_min_area_frac: float = 0.00050,
    power: float = 2.0,
) -> List[DetBox]:
    """
    Keep smaller boxes near the top of the image (far players),
    require larger boxes near the bottom (near players).

    This helps if downstream code is otherwise tempted to globally drop "small" detections.
    """
    if not boxes:
        return []

    denom = float(max(1, img_w * img_h))
    out: List[DetBox] = []
    for b in boxes:
        y = float(np.clip(b.cy / max(1, img_h), 0.0, 1.0))  # 0=top, 1=bottom
        min_frac = float(far_min_area_frac) + (float(near_min_area_frac) - float(far_min_area_frac)) * (y ** float(power))
        if (b.area / denom) >= min_frac:
            out.append(b)
    return out


# ----------------------------
# Detector wrapper
# ----------------------------

class Detector:
    def __init__(self, player_model_path: str, ball_model_path: str):
        self.player_model = YOLO(player_model_path)
        self.ball_model = YOLO(ball_model_path)

        # Cache names dicts (Ultralytics)
        self.player_names = getattr(self.player_model, "names", {})
        self.ball_names = getattr(self.ball_model, "names", {})

    def _infer(self, model: YOLO, img_bgr: np.ndarray, imgsz: int, conf: float):
        return model(img_bgr, imgsz=imgsz, conf=conf, verbose=False)[0]

    # ------------------------
    # Base (single-pass) detections
    # ------------------------

    def detect_players(self, img_bgr: np.ndarray, imgsz: int = 960, conf: float = 0.35) -> List[DetBox]:
        res = self._infer(self.player_model, img_bgr, imgsz=imgsz, conf=conf)
        boxes: List[DetBox] = []
        for b in res.boxes:
            cls_id = int(b.cls[0])
            cls_name = self.player_names.get(cls_id, str(cls_id))
            # COCO "person"
            if cls_name != "person":
                continue
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            cf = float(b.conf[0])
            boxes.append(DetBox(x1, y1, x2, y2, conf=cf, cls_id=cls_id))
        return boxes

    def detect_ball(self, img_bgr: np.ndarray, imgsz: int = 640, conf: float = 0.20) -> List[DetBox]:
        res = self._infer(self.ball_model, img_bgr, imgsz=imgsz, conf=conf)
        boxes: List[DetBox] = []
        for b in res.boxes:
            cls_id = int(b.cls[0])
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            cf = float(b.conf[0])
            boxes.append(DetBox(x1, y1, x2, y2, conf=cf, cls_id=cls_id))
        return boxes

    # ------------------------
    # Two-pass small-player detection
    # ------------------------

    def detect_players_two_pass(
        self,
        img_bgr: np.ndarray,
        imgsz_lo: int = 960,
        imgsz_hi: int = 1280,
        conf: float = 0.35,
        conf_p2: Optional[float] = None,
        # NMS merge across tiles + pass1
        nms_iou: float = 0.55,
        center_dist_frac: float = 0.35,
        # tiles
        tile_w: int = 960,
        tile_h: int = 540,
        overlap: float = 0.20,
        # when to trigger pass2
        trigger_min_players: int = 10,
        trigger_median_area_frac: float = 0.0025,
        # optional pitch mask in perspective coords
        mask_bin: Optional[np.ndarray] = None,
        tile_pitch_min_coverage: float = 0.05,
        # limit pass2 to far band (top part of image)
        p2_ymax_frac: float = 0.55,
        # optional throttle
        every_n_frames: int = 1,
        frame_index: Optional[int] = None,
        # optional y-adaptive min-area filtering (applied at the very end)
        apply_y_adaptive_filter: bool = False,
        far_min_area_frac: float = 0.00008,
        near_min_area_frac: float = 0.00050,
        y_area_power: float = 2.0,
    ) -> List[DetBox]:
        """
        Pass1: full frame @ imgsz_lo.
        Trigger pass2 if:
          - too few players OR
          - median bbox area too small (players are far away / zoomed out)

        Pass2: tiled inference (restricted to top p2_ymax_frac band), merged via NMS.

        mask_bin (H,W bool) can be used to skip tiles with little pitch content.
        """

        h, w = img_bgr.shape[:2]

        # ---- pass 1 ----
        p1 = self.detect_players(img_bgr, imgsz=imgsz_lo, conf=conf)

        # decide if pass2 should run
        run_p2 = False
        if len(p1) < trigger_min_players:
            run_p2 = True
        else:
            areas = np.array([b.area for b in p1], dtype=np.float32)
            if len(areas) > 0:
                med_area = float(np.median(areas))
                med_area_frac = med_area / float(w * h)
                if med_area_frac < trigger_median_area_frac:
                    run_p2 = True

        # throttle pass2 if requested
        if run_p2 and every_n_frames > 1 and frame_index is not None:
            if (frame_index % every_n_frames) != 0:
                run_p2 = False

        if not run_p2:
            if apply_y_adaptive_filter:
                return filter_by_y_adaptive_area(
                    p1, w, h,
                    far_min_area_frac=far_min_area_frac,
                    near_min_area_frac=near_min_area_frac,
                    power=y_area_power,
                )
            return p1

        # ---- pass 2: tiles (restricted to far band) ----
        p2_ymax = int(max(64, min(h, round(h * float(p2_ymax_frac)))))
        p2_conf = float(conf if conf_p2 is None else conf_p2)

        tiles = _tile_grid(w, p2_ymax, tile_w, tile_h, overlap)
        p2: List[DetBox] = []

        for (x1, y1, x2, y2) in tiles:
            # if pitch mask is given, skip tiles with tiny pitch coverage
            if mask_bin is not None:
                mh, mw = mask_bin.shape[:2]
                # mask_bin should match img size; if not, just ignore
                if (mw == w) and (mh == h):
                    cov = float(mask_bin[y1:y2, x1:x2].mean())
                    if cov < tile_pitch_min_coverage:
                        continue

            tile = img_bgr[y1:y2, x1:x2]
            if tile.size == 0:
                continue

            dets = self.detect_players(tile, imgsz=imgsz_hi, conf=p2_conf)
            # map back to full-frame coordinates
            for b in dets:
                p2.append(
                    DetBox(
                        x1=b.x1 + x1,
                        y1=b.y1 + y1,
                        x2=b.x2 + x1,
                        y2=b.y2 + y1,
                        conf=b.conf,
                        cls_id=b.cls_id,
                    )
                )

        # ---- merge pass1 + pass2 ----
        merged = p1 + p2
        merged = nms_people_xyxy(
            merged,
            iou_thresh=nms_iou,
            center_dist_frac=center_dist_frac,
        )

        if apply_y_adaptive_filter:
            merged = filter_by_y_adaptive_area(
                merged, w, h,
                far_min_area_frac=far_min_area_frac,
                near_min_area_frac=near_min_area_frac,
                power=y_area_power,
            )

        return merged


# ----------------------------
# Ball selection helper
# ----------------------------

def pick_best_ball(
    ball_boxes: List[DetBox],
    max_area_frac: float,
    out_w: int,
    out_h: int
) -> Optional[DetBox]:
    """
    Pick best ball by (conf, area) descending, with a simple max-area filter.
    """
    if not ball_boxes:
        return None

    max_area = float(out_w * out_h) * float(max_area_frac)
    candidates = []
    for b in ball_boxes:
        if b.area <= max_area:
            candidates.append(b)

    if not candidates:
        return None

    candidates.sort(key=lambda b: (float(b.conf), float(b.area)), reverse=True)
    return candidates[0]
