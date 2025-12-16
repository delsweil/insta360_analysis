from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from ultralytics import YOLO

@dataclass
class Box:
    x1: int; y1: int; x2: int; y2: int
    conf: float

    @property
    def cx(self) -> int: return (self.x1 + self.x2) // 2
    @property
    def cy(self) -> int: return (self.y1 + self.y2) // 2
    @property
    def foot_y(self) -> int: return self.y2
    @property
    def area(self) -> int: return max(1, (self.x2 - self.x1) * (self.y2 - self.y1))

class Detector:
    def __init__(self, player_model_path: str, ball_model_path: str):
        self.player_model = YOLO(player_model_path)
        self.ball_model = YOLO(ball_model_path)

    def detect_players(self, frame: np.ndarray, imgsz: int, conf: float) -> List[Box]:
        res = self.player_model(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
        names = self.player_model.names
        out: List[Box] = []
        for b in res.boxes:
            cls_id = int(b.cls[0])
            cls_name = names.get(cls_id, str(cls_id))
            if cls_name != "person":
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            out.append(Box(x1, y1, x2, y2, float(b.conf[0])))
        return out

    def detect_ball(self, frame: np.ndarray, imgsz: int, conf: float) -> List[Box]:
        res = self.ball_model(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
        out: List[Box] = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            out.append(Box(x1, y1, x2, y2, float(b.conf[0])))
        return out

def pick_best_ball(cands: List[Box], max_area_frac: float, out_w: int, out_h: int) -> Optional[Box]:
    if not cands:
        return None
    max_area = out_w * out_h * max_area_frac
    filtered = [b for b in cands if b.area <= max_area]
    if not filtered:
        return None
    filtered.sort(key=lambda b: (b.conf, b.area), reverse=True)
    return filtered[0]
