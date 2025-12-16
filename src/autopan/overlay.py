# src/autopan/overlay.py
import cv2
import numpy as np
from typing import List, Tuple


def draw_boxes(img, boxes_xyxy: List[Tuple[int,int,int,int]], color=(0,255,0), thickness=2):
    for (x1,y1,x2,y2) in boxes_xyxy:
        cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), color, thickness)


def draw_circle(img, xy: Tuple[int,int], r=5, color=(255,255,255), thickness=2):
    cv2.circle(img, (int(xy[0]), int(xy[1])), int(r), color, int(thickness))


def draw_text(img, text: str, org: Tuple[int,int], scale=0.7, color=(255,255,255), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, float(scale), color, int(thickness), cv2.LINE_AA)


def draw_mask_contour(img, mask: np.ndarray, color=(255,255,0), thickness=2):
    """
    Draw the outline of a binary mask (mask in perspective coords).
    """
    if mask is None:
        return
    m = (mask > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(img, cnts, -1, color, thickness)
