#!/usr/bin/env python3
"""Scan full equirectangular Insta360 frames for the ball.

This addresses the "camera must already see the ball" failure mode by running
the ball detector over several overlapping perspective views and merging the
results back into longitude/latitude coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from py360convert import e2p

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autopan.perception import DetBox, nms_boxes_xyxy, pick_best_ball
from ball_tracker_equirect import BallMeasurement, MultiHypothesisEquirectTracker, shortest_lon_delta, wrap_lon


@dataclass
class ScanDetection:
    lon: float
    lat: float
    conf: float
    source_yaw: float
    frame_x: float
    frame_y: float
    bbox: Tuple[float, float, float, float]


def parse_scan_yaws(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def angular_dist_deg(a: ScanDetection, b: ScanDetection) -> float:
    return math.hypot(shortest_lon_delta(a.lon, b.lon), a.lat - b.lat)


def merge_scan_detections(detections: Sequence[ScanDetection], merge_dist_deg: float = 1.5) -> List[ScanDetection]:
    """Merge duplicate detections from overlapping perspective views."""
    if not detections:
        return []
    ordered = sorted(detections, key=lambda d: d.conf, reverse=True)
    clusters: List[List[ScanDetection]] = []
    for det in ordered:
        for cluster in clusters:
            if angular_dist_deg(det, cluster[0]) <= merge_dist_deg:
                cluster.append(det)
                break
        else:
            clusters.append([det])

    merged: List[ScanDetection] = []
    for cluster in clusters:
        weights = np.array([max(1e-6, d.conf) for d in cluster], dtype=np.float64)
        lons = np.array([cluster[0].lon + shortest_lon_delta(d.lon, cluster[0].lon) for d in cluster], dtype=np.float64)
        lats = np.array([d.lat for d in cluster], dtype=np.float64)
        best = max(cluster, key=lambda d: d.conf)
        merged.append(
            ScanDetection(
                lon=wrap_lon(float(np.average(lons, weights=weights))),
                lat=float(np.average(lats, weights=weights)),
                conf=float(max(d.conf for d in cluster)),
                source_yaw=best.source_yaw,
                frame_x=best.frame_x,
                frame_y=best.frame_y,
                bbox=best.bbox,
            )
        )
    merged.sort(key=lambda d: d.conf, reverse=True)
    return merged


def derive_tilt_fov(calib_path: str) -> Tuple[float, float]:
    with open(calib_path) as f:
        d = json.load(f)
    raw = d.get("pitch_polygon") or d.get("pixel_polygon") or d.get("auto_polygon")
    if not raw:
        raise ValueError(f"No pitch polygon found in {calib_path}")
    if isinstance(raw[0], dict):
        poly = np.array([[p["x"] * 2880, p["y"] * 1440] for p in raw], dtype=np.float32)
    else:
        poly = np.array(raw, dtype=np.float32)
    tilt_deg = (0.5 - poly[:, 1] / 1440) * 180
    far_tilt = float(np.mean(tilt_deg[: max(2, len(tilt_deg) // 2)]))
    near_tilt = float(np.mean(tilt_deg[max(2, len(tilt_deg) // 2) :]))
    e2p_tilt = ((far_tilt + near_tilt) / 2) * 1.20
    e2p_fov = float(np.clip(abs(near_tilt - far_tilt) * 1.85, 100, 130))
    return e2p_tilt, e2p_fov


def perspective_pixel_to_lon_lat(
    px: float,
    py: float,
    cam_yaw: float,
    cam_pitch: float,
    fov_deg: float,
    width: int,
    height: int,
) -> Tuple[float, float]:
    nx = (float(px) - width / 2.0) / (width / 2.0)
    ny = (float(py) - height / 2.0) / (height / 2.0)
    half_h = math.radians(float(fov_deg) / 2.0)
    half_v = math.atan(math.tan(half_h) * (height / max(1.0, width)))
    lon_delta = math.degrees(math.atan(nx * math.tan(half_h)))
    lat_delta = math.degrees(math.atan(ny * math.tan(half_v)))
    return wrap_lon(float(cam_yaw) + lon_delta), float(cam_pitch) - lat_delta


def lon_lat_to_perspective_pixel(
    lon: float,
    lat: float,
    cam_yaw: float,
    cam_pitch: float,
    fov_deg: float,
    width: int,
    height: int,
) -> Tuple[float, float]:
    dlon = wrap_lon(float(lon) - float(cam_yaw))
    half_h = math.radians(float(fov_deg) / 2.0)
    half_v = math.atan(math.tan(half_h) * (height / max(1.0, width)))
    x = width / 2.0 + (math.tan(math.radians(dlon)) / math.tan(half_h)) * (width / 2.0)
    dlat = float(cam_pitch) - float(lat)
    y = height / 2.0 + (math.tan(math.radians(dlat)) / math.tan(half_v)) * (height / 2.0)
    return x, y


def _tile_grid(w: int, h: int, tile_w: int, tile_h: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    tile_w = max(64, min(int(tile_w), w))
    tile_h = max(64, min(int(tile_h), h))
    ox = int(tile_w * overlap)
    oy = int(tile_h * overlap)
    step_x = max(32, tile_w - ox)
    step_y = max(32, tile_h - oy)
    tiles: List[Tuple[int, int, int, int]] = []
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


def detect_ball_boxes(
    frame_bgr: np.ndarray,
    model,
    device: str,
    imgsz: int,
    conf: float,
) -> List[DetBox]:
    result = model(frame_bgr, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    boxes: List[DetBox] = []
    for b in result.boxes:
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        boxes.append(DetBox(x1=x1, y1=y1, x2=x2, y2=y2, conf=float(b.conf[0]), cls_id=int(b.cls[0])))
    return boxes


def detect_ball_boxes_sliced(
    frame_bgr: np.ndarray,
    model,
    device: str,
    imgsz: int,
    conf: float,
    tile_size: int = 640,
    overlap: float = 0.25,
    batch: int = 8,
) -> List[DetBox]:
    h, w = frame_bgr.shape[:2]
    tiles = _tile_grid(w, h, tile_size, tile_size, overlap)
    tile_imgs = [frame_bgr[y1:y2, x1:x2] for x1, y1, x2, y2 in tiles]
    boxes: List[DetBox] = []
    for i in range(0, len(tile_imgs), max(1, batch)):
        chunk = tile_imgs[i : i + batch]
        chunk_tiles = tiles[i : i + batch]
        results = model(chunk, imgsz=imgsz, conf=conf, device=device, verbose=False)
        for result, (tx1, ty1, _, _) in zip(results, chunk_tiles):
            for b in result.boxes:
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                boxes.append(
                    DetBox(
                        x1=x1 + tx1,
                        y1=y1 + ty1,
                        x2=x2 + tx1,
                        y2=y2 + ty1,
                        conf=float(b.conf[0]),
                        cls_id=int(b.cls[0]),
                    )
                )
    return nms_boxes_xyxy(boxes, iou_thresh=0.45)


def scan_equirect_frame(
    eq_bgr: np.ndarray,
    model,
    device: str,
    scan_yaws: Sequence[float],
    tilt_deg: float,
    fov_deg: float,
    out_w: int = 960,
    out_h: int = 540,
    imgsz: int = 640,
    conf: float = 0.20,
    use_sliced: bool = True,
    merge_dist_deg: float = 1.5,
) -> List[ScanDetection]:
    rgb = cv2.cvtColor(eq_bgr, cv2.COLOR_BGR2RGB)
    detections: List[ScanDetection] = []

    for yaw in scan_yaws:
        persp_rgb = e2p(
            rgb,
            fov_deg=fov_deg,
            u_deg=-float(yaw),
            v_deg=float(tilt_deg),
            out_hw=(out_h, out_w),
            mode="bilinear",
        )
        persp_bgr = cv2.cvtColor(persp_rgb, cv2.COLOR_RGB2BGR)
        boxes = (
            detect_ball_boxes_sliced(persp_bgr, model, device, imgsz, conf)
            if use_sliced
            else detect_ball_boxes(persp_bgr, model, device, imgsz, conf)
        )
        best = pick_best_ball(boxes, max_area_frac=0.02, out_w=out_w, out_h=out_h)
        if best is None:
            continue
        lon, lat = perspective_pixel_to_lon_lat(best.cx, best.cy, yaw, tilt_deg, fov_deg, out_w, out_h)
        detections.append(
            ScanDetection(
                lon=lon,
                lat=lat,
                conf=float(best.conf),
                source_yaw=float(yaw),
                frame_x=float(best.cx),
                frame_y=float(best.cy),
                bbox=(float(best.x1), float(best.y1), float(best.x2), float(best.y2)),
            )
        )

    return merge_scan_detections(detections, merge_dist_deg=merge_dist_deg)


def open_equirect_stream(insv_path: str, start_s: float, duration_s: float):
    cmd = [
        "ffmpeg",
        "-ss",
        str(max(0.0, start_s)),
        "-i",
        insv_path,
        "-t",
        str(duration_s),
        "-vf",
        "rotate=PI/2*3,v360=fisheye:equirect:ih_fov=200:iv_fov=200,scale=2880:1440",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-an",
        "pipe:1",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def read_equirect_frame(proc, w: int = 2880, h: int = 1440) -> Optional[np.ndarray]:
    nbytes = w * h * 3
    data = proc.stdout.read(nbytes)
    if len(data) < nbytes:
        return None
    return np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3)).copy()


def _best_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--insv", required=True, help="Input .insv file")
    parser.add_argument("--calib", required=True, help="Calibration JSON")
    parser.add_argument("--ball", default="models/ball_v4.pt", help="Ball model path")
    parser.add_argument("--device", default=None, help="cpu, cuda, or mps")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--every", type=int, default=15, help="Scan every N decoded frames")
    parser.add_argument("--scan-yaws", default="-45,-30,-15,0,15,30,45")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--out-w", type=int, default=960)
    parser.add_argument("--out-h", type=int, default=540)
    parser.add_argument("--merge-dist-deg", type=float, default=1.5)
    parser.add_argument("--no-sliced", action="store_true", help="Disable SAHI-style tiled inference")
    parser.add_argument("--out-jsonl", default=None, help="Optional JSONL detections output")
    args = parser.parse_args()

    if not Path(args.insv).exists():
        raise FileNotFoundError(args.insv)
    if not Path(args.calib).exists():
        raise FileNotFoundError(args.calib)
    if not Path(args.ball).exists():
        raise FileNotFoundError(args.ball)

    device = args.device or _best_device()
    tilt, fov = derive_tilt_fov(args.calib)
    scan_yaws = parse_scan_yaws(args.scan_yaws)

    from ultralytics import YOLO

    model = YOLO(args.ball)
    tracker = MultiHypothesisEquirectTracker()
    proc = open_equirect_stream(args.insv, args.start, args.duration)
    out_f = open(args.out_jsonl, "w", encoding="utf-8") if args.out_jsonl else None

    frame_idx = 0
    scanned = 0
    hits = 0
    try:
        while True:
            frame = read_equirect_frame(proc)
            if frame is None:
                break
            if frame_idx % max(1, args.every) == 0:
                scanned += 1
                dets = scan_equirect_frame(
                    frame,
                    model,
                    device,
                    scan_yaws=scan_yaws,
                    tilt_deg=tilt,
                    fov_deg=fov,
                    out_w=args.out_w,
                    out_h=args.out_h,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    use_sliced=not args.no_sliced,
                    merge_dist_deg=args.merge_dist_deg,
                )
                tracker.predict(max(1, args.every))
                tracker.update(
                    BallMeasurement(d.lon, d.lat, d.conf, source=f"scan:{d.source_yaw}")
                    for d in dets[:3]
                )
                snap = tracker.snapshot()
                if dets:
                    hits += 1
                row = {
                    "frame_idx": frame_idx,
                    "time_s": args.start + frame_idx / 29.97,
                    "detections": [asdict(d) for d in dets],
                    "tracker": asdict(snap) if snap else None,
                }
                if out_f:
                    out_f.write(json.dumps(row) + "\n")
                best = dets[0] if dets else None
                if best:
                    print(
                        f"frame={frame_idx} lon={best.lon:+.1f} lat={best.lat:+.1f} "
                        f"conf={best.conf:.2f} yaw={best.source_yaw:+.0f}"
                    )
                else:
                    print(f"frame={frame_idx} no ball")
            frame_idx += 1
    finally:
        if out_f:
            out_f.close()
        if proc.stdout:
            proc.stdout.close()
        proc.wait()

    rate = 100.0 * hits / max(1, scanned)
    print(f"Scanned {scanned} frames, detections in {hits} ({rate:.1f}%).")


if __name__ == "__main__":
    main()
