"""
autopan_simple.py
-----------------
Streamlined autopan pipeline — single-pass YOLO detection, no team
clustering, no multi-pass tiling, FFmpeg pipe output.

Designed to run fast on Apple Silicon (M4/M5) with MPS acceleration.
On Intel Mac: ~0.5-1 fps (usable overnight for a full game)
On M4 Mac Mini: ~15-25 fps (full game in 15-20 minutes)

Usage:
    python autopan_simple.py \
        --input  /path/to/VID_20250701_174151_00_002.insv \
        --output /path/to/output.mp4 \
        --calib  calibration/pitch.json \
        --players models/yolo11n.pt \
        --ball    models/ball.pt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── Try to import py360convert ──────────────────────────────────
try:
    from py360convert import e2p
    HAS_PY360 = True
except ImportError:
    HAS_PY360 = False
    print("[WARN] py360convert not installed: pip install py360convert")

# ── Try to use MPS (Apple Silicon) or CUDA, fall back to CPU ────
def _best_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ──────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # I/O
    input_path: str = ""
    output_path: str = "output_simple.mp4"
    calib_path: str = "calibration/pitch.json"

    # Models
    player_model_path: str = "models/yolo11n.pt"
    ball_model_path: str = "models/ball.pt"

    # Detection
    imgsz: int = 1280
    conf_players: float = 0.20
    conf_ball: float = 0.35    # higher = fewer false positives
    ball_static_thresh: float = 5.0  # pixels — reject if ball hasn't moved this much
    sample_every: int = 1        # run detection every N frames (1 = every frame)

    # Output video
    out_w: int = 1280
    out_h: int = 960
    fov_deg: float = 80.0
    pitch_deg: float = -20.0     # vertical tilt of output view
    out_fps: Optional[float] = None   # None = match input fps
    out_bitrate: str = "8000k"
    rotate_input: bool = True   # rotate 90° CCW (needed for raw INSV files)

    # Camera motion
    yaw_init: float = 0.0
    pitch_init: float = 0.0   # kept for compatibility
    yaw_gain: float = 0.20   # reduced to prevent wild swings
    pitch_gain: float = 0.22
    max_yaw_step: float = 1.0  # smaller max step = smoother
    max_pitch_step: float = 1.2
    target_alpha: float = 0.08  # slower smoothing = less jerk
    vel_alpha: float = 0.15
    max_yaw_dev: float = 40.0   # max pan range
    max_pitch_dev: float = 14.0
    deadband_x: float = 0.08
    deadband_y: float = 0.06

    # Ball gating
    ball_confirm_frames: int = 5    # more confirms = less jitter
    ball_miss_short: int = 30   # stay on last ball position longer
    ball_miss_long: int = 90    # stay on players longer before returning to centre
    ball_max_jump: float = 0.25     # max ball movement as fraction of frame width

    # Debug
    draw_overlay: bool = False
    print_every: int = 50


# ──────────────────────────────────────────────────────────────────
# Pitch calibration / mask
# ──────────────────────────────────────────────────────────────────

def load_pitch_mask(calib_path: str, h360: int, w360: int) -> Optional[np.ndarray]:
    """
    Load pitch polygon from calibration JSON and build a 360° binary mask.
    Returns None if no calibration found.
    """
    if not calib_path or not os.path.exists(calib_path):
        return None
    with open(calib_path) as f:
        calib = json.load(f)

    poly = calib.get("pitch_polygon_360") or calib.get("pitch_polygon")
    if not poly or len(poly) < 3:
        return None

    pts = np.array(poly, dtype=np.int32)
    mask = np.zeros((h360, w360), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def project_mask(
    mask360: np.ndarray,
    yaw: float,
    pitch: float,
    fov_deg: float,
    out_h: int,
    out_w: int,
) -> Optional[np.ndarray]:
    """Project the 360° pitch mask to perspective view."""
    if not HAS_PY360 or mask360 is None:
        return None
    mask_rgb = cv2.cvtColor(mask360, cv2.COLOR_GRAY2RGB)
    persp = e2p(mask_rgb, fov_deg=fov_deg, u_deg=yaw, v_deg=pitch,
                out_hw=(out_h, out_w), mode='nearest')
    gray = cv2.cvtColor(persp, cv2.COLOR_RGB2GRAY)
    return gray > 127


def in_mask(mask_bin: Optional[np.ndarray], x: float, y: float) -> bool:
    if mask_bin is None:
        return True
    h, w = mask_bin.shape[:2]
    xi = int(np.clip(round(x), 0, w - 1))
    yi = int(np.clip(round(y), 0, h - 1))
    return bool(mask_bin[yi, xi])


# ──────────────────────────────────────────────────────────────────
# Frame projection
# ──────────────────────────────────────────────────────────────────

def project_frame(
    frame360: np.ndarray,
    yaw: float,
    pitch: float,
    fov_deg: float,
    out_h: int,
    out_w: int,
    rotate_ccw: bool = False,
) -> np.ndarray:
    """Project equirectangular frame to perspective view."""
    if not HAS_PY360:
        return cv2.resize(frame360, (out_w, out_h))
    if rotate_ccw:
        frame360 = cv2.rotate(frame360, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rgb = cv2.cvtColor(frame360, cv2.COLOR_BGR2RGB)
    persp = e2p(rgb, fov_deg=fov_deg, u_deg=yaw, v_deg=pitch,
                out_hw=(out_h, out_w), mode='bilinear')
    return cv2.cvtColor(persp, cv2.COLOR_RGB2BGR)


# ──────────────────────────────────────────────────────────────────
# Detection
# ──────────────────────────────────────────────────────────────────

@dataclass
class Box:
    x1: float; y1: float; x2: float; y2: float; conf: float

    @property
    def cx(self) -> float: return 0.5 * (self.x1 + self.x2)
    @property
    def cy(self) -> float: return 0.5 * (self.y1 + self.y2)
    @property
    def foot_y(self) -> float: return float(self.y2)
    @property
    def area(self) -> float: return max(0, (self.x2-self.x1)*(self.y2-self.y1))


class Detector:
    def __init__(self, player_path: str, ball_path: str, device: str = "cpu"):
        from ultralytics import YOLO
        print(f"Loading models on {device}...")
        self.player_model = YOLO(player_path)
        self.ball_model = YOLO(ball_path)
        self.device = device
        self._player_names = getattr(self.player_model, "names", {})
        self._ball_names = getattr(self.ball_model, "names", {})
        print("Models loaded.")

    def detect_players(
        self,
        frame: np.ndarray,
        imgsz: int,
        conf: float,
        mask_bin: Optional[np.ndarray] = None,
    ) -> List[Box]:
        results = self.player_model(
            frame, imgsz=imgsz, conf=conf,
            device=self.device, verbose=False
        )[0]
        boxes = []
        for b in results.boxes:
            cls = int(b.cls[0])
            if self._player_names.get(cls, "") != "person":
                continue
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            cf = float(b.conf[0])
            box = Box(x1, y1, x2, y2, cf)
            if mask_bin is not None and not in_mask(mask_bin, box.cx, box.foot_y):
                continue
            boxes.append(box)
        return boxes

    def detect_ball(
        self,
        frame: np.ndarray,
        imgsz: int,
        conf: float,
    ) -> List[Box]:
        results = self.ball_model(
            frame, imgsz=imgsz, conf=conf,
            device=self.device, verbose=False
        )[0]
        boxes = []
        for b in results.boxes:
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            boxes.append(Box(x1, y1, x2, y2, float(b.conf[0])))
        return boxes


def pick_best_ball(
    balls: List[Box],
    out_w: int,
    out_h: int,
    max_area_frac: float = 0.02,
) -> Optional[Box]:
    max_area = out_w * out_h * max_area_frac
    candidates = [b for b in balls if b.area <= max_area]
    if not candidates:
        return None
    return max(candidates, key=lambda b: (b.conf, -b.area))


# ──────────────────────────────────────────────────────────────────
# Camera control (same logic as v5, stripped of team stuff)
# ──────────────────────────────────────────────────────────────────

@dataclass
class CamState:
    yaw: float = 0.0
    pitch: float = 0.0
    yaw_vel: float = 0.0
    pitch_vel: float = 0.0
    sm_tx: float = 640.0
    sm_ty: float = 360.0


@dataclass
class BallGate:
    confirm: int = 0
    trusted: bool = False
    last_pos: Optional[Tuple[float, float]] = None
    frames_since: int = 9999
    velocity: Tuple[float, float] = (0.0, 0.0)  # pixels/frame velocity


def player_centroid(boxes: List[Box]) -> Optional[Tuple[float, float]]:
    if not boxes:
        return None
    xs = [b.cx for b in boxes]
    ys = [b.cy for b in boxes]
    return float(np.mean(xs)), float(np.mean(ys))


def choose_target(
    out_w: int, out_h: int,
    ball_gate: BallGate,
    ball_pos: Optional[Tuple[float, float]],
    player_boxes: List[Box],
    miss_short: int,
    miss_long: int,
) -> Tuple[float, float, str]:
    """
    Weighted blend of all signals rather than hard mode switching.
    This produces smooth natural camera movement.
    """
    centre_x, centre_y = out_w / 2, out_h * 0.42

    # ── Compute each signal and its weight ──

    # 1. Ball signal — high weight when trusted, decays exponentially when lost
    ball_x, ball_y, w_ball = centre_x, centre_y, 0.0
    if ball_gate.trusted and ball_pos is not None:
        ball_x, ball_y = ball_pos
        w_ball = 3.0  # strong when actively detected
    elif ball_gate.last_pos is not None:
        # Decay weight based on how long ago we saw the ball
        decay = max(0.0, 1.0 - ball_gate.frames_since / miss_short)
        w_ball = 1.5 * decay
        # Use velocity prediction if ball was moving fast
        vx, vy = ball_gate.velocity
        speed = (vx**2 + vy**2) ** 0.5
        if speed > 2.0 and ball_gate.frames_since < 15:
            ball_x = ball_gate.last_pos[0] + vx * ball_gate.frames_since * 0.5
            ball_y = ball_gate.last_pos[1] + vy * ball_gate.frames_since * 0.5
            ball_x = float(np.clip(ball_x, 0, out_w))
            ball_y = float(np.clip(ball_y, 0, out_h))
        else:
            ball_x, ball_y = ball_gate.last_pos

    # 2. Player centroid signal — always present when players detected
    player_x, player_y, w_players = centre_x, centre_y, 0.0
    if player_boxes:
        xs = [b.cx for b in player_boxes]
        ys = [b.cy for b in player_boxes]
        player_x = float(np.mean(xs))
        player_y = float(np.mean(ys))
        w_players = 1.0  # steady background signal

    # 3. Centre — weak constant pull to prevent camera drifting off pitch
    w_centre = 0.2

    # ── Blend all signals ──
    total_w = w_ball + w_players + w_centre
    tx = (w_ball * ball_x + w_players * player_x + w_centre * centre_x) / total_w
    ty = (w_ball * ball_y + w_players * player_y + w_centre * centre_y) / total_w

    # Determine mode label for display
    if w_ball > 2.0:
        mode = "ball"
    elif w_ball > 0.5:
        mode = "last_ball"
    elif w_players > 0:
        mode = "players"
    else:
        mode = "centre"

    return tx, ty, mode


def update_camera(
    cam: CamState,
    tx: float, ty: float,
    cfg: Config,
) -> None:
    """Update camera yaw/pitch toward target with velocity smoothing."""
    # Normalised error
    err_x = (tx - cam.sm_tx) / (cfg.out_w * 0.5)
    err_y = (ty - cam.sm_ty) / (cfg.out_h * 0.5)

    # Deadband
    if abs(err_x) < cfg.deadband_x: err_x = 0.0
    if abs(err_y) < cfg.deadband_y: err_y = 0.0

    # Target velocity
    target_yaw_vel = err_x * cfg.fov_deg * cfg.yaw_gain
    target_pitch_vel = err_y * cfg.fov_deg * cfg.pitch_gain * 0.5

    # Smooth velocity — faster recovery when far from target
    recovery = min(3.0, abs(err_x))  # boost when far off
    cam.yaw_vel += cfg.vel_alpha * recovery * (target_yaw_vel - cam.yaw_vel)
    cam.pitch_vel += cfg.vel_alpha * (target_pitch_vel - cam.pitch_vel)

    # Clamp step
    yaw_step = float(np.clip(cam.yaw_vel, -cfg.max_yaw_step, cfg.max_yaw_step))
    pitch_step = float(np.clip(cam.pitch_vel, -cfg.max_pitch_step, cfg.max_pitch_step))

    # Apply
    cam.yaw = float(np.clip(
        cam.yaw + yaw_step,
        cfg.yaw_init - cfg.max_yaw_dev,
        cfg.yaw_init + cfg.max_yaw_dev,
    ))
    cam.pitch = float(np.clip(
        cam.pitch + pitch_step,
        cfg.pitch_deg - cfg.max_pitch_dev,
        cfg.pitch_deg + cfg.max_pitch_dev,
    ))

    # Smooth target
    cam.sm_tx += cfg.target_alpha * (tx - cam.sm_tx)
    cam.sm_ty += cfg.target_alpha * (ty - cam.sm_ty)


# ──────────────────────────────────────────────────────────────────
# FFmpeg output pipe
# ──────────────────────────────────────────────────────────────────

def open_ffmpeg_writer(
    output_path: str,
    fps: float,
    out_w: int,
    out_h: int,
    bitrate: str = "8000k",
) -> subprocess.Popen:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{out_w}x{out_h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "h264_videotoolbox",   # hardware encode on Apple Silicon / Intel Mac
        "-b:v", bitrate,
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


# ──────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────

def run(cfg: Config) -> None:
    if not HAS_PY360:
        raise RuntimeError("pip install py360convert")

    device = _best_device()
    print(f"Device: {device}")

    # ── video input ──
    cap = cv2.VideoCapture(cfg.input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {cfg.input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = cfg.out_fps or fps

    print(f"Input : {in_w}x{in_h} @ {fps:.2f}fps  {n_frames} frames")
    print(f"Output: {cfg.out_w}x{cfg.out_h} @ {out_fps:.2f}fps  {cfg.output_path}")

    # ── pitch mask ──
    mask360 = load_pitch_mask(cfg.calib_path, in_h, in_w)
    if mask360 is not None:
        print(f"Pitch mask loaded from {cfg.calib_path}")
    else:
        print("No pitch mask — all detections used")

    # ── models ──
    det = Detector(cfg.player_model_path, cfg.ball_model_path, device)

    # ── output ──
    writer = open_ffmpeg_writer(cfg.output_path, out_fps, cfg.out_w, cfg.out_h, cfg.out_bitrate)

    # ── state ──
    cam = CamState(
        yaw=cfg.yaw_init, pitch=cfg.pitch_deg,
        sm_tx=cfg.out_w / 2, sm_ty=cfg.out_h / 2,
    )
    ball_gate = BallGate()

    # cached detection results (reused between sampled frames)
    cached_players: List[Box] = []
    cached_ball: Optional[Box] = None
    cached_mode = "centre"
    warmup_frames = 30  # use player centroid to initialise camera position

    t0 = time.time()
    idx = 0

    while True:
        ok, frame360 = cap.read()
        if not ok:
            break

        # ── project to perspective ──
        persp = project_frame(
            frame360, cam.yaw, cam.pitch,
            cfg.fov_deg, cfg.out_h, cfg.out_w,
            rotate_ccw=cfg.rotate_input,
        )

        # ── detection (sampled) ──
        if idx % cfg.sample_every == 0:
            mask_bin = project_mask(
                mask360, cam.yaw, cam.pitch,
                cfg.fov_deg, cfg.out_h, cfg.out_w,
            )

            cached_players = det.detect_players(
                persp, cfg.imgsz, cfg.conf_players, mask_bin
            )

            ball_boxes = det.detect_ball(persp, cfg.imgsz, cfg.conf_ball)
            cached_ball = pick_best_ball(ball_boxes, cfg.out_w, cfg.out_h)

            # ball gating
            ball_pos = None
            if cached_ball is not None:
                bx, by = cached_ball.cx, cached_ball.cy
                # Reject large detections in bottom 10% (shoes/feet near camera)
                # Small detections (real ball) are allowed through
                ball_area = cached_ball.area
                max_ball_area = (cfg.out_w * cfg.out_h) * 0.001  # 0.1% of frame
                in_bottom = by > cfg.out_h * 0.90
                if not (in_bottom and ball_area > max_ball_area):
                    if mask_bin is None or in_mask(mask_bin, bx, by):
                        ball_pos = (bx, by)

            if ball_pos:
                # Reject if ball jumped too far from last known position
                if ball_gate.last_pos is not None:
                    dx = abs(ball_pos[0] - ball_gate.last_pos[0]) / cfg.out_w
                    dy = abs(ball_pos[1] - ball_gate.last_pos[1]) / cfg.out_h
                    if dx > cfg.ball_max_jump or dy > cfg.ball_max_jump:
                        ball_pos = None  # reject - too far jump
                    # Reject static objects (goal weights, corner flags etc)
                    elif (abs(ball_pos[0] - ball_gate.last_pos[0]) < cfg.ball_static_thresh and
                          abs(ball_pos[1] - ball_gate.last_pos[1]) < cfg.ball_static_thresh and
                          ball_gate.frames_since == 0):
                        ball_pos = None  # reject - not moving

            if ball_pos:
                # Update velocity
                if ball_gate.last_pos is not None:
                    vx = ball_pos[0] - ball_gate.last_pos[0]
                    vy = ball_pos[1] - ball_gate.last_pos[1]
                    ball_gate.velocity = (
                        0.7 * ball_gate.velocity[0] + 0.3 * vx,
                        0.7 * ball_gate.velocity[1] + 0.3 * vy,
                    )
                ball_gate.confirm += 1
                if ball_gate.confirm >= cfg.ball_confirm_frames:
                    ball_gate.trusted = True
                ball_gate.last_pos = ball_pos
                ball_gate.frames_since = 0
            else:
                ball_gate.frames_since += cfg.sample_every
                ball_gate.confirm = 0

            tx, ty, cached_mode = choose_target(
                cfg.out_w, cfg.out_h,
                ball_gate, ball_pos,
                cached_players,
                cfg.ball_miss_short,
                cfg.ball_miss_long,
            )
        else:
            # between detections: use cached target
            tx = cam.sm_tx
            ty = cam.sm_ty

        # ── camera update ──
        # On very first detection, snap smoothed target to avoid initial jerk
        if idx == 0 and cached_players:
            xs = [b.cx for b in cached_players]
            ys = [b.cy for b in cached_players]
            cam.sm_tx = float(np.mean(xs))
            cam.sm_ty = float(np.mean(ys))
        update_camera(cam, tx, ty, cfg)

        # ── debug overlay ──
        if cfg.draw_overlay:
            for b in cached_players:
                cv2.rectangle(persp,
                    (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)),
                    (0, 255, 0), 1)
            if cached_ball:
                cv2.circle(persp,
                    (int(cached_ball.cx), int(cached_ball.cy)),
                    8, (0, 255, 255), 2)
            cv2.putText(persp, f"{idx} {cached_mode} y={cam.yaw:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # ── write frame ──
        writer.stdin.write(persp.tobytes())

        idx += 1
        if idx % cfg.print_every == 0:
            dt = time.time() - t0
            pct = 100 * idx / max(1, n_frames)
            eta = (dt / idx) * (n_frames - idx)
            print(f"  [{pct:5.1f}%] frame {idx:5d}  "
                  f"{idx/dt:5.1f} fps  "
                  f"mode={cached_mode:<10}  "
                  f"yaw={cam.yaw:+.1f}°  "
                  f"ETA {eta/60:.0f}min")

    cap.release()
    writer.stdin.close()
    writer.wait()

    dt = time.time() - t0
    print(f"\nDone. {idx} frames in {dt:.1f}s ({idx/dt:.1f} fps)")
    print(f"Output: {cfg.output_path}")


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Simple autopan pipeline")
    p.add_argument("--input",   required=True,  help="Input equirect video")
    p.add_argument("--output",  default="output_simple.mp4")
    p.add_argument("--calib",   default="calibration/pitch.json")
    p.add_argument("--players", default="models/yolo11n.pt")
    p.add_argument("--ball",    default="models/ball.pt")
    p.add_argument("--imgsz",   type=int,   default=1280)
    p.add_argument("--sample",  type=int,   default=1,    help="Detect every N frames")
    p.add_argument("--out-w",   type=int,   default=1280)
    p.add_argument("--out-h",   type=int,   default=720)
    p.add_argument("--fov",     type=float, default=90.0)
    p.add_argument("--bitrate", default="8000k")
    p.add_argument("--overlay", action="store_true", help="Draw debug overlay")
    p.add_argument("--yaw",     type=float, default=0.0,  help="Initial yaw")
    p.add_argument("--pitch",   type=float, default=-20.0,  help="Initial pitch (negative = tilt up)")
    p.add_argument("--no-rotate", action="store_true", help="Don't rotate input (use for pre-converted equirect)")
    p.add_argument("--device",  default=None, help="Force device (cpu/mps/cuda)")
    a = p.parse_args()

    cfg = Config(
        input_path=a.input,
        output_path=a.output,
        calib_path=a.calib,
        player_model_path=a.players,
        ball_model_path=a.ball,
        imgsz=a.imgsz,
        sample_every=a.sample,
        out_w=a.out_w,
        out_h=a.out_h,
        fov_deg=a.fov,
        out_bitrate=a.bitrate,
        draw_overlay=a.overlay,
        yaw_init=a.yaw,
        pitch_deg=a.pitch,
        rotate_input=not a.no_rotate,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
