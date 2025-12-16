from dataclasses import dataclass
from typing import Optional, Tuple, List
import math

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def soft_step(angle: float, delta: float, max_step: float) -> float:
    return angle + clamp(delta, -max_step, max_step)

def clamp_around_center(angle: float, center: float, max_dev: float) -> float:
    return clamp(angle, center - max_dev, center + max_dev)

@dataclass
class BallState:
    last_ball: Optional[Tuple[int,int]] = None
    confirm: int = 0
    trusted: bool = False
    frames_since: int = 999999

@dataclass
class TargetState:
    sm_tx: float
    sm_ty: float

@dataclass
class VelocityState:
    yaw_vel: float = 0.0
    pitch_vel: float = 0.0

def choose_target(
    out_w: int, out_h: int,
    ball_used: bool,
    ball_pos: Optional[Tuple[int,int]],
    last_ball: Optional[Tuple[int,int]],
    frames_since_ball: int,
    players_centroids: List[Tuple[int,int]],
    miss_short: int,
    miss_long: int
) -> Tuple[float, float, str]:
    tx, ty = out_w / 2, out_h / 2
    mode = "Center"

    if ball_used and ball_pos is not None:
        tx, ty = ball_pos
        mode = "Ball"
    elif last_ball is not None and frames_since_ball < miss_short:
        tx, ty = last_ball
        mode = "Last ball"
    elif players_centroids and frames_since_ball < miss_long:
        sx = sum(p[0] for p in players_centroids)
        sy = sum(p[1] for p in players_centroids)
        tx = sx / len(players_centroids)
        ty = sy / len(players_centroids)
        mode = "Players"
    else:
        mode = "Home drift"

    return tx, ty, mode

def update_smoothed_target(sm_tx: float, sm_ty: float, tx: float, ty: float, alpha: float) -> Tuple[float,float]:
    sm_tx = (1 - alpha) * sm_tx + alpha * tx
    sm_ty = (1 - alpha) * sm_ty + alpha * ty
    return sm_tx, sm_ty

def compute_errors(sm_tx: float, sm_ty: float, out_w: int, out_h: int, dead_x: float, dead_y: float) -> Tuple[float,float]:
    tx_norm = sm_tx / out_w
    ty_norm = sm_ty / out_h
    err_x = tx_norm - 0.5
    err_y = ty_norm - 0.5
    if abs(err_x) < dead_x: err_x = 0.0
    if abs(err_y) < dead_y: err_y = 0.0
    return err_x, err_y

def update_camera(
    yaw: float, pitch: float,
    yaw_center: float, pitch_center: float,
    err_x: float, err_y: float,
    fov_deg: float,
    yaw_gain: float, pitch_gain: float,
    max_yaw_step: float, max_pitch_step: float,
    max_yaw_dev: float, max_pitch_dev: float,
) -> Tuple[float,float]:
    d_yaw = yaw_gain * err_x * fov_deg
    d_pitch = pitch_gain * err_y * fov_deg * -1.0

    yaw = soft_step(yaw, d_yaw, max_yaw_step)
    pitch = soft_step(pitch, d_pitch, max_pitch_step)

    yaw = clamp_around_center(yaw, yaw_center, max_yaw_dev)
    pitch = clamp_around_center(pitch, pitch_center, max_pitch_dev)

    return yaw, pitch

def update_camera_with_velocity(
    yaw: float, pitch: float,
    yaw_center: float, pitch_center: float,
    err_x: float, err_y: float,
    fov_deg: float,
    yaw_gain: float, pitch_gain: float,
    max_yaw_step: float, max_pitch_step: float,
    max_yaw_dev: float, max_pitch_dev: float,
    yaw_vel: float, pitch_vel: float,
    vel_alpha: float
) -> Tuple[float,float,float,float]:
    desired_yaw_vel = yaw_gain * err_x * fov_deg
    desired_pitch_vel = pitch_gain * err_y * fov_deg * -1.0

    yaw_vel = (1 - vel_alpha) * yaw_vel + vel_alpha * desired_yaw_vel
    pitch_vel = (1 - vel_alpha) * pitch_vel + vel_alpha * desired_pitch_vel

    yaw = soft_step(yaw, yaw_vel, max_yaw_step)
    pitch = soft_step(pitch, pitch_vel, max_pitch_step)

    yaw = clamp_around_center(yaw, yaw_center, max_yaw_dev)
    pitch = clamp_around_center(pitch, pitch_center, max_pitch_dev)

    return yaw, pitch, yaw_vel, pitch_vel
