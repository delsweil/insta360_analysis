from dataclasses import dataclass

@dataclass
class VideoConfig:
    input_path: str
    output_path: str
    out_w: int = 1280
    out_h: int = 720
    fov_deg: float = 90.0

@dataclass
class ModelConfig:
    player_model_path: str = "models/yolo11n.pt"
    ball_model_path: str = "models/ball.pt"
    imgsz_players: int = 960
    imgsz_ball: int = 640
    conf_players: float = 0.35
    conf_ball: float = 0.20

@dataclass
class MotionConfig:
    yaw_init: float = 0.0
    pitch_init: float = 0.0

    yaw_gain: float = 0.30
    pitch_gain: float = 0.22
    max_yaw_step: float = 1.6
    max_pitch_step: float = 1.2

    deadband_x: float = 0.08
    deadband_y: float = 0.06

    # “corridor” around starting view
    max_yaw_dev: float = 40.0
    max_pitch_dev: float = 14.0

    # target smoothing
    target_alpha: float = 0.12

    # velocity smoothing (inertia)
    vel_alpha: float = 0.15  # 0.1–0.2

@dataclass
class BallGatingConfig:
    confirm_frames: int = 3
    miss_short_fallback: int = 10
    miss_long_fallback: int = 90

@dataclass
class WorldConfig:
    calib_path: str = "calibration/pitch.json"
    # if True: hard-filter detections by pitch polygon
    hard_pitch_filter: bool = True

@dataclass
class DebugConfig:
    draw: bool = True
    print_every: int = 100
