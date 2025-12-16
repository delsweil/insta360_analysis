# src/autopan/autopan_v5_refactor.py
import os
import time
import numpy as np
import cv2

from .config import VideoConfig, ModelConfig, MotionConfig, BallGatingConfig, WorldConfig, DebugConfig
from .io import open_video, make_writer
from .view import CameraState, project_e2p, project_mask_e2p
from .world import load_pitch_polygon
from .perception import Detector, pick_best_ball
from .control import BallState, TargetState, VelocityState, choose_target, update_smoothed_target, compute_errors, update_camera_with_velocity
from .overlay import draw_boxes, draw_circle, draw_text, draw_mask_contour


def _in_mask(mask_bin: np.ndarray, x: float, y: float) -> bool:
    if mask_bin is None:
        return True
    h, w = mask_bin.shape[:2]
    xi = int(np.clip(int(round(x)), 0, w - 1))
    yi = int(np.clip(int(round(y)), 0, h - 1))
    return bool(mask_bin[yi, xi])


def main():
    # ---- Edit these ----
    vid = VideoConfig(
        input_path="/Users/davidelsweiler/Desktop/test_run.mp4",
        output_path="data/processed/autopan_refactor_v5.mp4",
        out_w=1280, out_h=720, fov_deg=90
    )
    models = ModelConfig(
        player_model_path="models/yolo11n.pt",
        ball_model_path="models/ball.pt",
        imgsz_players=960,
        imgsz_ball=640,
        conf_players=0.35,
        conf_ball=0.20
    )
    motion = MotionConfig(
        yaw_init=0.0, pitch_init=0.0,
        yaw_gain=0.30, pitch_gain=0.22,
        max_yaw_step=1.6, max_pitch_step=1.2,
        target_alpha=0.12,
        vel_alpha=0.15,
        max_yaw_dev=40.0,
        max_pitch_dev=14.0,
        deadband_x=0.08,
        deadband_y=0.06
    )
    ballcfg = BallGatingConfig(confirm_frames=3, miss_short_fallback=10, miss_long_fallback=90)
    worldcfg = WorldConfig(calib_path="calibration/pitch.json", hard_pitch_filter=True)
    dbg = DebugConfig(draw=True, print_every=100)

    # ---- sanity ----
    for p in [models.player_model_path, models.ball_model_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing model file: {p}")
    os.makedirs(os.path.dirname(vid.output_path), exist_ok=True)

    # ---- video io ----
    cap = open_video(vid.input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input: {in_w}x{in_h} @ {fps:.2f} fps, {n_frames} frames")

    out = make_writer(vid.output_path, fps, vid.out_w, vid.out_h)

    # ---- world (360 polygon) ----
    pitch360 = load_pitch_polygon(worldcfg.calib_path)
    mask360 = None
    if pitch360 is not None:
        if pitch360.in_w != in_w or pitch360.in_h != in_h:
            print(
                f"[WARN] Calibration equirect dims ({pitch360.in_w}x{pitch360.in_h}) "
                f"do not match video ({in_w}x{in_h}). Recalibrate on this video."
            )
            pitch360 = None
        else:
            mask360 = pitch360.build_mask360()

    # ---- models ----
    print("Loading models...")
    det = Detector(models.player_model_path, models.ball_model_path)
    print("Models loaded.")

    # ---- state ----
    cam = CameraState(
        yaw=motion.yaw_init, pitch=motion.pitch_init,
        yaw_center=motion.yaw_init, pitch_center=motion.pitch_init
    )
    ball_state = BallState()
    target_state = TargetState(sm_tx=vid.out_w / 2, sm_ty=vid.out_h / 2)
    vel_state = VelocityState()

    t0 = time.time()
    idx = 0

    while True:
        ok, frame360 = cap.read()
        if not ok:
            break

        # ---- project view ----
        persp = project_e2p(frame360, cam.yaw, cam.pitch, vid.fov_deg, vid.out_h, vid.out_w)

        # ---- project pitch mask (if available) ----
        mask_persp = None
        mask_bin = None
        coverage = None
        if mask360 is not None:
            mask_persp = project_mask_e2p(mask360, cam.yaw, cam.pitch, vid.fov_deg, vid.out_h, vid.out_w)
            mask_bin = (mask_persp > 127)
            coverage = float(mask_bin.mean())

        # ---- detect players ----
        player_boxes = det.detect_players(persp, imgsz=models.imgsz_players, conf=models.conf_players)

        filtered_player_boxes = []
        players_centroids = []
        for b in player_boxes:
            cx, cy, foot_y = b.cx, b.cy, b.foot_y
            if worldcfg.hard_pitch_filter and mask_bin is not None:
                if not _in_mask(mask_bin, cx, foot_y):
                    continue
            filtered_player_boxes.append(b)
            players_centroids.append((cx, cy))

        # ---- detect ball ----
        ball_boxes = det.detect_ball(persp, imgsz=models.imgsz_ball, conf=models.conf_ball)
        best_ball = pick_best_ball(ball_boxes, max_area_frac=0.02, out_w=vid.out_w, out_h=vid.out_h)

        ball_pos = None
        used_ball = False

        if best_ball is not None:
            cx, cy = best_ball.cx, best_ball.cy
            if worldcfg.hard_pitch_filter and mask_bin is not None:
                if _in_mask(mask_bin, cx, cy):
                    ball_pos = (cx, cy)
            else:
                ball_pos = (cx, cy)

        if ball_pos is not None:
            ball_state.confirm += 1
            if ball_state.confirm >= ballcfg.confirm_frames:
                ball_state.trusted = True
            ball_state.last_ball = ball_pos
            ball_state.frames_since = 0
            used_ball = ball_state.trusted
        else:
            ball_state.frames_since += 1
            ball_state.confirm = 0

        # ---- choose target ----
        tx, ty, mode = choose_target(
            vid.out_w, vid.out_h,
            ball_used=used_ball,
            ball_pos=ball_pos,
            last_ball=ball_state.last_ball,
            frames_since_ball=ball_state.frames_since,
            players_centroids=players_centroids,
            miss_short=ballcfg.miss_short_fallback,
            miss_long=ballcfg.miss_long_fallback
        )

        # ---- smooth target ----
        target_state.sm_tx, target_state.sm_ty = update_smoothed_target(
            target_state.sm_tx, target_state.sm_ty,
            tx, ty,
            alpha=motion.target_alpha
        )

        # ---- compute errors ----
        err_x, err_y = compute_errors(
            target_state.sm_tx, target_state.sm_ty,
            vid.out_w, vid.out_h,
            dead_x=motion.deadband_x,
            dead_y=motion.deadband_y
        )

        # ---- update camera ----
        cam.yaw, cam.pitch, vel_state.yaw_vel, vel_state.pitch_vel = update_camera_with_velocity(
            cam.yaw, cam.pitch,
            cam.yaw_center, cam.pitch_center,
            err_x, err_y,
            fov_deg=vid.fov_deg,
            yaw_gain=motion.yaw_gain,
            pitch_gain=motion.pitch_gain,
            max_yaw_step=motion.max_yaw_step,
            max_pitch_step=motion.max_pitch_step,
            max_yaw_dev=motion.max_yaw_dev,
            max_pitch_dev=motion.max_pitch_dev,
            yaw_vel=vel_state.yaw_vel,
            pitch_vel=vel_state.pitch_vel,
            vel_alpha=motion.vel_alpha
        )

        # ---- debug draw ----
        if dbg.draw:
            if mask_persp is not None:
                draw_mask_contour(persp, mask_persp, color=(255, 255, 0), thickness=2)

            draw_boxes(persp, [(b.x1, b.y1, b.x2, b.y2) for b in filtered_player_boxes], color=(0, 255, 0), thickness=2)

            if ball_pos is not None:
                col = (0, 255, 255) if used_ball else (0, 140, 255)
                draw_circle(persp, ball_pos, r=8, color=col, thickness=2)

            draw_circle(persp, (int(target_state.sm_tx), int(target_state.sm_ty)), r=5, color=(255, 255, 255), thickness=2)

            draw_text(persp, f"Frame {idx}  yaw={cam.yaw:.1f} pitch={cam.pitch:.1f}", (20, 40), 0.7, (255, 255, 255), 2)
            draw_text(persp, f"Mode: {mode}  ball_miss={ball_state.frames_since}", (20, 70), 0.7, (0, 255, 255), 2)
            if coverage is not None:
                draw_text(persp, f"Pitch coverage: {coverage:.3f}", (20, 100), 0.7, (255, 255, 0), 2)

        out.write(persp)
        idx += 1

        if idx % dbg.print_every == 0:
            elapsed = time.time() - t0
            eff = idx / elapsed if elapsed > 0 else 0
            msg = f"Processed {idx}/{n_frames} frames ({eff:.2f} fps)"
            if coverage is not None:
                msg += f"  coverage={coverage:.3f}"
            print(msg)

    cap.release()
    out.release()

    elapsed = time.time() - t0
    eff = idx / elapsed if elapsed > 0 else 0
    print("\n--- AutoPan refactor complete ---")
    print(f"Frames: {idx}")
    print(f"Time: {elapsed:.2f}s  Effective FPS: {eff:.2f}")
    print(f"Output: {vid.output_path}")


if __name__ == "__main__":
    main()
