# src/autopan/autopan_v5_refactor.py
import os
import time
import numpy as np
import cv2

from .config import (
    VideoConfig, ModelConfig, MotionConfig,
    BallGatingConfig, WorldConfig, DebugConfig
)
from .io import open_video, make_writer
from .view import CameraState, project_e2p, project_mask_e2p
from .world import load_pitch_polygon
from .perception import Detector, pick_best_ball
from .control import (
    BallState, TargetState, VelocityState,
    choose_target, update_smoothed_target,
    compute_errors, update_camera_with_velocity
)
from .overlay import draw_circle, draw_text, draw_mask_contour

from .team_cluster import TeamClusterer, TeamClusterConfig
from .tracking import TrackManager


# -------------------------
# helpers
# -------------------------

def _in_mask(mask_bin: np.ndarray, x: float, y: float) -> bool:
    if mask_bin is None:
        return True
    h, w = mask_bin.shape[:2]
    xi = int(np.clip(int(round(x)), 0, w - 1))
    yi = int(np.clip(int(round(y)), 0, h - 1))
    return bool(mask_bin[yi, xi])


def _color_for_label(lab: str | None):
    if lab == "team0":
        return (0, 255, 0)
    if lab == "team1":
        return (255, 200, 0)
    return (180, 180, 180)  # unknown / unstable


# -------------------------
# main
# -------------------------

def main():
    # ---- video / models ----
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

    ballcfg = BallGatingConfig(
        confirm_frames=3,
        miss_short_fallback=10,
        miss_long_fallback=90
    )

    worldcfg = WorldConfig(
        calib_path="calibration/pitch.json",
        hard_pitch_filter=True
    )

    dbg = DebugConfig(draw=True, print_every=100)

    # ---- team clustering ----
    TEAM_ENABLED = True

    teamcfg = TeamClusterConfig(
        bootstrap_frames=250  # ~8 seconds
    )

    team_cluster = TeamClusterer(teamcfg)

    # margin gate from your eval (p10 ≈ 0.056)
    MIN_MARGIN_TO_VOTE = 0.08

    # ---- tracker ----
    tracker = TrackManager(
        iou_thresh=0.30,
        max_missed=10,
        vote_window=15,
        vote_ratio=0.7
    )

    # ---- sanity ----
    for p in [models.player_model_path, models.ball_model_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    os.makedirs(os.path.dirname(vid.output_path), exist_ok=True)

    # ---- video io ----
    cap = open_video(vid.input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input: {in_w}x{in_h} @ {fps:.2f} fps, {n_frames} frames")
    out = make_writer(vid.output_path, fps, vid.out_w, vid.out_h)

    # ---- pitch mask ----
    pitch360 = load_pitch_polygon(worldcfg.calib_path)
    mask360 = pitch360.build_mask360() if pitch360 else None

    # ---- models ----
    print("Loading models...")
    det = Detector(models.player_model_path, models.ball_model_path)
    print("Models loaded.")

    # ---- state ----
    cam = CameraState(
        yaw=motion.yaw_init,
        pitch=motion.pitch_init,
        yaw_center=motion.yaw_init,
        pitch_center=motion.pitch_init
    )

    ball_state = BallState()
    target_state = TargetState(
        sm_tx=vid.out_w / 2,
        sm_ty=vid.out_h / 2
    )
    vel_state = VelocityState()

    t0 = time.time()
    idx = 0

    # =========================
    # main loop
    # =========================

    while True:
        ok, frame360 = cap.read()
        if not ok:
            break

        persp = project_e2p(
            frame360, cam.yaw, cam.pitch,
            vid.fov_deg, vid.out_h, vid.out_w
        )

        # ---- pitch mask projection ----
        mask_bin = None
        if mask360 is not None:
            mask_p = project_mask_e2p(
                mask360, cam.yaw, cam.pitch,
                vid.fov_deg, vid.out_h, vid.out_w
            )
            mask_bin = mask_p > 127

        # ---- players ----
        player_boxes = det.detect_players_two_pass(
            persp,
            imgsz_lo=models.imgsz_players,     # pass 1 resolution
            imgsz_hi=1280,                      # pass 2 (small players)
            conf=models.conf_players,
            conf_p2=0.20,                       # slightly lower for small players
            mask_bin=mask_bin,                  # pitch mask if available
            every_n_frames=5,                   # <-- THIS restores your old behaviour
            frame_index=idx
        )

        filt_boxes = []
        centroids = []
        for b in player_boxes:
            if worldcfg.hard_pitch_filter and mask_bin is not None:
                if not _in_mask(mask_bin, b.cx, b.foot_y):
                    continue
            filt_boxes.append(b)
            centroids.append((b.cx, b.cy))

        # ---- team clustering + tracking ----
        stable_labels = None
        if TEAM_ENABLED and filt_boxes:
            if not team_cluster.ready:
                team_cluster.update_with_boxes(persp, filt_boxes)

            bboxes = [(int(b.x1), int(b.y1), int(b.x2), int(b.y2)) for b in filt_boxes]
            track_ids = tracker.update(bboxes)

            if team_cluster.ready:
                results = team_cluster.classify_boxes_with_margins(persp, filt_boxes)

                for tid, (lab, margin, _, _) in zip(track_ids, results):
                    if lab is None:
                        continue
                    if margin >= MIN_MARGIN_TO_VOTE:
                        tracker.vote(tid, lab)

                stable_labels = [tracker.stable_label(tid) for tid in track_ids]

        # ---- ball ----
        ball_boxes = det.detect_ball(
            persp,
            imgsz=models.imgsz_ball,
            conf=models.conf_ball
        )
        best_ball = pick_best_ball(
            ball_boxes,
            max_area_frac=0.02,
            out_w=vid.out_w,
            out_h=vid.out_h
        )

        ball_pos = None
        used_ball = False
        if best_ball is not None:
            if mask_bin is None or _in_mask(mask_bin, best_ball.cx, best_ball.cy):
                ball_pos = (best_ball.cx, best_ball.cy)

        if ball_pos:
            ball_state.confirm += 1
            if ball_state.confirm >= ballcfg.confirm_frames:
                ball_state.trusted = True
            ball_state.last_ball = ball_pos
            ball_state.frames_since = 0
            used_ball = ball_state.trusted
        else:
            ball_state.frames_since += 1
            ball_state.confirm = 0

        # ---- target ----
        tx, ty, mode = choose_target(
            vid.out_w, vid.out_h,
            used_ball, ball_pos,
            ball_state.last_ball,
            ball_state.frames_since,
            centroids,
            ballcfg.miss_short_fallback,
            ballcfg.miss_long_fallback
        )

        target_state.sm_tx, target_state.sm_ty = update_smoothed_target(
            target_state.sm_tx, target_state.sm_ty,
            tx, ty, motion.target_alpha
        )

        err_x, err_y = compute_errors(
            target_state.sm_tx, target_state.sm_ty,
            vid.out_w, vid.out_h,
            motion.deadband_x,
            motion.deadband_y
        )

        cam.yaw, cam.pitch, vel_state.yaw_vel, vel_state.pitch_vel = update_camera_with_velocity(
            cam.yaw, cam.pitch,
            cam.yaw_center, cam.pitch_center,
            err_x, err_y,
            vid.fov_deg,
            motion.yaw_gain,
            motion.pitch_gain,
            motion.max_yaw_step,
            motion.max_pitch_step,
            motion.max_yaw_dev,
            motion.max_pitch_dev,
            vel_state.yaw_vel,
            vel_state.pitch_vel,
            motion.vel_alpha
        )

        # ---- debug ----
        if dbg.draw:
            if mask_bin is not None:
                draw_mask_contour(persp, mask_bin.astype(np.uint8) * 255)

            for i, b in enumerate(filt_boxes):
                lab = stable_labels[i] if stable_labels else None
                col = _color_for_label(lab)
                cv2.rectangle(
                    persp,
                    (int(b.x1), int(b.y1)),
                    (int(b.x2), int(b.y2)),
                    col, 2
                )

            if ball_pos:
                draw_circle(persp, ball_pos, 8, (0, 255, 255), 2)

            draw_text(persp, f"Frame {idx}", (20, 40))
            draw_text(persp, f"Mode: {mode}", (20, 70))

        out.write(persp)
        idx += 1

        if idx % dbg.print_every == 0:
            dt = time.time() - t0
            print(f"Processed {idx}/{n_frames} ({idx/dt:.2f} fps)")

    cap.release()
    out.release()

    dt = time.time() - t0
    print(f"\nDone. {idx} frames in {dt:.1f}s ({idx/dt:.2f} fps)")
    print(f"Output: {vid.output_path}")


if __name__ == "__main__":
    main()
