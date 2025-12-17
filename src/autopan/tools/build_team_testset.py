# src/autopan/tools/build_team_testset.py
from __future__ import annotations
import os
import json
from pathlib import Path

import cv2
import numpy as np

from src.autopan.perception import Detector
from src.autopan.team_cluster import TeamClusterConfig, extract_kit_feature
from src.autopan.world import load_pitch_polygon
from src.autopan.view import project_e2p, project_mask_e2p


# ---------- helpers ----------
def clamp_int(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, v)))


def _in_mask(mask_bin: np.ndarray | None, x: float, y: float) -> bool:
    if mask_bin is None:
        return True
    h, w = mask_bin.shape[:2]
    xi = int(np.clip(int(round(x)), 0, w - 1))
    yi = int(np.clip(int(round(y)), 0, h - 1))
    return bool(mask_bin[yi, xi])


def torso_crop(img_bgr: np.ndarray, box, cfg: TeamClusterConfig) -> np.ndarray | None:
    """Crop torso area from a detection box."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)

    x1 = clamp_int(x1, 0, w - 1)
    x2 = clamp_int(x2, 0, w - 1)
    y1 = clamp_int(y1, 0, h - 1)
    y2 = clamp_int(y2, 0, h - 1)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    ry1 = y1 + int(cfg.torso_y1 * bh)
    ry2 = y1 + int(cfg.torso_y2 * bh)
    xpad = int(cfg.torso_xpad * bw)
    rx1 = x1 + xpad
    rx2 = x2 - xpad

    rx1 = clamp_int(rx1, 0, w - 1)
    rx2 = clamp_int(rx2, 0, w - 1)
    ry1 = clamp_int(ry1, 0, h - 1)
    ry2 = clamp_int(ry2, 0, h - 1)

    if rx2 <= rx1 or ry2 <= ry1:
        return None
    crop = img_bgr[ry1:ry2, rx1:rx2].copy()
    if crop.size == 0:
        return None
    return crop


def draw_help(img):
    lines = [
        "Click this window once so it has focus!",
        "LABEL: 0=team0  1=team1  r=ref  o=other  s=skip  q=quit",
        "NAV:   a=prev   d=next   (labels saved immediately)",
        "Tip: press 's' aggressively for unclear / tiny / blurred crops.",
    ]
    y = 26
    for t in lines:
        cv2.putText(img, t, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        y += 26


# ---------- main ----------
def main():
    # ---- configure these ----
    VIDEO_PATH = "/Users/davidelsweiler/Desktop/test_run.mp4"
    OUT_DIR = "data/team_testset_v1"
    PLAYER_MODEL = "models/yolo11n.pt"
    BALL_MODEL = "models/ball.pt"  # Detector expects both

    # Sampling
    STRIDE = 15                 # ~2 fps from 30 fps
    MAX_CROPS = 125             # target number of crops to label

    # Detection
    IMGSZ_PLAYERS = 960
    CONF_PLAYERS = 0.35
    MAX_BOXES_PER_FRAME = 6     # keep only the largest boxes each sampled frame
    MIN_BOX_AREA = 35 * 35      # reject tiny distant detections

    # Pitch filtering (IMPORTANT: prevents coaches/supporters)
    USE_PITCH_MASK = True
    CALIB_PATH = "calibration/pitch.json"

    # Perspective used for testset building (match your autopan view)
    # If you want, set these to the SAME yaw/pitch/fov/out_w/out_h you use in autopan
    OUT_W, OUT_H = 1280, 720
    FOV_DEG = 90
    YAW_DEG = 0.0
    PITCH_DEG = 0.0

    # Optional: quality gate – skip crops that won't produce a stable kit feature
    FEATURE_GATE = True

    # -------------------------

    cfg = TeamClusterConfig()

    out_dir = Path(OUT_DIR)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.jsonl"

    # append mode so you can resume
    fout = open(labels_path, "a", encoding="utf-8")

    det = Detector(PLAYER_MODEL, BALL_MODEL)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {n_frames} frames @ {fps:.2f} fps  ({in_w}x{in_h})")

    # existing crop ids (resume-safe)
    existing = set()
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as fin:
            for line in fin:
                try:
                    obj = json.loads(line)
                    existing.add(obj["crop_file"])
                except Exception:
                    pass

    # load 360 polygon mask if available
    mask360 = None
    if USE_PITCH_MASK:
        pitch360 = load_pitch_polygon(CALIB_PATH)
        if pitch360 is None:
            print(f"[WARN] No pitch polygon at {CALIB_PATH}. Proceeding without pitch filtering.")
        else:
            if pitch360.in_w != in_w or pitch360.in_h != in_h:
                print(
                    f"[WARN] pitch.json equirect dims ({pitch360.in_w}x{pitch360.in_h}) "
                    f"!= video ({in_w}x{in_h}). Recalibrate for this video."
                )
            else:
                mask360 = pitch360.build_mask360()
                #print(f"[OK] Loaded 360 pitch polygon with {len(pitch360.poly)} points ({in_w}x{in_h})")
                print(f"[OK] Loaded 360 pitch polygon ({in_w}x{in_h})")

    labeled = 0
    frame_i = 0
    kept_items = []  # list[(meta, crop_img)]

    # ---- collect crops (fast) ----
    while len(kept_items) < MAX_CROPS and frame_i < n_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
        ok, frame360 = cap.read()
        if not ok:
            break

        # Build a fixed perspective view for consistent cropping (crucial)
        persp = project_e2p(frame360, YAW_DEG, PITCH_DEG, FOV_DEG, OUT_H, OUT_W)

        # Project mask to the same perspective view, so filtering matches production
        mask_bin = None
        if mask360 is not None:
            mask_p = project_mask_e2p(mask360, YAW_DEG, PITCH_DEG, FOV_DEG, OUT_H, OUT_W)
            mask_bin = (mask_p > 127)

        boxes = det.detect_players(persp, imgsz=IMGSZ_PLAYERS, conf=CONF_PLAYERS)

        # Keep only largest boxes first
        boxes = sorted(boxes, key=lambda b: float((b.x2 - b.x1) * (b.y2 - b.y1)), reverse=True)[:MAX_BOXES_PER_FRAME]

        for bi, b in enumerate(boxes):
            bw = max(1, int(b.x2 - b.x1))
            bh = max(1, int(b.y2 - b.y1))
            if bw * bh < MIN_BOX_AREA:
                continue

            # Pitch filtering using FOOT point (best proxy)
            cx = 0.5 * (float(b.x1) + float(b.x2))
            foot_y = float(b.y2)
            if mask_bin is not None:
                if not _in_mask(mask_bin, cx, foot_y):
                    continue

            crop = torso_crop(persp, b, cfg)
            if crop is None:
                continue

            # Quality gate: if feature extractor returns None, this crop is likely useless/unstable
            if FEATURE_GATE:
                # fake box spanning entire crop
                class FakeBox:
                    def __init__(self, w, h):
                        self.x1, self.y1, self.x2, self.y2 = 0, 0, w, h
                fb = FakeBox(crop.shape[1], crop.shape[0])
                f = extract_kit_feature(crop, fb, cfg)
                if f is None:
                    continue

            crop_name = f"f{frame_i:06d}_b{bi:02d}.jpg"
            if crop_name in existing:
                continue

            meta = {
                "video": VIDEO_PATH,
                "frame_index": frame_i,
                "perspective": {
                    "out_w": OUT_W, "out_h": OUT_H, "fov_deg": FOV_DEG,
                    "yaw_deg": YAW_DEG, "pitch_deg": PITCH_DEG
                },
                "box": {"x1": int(b.x1), "y1": int(b.y1), "x2": int(b.x2), "y2": int(b.y2)},
                "crop_file": crop_name,
                "cfg": {"torso_y1": cfg.torso_y1, "torso_y2": cfg.torso_y2, "torso_xpad": cfg.torso_xpad},
            }
            kept_items.append((meta, crop))

            if len(kept_items) >= MAX_CROPS:
                break

        frame_i += STRIDE

    cap.release()
    print(f"Collected {len(kept_items)} crops (pre-label). Now labeling...")

    # ---- interactive labeling ----
    idx = 0
    keymap = {
        ord("0"): "team0",
        ord("1"): "team1",
        ord("r"): "ref",
        ord("o"): "other",
        ord("s"): "skip",
    }

    while 0 <= idx < len(kept_items):
        meta, crop = kept_items[idx]

        show = crop.copy()
        draw_help(show)
        cv2.putText(show, f"{idx+1}/{len(kept_items)}  file={meta['crop_file']}", (18, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

        cv2.imshow("Team testset labeler", show)
        k = cv2.waitKey(0) & 0xFF

        if k == ord("q"):
            break
        if k == ord("a"):
            idx -= 1
            continue
        if k == ord("d"):
            idx += 1
            continue

        if k in keymap:
            lab = keymap[k]

            # Save only non-skip labels
            if lab != "skip":
                crop_path = crops_dir / meta["crop_file"]
                cv2.imwrite(str(crop_path), crop)

                row = dict(meta)
                row["label"] = lab
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                labeled += 1

            idx += 1

    fout.close()
    cv2.destroyAllWindows()
    print(f"Done. Labeled {labeled} crops.")
    print(f"Saved to: {labels_path}")
    print(f"Crops in: {crops_dir}")


if __name__ == "__main__":
    main()
