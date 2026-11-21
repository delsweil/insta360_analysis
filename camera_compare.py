#!/usr/bin/env python3
"""
camera_compare.py — the same detector on the same kind of footage from two cameras (no lens fit needed).

For each clip it decodes frames at --fps (default 3, like the pipeline), cuts every frame into overlapping
1280x720 tiles at the file's NATIVE resolution, runs the player model and the ball model on the tiles, merges
duplicates from the overlaps, and counts what was found. Because the tiles are not resized, a camera with
more pixels shows its extra detail to the detector.

A clip wider than --match-width (default 2560) is also run a second time scaled down to that width
("<label>@2560"). That is roughly what the current pipeline does with it (its views are 1280x720 for 100 degrees),
so the gap between "<label>" and "<label>@2560" is the part of the extra detail that the current views would throw away.

Usage (from the repo root, with the project's venv):
  .venv/bin/python camera_compare.py \
      --clip x2=~/Desktop/camtest/x2/VID_xxx_10s.mp4 \
      --clip x5_015=~/Desktop/camtest/x5/VID_20261118_150229_00_015_10s.mp4 \
      --out ~/Desktop/camtest/compare --hwaccel

Prints one table, writes summary.json and annotated pictures into --out.
Nothing here changes the pipeline, the models or any other file.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import cv2
import numpy as np

TILE_W, TILE_H = 1280, 720
BALL_MAX_AREA = TILE_W * TILE_H * 0.02            # same size cap as detect_balls in extract_detections.py


# ── video ────────────────────────────────────────────────────────────────────

def probe(path):
    """-> (width, height, duration) as displayed (the rotate flag is applied, as ffmpeg does when decoding)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height:stream_side_data=rotation:format=duration",
                        "-of", "json", path], capture_output=True, text=True, check=True)
    j = json.loads(r.stdout)
    s = j["streams"][0]
    w, h = int(s["width"]), int(s["height"])
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    if abs(rot) % 180 == 90:
        w, h = h, w
    return w, h, float(j["format"]["duration"])


def read_exact(f, n):
    buf = b""
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def frames(path, w, h, fps, start, duration, hwaccel, out_w):
    """Yield (t, bgr frame) at fps, optionally scaled to out_w wide."""
    out_h = h
    vf = f"fps={fps}"
    if out_w and out_w != w:
        out_h = int(round(h * out_w / w / 2.0)) * 2
        vf += f",scale={out_w}:{out_h}:flags=lanczos"
    else:
        out_w = w
    cmd = ["ffmpeg", "-v", "error"]
    if hwaccel:
        cmd += ["-hwaccel", "videotoolbox"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", path]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "pipe:1"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n, i = out_w * out_h * 3, 0
    try:
        while True:
            buf = read_exact(p.stdout, n)
            if buf is None:
                break
            yield i / fps, np.frombuffer(buf, np.uint8).reshape(out_h, out_w, 3)
            i += 1
    finally:
        p.stdout.close()
        p.terminate()


# ── tiling and merging ───────────────────────────────────────────────────────

def starts(total, size, stride):
    if total <= size:
        return [0]
    xs = list(range(0, total - size, stride))
    xs.append(total - size)
    return xs


def tiles_of(frame, stride_x=960, stride_y=540):
    H, W = frame.shape[:2]
    out = []
    for y in starts(H, TILE_H, stride_y):
        for x in starts(W, TILE_W, stride_x):
            out.append((x, y, frame[y:y + TILE_H, x:x + TILE_W]))
    return out


def merge(boxes, thr=0.5, contain=0.8):
    """Greedy duplicate removal across tile overlaps. boxes: (x1, y1, x2, y2, conf)."""
    keep = []
    for b in sorted(boxes, key=lambda b: -b[4]):
        ok = True
        for k in keep:
            ix = max(0.0, min(b[2], k[2]) - max(b[0], k[0]))
            iy = max(0.0, min(b[3], k[3]) - max(b[1], k[1]))
            inter = ix * iy
            a, c = (b[2] - b[0]) * (b[3] - b[1]), (k[2] - k[0]) * (k[3] - k[1])
            if inter / max(a + c - inter, 1e-9) > thr or inter / max(min(a, c), 1e-9) > contain:
                ok = False
                break
        if ok:
            keep.append(b)
    return keep


def ball_class_ids(model):
    names = dict(model.names)
    if len(names) <= 1:
        return None
    ids = {i for i, n in names.items() if "ball" in str(n).lower()}
    if not ids:
        raise SystemExit(f"ball model classes {sorted(names.values())}: none looks like a ball")
    return ids


def infer(model, tiles, imgsz, conf, device, want):
    """Run a model on the tiles; want(cls_id, name) -> bool. -> boxes in frame coordinates."""
    out = []
    names = model.names
    for s in range(0, len(tiles), 8):
        chunk = tiles[s:s + 8]
        results = model([t[2] for t in chunk], imgsz=imgsz, conf=conf, device=device, verbose=False)
        for (ox, oy, _), res in zip(chunk, results):
            for b in res.boxes:
                c = int(b.cls[0])
                if not want(c, str(names.get(c, ""))):
                    continue
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                out.append((x1 + ox, y1 + oy, x2 + ox, y2 + oy, float(b.conf[0])))
    return out


# ── one clip, one variant ────────────────────────────────────────────────────

def analyse(path, label, players_model, ball_model, device, fps, start, duration, hwaccel, out_w,
            player_conf, ball_conf, imgsz, outdir, save_at):
    w, h, dur = probe(path)
    ball_ids = ball_class_ids(ball_model) if ball_model is not None else None
    per, shown = [], None
    t0 = time.time()
    for t, fr in frames(path, w, h, fps, start, duration, hwaccel, out_w):
        tl = tiles_of(fr)
        pl = merge(infer(players_model, tl, imgsz, player_conf, device, lambda c, n: n == "person"))
        bl = []
        if ball_model is not None:
            bl = merge(infer(ball_model, tl, imgsz, ball_conf, device,
                             lambda c, n: ball_ids is None or c in ball_ids), thr=0.3)
            bl = [b for b in bl if (b[2] - b[0]) * (b[3] - b[1]) <= BALL_MAX_AREA]
        per.append({"t": round(t, 2), "players": pl, "balls": bl})
        if shown is None and t >= save_at:
            shown = (fr.copy(), pl, bl)
    n = len(per)
    if n == 0:
        raise SystemExit(f"{path}: no frames decoded")
    secs = time.time() - t0
    pconf = [b[4] for f in per for b in f["players"]]
    ph = [b[3] - b[1] for f in per for b in f["players"]]
    bconf = [b[4] for f in per for b in f["balls"]]
    top_ball = [max((b[4] for b in f["balls"]), default=0.0) for f in per]
    res = {
        "label": label, "file": os.path.basename(path), "frame": f"{(out_w or w)}x{int(round(h * (out_w or w) / w))}",
        "frames": n, "tiles_per_frame": len(tl),
        "players_per_frame": round(sum(len(f["players"]) for f in per) / n, 2),
        "frames_with_player": round(sum(1 for f in per if f["players"]) / n, 3),
        "player_conf_mean": round(float(np.mean(pconf)), 3) if pconf else None,
        "player_height_px_median": round(float(np.median(ph)), 1) if ph else None,
        "frames_with_ball": round(sum(1 for f in per if f["balls"]) / n, 3),
        "ball_cands_per_frame": round(len(bconf) / n, 2),
        "ball_conf_mean": round(float(np.mean(bconf)), 3) if bconf else None,
        "ball_conf_best_mean": round(float(np.mean(top_ball)), 3),
        "seconds_per_frame": round(secs / n, 2),
    }
    if shown is not None:
        fr, pl, bl = shown
        for x1, y1, x2, y2, c in pl:
            cv2.rectangle(fr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
        for x1, y1, x2, y2, c in bl:
            cv2.circle(fr, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 22, (0, 0, 255), 2)
        os.makedirs(outdir, exist_ok=True)
        base = os.path.join(outdir, label.replace("@", "_at_"))
        small = cv2.resize(fr, (1920, int(round(fr.shape[0] * 1920 / fr.shape[1]))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(base + "_frame.jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if pl:                                        # a 640x480 crop (native pixels) around the most confident person
            x1, y1, x2, y2, c = max(pl, key=lambda b: b[4])
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            x0 = int(np.clip(cx - 320, 0, max(0, fr.shape[1] - 640)))
            y0 = int(np.clip(cy - 240, 0, max(0, fr.shape[0] - 480)))
            cv2.imwrite(base + "_player_crop.jpg", fr[y0:y0 + 480, x0:x0 + 640], [cv2.IMWRITE_JPEG_QUALITY, 95])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--out", default="camera_compare_out")
    ap.add_argument("--players", default="models/yolo11s.pt")
    ap.add_argument("--ball", default="models/ball_v4.pt")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--match-width", type=int, default=2560)
    ap.add_argument("--player-conf", type=float, default=0.20)   # CONF_PLAYERS in autopan_infer.py
    ap.add_argument("--ball-conf", type=float, default=0.40)     # CONF_BALL in autopan_infer.py
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--hwaccel", action="store_true", help="ffmpeg -hwaccel videotoolbox (macOS)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    from ultralytics import YOLO
    device = a.device
    if device is None:
        try:
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    pm, bm = YOLO(a.players), YOLO(a.ball)
    out = os.path.expanduser(a.out)
    rows = []
    for spec in a.clip:
        label, _, path = spec.partition("=")
        path = os.path.expanduser(path)
        w, _, _ = probe(path)
        variants = [(label, None)]
        if w > a.match_width:
            variants.append((f"{label}@{a.match_width}", a.match_width))
        for lab, ow in variants:
            print(f"… {lab}", flush=True)
            rows.append(analyse(path, lab, pm, bm, device, a.fps, a.start, a.duration, a.hwaccel, ow,
                                a.player_conf, a.ball_conf, a.imgsz, out, save_at=a.duration / 2))
    cols = [("label", "variant", 16), ("frame", "frame", 11), ("frames", "n", 3), ("players_per_frame", "ppl/frm", 8),
            ("frames_with_player", "frm w/ ppl", 10), ("player_conf_mean", "ppl conf", 8),
            ("player_height_px_median", "ppl h px", 8), ("frames_with_ball", "frm w/ ball", 11),
            ("ball_cands_per_frame", "ball/frm", 8), ("ball_conf_best_mean", "ball conf*", 10),
            ("seconds_per_frame", "s/frame", 7)]
    print()
    print("  ".join(f"{h:<{wd}}" for _, h, wd in cols))
    for r in rows:
        print("  ".join(f"{str(r[k]):<{wd}}" for k, _, wd in cols))
    print("\n* mean over frames of the best ball candidate's confidence (0 when none)")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {os.path.join(out, 'summary.json')} and the pictures next to it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
