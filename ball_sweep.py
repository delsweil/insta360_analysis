#!/usr/bin/env python3
"""
ball_sweep.py — does a bigger detector input find more of the ball?  (companion of detect_sweep.py)

Samples frames from a game, runs the ball model through the app's own detection views at several
input sizes, and reports for each size and confidence:
    frames with a ball    share of frames with at least one candidate on the pitch
    candidates / frame    clutter: far more than ~2 means false positives (boots, line marks, ...)
and saves a contact sheet of candidates that only the largest size finds, so you can see by eye
whether they are the ball.  Contact-sheet crops are labelled with the confidence.

    python ball_sweep.py --job /path/to/job_folder --device mps
        [--sizes 1280 1600 1920 2560] [--conf 0.25] [--frames 30] [--step 20] [--start S]

The app's own setting is size 1280, conf 0.40 (extract_detections.CONF_BALL).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import extract_detections as ED
import kf_core as K


def candidates(frame, views, model, device, mask, imgsz, conf, class_ids):
    """-> [(lon, lat, conf, view_index, (cx, cy))], on the pitch only, owner-view rule as in the extractor."""
    out = []
    for vi, v in enumerate(views):
        img = v.sample(frame)
        vmask = v.sample_mask(mask) if mask is not None else None
        res = model(img, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        for b in res.boxes:
            if class_ids is not None and int(b.cls[0]) not in class_ids:
                continue
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            if (x2 - x1) * (y2 - y1) > ED.VIEW_W * ED.VIEW_H * 0.02:
                continue
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            lon, lat = v.lonlat(cx, cy)
            if ED.nearest_view(views, lon) != vi:
                continue
            if vmask is not None and not vmask[int(np.clip(cy, 0, vmask.shape[0] - 1)), int(np.clip(cx, 0, vmask.shape[1] - 1))]:
                continue
            out.append((lon, lat, float(b.conf[0]), vi, (cx, cy)))
    return out


def near(a, lst, tol=0.5):
    return any(abs(a[0] - q[0]) < tol and abs(a[1] - q[1]) < tol for q in lst)


def main(argv=None, model=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job", required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--ball", default=None, help="default: the model the app uses (pipeline.BALL_MODEL)")
    p.add_argument("--device", default=None)
    p.add_argument("--start", type=float, default=None)
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--step", type=float, default=20.0)
    p.add_argument("--sizes", type=int, nargs="+", default=[1280, 1600, 1920, 2560])
    p.add_argument("--conf", type=float, default=0.25, help="low confidence used to look for extra candidates")
    p.add_argument("--app-conf", type=float, default=ED.CONF_BALL)
    p.add_argument("--events", default=None,
                   help="the events CSV exported from the web player (times are in the MERGED video): samples the middle "
                        "of each shot/goal/set piece of this job's video instead of evenly spaced frames")
    p.add_argument("--kinds", nargs="+", default=["shot", "goal", "setpiece"])
    p.add_argument("--out", default="ball_sweep_out")
    a = p.parse_args(argv)

    jd = os.path.abspath(a.job)
    cams = json.load(open(os.path.join(jd, "camera.json")))
    spec = json.load(open(os.path.join(jd, "job.json")))
    i = a.index
    video = cams[i]["path"]
    man, auto = os.path.join(jd, f"pitch{i}.manual.json"), os.path.join(jd, f"pitch{i}.json")
    calib = man if os.path.isfile(man) else auto if os.path.isfile(auto) else None
    start = a.start if a.start is not None else float(spec["videos"][i].get("start") or 0.0) + 120.0
    iv_fov, tilt = 112.0, -22.0                                   # pipeline.EXTRACT_ARGS
    print(f"video {video}\n  outline {calib or 'none'}  tilt {tilt}  iv_fov {iv_fov}")

    info = K.probe_video(video)
    raw = not K.is_equirect_size(info["width"], info["height"])
    poly = ED.load_polygon(calib) if calib else None
    mask = ED.polygon_mask(poly) if poly is not None else None
    views = ED.make_views(-80.0, 80.0, tilt, 40.0, fov=ED.VIEW_FOV, legacy_fov=False)
    device = ED.pick_device(a.device)
    if model is None:
        from ultralytics import YOLO
        here = os.path.dirname(os.path.abspath(__file__))
        model = YOLO(a.ball or os.path.join(here, "models", "ball_v4.pt"))
    class_ids = ED.ball_class_ids(model)
    os.makedirs(a.out, exist_ok=True)

    times = [start + k * a.step for k in range(a.frames)]
    if a.events:
        import csv
        # merged time -> time in the source video i: the merged video is the job's videos one after the other
        offs, acc = [], 0.0
        for j, v in enumerate(spec["videos"]):
            s0 = float(v.get("start") or 0.0)
            e0 = float(v["end"]) if v.get("end") else float(cams[j]["duration"])
            offs.append((acc, acc + (e0 - s0), s0))
            acc += e0 - s0
        lo_m, hi_m, s0 = offs[i]
        times = []
        for r in csv.DictReader(open(a.events)):
            if r["kind"] in a.kinds:
                mid = 0.5 * (float(r["video_t"]) + float(r["end_t"] or r["video_t"]))
                if lo_m <= mid < hi_m:
                    times.append(s0 + mid - lo_m)
        times = times[:a.frames]
        print(f"{len(times)} events of kind {a.kinds} fall in video {i + 1} (merged {lo_m:.0f}-{hi_m:.0f} s)")

    sizes = sorted(set(a.sizes))
    runs = {s: [] for s in sizes}
    secs = {s: 0.0 for s in sizes}
    frames = []
    for k, t in enumerate(times):
        if t >= info["duration"]:
            break
        proc = ED.open_frames(video, t, 1.0, 1.0, raw, iv_fov)
        fr = ED.read_frame(proc)
        proc.stdout.close(); proc.terminate()
        if fr is None:
            break
        frames.append((t, fr.copy()))
        for s in sizes:
            t0 = time.time()
            runs[s].append(candidates(fr, views, model, device, mask, s, a.conf, class_ids))
            secs[s] += time.time() - t0
        print(f"  frame {k + 1}/{len(times)}  t={t:.0f}s", end="\r", flush=True)
    n = max(1, len(frames))
    print()

    report = {"frames": len(frames), "device": device, "settings": {}}
    print(f"{len(frames)} frames, {device}; ball candidates on the pitch")
    print(f"{'imgsz':>6} | {'conf>=' + str(a.app_conf):>10} {'frames':>7} {'cand/f':>7} | {'conf>=' + str(a.conf):>10} {'frames':>7} {'cand/f':>7} | {'s/frame':>8}")
    for s in sizes:
        row = {}
        for tag, c in (("app", a.app_conf), ("low", a.conf)):
            per = [[d for d in f if d[2] >= c] for f in runs[s]]
            row[tag] = {"frames_with_ball_pct": round(100 * sum(1 for f in per if f) / n, 1),
                        "candidates_per_frame": round(sum(len(f) for f in per) / n, 2)}
        row["sec_per_frame"] = round(secs[s] / n, 2)
        report["settings"][s] = row
        print(f"{s:>6} | {'':>10} {str(row['app']['frames_with_ball_pct']) + '%':>7} {row['app']['candidates_per_frame']:>7} | "
              f"{'':>10} {str(row['low']['frames_with_ball_pct']) + '%':>7} {row['low']['candidates_per_frame']:>7} | {row['sec_per_frame']:>8}")

    import cv2
    base, top = sizes[0], sizes[-1]
    # what the tracker would follow: the most confident candidate per frame, base size next to the others
    def best_crop(i2, fr, s_):
        c = [d for d in runs[s_][i2] if d[2] >= a.conf]
        if not c:
            return np.zeros((160, 160, 3), np.uint8)
        d = max(c, key=lambda q: q[2])
        img = views[d[3]].sample(fr)
        cx, cy = int(d[4][0]), int(d[4][1])
        x = cv2.resize(img[max(0, cy - 40):cy + 40, max(0, cx - 40):cx + 40], (160, 160), interpolation=cv2.INTER_CUBIC)
        cv2.putText(x, f"{s_} {d[2]:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.circle(x, (80, 80), 22, (0, 255, 255), 1)
        return x
    rows = []
    for i2, (t, fr) in enumerate(frames[:24]):
        rows.append(np.hstack([best_crop(i2, fr, s_) for s_ in sizes]))
    if rows:
        per = 2 if len(sizes) <= 4 else 1
        while len(rows) % per:
            rows.append(np.zeros_like(rows[0]))
        cv2.imwrite(os.path.join(a.out, "ball_best_per_frame.jpg"),
                    np.vstack([np.hstack(rows[r:r + per]) for r in range(0, len(rows), per)]))
        print(f"look at {a.out}/ball_best_per_frame.jpg — one row per frame, one crop per size: is the circled thing the ball?")
    crops = []
    for i2, (t, fr) in enumerate(frames):
        for d in runs[top][i2]:
            if d[2] >= a.conf and not near(d, runs[base][i2]) and len(crops) < 48:
                img = views[d[3]].sample(fr)
                cx, cy = int(d[4][0]), int(d[4][1])
                c = img[max(0, cy - 40):cy + 40, max(0, cx - 40):cx + 40]
                if c.size:
                    c = cv2.resize(c, (160, 160), interpolation=cv2.INTER_CUBIC)
                    cv2.putText(c, f"{d[2]:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.circle(c, (80, 80), 22, (0, 255, 255), 1)
                    crops.append(c)
    if crops:
        cols = 8
        while len(crops) % cols:
            crops.append(np.zeros_like(crops[0]))
        cv2.imwrite(os.path.join(a.out, f"ball_extra_{base}_vs_{top}.jpg"),
                    np.vstack([np.hstack(crops[r:r + cols]) for r in range(0, len(crops), cols)]))
        print(f"look at {a.out}/ball_extra_{base}_vs_{top}.jpg — a ball in the circle = the bigger size earns its time")
    json.dump(report, open(os.path.join(a.out, "ball_sweep_report.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
