#!/usr/bin/env python3
"""
detect_sweep.py — how much does the player detector miss at the current settings?

The detection views are 1280x720 and are fed to the detector at imgsz=960, i.e. shrunk to 75 %.
A far-side player is only ~15-25 px tall in the source, so that shrink hurts most there. This
tool samples frames from a clip, runs the SAME views/owner rules as extract_detections.py at
several settings, and reports for each setting how many players it finds, how many *small* ones
(the far side), the time per frame, and — against the union of all settings — how many of the
small players the current setting misses.  It also saves a contact sheet of the players that only
the larger settings find, so you can judge by eye whether they are real people or ghosts.

    python detect_sweep.py --job /path/to/job_folder --device mps      # the app's own setup (recommended)

    python detect_sweep.py --video GAME.mp4 --calib calibration/pitch.json \
        --start 600 --frames 30 --step 20 --device mps
        [--sizes 960 1280 1600 1920] [--small-px 40] [--players models/yolo11s.pt]

Use the same --calib / --tilt / --iv-fov / --view-fov you use for the board pass.
Nothing is written except <out>/sweep_report.json and <out>/sweep_extra_*.jpg.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import extract_detections as ED
import kf_core as K


def owned_players(frame, views, model, device, mask, imgsz, conf):
    """-> [(lon, lat, box_h, conf, view_index, (x1,y1,x2,y2))] using the extractor's own ownership rule."""
    ED.IMGSZ_PLAYERS = imgsz
    out = []
    for vi, v in enumerate(views):
        img = v.sample(frame)
        vmask = v.sample_mask(mask) if mask is not None else None
        res = model(img, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        for b in res.boxes:
            if model.names.get(int(b.cls[0]), "") != "person":
                continue
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            cx, foot = (x1 + x2) / 2, y2
            lon, lat = v.lonlat(cx, foot)
            if ED.nearest_view(views, lon) != vi:
                continue
            on = True
            if vmask is not None:
                on = bool(vmask[int(np.clip(foot, 0, vmask.shape[0] - 1)), int(np.clip(cx, 0, vmask.shape[1] - 1))])
            if not on:
                continue                                  # only players on the pitch matter for the board
            out.append((lon, lat, y2 - y1, float(b.conf[0]), vi, (x1, y1, x2, y2)))
    return out


def match(a, b, tol_deg=0.8):
    """Is detection a (lon, lat, ...) within tol of anything in list b?"""
    return any(abs(a[0] - q[0]) < tol_deg and abs(a[1] - q[1]) < tol_deg for q in b)


def main(argv=None, model=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default=None)
    p.add_argument("--job", default=None,
                   help="an app job folder: takes the video, the pitch outline (hand-drawn if there is one), "
                        "tilt, unwrapping and lon range exactly as the app's player pass uses them")
    p.add_argument("--index", type=int, default=0, help="which video of the job (0 = first)")
    p.add_argument("--calib", default=None)
    p.add_argument("--tilt", type=float, default=None)
    p.add_argument("--iv-fov", type=float, default=None)
    p.add_argument("--view-fov", type=float, default=ED.VIEW_FOV)
    p.add_argument("--view-spacing", type=float, default=40.0)
    p.add_argument("--lon-range", type=float, nargs=2, default=None)
    p.add_argument("--players", default="models/yolo11s.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--start", type=float, default=None)
    p.add_argument("--frames", type=int, default=30, help="how many frames to sample")
    p.add_argument("--step", type=float, default=20.0, help="seconds between sampled frames")
    p.add_argument("--sizes", type=int, nargs="+", default=[960, 1280, 1600, 1920])
    p.add_argument("--conf", type=float, default=ED.CONF_PLAYERS)
    p.add_argument("--small-px", type=float, default=40.0, help="a box shorter than this counts as far-side")
    p.add_argument("--out", default="sweep_out")
    a = p.parse_args(argv)

    if a.job:
        import json as _j
        jd = os.path.abspath(a.job)
        cams = _j.load(open(os.path.join(jd, "camera.json")))
        spec = _j.load(open(os.path.join(jd, "job.json")))
        i = a.index
        a.video = a.video or cams[i]["path"]
        man, auto = os.path.join(jd, f"pitch{i}.manual.json"), os.path.join(jd, f"pitch{i}.json")
        a.calib = a.calib or (man if os.path.isfile(man) else auto if os.path.isfile(auto) else None)
        a.iv_fov = 112.0 if a.iv_fov is None else a.iv_fov          # pipeline.EXTRACT_ARGS
        a.tilt = -22.0 if a.tilt is None else a.tilt                 # pipeline.DETECT_TILT
        a.lon_range = a.lon_range or (-80.0, 80.0)
        if a.start is None:
            a.start = float(spec["videos"][i].get("start") or 0.0) + 120.0   # a couple of minutes into the half
        print(f"job {jd}\n  video {a.video}\n  pitch outline {a.calib or 'none'}  tilt {a.tilt}  iv_fov {a.iv_fov}")
    if not a.video:
        p.error("give --video or --job")
    if a.start is None:
        a.start = 0.0
    info = K.probe_video(a.video)
    raw = not K.is_equirect_size(info["width"], info["height"])
    iv_fov = K.DEFAULT_IV_FOV if a.iv_fov is None else a.iv_fov
    poly = ED.load_polygon(a.calib) if a.calib else None
    mask = ED.polygon_mask(poly) if poly is not None else None
    tilt = a.tilt if a.tilt is not None else (K.tilt_from_calib(a.calib) if a.calib else -15.0)
    lo, hi = a.lon_range or (ED.polygon_lon_range(poly) if poly is not None else (-80.0, 80.0))
    views = ED.make_views(lo, hi, tilt, a.view_spacing, fov=a.view_fov, legacy_fov=iv_fov == K.DEFAULT_IV_FOV)
    device = ED.pick_device(a.device)
    if model is None:
        from ultralytics import YOLO
        model = YOLO(a.players)
    os.makedirs(a.out, exist_ok=True)

    sizes = sorted(set(a.sizes))
    runs = {s: [] for s in sizes}                       # size -> per frame detections
    secs = {s: 0.0 for s in sizes}
    frames = []
    for k in range(a.frames):
        t = a.start + k * a.step
        if t >= info["duration"]:
            break
        proc = ED.open_frames(a.video, t, 1.0, 1.0, raw, iv_fov)
        fr = ED.read_frame(proc)
        proc.stdout.close(); proc.terminate()
        if fr is None:
            break
        frames.append((t, fr.copy()))
        for s in sizes:
            t0 = time.time()
            runs[s].append(owned_players(fr, views, model, device, mask, s, a.conf))
            secs[s] += time.time() - t0
        print(f"  frame {k + 1}/{a.frames}  t={t:.0f}s", end="\r", flush=True)
    n = max(1, len(frames))
    print()

    # union of all settings = the best guess of who is really there (biased towards the big sizes)
    report = {"frames": len(frames), "small_px": a.small_px, "device": device, "settings": {}}
    union = []
    for i in range(len(frames)):
        u = []
        for s in sizes:
            for d in runs[s][i]:
                if not match(d, u):
                    u.append(d)
        union.append(u)
    n_small_union = sum(1 for u in union for d in u if d[2] < a.small_px)
    for s in sizes:
        tot = sum(len(f) for f in runs[s])
        small = sum(1 for f in runs[s] for d in f if d[2] < a.small_px)
        miss = sum(1 for i, u in enumerate(union) for d in u if d[2] < a.small_px and not match(d, runs[s][i]))
        report["settings"][s] = {"players_per_frame": round(tot / n, 2), "small_per_frame": round(small / n, 2),
                                 "small_missed_vs_union_pct": round(100 * miss / max(1, n_small_union), 1),
                                 "sec_per_frame": round(secs[s] / n, 2)}
    report["union_small_per_frame"] = round(n_small_union / n, 2)

    # contact sheet: players found at the biggest size but not at the current (smallest) one
    import cv2
    base, top = sizes[0], sizes[-1]
    crops = []
    for i, (t, fr) in enumerate(frames):
        for d in runs[top][i]:
            if not match(d, runs[base][i]) and len(crops) < 48:
                v = views[d[4]]
                img = v.sample(fr)
                x1, y1, x2, y2 = [int(round(c)) for c in d[5]]
                pad = max(12, int(0.6 * (y2 - y1)))
                c = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
                if c.size:
                    c = cv2.resize(c, (160, 160), interpolation=cv2.INTER_CUBIC)
                    cv2.putText(c, f"{d[3]:.2f} h{d[2]:.0f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    crops.append(c)
    if crops:
        cols = 8
        while len(crops) % cols:
            crops.append(np.zeros_like(crops[0]))
        sheet = np.vstack([np.hstack(crops[r:r + cols]) for r in range(0, len(crops), cols)])
        cv2.imwrite(os.path.join(a.out, f"sweep_extra_{base}_vs_{top}.jpg"), sheet)
    with open(os.path.join(a.out, "sweep_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"{len(frames)} frames, {device}, small = box shorter than {a.small_px:g}px")
    print(f"{'imgsz':>6} {'players/f':>10} {'small/f':>8} {'small missed':>13} {'s/frame':>8}")
    for s in sizes:
        r = report["settings"][s]
        print(f"{s:>6} {r['players_per_frame']:>10} {r['small_per_frame']:>8} "
              f"{str(r['small_missed_vs_union_pct']) + '%':>13} {r['sec_per_frame']:>8}")
    if crops:
        print(f"look at {a.out}/sweep_extra_{base}_vs_{top}.jpg — real people there = the bigger size earns its time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
