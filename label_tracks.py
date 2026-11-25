#!/usr/bin/env python3
"""label_tracks.py — make ground truth for the tracker: you follow a few players through a stretch of the rendered video.

The tracker cuts every player into pieces and, at the end of each piece, guesses which piece is the continuation. Nobody has
ever checked those guesses. This tool makes a web page (one folder, open index.html in a browser; no server) that shows the real
video frames with the detector's own marks, and asks you the same question the tracker answers: "he was here; which of these
is him afterwards?", with close-up pictures cut out of the rendered video. Your answers are the truth the tracker is scored on.

  cd <your insta360_analysis folder>
  python3 label_tracks.py /Users/davidelsweiler/FootballAutopan/TSV_Buch_A-Jugend --part 1 --from 21:30 --to 23:30 --assign 6=B 5=REF
      -> writes <game folder>/labels_part1_21-30/index.html  (about 2 minutes of work for 2 minutes of video; open it in Chrome / Safari)

  python3 label_tracks.py --score labels_part1_21-30/labels.json     # after you pressed "Download labels" and moved the file there

--from / --to are times in the PART's video (mm:ss or seconds), the part given by --part (1 = first part), not the joined video.
The part video is found as <game folder>/*part<N>*autopan*.mp4 (name it with --video if yours differs).

What is drawn: the picture position of every detected player's FEET is computed with the renderer's own camera maths (pan, tilt, fov,
projection and lens from the keyframe files), not estimated from the board. The check for it is  --selftest  (makes a clip with ffmpeg and
looks for the marks).

Needs track_stitch.py, board_tracks.py, ffmpeg and opencv in the same environment as the other scripts.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

K_ANGLE = 0.75
MIN_CONF = 0.3
FORMAT = "football-autopan-labels"
WINDOW, REACH, TOP = 20.0, 1.6, 6      # generous: the labelling should see the right piece if it exists (--window, --reach, --top)


def parse_time(s: str) -> float:
    s = str(s).strip()
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        out = 0.0
        for p in parts:
            out = out * 60 + p
        return out
    return float(s)


def fmt(sec: float) -> str:
    s = max(0, int(round(sec)))
    return f"{s // 60}:{s % 60:02d}"


# ── camera ───────────────────────────────────────────────────────────────────

class Camera:
    """Where a (lon, lat) of the detections falls in the rendered picture, with the maths of render_keyframes.py."""

    def __init__(self, kf_path: str, calib: str | None, width: int, height: int):
        import kf_core as K
        self.K = K
        meta = K.load_meta(kf_path)
        self.meta = meta
        tilt = meta.get("tilt_deg")
        self.tilt_source = "meta"
        if tilt is None and calib:
            tilt = K.tilt_from_calib(calib)
            self.tilt_source = "pitch outline"
        if tilt is None:
            tilt = 0.0
            self.tilt_source = "none (0)"
        self.legacy_fov = bool(meta.get("legacy_fov", False))
        self.projection = meta.get("projection", "flat" if meta else "cylindrical")
        if self.projection not in K.PROJECTIONS:
            self.projection = "flat"
        self.lens = meta.get("lens", "legacy")
        self.iv_fov = float(meta["iv_fov"]) if meta.get("iv_fov") is not None else K.DEFAULT_IV_FOV
        self.interp = meta.get("interp", "pchip")
        self.path = K.Path(K.KeyframeSet.load(kf_path).items, self.interp, base_tilt=float(tilt))
        self.base_tilt = float(tilt)
        self.W, self.H = width, height
        self.L = None
        if self.lens != "legacy":
            import lens as L
            self.L = L

    def describe(self) -> str:
        return (f"projection {self.projection}, lens {self.lens}, {'legacy' if self.legacy_fov else 'square-pixel'} fov, "
                f"tilt {self.base_tilt:+.1f} ({self.tilt_source}){' with per-keyframe tilt' if getattr(self.path, 'has_tilt', False) else ''}")

    def view_at(self, t: float):
        """-> (yaw, pitch, h_fov) of the picture at raw time t, in the geometry the video was rendered with."""
        K = self.K
        tq = np.array([float(t)])
        pan, fov = self.path.at(tq)
        tilt = self.path.tilt_at(tq)[0] if getattr(self.path, "has_tilt", False) else self.base_tilt
        p = K.to_v360_params(float(pan[0]), float(tilt), float(fov[0]), self.W, self.H, self.legacy_fov, self.projection)
        if self.L is not None:
            c = self.L.convert_view(p["yaw"], p["pitch"], p["h_fov"], self.H / self.W, self.iv_fov)
            return c["yaw"], c["pitch"], c["h_fov"]
        return p["yaw"], p["pitch"], p["h_fov"]

    def project(self, lon: float, lat: float, view):
        """-> (u, v) as fractions of the picture (0..1, v downwards) or None when the point is behind the camera."""
        K = self.K
        yaw, pitch, h_fov = view
        lo, la = math.radians(lon), math.radians(lat)
        p = np.array([math.cos(la) * math.sin(lo), math.sin(la), math.cos(la) * math.cos(lo)])
        if self.L is not None:                      # the detections are on the legacy unwrapping; the render looks at the true lens
            px, py = self.L.legacy_dir_to_raw(p, self.iv_fov)
            p = np.asarray(self.L.raw_to_true_dir(np.clip(px, 0, self.L.RAW_W - 1), np.clip(py, 0, self.L.RAW_H - 1)), float)
        y_, pt = math.radians(yaw), math.radians(pitch)
        f = np.array([math.cos(pt) * math.sin(y_), math.sin(pt), math.cos(pt) * math.cos(y_)])
        r = np.array([math.cos(y_), 0.0, -math.sin(y_)])
        u = np.cross(f, r)
        xm, ym = K.plane_extent(h_fov, self.W, self.H, self.legacy_fov, self.projection)
        if self.projection == "cylindrical":
            a, b = float(p @ r), float(p @ f)
            rho = math.hypot(a, b)
            if rho <= 1e-6:
                return None
            x, yy = math.atan2(a, b), float(p @ u) / rho
        else:
            d = float(p @ f)
            if d <= 1e-6:
                return None
            x, yy = float(p @ r) / d, float(p @ u) / d
        return 0.5 + x / (2 * xm), 0.5 - yy / (2 * ym)


# ── reading the part ─────────────────────────────────────────────────────────

def find_video(folder: str, part: int, override: str | None) -> str:
    if override:
        return override
    hits = sorted(glob.glob(os.path.join(folder, f"*part{part}*autopan*.mp4")) + glob.glob(os.path.join(folder, f"*part{part}*.mp4")))
    hits = [h for h in hits if "full_game" not in os.path.basename(h)]
    if not hits:
        sys.exit(f"No video for part {part} in {folder} (looked for *part{part}*autopan*.mp4); name it with --video.")
    return hits[0]


def probe(video: str):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", video], capture_output=True, text=True).stdout.strip().split(",")
    return int(out[0]), int(out[1])


def load_part(det: str, kf: str, calib: str | None, t0: float, t1: float, W: int, H: int, clusters, min_conf: float):
    """-> dict of arrays for the detections between t0 and t1 (seconds in the part video)."""
    import auto_keyframes as AK
    import kf_core as K
    import scene_check as S
    hdr, frames = AK.load_detections(det)
    meta = K.load_meta(kf)
    t_in = float(meta["trim_in"]) if meta.get("trim_in") is not None else float(hdr.get("start", 0.0))
    fy = S.view_fy(hdr)
    views = np.array(hdr["view_lons"], float)
    cam = Camera(kf, calib, W, H)
    out = {k: [] for k in ("fr", "vt_m", "lon", "lat", "h", "conf", "x", "y", "team", "u", "v", "hf")}
    moments = []
    fn = getattr(clusters, "team_of_rescue", None) if clusters is not None else None
    for r in frames:
        vt = float(r["t"]) - t_in
        if vt < t0 or vt > t1:
            continue
        mi = len(moments)
        moments.append(vt)
        view = cam.view_at(float(r["t"]))
        for p in r["players"]:
            if p[3] < min_conf or (len(p) > 4 and p[4] is False):
                continue
            lon, lat, h = float(p[0]), float(p[1]), float(p[2])
            centre = float(views[np.argmin(np.abs(views - lon))])
            x, y = S.ground_position(lon, h, centre, fy, K_ANGLE)
            pos = cam.project(lon, lat, view)
            if pos is None:
                continue
            head = cam.project(lon, lat + math.degrees(h / fy), view)
            hf = abs(head[1] - pos[1]) if head else 0.1
            team = 2
            if fn is not None and len(p) > 7 and p[5] is not None and h >= 14:
                t_ = int(fn(np.array([[p[5], p[6], p[7]]], float), 1.5)[0])
                team = t_ if t_ in (0, 1) else 2
            for k, v in (("fr", mi), ("lon", lon), ("lat", lat), ("h", h), ("conf", float(p[3])), ("x", x), ("y", y),
                         ("team", team), ("u", pos[0]), ("v", pos[1]), ("hf", hf)):
                out[k].append(v)
    out["vt_m"] = np.array(moments, float)
    for k in out:
        if k != "vt_m":
            out[k] = np.array(out[k])
    out["cam"] = cam
    return out


def extract_frames(video: str, t0: float, t1: float, fps: float, tmp: str, max_w: int = 1920):
    os.makedirs(tmp, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", video,
           "-vf", f"fps={fps:g},scale='min({max_w},iw)':-2", "-q:v", "3", os.path.join(tmp, "f_%05d.jpg")]
    subprocess.run(cmd, check=True)
    return len(glob.glob(os.path.join(tmp, "f_*.jpg")))


class Frames:
    def __init__(self, tmp: str, t0: float, fps: float):
        import cv2
        self.cv2 = cv2
        self.tmp, self.t0, self.fps = tmp, t0, fps
        self.cache = {}
        self.n = len(glob.glob(os.path.join(tmp, "f_*.jpg")))

    def at(self, vt: float):
        i = int(np.clip(round((vt - self.t0) * self.fps) + 1, 1, self.n))
        if i not in self.cache:
            if len(self.cache) > 40:
                self.cache.clear()
            self.cache[i] = self.cv2.imread(os.path.join(self.tmp, f"f_{i:05d}.jpg"))
        return self.cache[i]

    def crop(self, vt, cx, cy, w, h, ow, oh, mark=None):
        """cut (cx, cy, w, h) in fractions-of-pixel units of the frame, scaled to ow x oh"""
        cv2 = self.cv2
        img = self.at(vt)
        sx, sy = ow / w, oh / h
        M = np.array([[sx, 0, -(cx - w / 2) * sx], [0, sy, -(cy - h / 2) * sy]], float)
        out = cv2.warpAffine(img, M, (ow, oh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(70, 70, 70))
        if mark is not None:
            mx, my = (mark[0] - (cx - w / 2)) * sx, (mark[1] - (cy - h / 2)) * sy
            cv2.circle(out, (int(mx), int(my)), 5, (0, 215, 255), 2, cv2.LINE_AA)
        return out

    def save(self, path, img):
        self.cv2.imwrite(path, img, [self.cv2.IMWRITE_JPEG_QUALITY, 88])

    @property
    def size(self):
        im = self.at(self.t0)
        return im.shape[1], im.shape[0]


# ── pieces ───────────────────────────────────────────────────────────────────

def build_pieces(P):
    import board_tracks as BT
    fr = P["fr"].astype(int)
    pid, pieces = BT.build(P["vt_m"], fr, P["x"], P["y"], P["team"])
    nxt, prv = BT.candidates(pieces, P["vt_m"], fr, P["x"], P["y"], pid, window=WINDOW, loose=True, reach=REACH, top=TOP)
    return pid, pieces, nxt, prv, BT


def make_site(a):
    import track_stitch as TS
    folder = a.folder
    part = a.part
    dets = TS.find_parts(folder)
    if len(dets) < part:
        sys.exit(f"Only {len(dets)} parts of detections found in {folder}.")
    det, kf = dets[part - 1]
    if not kf:
        sys.exit(f"No keyframes file found for {os.path.basename(det)}.")
    cal = TS.find_outlines(folder, dets, a.calib)[part - 1]
    video = find_video(folder, part, a.video)
    W, H = probe(video)
    t0, t1 = parse_time(a.t_from), parse_time(a.t_to)
    if t1 <= t0:
        sys.exit("--to must be later than --from.")
    out_dir = a.out or os.path.join(folder, f"labels_part{part}_{a.t_from.replace(':', '-')}")
    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)

    lab_path = a.labels or next(iter(sorted(glob.glob(os.path.join(folder, "*team_labels.json")))), None)
    clusters = None
    if lab_path:
        import team_labels as TL
        clusters = TL.Labelled(lab_path)
        for item in a.assign or []:
            g, who = item.split("="); g = int(g); who = who.upper()
            if who in ("A", "B", "OTHER"):
                clusters.group_team[g] = {"A": 0, "B": 1, "OTHER": -1}[who]; clusters.group_role[g] = -1
            elif who in ("REF", "GKA", "GKB"):
                clusters.group_team[g] = -1; clusters.group_role[g] = {"REF": 3, "GKA": 4, "GKB": 5}[who]; clusters.has_roles = True
    print(f"part {part}: {os.path.basename(det)}, video {os.path.basename(video)} ({W}x{H})")
    P = load_part(det, kf, cal, t0, t1, W, H, clusters, MIN_CONF)
    print("camera:", P["cam"].describe())
    print(f"{len(P['vt_m'])} moments, {len(P['fr'])} detections between {fmt(t0)} and {fmt(t1)}")
    if len(P["fr"]) == 0:
        sys.exit("No detections in that range (are --from/--to times in the part's own video?).")
    pid, pieces, nxt, prv, BT = build_pieces(P)

    tmp = os.path.join(out_dir, "_frames")
    shutil.rmtree(tmp, ignore_errors=True)
    print("cutting frames out of the video ...", flush=True)
    n = extract_frames(video, t0, t1, a.fps, tmp)
    FR = Frames(tmp, t0, a.fps)
    fw, fh = FR.size
    print(f"{n} frames of {fw}x{fh}")

    # for each piece: first and last detection
    first, last = {}, {}
    order = np.lexsort((np.arange(len(pid)), P["fr"]))
    for i in order:
        k = int(pid[i])
        first.setdefault(k, i)
        last[k] = i
    keep = [p for p in pieces if (P["vt_m"][p[3]] - P["vt_m"][p[2]]) >= a.min_piece or p[0] in nxt or p[0] in prv]
    keep_ids = {p[0] for p in keep}
    meta = []
    for p in keep:
        k = p[0]
        rec = {"id": k, "side": p[1], "s": round(float(P["vt_m"][p[2]]), 2), "e": round(float(P["vt_m"][p[3]]), 2)}
        for tag, idx in (("s", first[k]), ("e", last[k])):
            vt = float(P["vt_m"][P["fr"][idx]])
            u, v, hf = float(P["u"][idx]), float(P["v"][idx]), float(P["hf"][idx])
            hp = max(hf * fh, 14.0)
            cx, cy = u * fw, v * fh
            ch = 1.4 * hp
            FR.save(os.path.join(out_dir, "img", f"p{k}_{tag}.jpg"), FR.crop(vt, cx, cy - 0.5 * hp, ch * 0.75, ch, 150, 200))
            cw = max(8 * hp, 240.0)
            FR.save(os.path.join(out_dir, "img", f"p{k}_{tag}_c.jpg"), FR.crop(vt, cx, cy - 0.5 * hp, cw, cw * 0.75, 320, 240, mark=(cx, cy)))
            rec[tag + "u"], rec[tag + "v"] = round(u, 4), round(v, 4)
        meta.append(rec)

    # frames to pick a player from: one every 10 s with a mark on the pieces alive then
    picks = []
    vm = P["vt_m"]
    for k_, tt in enumerate(np.arange(t0, t1, a.pick_every)):
        mi = int(np.argmin(np.abs(vm - tt)))
        if abs(vm[mi] - tt) > 1.5:
            continue
        name = f"full_{k_}.jpg"
        FR.save(os.path.join(out_dir, "img", name), FR.at(vm[mi]))
        marks = [{"p": int(pid[i]), "u": round(float(P["u"][i]), 4), "v": round(float(P["v"][i]), 4), "t": int(P["team"][i])}
                 for i in np.flatnonzero(P["fr"] == mi) if int(pid[i]) in keep_ids]
        picks.append({"t": round(float(vm[mi]), 2), "img": f"img/{name}", "marks": marks})
    shutil.rmtree(tmp, ignore_errors=True)

    def cands(d):
        return {str(k): {"c": [c for c in v["c"] if c[0] in keep_ids], "s": v["s"]} for k, v in d.items() if k in keep_ids} if False else None

    import board_tracks as BT2
    def pack(d):
        out = {}
        for k, lst in d.items():
            if k not in keep_ids:
                continue
            c = [[b, p_, dd, gg] for b, p_, dd, gg in lst if b in keep_ids]
            if c:
                top = c[0]
                out[str(k)] = {"c": c, "s": 1 if (top[1] >= BT2.SURE and top[2] <= 3 + 2 * top[3]) else 0}
        return out

    data = {"format": FORMAT, "game": os.path.basename(os.path.abspath(folder)), "part": part, "from": t0, "to": t1,
            "tracker": BT2.TRACKER + " (detections, loose sides)", "camera": P["cam"].describe(),
            "pieces": meta, "next": pack(nxt), "picks": picks}
    with open(os.path.join(out_dir, "data.js"), "w") as fh_:
        fh_.write("const DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    with open(os.path.join(out_dir, "index.html"), "w") as fh_:
        fh_.write(HTML)
    print(f"\nwrote {out_dir}/index.html  ({len(meta)} pieces, {len(picks)} frames to pick from, "
          f"{sum(len(v['c']) > 0 for v in data['next'].values())} piece ends with candidates)")
    print("Open index.html in a browser, label a few players, press 'Download labels', move labels.json into that folder, then run with --score.")


# ── scoring ──────────────────────────────────────────────────────────────────

def score(path: str):
    d = json.load(open(path))
    qs = d["questions"]
    print(f"{d['game']} part {d['part']}  {fmt(d['from'])}–{fmt(d['to'])}   tracker: {d['tracker']}")
    print(f"{len(d['targets'])} players followed, {len(qs)} questions answered\n")
    def line(name, sel):
        if not sel:
            print(f"{name:34s} -"); return
        ok = sum(1 for q in sel if q["answer"] == q["cands"][0]["piece"])
        none = sum(1 for q in sel if q["answer"] in ("none", "unsure"))
        manual = sum(1 for q in sel if isinstance(q["answer"], dict))
        print(f"{name:34s} {len(sel):4d} questions: tracker's first choice right {100 * ok / len(sel):4.0f}%, "
              f"right one not offered {100 * manual / len(sel):3.0f}%, none/unsure {100 * none / len(sel):3.0f}%")
    line("all questions", qs)
    line("tracker said 'sure'", [q for q in qs if q["sure"]])
    line("tracker said 'unsure'", [q for q in qs if not q["sure"]])
    line("only one candidate", [q for q in qs if len(q["cands"]) == 1])
    line("two or more candidates", [q for q in qs if len(q["cands"]) >= 2])
    ranks = [next((i for i, c in enumerate(q["cands"]) if c["piece"] == q["answer"]), None) for q in qs if isinstance(q["answer"], int)]
    if ranks:
        print("\nrank of the right candidate: " + ", ".join(f"#{r + 1}: {sum(1 for x in ranks if x == r)}" for r in range(4) if any(x == r for x in ranks)))
    man = [q for q in qs if isinstance(q["answer"], dict)]
    if man:
        print("\nWHY WAS THE RIGHT PIECE NOT OFFERED? (the %d answers where you picked him yourself)" % len(man))
        for q in man:
            ch = next((c for c in d["targets"][q["target"]]["chain"] if c["how"] == "manual" and c["piece"] == q["answer"]["manual"]), None)
            if ch:
                gap = ch["from"] - q["t"]
                print(f"  player {q['target'] + 1}: lost at {fmt(q['t'])}, he reappears as piece {ch['piece']} {gap:+.1f} s later; "
                      f"{len(q['cands'])} candidates were offered" + ("  (a gap beyond the search window)" if gap > 20 else "  (overlaps in time with the old piece: a swap or a double)" if gap < -0.4 else ""))
    n_sure_wrong = sum(1 for q in qs if q["sure"] and isinstance(q["answer"], int) and q["answer"] != q["cands"][0]["piece"])
    print(f"\nlinks the tracker would make by itself ('sure') that are wrong: {n_sure_wrong} of {sum(1 for q in qs if q['sure'])}")
    print("How to read it: 'first choice right' is the share of joins a linker using that rule gets right where the right piece existed.\n"
          "'right one not offered' = the tracker had no piece for him (lost detection or too far): more detections, not a better rule, fix that.")


# ── self test ────────────────────────────────────────────────────────────────

def selftest():
    """Render markers with ffmpeg's v360 and check that Camera.project finds them."""
    import cv2
    import kf_core as K
    tmp = tempfile.mkdtemp()
    W, H = 1280, 720
    ok_all = True
    for projection in ("flat", "cylindrical"):
        # an equirect image with a white square at known lon/lat, rendered through the same v360 parameters the renderer uses
        eq = np.zeros((1440, 2880, 3), np.uint8)
        pts = [(-20.0, -12.0), (15.0, -8.0), (35.0, -20.0), (-5.0, -5.0)]
        for lon, lat in pts:
            x = int((lon + 180) / 360 * 2880); y = int((90 - lat) / 180 * 1440)
            cv2.rectangle(eq, (x - 9, y - 9), (x + 9, y + 9), (255, 255, 255), -1)
        cv2.imwrite(os.path.join(tmp, "eq.png"), eq)
        kf = os.path.join(tmp, "k.kf.json")
        json.dump([{"t": 0.0, "pan_deg": -5.0, "fov_deg": 70.0, "tilt_deg": -12.0}, {"t": 10.0, "pan_deg": -5.0, "fov_deg": 70.0, "tilt_deg": -12.0}], open(kf, "w"))
        json.dump({"projection": projection, "tilt_deg": -12.0}, open(os.path.join(tmp, "k.kf.meta.json"), "w"))
        cam = Camera(kf, None, W, H)
        view = cam.view_at(1.0)
        p = K.to_v360_params(-5.0, -12.0, 70.0, W, H, False, projection)
        vf = f"v360=equirect:{projection}:yaw={p['yaw']}:pitch={p['pitch']}:h_fov={p['h_fov']}:v_fov={p['v_fov']}:w={W}:h={H}"
        png = os.path.join(tmp, f"{projection}.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", os.path.join(tmp, "eq.png"), "-vf", vf, "-frames:v", "1", png], check=True)
        img = cv2.imread(png, 0)
        print(f"{projection}:")
        for lon, lat in pts:
            pos = cam.project(lon, lat, view)
            if pos is None or not (0 < pos[0] < 1 and 0 < pos[1] < 1):
                print(f"  ({lon:+.0f}, {lat:+.0f}) outside the picture"); continue
            px, py = int(pos[0] * W), int(pos[1] * H)
            win = img[max(0, py - 14): py + 15, max(0, px - 14): px + 15]
            ys, xs = np.nonzero(win > 128)
            if len(xs) == 0:
                print(f"  ({lon:+.0f}, {lat:+.0f}) predicted at ({px},{py}): NO marker there"); ok_all = False; continue
            fx, fy_ = xs.mean() + max(0, px - 14), ys.mean() + max(0, py - 14)
            err = math.hypot(fx - px, fy_ - py)
            print(f"  ({lon:+.0f}, {lat:+.0f}) predicted ({px},{py}), found ({fx:.0f},{fy_:.0f}), off by {err:.1f} px")
            ok_all &= err < 4
    shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST " + ("passed" if ok_all else "FAILED"))
    return 0 if ok_all else 1


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Label players</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#f4f6fb;color:#1a2340}
header{background:#0f2972;color:#fff;padding:8px 14px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
header b{font-size:16px}button{font:inherit;border:0;border-radius:99px;padding:5px 12px;cursor:pointer;background:#e8edf8;color:#0f2972;font-weight:600}
button.p{background:#e8780a;color:#fff}button.g{background:#2fa36b;color:#fff}button:disabled{opacity:.4;cursor:default}
main{padding:12px 14px;max-width:1400px;margin:auto}.card{background:#fff;border:1px solid #e4e6ee;border-radius:12px;padding:10px;margin-bottom:10px}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-start}.cand{width:340px}.cand img{display:block;border-radius:6px;margin-bottom:4px}
.pair{display:flex;gap:4px}.small{font-size:12px;color:#5b6377}.stage{position:relative;display:inline-block;line-height:0;max-width:100%}
.stage img{max-width:100%;display:block;border-radius:8px}.mark{position:absolute;width:26px;height:26px;margin:-13px;border-radius:50%;border:2px solid #fff;
box-shadow:0 0 0 1px #000;cursor:pointer;opacity:.9}.mark:hover{outline:3px solid #ffd400}
</style></head><body>
<header><b>Label players</b><span id="where"></span><span style="flex:1"></span>
<button id="new">+ New player</button><button id="undo">Undo</button><button class="g" id="dl">Download labels</button></header>
<main><div id="view"></div>
<div class="small" style="margin-top:8px">Your answers are kept in this browser while you work. 'Download labels' saves labels.json; put it next to index.html and run <code>python3 label_tracks.py --score labels.json</code>.</div></main>
<script src="data.js"></script>
<script>
const KEY='labels:'+DATA.game+':'+DATA.part+':'+DATA.from;
let S={targets:[],questions:[],cur:-1,mode:'idle',pickIdx:0,manualFor:null};
try{const v=localStorage.getItem(KEY);if(v)S=JSON.parse(v)}catch(e){}
const byId={};DATA.pieces.forEach(p=>byId[p.id]=p);
const $=id=>document.getElementById(id);
const fmt=s=>{s=Math.max(0,Math.round(s));return Math.floor(s/60)+':'+String(s%60).padStart(2,'0')};
const vt=s=>fmt(s);       // piece times are seconds in the part's video
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}};
const TEAMCOL=['#d62828','#1d4ed8','#9aa3b8','#ffd400'];
function target(){return S.targets[S.cur]}
function lastPiece(t){return t.chain[t.chain.length-1].piece}
function snapshot(){S.hist=S.hist||[];S.hist.push(JSON.stringify({targets:S.targets,questions:S.questions,cur:S.cur,mode:S.mode,pickIdx:S.pickIdx,manualFor:S.manualFor}));if(S.hist.length>60)S.hist.shift()}
function render(){
  save();$('where').textContent=DATA.game+' · part '+DATA.part+' · '+fmt(DATA.from)+'–'+fmt(DATA.to)+' · '+S.targets.length+' player(s), '+S.questions.length+' answers';
  const v=$('view');v.innerHTML='';
  if(S.mode==='pick'||S.mode==='manual'){return renderPick(v)}
  if(S.cur<0||!target()){v.innerHTML='<div class="card">Press <b>+ New player</b>, then click the player you want to follow on a frame of the video.</div>';return}
  const t=target();
  const tabs=document.createElement('div');tabs.className='card';
  tabs.innerHTML='<b>'+t.name+'</b> &nbsp; '+S.targets.map((x,i)=>'<button data-i="'+i+'" '+(i===S.cur?'class="p"':'')+'>'+x.name+(x.ended?' ✓':'')+'</button>').join(' ')+
    ' <span class="small"> followed through '+t.chain.length+' pieces, up to '+vt(byId[lastPiece(t)].e)+'</span>';
  v.appendChild(tabs);tabs.querySelectorAll('button').forEach(b=>b.onclick=()=>{S.cur=+b.dataset.i;render()});
  if(t.ended){v.insertAdjacentHTML('beforeend','<div class="card">This player is done ('+t.ended+'). Start another with <b>+ New player</b>.</div>');return}
  const p=byId[lastPiece(t)];
  if(p.e>=DATA.to-0.7){t.ended='reached the end of the stretch';render();return}
  const nx=DATA.next[String(p.id)];
  const last=document.createElement('div');last.className='card';
  last.innerHTML='<b>Last seen: '+vt(p.e)+'</b><div class="pair"><img src="img/p'+p.id+'_e.jpg"><img src="img/p'+p.id+'_e_c.jpg"></div><div class="small">The yellow ring marks his feet. Which of these is the same player afterwards?</div>';
  v.appendChild(last);
  const row=document.createElement('div');row.className='row';
  if(nx){nx.c.forEach((c,i)=>{const q=byId[c[0]];const d=document.createElement('div');d.className='card cand';
    d.innerHTML='<b>Candidate '+(i+1)+'</b> <span class="small">starts '+vt(q.s)+' ('+(c[3]<0.05?'right away':c[3].toFixed(1)+' s later')+', until '+vt(q.e)+')</span>'+
      '<div class="pair"><img src="img/p'+q.id+'_s.jpg"><img src="img/p'+q.id+'_s_c.jpg"></div><div class="pair"><img style="width:75px" src="img/p'+q.id+'_e.jpg"><span class="small">where he is at the end of this piece</span></div>'+
      '<button class="g">This is him</button>';
    d.querySelector('button').onclick=()=>answer(c[0]);row.appendChild(d)})}
  else{row.innerHTML='<div class="card">The tracker found no continuation for this piece.</div>'}
  v.appendChild(row);
  const act=document.createElement('div');act.className='card';
  act.innerHTML='<button id="manual">He is on the picture but not among these: pick him</button> <button id="lost">He left the picture / lost him</button> <button id="unsure">Cannot tell</button>';
  v.appendChild(act);
  $('manual').onclick=()=>{S.mode='manual';S.manualFor=p.id;S.pickIdx=nearestPick(p.e);render()};
  $('lost').onclick=()=>answer('none');$('unsure').onclick=()=>answer('unsure');
}
function nearestPick(t){let b=0,bd=1e9;DATA.picks.forEach((k,i)=>{const d=Math.abs(k.t-t);if(d<bd){bd=d;b=i}});return b}
function candList(pid){const nx=DATA.next[String(pid)];return nx?nx.c.map((c,i)=>({piece:c[0],rank:i,chance:c[1],dist:c[2],gap:c[3]})):[]}
function answer(a){snapshot();const t=target(),p=lastPiece(t);
  S.questions.push({target:S.cur,from:p,t:byId[p].e,cands:candList(p),sure:DATA.next[String(p)]?DATA.next[String(p)].s:0,answer:a});
  if(typeof a==='number')t.chain.push({piece:a,how:'answer'});else t.ended=a==='none'?'lost / left the picture':'could not tell';
  render()}
function renderPick(v){
  const k=DATA.picks[S.pickIdx];
  const hd=document.createElement('div');hd.className='card';
  hd.innerHTML=(S.mode==='manual'?'<b>Click him</b> on a frame, as late as you can find him after '+vt(byId[S.manualFor].e)+'. ':'<b>Click the player you want to follow.</b> ')+
    '<button id="pv">◀ earlier</button> <b>'+vt(k.t)+'</b> <button id="nx">later ▶</button> <button id="cn">Cancel</button>';
  v.appendChild(hd);
  const st=document.createElement('div');st.className='stage';st.innerHTML='<img src="'+k.img+'">';
  k.marks.forEach(m=>{const e=document.createElement('div');e.className='mark';e.style.left=(m.u*100)+'%';e.style.top=(m.v*100)+'%';e.style.background=TEAMCOL[m.t]||'#9aa3b8';
    e.title='click: this is him';e.onclick=()=>pickPiece(m.p);st.appendChild(e)});
  v.appendChild(st);
  $('pv').onclick=()=>{S.pickIdx=Math.max(0,S.pickIdx-1);render()};$('nx').onclick=()=>{S.pickIdx=Math.min(DATA.picks.length-1,S.pickIdx+1);render()};
  $('cn').onclick=()=>{S.mode='idle';render()};
}
function pickPiece(pid){snapshot();
  if(S.mode==='pick'){S.targets.push({name:'Player '+(S.targets.length+1),chain:[{piece:pid,how:'pick'}],ended:null});S.cur=S.targets.length-1}
  else{const t=target(),p=lastPiece(t);
    S.questions.push({target:S.cur,from:p,t:byId[p].e,cands:candList(p),sure:DATA.next[String(p)]?DATA.next[String(p)].s:0,answer:{manual:pid}});
    if(pid!==p)t.chain.push({piece:pid,how:'manual'})}
  S.mode='idle';render()}
$('new').onclick=()=>{S.mode='pick';S.pickIdx=0;render()};
$('undo').onclick=()=>{if(S.hist&&S.hist.length){const h=JSON.parse(S.hist.pop());Object.assign(S,h);render()}};
$('dl').onclick=()=>{const out={format:'football-autopan-labels',version:1,game:DATA.game,part:DATA.part,from:DATA.from,to:DATA.to,tracker:DATA.tracker,camera:DATA.camera,
  targets:S.targets.map(t=>({name:t.name,chain:t.chain.map(c=>({piece:c.piece,how:c.how,from:byId[c.piece].s,to:byId[c.piece].e})),ended:t.ended})),questions:S.questions};
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'}));a.download='labels.json';a.click()};
render();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?")
    ap.add_argument("--part", type=int, default=1)
    ap.add_argument("--from", dest="t_from", default="0:00")
    ap.add_argument("--to", dest="t_to", default="2:00")
    ap.add_argument("--video"); ap.add_argument("--out"); ap.add_argument("--labels"); ap.add_argument("--calib")
    ap.add_argument("--assign", nargs="+", metavar="GROUP=WHO")
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--min-piece", type=float, default=1.0, help="shortest piece (s) that gets pictures unless it has candidates")
    ap.add_argument("--pick-every", type=float, default=10.0, help="seconds between the frames you can pick a player from")
    ap.add_argument("--window", type=float, default=20.0, help="how many seconds after a piece ends a continuation may start")
    ap.add_argument("--reach", type=float, default=1.6, help="how far (times the tracker's own gate) a continuation may be")
    ap.add_argument("--top", type=int, default=6, help="most candidates shown per piece end")
    ap.add_argument("--score", metavar="LABELS_JSON")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.score:
        return score(a.score)
    if not a.folder:
        ap.error("give the game folder (or --score / --selftest)")
    global WINDOW, REACH, TOP
    WINDOW, REACH, TOP = a.window, a.reach, a.top
    make_site(a)


if __name__ == "__main__":
    main()
