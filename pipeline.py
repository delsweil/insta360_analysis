#!/usr/bin/env python3
"""pipeline.py — one game, start to finish, for people who do not use the command line.

A *job* is a folder holding job.json (what the coach asked for) and status.json (what has happened). The stages:

  check      look at the video files (size, length, which camera layout) and save a picture of the unwrapped view
  teams      find the colour groups of the shirts in ~40 moments spread over the game (the coach confirms them)
  extract    find the players, 3 times a second           (the long step: ~0.75 s per second of video)
  keyframes  turn the detections into the camera path
  charts     where the play happened, game statistics
  render     the panning video                            (~0.8 s per second of video)
  stats      team charts (needs the confirmed teams; if they are missing the job waits and carries on after)
  board      positions of all players in the time of the joined video, for the tactical board of the web player
  package    list of results (results.json) and a short README

Every finished stage leaves a marker in <job>/.done, so a job that stopped (closed lid, error, cancelled) continues
where it was. Use it from app.py, or on its own:  python pipeline.py --job ~/FootballAutopan/mygame run
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".insv")
KNOWN_RAW_SIZES = {(2560, 1440), (1440, 2560)}          # Insta360 X2 single-lens recording; tested
DETECT_TILT = -22.0
CROP_TILT = -20.0
EXTRACT_ARGS = ["--lon-range", "-80", "80", "--no-ball", "--iv-fov", "112", "--tilt", str(DETECT_TILT), "--colours"]
KEY_ARGS = ["--projection", "flat", "--fov-max", "100", "--fov-min", "70", "--tilt", str(CROP_TILT),
            "--max-dip", "8", "--tilt-margin", "4"]
# seconds of computing per second of video (Apple silicon), used for the progress bar and the estimate
COST = {"check": 0.0, "extract": 0.75, "keyframes": 0.0, "render": 0.8, "charts": 0.0, "stats": 0.0, "board": 0.0, "teams": 0.0}
FIXED_SECONDS = {"check": 15, "pitch": 20, "merge": 60, "teams": 120, "keyframes": 10, "charts": 30, "stats": 40, "board": 15, "package": 2}


VERIFY_RENDER = True        # look into every finished video before it replaces the old one


def verify_video(path: str, positions: int = 6, seconds: float = 3.0) -> List[str]:
    """Play a few seconds at several places of a finished video; -> the problems found (empty = looks fine)."""
    import kf_core as K
    try:
        dur = float(K.probe_video(path)["duration"])
    except Exception as e:
        return [f"cannot be opened ({str(e)[-100:]})"]
    problems = []
    for k in range(positions):
        t = dur * (k + 0.5) / positions
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-t", f"{seconds:g}", "-i", path, "-f", "null", "-"],
                           capture_output=True, text=True)
        err = (r.stderr or "").strip()
        if r.returncode != 0 or err:
            problems.append(f"at {t / 60:.1f} min: {err.splitlines()[0][:100] if err else 'decoding failed'}")
    return problems


def safe_name(s: str) -> str:
    s = re.sub(r"[^\w\-. ]+", "", s, flags=re.UNICODE).strip().replace(" ", "_")
    return s or "game"


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def read_json(path: str, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def camera_info(info: dict, path: str) -> dict:
    """What we know about a recording. Only the X2 single-lens layout has been tested; anything else is tried
    anyway and the coach is asked to look at the picture of the unwrapped view."""
    w, h = int(info["width"]), int(info["height"])
    import kf_core as K
    if K.is_equirect_size(w, h):
        return {"layout": "equirect", "tested": True, "note": "ready-made 360 picture"}
    if (w, h) in KNOWN_RAW_SIZES:
        return {"layout": "raw", "tested": True, "note": "Insta360 single-lens recording (X2 layout)"}
    return {"layout": "raw", "tested": False,
            "note": f"{w}x{h}: not tested yet - look at the picture of the unwrapped view before you trust the result"}


def estimate_seconds(durations: List[float]) -> float:
    """Computing time for a whole game, from the lengths of its video files."""
    tot = sum(durations)
    return tot * (COST["extract"] + COST["render"]) + sum(FIXED_SECONDS.values()) + 5 * len(durations)


class Cancelled(Exception):
    pass


class Job:
    def __init__(self, folder: str):
        self.dir = os.path.abspath(folder)
        self.spec = read_json(os.path.join(self.dir, "job.json"), {})
        self.cancel_flag = False
        self.proc: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        self.status = read_json(os.path.join(self.dir, "status.json"), None) or {
            "state": "idle", "message": "", "stages": [], "overall": 0.0, "remaining_s": None, "updated": time.time()}

    # ── paths and files ──────────────────────────────────────────────────────
    def p(self, *parts: str) -> str:
        return os.path.join(self.dir, *parts)

    @staticmethod
    def create(root: str, spec: dict) -> "Job":
        base = safe_name(spec.get("title") or "game")
        folder, n = os.path.join(root, base), 2
        while os.path.exists(folder):
            folder, n = os.path.join(root, f"{base}_{n}"), n + 1
        os.makedirs(os.path.join(folder, ".done"))
        spec = dict(spec)
        spec.setdefault("lang", "en")
        spec["created"] = time.time()
        write_json(os.path.join(folder, "job.json"), spec)
        return Job(folder)

    def save_status(self) -> None:
        self.status["updated"] = time.time()
        write_json(self.p("status.json"), self.status)

    def done(self, sid: str) -> bool:
        return os.path.exists(self.p(".done", sid))

    def mark_done(self, sid: str) -> None:
        open(self.p(".done", sid), "w").close()

    def log(self, line: str) -> None:
        with open(self.p("job.log"), "a") as fh:
            fh.write(line.rstrip("\n") + "\n")

    # ── the plan ─────────────────────────────────────────────────────────────
    def videos(self) -> List[dict]:
        out = []
        for v in self.spec["videos"]:
            out.append({"path": v["path"], "start": float(v.get("start") or 0.0), "end": v.get("end"),
                        "half": int(v.get("half") or 1), "outline": bool(v.get("outline", True))})
        return out

    def migrate(self) -> None:
        """Jobs made before halves existed: first half = the first ceil(n/2) videos; their charts are made again."""
        if self.spec.get("layout") == 2:
            return
        n = len(self.spec["videos"])
        for i, v in enumerate(self.spec["videos"]):
            v.setdefault("half", 1 if i < (n + 1) // 2 else 2)
            v.setdefault("outline", True)
        self.spec["layout"] = 2
        self.spec.pop("halves", None)
        write_json(self.p("job.json"), self.spec)
        self.clear("charts", "stats", "board")

    def clear(self, *ids: str) -> List[str]:
        """Forget that these steps are done (they run again). -> the ones that really were done."""
        gone = []
        for sid in ids:
            try:
                os.remove(self.p(".done", sid))
                gone.append(sid)
            except OSError:
                pass
        return gone

    def halves(self) -> List[tuple]:
        """[(1, [video indices]), (2, [...])] for the halves that have videos."""
        vs = self.videos()
        return [(h, [i for i, v in enumerate(vs) if v["half"] == h]) for h in (1, 2) if any(v["half"] == h for v in vs)]

    def reconfigure(self, cfg: List[dict], redo: List[int]) -> List[str]:
        """The coach changed which half a video belongs to, or whether the pitch outline is used, or wants the
        camera of a video planned again: whatever depends on it is made again, the rest is kept."""
        old = self.videos()
        gone: List[str] = []
        if len(cfg) != len(old):
            raise ValueError("wrong number of videos")
        for i, c in enumerate(cfg):
            half = int(c.get("half", old[i]["half"]))
            if half not in (1, 2):
                raise ValueError("half must be 1 or 2")
            outline = bool(c.get("outline", old[i]["outline"]))
            if half != old[i]["half"]:
                gone += self.clear("charts", "stats", "board")
            if outline != old[i]["outline"] or i in redo:
                gone += self.clear(f"pitch{i}", f"keyframes{i}", f"render{i}", "merge", "charts", "stats", "board")
            self.spec["videos"][i]["half"], self.spec["videos"][i]["outline"] = half, outline
        write_json(self.p("job.json"), self.spec)
        return list(dict.fromkeys(gone))

    def det_file(self, i: int) -> str:
        """The detections the camera and the statistics are made from: without people beside the pitch if the outline was used."""
        info = read_json(self.p(f"pitch{i}.info.json"), {})
        f = self.p(f"v{i}.onpitch.detections.jsonl")
        return f if info.get("used") and os.path.isfile(f) else self.p(f"v{i}.detections.jsonl")

    def info(self) -> List[dict]:
        return read_json(self.p("camera.json"), [])

    def span(self, i: int) -> tuple:
        """(start, duration) in seconds that is analysed for video i."""
        v = self.videos()[i]
        info = self.info()
        total = float(info[i]["duration"]) if i < len(info) else 0.0
        end = float(v["end"]) if v.get("end") else total
        return v["start"], max(1.0, end - v["start"])

    def stages(self) -> List[dict]:
        n = len(self.spec["videos"])
        st = [{"id": "check", "label": "Look at the videos"}, {"id": "teams", "label": "Find the shirt colours"}]
        for i in range(n):
            st.append({"id": f"extract{i}", "label": f"Find the players{self._of(i, n)}", "video": i})
        for i in range(n):
            st.append({"id": f"pitch{i}", "label": f"Find the pitch lines{self._of(i, n)}", "video": i})
        for i in range(n):
            st.append({"id": f"keyframes{i}", "label": f"Plan the camera{self._of(i, n)}", "video": i})
        st.append({"id": "charts", "label": "Where the play was"})
        for i in range(n):
            st.append({"id": f"render{i}", "label": f"Make the video{self._of(i, n)}", "video": i})
        if n > 1:
            st.append({"id": "merge", "label": "Join the videos for YouTube"})
        st.append({"id": "stats", "label": "Team statistics"})
        st.append({"id": "board", "label": "Board data for the web player"})
        st.append({"id": "package", "label": "Collect the results"})
        return st

    @staticmethod
    def _of(i: int, n: int) -> str:
        return f" ({i + 1}/{n})" if n > 1 else ""

    def weight(self, st: dict) -> float:
        sid = st["id"]
        kind = re.sub(r"\d+$", "", sid)
        if "video" in st:
            try:
                dur = self.span(st["video"])[1]
            except (IndexError, KeyError):
                dur = 600.0
            return max(1.0, COST.get(kind, 0.0) * dur + FIXED_SECONDS.get(kind, 0))
        return float(FIXED_SECONDS.get(kind, 5))

    # ── leftovers of an earlier run ──
    def reap(self) -> bool:
        """Stop a step that an earlier run of the app left running (the app was closed while it worked). Two runs
        writing the same file would spoil it. -> True if something was stopped."""
        info = read_json(self.p(".proc"))
        stopped = False
        if info and os.name == "posix":
            pid = int(info.get("pid", 0))
            try:
                os.kill(pid, 0)
                cmdline = subprocess.run(["ps", "-ww", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout
                if self.dir in cmdline:                              # really ours, not a reused process number
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGTERM)
                    except OSError:
                        os.kill(pid, signal.SIGTERM)
                    for _ in range(50):
                        time.sleep(0.1)
                        try:
                            os.kill(pid, 0)
                        except OSError:
                            break
                    else:
                        os.kill(pid, signal.SIGKILL)
                    stopped = True
            except (OSError, ValueError):
                pass
        try:
            os.remove(self.p(".proc"))
        except OSError:
            pass
        return stopped

    # ── running ──────────────────────────────────────────────────────────────
    def cancel(self) -> None:
        self.cancel_flag = True
        p = self.proc
        if p and p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except (OSError, AttributeError):
                p.terminate()

    def run(self) -> str:
        """Run every stage that is not done yet. Returns the final state."""
        self.cancel_flag = False
        if self.reap():
            self.log("stopped a step that an earlier run had left running")
        self.migrate()
        stages = self.stages()
        known = {s["id"]: s for s in self.status.get("stages", [])}
        self.status["stages"] = []
        for s in stages:
            old = known.get(s["id"], {})
            state = "done" if self.done(s["id"]) and s["id"] != "package" else "pending"
            self.status["stages"].append({"id": s["id"], "label": s["label"], "state": state,
                                          "progress": 1.0 if state == "done" else 0.0, "note": "",
                                          "weight": self.weight(s), "took": old.get("took", 0.0) if state == "done" else 0.0})
        self.status.update(state="running", message="")
        self.save_status()
        fns: Dict[str, Callable] = {"check": self.s_check, "teams": self.s_teams, "charts": self.s_charts,
                                    "stats": self.s_stats, "board": self.s_board, "package": self.s_package}
        waiting_stats = False
        try:
            for s in stages:
                rec = self._rec(s["id"])
                if rec["state"] == "done":
                    continue
                if self.cancel_flag:
                    raise Cancelled()
                kind = re.sub(r"\d+$", "", s["id"])
                fn = fns.get(s["id"]) or getattr(self, "s_" + kind)
                rec["state"] = "running"
                self.save_status()
                t0 = time.time()
                try:
                    res = fn(s) if kind in ("extract", "pitch", "keyframes", "render") else fn()
                except Cancelled:
                    raise
                except Exception as e:                                              # the stage failed
                    rec.update(state="failed", note=str(e)[-400:])
                    self.status.update(state="failed", message=f"{s['label']}: {str(e)[-300:]}")
                    self.log(f"FAILED {s['id']}: {e}")
                    self.save_status()
                    return "failed"
                if res == "waiting":
                    rec.update(state="waiting", note="waiting for you to confirm the teams")
                    waiting_stats = True
                    self.save_status()
                    continue
                rec.update(state="done", progress=1.0, took=time.time() - t0)
                if s["id"] != "package":                      # the list of results is made again every time
                    self.mark_done(s["id"])
                self._overall()
                self.save_status()
            # the team stage may have been waiting while the long steps ran: try once more
            if waiting_stats and self.labels_ready():
                for sid, fn in (("stats", self.s_stats), ("board", self.s_board)):
                    rec = self._rec(sid)
                    if rec["state"] == "done":
                        continue
                    rec["state"] = "running"
                    self.save_status()
                    fn()
                    rec.update(state="done", progress=1.0, note="")
                    self.mark_done(sid)
                self.s_package()
                waiting_stats = False
        except Cancelled:
            self.status.update(state="cancelled", message="stopped")
            for r in self.status["stages"]:
                if r["state"] == "running":
                    r["state"] = "pending"
            self.save_status()
            return "cancelled"
        except Exception as e:
            self.status.update(state="failed", message=str(e)[-300:])
            self.save_status()
            return "failed"
        state = "needs_input" if waiting_stats else "done"
        self.status.update(state=state, message="", overall=1.0 if state == "done" else self.status.get("overall", 0.0),
                           remaining_s=0 if state == "done" else None)
        self.save_status()
        return state

    def _rec(self, sid: str) -> dict:
        return next(r for r in self.status["stages"] if r["id"] == sid)

    def _overall(self, live: float = 0.0, live_id: Optional[str] = None) -> None:
        recs = self.status["stages"]
        tot = sum(r["weight"] for r in recs) or 1.0
        done = sum(r["weight"] * (1.0 if r["state"] == "done" else (r["progress"] if r["id"] == live_id else 0.0)) for r in recs)
        self.status["overall"] = min(1.0, done / tot)
        fin = [r for r in recs if r["state"] == "done" and r.get("took", 0) > 0 and r["weight"] > 20]
        ratio = (sum(r["took"] for r in fin) / sum(r["weight"] for r in fin)) if fin else 1.0
        ratio = min(max(ratio, 0.3), 4.0)
        self.status["remaining_s"] = max(0.0, (tot - done) * ratio)

    def set_progress(self, sid: str, frac: float, note: str = "") -> None:
        try:
            rec = self._rec(sid)
        except StopIteration:                                  # a step run on its own, not through run()
            return
        rec["progress"] = max(0.0, min(0.999, frac))
        if note:
            rec["note"] = note
        self._overall(rec["progress"], sid)
        self.save_status()

    def exec(self, cmd: List[str], sid: str, parse: Optional[Callable[[str], Optional[float]]] = None) -> None:
        """Run a command, stream its output into job.log, turn progress lines into a fraction. Raises on failure."""
        self.log("$ " + " ".join(cmd))
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        kw = {"start_new_session": True} if os.name == "posix" else {}
        self.proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, bufsize=0, **kw)
        write_json(self.p(".proc"), {"pid": self.proc.pid, "step": sid})
        tail: List[str] = []
        buf, last_save = b"", 0.0
        fd = self.proc.stdout.fileno()
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            buf += chunk
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for raw in parts:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                f = parse(line) if parse else None
                if f is None:
                    self.log(line)
                    tail = (tail + [line])[-12:]
                elif time.time() - last_save > 1.0:
                    last_save = time.time()
                    self.set_progress(sid, f, line[:90])
        rc = self.proc.wait()
        self.proc = None
        try:
            os.remove(self.p(".proc"))
        except OSError:
            pass
        if self.cancel_flag:
            raise Cancelled()
        if rc != 0:
            raise RuntimeError(" | ".join(tail[-4:]) or f"exit code {rc}")

    def py(self, script: str, *args: str) -> List[str]:
        return [sys.executable, "-u", os.path.join(HERE, script), *args]

    # ── stages ───────────────────────────────────────────────────────────────
    def s_check(self) -> None:
        import kf_core as K
        infos = []
        for v in self.videos():
            if not os.path.isfile(v["path"]):
                raise RuntimeError(f"file not found: {v['path']}")
            if not v["path"].lower().endswith(VIDEO_EXT):
                raise RuntimeError(f"not a video file: {os.path.basename(v['path'])}")
            info = K.probe_video(v["path"])
            infos.append({"path": v["path"], "width": info["width"], "height": info["height"], "fps": info["fps"],
                          "duration": float(info["duration"]), **camera_info(info, v["path"])})
        write_json(self.p("camera.json"), infos)
        # a picture of the unwrapped view in the middle of the first video: the coach checks that the pitch looks right
        import cv2
        import scene_check as S
        first = infos[0]
        raw = first["layout"] == "raw"
        s0, d0 = self.span(0)
        fr = S.grab_frame(first["path"], s0 + d0 * 0.4, 112.0 if raw else None, raw)
        if fr is not None:
            small = cv2.resize(fr, (1440, 720), interpolation=cv2.INTER_AREA)
            cv2.imwrite(self.p("preview.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 85])

    def s_teams(self) -> None:
        """Colour groups of the shirts from ~40 moments spread over the first video, for the coach to confirm."""
        info = self.info()[0]
        s0, d0 = self.span(0)
        raw = info["layout"] == "raw"
        step = max(1.0, d0 * 0.9 / 60.0)
        t = s0 + d0 * 0.05
        frames = []
        while t < s0 + d0 * 0.95:
            frames.append({"type": "frame", "t": round(t, 2), "players": []})
            t += step
        header = {"type": "header", "video": os.path.basename(info["path"]), "view_lons": [-60, -20, 20, 60],
                  "tilt": DETECT_TILT, "view_fov": 40.0, "raw_input": raw, "iv_fov": 112.0, "square_views": True}
        samp = self.p("sample.detections.jsonl")
        with open(samp, "w") as fh:
            fh.write(json.dumps(header) + "\n")
            for fr in frames:
                fh.write(json.dumps(fr) + "\n")
        pat = re.compile(r"frame (\d+)/(\d+)")

        def parse(line):
            m = pat.search(line)
            return int(m.group(1)) / int(m.group(2)) if m else None
        self.exec(self.py("team_labels.py", "make", "--video", info["path"], "--detections", samp,
                          "--out", self.p("teams"), "--frames", "40", "--iv-fov", "112"), "teams", parse)

    def _frames_re(self):
        pat = re.compile(r"(\d+)/(\d+) frames")
        return lambda line: (int(pat.search(line).group(1)) / max(1, int(pat.search(line).group(2)))) if pat.search(line) else None

    def s_extract(self, st: dict) -> None:
        i = st["video"]
        s0, d0 = self.span(i)
        self.exec(self.py("extract_detections.py", "--video", self.videos()[i]["path"], *EXTRACT_ARGS,
                          "--start", f"{s0:.2f}", "--duration", f"{d0:.2f}",
                          "--out", self.p(f"v{i}.detections.jsonl")), st["id"], self._frames_re())

    def s_keyframes(self, st: dict) -> None:
        i = st["video"]
        self.exec(self.py("auto_keyframes.py", "--detections", self.det_file(i), *KEY_ARGS,
                          "--out", self.p(f"v{i}.auto.keyframes.json")), st["id"])

    # ── the pitch outline: automatic, or drawn by the coach ──────────────────
    def manual_file(self, i: int) -> str:
        return self.p(f"pitch{i}.manual.json")

    def ensure_frame(self, i: int) -> str:
        """The unwrapped picture (2880x1440) the outline is found in / drawn on."""
        png = self.p(f"pitch{i}_frame.png")
        if not os.path.isfile(png):
            import cv2
            import scene_check as S
            cam = self.info()[i]
            raw = cam["layout"] == "raw"
            s0, d0 = self.span(i)
            fr = S.grab_frame(cam["path"], s0 + d0 * 0.5, 112.0 if raw else None, raw)
            if fr is None:
                raise RuntimeError("no picture")
            cv2.imwrite(png, fr)
        return png

    def outline_points(self, i: int) -> tuple:
        """(points, source) for the editor: the coach's own outline, else the automatic one made simple, else a rough start."""
        import numpy as np
        import extract_detections as X
        for src, f in (("manual", self.manual_file(i)), ("auto", self.p(f"pitch{i}.json"))):
            if os.path.isfile(f):
                try:
                    poly = np.asarray(X.load_polygon(f), np.float32).reshape(-1, 2)
                except Exception:
                    continue
                if src == "auto":
                    import cv2
                    eps = 4.0
                    pts = poly
                    while eps < 200:
                        pts = cv2.approxPolyDP(poly.reshape(-1, 1, 2), eps, True).reshape(-1, 2)
                        if len(pts) <= 24:
                            break
                        eps *= 1.5
                    poly = pts
                return [[round(float(x)), round(float(y))] for x, y in poly], src
        return [[500, 870], [1440, 800], [2380, 870], [2700, 1000], [1440, 1130], [180, 1000]], "default"

    def draw_outline(self, i: int, pts) -> None:
        """The picture shown in the app for a hand-drawn outline: the line on the unwrapped view, and the players
        seen (green = inside, red = outside)."""
        import cv2
        import numpy as np
        fr = cv2.imread(self.ensure_frame(i))
        if fr is None:
            return
        poly = np.array(pts, np.int32).reshape(-1, 1, 2)
        if os.path.isfile(self.p(f"v{i}.detections.jsonl")):
            inside = np.zeros(fr.shape[:2], np.uint8)
            cv2.fillPoly(inside, [poly], 255)
            for x, y in self.feet(i, 1500):
                ok = 0 <= y < inside.shape[0] and 0 <= x < inside.shape[1] and inside[y, x] > 0
                cv2.circle(fr, (x, y), 7, (80, 200, 60) if ok else (60, 60, 230), -1)
        cv2.polylines(fr, [poly], True, (60, 220, 240), 6)
        for p in poly.reshape(-1, 2):
            cv2.circle(fr, (int(p[0]), int(p[1])), 12, (255, 255, 255), -1)
        cv2.putText(fr, "drawn by you", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (60, 220, 240), 5)
        cv2.imwrite(self.p(f"pitch{i}_outline.jpg"), cv2.resize(fr, (1440, 720), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

    def set_manual_outline(self, i: int, pts: List[List[float]]) -> None:
        if not 0 <= i < len(self.spec["videos"]):
            raise ValueError("no such video")
        pts = [[float(x), float(y)] for x, y in pts]
        if len(pts) < 3 or any(not (0 <= x <= 2880 and 0 <= y <= 1440) for x, y in pts):
            raise ValueError("the outline needs at least 3 points inside the picture")
        write_json(self.manual_file(i), {"pixel_polygon": pts, "manual": True})
        try:
            self.draw_outline(i, pts)                          # the new line is visible at once
        except Exception as e:
            self.log(f"outline picture not made: {e}")
        self.spec["videos"][i]["outline"] = True
        write_json(self.p("job.json"), self.spec)
        self.clear(f"pitch{i}", f"keyframes{i}", f"render{i}", "merge", "charts", "stats", "board")

    def clear_manual_outline(self, i: int) -> None:
        for f in (self.manual_file(i), self.p(f"pitch{i}_outline.jpg")):     # the picture comes back with the automatic outline
            try:
                os.remove(f)
            except OSError:
                pass
        self.clear(f"pitch{i}", f"keyframes{i}", f"render{i}", "merge", "charts", "stats", "board")

    def feet(self, i: int, max_pts: int = 3000) -> List[List[int]]:
        """Where the feet of the detected players are in the picture (a sample), to see what an outline keeps."""
        import kf_core as K
        import extract_detections as X
        with open(self.p(f"v{i}.detections.jsonl")) as fh:
            fh.readline()
            rows = [json.loads(l) for l in fh if l.strip()]
        step = max(1, len(rows) // 400)
        pts = []
        for r in rows[::step]:
            for p in r["players"]:
                if len(p) > 3 and p[3] >= 0.5:
                    x, y = K.lonlat_to_xy(p[0], p[1], X.EQ_W, X.EQ_H)
                    pts.append([int(round(float(x))), int(round(float(y)))])
        return pts[:: max(1, len(pts) // max_pts)]

    def s_pitch(self, st: dict) -> None:
        """Find the pitch outline in one picture of this video and leave out players standing outside it (coaches,
        substitutes, spectators): they pull the camera sideways. If the outline looks wrong the detections are used as they are."""
        i = st["video"]
        info = {"used": False, "reason": "", "removed_pct": 0.0}
        try:
            if not self.videos()[i]["outline"]:
                info["reason"] = "switched off"
                return
            import extract_detections as X
            import pitch_filter as PF
            manual = os.path.isfile(self.manual_file(i))
            info["manual"] = manual
            if manual:                                          # drawn by the coach: trusted
                calib = self.manual_file(i)
                try:
                    self.draw_outline(i, read_json(calib)["pixel_polygon"])
                except Exception as e:
                    self.log(f"outline picture not made: {e}")
            else:
                png = self.ensure_frame(i)
                calib = self.p(f"pitch{i}.json")
                self.exec(self.py("calibrate_auto.py", "--input", png, "--output", calib,
                                  "--vis", self.p(f"pitch{i}_outline.jpg"), "--quiet"), st["id"])
            share = float((X.polygon_mask(X.load_polygon(calib)) > 0).mean())
            if not (0.02 if manual else 0.05) <= share <= 0.95:
                info["reason"] = f"outline implausible ({100 * share:.0f} % of the picture)"
                return
            out = self.p(f"v{i}.onpitch.detections.jsonl")
            n_all, n_off = PF.apply(self.p(f"v{i}.detections.jsonl"), out, calib, 25.0)
            info["removed_pct"] = round(100.0 * n_off / max(1, n_all), 1)
            if n_all == 0 or (n_off / n_all > 0.5 and not manual):
                info["reason"] = f"outline would remove {info['removed_pct']} % of all players"
                os.remove(out)
                return
            info["used"] = True
        except Cancelled:
            raise
        except Exception as e:                                 # never stop the game for this: carry on without the outline
            info["reason"] = f"no outline ({str(e)[-120:]})"
        finally:
            write_json(self.p(f"pitch{i}.info.json"), info)
            self.log(f"pitch outline video {i + 1}: {info}")

    def tag(self, h: int) -> str:
        return f"{safe_name(self.spec.get('title') or 'game')}_half{h}"

    def s_charts(self) -> None:
        title, lang = self.spec.get("title", "Game"), ("de" if self.spec.get("lang") == "de" else "en")
        for h, idx in self.halves():
            dets = [self.det_file(i) for i in idx]
            kfs = [self.p(f"v{i}.auto.keyframes.json") for i in idx]
            t = f"{title} - {'2. Halbzeit' if h == 2 else '1. Halbzeit'}" if lang == "de" else f"{title} - {'second' if h == 2 else 'first'} half"
            self.exec(self.py("play_map.py", "--detections", *dets, "--keyframes", *kfs,
                              "--out", self.p(f"{self.tag(h)}_play.png"), "--title", t, "--lang", lang), "charts")
            self.exec(self.py("game_stats.py", "--detections", *dets, "--keyframes", *kfs,
                              "--out", self.p(f"{self.tag(h)}_stats"), "--title", t, "--lang", lang), "charts")

    def stem(self, i: int) -> str:
        n = len(self.spec["videos"])
        base = safe_name(self.spec.get("title") or "game")
        return base + (f"_part{i + 1}" if n > 1 else "")

    def encoder(self) -> List[str]:
        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        except OSError:
            out = ""
        if sys.platform == "darwin" and "h264_videotoolbox" in out:
            return ["--encoder", "h264_videotoolbox", "--bitrate", "10000k"]
        return ["--encoder", "libx264", "--crf", "20"]

    def s_render(self, st: dict) -> None:
        i = st["video"]
        s0, d0 = self.span(i)
        pat = re.compile(r"time=(\d+):(\d+):([\d.]+)")

        def parse(line):
            m = pat.search(line)
            return (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))) / d0 if m else None
        out = self.p(f"{self.stem(i)}_autopan.mp4")
        self.exec(self.py("render_keyframes.py", "--input", self.videos()[i]["path"], "--keyframes",
                          self.p(f"v{i}.auto.keyframes.json"), "--output", out + ".part.mp4", "--audio", *self.encoder()),
                  st["id"], parse)
        if VERIFY_RENDER:
            problems = verify_video(out + ".part.mp4")
            if problems:
                os.remove(out + ".part.mp4")
                raise RuntimeError(f"The new video is damaged ({problems[0]}). Make it again with 'Plan camera again'.")
        os.replace(out + ".part.mp4", out)

    def s_merge(self) -> None:
        """One video of the whole game for YouTube: the finished videos in playing order, joined without re-encoding
        (they all come out of the same settings), plus a text file with the start of each part for YouTube chapters."""
        order = [i for _, idx in self.halves() for i in idx]
        files = [self.p(f"{self.stem(i)}_autopan.mp4") for i in order]
        for f in files:
            if not os.path.isfile(f):
                raise RuntimeError(f"missing video: {os.path.basename(f)}")
        base = safe_name(self.spec.get("title") or "game")
        out = self.p(f"{base}_full_game.mp4")
        lst = self.p(".merge.txt")
        with open(lst, "w") as fh:
            for f in files:
                fh.write("file '" + f.replace("'", "'\\''") + "'\n")
        total = sum(self.span(i)[1] for i in order)
        pat = re.compile(r"time=(\d+):(\d+):([\d.]+)")

        def parse(line):
            m = pat.search(line)
            return (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))) / max(1.0, total) if m else None
        self.exec(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-stats", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                   "-c", "copy", "-movflags", "+faststart", out + ".part.mp4"], "merge", parse)
        if VERIFY_RENDER:
            problems = verify_video(out + ".part.mp4")
            if problems:
                os.remove(out + ".part.mp4")
                raise RuntimeError(f"The joined video is damaged ({problems[0]}).")
        os.replace(out + ".part.mp4", out)
        try:                                                    # chapters: where each part starts (YouTube wants 0:00 first)
            import kf_core as K
            t, lines = 0.0, []
            for n, (i, f) in enumerate(zip(order, files)):
                h = self.videos()[i]["half"]
                label = f"{'Second' if h == 2 else 'First'} half" + (f" (part {n + 1})" if len(files) > 2 else "")
                lines.append(f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{int(t % 60):02d} {label}" if t >= 3600 else
                             f"{int(t // 60)}:{int(t % 60):02d} {label}")
                t += float(K.probe_video(f)["duration"])
            with open(self.p(f"{base}_chapters.txt"), "w") as fh:
                fh.write("\n".join(lines) + "\n")
        except Exception as e:
            self.log(f"chapters not written: {e}")

    def labels(self) -> Optional[dict]:
        return read_json(self.p("teams.team_labels.json"))

    def labels_ready(self) -> bool:
        m = self.labels()
        return bool(m and m["assign"]["A"] and m["assign"]["B"])

    def s_stats(self):
        if not self.labels_ready():
            return "waiting"
        lang = self.spec.get("lang", "en")
        title = self.spec.get("title", "Game")
        for h, idx in self.halves():
            dets = [self.det_file(i) for i in idx]
            kfs = [self.p(f"v{i}.auto.keyframes.json") for i in idx]
            t = f"{title} - {'2. Halbzeit' if h == 2 else '1. Halbzeit'}" if lang == "de" else f"{title} - {'second' if h == 2 else 'first'} half"
            self.exec(self.py("team_stats.py", "--detections", *dets, "--keyframes", *kfs,
                              "--team-labels", self.p("teams.team_labels.json"), "--title", t,
                              "--lang", "de" if lang == "de" else "en", "--out", self.p(f"{self.tag(h)}_teams"),
                              "--data-json", self.p(f"{self.tag(h)}_chartdata.json")), "stats")
        return None

    def s_board(self):
        """<game>_board.json: every player position (3 a second) in the time of the joined video that goes to YouTube,
        for the tactical board of the web player. Needs the confirmed teams, like the statistics."""
        if not self.labels_ready():
            return "waiting"
        import kf_core as K
        dets, kfs, halves, durs, charts, calibs = [], [], [], [], [], []
        for h, idx in self.halves():
            for i in idx:
                f = self.p(f"{self.stem(i)}_autopan.mp4")
                try:
                    dur = float(K.probe_video(f)["duration"])
                except Exception:
                    dur = self.span(i)[1]
                info = read_json(self.p(f"pitch{i}.info.json"), {})
                raw = self.p(f"v{i}.detections.jsonl")          # the board has its own outline test (metres, not degrees)
                dets.append(raw if info.get("used") else self.det_file(i))
                calibs.append((self.manual_file(i) if info.get("manual") else self.p(f"pitch{i}.json")) if info.get("used") else "-")
                kfs.append(self.p(f"v{i}.auto.keyframes.json"))
                halves.append(h)
                durs.append(dur)
        for h, _ in self.halves():
            cd = self.p(f"{self.tag(h)}_chartdata.json")
            if os.path.isfile(cd):
                charts.append(cd)
        lang = "de" if self.spec.get("lang") == "de" else "en"
        base = safe_name(self.spec.get("title") or "game")
        cmd = self.py("board_data.py", "--detections", *dets, "--keyframes", *kfs, "--halves", *map(str, halves),
                      "--durations", *[f"{d:.2f}" for d in durs], "--team-labels", self.p("teams.team_labels.json"),
                      "--title", self.spec.get("title", "Game"), "--lang", lang, "--out", self.p(f"{base}_board.json"))
        cmd += ["--calib", *calibs]
        if charts:
            cmd += ["--chart-data", *charts]
        self.exec(cmd, "board")
        return None

    def s_package(self) -> None:
        files = self.results()
        write_json(self.p("results.json"), {"title": self.spec.get("title"), "files": files})
        with open(self.p("README.txt"), "w") as fh:
            fh.write(f"{self.spec.get('title', 'Game')}\n\n"
                     "*_autopan.mp4          the video with the camera following the play (one per file)\n"
                     "*_full_game.mp4        all parts joined in playing order: upload this one to YouTube\n"
                     "*_chapters.txt         where each part starts, to paste into the YouTube description\n"
                     "*_halfN_play.png / _stats.*  where the play was, game statistics (N = 1 first half, 2 second half)\n"
                     "*_halfN_teams.png / .csv     the two teams: heat map, territory, spread, tempo (CSV = numbers per 5 minutes)\n"
                     "*_halfN_chartdata.json       the numbers the charts of that half are drawn from (heat map grid, curves)\n"
                     "*_board.json           player positions in the time of the joined video: upload it in the web player (tactical board)\n"
                     "v*.detections.jsonl     every player found, 3 times a second (shirt colours included)\n"
                     "v*.auto.keyframes.json  the camera path (pan, zoom, tilt over time)\n"
                     "teams.team_labels.json  which shirt colours you called team A and B\n")

    def results(self) -> List[dict]:
        keep = (".mp4", ".png", ".csv", ".json", ".jsonl", ".txt", ".jpg")
        skip = {"job.json", "status.json", "camera.json", "sample.detections.jsonl", "results.json", "teams.team_labels.json"}
        out = []
        for name in sorted(os.listdir(self.dir)):
            fp = self.p(name)
            if not os.path.isfile(fp) or name in skip or name.startswith(".") or not name.endswith(keep) or ".part" in name:
                continue
            if name in ("preview.jpg", "job.log") or name.startswith("pitch") or ".onpitch." in name:
                continue
            out.append({"name": name, "size": os.path.getsize(fp)})
        return out


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job", required=True, help="job folder")
    p.add_argument("cmd", choices=["run", "status"])
    a = p.parse_args(argv)
    job = Job(a.job)
    if a.cmd == "status":
        print(json.dumps(job.status, indent=1))
        return 0
    print(job.run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
