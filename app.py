#!/usr/bin/env python3
"""app.py — the Football Autopan app for coaches. One page in the browser, everything runs on this computer.

    python app.py            (or double-click start.command on the Mac)

It opens http://127.0.0.1:8750 . Pick the video files of a game, press start, confirm which shirt colours are the
two teams on a picture, and collect the finished video and statistics from the game's folder
(default ~/FootballAutopan/<game>/ ; change with --folder).
Nothing leaves the computer. Only the standard library is needed for the app itself.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8750
STATE = {"root": os.path.expanduser("~/FootballAutopan"), "running": {}, "threads": {}}
LOCK = threading.Lock()


def jobs_root() -> str:
    os.makedirs(STATE["root"], exist_ok=True)
    return STATE["root"]


def job_dir(jid: str) -> str:
    if not jid or "/" in jid or "\\" in jid or jid.startswith("."):
        raise KeyError(jid)
    d = os.path.join(jobs_root(), jid)
    if not os.path.isfile(os.path.join(d, "job.json")):
        raise KeyError(jid)
    return d


def get_job(jid: str) -> P.Job:
    with LOCK:
        j = STATE["running"].get(jid)
    return j or P.Job(job_dir(jid))


def is_running(jid: str) -> bool:
    t = STATE["threads"].get(jid)
    return bool(t and t.is_alive())


def start_job(jid: str) -> str:
    with LOCK:
        busy = [k for k in STATE["threads"] if STATE["threads"][k].is_alive()]
        if busy:
            return f"another game is being processed ({busy[0]}); wait until it has finished or stop it first"
        job = P.Job(job_dir(jid))
        STATE["running"][jid] = job
        t = threading.Thread(target=job.run, daemon=True)
        STATE["threads"][jid] = t
        t.start()
    return ""


def stop_job(jid: str) -> bool:
    """Stop a game that is being processed (if it is) and wait until it has stopped."""
    j, t = STATE["running"].get(jid), STATE["threads"].get(jid)
    if j is not None and t is not None and t.is_alive():
        j.cancel()
        t.join(60)
        return not t.is_alive()
    return True


def list_dir(path: str) -> dict:
    path = os.path.abspath(os.path.expanduser(path or "~"))
    if not os.path.isdir(path):
        path = os.path.dirname(path) if os.path.isdir(os.path.dirname(path)) else os.path.expanduser("~")
    dirs, files = [], []
    try:
        names = sorted(os.listdir(path), key=str.lower)
    except OSError as e:
        return {"path": path, "parent": os.path.dirname(path), "dirs": [], "files": [], "error": str(e)}
    for n in names:
        if n.startswith("."):
            continue
        fp = os.path.join(path, n)
        try:
            if os.path.isdir(fp):
                dirs.append(n)
            elif n.lower().endswith(P.VIDEO_EXT):
                files.append({"name": n, "size": os.path.getsize(fp), "mtime": os.path.getmtime(fp)})
        except OSError:
            continue
    return {"path": path, "parent": os.path.dirname(path), "dirs": dirs, "files": files}


def probe(paths: list) -> dict:
    import kf_core as K
    out, durs = [], []
    for fp in paths:
        try:
            info = K.probe_video(fp)
            cam = P.camera_info(info, fp)
            out.append({"path": fp, "name": os.path.basename(fp), "duration": float(info["duration"]),
                        "width": info["width"], "height": info["height"], **cam})
            durs.append(float(info["duration"]))
        except Exception as e:
            out.append({"path": fp, "name": os.path.basename(fp), "error": str(e)[-200:]})
    return {"videos": out, "estimate_s": P.estimate_seconds(durs) if durs else 0}


def job_view(jid: str) -> dict:
    job = get_job(jid)
    st = dict(job.status)
    if st["state"] == "running" and not is_running(jid):
        st["state"] = "stopped"                                   # the app was closed while it was working
    st["can_continue"] = False
    if st["state"] != "running":
        old = {r["id"]: r for r in st.get("stages", [])}
        merged, missing = [], False
        for s in job.stages():                                    # steps added by a newer version of the app show up as waiting
            r = dict(old.get(s["id"]) or {"id": s["id"], "label": s["label"], "state": "pending", "progress": 0.0, "note": ""})
            r["label"] = s["label"]
            if job.done(s["id"]):
                r["state"], r["progress"] = "done", 1.0
            elif r.get("state") == "done" and s["id"] != "package":
                r["state"], r["progress"] = "pending", 0.0
            merged.append(r)
            if s["id"] != "package" and r["state"] in ("pending", "failed"):
                missing = True
        st["stages"] = merged
        if missing and st["state"] in ("done", "needs_input", "idle"):
            st["state"] = "incomplete" if st["state"] == "done" else st["state"]
        st["can_continue"] = missing
    meta = job.labels()
    teams = None
    if meta:
        import team_labels as TL
        teams = {"groups": [{"index": j, "size": int(meta["sizes"][j]),
                             "share": round(100.0 * meta["sizes"][j] / max(1, sum(meta["sizes"])), 1),
                             "bgr": [round(x) for x in meta["mean_bgr"][j]]} for j in range(len(meta["sizes"]))],
                 "assign": {"A": meta["assign"]["A"], "B": meta["assign"]["B"], "REF": meta["assign"].get("REF", []),
                            "GKA": meta["assign"].get("GKA", []), "GKB": meta["assign"].get("GKB", [])}, "suggest": meta.get("suggest", {"A": [], "B": []}),
                 "names": meta["names"], "colours": meta["colours"], "quality": TL.quality(meta),
                 "attack": job.spec.get("attack") or {}}
    job.migrate()
    vids = []
    for i, v in enumerate(job.videos()):
        cam = job.info()[i] if i < len(job.info()) else {}
        vids.append({"name": os.path.basename(v["path"]), "duration": cam.get("duration"), "half": v["half"],
                     "outline": v["outline"], "manual": os.path.isfile(job.manual_file(i)), "pitch": P.read_json(job.p(f"pitch{i}.info.json")),
                     "has_outline_picture": os.path.isfile(job.p(f"pitch{i}_outline.jpg")),
                     "outline_t": int(os.path.getmtime(job.p(f"pitch{i}_outline.jpg"))) if os.path.isfile(job.p(f"pitch{i}_outline.jpg")) else 0})
    return {"id": jid, "videos": vids, "title": job.spec.get("title"), "spec": job.spec, "status": st, "teams": teams,
            "cameras": job.info(), "has_preview": os.path.isfile(job.p("preview.jpg")),
            "files": job.results(), "folder": job.dir}


def list_jobs() -> list:
    out = []
    root = jobs_root()
    for n in sorted(os.listdir(root)):
        jp = os.path.join(root, n, "job.json")
        if os.path.isfile(jp):
            spec = P.read_json(jp, {})
            st = P.read_json(os.path.join(root, n, "status.json"), {}) or {}
            state = st.get("state", "idle")
            if state == "running" and not is_running(n):
                state = "stopped"
            out.append({"id": n, "title": spec.get("title", n), "state": state, "overall": st.get("overall", 0.0),
                        "created": spec.get("created", 0), "videos": len(spec.get("videos", []))})
    out.sort(key=lambda j: -j["created"])
    return out


def assign_teams(jid: str, body: dict) -> None:
    import team_labels as TL
    job = get_job(jid)
    path = job.p("teams.team_labels.json")
    if not os.path.isfile(path):
        raise ValueError("the shirt colours have not been found yet")

    class A:  # same fields as the command line
        labels = path
        a = [int(x) for x in body.get("a", [])]
        b = [int(x) for x in body.get("b", [])]
        ref = [int(x) for x in body.get("ref", [])]
        gk_a = [int(x) for x in body.get("gk_a", [])]
        gk_b = [int(x) for x in body.get("gk_b", [])]
        name_a = (body.get("name_a") or "").strip() or None
        name_b = (body.get("name_b") or "").strip() or None
        colour_a = body.get("colour_a") or None
        colour_b = body.get("colour_b") or None
    if not A.a or not A.b:
        raise ValueError("pick at least one colour group for each team")
    before = (P.read_json(path) or {})
    old_attack = dict(job.spec.get("attack") or {})
    new_attack = {h: v for h, v in (body.get("attack") or {}).items() if h in ("1", "2") and v in ("right", "left")}
    if "attack" in body:
        job.spec["attack"] = new_attack
        P.write_json(job.p("job.json"), job.spec)
    TL.assign(A)
    after = P.read_json(path) or {}
    if any(before.get(k) != after.get(k) for k in ("assign", "names", "colours")) or ("attack" in body and old_attack != new_attack):
        # the statistics and the board were made from the old labels: make them again (the video does not depend on them)
        job.clear("stats", "board", "package")


class Handler(BaseHTTPRequestHandler):
    server_version = "FootballAutopan"

    def log_message(self, *a):  # quiet
        pass

    # ── helpers ──
    def _ok_host(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost")

    def _send(self, code: int, body: bytes, ctype: str = "application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode())

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    # ── routes ──
    def do_GET(self):
        if not self._ok_host():
            return self._send(403, b"forbidden", "text/plain")
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                with open(os.path.join(HERE, "app.html"), "rb") as fh:
                    return self._send(200, fh.read(), "text/html; charset=utf-8")
            if u.path == "/api/ls":
                return self._json(list_dir(q.get("path", "")))
            if u.path == "/api/jobs":
                return self._json({"jobs": list_jobs(), "root": STATE["root"]})
            if u.path.startswith("/api/jobs/") and u.path.endswith("/outline"):
                job = get_job(u.path.split("/")[3])
                i = int(q.get("video", 0))
                job.ensure_frame(i)
                pts, src = job.outline_points(i)
                return self._json({"points": pts, "source": src, "frame": f"pitch{i}_frame.png", "size": [2880, 1440],
                                   "feet": job.feet(i) if os.path.isfile(job.p(f"v{i}.detections.jsonl")) else []})
            if u.path.startswith("/api/jobs/"):
                return self._json(job_view(u.path.split("/")[3]))
            if u.path.startswith("/files/"):
                return self._file(u.path)
            return self._send(404, b"not found", "text/plain")
        except KeyError:
            return self._send(404, b"not found", "text/plain")
        except Exception as e:
            return self._json({"error": str(e)}, 500)

    def _file(self, path: str):
        parts = [unquote(x) for x in path.split("/")[2:]]
        if len(parts) < 2 or any(p in ("", ".", "..") for p in parts) or any("/" in p or "\\" in p for p in parts[:1]):
            return self._send(404, b"not found", "text/plain")
        base = job_dir(parts[0])
        fp = os.path.realpath(os.path.join(base, *parts[1:]))
        if not fp.startswith(os.path.realpath(base) + os.sep) or not os.path.isfile(fp):
            return self._send(404, b"not found", "text/plain")
        size = os.path.getsize(fp)
        ctype = mimetypes.guess_type(fp)[0] or "application/octet-stream"
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else max(0, size - int(b))
            end = int(b) if (a and b) else size - 1
            code = 206
        end = min(end, size - 1)
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if self.path.endswith("?download=1") or "download=1" in (urlparse(self.path).query):
            self.send_header("Content-Disposition", f'attachment; filename="{os.path.basename(fp)}"')
        self.end_headers()
        try:
            with open(fp, "rb") as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = fh.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        if not self._ok_host() or self.headers.get("X-App") != "football":
            return self._send(403, b"forbidden", "text/plain")
        u = urlparse(self.path)
        try:
            body = self._body()
            if u.path == "/api/probe":
                return self._json(probe(body.get("paths", [])))
            if u.path == "/api/jobs":
                vids = [{"path": v} if isinstance(v, str) else v for v in body.get("videos", [])]
                n_v = len(vids)
                for i, v in enumerate(vids):                      # the coach's choice, else first half = first ceil(n/2) files
                    v["half"] = int(v.get("half") or (1 if i < (n_v + 1) // 2 else 2))
                    v.setdefault("outline", True)
                if not vids:
                    return self._json({"error": "choose at least one video"}, 400)
                bad = [v["path"] for v in vids if not os.path.isfile(v["path"])]
                if bad:
                    return self._json({"error": f"file not found: {bad[0]}"}, 400)
                job = P.Job.create(jobs_root(), {"title": (body.get("title") or "").strip() or "Game", "videos": vids,
                                                 "lang": body.get("lang", "en"), "layout": 2})
                err = start_job(os.path.basename(job.dir))
                return self._json({"id": os.path.basename(job.dir), "error": err})
            parts = u.path.split("/")
            if len(parts) == 5 and parts[1] == "api" and parts[2] == "jobs":
                jid, action = parts[3], parts[4]
                job_dir(jid)
                if action == "teams":
                    assign_teams(jid, body)
                    err = "" if is_running(jid) else start_job(jid)             # a job that was waiting goes on
                    return self._json({"ok": True, "error": err})
                if action == "outline":
                    if not stop_job(jid):
                        return self._json({"error": "could not stop the running job - try Stop first"}, 400)
                    job = P.Job(job_dir(jid))
                    i = int(body.get("video", -1))
                    if body.get("reset"):
                        job.clear_manual_outline(i)
                    else:
                        job.set_manual_outline(i, body.get("points", []))
                    return self._json({"ok": True, "error": start_job(jid)})
                if action == "setup":
                    if not stop_job(jid):
                        return self._json({"error": "could not stop the running job - try Stop first"}, 400)
                    job = P.Job(job_dir(jid))
                    gone = job.reconfigure(body.get("videos", []), [int(x) for x in body.get("redo", [])])
                    err = start_job(jid)
                    return self._json({"ok": True, "error": err, "redo": gone})
                if action == "resume":
                    return self._json({"error": start_job(jid)})
                if action == "cancel":
                    j = STATE["running"].get(jid)
                    if j:
                        j.cancel()
                    return self._json({"ok": True})
                if action == "open":
                    d = job_dir(jid)
                    cmd = ["open", d] if sys.platform == "darwin" else ["explorer", d] if os.name == "nt" else ["xdg-open", d]
                    subprocess.Popen(cmd)
                    return self._json({"ok": True})
            return self._send(404, b"not found", "text/plain")
        except KeyError:
            return self._send(404, b"not found", "text/plain")
        except (ValueError, SystemExit) as e:
            return self._json({"error": str(e)}, 400)
        except Exception as e:
            return self._json({"error": str(e)}, 500)


def make_server(port: int = PORT) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default=STATE["root"], help="where the games' results are kept")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    STATE["root"] = os.path.abspath(os.path.expanduser(a.folder))
    os.makedirs(STATE["root"], exist_ok=True)
    for n in os.listdir(STATE["root"]):                  # steps left running by an earlier run of the app
        if os.path.isfile(os.path.join(STATE["root"], n, "job.json")):
            if P.Job(os.path.join(STATE["root"], n)).reap():
                print(f"stopped a step of '{n}' that an earlier run had left running")
    srv = make_server(a.port)
    url = f"http://127.0.0.1:{a.port}"
    print(f"Football Autopan is running at {url}   (results in {STATE['root']})   Ctrl+C to stop")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    import signal

    def bye(*_):
        raise KeyboardInterrupt
    for sg in (signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM)):
        signal.signal(sg, bye)                           # closing the terminal window stops the steps too
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for jid, j in list(STATE["running"].items()):    # do not leave steps running after the app is gone
            j.cancel()
        for t in list(STATE["threads"].values()):
            t.join(15)
    return 0


if __name__ == "__main__":
    sys.exit(main())
