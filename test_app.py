"""Tests for the coach app: job runner (pipeline.py) and local server (app.py). The heavy programs are replaced by
a fake that writes the files they would write, so this runs without footage or a detector."""
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

import numpy as np

import app as APP
import pipeline as P


def make_video(path, w=2560, h=1440, secs=3):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={w}x{h}:rate=10:duration={secs}",
                    "-pix_fmt", "yuv420p", path], check=True)


def _det_frames():
    """players on the pitch (lon 0) and beside it (lon 80), one frame"""
    import kf_core as K
    return json.dumps({"t": 0.0, "players": [[0.0, -10.0, 40.0, 0.9, None, [200, 200, 200]],
                                             [80.0, -10.0, 40.0, 0.9, None, [200, 200, 200]]], "balls": []}) + "\n"


DET_FRAMES = _det_frames()


def arg(cmd, name):
    return cmd[cmd.index(name) + 1]


def fake_exec(self, cmd, sid, parse=None):
    """Writes what the real script would write; feeds one progress line through the parser."""
    script = os.path.basename(cmd[2])
    if parse:
        parse("  10/20 frames  (0.2s each, ~1.0 min left)")
    if script == "team_labels.py":
        out = arg(cmd, "--out")
        os.makedirs(out + ".groups", exist_ok=True)
        for j in range(3):
            open(f"{out}.groups/g{j}.png", "wb").write(b"png")
        json.dump({"centres": [[0, 0, 0], [30, 0, 0], [60, 0, 0]], "sizes": [50, 40, 10],
                   "mean_bgr": [[250, 250, 250], [200, 80, 20], [40, 40, 40]], "suggest": {"A": [0], "B": [1]},
                   "assign": {"A": [], "B": []}, "names": {"A": "team A", "B": "team B"},
                   "colours": {"A": "#ececec", "B": "#d9392b"}}, open(out + ".team_labels.json", "w"))
    elif script == "calibrate_auto.py":
        json.dump({"pixel_polygon": [[400, 600], [1800, 600], [1800, 1100], [400, 1100]]}, open(arg(cmd, "--output"), "w"))
        open(arg(cmd, "--vis"), "wb").write(b"jpg")
    elif script == "extract_detections.py":
        open(arg(cmd, "--out"), "w").write(json.dumps({"type": "header"}) + "\n" + DET_FRAMES)
    elif script == "auto_keyframes.py":
        json.dump({"keyframes": []}, open(arg(cmd, "--out"), "w"))
    elif script == "render_keyframes.py":
        open(arg(cmd, "--output"), "wb").write(b"mp4")
    elif script == "play_map.py":
        open(arg(cmd, "--out"), "wb").write(b"png")
    elif script in ("game_stats.py", "team_stats.py"):
        open(arg(cmd, "--out") + ".png", "wb").write(b"png")
        if script == "team_stats.py":
            open(arg(cmd, "--out") + ".csv", "w").write("minute\n")
            if "--data-json" in cmd:
                open(arg(cmd, "--data-json"), "w").write("{}")
            self.team_cmds = getattr(self, "team_cmds", []) + [cmd]
            self.last_team_cmd = cmd
    else:
        raise AssertionError(script)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.vid = os.path.join(self.tmp, "half1.mp4")
        make_video(self.vid)
        self.root = os.path.join(self.tmp, "out")
        os.makedirs(self.root)
        self._exec = P.Job.exec
        P.Job.exec = fake_exec

    def tearDown(self):
        P.Job.exec = self._exec
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestPipeline(Base):
    def job(self, n=1, **kw):
        return P.Job.create(self.root, {"title": "Altenberg Cup", "videos": [{"path": self.vid}] * n, **kw})

    def test_stage_plan_and_estimate(self):
        j = self.job(2)
        ids = [s["id"] for s in j.stages()]
        self.assertEqual(ids[:2], ["check", "teams"])
        self.assertIn("render1", ids)
        self.assertIn("pitch1", ids)
        self.assertEqual(ids[-2:], ["stats", "package"])
        # two hours of video: roughly 3.2 hours of computing
        self.assertAlmostEqual(P.estimate_seconds([3600, 3600]) / 3600, 3.2, delta=0.2)

    def test_camera_info(self):
        j = self.job()
        j.s_check()
        cam = j.info()[0]
        self.assertEqual(cam["layout"], "raw")
        self.assertTrue(cam["tested"])
        self.assertTrue(os.path.isfile(j.p("preview.jpg")))
        info = {"width": 3840, "height": 2160, "fps": 30, "duration": 10}
        self.assertFalse(P.camera_info(info, "x.mp4")["tested"])
        self.assertEqual(P.camera_info({"width": 3840, "height": 1920}, "x.mp4")["layout"], "equirect")

    def test_missing_and_wrong_files_fail_clearly(self):
        j = P.Job.create(self.root, {"title": "bad", "videos": [{"path": os.path.join(self.tmp, "nope.mp4")}]})
        self.assertEqual(j.run(), "failed")
        self.assertIn("file not found", j.status["message"])
        txt = os.path.join(self.tmp, "a.txt")
        open(txt, "w").write("x")
        j2 = P.Job.create(self.root, {"title": "bad2", "videos": [{"path": txt}]})
        self.assertEqual(j2.run(), "failed")
        self.assertIn("not a video file", j2.status["message"])

    def test_run_waits_for_teams_then_finishes(self):
        j = self.job(2, lang="de")
        self.assertEqual(j.run(), "needs_input")
        self.assertEqual(j._rec("stats")["state"], "waiting")
        self.assertTrue(os.path.isfile(j.p("Altenberg_Cup_part2_autopan.mp4")))
        self.assertTrue(os.path.isfile(j.p("teams.groups", "g0.png")))
        # the coach confirms the suggestion -> only the missing stages run
        m = json.load(open(j.p("teams.team_labels.json")))
        m["assign"] = {"A": [0], "B": [1]}
        json.dump(m, open(j.p("teams.team_labels.json"), "w"))
        j2 = P.Job(j.dir)
        self.assertEqual(j2.run(), "done")
        cmd = j2.last_team_cmd
        self.assertEqual(arg(cmd, "--lang"), "de")
        self.assertNotIn("--switch-at", cmd)                    # each half is analysed on its own
        self.assertEqual(len(j2.team_cmds), 2)
        for c in j2.team_cmds:
            self.assertEqual(len(c[c.index("--detections") + 1:c.index("--keyframes")]), 1)
        names = [f["name"] for f in j2.results()]
        for h in (1, 2):
            self.assertIn(f"Altenberg_Cup_half{h}_teams.png", names)
            self.assertIn(f"Altenberg_Cup_half{h}_teams.csv", names)
            self.assertIn(f"Altenberg_Cup_half{h}_chartdata.json", names)
        self.assertNotIn("pitch0_outline.jpg", names)
        self.assertEqual(j2.status["overall"], 1.0)
        self.assertTrue(os.path.isfile(j2.p("README.txt")))
        self.assertNotIn(".part.mp4", " ".join(names))

    def test_finished_stages_are_not_repeated(self):
        j = self.job()
        j.run()
        calls = []
        P.Job.exec = lambda self, cmd, sid, parse=None: calls.append(sid)
        P.Job(j.dir).run()
        self.assertEqual([c for c in calls if c.startswith(("extract", "render"))], [])

    def test_failure_in_a_stage_stops_with_message(self):
        def boom(self, cmd, sid, parse=None):
            if "extract_detections.py" in cmd[2]:
                raise RuntimeError("CUDA out of memory")
            return fake_exec(self, cmd, sid, parse)
        P.Job.exec = boom
        j = self.job()
        self.assertEqual(j.run(), "failed")
        self.assertIn("CUDA out of memory", j.status["message"])
        self.assertFalse(j.done("extract0"))
        P.Job.exec = fake_exec
        self.assertEqual(P.Job(j.dir).run(), "needs_input")       # carries on from the failure

    def test_trim_range_is_used(self):
        j = P.Job.create(self.root, {"title": "t", "videos": [{"path": self.vid, "start": 1.0, "end": 2.5}]})
        j.s_check()
        s, d = j.span(0)
        self.assertEqual((s, round(d, 1)), (1.0, 1.5))


class TestHalvesAndOutline(Base):
    def job(self, n, cfg=None):
        cfg = cfg or {}
        return P.Job.create(self.root, {"title": "Alt", "videos": [{"path": self.vid, **cfg.get(i, {})} for i in range(n)],
                                        "layout": 2})

    def test_default_halves(self):
        for n, want in ((1, [1]), (2, [1, 2]), (3, [1, 1, 2]), (4, [1, 1, 2, 2])):
            j = P.Job.create(self.root, {"title": f"g{n}", "videos": [{"path": self.vid}] * n})
            j.migrate()
            self.assertEqual([v["half"] for v in j.videos()], want)

    def test_one_video_for_the_second_half(self):
        """the camera stopped: two files for the first half, one for the second"""
        j = self.job(3, {0: {"half": 1}, 1: {"half": 1}, 2: {"half": 2}})
        self.assertEqual(j.halves(), [(1, [0, 1]), (2, [2])])
        self.assertEqual(j.run(), "needs_input")
        m = json.load(open(j.p("teams.team_labels.json")))
        m["assign"] = {"A": [0], "B": [1]}
        json.dump(m, open(j.p("teams.team_labels.json"), "w"))
        j = P.Job(j.dir)
        self.assertEqual(j.run(), "done")
        h1 = [c for c in j.team_cmds if arg(c, "--out").endswith("half1_teams")][0]
        self.assertEqual(len(h1[h1.index("--detections") + 1:h1.index("--keyframes")]), 2)
        h2 = [c for c in j.team_cmds if arg(c, "--out").endswith("half2_teams")][0]
        self.assertEqual(len(h2[h2.index("--detections") + 1:h2.index("--keyframes")]), 1)

    def test_only_second_half(self):
        j = self.job(1, {0: {"half": 2}})
        self.assertEqual(j.halves(), [(2, [0])])

    def test_outline_removes_people_beside_the_pitch(self):
        j = self.job(1)
        j.run()
        info = json.load(open(j.p("pitch0.info.json")))
        self.assertTrue(info["used"], info)
        self.assertAlmostEqual(info["removed_pct"], 50.0, delta=1)         # one of two players stood beside the pitch
        # the camera is planned from the file without them
        self.assertEqual(j.det_file(0), j.p("v0.onpitch.detections.jsonl"))
        rows = [json.loads(l) for l in open(j.det_file(0))][1:]
        self.assertEqual([p[4] for p in rows[0]["players"]], [True, False])

    def test_bad_outline_is_ignored(self):
        def small(self, cmd, sid, parse=None):
            if "calibrate_auto.py" in cmd[2]:
                json.dump({"pixel_polygon": [[10, 10], [60, 10], [60, 40], [10, 40]]}, open(arg(cmd, "--output"), "w"))
                open(arg(cmd, "--vis"), "wb").write(b"jpg")
                return
            return fake_exec(self, cmd, sid, parse)
        P.Job.exec = small
        j = self.job(1)
        j.run()
        info = json.load(open(j.p("pitch0.info.json")))
        self.assertFalse(info["used"])
        self.assertIn("implausible", info["reason"])
        self.assertEqual(j.det_file(0), j.p("v0.detections.jsonl"))

    def test_manual_outline_overrides_automatic_one(self):
        j = self.job(1)
        j.run()
        self.assertTrue(j.done("keyframes0"))
        calls = []

        def spy(self, cmd, sid, parse=None):
            calls.append(os.path.basename(cmd[2]))
            return fake_exec(self, cmd, sid, parse)
        P.Job.exec = spy
        j = P.Job(j.dir)
        with self.assertRaises(ValueError):
            j.set_manual_outline(0, [[0, 0], [10, 10]])                 # needs three points
        with self.assertRaises(ValueError):
            j.set_manual_outline(0, [[0, 0], [9999, 10], [10, 10]])    # outside the picture
        j.set_manual_outline(0, [[400, 600], [1800, 600], [1800, 1100], [400, 1100]])
        self.assertFalse(j.done("pitch0") or j.done("keyframes0") or j.done("render0") or j.done("charts"))
        j.run()
        self.assertNotIn("calibrate_auto.py", calls)                    # the coach's line is used as drawn
        info = P.read_json(j.p("pitch0.info.json"))
        self.assertTrue(info["used"] and info["manual"], info)
        pts, src = j.outline_points(0)
        self.assertEqual(src, "manual")
        self.assertEqual(len(pts), 4)
        j.clear_manual_outline(0)
        self.assertEqual(j.outline_points(0)[1], "auto")
        self.assertFalse(j.done("pitch0"))

    def test_editor_data(self):
        j = self.job(1)
        j.run()
        pts, src = j.outline_points(0)
        self.assertEqual(src, "auto")
        self.assertGreaterEqual(len(pts), 3)
        feet = j.feet(0)
        self.assertEqual(sorted(f[0] for f in feet)[0] < sorted(f[0] for f in feet)[-1], True)   # on and beside the pitch
        self.assertTrue(os.path.isfile(j.p("pitch0_frame.png")))

    def test_failing_calibration_does_not_stop_the_game(self):
        def boom(self, cmd, sid, parse=None):
            if "calibrate_auto.py" in cmd[2]:
                raise RuntimeError("no pitch found")
            return fake_exec(self, cmd, sid, parse)
        P.Job.exec = boom
        j = self.job(1)
        self.assertEqual(j.run(), "needs_input")
        self.assertFalse(json.load(open(j.p("pitch0.info.json")))["used"])

    def test_reconfigure_redoes_only_what_depends_on_it(self):
        j = self.job(2, {0: {"half": 1}, 1: {"half": 2}})
        j.run()
        for sid in ("charts", "keyframes0", "keyframes1", "render1", "pitch1"):
            self.assertTrue(j.done(sid), sid)
        j.reconfigure([{"half": 1, "outline": True}, {"half": 1, "outline": True}], [])     # only the half changed
        self.assertFalse(j.done("charts"))
        self.assertTrue(j.done("keyframes1") and j.done("render1"))
        j.run()
        j.reconfigure([{"half": 1, "outline": True}, {"half": 1, "outline": False}], [])    # outline off for video 2
        self.assertFalse(j.done("keyframes1") or j.done("render1") or j.done("pitch1"))
        self.assertTrue(j.done("keyframes0") and j.done("render0"))
        j.run()
        self.assertFalse(P.read_json(j.p("pitch1.info.json"))["used"])
        j.reconfigure([{"half": 1, "outline": True}, {"half": 1, "outline": False}], [0])   # plan camera of video 1 again
        self.assertFalse(j.done("keyframes0"))
        with self.assertRaises(ValueError):
            j.reconfigure([{"half": 3}, {"half": 1}], [])

    def test_old_jobs_are_migrated(self):
        j = P.Job.create(self.root, {"title": "old", "videos": [{"path": self.vid}] * 3, "halves": True})
        os.makedirs(j.p(".done"), exist_ok=True)
        j.mark_done("charts")
        j.mark_done("stats")
        j.mark_done("keyframes2")
        j.migrate()
        self.assertEqual([v["half"] for v in j.videos()], [1, 1, 2])
        self.assertFalse(j.done("charts") or j.done("stats"))
        self.assertTrue(j.done("keyframes2"))


class TestQuality(unittest.TestCase):
    def test_suggest_and_quality(self):
        import team_labels as TL
        rng = np.random.default_rng(0)
        centres = np.array([[40, 5, 5], [25, 40, 10], [38, 8, 4], [20, 35, 12], [60, 0, 0]], float)
        sizes = np.array([40, 30, 15, 10, 5])
        s = TL.suggest(centres, sizes)
        self.assertEqual(s["A"][0], 0)
        self.assertIn(1, s["B"])
        self.assertIn(2, s["A"])                                  # near A's colour
        self.assertIn(3, s["B"])
        meta = {"sizes": sizes.tolist(), "assign": {"A": [0, 2], "B": [1, 3]}}
        q = TL.quality(meta)
        self.assertAlmostEqual(q["a"], 0.55)
        self.assertEqual(q["warnings"], [])
        meta["assign"] = {"A": [0, 2], "B": [4]}
        self.assertIn("unbalanced", TL.quality(meta)["warnings"])
        meta["assign"] = {"A": [], "B": []}
        self.assertIn("not_assigned", TL.quality(meta)["warnings"])


class TestServer(Base):
    def setUp(self):
        super().setUp()
        APP.STATE.update(root=self.root, running={}, threads={})
        self.srv = APP.make_server(0)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        for t in list(APP.STATE["threads"].values()):
            t.join(10)
        super().tearDown()

    def call(self, path, body=None, headers=None, raw=False):
        h = {"X-App": "football", "Content-Type": "application/json", **(headers or {})}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                     data=None if body is None else json.dumps(body).encode(), headers=h)
        try:
            with urllib.request.urlopen(req) as r:
                data = r.read()
                return (r.status, data) if raw else (r.status, json.loads(data))
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def until(self, jid, cond, timeout=30):
        t0 = time.time()
        while time.time() - t0 < timeout:
            code, v = self.call(f"/api/jobs/{jid}")
            if cond(v):
                return v
            time.sleep(0.2)
        self.fail("condition not reached")

    def wait(self, jid, states, timeout=30):
        t0 = time.time()
        while time.time() - t0 < timeout:
            code, v = self.call(f"/api/jobs/{jid}")
            if v["status"]["state"] in states:
                return v
            time.sleep(0.2)
        self.fail(f"job did not reach {states}: {v['status']}")

    def test_page_and_browse_and_probe(self):
        code, html = self.call("/", raw=True)
        self.assertEqual(code, 200)
        self.assertIn(b"Football Autopan", html)
        code, ls = self.call("/api/ls?path=" + self.tmp)
        self.assertEqual([f["name"] for f in ls["files"]], ["half1.mp4"])
        code, p = self.call("/api/probe", {"paths": [self.vid, "/no/such.mp4"]})
        self.assertEqual(p["videos"][0]["layout"], "raw")
        self.assertIn("error", p["videos"][1])
        self.assertGreater(p["estimate_s"], 0)

    def test_full_game_through_the_api(self):
        code, r = self.call("/api/jobs", {"title": "Heim gegen Gast", "videos": [self.vid], "lang": "de"})
        self.assertEqual(code, 200, r)
        jid = r["id"]
        v = self.wait(jid, ("needs_input", "failed"))
        self.assertEqual(v["status"]["state"], "needs_input", v["status"])
        self.assertEqual(v["teams"]["suggest"], {"A": [0], "B": [1]})
        self.assertTrue(v["has_preview"])
        code, png = self.call(f"/files/{jid}/teams.groups/g0.png", raw=True)
        self.assertEqual((code, png), (200, b"png"))
        # a second game while the first one waits is allowed (the first has finished its work)
        code, r = self.call(f"/api/jobs/{jid}/teams", {"a": [0], "b": [1], "name_a": "FCA", "name_b": "Gast", "colour_a": "white", "colour_b": "#3d7bd9"})
        self.assertEqual(code, 200, r)
        v = self.wait(jid, ("done", "failed"))
        self.assertEqual(v["status"]["state"], "done", v["status"])
        self.assertEqual(v["teams"]["names"], {"A": "FCA", "B": "Gast"})
        self.assertEqual(v["teams"]["colours"]["A"], "#ececec")
        names = [f["name"] for f in v["files"]]
        self.assertIn("Heim_gegen_Gast_autopan.mp4", names)
        self.assertIn("Heim_gegen_Gast_half1_teams.png", names)
        self.assertNotIn("Heim_gegen_Gast_half2_teams.png", names)
        code, jobs = self.call("/api/jobs")
        self.assertEqual(jobs["jobs"][0]["state"], "done")

    def test_setup_endpoint_changes_halves(self):
        code, r = self.call("/api/jobs", {"title": "s2", "videos": [{"path": self.vid, "half": 1}, {"path": self.vid, "half": 1}]})
        jid = r["id"]
        v = self.wait(jid, ("needs_input",))
        self.assertEqual([x["half"] for x in v["videos"]], [1, 1])
        self.assertTrue(v["videos"][0]["has_outline_picture"])
        code, r = self.call(f"/api/jobs/{jid}/setup", {"videos": [{"half": 1, "outline": True}, {"half": 2, "outline": True}], "redo": [1]})
        self.assertEqual(code, 200, r)
        v = self.wait(jid, ("needs_input", "failed"))
        self.assertEqual([x["half"] for x in v["videos"]], [1, 2])
        code, r = self.call(f"/api/jobs/{jid}/setup", {"videos": [{"half": 9}, {"half": 1}]})
        self.assertEqual(code, 400)

    def test_outline_endpoints(self):
        code, r = self.call("/api/jobs", {"title": "o", "videos": [self.vid]})
        jid = r["id"]
        self.wait(jid, ("needs_input",))
        code, d = self.call(f"/api/jobs/{jid}/outline?video=0")
        self.assertEqual(code, 200, d)
        self.assertEqual(d["source"], "auto")
        self.assertTrue(d["feet"])
        code, r = self.call(f"/api/jobs/{jid}/outline", {"video": 0, "points": [[400, 600], [1800, 600], [1800, 1100]]})
        self.assertEqual(code, 200, r)
        v = self.until(jid, lambda v: v["videos"][0]["pitch"] and v["videos"][0]["pitch"].get("manual") and v["status"]["state"] != "running")
        self.assertTrue(v["videos"][0]["manual"])
        code, r = self.call(f"/api/jobs/{jid}/outline", {"video": 0, "points": [[1, 1]]})
        self.assertEqual(code, 400)
        code, r = self.call(f"/api/jobs/{jid}/outline", {"video": 0, "reset": True})
        v = self.until(jid, lambda v: v["videos"][0]["pitch"] and not v["videos"][0]["pitch"].get("manual") and v["status"]["state"] != "running")
        self.assertFalse(v["videos"][0]["manual"])

    def test_outline_saved_while_the_game_is_running(self):
        """changing the outline of one video while another step is working stops the job and starts it again"""
        gate = threading.Event()

        def slow(self, cmd, sid, parse=None):
            if "render_keyframes.py" in cmd[2] and not gate.is_set():
                while not self.cancel_flag:
                    time.sleep(0.05)
                raise P.Cancelled()
            return fake_exec(self, cmd, sid, parse)
        P.Job.exec = slow
        code, r = self.call("/api/jobs", {"title": "run", "videos": [self.vid, self.vid]})
        jid = r["id"]
        self.until(jid, lambda v: v["status"]["state"] == "running" and any(s["id"].startswith("render") and s["state"] == "running" for s in v["status"]["stages"]))
        gate.set()
        code, r = self.call(f"/api/jobs/{jid}/outline", {"video": 1, "points": [[400, 600], [1800, 600], [1800, 1100]]})
        self.assertEqual(code, 200, r)
        self.assertEqual(r["error"], "")
        v = self.until(jid, lambda v: v["status"]["state"] in ("needs_input", "done"))
        self.assertTrue(v["videos"][1]["manual"] and v["videos"][1]["pitch"]["manual"])
        self.assertFalse(v["videos"][0]["manual"])

    def test_team_confirmation_needs_both_teams(self):
        code, r = self.call("/api/jobs", {"title": "g", "videos": [self.vid]})
        jid = r["id"]
        self.wait(jid, ("needs_input",))
        code, r = self.call(f"/api/jobs/{jid}/teams", {"a": [0], "b": []})
        self.assertEqual(code, 400)
        code, r = self.call(f"/api/jobs/{jid}/teams", {"a": [0], "b": [0]})
        self.assertEqual(code, 400)                               # a group cannot be in both teams

    def test_range_requests_for_video(self):
        code, r = self.call("/api/jobs", {"title": "r", "videos": [self.vid]})
        jid = r["id"]
        self.wait(jid, ("needs_input",))
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/files/{jid}/r_autopan.mp4", headers={"Range": "bytes=1-2"})
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 206)
            self.assertEqual(resp.read(), b"p4")
            self.assertEqual(resp.headers["Content-Range"], "bytes 1-2/3")

    def test_safety(self):
        code, _ = self.call("/api/jobs", {"title": "x", "videos": [self.vid]}, headers={"X-App": ""})
        self.assertEqual(code, 403)                               # no header: a web page elsewhere cannot start jobs
        code, _ = self.call("/api/probe", {"paths": []}, headers={"Host": "evil.example"})
        self.assertEqual(code, 403)
        code, r = self.call("/api/jobs", {"title": "s", "videos": [self.vid]})
        jid = r["id"]
        self.wait(jid, ("needs_input",))
        for bad in (f"/files/{jid}/../../etc/passwd", f"/files/{jid}/%2e%2e/%2e%2e/etc/passwd", "/files/../job.json", "/api/jobs/..%2f.."):
            code, _ = self.call(bad, raw=True)
            self.assertEqual(code, 404, bad)

    def test_one_game_at_a_time(self):
        gate = threading.Event()

        def slow(self, cmd, sid, parse=None):
            gate.wait(10)
            return fake_exec(self, cmd, sid, parse)
        P.Job.exec = slow
        code, r = self.call("/api/jobs", {"title": "first", "videos": [self.vid]})
        code, r2 = self.call("/api/jobs", {"title": "second", "videos": [self.vid]})
        self.assertIn("another game", r2["error"])
        gate.set()
        self.wait(r["id"], ("needs_input", "failed"))

    def test_cancel(self):
        gate = threading.Event()

        def slow(self, cmd, sid, parse=None):
            while not gate.is_set() and not self.cancel_flag:
                time.sleep(0.05)
            if self.cancel_flag:
                raise P.Cancelled()
            return fake_exec(self, cmd, sid, parse)
        P.Job.exec = slow
        code, r = self.call("/api/jobs", {"title": "c", "videos": [self.vid]})
        time.sleep(1.0)
        self.call(f"/api/jobs/{r['id']}/cancel", {})
        v = self.wait(r["id"], ("cancelled", "stopped"))
        self.assertIn(v["status"]["state"], ("cancelled", "stopped"))
        gate.set()


if __name__ == "__main__":
    unittest.main()
