"""
Tests for kf_core / annotate / render_keyframes / make_proxy.

    python -m pytest -q test_keyframe_tools.py        # or:  python test_keyframe_tools.py

Needs ffmpeg + ffprobe + opencv + numpy. The cross-checks against py360convert and scipy
are skipped when those are not installed.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annotate as A          # noqa: E402
import kf_core as K           # noqa: E402
import make_proxy             # noqa: E402
import render_keyframes as R  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
try:
    import py360convert  # noqa: F401
    HAVE_P360 = True
except Exception:
    HAVE_P360 = False
try:
    import scipy.interpolate as _si  # noqa: F401
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def ffmpeg_write_gray(path, frames, w, h, fps, pix="gray", codec=("-c:v", "ffv1"), extra=()):
    """frames: iterable of uint8/uint16 (h, w) arrays -> video at `path`."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", *extra, *codec, path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    assert p.wait() == 0


def decode_gray16(path, w, h):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray16le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (w * h * 2)
    return np.frombuffer(raw, dtype="<u2")[: n * w * h].reshape(n, h, w)


# ─────────────────────────────────────────────────────────────────────────────
class TestInterpolation(unittest.TestCase):
    def kfs(self, rows):
        return [K.Keyframe(t, p, f) for t, p, f in rows]

    @unittest.skipUnless(HAVE_SCIPY, "scipy not installed")
    def test_pchip_matches_scipy(self):
        rng = np.random.default_rng(1)
        t = np.cumsum(rng.uniform(0.5, 4, 12))
        pan = rng.uniform(-50, 50, 12)
        fov = rng.uniform(50, 100, 12)
        path = K.Path(self.kfs(zip(t, pan, fov)), "pchip")
        tq = np.linspace(t[0] - 1, t[-1] + 1, 500)
        got_pan, got_fov = path.at(tq)
        ref_pan = _si.PchipInterpolator(t, pan, extrapolate=False)(np.clip(tq, t[0], t[-1]))
        ref_fov = _si.PchipInterpolator(t, fov, extrapolate=False)(np.clip(tq, t[0], t[-1]))
        np.testing.assert_allclose(got_pan, ref_pan, atol=1e-9)
        np.testing.assert_allclose(got_fov, ref_fov, atol=1e-9)

    def test_pchip_never_overshoots_and_hits_keyframes(self):
        rows = [(0, 0, 80), (2, 30, 80), (3, 30, 60), (6, -20, 90)]
        path = K.Path(self.kfs(rows), "pchip")
        tq = np.linspace(0, 6, 2001)
        pan, fov = path.at(tq)
        self.assertLessEqual(pan.max(), 30 + 1e-9)       # plateau at 30 not exceeded
        self.assertGreaterEqual(pan.min(), -20 - 1e-9)
        self.assertGreaterEqual(fov.min(), 60 - 1e-9)
        self.assertLessEqual(fov.max(), 90 + 1e-9)
        for t, p, f in rows:
            gp, gf = path.at(float(t))
            self.assertAlmostEqual(gp, p, places=9)
            self.assertAlmostEqual(gf, f, places=9)

    def test_holds_outside_range_and_single_keyframe(self):
        path = K.Path(self.kfs([(2, 10, 70), (4, -10, 90)]), "pchip")
        self.assertEqual(path.at(0.0), (10.0, 70.0))
        self.assertEqual(path.at(99.0), (-10.0, 90.0))
        one = K.Path(self.kfs([(5, 7, 66)]), "pchip")
        self.assertEqual(one.at(0.0), (7.0, 66.0))
        self.assertEqual(one.at(50.0), (7.0, 66.0))

    def test_smoothstep_and_linear(self):
        rows = [(0, 0, 80), (2, 20, 60)]
        lin = K.Path(self.kfs(rows), "linear")
        ss = K.Path(self.kfs(rows), "smoothstep")
        self.assertAlmostEqual(lin.at(1.0)[0], 10.0)
        self.assertAlmostEqual(ss.at(1.0)[0], 10.0)
        self.assertLess(ss.at(0.2)[0], lin.at(0.2)[0])    # eases in

    def test_pchip_velocity_is_continuous(self):
        rows = [(0, 0, 80), (1, 10, 80), (2, 25, 80), (3, 25, 80), (4, 5, 80)]
        path = K.Path(self.kfs(rows), "pchip")
        eps = 1e-6
        for t in (1.0, 2.0, 3.0):
            a = (path.at(t)[0] - path.at(t - eps)[0]) / eps
            b = (path.at(t + eps)[0] - path.at(t)[0]) / eps
            self.assertAlmostEqual(a, b, places=3)


# ─────────────────────────────────────────────────────────────────────────────
class TestKeyframeSet(unittest.TestCase):
    def test_toggle_undo_redo(self):
        ks = K.KeyframeSet(tol=1 / 60)
        self.assertTrue(ks.toggle(K.Keyframe(1.0, 5, 80)))
        self.assertTrue(ks.toggle(K.Keyframe(0.5, -5, 70)))
        self.assertEqual([k.t for k in ks], [0.5, 1.0])                    # sorted
        self.assertFalse(ks.toggle(K.Keyframe(1.0 + 1 / 120, 0, 0)))        # within tol -> removes
        self.assertEqual([k.t for k in ks], [0.5])
        self.assertTrue(ks.undo())
        self.assertEqual([k.t for k in ks], [0.5, 1.0])
        self.assertTrue(ks.undo())
        self.assertTrue(ks.undo())
        self.assertEqual(len(ks), 0)
        self.assertFalse(ks.undo())
        self.assertTrue(ks.redo())
        self.assertEqual(len(ks), 1)

    def test_upsert_replaces_and_remove_between(self):
        ks = K.KeyframeSet(tol=0.01)
        for t in (1, 2, 3, 4, 5):
            ks.upsert(K.Keyframe(t, t, 80))
        ks.upsert(K.Keyframe(3.004, 99, 80))
        self.assertEqual(len(ks), 5)
        self.assertEqual(ks.at(3.0).pan_deg, 99)
        self.assertEqual(ks.remove_between(1, 5), 3)                        # exclusive of both ends
        self.assertEqual([k.t for k in ks], [1, 5])

    def test_json_roundtrip_matches_spec(self):
        ks = K.KeyframeSet()
        ks.upsert(K.Keyframe(12.3456789, -12.34567, 77.7777777))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "k.json")
            ks.save(p)
            with open(p) as fh:
                rows = json.load(fh)
            self.assertEqual(rows, [{"t": 12.3457, "pan_deg": -12.3457, "fov_deg": 77.7778}])
            back = K.KeyframeSet.load(p)
            self.assertAlmostEqual(back.items[0].t, 12.3457)
            self.assertEqual(os.listdir(d), ["k.json"])                     # no temp litter

    def test_load_rejects_bad_data(self):
        for bad in ([{"t": 1, "pan_deg": 2}], [{"t": "x", "pan_deg": 1, "fov_deg": 80}],
                    [{"t": -1, "pan_deg": 1, "fov_deg": 80}],
                    [{"t": 1, "pan_deg": float("nan"), "fov_deg": 80}]):
            with self.assertRaises(ValueError):
                K.KeyframeSet.from_list(bad)


# ─────────────────────────────────────────────────────────────────────────────
class TestGeometry(unittest.TestCase):
    def test_sign_conventions(self):
        # positive pan looks LEFT (centre longitude = -pan), as in autopan_infer.py
        self.assertEqual(K.centre_lonlat(30, -5), (-30.0, -5.0))
        lon, lat = K.crop_outline(0, 0, 80)
        self.assertAlmostEqual(float(lon.min()), -40.0, delta=0.5)
        self.assertAlmostEqual(float(lon.max()), 40.0, delta=3.0)   # right edge sampled at endpoint=False
        self.assertTrue(K.point_in_crop(0, 0, 0, 0, 80))
        self.assertTrue(K.point_in_crop(-30, 0, 30, 0, 80))          # pan=+30 covers lon=-30
        self.assertFalse(K.point_in_crop(30, 0, 30, 0, 80))
        self.assertFalse(K.point_in_crop(180, 0, 0, 0, 80))          # behind the camera

    def test_vertical_fov_is_rectilinear_16_9(self):
        self.assertAlmostEqual(K.vfov_deg(80), 2 * math.degrees(math.atan(math.tan(math.radians(40)) * 9 / 16)))
        self.assertAlmostEqual(K.vfov_deg(80, legacy_fov=True), 80.0)

    @unittest.skipUnless(HAVE_P360, "py360convert not installed")
    def test_matches_py360convert_used_by_autopan_infer(self):
        from py360convert import e2p
        W, H, oh, ow = 2880, 1440, 360, 640
        xs = np.tile(np.arange(W, dtype=np.float32)[None, :], (H, 1))
        ys = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))
        img = np.stack([xs, ys], -1)
        for pan, tilt, fov in [(0, 0, 80), (25, -8, 80), (-40, -12, 60), (55, 5, 100), (-55, -20, 70)]:
            ref = e2p(img, fov_deg=fov, u_deg=-pan, v_deg=tilt, out_hw=(oh, ow), mode="nearest")
            mx, my = K.build_remap(pan, tilt, fov, W, H, ow, oh, legacy_fov=True)
            dx = np.abs(((mx - ref[..., 0]) + W / 2) % W - W / 2)
            dy = np.abs(my - ref[..., 1])
            self.assertLessEqual(dx.max(), 0.51, (pan, tilt, fov))   # nearest-neighbour rounding only
            self.assertLessEqual(dy.max(), 0.51, (pan, tilt, fov))

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
    def test_matches_ffmpeg_v360(self):
        W, H, oh, ow = 2880, 1440, 360, 640
        xs = np.tile(np.arange(W, dtype=np.float64)[None, :], (H, 1))
        ys = np.tile(np.arange(H, dtype=np.float64)[:, None], (1, W))
        img = np.zeros((H, W, 3), np.uint16)
        img[..., 2] = np.round(xs / (W - 1) * 65535)
        img[..., 1] = np.round(ys / (H - 1) * 65535)
        with tempfile.TemporaryDirectory() as d:
            src, out = os.path.join(d, "eq.png"), os.path.join(d, "o.png")
            cv2.imwrite(src, img)
            for pan, tilt, fov in [(0, 0, 80), (25, -8, 80), (-40, -12, 60), (55, 5, 100), (10, -15, 45)]:
                q = K.to_v360_params(pan, tilt, fov, ow, oh)
                vf = (f"format=gbrp16le,v360=input=equirect:output=flat:w={ow}:h={oh}:interp=nearest:"
                      f"yaw={q['yaw']}:pitch={q['pitch']}:h_fov={q['h_fov']}:v_fov={q['v_fov']},format=rgb48be")
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", vf, "-frames:v", "1", out],
                               check=True)
                o = cv2.imread(out, cv2.IMREAD_UNCHANGED).astype(np.float64)
                X, Y = o[..., 2] / 65535 * (W - 1), o[..., 1] / 65535 * (H - 1)
                mx, my = K.build_remap(pan, tilt, fov, W, H, ow, oh)
                dx = np.abs(((mx - X) + W / 2) % W - W / 2)
                self.assertLessEqual(dx.max(), 1.6, (pan, tilt, fov))
                self.assertLessEqual(np.abs(my - Y).max(), 1.6, (pan, tilt, fov))

    def test_coarse_remap_matches_exact(self):
        W, H, pw, ph = 1440, 720, 460, 259
        for pan, tilt, fov in [(0, 0, 80), (25, -8, 80), (-40, -12, 60), (55, 5, 120), (170, -10, 90)]:
            ex, ey = K.build_remap(pan, tilt, fov, W, H, pw, ph)
            cx, cy = K.build_remap(pan, tilt, fov, W, H, pw, ph, step=4)
            self.assertEqual(cx.shape, ex.shape)
            self.assertLess(np.abs(cx - ex).max(), 0.3, (pan, tilt, fov))
            self.assertLess(np.abs(cy - ey).max(), 0.3, (pan, tilt, fov))

    def test_remap_wraps_across_the_seam(self):
        W, H = 720, 360
        img = np.tile((np.arange(W) % 256).astype(np.uint8)[None, :], (H, 1))
        mx, my = K.build_remap(0, 0, 80, W, H, 160, 90)           # looks straight ahead: lon 0 = x 360
        view = cv2.remap(img, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
        self.assertTrue(abs(int(view[45, 80]) - 360 % 256) <= 2)
        mx2, my2 = K.build_remap(-180, 0, 80, W, H, 160, 90)      # centred on the seam
        self.assertTrue(mx2.min() < 0 or mx2.max() > W)            # unwrapped, continuous
        view2 = cv2.remap(img, mx2, my2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        self.assertEqual(view2.shape, (90, 160))


# ─────────────────────────────────────────────────────────────────────────────
@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class TestRender(unittest.TestCase):
    """Frame-accurate end-to-end check of the sendcmd/v360 render."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="kftest_")
        W, H, fps, n = 1440, 720, 30, 150
        row = np.round(np.arange(W) / (W - 1) * 65535).astype("<u2")
        frame = np.tile(row[None, :], (H, 1))
        cls.src = os.path.join(cls.tmp, "lon16.mkv")
        ffmpeg_write_gray(cls.src, (frame for _ in range(n)), W, H, fps, pix="gray16le")
        cls.W, cls.fps = W, fps
        cls.kf = os.path.join(cls.tmp, "k.json")
        with open(cls.kf, "w") as fh:
            json.dump([{"t": 0.5, "pan_deg": 0, "fov_deg": 80},
                       {"t": 1.5, "pan_deg": -45, "fov_deg": 60},    # fast: ~2 deg / frame
                       {"t": 3.0, "pan_deg": 20, "fov_deg": 100}], fh)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_every_frame_follows_the_interpolated_path(self):
        out = os.path.join(self.tmp, "out.mkv")
        rc = R.main(["--input", self.src, "--keyframes", self.kf, "--output", out, "--encoder", "ffv1",
                     "--tilt", "0", "--interp-filter", "nearest"])
        self.assertEqual(rc, 0)
        frames = decode_gray16(out, 1280, 720)
        self.assertEqual(len(frames), 150)
        path = K.Path(K.KeyframeSet.load(self.kf).items, "pchip")
        W = self.W
        lon = lambda v: (v / 65535 * (W - 1) / W - 0.5) * 360.0       # noqa: E731
        worst_pan = worst_fov = 0.0
        for i, f in enumerate(frames):
            row = f[359].astype(np.float64)                             # centre row (tilt 0)
            pan, fov = path.at(i / self.fps)
            c = lon(row[640])
            worst_pan = max(worst_pan, abs(c - (-pan)))
            ext = lon(row[-1]) - lon(row[0])
            worst_fov = max(worst_fov, abs(ext - fov))
        # nearest-neighbour sampling of a 4 px/deg source limits accuracy to ~0.15 deg; being a whole
        # frame late/early at the fastest pan (~2 deg/frame) would be far outside this bound
        self.assertLess(worst_pan, 0.35, f"pan error {worst_pan:.3f} deg")
        self.assertLess(worst_fov, 0.6, f"fov error {worst_fov:.3f} deg")

    def test_constant_camera_needs_no_commands(self):
        path = K.Path([K.Keyframe(0, 12, 80), K.Keyframe(10, 12, 80)], "pchip")
        text, n, first = R.build_sendcmd(path, -5.0, 300, 30.0, 0.0, 1280, 720, False)
        self.assertEqual((text, n), ("", 0))
        self.assertAlmostEqual(first["yaw"], -12.0)

    def test_dry_run_command_uses_unchanged_pipeline_filter_for_raw_input(self):
        raw = os.path.join(self.tmp, "raw.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "testsrc2=size=1440x2560:rate=30:duration=1", "-pix_fmt", "yuv420p", raw], check=True)
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = R.main(["--input", raw, "--keyframes", self.kf, "--output", "x.mp4", "--dry-run"])
        txt = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("source=raw", txt)
        self.assertIn("v360=fisheye:equirect:ih_fov=200:iv_fov=200,scale=2880:1440", txt)
        self.assertNotIn("noautorotate", txt)
        self.assertNotIn("rotate=", txt)

    def test_render_picks_up_tilt_and_interp_saved_by_the_annotator(self):
        import contextlib
        import io
        kf = os.path.join(self.tmp, "withmeta.keyframes.json")
        shutil.copy(self.kf, kf)
        K.save_meta(kf, tilt_deg=-7.5, interp="linear", legacy_fov=False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            R.main(["--input", self.src, "--keyframes", kf, "--output", "x.mkv", "--dry-run"])
        self.assertIn("tilt=-7.50", buf.getvalue())
        self.assertIn("interp=linear", buf.getvalue())
        buf = io.StringIO()                                        # explicit flags win over the sidecar
        with contextlib.redirect_stdout(buf):
            R.main(["--input", self.src, "--keyframes", kf, "--output", "x.mkv", "--dry-run",
                    "--tilt", "3", "--interp", "smoothstep"])
        self.assertIn("tilt=+3.00", buf.getvalue())
        self.assertIn("interp=smoothstep", buf.getvalue())

    def test_static_camera_renders(self):
        """No camera movement in the rendered range -> no sendcmd file; must not crash."""
        out = os.path.join(self.tmp, "static.mkv")
        rc = R.main(["--input", self.src, "--keyframes", self.kf, "--output", out, "--encoder", "ffv1",
                     "--duration", "0.4"])                    # before the first keyframe: camera holds
        self.assertEqual(rc, 0)
        self.assertEqual(len(decode_gray16(out, 1280, 720)), 12)

    def test_raw_input_renders_end_to_end(self):
        raw = os.path.join(self.tmp, "raw2.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "testsrc2=size=1440x2560:rate=30:duration=1", "-pix_fmt", "yuv420p", raw], check=True)
        out = os.path.join(self.tmp, "raw_out.mp4")
        rc = R.main(["--input", raw, "--keyframes", self.kf, "--output", out, "--duration", "0.5"])
        self.assertEqual(rc, 0)
        info = K.probe_video(out)
        self.assertEqual((info["width"], info["height"]), (1280, 720))
        self.assertGreaterEqual(info["nb_frames"], 14)

    def test_proxy_conversion(self):
        raw = os.path.join(self.tmp, "raw3.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "testsrc2=size=1440x2560:rate=30000/1001:duration=1.5", "-pix_fmt", "yuv420p", raw],
                       check=True)
        out = os.path.join(self.tmp, "proxy.mp4")
        self.assertEqual(make_proxy.main([raw, out, "--width", "640"]), 0)
        info = K.probe_video(out)
        self.assertEqual((info["width"], info["height"]), (640, 320))
        self.assertAlmostEqual(info["fps"], 30000 / 1001, places=3)
        self.assertTrue(K.is_equirect_size(info["width"], info["height"]))


# ─────────────────────────────────────────────────────────────────────────────
@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class TestAnnotator(unittest.TestCase):
    FPS, N = 30, 120

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="kfui_")
        W, H = 640, 320
        # frame i has a flat grey level of 2*i (+ a horizon band) so seeks can be verified
        frames = []
        for i in range(cls.N):
            f = np.full((H, W), 2 * i + 4, np.uint8)
            frames.append(f)
        cls.video = os.path.join(cls.tmp, "idx.mp4")
        ffmpeg_write_gray(cls.video, frames, W, H, cls.FPS, pix="gray",
                          codec=("-c:v", "libx264", "-preset", "veryfast", "-crf", "10", "-g", "10",
                                 "-bf", "0", "-pix_fmt", "yuv420p"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def app(self, name="a", **kw):
        kfp = os.path.join(self.tmp, f"{name}.keyframes.json")
        for p in (kfp, K.meta_path(kfp)):
            if os.path.exists(p):
                os.unlink(p)
        return A.Annotator(self.video, kfp, disp_w=640, **kw)

    # -- video access --------------------------------------------------------
    def test_random_access_is_frame_accurate(self):
        src = A.FrameSource(self.video, 640, cache_size=4)
        order = [0, 1, 2, 3, 10, 11, 5, 4, 3, 60, 59, 58, 119, 0, 87, 30, 31, 90, 12, 12, 7]
        for idx in order:
            f = src.get(idx)
            self.assertIsNotNone(f)
            self.assertAlmostEqual(float(f[..., 0].mean()), 2 * idx + 4, delta=2.5, msg=f"frame {idx}")
        src.close()

    def test_backward_stepping_decodes_blocks_not_one_seek_per_frame(self):
        src = A.FrameSource(self.video, 640, cache_size=48)

        class CountingCap:
            def __init__(self, cap):
                self.cap, self.seeks, self.reads = cap, 0, 0

            def set(self, *a):
                self.seeks += 1
                return self.cap.set(*a)

            def read(self):
                self.reads += 1
                return self.cap.read()

            def __getattr__(self, name):
                return getattr(self.cap, name)

        for i in range(0, 61):                       # play forward to frame 60
            src.get(i)
        src._cache.clear()                           # as if it had been evicted
        src.cap = cc = CountingCap(src.cap)
        for idx in range(59, 40, -1):                # then step back 19 frames
            f = src.get(idx)
            self.assertAlmostEqual(float(f[..., 0].mean()), 2 * idx + 4, delta=2.5, msg=f"frame {idx}")
        self.assertLessEqual(cc.seeks, 2)            # 19 steps cost two block decodes, not 19 seeks
        self.assertLessEqual(cc.reads, 2 * A.FrameSource.BACK_BLOCK + 2)
        src.close()

    # -- navigation ----------------------------------------------------------
    def test_navigation_keys(self):
        a = self.app()
        a.handle_key(ord("d")); a.handle_key(ord("d")); a.handle_key(ord("d"))
        self.assertEqual(a.idx, 3)
        a.handle_key(ord("a"))
        self.assertEqual(a.idx, 2)
        a.handle_key(ord("e"))                       # +5 s = +150 frames -> clamps to the end
        self.assertEqual(a.idx, self.N - 1)
        a.handle_key(ord("Q"))                       # -10 s = -300 frames -> clamps to 0
        self.assertEqual(a.idx, 0)
        a.handle_key(63235)                          # macOS right arrow
        self.assertEqual(a.idx, 1)
        a.handle_key(65361)                          # Linux left arrow
        self.assertEqual(a.idx, 0)

    def test_playback_follows_wall_clock_and_speed(self):
        a = self.app()
        a.handle_key(ord("3"))                       # 2x
        self.assertEqual(a.speed, 2.0)
        a.handle_key(ord(" "))
        t0 = a._play_wall0
        a.process(now=t0 + 0.5)                      # 0.5 s * 30 fps * 2x = frame 30
        self.assertEqual(a.idx, 30)
        a.process(now=t0 + 100)                      # runs off the end -> stops on the last frame
        self.assertEqual(a.idx, self.N - 1)
        self.assertFalse(a.playing)
        a.handle_key(ord("0"))
        self.assertEqual(a.speed, 0.5)

    def test_timeline_click_and_scrub(self):
        a = self.app()
        x = A.PAD + (a.disp_w - 2 * A.PAD) // 2
        a.on_mouse(cv2.EVENT_LBUTTONDOWN, x, a.eq_h + 10, 0)
        a.process(now=100.0)
        self.assertAlmostEqual(a.idx, (a.n - 1) / 2, delta=1)
        a.on_mouse(cv2.EVENT_MOUSEMOVE, A.PAD, a.eq_h + 10, cv2.EVENT_FLAG_LBUTTON)
        a.on_mouse(cv2.EVENT_LBUTTONUP, A.PAD, a.eq_h + 10, 0)
        a.process(now=101.0)
        self.assertEqual(a.idx, 0)

    # -- crop editing --------------------------------------------------------
    def test_mouse_drag_moves_crop_with_autopan_sign(self):
        a = self.app()
        cx, cy = a.disp_w // 2, int(a.eq_h * 0.6)     # inside the crop (pan 0 -> centre lon 0)
        a.on_mouse(cv2.EVENT_LBUTTONDOWN, cx, cy, 0)
        a.on_mouse(cv2.EVENT_MOUSEMOVE, cx + 64, cy, cv2.EVENT_FLAG_LBUTTON)
        a.on_mouse(cv2.EVENT_LBUTTONUP, cx + 64, cy, 0)
        self.assertAlmostEqual(a.pan, -64 / 640 * 360, places=6)    # dragged right -> pan decreases
        self.assertTrue(a.hold)
        self.assertEqual(len(a.kfs), 0)

    def test_click_outside_recentres_crop_there(self):
        a = self.app()
        x = int(round(a.disp_w * (0.5 + 100 / 360)))   # lon ~ +100, far outside the crop
        a.on_mouse(cv2.EVENT_LBUTTONDOWN, x, int(a.eq_h * 0.6), 0)
        a.on_mouse(cv2.EVENT_LBUTTONUP, x, int(a.eq_h * 0.6), 0)
        self.assertAlmostEqual(a.pan, -(x / a.disp_w - 0.5) * 360, places=6)
        self.assertAlmostEqual(a.pan, -100.0, delta=0.3)

    def test_scroll_zoom(self):
        a = self.app()
        f0 = a.fov
        flags_up = 120 << 16
        a.on_mouse(cv2.EVENT_MOUSEWHEEL, 100, 100, flags_up)
        self.assertLess(a.fov, f0)
        for _ in range(200):                          # a trackpad burst within ~1 ms changes it only once
            a.on_mouse(cv2.EVENT_MOUSEWHEEL, 100, 100, flags_up)
        self.assertGreater(a.fov, f0 * 0.97)
        a._last_wheel = -1.0
        a.on_mouse(cv2.EVENT_MOUSEWHEEL, 100, 100, (-120 & 0xFFFF) << 16)
        self.assertGreater(a.fov, f0 * 0.99 - 1e-6)
        for _ in range(500):
            a.zoom(0.9)
        self.assertEqual(a.fov, K.FOV_MIN)
        for _ in range(500):
            a.zoom(1.1)
        self.assertEqual(a.fov, K.FOV_MAX)

    def test_keyframe_toggle_hold_and_live_edit(self):
        a = self.app()
        a.goto(30)
        a.handle_key(ord("L")); a.handle_key(ord("i"))
        self.assertTrue(a.hold)
        a.handle_key(ord("k"))                        # commit
        self.assertFalse(a.hold)
        self.assertEqual(len(a.kfs), 1)
        kf = a.kfs.items[0]
        self.assertAlmostEqual(kf.t, 1.0)
        self.assertAlmostEqual(kf.pan_deg, -5.0)      # L = 5 deg right = pan -5
        self.assertAlmostEqual(kf.fov_deg, 80 * 0.96)
        a.handle_key(ord("J"))                        # parked on the keyframe -> edits it live
        self.assertFalse(a.hold)
        self.assertAlmostEqual(a.kfs.items[0].pan_deg, 0.0)
        a.handle_key(ord("k"))                        # again -> removes
        self.assertEqual(len(a.kfs), 0)
        a.handle_key(ord("z"))                        # undo the removal ...
        self.assertEqual(len(a.kfs), 1)
        self.assertAlmostEqual(a.kfs.items[0].pan_deg, 0.0)   # ... restored incl. the live edit
        a.handle_key(ord("z"))                        # undo the live edit (one step for J)
        self.assertAlmostEqual(a.kfs.items[0].pan_deg, -5.0)
        a.handle_key(ord("y"))
        self.assertAlmostEqual(a.kfs.items[0].pan_deg, 0.0)

    def test_k_on_existing_keyframe_after_moving_crop_updates_it(self):
        a = self.app()
        a.goto(30); a.handle_key(ord("k"))
        a.goto(31); a.goto(30)
        a.hold = True; a.pan = 17.0                   # user steered while parked off-path
        a.handle_key(ord("k"))
        self.assertEqual(len(a.kfs), 1)
        self.assertAlmostEqual(a.kfs.items[0].pan_deg, 17.0)

    def test_crop_follows_interpolated_path_until_user_moves_it(self):
        a = self.app()
        a.goto(0); a.pan = 0; a.handle_key(ord("k"))
        a.goto(60); a.pan = -30; a.hold = True; a.handle_key(ord("k"))
        a.goto(30)
        self.assertAlmostEqual(a.pan, -15.0, delta=0.5)         # midpoint of the path
        a.handle_key(ord("L"))                                   # user moves it -> held
        self.assertTrue(a.hold)
        held = a.pan
        a.goto(35)
        self.assertEqual(a.pan, held)                            # hold survives stepping
        a.handle_key(ord("g"))                                   # discard -> back to the path
        self.assertFalse(a.hold)
        self.assertNotEqual(a.pan, held)

    def test_jump_between_keyframes_and_neighbours(self):
        a = self.app()
        for i in (10, 40, 70):
            a.goto(i); a.hold = True; a.handle_key(ord("k"))
        a.goto(0)
        a.handle_key(ord(".")); self.assertEqual(a.idx, 10)
        a.handle_key(ord(".")); self.assertEqual(a.idx, 40)
        a.handle_key(ord(",")); self.assertEqual(a.idx, 10)
        a.handle_key(ord(",")); self.assertEqual(a.idx, 10)      # none before -> stays

    # -- REC -----------------------------------------------------------------
    def test_rec_autokeyframes_while_steering(self):
        a = self.app(rec_interval=0.5)
        a.handle_key(ord("r"))
        a.handle_key(ord(" "))
        t0 = a._play_wall0
        for step in range(1, 91):                     # 3 s of playback in 1/30 s steps
            now = t0 + step / 30.0
            a.set_crop(pan=-step * 0.3)               # steering all the time
            a.process(now=now)
        times = [k.t for k in a.kfs]
        self.assertGreaterEqual(len(times), 5)
        gaps = np.diff(times)
        self.assertTrue(np.all(gaps >= 0.5 - 1e-6), gaps)
        self.assertTrue(np.all(gaps < 0.6), gaps)
        pans = [k.pan_deg for k in a.kfs]
        self.assertTrue(np.all(np.diff(pans) < 0))                # monotone track of the steering
        n_before = len(a.kfs)
        a.handle_key(ord("z"))                                    # the whole take is one undo step
        self.assertEqual(len(a.kfs), 0)
        self.assertGreater(n_before, 0)

    def test_rec_overwrites_keyframes_it_passes_over(self):
        a = self.app(rec_interval=0.5)
        for i in (10, 12, 14):                        # stale keyframes in the take's way
            a.goto(i); a.hold = True; a.handle_key(ord("k"))
        a.goto(0)
        a.handle_key(ord("r")); a.handle_key(ord(" "))
        t0 = a._play_wall0
        for step in range(1, 31):
            a.set_crop(pan=5.0)
            a.process(now=t0 + step / 30.0)
        stale = [k for k in a.kfs if abs(k.t - 12 / 30) < 1e-6]
        self.assertEqual(stale, [])

    # -- persistence ---------------------------------------------------------
    def test_autosave_resume_and_meta(self):
        a = self.app("persist", tilt=-8.0)
        a.goto(45); a.handle_key(ord("k"))
        a.process(now=50.0)                           # autosave tick
        with open(a.kf_path) as fh:
            rows = json.load(fh)
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), {"t", "pan_deg", "fov_deg"})
        a.handle_key(ord("w")); a.handle_key(ord("w"))
        self.assertAlmostEqual(a.tilt, -7.0)
        a.save(force=True)
        meta = K.load_meta(a.kf_path)
        self.assertEqual(meta["tilt_deg"], -7.0)
        b = A.Annotator(self.video, a.kf_path, disp_w=640)       # reopen
        self.assertEqual(len(b.kfs), 1)
        self.assertAlmostEqual(b.tilt, -7.0)
        self.assertEqual(b.idx, 45)                              # resumed where we left off

    def test_rejects_non_equirect_video(self):
        bad = os.path.join(self.tmp, "portrait.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "testsrc2=size=360x640:rate=30:duration=0.5", "-pix_fmt", "yuv420p", bad], check=True)
        with self.assertRaises(SystemExit) as cm:
            A.Annotator(bad, os.path.join(self.tmp, "bad.json"), disp_w=640)
        self.assertIn("make_proxy.py", str(cm.exception))

    # -- drawing -------------------------------------------------------------
    def test_render_produces_expected_canvas_in_every_mode(self):
        a = self.app()
        a.goto(20); a.hold = True; a.handle_key(ord("k"))
        for toggle in (None, ord("f"), ord("v"), ord("h")):
            if toggle:
                a.handle_key(toggle)
            img = a.render()
            self.assertEqual(img.shape, (a.size[1], a.size[0], 3))
            self.assertEqual(img.dtype, np.uint8)
        a.show_help = False
        a.pan = 175.0                                  # crop straddling the +-180 seam must not crash
        self.assertEqual(a.render().shape, (a.size[1], a.size[0], 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
