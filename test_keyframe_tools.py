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
import extract_detections as X  # noqa: E402
import auto_keyframes as AK
import tune_auto as TU  # noqa: E402

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

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
    def test_cylindrical_matches_ffmpeg_v360(self):
        W, H, oh, ow = 2880, 1440, 360, 640
        xs = np.tile(np.arange(W, dtype=np.float64)[None, :], (H, 1))
        ys = np.tile(np.arange(H, dtype=np.float64)[:, None], (1, W))
        img = np.zeros((H, W, 3), np.uint16)
        img[..., 2] = np.round(xs / (W - 1) * 65535)
        img[..., 1] = np.round(ys / (H - 1) * 65535)
        with tempfile.TemporaryDirectory() as d:
            src, out = os.path.join(d, "eq.png"), os.path.join(d, "o.png")
            cv2.imwrite(src, img)
            for pan, tilt, fov in [(0, 0, 100), (30, -10, 110), (-45, -10, 120), (10, 15, 90), (60, -5, 70)]:
                q = K.to_v360_params(pan, tilt, fov, ow, oh, projection="cylindrical")
                vf = (f"format=gbrp16le,v360=input=equirect:output=cylindrical:w={ow}:h={oh}:interp=nearest:"
                      f"yaw={q['yaw']}:pitch={q['pitch']}:h_fov={q['h_fov']}:v_fov={q['v_fov']},format=rgb48be")
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", vf, "-frames:v", "1", out],
                               check=True)
                o = cv2.imread(out, cv2.IMREAD_UNCHANGED).astype(np.float64)
                X, Y = o[..., 2] / 65535 * (W - 1), o[..., 1] / 65535 * (H - 1)
                mx, my = K.build_remap(pan, tilt, fov, W, H, ow, oh, projection="cylindrical")
                dx = np.abs(((mx - X) + W / 2) % W - W / 2)
                self.assertLessEqual(dx.max(), 1.8, (pan, tilt, fov))
                self.assertLessEqual(np.abs(my - Y).max(), 1.8, (pan, tilt, fov))

    def test_cylindrical_properties(self):
        # square pixels at the centre; straight vertical edges; far less edge stretch than flat
        xm, ym = K.plane_extent(110, 1280, 720, projection="cylindrical")
        self.assertAlmostEqual(xm * 2 / 1280, ym * 2 / 720, places=9)
        lon, lat = K.crop_outline(0, 0, 110, 1280, 720, n=40, projection="cylindrical")
        left = lon[3 * 40:4 * 40]                       # left edge, bottom -> top
        self.assertLess(np.ptp(left), 1e-3)             # a vertical line stays at one longitude
        self.assertAlmostEqual(float(-left[0]), 55.0, places=3)
        flat_v = K.vfov_deg(110, 1280, 720)
        cyl_v = K.vfov_deg(110, 1280, 720, projection="cylindrical")
        self.assertLess(cyl_v, flat_v)
        # a point just inside / outside the edge
        self.assertTrue(K.point_in_crop(54, 0, 0, 0, 110, projection="cylindrical"))
        self.assertFalse(K.point_in_crop(56, 0, 0, 0, 110, projection="cylindrical"))
        self.assertFalse(K.point_in_crop(180, 0, 0, 0, 110, projection="cylindrical"))
        self.assertTrue(K.point_in_crop(0, 0.9 * cyl_v / 2, 0, 0, 110, projection="cylindrical"))
        self.assertFalse(K.point_in_crop(0, 1.1 * cyl_v / 2, 0, 0, 110, projection="cylindrical"))

    def test_coarse_remap_matches_exact(self):
        W, H, pw, ph = 1440, 720, 460, 259
        for proj in K.PROJECTIONS:
            for pan, tilt, fov in [(0, 0, 80), (25, -8, 80), (-40, -12, 60), (55, 5, 120), (170, -10, 90)]:
                ex, ey = K.build_remap(pan, tilt, fov, W, H, pw, ph, projection=proj)
                cx, cy = K.build_remap(pan, tilt, fov, W, H, pw, ph, step=4, projection=proj)
                self.assertEqual(cx.shape, ex.shape)
                self.assertLess(np.abs(cx - ex).max(), 0.3, (proj, pan, tilt, fov))
                self.assertLess(np.abs(cy - ey).max(), 0.3, (proj, pan, tilt, fov))

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

    def test_cylindrical_render_follows_the_interpolated_path(self):
        out = os.path.join(self.tmp, "cyl.mkv")
        rc = R.main(["--input", self.src, "--keyframes", self.kf, "--output", out, "--encoder", "ffv1",
                     "--tilt", "0", "--interp-filter", "nearest", "--projection", "cylindrical"])
        self.assertEqual(rc, 0)
        frames = decode_gray16(out, 1280, 720)
        path = K.Path(K.KeyframeSet.load(self.kf).items, "pchip")
        W = self.W
        lon = lambda v: (v / 65535 * (W - 1) / W - 0.5) * 360.0       # noqa: E731
        worst_pan = worst_fov = 0.0
        for i, f in enumerate(frames):
            row = f[359].astype(np.float64)
            pan, fov = path.at(i / self.fps)
            worst_pan = max(worst_pan, abs(lon(row[640]) - (-pan)))
            worst_fov = max(worst_fov, abs(lon(row[-1]) - lon(row[0]) - fov))
        self.assertLess(worst_pan, 0.35, f"pan error {worst_pan:.3f} deg")
        self.assertLess(worst_fov, 0.6, f"fov error {worst_fov:.3f} deg")
        # in the cylindrical view the longitude is linear in x across the row
        mid = frames[75][359].astype(np.float64)
        lons = np.array([lon(v) for v in mid[::160]])
        steps = np.diff(lons)
        self.assertLess(np.ptp(steps), 0.1 * abs(steps.mean()) + 0.2)

    def test_render_picks_up_projection_saved_by_the_annotator(self):
        import contextlib
        import io
        kf = os.path.join(self.tmp, "cylmeta.keyframes.json")
        shutil.copy(self.kf, kf)
        K.save_meta(kf, tilt_deg=-7.5, interp="pchip", legacy_fov=False, projection="cylindrical")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            R.main(["--input", self.src, "--keyframes", kf, "--output", "x.mkv", "--dry-run"])
        self.assertIn("cylindrical", buf.getvalue())
        self.assertIn("output=cylindrical", buf.getvalue())

    def test_output_has_square_pixels(self):
        out = os.path.join(self.tmp, "sar.mp4")
        self.assertEqual(R.main(["--input", self.src, "--keyframes", self.kf, "--output", out,
                                 "--tilt", "0", "--duration", "0.5"]), 0)
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=sample_aspect_ratio", "-of", "csv=p=0", out],
                           capture_output=True, text=True).stdout.strip()
        self.assertIn(r, ("1:1", "N/A"))

    def test_render_honours_game_marks_and_explicit_overrides(self):
        kf = os.path.join(self.tmp, "marks.keyframes.json")
        shutil.copy(self.kf, kf)
        K.save_meta(kf, tilt_deg=0.0, interp="pchip", legacy_fov=False, projection="flat",
                    trim_in=1.0, trim_out=2.0)
        out = os.path.join(self.tmp, "marks.mkv")
        self.assertEqual(R.main(["--input", self.src, "--keyframes", kf, "--output", out, "--encoder", "ffv1",
                                 "--interp-filter", "nearest"]), 0)
        frames = decode_gray16(out, 1280, 720)
        self.assertEqual(len(frames), 30)                                  # exactly the marked second
        path = K.Path(K.KeyframeSet.load(kf).items, "pchip")
        W = self.W
        lon = lambda v: (v / 65535 * (W - 1) / W - 0.5) * 360.0          # noqa: E731
        for i in (0, 15, 29):
            pan, _ = path.at(1.0 + i / self.fps)                           # the path continues from t = IN
            self.assertLess(abs(lon(frames[i][359].astype(np.float64)[640]) - (-pan)), 0.4)
        out2 = os.path.join(self.tmp, "marks2.mkv")                       # explicit --duration beats the marks
        R.main(["--input", self.src, "--keyframes", kf, "--output", out2, "--encoder", "ffv1",
                "--duration", "0.5"])
        self.assertEqual(len(decode_gray16(out2, 1280, 720)), 15)
        out3 = os.path.join(self.tmp, "marks3.mkv")                       # --no-trim renders the whole file
        R.main(["--input", self.src, "--keyframes", kf, "--output", out3, "--encoder", "ffv1", "--no-trim"])
        self.assertEqual(len(decode_gray16(out3, 1280, 720)), 150)

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

    def test_widest_zoom_out_is_140_unless_raised_and_is_remembered(self):
        a = self.app("fovmax_default")
        a.set_crop(fov=170.0)
        self.assertEqual(a.fov, 140.0)                                   # unchanged default
        b = self.app("fovmax_wide", fov_max=170.0)
        b.set_crop(fov=165.0)
        self.assertEqual(b.fov, 165.0)
        b.set_crop(fov=400.0)
        self.assertEqual(b.fov, 170.0)
        self.assertEqual(b.render().shape, (b.size[1], b.size[0], 3))
        b.goto(10); b.hold = True; b.handle_key(ord("k"))
        b.save(force=True)
        self.assertEqual(K.load_meta(b.kf_path)["fov_max"], 170.0)
        c = A.Annotator(self.video, b.kf_path, disp_w=640)               # reopened without the flag: remembered
        self.assertEqual(c.fov_max, 170.0)
        d = A.Annotator(self.video, b.kf_path, disp_w=640, fov_max=999)  # absurd values are capped
        self.assertEqual(d.fov_max, K.FOV_HARD_MAX)

    def test_cylindrical_projection_persists_and_renders(self):
        a = self.app("cyl", projection="cylindrical")
        a.goto(20); a.hold = True; a.handle_key(ord("k"))
        a.set_crop(fov=120.0)
        self.assertEqual(a.render().shape, (a.size[1], a.size[0], 3))
        a.save(force=True)
        self.assertEqual(K.load_meta(a.kf_path)["projection"], "cylindrical")
        b = A.Annotator(self.video, a.kf_path, disp_w=640)             # picked up from the sidecar
        self.assertEqual(b.projection, "cylindrical")
        c = A.Annotator(self.video, a.kf_path, disp_w=640, projection="flat")   # CLI overrides
        self.assertEqual(c.projection, "flat")
        # clicking outside the crop recentres it there, in the cylindrical view too
        b.goto(30)
        b.set_crop(fov=40.0)
        target_lon = K.centre_lonlat(b.pan, b.tilt)[0] + 60.0
        x, y = K.lonlat_to_xy(target_lon, b.tilt, b.disp_w, b.eq_h)
        b.on_mouse(cv2.EVENT_LBUTTONDOWN, int(round(x)), int(round(y)), 0)
        b.on_mouse(cv2.EVENT_LBUTTONUP, int(round(x)), int(round(y)), 0)
        self.assertAlmostEqual(b.pan, K.PAN_SIGN * float(K.wrap180(target_lon)), delta=1.0)

    def test_game_start_end_marks(self):
        a = self.app("trim")
        a.goto(30); a.handle_key(ord("["))
        a.goto(90); a.handle_key(ord("]"))
        self.assertAlmostEqual(a.trim_in, 30 / self.FPS, places=3)
        self.assertAlmostEqual(a.trim_out, 90 / self.FPS, places=3)
        a.goto(10); a.handle_key(ord("]"))                         # OUT before IN is refused
        self.assertAlmostEqual(a.trim_out, 90 / self.FPS, places=3)
        a.goto(90); a.handle_key(ord("]"))                         # same key on the mark clears it
        self.assertIsNone(a.trim_out)
        a.goto(100); a.handle_key(ord("]"))
        self.assertEqual(a.render().shape, (a.size[1], a.size[0], 3))   # timeline with marks draws
        a.save(force=True)
        meta = K.load_meta(a.kf_path)
        self.assertAlmostEqual(meta["trim_in"], 1.0, places=3)
        self.assertAlmostEqual(meta["trim_out"], 100 / self.FPS, places=3)
        b = A.Annotator(self.video, a.kf_path, disp_w=640)                 # reopened
        self.assertAlmostEqual(b.trim_in, 1.0, places=3)
        self.assertAlmostEqual(b.trim_out, 100 / self.FPS, places=3)

    def test_new_projects_start_cylindrical_old_ones_stay_flat(self):
        a = self.app("newproj")
        self.assertEqual(a.projection, "cylindrical")
        kfp = os.path.join(self.tmp, "legacy.keyframes.json")
        K.save_meta(kfp, tilt_deg=-5.0, interp="pchip")            # sidecar from before projections existed
        b = A.Annotator(self.video, kfp, disp_w=640)
        self.assertEqual(b.projection, "flat")

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


# ─────────────────────────────────────────────────────────────────────────────
class _Box:
    def __init__(self, cls, xyxy, conf):
        self.cls, self.xyxy, self.conf = [cls], [np.array(xyxy, float)], [conf]


class _Res:
    def __init__(self, boxes):
        self.boxes = boxes


class BlobModel:
    """Stand-in for a YOLO model: reports every bright blob in the picture as a box."""
    names = {0: "person"}

    def __init__(self, min_v=200):
        self.min_v = min_v

    def __call__(self, img, imgsz=None, conf=0.0, device=None, verbose=False):
        g = img.min(axis=2)
        n, _, st, _ = cv2.connectedComponentsWithStats((g > self.min_v).astype(np.uint8))
        boxes = [_Box(0, (x, y, x + w, y + h), 0.9) for x, y, w, h, a in st[1:] if a > 30]
        return [_Res(boxes)]


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class TestExtractDetections(unittest.TestCase):
    """Geometry of the detections: a marker at a known longitude/latitude must come back there,
    from whichever fixed view sees it, exactly once."""

    MARKS = [(-35.0, -8.0), (12.0, -12.0), (20.0, -10.0), (55.0, -6.0)]   # (20, -10) sits near a view boundary

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="kfx_")
        img = np.zeros((X.EQ_H, X.EQ_W, 3), np.uint8)
        for lon, lat in cls.MARKS:
            x, y = K.lonlat_to_xy(lon, lat, X.EQ_W, X.EQ_H)
            cv2.circle(img, (int(round(x)), int(round(y))), 16, (255, 255, 255), -1)
        cls.video = os.path.join(cls.tmp, "marks.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                        f"{X.EQ_W}x{X.EQ_H}", "-r", "10", "-i", "-", "-frames:v", "20", "-c:v", "libx264",
                        "-crf", "12", "-pix_fmt", "yuv444p", cls.video],
                       input=b"".join(img.tobytes() for _ in range(20)), check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_each_marker_found_once_at_the_right_lon_lat(self):
        out = os.path.join(self.tmp, "d.jsonl")
        rc = X.main(["--video", self.video, "--out", out, "--fps", "2", "--tilt", "-10",
                     "--lon-range", "-80", "80", "--duration", "1.5"],
                    player_model=BlobModel(), ball_model=BlobModel())
        self.assertEqual(rc, 0)
        with open(out) as fh:
            lines = [json.loads(l) for l in fh]
        self.assertEqual(lines[0]["type"], "header")
        frames = lines[1:]
        self.assertGreaterEqual(len(frames), 3)
        for fr in frames:
            self.assertEqual(len(fr["players"]), len(self.MARKS), fr["players"])    # none missed or doubled
            self.assertEqual(len(fr["balls"]), len(self.MARKS))
            # players are located by their feet (bottom of the box): a 16 px (2 deg) disc has them ~2 deg lower
            found = sorted((p[0], p[1]) for p in fr["players"])
            for (elon, elat), (lon, lat) in zip(sorted(self.MARKS), found):
                self.assertLess(abs(lon - elon), 0.6, (elon, lon))
                self.assertLess(abs(lat - (elat - 2.0)), 0.9, (elat, lat))
            # balls are located by their centre
            for (elon, elat), (lon, lat, _c) in zip(sorted(self.MARKS), sorted(fr["balls"])):
                self.assertLess(abs(lon - elon), 0.6, (elon, lon))
                self.assertLess(abs(lat - elat), 0.8, (elat, lat))

    def test_multi_class_ball_model_only_counts_ball_classes(self):
        class Multi(BlobModel):
            names = {0: "person", 32: "sports ball"}

            def __call__(self, img, **kw):
                res = BlobModel.__call__(self, img, **kw)[0]
                for i, b in enumerate(res.boxes):
                    b.cls = [32 if i % 2 else 0]         # alternate: person, ball, person, ball
                return [res]
        m = Multi()
        ids = X.ball_class_ids(m)
        self.assertEqual(ids, {32})
        self.assertIsNone(X.ball_class_ids(BlobModel()))             # one-class model: everything
        self.assertEqual(X.ball_class_ids(m, "person"), {0})
        img = np.zeros((720, 1280, 3), np.uint8)
        for x in (100, 300, 500, 700):
            cv2.circle(img, (x, 300), 14, (255, 255, 255), -1)
        self.assertEqual(len(X.detect_balls(img, m, "cpu", class_ids=ids)), 2)
        self.assertEqual(len(X.detect_balls(img, m, "cpu", class_ids=None)), 4)

    def test_range_comes_from_the_annotated_stretch(self):
        kf = os.path.join(self.tmp, "k.json")
        with open(kf, "w") as fh:
            json.dump([{"t": 0.4, "pan_deg": 0, "fov_deg": 80}, {"t": 0.8, "pan_deg": 5, "fov_deg": 80}], fh)
        out = os.path.join(self.tmp, "d2.jsonl")
        X.main(["--video", self.video, "--out", out, "--fps", "5", "--keyframes", kf, "--margin", "0.2",
                "--tilt", "-10"], player_model=BlobModel(), ball_model=None)
        lines = [json.loads(l) for l in open(out)]
        ts = [l["t"] for l in lines[1:]]
        self.assertAlmostEqual(lines[0]["start"], 0.2, places=3)
        self.assertTrue(all(0.19 <= t <= 1.05 for t in ts), ts)
        self.assertTrue(all(fr["balls"] == [] for fr in lines[1:]))

    def test_game_marks_win_over_keyframes_and_reach_the_automatic_output(self):
        kf = os.path.join(self.tmp, "m.keyframes.json")                      # marks only, no keyframes yet
        with open(kf, "w") as fh:
            json.dump([], fh)
        K.save_meta(kf, tilt_deg=-10.0, trim_in=0.5, trim_out=1.5)
        out = os.path.join(self.tmp, "d3.jsonl")
        X.main(["--video", self.video, "--out", out, "--fps", "5", "--keyframes", kf, "--tilt", "-10"],
               player_model=BlobModel(), ball_model=None)
        lines = [json.loads(l) for l in open(out)]
        self.assertAlmostEqual(lines[0]["start"], 0.5, places=3)
        self.assertAlmostEqual(lines[0]["duration"], 1.0, places=3)
        self.assertIn("video_duration", lines[0])
        with open(kf, "w") as fh:                                              # marks AND keyframes: marks win
            json.dump([{"t": 0.1, "pan_deg": 0, "fov_deg": 80}, {"t": 1.9, "pan_deg": 5, "fov_deg": 80}], fh)
        out2 = os.path.join(self.tmp, "d4.jsonl")
        X.main(["--video", self.video, "--out", out2, "--fps", "5", "--keyframes", kf, "--tilt", "-10"],
               player_model=BlobModel(), ball_model=None)
        h = json.loads(open(out2).readline())
        self.assertAlmostEqual(h["start"], 0.5, places=3)
        # the automatic pass hands the marks on to the keyframes it writes (only for a partial extraction)
        t_in, t_out = AK.game_marks({"start": 60.0, "duration": 1380.0, "video_duration": 1500.0})
        self.assertEqual((t_in, t_out), (60.0, 1440.0))
        self.assertEqual(AK.game_marks({"start": 0.0, "duration": 1500.0, "video_duration": 1500.0}), (None, None))
        self.assertEqual(AK.game_marks({"fps": 3.0}), (None, None))             # older detections files: nothing

    def test_pitch_flag_uses_the_calibration_polygon(self):
        poly = [[1700, 700], [2300, 700], [2300, 800], [1700, 800]]          # lon about -29..+13 at the equator band
        cal = os.path.join(self.tmp, "cal.json")
        json.dump({"pitch_polygon": poly}, open(cal, "w"))
        out = os.path.join(self.tmp, "d3.jsonl")
        X.main(["--video", self.video, "--out", out, "--fps", "2", "--calib", cal, "--tilt", "-10",
                "--lon-range", "-80", "80", "--duration", "1"], player_model=BlobModel(), ball_model=None)
        fr = [json.loads(l) for l in open(out)][1]
        flags = {round(p[0]): p[4] for p in fr["players"]}
        self.assertTrue(flags[-35] in (False, None) or flags[-35] is False)
        self.assertTrue(any(v is True for v in flags.values()))
        self.assertTrue(any(v is False for v in flags.values()))


# ─────────────────────────────────────────────────────────────────────────────
class TestAutoKeyframes(unittest.TestCase):
    """The automatic pass, on synthetic detections where the right answer is known."""

    @staticmethod
    def make(centre_fn, n_players=14, seconds=120, rate=3.0, seed=1, floodlight=True, drop=()):
        rng = np.random.default_rng(seed)
        header = {"type": "header", "fps": rate, "tilt": -20.0, "video": "x.mp4"}
        frames = []
        offs = rng.uniform(-25, 25, n_players)
        offs = offs - offs.mean()                                        # the group's centre is exactly centre_fn(t)
        for i in range(int(seconds * rate)):
            t = i / rate
            c = centre_fn(t)
            pl = [] if any(a <= t < b for a, b in drop) else \
                [[float(c + o + rng.normal(0, 1.5)), -20.0 + float(rng.normal(0, 4)), 30.0, 0.6, None] for o in offs]
            balls = [[42.0, -2.0, 0.8]] if floodlight else []
            frames.append({"t": t, "players": pl, "balls": balls})
        return header, frames

    def test_follows_the_action_with_the_right_sign_and_no_lag(self):
        f = lambda t: 30 * np.sin(2 * np.pi * t / 60)                    # slow sweep, lon degrees        # noqa: E731
        header, frames = self.make(f)
        r = AK.build_path(header, frames, deadband=0.0, smooth_s=4.0)
        err = r["pan"] - K.PAN_SIGN * np.array([f(t) for t in r["t"]])
        self.assertLess(np.abs(err[10:-10]).mean(), 2.5)                 # tracks it
        self.assertLess(abs(err[10:-10].mean()), 1.5)                    # centred: no lag-induced bias
        self.assertGreater(np.corrcoef(r["pan"], [-f(t) for t in r["t"]])[0, 1], 0.99)   # pan = -longitude

    def test_look_ahead_has_no_delay(self):
        f = lambda t: 0.0 if t < 60 else 25.0                            # play jumps at t = 60      # noqa: E731
        header, frames = self.make(f)
        r = AK.build_path(header, frames, deadband=0.0, smooth_s=4.0)
        i_mid = int(np.argmin(np.abs(r["t"] - 60)))
        self.assertAlmostEqual(r["pan"][i_mid], -12.5, delta=3.0)        # half way exactly at the jump
        self.assertLess(r["pan"][int(np.argmin(np.abs(r["t"] - 59)))], -1.0)   # already moving a second before it

    def test_deadband_keeps_the_camera_still_for_small_drifts(self):
        f = lambda t: 3.0 * np.sin(2 * np.pi * t / 7)                    # jitter of +-3 deg        # noqa: E731
        header, frames = self.make(f, seconds=90)
        r = AK.build_path(header, frames, deadband=6.0, smooth_s=3.0)
        self.assertLess(np.ptp(r["pan"][20:]), 1.5)

    def test_floodlight_is_not_a_ball_and_real_ball_is_used_on_request(self):
        f = lambda t: 0.0                                                # noqa: E731
        header, frames = self.make(f, seconds=60)
        for i, fr in enumerate(frames):                                  # a real, moving ball to the right of the group
            fr["balls"].append([10.0 + (i * 0.37) % 30.0, -20.0, 0.9])
        off = AK.build_path(header, frames, deadband=0.0, use_ball=False)
        on = AK.build_path(header, frames, deadband=0.0, use_ball=True)
        self.assertGreaterEqual(off["static_ball_cells"], 1)
        self.assertLess(abs(off["pan"][30:-30].mean()), 1.0)             # floodlight at +42 never pulls
        self.assertLess(on["pan"][30:-30].mean(), -6.0)                  # the ball pulls the crop right (pan < 0)
        self.assertGreater(on["pan"][30:-30].mean(), -14.0)              # but only part of the way (weight 0.4)

    def test_gaps_without_players_are_bridged_and_long_gaps_hold(self):
        f = lambda t: 20.0 * (t > 40)                                    # noqa: E731
        header, frames = self.make(f, seconds=100, drop=[(20, 24), (50, 70)])
        r = AK.build_path(header, frames, deadband=0.0, max_gap_s=8.0)
        self.assertFalse(np.isnan(r["pan"]).any())
        self.assertFalse(np.isnan(r["fov"]).any())

    def test_never_shows_beyond_the_lens_coverage(self):
        f = lambda t: 85.0                                               # action far to the side       # noqa: E731
        header, frames = self.make(f, seconds=60)
        r = AK.build_path(header, frames, edge_limit=100.0)
        centre = K.PAN_SIGN * r["pan"]
        self.assertTrue(np.all(np.abs(centre) + r["fov"] / 2 <= 100.0 + 1e-6))

    def test_zoom_widens_when_the_players_spread_out(self):
        rng = np.random.default_rng(3)
        header = {"type": "header", "fps": 3.0}
        frames = []
        for i in range(600):
            t = i / 3
            half = 15 if t < 100 else 45                                 # compact first, then spread
            x = rng.uniform(-half, half, 14)
            frames.append({"t": t, "players": [[float(v), -20.0, 30.0, 0.6, None] for v in x], "balls": []})
        r = AK.build_path(header, frames)
        self.assertGreater(r["fov"][-5], r["fov"][5] + 5.0)
        self.assertTrue(np.all((r["fov"] >= 90.0 - 1e-9) & (r["fov"] <= 140.0 + 1e-9)))
        fixed = AK.build_path(header, frames, fov_fixed=110.0)
        self.assertTrue(np.allclose(fixed["fov"], 110.0))

    def test_simplified_keyframes_reproduce_the_path_and_load_in_the_tools(self):
        f = lambda t: 30 * np.sin(2 * np.pi * t / 50)                    # noqa: E731
        header, frames = self.make(f, seconds=150)
        r = AK.build_path(header, frames)
        idx = AK.simplify(r["t"], r["pan"], r["fov"], 1.5, 3.0, 1.0)
        self.assertLess(len(idx), len(r["t"]) // 6)                      # far fewer than samples
        self.assertEqual((idx[0], idx[-1]), (0, len(r["t"]) - 1))
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "a.keyframes.json")
            AK.write_keyframes(out, r["t"], r["pan"], r["fov"], idx, -20.0, "cylindrical")
            kfs = K.KeyframeSet.load(out)                                # valid spec: t, pan_deg, fov_deg
            self.assertEqual(len(kfs), len(idx))
            meta = K.load_meta(out)
            self.assertEqual((meta["projection"], meta["tilt_deg"]), ("cylindrical", -20.0))
            p, fv = K.Path(kfs.items, "pchip").at(r["t"])
            self.assertLess(np.abs(p - r["pan"]).max(), 1.6)
            self.assertLess(np.abs(fv - r["fov"]).max(), 3.2)
            c = AK.compare(r["t"], r["pan"], r["fov"], out, r["lon_raw"])
            self.assertLess(c["pan_mae"], 0.6)

    def test_command_line_end_to_end(self):
        header, frames = self.make(lambda t: 20 * np.sin(t / 8), seconds=60)
        with tempfile.TemporaryDirectory() as d:
            det = os.path.join(d, "g.detections.jsonl")
            with open(det, "w") as fh:
                fh.write(json.dumps(header) + "\n")
                for fr in frames:
                    fh.write(json.dumps(fr) + "\n")
            out = os.path.join(d, "g.auto.keyframes.json")
            self.assertEqual(AK.main(["--detections", det, "--out", out]), 0)
            self.assertGreater(len(K.KeyframeSet.load(out)), 3)
            self.assertEqual(AK.main(["--detections", det, "--out", out + "2", "--compare", out]), 0)


class TestAutoKeyframesMarks(unittest.TestCase):
    def test_main_writes_game_marks_into_the_meta(self):
        header, frames = TestAutoKeyframes.make(lambda t: 10.0, seconds=60)
        header.update({"start": 30.0, "duration": 60.0, "video_duration": 200.0})
        for fr in frames:
            fr["t"] += 30.0
        with tempfile.TemporaryDirectory() as d:
            det = os.path.join(d, "g.detections.jsonl")
            with open(det, "w") as fh:
                fh.write(json.dumps(header) + "\n")
                for fr in frames:
                    fh.write(json.dumps(fr) + "\n")
            out = os.path.join(d, "g.json")
            self.assertEqual(AK.main(["--detections", det, "--out", out]), 0)
            meta = K.load_meta(out)
            self.assertEqual((meta["trim_in"], meta["trim_out"]), (30.0, 90.0))


class TestKeyframeTilt(unittest.TestCase):
    """Optional per-keyframe tilt: file format, interpolation, FFmpeg render, annotator keys, auto pass."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="kftilt_")
        W, H, fps, n = 1440, 720, 30, 150
        col = np.round(np.arange(H) / (H - 1) * 65535).astype("<u2")           # value encodes latitude (rows)
        frame = np.tile(col[:, None], (1, W))
        cls.src = os.path.join(cls.tmp, "lat16.mkv")
        ffmpeg_write_gray(cls.src, (frame for _ in range(n)), W, H, fps, pix="gray16le")
        cls.H, cls.fps = H, fps

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # -- file format ---------------------------------------------------------
    def test_files_without_tilt_are_unchanged_and_tilt_round_trips(self):
        old = K.KeyframeSet.from_list([{"t": 1, "pan_deg": 2, "fov_deg": 90}])
        self.assertEqual(set(old.to_list()[0]), {"t", "pan_deg", "fov_deg"})       # nothing new in old files
        self.assertIsNone(old.items[0].tilt_deg)
        new = K.KeyframeSet.from_list([{"t": 1, "pan_deg": 2, "fov_deg": 90, "tilt_deg": -31.5},
                                       {"t": 3, "pan_deg": 2, "fov_deg": 90, "tilt_deg": None}])
        self.assertEqual(new.items[0].tilt_deg, -31.5)
        self.assertIsNone(new.items[1].tilt_deg)
        d = os.path.join(self.tmp, "rt.json")
        new.save(d)
        self.assertEqual(K.KeyframeSet.load(d).items, new.items)
        with self.assertRaises(ValueError):
            K.KeyframeSet.from_list([{"t": 1, "pan_deg": 2, "fov_deg": 90, "tilt_deg": float("nan")}])

    # -- interpolation -------------------------------------------------------
    def test_tilt_path_dips_and_eases_back_without_overshoot(self):
        kfs = [K.Keyframe(0, 0, 100), K.Keyframe(10, 0, 100, -30.0), K.Keyframe(20, 0, 100), K.Keyframe(30, 0, 100)]
        path = K.Path(kfs, "pchip", base_tilt=-15.0)
        self.assertTrue(path.has_tilt)
        self.assertAlmostEqual(path.tilt_at(0.0), -15.0)
        self.assertAlmostEqual(path.tilt_at(10.0), -30.0)
        self.assertAlmostEqual(path.tilt_at(20.0), -15.0)
        self.assertAlmostEqual(path.tilt_at(99.0), -15.0)                           # holds after the last keyframe
        ts = np.linspace(0, 30, 601)
        tl = path.tilt_at(ts)
        self.assertGreaterEqual(tl.min(), -30.0 - 1e-9)
        self.assertLessEqual(tl.max(), -15.0 + 1e-9)
        plain = K.Path([K.Keyframe(0, 0, 100), K.Keyframe(10, 5, 90)], "pchip", base_tilt=-7.0)
        self.assertFalse(plain.has_tilt)
        self.assertTrue(np.all(plain.tilt_at(ts) == -7.0))
        self.assertEqual(K.Path([], "pchip", base_tilt=-7.0).tilt_at(1.0), -7.0)

    # -- renderer ------------------------------------------------------------
    def test_sendcmd_is_unchanged_without_tilt_and_varies_with_it(self):
        plain = K.Path([K.Keyframe(0, 12, 80), K.Keyframe(10, -12, 100)], "pchip")
        text, n, first = R.build_sendcmd(plain, -15.0, 300, 30.0, 0.0, 1280, 720, False, projection="cylindrical")
        self.assertNotIn("pitch", text)                                              # constant tilt: no pitch commands
        self.assertAlmostEqual(first["pitch"], -15.0)
        tilted = K.Path([K.Keyframe(0, 12, 80), K.Keyframe(5, 12, 80, -30.0), K.Keyframe(10, 12, 80)],
                        "pchip", base_tilt=-15.0)
        text, n, first = R.build_sendcmd(tilted, -15.0, 300, 30.0, 0.0, 1280, 720, False, projection="cylindrical")
        self.assertIn("pitch", text)
        vals = [float(l.split("pitch")[1].split(",")[0].split(";")[0]) for l in text.splitlines() if "pitch" in l]
        self.assertAlmostEqual(min(vals), -30.0, delta=0.2)

    def test_render_follows_per_keyframe_tilt_frame_by_frame(self):
        kf = os.path.join(self.tmp, "t.keyframes.json")
        with open(kf, "w") as fh:
            json.dump([{"t": 0.5, "pan_deg": 0, "fov_deg": 90},
                       {"t": 2.5, "pan_deg": 0, "fov_deg": 90, "tilt_deg": -35.0},
                       {"t": 4.5, "pan_deg": 0, "fov_deg": 90}], fh)
        K.save_meta(kf, tilt_deg=-10.0, interp="pchip", legacy_fov=False, projection="cylindrical")
        out = os.path.join(self.tmp, "t.mkv")
        rc = R.main(["--input", self.src, "--keyframes", kf, "--output", out, "--encoder", "ffv1",
                     "--interp-filter", "nearest"])
        self.assertEqual(rc, 0)
        frames = decode_gray16(out, 1280, 720)
        self.assertEqual(len(frames), 150)
        path = K.Path(K.KeyframeSet.load(kf).items, "pchip", base_tilt=-10.0)
        H = self.H
        lat = lambda v: (0.5 - (v / 65535 * (H - 1) + 0.5) / H) * 180.0              # noqa: E731
        worst = 0.0
        for i, f in enumerate(frames):
            c = 0.5 * (f[359, 640].astype(np.float64) + f[360, 640].astype(np.float64))
            worst = max(worst, abs(lat(c) - path.tilt_at(i / self.fps)))
        self.assertLess(worst, 0.7, f"tilt error {worst:.3f} deg")
        self.assertAlmostEqual(path.tilt_at(2.5), -35.0)

    # -- annotator -----------------------------------------------------------
    def test_annotator_tilts_one_keyframe_and_leaves_the_project_tilt_alone(self):
        d = tempfile.mkdtemp(dir=self.tmp)
        video = os.path.join(d, "v.mp4")
        ffmpeg_write_gray(video, [np.full((320, 640), 2 * i + 4, np.uint8) for i in range(120)], 640, 320, 30,
                          pix="gray", codec=("-c:v", "libx264", "-preset", "veryfast", "-crf", "10", "-g", "10",
                                             "-bf", "0", "-pix_fmt", "yuv420p"))
        a = A.Annotator(video, os.path.join(d, "v.keyframes.json"), tilt=-12.0, disp_w=640)
        for idx in (0, 60, 119):
            a.goto(idx); a.hold = True; a.handle_key(ord("k"))
        self.assertTrue(all(k.tilt_deg is None for k in a.kfs))                     # no tilt written by default
        a.goto(60)
        for _ in range(8):
            a.handle_key(ord("b"))                                                  # tilt this keyframe down 4 deg
        self.assertAlmostEqual(a.tilt, -16.0)
        self.assertAlmostEqual(a.base_tilt, -12.0)
        self.assertAlmostEqual(a.kfs.at(60 / 30).tilt_deg, -16.0)
        self.assertIsNone(a.kfs.at(0).tilt_deg)
        a.goto(30)                                                                   # half way: eased, between both
        self.assertTrue(-16.0 < a.tilt < -12.0)
        a.goto(119)
        self.assertAlmostEqual(a.tilt, -12.0)
        a.goto(60)
        a.handle_key(ord("x"))                                                       # clear it
        self.assertIsNone(a.kfs.at(60 / 30).tilt_deg)
        for _ in range(2):
            a.handle_key(ord("t"))
        self.assertAlmostEqual(a.kfs.at(60 / 30).tilt_deg, -11.0)
        a.handle_key(ord("w"))                                                       # W/S still change the project tilt
        self.assertAlmostEqual(a.base_tilt, -11.5)
        self.assertAlmostEqual(a.kfs.at(60 / 30).tilt_deg, -11.0)                    # an explicit keyframe tilt stays
        self.assertEqual(a.render().shape, (a.size[1], a.size[0], 3))
        a.save(force=True)
        self.assertEqual(K.load_meta(a.kf_path)["tilt_deg"], -11.5)
        b = A.Annotator(video, a.kf_path, disp_w=640)
        self.assertAlmostEqual(b.base_tilt, -11.5)
        b.goto(60)
        self.assertAlmostEqual(b.tilt, -11.0)

    def test_old_project_still_behaves_as_before_in_the_annotator(self):
        d = tempfile.mkdtemp(dir=self.tmp)
        video = os.path.join(d, "v.mp4")
        ffmpeg_write_gray(video, [np.full((320, 640), 2 * i + 4, np.uint8) for i in range(60)], 640, 320, 30,
                          pix="gray", codec=("-c:v", "libx264", "-preset", "veryfast", "-crf", "10", "-g", "10",
                                             "-bf", "0", "-pix_fmt", "yuv420p"))
        kfp = os.path.join(d, "v.keyframes.json")
        with open(kfp, "w") as fh:
            json.dump([{"t": 0.5, "pan_deg": 3, "fov_deg": 90}, {"t": 1.5, "pan_deg": -3, "fov_deg": 100}], fh)
        K.save_meta(kfp, tilt_deg=-9.0, interp="pchip", legacy_fov=False, projection="cylindrical")
        a = A.Annotator(video, kfp, disp_w=640)
        for idx in (0, 20, 40):
            a.goto(idx)
            self.assertAlmostEqual(a.tilt, -9.0)
        a.save(force=True)
        with open(kfp) as fh:
            rows = json.load(fh)
        self.assertTrue(all(set(r) == {"t", "pan_deg", "fov_deg"} for r in rows))

    # -- automatic pass ------------------------------------------------------
    @staticmethod
    def _detections(near_feet_lat, seconds=90, rate=3.0, near_from=30, near_to=60):
        """Players around lat -20, plus (between near_from and near_to) one player close to the camera."""
        rng = np.random.default_rng(5)
        frames = []
        for i in range(int(seconds * rate)):
            t = i / rate
            pl = [[float(o), -20.0 + float(rng.normal(0, 1)), 30.0, 0.7, None] for o in np.linspace(-20, 20, 10)]
            if near_from <= t < near_to:
                pl.append([0.0, near_feet_lat, 120.0, 0.9, None])
            frames.append({"t": t, "players": pl, "balls": []})
        return {"type": "header", "fps": rate, "tilt": -15.0, "video": "x.mp4"}, frames

    def test_adaptive_tilt_dips_only_while_the_near_player_would_be_cut_off(self):
        header, frames = self._detections(near_feet_lat=-50.0)
        flat = AK.build_path(header, frames, fov_fixed=110, deadband=0.0)
        self.assertIsNone(flat["tilt"])                                              # off by default
        r = AK.build_path(header, frames, fov_fixed=110, deadband=0.0, adaptive_tilt_on=True, base_tilt=-15.0)
        t, tilt = r["t"], r["tilt"]
        self.assertAlmostEqual(float(tilt[t < 15].mean()), -15.0, delta=0.3)          # before: untouched
        self.assertAlmostEqual(float(tilt[t > 75].mean()), -15.0, delta=0.3)          # after: back to base
        self.assertLess(float(tilt[(t > 38) & (t < 52)].mean()), -20.0)               # dips while he is there
        self.assertGreaterEqual(float(tilt.min()), -15.0 - 20.0 - 1e-9)               # never beyond max_dip
        self.assertLessEqual(float(tilt.max()), -15.0 + 1e-9)                         # never tilts above base
        c0 = AK.feet_coverage(frames, t, r["pan"], r["fov"], np.full(len(t), -15.0), "cylindrical")
        c1 = AK.feet_coverage(frames, t, r["pan"], r["fov"], tilt, "cylindrical")
        self.assertGreater(c1, c0 + 0.01)
        self.assertGreater(c1, 0.98)

    def test_adaptive_tilt_main_writes_a_file_the_renderer_and_annotator_read(self):
        header, frames = self._detections(near_feet_lat=-50.0)
        d = tempfile.mkdtemp(dir=self.tmp)
        det = os.path.join(d, "g.detections.jsonl")
        with open(det, "w") as fh:
            fh.write(json.dumps(header) + "\n")
            for fr in frames:
                fh.write(json.dumps(fr) + "\n")
        plain, adapt = os.path.join(d, "plain.json"), os.path.join(d, "adapt.json")
        self.assertEqual(AK.main(["--detections", det, "--out", plain, "--fov", "110"]), 0)
        self.assertEqual(AK.main(["--detections", det, "--out", adapt, "--fov", "110", "--adaptive-tilt"]), 0)
        with open(plain) as fh:
            self.assertTrue(all("tilt_deg" not in r for r in json.load(fh)))          # default output has no tilt
        kfs = K.KeyframeSet.load(adapt)
        self.assertTrue(any(k.tilt_deg is not None and k.tilt_deg < -20 for k in kfs))
        self.assertEqual(K.load_meta(adapt)["tilt_deg"], -15.0)
        path = K.Path(kfs.items, "pchip", base_tilt=-15.0)
        self.assertTrue(path.has_tilt)
        self.assertAlmostEqual(path.tilt_at(5.0), -15.0, delta=0.3)
        self.assertLess(path.tilt_at(45.0), -20.0)


class TestLeadAndProfile(unittest.TestCase):
    def test_lead_shift(self):
        y = np.arange(5.0)
        self.assertEqual(list(AK.lead_shift(y, 2)), [2, 3, 4, 4, 4])
        self.assertEqual(list(AK.lead_shift(y, -2)), [0, 0, 0, 1, 2])
        self.assertEqual(list(AK.lead_shift(y, 0)), list(y))
        self.assertEqual(len(AK.lead_shift(y, 99)), 5)

    def test_lead_makes_the_camera_arrive_before_the_play(self):
        f = lambda t: 0.0 if t < 60 else 25.0                            # noqa: E731
        header, frames = TestAutoKeyframes.make(f)
        base = AK.build_path(header, frames, deadband=0.0, smooth_s=4.0)
        led = AK.build_path(header, frames, deadband=0.0, smooth_s=4.0, lead_s=2.0)
        again = AK.build_path(header, frames, deadband=0.0, smooth_s=4.0, lead_s=0.0)
        self.assertTrue(np.array_equal(base["pan"], again["pan"]))        # lead 0 changes nothing
        i = int(np.argmin(np.abs(base["t"] - 58)))
        self.assertLess(led["pan"][i], base["pan"][i] - 3.0)              # already well on its way, ahead of the jump

    def test_profile_is_loaded_and_command_line_flags_win(self):
        header, frames = TestAutoKeyframes.make(lambda t: 10.0, seconds=60)
        with tempfile.TemporaryDirectory() as d:
            det = os.path.join(d, "g.detections.jsonl")
            with open(det, "w") as fh:
                fh.write(json.dumps(header) + "\n")
                for fr in frames:
                    fh.write(json.dumps(fr) + "\n")
            prof = os.path.join(d, "p.json")
            with open(prof, "w") as fh:
                json.dump({"settings": {"smooth": 9.0, "lead": 1.0, "tilt": -12.0, "bogus": 1}}, fh)
            a, b = os.path.join(d, "a.json"), os.path.join(d, "b.json")
            self.assertEqual(AK.main(["--detections", det, "--out", a, "--profile", prof]), 0)
            self.assertEqual(K.load_meta(a)["tilt_deg"], -12.0)            # the profile's tilt is used ...
            self.assertEqual(AK.main(["--detections", det, "--out", b, "--profile", prof, "--tilt", "-5"]), 0)
            self.assertEqual(K.load_meta(b)["tilt_deg"], -5.0)             # ... unless a flag says otherwise
            with self.assertRaises(SystemExit):
                AK.main(["--detections", det, "--out", a, "--profile", os.path.join(d, "missing.json")])


class TestTuner(unittest.TestCase):
    """The tuner must recover settings from a reference that was made with known ones."""

    @staticmethod
    def _case(d, seconds, truth, tilt_truth=None, near=False, seed=7):
        centre = lambda t: 25 * np.sin(2 * np.pi * t / 70) + 10 * np.sin(2 * np.pi * t / 23)    # noqa: E731
        header, frames = TestAutoKeyframes.make(centre, seconds=seconds, seed=seed)
        for fr in frames:
            k = 0.6 + 0.8 * (np.sin(fr["t"] / 40) + 1) / 2
            c = centre(fr["t"])
            fr["players"] = [[c + (p[0] - c) * k, p[1], p[2], p[3], p[4]] for p in fr["players"]]
            if near and seconds * 0.3 <= fr["t"] < seconds * 0.6:
                fr["players"].append([c, -55.0, 120.0, 0.9, None])
        header["tilt"] = -10.0
        det = os.path.join(d, f"g{seed}.detections.jsonl")
        with open(det, "w") as fh:
            fh.write(json.dumps(header) + "\n")
            for f in frames:
                fh.write(json.dumps(f) + "\n")
        kw = dict(deadband=truth["deadband"], smooth_s=truth["smooth"], zoom_s=truth["zoom_smooth"],
                  fov_a=truth["fov_a"], fov_b=truth["fov_b"], lead_s=truth["lead"])
        if tilt_truth:
            kw.update(adaptive_tilt_on=True, base_tilt=-10.0, tilt_margin=tilt_truth["tilt_margin"],
                      max_dip=tilt_truth["max_dip"])
        r = AK.build_path(header, frames, **kw)
        idx = AK.simplify(r["t"], r["pan"], r["fov"], 0.3, 0.6, 0.5, tilt=r["tilt"], tol_tilt=0.5, base_tilt=-10.0)
        ref = os.path.join(d, f"ref{seed}.keyframes.json")
        AK.write_keyframes(ref, r["t"], r["pan"], r["fov"], idx, -10.0, "cylindrical", tilt_path=r["tilt"])
        return det, ref

    TRUTH = dict(deadband=2.0, smooth=6.0, zoom_smooth=8.0, fov_a=0.5, fov_b=80.0, lead=2.0)

    def test_recovers_pan_and_zoom_settings_and_beats_the_defaults_on_held_out_data(self):
        with tempfile.TemporaryDirectory() as d:
            det, ref = self._case(d, 150, self.TRUTH)
            out = os.path.join(d, "p.json")
            self.assertEqual(TU.main(["--detections", det, "--reference", ref, "--out", out]), 0)
            j = json.load(open(out))
            s, rep = j["settings"], j["report"]
            self.assertFalse(rep["tilt_fitted"])
            self.assertNotIn("adaptive_tilt", s)                          # no tilt in the reference -> no tilt fitted
            self.assertLess(rep["tuned"]["pan"], 0.5 * rep["defaults"]["pan"])
            self.assertLess(rep["heldout_tuned"]["pan"], 0.6 * rep["heldout_defaults"]["pan"])
            self.assertGreaterEqual(s["lead"], 1.0)                       # it found that the operator leads the play
            self.assertAlmostEqual(s["fov_a"], 0.5, delta=0.15)
            # the profile it wrote is accepted by auto_keyframes and gives a path close to the reference
            outk = os.path.join(d, "auto.json")
            self.assertEqual(AK.main(["--detections", det, "--profile", out, "--out", outk, "--compare", ref]), 0)

    def test_fits_tilt_only_when_the_reference_uses_it(self):
        with tempfile.TemporaryDirectory() as d:
            det, ref = self._case(d, 120, self.TRUTH, tilt_truth=dict(tilt_margin=6, max_dip=30), near=True, seed=11)
            self.assertTrue(any(k.tilt_deg is not None for k in K.KeyframeSet.load(ref)))
            out = os.path.join(d, "p.json")
            self.assertEqual(TU.main(["--detections", det, "--reference", ref, "--out", out]), 0)
            j = json.load(open(out))
            self.assertTrue(j["report"]["tilt_fitted"])
            self.assertTrue(j["settings"]["adaptive_tilt"])
            self.assertEqual(j["settings"]["tilt"], -10.0)
            self.assertLess(j["report"]["tuned"]["tilt"], j["report"]["defaults"]["tilt"])
            self.assertGreaterEqual(j["settings"]["max_dip"], 30)         # needs the deeper dip the operator used

    def test_two_games_can_be_fitted_together_and_a_mismatched_pair_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d1, r1 = self._case(d, 90, self.TRUTH, seed=3)
            d2, r2 = self._case(d, 90, self.TRUTH, seed=4)
            out = os.path.join(d, "p.json")
            self.assertEqual(TU.main(["--detections", d1, d2, "--reference", r1, r2, "--out", out]), 0)
            with open(out) as fh:
                samples = json.load(fh)["report"]["samples"]
            self.assertEqual(samples, 2 * len(
                [1 for r in AK.load_detections(d1)[1] if r["t"] >= K.KeyframeSet.load(r1).items[0].t
                 and r["t"] <= K.KeyframeSet.load(r1).items[-1].t]))
            with self.assertRaises(SystemExit):
                TU.main(["--detections", d1, d2, "--reference", r1])

    def test_reference_that_does_not_overlap_the_detections_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            det, _ = self._case(d, 60, self.TRUTH, seed=5)
            far = os.path.join(d, "far.keyframes.json")
            with open(far, "w") as fh:
                json.dump([{"t": 500, "pan_deg": 0, "fov_deg": 90}, {"t": 520, "pan_deg": 5, "fov_deg": 90}], fh)
            with self.assertRaises(SystemExit):
                TU.main(["--detections", det, "--reference", far, "--out", os.path.join(d, "x.json")])


class TestIvFov(unittest.TestCase):
    """The vertical field of the fisheye unwrapping is one setting, carried from extraction to render."""

    ORIGINAL = "v360=fisheye:equirect:ih_fov=200:iv_fov=200,scale=2880:1440"

    def test_default_is_exactly_the_original_filter(self):
        self.assertEqual(K.raw_to_equirect(), self.ORIGINAL)
        self.assertEqual(K.raw_to_equirect(None), self.ORIGINAL)
        self.assertEqual(K.raw_to_equirect(200), self.ORIGINAL)
        self.assertEqual(X.RAW_TO_EQUIRECT, self.ORIGINAL)
        self.assertEqual(R.RAW_TO_EQUIRECT, self.ORIGINAL)

    def test_other_value_changes_only_iv_fov(self):
        self.assertEqual(K.raw_to_equirect(112), "v360=fisheye:equirect:ih_fov=200:iv_fov=112,scale=2880:1440")
        self.assertIn("iv_fov=112.5", K.raw_to_equirect(112.5))

    def test_extraction_filter_follows_the_setting(self):
        import unittest.mock as mock
        seen = []
        with mock.patch.object(X.subprocess, "Popen", lambda cmd, **kw: seen.append(cmd) or None):
            X.open_frames("v.mp4", 0.0, 10.0, 3.0, True)
            X.open_frames("v.mp4", 0.0, 10.0, 3.0, True, 112)
            X.open_frames("v.mp4", 0.0, 10.0, 3.0, False, 112)             # equirect input is never unwrapped
        vf = [c[c.index("-vf") + 1] for c in seen]
        self.assertTrue(vf[0].startswith(self.ORIGINAL))
        self.assertTrue(vf[1].startswith("v360=fisheye:equirect:ih_fov=200:iv_fov=112,"))
        self.assertNotIn("fisheye", vf[2])

    def _dry_run(self, kf, *extra):
        import contextlib
        import io
        raw = os.path.join(os.path.dirname(kf), "raw.mp4")
        if not os.path.exists(raw):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                            "testsrc2=size=1440x2560:rate=30:duration=1", "-pix_fmt", "yuv420p", raw], check=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = R.main(["--input", raw, "--keyframes", kf, "--output", "x.mp4", "--dry-run", *extra])
        self.assertEqual(rc, 0)
        return buf.getvalue()

    def test_render_takes_the_value_from_the_keyframe_file_and_the_flag_wins(self):
        with tempfile.TemporaryDirectory() as d:
            kf = os.path.join(d, "a.keyframes.json")
            with open(kf, "w") as fh:
                json.dump([{"t": 0, "pan_deg": 0, "fov_deg": 90}, {"t": 5, "pan_deg": 10, "fov_deg": 90}], fh)
            self.assertIn(self.ORIGINAL, self._dry_run(kf))                       # no setting anywhere: unchanged
            K.save_meta(kf, tilt_deg=-10.0, projection="flat", iv_fov=112.0)
            txt = self._dry_run(kf)
            self.assertIn("iv_fov=112,scale=2880:1440", txt)
            self.assertNotIn("iv_fov=200", txt)
            self.assertIn("iv_fov=150,scale=2880:1440", self._dry_run(kf, "--iv-fov", "150"))

    def test_views_are_square_pixel_only_for_a_non_default_unwrapping(self):
        old = X.View(0.0, -18.0)                                           # original: vertical field = horizontal field
        sq = X.View(0.0, -18.0, legacy_fov=False)
        span = lambda v: float(v.lat.max() - v.lat.min())                  # noqa: E731
        self.assertAlmostEqual(span(old), 100.0, delta=8.0)
        self.assertAlmostEqual(span(sq), 67.7, delta=3.0)                  # flat 100 deg across 1280 px -> 67.7 deg across 720 px
        self.assertEqual([v.fov for v in X.make_views(-80, 80, -18.0, legacy_fov=False)], [100.0] * 4)
        self.assertAlmostEqual(span(X.make_views(-80, 80, -18.0)[0]), span(old), delta=0.5)   # default call unchanged

    def test_game_stats_speed_ruler_follows_the_view_geometry(self):
        import game_stats as GS
        # one player 50 px tall, 1 m per frame sideways at 3 frames/s = 3 m/s, seen from 1.7 * fy / 50 metres away
        for fy in (GS.VIEW_FY, GS.SQUARE_FY):
            dist = 1.7 * fy / 50.0
            step_deg = math.degrees(1.0 / dist)
            frames = [{"t": i / 3.0, "players": [[10.0 + i * step_deg, -20.0, 50.0, 0.9, None]], "balls": []}
                      for i in range(4)]
            sp = GS.player_speeds(frames, np.array([f["t"] for f in frames]), 0.3, fy)
            self.assertAlmostEqual(float(sp[0][0]), 3.0, delta=0.05)
        self.assertAlmostEqual(GS.SQUARE_FY, 537.0, delta=0.5)

    def test_setting_travels_from_detections_header_to_keyframe_meta(self):
        with tempfile.TemporaryDirectory() as d:
            header, frames = TestAutoKeyframes.make(lambda t: 10.0 * np.sin(t / 9.0), seconds=60)
            for name, extra in (("plain", {}), ("iv", {"iv_fov": 112.0})):
                det = os.path.join(d, name + ".jsonl")
                with open(det, "w") as fh:
                    fh.write(json.dumps({**header, **extra}) + "\n")
                    for r in frames:
                        fh.write(json.dumps(r) + "\n")
                out = os.path.join(d, name + ".keyframes.json")
                self.assertEqual(AK.main(["--detections", det, "--out", out]), 0)
                meta = K.load_meta(out)
                if extra:
                    self.assertEqual(meta["iv_fov"], 112.0)
                else:
                    self.assertNotIn("iv_fov", meta)                               # old files are written exactly as before


class TestSceneCheck(unittest.TestCase):
    """scene_check.py: colour clusters, position estimate, scene picking, and the whole run on a stub detector."""

    def setUp(self):
        import scene_check as S
        self.S = S

    def test_position_follows_the_ruler_and_the_angle(self):
        S = self.S
        x, y = S.ground_position(0.0, 40.0, 0.0, S.SQUARE_FY, 1.0)
        self.assertAlmostEqual(x, 0.0, places=6)
        self.assertAlmostEqual(y, 1.7 * S.SQUARE_FY / 40.0, places=6)
        x2, y2 = S.ground_position(0.0, 20.0, 0.0, S.SQUARE_FY, 1.0)
        self.assertAlmostEqual(y2, 2 * y, places=6)                         # half the height = twice as far
        xs, _ = S.ground_position(30.0, 40.0, 30.0, S.SQUARE_FY, 0.75)
        self.assertGreater(xs, 0)
        self.assertLess(S.ground_position(-30.0, 40.0, -30.0, S.SQUARE_FY, 0.75)[0], 0)

    def test_torso_colour_ignores_grass_and_clusters_separate_teams(self):
        S = self.S
        img = np.full((100, 60, 3), (60, 140, 50), np.uint8)
        cv2.rectangle(img, (15, 15), (45, 50), (40, 40, 220), -1)             # red shirt
        c = S.torso_colour(img, (10, 10, 50, 90))
        self.assertGreater(c[2], 180)
        self.assertLess(c[1], 80)
        rng = np.random.default_rng(0)
        red = np.array([40, 40, 220]) + rng.normal(0, 6, (30, 3))
        blue = np.array([210, 70, 30]) + rng.normal(0, 6, (25, 3))
        cl = S.Clusters(np.vstack([red, blue]), 2, 0)
        self.assertEqual(sorted(cl.sizes.tolist()), [25, 30])
        self.assertEqual(cl.sizes[0], 30)                                     # biggest first
        lab, amb = cl.assign(np.array([[40, 40, 220], [210, 70, 30]]))
        self.assertNotEqual(lab[0], lab[1])
        self.assertTrue((amb < 0.3).all())

    def test_scene_picking_is_seeded_and_spaced(self):
        S = self.S
        frames = [{"t": float(t), "players": [[0, 0, 30, 0.9, None]] * 16} for t in range(0, 1000)]
        a, b = S.pick_scenes(frames, 4, 7), S.pick_scenes(frames, 4, 7)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 4)
        self.assertTrue(all(y - x >= 90 for x, y in zip(a, a[1:])))
        self.assertNotEqual(a, S.pick_scenes(frames, 4, 8))

    def test_whole_run_with_a_stub_detector(self):
        S = self.S

        class Box:
            def __init__(s, b, c):
                s.xyxy, s.cls, s.conf = [np.array(b, float)], [0], [c]

        class Stub:
            names = {0: "person"}

            def __call__(s, img, **kw):
                m = (np.abs(img.astype(int) - np.array([60, 140, 50])).sum(-1) > 90).astype(np.uint8)
                n, lab, st, _ = cv2.connectedComponentsWithStats(m)
                res = type("R", (), {})()
                res.boxes = [Box((x, y, x + w, y + h), .9) for x, y, w, h, a in st[1:] if a > 150]
                return [res]

        def frame(seed):
            r = np.random.default_rng(seed)
            f = np.full((X.EQ_H, X.EQ_W, 3), (60, 140, 50), np.uint8)
            for i in range(10):
                x, y = 900 + 100 * i, int(r.integers(780, 940))
                cv2.rectangle(f, (x, y), (x + 22, y + 34), (40, 40, 220) if i % 2 else (210, 70, 30), -1)
                cv2.rectangle(f, (x + 2, y + 34), (x + 20, y + 62), (20, 20, 20), -1)
            return f

        with tempfile.TemporaryDirectory() as d:
            det = os.path.join(d, "d.jsonl")
            with open(det, "w") as fh:
                fh.write(json.dumps({"type": "header", "tilt": -22.0, "view_lons": [-60, -20, 20, 60],
                                     "iv_fov": 112.0, "square_views": True, "raw_input": True}) + "\n")
                for t in range(0, 300):
                    fh.write(json.dumps({"t": float(t), "players": [[0, 0, 40, .9, None]] * 16, "balls": []}) + "\n")
            out = os.path.join(d, "o.png")
            rc = S.main(["--video", "x.mp4", "--detections", det, "--out", out, "--n", "2", "--fit-frames", "6",
                         "--k", "2", "--device", "cpu"], model=Stub(), grab=lambda t: frame(int(t)))
            self.assertEqual(rc, 0)
            self.assertGreater(os.path.getsize(out), 10000)


class TestColourExtraction(unittest.TestCase):
    def test_shirt_colour_survives_dry_yellow_ground(self):
        ground = (110, 180, 200)                                   # dry yellow-brown pitch (b, g, r)
        img = np.full((300, 300, 3), ground, np.uint8)
        cv2.rectangle(img, (130, 100), (170, 140), (150, 50, 20), -1)      # blue shirt
        cv2.rectangle(img, (135, 140), (165, 190), (30, 30, 30), -1)       # dark shorts and legs
        c = X.torso_colour(img, (120, 95, 180, 195))
        self.assertGreater(c[0], 120)                              # blue channel high: the shirt, not the ground
        self.assertLess(c[2], 80)
        green = np.full((300, 300, 3), (60, 140, 50), np.uint8)    # green pitch still works
        cv2.rectangle(green, (130, 100), (170, 140), (40, 40, 220), -1)
        g = X.torso_colour(green, (120, 95, 180, 195))
        self.assertGreater(g[2], 180)

    def test_colours_are_an_optional_sixth_value(self):
        img = np.full((720, 1280, 3), (60, 140, 50), np.uint8)
        cv2.rectangle(img, (600, 300), (630, 340), (40, 40, 220), -1)

        class Box:
            xyxy, cls, conf = [np.array([595.0, 295.0, 635.0, 400.0])], [0], [0.9]

        class M:
            names = {0: "person"}

            def __call__(s, im, **kw):
                return [type("R", (), {"boxes": [Box()]})()]

        plain = X.detect_players(img, M(), "cpu")
        self.assertEqual(len(plain[0]), 5)                                    # default output unchanged
        rich = X.detect_players(img, M(), "cpu", colours=True)
        self.assertEqual(len(rich[0]), 6)
        self.assertGreater(rich[0][5][2], 150)                                # red shirt: high R in [b, g, r]


class TestPitchFilter(unittest.TestCase):
    def test_players_outside_the_outline_are_flagged_and_dropped_by_the_loader(self):
        import pitch_filter as PF
        with tempfile.TemporaryDirectory() as d:
            # outline: the frame area around lon -40..40, lat -40..-5
            x0, y0 = K.lonlat_to_xy(-40, -5, X.EQ_W, X.EQ_H)
            x1, y1 = K.lonlat_to_xy(40, -40, X.EQ_W, X.EQ_H)
            cal = os.path.join(d, "pitch.json")
            with open(cal, "w") as fh:
                json.dump({"pitch_polygon": [[float(x0), float(y0)], [float(x1), float(y0)],
                                             [float(x1), float(y1)], [float(x0), float(y1)]]}, fh)
            src, dst = os.path.join(d, "a.jsonl"), os.path.join(d, "b.jsonl")
            with open(src, "w") as fh:
                fh.write(json.dumps({"type": "header", "video": "v"}) + "\n")
                fh.write(json.dumps({"t": 0.0, "balls": [], "players": [
                    [0.0, -20.0, 40.0, 0.9, None, [10, 20, 30]],            # on the pitch, with a colour
                    [90.0, -20.0, 40.0, 0.9, None],                         # far to the side
                    [0.0, 5.0, 40.0, 0.9, None]]}) + "\n")                 # above the horizon: stand / trees
            n, off = PF.apply(src, dst, cal, margin=10)
            self.assertEqual((n, off), (3, 2))
            rows = [json.loads(line) for line in open(dst)]
            self.assertEqual([p[4] for p in rows[1]["players"]], [True, False, False])
            self.assertEqual(rows[1]["players"][0][5], [10, 20, 30])         # colour kept
            self.assertIn("pitch_filter", rows[0])
            _, frames = AK.load_detections(dst)
            self.assertEqual(len(frames[0]["players"]), 1)                  # the rest of the pipeline never sees them
            _, plain = AK.load_detections(src)
            self.assertEqual(len(plain[0]["players"]), 3)                   # unflagged files behave as before


class TestTeamStats(unittest.TestCase):
    def test_two_teams_found_and_spread_recovered(self):
        import team_stats as TS
        rng = np.random.default_rng(1)
        fy = TS.S.SQUARE_FY
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.jsonl")
            with open(path, "w") as fh:
                fh.write(json.dumps({"type": "header", "start": 0.0, "fps": 3.0, "view_lons": [-60, -20, 20, 60],
                                     "square_views": True, "colours": True}) + "\n")
                for i in range(300):
                    pl = []
                    for bgr, cx in (((200, 200, 200), -8.0), ((120, 60, 30), 8.0)):
                        for _ in range(9):
                            x, y = cx + rng.normal(0, 8), 30 + rng.normal(0, 6)
                            lon = math.degrees(math.atan2(x, y)) / 0.75
                            c = min([-60, -20, 20, 60], key=lambda v: abs(v - lon))
                            D = math.hypot(x, y)
                            h = 1.7 * fy / (0.75 * D * math.cos(math.radians(0.75 * (lon - c))))
                            pl.append([lon, -10, h, 0.9, None, [int(v + rng.normal(0, 8)) for v in bgr]])
                    fh.write(json.dumps({"t": i / 3, "players": pl, "balls": []}) + "\n")
            out = os.path.join(d, "t")
            self.assertEqual(TS.main(["--detections", path, "--out", out, "--k", "3"]), 0)
            self.assertTrue(os.path.getsize(out + ".png") > 10000)
            rows = list(__import__("csv").DictReader(open(out + ".csv")))
            for col in ("A_spread_m", "B_spread_m"):                 # players scattered 8 m sideways, 6 m in depth
                self.assertAlmostEqual(float(rows[0][col]), 10.0, delta=2.0)
            self.assertIn("A_tempo_ms", rows[0])
            self.assertIn("A_territory_pct", rows[0])

    def test_chart_data_json(self):
        import team_stats as TS
        rng = np.random.default_rng(2)
        fy = TS.S.SQUARE_FY
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.jsonl")
            with open(path, "w") as fh:
                fh.write(json.dumps({"type": "header", "start": 0.0, "fps": 3.0, "view_lons": [-60, -20, 20, 60],
                                     "square_views": True, "colours": True}) + "\n")
                for i in range(240):
                    pl = []
                    for bgr, cx in (((200, 200, 200), -8.0), ((120, 60, 30), 8.0)):
                        for _ in range(9):
                            x, y = cx + rng.normal(0, 8), 30 + rng.normal(0, 6)
                            lon = math.degrees(math.atan2(x, y)) / 0.75
                            c = min([-60, -20, 20, 60], key=lambda v: abs(v - lon))
                            h = 1.7 * fy / (0.75 * math.hypot(x, y) * math.cos(math.radians(0.75 * (lon - c))))
                            pl.append([lon, -10, h, 0.9, None, [int(v + rng.normal(0, 8)) for v in bgr]])
                    fh.write(json.dumps({"t": i / 3, "players": pl, "balls": []}) + "\n")
            out, js = os.path.join(d, "t"), os.path.join(d, "t.json")
            self.assertEqual(TS.main(["--detections", path, "--out", out, "--k", "3", "--data-json", js]), 0)
            data = json.load(open(js))                                     # strict JSON: no NaN
            self.assertEqual(data["format"], "football-autopan-chart-data")
            self.assertEqual(set(data["teams"]), {"A", "B"})
            cnt = np.array(data["heat_map"]["counts"])
            self.assertEqual(cnt.shape, (len(data["heat_map"]["y_edges_m"]) - 1, len(data["heat_map"]["x_edges_m"]) - 1))
            self.assertGreater(cnt.sum(), 1000)
            self.assertEqual(len(data["territory"]["minutes"]), len(data["territory"]["value"]))
            self.assertAlmostEqual(data["teams"]["A"]["spread_m"][0], 10.0, delta=2.5)

    def test_teams_labelled_by_eye(self):
        import team_labels as TL
        import team_stats as TS
        rng = np.random.default_rng(3)

        class Box:
            def __init__(s, b, c):
                s.xyxy, s.cls, s.conf = [np.array(b, float)], [0], [c]

        class Stub:
            names = {0: "person"}

            def __call__(s, img, **kw):
                m = (np.abs(img.astype(int) - np.array([60, 140, 50])).sum(-1) > 90).astype(np.uint8)
                n, lab, st, _ = cv2.connectedComponentsWithStats(m)
                res = type("R", (), {})()
                res.boxes = [Box((x, y, x + w, y + h), .9) for x, y, w, h, a in st[1:] if a > 150]
                return [res]

        shirts = [(200, 200, 200), (190, 190, 215), (210, 70, 30), (150, 50, 20)]   # team A: two lights; team B: two blues

        def frame(seed):
            r = np.random.default_rng(seed)
            f = np.full((X.EQ_H, X.EQ_W, 3), (60, 140, 50), np.uint8)
            for i in range(12):
                x, y = 900 + 80 * i, int(r.integers(780, 940))
                cv2.rectangle(f, (x, y), (x + 22, y + 34), shirts[i % 4], -1)
                cv2.rectangle(f, (x + 2, y + 34), (x + 20, y + 62), (20, 20, 20), -1)
            return f

        with tempfile.TemporaryDirectory() as d:
            det = os.path.join(d, "d.jsonl")
            with open(det, "w") as fh:
                fh.write(json.dumps({"type": "header", "tilt": -22.0, "view_lons": [-60, -20, 20, 60], "iv_fov": 112.0,
                                     "square_views": True, "raw_input": True, "colours": True, "start": 0.0, "fps": 3.0}) + "\n")
                for t in range(0, 300):
                    fh.write(json.dumps({"t": float(t), "players": [[0, -10, 40, .9, None, [1, 2, 3]]] * 12, "balls": []}) + "\n")
            out = os.path.join(d, "g")
            self.assertEqual(TL.main(["make", "--video", "x.mp4", "--detections", det, "--out", out, "--k", "4",
                                      "--frames", "10", "--device", "cpu"], model=Stub(), grab=lambda t: frame(int(t))), 0)
            self.assertTrue(os.path.getsize(out + ".teams_sheet.png") > 5000)
            meta = json.load(open(out + ".team_labels.json"))
            # find which groups hold which shirt: nearest centre to each known shirt colour
            import scene_check as S2
            feats = S2.colour_features(np.array([[float(v) for v in s] for s in shirts]))
            grp = [int(np.argmin(((np.array(meta["centres"]) - f) ** 2).sum(1))) for f in feats]
            a_groups, b_groups = sorted(set(grp[:2])), sorted(set(grp[2:]))
            self.assertEqual(TL.main(["assign", "--labels", out + ".team_labels.json", "--a", *map(str, a_groups),
                                      "--b", *map(str, b_groups), "--name-a", "Light", "--colour-b", "blue"]), 0)
            lab = TL.Labelled(out + ".team_labels.json")
            self.assertEqual(lab.names[0], "Light")
            self.assertEqual(lab.colours[1], TL.COLOURS["blue"])
            team = lab.team_of(np.array([[200, 200, 200], [190, 190, 215], [210, 70, 30], [150, 50, 20]], float))
            self.assertEqual(team.tolist(), [0, 0, 1, 1])
            with self.assertRaises(SystemExit):
                TL.main(["assign", "--labels", out + ".team_labels.json", "--a", "99"])

    def test_play_centre_follows_the_camera_not_the_detector(self):
        import team_stats as TS
        with tempfile.TemporaryDirectory() as d:
            det, kf = os.path.join(d, "a.jsonl"), os.path.join(d, "k.json")
            with open(det, "w") as fh:
                fh.write(json.dumps({"type": "header", "start": 0.0, "fps": 3.0, "view_lons": [-60, -20, 20, 60],
                                     "square_views": True, "colours": True}) + "\n")
                for i in range(30):
                    # one player straight ahead of the camera (lon 0), one far outside the crop (lon 80)
                    fh.write(json.dumps({"t": i / 3.0, "balls": [], "players": [
                        [0.0, -10, 60.0, 0.9, None, [1, 2, 3]], [80.0, -10, 60.0, 0.9, None, [1, 2, 3]]]}) + "\n")
            with open(kf, "w") as fh:
                json.dump([{"t": 0, "pan_deg": 0, "fov_deg": 60}, {"t": 50, "pan_deg": 0, "fov_deg": 60}], fh)
            t, x, y = TS.load_play_centres([det], [kf], 98.0, 0.3, 0.75)
            self.assertEqual(len(t), 30)
            self.assertTrue(np.all(np.abs(x) < 0.5))               # only the player inside the crop counts, at lon 0

    def test_tempo_recovers_a_known_speed(self):
        import team_stats as TS
        n = 60
        t, x, y, fr, tm = [], [], [], [], []
        for f in range(n):
            for k in range(6):                                    # six players of one team running 2 m/s in x
                t.append(f / 3.0)
                x.append(10.0 * k + 2.0 * f / 3.0)
                y.append(20.0 + 3.0 * k)
                fr.append(f)
                tm.append(0)
        game = {"t": np.array(t), "x": np.array(x), "y": np.array(y), "frame": np.array(fr), "duration": n / 3.0,
                "n_frames": n}
        v = TS.tempo_per_minute(game, np.array(tm))
        self.assertAlmostEqual(float(v[0][0]), 2.0, delta=0.05)
        self.assertTrue(np.isnan(v[1][0]))                         # no second team

    def test_team_colours_follow_the_shirts(self):
        import football_style as FS
        self.assertEqual(FS.colour_name([150, 148, 138]), "white")             # shaded white shirt (b, g, r)
        self.assertEqual(FS.colour_name([111, 64, 36]), "blue")                # navy
        self.assertEqual(FS.display_colour([150, 148, 138]), "#ececec")
        h = FS.display_colour([111, 64, 36])                                   # navy -> bright blue, still blue
        r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
        self.assertGreater(b, 180)
        self.assertGreater(b, r + 40)

    def test_file_without_colours_is_refused(self):
        import team_stats as TS
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "n.jsonl")
            with open(path, "w") as fh:
                fh.write(json.dumps({"type": "header", "start": 0.0, "fps": 3.0, "view_lons": [0]}) + "\n")
                fh.write(json.dumps({"t": 0.0, "players": [], "balls": []}) + "\n")
            with self.assertRaises(SystemExit):
                TS.load_game([path], 98, 0.5, 18, 0.75)


if __name__ == "__main__":
    unittest.main(verbosity=2)
