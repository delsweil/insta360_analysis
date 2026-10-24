#!/usr/bin/env python3
"""Tests for lens.py (no video needed)."""
import math
import sys

import numpy as np

import lens as L


def check(name, ok, info=""):
    print(("OK   " if ok else "FAIL ") + name + (f"  {info}" if info else ""))
    return ok


def main() -> int:
    ok = True
    # raw pixel <-> true direction round trip
    px, py = np.array([5.0, 1280.0, 2500.0]), np.array([10.0, 720.0, 1400.0])
    a, b = L.true_dir_to_raw(L.raw_to_true_dir(px, py))
    ok &= check("true lens round trip", np.abs(a - px).max() < 1e-6 and np.abs(b - py).max() < 1e-6)

    # the legacy model is FFmpeg's fisheye unwrap: values read off a real v360 run (16-bit coordinate image)
    def v_of(lon, lat):
        lo, la = math.radians(lon), math.radians(lat)
        return np.array([math.cos(la) * math.sin(lo), math.sin(la), math.cos(la) * math.cos(lo)])
    for (i, j), (ex, ey) in {(1000, 900): (618.0, 1056.6), (1900, 1000): (1908.7, 1244.9), (800, 800): (271.3, 902.0)}.items():
        lon, lat = (i + 0.5) / 2880 * 360 - 180, 90 - (j + 0.5) / 1440 * 180
        gx, gy = L.legacy_dir_to_raw(v_of(lon, lat), 112.0)
        ok &= check(f"legacy model matches v360 at equirect ({i},{j})", abs(gx - ex) < 1.0 and abs(gy - ey) < 1.0, f"{gx:.1f},{gy:.1f}")

    # a straight 3D line stays straight: points of a line seen by the true lens are coplanar with the camera
    p0, d = np.array([-30.0, -4.0, 8.0]), np.array([1.0, 0.0, 0.0])
    pts = np.array([p0 + t * d for t in np.linspace(0, 60, 40)])
    v = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    gx, gy = L.true_dir_to_raw(v)
    back = L.raw_to_true_dir(gx, gy)
    s = np.linalg.svd(back, compute_uv=False)
    ok &= check("straight line is coplanar through the lens", s[-1] < 1e-9, f"{s[-1]:.2e}")

    # views
    c = L.convert_view(0, -20, 90, 9 / 16, 112.0)
    ok &= check("centre view keeps its size and pitch", abs(c["yaw"]) < 1e-6 and 70 < c["h_fov"] < 80 and -19 < c["pitch"] < -15 and c["scale"] == 1.0, str({k: round(v, 2) for k, v in c.items()}))
    l = L.convert_view(-40, -20, 100, 9 / 16, 112.0)
    r = L.convert_view(40, -20, 100, 9 / 16, 112.0)
    ok &= check("left/right views mirror each other", abs(l["yaw"] + r["yaw"]) < 1e-6 and abs(l["h_fov"] - r["h_fov"]) < 1e-6)
    far = L.convert_view(-60, -20, 100, 9 / 16, 112.0)
    qx, qy = L.true_dir_to_raw(np.stack([_ for _ in [L._unit(L._rays(far["yaw"], far["pitch"], math.tan(math.radians(far["h_fov"]) / 2), math.tan(math.radians(far["h_fov"]) / 2) * 9 / 16))][0]]))
    ok &= check("a view near the edge is narrowed to stay inside the lens image", far["scale"] < 1.0 and L._inside(qx, qy, 0.0), f"scale {far['scale']:.2f}")
    ih, iv = L.sg_fovs()
    ok &= check("v360 sg fields reproduce the fitted focal length", abs((L.RAW_W / 2) / (2 * math.tan(math.radians(ih) / 4)) - L.F_PX) < 1e-6)
    print("\nALL OK" if ok else "\nFAILURES")
    return 0 if ok else 1




def test_view_render_maps():
    import numpy as np
    try:
        import view_render as VR
    except Exception:
        print("skip view_render (no cv2)"); return
    if not VR.have_cv2():
        print("skip view_render (no cv2)"); return
    yaw, pitch, hf, W, H = 20.0, -15.0, 60.0, 640, 360
    mx, my = VR.make_maps(yaw, pitch, hf, W, H)
    assert mx.shape == (H, W) and mx.dtype == np.float32
    # exact projection at a few pixels vs the coarse+bilinear map
    import math
    tx = math.tan(math.radians(hf) / 2); ty = tx * H / W
    f, r, u = L._basis(yaw, pitch)
    for (i, j) in [(0, 0), (H // 2, W // 2), (H - 1, W - 1), (37, 411)]:
        gx = ((2 * j + 1) / W - 1) * tx; gy = (1 - (2 * i + 1) / H) * ty
        ray = f + gx * r + gy * u; ray = ray / np.linalg.norm(ray)
        px, py = L.true_dir_to_raw(ray[None, :])
        assert abs(float(np.ravel(px)[0]) - mx[i, j]) < 0.2 and abs(float(np.ravel(py)[0]) - my[i, j]) < 0.2, (i, j)
    print("view_render maps OK")


if __name__ == "__main__":
    test_view_render_maps()
    sys.exit(main())
