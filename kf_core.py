"""
kf_core.py — shared core for the keyframe annotation + render tools.

Used by:
    annotate.py          interactive annotation tool (OpenCV)
    render_keyframes.py  FFmpeg render driven by the keyframe JSON
    make_proxy.py        raw MP4 -> equirect proxy (only uses probe helpers)

Contents
    * Camera conventions        (pan/tilt/fov  <->  where the crop sits on the equirect)
    * Projection geometry       (crop outline, hit-testing, remap maps for the live preview)
    * Keyframe store            (toggle / upsert / undo / redo / JSON load+save, atomic)
    * Interpolation             (PCHIP default, smoothstep, linear)

Conventions (verified against py360convert 1.0.4 as used by autopan_infer.py, and
against FFmpeg's v360 filter; see test_keyframe_tools.py):

    pan_deg   == cam.yaw in autopan_infer.py. The crop is centred on equirect
                 longitude  -pan_deg  (so *positive pan looks LEFT* on the equirect,
                 because autopan_infer.py calls e2p(u_deg=-yaw)).
    tilt_deg  == cam.pitch / e2p v_deg. Positive = up. A project default (meta sidecar) that
              a keyframe may override with its own optional `tilt_deg`; keyframes without one
              follow the project default, so older files behave exactly as before.
    fov_deg   == HORIZONTAL field of view of the output frame. The vertical FOV is derived
                 from the output aspect ratio (square pixels, true rectilinear).
                 Pass legacy_fov=True to reproduce py360convert's scalar-fov behaviour
                 (vertical FOV == horizontal FOV, i.e. an anamorphic 16:9 image).

Equirect frame: x = (lon/360 + 0.5) * W,  y = (0.5 - lat/180) * H.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ── Defaults ──────────────────────────────────────────────────────────────────
OUT_W, OUT_H = 1280, 720          # same as autopan_infer.py
DEFAULT_FOV = 80.0                # autopan_infer.OUTPUT_FOV
FOV_MIN, FOV_MAX = 20.0, 140.0
PAN_SIGN = -1.0                   # centre longitude = PAN_SIGN * pan_deg


# ── Small helpers ─────────────────────────────────────────────────────────────

def wrap180(a):
    """Wrap degrees into [-180, 180)."""
    return (np.asarray(a, dtype=np.float64) + 180.0) % 360.0 - 180.0


def fmt_time(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - h * 3600 - m * 60
    return f"{h}:{m:02d}:{s:06.3f}"


def probe_video(path: str) -> dict:
    """ffprobe -> {width, height, fps, duration, nb_frames}. Rotation is applied
    (FFmpeg autorotates via the Display Matrix), so portrait files report the
    *displayed* size."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:"
                            "stream_side_data=rotation:format=duration",
           "-of", "json", path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {r.stderr.strip()[-300:]}")
    d = json.loads(r.stdout)
    st = d["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    for sd in st.get("side_data_list", []) or []:
        rot = sd.get("rotation")
        if rot is not None and int(round(float(rot))) % 180 == 90:
            w, h = h, w

    def rate(s):
        try:
            n, dd = s.split("/")
            return float(n) / float(dd) if float(dd) else 0.0
        except Exception:
            return 0.0

    fps = rate(st.get("avg_frame_rate", "0/1")) or rate(st.get("r_frame_rate", "0/1"))
    dur = st.get("duration")
    if dur in (None, "N/A"):
        dur = d.get("format", {}).get("duration")
    dur = float(dur) if dur not in (None, "N/A") else 0.0
    nb = st.get("nb_frames")
    nb = int(nb) if nb not in (None, "N/A") else int(round(dur * fps))
    return {"width": w, "height": h, "fps": fps, "duration": dur, "nb_frames": nb}


def is_equirect_size(w: int, h: int, tol: float = 0.03) -> bool:
    return h > 0 and abs(w / h - 2.0) <= 2.0 * tol


# ── Projection geometry ───────────────────────────────────────────────────────

PROJECTIONS = ("flat", "cylindrical")


def plane_extent(fov_deg: float, out_w: int = OUT_W, out_h: int = OUT_H,
                 legacy_fov: bool = False, projection: str = "flat") -> Tuple[float, float]:
    """Half-extent of the output image in the projection's own coordinates.

    flat        : (tan(h/2), tan(v/2)) on the image plane at unit distance.
    cylindrical : (h/2 in radians, tan(v/2)) — horizontal angle is linear in x, vertical is
                  tan(latitude) (what FFmpeg's v360=output=cylindrical does). Vertical extent
                  is chosen so pixels are square at the image centre."""
    h = math.radians(fov_deg)
    if projection == "cylindrical":
        xm = h / 2.0
        ym = math.tan(xm) if legacy_fov else xm * out_h / out_w
        return xm, ym
    xm = math.tan(h / 2.0)
    ym = xm if legacy_fov else xm * out_h / out_w
    return xm, ym


def vfov_deg(fov_deg: float, out_w: int = OUT_W, out_h: int = OUT_H,
             legacy_fov: bool = False, projection: str = "flat") -> float:
    _, ym = plane_extent(fov_deg, out_w, out_h, legacy_fov, projection)
    return math.degrees(2.0 * math.atan(ym))


def _rays(gx, gy, basis, projection: str = "flat"):
    """Rays for grid coordinates (gx along x, gy along y; broadcastable). Last axis = xyz."""
    f, r, u = basis
    if projection == "cylindrical":
        return (np.cos(gx)[..., None] * f + np.sin(gx)[..., None] * r + gy[..., None] * u)
    return f + gx[..., None] * r + gy[..., None] * u


def centre_lonlat(pan_deg: float, tilt_deg: float) -> Tuple[float, float]:
    return float(wrap180(PAN_SIGN * pan_deg)), float(tilt_deg)


def _basis(pan_deg: float, tilt_deg: float):
    """forward / right / up unit vectors of the virtual camera (yaw about the world
    vertical applied after pitch, i.e. the horizon stays level)."""
    lon = math.radians(PAN_SIGN * pan_deg)
    lat = math.radians(tilt_deg)
    f = np.array([math.cos(lat) * math.sin(lon), math.sin(lat), math.cos(lat) * math.cos(lon)])
    r = np.array([math.cos(lon), 0.0, -math.sin(lon)])
    u = np.cross(f, r)
    return f, r, u


def rays_to_lonlat(rays: np.ndarray, ref_lon_deg: float):
    """(...,3) rays -> (lon_deg unwrapped around ref_lon, lat_deg)."""
    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    lon = np.degrees(np.arctan2(x, z))
    lat = np.degrees(np.arctan2(y, np.hypot(x, z)))
    lon = ref_lon_deg + wrap180(lon - ref_lon_deg)
    return lon, lat


def view_rays(pan_deg: float, tilt_deg: float, fov_deg: float,
              out_w: int, out_h: int, legacy_fov: bool = False,
              projection: str = "flat") -> np.ndarray:
    """(out_h, out_w, 3) world-space rays; pixel grid identical to py360convert
    (edge pixels sit exactly on the frustum boundary)."""
    xm, ym = plane_extent(fov_deg, out_w, out_h, legacy_fov, projection)
    gx = np.linspace(-xm, xm, out_w, dtype=np.float32)
    gy = np.linspace(ym, -ym, out_h, dtype=np.float32)       # row 0 = top
    basis = tuple(v.astype(np.float32) for v in _basis(pan_deg, tilt_deg))
    return _rays(gx[None, :], gy[:, None], basis, projection)


def lonlat_to_xy(lon_deg, lat_deg, w: int, h: int):
    x = (np.asarray(lon_deg) / 360.0 + 0.5) * w - 0.5
    y = (0.5 - np.asarray(lat_deg) / 180.0) * h - 0.5
    return x, y


def xy_to_lonlat(x, y, w: int, h: int):
    lon = ((np.asarray(x, dtype=np.float64) + 0.5) / w - 0.5) * 360.0
    lat = (0.5 - (np.asarray(y, dtype=np.float64) + 0.5) / h) * 180.0
    return lon, lat


def build_remap(pan_deg: float, tilt_deg: float, fov_deg: float,
                src_w: int, src_h: int, out_w: int, out_h: int,
                legacy_fov: bool = False, step: int = 1,
                projection: str = "flat") -> Tuple[np.ndarray, np.ndarray]:
    """cv2.remap maps (map_x, map_y) projecting an equirect frame of size
    (src_w, src_h) to the perspective view. x is left unwrapped (may be <0 or >src_w);
    use borderMode=cv2.BORDER_WRAP.

    step > 1 evaluates the projection on a grid `step` times coarser and bilinearly
    upsamples the (smooth, seam-free) maps: ~step^2 faster, error well under 0.3 px for
    step <= 4. Used for the live preview; the exact path (step=1) is the reference."""
    c_lon, _ = centre_lonlat(pan_deg, tilt_deg)
    if step <= 1:
        rays = view_rays(pan_deg, tilt_deg, fov_deg, out_w, out_h, legacy_fov, projection)
        lon, lat = rays_to_lonlat(rays, c_lon)
        mx, my = lonlat_to_xy(lon, lat, src_w, src_h)
        return mx.astype(np.float32), my.astype(np.float32)

    import cv2                                  # lazy: only the preview needs it
    cw, ch = -(-out_w // step), -(-out_h // step)
    # coarse samples sit where cv2.resize expects them, padded by one cell on each side so
    # the frame edges interpolate instead of replicating
    px = (np.arange(-1, cw + 1) + 0.5) * step - 0.5
    py = (np.arange(-1, ch + 1) + 0.5) * step - 0.5
    xm, ym = plane_extent(fov_deg, out_w, out_h, legacy_fov, projection)
    gx = (-xm + 2 * xm * px / (out_w - 1)).astype(np.float32)
    gy = (ym - 2 * ym * py / (out_h - 1)).astype(np.float32)
    basis = tuple(v.astype(np.float32) for v in _basis(pan_deg, tilt_deg))
    rays = _rays(gx[None, :], gy[:, None], basis, projection)
    lon, lat = rays_to_lonlat(rays, c_lon)
    mx, my = lonlat_to_xy(lon, lat, src_w, src_h)
    size = ((cw + 2) * step, (ch + 2) * step)
    out = []
    for m in (mx, my):
        big = cv2.resize(m.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
        out.append(np.ascontiguousarray(big[step:step + out_h, step:step + out_w]))
    return out[0], out[1]


def crop_outline(pan_deg: float, tilt_deg: float, fov_deg: float,
                 out_w: int = OUT_W, out_h: int = OUT_H,
                 legacy_fov: bool = False, n: int = 20,
                 projection: str = "flat") -> Tuple[np.ndarray, np.ndarray]:
    """Boundary of the crop on the equirect as (lon_deg, lat_deg) arrays, in order
    around the frame, longitudes unwrapped around the crop centre (so continuous)."""
    xm, ym = plane_extent(fov_deg, out_w, out_h, legacy_fov, projection)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    xs = np.concatenate([-xm + 2 * xm * t, np.full(n, xm), xm - 2 * xm * t, np.full(n, -xm)])
    ys = np.concatenate([np.full(n, ym), ym - 2 * ym * t, np.full(n, -ym), -ym + 2 * ym * t])
    rays = _rays(xs, ys, _basis(pan_deg, tilt_deg), projection)
    c_lon, _ = centre_lonlat(pan_deg, tilt_deg)
    return rays_to_lonlat(rays, c_lon)


def point_in_crop(lon_deg: float, lat_deg: float, pan_deg: float, tilt_deg: float,
                  fov_deg: float, out_w: int = OUT_W, out_h: int = OUT_H,
                  legacy_fov: bool = False, projection: str = "flat") -> bool:
    lo, la = math.radians(lon_deg), math.radians(lat_deg)
    p = np.array([math.cos(la) * math.sin(lo), math.sin(la), math.cos(la) * math.cos(lo)])
    f, r, u = _basis(pan_deg, tilt_deg)
    xm, ym = plane_extent(fov_deg, out_w, out_h, legacy_fov, projection)
    if projection == "cylindrical":
        px, pz = float(p @ r), float(p @ f)
        rho = math.hypot(px, pz)
        if rho <= 1e-6:
            return False
        return abs(math.atan2(px, pz)) <= xm and abs(float(p @ u) / rho) <= ym
    d = float(p @ f)
    if d <= 1e-6:
        return False
    return abs(float(p @ r) / d) <= xm and abs(float(p @ u) / d) <= ym


def to_v360_params(pan_deg: float, tilt_deg: float, fov_deg: float,
                   out_w: int = OUT_W, out_h: int = OUT_H,
                   legacy_fov: bool = False, projection: str = "flat") -> dict:
    """Map (pan, tilt, fov) to the runtime options of FFmpeg's v360=equirect:<projection>."""
    return {
        "yaw": float(wrap180(PAN_SIGN * pan_deg)),   # v360: +yaw looks right
        "pitch": float(tilt_deg),                    # v360: +pitch looks up
        "h_fov": float(fov_deg),
        "v_fov": vfov_deg(fov_deg, out_w, out_h, legacy_fov, projection),
    }


# ── Tilt from calibration (same maths as autopan_infer.derive_tilt_fov) ───────

def tilt_from_calib(calib_path: str) -> float:
    with open(calib_path) as f:
        d = json.load(f)
    raw = d.get("pitch_polygon") or d.get("pixel_polygon") or d.get("auto_polygon")
    if isinstance(raw[0], dict):
        ys = np.array([p["y"] * 1440 for p in raw], dtype=np.float64)
    else:
        ys = np.array([p[1] for p in raw], dtype=np.float64)
    tilt = (0.5 - ys / 1440.0) * 180.0
    far, near = float(np.mean(tilt[:7])), float(np.mean(tilt[7:]))
    return ((far + near) / 2.0) * 1.20


# ── Project sidecar (tilt etc. that the keyframe list itself does not carry) ──

def meta_path(kf_path: str) -> str:
    """foo.keyframes.json -> foo.keyframes.meta.json"""
    root, _ = os.path.splitext(kf_path)
    return root + ".meta.json"


def save_meta(kf_path: str, **meta) -> None:
    p = meta_path(kf_path)
    d = os.path.dirname(os.path.abspath(p))
    fd, tmp = tempfile.mkstemp(prefix=".meta_", suffix=".json", dir=d)
    with os.fdopen(fd, "w") as f:
        json.dump(meta, f, indent=1)
        f.write("\n")
    os.replace(tmp, p)


def load_meta(kf_path: str) -> dict:
    p = meta_path(kf_path)
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# ── Keyframes ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Keyframe:
    t: float
    pan_deg: float
    fov_deg: float
    tilt_deg: Optional[float] = None       # None -> the project tilt (meta sidecar)

    def as_dict(self) -> dict:
        d = {"t": round(self.t, 4), "pan_deg": round(self.pan_deg, 4),
             "fov_deg": round(self.fov_deg, 4)}
        if self.tilt_deg is not None:        # only written when set: old files stay byte-identical
            d["tilt_deg"] = round(self.tilt_deg, 4)
        return d


class KeyframeSet:
    """Sorted keyframes with toggle/upsert/remove, undo/redo and atomic JSON IO.
    `tol` is the time window (seconds) within which two keyframes count as the
    same one — callers pass half a frame."""

    def __init__(self, tol: float = 1 / 60):
        self.tol = tol
        self._k: List[Keyframe] = []
        self._undo: List[Tuple[Keyframe, ...]] = []
        self._redo: List[Tuple[Keyframe, ...]] = []
        self.version = 0                      # bumps on every change (cache key)

    # -- queries
    def __len__(self):
        return len(self._k)

    def __iter__(self):
        return iter(self._k)

    @property
    def items(self) -> List[Keyframe]:
        return list(self._k)

    def times(self) -> np.ndarray:
        return np.array([k.t for k in self._k], dtype=np.float64)

    def index_at(self, t: float) -> Optional[int]:
        for i, k in enumerate(self._k):
            if abs(k.t - t) <= self.tol:
                return i
        return None

    def at(self, t: float) -> Optional[Keyframe]:
        i = self.index_at(t)
        return None if i is None else self._k[i]

    def prev_before(self, t: float) -> Optional[Keyframe]:
        c = [k for k in self._k if k.t < t - self.tol]
        return c[-1] if c else None

    def next_after(self, t: float) -> Optional[Keyframe]:
        for k in self._k:
            if k.t > t + self.tol:
                return k
        return None

    # -- undo plumbing
    def snapshot(self):
        """Push an undo point (call once at the start of a compound edit)."""
        self._undo.append(tuple(self._k))
        if len(self._undo) > 300:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(tuple(self._k))
        self._k = list(self._undo.pop())
        self.version += 1
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(tuple(self._k))
        self._k = list(self._redo.pop())
        self.version += 1
        return True

    # -- edits (these do NOT snapshot; callers decide the undo granularity)
    def upsert(self, kf: Keyframe) -> None:
        i = self.index_at(kf.t)
        if i is None:
            self._k.append(kf)
            self._k.sort(key=lambda k: k.t)
        else:
            self._k[i] = kf
        self.version += 1

    def remove_at(self, t: float) -> bool:
        i = self.index_at(t)
        if i is None:
            return False
        del self._k[i]
        self.version += 1
        return True

    def remove_between(self, t0: float, t1: float) -> int:
        """Remove keyframes with t0 < t < t1 (exclusive, with tolerance)."""
        keep = [k for k in self._k if not (t0 + self.tol < k.t < t1 - self.tol)]
        n = len(self._k) - len(keep)
        if n:
            self._k = keep
            self.version += 1
        return n

    def toggle(self, kf: Keyframe) -> bool:
        """Add keyframe, or remove the one at that time. Returns True if added."""
        self.snapshot()
        if self.remove_at(kf.t):
            return False
        self.upsert(kf)
        return True

    # -- IO
    def to_list(self) -> List[dict]:
        return [k.as_dict() for k in self._k]

    def save(self, path: str) -> None:
        d = os.path.dirname(os.path.abspath(path))
        fd, tmp = tempfile.mkstemp(prefix=".kf_", suffix=".json", dir=d)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.to_list(), f, indent=1)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    @classmethod
    def from_list(cls, rows: Sequence[dict], tol: float = 1 / 60) -> "KeyframeSet":
        ks = cls(tol)
        for i, r in enumerate(rows):
            try:
                tl = r.get("tilt_deg")
                kf = Keyframe(float(r["t"]), float(r["pan_deg"]), float(r["fov_deg"]),
                              None if tl is None else float(tl))
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"keyframe #{i} must have numeric t, pan_deg, fov_deg: {r!r}") from e
            vals = [kf.t, kf.pan_deg, kf.fov_deg] + ([] if kf.tilt_deg is None else [kf.tilt_deg])
            if not all(math.isfinite(v) for v in vals) or kf.t < 0:
                raise ValueError(f"keyframe #{i} has an invalid value: {r!r}")
            ks._k.append(kf)
        ks._k.sort(key=lambda k: k.t)
        return ks

    @classmethod
    def load(cls, path: str, tol: float = 1 / 60) -> "KeyframeSet":
        with open(path) as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"{path}: expected a JSON list of keyframes")
        return cls.from_list(rows, tol)


# ── Interpolation ─────────────────────────────────────────────────────────────

def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch–Carlson monotone slopes (same formulas as scipy's PchipInterpolator)."""
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(n)
    if n == 2:
        d[:] = delta[0]
        return d
    w1 = 2 * h[1:] + h[:-1]
    w2 = h[1:] + 2 * h[:-1]
    same = delta[:-1] * delta[1:] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        hm = (w1 + w2) / (w1 / delta[:-1] + w2 / delta[1:])
    d[1:-1] = np.where(same, hm, 0.0)

    def end(h0, h1, d0, d1):
        s = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(s) != np.sign(d0):
            return 0.0
        if np.sign(d0) != np.sign(d1) and abs(s) > 3 * abs(d0):
            return 3 * d0
        return s

    d[0] = end(h[0], h[1], delta[0], delta[1])
    d[-1] = end(h[-1], h[-2], delta[-1], delta[-2])
    return d


class Path:
    """Interpolated camera path (pan_deg, fov_deg, and tilt) over time from a KeyframeSet.
    Holds the first value before the first keyframe and the last after the last.

    Tilt: a keyframe without its own tilt_deg uses `base_tilt` (the project tilt), so a
    path with no per-keyframe tilt at all (`has_tilt` False) is just the constant base."""

    METHODS = ("pchip", "smoothstep", "linear")

    def __init__(self, kfs: Sequence[Keyframe], method: str = "pchip", base_tilt: float = 0.0):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.method = method
        ks = sorted(kfs, key=lambda k: k.t)
        # collapse duplicate timestamps (keep the last)
        uniq = {}
        for k in ks:
            uniq[round(k.t, 6)] = k
        ks = [uniq[t] for t in sorted(uniq)]
        self.t = np.array([k.t for k in ks], dtype=np.float64)
        self.pan = np.array([k.pan_deg for k in ks], dtype=np.float64)
        self.fov = np.array([k.fov_deg for k in ks], dtype=np.float64)
        self.base_tilt = float(base_tilt)
        self.has_tilt = any(k.tilt_deg is not None for k in ks)
        self.tilt = np.array([self.base_tilt if k.tilt_deg is None else k.tilt_deg for k in ks],
                             dtype=np.float64)
        self._dpan = self._dfov = self._dtilt = None
        if method == "pchip" and len(ks) >= 2:
            self._dpan = _pchip_slopes(self.t, self.pan)
            self._dfov = _pchip_slopes(self.t, self.fov)
            if self.has_tilt:
                self._dtilt = _pchip_slopes(self.t, self.tilt)

    def __bool__(self):
        return len(self.t) > 0

    def _eval(self, tq: np.ndarray, y: np.ndarray, d: Optional[np.ndarray]) -> np.ndarray:
        n = len(self.t)
        if n == 1:
            return np.full_like(tq, y[0], dtype=np.float64)
        tc = np.clip(tq, self.t[0], self.t[-1])
        i = np.clip(np.searchsorted(self.t, tc, side="right") - 1, 0, n - 2)
        h = self.t[i + 1] - self.t[i]
        s = (tc - self.t[i]) / h
        y0, y1 = y[i], y[i + 1]
        if self.method == "linear":
            return y0 + (y1 - y0) * s
        if self.method == "smoothstep":
            return y0 + (y1 - y0) * (s * s * (3 - 2 * s))
        s2, s3 = s * s, s * s * s
        h00 = 2 * s3 - 3 * s2 + 1
        h10 = s3 - 2 * s2 + s
        h01 = -2 * s3 + 3 * s2
        h11 = s3 - s2
        return h00 * y0 + h10 * h * d[i] + h01 * y1 + h11 * h * d[i + 1]

    def at(self, t):
        """Vectorised: returns (pan_deg, fov_deg) arrays (or floats for scalar t)."""
        scalar = np.isscalar(t)
        tq = np.atleast_1d(np.asarray(t, dtype=np.float64))
        pan = self._eval(tq, self.pan, self._dpan)
        fov = self._eval(tq, self.fov, self._dfov)
        if scalar:
            return float(pan[0]), float(fov[0])
        return pan, fov

    def tilt_at(self, t):
        """Tilt in degrees at time(s) t (the constant project tilt when no keyframe sets one)."""
        scalar = np.isscalar(t)
        tq = np.atleast_1d(np.asarray(t, dtype=np.float64))
        if not self.has_tilt or len(self.t) == 0:
            out = np.full(tq.shape, self.base_tilt, dtype=np.float64)
        else:
            out = self._eval(tq, self.tilt, self._dtilt)
        return float(out[0]) if scalar else out
