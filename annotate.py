#!/usr/bin/env python3
"""
annotate.py — fast, keyboard-driven keyframe annotation for broadcast-style panning.

    python make_proxy.py VID_xxx.mp4                 # once: raw MP4 -> equirect proxy
    python annotate.py   VID_xxx.equirect.mp4        # annotate (autosaves as you go)
    python render_keyframes.py --input VID_xxx.mp4 \
        --keyframes VID_xxx.equirect.keyframes.json --output game.mp4

The window shows the equirect frame with the broadcast crop outlined (everything
outside it is dimmed), a live preview of the actual output frame, and a timeline with
your keyframes, the interpolated pan curve and warnings for un-annotated stretches.

WORKFLOW
  Play at 2x, steer the crop with the mouse, press K to drop a keyframe whenever the
  crop is right. Or press R (REC): play and just *steer* — a keyframe is dropped
  automatically every 0.5 s while you move the crop. Everything autosaves.

  The crop always follows the interpolated path between keyframes. The moment you
  move it yourself it is "held" (orange outline) until you commit it with K (or snap
  back to the path with G). Parked exactly on a keyframe you edit it live (green).

KEYS  (H shows this in the window)
  SPACE play/pause        1 / 2 / 3  speed 1x / 1.5x / 2x      0  speed 0.5x
  A / D  or  <- / ->      step 1 frame back / forward
  Q / E                   jump 5 s back / forward   (shift: 10 s)       Up / Down  +/- 5 s
  , / .                   previous / next keyframe
  K                       add keyframe here with the current crop; again removes it
                          (if you have moved the crop it updates the keyframe instead)
  J / L                   nudge crop left / right 1 deg   (shift: 5 deg)
  I / O                   zoom in / out
  W / S                   tilt up / down 0.5 deg  (the project tilt: global for the whole video)
  T / B                   tilt THIS keyframe up / down 0.5 deg (e.g. look lower when the play is
                          on the near touchline); it eases back to the project tilt at the next
                          keyframe that has no tilt of its own.   X  clear this keyframe's tilt
  G                       drop your unsaved crop, snap to the interpolated path
  R                       REC: auto-keyframe while playing and steering
  Z / Y                   undo / redo keyframe edits
  F  live preview on/off     V  neighbour-keyframe outlines on/off
  shift+S                 save now        ESC  save and quit
MOUSE
  drag on the picture     move the crop (click outside it to re-centre it there)
  scroll wheel            zoom in / out
  click / drag timeline   jump / scrub

Keyframes are saved as [{"t": seconds, "pan_deg": .., "fov_deg": ..}, ...] (plus "tilt_deg"
on keyframes whose tilt differs from the project tilt).
Conventions (identical to autopan_infer.py): pan_deg is cam.yaw, so positive pan looks
LEFT on the equirect; fov_deg is the horizontal field of view of the 1280x720 output.
The project tilt is stored next to the keyframes in <name>.meta.json.
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

import kf_core as K

WIN = "keyframe annotator"
TL_H = 76
HUD_H = 58
PAD = 14
SKIP_AHEAD_MAX = 60          # decode-and-discard up to this many frames instead of seeking

# keys that arrive as waitKeyEx() codes on the three desktop backends
ARROW_CODES = {
    2424832: "left", 2555904: "right", 2490368: "up", 2621440: "down",   # Windows
    65361: "left", 65363: "right", 65362: "up", 65364: "down",           # Linux
    63234: "left", 63235: "right", 63232: "up", 63233: "down",           # macOS
    2: "left", 3: "right", 0: "up", 1: "down",                           # older macOS
}

HELP_LINES = [
    "SPACE play/pause     1/2/3 speed 1x/1.5x/2x     0 speed 0.5x",
    "A/D or arrows        step 1 frame",
    "Q/E  jump 5 s  (shift = 10 s)       up/down  5 s",
    ",  .                 previous / next keyframe",
    "K    add/remove keyframe (updates it if you moved the crop)",
    "J/L  nudge crop 1 deg (shift 5)     I/O  zoom in/out     W/S  project tilt",
    "T/B  tilt this keyframe up/down     X  clear this keyframe's tilt",
    "[  ]  mark where the game starts / ends (press again on the mark to clear it)",
    "G    drop my unsaved crop, snap to path",
    "R    REC: auto-keyframe every 0.5 s while playing + steering",
    "Z/Y  undo/redo           F  preview          V  neighbour outlines",
    "shift+S save now         ESC save + quit",
    "mouse: drag = move crop (click outside = re-centre)   wheel = zoom",
    "       click/drag timeline = jump/scrub",
]

C_CROP = (0, 255, 255)       # yellow     : crop follows the path
C_HOLD = (0, 150, 255)       # orange     : crop moved by you, not committed
C_LIVE = (0, 255, 0)         # green      : parked on a keyframe, edits update it
C_PREV = (255, 170, 0)
C_TRIM = (80, 255, 80)       # green-ish : game start / end marks
C_NEXT = (255, 0, 255)
C_KF = (0, 200, 255)


# ─────────────────────────────────────────────────────────────────────────────
# Video access
# ─────────────────────────────────────────────────────────────────────────────

class FrameSource:
    """Random-access frames, resized to display size, with an LRU cache so
    stepping backwards and re-visiting frames is instant. Forward playback decodes
    sequentially (skipped frames are only grabbed, never converted/resized)."""

    BACK_BLOCK = 12            # frames decoded per backwards cache miss (>= the proxy's GOP)

    def __init__(self, path: str, disp_w: int, cache_size: int = 48):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self.path = path
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.n <= 0:
            raise RuntimeError("video reports no frames")
        self.disp_w = disp_w
        self.disp_h = disp_w // 2
        self._pos = 0                       # index of the frame the next read() returns
        self._cache: "collections.OrderedDict[int, np.ndarray]" = collections.OrderedDict()
        self._cache_size = cache_size

    def get(self, idx: int) -> Optional[np.ndarray]:
        idx = max(0, min(self.n - 1, idx))
        hit = self._cache.get(idx)
        if hit is not None:
            self._cache.move_to_end(idx)
            return hit
        if idx < self._pos:
            # Stepping backwards: re-decode a short block ending at idx and cache all of it,
            # so the next BACK_BLOCK-1 steps back are instant instead of a seek each.
            start = max(0, idx - self.BACK_BLOCK + 1)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            self._pos = start
            while self._pos < idx:
                ok, f = self.cap.read()
                if not ok or f is None:
                    return None
                self._store(self._pos, f)
                self._pos += 1
        elif idx > self._pos + SKIP_AHEAD_MAX:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._pos = idx
        while self._pos < idx:                  # short hop forward: decode, don't convert
            if not self.cap.grab():
                return None
            self._pos += 1
        ok, f = self.cap.read()
        if not ok or f is None:
            return None
        self._pos += 1
        return self._store(idx, f)

    def _store(self, idx: int, f: np.ndarray) -> np.ndarray:
        if f.shape[1] != self.disp_w:
            f = cv2.resize(f, (self.disp_w, self.disp_h), interpolation=cv2.INTER_AREA)
        self._cache[idx] = f
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return f

    def close(self):
        self.cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

FONT = cv2.FONT_HERSHEY_SIMPLEX


def put(img, text, org, scale=0.55, color=(255, 255, 255), thick=1):
    """Text with a dark outline. The outline is drawn as offset copies at the *same*
    thickness (a thicker stroke would advance wider and ghost past the real text)."""
    x, y = org
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cv2.putText(img, text, (x + dx, y + dy), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def text_w(text, scale=0.55, thick=1) -> int:
    return cv2.getTextSize(text, FONT, scale, thick)[0][0]


# ─────────────────────────────────────────────────────────────────────────────
# The application (no window calls in here — run() owns the window)
# ─────────────────────────────────────────────────────────────────────────────

def _opt_float(v) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


class Annotator:
    SPEEDS = {"1": 1.0, "2": 1.5, "3": 2.0, "0": 0.5}

    def __init__(self, video: str, kf_path: Optional[str] = None, tilt: Optional[float] = None,
                 disp_w: int = 1440, legacy_fov: bool = False, interp: str = "pchip",
                 projection: Optional[str] = None,
                 gap_warn: float = 20.0, rec_interval: float = 0.5,
                 out_w: int = K.OUT_W, out_h: int = K.OUT_H, start: Optional[float] = None,
                 fov_max: Optional[float] = None):
        self.video = video
        self.kf_path = kf_path or (os.path.splitext(video)[0] + ".keyframes.json")
        self.src = FrameSource(video, disp_w)
        if not K.is_equirect_size(self.src.src_w, self.src.src_h):
            self.src.close()
            raise SystemExit(
                f"{video} is {self.src.src_w}x{self.src.src_h}, not a 2:1 equirectangular video.\n"
                f"Convert the raw MP4 first:  python make_proxy.py {video}")
        self.fps = self.src.fps
        self.n = self.src.n
        self.duration = (self.n - 1) / self.fps
        self.disp_w = disp_w
        self.eq_h = self.src.disp_h
        self.size = (disp_w, self.eq_h + TL_H + HUD_H)
        self.out_w, self.out_h = out_w, out_h
        self.legacy_fov = legacy_fov
        self.interp = interp
        self.gap_warn = gap_warn
        self.rec_interval = rec_interval

        meta = K.load_meta(self.kf_path)
        fm = _opt_float(fov_max if fov_max is not None else meta.get("fov_max"))
        self.fov_max = float(min(K.FOV_HARD_MAX, max(60.0, fm))) if fm is not None else K.FOV_MAX    # widest zoom-out allowed
        self.base_tilt = float(tilt if tilt is not None else meta.get("tilt_deg", 0.0))   # project tilt
        self.tilt = self.base_tilt                                  # tilt of the view being shown / edited
        # a project saved before projections existed was annotated in the flat view; new ones start cylindrical
        self.projection = projection or meta.get("projection", "flat" if meta else "cylindrical")
        if self.projection not in K.PROJECTIONS:
            self.projection = "flat"
        self.trim_in: Optional[float] = _opt_float(meta.get("trim_in"))
        self.trim_out: Optional[float] = _opt_float(meta.get("trim_out"))
        self.kfs = (K.KeyframeSet.load(self.kf_path, tol=0.5 / self.fps)
                    if os.path.exists(self.kf_path) else K.KeyframeSet(tol=0.5 / self.fps))
        self._saved_version = self.kfs.version
        self._meta_dirty = True
        self._last_save = 0.0

        # state
        self.idx = 0
        self.playing = False
        self.speed = 1.0
        self.pan, self.fov = 0.0, K.DEFAULT_FOV
        self.hold = False
        self.rec = False
        self.show_preview = True
        self.show_ghosts = True
        self.show_help = False
        self.mouse = (-1, -1)
        self.drag: Optional[str] = None
        self._grab_dlon = 0.0
        self._pending_idx: Optional[int] = None
        self._last_scrub = 0.0
        self._play_wall0 = 0.0
        self._play_idx0 = 0
        self._rec_last_t = 0.0
        self._last_edit = -10.0
        self._toast = ("", 0.0)
        self._path_cache = (None, None)
        self._curve_cache = (None, None)
        self._pv_cache = (None, None, None)
        self._last_wheel = -1.0
        self._canvas = np.zeros((self.size[1], self.size[0], 3), np.uint8)
        self.dirty = True

        self._path_obj()
        self._sync_to_path()
        resume = None
        if start is not None:
            resume = start
        elif "last_t" in meta:
            resume = float(meta["last_t"])
        if resume:
            self.idx = max(0, min(self.n - 1, int(round(resume * self.fps))))
            self._sync_to_path()
            if start is None:
                self.toast(f"resumed at {K.fmt_time(self.t)}", 2.5)

    # ── derived state ────────────────────────────────────────────────────────
    @property
    def t(self) -> float:
        return self.idx / self.fps

    def _path_obj(self) -> K.Path:
        key = (self.kfs.version, self.interp, round(self.base_tilt, 4))
        if self._path_cache[0] != key:
            self._path_cache = (key, K.Path(self.kfs.items, self.interp, base_tilt=self.base_tilt))
        return self._path_cache[1]

    def _tilt_of(self, kf: K.Keyframe) -> float:
        return self.base_tilt if kf.tilt_deg is None else kf.tilt_deg

    def kf_here(self) -> Optional[K.Keyframe]:
        return self.kfs.at(self.t)

    def toast(self, msg: str, secs: float = 1.6):
        self._toast = (msg, time.perf_counter() + secs)
        self.dirty = True

    # ── game start / end marks ───────────────────────────────────────────────
    def mark_trim(self, which: str):
        """'in' / 'out': set the mark here; pressing it again while parked on that mark clears it."""
        t = round(self.t, 3)
        cur = self.trim_in if which == "in" else self.trim_out
        if cur is not None and abs(cur - t) < 0.5 / self.fps:
            new = None
        else:
            new = t
            other = self.trim_out if which == "in" else self.trim_in
            if other is not None and ((which == "in" and t >= other) or (which == "out" and t <= other)):
                self.toast(f"{which.upper()} must be {'before' if which == 'in' else 'after'} "
                            f"{'OUT' if which == 'in' else 'IN'} ({K.fmt_time(other)})", 2.5)
                return
        if which == "in":
            self.trim_in = new
        else:
            self.trim_out = new
        self._meta_dirty = True
        self.toast(f"{which.upper()} cleared" if new is None else f"{which.upper()} set at {K.fmt_time(new)}", 2.0)

    # ── navigation ───────────────────────────────────────────────────────────
    def _sync_to_path(self):
        path = self._path_obj()
        if path:
            self.pan, self.fov = path.at(self.t)
            self.tilt = path.tilt_at(self.t)

    def goto(self, idx: int):
        idx = max(0, min(self.n - 1, int(idx)))
        if idx == self.idx:
            return
        self._last_edit = -10.0                 # moving ends the current undo group
        self.idx = idx
        if not self.hold:
            self._sync_to_path()
        if self.playing:
            self._play_wall0, self._play_idx0 = time.perf_counter(), self.idx
        self._rec_last_t = self.t if not self.playing else self._rec_last_t
        self.dirty = True

    def step(self, n: int):
        self.playing = False
        self.goto(self.idx + n)

    def jump(self, seconds: float):
        self.goto(self.idx + int(round(seconds * self.fps)))

    def jump_keyframe(self, direction: int):
        kf = self.kfs.next_after(self.t) if direction > 0 else self.kfs.prev_before(self.t)
        if kf is None:
            self.toast("no more keyframes")
            return
        self.goto(int(round(kf.t * self.fps)))

    def toggle_play(self):
        if self.playing:
            self.playing = False
            if self.rec and self.hold:
                self._rec_commit()
        else:
            if self.idx >= self.n - 1:
                self.idx = 0
            self.playing = True
            self._play_wall0, self._play_idx0 = time.perf_counter(), self.idx
            self._rec_last_t = self.t
        self.dirty = True

    def set_speed(self, s: float):
        self.speed = s
        if self.playing:
            self._play_wall0, self._play_idx0 = time.perf_counter(), self.idx
        self.toast(f"speed {s:g}x")

    # ── crop editing ─────────────────────────────────────────────────────────
    def _begin_edit(self, now: Optional[float] = None, force: bool = False):
        """Open an undo step for live-keyframe edits; consecutive edits within 1 s
        (a drag, a run of nudges) share one step."""
        now = time.perf_counter() if now is None else now
        if self.kf_here() is not None and not self.playing and (force or now - self._last_edit > 1.0):
            self.kfs.snapshot()
        self._last_edit = now

    def _kf_tilt(self) -> Optional[float]:
        """The tilt a keyframe made from the current view should carry: None (= follow the
        project tilt) unless this view was deliberately tilted away from it."""
        return None if abs(self.tilt - self.base_tilt) < 0.01 else round(self.tilt, 3)

    def set_crop(self, pan: Optional[float] = None, fov: Optional[float] = None):
        if pan is not None:
            self.pan = float(K.wrap180(pan))
        if fov is not None:
            self.fov = float(min(self.fov_max, max(K.FOV_MIN, fov)))
        kf = self.kf_here()
        if kf is not None and not self.playing:
            self.kfs.upsert(K.Keyframe(kf.t, self.pan, self.fov, self._kf_tilt()))      # live edit
            self.hold = False
        else:
            self.hold = True            # uncommitted until K (or REC) commits it
        self.dirty = True

    def nudge_lon(self, dlon: float):
        self._begin_edit()
        self.set_crop(pan=self.pan + K.PAN_SIGN * dlon)

    def zoom(self, factor: float):
        self._begin_edit()
        self.set_crop(fov=self.fov * factor)

    def set_tilt(self, tilt: float):
        """W / S: change the project tilt. The view follows unless it carries its own tilt."""
        old = self.base_tilt
        self.base_tilt = float(max(-60.0, min(60.0, tilt)))
        if abs(self.tilt - old) < 0.01:
            self.tilt = self.base_tilt
        self._meta_dirty = True
        self.dirty = True

    def set_view_tilt(self, tilt: float):
        """T / B / X: tilt only this view. On a keyframe it edits that keyframe; elsewhere it is
        held (orange) until K or REC commits it, exactly like a moved crop."""
        self._begin_edit()
        self.tilt = float(max(-60.0, min(60.0, tilt)))
        kf = self.kf_here()
        if kf is not None and not self.playing:
            self.kfs.upsert(K.Keyframe(kf.t, self.pan, self.fov, self._kf_tilt()))
            self.hold = False
        else:
            self.hold = True
        self.dirty = True

    def toggle_keyframe(self):
        self._last_edit = -10.0                 # a discrete action ends the current undo group
        t = self.idx / self.fps
        kf = self.kfs.at(t)
        if kf is not None and not self.hold:
            self.kfs.snapshot()
            self.kfs.remove_at(t)
            self._sync_to_path()
            self.toast("keyframe removed")
        else:
            self.kfs.snapshot()
            self.kfs.upsert(K.Keyframe(t, self.pan, self.fov, self._kf_tilt()))
            self.toast("keyframe updated" if kf is not None else f"keyframe #{len(self.kfs)} added")
        self.hold = False
        self.dirty = True

    def snap_to_path(self):
        self.hold = False
        self._sync_to_path()
        self.toast("snapped to path")
        self.dirty = True

    def _rec_commit(self):
        t = self.t
        gap = t - self._rec_last_t
        if 0 < gap <= max(4 * self.rec_interval, 2.0):
            self.kfs.remove_between(self._rec_last_t, t)       # overwrite what we passed over
        self.kfs.upsert(K.Keyframe(t, self.pan, self.fov, self._kf_tilt()))
        self.hold = False
        self._rec_last_t = t

    def toggle_rec(self):
        self.rec = not self.rec
        if self.rec:
            self.kfs.snapshot()                                 # whole take = one undo step
            self._rec_last_t = self.t
        self.toast("REC on: steer while playing" if self.rec else "REC off")
        self.dirty = True

    def undo(self):
        self._last_edit = -10.0
        if self.kfs.undo():
            self.hold = False
            self._sync_to_path()
            self.toast("undo")
        else:
            self.toast("nothing to undo")
        self.dirty = True

    def redo(self):
        self._last_edit = -10.0
        if self.kfs.redo():
            self.hold = False
            self._sync_to_path()
            self.toast("redo")
        else:
            self.toast("nothing to redo")
        self.dirty = True

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, force: bool = False, now: Optional[float] = None) -> bool:
        now = time.perf_counter() if now is None else now
        changed = self.kfs.version != self._saved_version
        if not (changed or self._meta_dirty or force):
            return False
        if not force and now - self._last_save < 0.35:
            return False
        if changed or force or not os.path.exists(self.kf_path):
            self.kfs.save(self.kf_path)
            self._saved_version = self.kfs.version
        # keep what other tools put in the meta file (iv_fov, lens, ...): the render needs them
        meta = dict(K.load_meta(self.kf_path))
        meta.update(tilt_deg=round(self.base_tilt, 3), legacy_fov=self.legacy_fov, projection=self.projection,
                    fov_max=self.fov_max,
                    interp=self.interp, video=os.path.basename(self.video), fps=self.fps,
                    last_t=round(self.t, 3), trim_in=self.trim_in, trim_out=self.trim_out)
        K.save_meta(self.kf_path, **meta)
        self._meta_dirty = False
        self._last_save = now
        return True

    # ── per-iteration update ─────────────────────────────────────────────────
    def process(self, now: Optional[float] = None) -> bool:
        """Advance playback / scrubbing / autosave. Returns True if a redraw is due."""
        now = time.perf_counter() if now is None else now
        if self._pending_idx is not None and now - self._last_scrub > 0.04:
            idx, self._pending_idx = self._pending_idx, None
            self._last_scrub = now
            self.goto(idx)
        if self.playing:
            target = self._play_idx0 + int((now - self._play_wall0) * self.fps * self.speed)
            if target >= self.n - 1:
                target = self.n - 1
                self.playing = False
            if target != self.idx:
                self.idx = target
                if not self.hold:
                    self._sync_to_path()
                if self.rec and self.hold and self.t - self._rec_last_t >= self.rec_interval:
                    self._rec_commit()
                self.dirty = True
            if not self.playing and self.rec and self.hold:
                self._rec_commit()
        if self._toast[0] and now > self._toast[1]:
            self._toast = ("", 0.0)
            self.dirty = True
        self.save(now=now)
        d, self.dirty = self.dirty, False
        return d

    # ── input ────────────────────────────────────────────────────────────────
    def handle_key(self, code: int) -> bool:
        """Returns False when the user wants to quit."""
        if code in (-1, 255):
            return True
        if code == 27:
            return False
        name = ARROW_CODES.get(code)
        ch = chr(code) if name is None and 0 <= code < 128 else ""
        self.dirty = True
        if ch == " ":
            self.toggle_play()
        elif ch == "a" or name == "left":
            self.step(-1)
        elif ch == "d" or name == "right":
            self.step(1)
        elif ch == "q":
            self.jump(-5)
        elif ch == "e":
            self.jump(5)
        elif ch == "Q":
            self.jump(-10)
        elif ch == "E":
            self.jump(10)
        elif name == "up":
            self.jump(5)
        elif name == "down":
            self.jump(-5)
        elif ch in self.SPEEDS:
            self.set_speed(self.SPEEDS[ch])
        elif ch == "k":
            self.toggle_keyframe()
        elif ch == ",":
            self.jump_keyframe(-1)
        elif ch == ".":
            self.jump_keyframe(1)
        elif ch == "j":
            self.nudge_lon(-1)
        elif ch == "l":
            self.nudge_lon(1)
        elif ch == "J":
            self.nudge_lon(-5)
        elif ch == "L":
            self.nudge_lon(5)
        elif ch == "i":
            self.zoom(0.96)
        elif ch == "o":
            self.zoom(1 / 0.96)
        elif ch == "w":
            self.set_tilt(self.base_tilt + 0.5)
        elif ch == "s":
            self.set_tilt(self.base_tilt - 0.5)
        elif ch == "t":
            self.set_view_tilt(self.tilt + 0.5)
        elif ch == "b":
            self.set_view_tilt(self.tilt - 0.5)
        elif ch == "x":
            self.set_view_tilt(self.base_tilt)
            self.toast("tilt back to project tilt")
        elif ch == "[":
            self.mark_trim("in")
        elif ch == "]":
            self.mark_trim("out")
        elif ch == "g":
            self.snap_to_path()
        elif ch == "r":
            self.toggle_rec()
        elif ch == "z":
            self.undo()
        elif ch == "y":
            self.redo()
        elif ch == "f":
            self.show_preview = not self.show_preview
        elif ch == "v":
            self.show_ghosts = not self.show_ghosts
        elif ch in ("h", "H", "?"):
            self.show_help = not self.show_help
        elif ch == "S":
            self.save(force=True)
            self.toast(f"saved {len(self.kfs)} keyframes")
        else:
            self.dirty = False
        return True

    def _tl_x(self, t: float) -> float:
        return PAD + (t / self.duration if self.duration > 0 else 0.0) * (self.disp_w - 2 * PAD)

    def _tl_idx(self, x: float) -> int:
        f = (x - PAD) / max(1, self.disp_w - 2 * PAD)
        return int(round(min(1.0, max(0.0, f)) * (self.n - 1)))

    def _mouse_lonlat(self, x, y):
        lon = (x / self.disp_w - 0.5) * 360.0
        lat = (0.5 - y / self.eq_h) * 180.0
        return lon, lat

    def on_mouse(self, ev, x, y, flags, _param=None):
        self.mouse = (x, y)
        in_eq = 0 <= y < self.eq_h
        in_tl = self.eq_h <= y < self.eq_h + TL_H
        if ev == cv2.EVENT_LBUTTONDOWN:
            if in_eq:
                self._begin_edit(force=True)
                lon, lat = self._mouse_lonlat(x, y)
                c_lon, _ = K.centre_lonlat(self.pan, self.tilt)
                inside = K.point_in_crop(lon, lat, self.pan, self.tilt, self.fov,
                                         self.out_w, self.out_h, self.legacy_fov, self.projection)
                self._grab_dlon = float(K.wrap180(lon - c_lon)) if inside else 0.0
                self.drag = "crop"
                self._drag_crop(x)
            elif in_tl:
                self.drag = "scrub"
                self._pending_idx = self._tl_idx(x)
                self._last_scrub = 0.0
                self.dirty = True
        elif ev == cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_LBUTTON and self.drag == "crop":
                self._drag_crop(x)
            elif flags & cv2.EVENT_FLAG_LBUTTON and self.drag == "scrub":
                self._pending_idx = self._tl_idx(x)
                self.dirty = True
            elif in_tl or self.drag:
                self.dirty = True
        elif ev == cv2.EVENT_LBUTTONUP:
            if self.drag == "scrub":
                self._pending_idx = self._tl_idx(x)
                self._last_scrub = 0.0
            self.drag = None
            self.dirty = True
        elif ev == getattr(cv2, "EVENT_MOUSEWHEEL", 10) and in_eq:
            try:
                delta = cv2.getMouseWheelDelta(flags)
            except AttributeError:
                delta = ctypes_signed16(flags >> 16)
            now = time.perf_counter()
            if delta and now - self._last_wheel >= 0.04:     # trackpads fire bursts of events
                self._last_wheel = now
                self.zoom(0.99 if delta > 0 else 1 / 0.99)

    def _drag_crop(self, x):
        lon, _ = self._mouse_lonlat(x, 0)
        centre = lon - self._grab_dlon
        self.set_crop(pan=K.PAN_SIGN * float(K.wrap180(centre)))

    # ── rendering ────────────────────────────────────────────────────────────
    def _outline_pts(self, pan, fov, tilt=None):
        lon, lat = K.crop_outline(pan, self.tilt if tilt is None else tilt, fov, self.out_w, self.out_h, self.legacy_fov, n=18,
                                 projection=self.projection)
        x, y = K.lonlat_to_xy(lon, lat, self.disp_w, self.eq_h)
        return np.stack([x, y], 1)

    def _draw_outline(self, img, pan, fov, color, thick=2, tilt=None):
        pts = self._outline_pts(pan, fov, tilt)
        for shift in (-self.disp_w, 0, self.disp_w):
            p = np.round(pts + [shift, 0]).astype(np.int32)
            if p[:, 0].max() < 0 or p[:, 0].min() > self.disp_w:
                continue
            cv2.polylines(img, [p.reshape(-1, 1, 2)], True, color, thick, cv2.LINE_AA)

    def _crop_color(self):
        if self.hold:
            return C_HOLD
        if self.kf_here() is not None:
            return C_LIVE
        return C_CROP

    def _preview_maps(self, pw: int, ph: int):
        key = (round(self.pan, 3), round(self.fov, 3), round(self.tilt, 3), pw, ph, self.legacy_fov, self.projection)
        if self._pv_cache[0] != key:
            mx, my = K.build_remap(self.pan, self.tilt, self.fov, self.disp_w, self.eq_h, pw, ph,
                                   self.legacy_fov, step=4, projection=self.projection)
            self._pv_cache = (key, mx, my)
        return self._pv_cache[1], self._pv_cache[2]

    def _render_equirect(self, frame: Optional[np.ndarray], out: Optional[np.ndarray] = None) -> np.ndarray:
        W, H = self.disp_w, self.eq_h
        if frame is None:
            frame = np.zeros((H, W, 3), np.uint8)
            put(frame, "cannot decode this frame", (20, 40))
        mask = np.zeros((H, W), np.uint8)
        pts = self._outline_pts(self.pan, self.fov)
        for shift in (-W, 0, W):
            p = np.round(pts + [shift, 0]).astype(np.int32)
            if p[:, 0].max() < 0 or p[:, 0].min() > W:
                continue
            cv2.fillPoly(mask, [p.reshape(-1, 1, 2)], 255)
        img = out if out is not None else np.empty((H, W, 3), np.uint8)
        cv2.convertScaleAbs(frame, alpha=0.5, dst=img)            # dim everything ...
        cv2.copyTo(frame, mask, img)                              # ... except inside the crop

        if self.show_ghosts:
            prev = self.kfs.prev_before(self.t)
            nxt = self.kfs.next_after(self.t)
            if prev is not None:
                self._draw_outline(img, prev.pan_deg, prev.fov_deg, C_PREV, 1, self._tilt_of(prev))
            if nxt is not None:
                self._draw_outline(img, nxt.pan_deg, nxt.fov_deg, C_NEXT, 1, self._tilt_of(nxt))
            if self.hold:
                path = self._path_obj()
                if path:
                    pp, pf = path.at(self.t)
                    self._draw_outline(img, pp, pf, (200, 200, 200), 1, path.tilt_at(self.t))
        col = self._crop_color()
        self._draw_outline(img, self.pan, self.fov, col, 2)
        c_lon, c_lat = K.centre_lonlat(self.pan, self.tilt)
        cx, cy = K.lonlat_to_xy(c_lon, c_lat, W, H)
        for shift in (-W, 0, W):
            x = int(round(float(cx))) + shift
            if -10 <= x <= W + 10:
                cv2.drawMarker(img, (x, int(round(float(cy)))), col, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)

        if self.show_preview:
            pw = int(W * 0.32)
            ph = int(round(pw * self.out_h / self.out_w))
            mx, my = self._preview_maps(pw, ph)
            pv = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            x0, y0 = W - pw - 8, 8
            img[y0:y0 + ph, x0:x0 + pw] = pv
            cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + pw, y0 + ph), col, 2)
            put(img, "output", (x0 + 6, y0 + 18), 0.45)

        if self.show_help:
            h = 28 * len(HELP_LINES) + 24
            x1, y1 = min(W - 20, 760), 20 + h
            roi = img[20:y1, 20:x1]
            roi[:] = cv2.addWeighted(roi, 0.22, np.zeros_like(roi), 0.78, 0)    # darken the panel area
            for i, line in enumerate(HELP_LINES):
                put(img, line, (34, 52 + 28 * i), 0.55)
        return img

    def _nice_step(self, px_per_s: float) -> float:
        for s in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600):
            if s * px_per_s >= 90:
                return float(s)
        return 3600.0

    def _curve(self, width: int):
        key = (self.kfs.version, self.interp, width)
        if self._curve_cache[0] != key:
            path = self._path_obj()
            if path:
                ts = np.linspace(0, self.duration, width)
                pan, _ = path.at(ts)
            else:
                pan = None
            self._curve_cache = (key, pan)
        return self._curve_cache[1]

    def _render_timeline(self) -> np.ndarray:
        W = self.disp_w
        img = np.full((TL_H, W, 3), 26, np.uint8)
        x0, x1 = PAD, W - PAD
        bar_t, bar_b = 6, TL_H - 22                       # main track
        cv2.rectangle(img, (x0, bar_t), (x1, bar_b), (40, 40, 40), -1)

        # coverage warning band under the track
        if self.gap_warn > 0:
            times = [0.0] + [k.t for k in self.kfs] + [self.duration]
            if len(self.kfs) == 0:
                times = [0.0, self.duration]
            for a, b in zip(times[:-1], times[1:]):
                if b - a > self.gap_warn:
                    xa, xb = int(self._tl_x(a)), int(self._tl_x(b))
                    cv2.rectangle(img, (xa, bar_b + 2), (max(xa + 1, xb), bar_b + 7), (40, 40, 200), -1)

        pan = self._curve(x1 - x0 + 1)
        if pan is not None:
            lim = max(30.0, math.ceil(float(np.max(np.abs(pan))) / 10.0) * 10.0)
            mid = (bar_t + bar_b) / 2
            ys = mid - pan / lim * (bar_b - bar_t - 6) / 2
            xs = np.arange(x0, x1 + 1)
            line = np.stack([xs, ys], 1).round().astype(np.int32)
            cv2.polylines(img, [line.reshape(-1, 1, 2)], False, (170, 170, 170), 1, cv2.LINE_AA)
            put(img, f"pan +/-{lim:.0f}", (x0 + 4, bar_t + 13), 0.4, (150, 150, 150))

        here = self.kf_here()
        for k in self.kfs:
            x = int(round(self._tl_x(k.t)))
            col = C_LIVE if (here is not None and k is here) else C_KF
            cv2.line(img, (x, bar_t), (x, bar_b), col, 1)
            tri = np.array([[x - 4, bar_t], [x + 4, bar_t], [x, bar_t + 7]], np.int32)
            cv2.fillConvexPoly(img, tri, col)
            if k.tilt_deg is not None:                       # this keyframe carries its own tilt
                cv2.circle(img, (x, bar_b - 5), 3, (255, 255, 0), -1, cv2.LINE_AA)

        # outside the game (before IN / after OUT): dimmed, with a marker at each mark
        for lo, hi in ((0.0, self.trim_in), (self.trim_out, self.duration)):
            if lo is not None and hi is not None and hi > lo:
                xa, xb = int(self._tl_x(lo)), int(self._tl_x(hi))
                sub = img[bar_t:bar_b + 1, xa:xb + 1]
                sub[:] = (sub * 0.35).astype(np.uint8)
        for tm, lab in ((self.trim_in, "IN"), (self.trim_out, "OUT")):
            if tm is not None:
                x = int(round(self._tl_x(tm)))
                cv2.line(img, (x, bar_t), (x, bar_b + 7), C_TRIM, 2)
                put(img, lab, (x + 3 if lab == "IN" else x - 3 - text_w(lab, 0.4), bar_b + 6), 0.4, C_TRIM)

        # ticks
        step = self._nice_step((x1 - x0) / max(self.duration, 1e-6))
        tt = 0.0
        while tt <= self.duration:
            x = int(round(self._tl_x(tt)))
            cv2.line(img, (x, bar_b + 8), (x, bar_b + 12), (130, 130, 130), 1)
            lab = f"{int(tt // 3600)}:{int(tt % 3600 // 60):02d}:{int(tt % 60):02d}" if self.duration >= 3600 \
                else f"{int(tt // 60)}:{int(tt % 60):02d}"
            put(img, lab, (x + 3, TL_H - 3), 0.38, (150, 150, 150))
            tt += step

        # hover line
        mx, my = self.mouse
        if self.eq_h <= my < self.eq_h + TL_H and x0 <= mx <= x1:
            cv2.line(img, (mx, 0), (mx, bar_b), (200, 200, 200), 1)
            lab = K.fmt_time(self._tl_idx(mx) / self.fps)
            put(img, lab, (min(mx + 6, W - 90), bar_b + 8 + 0), 0.45)
        xp = int(round(self._tl_x(self.t)))
        cv2.line(img, (xp, 0), (xp, bar_b + 3), (60, 60, 255), 2)
        return img

    def _render_hud(self) -> np.ndarray:
        W = self.disp_w
        img = np.full((HUD_H, W, 3), 18, np.uint8)
        state = "REC " if (self.rec and self.playing) else ("PLAY" if self.playing else "PAUSE")
        scol = (60, 60, 255) if self.rec and self.playing else ((120, 255, 120) if self.playing else (200, 200, 200))
        put(img, state, (PAD, 22), 0.6, scol, 2)
        kf = self.kf_here()
        idxs = ""
        if kf is not None:
            idxs = f"   KEYFRAME #{self.kfs.index_at(self.t) + 1}"
        line1 = (f"{self.speed:g}x   t {K.fmt_time(self.t)} / {K.fmt_time(self.duration)}"
                 f"   frame {self.idx}/{self.n - 1}   keyframes {len(self.kfs)}{idxs}")
        if self.trim_in is not None or self.trim_out is not None:
            line1 += (f"   game {K.fmt_time(self.trim_in) if self.trim_in is not None else 'start'}"
                      f" - {K.fmt_time(self.trim_out) if self.trim_out is not None else 'end'}")
        put(img, line1, (PAD + 80, 22), 0.55)
        vf = K.vfov_deg(self.fov, self.out_w, self.out_h, self.legacy_fov, self.projection)
        side = "left" if self.pan > 0.05 else ("right" if self.pan < -0.05 else "centre")
        tl_tag = "" if abs(self.tilt - self.base_tilt) < 0.01 else f" (project {self.base_tilt:+.1f})"
        line2 = f"pan {self.pan:+6.1f} deg ({side})   fov {self.fov:5.1f} (v {vf:4.1f})   tilt {self.tilt:+5.1f}{tl_tag}   {self.projection}"
        put(img, line2, (PAD, 48), 0.55)
        x = PAD + text_w(line2) + 24
        if self.hold:
            put(img, "HOLD - K to commit, G to discard", (x, 48), 0.55, C_HOLD, 1)
        elif kf is not None:
            put(img, "editing keyframe live", (x, 48), 0.55, C_LIVE, 1)
        msg, until = self._toast
        if msg:
            put(img, msg, (W - PAD - text_w(msg, 0.6, 2), 22), 0.6, (0, 255, 255), 2)
        hint = "H = help"
        put(img, hint, (W - PAD - text_w(hint), 48), 0.5, (150, 150, 150))
        return img

    def render(self) -> np.ndarray:
        """Draw the whole window into one preallocated buffer (returned; copy it if you
        need to keep it past the next call)."""
        frame = self.src.get(self.idx)
        if frame is None and self.playing:
            self.playing = False
        cv = self._canvas
        self._render_equirect(frame, out=cv[:self.eq_h])
        cv[self.eq_h:self.eq_h + TL_H] = self._render_timeline()
        cv[self.eq_h + TL_H:] = self._render_hud()
        return cv

    def render_command(self) -> str:
        return (f"python render_keyframes.py --input <raw or equirect video> "
                f"--keyframes {self.kf_path} --output game.mp4")


def ctypes_signed16(v: int) -> int:
    v &= 0xFFFF
    return v - 0x10000 if v & 0x8000 else v


# ─────────────────────────────────────────────────────────────────────────────
# Window loop
# ─────────────────────────────────────────────────────────────────────────────

def run(app: Annotator) -> None:
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, app.on_mouse)
    cv2.imshow(WIN, app.render())
    try:
        while True:
            if app.process():
                cv2.imshow(WIN, app.render())
            key = cv2.waitKeyEx(1 if app.playing else 12)
            if key != -1 and not app.handle_key(key):
                break
            try:
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        app.save(force=True)
        app.src.close()
        cv2.destroyAllWindows()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="equirectangular (2:1) video; make one with make_proxy.py")
    p.add_argument("--keyframes", default=None, help="keyframe JSON (default: <video>.keyframes.json)")
    p.add_argument("--tilt", type=float, default=None, help="vertical aim in degrees (+ = up)")
    p.add_argument("--calib", default=None, help="pitch.json; derive tilt like autopan_infer.py")
    p.add_argument("--width", type=int, default=1440, help="window width in px (default 1440)")
    p.add_argument("--interp", choices=K.Path.METHODS, default="pchip")
    p.add_argument("--legacy-fov", action="store_true",
                   help="vertical FOV == horizontal FOV, like py360convert's scalar fov (anamorphic)")
    p.add_argument("--projection", choices=K.PROJECTIONS, default=None,
                   help="output view: flat (rectilinear) or cylindrical (straight verticals, far less "
                        "edge stretch at wide fov). Default: the saved choice, else cylindrical")
    p.add_argument("--fov-max", type=float, default=None,
                   help=f"widest zoom-out the editor allows, in degrees (default {K.FOV_MAX:g}, saved with the project; "
                        f"at most {K.FOV_HARD_MAX:g}). The lens covers about +/-100 deg, so very wide crops show the blank edges")
    p.add_argument("--gap-warn", type=float, default=20.0,
                   help="flag stretches without a keyframe longer than this many seconds (0 = off)")
    p.add_argument("--rec-interval", type=float, default=0.5, help="REC keyframe spacing in seconds")
    p.add_argument("--start", type=float, default=None, help="start at this many seconds")
    a = p.parse_args(argv)

    tilt = a.tilt                       # None -> use the saved value, else 0
    if tilt is None and a.calib:
        tilt = K.tilt_from_calib(a.calib)
    app = Annotator(a.video, a.keyframes, tilt=tilt, disp_w=a.width, legacy_fov=a.legacy_fov, projection=a.projection,
                    interp=a.interp, gap_warn=a.gap_warn, rec_interval=a.rec_interval, start=a.start,
                    fov_max=a.fov_max)
    print(f"{a.video}: {app.n} frames @ {app.fps:.3f} fps ({K.fmt_time(app.duration)})")
    print(f"keyframes -> {app.kf_path}  ({len(app.kfs)} loaded)   tilt {app.tilt:+.1f} deg   press H for keys")
    run(app)
    print(f"saved {len(app.kfs)} keyframes to {app.kf_path}")
    print("render with:  " + app.render_command())
    return 0


if __name__ == "__main__":
    sys.exit(main())
