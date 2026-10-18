#!/usr/bin/env python3
"""
auto_keyframes.py — turn extracted detections into a keyframes file you can open in annotate.py.

    python auto_keyframes.py --detections game.detections.jsonl --out game.auto.keyframes.json \
        [--compare my_annotation.keyframes.json]

The detections come from extract_detections.py. Because this runs offline it can look ahead,
which a live controller can't: every moment is smoothed with the moments before AND after it,
so the camera starts moving *before* the play gets there instead of chasing it.

Steps
  1. Where is the action?   Per sample, the confidence-weighted average longitude of the players
                            (the simple rule that follows human operators best on real footage).
                            Optional --use-ball blends in the ball, after throwing away fixed
                            "balls" (floodlights, marks) and anything away from the players.
  2. Hold still when it's fine: a dead-band ("backlash") so small drifts don't move the camera.
  3. Smooth, with look-ahead: a smoothing spline (second-difference penalty) over the whole file.
  4. Zoom: the spread of the players sets a slow zoom (default 0.36 * spread + 90 deg, from your
                            annotated footage; clamp --fov-min/--fov-max, or fix it with --fov).
  5. Keep the picture inside the lens: the crop never reaches past --edge-limit degrees.
  6. Tilt (only with --adaptive-tilt): dip the view when the nearest players' feet would fall
                            out of the bottom of the frame, and come back up afterwards. Without the
                            flag the output carries no tilt at all, exactly as before.
  7. Simplify to a few keyframes (a pchip path through them reproduces the smooth path to within
                            --tol-pan / --tol-fov), so there is something you can actually edit.

Settings: every knob below has a default; `--profile my.profile.json` loads a set fitted to your own
annotations (see tune_auto.py), and any flag you also give on the command line wins over the profile.

Writes <out> (+ the .meta.json sidecar annotate.py and render_keyframes.py read: tilt, cylindrical).
--compare prints how closely the result follows an annotation you made by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

import kf_core as K


# ── detections ───────────────────────────────────────────────────────────────

def load_detections(path: str, max_lon: Optional[float] = None) -> Tuple[dict, List[dict]]:
    """header, frames. With max_lon, players beyond +/-max_lon degrees are dropped: that is outside the
    lens's usable range (about +/-100), and a fixed phantom there ("player" at -110 deg in most frames of
    the Altenberg games) drags the action centre sideways."""
    with open(path) as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    if not rows or rows[0].get("type") != "header":
        raise SystemExit(f"{path}: not a detections file (no header line)")
    frames = rows[1:]
    if max_lon is not None:
        for r in frames:
            r["players"] = [p for p in r["players"] if abs(p[0]) <= max_lon]
    return rows[0], frames


def static_cells(frames: List[dict], cell: float = 2.0, frac: float = 0.2) -> set:
    """2-degree cells in which a 'ball' shows up in more than `frac` of the frames: those are
    fixed objects (floodlights, line marks), not a ball."""
    counts: dict = {}
    for r in frames:
        seen = {(round(b[0] / cell), round(b[1] / cell)) for b in r["balls"]}
        for c in seen:
            counts[c] = counts.get(c, 0) + 1
    n = max(len(frames), 1)
    return {c for c, k in counts.items() if k / n > frac}


def _near_static(b, cells, cell=2.0) -> bool:
    cx, cy = round(b[0] / cell), round(b[1] / cell)
    return any((cx + dx, cy + dy) in cells for dx in (-1, 0, 1) for dy in (-1, 0, 1))


def frame_features(r: dict, cells: set, lat_band: Tuple[float, float], min_conf: float,
                   use_ball: bool, ball_weight: float, ball_conf: float, ball_reach: float):
    """-> (lon of the action or nan, spread p5..p95 in degrees or nan, number of players)."""
    P = [p for p in r["players"] if p[3] >= min_conf]
    if not P:
        return np.nan, np.nan, 0
    x = np.array([p[0] for p in P])
    w = np.array([p[3] for p in P])
    lon = float((x * w).sum() / w.sum())
    spread = float(np.percentile(x, 95) - np.percentile(x, 5)) if len(x) >= 4 else np.nan
    if use_ball:
        cand = [b for b in r["balls"] if b[2] >= ball_conf and not _near_static(b, cells)
                and lat_band[0] <= b[1] <= lat_band[1] and abs(b[0] - lon) <= ball_reach]
        if cand:
            b = max(cand, key=lambda b: b[2])
            lon = (1 - ball_weight) * lon + ball_weight * b[0]
    return lon, spread, len(P)


# ── signal processing ────────────────────────────────────────────────────────

def fill_gaps(y: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation over missing values; beyond `max_gap` samples the last value is held."""
    y = y.copy()
    ok = ~np.isnan(y)
    if not ok.any():
        return np.zeros_like(y)
    idx = np.arange(len(y))
    filled = np.interp(idx, idx[ok], y[ok])
    # long gaps: hold the value from before the gap instead of drawing a slow line across it
    last_ok = -10 ** 9
    for i in range(len(y)):
        if ok[i]:
            last_ok = i
        elif i - last_ok > max_gap and last_ok >= 0:
            filled[i] = y[last_ok]
    return filled


def backlash(y: np.ndarray, band: float) -> np.ndarray:
    """Dead-band: the output only moves once the input is more than `band` away from it."""
    if band <= 0:
        return y.copy()
    out = np.empty_like(y)
    h = y[0]
    for i, v in enumerate(y):
        if v - h > band:
            h = v - band
        elif h - v > band:
            h = v + band
        out[i] = h
    return out


def smooth_series(y: np.ndarray, rate: float, seconds: float) -> np.ndarray:
    """Smoothing spline: minimise |x - y|^2 + lam |D2 x|^2 over the whole series (look-ahead and
    look-behind alike). `seconds` is the period below which wiggles are removed."""
    n = len(y)
    if n < 4 or seconds <= 0:
        return y.copy()
    lam = (seconds * rate / (2 * np.pi)) ** 4
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    A = (sparse.identity(n, format="csc") + lam * (D.T @ D)).tocsc()
    return np.asarray(spsolve(A, y))


def _kf_tilt(tilt: Optional[np.ndarray], i: int, base: float) -> Optional[float]:
    """Tilt a keyframe should carry: None when it is (nearly) the project tilt."""
    if tilt is None or abs(float(tilt[i]) - base) < 0.25:
        return None
    return float(tilt[i])


def simplify(t: np.ndarray, pan: np.ndarray, fov: np.ndarray, tol_pan: float, tol_fov: float,
             min_gap: float, method: str = "pchip", tilt: Optional[np.ndarray] = None,
             tol_tilt: float = 1.5, base_tilt: float = 0.0) -> List[int]:
    """Greedy keyframe picking: start with the two ends, repeatedly add the sample where the
    path through the chosen keyframes is furthest from the target, until all within tolerance.
    With `tilt`, the tilt path has to stay within `tol_tilt` degrees as well."""
    n = len(t)
    chosen = [0, n - 1] if n > 1 else [0]
    while True:
        items = [K.Keyframe(float(t[i]), float(pan[i]), float(fov[i]), _kf_tilt(tilt, i, base_tilt)) for i in chosen]
        path = K.Path(items, method, base_tilt=base_tilt)
        p, f = path.at(t) if len(items) > 1 else (np.full(n, pan[0]), np.full(n, fov[0]))
        err = np.maximum(np.abs(p - pan) / tol_pan, np.abs(f - fov) / tol_fov)
        if tilt is not None:
            tt = path.tilt_at(t) if len(items) > 1 else np.full(n, items[0].tilt_deg if items[0].tilt_deg is not None else base_tilt)
            err = np.maximum(err, np.abs(tt - tilt) / tol_tilt)
        for i in chosen:                                  # taken
            err[i] = 0
        ct = t[chosen]
        for i in range(n):                                # too close to an existing keyframe
            if err[i] > 0 and np.min(np.abs(ct - t[i])) < min_gap:
                err[i] = 0
        w = int(np.argmax(err))
        if err[w] <= 1.0:
            break
        chosen = sorted(chosen + [w])
    return chosen


# ── tilt ─────────────────────────────────────────────────────────────────────

def adaptive_tilt(frames: List[dict], centre: np.ndarray, fov: np.ndarray, rate: float, *,
                  base: float, projection: str, min_conf=0.3, margin=3.0, head_allow=5.0,
                  max_dip=20.0, smooth_s=3.0, max_gap=24) -> np.ndarray:
    """Per-sample tilt: the project tilt, dipping lower only while the feet of the players inside
    the crop would otherwise be cut off at the bottom of the frame.

    A player's foot point is at latitude `lat`; the crop shows tilt +/- vfov/2, so the feet are in
    when  tilt <= lat + vfov/2 - margin.  Dipping must not push the far players' heads
    (foot lat + head_allow) out of the top, which sets a floor. Never tilts above `base`,
    never dips more than `max_dip` degrees, and moves as smoothly as the pan."""
    n = len(frames)
    foot_lo = np.full(n, np.nan)
    foot_hi = np.full(n, np.nan)
    for i, r in enumerate(frames):
        half = fov[i] / 2.0
        lats = [p[1] for p in r["players"]
                if p[3] >= min_conf and p[4] is not False
                and abs(float(K.wrap180(p[0] - centre[i]))) <= half]
        if lats:
            foot_lo[i], foot_hi[i] = min(lats), max(lats)
    vh = np.array([K.vfov_deg(float(f), K.OUT_W, K.OUT_H, False, projection) for f in fov]) / 2.0
    want = np.minimum(base, foot_lo + vh - margin)              # lowest feet still in the frame
    floor = foot_hi + head_allow - vh                            # highest heads still in the frame
    want = np.clip(np.maximum(want, np.minimum(floor, base)), base - max_dip, base)   # nan stays nan
    if np.isnan(want).all():
        return np.full(n, float(base))
    want = fill_gaps(want, max_gap)                              # no players in the crop: hold
    want = np.where(base - want < 1.5, float(base), want)         # ignore dips too small to matter
    out = smooth_series(want, rate, smooth_s)
    return np.clip(out, base - max_dip, base)


def feet_coverage(frames: List[dict], t, pan, fov, tilt, projection: str, min_conf=0.3) -> float:
    """Share of the players that are inside the crop horizontally whose feet are also inside it
    vertically (1.0 = nobody is cut off at the top or bottom)."""
    hit = tot = 0
    for i, r in enumerate(frames):
        centre = K.PAN_SIGN * pan[i]
        for p in r["players"]:
            if p[3] < min_conf or p[4] is False:
                continue
            if abs(float(K.wrap180(p[0] - centre))) > fov[i] / 2.0:
                continue
            tot += 1
            hit += K.point_in_crop(p[0], p[1], float(pan[i]), float(tilt[i]), float(fov[i]),
                                   K.OUT_W, K.OUT_H, False, projection)
    return hit / tot if tot else float("nan")


# ── the whole pass ───────────────────────────────────────────────────────────

def lead_shift(y: np.ndarray, k: int) -> np.ndarray:
    """y[i] -> y[i + k]: with k > 0 the camera is `k` samples ahead of the action (anticipation)."""
    if k == 0 or len(y) == 0:
        return y
    k = int(np.clip(k, -(len(y) - 1), len(y) - 1))
    if k > 0:
        return np.concatenate([y[k:], np.full(k, y[-1])])
    return np.concatenate([np.full(-k, y[0]), y[:k]])


def build_path(header: dict, frames: List[dict], *, use_ball=False, min_conf=0.3, deadband=5.0,
               smooth_s=4.0, zoom_s=12.0, fov_fixed: Optional[float] = None, fov_a=0.36, fov_b=90.0,
               fov_min=90.0, fov_max=140.0, edge_limit=100.0, ball_weight=0.4, ball_conf=0.5,
               ball_reach=45.0, max_gap_s=8.0, adaptive_tilt_on=False, base_tilt=-15.0,
               projection="cylindrical", tilt_margin=3.0, max_dip=20.0, tilt_s=3.0, lead_s=0.0) -> dict:
    rate = float(header.get("fps", 3.0))
    t = np.array([r["t"] for r in frames], float)
    cells = static_cells(frames)
    lats = [p[1] for r in frames for p in r["players"] if p[3] >= min_conf]
    lat_band = (min(lats) - 8.0, max(lats) + 8.0) if lats else (-90.0, 90.0)
    feats = [frame_features(r, cells, lat_band, min_conf, use_ball, ball_weight, ball_conf, ball_reach)
             for r in frames]
    lon = np.array([f[0] for f in feats])
    spread = np.array([f[1] for f in feats])
    n_pl = np.array([f[2] for f in feats])
    max_gap = int(max_gap_s * rate)

    lon_f = fill_gaps(lon, max_gap)
    centre = smooth_series(backlash(lon_f, deadband), rate, smooth_s)

    if fov_fixed is not None:
        fov = np.full(len(t), float(fov_fixed))
    else:
        sp = fill_gaps(spread, max_gap)
        if np.isnan(spread).all():
            sp = np.full(len(t), 80.0)
        fov = np.clip(smooth_series(fov_a * sp + fov_b, rate, zoom_s), fov_min, fov_max)

    # keep the crop inside the part of the picture the lens covers
    room = np.maximum(edge_limit - fov / 2.0, 0.0)
    centre = np.clip(centre, -room, room)
    tilt = None
    if adaptive_tilt_on:
        tilt = adaptive_tilt(frames, centre, fov, rate, base=base_tilt, projection=projection,
                             min_conf=min_conf, margin=tilt_margin, max_dip=max_dip, smooth_s=tilt_s,
                             max_gap=max_gap)
    k = int(round(lead_s * rate))                         # anticipation: the camera runs `lead_s` ahead
    if k:
        centre, fov = lead_shift(centre, k), lead_shift(fov, k)
        if tilt is not None:
            tilt = lead_shift(tilt, k)
    pan = K.PAN_SIGN * centre                             # pan_deg = -(longitude of the crop centre)
    return {"t": t, "pan": pan, "fov": fov, "tilt": tilt, "lon_raw": lon, "spread": spread,
            "n_players": n_pl, "static_ball_cells": len(cells)}


def write_keyframes(path: str, t, pan, fov, idx: List[int], tilt: float, projection: str, video: str = "",
                    tilt_path: Optional[np.ndarray] = None, trim_in: Optional[float] = None,
                    trim_out: Optional[float] = None, iv_fov: Optional[float] = None) -> None:
    """`tilt` is the project tilt (meta); `tilt_path`, if given, adds a per-keyframe tilt wherever
    it differs from the project tilt. `iv_fov` (only when not the default) records the fisheye model
    the detections were made with, so render_keyframes.py unwraps the video the same way."""
    rows = []
    for i in idx:
        row = {"t": round(float(t[i]), 4), "pan_deg": round(float(pan[i]), 4), "fov_deg": round(float(fov[i]), 4)}
        tl = _kf_tilt(tilt_path, i, float(tilt))
        if tl is not None:
            row["tilt_deg"] = round(tl, 4)
        rows.append(row)
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=1)
        fh.write("\n")
    extra = {} if iv_fov is None else {"iv_fov": float(iv_fov)}
    K.save_meta(path, tilt_deg=round(float(tilt), 3), legacy_fov=False, projection=projection,
                interp="pchip", video=video, trim_in=trim_in, trim_out=trim_out,
                note="written by auto_keyframes.py", **extra)


def game_marks(header: dict) -> Tuple[Optional[float], Optional[float]]:
    """IN / OUT marks for the output: the stretch the detections were extracted from, when that
    was only part of the video (so render_keyframes.py renders the game, not the setup and take-down)."""
    start = float(header.get("start") or 0.0)
    dur = header.get("duration")
    vd = header.get("video_duration")
    t_in = round(start, 3) if start > 0.5 else None
    t_out = None
    if dur is not None and vd is not None and start + float(dur) < float(vd) - 0.5:
        t_out = round(start + float(dur), 3)
    return t_in, t_out


def compare(t, pan, fov, ref_path: str, lon_raw=None) -> dict:
    ref = K.Path(K.KeyframeSet.load(ref_path).items, "pchip")
    rp, rf = ref.at(t)
    ep = pan - rp
    out = {"pan_mae": float(np.mean(np.abs(ep))), "pan_p90": float(np.percentile(np.abs(ep), 90)),
           "pan_bias": float(np.mean(ep)), "fov_mae": float(np.mean(np.abs(fov - rf)))}
    # would the automatic crop still contain where the human's crop was centred?
    out["ref_centre_in_auto_crop"] = float(np.mean(np.abs(ep) <= fov / 2.0))
    if lon_raw is not None:
        ok = ~np.isnan(lon_raw)
        out["players_in_auto_crop"] = float(np.mean(np.abs(lon_raw[ok] - K.PAN_SIGN * pan[ok]) <= fov[ok] / 2.0))
    return out


PROFILE_KEYS = ("deadband", "smooth", "zoom_smooth", "fov_a", "fov_b", "fov_min", "fov_max", "edge_limit",
                "tilt", "tilt_margin", "max_dip", "tilt_smooth", "lead", "adaptive_tilt", "min_conf")


def load_profile(path: str) -> dict:
    """A profile is a JSON object whose keys are auto_keyframes.py option names (underscored)."""
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, ValueError) as e:
        raise SystemExit(f"cannot read profile {path}: {e}")
    if not isinstance(d, dict):
        raise SystemExit(f"{path}: a profile is a JSON object")
    d = d.get("settings", d)                       # tune_auto.py nests the settings next to its report
    bad = [k for k in d if k not in PROFILE_KEYS]
    if bad:
        print(f"profile  : ignoring unknown keys {bad}", file=sys.stderr)
    return {k: v for k, v in d.items() if k in PROFILE_KEYS}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detections", required=True)
    p.add_argument("--out", default=None, help="default: <detections>.auto.keyframes.json")
    p.add_argument("--compare", default=None, help="your own annotation, to measure how close the result is")
    p.add_argument("--tilt", type=float, default=None, help="output tilt (default: the one the detections used)")
    p.add_argument("--projection", default="cylindrical", choices=K.PROJECTIONS)
    p.add_argument("--use-ball", action="store_true", help="blend in plausible ball detections")
    p.add_argument("--min-conf", type=float, default=0.3, help="ignore players below this confidence")
    p.add_argument("--max-lon", type=float, default=98.0,
                   help="ignore players beyond +/- this longitude (outside the lens; 0 = keep all)")
    p.add_argument("--deadband", type=float, default=5.0, help="degrees the action may drift before the camera moves")
    p.add_argument("--smooth", type=float, default=4.0, help="pan smoothing period in seconds (larger = calmer)")
    p.add_argument("--zoom-smooth", type=float, default=12.0, help="zoom smoothing period in seconds")
    p.add_argument("--fov", type=float, default=None, help="fixed field of view instead of the adaptive one")
    p.add_argument("--fov-a", type=float, default=0.36, help="zoom: fov = a * player spread + b")
    p.add_argument("--fov-b", type=float, default=90.0)
    p.add_argument("--fov-min", type=float, default=90.0)
    p.add_argument("--fov-max", type=float, default=140.0)
    p.add_argument("--edge-limit", type=float, default=100.0, help="never show beyond this longitude (lens coverage)")
    p.add_argument("--tol-pan", type=float, default=1.5, help="keyframe simplification tolerance (deg)")
    p.add_argument("--tol-fov", type=float, default=3.0)
    p.add_argument("--min-gap", type=float, default=1.0, help="minimum seconds between keyframes")
    p.add_argument("--adaptive-tilt", action="store_true",
                   help="dip the view (per-keyframe tilt) while the nearest players' feet would be cut off")
    p.add_argument("--max-dip", type=float, default=20.0, help="most the tilt may drop below --tilt (deg)")
    p.add_argument("--tilt-margin", type=float, default=3.0, help="degrees of room kept below the lowest feet")
    p.add_argument("--tilt-smooth", type=float, default=3.0, help="tilt smoothing period in seconds")
    p.add_argument("--tol-tilt", type=float, default=1.5, help="keyframe simplification tolerance for tilt (deg)")
    p.add_argument("--lead", type=float, default=0.0,
                   help="seconds the camera runs ahead of the action (anticipation; 0 = centred on it)")
    p.add_argument("--profile", default=None,
                   help="settings fitted by tune_auto.py; flags given on the command line override it")
    pre, _ = p.parse_known_args(argv)
    if pre.profile:
        prof = load_profile(pre.profile)
        known = {act.dest for act in p._actions}
        p.set_defaults(**{k: v for k, v in prof.items() if k in known})
    a = p.parse_args(argv)
    if a.profile:
        print(f"profile  : {a.profile}")

    header, frames = load_detections(a.detections, max_lon=a.max_lon if a.max_lon > 0 else None)
    if len(frames) < 4:
        raise SystemExit("too few frames in the detections file")
    tilt = a.tilt if a.tilt is not None else float(header.get("tilt", -15.0))
    res = build_path(header, frames, use_ball=a.use_ball, min_conf=a.min_conf, deadband=a.deadband,
                     smooth_s=a.smooth, zoom_s=a.zoom_smooth, fov_fixed=a.fov, fov_a=a.fov_a, fov_b=a.fov_b,
                     fov_min=a.fov_min, fov_max=a.fov_max, edge_limit=a.edge_limit,
                     adaptive_tilt_on=a.adaptive_tilt, base_tilt=tilt, projection=a.projection,
                     tilt_margin=a.tilt_margin, max_dip=a.max_dip, tilt_s=a.tilt_smooth, lead_s=a.lead)
    idx = simplify(res["t"], res["pan"], res["fov"], a.tol_pan, a.tol_fov, a.min_gap,
                   tilt=res["tilt"], tol_tilt=a.tol_tilt, base_tilt=tilt)
    out = a.out or os.path.splitext(a.detections)[0] + ".auto.keyframes.json"
    write_keyframes(out, res["t"], res["pan"], res["fov"], idx, tilt, a.projection, header.get("video", ""),
                    tilt_path=res["tilt"], trim_in=game_marks(header)[0], trim_out=game_marks(header)[1],
                    iv_fov=header.get("iv_fov"))

    t = res["t"]
    print(f"{len(frames)} samples over {t[-1] - t[0]:.0f}s -> {len(idx)} keyframes "
          f"(one per {(t[-1] - t[0]) / max(len(idx) - 1, 1):.1f}s)   tilt {tilt:+.1f}  {a.projection}")
    print(f"pan {res['pan'].min():+.1f}..{res['pan'].max():+.1f}   fov {res['fov'].min():.0f}..{res['fov'].max():.0f}"
          f"   frames without players: {int(np.sum(res['n_players'] == 0))}"
          + (f"   fixed objects ignored as 'ball': {res['static_ball_cells']} spot(s)" if res["static_ball_cells"] else ""))
    base_arr = np.full(len(t), float(tilt))
    cov0 = feet_coverage(frames, t, res["pan"], res["fov"], base_arr, a.projection, a.min_conf)
    if res["tilt"] is not None:
        cov1 = feet_coverage(frames, t, res["pan"], res["fov"], res["tilt"], a.projection, a.min_conf)
        dip = base_arr - res["tilt"]
        print(f"tilt: adaptive, {res['tilt'].min():+.1f}..{res['tilt'].max():+.1f}  (project {tilt:+.1f}); dipped more than 2 deg "
              f"for {100 * np.mean(dip > 2):.0f}% of the time")
        print(f"players inside the crop whose feet are in frame: {100 * cov0:.1f}% with fixed tilt -> {100 * cov1:.1f}% adaptive")
    else:
        print(f"players inside the crop whose feet are in frame, tilt fixed at {tilt:+.1f}: {100 * cov0:.1f}%"
              f"   (try --adaptive-tilt if that is low)")
    if a.compare:
        c = compare(t, res["pan"], res["fov"], a.compare, res["lon_raw"])
        print(f"vs {os.path.basename(a.compare)}: pan error {c['pan_mae']:.1f} deg on average, {c['pan_p90']:.1f} deg "
              f"for the worst 10%  (bias {c['pan_bias']:+.1f}); zoom differs by {c['fov_mae']:.1f} deg")
        print(f"   your crop centre is inside the automatic crop {100 * c['ref_centre_in_auto_crop']:.0f}% of the time; "
              f"the players' centre is inside it {100 * c.get('players_in_auto_crop', float('nan')):.0f}%")
    t_in, t_out = game_marks(header)
    if t_in is not None or t_out is not None:
        print(f"game marks carried over: IN {K.fmt_time(t_in) if t_in is not None else 'start'}  "
              f"OUT {K.fmt_time(t_out) if t_out is not None else 'end'}  (render_keyframes.py uses them by default)")
    print(f"wrote {out}\nopen it:  python annotate.py <video> --keyframes {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
