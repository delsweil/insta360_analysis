"""
ball_track.py — turn the raw per-frame "ball" detections into one ball path.

The ball detector fires on many things (spare balls, white marks, boots, floodlights): on a real game
there are ~1.8 "balls" per frame and a fixed spot can be hit in half of the frames. Three steps:
  1. drop fixed spots  (the same 3-degree cell in more than `static_frac` of all frames)
  2. keep only confident detections (`conf`)
  3. choose ONE candidate (or "none") per frame so that the path is steady: a Viterbi search over the
     whole clip that rewards confidence and punishes jumps.

track_ball(frames) -> np.ndarray of longitudes, nan where no ball was followed.
hold(x, n)         -> carries the last position over up to n frames (the ball is hidden behind players).
"""

from __future__ import annotations

from typing import List

import numpy as np

CELL = 3.0           # degrees: size of the cells used to find fixed spots
NONE_COST = 0.3      # staying without a ball
LOSE_COST = 1.0      # losing the ball
FIND_COST = 0.5      # picking a ball up again
JUMP_FREE = 40.0     # degrees of movement between frames that are still a plausible ball path
CONF_GAIN = 3.0      # reward per unit of confidence


def static_spots(frames: List[dict], frac: float = 0.06) -> set:
    counts: dict = {}
    for r in frames:
        for c in {(round(b[0] / CELL), round(b[1] / CELL)) for b in r.get("balls", [])}:
            counts[c] = counts.get(c, 0) + 1
    n = max(len(frames), 1)
    return {c for c, k in counts.items() if k / n > frac}


def track_ball(frames: List[dict], conf: float = 0.4, static_frac: float = 0.06,
               near: float = 15.0, min_near: int = 2, min_run: int = 3) -> np.ndarray:
    """-> longitude of the followed ball per frame (nan where there is none).
    A ball must have `min_near` players within `near` degrees of it (a ball with nobody around it is a
    mark or a spare ball), and a followed stretch shorter than `min_run` frames is dropped as a false hit."""
    n = len(frames)
    out = np.full(n, np.nan)
    if n == 0 or not any(r.get("balls") for r in frames):
        return out
    st = static_spots(frames, static_frac)
    def supported(r, b):
        xs = [p[0] for p in r.get("players", []) if len(p) < 5 or p[4] is not False]
        return sum(1 for x in xs if abs(x - b[0]) <= near) >= min_near

    cand = [[b for b in r.get("balls", []) if b[2] >= conf and (round(b[0] / CELL), round(b[1] / CELL)) not in st
             and supported(r, b)] for r in frames]
    score = [np.array([0.0] + [-CONF_GAIN * b[2] for b in cand[0]])]
    back: List[np.ndarray] = [np.zeros(len(score[0]), int)]
    for i in range(1, n):
        prev, cur = [None] + cand[i - 1], [None] + cand[i]
        cost = np.zeros((len(prev), len(cur)))
        for a, pb in enumerate(prev):
            for c, b in enumerate(cur):
                if pb is None and b is None:
                    cost[a, c] = NONE_COST
                elif pb is None:
                    cost[a, c] = FIND_COST
                elif b is None:
                    cost[a, c] = LOSE_COST
                else:
                    d = float(np.hypot(b[0] - pb[0], (b[1] - pb[1]) * 1.5))
                    cost[a, c] = d / 10.0 if d < JUMP_FREE else 5.0 + d / 10.0
        tot = score[-1][:, None] + cost
        arg = tot.argmin(axis=0)
        s = tot[arg, np.arange(len(cur))] + np.array([0.0] + [-CONF_GAIN * b[2] for b in cand[i]])
        score.append(s)
        back.append(arg)
    k = int(np.argmin(score[-1]))
    for i in range(n - 1, -1, -1):
        if k > 0:
            out[i] = cand[i][k - 1][0]
        k = int(back[i][k])
    # drop very short stretches
    i = 0
    while i < n:
        if np.isnan(out[i]):
            i += 1
            continue
        j = i
        while j < n and not np.isnan(out[j]) and (j == i or abs(out[j] - out[j - 1]) < JUMP_FREE):
            j += 1
        if j - i < min_run:
            out[i:j] = np.nan
        i = j
    return out


def hold(x: np.ndarray, n: int, drift: int = 0, rate: float = 3.0, vmax: float = 40.0) -> np.ndarray:
    """Keep the last known value for up to `n` frames after the ball is lost. With `drift` > 0 the ball is
    assumed to keep moving the way it was going for that many frames (at most `vmax` degrees/second):
    a ball that disappears at the touchline is usually still travelling towards the goal line."""
    o = x.copy()
    last, age, vel, run = np.nan, 10 ** 9, 0.0, []
    for i, v in enumerate(x):
        if not np.isnan(v):
            run = (run + [v])[-4:]
            vel = (run[-1] - run[0]) / max(len(run) - 1, 1) if len(run) > 1 else 0.0       # degrees per frame
            vel = float(np.clip(vel, -vmax / rate, vmax / rate))
            last, age = v, 0
            o[i] = v
        else:
            age += 1
            if age <= n:
                o[i] = last + vel * min(age, drift)
            else:
                o[i] = np.nan
    return np.clip(o, -100.0, 100.0)


def ease(x: np.ndarray, width: int = 3) -> np.ndarray:
    """Moving average over `width` frames that ignores missing values (and leaves gaps as gaps)."""
    if width <= 1:
        return x.copy()
    ok = ~np.isnan(x)
    v = np.where(ok, x, 0.0)
    k = np.ones(width)
    num, den = np.convolve(v, k, "same"), np.convolve(ok.astype(float), k, "same")
    out = np.full(len(x), np.nan)
    m = ok & (den > 0)
    out[m] = num[m] / den[m]
    return out
