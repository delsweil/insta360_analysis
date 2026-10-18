#!/usr/bin/env python3
"""game_stats.py - a few simple numbers about how a game was played, block by block.

    python game_stats.py --detections g030.detections.jsonl g031.detections.jsonl \
                         --keyframes  g030.auto.keyframes.json g031.auto.keyframes.json \
                         --out altenberg.stats            # writes altenberg.stats.csv and altenberg.stats.png

One row per block of --block minutes (default 5). What is measured, and how much to trust it:

  tempo         how fast the play moved: speed of the camera path in degrees per second (the camera follows
                the play). Reliable.
  switches      times the play swung from one side to the other (a move of at least --swing degrees that
                then reverses), per block. Reliable.
  ball in play  share of the time that was NOT a stoppage. A stoppage is a stretch where the typical
                player barely moves (median speed below --still m/s) for at least 5 seconds. A rule of
                thumb: throw-ins, free kicks and goal kicks show up, but slow build-up play can too.
  running       share of player-time above 2.5 m/s, and above 4.5 m/s. Speeds come from following each
                player from one frame to the next, using the player's own height (1.7 m) as a ruler. Only
                sideways movement is seen, so true speeds are higher; far-side players are noisy.
                Compare blocks with each other, not with a stopwatch.
  spread        how wide the players were (middle 80% of the players, in camera degrees). Detector-dependent.
  near the play average number of players within 12 degrees of the play centre. Detector-dependent.
  found         players found per frame: a reminder of how much of the pitch the detector saw.
"""
import argparse
import csv
import math
from typing import List

import numpy as np

import auto_keyframes as AK
import play_map as PM

PLAYER_H = 1.7          # metres, the ruler
VIEW_FY = 302.0         # vertical focal length in pixels of the 1280x720 detection views (100 deg scalar fov)
SQUARE_FY = 640.0 / math.tan(math.radians(50.0))   # the same for the square-pixel views made with --iv-fov (about 537)
RUN, SPRINT = 2.5, 4.5  # m/s


def smooth(v: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(v, np.nan, dtype=float)
    for i in range(len(v)):
        s = v[max(0, i - n):i + n + 1]
        s = s[~np.isnan(s)]
        if len(s):
            out[i] = s.mean()
    return out


def player_speeds(frames: List[dict], tg: np.ndarray, min_conf: float, fy: float = VIEW_FY):
    """-> list (per frame) of arrays of sideways speeds (m/s) of players followed to the next frame."""
    out = [np.zeros(0) for _ in frames]
    for i in range(len(frames) - 1):
        dt = tg[i + 1] - tg[i]
        if not 0.15 < dt < 0.6:
            continue
        A = np.array([[p[0], p[1], p[2]] for p in frames[i]["players"] if p[3] >= min_conf], float).reshape(-1, 3)
        B = np.array([[p[0], p[1], p[2]] for p in frames[i + 1]["players"] if p[3] >= min_conf], float).reshape(-1, 3)
        if not len(A) or not len(B):
            continue
        D = np.hypot(A[:, None, 0] - B[None, :, 0], A[:, None, 1] - B[None, :, 1])
        ja, ib = D.argmin(axis=1), D.argmin(axis=0)
        v = []
        for a, b in enumerate(ja):
            if ib[b] != a or D[a, b] > 6.0:                      # mutual nearest neighbours only
                continue
            h = max((A[a, 2] + B[b, 2]) / 2.0, 8.0)
            dist_m = PLAYER_H * fy / h                       # how far away, from the player's height on screen
            speed = dist_m * math.radians(abs(B[b, 0] - A[a, 0])) / dt
            if speed <= 9.0:
                v.append(speed)
        out[i] = np.array(v)
    return out


def swings(cam: np.ndarray, t: np.ndarray, amp: float):
    """times of the turning points of a zig-zag of the camera longitude with at least `amp` degrees."""
    marks = []
    if len(cam) < 3:
        return marks
    trend, hi, lo, hi_i, lo_i = 0, cam[0], cam[0], 0, 0
    for i, x in enumerate(cam):
        if np.isnan(x):
            continue
        if trend == 0:
            if x > hi: hi, hi_i = x, i
            if x < lo: lo, lo_i = x, i
            if hi - x >= amp:
                trend, lo, lo_i = -1, x, i
            elif x - lo >= amp:
                trend, hi, hi_i = 1, x, i
        elif trend == 1:
            if x > hi:
                hi, hi_i = x, i
            elif hi - x >= amp:
                marks.append(t[hi_i]); trend, lo, lo_i = -1, x, i
        else:
            if x < lo:
                lo, lo_i = x, i
            elif x - lo >= amp:
                marks.append(t[lo_i]); trend, hi, hi_i = 1, x, i
    return marks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detections", nargs="+", required=True)
    ap.add_argument("--keyframes", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="prefix: writes <out>.csv and <out>.png")
    ap.add_argument("--block", type=float, default=5.0)
    ap.add_argument("--swing", type=float, default=25.0, help="degrees the play must move to count as a switch")
    ap.add_argument("--still", type=float, default=0.9, help="m/s: typical player speed below this = a stoppage")
    ap.add_argument("--near", type=float, default=12.0, help="degrees around the play centre for 'players near the play'")
    ap.add_argument("--max-lon", type=float, default=98.0)
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--title", default=None)
    ap.add_argument("--lang", choices=["en", "de"], default="en", help="language of the chart (en or de)")
    a = ap.parse_args()
    L = (lambda en, de: de if a.lang == "de" else en)
    if len(a.detections) != len(a.keyframes):
        raise SystemExit("give one --keyframes file for each --detections file, in the same order")

    # one continuous game clock across the files
    T, CAM, LON, LAT = [], [], [], []
    SPD, FOUND, WIDTH, NEAR = [], [], [], []
    off = 0.0
    for d, k in zip(a.detections, a.keyframes):
        p = PM.per_file(d, k, off, a.max_lon if a.max_lon > 0 else None, a.min_conf)
        hdr, frames = AK.load_detections(d, max_lon=a.max_lon if a.max_lon > 0 else None)
        fy = SQUARE_FY if hdr.get("square_views") else VIEW_FY       # detection views differ with the unwrapping
        tg = p["t"]
        T.append(tg); CAM.append(p["cam"]); LON.append(p["lon"]); LAT.append(p["lat"])
        SPD += player_speeds(frames, tg, a.min_conf, fy)
        for i, r in enumerate(frames):
            P = np.array([[q[0], q[1]] for q in r["players"] if q[3] >= a.min_conf], float).reshape(-1, 2)
            FOUND.append(len(P))
            WIDTH.append(np.percentile(P[:, 0], 90) - np.percentile(P[:, 0], 10) if len(P) >= 5 else np.nan)
            if len(P) and not np.isnan(p["lon"][i]):
                NEAR.append(float((np.hypot(P[:, 0] - p["lon"][i], P[:, 1] - p["lat"][i]) <= a.near).sum()))
            else:
                NEAR.append(np.nan)
        off += p["span"]
    t = np.concatenate(T); cam = np.concatenate(CAM)
    found, width, near = np.array(FOUND, float), np.array(WIDTH), np.array(NEAR)
    med_speed = np.array([np.median(s) if len(s) >= 3 else np.nan for s in SPD])
    fps = len(t) / max(t[-1] - t[0], 1.0)
    n3 = max(1, int(round(1.5 * fps)))                                    # about 3 s window
    cam_s = smooth(cam, n3)
    tempo = np.abs(np.gradient(cam_s, t))
    still = smooth(med_speed, n3) < a.still
    # a stoppage needs at least 5 seconds in a row
    run_len = int(round(5 * fps))
    stop = np.zeros(len(t), bool)
    i = 0
    while i < len(t):
        if still[i] and not np.isnan(med_speed[i]):
            j = i
            while j < len(t) and still[j]:
                j += 1
            if j - i >= run_len:
                stop[i:j] = True
            i = j
        else:
            i += 1
    sw = np.array(swings(cam_s, t, a.swing))

    total = t[-1]
    nb = max(1, int(math.ceil(total / (a.block * 60))))
    rows = []
    for b in range(nb):
        lo, hi = b * a.block * 60, (b + 1) * a.block * 60
        s = (t >= lo) & (t < hi)
        if s.sum() < 20 or (min(hi, total) - lo) < 60:       # skip a stub of under a minute at the end
            continue
        sp = np.concatenate([SPD[i] for i in np.where(s)[0]]) if s.any() else np.zeros(0)
        mins = (min(hi, total) - lo) / 60.0
        rows.append(dict(
            block=f"{int(lo // 60)}-{int(min(hi, total) // 60)}",
            minutes=round(mins, 1),
            tempo_deg_per_s=round(float(np.nanmean(tempo[s])), 2),
            switches_per_5min=round(float(((sw >= lo) & (sw < hi)).sum()) * 5.0 / max(mins, 0.1), 1),
            ball_in_play_pct=round(100 * (1 - stop[s].mean()), 0),
            running_pct=round(100 * float((sp > RUN).mean()), 1) if len(sp) else float("nan"),
            sprint_pct=round(100 * float((sp > SPRINT).mean()), 1) if len(sp) else float("nan"),
            median_speed_mps=round(float(np.median(sp)), 2) if len(sp) else float("nan"),
            spread_deg=round(float(np.nanmedian(width[s])), 1),
            players_near_play=round(float(np.nanmean(near[s])), 1),
            players_found=round(float(found[s].mean()), 1),
        ))
    if not rows:
        raise SystemExit("not enough frames")

    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"{'block':>8} {'tempo':>6} {'switch':>6} {'inplay%':>7} {'run%':>5} {'sprint%':>7} {'spread':>6} {'near':>5} {'found':>5}")
    for r in rows:
        print(f"{r['block']:>8} {r['tempo_deg_per_s']:6.2f} {r['switches_per_5min']:6.1f} {r['ball_in_play_pct']:7.0f} "
              f"{r['running_pct']:5.1f} {r['sprint_pct']:7.1f} {r['spread_deg']:6.1f} {r['players_near_play']:5.1f} {r['players_found']:5.1f}")
    print(f"\nwhole game: ball in play {100 * (1 - stop.mean()):.0f}% of the time, {len(sw)} switches of play, "
          f"mean tempo {np.nanmean(tempo):.2f} deg/s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, soft, blue, red = "#1c1f26", "#8a8f9c", "#4f74c4", "#c0452f"
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.95))
    xs = np.arange(len(rows)); lab = [r["block"] for r in rows]
    panels = [
        (L("Tempo of play", "Tempo des Spiels"), L("camera speed (°/s)", "Kamerageschwindigkeit (°/s)"), "tempo_deg_per_s", None,
         L("Faster = the play moved around more quickly.", "Höher = das Spiel wurde schneller verlagert.")),
        (L("Switches of play", "Seitenwechsel"), L("swings per 5 min", "Wechsel pro 5 Min."), "switches_per_5min", None,
         L(f"Moves of at least {a.swing:g}° that then reverse.", f"Verlagerungen um mindestens {a.swing:g}° mit anschließender Umkehr.")),
        (L("Ball in play", "Ball im Spiel"), L("% of the time", "% der Zeit"), "ball_in_play_pct", None,
         L("The rest is stoppages of 5 s or more.", "Der Rest sind Unterbrechungen ab 5 s.")),
        (L("Running", "Laufen"), L("% of player-time above 2.5 m/s", "% der Spielerzeit über 2,5 m/s"), "running_pct", "sprint_pct",
         L("Dark bars: above 4.5 m/s. Sideways speed only.", "Dunkel: über 4,5 m/s. Nur seitliche Geschwindigkeit.")),
        (L("How spread out the players were", "Wie breit die Spieler verteilt waren"),
         L("width of the middle 80% (°)", "Breite der mittleren 80 % (°)"), "spread_deg", None,
         L("Larger = more stretched across the pitch.", "Größer = mehr über das Feld verteilt.")),
        (L("Players near the play", "Spieler nahe am Spielgeschehen"),
         L(f"players within {a.near:g}° of the play", f"Spieler im Umkreis von {a.near:g}° ums Spiel"), "players_near_play", None,
         L("Larger = a more crowded ball area.", "Größer = mehr Gedränge am Ball.")),
    ]
    for ax, (title, yl, key, key2, note) in zip(axs.ravel(), panels):
        ax.bar(xs, [r[key] for r in rows], color=blue, width=0.75)
        if key2:
            ax.bar(xs, [r[key2] for r in rows], color="#1f3470", width=0.75)
        ax.set_title(title, loc="left", fontsize=12, color=ink)
        ax.set_ylabel(yl, fontsize=9); ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=8, rotation=0)
        ax.set_xlabel(L("minutes", "Minuten"), fontsize=9)
        ax.text(0.0, -0.27, note, transform=ax.transAxes, fontsize=8, color=soft)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
    fig.suptitle(a.title or L("How the game was played", "Wie das Spiel gespielt wurde"), x=0.01, ha="left", fontsize=14, color=ink)
    fig.text(0.01, 0.003, L(
        f"Estimates from the video. Players found per frame: {found.mean():.1f} on average (the detector misses some, "
        "especially on the far side). Running speeds are sideways only and approximate: compare blocks, not stopwatches.",
        f"Schätzungen aus dem Video. Gefundene Spieler pro Bild: im Schnitt {found.mean():.1f} (das Programm übersieht einige, "
        "besonders auf der abgewandten Seite). Laufgeschwindigkeiten sind nur seitlich und ungefähr: Blöcke vergleichen, nicht stoppen."),
        fontsize=8, color=soft, ha="left", va="bottom")
    fig.savefig(a.out + ".png", dpi=130)
    print(f"-> {a.out}.csv  {a.out}.png")


if __name__ == "__main__":
    main()
