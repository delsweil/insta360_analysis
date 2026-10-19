#!/usr/bin/env python3
"""team_stats.py — team-wise analysis from detections that carry shirt colours (extract_detections --colours).

The shirt colours of the whole game are grouped (k-means in Lab, like scene_check.py): the two biggest
clusters are the two teams, the smaller ones are referee, goalkeepers and the like (left out of the team
numbers). Every detected player gets a team and a top-down position (metres, camera at the origin), then:

  * one heat map of the whole game on a pitch (everybody seen; people standing still all game are left out),
  * a territory curve like a momentum chart: who has the play in their attacking half,
  * spread per minute: how far each team stretches (rms distance of its players from the team's centre),
  * tempo per minute: average running speed of each team's players,
  * a card with the key numbers, and a table per 5-minute block (CSV).

Positions are approximate (box height as a ruler, see scene_check.py): trust the shapes and the trends,
not the last metre.

  python team_stats.py --detections h034c.detections.jsonl h035c.detections.jsonl --out laufamholz.teams
  options: --lang de  --title "..."  --k 4  --angle-factor 0.75  --seed 1
"""
from __future__ import annotations

import argparse
import csv
import sys
from typing import List

import numpy as np

import auto_keyframes as AK
import scene_check as S
import football_style as FS

TEXT = {
    "en": {"heat": "Where the game was played", "spread": "Spread: how far each team stretches (m)",
           "tempo": "Tempo: typical running speed of the players (m/s)", "minute": "minute of play",
           "team": "team", "momentum": "Territory: who has the play in their attacking half",
           "attacks": "{t} attacks to the {d}", "right": "right", "left": "left",
           "colour_team": "{c} team", "colours": {"white": "white", "black": "dark", "red": "red", "orange": "orange",
                                                  "yellow": "yellow", "green": "green", "blue": "blue", "purple": "purple"},
           "card": "Key numbers", "c_terr": "Territory (time with the play in their attacking half)",
           "c_spread": "Average spread", "c_tempo": "Typical tempo", "c_n": "Players seen per moment",
           "foot": "Teams are told apart by shirt colour; positions are approximate (box height as ruler); "
                   "pitch markings are fitted to where players were seen; the tempo includes measuring noise, "
                   "so compare the teams and the minutes, not the absolute numbers."},
    "de": {"heat": "Wo gespielt wurde", "spread": "Streuung: wie weit sich jedes Team auseinanderzieht (m)",
           "tempo": "Tempo: typische Laufgeschwindigkeit der Spieler (m/s)", "minute": "Spielminute",
           "team": "Team", "momentum": "Territorium: wer spielt in der gegnerischen Hälfte",
           "attacks": "{t} greift nach {d} an", "right": "rechts", "left": "links",
           "colour_team": "{c}", "colours": {"white": "Weiß", "black": "Dunkel", "red": "Rot", "orange": "Orange",
                                              "yellow": "Gelb", "green": "Grün", "blue": "Blau", "purple": "Lila"},
           "card": "Kennzahlen", "c_terr": "Territorium (Zeit mit dem Spiel in der gegnerischen Hälfte)",
           "c_spread": "Mittlere Streuung", "c_tempo": "Typisches Tempo", "c_n": "Spieler pro Moment",
           "foot": "Teams werden an der Trikotfarbe unterschieden; Positionen sind Näherungen (Boxhöhe als Maßstab); "
                   "Spielfeldlinien sind an die beobachteten Positionen angepasst; das Tempo enthält Messrauschen, "
                   "vergleiche daher Teams und Minuten, nicht die absoluten Werte."},
}


def load_game(paths: List[str], max_lon: float, min_conf: float, min_height: float, k_angle: float):
    """-> dict of arrays, one row per detected player: t (s of play), x, y (m), colour (bgr), frame index."""
    rows = []
    offset, frame_no, n_frames = 0.0, 0, 0
    fy = S.SQUARE_FY
    for path in paths:
        hdr, frames = AK.load_detections(path, max_lon=max_lon)
        if not hdr.get("colours"):
            raise SystemExit(f"{path}: no shirt colours in this file (extract it with --colours)")
        fy = S.SQUARE_FY if hdr.get("square_views") else S.LEGACY_FY
        views = np.array(hdr["view_lons"], float)
        t0 = float(hdr.get("start", 0.0))
        fps = float(hdr.get("fps", 3.0))
        for r in frames:
            for p in r["players"]:
                if p[3] < min_conf or p[2] < min_height or len(p) < 6 or p[5] is None:
                    continue
                centre = float(views[np.argmin(np.abs(views - p[0]))])
                x, y = S.ground_position(p[0], p[2], centre, fy, k_angle)
                rows.append((offset + (r["t"] - t0), x, y, p[5][0], p[5][1], p[5][2], frame_no))
            frame_no += 1
        offset += (frames[-1]["t"] - t0 + 1.0 / fps) if frames else 0.0
        n_frames = frame_no
    a = np.array(rows, float).reshape(-1, 7)
    return {"t": a[:, 0], "x": a[:, 1], "y": a[:, 2], "bgr": a[:, 3:6], "frame": a[:, 6].astype(int),
            "n_frames": n_frames, "duration": offset}


def shape(x: np.ndarray, y: np.ndarray):
    """centre, spread (rms distance to the centre), width and depth (std) of one team in one moment."""
    cx, cy = x.mean(), y.mean()
    return cx, cy, float(np.sqrt(((x - cx) ** 2 + (y - cy) ** 2).mean())), float(x.std()), float(y.std())


def per_minute(game, team, minute_len: float = 60.0, min_players: int = 4):
    """-> dict of arrays per minute: cx, cy, spread, width, depth for team 0 and 1 and the number of moments used."""
    nmin = int(game["duration"] // minute_len) + 1
    out = {m: {k: np.full(nmin, np.nan) for k in ("cx", "cy", "spread", "width", "depth", "n")} for m in (0, 1)}
    tmin = (game["t"] // minute_len).astype(int)
    for m in (0, 1):
        sel = team == m
        for mi in range(nmin):
            s = sel & (tmin == mi)
            if not s.any():
                continue
            vals = []
            for f in np.unique(game["frame"][s]):
                q = s & (game["frame"] == f)
                if q.sum() >= min_players:
                    vals.append(shape(game["x"][q], game["y"][q]))
            if vals:
                v = np.array(vals).mean(0)
                for key, val in zip(("cx", "cy", "spread", "width", "depth"), v):
                    out[m][key][mi] = val
                out[m]["n"][mi] = len(vals)
    return out


def territory(game, team, box, switch_min: float = None, min_players: int = 6, bin_s: float = 5.0, sigma_bins: float = 3.0):
    """Momentum proxy per 5 s: where the play is, relative to the middle of the pitch, seen from team A.

    Mean sideways position of all players of the two teams -> -1 (play at the left goal line) .. +1 (right).
    Team A is taken to attack to the right when its centre of gravity is, on average, left of team B's
    (it defends the left goal); after `switch_min` minutes the teams have changed ends. + = team A has the
    play in its attacking half. -> (minutes, value in -1..1, A attacks right?)"""
    sel = team >= 0
    fr = game["frame"][sel]
    tt, xx, tm = game["t"][sel], game["x"][sel], team[sel]
    ok = np.isin(fr, [f for f, c in zip(*np.unique(fr, return_counts=True)) if c >= min_players])
    nb = int(game["duration"] // bin_s) + 1
    sx, sn = np.zeros(nb), np.zeros(nb)
    bi = np.minimum((tt[ok] // bin_s).astype(int), nb - 1)
    np.add.at(sx, bi, xx[ok])
    np.add.at(sn, bi, 1)
    mid, half = (box[0] + box[1]) / 2, (box[1] - box[0]) / 2
    x = np.where(sn > 0, sx / np.maximum(sn, 1), np.nan)
    terr = np.clip((x - mid) / half, -1, 1)
    minutes = (np.arange(nb) + 0.5) * bin_s / 60.0
    good = np.isfinite(terr)
    terr = np.interp(minutes, minutes[good], terr[good]) if good.any() else np.zeros(nb)
    # which way does team A attack? A's mean sideways position minus B's, over the first stretch before any switch
    first = tt < (switch_min * 60.0 if switch_min else 1e12)
    ax_ = xx[first & (tm == 0)]
    bx_ = xx[first & (tm == 1)]
    a_right = bool(ax_.size and bx_.size and ax_.mean() < bx_.mean())      # A on the left side = attacks right
    sign = 1.0 if a_right else -1.0
    value = terr * sign
    if switch_min:
        value = np.where(minutes >= switch_min, -value, value)
    return minutes, FS.smooth(value, sigma_bins), a_right


def tempo_per_minute(game, team, stride: int = 3, max_step_m: float = 7.0, minute_len: float = 60.0):
    """Typical (median) running speed (m/s) of each team per minute.

    Players are followed over `stride` moments (1 s at 3 per second): within a team, each player is matched
    to the nearest player of the later moment (both ways, i.e. mutual nearest neighbours) and the distance
    walked divided by the time is the speed. Matches farther than `max_step_m` are not trusted. -> {0: array, 1: array}"""
    nmin = int(game["duration"] // minute_len) + 1
    vals = {m: [[] for _ in range(nmin)] for m in (0, 1)}
    order = np.argsort(game["frame"], kind="stable")
    fr = game["frame"][order]
    starts = np.r_[0, np.where(np.diff(fr) != 0)[0] + 1]
    ends = np.r_[starts[1:], len(fr)]
    where = {int(fr[s]): (order[s:e]) for s, e in zip(starts, ends)}
    for f, idx in where.items():
        g = where.get(f + stride)
        if g is None:
            continue
        dt = float(game["t"][g[0]] - game["t"][idx[0]])
        if not (0.5 < dt < 2.0 * stride / 3.0 + 0.5):
            continue
        mi = min(int(game["t"][idx[0]] // minute_len), nmin - 1)
        for m in (0, 1):
            a_ = idx[team[idx] == m]
            b_ = g[team[g] == m]
            if len(a_) < 2 or len(b_) < 2:
                continue
            d = np.hypot(game["x"][a_][:, None] - game["x"][b_][None], game["y"][a_][:, None] - game["y"][b_][None])
            ia, ib = d.argmin(1), d.argmin(0)
            for i, j in enumerate(ia):
                if ib[j] == i and d[i, j] <= max_step_m:
                    vals[m][mi].append(d[i, j] / dt)
    return {m: np.array([np.median(v) if len(v) >= 8 else np.nan for v in vals[m]]) for m in (0, 1)}


def static_spots(game, cell: float = 2.0, frac: float = 0.35):
    """Cells in which somebody stands in more than `frac` of all moments (coaches, bench, a keeper)."""
    ix, iy = np.floor(game["x"] / cell).astype(int), np.floor(game["y"] / cell).astype(int)
    seen = {}
    for f, a_, b_ in zip(game["frame"], ix, iy):
        seen.setdefault((a_, b_), set()).add(int(f))
    n = max(1, game["n_frames"])
    return {k for k, v in seen.items() if len(v) / n > frac}, cell


def draw(game, team, clusters, pm, tempo, out: str, title: str, lang: str, box, mom, switch_min=None,
         keep=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = TEXT[lang]
    col = [FS.display_colour(clusters.mean_bgr[m]) for m in (0, 1)]       # the real shirt colours, lifted for the dark background
    cn = [FS.colour_name(clusters.mean_bgr[m]) for m in (0, 1)]
    if cn[0] != cn[1] and all(c in T["colours"] for c in cn):
        names = [T["colour_team"].format(c=T["colours"][c]) for c in cn]
    else:
        names = [f"{T['team']} A", f"{T['team']} B"]

    fig = plt.figure(figsize=(14, 15.5), facecolor=FS.BG)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.5, 0.8, 0.7, 0.7], hspace=0.40, wspace=0.10, top=0.925, bottom=0.07)
    fig.text(0.07, 0.965, title, fontsize=17, color=FS.TEXT, fontweight="bold", ha="left")

    # combined heat map: everybody seen
    ax = fig.add_subplot(gs[0, 0])
    FS.style_axes(ax)
    sel = keep if keep is not None else np.ones(len(game["x"]), bool)
    FS.draw_heat(ax, game["x"][sel], game["y"][sel], box)
    FS.draw_pitch(ax, box)
    ax.scatter([0], [box[2] - 0.03 * (box[3] - box[2])], marker="^", c=FS.MUTED, s=50, zorder=4, clip_on=False)
    ax.set_title(T["heat"], loc="left", fontsize=13, color=FS.TEXT, pad=8)

    # key numbers
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(FS.BG)
    ax.axis("off")
    minutes, value, a_right = mom
    share = float((value > 0).mean())
    sp = [np.nanmean(pm[m]["spread"]) for m in (0, 1)]
    tp = [np.nanmean(tempo[m]) for m in (0, 1)]
    n_seen = [len(game["x"][team == m]) / max(1, game["n_frames"]) for m in (0, 1)]
    ax.text(0.0, 0.97, T["card"], fontsize=13, color=FS.TEXT, va="top", transform=ax.transAxes)
    rows = [(T["c_terr"], [f"{100 * share:.0f} %", f"{100 * (1 - share):.0f} %"]),
            (T["c_spread"], [f"{v:.1f} m" for v in sp]), (T["c_tempo"], [f"{v:.2f} m/s" for v in tp]),
            (T["c_n"], [f"{v:.1f}" for v in n_seen])]
    y = 0.84
    for label, vals in rows:
        ax.text(0.0, y, label, fontsize=10, color=FS.MUTED, va="top", transform=ax.transAxes)
        for m in (0, 1):
            ax.text(0.08 + 0.42 * m, y - 0.055, vals[m], fontsize=20, color=col[m], va="top", fontweight="bold",
                    transform=ax.transAxes)
        y -= 0.19
    for m in (0, 1):
        ax.text(0.08 + 0.42 * m, 0.0, names[m], fontsize=11, color=col[m], va="bottom", transform=ax.transAxes)

    # territory (momentum)
    ax = fig.add_subplot(gs[1, :])
    FS.style_axes(ax)
    marks = [(switch_min, "HT")] if switch_min and minutes[0] < switch_min < minutes[-1] else []
    FS.draw_momentum(ax, minutes, value, col[0], col[1], marks)
    ax.set_title(T["momentum"], fontsize=13, color=FS.TEXT, pad=26)
    ax.set_xlabel(T["minute"])
    ax.text(0.0, 1.02, f"▲ {names[0]}", transform=ax.transAxes, color=col[0], fontsize=11, va="bottom", fontweight="bold")
    ax.text(0.12, 1.02, f"▼ {names[1]}", transform=ax.transAxes, color=col[1], fontsize=11, va="bottom", fontweight="bold")
    ax.text(0.995, 1.02, T["attacks"].format(t=names[0], d=T["right"] if a_right else T["left"]),
            transform=ax.transAxes, color=FS.MUTED, fontsize=9, va="bottom", ha="right")

    # spread and tempo per minute
    minutes_m = np.arange(len(pm[0]["spread"])) + 0.5
    for pos, label, series in ((gs[2, :], T["spread"], [pm[0]["spread"], pm[1]["spread"]]),
                               (gs[3, :], T["tempo"], [tempo[0], tempo[1]])):
        ax = fig.add_subplot(pos)
        FS.style_axes(ax)
        for m in (0, 1):
            v = np.asarray(series[m], float)
            ax.plot(minutes_m[: len(v)], v, color=col[m], lw=1.0, alpha=0.35)
            ax.plot(minutes_m[: len(v)], _roll(v, 3), color=col[m], lw=2.4, label=names[m])
        ax.set_title(label, loc="left", fontsize=12, color=FS.TEXT)
        ax.set_xlabel(T["minute"])
        ax.grid(color=FS.LINE, alpha=0.25, lw=0.6)
        ax.set_xlim(0, max(minutes_m[-1] + 0.5, 1))
    fig.text(0.07, 0.015, T["foot"], fontsize=8.5, color=FS.MUTED, wrap=True)
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor=FS.BG)
    plt.close(fig)


def _roll(v: np.ndarray, w: int) -> np.ndarray:
    """Centred moving average that ignores gaps."""
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        seg = v[max(0, i - w // 2): i + w // 2 + 1]
        if np.isfinite(seg).any():
            out[i] = np.nanmean(seg)
    return out


def write_csv(path: str, pm, tempo, mom, block: int = 5):
    nmin = len(pm[0]["spread"])
    minutes, value, _ = mom
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["block_min", "A_spread_m", "B_spread_m", "A_tempo_ms", "B_tempo_ms", "A_territory_pct"])
        for b in range(0, nmin, block):
            sl = slice(b, min(nmin, b + block))
            def nm(v):
                v = np.asarray(v[sl], float)
                return f"{np.nanmean(v):.2f}" if np.isfinite(v).any() else ""
            blk = (minutes >= b) & (minutes < min(nmin, b + block))
            terr = f"{100 * float((value[blk] > 0).mean()):.0f}" if blk.any() else ""
            w.writerow([f"{b}-{min(nmin, b + block)}", nm(pm[0]["spread"]), nm(pm[1]["spread"]),
                        nm(tempo[0]), nm(tempo[1]), terr])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detections", nargs="+", required=True, help="detections with colours, in playing order")
    p.add_argument("--out", required=True, help="output prefix: <out>.png and <out>.csv")
    p.add_argument("--title", default="Team analysis")
    p.add_argument("--lang", choices=["en", "de"], default="en")
    p.add_argument("--box", type=float, nargs=4, default=None, metavar=("X0", "X1", "Y0", "Y1"),
                   help="pitch rectangle in metres (default: estimated from where players were seen)")
    p.add_argument("--switch-at", type=float, default=None, metavar="MIN",
                   help="minute of play at which the teams change ends (a game given as both halves)")
    p.add_argument("--tempo-stride", type=int, default=3, help="moments between the two positions a speed is measured over (3 = 1 s)")
    p.add_argument("--keep-static", action="store_true", help="draw people who stand still all game into the heat map too")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--angle-factor", type=float, default=0.75)
    p.add_argument("--max-lon", type=float, default=98.0)
    p.add_argument("--min-conf", type=float, default=0.5)
    p.add_argument("--min-height", type=float, default=18.0, help="smallest box (px) whose shirt colour is trusted")
    p.add_argument("--ambiguous", type=float, default=0.8, help="distance ratio above which a colour is left unassigned")
    p.add_argument("--fit-max", type=int, default=8000, help="colours used to learn the clusters (random subset)")
    a = p.parse_args(argv)

    game = load_game(a.detections, a.max_lon, a.min_conf, a.min_height, a.angle_factor)
    if len(game["t"]) < 50:
        raise SystemExit("too few players with colours")
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(len(game["t"]), min(a.fit_max, len(game["t"])), replace=False)
    clusters = S.Clusters(game["bgr"][idx], a.k, a.seed)
    lab, amb = clusters.assign(game["bgr"])
    team = np.where(amb > a.ambiguous, -1, lab)
    team = np.where(team > 1, -1, team)                      # referee, goalkeepers, ...: not a team
    print("clusters (fit sample): " + ", ".join(f"{clusters.name(j)} {clusters.sizes[j]}" for j in range(len(clusters.centres))))
    print(f"{len(team)} player sightings in {game['n_frames']} moments: team A {int((team == 0).sum())}, "
          f"team B {int((team == 1).sum())}, other/ambiguous {int((team < 0).sum())}")
    pm = per_minute(game, team)
    tempo = tempo_per_minute(game, team, a.tempo_stride)
    ok = team >= 0
    box = tuple(a.box) if a.box else FS.estimate_box(game["x"][ok], game["y"][ok])
    mom = territory(game, team, box, a.switch_at)
    print(f"pitch box (x0 x1 y0 y1, m): {', '.join(f'{v:.1f}' for v in box)}   team A attacks "
          f"{'right' if mom[2] else 'left'}")
    keep = np.ones(len(game["x"]), bool)
    if not a.keep_static:
        cells, cell = static_spots(game)
        if cells:
            cid = list(zip(np.floor(game["x"] / cell).astype(int), np.floor(game["y"] / cell).astype(int)))
            keep = np.array([c not in cells for c in cid])
            print(f"heat map: {len(cells)} static spot(s) left out (somebody stands there in >35% of the moments: "
                  f"coaches, bench); use --keep-static to include them")
    draw(game, team, clusters, pm, tempo, a.out + ".png", a.title, a.lang, box, mom, a.switch_at, keep)
    write_csv(a.out + ".csv", pm, tempo, mom)
    print(f"wrote {a.out}.png and {a.out}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
