#!/usr/bin/env python3
"""play_map.py - where was the PLAY? Uses the camera path as a stand-in for the ball.

    python play_map.py --detections g030.detections.jsonl g031.detections.jsonl \
                       --keyframes  g030.auto.keyframes.json g031.auto.keyframes.json --out play.png
    # optional: check the path against a human-annotated stretch
    python play_map.py ... --reference g030.ref.keyframes.json

The camera path is the one thing in this project that has been checked against people, and the ball is
nearly always where the camera points. So instead of counting every player the detector happened to
find (which over-counts what is easy to detect), this map has ONE point per frame: the play centre,
the middle of the players inside the crop, weighted toward the crop centre. Every moment counts the
same, whatever the detector found.

Panels:
  * play-centre heat map   (camera longitude across, foot latitude of the play centre up/down)
  * the camera over time   (where the camera looked, and the play centre next to it)
  * territory              per block of minutes: how long the play was in the left, middle and right part
                           of the pitch (the parts are the thirds of where play went over the whole game)
  * with --reference       how far the camera was from a human-annotated path (the accuracy check)

Both --keyframes and --reference accept the files the annotator writes, so you can also draw this from a
path you corrected by hand.
"""
import argparse
import math
from typing import List

import numpy as np

import auto_keyframes as AK
import kf_core as K


def per_file(det_path: str, kf_path: str, offset: float, max_lon: float, min_conf: float):
    hdr, frames = AK.load_detections(det_path, max_lon=max_lon)
    kfs = K.KeyframeSet.load(kf_path)
    path = K.Path(kfs.items, "pchip")
    t = np.array([r["t"] for r in frames])
    pan, fov = path.at(t)
    cam = np.array([K.wrap180(K.PAN_SIGN * p) for p in pan])
    play_lon = np.full(len(t), np.nan)
    play_lat = np.full(len(t), np.nan)
    n_in = np.zeros(len(t), int)
    for i, r in enumerate(frames):
        P = np.array([[p[0], p[1], p[3]] for p in r["players"] if p[3] >= min_conf], float).reshape(-1, 3)
        if not len(P):
            continue
        d = np.array([K.wrap180(x - cam[i]) for x in P[:, 0]])
        inside = np.abs(d) <= fov[i] / 2
        if not inside.any():
            continue
        sig = max(fov[i] / 4.0, 5.0)
        w = np.exp(-0.5 * (d[inside] / sig) ** 2) * P[inside, 2]
        play_lon[i] = float(cam[i] + np.average(d[inside], weights=w))
        play_lat[i] = float(np.average(P[inside, 1], weights=w))
        n_in[i] = int(inside.sum())
    t0 = t[0]
    return dict(t=t - t0 + offset, t_abs=t, cam=cam, fov=fov, lon=play_lon, lat=play_lat, n=n_in,
                span=float(t[-1] - t0) + 1.0)


def blur(a, sigma):
    k = np.arange(-int(3 * sigma), int(3 * sigma) + 1)
    k = np.exp(-0.5 * (k / sigma) ** 2); k /= k.sum()
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detections", nargs="+", required=True)
    ap.add_argument("--keyframes", nargs="+", required=True, help="one keyframe file per detections file, same order")
    ap.add_argument("--reference", nargs="*", default=None, help="human-annotated keyframes (any stretch) to score the path against")
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=float, default=5.0, help="minutes per territory bar")
    ap.add_argument("--max-lon", type=float, default=98.0)
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--title", default=None)
    ap.add_argument("--lang", choices=["en", "de"], default="en", help="language of the chart (en or de)")
    a = ap.parse_args()
    L = (lambda en, de: de if a.lang == "de" else en)
    if len(a.detections) != len(a.keyframes):
        raise SystemExit("give one --keyframes file for each --detections file, in the same order")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parts, off = [], 0.0
    for d, k in zip(a.detections, a.keyframes):
        p = per_file(d, k, off, a.max_lon if a.max_lon > 0 else None, a.min_conf)
        parts.append(p); off += p["span"]
    cat = lambda key: np.concatenate([p[key] for p in parts])
    t, cam, lon, lat = cat("t"), cat("cam"), cat("lon"), cat("lat")
    ok = ~np.isnan(lon)
    if ok.sum() < 50:
        raise SystemExit("too few frames with players inside the crop")
    total = t[-1]
    ink, soft, blue, red = "#1c1f26", "#8a8f9c", "#2b4a9b", "#c0452f"

    lo3, hi3 = np.percentile(lon[ok], [33.3, 66.7])
    zone = np.where(lon < lo3, 0, np.where(lon < hi3, 1, 2))
    names = [L("left third", "linkes Drittel"), L("middle third", "mittleres Drittel"), L("right third", "rechtes Drittel")]

    nb = max(1, int(math.ceil(total / (a.block * 60))))
    fig = plt.figure(figsize=(15, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])
    fig.get_layout_engine().set(rect=(0, 0.025, 1, 0.97))

    # 1 heat map of the play centre
    ax = fig.add_subplot(gs[0, 0])
    xr = (float(np.percentile(lon[ok], 0.5)) - 5, float(np.percentile(lon[ok], 99.5)) + 5)
    yr = (float(np.percentile(lat[ok], 0.5)) - 3, float(np.percentile(lat[ok], 99.5)) + 3)
    H, _, _ = np.histogram2d(lon[ok], lat[ok], bins=(80, 36), range=[xr, yr])
    H = blur(H / H.sum(), 1.6).T
    ax.imshow(H, origin="lower", extent=[xr[0], xr[1], yr[0], yr[1]], aspect="auto", cmap="magma_r")
    for v in (lo3, hi3):
        ax.axvline(v, color="white", lw=0.8, ls="--")
    ax.set_title(L("Where the play was (one point per moment, whole game)", "Wo das Spiel stattfand (ein Punkt pro Moment, ganzes Spiel)"), loc="left", fontsize=12, color=ink)
    ax.set_xlabel(L("camera longitude (°)   ← left        right →", "Kamera-Längengrad (°)   ← links        rechts →"))
    ax.set_ylabel(L("foot latitude of the play (°)   far ↑   ↓ near", "Fußpunkt-Breitengrad des Spiels (°)   weit ↑   ↓ nah"))

    # 2 the camera over time
    ax2 = fig.add_subplot(gs[0, 1])
    m = t / 60.0
    ax2.plot(m, np.where(ok, lon, np.nan), color=red, lw=0.7, alpha=0.8, label=L("play centre (players in the crop)", "Spielmitte (Spieler im Bildausschnitt)"))
    ax2.plot(m, cam, color=blue, lw=1.0, label=L("where the camera looked", "wohin die Kamera schaute"))
    for v in (lo3, hi3):
        ax2.axhline(v, color=soft, lw=0.6, ls=":")
    for p in parts[:-1]:
        ax2.axvline((p["t"][-1]) / 60.0, color=soft, lw=0.8)
    ax2.set_xlim(0, m[-1]); ax2.set_xlabel(L("minutes", "Minuten")); ax2.set_ylabel(L("longitude (°)", "Längengrad (°)"))
    ax2.set_title(L("The camera over time", "Die Kamera im Zeitverlauf"), loc="left", fontsize=12, color=ink)
    ax2.legend(loc="upper right", fontsize=8, frameon=False)

    # 3 territory per block
    ax3 = fig.add_subplot(gs[1, :] if not a.reference else gs[1, 0])
    cols = ["#9db4e3", "#d9dde8", "#e8a99b"]
    shares = np.zeros((nb, 3))
    for i in range(nb):
        s = (t >= i * a.block * 60) & (t < (i + 1) * a.block * 60) & ok
        if s.sum():
            for z in range(3):
                shares[i, z] = (zone[s] == z).mean()
    bottom = np.zeros(nb)
    xs = np.arange(nb)
    for z in range(3):
        ax3.bar(xs, shares[:, z] * 100, bottom=bottom, color=cols[z], label=names[z], width=0.8)
        bottom += shares[:, z] * 100
    ax3.set_xticks(xs); ax3.set_xticklabels([f"{int(i * a.block)}–{int(min((i + 1) * a.block, total / 60))}" for i in xs], fontsize=8)
    ax3.set_xlabel(L("minutes", "Minuten")); ax3.set_ylabel(L("% of the time the play was in…", "% der Zeit"))
    ax3.set_title(L("Territory: where the play was, block by block", "Raumgewinn: wo das Spiel war, Block für Block"), loc="left", fontsize=12, color=ink)
    ax3.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)

    # 4 accuracy against a human path
    note = ""
    if a.reference:
        ax4 = fig.add_subplot(gs[1, 1])
        refs = []
        for rp in a.reference:
            kf = K.KeyframeSet.load(rp); refs.append((K.Path(kf.items, "pchip"), kf.items[0].t, kf.items[-1].t))
        errs, tt = [], []
        for p in parts:
            tabs = p["t_abs"]
            for path, r0, r1 in refs:
                s = (tabs >= r0) & (tabs <= r1)
                if not s.any():
                    continue
                rpan, _ = path.at(tabs[s])
                rlon = np.array([K.wrap180(K.PAN_SIGN * v) for v in rpan])
                errs.append(K.wrap180(p["cam"][s] - rlon) if np.isscalar(rlon) else
                            np.array([K.wrap180(x) for x in (p["cam"][s] - rlon)]))
                tt.append(p["t"][s] / 60.0)
        if errs:
            e = np.concatenate(errs); tm = np.concatenate(tt)
            ax4.plot(tm, e, color=blue, lw=0.8)
            ax4.axhline(0, color=soft, lw=0.6)
            ax4.set_xlabel(L("minutes", "Minuten")); ax4.set_ylabel(L("camera − human (°)", "Kamera − Mensch (°)"))
            ax4.set_title(L(f"Against the human path: mean error {np.mean(np.abs(e)):.1f}°, bias {np.mean(e):+.1f}°", f"Gegen den Verlauf von Hand: Fehler {np.mean(np.abs(e)):.1f}°, Abweichung {np.mean(e):+.1f}°"),
                          loc="left", fontsize=12, color=ink)
            note = L(f"  Accuracy: {np.mean(np.abs(e)):.1f}° mean error over {len(e) / 3 / 60:.1f} min of human-annotated play (at 3 frames/s).",
                     f"  Genauigkeit: {np.mean(np.abs(e)):.1f}° mittlerer Fehler über {len(e) / 3 / 60:.1f} Min. von Hand gesetzten Verlaufs (bei 3 Bildern/s).")
        else:
            ax4.text(0.5, 0.5, "the reference does not overlap the game", ha="center", transform=ax4.transAxes)
    fig.suptitle(a.title or L("Where the play was", "Wo das Spiel stattfand"), x=0.01, ha="left", fontsize=14, color=ink)
    fig.text(0.01, 0.003, L("Play centre = middle of the players inside the crop, weighted to the centre; an estimate of where the ball "
             "was, not a ball track.", "Spielmitte = Mitte der Spieler im Bildausschnitt, zur Bildmitte hin stärker gewichtet; eine Schätzung, "
             "wo der Ball war, keine Ballverfolgung.") + note, fontsize=8, color=soft, ha="left", va="bottom")
    fig.savefig(a.out, dpi=130)
    print(f"{ok.sum()} of {len(t)} moments with players in the crop, {total / 60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
