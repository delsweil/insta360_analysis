#!/usr/bin/env python3
"""
tune_auto.py — fit the automatic pass to how YOU frame the game.

    python tune_auto.py --detections g030_5min.detections.jsonl --reference g030.ref.keyframes.json \
        --out altenberg.profile.json

You annotate a stretch by hand in annotate.py (that is the reference, including the tilt you set with
W/S and T/B), extract detections for the same stretch (extract_detections.py), and this tool searches
the settings of auto_keyframes.py — how calm the pan is, how much the camera may drift before it moves,
how far ahead of the play it runs, how wide it zooms, how readily it tilts down for the near side — for
the set that follows your reference most closely.

It does NOT just learn your five minutes by heart: the stretch is split in two halves, the settings are
fitted on one half and scored on the other (both ways round). If the gain on the held-out half is much
smaller than on the data it was fitted on, there is too little data and it says so.

The result is a profile for auto_keyframes.py:

    python auto_keyframes.py --detections game.detections.jsonl --profile altenberg.profile.json ...

Several --detections / --reference pairs (in the same order) are fitted together, which is how to get
settings that hold across games.

Tilt is only fitted if your reference actually uses per-keyframe tilt (T/B in the editor) or you pass
--fit-tilt; otherwise the automatic tilt stays off.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import auto_keyframes as AK
import kf_core as K

DEFAULTS: Dict[str, float] = dict(deadband=5.0, smooth=4.0, zoom_smooth=12.0, fov_a=0.36, fov_b=90.0,
                                  fov_min=90.0, fov_max=140.0, lead=0.0,
                                  tilt_margin=3.0, max_dip=20.0, tilt_smooth=3.0)

GRIDS: Dict[str, List[float]] = dict(
    deadband=[0, 1, 2, 3, 4, 5, 6, 8],
    smooth=[1.5, 2, 3, 4, 5, 6, 8, 10],
    lead=[-1, 0, 0.5, 1, 1.5, 2, 3, 4],
    zoom_smooth=[4, 8, 12, 20, 30],
    fov_a=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7],
    fov_b=[60, 70, 80, 90, 100, 110, 120],
    fov_min=[60, 70, 80, 90],
    fov_max=[110, 120, 130, 140, 150, 160, 170],
    tilt_margin=[0, 2, 3, 4, 6, 8, 10],
    max_dip=[10, 20, 30, 40],
    tilt_smooth=[1.5, 3, 5],
)
GROUPS_PAN = [("deadband",), ("smooth",), ("lead",), ("zoom_smooth",), ("fov_a", "fov_b"), ("fov_min",), ("fov_max",)]
GROUPS_TILT = [("tilt_margin",), ("max_dip",), ("tilt_smooth",)]

W_FOV = 0.35          # a degree of zoom error is less visible than a degree of pan error
W_TILT = 0.5
MIN_GAIN = 0.005      # a change must improve the loss by at least 0.5 % to be accepted


class Case:
    """One detections file + the reference annotation made for the same stretch."""

    def __init__(self, det_path: str, ref_path: str):
        self.name = os.path.basename(det_path)
        self.header, self.frames = AK.load_detections(det_path)
        self.t = np.array([r["t"] for r in self.frames], float)
        kfs = K.KeyframeSet.load(ref_path)
        if len(kfs) < 2:
            raise SystemExit(f"{ref_path}: the reference needs at least 2 keyframes")
        meta = K.load_meta(ref_path)
        self.base_tilt = float(meta.get("tilt_deg", self.header.get("tilt", -15.0)))
        self.projection = meta.get("projection", "cylindrical")
        path = K.Path(kfs.items, "pchip", base_tilt=self.base_tilt)
        self.ref_pan, self.ref_fov = path.at(self.t)
        self.ref_tilt = path.tilt_at(self.t)
        self.has_tilt = path.has_tilt
        first, last = kfs.items[0].t, kfs.items[-1].t
        self.window = (self.t >= first) & (self.t <= last)
        if self.window.sum() < 30:
            raise SystemExit(f"{ref_path}: only {int(self.window.sum())} detection samples fall inside the "
                             f"annotated stretch ({K.fmt_time(first)} - {K.fmt_time(last)}); "
                             f"extract detections for the same stretch")

    def halves(self) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.flatnonzero(self.window)
        mid = self.t[idx[len(idx) // 2]]
        return self.window & (self.t < mid), self.window & (self.t >= mid)


def run(case: Case, s: Dict[str, float], fit_tilt: bool, projection: str) -> dict:
    return AK.build_path(case.header, case.frames, deadband=s["deadband"], smooth_s=s["smooth"],
                         zoom_s=s["zoom_smooth"], fov_a=s["fov_a"], fov_b=s["fov_b"], fov_min=s["fov_min"],
                         fov_max=s["fov_max"], lead_s=s["lead"], adaptive_tilt_on=fit_tilt,
                         base_tilt=case.base_tilt, projection=projection, tilt_margin=s["tilt_margin"],
                         max_dip=s["max_dip"], tilt_s=s["tilt_smooth"])


def errors(case: Case, res: dict, mask: np.ndarray, fit_tilt: bool) -> dict:
    ep = np.abs(res["pan"] - case.ref_pan)[mask]
    ef = np.abs(res["fov"] - case.ref_fov)[mask]
    out = {"n": int(mask.sum()), "pan": float(ep.mean()), "p90": float(np.percentile(ep, 90)), "fov": float(ef.mean())}
    if fit_tilt:
        out["tilt"] = float(np.abs(res["tilt"] - case.ref_tilt)[mask].mean())
    return out


def combine(parts: List[dict]) -> dict:
    n = sum(p["n"] for p in parts)
    out = {"n": n}
    for k in ("pan", "p90", "fov", "tilt"):
        if all(k in p for p in parts):
            out[k] = sum(p[k] * p["n"] for p in parts) / n
    return out


def loss_of(e: dict) -> float:
    return e["pan"] + W_FOV * e["fov"] + W_TILT * e.get("tilt", 0.0)


def evaluate(cases: List[Case], masks: List[np.ndarray], s: Dict[str, float], fit_tilt: bool) -> dict:
    return combine([errors(c, run(c, s, fit_tilt, c.projection), m, fit_tilt) for c, m in zip(cases, masks)])


def fit(cases: List[Case], masks: List[np.ndarray], fit_tilt: bool, rounds: int = 3,
        start: Optional[Dict[str, float]] = None, log=None) -> Dict[str, float]:
    """Coordinate search: sweep each setting (or the zoom pair) over its grid, keep the best, repeat."""
    cur = dict(start or DEFAULTS)
    cache: Dict[tuple, float] = {}
    keys = tuple(sorted(cur))

    def score(s):
        k = tuple(s[x] for x in keys)
        if k not in cache:
            cache[k] = loss_of(evaluate(cases, masks, s, fit_tilt))
        return cache[k]

    groups = GROUPS_PAN + (GROUPS_TILT if fit_tilt else [])
    best = score(cur)
    for r in range(rounds):
        improved = False
        for g in groups:
            cand_best, cand_s = best, None
            for vals in itertools.product(*[GRIDS[x] for x in g]):
                trial = dict(cur)
                trial.update(dict(zip(g, vals)))
                if trial["fov_min"] > trial["fov_max"]:
                    continue
                v = score(trial)
                if v < cand_best - MIN_GAIN * max(best, 1e-9):
                    cand_best, cand_s = v, trial
            if cand_s is not None:
                cur, best, improved = cand_s, cand_best, True
                if log:
                    log(f"  round {r + 1}: {', '.join(f'{x}={cur[x]:g}' for x in g)}   loss {best:.2f}")
        if not improved:
            break
    return cur


def fmt_row(label: str, e: dict, fit_tilt: bool) -> str:
    row = f"  {label:<24}{e['pan']:6.1f}  {e['p90']:6.1f}  {e['fov']:6.1f}"
    row += f"  {e['tilt']:6.1f}" if fit_tilt else ""
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detections", nargs="+", required=True, help="extract_detections.py output(s)")
    ap.add_argument("--reference", nargs="+", required=True,
                    help="your keyframes for the same stretch(es), same order")
    ap.add_argument("--out", default=None, help="profile to write (default: <first reference>.profile.json)")
    ap.add_argument("--fit-tilt", action="store_true", help="fit the automatic tilt even if the reference has no tilt keyframes")
    ap.add_argument("--no-fit-tilt", action="store_true", help="never fit the automatic tilt")
    ap.add_argument("--rounds", type=int, default=3, help="search sweeps (default 3)")
    a = ap.parse_args(argv)
    if len(a.detections) != len(a.reference):
        ap.error("give one --reference for each --detections, in the same order")

    cases = [Case(d, r) for d, r in zip(a.detections, a.reference)]
    fit_tilt = (any(c.has_tilt for c in cases) or a.fit_tilt) and not a.no_fit_tilt
    base_tilt = float(np.mean([c.base_tilt for c in cases]))
    t0 = time.time()
    full = [c.window for c in cases]
    n = int(sum(m.sum() for m in full))
    secs = sum(float(c.t[m][-1] - c.t[m][0]) for c, m in zip(cases, full))
    print(f"{len(cases)} case(s), {n} samples ({secs:.0f} s of reference)   "
          f"tilt {'fitted' if fit_tilt else 'not fitted (the reference has no tilt keyframes)'}   "
          f"base tilt {base_tilt:+.1f}")

    # held-out check: fit on one half, score on the other, both ways
    halves = [c.halves() for c in cases]
    mA, mB = [h[0] for h in halves], [h[1] for h in halves]
    print("fitting on the first half ...")
    sA = fit(cases, mA, fit_tilt, a.rounds)
    print("fitting on the second half ...")
    sB = fit(cases, mB, fit_tilt, a.rounds)
    hd = combine([evaluate(cases, mB, DEFAULTS, fit_tilt), evaluate(cases, mA, DEFAULTS, fit_tilt)])
    ht = combine([evaluate(cases, mB, sA, fit_tilt), evaluate(cases, mA, sB, fit_tilt)])

    print("fitting on everything ...")
    best = fit(cases, full, fit_tilt, a.rounds, log=print)
    d_all = evaluate(cases, full, DEFAULTS, fit_tilt)
    t_all = evaluate(cases, full, best, fit_tilt)

    hdr = "                          pan    worst10%   zoom" + ("    tilt" if fit_tilt else "")
    print("\naverage error against your reference (degrees)\n" + hdr)
    print(fmt_row("defaults", d_all, fit_tilt))
    print(fmt_row("tuned (same data)", t_all, fit_tilt))
    print(fmt_row("held-out: defaults", hd, fit_tilt))
    print(fmt_row("held-out: tuned", ht, fit_tilt))
    gain_in = loss_of(d_all) - loss_of(t_all)
    gain_out = loss_of(hd) - loss_of(ht)
    print(f"\nthe tuned settings are {100 * gain_out / max(loss_of(hd), 1e-9):.0f}% better on data they were not fitted on "
          f"({100 * gain_in / max(loss_of(d_all), 1e-9):.0f}% on the data itself)")
    if gain_in > 0 and gain_out < 0.4 * gain_in:
        print("  -> much less gain on held-out data: this is too little reference to trust every setting. Annotate "
              "more (ideally from another part of the game or another game) and pass several --detections/--reference pairs.")
    if gain_out <= 0:
        print("  -> no gain on held-out data: keep the defaults, or add more reference before using this profile.")

    changed = {k: best[k] for k in DEFAULTS if abs(best[k] - DEFAULTS[k]) > 1e-9
               and (fit_tilt or not k.startswith("tilt_") and k != "max_dip")}
    if changed:
        print("settings that moved from the defaults: " + ", ".join(f"{k} {DEFAULTS[k]:g} -> {v:g}" for k, v in changed.items()))
    else:
        print("the defaults were already the best of everything searched")

    settings = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()
                if fit_tilt or not (k.startswith("tilt_") or k == "max_dip")}
    settings["tilt"] = round(base_tilt, 3)
    if fit_tilt:
        settings["adaptive_tilt"] = True
    out = a.out or os.path.splitext(a.reference[0])[0] + ".profile.json"
    with open(out, "w") as fh:
        json.dump({"settings": settings,
                   "report": {"defaults": d_all, "tuned": t_all, "heldout_defaults": hd, "heldout_tuned": ht,
                              "samples": n, "reference_seconds": round(secs, 1), "tilt_fitted": fit_tilt},
                   "detections": [os.path.basename(d) for d in a.detections],
                   "reference": [os.path.basename(r) for r in a.reference]}, fh, indent=1)
        fh.write("\n")
    print(f"\nwrote {out}   ({time.time() - t0:.0f} s)\nuse it:  python auto_keyframes.py --detections <game>.detections.jsonl "
          f"--profile {out} --out <game>.auto.keyframes.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
