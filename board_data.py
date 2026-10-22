#!/usr/bin/env python3
"""board_data.py — everything the tactical board of the web player needs, in one JSON file.

The web player shows the YouTube video; next to it a top-down board draws where the players were at the moment
the viewer has paused (or is watching). For that it needs, for every detected moment (3 per second):

  * the time in the *joined video that is on YouTube* (not the time in the raw footage),
  * every player as pitch position (metres, camera at the origin, y = distance from the camera) and team,
  * where the camera was looking (pan + zoom), so the board can show what the video shows.

The file is written next to the other results (`<game>_board.json`) and is uploaded once in the web player.

  python board_data.py --detections v0.det.jsonl v1.det.jsonl --keyframes v0.kf.json v1.kf.json \
      --halves 1 2 --team-labels teams.team_labels.json --chart-data g_half1_chartdata.json g_half2_chartdata.json \
      --durations 1740.9 1650.0 --out game_board.json

Positions are approximate (box height as a ruler, see scene_check.py): the board shows the shape of the teams,
not exact distances.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

import auto_keyframes as AK
import scene_check as S

FORMAT = "football-autopan-board"
VERSION = 1
OTHER = 2                                   # referee, goalkeepers, shirt colour unknown
# The board has its own, looser thresholds than the camera planning: the detections file keeps everything found at
# confidence >= 0.2, the camera path is made from it by other code and is not affected by what is shown on the board.
MIN_CONF = 0.3
MIN_HEIGHT = 14.0                           # smallest box (px) whose shirt colour is trusted
MARGIN_M = 1.5                              # a foot this close (metres) to the pitch outline still counts as on the pitch
RESCUE = 1.5                                # a colour closer to a team than 1.5 x to its 'other' group still counts for the team


def part_frames(det_path: str, kf_path: Optional[str], clusters, video_start: float, max_lon: float, min_conf: float,
                min_height: float, k_angle: float, rescue: float = RESCUE, calib: Optional[str] = None,
                margin_m: float = MARGIN_M) -> List[dict]:
    """One part (one detections file) -> frames {vt, pan, fov, players:[(x, y, team)]}; vt = seconds in the joined video."""
    hdr, frames = AK.load_detections(det_path, max_lon=max_lon)
    fy = S.view_fy(hdr)
    tester = None
    if calib:                                    # the outline: people beside the pitch are left out, by metres not degrees
        import pitch_filter as PF
        tester = PF.FootTester(calib, margin_m=margin_m, fy=fy, k=k_angle)
    views = np.array(hdr["view_lons"], float)
    t_in = float(hdr.get("start", 0.0))
    path = None
    if kf_path:
        import kf_core as K
        try:
            meta = K.load_meta(kf_path)
            if meta.get("trim_in") is not None:
                t_in = float(meta["trim_in"])         # the rendered video starts here (see render_keyframes.py)
            path = K.Path(K.KeyframeSet.load(kf_path).items, "pchip")
        except (OSError, ValueError, SystemExit):
            path = None
    ts = np.array([r["t"] for r in frames], float)
    pan = fov = None
    if path is not None and len(ts):
        import kf_core as K
        p, f = path.at(ts)
        pan = np.array([K.wrap180(K.PAN_SIGN * v) for v in p])
        fov = np.asarray(f, float)

    rows, owner = [], []
    for i, r in enumerate(frames):
        for p in r["players"]:
            if p[3] < min_conf or (tester is not None and not tester(p[0], p[1], p[2])):
                continue
            centre = float(views[np.argmin(np.abs(views - p[0]))])
            x, y = S.ground_position(p[0], p[2], centre, fy, k_angle)
            has_colour = len(p) > 5 and p[5] is not None and p[2] >= min_height
            rows.append((x, y, *(p[5] if has_colour else (-1, -1, -1))))
            owner.append(i)
    team = np.full(len(rows), OTHER, int)
    if rows:
        a = np.array(rows, float)
        ok = a[:, 2] >= 0
        if ok.any():
            fn = getattr(clusters, "team_of_rescue", None)
            t = fn(a[ok, 2:5], rescue) if fn is not None else clusters.team_of(a[ok, 2:5])
            team[ok] = np.where(t >= 0, t, OTHER)
    out = [{"vt": float(r["t"]) - t_in + video_start,
            "pan": None if pan is None else float(pan[i]), "fov": None if fov is None else float(fov[i]), "players": []}
           for i, r in enumerate(frames)]
    for k, i in enumerate(owner):
        out[i]["players"].append((float(rows[k][0]), float(rows[k][1]), int(team[k])))
    return [f for f in out if f["vt"] >= video_start - 1e-6]


def build(parts: List[dict], clusters, title: str = "", lang: str = "en", max_lon: float = 98.0, min_conf: float = MIN_CONF,
          min_height: float = MIN_HEIGHT, k_angle: float = 0.75, rescue: float = RESCUE,
          margin_m: float = MARGIN_M) -> dict:
    """parts: [{det, kf, half, video_start, duration, box:[x0,x1,y0,y1], a_attacks_right}] in playing order."""
    frames, meta_parts = [], []
    for p in parts:
        fr = part_frames(p["det"], p.get("kf"), clusters, float(p["video_start"]), max_lon, min_conf, min_height, k_angle, rescue, p.get("calib"), margin_m)
        end = float(p["video_start"]) + float(p["duration"]) if p.get("duration") else None
        if end is not None:
            fr = [f for f in fr if f["vt"] <= end + 1e-6]
        frames.extend(fr)
        meta_parts.append({"half": int(p["half"]), "video_start": round(float(p["video_start"]), 2),
                           "duration": None if end is None else round(float(p["duration"]), 2),
                           "box": [round(float(v), 2) for v in p["box"]] if p.get("box") else None,
                           "a_attacks_right": p.get("a_attacks_right")})
    if not frames:
        raise SystemExit("no moments for the board (no players found)")
    frames.sort(key=lambda f: f["vt"])
    return {"format": FORMAT, "version": VERSION, "title": title, "lang": lang, "k_angle": k_angle,
            "teams": {"A": {"name": clusters.names[0], "colour": clusters.colours[0]},
                      "B": {"name": clusters.names[1], "colour": clusters.colours[1]}},
            "parts": meta_parts,
            "note": "frames[i] = [t, pan, fov, x1, y1, team1, x2, y2, team2, ...]: t = seconds in the joined video; pan/fov = "
                    "camera direction and zoom (degrees); x (sideways) and y (away from the camera) in decimetres; team 0 = A, "
                    "1 = B, 2 = referee/goalkeeper/unknown",
            "frames": [[round(f["vt"], 2), None if f["pan"] is None else round(f["pan"], 1),
                        None if f["fov"] is None else round(f["fov"], 1)]
                       + [v for (x, y, t) in f["players"] for v in (int(round(x * 10)), int(round(y * 10)), t)]
                       for f in frames]}


def write(path: str, data: dict) -> None:
    with open(path, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detections", nargs="+", required=True, help="detections of each part, in playing order")
    p.add_argument("--keyframes", nargs="+", default=None, help="camera path of each part (same order)")
    p.add_argument("--halves", nargs="+", type=int, required=True, help="half (1 or 2) of each part")
    p.add_argument("--durations", nargs="+", type=float, required=True, help="length in seconds of each rendered part")
    p.add_argument("--team-labels", required=True)
    p.add_argument("--calib", nargs="+", default=None,
                   help="pitch outline file of each part (same order; '-' = none): people beside the pitch are left out")
    p.add_argument("--margin-m", type=float, default=MARGIN_M, help="metres outside the outline that still count as the pitch")
    p.add_argument("--chart-data", nargs="+", default=None, metavar="JSON",
                   help="<game>_halfN_chartdata.json files (pitch box and attacking direction), any order")
    p.add_argument("--title", default="")
    p.add_argument("--lang", choices=["en", "de"], default="en")
    p.add_argument("--out", required=True)
    p.add_argument("--max-lon", type=float, default=98.0)
    p.add_argument("--min-conf", type=float, default=MIN_CONF, help="lowest detector confidence shown on the board")
    p.add_argument("--min-height", type=float, default=MIN_HEIGHT, help="smallest box (px) whose shirt colour is trusted")
    p.add_argument("--angle-factor", type=float, default=0.75)
    p.add_argument("--rescue-ratio", type=float, default=RESCUE,
                   help="a colour whose nearest group is neither team still gets a team if a team group is within this factor "
                        "of that distance (1 = off); makes small, washed-out far-side shirts count")
    a = p.parse_args(argv)
    n = len(a.detections)
    if not (len(a.halves) == len(a.durations) == n) or (a.keyframes and len(a.keyframes) != n) or (a.calib and len(a.calib) != n):
        raise SystemExit("give one --halves, --durations (and --keyframes) value for each --detections file")
    import team_labels as TL
    clusters = TL.Labelled(a.team_labels)
    by_half: Dict[int, dict] = {}
    for path in a.chart_data or []:
        with open(path) as fh:
            d = json.load(fh)
        h = 2 if "_half2_" in os.path.basename(path) else 1
        by_half[h] = d
    parts, t = [], 0.0
    for i in range(n):
        d = by_half.get(a.halves[i], {})
        parts.append({"det": a.detections[i], "kf": a.keyframes[i] if a.keyframes else None, "half": a.halves[i],
                      "video_start": t, "duration": a.durations[i], "box": d.get("pitch_box_m"),
                      "calib": a.calib[i] if a.calib and a.calib[i] != "-" else None,
                      "a_attacks_right": d.get("team_a_attacks_right")})
        t += a.durations[i]
    data = build(parts, clusters, a.title, a.lang, a.max_lon, a.min_conf, a.min_height, a.angle_factor, a.rescue_ratio, a.margin_m)
    write(a.out, data)
    print(f"wrote {a.out}: {len(data['frames'])} moments, {sum((len(f) - 3) // 3 for f in data['frames'])} player positions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
