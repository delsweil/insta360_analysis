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
import board_clean as BC
import board_smooth as BSM
import ground_model as GM
import scene_check as S

FORMAT = "football-autopan-board"
VERSION = 1
OTHER = 2                                   # referee, goalkeepers, shirt colour unknown
# The board has its own, looser thresholds than the camera planning: the detections file keeps everything found at
# confidence >= 0.2, the camera path is made from it by other code and is not affected by what is shown on the board.
MIN_CONF = 0.3
LOW_CONF = 0.2                              # weaker detections stay when they belong to a steady track (board_clean.confirm_low)
MIN_HEIGHT = 14.0                           # smallest box (px) whose shirt colour is trusted
MARGIN_M = 1.5                              # a foot this close (metres) to the pitch outline still counts as on the pitch
RESCUE = 1.5                                # a colour closer to a team than 1.5 x to its 'other' group still counts for the team
MAX_TEAM = 11                               # players of one team on the pitch at one moment
GK_END_M = 10.0                             # a goalkeeper this far (metres) from the middle line stands at his own goal: that decides whose he is
TEAM_MARGIN_M = 0.5                         # a team player this far (metres) outside the outline is shown as 'other'


def part_frames(det_path: str, kf_path: Optional[str], clusters, video_start: float, max_lon: float, min_conf: float,
                min_height: float, k_angle: float, rescue: float = RESCUE, calib: Optional[str] = None,
                margin_m: float = MARGIN_M, tidy: bool = True, max_team: int = MAX_TEAM,
                team_margin_m: float = TEAM_MARGIN_M, stats: Optional[dict] = None, feet: bool = True,
                smooth: bool = True, a_attacks_right: Optional[bool] = None,
                box: Optional[list] = None, low_conf: float = LOW_CONF, fill: bool = True) -> List[dict]:
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

    kept, centres = [], []
    for i, r in enumerate(frames):
        for p in r["players"]:
            if p[3] < min(min_conf, low_conf) or (tester is not None and not tester(p[0], p[1], p[2])):
                continue
            kept.append((i, p))
            centres.append(float(views[np.argmin(np.abs(views - p[0]))]))
    # box height alone is a coarse ruler far away; where the feet are in the picture is a steadier one (ground_model.py)
    h_use = np.array([p[2] for _, p in kept], float)
    gm = None
    if feet and kept:
        arr = np.array([[p[0], p[1], p[2]] for r in frames for p in r["players"] if p[3] >= 0.5], float).reshape(-1, 3)
        if len(arr):
            cen_all = views[np.argmin(np.abs(views[None, :] - arr[:, 0:1]), axis=1)]
            f = GM.fit(arr[:, 0], arr[:, 1], arr[:, 2], cen_all)
            if f is not None and f[2] < 0.15:
                gm = f
    if gm is not None:
        lo_ = np.array([p[0] for _, p in kept]); la_ = np.array([p[1] for _, p in kept])
        h_use, used = GM.fuse(lo_, la_, h_use, np.array(centres), gm[0], gm[1])
        if stats is not None:
            stats["feet_used"] = stats.get("feet_used", 0) + int(used.sum())
            stats["feet_gated"] = stats.get("feet_gated", 0) + int((used == 0).sum())
    rows, owner, conf, outm = [], [], [], []
    for k, (i, p) in enumerate(kept):
        x, y = S.ground_position(p[0], float(h_use[k]), centres[k], fy, k_angle)
        has_colour = len(p) > 5 and p[5] is not None and p[2] >= min_height
        rows.append((x, y, *(p[5] if has_colour else (-1, -1, -1))))
        owner.append(i)
        conf.append(float(p[3]))
        outm.append(0.0 if tester is None else tester.outside_m(p[0], p[1], p[2]))
    if rows and low_conf < min_conf:             # weak detections only where a steady track vouches for them
        a0 = np.array(rows, float)
        vt0 = np.array([float(r["t"]) for r in frames])
        trk0 = BC.link_tracks(np.array(owner), a0[:, 0], a0[:, 1], vt0)
        keep = BC.confirm_low(trk0, np.array(conf), min_conf)
        if stats is not None:
            weak = np.array(conf) < min_conf
            stats["weak_kept"] = stats.get("weak_kept", 0) + int((keep & weak).sum())
            stats["weak_dropped"] = stats.get("weak_dropped", 0) + int((~keep & weak).sum())
        rows = [r for r, k in zip(rows, keep) if k]
        owner = [o for o, k in zip(owner, keep) if k]
        conf = [c for c, k in zip(conf, keep) if k]
        outm = [o for o, k in zip(outm, keep) if k]
    team = np.full(len(rows), OTHER, int)
    weight = np.zeros(len(rows))                 # how much the detection's own label counts as a vote
    margin = np.ones(len(rows))                  # 0 = the colour is as near one team as the other, 1 = sure
    gk_label = np.full(len(rows), -1)            # 4 / 5 = the goalkeeper group (of A / B) the colour is nearest to
    if rows:
        a = np.array(rows, float)
        c = np.array(conf)
        ok = a[:, 2] >= 0
        if ok.any():
            fn = getattr(clusters, "team_of_rescue", None)
            t = fn(a[ok, 2:5], rescue) if fn is not None else clusters.team_of(a[ok, 2:5])
            team[ok] = np.where(t >= 0, t, OTHER)
            dist = getattr(clusters, "team_distances", None)
            m = np.ones(int(ok.sum()))
            if dist is not None:
                d = dist(a[ok, 2:5])
                dd = np.where(np.isfinite(d[:, :2]), d[:, :2], 1e3)
                m = np.abs(dd[:, 0] - dd[:, 1]) / np.maximum(dd[:, 0] + dd[:, 1], 1e-6)
            margin[ok] = m
            sure_other = team[ok] == OTHER                       # a colour no team is near (referee, goalkeeper)
            weight[ok] = np.where(sure_other, 0.5, np.maximum(m, 0.05)) * c[ok]
            if getattr(clusters, "has_roles", False):         # referee / goalkeeper groups: the nearest group decides
                rl = clusters.role_of(a[ok, 2:5])
                gk_label[ok] = np.where(rl >= 4, rl, -1)
                team[ok] = np.where(rl == 3, BC.REFEREE, np.where(rl >= 4, BC.GK, team[ok]))
                weight[ok] = np.where(rl >= 3, 0.5 * c[ok], weight[ok])
        if tidy:
            vt = np.array([float(r["t"]) for r in frames])
            team, st = BC.clean(np.array(owner), vt, a[:, 0], a[:, 1], team, weight, margin, np.array(outm),
                                max_team, team_margin_m)
            if stats is not None:
                for k, v in st.items():
                    stats[k] = stats.get(k, 0) + v
                stats["players"] = stats.get("players", 0) + len(rows)
    if smooth and rows:                          # Kalman/RTS smoothing along the tracks (board_smooth.py)
        a2 = np.array(rows, float)
        vt_all = np.array([float(r["t"]) for r in frames])
        fr_idx = np.array(owner)
        trk = BC.link_tracks(fr_idx, a2[:, 0], a2[:, 1], vt_all)
        sx, sy, sst = BSM.smooth(fr_idx, vt_all, a2[:, 0], a2[:, 1], trk)
        rows = [(float(sx[k]), float(sy[k]), *rows[k][2:]) for k in range(len(rows))]
        if stats is not None:
            stats["smoothed"] = stats.get("smoothed", 0) + sst["smoothed"]
            stats["gated"] = stats.get("gated", 0) + sst["gated"]
    if rows and (team >= BC.GK).any():             # which goalkeeper? the one whose goal he stands at, else his group's label
        xs = np.array([r[0] for r in rows], float)
        code = np.where(gk_label >= 4, gk_label, BC.OTHER)
        if a_attacks_right is not None:
            c0 = 0.5 * (float(box[0]) + float(box[1])) if box else 0.0
            a_home_left = bool(a_attacks_right)                # A attacks right -> A defends the left goal
            near_a = (xs < c0) == a_home_left
            code = np.where(np.abs(xs - c0) > GK_END_M, np.where(near_a, 4, 5), code)
        team = np.where(team == BC.GK, code, team)
        if stats is not None:
            stats["keepers"] = stats.get("keepers", 0) + int(((team == 4) | (team == 5)).sum())
    if stats is not None and rows:
        stats["referees"] = stats.get("referees", 0) + int((team == BC.REFEREE).sum())
    extra = None
    if fill and tidy and rows:                     # short holes in a track are bridged (board_clean.fill_gaps)
        a3 = np.array(rows, float)
        trk3 = BC.link_tracks(np.array(owner), a3[:, 0], a3[:, 1], np.array([float(r["t"]) for r in frames]))
        extra = BC.fill_gaps(np.array(owner), np.array([float(r["t"]) for r in frames]), a3[:, 0], a3[:, 1], team, trk3)
        if stats is not None:
            stats["filled"] = stats.get("filled", 0) + len(extra[0])
    out = [{"vt": float(r["t"]) - t_in + video_start,
            "pan": None if pan is None else float(pan[i]), "fov": None if fov is None else float(fov[i]), "players": []}
           for i, r in enumerate(frames)]
    for k, i in enumerate(owner):
        out[i]["players"].append((float(rows[k][0]), float(rows[k][1]), int(team[k])))
    if extra is not None:
        for f, x, y, t in zip(*extra):
            out[int(f)]["players"].append((float(x), float(y), int(t)))
    return [f for f in out if f["vt"] >= video_start - 1e-6]


def build(parts: List[dict], clusters, title: str = "", lang: str = "en", max_lon: float = 98.0, min_conf: float = MIN_CONF,
          min_height: float = MIN_HEIGHT, k_angle: float = 0.75, rescue: float = RESCUE,
          margin_m: float = MARGIN_M, tidy: bool = True, max_team: int = MAX_TEAM,
          team_margin_m: float = TEAM_MARGIN_M, feet: bool = True, smooth: bool = True) -> dict:
    """parts: [{det, kf, half, video_start, duration, box:[x0,x1,y0,y1], a_attacks_right}] in playing order."""
    frames, meta_parts = [], []
    stats: dict = {}
    for p in parts:
        fr = part_frames(p["det"], p.get("kf"), clusters, float(p["video_start"]), max_lon, min_conf, min_height, k_angle, rescue, p.get("calib"), margin_m,
                         tidy, max_team, team_margin_m, stats, feet, smooth, p.get("a_attacks_right"), p.get("box"))
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
    return {"format": FORMAT, "version": VERSION, "title": title, "lang": lang, "k_angle": k_angle, "roles": bool(getattr(clusters, "has_roles", False)), "tidy": stats or None,
            "teams": {"A": {"name": clusters.names[0], "colour": clusters.colours[0]},
                      "B": {"name": clusters.names[1], "colour": clusters.colours[1]}},
            "parts": meta_parts,
            "note": "frames[i] = [t, pan, fov, x1, y1, team1, x2, y2, team2, ...]: t = seconds in the joined video; pan/fov = "
                    "camera direction and zoom (degrees); x (sideways) and y (away from the camera) in decimetres; team 0 = A, "
                    "1 = B, 2 = other / unknown (not counted), 3 = referee, 4 = goalkeeper of A, 5 = goalkeeper of B",
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
    p.add_argument("--no-tidy", action="store_true", help="skip the clean-up of the team labels (tracks, pitch outline, 11 a side)")
    p.add_argument("--no-feet", action="store_true", help="place players by box height only (not combined with the foot position)")
    p.add_argument("--no-smooth", action="store_true", help="show the raw position of every detection (no Kalman smoothing)")
    p.add_argument("--max-team", type=int, default=MAX_TEAM, help="most players of one team at one moment")
    p.add_argument("--team-margin-m", type=float, default=TEAM_MARGIN_M,
                   help="a team player more than this many metres outside the outline is shown as 'other'")
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
    data = build(parts, clusters, a.title, a.lang, a.max_lon, a.min_conf, a.min_height, a.angle_factor, a.rescue_ratio, a.margin_m,
                 not a.no_tidy, a.max_team, a.team_margin_m, not a.no_feet, not a.no_smooth)
    write(a.out, data)
    print(f"wrote {a.out}: {len(data['frames'])} moments, {sum((len(f) - 3) // 3 for f in data['frames'])} player positions")
    st = data.get("tidy")
    if st and "feet_used" in st:
        print(f"positions: {st['feet_used']} combined with the foot position, {st.get('feet_gated', 0)} kept at box height only")
    if st and st.get("smoothed"):
        print(f"smoothed: {st['smoothed']} positions along tracks, {st.get('gated', 0)} wild measurements down-weighted")
    if st:
        print(f"tidied: {st.get('relabelled', 0)} team labels changed by their track, {st.get('recovered', 0)} unreadable players "
              f"given a team, {st.get('outside', 0)} outside the pitch shown as other, {st.get('to_other_team', 0) + st.get('to_other', 0)} "
              f"over {a.max_team} a side moved (of {st.get('players', 0)} detections)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
