#!/usr/bin/env python3
"""suggest.py — suggest the moments of a game that are probably a shot or a goal.

Reads the tactical-board data of a game (`<game>_board.json`, made by board_data.py) and finds the stretches where the ball is
far in one team's attacking end for a couple of seconds. Each one is written as a suggestion: the moment the ball is furthest
forward, and a clip that starts 12 s earlier (the lead-in the coach uses) and ends 8 s later. The web player loads them as
ordinary annotations (see the import button next to the Veo import) that the coach checks, moves or deletes.

    python suggest.py --board game_board.json --out game_suggestions.json
    python suggest.py --csv game_frames.csv --events game_events.csv      # the web export, to try the rule on an annotated game

Needs the ball in the board data (board_data.py with board_ball.py); without it nothing is suggested, because the players alone
find only about one real shot in six suggestions (measured on two games).

How good is it?  On the one game with the ball (20 shots and goals, 85 minutes): 37 home suggestions, 11-12 of the 20 events found
(11 when a suggestion must be within 8 s of the annotation + 12 s, 12 within 12 s), about 3 suggestions in 10 right, against about
a quarter found by chance. One game and thresholds chosen on it: expect less on the next.
Home = team A (the shirt group A of the board), as in the web player's labels shot_home / shot_away.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

import numpy as np

MODEL = "ball-end-v2"
END_FRAC = 0.5           # the ball must be this far (fraction of the half-length from the middle) towards a goal
MIN_S = 2.0              # for at least this long (smoothed)
MERGE_S = 15.0           # episodes closer than this are one
LEAD_S = 12.0            # the clip starts this long before the ball arrives (the coach's own lead-in)
TAIL_S = 8.0
PLAYERS_THR = 2.0        # players-only candidates: play centre + opponent's last defender pushed this many sd above the game's mean
PLAYERS_SEP = 40.0       # at least this many seconds between two of them
SMOOTH = 9               # moments (3 s) of running median: the track jumps now and then


def table_from_board(board: dict):
    """-> dict of arrays per moment: t (board seconds), att (ball, metres towards team A's goal end; nan = no ball),
    half_len (metres), cog (play centre towards A's goal), b_deep (B's furthest-back player, towards A's goal)."""
    frames, ball = board["frames"], board.get("ball")
    if not ball:
        raise SystemExit("this board file has no ball (made without board_ball.py, or from detections without the ball)")
    parts = board.get("parts") or []
    n = len(frames)
    t = np.array([f[0] for f in frames], float)
    att = np.full(n, np.nan)
    cog = np.full(n, np.nan)
    bdeep = np.full(n, np.nan)
    half_len = np.full(n, np.nan)
    for i, f in enumerate(frames):
        p = None
        for q in parts:
            if q["video_start"] <= t[i] + 0.01:
                p = q
        box = (p or {}).get("box") or [-52, 52, 0, 0]
        cx = 0.5 * (box[0] + box[1])
        s = 1.0 if (p or {}).get("a_attacks_right", True) else -1.0
        half_len[i] = 0.5 * (box[1] - box[0])
        if ball[i] is not None:
            att[i] = s * (ball[i][0] / 10.0 - cx)
        xa, xb = [], []
        for k in range(3, len(f) - 2, 3):
            tm = f[k + 2]
            (xa if tm in (0, 4) else xb if tm in (1, 5) else []).append(f[k] / 10.0 - cx)
        if xa and xb:
            cog[i] = s * (np.mean(xa) * len(xa) + np.mean(xb) * len(xb)) / (len(xa) + len(xb))
        if xb:
            bdeep[i] = s * (max(xb) if s > 0 else min(xb))
    return {"t": t, "att": att, "half_len": half_len, "cog": cog, "b_deep": bdeep}


def table_from_csv(path: str):
    """The web export (`_frames.csv`) as the same table, to try the rule on a game that is annotated."""
    import pandas as pd
    f = pd.read_csv(path).sort_values("video_t").reset_index(drop=True)
    s = np.where(f.A_attacks_right == 1, 1.0, -1.0)
    na, nb = f.n_A + f.n_GK_A, f.n_B + f.n_GK_B
    cog = s * (f.A_mean_x.fillna(0) * na + f.B_mean_x.fillna(0) * nb) / (na + nb).replace(0, np.nan)
    bdeep = np.where(s > 0, f.B_x_max, -f.B_x_min)
    half_len = np.full(len(f), 52.0)           # the export has no pitch box: a normal 105 m pitch
    return {"t": f.video_t.values.astype(float), "att": (s * f.ball_x).values.astype(float), "half_len": half_len,
            "cog": cog.values.astype(float), "b_deep": bdeep.astype(float), "time_base": "video"}


def _median(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    h = w // 2
    for i in range(len(x)):
        seg = x[max(0, i - h): i + h + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) >= 3:
            out[i] = np.median(seg)
    return out


def _episodes(t: np.ndarray, sig: np.ndarray, thr: np.ndarray, min_s: float, merge_s: float) -> List[List[int]]:
    on = np.nan_to_num(sig, nan=-1e9) >= thr
    eps, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            if t[j] - t[i] >= min_s:
                eps.append([i, j])
            i = j + 1
        else:
            i += 1
    out: List[List[int]] = []
    for a, b in eps:
        if out and t[a] - t[out[-1][1]] < merge_s:
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


def _players_peaks(tab: dict, thr: float, sep: float) -> List[int]:
    """Moments where the play is pushed towards team A's goal and the opponent's last defender is deep: indices into the table."""
    def smooth(x):
        s = np.asarray(x, float).copy()
        k = 30
        idx = np.arange(len(s))
        ok = ~np.isnan(s)
        if ok.sum() < 10:
            return np.full(len(s), np.nan)
        s = np.interp(idx, idx[ok], s[ok])
        return np.convolve(s, np.ones(k) / k, mode="same")
    c, b = smooth(tab["cog"]), smooth(tab["b_deep"])
    if np.isnan(c).all() or np.isnan(b).all() or np.nanstd(c) < 1e-6 or np.nanstd(b) < 1e-6:
        return []
    sc = (c - np.nanmean(c)) / np.nanstd(c) + (b - np.nanmean(b)) / np.nanstd(b)
    t, keep = tab["t"], []
    for i in np.argsort(-np.nan_to_num(sc, nan=-9)):
        if sc[i] < thr:
            break
        if all(abs(t[i] - t[j]) >= sep for j in keep):
            keep.append(int(i))
    return sorted(keep)


def suggest(tab: dict, end_frac: float = END_FRAC, teams: str = "home", players: bool = True,
            players_thr: float = PLAYERS_THR) -> List[dict]:
    """-> suggestions, in time order: {id, label, t_peak, t_start, t_end, score, ...}; times in the table's own seconds."""
    t, att, hl = tab["t"], tab["att"], tab["half_len"]
    out: List[dict] = []
    for side, sign, label in (("home", 1.0, "shot_home"), ("away", -1.0, "shot_away")):
        if teams not in ("both", side):
            continue
        sm = _median(sign * att, SMOOTH)
        thr = end_frac * hl
        for a, b in _episodes(t, sm, thr, MIN_S, MERGE_S):
            seg = np.nan_to_num(sm[a:b + 1], nan=-1e9)
            k = a + int(np.argmax(seg))
            reach = float(sm[k])
            # how sure: further forward and longer in the end = more; players agreeing (play centre pushed that way) = more
            depth = min(1.0, max(0.0, (reach - thr[k]) / max(1.0, hl[k] - thr[k])))
            dwell = min(1.0, (t[b] - t[a]) / 8.0)
            c = tab["cog"][max(0, k - 15): k + 16]
            c = c[~np.isnan(c)]
            push = 0.5 if len(c) == 0 else float(1 / (1 + np.exp(-sign * np.mean(c) / 10.0)))
            score = round(0.4 * depth + 0.3 * dwell + 0.3 * push, 2)
            out.append({"label": label, "t_peak": round(float(t[k]), 1), "t_start": round(float(t[k]) - LEAD_S, 1),
                        "t_end": round(float(t[k]) + TAIL_S, 1), "score": score, "reach_m": round(reach, 1),
                        "dwell_s": round(float(t[b] - t[a]), 1), "source": "ball"})
    if players and teams in ("both", "home"):             # the ball is lost on some attacks: the players still show the push forward
        have = [s["t_peak"] for s in out if s["label"] == "shot_home"]
        for k in _players_peaks(tab, players_thr, PLAYERS_SEP):
            if all(abs(float(t[k]) - p) > 15.0 for p in have):
                out.append({"label": "shot_home", "t_peak": round(float(t[k]), 1), "t_start": round(float(t[k]) - LEAD_S, 1),
                            "t_end": round(float(t[k]) + TAIL_S, 1), "score": 0.2, "reach_m": None, "dwell_s": None,
                            "source": "players"})
    out.sort(key=lambda s: s["t_peak"])
    for i, s in enumerate(out):
        s["id"] = f"s{i + 1:03d}"
    return out


def evaluate(sugg: List[dict], events_csv: str, lead: float = 12.0, tol: float = 8.0, labels=("shot_home", "goal_home")) -> dict:
    """Compare with the coach's annotations: an annotation is found when a suggestion's moment is within `tol` s of annotation + lead."""
    import pandas as pd
    e = pd.read_csv(events_csv)
    ev = e[e.label.isin(labels)].video_t.values
    P = [s["t_peak"] for s in sugg if s["label"] in ("shot_home",)] if "shot_home" in labels else [s["t_peak"] for s in sugg]
    found = sum(any(abs(p - (x + lead)) <= tol for p in P) for x in ev)
    right = sum(any(abs(p - (x + lead)) <= tol for x in ev) for p in P)
    return {"events": int(len(ev)), "found": int(found), "suggestions": len(P), "right": int(right)}


def review(suggestions_json: str, events_csv: str, tol: float = 12.0) -> dict:
    """What happened to the suggestions once the coach had worked on them? Compares the suggestions file with the events export
    (the web player's Export, whose `note` column carries 'Suggested (model id, score, moment)'). For every suggestion:
      kept      an annotation with its id is still there, and its start has not moved more than `tol` s
      moved     ... and its start has moved by more than that
      relabelled the label was changed (shot -> goal is the usual one: a good sign)
      deleted   no annotation carries its id any more: the coach threw it away
    and the coach's own annotations without a suggestion near them (a shot or goal that was missed). -> counts + rows"""
    import re
    import pandas as pd
    with open(suggestions_json) as fh:
        sg = json.load(fh)["suggestions"]
    e = pd.read_csv(events_csv)
    rx = re.compile(r"Suggested \([\w.\-]+ (s\d+), score ([\d.]+), moment ([\d.]+)\)")
    by_id = {}
    for _, r in e.iterrows():
        m = rx.search(str(r.get("note", "")))
        if m:
            by_id[m.group(1)] = r
    rows = []
    for s in sg:
        r = by_id.get(s["id"])
        if r is None:
            status = "deleted"
        elif r["label"] != s["label"]:
            status = "relabelled"
        elif abs(float(r["video_t"]) - s["t_start"]) > tol:
            status = "moved"
        else:
            status = "kept"
        rows.append({"id": s["id"], "source": s.get("source", "ball"), "suggested_label": s["label"], "final_label": None if r is None else r["label"],
                     "t_peak": s["t_peak"], "score": s["score"], "status": status})
    manual = e[~e.note.fillna("").str.contains("Suggested \\(")]
    manual = manual[manual.kind.isin(["shot", "goal"])]
    peaks = np.array([s["t_peak"] for s in sg]) if sg else np.array([])
    missed = [float(t) for t in manual.video_t if not len(peaks) or np.min(np.abs(peaks - (float(t) + LEAD_S))) > 8.0]
    counts = {k: sum(1 for r in rows if r["status"] == k) for k in ("kept", "moved", "relabelled", "deleted")}
    by_source = {}
    for src in sorted({r["source"] for r in rows}):
        rs = [r for r in rows if r["source"] == src]
        by_source[src] = {"suggestions": len(rs), "deleted": sum(r["status"] == "deleted" for r in rs),
                          "kept_or_changed": sum(r["status"] != "deleted" for r in rs)}
    return {"counts": counts, "by_source": by_source, "rows": rows, "missed_by_suggestions": missed, "manual_shots_goals": int(len(manual))}


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "review":
        q = argparse.ArgumentParser(prog="suggest.py review", description=review.__doc__)
        q.add_argument("--suggestions", required=True)
        q.add_argument("--events", required=True)
        q.add_argument("--out", help="CSV of what happened to each suggestion")
        b = q.parse_args(argv[1:])
        r = review(b.suggestions, b.events)
        c = r["counts"]
        n = sum(c.values())
        print(f"{n} suggestions: {c['kept']} kept, {c['moved']} moved, {c['relabelled']} relabelled, {c['deleted']} deleted")
        for src, v in r["by_source"].items():
            print(f"  from the {src}: {v['suggestions']} suggestions, {v['kept_or_changed']} kept (moved or relabelled included), {v['deleted']} deleted")
        print(f"{len(r['missed_by_suggestions'])} of {r['manual_shots_goals']} shots and goals you added yourself had no suggestion near them")
        if b.out:
            import pandas as pd
            pd.DataFrame(r["rows"]).to_csv(b.out, index=False)
            print(f"wrote {b.out}")
        return 0
    return _main(argv)


def _main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--board", help="<game>_board.json")
    src.add_argument("--csv", help="<game>_frames.csv from the web export")
    p.add_argument("--events", help="<game>_events.csv: report how many of the coach's shots and goals were found")
    p.add_argument("--out", help="suggestions file (JSON)")
    p.add_argument("--teams", choices=["both", "home", "away"], default="home",
                   help="home = team A only (default: the away suggestions are unchecked, no annotated away shots with a ball yet)")
    p.add_argument("--no-players", action="store_true", help="only suggestions from the ball (fewer, more often right)")
    p.add_argument("--players-thr", type=float, default=PLAYERS_THR, help="higher = fewer players-only suggestions")
    p.add_argument("--end-frac", type=float, default=END_FRAC, help="how far towards the goal end the ball must be (fraction of half-length)")
    a = p.parse_args(argv)
    if a.board:
        with open(a.board) as fh:
            tab = table_from_board(json.load(fh))
    else:
        tab = table_from_csv(a.csv)
    sg = suggest(tab, a.end_frac, a.teams, not a.no_players, a.players_thr)
    time_base = tab.get("time_base", "board")
    dur = float(np.nanmax(tab["t"]) - np.nanmin(tab["t"]))
    print(f"{len(sg)} suggestions in {dur / 60:.0f} min ({len(sg) / max(dur / 600, 1e-9):.1f} per 10 min): "
          f"{sum(s['label'] == 'shot_home' for s in sg)} for home, {sum(s['label'] == 'shot_away' for s in sg)} for away; "
          f"{sum(s.get('source') == 'ball' for s in sg)} from the ball, {sum(s.get('source') == 'players' for s in sg)} from the players alone")
    if a.events:
        r = evaluate(sg, a.events)
        print(f"against your annotations (home shots and goals): found {r['found']} of {r['events']}; "
              f"{r['right']} of {sum(s['label'] == 'shot_home' for s in sg)} home suggestions match one")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"format": "football-autopan-suggestions", "version": 1, "model": MODEL,
                       "time_base": time_base,
                       "note": "time_base 'board': the board's seconds, the web player subtracts its saved sync shift; "
                               "'video': already the video time of the web player (made from its CSV export)",
                       "suggestions": sg}, fh, indent=1)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
