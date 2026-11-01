#!/usr/bin/env python3
"""add_ball_web.py — the two small edits in web/lib/board.ts that carry the ball, made in place.

Run from the repository root:   python3 add_ball_web.py
(The full web/lib/exportEvents.ts is a separate file: copy it over the old one.)
Safe to run twice; it says what it did and changes nothing it cannot find.
"""
import re
import sys

path = "web/lib/board.ts"
s = open(path).read()
if "ball: (number[] | null)[] | null" in s:
    print("board.ts already has the ball: nothing to do")
    sys.exit(0)

# 1) the Board type: a `ball` property right after `frames: ...[][]`
m = re.search(r"^(\s*)frames:\s*\(number \| null\)\[\]\[\][^\n]*\n", s, re.M)
done = []
if m:
    add = (f"{m.group(1)}/** ball[i] belongs to frames[i]: [x, y, confidence %] (x, y in decimetres) or null; null when the file has no ball */\n"
           f"{m.group(1)}ball: (number[] | null)[] | null\n")
    s = s[:m.end()] + add + s[m.end():]
    done.append("type")
else:
    print("could not find 'frames: (number | null)[][]' in the Board type: add\n  ball: (number[] | null)[] | null\nunder it by hand")

# 2) parseBoard(): `ball: ...` right after the shorthand `frames,` of the returned object
m = re.search(r"^(\s*)frames,\s*\n", s, re.M)
if m:
    add = f"{m.group(1)}ball: Array.isArray(raw.ball) && raw.ball.length === frames.length ? (raw.ball as (number[] | null)[]) : null,\n"
    s = s[:m.end()] + add + s[m.end():]
    done.append("parse")
else:
    print("could not find a line 'frames,' in parseBoard(): add\n"
          "  ball: Array.isArray(raw.ball) && raw.ball.length === frames.length ? (raw.ball as (number[] | null)[]) : null,\n"
          "after it by hand")

if len(done) == 2:
    open(path, "w").write(s)
    print("board.ts updated:", ", ".join(done))
else:
    print("board.ts NOT written (only some edits found); nothing changed")
    sys.exit(1)
