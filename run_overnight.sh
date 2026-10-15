#!/usr/bin/env bash
# run_overnight.sh — extract players, build keyframes (tuned profile) and render, for each video,
# one after the other, with no supervision.
#
#   cd /Users/davidelsweiler/Dropbox/development/insta360_analysis
#   nohup bash run_overnight.sh < /dev/null > /dev/null 2>&1 &
#   tail -f /Users/davidelsweiler/Desktop/altenberg/overnight.log        # optional, to look in
#
# Each video is handled on its own; a failure in one never stops the others. Finished steps leave a
# ".done" marker, so running the script again after an interruption picks up where it stopped
# (delete a marker to redo that step). Keep the Mac plugged in with the lid OPEN: a closed lid sleeps
# the machine whatever the script does.

# ── settings ────────────────────────────────────────────────────────────────
REPO=/Users/davidelsweiler/Dropbox/development/insta360_analysis
D=/Users/davidelsweiler/Desktop/altenberg
PROFILE=$D/altenberg.profile.json
PLAYERS=models/yolo11s.pt
JOIN_OUT=$D/altenberg_full.mp4   # all renders joined in the order of JOBS ("" = skip joining)
BITRATE=10000k          # hardware encoder; about 1.9 GB per 25 minutes

# "video-file  start-seconds  output-prefix"   (one job per line)
JOBS=(
  "$D/VID_20260608_201023_10_030.mp4 47 $D/g030"
  "$D/VID_20260608_203929_10_031.mp4 0  $D/g031"
)
# ────────────────────────────────────────────────────────────────────────────

# never touch the terminal: a background job that reads it gets frozen (state "T") by the shell
exec < /dev/null
LOG=$D/overnight.log
mkdir -p "$D"
cd "$REPO" || { echo "cannot cd to $REPO" >> "$LOG"; exit 1; }
[ -f .venv/bin/activate ] && source .venv/bin/activate

say() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }

# keep the Mac awake for as long as this script runs
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i -w $$ &
fi

FAILED=()

# step <marker> <description> <command...>   runs once; logs; records success with a marker file
step() {
  local marker=$1 desc=$2; shift 2
  if [ -f "$marker" ]; then say "skip   $desc (already done)"; return 0; fi
  say "start  $desc"
  local t0=$SECONDS
  "$@" >> "$LOG" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    touch "$marker"
    say "done   $desc  ($(( (SECONDS - t0) / 60 )) min)"
  else
    say "FAILED $desc  (exit $rc; see the lines above in this log)"
    FAILED+=("$desc")
  fi
  return $rc
}

say "=== overnight run: ${#JOBS[@]} video(s) ==="

# 1) players + keyframes for every video first, so the keyframes exist even if a long render fails later
for job in "${JOBS[@]}"; do
  read -r VIDEO START PREFIX <<< "$job"
  [ -f "$VIDEO" ] || { say "FAILED missing video $VIDEO"; FAILED+=("missing $VIDEO"); continue; }
  step "$PREFIX.detections.done" "extract players: $(basename "$VIDEO")" \
    python extract_detections.py --video "$VIDEO" --lon-range -80 80 --no-ball --players "$PLAYERS" \
      --start "$START" --out "$PREFIX.detections.jsonl" \
  && step "$PREFIX.keyframes.done" "keyframes: $(basename "$VIDEO")" \
    python auto_keyframes.py --detections "$PREFIX.detections.jsonl" --profile "$PROFILE" \
      --out "$PREFIX.auto.keyframes.json"
done

# 2) the renders
for job in "${JOBS[@]}"; do
  read -r VIDEO START PREFIX <<< "$job"
  [ -f "$PREFIX.auto.keyframes.json" ] || { say "skip   render $(basename "$VIDEO"): no keyframes"; continue; }
  step "$PREFIX.render.done" "render: $(basename "$VIDEO")" \
    python render_keyframes.py --input "$VIDEO" --keyframes "$PREFIX.auto.keyframes.json" \
      --output "$PREFIX.auto.mp4" --audio --encoder h264_videotoolbox --bitrate "$BITRATE"
done

# 3) join the finished renders into one file (same encoder and size, so no re-encoding: takes seconds)
join_all() {
  local list=$D/.join.txt; : > "$list"
  for job in "${JOBS[@]}"; do
    read -r VIDEO START PREFIX <<< "$job"
    [ -f "$PREFIX.auto.mp4" ] || { echo "missing $PREFIX.auto.mp4"; return 1; }
    echo "file '$PREFIX.auto.mp4'" >> "$list"
  done
  ffmpeg -y -v error -f concat -safe 0 -i "$list" -c copy "$JOIN_OUT"
}
[ -n "$JOIN_OUT" ] && step "$JOIN_OUT.done" "join: $(basename "$JOIN_OUT")" join_all

say "=== finished ==="
for job in "${JOBS[@]}"; do
  read -r VIDEO START PREFIX <<< "$job"
  [ -f "$PREFIX.auto.mp4" ] && say "output  $(ls -lh "$PREFIX.auto.mp4" | awk '{print $5}')  $PREFIX.auto.mp4"
done
[ -n "$JOIN_OUT" ] && [ -f "$JOIN_OUT" ] && say "output  $(ls -lh "$JOIN_OUT" | awk '{print $5}')  $JOIN_OUT"
if [ ${#FAILED[@]} -gt 0 ]; then
  say "${#FAILED[@]} step(s) failed:"
  for f in "${FAILED[@]}"; do say "  - $f"; done
  exit 1
fi
say "all steps succeeded"
