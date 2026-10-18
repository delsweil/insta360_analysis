#!/usr/bin/env bash
# run_laufamholz.sh — one half of a game recorded in two video files: extract the players, build the
# camera path, draw the heat map and the game statistics, render the video, join the two parts.
#
#   cd /Users/davidelsweiler/Dropbox/development/insta360_analysis
#   nohup bash run_laufamholz.sh < /dev/null > /dev/null 2>&1 &
#   tail -f "/Users/davidelsweiler/Desktop/erste Mannschaft/laufamholz/overnight.log"      # optional, to look in
#
# Order: players + keyframes for both files -> charts and statistics (they only need the detections and
# the keyframes, so they are ready long before the renders) -> renders -> join.
# Each step leaves a ".done" marker, so running the script again after an interruption picks up where it
# stopped (delete a marker to redo that step). A failure in one step never stops the others.
# Keep the Mac plugged in with the lid OPEN: a closed lid sleeps the machine whatever the script does.
#
# Same method as the 112 trial: the raw video is unwrapped with iv_fov=112 (square pixels), the crop is
# flat 100, tilt -20. Pan and zoom smoothing come from the Altenberg profile.

# ── settings ────────────────────────────────────────────────────────────────
REPO=${REPO:-/Users/davidelsweiler/Dropbox/development/insta360_analysis}
D=${D:-"/Users/davidelsweiler/Desktop/erste Mannschaft/laufamholz"}
PROFILE=${PROFILE:-/Users/davidelsweiler/Desktop/altenberg/altenberg.profile.json}
PLAYERS=models/yolo11s.pt
PLAYERS_ON_PITCH=22                  # for the head-count check
JOIN_OUT=${JOIN_OUT-"$D/laufamholz_half.mp4"}      # both renders joined in the order of JOBS ("" = skip joining)
BITRATE=10000k                       # hardware encoder
TITLE_EN="Laufamholz - one half"
TITLE_DE="Laufamholz - eine Halbzeit"

EXTRACT_ARGS=(--lon-range -80 80 --no-ball --iv-fov 112 --tilt -22)
KEY_ARGS=(--projection flat --fov-max 100 --fov-min 70 --tilt -20 --max-dip 8 --tilt-margin 4)

# "video-file | start-seconds | duration-seconds (empty = to the end of the video) | output-prefix"
# the second file ends at the full-time whistle, 18:29 = 1109 s
JOBS=(
  "$D/VID_20260608_190303_10_034.mp4|70||$D/h034"
  "$D/VID_20260608_193209_10_035.mp4|0|1109|$D/h035"
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

# the current job (set by the loops below)
VIDEO=; START=; DUR=; PREFIX=
split_job() { IFS='|' read -r VIDEO START DUR PREFIX <<< "$1"; }

do_extract() {
  local dur=()
  [ -n "$DUR" ] && dur=(--duration "$DUR")
  python extract_detections.py --video "$VIDEO" "${EXTRACT_ARGS[@]}" --players "$PLAYERS" \
    --start "$START" "${dur[@]}" --out "$PREFIX.detections.jsonl"
}
do_keyframes() {
  python auto_keyframes.py --detections "$PREFIX.detections.jsonl" --profile "$PROFILE" \
    "${KEY_ARGS[@]}" --out "$PREFIX.auto.keyframes.json"
}
do_render() {
  python render_keyframes.py --input "$VIDEO" --keyframes "$PREFIX.auto.keyframes.json" \
    --output "$PREFIX.auto.mp4" --audio --encoder h264_videotoolbox --bitrate "$BITRATE"
}
do_checks() {                                    # per file: how many players were found, flicker, spot checks
  local job
  for job in "${JOBS[@]}"; do
    split_job "$job"
    python detection_check.py --detections "$PREFIX.detections.jsonl" --players-on-pitch "$PLAYERS_ON_PITCH" \
      > "$PREFIX.detection_check.txt" 2>&1 || return 1
  done
}
do_charts() {                                    # heat map / territory (play_map) and game statistics, English and German
  local job det=() kf=()
  for job in "${JOBS[@]}"; do
    split_job "$job"
    det+=("$PREFIX.detections.jsonl"); kf+=("$PREFIX.auto.keyframes.json")
  done
  python play_map.py --detections "${det[@]}" --keyframes "${kf[@]}" --out "$D/laufamholz.play.png" \
      --title "$TITLE_EN" \
  && python play_map.py --detections "${det[@]}" --keyframes "${kf[@]}" --out "$D/laufamholz.play.de.png" \
      --lang de --title "$TITLE_DE" \
  && python game_stats.py --detections "${det[@]}" --keyframes "${kf[@]}" --out "$D/laufamholz.stats" \
      --title "$TITLE_EN" \
  && python game_stats.py --detections "${det[@]}" --keyframes "${kf[@]}" --out "$D/laufamholz.stats.de" \
      --lang de --title "$TITLE_DE"
}
join_all() {                                     # same encoder and size, so no re-encoding: takes seconds
  local list=$D/.join.txt job; : > "$list"
  for job in "${JOBS[@]}"; do
    split_job "$job"
    [ -f "$PREFIX.auto.mp4" ] || { echo "missing $PREFIX.auto.mp4"; return 1; }
    echo "file '$PREFIX.auto.mp4'" >> "$list"
  done
  ffmpeg -y -v error -f concat -safe 0 -i "$list" -c copy "$JOIN_OUT"
}

say "=== laufamholz run: ${#JOBS[@]} video(s) ==="

# 1) players + keyframes for every video first, so the charts exist even if a long render fails later
for job in "${JOBS[@]}"; do
  split_job "$job"
  [ -f "$VIDEO" ] || { say "FAILED missing video $VIDEO"; FAILED+=("missing $VIDEO"); continue; }
  step "$PREFIX.detections.done" "extract players: $(basename "$VIDEO")" do_extract \
  && step "$PREFIX.keyframes.done" "keyframes: $(basename "$VIDEO")" do_keyframes
done

# 2) head-count checks, heat map and statistics (need only the detections and the keyframes)
ALL_KF=1
for job in "${JOBS[@]}"; do split_job "$job"; [ -f "$PREFIX.auto.keyframes.json" ] || ALL_KF=0; done
if [ $ALL_KF = 1 ]; then
  step "$D/checks.done" "detection checks" do_checks
  step "$D/charts.done" "heat map and statistics" do_charts
else
  say "skip   charts: not every file has keyframes"
fi

# 3) the renders
for job in "${JOBS[@]}"; do
  split_job "$job"
  [ -f "$PREFIX.auto.keyframes.json" ] || { say "skip   render $(basename "$VIDEO"): no keyframes"; continue; }
  step "$PREFIX.render.done" "render: $(basename "$VIDEO")" do_render
done

# 4) join the finished renders into one file
[ -n "$JOIN_OUT" ] && step "$JOIN_OUT.done" "join: $(basename "$JOIN_OUT")" join_all

say "=== finished ==="
for job in "${JOBS[@]}"; do
  split_job "$job"
  [ -f "$PREFIX.auto.mp4" ] && say "output  $(ls -lh "$PREFIX.auto.mp4" | awk '{print $5}')  $PREFIX.auto.mp4"
done
[ -n "$JOIN_OUT" ] && [ -f "$JOIN_OUT" ] && say "output  $(ls -lh "$JOIN_OUT" | awk '{print $5}')  $JOIN_OUT"
for f in "$D/laufamholz.play.png" "$D/laufamholz.play.de.png" "$D/laufamholz.stats.png" "$D/laufamholz.stats.de.png"; do
  [ -f "$f" ] && say "output  $f"
done
if [ ${#FAILED[@]} -gt 0 ]; then
  say "${#FAILED[@]} step(s) failed:"
  for f in "${FAILED[@]}"; do say "  - $f"; done
  exit 1
fi
say "all steps succeeded"
