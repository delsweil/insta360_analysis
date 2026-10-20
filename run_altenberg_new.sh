#!/usr/bin/env bash
# run_altenberg_new.sh — the current method (iv_fov 112 unwrapping, square detection views, flat 100 crop,
# shirt colours, team analysis) on the 5-minute sample of the Altenberg game, next to the old results.
#
#   cd /Users/davidelsweiler/Dropbox/development/insta360_analysis
#   nohup bash run_altenberg_new.sh < /dev/null > /dev/null 2>&1 &
#   tail -f /Users/davidelsweiler/Desktop/altenberg/new_method.log
#
# Which 5 minutes: the same stretch as the old sample. If $D/g030_5min.detections.jsonl exists its header
# gives start and duration; otherwise START and DURATION below are used. Everything new is written with the
# prefix g030n_5min, so the old g030_5min files are not touched:
#   g030n_5min.detections.jsonl  .auto.keyframes.json  .auto.mp4  .detection_check.txt
#   g030n_5min.play.png  .stats.png  .teams.png  (+ .de versions, .csv)
# About 12 minutes in all (extract ~4, render ~6, charts ~1).

REPO=${REPO:-/Users/davidelsweiler/Dropbox/development/insta360_analysis}
D=${D:-/Users/davidelsweiler/Desktop/altenberg}
VIDEO=${VIDEO:-$D/VID_20260608_201023_10_030.mp4}
OLD=${OLD:-$D/g030_5min}
START=47            # seconds; used only when the old sample's detections file is not there
DURATION=300
PREFIX=$D/g030n_5min
PROFILE=${PROFILE:-$D/altenberg.profile.json}
PLAYERS=models/yolo11s.pt
BITRATE=10000k
TILT_DETECT=-22     # tilt of the detection views; the crop tilt below. Laufamholz values: check the picture, adjust if the near side is cut.
TILT_CROP=-20
EXTRACT_ARGS=(--lon-range -80 80 --no-ball --iv-fov 112 --tilt "$TILT_DETECT" --colours)
KEY_ARGS=(--projection flat --fov-max 100 --fov-min 70 --tilt "$TILT_CROP" --max-dip 8 --tilt-margin 4)

exec < /dev/null
LOG=$D/new_method.log
cd "$REPO" || { echo "cannot cd to $REPO" >> "$LOG"; exit 1; }
[ -f .venv/bin/activate ] && source .venv/bin/activate
say() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }
command -v caffeinate >/dev/null 2>&1 && caffeinate -i -w $$ &

if [ -f "$OLD.detections.jsonl" ]; then
  read -r START DURATION < <(python - "$OLD.detections.jsonl" <<'PY'
import json, sys
h = json.loads(open(sys.argv[1]).readline())
print(h.get("start", 47), h.get("duration", 300))
PY
)
  say "same stretch as the old sample: start ${START}s, duration ${DURATION}s"
else
  say "no old sample found at $OLD.detections.jsonl; using start ${START}s, duration ${DURATION}s"
fi

FAILED=()
step() {
  local marker=$1 desc=$2; shift 2
  if [ -f "$marker" ]; then say "skip   $desc (already done)"; return 0; fi
  say "start  $desc"; local t0=$SECONDS
  "$@" >> "$LOG" 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then touch "$marker"; say "done   $desc  ($(( (SECONDS - t0) / 60 )) min)"
  else say "FAILED $desc  (exit $rc; see the lines above in this log)"; FAILED+=("$desc"); fi
  return $rc
}

do_extract()   { python extract_detections.py --video "$VIDEO" "${EXTRACT_ARGS[@]}" --players "$PLAYERS" \
                   --start "$START" --duration "$DURATION" --out "$PREFIX.detections.jsonl"; }
do_keyframes() { python auto_keyframes.py --detections "$PREFIX.detections.jsonl" --profile "$PROFILE" "${KEY_ARGS[@]}" \
                   --out "$PREFIX.auto.keyframes.json"; }
do_checks()    { python detection_check.py --detections "$PREFIX.detections.jsonl" --players-on-pitch 22 > "$PREFIX.detection_check.txt" 2>&1 \
                 && { [ ! -f "$OLD.detections.jsonl" ] || python detection_check.py --detections "$OLD.detections.jsonl" --players-on-pitch 22 > "$OLD.detection_check.new_run.txt" 2>&1; }; }
do_render()    { python render_keyframes.py --input "$VIDEO" --keyframes "$PREFIX.auto.keyframes.json" --output "$PREFIX.auto.mp4" \
                   --audio --encoder h264_videotoolbox --bitrate "$BITRATE"; }
do_charts() {
  python play_map.py --detections "$PREFIX.detections.jsonl" --keyframes "$PREFIX.auto.keyframes.json" --out "$PREFIX.play.png" --title "Altenberg 5 min (new method)" \
  && python game_stats.py --detections "$PREFIX.detections.jsonl" --keyframes "$PREFIX.auto.keyframes.json" --out "$PREFIX.stats" --title "Altenberg 5 min (new method)" \
  && python team_stats.py --detections "$PREFIX.detections.jsonl" --keyframes "$PREFIX.auto.keyframes.json" --out "$PREFIX.teams" --title "Altenberg 5 min (new method)" \
  && python team_stats.py --detections "$PREFIX.detections.jsonl" --keyframes "$PREFIX.auto.keyframes.json" --out "$PREFIX.teams.de" --lang de --title "Altenberg 5 Minuten (neue Methode)"
}

say "=== Altenberg sample, new method ==="
step "$PREFIX.extract.done" "extract players with colours" do_extract \
 && step "$PREFIX.keyframes.done" "keyframes" do_keyframes \
 && step "$PREFIX.checks.done" "detection checks" do_checks \
 && step "$PREFIX.charts.done" "heat map, statistics, teams" do_charts
step "$PREFIX.render.done" "render" do_render
say "=== finished ==="
for f in "$PREFIX.auto.mp4" "$PREFIX.play.png" "$PREFIX.stats.png" "$PREFIX.teams.png" "$PREFIX.detection_check.txt"; do
  [ -f "$f" ] && say "output  $f"
done
if [ ${#FAILED[@]} -gt 0 ]; then say "${#FAILED[@]} step(s) failed:"; for f in "${FAILED[@]}"; do say "  - $f"; done; exit 1; fi
say "all steps succeeded"
