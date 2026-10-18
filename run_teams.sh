#!/usr/bin/env bash
# run_teams.sh — team-wise analysis of the Laufamholz half: extract the players again, this time with their
# shirt colours (the camera path and the render stay as they are), then draw the team charts.
#
#   cd /Users/davidelsweiler/Dropbox/development/insta360_analysis
#   nohup bash run_teams.sh < /dev/null > /dev/null 2>&1 &
#   tail -f "/Users/davidelsweiler/Desktop/erste Mannschaft/laufamholz/teams.log"      # optional
#
# About 40 minutes (23 + 15 for the two files). Results next to the videos:
#   laufamholz.teams.png / .csv   and   laufamholz.teams.de.png / .csv
# Each step leaves a ".done" marker, so running it again continues where it stopped.
# Keep the Mac plugged in with the lid OPEN.

REPO=${REPO:-/Users/davidelsweiler/Dropbox/development/insta360_analysis}
D=${D:-"/Users/davidelsweiler/Desktop/erste Mannschaft/laufamholz"}
PLAYERS=models/yolo11s.pt
TITLE_EN="Laufamholz - one half, by team"
TITLE_DE="Laufamholz - eine Halbzeit, nach Teams"
EXTRACT_ARGS=(--lon-range -80 80 --no-ball --iv-fov 112 --tilt -22 --colours)

# "video-file | start-seconds | duration-seconds (empty = to the end) | output-prefix"
JOBS=(
  "$D/VID_20260608_190303_10_034.mp4|70||$D/h034c"
  "$D/VID_20260608_193209_10_035.mp4|0|1109|$D/h035c"
)

exec < /dev/null
LOG=$D/teams.log
cd "$REPO" || { echo "cannot cd to $REPO" >> "$LOG"; exit 1; }
[ -f .venv/bin/activate ] && source .venv/bin/activate
say() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }
command -v caffeinate >/dev/null 2>&1 && caffeinate -i -w $$ &

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
VIDEO=; START=; DUR=; PREFIX=
split_job() { IFS='|' read -r VIDEO START DUR PREFIX <<< "$1"; }
do_extract() {
  local dur=(); [ -n "$DUR" ] && dur=(--duration "$DUR")
  python extract_detections.py --video "$VIDEO" "${EXTRACT_ARGS[@]}" --players "$PLAYERS" \
    --start "$START" "${dur[@]}" --out "$PREFIX.detections.jsonl"
}
do_teams() {
  local job det=(); for job in "${JOBS[@]}"; do split_job "$job"; det+=("$PREFIX.detections.jsonl"); done
  python team_stats.py --detections "${det[@]}" --out "$D/laufamholz.teams" --title "$TITLE_EN" \
  && python team_stats.py --detections "${det[@]}" --out "$D/laufamholz.teams.de" --lang de --title "$TITLE_DE"
}

say "=== laufamholz team run: ${#JOBS[@]} video(s) ==="
ALL=1
for job in "${JOBS[@]}"; do
  split_job "$job"
  [ -f "$VIDEO" ] || { say "FAILED missing video $VIDEO"; FAILED+=("missing $VIDEO"); ALL=0; continue; }
  step "$PREFIX.detections.done" "extract players with colours: $(basename "$VIDEO")" do_extract || ALL=0
done
[ $ALL = 1 ] && step "$D/teams.done" "team charts" do_teams
say "=== finished ==="
for f in "$D/laufamholz.teams.png" "$D/laufamholz.teams.de.png" "$D/laufamholz.teams.csv"; do
  [ -f "$f" ] && say "output  $f"
done
if [ ${#FAILED[@]} -gt 0 ]; then say "${#FAILED[@]} step(s) failed:"; for f in "${FAILED[@]}"; do say "  - $f"; done; exit 1; fi
say "all steps succeeded"
