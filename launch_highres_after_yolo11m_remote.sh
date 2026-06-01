#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/root/insta360_ball_v5_20260601}"
WAIT_PID="${WAIT_PID:-6446}"
SCRIPT="${SCRIPT:-queue_ball_v5_highres_after_yolo11m_remote.sh}"
PROJECT="${PROJECT:-$WORKSPACE/runs/ball_v5_highres_after_yolo11m}"
NAME="${NAME:-ball_v5_yolo11s_1536_after_yolo11m}"

cd "$WORKSPACE"
mkdir -p logs

ts="$(date -u +%Y%m%d-%H%M%S)"
log="logs/highres_after_yolo11m_${ts}.log"
pidf="logs/highres_after_yolo11m_${ts}.pid"

nohup env \
  WAIT_PID="$WAIT_PID" \
  PROJECT="$PROJECT" \
  NAME="$NAME" \
  bash "$SCRIPT" > "$log" 2>&1 &
child="$!"
echo "$child" > "$pidf"
printf 'pid=%s log=%s pidfile=%s\n' "$child" "$log" "$pidf"
