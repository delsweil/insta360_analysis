#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/root/insta360_ball_v5_20260601}"
FINALIZER_PID="${FINALIZER_PID:-}"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE/.venv_sys/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_ball_v5.py}"
if [[ -f "$WORKSPACE/train_ball_v5.py.next" ]]; then
  TRAIN_SCRIPT="${TRAIN_SCRIPT_NEXT:-train_ball_v5.py.next}"
fi

DATA_YAML="${DATA_YAML:-data/ball_v5/data.yaml}"
MODEL="${MODEL:-yolo11m.pt}"
PROJECT="${PROJECT:-$WORKSPACE/runs/ball_v5_followup}"
NAME="${NAME:-ball_v5_yolo11m_1280_followup}"
EPOCHS="${EPOCHS:-220}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"

cd "$WORKSPACE"
mkdir -p models results logs "$PROJECT"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$WORKSPACE/.yolo_config}"

echo "[followup start] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[workspace] $WORKSPACE"
echo "[finalizer_pid] ${FINALIZER_PID:-none}"

pid_is_running() {
  local pid="$1"
  local stat
  stat="$(ps -p "$pid" -o stat= 2>/dev/null | awk '{print $1}')"
  [[ -n "$stat" && "$stat" != Z* ]]
}

if [[ -n "$FINALIZER_PID" ]]; then
  while pid_is_running "$FINALIZER_PID"; do
    echo "[followup wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) finalizer=$FINALIZER_PID still running"
    sleep "${WAIT_INTERVAL_SECONDS:-180}"
  done
  echo "[followup wait] finalizer=$FINALIZER_PID exited"
fi

if [[ -f results/ball_v5_finalize_status.json ]]; then
  STATUS="$("$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
status = json.loads(Path("results/ball_v5_finalize_status.json").read_text(encoding="utf-8"))
print(status.get("status", "unknown"))
PY
)"
  echo "[followup status] previous finalize status=$STATUS"
  if [[ "$STATUS" == "passed" ]]; then
    echo "[followup skip] existing run passed target gate"
    exit 0
  fi
fi

echo "[followup train] model=$MODEL name=$NAME epochs=$EPOCHS batch=$BATCH"
nvidia-smi || true

"$PYTHON_BIN" "$TRAIN_SCRIPT" \
  --data "$DATA_YAML" \
  --model "$MODEL" \
  --project "$PROJECT" \
  --name "$NAME" \
  --imgsz 1280 \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --patience 60 \
  --device "$DEVICE" \
  --optimizer AdamW \
  --lr0 0.0006 \
  --lrf 0.01 \
  --warmup-epochs 6 \
  --weight-decay 0.0006 \
  --amp \
  --mosaic 0.6 \
  --copy-paste 0.05 \
  --scale 0.55 \
  --close-mosaic 30

PROJECT="$PROJECT" NAME="$NAME" BATCH="$BATCH" WORKERS="$WORKERS" DEVICE="$DEVICE" \
  bash finalize_ball_v5_remote.sh

echo "[followup complete] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
