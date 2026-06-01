#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-$(pwd)}"
cd "$WORKSPACE"

PYTHON_BIN="${PYTHON_BIN:-.venv_sys/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_ball_v5.py}"
if [[ -f train_ball_v5.py.next ]]; then
  TRAIN_SCRIPT="${TRAIN_SCRIPT_NEXT:-train_ball_v5.py.next}"
fi

DATA_YAML="${DATA_YAML:-data/ball_v5/data.yaml}"
MODEL="${MODEL:-yolo11s.pt}"
PROJECT="${PROJECT:-$WORKSPACE/runs/ball_v5_stable}"
NAME="${NAME:-ball_v5_yolo11s_1280_noamp_lr001}"
EPOCHS="${EPOCHS:-180}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-4}"
DEVICE="${DEVICE:-0}"

mkdir -p models results "$PROJECT"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$WORKSPACE/.yolo_config}"

echo "[stable start] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[workspace] $WORKSPACE"
echo "[script] $TRAIN_SCRIPT"
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
  --patience 45 \
  --device "$DEVICE" \
  --optimizer AdamW \
  --lr0 0.001 \
  --lrf 0.01 \
  --warmup-epochs 5 \
  --weight-decay 0.0005 \
  --no-amp \
  --mosaic 0.8 \
  --copy-paste 0.10 \
  --scale 0.70

BEST_PT="$PROJECT/$NAME/weights/best.pt"
STABLE_PT="models/ball_v5_stable.pt"
STABLE_JSON="results/ball_v5_stable_eval.json"

cp "$BEST_PT" "$STABLE_PT"

"$PYTHON_BIN" "$TRAIN_SCRIPT" \
  --eval-only \
  --data "$DATA_YAML" \
  --weights "$STABLE_PT" \
  --split test \
  --project "$PROJECT" \
  --name ball_v5_stable_eval \
  --imgsz 1280 \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --device "$DEVICE" \
  --metrics-json "$STABLE_JSON"

"$PYTHON_BIN" - <<'PY'
import json
import os
import shutil
from pathlib import Path

metrics_path = Path("results/ball_v5_stable_eval.json")
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
map50 = float(metrics.get("map50", metrics.get("mAP50", 0.0)))
map5095 = float(metrics.get("map50_95", metrics.get("mAP50-95", metrics.get("map", 0.0))))
print(f"[stable metrics] map50={map50:.4f} map50_95={map5095:.4f}")
if os.environ.get("PROMOTE_ON_PASS", "1") != "0" and map50 >= 0.90 and map5095 >= 0.60:
    shutil.copy2("models/ball_v5_stable.pt", "models/ball_v5.pt")
    shutil.copy2(metrics_path, "results/ball_v5_eval.json")
    print("[stable promote] copied stable artifacts to ball_v5 final paths")
else:
    print("[stable promote] final artifacts not promoted")
PY

echo "[stable complete] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
