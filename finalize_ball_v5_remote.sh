#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/root/insta360_ball_v5_20260601}"
WAIT_PID="${WAIT_PID:-}"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE/.venv_sys/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_ball_v5.py}"
if [[ -f "$WORKSPACE/train_ball_v5.py.next" ]]; then
  TRAIN_SCRIPT="${TRAIN_SCRIPT_NEXT:-train_ball_v5.py.next}"
fi

DATA_YAML="${DATA_YAML:-data/ball_v5/data.yaml}"
PROJECT="${PROJECT:-$WORKSPACE/runs/ball_v5_stable}"
NAME="${NAME:-ball_v5_yolo11s_1280_speed}"
RUN_DIR="${RUN_DIR:-$PROJECT/$NAME}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"

cd "$WORKSPACE"
mkdir -p models results logs

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$WORKSPACE/.yolo_config}"

echo "[finalize start] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[workspace] $WORKSPACE"
echo "[wait_pid] ${WAIT_PID:-none}"
echo "[run_dir] $RUN_DIR"

pid_is_running() {
  local pid="$1"
  local stat
  stat="$(ps -p "$pid" -o stat= 2>/dev/null | awk '{print $1}')"
  [[ -n "$stat" && "$stat" != Z* ]]
}

if [[ -n "$WAIT_PID" ]]; then
  while pid_is_running "$WAIT_PID"; do
    echo "[finalize wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$WAIT_PID still running"
    sleep "${WAIT_INTERVAL_SECONDS:-120}"
  done
  echo "[finalize wait] pid=$WAIT_PID exited"
fi

BEST_PT="$RUN_DIR/weights/best.pt"
STABLE_PT="models/ball_v5_stable.pt"
STABLE_JSON="results/ball_v5_stable_eval.json"
STABLE_DOMAIN_JSON="results/ball_v5_stable_domain_eval.json"
STATUS_JSON="results/ball_v5_finalize_status.json"

if [[ ! -s "$BEST_PT" ]]; then
  echo "[finalize error] missing best checkpoint: $BEST_PT" >&2
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
Path("results").mkdir(exist_ok=True)
Path("results/ball_v5_finalize_status.json").write_text(json.dumps({
    "status": "error",
    "reason": "missing_best_checkpoint",
}, indent=2), encoding="utf-8")
PY
  exit 2
fi

cp "$BEST_PT" "$STABLE_PT"
echo "[finalize copy] $BEST_PT -> $STABLE_PT"

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
  --metrics-json "$STABLE_JSON" \
  --eval-domains \
  --domain-metrics-json "$STABLE_DOMAIN_JSON"

"$PYTHON_BIN" - <<'PY'
import json
import os
import shutil
from pathlib import Path

metrics_path = Path("results/ball_v5_stable_eval.json")
domain_metrics_path = Path("results/ball_v5_stable_domain_eval.json")
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
map50 = float(metrics.get("map50", metrics.get("mAP50", 0.0)))
map5095 = float(metrics.get("map50_95", metrics.get("mAP50-95", metrics.get("map", 0.0))))
target_map50 = float(os.environ.get("TARGET_MAP50", "0.90"))
target_map5095 = float(os.environ.get("TARGET_MAP50_95", "0.60"))
target_domain_recall = float(os.environ.get("TARGET_INSTA360_RECALL", "0.75"))
domain = os.environ.get("TARGET_DOMAIN", "insta360_style")
domain_recall = None
if domain_metrics_path.is_file():
    domain_metrics = json.loads(domain_metrics_path.read_text(encoding="utf-8"))
    for row in domain_metrics.get("domains", []):
        if row.get("domain") == domain:
            domain_recall = float(row.get("recall", row.get("metrics/recall(B)", 0.0)))
            break

metric_passed = map50 >= target_map50 and map5095 >= target_map5095
domain_passed = domain_recall is not None and domain_recall >= target_domain_recall
passed = metric_passed and domain_passed
if not metric_passed:
    status_name = "failed_metric_gate"
elif not domain_passed:
    status_name = "failed_domain_gate"
else:
    status_name = "passed"
status = {
    "status": status_name,
    "map50": map50,
    "map50_95": map5095,
    "target_map50": target_map50,
    "target_map50_95": target_map5095,
    "domain": domain,
    "domain_recall": domain_recall,
    "target_domain_recall": target_domain_recall,
    "promoted": False,
}
print(f"[finalize metrics] map50={map50:.4f} map50_95={map5095:.4f} {domain}_recall={domain_recall}")
if os.environ.get("PROMOTE_ON_PASS", "1") != "0" and passed:
    shutil.copy2("models/ball_v5_stable.pt", "models/ball_v5.pt")
    shutil.copy2(metrics_path, "results/ball_v5_eval.json")
    shutil.copy2(domain_metrics_path, "results/ball_v5_domain_eval.json")
    status["promoted"] = True
    print("[finalize promote] copied stable artifacts to final ball_v5 paths")
else:
    print("[finalize promote] final artifacts not promoted")
Path("results/ball_v5_finalize_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
PY

echo "[finalize complete] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
