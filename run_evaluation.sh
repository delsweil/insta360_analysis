#!/usr/bin/env bash
# run_evaluation.sh — Multi-clip evaluation with per-clip calibrations
set -e
cd ~/insta360_analysis
source .venv/bin/activate

PLAYERS="models/yolo11s.pt"
BALL="models/ball_v4.pt"
SEG_DURATION=45
N_SEGMENTS=6
SEED=99
OUTDIR="/tmp/eval_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUTDIR"

echo "Output: $OUTDIR"
echo "Segments: ${N_SEGMENTS} x ${SEG_DURATION}s"
echo ""

run_clip() {
    local KEY=$1
    local INSV=$2
    local INSPRJ=$3
    local CALIB=$4

    if [ ! -f "$INSV" ]; then
        echo "SKIP $KEY — insv not found: $INSV"
        return
    fi
    if [ ! -f "$CALIB" ]; then
        echo "SKIP $KEY — calibration not found: $CALIB"
        return
    fi

    echo "=== Clip $KEY: $(basename $INSV) ==="
    local CSV="$OUTDIR/${KEY}_pan.csv"
    local PNG="$OUTDIR/${KEY}_comparison.png"

    python3 autopan_infer.py \
        --insv "$INSV" \
        --calib "$CALIB" \
        --players "$PLAYERS" \
        --ball "$BALL" \
        --segments $N_SEGMENTS \
        --seg-duration $SEG_DURATION \
        --seed $SEED \
        --log-csv "$CSV" \
        --output "$OUTDIR/${KEY}_infer.mp4" \
        --device mps

    python3 compare_pan.py \
        --insprj "$INSPRJ" \
        --predicted "$CSV" \
        --seg-duration $SEG_DURATION \
        --out "$PNG"

    echo ""
}

run_clip "001" \
    "/Volumes/Sickis disk/DCIM/Camera01/VID_20241028_144020_10_001.insv" \
    "VID_20241028_144020_00_001.insv.insprj" \
    "calibration/pitch_VID_20241028_144020_10_001.insv.json"

run_clip "002" \
    "/Volumes/Sickis disk/DCIM/Camera01/VID_20241028_150939_10_002.insv" \
    "VID_20241028_150939_00_002.insv.insprj" \
    "calibration/pitch_VID_20241028_150939_10_002.insv.json"

run_clip "003" \
    "/Volumes/Sickis disk/DCIM/Camera01/VID_20241028_153158_10_003.insv" \
    "VID_20241028_153158_00_003.insv.insprj" \
    "calibration/pitch_VID_20241028_153158_10_003.insv.json"

run_clip "004" \
    "/Volumes/Sickis disk/DCIM/Camera01/VID_20241028_160117_10_004.insv" \
    "VID_20241028_160117_00_004.insv.insprj" \
    "calibration/pitch_VID_20241028_160117_10_004.insv.json"

run_clip "005" \
    "/Volumes/Sickis disk/DCIM/Camera01/VID_20241028_174053_10_005.insv" \
    "VID_20241028_174053_00_005.insv.insprj" \
    "calibration/pitch_VID_20241028_174053_10_005.insv.json"

echo "Done. Results in: $OUTDIR"
open "$OUTDIR"
