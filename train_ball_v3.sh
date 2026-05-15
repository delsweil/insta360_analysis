#!/usr/bin/env bash
# train_ball_v3.sh — Fine-tune ball detector on new Roboflow dataset
# Usage: bash train_ball_v3.sh ~/Downloads/ball_detector_merged.zip

set -e
ZIPFILE=${1:-~/Downloads/ball_detector_merged.zip}
DATASET_DIR=~/ball_dataset_v3

echo "Unzipping dataset..."
rm -rf "$DATASET_DIR"
unzip "$ZIPFILE" -d "$DATASET_DIR"

echo "Dataset contents:"
ls "$DATASET_DIR"
cat "$DATASET_DIR/data.yaml"

echo "Starting training..."
cd ~/insta360_analysis && source .venv/bin/activate

yolo detect train \
    data="$DATASET_DIR/data.yaml" \
    model=models/ball_v2.pt \
    epochs=50 \
    imgsz=640 \
    batch=16 \
    device=mps \
    project=models/finetune \
    name=ball_v3 \
    patience=10 \
    save=true \
    plots=true

echo "Training complete!"
echo "Best model at: models/finetune/ball_v3/weights/best.pt"
