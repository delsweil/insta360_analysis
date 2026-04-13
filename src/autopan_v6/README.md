# autopan v6

Insta360 football autopan pipeline. Converts raw `.insv` files into a
smooth, autopanned MP4 suitable for YouTube upload.

## What's new vs v5

- **No `py360convert` per frame** — the most expensive operation in v5.
  FFmpeg's `v360` filter handles all projection during the final render.
- **Sampled detection** — YOLO runs on 1-in-8 frames by default, not every frame.
- **Savitzky-Golay smoothing** — produces smoother pan than per-frame EMA,
  with better handling of fast action and ball disappearances.
- **Two-phase pipeline** — detection and render are separate steps.
  You can re-render with different settings without re-running detection.
- **Works directly on `.insv`** — no Insta360 Studio export step required.
- **Flexible camera support** — X2, X3, X4, ONE RS, ONE X all supported.

## Setup

### 1. Install FFmpeg

```bash
# Mac
brew install ffmpeg

# Linux
sudo apt install ffmpeg

# Verify
ffmpeg -version
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Download YOLO models

```bash
# Creates models/ directory with yolov8n.pt
python -c "
from ultralytics import YOLO
import os
os.makedirs('models', exist_ok=True)
YOLO('yolov8n.pt')  # downloads ~6MB nano model
import shutil; shutil.move('yolov8n.pt', 'models/yolo11n.pt')
"
```

For the ball model, place your `ball.pt` in `models/`. If you don't have
a custom ball model, the pipeline falls back to player-centroid targeting.

## Usage

### Quick start (full pipeline)

```bash
python main.py run /path/to/game_folder/ \
    --model "ONE X2" \
    --output output/game.mp4
```

### Step 1: Calibrate pitch boundary (recommended, do once per pitch)

```bash
python main.py calibrate /path/to/segment.insv
```

Click the leftmost and rightmost points of the pitch boundary on the
fisheye frame. Press S to save to `calibration/pitch.json`.

### Step 2: Probe files

```bash
python main.py probe /path/to/game_folder/
```

Shows resolution, fps, duration, estimated processing time.

### Step 3: Detect (slow phase)

```bash
python main.py detect /path/to/game_folder/ \
    --model "ONE X2" \
    --sample-every 8 \
    --output data/yaw_schedule.csv
```

On a 2019 Intel MBP, expect ~90-120 minutes for a 90-minute game
with `--sample-every 8`. Use `--sample-every 12` to halve the time
with minimal quality impact.

### Step 4: Render (fast phase)

```bash
# First smooth the schedule
python main.py smooth data/yaw_schedule.csv \
    --fps 30 \
    --total-frames 162000 \
    --output data/yaw_curve.npz

# Then render
python main.py render /path/to/game_folder/ data/yaw_curve.npz \
    --output output/game.mp4
```

Render typically takes 15-30 minutes on the Intel Mac (VideoToolbox).

## Performance tuning

| Setting | Default | Faster | Slower/Better |
|---------|---------|--------|---------------|
| `--sample-every` | 8 | 12-16 | 4-6 |
| `--detection-width` | 1280 | 960 | 1920 |
| `--encoder` | auto | libx264 faster | h264_videotoolbox |

## Project structure

```
autopan_v6/
├── main.py                 # CLI entry point
├── requirements.txt
├── pipeline/
│   ├── lens_models.py      # Camera parameter table
│   ├── probe.py            # File inspection and metadata
│   ├── decode.py           # FFmpeg frame extraction
│   ├── detect.py           # YOLO detection + yaw schedule
│   ├── smooth.py           # Savitzky-Golay curve fitting
│   ├── render.py           # FFmpeg v360 render
│   └── calibrate.py        # Interactive pitch calibration
├── models/
│   ├── yolo11n.pt          # Player detection model
│   └── ball.pt             # Ball detection model
├── calibration/
│   └── pitch.json          # Pitch boundary calibration
├── data/
│   ├── yaw_schedule.csv    # Sparse detection output
│   └── yaw_curve.npz       # Dense smoothed curve
└── output/
    └── game.mp4            # Final output
```
