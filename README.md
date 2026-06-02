# ASN Pfeil Phönix — Insta360 Analysis Platform

Football analysis platform for ASN Pfeil Phönix Nürnberg. Processes Insta360 360° footage into autopanned game videos with a web-based review and annotation interface.

---

## Architecture

| Component | What it does | Where it runs |
|---|---|---|
| **Web app** | Next.js frontend — game library, annotation, calibration UI | `localhost:3000` |
| **Worker server** | FastAPI — SD card scanning, ffmpeg extraction, auto-calibration | `localhost:8765` |
| **Supabase** | Database + auth + file storage | Cloud (always on) |
| **Vercel** | Production hosting of the web app | Cloud (auto-deploys on push) |

---

## Prerequisites

Install these once on a new Mac:

```bash
# 1. Xcode command line tools
xcode-select --install

# 2. Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

# 3. Dependencies
brew install node python@3.11 ffmpeg

# 4. Clone repo
git clone https://github.com/delsweil/insta360_analysis.git
cd insta360_analysis

# 5. Python virtual environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 6. Node dependencies
cd web
npm install
cd ..

# 7. Environment variables
echo "NEXT_PUBLIC_SUPABASE_URL=https://ljvocxqvlcfamckkusbg.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key_here" > web/.env.local
```

Get the `NEXT_PUBLIC_SUPABASE_ANON_KEY` from **Supabase dashboard → Project Settings → API → anon public**.

---

## Windows/Linux setup

Install Node.js, Python 3.11, and FFmpeg, then from the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd web
npm install
cd ..
```

For CUDA training or inference, install the PyTorch build that matches the machine's NVIDIA driver before installing the remaining requirements.

---

## Ball tracking v5 workflow

The upgraded ball pipeline is implemented around `autopan_infer.py`, `train_ball_v5.py`,
`equirect_ball_scanner.py`, `ball_tracker_equirect.py`, `pitch_model.py`, and
`team_classifier.py`.

Prepare or refresh the merged Roboflow YOLO dataset:

```bash
python train_ball_v5.py --zip ../ball_detector_merged.v3i.yolov8.zip --extract-dir data/ball_v5 --prepare-only
python audit_ball_dataset.py --data data/ball_v5/data.yaml --json-out results/ball_v5_dataset_audit.json
```

The audit reports duplicate label rows plus `insta360_style`/`veo_style`
composition per split. Use those split-specific counts when judging detector
metrics; aggregate validation/test scores are currently dominated by Veo-style
frames.

Evaluate detector checkpoints with source-domain subsets whenever comparing
candidates:

```bash
python train_ball_v5.py \
  --eval-only \
  --data data/ball_v5/data.yaml \
  --weights models/ball_v5_yolo11s_1280_candidate.pt \
  --split test \
  --eval-domains \
  --metrics-json results/ball_v5_candidate_eval.json \
  --domain-metrics-json results/ball_v5_domain_eval.json
```

`--eval-domains` builds hardlinked/copied YOLO subset datasets under
`runs/ball_v5/domain_subsets/` and evaluates `insta360_style` and `veo_style`
separately. Treat the Insta360 subset as the critical detector signal for the
autopan pipeline. `verify_plan.py` reads `results/ball_v5_domain_eval.json`
and requires at least `0.75` recall on `insta360_style` by default.

Check the remote GPU training job and fetch artifacts:

```bash
python sync_ball_v5_artifacts.py --status
python sync_ball_v5_artifacts.py --fetch-live --skip-live-best
python sync_ball_v5_artifacts.py --fetch
```

`sync_ball_v5_artifacts.py` defaults to the active continuation VM at
`root@69.30.85.25:22187` and
`/root/insta360_ball_v5_continue_20260602`. `--status` reports remote and local
artifact state separately, including the active continuation PID/log and latest
YOLO results row. Use explicit `--host`, `--port`, and `--remote-workspace`
arguments if you need to inspect the older VM workspace.

Final detector artifacts are only current once the remote finalizer has produced
`models/ball_v5.pt`, `models/ball_v5_stable.pt`, and the matching eval JSON
files, or once the m-continuation run has produced
`models/ball_v5_yolo11m_1280_continue_best.pt` plus
`results/ball_v5_yolo11m_1280_continue_eval.json`,
`results/ball_v5_yolo11m_1280_continue_domain_eval.json`, and
`results/ball_v5_yolo11m_1280_continue_status.json`. Live checkpoints are
fetched to `ball_v5_live_*` filenames and should not be treated as promoted
production weights. Use `--skip-live-best` for lightweight status/log/result
sync while a large checkpoint is still changing; the command writes
`results/ball_v5_live_manifest.json` with the active run name, paths, and latest
metrics. Fetch commands also write `results/ball_v5_candidates.json`, a local
manifest of final, stable, preserved candidate, stopped m, continuation m, and
live-cache artifacts. Completed finalizers fetch
`results/ball_v5_stable_domain_eval.json` and, on promotion,
`results/ball_v5_domain_eval.json` alongside the aggregate eval files.

Runs that miss the formal target can still be useful. Preserve those checkpoints
under explicit candidate names, for example
`models/ball_v5_yolo11s_1280_candidate.pt` with
`results/ball_v5_yolo11s_1280_candidate_eval.json`. The worker,
`autopan_infer.py --ball auto`, and the `evaluate.py --approach track_v2`
preset choose the best evaluated local detector candidate rather than blindly
preferring a stale final filename.

To test detector-guided re-anchoring without letting noisy ball detections drive
the Kalman tracker continuously, run:

```bash
python evaluate.py --approach reanchor_ball_v5 --ball-metrics
```

This preset uses the best evaluated local ball_v5 candidate, runs re-anchor mode,
and uses accepted ball detections only to trigger and target confirmed far-ball
re-anchors.

Compare existing approach summaries with:

```bash
python evaluate.py --compare track,reanchor_triggered,reanchor_ball_v5
```

The comparison writes `results/approach_comparison.json` and reports both
clip-mean RMSE, matching the historical table, and frame-weighted RMSE.

Run the implementation verifier:

```bash
python verify_plan.py --allow-blocked --run-python-compile --run-cli-smoke --run-metric-smoke --run-component-smoke --run-eval-smoke
```

To verify the completed m-continuation artifacts before promoting or renaming
them, point the detector gates at the explicit continuation files:

```bash
python verify_plan.py \
  --ball-v5 models/ball_v5_yolo11m_1280_continue_best.pt \
  --ball-v5-eval results/ball_v5_yolo11m_1280_continue_eval.json \
  --ball-v5-status results/ball_v5_yolo11m_1280_continue_status.json \
  --ball-domain-eval results/ball_v5_yolo11m_1280_continue_domain_eval.json
```

The detector metric gate requires `mAP50 >= 0.90` and `mAP50-95 >= 0.60`.
End-to-end pan/ball recall gates also require accessible `.insv` footage and
independently annotated ball ground-truth files. Detector-derived predictor
CSVs are useful as pseudo-label evidence, but they should not be used to pass
the ground-truth recall/RMSE gates.

When evaluation footage and ball GT are mounted, produce the scanner gate
artifact directly from the full-pitch scanner:

```bash
python equirect_ball_scanner.py \
  --insv /path/to/clip.insv \
  --calib calibration/pitch_clip.json \
  --ball auto \
  --out-jsonl results/equirect_ball_scanner_detections.jsonl \
  --summary-json results/equirect_ball_scanner_summary.json \
  --ball-groundtruth annotations/clip_ball_gt.csv
```

Without `--ball-groundtruth`, the scanner summary records only a proxy
`detection_rate`; the verifier requires GT-matched `on_pitch_detection_rate`
for the formal >80% gate.

---

## Running locally

You need **two terminal windows** open simultaneously.

### Terminal 1 — Worker server

```bash
cd ~/insta360_analysis
source .venv/bin/activate
uvicorn worker_server:app --port 8765 --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8765
INFO:     Application startup complete.
```

The worker handles SD card scanning, ffmpeg frame extraction, and auto-calibration. It must be running for the calibration page to work. The `--reload` flag restarts it automatically when you edit `worker_server.py`.

### Terminal 2 — Web app

```bash
cd ~/insta360_analysis/web
npm run dev
```

You should see:
```
▲ Next.js
- Local: http://localhost:3000
```

Then open **http://localhost:3000** in your browser.

---

## Calibration workflow

1. Insert the Insta360 SD card
2. Open **http://localhost:3000/calibrate**
3. The worker auto-detects the SD card — confirm the folder path and click **Scannen**
4. Select recordings to process (test clips are flagged automatically)
5. Fill in: date, opponent, home/away, venue
6. Click **Kalibrierung starten →**
7. For each recording:
   - ffmpeg extracts a frame at 4:00
   - `calibrate_auto.py` suggests a polygon
   - Review the polygon on the canvas — drag points to adjust
   - Use **Linse** to switch between the two lens files if the wrong one loaded
   - Use **Rotation** (CW / CCW / 180°) if the image is sideways or upside down
   - Click **✓ Akzeptieren + Weiter** to save and move to the next file
8. Polygon is saved to both `venues` (template) and `games` (per-game) in Supabase
9. Game status is set to `calibrated` — ready for autopan

---

## Folder structure

```
insta360_analysis/
├── worker_server.py          # FastAPI local worker
├── calibrate_auto.py         # Auto-calibration script
├── autopan_simple.py         # Autopan pipeline (in development)
├── .venv/                    # Python virtual environment
├── .worker_tmp/              # Extracted frames and calibration previews (gitignored)
└── web/                      # Next.js app
    ├── app/
    │   ├── calibrate/        # Calibration page
    │   ├── game/[id]/        # Game detail + annotation
    │   └── add-game/         # Add game manually
    ├── components/
    │   ├── PitchCanvas.tsx   # Interactive polygon editor
    │   └── Topbar.tsx        # Navigation
    └── .env.local            # Supabase credentials (not committed)
```

---

## Verify worker is running

Open **http://localhost:8765/health** in your browser. You should see:

```json
{
  "status": "ok",
  "repo_root": "/Users/.../insta360_analysis",
  "calibrate_script_exists": true
}
```

If `calibrate_script_exists` is `false`, the worker can't find `calibrate_auto.py` — make sure you started it from the repo root.

---

## Deployment

The web app auto-deploys to Vercel on every push to `main`. The worker server is local-only and never deployed — it only needs to run on the Mac Mini during calibration sessions.

```bash
git add .
git commit -m "your message"
git push   # triggers Vercel deploy automatically
```

---

## Games pipeline status

| Status | Meaning |
|---|---|
| `raw` | Game created, no calibration yet |
| `calibrated` | Pitch polygon accepted, ready for autopan |
| `autopanned` | Autopan pipeline complete, video ready |
| `uploaded` | Video uploaded to YouTube |
