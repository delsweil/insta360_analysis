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
pip install fastapi uvicorn python-multipart pillow numpy opencv-python

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
