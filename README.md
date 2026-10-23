# ASN Pfeil Phönix — Insta360 Analysis Platform

Football video analysis for ASN Pfeil Phönix Nürnberg. Raw Insta360 footage goes into a **local app** that makes the autopanned game video, team statistics and board data. The **web player** is where the team watches the game on YouTube, annotates it and uses the tactical board.

```
raw .mp4 files ──► Football Autopan (local app) ──► YouTube video + stats + board data
                                                          │
                                       Web player (Vercel + Supabase): annotations, sharing, tactical board
```

---

## Architecture

| Component | What it does | Where it runs |
|---|---|---|
| **Football Autopan** (`app.py`, `app.html`, `pipeline.py`) | Takes the game's video files, finds the players, plans the camera, renders the autopan video, joins the parts for YouTube, makes statistics and board data | Your own computer (`http://127.0.0.1:…`, opens in the browser) |
| **Web player** (`web/`) | Next.js: game library, YouTube player with annotations, share links, tactical board | `localhost:3000` for development, Vercel in production |
| **Supabase** | Database, login, file storage (`boards` bucket for board data) | Cloud |
| **Vercel** | Hosts the web app, deploys on every push to `main` | Cloud |

Nothing needs to run while the app is idle, and the old SD-card worker server is no longer part of the workflow (see *Retired* at the end).

---

## Football Autopan (for coaches)

Runs on your own computer, nothing is uploaded.

- **Start:** Mac: double-click `start.command` (first time: right-click → Open). Otherwise `python app.py`.
- **Use:** choose the video file(s) of a game and mark each as first or second half, press **Start**, confirm on a picture which shirt colours are the two teams (a first guess is marked), and optionally draw the pitch outline by hand if the automatic one is wrong.
- **Results** are in `~/FootballAutopan/<game>/`:

| File | What it is |
|---|---|
| `*_autopan.mp4` | the video with the camera following the play (one per input file) |
| `*_full_game.mp4`, `*_chapters.txt` | all parts joined in playing order: **upload this one to YouTube**; the chapters file goes into the video description |
| `*_halfN_play.png`, `*_halfN_stats.*` | where the play was, game statistics (N = 1 first half, 2 second half) |
| `*_halfN_teams.png/.csv`, `*_halfN_chartdata.json` | the two teams: heat map, territory, spread, tempo; the numbers behind the charts |
| `*_board.json` | player positions for the tactical board (see below) |
| `v*.detections.jsonl`, `v*.auto.keyframes.json` | every player found (3 a second) and the camera path |

- A stopped game continues where it was (press **Continue**). Computing takes about 2 minutes per minute of video on an Apple-silicon Mac.
- Python packages: `numpy opencv-python scipy matplotlib torch ultralytics` (plus `ffmpeg` from Homebrew).
- Tests: `python -m unittest test_app test_board test_keyframe_tools`.

---

## Tactical board in the web player

The app's last step writes `*_board.json`: every player position (3 a second) in the time of the joined video that is on YouTube, with team and camera direction.

1. One-time: run `web/supabase/boards.sql` in the Supabase SQL editor (creates the `boards` bucket; only coaches and admins may write).
2. In the web player, a coach opens the game, presses **Upload board data** in the Board card and picks the file.
3. The board follows the YouTube time: pause = freeze frame. There are Trails, enlarge, save as picture; share links show it too.
4. If the board is a fraction of a second off, use the Sync buttons and **Save for everyone**.

The board data is tidied before it is written (`board_clean.py`): a player's team is the vote of his short track over about 5 seconds (not just one frame's shirt colour), someone more than 0.5 m outside the pitch outline is shown as "other", and no team has more than 11 players at one moment. The app's log prints how many labels this changed; `--no-tidy` switches it off. Players are placed from both the height of their box and the position of their feet in the picture (`ground_model.py`), which steadies the far side; `--no-feet` uses the box height alone. `ground_check.py` shows how well the two agree for a video.

Positions are estimated from the video (a few metres off, more on the far side): read the shape, not exact distances. This assumes the one joined video is the one on YouTube.

---

## Web player: set-up and running

One-time set-up on a new Mac:

```bash
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
brew install node python@3.11 ffmpeg

git clone https://github.com/delsweil/insta360_analysis.git
cd insta360_analysis
python3.11 -m venv .venv
source .venv/bin/activate
pip install numpy opencv-python scipy matplotlib torch ultralytics

cd web
npm install
cd ..
echo "NEXT_PUBLIC_SUPABASE_URL=https://ljvocxqvlcfamckkusbg.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key_here" > web/.env.local
```

Get the anon key from **Supabase dashboard → Project Settings → API → anon public**.

Run the web app (only needed for development; players use the Vercel site):

```bash
cd ~/insta360_analysis/web
npm run dev
```

Then open **http://localhost:3000**.

---

## Folder structure

```
insta360_analysis/
├── app.py, app.html          # Football Autopan: local server and page
├── pipeline.py               # the steps of a game (check, teams, extract, pitch, keyframes, render, merge, stats, board)
├── extract_detections.py     # YOLO player detection
├── auto_keyframes.py         # camera path from the detections
├── render_keyframes.py       # renders the autopan video
├── team_labels.py, team_stats.py, play_map.py, game_stats.py   # teams, charts, statistics
├── board_data.py             # positions for the web player's board
├── calibrate_auto.py, pitch_filter.py   # automatic pitch outline and filter
├── test_app.py, test_board.py, test_keyframe_tools.py
├── start.command             # Mac double-click start
└── web/                      # Next.js web player
    ├── app/                  # pages: game/[id], share/[token], add-game, admin
    ├── components/           # BoardPanel, AnnotationCanvas, Topbar, ...
    ├── lib/                  # supabase.ts, board.ts
    └── supabase/boards.sql   # one-time storage set-up for the board
```

---

## Deployment

The web app deploys to Vercel on every push to `main`. The local app is never deployed.

```bash
git status --short
git add <the files you changed>
git commit -m "your message"
git push
```

---

## Retired

`worker_server.py` (FastAPI, SD-card scanning and extraction on port 8765) and the web page `/calibrate` that talked to it are replaced by Football Autopan and are no longer used. They are still in the repository; they can be deleted together with `web/app/calibrate/` and `web/components/PitchCanvas.tsx` once nothing links to them.
