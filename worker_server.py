"""
worker_server.py — Local FastAPI server for the ASN calibration pipeline.

Run from repo root in your venv:
    uvicorn worker_server:app --port 8765 --reload

Endpoints:
    GET  /health                  Health check
    GET  /scan-volumes            Auto-detect SD card mounts under /Volumes
    GET  /scan-folder?path=...    Scan folder for INSV recordings grouped by sequence
    POST /extract-frame           Extract equirect keyframe in-place (no file copy)
    POST /auto-calibrate          Run calibrate_auto.py on an extracted frame
    POST /autopan                 Stub — not yet implemented
"""

import json
import time
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─── Config ──────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.resolve()
WORK_DIR  = REPO_ROOT / ".worker_tmp"
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Files smaller than this are test clips — flagged but not hidden
TEST_CLIP_THRESHOLD_MB = 100

# Bitrate estimate for duration: ~170 MB/min for Insta360 X3 main footage
BITRATE_MB_PER_MIN = 170

app = FastAPI(title="ASN Calibration Worker", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=str(WORK_DIR)), name="files")


# ─── Models ──────────────────────────────────────────────────────────────────

class ExtractRequest(BaseModel):
    insv_path: str
    timestamp: str = "00:04:00"
    rotation: Optional[str] = None


class CalibrateRequest(BaseModel):
    frame_path: str
    scale: float = 1.0


class AutopanRequest(BaseModel):
    insv_path: str
    polygon: list


# ─── Helpers ─────────────────────────────────────────────────────────────────

def job_dir(insv_path: str) -> Path:
    slug = re.sub(r"[^\w\-]", "_", Path(insv_path).stem)
    d = WORK_DIR / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def pitch_in_lower_half(frame_path: str) -> bool:
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(frame_path).convert("RGB")
        _, h = img.size
        arr = np.array(img)
        green = (
            (arr[:, :, 1].astype(int) - arr[:, :, 0].astype(int) > 20) &
            (arr[:, :, 1].astype(int) - arr[:, :, 2].astype(int) > 20)
        )
        return green[h // 2:, :].sum() >= green[:h // 2, :].sum()
    except Exception:
        return True


def ffmpeg_extract(insv_path: str, output_path: str, timestamp: str, rotation_flag: str) -> subprocess.CompletedProcess:
    return subprocess.run([
        "ffmpeg", "-y",
        "-ss", timestamp,
        "-i", insv_path,
        "-vf", f"rotate={rotation_flag},v360=fisheye:equirect:ih_fov=200:iv_fov=200,scale=2880:1440",
        "-frames:v", "1", "-update", "1",
        output_path,
    ], capture_output=True, text=True, timeout=120)


def extract_with_auto_rotation(insv_path: str, output_path: str, timestamp: str, rotation: Optional[str]) -> dict:
    ROTATIONS = {"cw": "PI/2*3", "ccw": "PI/2", "180": "PI"}

    if rotation in ROTATIONS:
        result = ffmpeg_extract(insv_path, output_path, timestamp, ROTATIONS[rotation])
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")
        return {"rotation": rotation, "auto_detected": False}

    for rot_name, flag in ROTATIONS.items():
        tmp = output_path + f".try_{rot_name}.jpg"
        result = ffmpeg_extract(insv_path, tmp, timestamp, flag)
        if result.returncode != 0:
            continue
        if pitch_in_lower_half(tmp):
            shutil.move(tmp, output_path)
            for other in ROTATIONS:
                p = Path(output_path + f".try_{other}.jpg")
                if p.exists():
                    p.unlink()
            return {"rotation": rot_name, "auto_detected": True}

    # Fallback
    result = ffmpeg_extract(insv_path, output_path, timestamp, ROTATIONS["cw"])
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")
    for other in ROTATIONS:
        p = Path(output_path + f".try_{other}.jpg")
        if p.exists():
            p.unlink()
    return {"rotation": "cw", "auto_detected": True, "fallback": True}


def format_duration(size_bytes: int) -> str:
    mins = size_bytes / (BITRATE_MB_PER_MIN * 1_048_576)
    if mins < 1:
        return f"~{int(mins * 60)}s"
    return f"~{int(mins)}min"


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "repo_root": str(REPO_ROOT),
        "calibrate_script_exists": (REPO_ROOT / "calibrate_auto.py").exists(),
    }


@app.get("/scan-volumes")
def scan_volumes():
    """
    Auto-detect likely SD card or camera mount points under /Volumes.
    Returns volumes that contain at least one .insv file somewhere inside.
    """
    volumes = Path("/Volumes")
    if not volumes.exists():
        return {"volumes": [], "warning": "/Volumes not found"}

    candidates = []
    for vol in sorted(volumes.iterdir()):
        if vol.name.startswith("."):
            continue
        # Quick check: does it contain any .insv file?
        try:
            found = next(vol.rglob("*.insv"), None)
            if found:
                # Return the folder containing the insv files, not the volume root
                candidates.append({
                    "volume": vol.name,
                    "path": str(found.parent),
                    "mount": str(vol),
                })
        except (PermissionError, OSError):
            continue

    return {"volumes": candidates}


@app.get("/scan-folder")
def scan_folder(path: str = Query(..., description="Absolute path to folder containing INSV files")):
    """
    Scan a folder for INSV recordings.
    Groups files by sequence number, filters LRV proxies and .bin metadata.
    Returns recordings sorted newest-first with size, estimated duration,
    and a test_clip flag for short recordings.
    """
    folder = Path(path)
    if not folder.exists():
        raise HTTPException(404, f"Folder not found: {path}")
    if not folder.is_dir():
        raise HTTPException(400, f"Not a directory: {path}")

    # Collect all INSV files, skip LRV proxies and .bin files
    insv_files = []
    try:
        for f in folder.iterdir():
            if f.suffix.lower() != ".insv":
                continue
            if f.name.startswith("."):        # skip macOS shadow files (._filename)
                continue
            if f.name.upper().startswith("LRV"):
                continue
            insv_files.append(f)
    except PermissionError:
        raise HTTPException(403, f"Permission denied: {path}")

    if not insv_files:
        return {"recordings": [], "folder": path, "warning": "No INSV files found (LRV proxies excluded)"}

    # Group by sequence number suffix — the trailing _NNN before .insv
    # e.g. VID_20250629_121009_00_005.insv → sequence key "005"
    # Falls back to grouping by mtime if no numeric suffix found
    def sequence_key(f: Path) -> str:
        # Try to find trailing _NNN pattern
        m = re.search(r'_(\d{3})(?:\.\w+)?$', f.stem)
        if m:
            return m.group(1)
        # Fallback: group by modification date rounded to nearest minute
        mtime = int(f.stat().st_mtime // 60)
        return str(mtime)

    groups: dict[str, list[Path]] = {}
    for f in insv_files:
        key = sequence_key(f)
        groups.setdefault(key, []).append(f)

    recordings = []
    for seq_key, files in sorted(groups.items(), reverse=True):
        files_sorted = sorted(files, key=lambda f: f.name)
        total_size = sum(f.stat().st_size for f in files_sorted)
        is_test = total_size < TEST_CLIP_THRESHOLD_MB * 1_048_576

        # Use the first file's mtime as the recording timestamp
        mtime = min(f.stat().st_mtime for f in files_sorted)
        from datetime import datetime
        recorded_at = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

        recordings.append({
            "sequence": seq_key,
            "recorded_at": recorded_at,
            "test_clip": is_test,
            "estimated_duration": format_duration(total_size),
            "files": [
                {
                    "name": f.name,
                    "path": str(f),
                    "size_mb": round(f.stat().st_size / 1_048_576, 1),
                }
                for f in files_sorted
            ],
        })

    return {"recordings": recordings, "folder": path}


@app.post("/extract-frame")
def extract_frame(req: ExtractRequest):
    """
    Extract an equirect keyframe directly from the file at its current location.
    No copying — works on SD card, local disk, anywhere.
    """
    insv = Path(req.insv_path)
    if not insv.exists():
        raise HTTPException(404, f"File not found: {req.insv_path}")

    jd = job_dir(req.insv_path)
    ts_slug = req.timestamp.replace(":", "-")
    frame_path = str(jd / f"frame_{ts_slug}.jpg")

    try:
        rotation_info = extract_with_auto_rotation(str(insv), frame_path, req.timestamp, req.rotation)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "ffmpeg timed out after 120s")

    if not Path(frame_path).exists():
        raise HTTPException(500, "ffmpeg completed but output file not found")

    rel = Path(frame_path).relative_to(WORK_DIR)
    return {
        "frame_path": frame_path,
        "frame_url": f"http://localhost:8765/files/{rel}?v={int(time.time())}",
        "timestamp": req.timestamp,
        **rotation_info,
    }


@app.post("/auto-calibrate")
def auto_calibrate(req: CalibrateRequest):
    frame = Path(req.frame_path)
    if not frame.exists():
        raise HTTPException(404, f"Frame not found: {req.frame_path}")

    calibrate_script = REPO_ROOT / "calibrate_auto.py"
    if not calibrate_script.exists():
        raise HTTPException(500, "calibrate_auto.py not found in repo root")

    jd = job_dir(req.frame_path)
    poly_out = str(jd / "pitch.json")
    vis_out  = str(jd / "calib_preview.jpg")

    try:
        result = subprocess.run([
            "python", str(calibrate_script),
            "--input",  req.frame_path,
            "--output", poly_out,
            "--vis",    vis_out,
            "--scale",  str(req.scale),
            "--quiet",
        ], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "calibrate_auto.py timed out after 60s")

    if not Path(poly_out).exists():
        detail = result.stderr[-500:] if result.stderr else "No output written"
        raise HTTPException(500, f"calibrate_auto.py produced no output. stderr: {detail}")

    try:
        with open(poly_out) as f:
            pitch_data = json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Invalid JSON from calibrate_auto.py: {e}")

    raw_poly = pitch_data.get("pitch_polygon") or pitch_data.get("polygon") or pitch_data.get("auto_polygon") or []
    img_w = pitch_data.get("frame_width", 2880)
    img_h = pitch_data.get("frame_height", 1440)

    def normalise(pt):
        if isinstance(pt, dict):
            x, y = pt.get("x", 0), pt.get("y", 0)
        else:
            x, y = pt[0], pt[1]
        if x <= 1.0 and y <= 1.0:
            return {"x": round(x, 4), "y": round(y, 4)}
        return {"x": round(x / img_w, 4), "y": round(y / img_h, 4)}

    polygon = [normalise(p) for p in raw_poly]
    confidence = pitch_data.get("confidence")
    if confidence is None:
        n = len(polygon)
        confidence = round(min(1.0, max(0.0, (n - 4) / 12)), 2) if n >= 4 else 0.0

    preview_url = None
    if Path(vis_out).exists():
        rel = Path(vis_out).relative_to(WORK_DIR)
        preview_url = f"http://localhost:8765/files/{rel}"

    return {
        "polygon": polygon,
        "confidence": confidence,
        "point_count": len(polygon),
        "preview_url": preview_url,
        "warnings": ["Low confidence — check polygon carefully"] if confidence < 0.6 else [],
        "exit_code": result.returncode,
    }


@app.post("/autopan")
def autopan(req: AutopanRequest):
    return {
        "status": "not_yet_implemented",
        "message": "Autopan pipeline is under development. Polygon has been saved.",
        "insv_path": req.insv_path,
    }
