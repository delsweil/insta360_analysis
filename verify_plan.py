#!/usr/bin/env python3
"""Verify implementation_plan.md requirements against the current checkout.

The implementation plan includes both code deliverables and empirical gates.
This verifier keeps those separate:

- static checks prove files, flags, and integration points exist
- external gates require footage, labels, trained weights, and result summaries

Exit codes:
  0: all checked gates passed, or blocked gates were allowed
  1: at least one implementation/static check failed
  2: only external-evidence gates are incomplete
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal


Status = Literal["pass", "fail", "blocked_external", "not_run"]


ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"


@dataclass
class Check:
    phase: str
    requirement: str
    status: Status
    evidence: str


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def has_all(path: Path, needles: Iterable[str]) -> bool:
    if not path.exists():
        return False
    text = read_text(path)
    return all(needle in text for needle in needles)


def check_file(path: str, phase: str, requirement: str) -> Check:
    p = ROOT / path
    return Check(
        phase,
        requirement,
        "pass" if p.exists() else "fail",
        f"{rel(p)} {'exists' if p.exists() else 'is missing'}",
    )


def check_markers(path: str, markers: Iterable[str], phase: str, requirement: str) -> Check:
    p = ROOT / path
    missing = []
    text = read_text(p) if p.exists() else ""
    for marker in markers:
        if marker not in text:
            missing.append(marker)
    status: Status = "pass" if p.exists() and not missing else "fail"
    evidence = f"{rel(p)} contains required markers" if not missing else f"{rel(p)} missing markers: {missing}"
    return Check(phase, requirement, status, evidence)


def run_cmd(cmd: list[str], cwd: Path, timeout_s: int = 120, env: dict[str, str] | None = None) -> Check:
    label = " ".join(cmd)
    resolved_cmd = cmd[:]
    found = shutil.which(resolved_cmd[0])
    if found is None and sys.platform == "win32" and Path(resolved_cmd[0]).suffix == "":
        found = shutil.which(f"{resolved_cmd[0]}.cmd") or shutil.which(f"{resolved_cmd[0]}.exe")
    if found is not None:
        resolved_cmd[0] = found
    try:
        result = subprocess.run(
            resolved_cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except Exception as exc:
        return Check("automated", label, "fail", str(exc))
    output = (result.stdout + "\n" + result.stderr).strip()
    if len(output) > 1200:
        output = output[:1200] + "...<truncated>"
    return Check("automated", label, "pass" if result.returncode == 0 else "fail", output or "no output")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_compile_files() -> list[str]:
    files = [
        "autopan_infer.py",
        "autopan_simple.py",
        "evaluate.py",
        "worker_server.py",
        "equirect_ball_scanner.py",
        "ball_tracker_equirect.py",
        "train_ball_v5.py",
        "pitch_model.py",
        "team_classifier.py",
        "extract_frames.py",
        "studio_replicate.py",
        "game_state.py",
        "sync_ball_v5_artifacts.py",
        "verify_plan.py",
        "audit_ball_dataset.py",
    ]
    files.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT / "src" / "autopan").glob("*.py")))
    return files


def external_asset_checks(args) -> list[Check]:
    checks: list[Check] = []
    checks.append(dataset_zip_gate(Path(args.dataset_zip), min_images=3000))
    ball_v5_candidates = [
        ROOT / args.ball_v5,
        ROOT / "models" / "ball_v5_yolo11s_1280_candidate.pt",
        ROOT / "models" / "ball_v5_stable.pt",
        ROOT / "models" / "ball_v5_live_best.pt",
    ]
    available_ball_v5 = [p for p in ball_v5_candidates if p.exists()]
    checks.append(Check(
        "phase1_detector",
        "trained ball_v5 weights or preserved candidates are available for mAP evaluation",
        "pass" if available_ball_v5 else "blocked_external",
        "available: " + ", ".join(rel(p) for p in available_ball_v5)
        if available_ball_v5 else f"none of {[rel(p) for p in ball_v5_candidates]} exist",
    ))
    checks.append(yolo_data_yaml_gate(ROOT / args.ball_data))

    clip_defs = load_clip_defs(args.clips_file)
    missing_insv = [cid for cid, clip in clip_defs.items() if not Path(clip.get("insv", "")).exists()]
    missing_insprj = [cid for cid, clip in clip_defs.items() if not Path(clip.get("insprj", "")).exists()]
    missing_calib = [cid for cid, clip in clip_defs.items() if not Path(clip.get("calib", "")).exists()]
    checks.append(Check(
        "phase2_phase5_eval",
        "default evaluation INSV footage is accessible",
        "pass" if not missing_insv else "blocked_external",
        "all configured INSV files exist" if not missing_insv else f"missing INSV clips: {missing_insv}",
    ))
    checks.append(Check(
        "phase5_eval",
        "Insta360 Studio keyframe ground truth is accessible",
        "pass" if not missing_insprj else "blocked_external",
        "all configured .insprj files exist" if not missing_insprj else f"missing insprj clips: {missing_insprj}",
    ))
    checks.append(Check(
        "phase5_eval",
        "pitch calibration files are accessible",
        "pass" if not missing_calib else "blocked_external",
        "all configured calibration files exist" if not missing_calib else f"missing calibration clips: {missing_calib}",
    ))

    gt_dir = Path(args.ball_groundtruth_dir) if args.ball_groundtruth_dir else None
    if gt_dir:
        missing_gt = []
        for cid in clip_defs:
            if not any((gt_dir / f"{cid}{suffix}").exists() for suffix in (".csv", ".json", ".jsonl")):
                missing_gt.append(cid)
        checks.append(Check(
            "phase5_eval",
            "ball ground-truth files exist for recall/RMSE metrics",
            "pass" if not missing_gt else "blocked_external",
            "all clip ball-GT files exist" if not missing_gt else f"missing ball-GT clips: {missing_gt}",
        ))
    else:
        checks.append(Check(
            "phase5_eval",
            "ball ground-truth files exist for recall/RMSE metrics",
            "blocked_external",
            "--ball-groundtruth-dir was not provided",
        ))

    return checks


def parse_simple_yaml_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def yolo_data_yaml_gate(path: Path) -> Check:
    if not path.exists():
        return Check(
            "phase1_detector",
            "local ball_v5 YOLO data.yaml points to accessible image folders",
            "blocked_external",
            f"{rel(path)} is missing",
        )
    values = parse_simple_yaml_values(path)
    base = Path(values.get("path", path.parent))
    if not base.is_absolute():
        base = path.parent / base
    missing = []
    for key in ("train", "val"):
        value = values.get(key)
        if not value:
            missing.append(f"{key}:<missing>")
            continue
        target = Path(value)
        if not target.is_absolute():
            target = base / target
        if not target.exists():
            missing.append(f"{key}:{target}")
    if missing:
        return Check(
            "phase1_detector",
            "local ball_v5 YOLO data.yaml points to accessible image folders",
            "blocked_external",
            f"{rel(path)} exists but image folders are not local/accessible: {missing}",
        )
    return Check(
        "phase1_detector",
        "local ball_v5 YOLO data.yaml points to accessible image folders",
        "pass",
        f"{rel(path)} references existing train/val image folders",
    )


def dataset_zip_gate(path: Path, min_images: int) -> Check:
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return Check(
            "phase1_detector",
            "expanded ball detector annotation ZIP is available",
            "blocked_external",
            f"{rel(path)} missing",
        )
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            images = [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
            labels = [n for n in names if n.lower().endswith(".txt") and "/labels/" in n]
            data_yaml = [n for n in names if n.endswith("data.yaml")]
    except Exception as exc:
        return Check("phase1_detector", "expanded ball detector annotation ZIP is readable", "fail", str(exc))
    enough = len(images) >= min_images and len(labels) >= min_images and bool(data_yaml)
    return Check(
        "phase1_detector",
        f"expanded ball detector annotation ZIP has at least {min_images} images/labels",
        "pass" if enough else "fail",
        f"{rel(path)} images={len(images)}, labels={len(labels)}, data_yaml={len(data_yaml)}",
    )


def load_clip_defs(path: str | None) -> dict:
    if path:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return {str(item.get("id") or item.get("clip") or i): item for i, item in enumerate(raw, start=1)}
        return raw

    from evaluate import DEFAULT_CLIPS

    return DEFAULT_CLIPS


def metric_gate_checks(args) -> list[Check]:
    checks: list[Check] = []
    results_dir = ROOT / args.results_dir
    track_v2_summary = results_dir / "track_v2_summary.json"
    checks.append(summary_rmse_gate(track_v2_summary, 15.0))
    checks.append(ball_metric_gate(track_v2_summary, "ball_gt_recall_deg3", 0.70, "Ball detection recall > 70%"))
    checks.append(ball_metric_gate(track_v2_summary, "median_ball_track_s", 5.0, "Track continuity median > 5s"))
    checks.append(scanner_gate(results_dir / "equirect_ball_scanner_summary.json", 0.80))
    checks.append(detector_gate(
        results_dir / "ball_v5_eval.json",
        map50=0.90,
        map5095=0.60,
        weights_path=ROOT / args.ball_v5,
        finalize_status_path=results_dir / "ball_v5_finalize_status.json",
    ))
    return checks


def ball_metric_smoke_check() -> Check:
    """Exercise evaluate_ball_metrics with synthetic pan CSV and ball GT."""
    try:
        from evaluate import evaluate_ball_metrics

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            pan_csv = tmpdir / "pan.csv"
            gt_csv = tmpdir / "gt.csv"
            pan_csv.write_text(
                "\n".join([
                    "timestamp_s,predicted_pan_deg,mode,ball_conf,ball_lon,ball_lat,tracker_lon,tracker_lat",
                    "0,0,ball,0.9,179,0,179,0",
                    "1,0,ball,0.8,178,0,178,0",
                    "2,0,players,,0,0,-179,0",
                    "3,0,equirect_kalman,0.7,11,0,11,0",
                    "4,0,hold,,11,0,11,0",
                    "",
                ]),
                encoding="utf-8",
            )
            gt_csv.write_text(
                "\n".join([
                    "timestamp_s,lon,lat",
                    "0,-179,0",
                    "1.5,-180,0",
                    "2,-178,0",
                    "4,10,0",
                    "",
                ]),
                encoding="utf-8",
            )
            metrics = evaluate_ball_metrics(str(pan_csv), fps=1.0, ball_gt_path=str(gt_csv))

        expected = {
            "ball_detection_rate": 3.0 / 5.0,
            "ball_mode_rate": 3.0 / 5.0,
            "longest_ball_track_s": 2.0,
            "median_ball_track_s": 1.5,
            "ball_gt_recall_deg3": 1.0,
        }
        failures = []
        for key, want in expected.items():
            got = float(metrics.get(key, float("nan")))
            if abs(got - want) > 1e-6:
                failures.append(f"{key}={got} expected {want}")
        if "ball_gt_rmse_deg" not in metrics or not math.isfinite(float(metrics["ball_gt_rmse_deg"])):
            failures.append("ball_gt_rmse_deg missing/non-finite")
        return Check(
            "automated",
            "synthetic ball metric smoke test",
            "fail" if failures else "pass",
            "; ".join(failures) if failures else json.dumps(metrics, sort_keys=True),
        )
    except Exception as exc:
        return Check("automated", "synthetic ball metric smoke test", "fail", str(exc))


def component_smoke_check() -> Check:
    """Exercise core v5 components on synthetic inputs."""
    try:
        import numpy as np

        from ball_tracker_equirect import (
            BallMeasurement,
            MultiHypothesisEquirectTracker,
            shortest_lon_delta,
        )
        from equirect_ball_scanner import (
            ScanDetection,
            lon_lat_to_perspective_pixel,
            merge_scan_detections,
            perspective_pixel_to_lon_lat,
            scan_summary_payload,
        )
        from game_state import GameStatePredictor
        from pitch_model import PitchAwareBallGate, PitchHomographyModel
        from team_classifier import TeamClassifier

        failures: list[str] = []

        tracker = MultiHypothesisEquirectTracker(max_hypotheses=3)
        tracker.update([
            BallMeasurement(179.0, 0.0, 0.9),
            BallMeasurement(-45.0, 0.0, 0.2),
        ])
        tracker.predict()
        tracker.update([BallMeasurement(-179.0, 0.2, 0.85)])
        snap = tracker.snapshot()
        if snap is None or abs(shortest_lon_delta(snap.lon, -179.0)) > 3.0:
            failures.append(f"tracker seam update failed: {snap}")

        lon, lat = perspective_pixel_to_lon_lat(620.0, 250.0, 12.0, -8.0, 110.0, 1280, 720)
        x, y = lon_lat_to_perspective_pixel(lon, lat, 12.0, -8.0, 110.0, 1280, 720)
        if abs(x - 620.0) > 1e-6 or abs(y - 250.0) > 1e-6:
            failures.append(f"scanner projection roundtrip failed: {(x, y)}")
        merged = merge_scan_detections([
            ScanDetection(179.5, 0.0, 0.8, 0.0, 10.0, 10.0, (1, 1, 2, 2)),
            ScanDetection(-179.4, 0.2, 0.7, 15.0, 11.0, 10.0, (1, 1, 2, 2)),
            ScanDetection(30.0, 0.0, 0.6, 30.0, 12.0, 10.0, (1, 1, 2, 2)),
        ], merge_dist_deg=2.0)
        if len(merged) != 2 or abs(shortest_lon_delta(merged[0].lon, 180.0)) > 1.0:
            failures.append(f"scan merge failed: {merged}")
        summary = scan_summary_payload(
            scanned_frames=10,
            detection_hits=8,
            gt_frames=5,
            gt_hits=4,
            gt_errors=[1.0, 2.0, 3.0, 4.0],
            args=type("Args", (), {
                "every": 15,
                "start": 0.0,
                "duration": 30.0,
                "conf": 0.2,
                "imgsz": 640,
                "no_sliced": False,
                "ball": "models/ball_v5.pt",
                "gt_match_deg": 3.0,
                "gt_max_dt": 0.25,
            })(),
            scan_yaws=[-45, 0, 45],
        )
        if abs(summary.get("detection_rate", 0.0) - 0.8) > 1e-9:
            failures.append(f"scanner detection summary failed: {summary}")
        if abs(summary.get("on_pitch_detection_rate", 0.0) - 0.8) > 1e-9:
            failures.append(f"scanner GT summary failed: {summary}")

        with tempfile.TemporaryDirectory() as tmp:
            calib = Path(tmp) / "pitch.json"
            calib.write_text(json.dumps({
                "source_frame": {"width": 2880, "height": 1440},
                "pixel_polygon": [[0, 0], [2880, 0], [2880, 1440], [0, 1440]],
            }), encoding="utf-8")
            pitch = PitchHomographyModel.from_calibration(calib)
            center = pitch.lon_lat_to_pitch(0.0, 0.0)
            if abs(center.x_m - 52.5) > 0.2 or abs(center.y_m - 34.0) > 0.2:
                failures.append(f"pitch center mapping failed: {center}")
            px, py = pitch.pitch_to_image(center.x_m, center.y_m)
            if abs(px - 1440.0) > 1.0 or abs(py - 720.0) > 1.0:
                failures.append(f"pitch roundtrip failed: {(px, py)}")
            gate = PitchAwareBallGate(pitch, max_speed_kmh=10.0)
            if not gate.check(0.0, 0.0, 0.0):
                failures.append("pitch gate rejected valid center")
            if gate.check(170.0, 0.0, 0.1):
                failures.append("pitch gate accepted implausible jump")

        predictor = GameStatePredictor()
        predictor.update_ball(10.0, 0.0, 0.0)
        predictor.update_ball(20.0, 0.0, 1.0)
        pred = predictor.predict(2.0)
        if pred is None or "ball_velocity" not in pred.reason or abs(shortest_lon_delta(pred.lon, 30.0)) > 1e-6:
            failures.append(f"game-state velocity prediction failed: {pred}")

        team = TeamClassifier()
        empty_assignments = team.update(np.zeros((32, 32, 3), dtype=np.uint8), [])
        if empty_assignments != []:
            failures.append(f"team classifier empty update failed: {empty_assignments}")

        candidate_model = ROOT / "models" / "ball_v5_yolo11s_1280_candidate.pt"
        if candidate_model.exists():
            import autopan_infer
            import argparse
            import evaluate
            import worker_server

            def score(path: Path) -> float:
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    return float(data.get("map50_95", data.get("mAP50-95", data.get("map", -1.0))))
                except Exception:
                    return -1.0

            candidate_score = score(ROOT / "results" / "ball_v5_yolo11s_1280_candidate_eval.json")
            final_score = score(ROOT / "results" / "ball_v5_eval.json")
            if candidate_score > final_score:
                expected = "models/ball_v5_yolo11s_1280_candidate.pt"
                if autopan_infer.preferred_ball_model() != expected:
                    failures.append(f"autopan auto model={autopan_infer.preferred_ball_model()} expected {expected}")
                if worker_server.preferred_ball_model() != expected:
                    failures.append(f"worker auto model={worker_server.preferred_ball_model()} expected {expected}")
                if evaluate.preferred_track_v2_ball_model() != expected:
                    failures.append(f"evaluate track_v2 model={evaluate.preferred_track_v2_ball_model()} expected {expected}")
                cmd = evaluate.build_cmd(
                    "track_v2",
                    {"insv": "clip.insv", "calib": "calib.json"},
                    "out.csv",
                    "out.mp4",
                    [],
                    argparse.Namespace(
                        ball=evaluate.BALL,
                        players=evaluate.PLAYERS,
                        scan_every=0,
                        ball_sahi=False,
                        field_opt=False,
                        segments=1,
                        seg_duration=1.0,
                        seed=1,
                        device="cpu",
                        player_detect_every=3,
                        ball_detect_every=2,
                        ball_sahi_every=6,
                    ),
                )
                if expected not in cmd:
                    failures.append(f"evaluate track_v2 command does not use {expected}: {cmd}")

        return Check(
            "automated",
            "synthetic v5 component smoke test",
            "fail" if failures else "pass",
            "; ".join(failures) if failures else "tracker/scanner/pitch/game-state/team components passed",
        )
    except Exception as exc:
        return Check("automated", "synthetic v5 component smoke test", "fail", str(exc))


def evaluation_summary_smoke_check() -> Check:
    """Exercise evaluate_csv segment aggregation and angle-wrap behavior."""
    try:
        from evaluate import evaluate_csv

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            csv_path = tmpdir / "pan.csv"
            insprj_path = tmpdir / "gt.insprj"

            rows = ["timestamp_s,predicted_pan_deg"]
            rows.extend(f"{t},10" for t in range(0, 60))
            rows.extend(f"{t},20" for t in range(200, 320))
            csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            insprj_path.write_text(
                """
<project>
  <recording>
    <keyframes>
      <keyframe time="0" pan="0" />
      <keyframe time="159000" pan="0" />
      <keyframe time="319000" pan="0" />
    </keyframes>
  </recording>
</project>
""".strip(),
                encoding="utf-8",
            )
            segments, overall_rmse, overall_mae = evaluate_csv(str(csv_path), str(insprj_path), seg_dur=100.0)

            wrap_csv = tmpdir / "pan_wrap.csv"
            wrap_gt = tmpdir / "gt_wrap.insprj"
            wrap_csv.write_text(
                "timestamp_s,predicted_pan_deg\n" + "\n".join(f"{t},179" for t in range(0, 60)) + "\n",
                encoding="utf-8",
            )
            wrap_gt.write_text(
                """
<project>
  <recording>
    <keyframes>
      <keyframe time="0" pan="-3.1241393611" />
      <keyframe time="59000" pan="-3.1241393611" />
      <keyframe time="100000" pan="-3.1241393611" />
    </keyframes>
  </recording>
</project>
""".strip(),
                encoding="utf-8",
            )
            _, wrap_rmse, wrap_mae = evaluate_csv(str(wrap_csv), str(wrap_gt), seg_dur=100.0)

            gt_wrap_csv = tmpdir / "pan_gt_wrap.csv"
            gt_wrap = tmpdir / "gt_cross_wrap.insprj"
            gt_wrap_csv.write_text(
                "timestamp_s,predicted_pan_deg\n" + "\n".join(f"{t},180" for t in range(0, 60)) + "\n",
                encoding="utf-8",
            )
            gt_wrap.write_text(
                """
<project>
  <recording>
    <keyframes>
      <keyframe time="0" pan="3.1241393611" />
      <keyframe time="59000" pan="-3.1241393611" />
      <keyframe time="100000" pan="-3.1241393611" />
    </keyframes>
  </recording>
</project>
""".strip(),
                encoding="utf-8",
            )
            _, gt_wrap_rmse, gt_wrap_mae = evaluate_csv(str(gt_wrap_csv), str(gt_wrap), seg_dur=100.0)

        failures = []
        expected_rmse = math.sqrt((60 * 10.0**2 + 100 * 20.0**2) / 160.0)
        expected_mae = (60 * 10.0 + 100 * 20.0) / 160.0
        if len(segments) != 2:
            failures.append(f"segments={len(segments)} expected 2")
        if abs(overall_rmse - expected_rmse) > 1e-6:
            failures.append(f"overall_rmse={overall_rmse} expected {expected_rmse}")
        if abs(overall_mae - expected_mae) > 1e-6:
            failures.append(f"overall_mae={overall_mae} expected {expected_mae}")
        if abs(wrap_rmse - 2.0) > 1e-3 or abs(wrap_mae - 2.0) > 1e-3:
            failures.append(f"angle wrap rmse/mae={(wrap_rmse, wrap_mae)} expected 2")
        if gt_wrap_rmse > 1.1 or gt_wrap_mae > 1.1:
            failures.append(f"GT unwrap rmse/mae={(gt_wrap_rmse, gt_wrap_mae)} expected <=1.1")
        return Check(
            "automated",
            "synthetic evaluation summary smoke test",
            "fail" if failures else "pass",
            "; ".join(failures) if failures else f"weighted_rmse={overall_rmse:.3f}, wrap_rmse={wrap_rmse:.3f}, gt_wrap_rmse={gt_wrap_rmse:.3f}",
        )
    except Exception as exc:
        return Check("automated", "synthetic evaluation summary smoke test", "fail", str(exc))


def summary_rmse_gate(path: Path, threshold: float) -> Check:
    if not path.exists():
        return Check("verification", "Mean RMSE < 15 degrees for track_v2", "blocked_external", f"{rel(path)} missing")
    data = json.loads(path.read_text(encoding="utf-8"))
    vals = [
        float(v["overall_rmse"])
        for v in data.get("clips", {}).values()
        if "overall_rmse" in v and not math.isnan(float(v["overall_rmse"]))
    ]
    if not vals:
        return Check("verification", "Mean RMSE < 15 degrees for track_v2", "fail", f"{rel(path)} has no RMSE values")
    mean = sum(vals) / len(vals)
    return Check(
        "verification",
        "Mean RMSE < 15 degrees for track_v2",
        "pass" if mean < threshold else "fail",
        f"{rel(path)} mean RMSE={mean:.2f} deg, threshold={threshold:.2f}",
    )


def ball_metric_gate(path: Path, key: str, threshold: float, label: str) -> Check:
    if not path.exists():
        return Check("verification", label, "blocked_external", f"{rel(path)} missing")
    data = json.loads(path.read_text(encoding="utf-8"))
    vals = []
    for clip in data.get("clips", {}).values():
        metric = clip.get("ball_metrics", {})
        if key in metric and metric[key] is not None:
            try:
                vals.append(float(metric[key]))
            except (TypeError, ValueError):
                pass
    if not vals:
        return Check("verification", label, "blocked_external", f"{rel(path)} has no {key} values")
    value = sum(vals) / len(vals)
    return Check(
        "verification",
        label,
        "pass" if value >= threshold else "fail",
        f"{rel(path)} mean {key}={value:.3f}, threshold={threshold:.3f}",
    )


def scanner_gate(path: Path, threshold: float) -> Check:
    if not path.exists():
        return Check("verification", "Full-pitch scanner detects ball in >80% on-pitch frames", "blocked_external", f"{rel(path)} missing")
    data = json.loads(path.read_text(encoding="utf-8"))
    value = data.get("on_pitch_detection_rate")
    if value is None:
        proxy = data.get("detection_rate")
        if proxy is not None:
            return Check(
                "verification",
                "Full-pitch scanner detects ball in >80% on-pitch frames",
                "blocked_external",
                f"{rel(path)} only has proxy detection_rate={float(proxy):.3f}; provide ball GT to compute on_pitch_detection_rate",
            )
        return Check("verification", "Full-pitch scanner detects ball in >80% on-pitch frames", "fail", f"{rel(path)} missing on_pitch_detection_rate")
    value = float(value)
    return Check(
        "verification",
        "Full-pitch scanner detects ball in >80% on-pitch frames",
        "pass" if value >= threshold else "fail",
        f"{rel(path)} on_pitch_detection_rate={value:.3f}, threshold={threshold:.3f}",
    )


def detector_gate(
    path: Path,
    map50: float,
    map5095: float,
    weights_path: Path | None = None,
    finalize_status_path: Path | None = None,
) -> Check:
    if finalize_status_path is not None and finalize_status_path.exists():
        status = json.loads(finalize_status_path.read_text(encoding="utf-8"))
        status_name = status.get("status")
        promoted = bool(status.get("promoted"))
        if status_name != "passed" or not promoted:
            got_map50 = float(status.get("map50", float("nan")))
            got_map = float(status.get("map50_95", float("nan")))
            return Check(
                "verification",
                "ball_v5 detector mAP targets",
                "fail",
                f"{rel(finalize_status_path)} status={status_name} promoted={promoted} "
                f"map50={got_map50:.3f}/{map50:.3f}, map50_95={got_map:.3f}/{map5095:.3f}",
            )
    if not path.exists():
        return Check("verification", "ball_v5 detector mAP targets", "blocked_external", f"{rel(path)} missing")
    data = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = data.get("weights_fingerprint")
    if weights_path is not None and isinstance(fingerprint, dict):
        if not weights_path.exists():
            return Check(
                "verification",
                "ball_v5 detector mAP targets",
                "blocked_external",
                f"{rel(path)} was generated with fingerprint metadata, but {rel(weights_path)} is missing",
            )
        expected_sha = fingerprint.get("sha256")
        if expected_sha:
            actual_sha = sha256_file(weights_path)
            if actual_sha != expected_sha:
                return Check(
                    "verification",
                    "ball_v5 detector mAP targets",
                    "blocked_external",
                    f"{rel(path)} does not match {rel(weights_path)} sha256={actual_sha[:12]} expected={expected_sha[:12]}; rerun detector eval",
                )
        elif fingerprint.get("size_bytes") is not None and weights_path.stat().st_size != int(fingerprint["size_bytes"]):
            return Check(
                "verification",
                "ball_v5 detector mAP targets",
                "blocked_external",
                f"{rel(path)} was generated for weights size={fingerprint['size_bytes']}, current {rel(weights_path)} size={weights_path.stat().st_size}",
            )
    got_map50 = float(data.get("map50", data.get("mAP50", float("nan"))))
    got_map = float(data.get("map50_95", data.get("mAP50-95", data.get("map", float("nan")))))
    passed = got_map50 >= map50 and got_map >= map5095
    split = data.get("split", "unknown")
    weights = data.get("weights", "unknown")
    return Check(
        "verification",
        "ball_v5 detector mAP targets",
        "pass" if passed else "fail",
        f"{rel(path)} split={split} weights={weights} map50={got_map50:.3f}/{map50:.3f}, map50_95={got_map:.3f}/{map5095:.3f}",
    )


def static_checks() -> list[Check]:
    checks = [
        check_markers(
            "autopan_infer.py",
            [
                "_pick_encoder",
                "PLAYER_DETECT_EVERY = 3",
                "BALL_DETECT_EVERY   = 2",
                "detect_ball_enhanced",
                "MultiHypothesisEquirectTracker",
                "PitchAwareBallGate",
                "GameStatePredictor",
                "--device",
                "--scan-every",
                "--ball-sahi",
                "--field-opt",
                "--no-game-state",
                "preferred_ball_model",
            ],
            "phase0_phase5",
            "main autopan pipeline integrates cross-platform encoder, device selection, v5 modules, equirect tracker, pitch gate, and game-state fallback",
        ),
        check_markers(
            "extract_frames.py",
            ["--clips-file", "--clip-root", "--use-insprj", "--groundtruth-window", "--samples-per-keyframe"],
            "phase0_phase1",
            "frame extraction supports custom clips and .insprj-guided sampling",
        ),
        check_markers(
            "train_ball_v5.py",
            ["--make-split", "--zip", "yolo11s.pt", "--imgsz", "--sweep", "--tracking", "copy_paste", "mosaic", "scale"],
            "phase1_detector",
            "ball_v5 training script supports splits/ZIP datasets, YOLO11s/1280, augmentations, sweeps, and tracking",
        ),
        check_markers(
            "equirect_ball_scanner.py",
            ["scan_equirect_frame", "parse_scan_yaws", "merge_scan_detections", "MultiHypothesisEquirectTracker", "--no-sliced", "--summary-json", "--ball-groundtruth"],
            "phase2_scanner",
            "full-pitch equirectangular scanner exists with multi-yaw scanning and SAHI option",
        ),
        check_markers(
            "ball_tracker_equirect.py",
            ["class EquirectIMMTracker", "class MultiHypothesisEquirectTracker", "constant-acceleration", "max_hypotheses"],
            "phase3_tracking",
            "equirectangular IMM tracker with multi-hypothesis tracking exists",
        ),
        check_markers(
            "pitch_model.py",
            ["class PitchHomographyModel", "class PitchAwareBallGate", "max_speed_kmh", "lon_lat_to_pitch"],
            "phase4_pitch",
            "homography pitch model and physical plausibility gate exist",
        ),
        check_markers(
            "game_state.py",
            ["class GameStatePredictor", "ball_velocity", "formation_shift", "corner_area", "goal_area", "throw_in_area"],
            "phase4_pitch",
            "game-state fallback uses ball velocity, formation shift, and pitch-area events",
        ),
        check_markers(
            "team_classifier.py",
            ["class TeamClassifier", "TeamClusterer", "TrackManager", "stable_label"],
            "phase4_team",
            "team classifier wraps HSV/LAB clustering and IoU tracks for stable labels",
        ),
        check_markers(
            "evaluate.py",
            ["DEFAULT_CLIPS", "--clips-file", "--clip-root", "normalize_clip", "track_v2", "evaluate_ball_metrics", "switching_latency_frames_median"],
            "phase5_eval",
            "evaluation supports expanded clip registry, track_v2 preset, and ball tracking metrics",
        ),
        check_markers(
            "worker_server.py",
            ["POST /autopan", "GET  /autopan/{job_id}", "WS   /autopan/{job_id}/events", "subprocess.Popen", "output_url", "preferred_ball_model"],
            "phase5_worker",
            "worker server can start autopan background jobs and report status/progress",
        ),
        check_markers(
            "web/app/calibrate/page.tsx",
            ["startAutopan", "watchAutopanJob", "WebSocket", "Autopan starten", "output_url"],
            "phase5_web",
            "calibrate page can trigger autopan jobs and display live status/preview",
        ),
    ]
    for file_path in [
        "ball_tracker_equirect.py",
        "equirect_ball_scanner.py",
        "pitch_model.py",
        "team_classifier.py",
        "train_ball_v5.py",
        "game_state.py",
    ]:
        checks.append(check_file(file_path, "new_artifacts", f"{file_path} exists"))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--allow-blocked", action="store_true", help="Return 0 when only external gates are blocked")
    parser.add_argument("--run-python-compile", action="store_true")
    parser.add_argument("--run-cli-smoke", action="store_true")
    parser.add_argument("--run-metric-smoke", action="store_true")
    parser.add_argument("--run-component-smoke", action="store_true")
    parser.add_argument("--run-eval-smoke", action="store_true")
    parser.add_argument("--run-web-lint", action="store_true")
    parser.add_argument("--run-web-build", action="store_true")
    parser.add_argument("--clips-file", default=None)
    parser.add_argument("--ball-groundtruth-dir", default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--dataset-zip", default="../ball_detector_merged.v3i.yolov8.zip")
    parser.add_argument("--ball-v5", default="models/ball_v5.pt")
    parser.add_argument("--ball-data", default="data/ball_v5/data.yaml")
    args = parser.parse_args()

    checks = static_checks()
    checks.extend(external_asset_checks(args))
    checks.extend(metric_gate_checks(args))

    if args.run_python_compile:
        checks.append(run_cmd([sys.executable, "-m", "py_compile", *default_compile_files()], ROOT))
    if args.run_cli_smoke:
        for script in (
            "autopan_infer.py",
            "equirect_ball_scanner.py",
            "evaluate.py",
            "extract_frames.py",
            "train_ball_v5.py",
            "sync_ball_v5_artifacts.py",
            "audit_ball_dataset.py",
        ):
            checks.append(run_cmd([sys.executable, script, "--help"], ROOT, timeout_s=60))
    if args.run_metric_smoke:
        checks.append(ball_metric_smoke_check())
    if args.run_component_smoke:
        checks.append(component_smoke_check())
    if args.run_eval_smoke:
        checks.append(evaluation_summary_smoke_check())
    if args.run_web_lint:
        checks.append(run_cmd(["npm", "run", "lint"], WEB, timeout_s=180))
    if args.run_web_build:
        import os

        env = dict(os.environ)
        env.setdefault("NEXT_PUBLIC_SUPABASE_URL", "https://example.supabase.co")
        env.setdefault("NEXT_PUBLIC_SUPABASE_ANON_KEY", "dummy-key")
        checks.append(run_cmd(["npm", "run", "build"], WEB, timeout_s=240, env=env))

    counts = {status: sum(1 for check in checks if check.status == status) for status in ("pass", "fail", "blocked_external", "not_run")}
    payload = {"counts": counts, "checks": [asdict(check) for check in checks]}

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("Implementation plan verification")
        print(json.dumps(counts, indent=2))
        for check in checks:
            print(f"[{check.status}] {check.phase}: {check.requirement}")
            print(f"  {check.evidence}")

    if counts["fail"] > 0:
        return 1
    if counts["blocked_external"] > 0:
        return 0 if args.allow_blocked else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
