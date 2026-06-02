#!/usr/bin/env python3
"""Check or fetch ball_v5 training artifacts from the GPU VM.

Default remote workspace:
  root@69.30.85.25:22187:/root/insta360_ball_v5_continue_20260602

Examples:
  python sync_ball_v5_artifacts.py --status
  python sync_ball_v5_artifacts.py --fetch
  python sync_ball_v5_artifacts.py --fetch-live

By default, the VM data.yaml is fetched to data/ball_v5/data.remote.yaml so it
does not overwrite the local ZIP-prepared dataset metadata.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_HOST = "root@69.30.85.25"
DEFAULT_PORT = 22187
DEFAULT_REMOTE_WORKSPACE = "/root/insta360_ball_v5_continue_20260602"


@dataclass
class RemoteFile:
    name: str
    remote: str
    local: Path
    exists: bool = False
    size: int | None = None


def ssh_base(args) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-i",
        str(Path(args.ssh_key).expanduser()),
        "-p",
        str(args.port),
        args.host,
    ]


def scp_base(args) -> list[str]:
    return [
        "scp",
        "-P",
        str(args.port),
        "-i",
        str(Path(args.ssh_key).expanduser()),
    ]


def run(cmd: list[str], timeout_s: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)


def local_file_info(path: Path, include_mtime: bool = True) -> dict:
    if not path.exists():
        return {"exists": False}
    info = {
        "exists": True,
        "size": path.stat().st_size,
    }
    if include_mtime:
        info["mtime"] = path.stat().st_mtime
    return info


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


def local_candidate_entry(name: str, model: str, eval_json: str | None = None,
                          status_json: str | None = None, results_csv: str | None = None,
                          domain_eval_json: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": name,
        "model": {
            "path": model,
            **local_file_info(ROOT / model, include_mtime=False),
        },
    }
    if eval_json:
        entry["eval_json"] = {
            "path": eval_json,
            **local_file_info(ROOT / eval_json, include_mtime=False),
            "metrics": read_json_if_exists(ROOT / eval_json),
        }
    if status_json:
        entry["status_json"] = {
            "path": status_json,
            **local_file_info(ROOT / status_json, include_mtime=False),
            "status": read_json_if_exists(ROOT / status_json),
        }
    if domain_eval_json:
        entry["domain_eval_json"] = {
            "path": domain_eval_json,
            **local_file_info(ROOT / domain_eval_json, include_mtime=False),
            "metrics": read_json_if_exists(ROOT / domain_eval_json),
        }
    if results_csv:
        entry["results_csv"] = {
            "path": results_csv,
            **local_file_info(ROOT / results_csv, include_mtime=False),
        }
    return entry


def write_candidate_manifest(live: dict | None = None) -> None:
    entries = [
        local_candidate_entry(
            "formal_final",
            "models/ball_v5.pt",
            "results/ball_v5_eval.json",
            "results/ball_v5_finalize_status.json",
            domain_eval_json="results/ball_v5_domain_eval.json",
        ),
        local_candidate_entry(
            "stable_yolo11s_1280",
            "models/ball_v5_stable.pt",
            "results/ball_v5_stable_eval.json",
            "results/ball_v5_finalize_status.json",
            domain_eval_json="results/ball_v5_stable_domain_eval.json",
        ),
        local_candidate_entry(
            "yolo11s_1280_candidate",
            "models/ball_v5_yolo11s_1280_candidate.pt",
            "results/ball_v5_yolo11s_1280_candidate_eval.json",
            "results/ball_v5_yolo11s_1280_candidate_status.json",
            "results/ball_v5_yolo11s_1280_candidate_results.csv",
        ),
        local_candidate_entry("live_best_cache", "models/ball_v5_live_best.pt", "results/ball_v5_live_eval.json", None, "results/ball_v5_live_results.csv"),
        local_candidate_entry(
            "yolo11m_1280_stopped_best",
            "models/ball_v5_yolo11m_1280_stopped_best.pt",
            results_csv="results/ball_v5_yolo11m_1280_stopped_results.csv",
        ),
        local_candidate_entry(
            "yolo11m_1280_continue_best",
            "models/ball_v5_yolo11m_1280_continue_best.pt",
            "results/ball_v5_yolo11m_1280_continue_eval.json",
            "results/ball_v5_yolo11m_1280_continue_status.json",
            domain_eval_json="results/ball_v5_yolo11m_1280_continue_domain_eval.json",
        ),
    ]
    payload: dict[str, Any] = {
        "candidates": entries,
        "live_run": live.get("run") if live else read_json_if_exists(ROOT / "results" / "ball_v5_live_manifest.json"),
    }
    write_json(ROOT / "results" / "ball_v5_candidates.json", payload)


def remote_files(args) -> list[RemoteFile]:
    ws = args.remote_workspace.rstrip("/")
    data_yaml_target = (
        ROOT / "data" / "ball_v5" / "data.yaml"
        if args.overwrite_local_data_yaml
        else ROOT / "data" / "ball_v5" / "data.remote.yaml"
    )
    return [
        RemoteFile("weights", f"{ws}/models/ball_v5.pt", ROOT / "models" / "ball_v5.pt"),
        RemoteFile("stable_weights", f"{ws}/models/ball_v5_stable.pt", ROOT / "models" / "ball_v5_stable.pt"),
        RemoteFile("data_yaml", f"{ws}/data/ball_v5/data.yaml", data_yaml_target),
        RemoteFile("eval_json", f"{ws}/results/ball_v5_eval.json", ROOT / "results" / "ball_v5_eval.json"),
        RemoteFile("stable_eval_json", f"{ws}/results/ball_v5_stable_eval.json", ROOT / "results" / "ball_v5_stable_eval.json"),
        RemoteFile("domain_eval_json", f"{ws}/results/ball_v5_domain_eval.json", ROOT / "results" / "ball_v5_domain_eval.json"),
        RemoteFile("stable_domain_eval_json", f"{ws}/results/ball_v5_stable_domain_eval.json", ROOT / "results" / "ball_v5_stable_domain_eval.json"),
        RemoteFile("finalize_status", f"{ws}/results/ball_v5_finalize_status.json", ROOT / "results" / "ball_v5_finalize_status.json"),
        RemoteFile("yolo11m_continue_weights", f"{ws}/models/ball_v5_yolo11m_1280_continue_best.pt", ROOT / "models" / "ball_v5_yolo11m_1280_continue_best.pt"),
        RemoteFile("yolo11m_continue_eval_json", f"{ws}/results/ball_v5_yolo11m_1280_continue_eval.json", ROOT / "results" / "ball_v5_yolo11m_1280_continue_eval.json"),
        RemoteFile("yolo11m_continue_domain_eval_json", f"{ws}/results/ball_v5_yolo11m_1280_continue_domain_eval.json", ROOT / "results" / "ball_v5_yolo11m_1280_continue_domain_eval.json"),
        RemoteFile("yolo11m_continue_status", f"{ws}/results/ball_v5_yolo11m_1280_continue_status.json", ROOT / "results" / "ball_v5_yolo11m_1280_continue_status.json"),
    ]


def inspect_remote(args, files: list[RemoteFile]) -> tuple[dict, str, dict]:
    remote_paths_json = json.dumps([f.remote for f in files])
    workspace_json = json.dumps(args.remote_workspace.rstrip("/"))
    script = f"""python3 - <<'PY'
import json
import subprocess
import csv
from pathlib import Path

def proc_state(pid):
    if not pid:
        return "unknown"
    result = subprocess.run(["ps", "-p", pid, "-o", "stat="], capture_output=True, text=True)
    stat = result.stdout.strip().split()
    if result.returncode != 0 or not stat:
        return "exited"
    return "exited" if stat[0].startswith("Z") else "running"

paths = json.loads({remote_paths_json!r})
workspace = Path(json.loads({workspace_json!r}))
for value in paths:
    path = Path(value)
    if path.is_file():
        print(f"{{value}}\\t{{path.stat().st_size}}")
    else:
        print(f"{{value}}\\tMISSING")

pid_patterns = ("train_*.pid", "continue_*.pid")
pidfiles = sorted(
    [p for pattern in pid_patterns for p in (workspace / "logs").glob(pattern)],
    key=lambda p: p.stat().st_mtime,
)
pidfile = None
if pidfiles:
    pidfile = pidfiles[-1]
    pid = pidfile.read_text().strip()
    state = proc_state(pid)
    print(f"__PID__\\t{{pid}}\\t{{state}}\\t{{pidfile}}")

logfile = pidfile.with_suffix(".log") if pidfile is not None else None
if logfile is None or not logfile.exists():
    log_patterns = ("train_*.log", "continue_*.log")
    logfiles = sorted(
        [p for pattern in log_patterns for p in (workspace / "logs").glob(pattern)],
        key=lambda p: p.stat().st_mtime,
    )
    logfile = logfiles[-1] if logfiles else None
if logfile is not None and logfile.exists():
    print(f"__LOG__\\t{{logfile}}\\t{{logfile.stat().st_size}}")
    try:
        text = logfile.read_text(encoding="utf-8", errors="replace")
        warnings = [
            line.strip()
            for line in text.replace("\\r", "\\n").splitlines()
            if "NaN/Inf" in line or "Skipping checkpoint save" in line
        ]
        if warnings:
            print("__LOG_WARNING__\\t" + json.dumps({{
                "count": len(warnings),
                "last": warnings[-1][-500:],
            }}))
    except Exception as exc:
        print("__LOG_ERROR__\\t" + str(exc))

finalizer_pidfiles = sorted((workspace / "logs").glob("finalize_*.pid"), key=lambda p: p.stat().st_mtime)
finalizer_pidfile = finalizer_pidfiles[-1] if finalizer_pidfiles else None
if finalizer_pidfile is not None:
    finalizer_pid = finalizer_pidfile.read_text().strip()
    state = proc_state(finalizer_pid)
    print(f"__FINALIZER__\\t{{finalizer_pid}}\\t{{state}}\\t{{finalizer_pidfile}}")

finalizer_logs = sorted((workspace / "logs").glob("finalize_*.log"), key=lambda p: p.stat().st_mtime)
finalizer_log = finalizer_logs[-1] if finalizer_logs else None
if finalizer_log is not None and finalizer_log.exists():
    print(f"__FINALIZER_LOG__\\t{{finalizer_log}}\\t{{finalizer_log.stat().st_size}}")

finalize_status = workspace / "results" / "ball_v5_finalize_status.json"
if finalize_status.is_file():
    try:
        print("__FINALIZE_STATUS__\\t" + json.dumps(json.loads(finalize_status.read_text(encoding="utf-8"))))
    except Exception as exc:
        print("__FINALIZE_STATUS_ERROR__\\t" + str(exc))

followup_pidfiles = sorted((workspace / "logs").glob("followup_*.pid"), key=lambda p: p.stat().st_mtime)
followup_pidfile = followup_pidfiles[-1] if followup_pidfiles else None
if followup_pidfile is not None:
    followup_pid = followup_pidfile.read_text().strip()
    state = proc_state(followup_pid)
    print(f"__FOLLOWUP__\\t{{followup_pid}}\\t{{state}}\\t{{followup_pidfile}}")

followup_logs = sorted((workspace / "logs").glob("followup_*.log"), key=lambda p: p.stat().st_mtime)
followup_log = followup_logs[-1] if followup_logs else None
if followup_log is not None and followup_log.exists():
    print(f"__FOLLOWUP_LOG__\\t{{followup_log}}\\t{{followup_log.stat().st_size}}")

highres_pidfiles = sorted((workspace / "logs").glob("highres_*.pid"), key=lambda p: p.stat().st_mtime)
highres_pidfile = highres_pidfiles[-1] if highres_pidfiles else None
if highres_pidfile is not None:
    highres_pid = highres_pidfile.read_text().strip()
    state = proc_state(highres_pid)
    print(f"__HIGHRES__\\t{{highres_pid}}\\t{{state}}\\t{{highres_pidfile}}")

highres_logs = sorted((workspace / "logs").glob("highres_*.log"), key=lambda p: p.stat().st_mtime)
highres_log = highres_logs[-1] if highres_logs else None
if highres_log is not None and highres_log.exists():
    print(f"__HIGHRES_LOG__\\t{{highres_log}}\\t{{highres_log.stat().st_size}}")

results_files = sorted(workspace.glob("runs/**/results.csv"), key=lambda p: p.stat().st_mtime)
if results_files:
    results_csv = results_files[-1]
    run_dir = results_csv.parent
    print(f"__LIVE_RUN__\\t{{run_dir}}\\t{{run_dir.parent.name}}\\t{{run_dir.name}}")
    for tag, path in (
        ("__LIVE_RESULTS__", results_csv),
        ("__LIVE_BEST__", run_dir / "weights" / "best.pt"),
        ("__LIVE_LAST__", run_dir / "weights" / "last.pt"),
    ):
        if path.is_file():
            print(f"{{tag}}\\t{{path}}\\t{{path.stat().st_size}}")
        else:
            print(f"{{tag}}\\t{{path}}\\tMISSING")
    try:
        with results_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            print("__LIVE_ROW__\\t" + json.dumps(rows[-1]))
    except Exception as exc:
        print("__LIVE_ROW_ERROR__\\t" + str(exc))
PY"""
    result = run([*ssh_base(args), script])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())

    pid_state = {}
    finalizer_state = {}
    live: dict = {"artifacts": {}}
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if not parts:
            continue
        if parts[0] == "__PID__":
            pid_state = {
                "pid": parts[1],
                "state": parts[2] if len(parts) > 2 else "unknown",
                "pidfile": parts[3] if len(parts) > 3 else None,
            }
            continue
        if parts[0] == "__FINALIZER__":
            finalizer_state = {
                "pid": parts[1],
                "state": parts[2] if len(parts) > 2 else "unknown",
                "pidfile": parts[3] if len(parts) > 3 else None,
            }
            continue
        if parts[0] == "__FINALIZER_LOG__" and len(parts) > 2:
            live["finalizer_log"] = {
                "remote": parts[1],
                "size": int(parts[2]),
            }
            continue
        if parts[0] == "__FINALIZE_STATUS__" and len(parts) > 1:
            try:
                live["finalize_status"] = json.loads(parts[1])
            except json.JSONDecodeError:
                live["finalize_status_raw"] = parts[1]
            continue
        if parts[0] == "__FINALIZE_STATUS_ERROR__" and len(parts) > 1:
            live["finalize_status_error"] = parts[1]
            continue
        if parts[0] == "__FOLLOWUP__":
            live["followup"] = {
                "pid": parts[1],
                "state": parts[2] if len(parts) > 2 else "unknown",
                "pidfile": parts[3] if len(parts) > 3 else None,
            }
            continue
        if parts[0] == "__FOLLOWUP_LOG__" and len(parts) > 2:
            live["followup_log"] = {
                "remote": parts[1],
                "size": int(parts[2]),
            }
            continue
        if parts[0] == "__HIGHRES__":
            live["highres"] = {
                "pid": parts[1],
                "state": parts[2] if len(parts) > 2 else "unknown",
                "pidfile": parts[3] if len(parts) > 3 else None,
            }
            continue
        if parts[0] == "__HIGHRES_LOG__" and len(parts) > 2:
            live["highres_log"] = {
                "remote": parts[1],
                "size": int(parts[2]),
            }
            continue
        if parts[0] == "__LIVE_RUN__" and len(parts) > 3:
            live["run"] = {
                "remote": parts[1],
                "project": parts[2],
                "name": parts[3],
            }
            continue
        if parts[0].startswith("__LIVE_"):
            tag = parts[0].strip("_").lower()
            if tag == "live_row" and len(parts) > 1:
                try:
                    live["last_metrics"] = json.loads(parts[1])
                except json.JSONDecodeError:
                    live["last_metrics_raw"] = parts[1]
            elif len(parts) > 2:
                live["artifacts"][tag] = {
                    "remote": parts[1],
                    "size": None if parts[2] == "MISSING" else int(parts[2]),
                    "exists": parts[2] != "MISSING",
                }
            continue
        if parts[0] == "__LOG__" and len(parts) > 2:
            live["log"] = {
                "remote": parts[1],
                "size": int(parts[2]),
            }
            continue
        if parts[0] == "__LOG_WARNING__" and len(parts) > 1:
            try:
                live["log_warning"] = json.loads(parts[1])
            except json.JSONDecodeError:
                live["log_warning_raw"] = parts[1]
            continue
        if parts[0] == "__LOG_ERROR__" and len(parts) > 1:
            live["log_error"] = parts[1]
            continue
        for f in files:
            if parts[0] == f.remote:
                if len(parts) > 1 and parts[1] != "MISSING":
                    f.exists = True
                    f.size = int(parts[1])
                break
    if finalizer_state:
        live["finalizer"] = finalizer_state
    return pid_state, result.stdout, live


def fetch(args, files: list[RemoteFile]) -> None:
    for f in files:
        if not f.exists:
            print(f"skip {f.name}: remote missing {f.remote}")
            continue
        f.local.parent.mkdir(parents=True, exist_ok=True)
        cmd = [*scp_base(args), f"{args.host}:{f.remote}", str(f.local)]
        result = run(cmd, timeout_s=600)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        print(f"fetched {f.name}: {f.local}")


def fetch_live(args, live: dict) -> None:
    """Fetch non-final artifacts from the currently active/latest run."""
    downloads: list[RemoteFile] = []
    artifacts = live.get("artifacts", {})
    live_results = artifacts.get("live_results", {})
    live_best = artifacts.get("live_best", {})
    live_last = artifacts.get("live_last", {})
    live_log = live.get("log", {})

    if live_results.get("exists"):
        downloads.append(RemoteFile(
            "live_results",
            live_results["remote"],
            ROOT / "results" / "ball_v5_live_results.csv",
            exists=True,
            size=live_results.get("size"),
        ))
    if live_best.get("exists") and not args.skip_live_best:
        downloads.append(RemoteFile(
            "live_best",
            live_best["remote"],
            ROOT / "models" / "ball_v5_live_best.pt",
            exists=True,
            size=live_best.get("size"),
        ))
    if args.include_live_last and live_last.get("exists"):
        downloads.append(RemoteFile(
            "live_last",
            live_last["remote"],
            ROOT / "models" / "ball_v5_live_last.pt",
            exists=True,
            size=live_last.get("size"),
        ))
    if live_log.get("remote"):
        downloads.append(RemoteFile(
            "live_log",
            live_log["remote"],
            ROOT / "logs" / "remote_ball_v5_train_live.log",
            exists=True,
            size=live_log.get("size"),
        ))
    for name, local_name in (
        ("finalizer_log", "remote_ball_v5_finalizer.log"),
        ("followup_log", "remote_ball_v5_followup.log"),
        ("highres_log", "remote_ball_v5_highres.log"),
    ):
        log_info = live.get(name, {})
        if log_info.get("remote"):
            downloads.append(RemoteFile(
                name,
                log_info["remote"],
                ROOT / "logs" / local_name,
                exists=True,
                size=log_info.get("size"),
            ))
    if live.get("finalize_status"):
        downloads.append(RemoteFile(
            "finalize_status",
            f"{args.remote_workspace.rstrip('/')}/results/ball_v5_finalize_status.json",
            ROOT / "results" / "ball_v5_finalize_status.json",
            exists=True,
        ))

    if not downloads:
        print("No live artifacts are available to fetch.")
        return
    fetch(args, downloads)
    write_json(ROOT / "results" / "ball_v5_live_manifest.json", {
        "run": live.get("run"),
        "last_metrics": live.get("last_metrics"),
        "artifacts": live.get("artifacts"),
        "logs": {
            "initial_train": live.get("log"),
            "finalizer": live.get("finalizer_log"),
            "followup": live.get("followup_log"),
            "highres": live.get("highres_log"),
        },
        "states": {
            "finalizer": live.get("finalizer"),
            "followup": live.get("followup"),
            "highres": live.get("highres"),
        },
    })
    print(f"wrote live manifest: {ROOT / 'results' / 'ball_v5_live_manifest.json'}")
    write_candidate_manifest(live)
    print(f"wrote candidate manifest: {ROOT / 'results' / 'ball_v5_candidates.json'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519_default")
    parser.add_argument("--remote-workspace", default=DEFAULT_REMOTE_WORKSPACE)
    parser.add_argument("--status", action="store_true", help="Print artifact status")
    parser.add_argument("--fetch", action="store_true", help="Fetch available artifacts")
    parser.add_argument("--fetch-live", action="store_true",
                        help="Fetch active run artifacts to non-final local filenames")
    parser.add_argument("--include-live-last", action="store_true",
                        help="With --fetch-live, also fetch the live last.pt checkpoint")
    parser.add_argument("--skip-live-best", action="store_true",
                        help="With --fetch-live, fetch metrics/logs but skip the large best.pt checkpoint")
    parser.add_argument("--overwrite-local-data-yaml", action="store_true",
                        help="Fetch VM data.yaml over data/ball_v5/data.yaml instead of data.remote.yaml")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    files = remote_files(args)
    pid_state, raw, live = inspect_remote(args, files)

    payload = {
        "remote_workspace": args.remote_workspace,
        "job": pid_state,
        "live_training": live,
        "files": [
            {
                **asdict(f),
                "local": str(f.local),
                "local_info": local_file_info(f.local),
            }
            for f in files
        ],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        if pid_state:
            print(f"job pid={pid_state.get('pid')} state={pid_state.get('state')}")
        for f in files:
            remote_state = f"{f.size} bytes" if f.exists else "missing"
            local = local_file_info(f.local)
            local_state = f"{local['size']} bytes" if local.get("exists") else "missing"
            print(f"{f.name}: remote={remote_state}; local={local_state} -> {f.local}")
        if live.get("last_metrics"):
            row = live["last_metrics"]
            epoch = row.get("epoch") or row.get("                  epoch") or row.get("                  Epoch")
            map50 = row.get("metrics/mAP50(B)") or row.get("metrics/mAP50")
            map5095 = row.get("metrics/mAP50-95(B)") or row.get("metrics/mAP50-95")
            print(f"live metrics: epoch={epoch} mAP50={map50} mAP50-95={map5095}")
        if live.get("run"):
            run_info = live["run"]
            print(f"live run: {run_info.get('project')}/{run_info.get('name')} at {run_info.get('remote')}")
        if live.get("log"):
            log = live["log"]
            print(f"initial train log: {log['size']} bytes at {log['remote']}")
        if live.get("finalizer"):
            finalizer = live["finalizer"]
            print(f"finalizer pid={finalizer.get('pid')} state={finalizer.get('state')}")
        if live.get("finalizer_log"):
            log = live["finalizer_log"]
            print(f"finalizer log: {log['size']} bytes at {log['remote']}")
        if live.get("finalize_status"):
            status = live["finalize_status"]
            print(
                "finalize status: "
                f"{status.get('status')} map50={status.get('map50')} "
                f"mAP50-95={status.get('map50_95')} promoted={status.get('promoted')}"
            )
        if live.get("followup"):
            followup = live["followup"]
            print(f"followup pid={followup.get('pid')} state={followup.get('state')}")
        if live.get("followup_log"):
            log = live["followup_log"]
            print(f"followup log: {log['size']} bytes at {log['remote']}")
        if live.get("highres"):
            highres = live["highres"]
            print(f"highres pid={highres.get('pid')} state={highres.get('state')}")
        if live.get("highres_log"):
            log = live["highres_log"]
            print(f"highres log: {log['size']} bytes at {log['remote']}")
        if live.get("log_warning"):
            warning = live["log_warning"]
            print(f"live log warnings: {warning.get('count')} checkpoint/NaN warnings; last={warning.get('last')}")
        for name, info in live.get("artifacts", {}).items():
            state = f"{info['size']} bytes" if info.get("exists") else "missing"
            print(f"live {name}: {state} at {info.get('remote')}")

    if args.fetch:
        fetch(args, files)
        write_candidate_manifest(live)
        print(f"wrote candidate manifest: {ROOT / 'results' / 'ball_v5_candidates.json'}")
    if args.fetch_live:
        fetch_live(args, live)
    if not (args.status or args.json or args.fetch or args.fetch_live):
        print("No action requested. Use --status or --fetch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
