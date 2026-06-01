#!/usr/bin/env python3
"""Train or evaluate the upgraded ball detector.

Examples:
    python train_ball_v5.py --data data/ball_v5/data.yaml --model yolo11s.pt
    python train_ball_v5.py --eval-only --data data/ball_v5/data.yaml --weights models/ball_v5.pt
    python train_ball_v5.py --make-split --source-dir exports/roboflow_pool --split-dir data/ball_v5
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import zipfile
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def has_yolo_split_dirs(dataset_dir: Path) -> bool:
    return (
        (dataset_dir / "train" / "images").is_dir()
        and (dataset_dir / "train" / "labels").is_dir()
        and (dataset_dir / "valid" / "images").is_dir()
        and (dataset_dir / "valid" / "labels").is_dir()
    )


def backup_metadata_only_dir(dataset_dir: Path) -> bool:
    """Move stale metadata out of a dataset dir before ZIP extraction."""
    entries = list(dataset_dir.iterdir()) if dataset_dir.exists() else []
    if not entries:
        return True
    metadata_names = {"data.yaml", "data.yml", ".extracted_from"}
    if any(p.is_dir() or p.name not in metadata_names for p in entries):
        return False

    backup_dir = dataset_dir.parent / f"{dataset_dir.name}_metadata_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        target = backup_dir / entry.name
        if target.exists():
            target = backup_dir / f"{entry.stem}_{entry.stat().st_mtime_ns}{entry.suffix}"
        shutil.move(str(entry), str(target))
    print(f"Moved stale dataset metadata to {backup_dir}")
    return True


def best_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def find_image_label_pairs(source_dir: Path) -> List[Tuple[Path, Path]]:
    images = [p for p in source_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    pairs: List[Tuple[Path, Path]] = []
    for img in images:
        candidates = [
            img.with_suffix(".txt"),
            source_dir / "labels" / f"{img.stem}.txt",
            source_dir / "labels" / img.relative_to(source_dir).with_suffix(".txt"),
        ]
        label = next((p for p in candidates if p.exists()), None)
        if label is not None:
            pairs.append((img, label))
    return pairs


def copy_pair(img: Path, label: Path, split_dir: Path, split: str) -> None:
    img_dst = split_dir / "images" / split / img.name
    label_dst = split_dir / "labels" / split / f"{img.stem}.txt"
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, img_dst)
    shutil.copy2(label, label_dst)


def make_split(source_dir: Path, split_dir: Path, val_frac: float, test_frac: float, seed: int) -> Path:
    pairs = find_image_label_pairs(source_dir)
    if not pairs:
        raise ValueError(f"No image/label pairs found under {source_dir}")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    buckets = {
        "test": pairs[:n_test],
        "val": pairs[n_test : n_test + n_val],
        "train": pairs[n_test + n_val :],
    }

    for split, split_pairs in buckets.items():
        for img, label in split_pairs:
            copy_pair(img, label, split_dir, split)

    yaml_path = split_dir / "data.yaml"
    yaml_path.write_text(
        "\n".join([
            f"path: {split_dir.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: ball",
            "",
        ]),
        encoding="utf-8",
    )

    print(f"Split written to {split_dir}")
    for split, split_pairs in buckets.items():
        print(f"  {split}: {len(split_pairs)}")
    return yaml_path


def merge_yolo_label_classes(labels_dir: Path, target_class: int = 0) -> int:
    """Rewrite YOLO labels to one detection class and bbox rows.

    Roboflow exports can contain a mix of detection rows
    (class cx cy w h) and segmentation rows (class x1 y1 x2 y2 ...).
    Ultralytics will ignore mixed segments at train time, but normalizing here
    keeps caches and warnings deterministic.
    """
    changed = 0
    for label_path in labels_dir.rglob("*.txt"):
        lines = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
        out_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            new_parts = [str(target_class)]
            if parts[0] != str(target_class):
                changed += 1
            if len(parts) > 5 and (len(parts) - 1) % 2 == 0:
                try:
                    coords = [float(v) for v in parts[1:]]
                except ValueError:
                    continue
                xs = coords[0::2]
                ys = coords[1::2]
                x1 = max(0.0, min(1.0, min(xs)))
                x2 = max(0.0, min(1.0, max(xs)))
                y1 = max(0.0, min(1.0, min(ys)))
                y2 = max(0.0, min(1.0, max(ys)))
                if x2 <= x1 or y2 <= y1:
                    continue
                new_parts.extend([
                    f"{(x1 + x2) / 2.0:.6f}",
                    f"{(y1 + y2) / 2.0:.6f}",
                    f"{x2 - x1:.6f}",
                    f"{y2 - y1:.6f}",
                ])
                changed += 1
            else:
                new_parts.extend(parts[1:5])
            out_lines.append(" ".join(new_parts))
        label_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return changed


def remove_yolo_caches(dataset_dir: Path) -> int:
    removed = 0
    for cache_path in dataset_dir.rglob("*.cache"):
        cache_path.unlink()
        removed += 1
    return removed


def write_normalized_data_yaml(dataset_dir: Path, merge_classes: bool = True) -> Path:
    """Write a data.yaml with paths relative to dataset_dir.

    Roboflow exports commonly use '../train/images' even when data.yaml is
    extracted next to train/, valid/, and test/. This normalizes the paths and
    optionally collapses Ball/ball class variants to one class.
    """
    names_block = "names:\n  0: ball" if merge_classes else "names:\n  0: Ball\n  1: ball"
    nc = 1 if merge_classes else 2
    lines = [
        f"path: {dataset_dir.resolve().as_posix()}",
        "train: train/images",
        "val: valid/images",
    ]
    if (dataset_dir / "test" / "images").exists():
        lines.append("test: test/images")
    lines.extend([
        f"nc: {nc}",
        names_block,
        "",
    ])
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    return yaml_path


def parse_simple_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def resolve_split_dirs(data_yaml: Path, split: str) -> tuple[Path, Path]:
    values = parse_simple_yaml(data_yaml)
    yaml_key = "val" if split in {"val", "valid"} else split
    if yaml_key not in values:
        raise ValueError(f"{data_yaml} has no {yaml_key!r} split")
    base = Path(values.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = data_yaml.parent / base
    images_dir = Path(values[yaml_key])
    if not images_dir.is_absolute():
        images_dir = base / images_dir
    labels_dir = Path(str(images_dir).replace(f"{Path('images')}", f"{Path('labels')}"))
    if images_dir.name == "images":
        labels_dir = images_dir.parent / "labels"
    return images_dir, labels_dir


def classify_source_domain_from_name(image_path: Path) -> str:
    name = image_path.name.lower()
    if name.startswith(("frame_", "vid_")):
        return "insta360_style"
    if name.startswith(("fortuna-", "veo-", "video-from-veo", "video5")):
        return "veo_style"
    return "unknown"


def link_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def write_domain_subset_yaml(
    data_yaml: str | Path,
    split: str,
    domain: str,
    output_root: str | Path,
) -> tuple[Path, dict]:
    """Create a YOLO data.yaml containing only one source-domain subset."""
    data_yaml = Path(data_yaml)
    images_dir, labels_dir = resolve_split_dirs(data_yaml, split)
    out_dir = Path(output_root) / f"{split}_{domain}"
    subset_images = out_dir / split / "images"
    subset_labels = out_dir / split / "labels"
    image_paths = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    selected = [p for p in image_paths if classify_source_domain_from_name(p) == domain]
    if not selected:
        raise ValueError(f"No {domain} images found for split {split} in {images_dir}")
    missing_labels = 0
    linked_labels = 0
    for image_path in selected:
        link_or_copy_file(image_path, subset_images / image_path.name)
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            link_or_copy_file(label_path, subset_labels / label_path.name)
            linked_labels += 1
        else:
            missing_labels += 1
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        "\n".join([
            f"path: {out_dir.resolve().as_posix()}",
            f"train: {split}/images",
            f"val: {split}/images",
            f"test: {split}/images",
            "nc: 1",
            "names:",
            "  0: ball",
            "",
        ]),
        encoding="utf-8",
    )
    manifest = {
        "source_data": str(data_yaml),
        "source_images": str(images_dir),
        "source_labels": str(labels_dir),
        "split": split,
        "domain": domain,
        "images": len(selected),
        "labels": linked_labels,
        "missing_labels": missing_labels,
        "data_yaml": str(yaml_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return yaml_path, manifest


def prepare_zip_dataset(zip_path: Path, extract_dir: Path, merge_classes: bool = True) -> Path:
    """Extract a YOLO ZIP dataset and return a normalized data.yaml path."""
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".extracted_from"
    if (
        marker.exists()
        and marker.read_text(encoding="utf-8") == str(zip_path.resolve())
        and has_yolo_split_dirs(extract_dir)
    ):
        print(f"Dataset already extracted: {extract_dir}")
    else:
        if any(extract_dir.iterdir()) and not backup_metadata_only_dir(extract_dir):
            raise SystemExit(
                f"{extract_dir} is not empty and was not created by this ZIP. "
                "Choose a fresh --extract-dir."
            )
        print(f"Extracting {zip_path} -> {extract_dir}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        marker.write_text(str(zip_path.resolve()), encoding="utf-8")

    if merge_classes:
        changed = merge_yolo_label_classes(extract_dir / "train" / "labels")
        changed += merge_yolo_label_classes(extract_dir / "valid" / "labels")
        changed += merge_yolo_label_classes(extract_dir / "test" / "labels")
        removed_caches = remove_yolo_caches(extract_dir)
        print(f"Normalized labels to class 0 detection boxes in {changed} rows")
        if removed_caches:
            print(f"Removed {removed_caches} stale YOLO cache files")
    yaml_path = write_normalized_data_yaml(extract_dir, merge_classes=merge_classes)
    print(f"Normalized data.yaml written to {yaml_path}")
    return yaml_path


def parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def resolve_project(value: str) -> str:
    return str(Path(value).expanduser().resolve())


def tracking_run(args, name_suffix: str = ""):
    """Return a context manager for optional experiment tracking."""
    if not args.tracking or args.tracking == "tensorboard":
        if args.tracking == "tensorboard":
            print("TensorBoard logging is written by Ultralytics under --project/--name.")
        return nullcontext()

    run_name = f"{args.name}{name_suffix}"
    if args.tracking == "wandb":
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit("Install wandb or omit --tracking wandb") from exc

        class WandbRun:
            def __enter__(self):
                return wandb.init(
                    project=Path(args.project).name,
                    name=run_name,
                    config=vars(args),
                )

            def __exit__(self, exc_type, exc, tb):
                wandb.finish(exit_code=1 if exc_type else 0)
                return False

        return WandbRun()

    if args.tracking == "mlflow":
        try:
            import mlflow
        except ImportError as exc:
            raise SystemExit("Install mlflow or omit --tracking mlflow") from exc

        class MlflowRun:
            def __enter__(self):
                mlflow.set_experiment(Path(args.project).name)
                mlflow.start_run(run_name=run_name)
                mlflow.log_params({
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                    if isinstance(v, (str, int, float, bool, type(None)))
                })
                return mlflow

            def __exit__(self, exc_type, exc, tb):
                mlflow.end_run(status="FAILED" if exc_type else "FINISHED")
                return False

        return MlflowRun()

    return nullcontext()


def yolo_metrics_dict(metrics) -> dict:
    box = getattr(metrics, "box", None)
    speed = getattr(metrics, "speed", None)
    return {
        "map50": float(getattr(box, "map50", float("nan"))) if box is not None else float("nan"),
        "map50_95": float(getattr(box, "map", float("nan"))) if box is not None else float("nan"),
        "precision": float(getattr(box, "mp", float("nan"))) if box is not None else float("nan"),
        "recall": float(getattr(box, "mr", float("nan"))) if box is not None else float("nan"),
        "fitness": float(getattr(metrics, "fitness", float("nan"))),
        "speed": speed if isinstance(speed, dict) else {},
    }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_fingerprint(path: str | Path, *, hash_contents: bool = False) -> dict:
    p = Path(path)
    out = {
        "path": str(p),
        "exists": p.exists(),
    }
    if not p.exists():
        return out
    stat = p.stat()
    out.update({
        "size_bytes": stat.st_size,
        "mtime_epoch": stat.st_mtime,
    })
    if hash_contents:
        out["sha256"] = sha256_file(p)
    return out


def write_metrics_json(metrics, path: str | None, extra: dict | None = None) -> None:
    if not path:
        return
    out = yolo_metrics_dict(metrics)
    out["generated_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if extra:
        out.update(extra)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Metrics JSON written to {out_path}")


def default_domain_metrics_path(metrics_json: str | None) -> str | None:
    if not metrics_json:
        return None
    path = Path(metrics_json)
    return str(path.with_name(f"{path.stem}_domains{path.suffix or '.json'}"))


def write_domain_metrics_json(rows: list[dict], path: str | None, extra: dict | None = None) -> None:
    if not path:
        return
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "domains": rows,
    }
    if extra:
        payload.update(extra)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Domain metrics JSON written to {out_path}")


def evaluate_domain_subsets(args, model, project: str) -> list[dict]:
    domains = args.eval_domain or ["insta360_style", "veo_style"]
    rows: list[dict] = []
    subset_root = Path(args.domain_subsets_dir)
    for domain in domains:
        subset_yaml, manifest = write_domain_subset_yaml(args.data, args.split, domain, subset_root)
        with tracking_run(args, name_suffix=f"_{args.split}_{domain}"):
            metrics = model.val(
                data=str(subset_yaml),
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                split="val",
                conf=args.conf,
                iou=args.iou,
                project=project,
                name=f"{args.name}_{args.split}_{domain}",
                exist_ok=True,
            )
        row = yolo_metrics_dict(metrics)
        row.update({
            "source_split": args.split,
            "eval_split": "val",
            "domain": domain,
            "subset": manifest,
            "subset_data": str(subset_yaml),
        })
        rows.append(row)
        print(json.dumps(row, indent=2))
    return rows


def train(args) -> None:
    from ultralytics import YOLO

    model = YOLO(args.model)
    project = resolve_project(args.project)
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": project,
        "name": args.name,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "cos_lr": True,
        "close_mosaic": args.close_mosaic,
        "mosaic": args.mosaic,
        "copy_paste": args.copy_paste,
        "scale": args.scale,
        "degrees": args.degrees,
        "fliplr": args.fliplr,
        "cache": args.cache,
        "exist_ok": True,
        "amp": args.amp,
    }
    if args.workers is not None:
        train_kwargs["workers"] = args.workers
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.lrf is not None:
        train_kwargs["lrf"] = args.lrf
    if args.weight_decay is not None:
        train_kwargs["weight_decay"] = args.weight_decay
    if args.warmup_epochs is not None:
        train_kwargs["warmup_epochs"] = args.warmup_epochs
    with tracking_run(args):
        model.train(**train_kwargs)


def evaluate(args) -> None:
    from ultralytics import YOLO

    model = YOLO(args.weights)
    project = resolve_project(args.project)
    with tracking_run(args, name_suffix=f"_{args.split}"):
        metrics = model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            split=args.split,
            conf=args.conf,
            iou=args.iou,
            project=project,
            name=f"{args.name}_{args.split}",
            exist_ok=True,
        )
    write_metrics_json(metrics, args.metrics_json, extra={
        "split": args.split,
        "weights": args.weights,
        "data": args.data,
        "weights_fingerprint": artifact_fingerprint(args.weights, hash_contents=True),
        "data_fingerprint": artifact_fingerprint(args.data, hash_contents=True),
    })
    if args.eval_domains:
        domain_rows = evaluate_domain_subsets(args, model, project)
        write_domain_metrics_json(domain_rows, args.domain_metrics_json or default_domain_metrics_path(args.metrics_json), extra={
            "split": args.split,
            "weights": args.weights,
            "data": args.data,
            "weights_fingerprint": artifact_fingerprint(args.weights, hash_contents=True),
            "data_fingerprint": artifact_fingerprint(args.data, hash_contents=True),
        })
    print(metrics)


def sweep(args) -> None:
    from ultralytics import YOLO

    model = YOLO(args.weights)
    confs = parse_float_list(args.sweep_conf)
    ious = parse_float_list(args.sweep_iou)
    out_dir = Path(resolve_project(args.project)) / f"{args.name}_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with tracking_run(args, name_suffix="_sweep") as tracker:
        for conf in confs:
            for iou in ious:
                metrics = model.val(
                    data=args.data,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    device=args.device,
                    split=args.split,
                    conf=conf,
                    iou=iou,
                    project=str(out_dir),
                    name=f"conf{conf:g}_iou{iou:g}",
                    exist_ok=True,
                )
                box = getattr(metrics, "box", None)
                row = {
                    "conf": conf,
                    "iou": iou,
                    "map50": float(getattr(box, "map50", float("nan"))) if box is not None else float("nan"),
                    "map50_95": float(getattr(box, "map", float("nan"))) if box is not None else float("nan"),
                    "precision": float(getattr(box, "mp", float("nan"))) if box is not None else float("nan"),
                    "recall": float(getattr(box, "mr", float("nan"))) if box is not None else float("nan"),
                }
                rows.append(row)
                if args.tracking == "wandb":
                    tracker.log(row)
                elif args.tracking == "mlflow":
                    tracker.log_metrics({f"conf{conf:g}_iou{iou:g}_{k}": v for k, v in row.items() if k not in {"conf", "iou"}})
                print(row)

    csv_path = out_dir / "threshold_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["conf", "iou", "map50", "map50_95", "precision", "recall"])
        writer.writeheader()
        writer.writerows(rows)
    json_path = out_dir / "threshold_sweep.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Sweep written to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None, help="YOLO data.yaml")
    parser.add_argument("--zip", default=None, help="Optional Roboflow/YOLO dataset ZIP to extract and normalize")
    parser.add_argument("--extract-dir", default="data/ball_v5", help="Extraction directory for --zip")
    parser.add_argument("--no-merge-ball-classes", action="store_true",
                        help="Keep exported class IDs instead of merging Ball/ball into class 0")
    parser.add_argument("--model", default="yolo11s.pt", help="Base model for training")
    parser.add_argument("--weights", default="models/ball_v5.pt", help="Weights for eval-only")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Prepare --zip/--make-split data and exit without training or evaluation")
    parser.add_argument("--sweep", action="store_true", help="Run confidence/NMS IoU sweep with --weights")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate override")
    parser.add_argument("--lrf", type=float, default=None, help="Final LR fraction override")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay override")
    parser.add_argument("--warmup-epochs", type=float, default=None, help="Warmup epochs override")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable mixed precision; use --no-amp if training shows NaN/Inf checkpoints")
    parser.add_argument("--project", default="runs/ball_v5")
    parser.add_argument("--name", default="ball_v5")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--tracking", choices=["wandb", "mlflow", "tensorboard"], default=None,
                        help="Enable Ultralytics experiment tracking backend when installed/configured")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--metrics-json", default=None, help="Optional JSON output for eval-only metrics")
    parser.add_argument("--eval-domains", action="store_true",
                        help="With --eval-only, also evaluate source-domain subsets such as insta360_style and veo_style")
    parser.add_argument("--eval-domain", action="append", choices=["insta360_style", "veo_style", "unknown"],
                        help="Source-domain subset to evaluate; repeat to choose multiple domains")
    parser.add_argument("--domain-subsets-dir", default="runs/ball_v5/domain_subsets",
                        help="Where to create hardlinked/copied YOLO subset datasets for --eval-domains")
    parser.add_argument("--domain-metrics-json", default=None,
                        help="Optional JSON output for --eval-domains metrics")
    parser.add_argument("--sweep-conf", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40")
    parser.add_argument("--sweep-iou", default="0.45,0.50,0.55,0.60,0.65")

    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--copy-paste", type=float, default=0.20)
    parser.add_argument("--scale", type=float, default=0.90)
    parser.add_argument("--degrees", type=float, default=3.0)
    parser.add_argument("--fliplr", type=float, default=0.50)
    parser.add_argument("--close-mosaic", type=int, default=20)

    parser.add_argument("--make-split", action="store_true")
    parser.add_argument("--source-dir", default=None, help="Flat/exported image+label pool")
    parser.add_argument("--split-dir", default="data/ball_v5")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.device = args.device or best_device()

    if args.zip:
        args.data = str(prepare_zip_dataset(
            Path(args.zip),
            Path(args.extract_dir),
            merge_classes=not args.no_merge_ball_classes,
        ))

    if args.make_split:
        if not args.source_dir:
            raise SystemExit("--source-dir is required with --make-split")
        args.data = str(make_split(Path(args.source_dir), Path(args.split_dir), args.val_frac, args.test_frac, args.seed))

    if not args.data:
        raise SystemExit("--data is required unless --make-split creates it")

    if args.prepare_only:
        print(f"Prepared dataset: {args.data}")
    elif args.sweep:
        sweep(args)
    elif args.eval_only:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
