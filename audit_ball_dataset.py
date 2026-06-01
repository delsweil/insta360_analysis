#!/usr/bin/env python3
"""Audit a YOLO ball detector dataset for training/evaluation decisions."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "p05": percentile(values, 0.05),
        "p25": percentile(values, 0.25),
        "median": median(values),
        "mean": mean(values),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def parse_simple_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def resolve_split_dir(data_yaml: Path, values: dict[str, str], key: str) -> tuple[Path, Path]:
    base = Path(values.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = data_yaml.parent / base
    images_dir = Path(values[key])
    if not images_dir.is_absolute():
        images_dir = base / images_dir
    labels_dir = Path(str(images_dir).replace(f"{Path('images')}", f"{Path('labels')}"))
    if images_dir.name == "images":
        labels_dir = images_dir.parent / "labels"
    return images_dir, labels_dir


def image_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def audit_split(images_dir: Path, labels_dir: Path) -> dict:
    images = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    label_paths = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []
    labels_by_stem = {p.stem: p for p in label_paths}

    classes: Counter[str] = Counter()
    dim_counts: Counter[str] = Counter()
    box_width_px: list[float] = []
    box_height_px: list[float] = []
    box_area_px: list[float] = []
    box_width_norm: list[float] = []
    box_height_norm: list[float] = []
    anomalies: list[dict] = []
    empty_labels = 0
    missing_labels = 0
    total_boxes = 0

    for image_path in images:
        dims = image_dimensions(image_path)
        if dims is None:
            anomalies.append({"image": str(image_path), "issue": "unreadable_image"})
            continue
        width, height = dims
        dim_counts[f"{width}x{height}"] += 1
        label_path = labels_by_stem.get(image_path.stem)
        if label_path is None:
            missing_labels += 1
            continue
        rows = [line.split() for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        if not rows:
            empty_labels += 1
        for row_index, parts in enumerate(rows, start=1):
            if len(parts) != 5:
                anomalies.append({
                    "label": str(label_path),
                    "row": row_index,
                    "issue": "non_bbox_row",
                    "columns": len(parts),
                })
                continue
            try:
                cls, cx, cy, bw, bh = parts
                cx_f = float(cx)
                cy_f = float(cy)
                bw_f = float(bw)
                bh_f = float(bh)
            except ValueError:
                anomalies.append({"label": str(label_path), "row": row_index, "issue": "non_numeric_row"})
                continue
            classes[cls] += 1
            total_boxes += 1
            box_width_norm.append(bw_f)
            box_height_norm.append(bh_f)
            box_width_px.append(bw_f * width)
            box_height_px.append(bh_f * height)
            box_area_px.append(bw_f * width * bh_f * height)
            if not (0.0 <= cx_f <= 1.0 and 0.0 <= cy_f <= 1.0 and 0.0 < bw_f <= 1.0 and 0.0 < bh_f <= 1.0):
                anomalies.append({"label": str(label_path), "row": row_index, "issue": "bbox_out_of_range"})

    orphan_labels = [str(p) for p in label_paths if p.stem not in {img.stem for img in images}]
    if orphan_labels:
        anomalies.extend({"label": p, "issue": "orphan_label"} for p in orphan_labels[:50])

    return {
        "images": len(images),
        "labels": len(label_paths),
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "orphan_labels": len(orphan_labels),
        "boxes": total_boxes,
        "boxes_per_image": total_boxes / len(images) if images else 0.0,
        "classes": dict(classes),
        "image_dimensions": dict(dim_counts),
        "bbox_width_px": summarize(box_width_px),
        "bbox_height_px": summarize(box_height_px),
        "bbox_area_px": summarize(box_area_px),
        "bbox_width_norm": summarize(box_width_norm),
        "bbox_height_norm": summarize(box_height_norm),
        "anomaly_count": len(anomalies),
        "anomaly_samples": anomalies[:50],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/ball_v5/data.yaml")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    data_yaml = Path(args.data)
    values = parse_simple_yaml(data_yaml)
    splits = [key for key in ("train", "val", "test") if key in values]
    report = {
        "data_yaml": str(data_yaml),
        "declared_nc": values.get("nc"),
        "splits": {},
    }
    for split in splits:
        images_dir, labels_dir = resolve_split_dir(data_yaml, values, split)
        report["splits"][split] = audit_split(images_dir, labels_dir)

    all_classes = Counter()
    for split_report in report["splits"].values():
        all_classes.update(split_report["classes"])
    report["all_classes"] = dict(all_classes)

    text = json.dumps(report, indent=2)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Dataset audit written to {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
