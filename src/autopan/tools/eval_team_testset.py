# src/autopan/tools/eval_team_testset.py
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import cv2

from src.autopan.team_cluster import TeamClusterConfig, extract_kit_feature


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def run_kmeans(X: np.ndarray, k: int, iters: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    C = X[rng.integers(0, n, size=k)].copy()

    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        newC = C.copy()
        for j in range(k):
            sel = X[labels == j]
            if len(sel) > 0:
                newC[j] = sel.mean(axis=0)
        newC = newC / (np.linalg.norm(newC, axis=1, keepdims=True) + 1e-8)
        if np.max(np.abs(newC - C)) < 1e-4:
            C = newC
            break
        C = newC
    return C, labels


def best_label_mapping(pred, gold, k):
    # brute-force mapping for small k
    import itertools
    uniq_gold = sorted(set(gold))
    # restrict to up to k gold labels
    uniq_gold = uniq_gold[:k]

    best = None
    best_acc = -1
    for perm in itertools.permutations(uniq_gold, k):
        mapping = {j: perm[j] for j in range(k)}
        mapped = [mapping[p] for p in pred]
        acc = sum(int(m == g) for m, g in zip(mapped, gold)) / len(gold)
        if acc > best_acc:
            best_acc = acc
            best = mapping
    return best, best_acc


def main():
    DATASET_DIR = Path("data/team_testset_v1")
    LABELS = DATASET_DIR / "labels.jsonl"
    CROPS = DATASET_DIR / "crops"

    cfg = TeamClusterConfig()

    rows = load_jsonl(LABELS)
    # use only team0/team1/ref/other (exclude skip)
    rows = [r for r in rows if r.get("label") in ("team0", "team1", "ref", "other")]
    print(f"Loaded {len(rows)} labeled crops")

    feats = []
    gold = []
    kept = 0
    dropped = 0

    # Since these are already torso crops, we can feed them as an image with a fake full-box
    class FakeBox:
        def __init__(self, w, h):
            self.x1, self.y1, self.x2, self.y2 = 0, 0, w, h

    for r in rows:
        img = cv2.imread(str(CROPS / r["crop_file"]))
        if img is None:
            dropped += 1
            continue

        b = FakeBox(img.shape[1], img.shape[0])
        f = extract_kit_feature(img, b, cfg)  # uses torso crop again; OK but redundant
        if f is None:
            dropped += 1
            continue

        feats.append(f)
        gold.append(r["label"])
        kept += 1

    print(f"Features kept={kept}, dropped={dropped} (due to masking/too-small/too-few pixels)")

    X = np.stack(feats, axis=0)

    # Evaluate k=2 on just team0/team1
    team_mask = [g in ("team0", "team1") for g in gold]
    X2 = X[team_mask]
    g2 = [g for g in gold if g in ("team0", "team1")]
    if len(g2) >= 30:
        C2, p2 = run_kmeans(X2, k=2)
        mapping, acc = best_label_mapping(p2.tolist(), g2, k=2)
        print(f"[k=2] best_acc={acc:.3f} mapping={mapping}")
    else:
        print("[k=2] not enough team samples")

    # Evaluate k=3 on team0/team1/ref (exclude other)
    tri_mask = [g in ("team0", "team1", "ref") for g in gold]
    X3 = X[tri_mask]
    g3 = [g for g in gold if g in ("team0", "team1", "ref")]
    if len(g3) >= 50:
        C3, p3 = run_kmeans(X3, k=3)
        mapping, acc = best_label_mapping(p3.tolist(), g3, k=3)
        print(f"[k=3] best_acc={acc:.3f} mapping={mapping}")
    else:
        print("[k=3] not enough (team/ref) samples")

    # Simple ambiguity metric: margin between closest and 2nd closest center (for k=2)
    if len(g2) >= 30:
        d2 = ((X2[:, None, :] - C2[None, :, :]) ** 2).sum(axis=2)
        srt = np.sort(d2, axis=1)
        margin = np.sqrt(srt[:, 1]) - np.sqrt(srt[:, 0])
        print(f"[Ambiguity k=2] margin mean={margin.mean():.3f}  p10={np.percentile(margin,10):.3f}")
        print("  (low margins => unstable assignments in video)")

    # distribution
    print("Gold distribution:", Counter(gold))


if __name__ == "__main__":
    main()
