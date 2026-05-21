#!/usr/bin/env python3
"""
compare_approaches.py — Compare results across approaches.

Usage:
    python3 compare_approaches.py
"""

import json
import math
import numpy as np
from pathlib import Path

results_dir = Path('results')
summaries = sorted(results_dir.glob('*_summary.json'))

if not summaries:
    print("No summary files found in results/")
    exit()

# Load all summaries
data = {}
all_clips = set()
for s in summaries:
    with open(s) as f:
        d = json.load(f)
    approach = d['approach']
    data[approach] = d
    all_clips.update(d['clips'].keys())

all_clips = sorted(all_clips)
approaches = sorted(data.keys())

# Print comparison table
print(f"\n{'Approach':<25}", end='')
for clip in all_clips:
    print(f"  {clip:>6}", end='')
print(f"  {'Mean':>6}  {'Valid':>5}")
print("-" * (25 + len(all_clips)*8 + 15))

for approach in approaches:
    d = data[approach]
    print(f"{approach:<25}", end='')
    clip_rmses = []
    for clip in all_clips:
        if clip in d['clips']:
            rmse = d['clips'][clip]['overall_rmse']
            if not math.isnan(rmse):
                print(f"  {rmse:>6.1f}", end='')
                clip_rmses.append(rmse)
            else:
                print(f"  {'nan':>6}", end='')
        else:
            print(f"  {'---':>6}", end='')
    mean = np.mean(clip_rmses) if clip_rmses else float('nan')
    print(f"  {mean:>6.1f}  {len(clip_rmses):>5}")

print()
