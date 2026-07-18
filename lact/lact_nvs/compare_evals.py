"""Paired per-scene comparison of eval.json files.

Usage: python compare_evals.py <baseline_eval.json> <other_eval.json> [...]
Prints mean PSNR/LPIPS, paired dPSNR/dLPIPS vs the baseline, and a paired
t-statistic (per-scene differences; scenes are matched by eval order, which is
deterministic for the standard protocol).
"""
import json
import sys

import numpy as np


def load(path):
    with open(path) as f:
        return json.load(f)


base = load(sys.argv[1])
bp = np.array(base["per_scene_psnr"])
bl = np.array(base["per_scene_lpips"])
print(f"{'exp':<40} {'PSNR':>8} {'LPIPS':>8} {'dPSNR':>8} {'dLPIPS':>9} {'t(dPSNR)':>9}")
print(f"{sys.argv[1]:<40} {bp.mean():8.3f} {bl.mean():8.4f} {'—':>8} {'—':>9} {'—':>9}")

for path in sys.argv[2:]:
    r = load(path)
    p = np.array(r["per_scene_psnr"])
    l = np.array(r["per_scene_lpips"])
    n = min(len(p), len(bp))
    dp = p[:n] - bp[:n]
    dl = l[:n] - bl[:n]
    t = dp.mean() / (dp.std(ddof=1) / np.sqrt(n) + 1e-12)
    print(f"{path:<40} {p.mean():8.3f} {l.mean():8.4f} {dp.mean():+8.3f} {dl.mean():+9.4f} {t:9.2f}")
