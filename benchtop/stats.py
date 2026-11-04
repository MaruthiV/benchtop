import math

import numpy as np

LABELS = ("paper_better", "baseline_better", "no_meaningful_difference")
Z = 1.96


def label(delta: float, se: float, min_effect: float, z: float = Z) -> str | None:
    # none means the data can't settle it, so the question gets dropped
    lo, hi = delta - z * se, delta + z * se
    if lo > min_effect:
        return "paper_better"
    if hi < -min_effect:
        return "baseline_better"
    if -min_effect < lo and hi < min_effect:
        return "no_meaningful_difference"
    return None


def point_label(delta: float, min_effect: float) -> str:
    if delta >= min_effect:
        return "paper_better"
    if delta <= -min_effect:
        return "baseline_better"
    return "no_meaningful_difference"


def cell_diff(a: dict[int, float], b: dict[int, float]) -> tuple[float, float]:
    shared = sorted(set(a) & set(b))
    if len(shared) >= 2:
        # arms share seeds (same split and init), so pair them
        d = np.array([a[s] - b[s] for s in shared])
        return float(d.mean()), float(d.var(ddof=1) / len(d))
    va, vb = np.array(list(a.values())), np.array(list(b.values()))
    var = (va.var(ddof=1) / len(va) if len(va) > 1 else 0.0) + (
        vb.var(ddof=1) / len(vb) if len(vb) > 1 else 0.0
    )
    return float(va.mean() - vb.mean()), var


def average_contrast(pairs: list[tuple[dict, dict]]) -> tuple[float, float]:
    # mean over contexts of (arm a - arm b), each context weighted equally
    diffs, variances = zip(*(cell_diff(a, b) for a, b in pairs), strict=False)
    n = len(diffs)
    return float(np.mean(diffs)), math.sqrt(sum(variances)) / n


def pooled_seed_std(groups: list[np.ndarray]) -> float:
    num, den = 0.0, 0
    for g in groups:
        if len(g) > 1:
            num += g.var(ddof=1) * (len(g) - 1)
            den += len(g) - 1
    return math.sqrt(num / den) if den else 0.0
