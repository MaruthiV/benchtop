import math

import numpy as np
from scipy.stats import t

LABELS = ("paper_better", "baseline_better", "no_meaningful_difference")
GRADER_ID = "seed_block_t_v1"
LEGACY_GRADER_ID = "historical_v0"


def confidence_interval(delta: float, se: float, n: int) -> tuple[float, float]:
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("a confidence interval requires at least two paired seeds")
    if not math.isfinite(delta) or not math.isfinite(se) or se < 0:
        raise ValueError("contrast estimates must be finite with nonnegative standard error")
    radius = float(t.ppf(0.975, n - 1)) * se
    return float(delta - radius), float(delta + radius)


def interval_label(lo: float, hi: float, min_effect: float) -> str | None:
    if not math.isfinite(min_effect) or min_effect < 0:
        raise ValueError("the effect margin must be finite and nonnegative")
    if lo > min_effect:
        return "paper_better"
    if hi < -min_effect:
        return "baseline_better"
    if -min_effect < lo and hi < min_effect:
        return "no_meaningful_difference"
    return None


def label(delta: float, se: float, min_effect: float, n: int) -> str | None:
    return interval_label(*confidence_interval(delta, se, n), min_effect)


def point_label(delta: float, min_effect: float) -> str:
    if delta >= min_effect:
        return "paper_better"
    if delta <= -min_effect:
        return "baseline_better"
    return "no_meaningful_difference"


def cell_diff(a: dict[int, float], b: dict[int, float]) -> tuple[float, float]:
    delta, se = average_contrast([(a, b)])
    return delta, se * se


def seed_means(pairs: list[tuple[dict, dict]]) -> np.ndarray:
    if not pairs:
        raise ValueError("a contrast requires at least one paired context")
    seeds = sorted(pairs[0][0])
    expected = set(seeds)
    if len(seeds) < 2:
        raise ValueError("a contrast requires at least two paired seeds")
    if any(set(a) != expected or set(b) != expected for a, b in pairs):
        raise ValueError("incomplete paired seed grid")
    values = np.array([[a[s] - b[s] for s in seeds] for a, b in pairs], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("paired differences must be finite")
    means = values.mean(axis=0)
    if not np.isfinite(means).all():
        raise ValueError("seed means must be finite")
    return means


def average_contrast(pairs: list[tuple[dict, dict]]) -> tuple[float, float]:
    means = seed_means(pairs)
    return float(means.mean()), float(means.std(ddof=1) / math.sqrt(len(means)))


def summarize_contrast(pairs: list[tuple[dict, dict]], min_effect: float) -> dict:
    means = seed_means(pairs)
    n = len(means)
    delta, se = float(means.mean()), float(means.std(ddof=1) / math.sqrt(n))
    interval = confidence_interval(delta, se, n)
    truth = interval_label(*interval, min_effect)
    rng = np.random.default_rng(1729)
    samples = means[rng.integers(0, n, size=(2000, n))].mean(axis=1)
    bootstrap = [float(v) for v in np.percentile(samples, [2.5, 97.5])]
    bootstrap_label = interval_label(*bootstrap, min_effect)
    return {
        "delta": delta,
        "se": se,
        "n_seeds": n,
        "interval": list(interval),
        "label": truth,
        "status": "settled" if truth is not None else "ambiguous",
        "bootstrap_interval": bootstrap,
        "bootstrap_label": bootstrap_label,
        "diagnostic_disagreement": bootstrap_label != truth,
    }


def pooled_seed_std(groups: list[np.ndarray]) -> float:
    num, den = 0.0, 0
    for g in groups:
        if len(g) > 1:
            num += g.var(ddof=1) * (len(g) - 1)
            den += len(g) - 1
    return math.sqrt(num / den) if den else 0.0
