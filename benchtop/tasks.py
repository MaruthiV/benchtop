import contextlib
import itertools
import json

import numpy as np

from benchtop.cache import task_binding, with_checksum
from benchtop.env import Env
from benchtop.space import Space, canon
from benchtop.stats import GRADER_ID, LABELS, average_contrast, pooled_seed_std, summarize_contrast

COMPONENT_LABELS = {
    "paper_better": "helps",
    "baseline_better": "hurts",
    "no_meaningful_difference": "no_meaningful_difference",
}


def grid_over(space: Space, skip: list[str], fixed: dict | None = None) -> list[dict]:
    fixed = fixed or {}
    free = [f for f in space.factors if f.name not in skip and f.name not in fixed]
    out = []
    for combo in itertools.product(*(f.levels for f in free)):
        out.append({**fixed, **dict(zip((f.name for f in free), combo, strict=False))})
    return out


def sign(space: Space) -> float:
    return 1.0 if space.goal == "max" else -1.0


def contrast_pairs(env: Env, factor: str, a, b, seeds: list[int], fixed: dict | None = None):
    space = env.space
    pairs = []
    k = sign(space)
    for ctx in grid_over(space, [factor], fixed):
        va = {s: v * k for s, v in env.table.by_seed({**ctx, factor: a}, seeds).items()}
        vb = {s: v * k for s, v in env.table.by_seed({**ctx, factor: b}, seeds).items()}
        if set(va) != set(seeds) or set(vb) != set(seeds):
            raise ValueError("incomplete qualification seed grid")
        pairs.append((va, vb))
    return pairs


def contrast(env: Env, factor: str, a, b, seeds: list[int], fixed: dict | None = None):
    return average_contrast(contrast_pairs(env, factor, a, b, seeds, fixed))


def top_share(env: Env, factor: str, a, b, seeds: list[int]) -> float:
    # how much of the average effect comes from the single biggest setting
    k, diffs = sign(env.space), []
    for ctx in grid_over(env.space, [factor]):
        va = env.table.values({**ctx, factor: a}, seeds) * k
        vb = env.table.values({**ctx, factor: b}, seeds) * k
        if len(va) and len(vb):
            diffs.append(abs(va.mean() - vb.mean()))
    total = sum(diffs)
    return float(max(diffs) / total) if total > 0 else 0.0


def config_means(env: Env, seeds: list[int]) -> dict[str, float]:
    means = {}
    for config in env.space.configs():
        v = env.table.values(config, seeds)
        if len(v):
            means[canon(config)] = float(v.mean()) * sign(env.space)
    return means


def build_task_set(env: Env, seeds: list[int] | None = None) -> dict:
    space = env.space
    held = list(env.heldout_seeds if seeds is None else seeds)
    tasks, diagnostics = [], []
    report = {
        "schema_version": 1,
        "grader_id": GRADER_ID,
        "seed_ids": held,
        "binding": task_binding(env, held),
        "interval_method": "student_t_95",
        "status": "complete",
        "issues": [],
        "tasks": tasks,
        "contrasts": diagnostics,
    }
    if len(held) < 2 or len(set(held)) != len(held):
        report.update(status="incomplete", issues=[{"kind": "insufficient_or_duplicate_seeds"}])
        return with_checksum(report, "artifact_sha256")
    missing = env.table.missing(held)
    if missing:
        report.update(status="incomplete", issues=[{"kind": "missing_results", "count": len(missing)}])
        return with_checksum(report, "artifact_sha256")
    try:
        values = np.array(
            [env.table.get(c, s)[space.metric] for c in space.configs() for s in held], dtype=float
        )
    except (KeyError, TypeError, ValueError):
        report.update(status="invalid", issues=[{"kind": "invalid_metric"}])
        return with_checksum(report, "artifact_sha256")
    if not np.isfinite(values).all():
        report.update(status="invalid", issues=[{"kind": "nonfinite_metric"}])
        return with_checksum(report, "artifact_sha256")

    def estimate(task_id, factor, a, b, fixed=None, level=None):
        out = summarize_contrast(contrast_pairs(env, factor, a, b, held, fixed), space.min_effect)
        diagnostics.append({"task_id": task_id, **({"level": level} if level is not None else {}), **out})
        return out

    t, p, b = space.treatment, space.paper_level, space.baseline_level
    noise = pooled_seed_std([env.table.values(c, held) for c in space.configs()])

    est = estimate("claim", t, p, b)
    if est["label"] is not None:
        tasks.append(
            {
                "id": "claim",
                "kind": "claim",
                "answer": est["label"],
                "delta": est["delta"],
                "se": est["se"],
                "interval": est["interval"],
                "seed_std": noise,
                "top_share": top_share(env, t, p, b, held),
            }
        )

    for f in space.context_factors():
        per_level, stats = {}, {}
        for level in f.levels:
            est = estimate(f"where:{f.name}", t, p, b, fixed={f.name: level}, level=canon(level))
            stats[canon(level)] = est
            if est["label"] is not None:
                per_level[canon(level)] = est["label"]
        if len(per_level) >= 2 and len(set(per_level.values())) >= 2:
            tasks.append(
                {
                    "id": f"where:{f.name}",
                    "kind": "where",
                    "factor": f.name,
                    "answer": per_level,
                    "stats": stats,
                    "seed_std": noise,
                }
            )

    for f in space.component_factors():
        if len(f.levels) != 2:
            continue
        est = estimate(f"component:{f.name}", f.name, f.levels[1], f.levels[0])
        if est["label"] is not None:
            tasks.append(
                {
                    "id": f"component:{f.name}",
                    "kind": "component",
                    "factor": f.name,
                    "answer": COMPONENT_LABELS[est["label"]],
                    "delta": est["delta"],
                    "se": est["se"],
                    "interval": est["interval"],
                    "seed_std": noise,
                }
            )

    means = config_means(env, held)
    if len(means) == space.size():
        values = np.array(list(means.values()))
        best_key = max(means, key=means.get)
        if values.max() - np.median(values) >= space.min_effect:
            tasks.append(
                {
                    "id": "best",
                    "kind": "best",
                    "answer": best_key,
                    "grading_rule": "heldout_best_mean",
                    "best": float(values.max()),
                    "median": float(np.median(values)),
                    "seed_std": noise,
                }
            )
    for task in tasks:
        task.update(grader_id=GRADER_ID, seed_ids=held, binding=report["binding"])
        task.update(with_checksum(task, "task_sha256"))
    return with_checksum(report, "artifact_sha256")


def build_tasks(env: Env, seeds: list[int] | None = None) -> list[dict]:
    report = build_task_set(env, seeds)
    if report["status"] != "complete":
        raise ValueError(f"qualification data is {report['status']}: {report['issues']}")
    return report["tasks"]


def norm_label(text) -> str:
    return str(text).strip().lower().replace("-", "_").replace(" ", "_")


def grade(env: Env, task: dict, answer) -> dict:
    kind = task["kind"]
    if isinstance(answer, str) and answer.strip().startswith("{"):
        with contextlib.suppress(json.JSONDecodeError):
            answer = json.loads(answer)
    if kind in ("claim", "component"):
        options = LABELS if kind == "claim" else tuple(COMPONENT_LABELS.values())
        given = norm_label(answer if not isinstance(answer, dict) else answer.get("answer", ""))
        if given not in options:
            return {"score": 0.0, "error": f"answer must be one of {list(options)}"}
        return {"score": float(given == task["answer"]), "given": given, "truth": task["answer"]}

    if kind == "where":
        f = env.space.factor(task["factor"])
        if not isinstance(answer, dict):
            return {"score": 0.0, "error": "answer must map each level to a label"}
        given = {}
        for k, v in answer.items():
            try:
                given[canon(f.match(_parse_level(k)))] = norm_label(v)
            except ValueError:
                continue
        hits = [given.get(level) == truth for level, truth in task["answer"].items()]
        return {"score": float(np.mean(hits)), "given": given, "truth": task["answer"]}

    if kind == "best":
        try:
            config = env.space.normalize(answer if isinstance(answer, dict) else {})
        except ValueError as e:
            return {"score": 0.0, "error": str(e)}
        means = config_means(env, task.get("seed_ids", env.heldout_seeds))
        got = means[canon(config)]
        spread = task["best"] - task["median"]
        score = float(np.clip((got - task["median"]) / spread, 0.0, 1.0))
        return {"score": score, "given": config, "truth": task["answer"], "regret": task["best"] - got}
    raise ValueError(f"unknown task kind {kind}")


def _parse_level(key):
    # levels come back as strings from json object keys
    for cast in (int, float):
        try:
            return cast(key)
        except (TypeError, ValueError):
            pass
    if str(key).lower() in ("true", "false"):
        return str(key).lower() == "true"
    return key
