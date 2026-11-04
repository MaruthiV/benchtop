import itertools

import numpy as np

from benchtop.episode import Episode
from benchtop.space import canon, config_key
from benchtop.stats import point_label
from benchtop.tasks import COMPONENT_LABELS, grid_over, sign


def _arms(ep: Episode) -> tuple[str, object, object]:
    space, task = ep.env.space, ep.task
    if task["kind"] == "component":
        f = space.factor(task["factor"])
        return f.name, f.levels[1], f.levels[0]
    return space.treatment, space.paper_level, space.baseline_level


def _observed(ep: Episode, runs: list[dict]) -> dict[str, list[float]]:
    obs: dict[str, list[float]] = {}
    for r in runs:
        obs.setdefault(config_key(r["config"]), []).append(r["metrics"][ep.env.space.metric])
    return obs


def _contrast(ep: Episode, obs: dict, fixed: dict | None = None) -> float | None:
    factor, a, b = _arms(ep)
    diffs = []
    for ctx in grid_over(ep.env.space, [factor], fixed):
        va, vb = obs.get(config_key({**ctx, factor: a})), obs.get(config_key({**ctx, factor: b}))
        if va and vb:
            diffs.append((np.mean(va) - np.mean(vb)) * sign(ep.env.space))
    return float(np.mean(diffs)) if diffs else None


def answer_from(ep: Episode, runs: list[dict]):
    space, task = ep.env.space, ep.task
    obs = _observed(ep, runs)
    if task["kind"] in ("claim", "component"):
        d = _contrast(ep, obs)
        lab = point_label(d, space.min_effect) if d is not None else "no_meaningful_difference"
        return COMPONENT_LABELS[lab] if task["kind"] == "component" else lab
    if task["kind"] == "where":
        f = space.factor(task["factor"])
        out = {}
        for level in f.levels:
            d = _contrast(ep, obs, fixed={f.name: level})
            out[str(level)] = (
                point_label(d, space.min_effect) if d is not None else "no_meaningful_difference"
            )
        return out
    means = {k: np.mean(v) * sign(space) for k, v in obs.items()}
    if not means:
        return space.default
    return dict(_from_key(max(means, key=means.get)))


def _from_key(key: str) -> dict:
    import json

    return json.loads(key)


def _spend(ep: Episode, configs: list[dict]) -> list[dict]:
    runs = []
    for config in configs:
        out = ep.run([config], seeds=1)
        runs.extend(out.get("results", []))
        if "budget exhausted" in out.get("notes", []) or "error" in out:
            break
    return runs


def prior_only(ep: Episode, rng=None):
    task, space = ep.task, ep.env.space
    if task["kind"] == "claim":
        return "paper_better"
    if task["kind"] == "component":
        return "helps"
    if task["kind"] == "where":
        return {str(level): "paper_better" for level in space.factor(task["factor"]).levels}
    return {**space.default, space.treatment: space.paper_level}


def random_design(ep: Episode, rng: np.random.Generator):
    configs = ep.env.space.configs()
    picks = [configs[i] for i in rng.integers(0, len(configs), 10_000)]
    # run() calls are capped, so batch the random picks
    runs, batch = [], max(1, len(picks) // ep.max_calls)
    for i in range(0, len(picks), batch):
        out = ep.run(picks[i : i + batch], seeds=1)
        runs.extend(out.get("results", []))
        if "budget exhausted" in out.get("notes", []) or "error" in out:
            break
    return answer_from(ep, runs)


def uniform_design(ep: Episode, rng: np.random.Generator):
    space, task = ep.env.space, ep.task
    if task["kind"] == "best":
        return _halving(ep, rng)
    factor, a, b = _arms(ep)
    contexts = grid_over(space, [factor])
    order = [contexts[i] for i in rng.permutation(len(contexts))]
    pairs = [[{**c, factor: a}, {**c, factor: b}] for c in order]
    runs = []
    for _ in range(len(ep.order)):
        batch = list(itertools.chain.from_iterable(pairs))
        out = ep.run(batch, seeds=1)
        runs.extend(out.get("results", []))
        if "budget exhausted" in out.get("notes", []) or "error" in out:
            break
    return answer_from(ep, runs)


def _halving(ep: Episode, rng: np.random.Generator):
    space = ep.env.space
    alive = [space.configs()[i] for i in rng.permutation(space.size())]
    runs: list[dict] = []
    while alive:
        out = ep.run(alive, seeds=1)
        runs.extend(out.get("results", []))
        if "budget exhausted" in out.get("notes", []) or "error" in out or len(alive) == 1:
            break
        obs = _observed(ep, runs)
        alive.sort(key=lambda c: -np.mean(obs.get(config_key(c), [-np.inf])) * sign(space))
        alive = alive[: max(1, len(alive) // 2)]
    return answer_from(ep, runs)


BASELINES = {"prior": prior_only, "random": random_design, "uniform": uniform_design}


def play(ep: Episode, name: str, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    answer = BASELINES[name](ep, rng)
    ep.submit(answer)
    return ep.result


def canon_answer(answer) -> str:
    return canon(answer)
