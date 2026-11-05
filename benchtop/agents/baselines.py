import itertools
from pathlib import Path

import numpy as np

from benchtop.agents.estimator import ESTIMATOR_ID, arms, estimate_answer, public_problem
from benchtop.cache import file_hash
from benchtop.episode import Episode
from benchtop.space import canon, config_key
from benchtop.stats import GRADER_ID, point_label
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


def paired_uniform_public(description: dict, run, rng: np.random.Generator) -> dict:
    """The original paired sweep schedule, using only public description and result callbacks."""
    space, question = public_problem(description)
    factor, a, b = arms(space, question)
    contexts = grid_over(space, [factor])
    order = [contexts[i] for i in rng.permutation(len(contexts))]
    batch = [{**context, factor: level} for context in order for level in (a, b)]
    limit = description["replicate_limit"]
    calls_left = description["budget"]["calls_left"]
    if type(limit) is not int or limit < 1 or type(calls_left) is not int or calls_left < 0:
        raise ValueError("invalid public replicate or call limit")
    observations, calls = [], 0
    stop = "call_limit" if calls_left < limit else "replicate_limit"
    for _ in range(min(limit, calls_left)):
        out = run(batch, seeds=1)
        calls += 1
        observations.extend(out.get("results", []))
        if "error" in out or out.get("errors"):
            stop = "experiment_error"
            break
        if "budget exhausted" in out.get("notes", []):
            stop = "budget_exhausted"
            break
        if not out.get("results"):
            stop = "no_observations"
            break
    estimate = estimate_answer(space, question, observations, allowed_ids=range(1, limit + 1))
    return {"policy_id": "paired_uniform_v1", "estimate": estimate, "answer": estimate["answer"],
            "observations": observations, "policy_calls": calls, "stop_reason": stop}


# Historical BASELINES/play remain unchanged, so their saved cohort and report retain their meaning.
RELEASE_BASELINES = {"paired_uniform": paired_uniform_public}


def play_release(ep: Episode, name: str = "paired_uniform", seed: int = 0) -> dict:
    """Trusted evaluation adapter. Scoring and private metadata are read only after acquisition ends."""
    if ep.env.grader_id != GRADER_ID:
        raise ValueError("release policies require corrected task artifacts; do not relabel historical tasks")
    if ep.done or ep.calls or ep.log:
        raise ValueError("release policies require a fresh episode")
    description = {**ep.describe(), "budget": ep.budget_left()}
    outcome = RELEASE_BASELINES[name](description, ep.run, np.random.default_rng(seed))
    ep.submit(outcome["answer"])
    if not ep.done:
        raise RuntimeError("release estimator produced an invalid submission")
    return {**outcome, "agent": outcome["policy_id"], "estimator_id": ESTIMATOR_ID,
            "episode": ep.id, "env": ep.env.id, "task": ep.task["id"], "kind": ep.task["kind"],
            "grader_id": ep.env.grader_id, "execution_id": ep.env.execution_id,
            "task_version": ep.env.task_version, "charging_id": ep.charging_id,
            "policy_seed": seed, "episode_seed": ep.seed, "budget": ep.budget, "spent": ep.spent,
            "budget_frac": ep.budget / ep.full_cost, "spent_frac": ep.result["spent_frac"],
            "calls": ep.calls, "score": ep.result["score"],
            "ran": [pair for entry in ep.log if entry["action"] == "run" for pair in entry["ran"]],
            "source_sha256": {"policy": file_hash(Path(__file__)),
                              "estimator": file_hash(Path(__file__).with_name("estimator.py")),
                              "description": file_hash(Path(__file__).parent.parent / "episode.py")}}
