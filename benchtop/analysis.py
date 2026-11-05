import contextlib
import json

import numpy as np

from benchtop.agents.baselines import answer_from, prior_only
from benchtop.env import Env
from benchtop.episode import Episode
from benchtop.tasks import grade, norm_label


def runs_from_pairs(env: Env, pairs: list[list]) -> list[dict]:
    runs = []
    for key, seed in pairs:
        config = json.loads(key)
        metrics = env.table.get(config, int(seed))
        if metrics is not None:
            runs.append({"config": config, "seed": int(seed), "metrics": metrics})
    return runs


def shared_estimate_from_pairs(env: Env, task_id: str, pairs: list[list]):
    """Explicit common-estimator diagnostic; historical reconstruction keeps its original default."""
    from benchtop.agents.estimator import estimate_answer

    for key, seed in pairs:
        if type(seed) is not int or seed not in env.visible_seeds:
            raise ValueError("trace reconstruction requires actual visible seed identities")
        config = json.loads(key)
        if env.table.get(config, seed) is None:
            raise ValueError("trace references an unavailable observation")
    observations = runs_from_pairs(env, pairs)
    observations = [{**row, "metrics": {key: round(value, 6) for key, value in row["metrics"].items()}}
                    for row in observations]
    task = env.task(task_id)
    question = {key: task[key] for key in ("kind", "factor") if key in task}
    return estimate_answer(env.space, question, observations, identity="seed", allowed_ids=env.visible_seeds)


def implied_answer(env: Env, task_id: str, pairs: list[list]):
    # what the statistician's rule says about the agent's own data
    ep = Episode(env, task_id, seed=0)
    return answer_from(ep, runs_from_pairs(env, pairs))


def same(env: Env, task: dict, a, b) -> bool:
    if task["kind"] in ("claim", "component"):
        return norm_label(a) == norm_label(b)
    if task["kind"] == "where":
        ga = grade(env, {**task, "answer": _as_truth(env, task, b)}, a)
        return ga.get("score") == 1.0
    return False


def _as_truth(env: Env, task: dict, answer) -> dict:
    from benchtop.space import canon

    f = env.space.factor(task["factor"])
    out = {}
    for k, v in (answer or {}).items():
        with contextlib.suppress(ValueError):
            out[canon(f.match(_parse(k)))] = norm_label(v)
    return {k: out.get(k) for k in task["answer"]}


def _parse(k):
    for cast in (int, float):
        try:
            return cast(k)
        except (TypeError, ValueError):
            pass
    return k


def decompose(env: Env, row: dict) -> str | None:
    task = env.task(row["task"])
    if task["kind"] not in ("claim", "where") or not row.get("submitted"):
        return None
    answer = row.get("answer")
    if isinstance(answer, dict) and "answer" in answer and task["kind"] == "claim":
        answer = answer["answer"]
    implied = implied_answer(env, row["task"], row.get("ran", []))
    prior = prior_only(Episode(env, row["task"], seed=0))
    right = row.get("score") == 1.0
    if right:
        return "right"
    if not same(env, task, answer, implied) and same(env, task, answer, prior):
        return "sided_with_paper_over_own_data"
    if same(env, task, answer, implied):
        return "followed_own_data_but_wrong"
    return "other_wrong"


def calibration(rows: list[dict]) -> dict:
    pairs = [
        (r["confidence"], float(r["score"] == 1.0))
        for r in rows
        if isinstance(r.get("confidence"), (int, float)) and r.get("score") is not None
    ]
    if not pairs:
        return {}
    conf, hit = np.array(pairs).T
    return {
        "n": len(pairs),
        "mean_confidence": float(conf.mean()),
        "accuracy": float(hit.mean()),
        "brier": float(((conf - hit) ** 2).mean()),
        "overconfidence": float(conf.mean() - hit.mean()),
    }
