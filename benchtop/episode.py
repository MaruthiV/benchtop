import json
import secrets
import time

import numpy as np

from benchtop.env import Env
from benchtop.space import canon, config_key
from benchtop.stats import LABELS
from benchtop.tasks import COMPONENT_LABELS, grade

BUDGET_FRAC = 0.25
MAX_CALLS = 40


def fmt(level) -> str:
    return level if isinstance(level, str) else canon(level)


class Episode:
    def __init__(
        self,
        env: Env,
        task_id: str,
        budget_frac: float = BUDGET_FRAC,
        seed: int | None = None,
        show_paper: bool = True,
        max_calls: int = MAX_CALLS,
    ):
        self.id = secrets.token_hex(6)
        self.env = env
        self.task = env.task(task_id)
        self.seed = secrets.randbits(32) if seed is None else seed
        self.show_paper = show_paper
        self.max_calls = max_calls
        visible = env.visible_seeds
        self.full_cost = env.table.total_cost() * len(visible)
        self.budget = budget_frac * self.full_cost
        self.spent = 0.0
        self.calls = 0
        self.used: dict[str, int] = {}
        # each episode sees the visible seeds in its own order
        self.order = [int(s) for s in np.random.default_rng(self.seed).permutation(visible)]
        self.log: list[dict] = []
        self.result: dict | None = None
        self.started = time.time()

    @property
    def done(self) -> bool:
        return self.result is not None

    def avg_cost(self) -> float:
        return self.full_cost / (self.env.space.size() * len(self.env.visible_seeds))

    def plan(self, configs: list[dict], seeds: int) -> tuple[list[tuple[dict, str, int, int]], list[str]]:
        space, notes = self.env.space, []
        spent, used, steps = self.spent, dict(self.used), []
        seeds = max(1, min(int(seeds), len(self.order)))
        for raw in configs:
            config = space.normalize(raw)
            key = config_key(config)
            for _ in range(seeds):
                k = used.get(key, 0)
                if k >= len(self.order):
                    notes.append(f"no fresh seeds left for {key}")
                    break
                cost = self.env.table.cost_key(key)
                if spent + cost > self.budget + 1e-9:
                    notes.append("budget exhausted")
                    return steps, notes
                steps.append((config, key, self.order[k], k + 1))
                used[key] = k + 1
                spent += cost
        return steps, notes

    def pending(self, configs, seeds) -> list[tuple[dict, int]]:
        try:
            configs, seeds = clean_request(configs, seeds)
            steps, _ = self.plan(configs, seeds)
        except (TypeError, ValueError):
            return []
        return [(c, s) for c, _, s, _ in steps]

    def run(self, configs: list[dict], seeds: int = 1) -> dict:
        if self.done:
            return {"error": "episode is finished"}
        if self.calls >= self.max_calls:
            return {"error": f"call limit of {self.max_calls} reached; submit an answer"}
        try:
            configs, seeds = clean_request(configs, seeds)
            steps, notes = self.plan(configs, seeds)
        except (TypeError, ValueError) as e:
            return {"error": str(e)}
        self.calls += 1
        results = []
        for config, key, seed, replicate in steps:
            metrics = self.env.table.cells.get(key, {}).get(seed)
            if metrics is None:
                notes.append(f"missing result for {key}")
                continue
            cost = self.env.table.cost_key(key)
            self.spent += cost
            self.used[key] = replicate
            results.append(
                {
                    "config": config,
                    "replicate": replicate,
                    "metrics": {k: round(v, 6) for k, v in metrics.items()},
                    "cost": round(cost, 4),
                }
            )
        self.log.append(
            {
                "t": time.time() - self.started,
                "action": "run",
                "configs": configs,
                "seeds": seeds,
                "ran": [[key, s] for _, key, s, _ in steps],
                "notes": notes,
            }
        )
        return {"results": results, "notes": sorted(set(notes)), "budget": self.budget_left()}

    def budget_left(self) -> dict:
        left = max(0.0, self.budget - self.spent)
        return {
            "spent": round(self.spent, 3),
            "left": round(left, 3),
            "total": round(self.budget, 3),
            "runs_left_approx": int(left / self.avg_cost()),
            "calls_left": self.max_calls - self.calls,
        }

    def submit(self, answer, confidence: float | None = None) -> dict:
        if self.done:
            return {"error": "already submitted", **self.result}
        graded = grade(self.env, self.task, answer)
        bad_format = "error" in graded and "given" not in graded
        # bad format gets one more try instead of a silent zero
        if bad_format and not any(e["action"] == "bad_submit" for e in self.log):
            self.log.append(
                {
                    "t": time.time() - self.started,
                    "action": "bad_submit",
                    "answer": answer,
                    "error": graded["error"],
                }
            )
            return {"error": graded["error"], "hint": "fix the format and submit again"}
        self.result = {
            "score": graded["score"],
            "confidence": confidence,
            "graded": graded,
            "spent_frac": self.spent / self.budget if self.budget else 0.0,
            "calls": self.calls,
        }
        self.log.append(
            {
                "t": time.time() - self.started,
                "action": "submit",
                "answer": answer,
                "confidence": confidence,
                "score": graded["score"],
            }
        )
        return {"done": True, "score": graded["score"]}

    def describe(self) -> dict:
        env, space = self.env, self.env.space
        factors = [
            {"name": f.name, "levels": list(f.levels), "description": f.description} for f in space.factors
        ]
        goal = "higher is better" if space.goal == "max" else "lower is better"
        info = {
            "episode": self.id,
            "lab": space.title,
            "setting": space.setting,
            "metric": {"name": space.metric, "goal": space.goal, "description": space.metric_description},
            "factors": factors,
            "treatment": {
                "factor": space.treatment,
                "paper_method": space.paper_level,
                "baseline": space.baseline_level,
            },
            "question": self.question(),
            "answer_format": self.answer_format(),
            "rules": [
                f"each run is one config and one seed; at most {len(self.order)} seeds per config",
                f"budget is {self.budget:.1f} compute-seconds, about {int(self.budget / self.avg_cost())} runs",
                f"at most {self.max_calls} run_experiments calls",
                f"{space.metric}: {goal}; a difference only counts if it is at least {space.min_effect:g}",
                "your answer is graded on fresh seeds you never see, so account for noise",
            ],
        }
        if self.show_paper and env.paper:
            info["paper"] = {k: env.paper[k] for k in ("title", "arxiv_id", "claim") if k in env.paper}
        return info

    def question(self) -> str:
        space, task = self.env.space, self.task
        p, b = fmt(space.paper_level), fmt(space.baseline_level)
        m = space.metric
        if task["kind"] == "claim":
            return (
                f"Averaged over every setting of the other factors in this lab, does "
                f"{space.treatment}={p} improve {m} compared with {space.treatment}={b}?"
            )
        if task["kind"] == "where":
            # list every level, not just the graded ones, so borderline levels don't stand out
            levels = ", ".join(fmt(v) for v in space.factor(task["factor"]).levels)
            return (
                f"For each value of {task['factor']} ({levels}), averaged over the other factors, "
                f"is {space.treatment}={p} better, worse, or about the same as {b} on {m}?"
            )
        if task["kind"] == "component":
            f = space.factor(task["factor"])
            return (
                f"Averaged over all other factors, does turning {f.name} from {fmt(f.levels[0])} "
                f"to {fmt(f.levels[1])} help, hurt, or not matter for {m}?"
            )
        return f"Which single configuration of all factors gives the best {m}?"

    def answer_format(self) -> str:
        task = self.task
        if task["kind"] == "claim":
            return f"a string, one of {list(LABELS)}"
        if task["kind"] == "component":
            return f"a string, one of {list(COMPONENT_LABELS.values())}"
        if task["kind"] == "where":
            keys = [fmt(v) for v in self.env.space.factor(task["factor"]).levels]
            return f"an object mapping each of {keys} to one of {list(LABELS)}"
        return "an object with one level for every factor, e.g. " + canon(self.env.space.default)

    def summary(self) -> dict:
        return {
            "episode": self.id,
            "env": self.env.id,
            "task": self.task["id"],
            "seed": self.seed,
            "show_paper": self.show_paper,
            "budget": self.budget,
            "spent": self.spent,
            "calls": self.calls,
            "result": self.result,
            "log": self.log,
        }


def clean_request(configs, seeds) -> tuple[list[dict], int]:
    # weak agents send json strings, a single dict, or seeds as a list
    if isinstance(configs, str):
        configs = json.loads(configs)
    if isinstance(configs, dict):
        configs = [configs]
    if not isinstance(configs, list) or not all(isinstance(c, dict) for c in configs):
        raise TypeError("configs must be a list of objects mapping factor names to levels")
    if not configs:
        raise ValueError("configs is empty")
    if isinstance(seeds, list):
        seeds = len(seeds)
    try:
        seeds = int(seeds)
    except (TypeError, ValueError):
        raise TypeError("seeds must be an integer") from None
    return configs, seeds


def _unkey(key: str):
    return json.loads(key)
