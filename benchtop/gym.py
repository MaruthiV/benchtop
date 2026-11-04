import json
from functools import cache

from benchtop.env import ENVS, Env
from benchtop.episode import Episode


@cache
def _env(env_id: str) -> Env:
    return Env.load(ENVS / env_id)


def list_episodes() -> list[tuple[str, str]]:
    out = []
    for p in sorted(ENVS.glob("*/tasks.json")):
        out += [(p.parent.name, t["id"]) for t in json.loads(p.read_text())]
    return out


class LabEnv:
    # text in, text out: observations are json strings so any llm policy can consume them
    def __init__(self, env_id: str, task_id: str, budget_frac: float = 0.25, show_paper: bool = True):
        self.env_id, self.task_id = env_id, task_id
        self.budget_frac, self.show_paper = budget_frac, show_paper
        self.ep: Episode | None = None

    def reset(self, seed: int | None = None) -> tuple[str, dict]:
        self.ep = Episode(
            _env(self.env_id),
            self.task_id,
            budget_frac=self.budget_frac,
            seed=seed,
            show_paper=self.show_paper,
        )
        return json.dumps({**self.ep.describe(), "budget": self.ep.budget_left()}), {"episode": self.ep.id}

    def step(self, action: dict) -> tuple[str, float, bool, dict]:
        tool = action.get("tool")
        if tool == "run_experiments":
            out = self.ep.run(action.get("configs", []), action.get("seeds", 1))
            return json.dumps(out), 0.0, False, {}
        if tool == "submit_answer":
            out = self.ep.submit(action.get("answer"), action.get("confidence"))
            if out.get("done"):
                return json.dumps({"received": True}), self.ep.result["score"], True, self.ep.result
            return json.dumps(out), 0.0, False, {}
        if tool == "describe_lab":
            return json.dumps({**self.ep.describe(), "budget": self.ep.budget_left()}), 0.0, False, {}
        return json.dumps({"error": f"unknown tool {tool!r}"}), 0.0, False, {}


def make(env_id: str, task_id: str, **kw) -> LabEnv:
    return LabEnv(env_id, task_id, **kw)
