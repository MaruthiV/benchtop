import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from benchtop.agents import baselines
from benchtop.agents.common import RESULTS, Client, record
from benchtop.env import ENVS, Env
from benchtop.episode import Episode


def accepted_ids() -> set[str]:
    from benchtop.pipeline import FUNNEL

    rows = [json.loads(x) for x in FUNNEL.read_text().splitlines()] if FUNNEL.exists() else []
    return {r["env"] for r in rows if r["outcome"] == "accepted"}


def all_envs(only_accepted: bool = True) -> list[Env]:
    ok = accepted_ids()
    dirs = [p.parent for p in sorted(ENVS.glob("*/tasks.json"))]
    return [Env.load(d) for d in dirs if not only_accepted or d.name in ok]


def run_baselines(episodes: int = 50, budget_frac: float = 0.25, name: str = "baselines.json") -> list[dict]:
    rows = []
    for env in all_envs():
        for task in env.tasks:
            for agent in baselines.BASELINES:
                scores = [
                    baselines.play(Episode(env, task["id"], seed=s, budget_frac=budget_frac), agent, seed=s)[
                        "score"
                    ]
                    for s in range(episodes)
                ]
                rows.append(
                    {
                        "env": env.id,
                        "task": task["id"],
                        "kind": task["kind"],
                        "agent": agent,
                        "mean": float(np.mean(scores)),
                        "n": episodes,
                        "budget_frac": budget_frac,
                    }
                )
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(rows, indent=1) + "\n")
    return rows


def episode_list(
    env_ids: list[str] | None = None, kinds: tuple[str, ...] = ("claim", "where", "component", "best")
):
    out = []
    for env in all_envs():
        if env_ids and env.id not in env_ids:
            continue
        out += [(env.id, t["id"]) for t in env.tasks if t["kind"] in kinds]
    return out


def run_rollouts(
    agent: str,
    pairs: list[tuple[str, str]],
    repeats: int = 1,
    show_paper=(True,),
    jobs: int = 2,
    base: str = "http://127.0.0.1:8787",
    tag: str = "",
    seed0: int = 0,
    group: bool = False,
    budget_frac: float = 0.25,
) -> list[dict]:
    backend, model = agent.split(":", 1)
    if backend == "claude":
        from benchtop.agents.claude import rollout
    elif backend == "ollama":
        from benchtop.agents.ollama import rollout
    elif backend == "codex":
        from benchtop.agents.codex import rollout
    else:
        raise ValueError(f"unknown agent backend {backend}")
    # group mode gives every repeat the same episode seed, like k samples of one prompt
    work = [
        (e, t, sp, seed0 if group else seed0 + r)
        for e, t in pairs
        for sp in show_paper
        for r in range(repeats)
    ]

    def one(item):
        env, task, sp, seed = item
        try:
            row = rollout(
                Client(base), env, task, model=model, show_paper=sp, seed=seed, budget_frac=budget_frac
            )
        except Exception as e:
            row = {
                "agent": agent,
                "env": env,
                "task": task,
                "show_paper": sp,
                "error": f"{type(e).__name__}: {e}",
            }
        row["tag"] = tag
        row["seed"] = seed
        row["repeat_of_group"] = group
        row["budget_frac"] = budget_frac
        record(row)
        print(
            f"{agent} {env[:40]:40s} {task:18s} paper={sp!s:5s} score={row.get('score')} "
            f"runs={row.get('runs')} secs={row.get('secs')}",
            flush=True,
        )
        return row

    with ThreadPoolExecutor(jobs) as pool:
        return list(pool.map(one, work))


def failed_pairs(agent: str, tag: str = "main", show_paper: bool = True) -> list[tuple[str, str]]:
    from benchtop.report import rollout_rows

    done = {
        (r["env"], r["task"])
        for r in rollout_rows(tag)
        if r["agent"] == agent and r.get("show_paper", True) == show_paper and r.get("submitted")
    }
    return [p for p in episode_list() if p not in done]
