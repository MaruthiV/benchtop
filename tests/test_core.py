import json
import sys
from pathlib import Path

import numpy as np
import pytest

from benchtop.agents.baselines import play
from benchtop.env import Env
from benchtop.episode import Episode
from benchtop.space import Space
from benchtop.stats import label
from benchtop.tasks import build_tasks, grade

SPACE = {
    "title": "toy",
    "metric": {"name": "acc", "goal": "max", "min_effect": 0.02},
    "factors": [
        {"name": "method", "role": "treatment", "levels": ["base", "new"]},
        {"name": "ctx", "role": "context", "levels": [0, 1, 2]},
        {"name": "width", "role": "context", "levels": [8, 64]},
    ],
    "treatment": {"factor": "method", "paper": "new", "baseline": "base"},
}
GAIN = {0: 0.0, 1: -0.05, 2: 0.08}


def toy_env(tmp_path: Path, seeds: int = 16) -> Env:
    space = Space.from_dict(SPACE)
    rows = []
    for config in space.configs():
        for seed in range(seeds):
            noise = np.random.default_rng(seed).normal(0, 0.01)
            acc = 0.7 + 0.1 * (config["width"] == 64) + noise
            if config["method"] == "new":
                acc += GAIN[config["ctx"]] + np.random.default_rng(1000 + seed).normal(0, 0.003)
            rows.append({"config": config, "seed": seed, "ok": True, "metrics": {"acc": acc}, "secs": 1.0})
    d = tmp_path / "toy"
    d.mkdir()
    (d / "space.json").write_text(json.dumps(SPACE))
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "manifest.json").write_text(json.dumps({"id": "toy", "seeds": seeds, "visible_seeds": 4}))
    env = Env.load(d)
    (d / "tasks.json").write_text(json.dumps(build_tasks(env)))
    return Env.load(d)


def test_levels_match_loosely():
    space = Space.from_dict(SPACE)
    assert space.normalize({"ctx": "2", "method": "NEW"}) == {"method": "new", "ctx": 2, "width": 8}
    with pytest.raises(ValueError):
        space.normalize({"ctx": 5})
    with pytest.raises(ValueError):
        space.normalize({"depth": 3})


def test_label_needs_confidence():
    assert label(0.05, 0.005, 0.02) == "paper_better"
    assert label(-0.05, 0.005, 0.02) == "baseline_better"
    assert label(0.001, 0.002, 0.02) == "no_meaningful_difference"
    assert label(0.021, 0.01, 0.02) is None


def test_tasks_and_grading(tmp_path):
    env = toy_env(tmp_path)
    tasks = {t["id"]: t for t in env.tasks}
    assert tasks["claim"]["answer"] == "no_meaningful_difference"
    where = tasks["where:ctx"]["answer"]
    assert where == {"0": "no_meaningful_difference", "1": "baseline_better", "2": "paper_better"}
    assert (
        grade(
            env,
            tasks["where:ctx"],
            {"0": "no_meaningful_difference", "1": "baseline_better", "2": "paper_better"},
        )["score"]
        == 1.0
    )
    assert grade(env, tasks["claim"], "paper better")["score"] == 0.0
    best = json.loads(tasks["best"]["answer"])
    assert best == {"method": "new", "ctx": 2, "width": 64}
    assert grade(env, tasks["best"], best)["score"] == 1.0


def test_episode_never_shows_heldout_seeds(tmp_path):
    env = toy_env(tmp_path)
    ep = Episode(env, "claim", seed=3, budget_frac=1.0)
    for _ in range(6):
        ep.run([{"method": "new", "ctx": 2}], seeds=4)
    seen = {s for e in ep.log if e["action"] == "run" for _, s in e["ran"]}
    assert seen <= set(env.visible_seeds)
    assert ep.used['{"ctx":2,"method":"new","width":8}'] == 4


def test_budget_is_enforced(tmp_path):
    env = toy_env(tmp_path)
    ep = Episode(env, "claim", seed=0, budget_frac=0.1)
    out = ep.run(env.space.configs(), seeds=4)
    assert "budget exhausted" in out["notes"]
    assert ep.spent <= ep.budget + 1e-9


def test_bad_requests_return_errors(tmp_path):
    env = toy_env(tmp_path)
    ep = Episode(env, "claim", seed=0)
    assert "error" in ep.run("not json", seeds=1)
    assert "error" in ep.run([{"ctx": 9}], seeds=1)
    assert ep.calls == 0
    assert "results" in ep.run({"ctx": 1}, seeds=[1, 2])


def test_bad_submit_gets_one_retry(tmp_path):
    env = toy_env(tmp_path)
    ep = Episode(env, "claim", seed=0)
    assert "error" in ep.submit("maybe")
    assert ep.submit("no_meaningful_difference")["done"]
    assert ep.result["score"] == 1.0


def test_baselines_run(tmp_path):
    env = toy_env(tmp_path)
    for name in ("prior", "random", "uniform"):
        for task in env.tasks:
            assert 0.0 <= play(Episode(env, task["id"], seed=1), name)["score"] <= 1.0


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS only")
def test_sandbox_blocks_network_and_writes(tmp_path):
    import asyncio

    from benchtop.sandbox import run_jobs

    lab = tmp_path / "lab.py"
    target = tmp_path / "outside.txt"
    lab.write_text(
        "import socket\n"
        "def run(config, seed):\n"
        "    out = {}\n"
        "    try:\n"
        "        socket.create_connection(('1.1.1.1', 80), timeout=2); out['net'] = 1\n"
        "    except Exception:\n"
        "        out['net'] = 0\n"
        "    try:\n"
        f"        open({str(target)!r}, 'w').write('x'); out['write'] = 1\n"
        "    except Exception:\n"
        "        out['write'] = 0\n"
        "    if config.get('hang'):\n"
        "        while True: pass\n"
        "    return out\n"
    )
    res = asyncio.run(run_jobs(lab, [({}, 0), ({"hang": True}, 1), ({}, 2)], workers=1, timeout=2))
    assert res[0]["metrics"] == {"net": 0.0, "write": 0.0}
    assert not res[1]["ok"] and "timeout" in res[1]["error"]
    assert res[2]["ok"]
    assert not target.exists()
