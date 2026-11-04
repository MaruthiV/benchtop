import asyncio
import json
import sys

import pytest

from benchtop.env import Env
from benchtop.live import LiveStore
from benchtop.server import Hub
from benchtop.tasks import build_tasks

LAB = """
import time
import numpy as np

def run(config, seed):
    time.sleep(0.05)
    rng = np.random.default_rng(seed)
    gain = 0.1 if config["method"] == "new" else 0.0
    return {"acc": 0.5 + gain * config["x"] + rng.normal(0, 0.01)}
"""

SPACE = {
    "title": "live toy",
    "metric": {"name": "acc", "goal": "max", "min_effect": 0.02},
    "factors": [
        {"name": "method", "role": "treatment", "levels": ["base", "new"]},
        {"name": "x", "role": "context", "levels": [0, 1]},
    ],
    "treatment": {"factor": "method", "paper": "new", "baseline": "base"},
}


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS only")
def test_identical_requests_share_one_run(tmp_path):
    d = tmp_path / "envs" / "toy"
    d.mkdir(parents=True)
    (d / "lab.py").write_text(LAB)
    (d / "space.json").write_text(json.dumps(SPACE))
    (d / "manifest.json").write_text(json.dumps({"id": "toy", "seeds": 16, "visible_seeds": 4}))

    async def main():
        env = Env.load(d)
        # precompute only the held-out seeds, like a grading-only sweep
        store = LiveStore(env, workers=2)
        await store.ensure([(c, s) for c in env.space.configs() for s in env.heldout_seeds])
        (d / "tasks.json").write_text(json.dumps(build_tasks(env)))
        await store.close()

        hub = Hub(tmp_path / "envs", live=True, live_workers=2)
        eps = [hub.create("toy", "claim", seed=7, budget_frac=1.0) for _ in range(20)]
        configs = [{"method": "new", "x": 1}, {"method": "base", "x": 1}]
        await asyncio.gather(*(hub.prepare(ep, configs, 2) for ep in eps))
        outs = [ep.run(configs, 2) for ep in eps]
        stats = hub.stores["toy"].stats
        await hub.stores["toy"].close()
        return outs, stats

    outs, stats = asyncio.run(main())
    assert all(len(o["results"]) == 4 for o in outs)
    assert stats["executed"] == 4
    assert stats["coalesced"] == 20 * 4 - 4
    assert len({json.dumps(o["results"]) for o in outs}) == 1
