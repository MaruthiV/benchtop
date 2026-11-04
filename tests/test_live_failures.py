import asyncio
import json

import pytest

from benchtop.agents.common import outcome
from benchtop.live import LiveStore
from benchtop.server import Hub
from benchtop.space import config_key
from tests.test_core import toy_env


class ControlledPool:
    def __init__(self, results=(), blocked=False):
        self.results = list(results)
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not blocked:
            self.release.set()
        self.closed = False

    async def run(self, config, seed):
        self.calls.append((config, seed))
        self.started.set()
        await self.release.wait()
        return self.results.pop(0) if self.results else {"ok": True, "metrics": {"acc": 0.9}, "secs": 10.0}

    async def __aexit__(self, *args):
        self.closed = True


def live_episode(tmp_path, **kw):
    env = toy_env(tmp_path)
    hub = Hub(tmp_path, live=True)
    hub.envs["toy"] = env
    episode = hub.create("toy", "claim", seed=0, **kw)
    config = env.space.configs()[0]
    for seed in env.visible_seeds:
        env.table.cells[config_key(config)].pop(seed)
    store = LiveStore(env)
    hub.stores["toy"] = store
    return hub, episode, config, store


def test_partial_failure_is_uncharged_and_retry_fills_the_same_seed_hole(tmp_path):
    async def scenario():
        hub, episode, config, store = live_episode(tmp_path)
        pool = ControlledPool([{"ok": False}])
        store.pool = pool
        first = await hub.run(episode, [config], seeds=2)
        assert [row["replicate"] for row in first["results"]] == [2]
        assert first["errors"] == [{"config": config, "replicate": 1,
                                   "error": {"code": "execution_failed", "retryable": True}}]
        assert episode.spent == 1.0
        assert episode.log[0]["status"] == "partial"
        assert len(episode.log[0]["attempted"]) == 2
        assert len(episode.log[0]["ran"]) == 1
        second = await hub.run(episode, [config])
        assert [row["replicate"] for row in second["results"]] == [1]
        assert [seed for _, seed in pool.calls] == [episode.order[0], episode.order[1], episode.order[0]]
        assert episode.calls == 2
        assert episode.spent == 2.0
        assert episode.used[config_key(config)] == 2
        await hub.submit(episode, "no_meaningful_difference")
        row = outcome(episode.summary(), "test", {})
        assert row["requested_runs"] == 3
        assert row["runs"] == 2
        assert row["charging_id"] == episode.charging_id
        assert store.stats["failed"] == 1
        assert store.stats["executed"] == 2
        assert len((episode.env.dir / "results.jsonl").read_text().splitlines()) == 194
        await hub.close()

    asyncio.run(scenario())


def test_failed_experiment_still_uses_a_call_slot(tmp_path):
    async def scenario():
        hub, episode, config, store = live_episode(tmp_path, max_calls=1)
        store.pool = ControlledPool([{"ok": False}])
        first = await hub.run(episode, [config])
        assert first["results"] == []
        assert first["errors"][0]["error"]["retryable"]
        assert episode.log[0]["status"] == "failed"
        assert episode.calls == 1
        assert episode.spent == 0
        assert episode.used == {}
        assert "error" in await hub.run(episode, [config])
        assert len(store.pool.calls) == 1
        await hub.close()

    asyncio.run(scenario())


def test_cancelled_run_keeps_shared_execution_and_retries_without_double_charging(tmp_path):
    async def scenario():
        hub, first, config, store = live_episode(tmp_path)
        second = hub.create("toy", "claim", seed=0)
        pool = ControlledPool(blocked=True)
        store.pool = pool
        owner = asyncio.create_task(hub.run(first, [config]))
        await asyncio.wait_for(pool.started.wait(), 2)
        waiter = asyncio.create_task(hub.run(second, [config]))
        for _ in range(100):
            if store.stats["coalesced"]:
                break
            await asyncio.sleep(0.001)
        assert store.stats["coalesced"] == 1
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert first.calls == 1
        assert first.spent == 0
        assert first.used == {}
        assert first.log[0]["status"] == "cancelled"
        pool.release.set()
        assert len((await waiter)["results"]) == 1
        retried = await hub.run(first, [config])
        assert retried["results"][0]["replicate"] == 1
        assert first.calls == 2
        assert first.spent == second.spent == 1
        assert len(pool.calls) == 1
        assert store.stats["executed"] == 1
        assert not store.inflight
        await hub.close()
        assert pool.closed

    asyncio.run(scenario())


def test_shutdown_cancels_owned_work_and_closes_the_pool(tmp_path):
    async def scenario():
        hub, episode, config, store = live_episode(tmp_path)
        pool = ControlledPool(blocked=True)
        store.pool = pool
        request = asyncio.create_task(hub.run(episode, [config]))
        await asyncio.wait_for(pool.started.wait(), 2)
        await hub.close()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert episode.log[0]["status"] == "cancelled"
        assert episode.spent == 0
        assert store.stats["cancelled"] == 1
        assert not store.inflight
        assert pool.closed
        assert await store.ensure([(config, episode.order[0])]) == [
            {"ok": False, "error": {"code": "store_closed", "retryable": False}},
        ]
        await hub.close()

    asyncio.run(scenario())


def test_failed_pool_startup_is_cleaned_up_and_can_be_retried(tmp_path, monkeypatch):
    instances = []

    class StartupPool(ControlledPool):
        def __init__(self, *args, **kwargs):
            super().__init__()
            instances.append(self)

        async def __aenter__(self):
            if len(instances) == 1:
                raise RuntimeError("startup failed")
            return self

    monkeypatch.setattr("benchtop.live.Pool", StartupPool)

    async def scenario():
        hub, episode, config, store = live_episode(tmp_path)
        first = await hub.run(episode, [config])
        assert first["errors"]
        assert store.pool is None
        assert instances[0].closed
        assert episode.spent == 0
        second = await hub.run(episode, [config])
        assert second["results"][0]["replicate"] == 1
        assert store.pool is instances[1]
        assert episode.spent == 1
        await hub.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("result", [
    {"ok": True, "metrics": {"other": 1}, "secs": 1},
    {"ok": True, "metrics": {"acc": float("nan")}, "secs": 1},
    {"ok": True, "metrics": {"acc": 0.9}, "secs": -1},
    {"ok": True, "metrics": {"acc": 0.9}, "secs": float("inf")},
])
def test_invalid_live_result_is_not_cached_or_charged(tmp_path, result):
    async def scenario():
        hub, episode, config, store = live_episode(tmp_path)
        original = (episode.env.dir / "results.jsonl").read_bytes()
        store.pool = ControlledPool([result])
        response = await hub.run(episode, [config])
        assert response["results"] == []
        assert response["errors"][0]["error"]["code"] == "execution_failed"
        assert episode.spent == 0
        assert episode.env.table.get(config, episode.order[0]) is None
        assert (episode.env.dir / "results.jsonl").read_bytes() == original
        await hub.close()

    asyncio.run(scenario())


def test_failed_persistence_does_not_deliver_or_charge_and_can_be_retried(tmp_path, monkeypatch):
    async def scenario():
        hub, episode, config, store = live_episode(tmp_path)
        store.pool = ControlledPool()
        results_path = episode.env.dir / "results.jsonl"
        original_open = type(results_path).open

        def unavailable(path, *args, **kwargs):
            if path == results_path:
                raise OSError("disk unavailable")
            return original_open(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(type(results_path), "open", unavailable)
            assert (await hub.run(episode, [config]))["results"] == []
        assert episode.spent == 0
        assert episode.env.table.get(config, episode.order[0]) is None
        assert (await hub.run(episode, [config]))["results"][0]["replicate"] == 1
        assert episode.spent == 1
        assert json.loads(results_path.read_text().splitlines()[-1])["seed"] == episode.order[0]
        await hub.close()

    asyncio.run(scenario())
