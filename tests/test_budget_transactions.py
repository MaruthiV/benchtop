import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest

from benchtop.episode import Episode
from benchtop.server import Hub, build_admin_app, build_app
from benchtop.space import config_key
from tests.test_core import toy_env
from tests.test_server import MCP_HEADERS


@pytest.fixture(autouse=True)
def simulated_backend_availability(monkeypatch):
    # These tests substitute stores; sandbox availability is tested with real workers elsewhere.
    monkeypatch.setattr("benchtop.server.sandbox_mode", lambda: "macos")
    monkeypatch.setattr("benchtop.live.sandbox_mode", lambda: "macos")


@asynccontextmanager
async def clients(hub):
    app, _ = build_app(hub)
    async with (
        app.app.router.lifespan_context(app.app),
        httpx.AsyncClient(transport=httpx.ASGITransport(build_admin_app(hub)), base_url="http://127.0.0.1:8787",
                          headers={"authorization": f"Bearer {hub.admin_token}"}) as control,
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://127.0.0.1:8788") as agent,
    ):
        yield control, agent


class HeldStore:
    def __init__(self, env, config):
        self.env = env
        key = config_key(config)
        self.metrics = {seed: env.table.cells[key].pop(seed) for seed in env.visible_seeds}
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.requested = []

    async def ensure(self, pairs):
        self.requested.extend(seed for _, seed in pairs)
        self.started.set()
        await self.release.wait()
        for config, seed in pairs:
            self.env.table.add(config, seed, self.metrics[seed], 1.0)
        return [{"ok": True} for _ in pairs]


def test_live_runtime_updates_cannot_reprice_an_episode(tmp_path):
    env = toy_env(tmp_path)
    config = env.space.configs()[0]
    first = Episode(env, "claim", budget_frac=0.1, seed=0)
    for seed in range(env.seeds):
        env.table.add(config, seed, dict(env.table.get(config, seed)), 100.0)
    result = first.run([config])
    assert len(result["results"]) == 1
    assert result["results"][0]["cost"] == 1.0
    second = Episode(env, "claim", budget_frac=0.1, seed=0)
    assert second.run([config])["results"][0]["cost"] == 1.0
    assert second.costs is first.costs
    assert second.charging_id == first.charging_id
    with pytest.raises(TypeError):
        first.costs[config_key(config)] = 99


def test_corrected_prices_use_only_qualification_timings(tmp_path):
    env = toy_env(tmp_path)
    config = env.space.configs()[0]
    # The qualification subset is explicit; neither visible nor future seeds price the lab.
    for task in env.tasks:
        task["seed_ids"] = [4, 5]
    for seed in range(env.seeds):
        env.table.add(config, seed, dict(env.table.get(config, seed)), 2.0 if seed in (4, 5) else 999.0)
    episode = Episode(env, "claim")
    assert episode.costs[config_key(config)] == 2.0
    assert episode.describe()["pricing"] == {
        "schedule_id": episode.charging_id, "unit": "compute_seconds", "by_config": dict(episode.costs),
    }


@pytest.mark.parametrize("budget", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_budgets_are_rejected(tmp_path, budget):
    with pytest.raises(ValueError, match="budget_frac"):
        Episode(toy_env(tmp_path), "claim", budget_frac=budget)


def test_zero_budget_cannot_use_rounding_to_run_a_tiny_cost(tmp_path):
    env = toy_env(tmp_path)
    config = env.space.configs()[0]
    for seed in env.heldout_seeds:
        env.table.add(config, seed, dict(env.table.get(config, seed)), 1e-14)
    episode = Episode(env, "claim", budget_frac=0)
    assert episode.run([config])["results"] == []
    assert episode.spent == 0


def test_submit_waits_for_an_inflight_run_on_the_other_interface(tmp_path):
    env = toy_env(tmp_path)
    config = env.space.configs()[0]

    async def scenario():
        hub = Hub(tmp_path, live=True)
        hub.envs["toy"] = env
        store = HeldStore(env, config)
        hub.stores["toy"] = store
        async with clients(hub) as (control, agent):
            created = (await control.post("/episodes", json={"env": "toy", "task": "claim", "seed": 0})).json()
            run = asyncio.create_task(control.post(f"/episodes/{created['id']}/run", json={"configs": [config]}))
            await asyncio.wait_for(store.started.wait(), 2)
            submit = asyncio.create_task(agent.post(
                created["mcp_url"], headers=MCP_HEADERS,
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                      "params": {"name": "submit_answer", "arguments": {"answer": "no_meaningful_difference"}}},
            ))
            await asyncio.sleep(0.02)
            waiting = not submit.done()
            store.release.set()
            ran, submitted = await asyncio.gather(run, submit)
            saved = (await control.get(f"/episodes/{created['id']}")).json()
            return waiting, ran.json(), submitted.json(), saved

    waiting, ran, submitted, saved = asyncio.run(scenario())
    assert waiting
    assert len(ran["results"]) == 1
    assert json.loads(submitted["result"]["content"][0]["text"]) == {"received": True}
    assert saved["spent"] == 1.0
    assert saved["result"]["calls"] == 1


def test_overlapping_runs_prepare_different_fresh_seeds(tmp_path):
    env = toy_env(tmp_path)
    config = env.space.configs()[0]

    async def scenario():
        hub = Hub(tmp_path, live=True)
        hub.envs["toy"] = env
        store = HeldStore(env, config)
        hub.stores["toy"] = store
        async with clients(hub) as (control, _):
            created = (await control.post("/episodes", json={"env": "toy", "task": "claim", "seed": 0, "budget_frac": 0.1})).json()
            path = f"/episodes/{created['id']}/run"
            first = asyncio.create_task(control.post(path, json={"configs": [config]}))
            await asyncio.wait_for(store.started.wait(), 2)
            second = asyncio.create_task(control.post(path, json={"configs": [config]}))
            await asyncio.sleep(0.02)
            store.release.set()
            responses = await asyncio.gather(first, second)
            return [response.json() for response in responses], store.requested, hub.get(created["id"])

    responses, requested, episode = asyncio.run(scenario())
    assert [len(response["results"]) for response in responses] == [1, 1]
    assert len(set(requested)) == 2
    assert episode.spent == 2.0
    assert episode.calls == 2


def test_concurrent_calls_cannot_spend_the_same_remaining_budget(tmp_path):
    env = toy_env(tmp_path)
    config = env.space.configs()[0]

    async def scenario():
        hub = Hub(tmp_path, live=True)
        hub.envs["toy"] = env
        episode = hub.create("toy", "claim", budget_frac=1 / 48, seed=0)
        store = HeldStore(env, config)
        hub.stores["toy"] = store
        first = asyncio.create_task(hub.run(episode, [config]))
        await asyncio.wait_for(store.started.wait(), 2)
        second = asyncio.create_task(hub.run(episode, [config]))
        await asyncio.sleep(0)
        store.release.set()
        return episode, store, await asyncio.gather(first, second)

    episode, store, responses = asyncio.run(scenario())
    assert [len(response["results"]) for response in responses] == [1, 0]
    assert len(store.requested) == 1
    assert episode.spent == episode.budget == 1.0


@pytest.mark.parametrize("first_interface", ["admin", "mcp"])
def test_simultaneous_submissions_keep_the_first_committed_answer(tmp_path, first_interface):
    env = toy_env(tmp_path)

    async def scenario():
        hub = Hub(tmp_path)
        hub.envs["toy"] = env
        episode = hub.create("toy", "claim", seed=0)
        async with clients(hub) as (control, agent):
            async def submit(interface, answer, confidence):
                if interface == "admin":
                    response = await control.post(f"/episodes/{episode.id}/submit", json={
                        "answer": answer, "confidence": confidence,
                    })
                    return response.json()
                response = await agent.post(hub.agent_url(episode), headers=MCP_HEADERS, json={
                    "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
                        "name": "submit_answer", "arguments": {"answer": answer, "confidence": confidence},
                    },
                })
                return json.loads(response.json()["result"]["content"][0]["text"])

            # Hold the shared lock until both transports are queued, then commit in queue order.
            lock = hub.locks[episode.id]
            await lock.acquire()
            first = asyncio.create_task(submit(first_interface, "no_meaningful_difference", 0.8))
            for _ in range(100):
                if lock._waiters:
                    break
                await asyncio.sleep(0.001)
            assert lock._waiters
            second_interface = "mcp" if first_interface == "admin" else "admin"
            second = asyncio.create_task(submit(second_interface, "paper_better", 0.1))
            for _ in range(100):
                if len(lock._waiters) == 2:
                    break
                await asyncio.sleep(0.001)
            queued = len(lock._waiters)
            lock.release()
            replies = await asyncio.gather(first, second)
            return episode, queued, replies

    episode, queued, replies = asyncio.run(scenario())
    assert queued == 2
    assert all(reply in ({"done": True}, {"received": True}) for reply in replies)
    assert episode.result["score"] == 1
    assert episode.result["confidence"] == 0.8
    assert [entry["answer"] for entry in episode.log] == ["no_meaningful_difference"]


def test_done_call_limited_and_invalid_requests_never_prepare_work(tmp_path):
    env = toy_env(tmp_path)
    config = env.space.configs()[0]

    async def scenario():
        hub = Hub(tmp_path, live=True)
        hub.envs["toy"] = env
        episode = hub.create("toy", "claim", max_calls=1)
        assert "error" in await hub.run(episode, [{"ctx": 99}])
        assert episode.calls == 0
        assert not hub.stores
        await hub.run(episode, [config])
        store = hub.stores["toy"]
        requested = store.stats["requested"]
        assert "error" in await hub.run(episode, [config])
        await hub.submit(episode, "no_meaningful_difference")
        assert "error" in await hub.run(episode, [config])
        assert store.stats["requested"] == requested
        await hub.close()

    asyncio.run(scenario())
