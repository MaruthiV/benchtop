import argparse
import asyncio
import json
import multiprocessing as mp
import random
import resource
import time

import httpx
import numpy as np

from benchtop.env import ENVS, Env
from benchtop.episode import Episode


def env_tasks() -> list[tuple[str, str, list[dict]]]:
    out = []
    for p in sorted(ENVS.glob("*/tasks.json")):
        env = Env.load(p.parent)
        configs = env.space.configs()
        out += [(env.id, t["id"], configs) for t in env.tasks]
    return out


class RawPool:
    # tiny keep-alive http/1.1 client; httpx costs more cpu than the server it's testing
    def __init__(self, base: str, size: int = 256):
        hostport = base.split("//", 1)[1]
        self.host, port = hostport.split(":")
        self.port = int(port)
        self.size = size
        self.free: asyncio.Queue = asyncio.Queue()
        self.opened = 0

    async def _conn(self):
        if self.free.empty() and self.opened < self.size:
            self.opened += 1
            return await asyncio.open_connection(self.host, self.port)
        return await self.free.get()

    async def request(self, method: str, url: str, json_body=None, **_):
        try:
            return await asyncio.wait_for(self._request(method, url, json_body), 30)
        except (asyncio.IncompleteReadError, ConnectionError):
            # an idle keep-alive socket the server already closed; retry once on a fresh one
            return await asyncio.wait_for(self._request(method, url, json_body), 30)

    async def _request(self, method: str, url: str, json_body=None):
        body = json.dumps(json_body).encode() if json_body is not None else b""
        reader, writer = await self._conn()
        try:
            writer.write(
                f"{method} {url} HTTP/1.1\r\nHost: {self.host}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n\r\n".encode()
                + body
            )
            await writer.drain()
            head = await reader.readuntil(b"\r\n\r\n")
            status = int(head.split(b" ", 2)[1])
            length = 0
            for line in head.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    length = int(line.split(b":")[1])
            data = await reader.readexactly(length)
        except BaseException:
            # timeouts land here as cancellation, so the socket never goes back in the pool
            writer.close()
            self.opened -= 1
            raise
        self.free.put_nowait((reader, writer))
        return _Resp(status, data)

    async def aclose(self):
        while not self.free.empty():
            _, w = self.free.get_nowait()
            w.close()


class _Resp:
    def __init__(self, status: int, data: bytes):
        self.status_code, self.data = status, data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return json.loads(self.data)


async def agent(http, work, rng: random.Random, steps: int, think: float, lat: list):
    env, task, configs = rng.choice(work)

    async def call(method, url, json=None):
        t = time.perf_counter()
        r = (
            await http.request(method, url, json_body=json)
            if isinstance(http, RawPool)
            else await http.request(method, url, json=json)
        )
        lat.append(time.perf_counter() - t)
        r.raise_for_status()
        return r.json()

    ep = await call("POST", "/episodes", json={"env": env, "task": task, "seed": rng.randrange(1 << 30)})
    for _ in range(steps):
        await asyncio.sleep(rng.expovariate(1 / think) if think else 0)
        batch = [rng.choice(configs) for _ in range(rng.randint(1, 4))]
        await call("POST", f"/episodes/{ep['id']}/run", json={"configs": batch, "seeds": rng.randint(1, 2)})
    await call("POST", f"/episodes/{ep['id']}/submit", json={"answer": "paper_better", "confidence": 0.5})


async def drive(bases: list[str], agents: int, steps: int, think: float, ramp: float, seed: int) -> dict:
    work = env_tasks()
    rng = random.Random(seed)
    lat: list[float] = []
    errors = 0
    kinds: dict[str, int] = {}
    clients = [RawPool(b) for b in bases]

    async def one(i):
        nonlocal errors
        await asyncio.sleep(ramp * i / max(1, agents))
        try:
            # episodes are independent, so shard them across server processes
            await agent(clients[i % len(clients)], work, random.Random(rng.random()), steps, think, lat)
        except Exception as e:
            errors += 1
            kinds[type(e).__name__] = kinds.get(type(e).__name__, 0) + 1

    started = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(agents)))
    wall = time.perf_counter() - started
    for c in clients:
        await c.aclose()
    return {"requests": len(lat), "errors": errors, "wall": wall, "lat": lat, "kinds": kinds}


def _proc(args):
    bases, agents, steps, think, ramp, seed = args
    return asyncio.run(drive(bases, agents, steps, think, ramp, seed))


def http_load(bases: list[str], agents: int, steps: int, think: float, procs: int, ramp: float) -> dict:
    per = [agents // procs + (i < agents % procs) for i in range(procs)]
    with mp.get_context("spawn").Pool(procs) as pool:
        parts = pool.map(_proc, [(bases, n, steps, think, ramp, i) for i, n in enumerate(per)])
    lat = np.array([x for p in parts for x in p["lat"]])
    wall = max(p["wall"] for p in parts)
    on_server = sum(httpx.get(f"{b}/health").json()["episodes"] for b in bases)
    return {
        "agents": agents,
        "servers": len(bases),
        "steps_per_agent": steps,
        "think_mean_s": think,
        "client_procs": procs,
        "requests": int(len(lat)),
        "errors": sum(p["errors"] for p in parts),
        "wall_s": round(wall, 1),
        "req_per_s": round(len(lat) / wall, 1),
        "p50_ms": round(1000 * float(np.percentile(lat, 50)), 2),
        "p99_ms": round(1000 * float(np.percentile(lat, 99)), 2),
        "max_ms": round(1000 * float(lat.max()), 1),
        "episodes_on_servers": on_server,
        "error_kinds": {k: sum(p["kinds"].get(k, 0) for p in parts) for p in parts for k in p["kinds"]},
    }


def inprocess(episodes: int, steps: int, seed: int = 0) -> dict:
    work = env_tasks()
    envs = {e: Env.load(ENVS / e) for e in {w[0] for w in work}}
    rng = random.Random(seed)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t = time.perf_counter()
    eps = []
    for _ in range(episodes):
        env, task, _ = rng.choice(work)
        eps.append(Episode(envs[env], task, seed=rng.randrange(1 << 30)))
    made = time.perf_counter() - t
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t = time.perf_counter()
    n = 0
    for _ in range(steps):
        for ep in eps:
            configs = ep.env.space.configs()
            ep.run([rng.choice(configs)], seeds=1)
            n += 1
    stepped = time.perf_counter() - t
    # ru_maxrss is bytes on macos
    return {
        "episodes": episodes,
        "create_per_s": round(episodes / made),
        "steps": n,
        "steps_per_s": round(n / stepped),
        "rss_growth_mb": round((after - before) / 1e6, 1),
        "kb_per_episode": round((after - before) / 1e3 / episodes, 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["http", "inprocess"])
    p.add_argument("--bases", default="http://127.0.0.1:8787", help="comma-separated server urls")
    p.add_argument("--agents", type=int, default=1000)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--think", type=float, default=2.0)
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--ramp", type=float, default=5.0)
    p.add_argument("--episodes", type=int, default=10000)
    a = p.parse_args()
    if a.mode == "http":
        out = http_load(a.bases.split(","), a.agents, a.steps, a.think, a.procs, a.ramp)
    else:
        out = inprocess(a.episodes, a.steps)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
