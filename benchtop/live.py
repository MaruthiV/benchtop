import asyncio
import json

from benchtop.env import STATE, Env
from benchtop.sandbox import Pool
from benchtop.space import config_key


class LiveStore:
    # runs missing (config, seed) pairs in the sandbox; identical concurrent requests share one run
    def __init__(self, env: Env, workers: int = 4, timeout: float = 60.0):
        self.env = env
        self.workers = workers
        self.timeout = timeout
        self.pool: Pool | None = None
        self.inflight: dict[tuple[str, int], asyncio.Future] = {}
        self.stats = {"requested": 0, "hits": 0, "coalesced": 0, "executed": 0, "failed": 0}
        self.lock = asyncio.Lock()

    async def _pool(self) -> Pool:
        async with self.lock:
            if self.pool is None:
                self.pool = Pool(
                    self.env.dir / "lab.py",
                    workers=self.workers,
                    timeout=self.timeout,
                    log_dir=STATE / "logs" / f"live-{self.env.id}",
                )
                await self.pool.__aenter__()
        return self.pool

    async def ensure(self, pairs: list[tuple[dict, int]]) -> None:
        await asyncio.gather(*(self._one(c, s) for c, s in pairs))

    async def _one(self, config: dict, seed: int) -> None:
        self.stats["requested"] += 1
        if self.env.table.get(config, seed) is not None:
            self.stats["hits"] += 1
            return
        key = (config_key(config), seed)
        if key in self.inflight:
            self.stats["coalesced"] += 1
            await self.inflight[key]
            return
        fut = asyncio.get_running_loop().create_future()
        self.inflight[key] = fut
        try:
            pool = await self._pool()
            result = await pool.run(config, seed)
            if result["ok"]:
                self.env.table.add(config, seed, result["metrics"], result["secs"])
                with (self.env.dir / "results.jsonl").open("a") as f:
                    f.write(json.dumps({"config": config, "seed": seed, **result}) + "\n")
                self.stats["executed"] += 1
            else:
                self.stats["failed"] += 1
        finally:
            fut.set_result(None)
            del self.inflight[key]

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.__aexit__(None, None, None)
            self.pool = None
