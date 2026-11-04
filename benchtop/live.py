import asyncio
import json
import math

from benchtop.cache import CacheError, seal_result
from benchtop.env import STATE, Env
from benchtop.sandbox import Pool, sandbox_mode
from benchtop.space import config_key


class LiveStore:
    # runs missing (config, seed) pairs in the sandbox; identical concurrent requests share one run
    def __init__(self, env: Env, workers: int = 4, timeout: float = 60.0):
        sandbox_mode()
        env.prepare_live()
        self.env = env
        self.workers = workers
        self.timeout = timeout
        self.pool: Pool | None = None
        self.inflight: dict[tuple[str, int], asyncio.Task] = {}
        self.stats = {"requested": 0, "hits": 0, "coalesced": 0, "executed": 0, "failed": 0, "cancelled": 0}
        self.lock = asyncio.Lock()
        self.closed = False

    async def _pool(self) -> Pool:
        async with self.lock:
            if self.pool is None:
                candidate = Pool(
                    self.env.dir / "lab.py",
                    workers=self.workers,
                    timeout=self.timeout,
                    log_dir=STATE / "logs" / f"live-{self.env.id}",
                )
                try:
                    await candidate.__aenter__()
                except BaseException:
                    await candidate.__aexit__(None, None, None)
                    raise
                self.pool = candidate
        return self.pool

    async def ensure(self, pairs: list[tuple[dict, int]]) -> list[dict]:
        return await asyncio.gather(*(self._one(c, s) for c, s in pairs))

    async def _one(self, config: dict, seed: int) -> dict:
        self.stats["requested"] += 1
        if self.closed:
            return {"ok": False, "error": {"code": "store_closed", "retryable": False}}
        try:
            self.env.assert_current()
        except CacheError:
            return {"ok": False, "error": {"code": "stale_cache", "retryable": False}}
        if self.env.table.get(config, seed) is not None:
            self.stats["hits"] += 1
            return {"ok": True}
        key = (config_key(config), seed)
        if key in self.inflight:
            self.stats["coalesced"] += 1
            task = self.inflight[key]
        else:
            task = asyncio.create_task(self._execute(config, seed))
            self.inflight[key] = task

            def completed(done):
                if self.inflight.get(key) is done:
                    self.inflight.pop(key)

            task.add_done_callback(completed)
        return await asyncio.shield(task)

    async def _execute(self, config: dict, seed: int) -> dict:
        try:
            self.env.assert_current()
            pool = await self._pool()
            result = await pool.run(config, seed)
            if not result["ok"]:
                raise ValueError("worker failed")
            metrics = {key: float(value) for key, value in result["metrics"].items()}
            if self.env.space.metric not in metrics or not all(math.isfinite(v) for v in metrics.values()):
                raise ValueError("worker returned invalid metrics")
            if not math.isfinite(result["secs"]) or result["secs"] < 0:
                raise ValueError("worker returned invalid timing")
            self.env.assert_current()
            row = seal_result(self.env.execution, self.env.space, config, seed, {**result, "metrics": metrics})
            with (self.env.dir / "results.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            self.env.table.add(config, seed, metrics, result["secs"])
            self.env.result_written()
            self.stats["executed"] += 1
            return {"ok": True}
        except asyncio.CancelledError:
            self.stats["cancelled"] += 1
            raise
        except CacheError:
            self.stats["failed"] += 1
            return {"ok": False, "error": {"code": "stale_cache", "retryable": False}}
        except Exception:
            self.stats["failed"] += 1
            return {"ok": False, "error": {"code": "execution_failed", "retryable": True}}

    async def close(self) -> None:
        self.closed = True
        tasks = list(self.inflight.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        async with self.lock:
            if self.pool is not None:
                await self.pool.__aexit__(None, None, None)
                self.pool = None
