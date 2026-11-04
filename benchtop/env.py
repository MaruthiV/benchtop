import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np

from benchtop import cache
from benchtop.space import Space, canon, config_key
from benchtop.stats import GRADER_ID, LEGACY_GRADER_ID

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / ".state"
ENVS = ROOT / "envs"
SEEDS = 16
VISIBLE = 4


def lab_sha(lab_path: Path) -> str:
    return hashlib.sha256(Path(lab_path).read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_catalog(envs_dir: Path) -> dict:
    path = Path(envs_dir) / "catalog.json"
    catalog = json.loads(path.read_text())
    if not isinstance(catalog, dict) or type(catalog.get("schema_version")) is not int or catalog["schema_version"] != 1:
        raise ValueError(f"unsupported environment catalog schema: {path}")
    if not isinstance(catalog.get("id"), str) or not catalog["id"]:
        raise ValueError(f"environment catalog needs an id: {path}")
    if catalog.get("grader_id") not in (GRADER_ID, LEGACY_GRADER_ID):
        raise ValueError(f"environment catalog needs a supported grader_id: {path}")
    ids = catalog.get("environments")
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"environment catalog needs a nonempty environments list: {path}")
    for name in ids:
        if not isinstance(name, str) or not name or name in (".", "..") or "/" in name or "\\" in name:
            raise ValueError(f"invalid environment identity in catalog: {name!r}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate environment identity in catalog: {path}")
    funnel = catalog.get("funnel")
    counts = ("screened", "selected", "built", "lab_written", "accepted")
    if not isinstance(funnel, dict) or any(type(funnel.get(k)) is not int or funnel[k] < 0 for k in counts):
        raise ValueError(f"environment catalog needs recorded funnel counts: {path}")
    if funnel["accepted"] != len(ids):
        raise ValueError(f"catalog membership and accepted funnel count differ: {path}")
    return catalog


class Table:
    def __init__(self, space: Space, rows: list[dict]):
        self.space = space
        self.cells: dict[str, dict[int, dict]] = {}
        self.secs: dict[str, dict[int, float]] = {}
        self._cost: dict[str, float] = {}
        self._total: float | None = None
        self._charges = None
        self._charge_seeds = None
        self.charging_id: str | None = None
        for row in rows:
            if not row.get("ok"):
                continue
            key = config_key(space.normalize(row["config"]))
            self.cells.setdefault(key, {})[int(row["seed"])] = row["metrics"]
            self.secs.setdefault(key, {})[int(row["seed"])] = float(row["secs"])

    def add(self, config: dict, seed: int, metrics: dict, secs: float) -> None:
        key = config_key(config)
        self.cells.setdefault(key, {})[int(seed)] = metrics
        self.secs.setdefault(key, {})[int(seed)] = float(secs)
        self._cost.pop(key, None)
        self._total = None

    def get(self, config: dict, seed: int) -> dict | None:
        return self.cells.get(config_key(config), {}).get(seed)

    def by_seed(self, config: dict, seeds: list[int]) -> dict[int, float]:
        cell = self.cells.get(config_key(config), {})
        return {s: cell[s][self.space.metric] for s in seeds if s in cell}

    def values(self, config: dict, seeds: list[int]) -> np.ndarray:
        cell = self.cells.get(config_key(config), {})
        return np.array([cell[s][self.space.metric] for s in seeds if s in cell], dtype=float)

    def cost(self, config: dict) -> float:
        return self.cost_key(config_key(config))

    def cost_key(self, key: str) -> float:
        if key not in self._cost:
            secs = list(self.secs.get(key, {}).values())
            self._cost[key] = statistics.median(secs) if secs else 1.0
        return self._cost[key]

    def total_cost(self) -> float:
        if self._total is None:
            self._total = sum(self.cost(c) for c in self.space.configs())
        return self._total

    def freeze_costs(self, seeds: list[int] | None = None):
        source = tuple(sorted(seeds)) if seeds is not None else None
        if self._charges is not None:
            if source != self._charge_seeds:
                raise ValueError("the charging seed source is already frozen")
            return self._charges, self.charging_id
        costs = {}
        for config in self.space.configs():
            key = config_key(config)
            timings = self.secs.get(key, {})
            values = list(timings.values()) if source is None else [timings[s] for s in source if s in timings]
            cost = float(statistics.median(values)) if values else 1.0
            if not math.isfinite(cost) or cost <= 0:
                raise ValueError(f"configuration cost must be finite and positive: {key}")
            costs[key] = cost
        self._charge_seeds = source
        self.charging_id = hashlib.sha256(canon({"costs": costs, "seed_ids": source}).encode()).hexdigest()
        self._charges = MappingProxyType(costs)
        return self._charges, self.charging_id

    def missing(self, seeds: list[int]) -> list[tuple[dict, int]]:
        return [(c, s) for c in self.space.configs() for s in seeds if self.get(c, s) is None]


@dataclass
class Env:
    dir: Path
    id: str
    space: Space
    raw_space: dict
    table: Table
    manifest: dict
    tasks: list[dict]
    execution: dict
    cache_status: str
    task_version: str = ""
    _tracked: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, env_dir: Path, *, replay: bool = False, load_tasks: bool = True) -> "Env":
        env_dir = Path(env_dir)
        raw = json.loads((env_dir / "space.json").read_text())
        space = Space.from_dict(raw)
        manifest_path = env_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        tasks_path = env_dir / "tasks.json"
        stored = json.loads(tasks_path.read_text()) if tasks_path.exists() and load_tasks else []
        artifact_grader = None
        if isinstance(stored, dict):
            if type(stored.get("schema_version")) is not int or stored["schema_version"] != 1 or stored.get("grader_id") != GRADER_ID:
                raise ValueError(f"unsupported task artifact: {tasks_path}")
            artifact_grader = stored["grader_id"]
            tasks = stored["tasks"]
            if stored.get("status") != "complete" and tasks:
                raise ValueError(f"incomplete task artifact contains eligible tasks: {tasks_path}")
        else:
            tasks = stored
        if not isinstance(tasks, list):
            raise ValueError(f"tasks must be a list: {tasks_path}")
        versions = {task.get("grader_id", artifact_grader or LEGACY_GRADER_ID) for task in tasks}
        if len(versions) > 1:
            raise ValueError(f"mixed grader versions in tasks: {tasks_path}")
        grader = next(iter(versions), artifact_grader or manifest.get("grader_id", LEGACY_GRADER_ID))
        if grader not in (GRADER_ID, LEGACY_GRADER_ID):
            raise ValueError(f"unknown task grader: {grader}")
        if any(v is not None and v != grader for v in (artifact_grader, manifest.get("grader_id"))):
            raise ValueError(f"task and manifest grader versions differ: {tasks_path}")
        manifest = {**manifest, "grader_id": grader}
        tasks = [{**task, "grader_id": grader} for task in tasks]
        rows = read_jsonl(env_dir / "results.jsonl")
        execution, status = cache.load_binding(env_dir, manifest, space, rows, replay=replay)
        table = Table(space, rows)
        env = cls(env_dir, manifest.get("id", env_dir.name), space, raw, table, manifest, tasks, execution, status)
        if grader == GRADER_ID and load_tasks:
            if isinstance(stored, dict) and not cache.valid_checksum(stored, "artifact_sha256"):
                raise cache.CacheError("task cache artifact checksum is invalid")
            if any(not cache.valid_checksum(task, "task_sha256") for task in tasks):
                raise cache.CacheError("task cache payload checksum is invalid")
            bindings = [task.get("binding") for task in tasks]
            if isinstance(stored, dict):
                bindings.append(stored.get("binding"))
            for binding in bindings:
                if not isinstance(binding, dict) or binding != cache.task_binding(env, binding.get("seed_ids", [])):
                    raise cache.CacheError("task cache inputs or generator changed; regenerate tasks from validated results")
            if any(task.get("seed_ids") != task["binding"]["seed_ids"] for task in tasks):
                raise cache.CacheError("task cache seed sources differ")
        env.task_version = cache.digest({"execution_id": env.execution_id, "grader_id": grader, "artifact": stored})
        tracked = [env_dir / name for name in (*cache.DATA_FILES, "cache.json")]
        tracked.append(env_dir.parent / "historical_cache.json")
        if status in ("current", "empty"):
            tracked.extend(cache.CODE / name for name in (*cache.ENGINE_FILES, "tasks.py", "stats.py"))
        env._tracked = {path: cache.file_signature(path) for path in tracked}
        return env

    @property
    def execution_id(self):
        return self.execution["id"]

    def assert_current(self):
        if any(cache.file_signature(path) != signature for path, signature in self._tracked.items()):
            raise cache.CacheError("environment cache files changed; reload or rebuild before continuing")
        if self.cache_status in ("current", "empty"):
            runtime = json.loads(json.dumps(cache.runtime_identity()))
            if self.execution["inputs"]["runtime"] != runtime:
                raise cache.CacheError("execution cache runtime changed; restart and rebuild")

    def prepare_live(self):
        self.assert_current()
        if self.cache_status in ("historical", "replay"):
            raise cache.CacheError("historical or replay cache is read-only; rebuild in a new directory for live execution")
        if self.cache_status == "empty":
            execution = cache.initialize_cache(self.dir, self.manifest)
            if execution != self.execution:
                raise cache.CacheError("execution cache inputs changed")
            self.cache_status = "current"
            self._tracked[self.dir / "cache.json"] = cache.file_signature(self.dir / "cache.json")

    def result_written(self):
        self._tracked[self.dir / "results.jsonl"] = cache.file_signature(self.dir / "results.jsonl")

    @property
    def seeds(self) -> int:
        return int(self.manifest.get("seeds", SEEDS))

    @property
    def grader_id(self) -> str:
        return self.manifest["grader_id"]

    def charging_schedule(self):
        seeds = None
        if self.grader_id == GRADER_ID:
            seeds = self.tasks[0].get("seed_ids", self.heldout_seeds) if self.tasks else self.heldout_seeds
        return self.table.freeze_costs(seeds)

    @property
    def visible_seeds(self) -> list[int]:
        return list(range(int(self.manifest.get("visible_seeds", VISIBLE))))

    @property
    def heldout_seeds(self) -> list[int]:
        return list(range(len(self.visible_seeds), self.seeds))

    @property
    def paper(self) -> dict:
        return self.raw_space.get("paper", {})

    def task(self, task_id: str) -> dict:
        for t in self.tasks:
            if t["id"] == task_id:
                return t
        raise KeyError(f"no task {task_id!r} in env {self.id}")
