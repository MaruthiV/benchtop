import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchtop.space import Space, config_key

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


class Table:
    def __init__(self, space: Space, rows: list[dict]):
        self.space = space
        self.cells: dict[str, dict[int, dict]] = {}
        self.secs: dict[str, list[float]] = {}
        self._cost: dict[str, float] = {}
        self._total: float | None = None
        for row in rows:
            if not row.get("ok"):
                continue
            key = config_key(space.normalize(row["config"]))
            self.cells.setdefault(key, {})[int(row["seed"])] = row["metrics"]
            self.secs.setdefault(key, []).append(float(row["secs"]))

    def add(self, config: dict, seed: int, metrics: dict, secs: float) -> None:
        key = config_key(config)
        self.cells.setdefault(key, {})[int(seed)] = metrics
        self.secs.setdefault(key, []).append(float(secs))
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
            secs = self.secs.get(key)
            self._cost[key] = statistics.median(secs) if secs else 1.0
        return self._cost[key]

    def total_cost(self) -> float:
        if self._total is None:
            self._total = sum(self.cost(c) for c in self.space.configs())
        return self._total

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

    @classmethod
    def load(cls, env_dir: Path) -> "Env":
        env_dir = Path(env_dir)
        raw = json.loads((env_dir / "space.json").read_text())
        space = Space.from_dict(raw)
        manifest_path = env_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        tasks_path = env_dir / "tasks.json"
        tasks = json.loads(tasks_path.read_text()) if tasks_path.exists() else []
        table = Table(space, read_jsonl(env_dir / "results.jsonl"))
        return cls(env_dir, manifest.get("id", env_dir.name), space, raw, table, manifest, tasks)

    @property
    def seeds(self) -> int:
        return int(self.manifest.get("seeds", SEEDS))

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
