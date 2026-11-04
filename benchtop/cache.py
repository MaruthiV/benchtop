"""Execution provenance for current caches and immutable historical replay."""

import hashlib
import json
import math
import platform
import re
import sys
from functools import lru_cache
from importlib.metadata import distributions
from pathlib import Path

from benchtop.space import config_key

CODE = Path(__file__).resolve().parent
ENGINE_FILES = ("_worker.py", "sandbox.py", "space.py", "cache.py")
DATA_FILES = ("lab.py", "space.json", "manifest.json", "results.jsonl", "tasks.json")


class CacheError(ValueError):
    pass


def digest(value, *, allow_nan=False):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=allow_nan).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def with_checksum(value, field):
    return {**value, field: digest({key: item for key, item in value.items() if key != field})}


def valid_checksum(value, field):
    return value.get(field) == with_checksum(value, field)[field]


@lru_cache(maxsize=1)
def runtime_identity():
    # The interpreter and installed environment are fixed for this process. Restart after installing packages.
    packages = sorted(
        (re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower(), d.version)
        for d in distributions() if d.metadata["Name"]
    )
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "packages": packages,
    }


def seed_partition(manifest):
    total, visible = manifest.get("seeds", 16), manifest.get("visible_seeds", 4)
    if type(total) is not int or type(visible) is not int or not 1 <= visible < total:
        raise CacheError("invalid cache seed partition")
    # Preserve additional split definitions when the fresh-cohort harness supplies them.
    return {"seeds": total, "visible_seeds": visible, "seed_splits": manifest.get("seed_splits")}


def execution_binding(env_dir, manifest):
    inputs = {
        "lab_sha256": file_hash(env_dir / "lab.py"),
        "space_sha256": file_hash(env_dir / "space.json"),
        "seed_partition": seed_partition(manifest),
        "runtime": runtime_identity(),
        "engine_sha256": {name: file_hash(CODE / name) for name in ENGINE_FILES},
    }
    # Roundtrip makes the stored binding independent of mutable memoized runtime data and tuple encoding.
    inputs = json.loads(json.dumps(inputs))
    return {"schema_version": 1, "id": digest(inputs), "inputs": inputs}


def initialize_cache(env_dir, manifest):
    """Initialize an empty environment; never relabel or overwrite existing observations."""
    path = env_dir / "cache.json"
    if path.exists():
        raise CacheError("cache already exists; validate or use a new directory")
    if any((env_dir / name).exists() and (env_dir / name).stat().st_size for name in ("results.jsonl", "tasks.json")):
        raise CacheError("unversioned cache is read-only; rebuild in a new directory")
    binding = execution_binding(env_dir, manifest)
    with path.open("x") as out:
        out.write(json.dumps(binding, indent=2, allow_nan=False) + "\n")
    return binding


def seal_result(binding, space, config, seed, result):
    if type(seed) is not int or not 0 <= seed < binding["inputs"]["seed_partition"]["seeds"]:
        raise CacheError("cache result seed is outside its bound partition")
    if type(result.get("ok")) is not bool:
        raise CacheError("cache result needs a boolean success status")
    row = {k: v for k, v in result.items() if k not in ("config", "seed", "execution_id", "row_sha256")}
    secs = float(row.get("secs", 0))
    if not math.isfinite(secs) or secs < 0:
        raise CacheError("cache result has invalid timing")
    row["secs"] = secs
    if row["ok"]:
        metrics = {key: float(value) for key, value in row["metrics"].items()}
        if space.metric not in metrics or not all(math.isfinite(v) for v in metrics.values()):
            raise CacheError("cache result has invalid metrics")
        row["metrics"] = metrics
    row.update(config=space.normalize(config), seed=seed, execution_id=binding["id"])
    row["row_sha256"] = digest(row)
    return row


def historical_binding(env_dir):
    path = env_dir.parent / "historical_cache.json"
    if not path.exists():
        raise CacheError("unversioned cache lacks historical provenance; rebuild in a new directory")
    index = json.loads(path.read_text())
    if type(index.get("schema_version")) is not int or index["schema_version"] != 1:
        raise CacheError("unsupported historical cache index")
    expected = index.get("environments", {}).get(env_dir.name)
    if not isinstance(expected, dict) or set(expected) != set(DATA_FILES):
        raise CacheError("unversioned cache is not a recorded historical environment")
    if any(not (env_dir / name).is_file() or file_hash(env_dir / name) != value for name, value in expected.items()):
        raise CacheError("historical cache inputs changed")
    return {"schema_version": 0, "id": "historical_v0:" + digest(expected), "inputs": expected}


def load_binding(env_dir, manifest, space, rows, *, replay=False):
    path = env_dir / "cache.json"
    if not path.exists():
        if rows or (env_dir / "tasks.json").exists():
            return historical_binding(env_dir), "historical"
        return execution_binding(env_dir, manifest), "empty"
    binding = json.loads(path.read_text())
    if type(binding.get("schema_version")) is not int or binding["schema_version"] != 1:
        raise CacheError("unsupported execution cache schema")
    if binding.get("id") != digest(binding.get("inputs")):
        raise CacheError("execution cache identity is corrupt")
    current = execution_binding(env_dir, manifest)
    if replay:
        # Replay keeps recorded execution provenance. It cannot authorize further execution.
        for key in ("runtime", "engine_sha256"):
            current["inputs"][key] = binding["inputs"][key]
        current["id"] = digest(current["inputs"])
    if binding != current:
        raise CacheError("execution cache inputs changed; rebuild in a new directory")
    for row in rows:
        if row.get("execution_id") != binding["id"]:
            raise CacheError("cache result belongs to a different execution")
        try:
            sealed = seal_result(binding, space, row["config"], row["seed"], row)
        except (KeyError, TypeError, ValueError) as error:
            raise CacheError("cache result is invalid") from error
        if sealed != row:
            raise CacheError("cache result checksum or configuration is invalid")
    return binding, "replay" if replay else "current"


def task_binding(env, seeds):
    rows = [
        [config_key(config), seed, env.table.get(config, seed)]
        for config in env.space.configs() for seed in seeds
    ]
    return {
        "execution_id": env.execution_id,
        "qualification_sha256": digest(rows, allow_nan=True),
        "generator_sha256": {name: file_hash(CODE / name) for name in ("tasks.py", "stats.py", "space.py", "cache.py")},
        "seed_ids": list(seeds),
    }


def file_signature(path):
    try:
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    except FileNotFoundError:
        return None
