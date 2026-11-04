import asyncio
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from benchtop.cache import (
    CODE,
    ENGINE_FILES,
    CacheError,
    file_signature,
    initialize_cache,
    load_binding,
    seal_result,
)
from benchtop.env import SEEDS, STATE, VISIBLE, Env, lab_sha, read_jsonl
from benchtop.sandbox import run_jobs, sandbox_mode
from benchtop.space import Space, config_key
from benchtop.stats import GRADER_ID
from benchtop.tasks import build_task_set


def sweep(
    env_dir: Path,
    seeds: int = SEEDS,
    workers: int = 8,
    timeout: float = 60.0,
    sandbox: str = "auto",
    log=print,
) -> dict:
    env_dir = Path(env_dir)
    space = Space.load(env_dir / "space.json")
    sha = lab_sha(env_dir / "lab.py")
    results_path = env_dir / "results.jsonl"
    manifest_path = env_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    manifest = {**manifest, "seeds": seeds, "visible_seeds": manifest.get("visible_seeds", VISIBLE)}
    original_rows = read_jsonl(results_path)
    binding, status = load_binding(env_dir, manifest, space, original_rows)
    if status == "historical":
        raise CacheError("historical cache is read-only; rebuild in a new directory")
    done = {
        (config_key(space.normalize(r["config"])), r["seed"]) for r in original_rows if r.get("ok")
    }
    jobs = [(c, s) for c in space.configs() for s in range(seeds) if (config_key(c), s) not in done]
    if jobs:
        sandbox_mode(sandbox)
    if status == "empty":
        binding = initialize_cache(env_dir, manifest)

    manifest.update(
        {
            "id": manifest.get("id", env_dir.name),
            "lab_sha256": sha,
            "seeds": seeds,
            "visible_seeds": manifest.get("visible_seeds", VISIBLE),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    inputs = [env_dir / name for name in ("lab.py", "space.json", "manifest.json", "cache.json")]
    inputs.extend(CODE / name for name in (*ENGINE_FILES, "tasks.py", "stats.py"))
    signatures = {path: file_signature(path) for path in inputs}

    def check_inputs():
        if any(file_signature(path) != signature for path, signature in signatures.items()):
            raise CacheError("execution cache inputs changed during sweep")

    started = time.perf_counter()
    errors = Counter()
    out = results_path.open("a")

    def record(config, seed, result):
        check_inputs()
        result = {key: value for key, value in result.items() if key != "trace"}
        try:
            row = seal_result(binding, space, config, seed, result)
        except (KeyError, TypeError, ValueError):
            row = seal_result(binding, space, config, seed, {"ok": False, "error": "invalid worker result", "secs": 0.0})
        out.write(json.dumps(row, allow_nan=False) + "\n")
        out.flush()
        if not row["ok"]:
            errors[row["error"][:160]] += 1

    log(f"sweep {env_dir.name}: {len(jobs)} runs to go ({space.size()} configs x {seeds} seeds)")
    try:
        if jobs:
            asyncio.run(
                run_jobs(
                    env_dir / "lab.py",
                    jobs,
                    workers,
                    timeout,
                    sandbox,
                    log_dir=STATE / "logs" / env_dir.name,
                    on_result=record,
                )
            )
    finally:
        out.close()
    wall = time.perf_counter() - started
    check_inputs()

    # failed rows are retried on the next sweep, so keep only the latest attempt per job
    rows = {}
    for r in read_jsonl(results_path):
        rows[(config_key(space.normalize(r["config"])), r["seed"])] = r
    results_path.write_text("".join(json.dumps(r) + "\n" for r in rows.values()))

    env = Env.load(env_dir, load_tasks=False)
    task_set = build_task_set(env)
    tasks = task_set["tasks"]
    (env_dir / "tasks.json").write_text(json.dumps(task_set, indent=2) + "\n")
    ok = sum(1 for r in rows.values() if r.get("ok"))
    summary = {
        "env": env_dir.name,
        "runs_ok": ok,
        "runs_failed": len(rows) - ok,
        "wall_secs": round(wall, 1),
        "cpu_secs": round(sum(r.get("secs", 0) for r in rows.values()), 1),
        "tasks": [t["id"] for t in tasks],
        "task_status": task_set["status"],
        "grader_id": GRADER_ID,
        "errors": dict(errors.most_common(5)),
        "swept_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    manifest["sweep"] = summary
    manifest["grader_id"] = GRADER_ID
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"sweep {env_dir.name}: {ok} ok, {len(rows) - ok} failed, {wall:.0f}s wall, tasks {summary['tasks']}")
    return summary
