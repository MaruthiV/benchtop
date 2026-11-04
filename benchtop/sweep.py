import asyncio
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from benchtop.env import SEEDS, STATE, VISIBLE, Env, lab_sha, read_jsonl
from benchtop.sandbox import run_jobs
from benchtop.space import Space, config_key
from benchtop.tasks import build_tasks


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

    if manifest.get("lab_sha256") != sha and results_path.exists():
        # lab changed, old numbers mean nothing now
        results_path.unlink()
    done = {
        (config_key(space.normalize(r["config"])), r["seed"]) for r in read_jsonl(results_path) if r.get("ok")
    }
    jobs = [(c, s) for c in space.configs() for s in range(seeds) if (config_key(c), s) not in done]

    manifest.update(
        {
            "id": manifest.get("id", env_dir.name),
            "lab_sha256": sha,
            "seeds": seeds,
            "visible_seeds": manifest.get("visible_seeds", VISIBLE),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    started = time.perf_counter()
    errors = Counter()
    out = results_path.open("a")

    def record(config, seed, result):
        row = {"config": config, "seed": seed, **result}
        row.pop("trace", None)
        out.write(json.dumps(row) + "\n")
        out.flush()
        if not result["ok"]:
            errors[result["error"][:160]] += 1

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

    # failed rows are retried on the next sweep, so keep only the latest attempt per job
    rows = {}
    for r in read_jsonl(results_path):
        rows[(config_key(space.normalize(r["config"])), r["seed"])] = r
    results_path.write_text("".join(json.dumps(r) + "\n" for r in rows.values()))

    env = Env.load(env_dir)
    tasks = build_tasks(env)
    (env_dir / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")
    ok = sum(1 for r in rows.values() if r.get("ok"))
    summary = {
        "env": env_dir.name,
        "runs_ok": ok,
        "runs_failed": len(rows) - ok,
        "wall_secs": round(wall, 1),
        "cpu_secs": round(sum(r.get("secs", 0) for r in rows.values()), 1),
        "tasks": [t["id"] for t in tasks],
        "errors": dict(errors.most_common(5)),
        "swept_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    manifest["sweep"] = summary
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"sweep {env_dir.name}: {ok} ok, {len(rows) - ok} failed, {wall:.0f}s wall, tasks {summary['tasks']}")
    return summary
