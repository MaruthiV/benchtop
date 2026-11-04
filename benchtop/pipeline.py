import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

from benchtop.env import ENVS, STATE, Env
from benchtop.sweep import sweep
from benchtop.synth import promote, synthesize

FUNNEL = STATE / "pipeline" / "funnel.jsonl"
MAX_SWEEP_CPU = 1200


def accept(env: Env, summary: dict) -> tuple[bool, str]:
    if summary["runs_failed"]:
        return False, f"sweep_failures: {summary['runs_failed']} runs failed ({list(summary['errors'])[:1]})"
    kinds = {t["kind"] for t in env.tasks}
    if not kinds & {"claim", "where", "component"}:
        return False, "no_settled_question: held-out seeds can't settle any claim or where question"
    if summary["cpu_secs"] > MAX_SWEEP_CPU:
        return False, f"too_slow: sweep used {summary['cpu_secs']:.0f} cpu seconds"
    return True, "ok"


def build_one(
    paper: dict, card: dict, backend: str = "claude", model: str = "sonnet", workers: int = 6
) -> dict:
    started = time.time()
    row = {"id": paper["id"], "title": paper["title"], "synth_backend": f"{backend}:{model}"}
    synth = synthesize(paper, card, backend=backend, model=model)
    row.update(
        {
            "synth_ok": synth["ok"],
            "attempts": synth["attempts"],
            "synth_errors": synth["history"][-1]["errors"] if synth["history"] else [],
        }
    )
    if not synth["ok"]:
        return _finish(row, "synth_failed", started)

    manifest = {
        "id": None,
        "source": "arxiv",
        "arxiv_id": paper["id"],
        "paper_title": paper["title"],
        "published": paper["published"],
        "synth": {"backend": backend, "model": model, "attempts": synth["attempts"]},
    }
    target = promote(synth["dir"], manifest)
    manifest["id"] = target.name
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = sweep(target, workers=workers, log=lambda *_: None)
    env = Env.load(target)
    ok, reason = accept(env, summary)
    row.update(
        {
            "env": target.name,
            "tasks": [t["id"] for t in env.tasks],
            "sweep_cpu": summary["cpu_secs"],
            "sweep_wall": summary["wall_secs"],
        }
    )
    if not ok:
        rejected = STATE / "pipeline" / "rejected" / target.name
        if rejected.exists():
            shutil.rmtree(rejected)
        rejected.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), rejected)
        return _finish(row, reason, started)
    return _finish(row, "accepted", started)


def _finish(row: dict, outcome: str, started: float) -> dict:
    row.update({"outcome": outcome.split(":")[0], "detail": outcome, "secs": round(time.time() - started, 1)})
    FUNNEL.parent.mkdir(parents=True, exist_ok=True)
    with FUNNEL.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def build_many(pairs: list[tuple[dict, dict]], jobs: int = 2, **kw) -> list[dict]:
    ENVS.mkdir(exist_ok=True)
    with ThreadPoolExecutor(jobs) as pool:
        return list(pool.map(lambda pc: build_one(*pc, **kw), pairs))
