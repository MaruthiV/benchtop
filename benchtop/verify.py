"""Reproduce engineering checks in a disposable copy of public repository inputs."""

import contextlib
import io
import json
import math
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from benchtop.cache import file_hash, runtime_identity
from benchtop.env import ROOT, Env

PUBLIC_DIRS = ("benchtop", "envs", "results", "benchmarks")
PUBLIC_FILES = ("README.md", "pyproject.toml", "uv.lock")

FIXTURE_LAB = """import random
def run(config, seed):
    noise = random.Random(seed).uniform(-0.001, 0.001)
    gain = 0.06 if config['method'] == 'new' and config['context'] == 1 else 0.0
    return {'value': 0.5 + gain + noise}
"""
FIXTURE_SPACE = {
    "title": "Foundation integration fixture", "setting": "Synthetic engineering fixture, not a paper lab",
    "metric": {"name": "value", "goal": "max", "min_effect": 0.01},
    "factors": [{"name": "method", "role": "treatment", "levels": ["base", "new"]},
                {"name": "context", "role": "context", "levels": [0, 1]}],
    "treatment": {"factor": "method", "paper": "new", "baseline": "base"},
}


class VerificationError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def compare_summary(expected, actual, path="summary"):
    """Compare every saved field, allowing only floating-point roundoff and new metadata."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"{path} is not an object")
        for key, value in expected.items():
            require(key in actual, f"{path}.{key} is missing")
            compare_summary(value, actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(expected) == len(actual), f"{path} has different length")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            compare_summary(left, right, f"{path}[{index}]")
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        require(isinstance(actual, (float, int)), f"{path} is not numeric")
        require((math.isnan(expected) and math.isnan(actual))
                or math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12), f"{path} differs")
    else:
        require(expected == actual, f"{path} differs")


def _check_episode(env, envs_dir):
    from benchtop.agents.baselines import play
    from benchtop.episode import Episode
    from benchtop.gym import make

    lab = make(env.id, "claim", budget_frac=1.0, show_paper=False, envs_dir=envs_dir)
    observation, _ = lab.reset(seed=7)
    require("paper" not in json.loads(observation), "paper hiding failed")
    observation, reward, done, _ = lab.step({"tool": "run_experiments", "configs": [env.space.configs()[0]]})
    require(len(json.loads(observation).get("results", [])) == 1 and reward == 0 and not done, "cached run failed")
    # Expected truth is used by the trusted checker only, never as an agent performance measurement.
    expected = env.task("claim")["answer"]
    observation, reward, done, _ = lab.step({"tool": "submit_answer", "answer": expected})
    require(json.loads(observation) == {"received": True} and reward == 1 and done, "trainer completion failed")
    _, reward, done, info = lab.step({"tool": "submit_answer", "answer": "paper_better"})
    require(reward == 0 and done and info == {}, "terminal reward was repeated")
    first, second = (Episode(env, "claim", seed=7, budget_frac=1.0) for _ in range(2))
    play(first, "uniform", seed=7)
    play(second, "uniform", seed=7)
    def runs(ep):
        return [entry["ran"] for entry in ep.log if entry["action"] == "run"]
    require(first.result == second.result and first.spent == second.spent and runs(first) == runs(second),
            "classical replay is not deterministic")
    return {"cached_observation": True, "trainer_completion": True, "terminal_reward_once": True,
            "deterministic_uniform_replay": True}


def _check_offline_copy():
    from benchtop.cli import main
    from benchtop.evaluate import all_envs, episode_list
    from benchtop.gym import list_episodes

    require(Path.cwd().resolve() == ROOT, "checker imported code outside the selected copy")
    require(not (ROOT / ".state").exists(), "copy unexpectedly contains private state")
    require(not (ROOT / "assets").exists(), "copy unexpectedly contains saved figures")
    expected = json.loads((ROOT / "results/summary.json").read_text())
    (ROOT / "results/summary.json").unlink()
    (ROOT / "assets").mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        main(["report"])
    actual = json.loads((ROOT / "results/summary.json").read_text())
    compare_summary(expected, actual)
    counts = {suffix: len(list((ROOT / "assets").glob(f"*.{suffix}"))) for suffix in ("svg", "png")}
    require(counts == {"svg": 5, "png": 5}, "report did not regenerate all figures")
    require(all(path.stat().st_size > 100 for path in (ROOT / "assets").iterdir()), "empty report figure")
    pairs = list_episodes()
    require(pairs == episode_list() and len(pairs) == 78, "discovery differs across interfaces")
    envs = all_envs(replay=True)
    require(len(envs) == 30, "historical cohort changed")
    reference = next(env for env in envs if any(task["id"] == "claim" for task in env.tasks))
    adapter = _check_episode(reference, ROOT / "envs")
    require(not (ROOT / ".state").exists(), "offline reproduction used private state")
    return {"status": "passed", "labs": len(envs), "questions": len(pairs),
            "historical_summary_fields_compared": len(expected), "figure_counts": counts,
            "private_state_absent": True, "adapter": adapter}


def _check_live_copy():
    from benchtop.cli import main
    from benchtop.gym import list_episodes
    from benchtop.stats import GRADER_ID

    require(Path.cwd().resolve() == ROOT, "live checker imported code outside the selected copy")
    envs_dir = ROOT / "integration_envs"
    fixture = envs_dir / "fixture"
    fixture.mkdir(parents=True)
    (fixture / "lab.py").write_text(FIXTURE_LAB)
    (fixture / "space.json").write_text(json.dumps(FIXTURE_SPACE))
    (fixture / "manifest.json").write_text(json.dumps({"id": "fixture", "seeds": 6, "visible_seeds": 2}))
    with contextlib.redirect_stdout(io.StringIO()):
        main(["sweep", str(fixture), "--seeds", "6", "--workers", "1"])
    env = Env.load(fixture)
    require(env.grader_id == GRADER_ID and len(env.table.missing(list(range(6)))) == 0, "fixture sweep failed")
    artifact = json.loads((fixture / "tasks.json").read_text())
    require(isinstance(artifact, dict) and artifact["status"] == "complete", "wrapped tasks missing")
    (envs_dir / "catalog.json").write_text(json.dumps({
        "schema_version": 1, "id": "foundation_fixture", "grader_id": GRADER_ID, "environments": ["fixture"],
        "funnel": {key: 1 for key in ("screened", "selected", "built", "lab_written", "accepted")},
    }))
    require(list_episodes(envs_dir=envs_dir) == [(env.id, task["id"]) for task in env.tasks], "wrapped task discovery failed")
    before = file_hash(fixture / "results.jsonl")
    with (patch("benchtop.sweep.run_jobs", side_effect=VerificationError("cached resume started workers")),
          contextlib.redirect_stdout(io.StringIO())):
        # No --seeds: the CLI must preserve the six-seed manifest rather than assume 16.
        main(["sweep", str(fixture), "--workers", "1"])
    require(file_hash(fixture / "results.jsonl") == before, "cached resume changed results")
    reloaded = Env.load(fixture)
    require(reloaded.seeds == 6 and reloaded.execution_id == env.execution_id
            and reloaded.task_version == env.task_version, "cached resume changed versions")
    return {"status": "passed", "fixture": "synthetic_engineering_only", "runs": 24,
            "execution_id": env.execution_id, "task_version": env.task_version,
            "wrapped_task_discovery": True, "resume_without_workers": True,
            "adapter": _check_episode(reloaded, envs_dir)}


def _run_copy(checkout, mode, work):
    script = """
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from benchtop.verify import _check_offline_copy, _check_live_copy
result = _check_live_copy() if sys.argv[1] == 'live' else _check_offline_copy()
print(json.dumps(result, allow_nan=False))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, mode], cwd=checkout, capture_output=True, text=True, timeout=60,
        env={**os.environ, "MPLCONFIGDIR": str(work / "matplotlib"), "MPLBACKEND": "Agg",
             "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"},
    )
    require(result.returncode == 0, f"{mode} copy check failed:\n{result.stderr[-6000:]}")
    return json.loads(result.stdout)


def verify_foundation(*, live=False, output: Path | None = None, root: Path = ROOT):
    root = Path(root).resolve()
    target = (root / output).resolve() if output is not None else None
    if target is not None:
        require(not target.is_relative_to(root) or target.is_relative_to(root / "artifacts"),
                "choose an output under artifacts/ or outside the repository")
    if live:
        from benchtop.sandbox import sandbox_mode

        sandbox_mode()
    started = time.monotonic()
    snapshot = root / "benchmarks/historical_v0"
    archived = runpy.run_path(str(snapshot / "verify.py"))["verify"](snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    protected = {name: item for name, item in manifest["files"].items()
                 if name.startswith(("envs/", "results/", "assets/"))}
    for name, item in protected.items():
        require(file_hash(root / name) == item["sha256"], f"historical file changed: {name}")
    files = [root / name for name in PUBLIC_FILES]
    for name in PUBLIC_DIRS:
        files.extend(path for path in (root / name).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    hashes = {str(path.relative_to(root)): file_hash(path) for path in sorted(files)}
    with tempfile.TemporaryDirectory(prefix="benchtop-verify-") as temporary:
        work = Path(temporary)
        checkout = work / "checkout"
        checkout.mkdir()
        for name in PUBLIC_DIRS:
            shutil.copytree(root / name, checkout / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in PUBLIC_FILES:
            shutil.copyfile(root / name, checkout / name)
        offline = _run_copy(checkout, "offline", work)
        execution = _run_copy(checkout, "live", work) if live else {"status": "not_requested"}
    require(all(file_hash(root / name) == sha for name, sha in hashes.items()), "source inputs changed during verification")
    require(all(file_hash(root / name) == item["sha256"] for name, item in protected.items()), "historical inputs changed")
    result = {
        "schema_version": 1, "status": "passed", "checked_at_utc": datetime.now(UTC).isoformat(),
        "scope": "foundation_engineering_checks", "seconds": round(time.monotonic() - started, 2),
        "archive_files_verified": archived, "original_data_files_verified": len(protected),
        "source_inputs_unchanged": True, "input_sha256": hashes, "runtime": runtime_identity(),
        "offline": offline, "live": execution, "model_sessions": 0, "qualia": "not_tested",
    }
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
