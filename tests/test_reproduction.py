import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchtop import gym
from benchtop.cli import main
from benchtop.env import Env
from benchtop.verify import VerificationError, compare_summary, verify_foundation
from tests.test_core import toy_env


def catalog(root, env):
    (root / "catalog.json").write_text(json.dumps({
        "schema_version": 1, "id": "integration", "grader_id": env.grader_id,
        "environments": [env.id],
        "funnel": {key: 1 for key in ("screened", "selected", "built", "lab_written", "accepted")},
    }))


def test_in_process_discovery_accepts_wrapped_tasks_and_uses_catalog(tmp_path, monkeypatch):
    env = toy_env(tmp_path)
    from benchtop.tasks import build_task_set

    (env.dir / "tasks.json").write_text(json.dumps(build_task_set(env)))
    catalog(tmp_path, env)
    shutil.copytree(env.dir, tmp_path / "unlisted")
    monkeypatch.setattr(gym, "ENVS", tmp_path)
    assert gym.list_episodes() == [(env.id, task["id"]) for task in env.tasks]


def test_in_process_terminal_reward_is_emitted_once(tmp_path, monkeypatch):
    toy_env(tmp_path)
    monkeypatch.setattr(gym, "ENVS", tmp_path)
    gym._env.cache_clear()
    lab = gym.make("toy", "claim")
    lab.reset(seed=0)
    first = lab.step({"tool": "submit_answer", "answer": "no_meaningful_difference"})
    repeated = lab.step({"tool": "submit_answer", "answer": "paper_better"})
    assert json.loads(first[0]) == {"received": True}
    assert first[1:3] == (1.0, True)
    assert repeated[1:] == (0.0, True, {})
    assert lab.step({"tool": "run_experiments", "configs": [{}]})[1:] == (0.0, True, {})
    assert len(lab.ep.log) == 1
    gym._env.cache_clear()


def test_cli_sweep_resumes_the_manifest_seed_count(tmp_path):
    env = toy_env(tmp_path, seeds=8)
    main(["sweep", str(env.dir), "--workers", "1"])
    assert Env.load(env.dir).seeds == 8


def test_cached_adapter_keeps_separate_roots_and_recorded_runtime(tmp_path, monkeypatch):
    from benchtop import cache

    roots = [tmp_path / "one", tmp_path / "two"]
    for index, root in enumerate(roots):
        root.mkdir()
        toy_env(root, seeds=8 + index)
    runtime = copy.deepcopy(cache.runtime_identity())
    runtime["packages"].append(("changed-runtime", "1"))
    monkeypatch.setattr(cache, "runtime_identity", lambda: runtime)
    first, second = [gym.make("toy", "claim", envs_dir=root) for root in roots]
    first.reset(seed=0)
    second.reset(seed=0)
    assert first.ep.env.dir != second.ep.env.dir
    assert first.ep.env.execution_id != second.ep.env.execution_id
    assert first.ep.env.cache_status == second.ep.env.cache_status == "replay"
    gym._env.cache_clear()


def test_step_before_reset_has_a_clear_error():
    with pytest.raises(RuntimeError, match="reset before step"):
        gym.make("unused", "claim").step({"tool": "describe_lab"})


def test_explicit_cli_seed_change_is_rejected_without_altering_the_cache(tmp_path):
    env = toy_env(tmp_path, seeds=8)
    main(["sweep", str(env.dir), "--seeds", "8", "--workers", "1"])
    before = {path.name: path.read_bytes() for path in env.dir.iterdir()}
    with pytest.raises(ValueError, match="cache inputs changed"):
        main(["sweep", str(env.dir), "--seeds", "16", "--workers", "1"])
    assert {path.name: path.read_bytes() for path in env.dir.iterdir()} == before


def test_verification_cannot_overwrite_repository_inputs(tmp_path):
    path = tmp_path / "results/summary.json"
    path.parent.mkdir()
    path.write_text("preserve this result")
    with pytest.raises(VerificationError, match="output under artifacts"):
        verify_foundation(root=tmp_path, output=path)
    assert path.read_text() == "preserve this result"


def test_verification_rejects_changed_historical_data_before_creating_output(tmp_path):
    from benchtop.env import ROOT

    target = tmp_path / "benchmarks/historical_v0"
    shutil.copytree(ROOT / "benchmarks/historical_v0", target)
    manifest = json.loads((target / "manifest.json").read_text())
    name = next(name for name in manifest["files"] if name.startswith(("envs/", "results/", "assets/")))
    changed = tmp_path / name
    changed.parent.mkdir(parents=True, exist_ok=True)
    changed.write_bytes(b"changed historical data")
    with pytest.raises(VerificationError, match="historical file changed"):
        verify_foundation(root=tmp_path, output=Path("artifacts/check.json"))
    assert not (tmp_path / "artifacts/check.json").exists()


def test_reproduction_comparison_detects_changed_metrics_beyond_cohort_counts():
    saved = {"envs": 30, "scores": [{"agent": "uniform", "accuracy": 0.94}]}
    changed = {"envs": 30, "scores": [{"agent": "uniform", "accuracy": 0.84}]}
    with pytest.raises(VerificationError, match=r"scores\[0\].accuracy differs"):
        compare_summary(saved, changed)


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS only")
def test_isolated_worker_uses_the_selected_checkout(tmp_path):
    from benchtop import sandbox

    code = tmp_path / "selected"
    code.mkdir()
    for name in ("__init__.py", "sandbox.py", "_worker.py"):
        shutil.copyfile(Path(sandbox.__file__).parent / name, code / name)
    worker = code / "_worker.py"
    worker.write_text("import os\nos.environ['BENCHTOP_SELECTED_WORKER'] = 'yes'\n" + worker.read_text())
    lab = tmp_path / "lab" / "lab.py"
    lab.parent.mkdir()
    lab.write_text("import os\ndef run(config, seed):\n    return {'selected': int(os.environ.get('BENCHTOP_SELECTED_WORKER') == 'yes')}\n")
    script = """
import asyncio, importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('selected_sandbox', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps(asyncio.run(module.run_jobs(Path(sys.argv[2]), [({}, 0)], workers=1))))
"""
    result = subprocess.run([sys.executable, "-I", "-c", script, str(code / "sandbox.py"), str(lab)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)[0]["metrics"] == {"selected": 1.0}
