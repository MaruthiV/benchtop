import asyncio
import copy
import json
import shutil
import sys

import pytest

from benchtop import cache, report
from benchtop.env import ENVS, Env
from benchtop.episode import Episode
from benchtop.live import LiveStore
from benchtop.sandbox import Pool, run_jobs, sandbox_mode
from benchtop.server import Hub
from benchtop.sweep import sweep
from tests.test_core import toy_env


@pytest.mark.parametrize("changed", ["lab.py", "space.json", "manifest.json"])
def test_changed_execution_inputs_reject_saved_results(tmp_path, changed):
    env = toy_env(tmp_path)
    path = env.dir / changed
    if changed == "lab.py":
        path.write_text("def run(config, seed):\n    return {'acc': -1.0}\n")
    else:
        content = json.loads(path.read_text())
        if changed == "space.json":
            content["metric"]["min_effect"] = 0.03
        else:
            content["visible_seeds"] = 3
        path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="cache|inputs"):
        Env.load(env.dir)


def test_unsupported_auto_sandbox_never_falls_back_to_unisolated_execution(monkeypatch):
    monkeypatch.setattr("benchtop.sandbox.sys.platform", "linux")
    monkeypatch.setattr("benchtop.sandbox.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="sandbox"):
        sandbox_mode()


@pytest.mark.parametrize("mode", ["none", "linux", "typo"])
def test_explicit_unsupported_sandbox_modes_are_rejected(mode):
    with pytest.raises(RuntimeError, match="sandbox"):
        sandbox_mode(mode)


def test_missing_macos_sandbox_is_rejected(monkeypatch):
    monkeypatch.setattr("benchtop.sandbox.sys.platform", "darwin")
    monkeypatch.setattr("benchtop.sandbox.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="sandbox-exec"):
        sandbox_mode("macos")


def test_runtime_change_requires_rebuild_but_explicit_replay_remains_read_only(tmp_path, monkeypatch):
    env = toy_env(tmp_path)
    other = copy.deepcopy(cache.runtime_identity())
    other["packages"].append(("changed-dependency", "2.0"))
    monkeypatch.setattr(cache, "runtime_identity", lambda: other)
    with pytest.raises(cache.CacheError, match="inputs"):
        Env.load(env.dir)
    with pytest.raises(cache.CacheError, match="runtime"):
        env.assert_current()
    replay = Env.load(env.dir, replay=True)
    assert replay.execution_id == env.execution_id
    assert replay.task_version == env.task_version
    assert Episode(replay, "claim").run([env.space.configs()[0]])["results"]
    with pytest.raises(cache.CacheError, match="read-only"):
        replay.prepare_live()


@pytest.mark.parametrize("changed", ["_worker.py", "tasks.py", "stats.py"])
def test_engine_and_task_generator_changes_are_distinguished(tmp_path, monkeypatch, changed):
    env = toy_env(tmp_path)
    code = tmp_path / "code"
    code.mkdir()
    for name in (*cache.ENGINE_FILES, "tasks.py", "stats.py"):
        shutil.copyfile(cache.CODE / name, code / name)
    with (code / changed).open("a") as out:
        out.write("\n# changed implementation\n")
    monkeypatch.setattr(cache, "CODE", code)
    with pytest.raises(cache.CacheError, match="inputs|generator"):
        Env.load(env.dir)
    if changed == "_worker.py":
        assert Env.load(env.dir, replay=True).execution_id == env.execution_id
    else:
        # Recomputing labels is explicit, and need not rerun the valid physical experiments.
        summary = sweep(env.dir, log=lambda *_: None)
        assert summary["task_status"] == "complete"
        assert Env.load(env.dir).task_version != env.task_version


@pytest.mark.parametrize("changed", ["execution_id", "metrics", "seed", "row_sha256"])
def test_modified_or_mixed_result_rows_are_rejected(tmp_path, changed):
    env = toy_env(tmp_path)
    path = env.dir / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if changed == "execution_id":
        rows[0][changed] = "another-execution"
    elif changed == "metrics":
        rows[0][changed]["acc"] += 0.1
    elif changed == "seed":
        rows[0][changed] = env.seeds + 1
    else:
        rows[0][changed] = "wrong-checksum"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(cache.CacheError, match="cache result"):
        Env.load(env.dir)


@pytest.mark.parametrize("qualification", [True, False])
def test_tasks_bind_qualification_observations_without_binding_visible_appends(tmp_path, qualification):
    env = toy_env(tmp_path)
    path = env.dir / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(row for row in rows if row["seed"] == (4 if qualification else 0))
    row["metrics"]["acc"] += 0.1
    row.update(cache.seal_result(env.execution, env.space, row["config"], row["seed"], row))
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if qualification:
        with pytest.raises(cache.CacheError, match="task cache inputs"):
            Env.load(env.dir)
    else:
        assert Env.load(env.dir).task_version == env.task_version


def test_task_payload_change_without_rebuilding_is_rejected(tmp_path):
    env = toy_env(tmp_path)
    path = env.dir / "tasks.json"
    tasks = json.loads(path.read_text())
    tasks[0]["answer"] = "paper_better"
    path.write_text(json.dumps(tasks))
    with pytest.raises(cache.CacheError, match="payload checksum"):
        Env.load(env.dir)


@pytest.mark.parametrize("changed", ["lab.py", "space.json", "runtime"])
def test_stale_sweep_preserves_all_existing_files(tmp_path, monkeypatch, changed):
    env = toy_env(tmp_path)
    if changed == "runtime":
        runtime = {**cache.runtime_identity(), "python": "another-python"}
        monkeypatch.setattr(cache, "runtime_identity", lambda: runtime)
    else:
        with (env.dir / changed).open("a") as out:
            out.write("\n")
    before = {path.name: path.read_bytes() for path in env.dir.iterdir()}
    with pytest.raises(cache.CacheError, match="inputs"):
        sweep(env.dir, log=lambda *_: None)
    assert {path.name: path.read_bytes() for path in env.dir.iterdir()} == before


def test_unbound_results_cannot_be_blessed_as_a_new_cache(tmp_path):
    env = toy_env(tmp_path)
    (env.dir / "cache.json").unlink()
    before = (env.dir / "results.jsonl").read_bytes()
    with pytest.raises(cache.CacheError, match="historical provenance"):
        Env.load(env.dir)
    with pytest.raises(cache.CacheError, match="unversioned cache"):
        cache.initialize_cache(env.dir, env.manifest)
    assert (env.dir / "results.jsonl").read_bytes() == before


def test_changed_files_in_an_active_episode_prevent_delivery_and_grading(tmp_path):
    env = toy_env(tmp_path)
    episode = Episode(env, "claim")
    with (env.dir / "lab.py").open("a") as out:
        out.write("\n# new code\n")
    assert episode.run([env.space.configs()[0]])["code"] == "stale_cache"
    assert episode.submit("no_meaningful_difference")["code"] == "stale_cache"
    assert episode.calls == episode.spent == 0
    assert not episode.done


def test_historical_snapshots_are_verified_and_read_only(tmp_path):
    name = "2610-36983-advancement-or-acceleration-how"
    shutil.copytree(ENVS / name, tmp_path / name)
    shutil.copyfile(ENVS / "historical_cache.json", tmp_path / "historical_cache.json")
    env = Env.load(tmp_path / name)
    assert env.cache_status == "historical"
    assert Episode(env, "claim").run([env.space.configs()[0]])["results"]
    with pytest.raises(cache.CacheError, match="read-only"):
        env.prepare_live()
    before = {path.name: path.read_bytes() for path in env.dir.iterdir()}
    with pytest.raises(cache.CacheError, match="read-only"):
        sweep(env.dir, log=lambda *_: None)
    assert {path.name: path.read_bytes() for path in env.dir.iterdir()} == before
    with (env.dir / "results.jsonl").open("a") as out:
        out.write("\n")
    with pytest.raises(cache.CacheError, match="historical cache inputs"):
        Env.load(env.dir)


@pytest.mark.parametrize("changed", ["execution_id", "task_version", "missing"])
def test_report_rejects_mismatched_result_bindings(tmp_path, monkeypatch, changed):
    env = toy_env(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    row = {"env": "toy", "task": "claim", "grader_id": env.grader_id,
           "execution_id": env.execution_id, "task_version": env.task_version}
    if changed == "missing":
        row.pop("execution_id")
    else:
        row[changed] = "another-version"
    (results / "baselines.json").write_text(json.dumps([row]))
    monkeypatch.setattr(report, "RESULTS", results)
    with pytest.raises(ValueError, match="result cache versions"):
        report.check_grading({"toy": env})


def test_unsupported_live_paths_fail_before_writing_or_starting_workers(tmp_path, monkeypatch):
    env = toy_env(tmp_path)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    for name in ("lab.py", "space.json", "manifest.json"):
        shutil.copyfile(env.dir / name, fresh / name)
    before = {path.name: path.read_bytes() for path in fresh.iterdir()}
    monkeypatch.setattr("benchtop.sandbox.sys.platform", "linux")
    monkeypatch.setattr("benchtop.sandbox.shutil.which", lambda _: None)
    for operation in (lambda: Hub(tmp_path, live=True), lambda: Pool(fresh / "lab.py"),
                      lambda: LiveStore(env), lambda: sweep(fresh, log=lambda *_: None)):
        with pytest.raises(RuntimeError, match="sandbox"):
            operation()
    assert {path.name: path.read_bytes() for path in fresh.iterdir()} == before
    historical = Env.load(ENVS / "label-smoothing")
    assert Episode(historical, "claim").run([historical.space.configs()[0]])["results"]


@pytest.mark.parametrize("corruption", ["schema_version", "id"])
def test_corrupt_cache_descriptor_is_rejected(tmp_path, corruption):
    env = toy_env(tmp_path)
    path = env.dir / "cache.json"
    descriptor = json.loads(path.read_text())
    descriptor[corruption] = 999 if corruption == "schema_version" else "wrong-id"
    path.write_text(json.dumps(descriptor))
    with pytest.raises(cache.CacheError, match="cache schema|identity is corrupt"):
        Env.load(env.dir)


def test_inputs_changed_during_live_work_are_not_persisted_or_charged(tmp_path, monkeypatch):
    from tests.test_live_failures import ControlledPool, live_episode

    monkeypatch.setattr("benchtop.server.sandbox_mode", lambda: "macos")
    monkeypatch.setattr("benchtop.live.sandbox_mode", lambda: "macos")

    async def scenario():
        hub, episode, config, store = live_episode(tmp_path)
        store.pool = pool = ControlledPool(blocked=True)
        original = (episode.env.dir / "results.jsonl").read_bytes()
        request = asyncio.create_task(hub.run(episode, [config]))
        await asyncio.wait_for(pool.started.wait(), 2)
        with (episode.env.dir / "lab.py").open("a") as out:
            out.write("\n# changed while running\n")
        pool.release.set()
        result = await request
        assert result["code"] == "stale_cache"
        assert episode.calls == 1
        assert episode.spent == 0
        assert episode.log[0]["status"] == "failed"
        assert (episode.env.dir / "results.jsonl").read_bytes() == original
        await hub.close()

    asyncio.run(scenario())


def test_worker_startup_refuses_changed_lab_copy(tmp_path, monkeypatch):
    lab = tmp_path / "lab.py"
    lab.write_text("def run(config, seed): return {'acc': 1}\n")
    monkeypatch.setattr("benchtop.sandbox.sandbox_mode", lambda *_: "macos")
    pool = Pool(lab, workers=1)
    lab.write_text("def run(config, seed): return {'acc': 2}\n")

    async def scenario():
        with pytest.raises(RuntimeError, match="lab changed"):
            await pool.__aenter__()
        assert pool.workers[0].tmp is None
        assert pool.workers[0].proc is None

    asyncio.run(scenario())


@pytest.mark.parametrize("changed", ["lab.py", "manifest.json", "cache.json"])
def test_sweep_detects_inflight_input_changes_before_recording_or_overwriting(tmp_path, monkeypatch, changed):
    env = toy_env(tmp_path)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    for name in ("lab.py", "space.json", "manifest.json"):
        shutil.copyfile(env.dir / name, fresh / name)
    monkeypatch.setattr("benchtop.sweep.sandbox_mode", lambda *_: "macos")

    async def changing_inputs(lab, jobs, *args, on_result, **kwargs):
        path = fresh / changed
        path.write_text(path.read_text() + "\n")
        edited = path.read_bytes()
        config, seed = jobs[0]
        try:
            on_result(config, seed, {"ok": True, "metrics": {"acc": 0.7}, "secs": 1.0})
        finally:
            assert path.read_bytes() == edited

    monkeypatch.setattr("benchtop.sweep.run_jobs", changing_inputs)
    with pytest.raises(cache.CacheError, match="inputs changed during sweep"):
        sweep(fresh, log=lambda *_: None)
    assert (fresh / "results.jsonl").read_bytes() == b""
    assert not (fresh / "tasks.json").exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS only")
def test_live_miss_and_reloaded_cache_match_the_deterministic_fixture(tmp_path):
    env = toy_env(tmp_path)
    config = env.space.configs()[0]
    seed = Episode(env, "claim", seed=0).order[0]
    expected = env.table.get(config, seed)
    rows = [json.loads(line) for line in (env.dir / "results.jsonl").read_text().splitlines()]
    rows = [row for row in rows if (row["config"], row["seed"]) != (config, seed)]
    (env.dir / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    async def scenario():
        hub = Hub(tmp_path, live=True, live_workers=1)
        episode = hub.create("toy", "claim", seed=0)
        response = await hub.run(episode, [config])
        assert response["results"][0]["metrics"] == {key: round(value, 6) for key, value in expected.items()}
        assert hub.stores["toy"].stats["executed"] == 1
        await hub.close()

    asyncio.run(scenario())
    reloaded = Env.load(env.dir)
    assert reloaded.table.get(config, seed) == expected
    assert reloaded.execution_id == env.execution_id
    assert reloaded.task_version == env.task_version


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS only")
def test_custom_lab_directory_is_unreadable_to_its_worker(tmp_path):
    secret = tmp_path / "results.jsonl"
    secret.write_text("held-out answers")
    lab = tmp_path / "lab.py"
    lab.write_text(
        "def run(config, seed):\n"
        f"    try:\n        open({str(secret)!r}).read()\n        return {{'read': 1}}\n"
        "    except PermissionError:\n        return {'read': 0}\n"
    )
    results = asyncio.run(run_jobs(lab, [({}, 0)], workers=1))
    assert results[0]["metrics"] == {"read": 0.0}


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS only")
def test_fresh_sweep_writes_bound_results_and_resumes_without_execution(tmp_path, monkeypatch):
    from tests.test_live import LAB, SPACE

    env_dir = tmp_path / "fresh"
    env_dir.mkdir()
    (env_dir / "lab.py").write_text(LAB)
    (env_dir / "space.json").write_text(json.dumps(SPACE))
    (env_dir / "manifest.json").write_text(json.dumps({"id": "fresh", "seeds": 6, "visible_seeds": 2}))
    first = sweep(env_dir, seeds=6, workers=2, log=lambda *_: None)
    env = Env.load(env_dir)
    assert first["runs_ok"] == 24
    assert first["runs_failed"] == 0
    assert first["task_status"] == "complete"
    results = (env_dir / "results.jsonl").read_bytes()
    rows = [json.loads(line) for line in results.splitlines()]
    assert all(row["execution_id"] == env.execution_id and row["row_sha256"] for row in rows)
    identity = env.execution_id

    async def unexpected_execution(*args, **kwargs):
        pytest.fail("a fully cached sweep must not start workers")

    monkeypatch.setattr("benchtop.sweep.run_jobs", unexpected_execution)
    second = sweep(env_dir, seeds=6, log=lambda *_: None)
    assert second["runs_ok"] == first["runs_ok"]
    assert Env.load(env_dir).execution_id == identity
    assert (env_dir / "results.jsonl").read_bytes() == results
