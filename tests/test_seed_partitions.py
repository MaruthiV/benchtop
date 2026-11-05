import copy
import json

import pytest

from benchtop import cache
from benchtop.admission import inspect_lab
from benchtop.cli import main
from benchtop.env import ROOT, Env, seed_groups
from benchtop.episode import Episode
from benchtop.sweep import sweep
from benchtop.tasks import build_task_set
from tests.test_admission import fixture

SPLITS = {"visible": [0, 1], "qualification": [2, 3, 4], "confirmation": [5, 6, 7]}
MANIFEST = {"seeds": 8, "visible_seeds": 2, "seed_splits": SPLITS}


@pytest.mark.parametrize("change", [
    {"seed_splits": []},
    {"seed_splits": {"visible": [0, 1], "qualification": [2, 3, 4]}},
    {"seed_splits": {**SPLITS, "extra": []}},
    {"seed_splits": {**SPLITS, "visible": [1, 0]}},
    {"seed_splits": {**SPLITS, "visible": [0, True]}},
    {"seed_splits": {**SPLITS, "qualification": [2, 2, 4]}},
    {"seed_splits": {**SPLITS, "qualification": [1, 3, 4]}},
    {"seed_splits": {**SPLITS, "qualification": [2, 3]}},
    {"seed_splits": {**SPLITS, "qualification": [2, 3, 9]}},
    {"seed_splits": {**SPLITS, "qualification": [2, 3, 4.0]}},
    {"seed_splits": {**SPLITS, "qualification": (2, 3, 4)}},
    {"seed_splits": {"visible": [0, 1], "qualification": [2], "confirmation": [3, 4, 5, 6, 7]}},
    {"seed_splits": {"visible": [0, 1], "qualification": [2, 3, 4, 5, 6], "confirmation": [7]}},
    {"seeds": True},
    {"visible_seeds": 2.0},
])
def test_invalid_partitions_fail_before_sweep_writes(tmp_path, change):
    directory = fixture(tmp_path)
    manifest = {**MANIFEST, **change}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(cache.CacheError):
        seed_groups(manifest)
    with pytest.raises(cache.CacheError):
        Env.load(directory)
    with pytest.raises(cache.CacheError):
        sweep(directory, log=lambda *_: None)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_explicit_groups_are_copied_and_legacy_pool_is_preserved(tmp_path):
    groups = seed_groups(MANIFEST)
    groups["qualification"].clear()
    assert SPLITS["qualification"] == [2, 3, 4]
    legacy = Env.load(fixture(tmp_path))
    assert legacy.qualification_seeds == legacy.heldout_seeds == [2, 3, 4, 5, 6, 7]
    assert legacy.confirmation_seeds == []
    assert seed_groups(legacy.manifest) is None


def test_confirmation_metrics_and_timings_cannot_change_qualification_or_prices(tmp_path):
    directory = fixture(tmp_path, splits=copy.deepcopy(SPLITS))
    env = Env.load(directory)
    before = build_task_set(env)
    subject = inspect_lab(directory)["review_subject"]
    prices, charging_id = env.charging_schedule()
    assert env.heldout_seeds == env.qualification_seeds == [2, 3, 4]
    assert env.confirmation_seeds == [5, 6, 7]
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    for row in rows:
        if row["seed"] in env.confirmation_seeds:
            row["metrics"]["value"] = -999 if row["config"]["method"] == "new" else 999
            row["secs"] = 999
            row.update(cache.seal_result(env.execution, env.space, row["config"], row["seed"], row))
    (directory / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    updated = Env.load(directory)
    assert build_task_set(updated) == before
    assert updated.task_version == env.task_version
    assert updated.charging_schedule() == (prices, charging_id)
    assert inspect_lab(directory)["review_subject"] == subject
    assert updated.task("claim")["answer"] == "paper_better"


@pytest.mark.parametrize("mixed", [False, True])
def test_confirmation_task_cannot_be_loaded_as_a_qualification_episode(tmp_path, mixed):
    directory = fixture(tmp_path, splits=copy.deepcopy(SPLITS))
    env = Env.load(directory)
    artifact = build_task_set(env, seeds=env.confirmation_seeds)
    if mixed:
        artifact["seed_ids"] = env.qualification_seeds
        artifact["binding"] = cache.task_binding(env, env.qualification_seeds)
        artifact = cache.with_checksum(artifact, "artifact_sha256")
    (directory / "tasks.json").write_text(json.dumps(artifact))
    with pytest.raises(cache.CacheError, match="qualification partition"):
        Env.load(directory)


def test_complete_confirmation_cannot_fill_missing_qualification_pairs(tmp_path):
    directory = fixture(tmp_path, splits=copy.deepcopy(SPLITS), missing=(0, 2))
    env = Env.load(directory)
    artifact = build_task_set(env)
    assert artifact["status"] == "incomplete" and artifact["tasks"] == []
    assert artifact["issues"] == [{"kind": "missing_results", "count": 1}]
    assert not env.table.missing(env.confirmation_seeds)
    assert not inspect_lab(directory)["qualification"]["claim_eligible"]


def test_agent_runs_are_still_limited_to_visible_replicates(tmp_path):
    env = Env.load(fixture(tmp_path, splits=copy.deepcopy(SPLITS)))
    episode = Episode(env, "claim", budget_frac=1, visible_order=[1, 0])
    response = episode.run(env.space.configs(), seeds=8)
    assert len(response["results"]) == env.space.size() * 2
    assert {seed for _, seed in episode.log[0]["ran"]} == {0, 1}
    assert "seed_splits" not in episode.describe()
    assert "confirmation_seeds" not in episode.describe()


def empty_fresh(root, monkeypatch):
    directory = fixture(root, splits=copy.deepcopy(SPLITS))
    (directory / "cache.json").unlink()
    (directory / "tasks.json").unlink()
    (directory / "results.jsonl").unlink()
    manifest = json.loads((directory / "manifest.json").read_text())
    spec = json.loads((ROOT / "benchmarks/release_v1/spec.json").read_text())
    groups = {name: spec["seed_partitions"][name] for name in SPLITS}
    manifest.update(seeds=28, visible_seeds=4, seed_splits=groups)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    calls = []

    async def controlled_jobs(lab, jobs, workers, timeout, sandbox, **kwargs):
        for config, seed in jobs:
            calls.append((config, seed))
            assert seed not in groups["confirmation"]
            value = 0.5 + (0.2 if config["method"] == "new" else 0)
            kwargs["on_result"](config, seed, {"ok": True, "metrics": {"value": value}, "secs": 1.0})

    monkeypatch.setattr("benchtop.sweep.run_jobs", controlled_jobs)
    monkeypatch.setattr("benchtop.sweep.sandbox_mode", lambda _: "controlled_unit_fixture")
    return directory, groups, calls


def test_default_fresh_sweep_reserves_confirmation_and_resume_starts_no_work(tmp_path, monkeypatch):
    directory, groups, calls = empty_fresh(tmp_path, monkeypatch)
    summary = sweep(directory, log=lambda *_: None)
    env = Env.load(directory)
    assert {seed for _, seed in calls} == set(groups["visible"] + groups["qualification"])
    assert len(calls) == env.space.size() * 16
    assert summary["task_status"] == "complete"
    assert summary["seed_stage"] == "visible_and_qualification" and not summary["confirmation_evaluated"]
    assert env.table.missing(env.confirmation_seeds) == [(c, s) for c in env.space.configs() for s in groups["confirmation"]]
    assert all(task["seed_ids"] == groups["qualification"] for task in env.tasks)
    before = (directory / "results.jsonl").read_bytes()
    calls.clear()
    sweep(directory, log=lambda *_: None)
    assert not calls and (directory / "results.jsonl").read_bytes() == before
    assert Env.load(directory).task_version == env.task_version


def test_cli_can_sweep_visible_then_qualification_without_confirmation(tmp_path, monkeypatch):
    directory, groups, calls = empty_fresh(tmp_path, monkeypatch)
    main(["sweep", str(directory), "--workers", "1", "--split", "visible"])
    assert {seed for _, seed in calls} == set(groups["visible"])
    artifact = json.loads((directory / "tasks.json").read_text())
    assert artifact["status"] == "incomplete" and artifact["tasks"] == []
    assert artifact["seed_ids"] == groups["qualification"]
    calls.clear()
    main(["sweep", str(directory), "--workers", "1", "--split", "qualification"])
    assert {seed for _, seed in calls} == set(groups["qualification"])
    assert Env.load(directory).task("claim")["answer"] == "paper_better"


def test_failed_qualification_pair_stays_incomplete_until_exact_job_resumes(tmp_path, monkeypatch):
    from benchtop import sweep as runner

    directory, groups, calls = empty_fresh(tmp_path, monkeypatch)
    original = runner.run_jobs
    failed = []

    async def fail_one(lab, jobs, workers, timeout, sandbox, **kwargs):
        config, seed = next(job for job in jobs if job[1] == groups["qualification"][0])
        failed.append((config, seed))
        kwargs["on_result"](config, seed, {"ok": False, "error": "Controlled fixture failure", "secs": 0})
        await original(lab, [job for job in jobs if job != (config, seed)], workers, timeout, sandbox, **kwargs)

    monkeypatch.setattr(runner, "run_jobs", fail_one)
    first = sweep(directory, log=lambda *_: None)
    assert first["runs_failed"] == 1 and first["task_status"] == "incomplete"
    assert Env.load(directory).tasks == []
    monkeypatch.setattr(runner, "run_jobs", original)
    calls.clear()
    resumed = sweep(directory, log=lambda *_: None)
    assert calls == failed
    assert resumed["runs_failed"] == 0 and resumed["task_status"] == "complete"
    assert Env.load(directory).table.missing(groups["confirmation"])


@pytest.mark.parametrize("split", ["confirmation", "typo"])
def test_ordinary_sweep_rejects_confirmation_or_unknown_stage(tmp_path, split):
    directory = fixture(tmp_path, splits=copy.deepcopy(SPLITS))
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(ValueError, match="locked study runner"):
        sweep(directory, split=split, log=lambda *_: None)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_named_split_rejects_legacy_and_seed_extension_preserves_fresh_cache(tmp_path):
    legacy = fixture(tmp_path / "legacy")
    with pytest.raises(ValueError, match="three-part"):
        sweep(legacy, split="visible", log=lambda *_: None)
    directory = fixture(tmp_path / "fresh", splits=copy.deepcopy(SPLITS))
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(cache.CacheError):
        sweep(directory, seeds=9, log=lambda *_: None)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_valid_partition_change_cannot_reuse_or_relabel_cache(tmp_path):
    directory = fixture(tmp_path, splits=copy.deepcopy(SPLITS))
    env = Env.load(directory)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["seed_splits"]["qualification"], manifest["seed_splits"]["confirmation"] = (
        manifest["seed_splits"]["confirmation"], manifest["seed_splits"]["qualification"])
    path.write_text(json.dumps(manifest))
    with pytest.raises(cache.CacheError):
        env.assert_current()
    with pytest.raises(cache.CacheError):
        Env.load(directory, replay=True)
    with pytest.raises(cache.CacheError):
        sweep(directory, log=lambda *_: None)
