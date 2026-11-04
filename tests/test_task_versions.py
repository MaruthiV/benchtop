import json

import pytest

from benchtop import report
from benchtop.agents.common import outcome
from benchtop.env import Env, lab_sha
from benchtop.episode import Episode
from benchtop.evaluate import all_envs
from benchtop.pipeline import accept
from benchtop.stats import GRADER_ID, LEGACY_GRADER_ID
from benchtop.sweep import sweep
from benchtop.tasks import build_task_set, build_tasks, contrast
from tests.test_core import toy_env


def test_ambiguous_claim_is_retained_as_diagnostic(tmp_path):
    env = toy_env(tmp_path)
    for config in env.space.configs():
        for seed in env.heldout_seeds:
            gain = 0.02 + (-0.05 if seed % 2 else 0.05)
            env.table.get(config, seed)["acc"] = 0.7 + (gain if config["method"] == "new" else 0)
    artifact = build_task_set(env)
    claim = next(item for item in artifact["contrasts"] if item["task_id"] == "claim")
    assert artifact["status"] == "complete"
    assert claim["status"] == "ambiguous"
    assert claim["label"] is None
    assert all(task["id"] != "claim" for task in artifact["tasks"])


@pytest.mark.parametrize("whole_context", [False, True])
def test_missing_qualification_data_is_explicit(tmp_path, whole_context):
    env = toy_env(tmp_path)
    config = env.space.configs()[0]
    held = env.heldout_seeds
    for seed in held if whole_context else held[:1]:
        env.table.cells[next(key for key in env.table.cells if json.loads(key) == config)].pop(seed)
    artifact = build_task_set(env)
    assert artifact["status"] == "incomplete"
    assert artifact["tasks"] == []
    assert artifact["issues"] == [{"kind": "missing_results", "count": 12 if whole_context else 1}]
    with pytest.raises(ValueError, match="qualification"):
        build_tasks(env)
    with pytest.raises(ValueError, match="qualification"):
        contrast(env, "method", "new", "base", held)


def test_nonfinite_metric_cannot_produce_eligible_tasks(tmp_path):
    env = toy_env(tmp_path)
    env.table.get(env.space.configs()[0], env.heldout_seeds[0])["acc"] = float("nan")
    artifact = build_task_set(env)
    assert artifact["status"] == "invalid"
    assert artifact["tasks"] == []
    assert artifact["issues"] == [{"kind": "nonfinite_metric"}]


def test_versioned_task_artifact_roundtrip(tmp_path):
    env = toy_env(tmp_path)
    artifact = build_task_set(env)
    (env.dir / "tasks.json").write_text(json.dumps(artifact))
    loaded = Env.load(env.dir)
    assert loaded.grader_id == GRADER_ID
    assert loaded.tasks == artifact["tasks"]
    assert all(task["seed_ids"] == env.heldout_seeds for task in loaded.tasks)
    summary = Episode(loaded, "claim").summary()
    assert summary["grader_id"] == GRADER_ID
    assert outcome(summary, "test", {})["grader_id"] == GRADER_ID


def test_unversioned_tasks_remain_historical(tmp_path):
    env = toy_env(tmp_path)
    path = env.dir / "tasks.json"
    original = json.dumps([{"id": "claim", "kind": "claim", "answer": "paper_better"}])
    path.write_text(original)
    loaded = Env.load(env.dir)
    assert loaded.grader_id == LEGACY_GRADER_ID
    assert loaded.task("claim")["grader_id"] == LEGACY_GRADER_ID
    assert path.read_text() == original


def test_manifest_cannot_upgrade_unversioned_labels(tmp_path):
    env = toy_env(tmp_path)
    (env.dir / "tasks.json").write_text(json.dumps([{"id": "claim", "kind": "claim", "answer": "paper_better"}]))
    path = env.dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["grader_id"] = GRADER_ID
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="grader"):
        Env.load(env.dir)


def test_mixed_task_graders_are_rejected(tmp_path):
    env = toy_env(tmp_path)
    tasks = env.tasks
    tasks[0]["grader_id"] = LEGACY_GRADER_ID
    (env.dir / "tasks.json").write_text(json.dumps(tasks))
    with pytest.raises(ValueError, match="mixed grader"):
        Env.load(env.dir)


def test_catalog_rejects_a_different_task_grader(tmp_path):
    env = toy_env(tmp_path)
    (env.dir / "lab.py").write_text("def run(config, seed):\n    return {'acc': 0.7}\n")
    catalog = {
        "schema_version": 1,
        "id": "historical",
        "grader_id": LEGACY_GRADER_ID,
        "environments": ["toy"],
        "funnel": {"screened": 1, "selected": 1, "built": 1, "lab_written": 1, "accepted": 1},
    }
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="catalog grader"):
        all_envs(envs_dir=tmp_path)


@pytest.mark.parametrize("name", ["baselines.json", "baselines_tight.json", "rollouts.jsonl"])
def test_report_refuses_historical_scores_for_corrected_tasks(tmp_path, monkeypatch, name):
    env = toy_env(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    row = {"env": "toy", "task": "claim", "kind": "claim", "agent": "uniform", "mean": 1.0, "score": 1.0}
    (results / name).write_text(json.dumps(row) + "\n" if name.endswith(".jsonl") else json.dumps([row]))
    monkeypatch.setattr(report, "env_index", lambda: {"toy": env})
    monkeypatch.setattr(report, "RESULTS", results)
    with pytest.raises(ValueError, match="result grader"):
        report.build_report()
    assert not (results / "summary.json").exists()


def test_cached_sweep_persists_corrected_artifact_and_manifest(tmp_path):
    env = toy_env(tmp_path)
    lab = env.dir / "lab.py"
    lab.write_text("def run(config, seed):\n    return {'acc': 0.7}\n")
    path = env.dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["lab_sha256"] = lab_sha(lab)
    path.write_text(json.dumps(manifest))
    summary = sweep(env.dir, log=lambda *_: None)
    artifact = json.loads((env.dir / "tasks.json").read_text())
    assert summary["grader_id"] == GRADER_ID
    assert summary["task_status"] == "complete"
    assert artifact["grader_id"] == GRADER_ID
    assert artifact["contrasts"]
    assert Env.load(env.dir).grader_id == GRADER_ID


@pytest.mark.parametrize("status", ["incomplete", "invalid"])
def test_pipeline_cannot_admit_invalid_qualification(tmp_path, status):
    env = toy_env(tmp_path)
    ok, reason = accept(env, {"runs_failed": 0, "task_status": status})
    assert not ok
    assert reason == f"invalid_qualification: {status}"


def test_catalog_cannot_silently_keep_a_lab_without_eligible_tasks(tmp_path):
    env = toy_env(tmp_path)
    (env.dir / "lab.py").write_text("def run(config, seed):\n    return {'acc': 0.7}\n")
    artifact = build_task_set(env)
    artifact.update(status="incomplete", tasks=[])
    (env.dir / "tasks.json").write_text(json.dumps(artifact))
    catalog = {
        "schema_version": 1,
        "id": "corrected",
        "grader_id": GRADER_ID,
        "environments": ["toy"],
        "funnel": {"screened": 1, "selected": 1, "built": 1, "lab_written": 1, "accepted": 1},
    }
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="no eligible tasks"):
        all_envs(envs_dir=tmp_path)
