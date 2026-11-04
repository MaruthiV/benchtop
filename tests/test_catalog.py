import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchtop import evaluate, report
from benchtop.stats import GRADER_ID
from tests.test_core import toy_env

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def catalog_tree(tmp_path, monkeypatch):
    root = tmp_path / "envs"
    root.mkdir()
    original = toy_env(root)
    for name in ["other", "unlisted"]:
        target = root / name
        shutil.copytree(original.dir, target)
        manifest = json.loads((target / "manifest.json").read_text())
        manifest["id"] = name
        (target / "manifest.json").write_text(json.dumps(manifest))
    catalog = {
        "schema_version": 1,
        "id": "test_cohort",
        "grader_id": GRADER_ID,
        "environments": ["toy", "other"],
        "funnel": {"screened": 3, "selected": 2, "built": 2, "lab_written": 2, "accepted": 2},
    }
    (root / "catalog.json").write_text(json.dumps(catalog))
    monkeypatch.setattr(evaluate, "ENVS", root)
    monkeypatch.setattr(report, "ENVS", root)
    from benchtop import pipeline

    monkeypatch.setattr(pipeline, "FUNNEL", tmp_path / ".state/pipeline/funnel.jsonl")
    return root, catalog


def test_catalog_controls_evaluation_and_report_without_private_state(catalog_tree):
    root, catalog = catalog_tree
    assert not (root.parent / ".state").exists()
    assert evaluate.accepted_ids() == {"toy", "other"}
    envs = evaluate.all_envs()
    assert [env.id for env in envs] == ["toy", "other"]
    assert evaluate.episode_list() == [(env.id, task["id"]) for env in envs for task in env.tasks]
    assert list(report.env_index()) == ["toy", "other"]
    assert report.funnel_stats() == catalog["funnel"]
    assert {env.id for env in evaluate.all_envs(only_accepted=False)} == {"toy", "other", "unlisted"}


def test_missing_catalog_does_not_fall_back_to_private_logs(catalog_tree):
    root, _ = catalog_tree
    (root / "catalog.json").unlink()
    log = root.parent / ".state/pipeline/funnel.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps({"env": "toy", "outcome": "accepted"}) + "\n")
    with pytest.raises(FileNotFoundError, match="catalog.json"):
        evaluate.all_envs()


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", 2), ("environments", ["toy", "toy"]), ("environments", ["../toy"])],
)
def test_invalid_catalog_membership_is_rejected(catalog_tree, field, value):
    root, catalog = catalog_tree
    catalog[field] = value
    (root / "catalog.json").write_text(json.dumps(catalog))
    with pytest.raises(ValueError):
        evaluate.all_envs()


@pytest.mark.parametrize("missing", ["tasks.json", "results.jsonl"])
def test_missing_selected_lab_artifact_is_rejected(catalog_tree, missing):
    root, _ = catalog_tree
    (root / "toy" / missing).unlink()
    with pytest.raises(FileNotFoundError, match=missing):
        evaluate.all_envs()


def test_selected_lab_identity_must_match_catalog(catalog_tree):
    root, _ = catalog_tree
    path = root / "toy/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["id"] = "unlisted"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity"):
        evaluate.all_envs()


def test_fresh_copy_discovers_labs_and_rebuilds_report(tmp_path):
    output = tmp_path / "foundation.json"
    result = subprocess.run(
        [sys.executable, "-m", "benchtop.cli", "verify", "--output", str(output)],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    actual = json.loads(result.stdout)
    saved = json.loads(output.read_text())
    assert actual == {key: value for key, value in saved.items() if key not in ("input_sha256", "runtime")}
    assert saved["input_sha256"] and saved["runtime"]
    baseline = json.loads((ROOT / "benchmarks/historical_v0/manifest.json").read_text())
    assert actual["offline"]["labs"] == len(baseline["collection"]["paper_environment_ids"])
    assert actual["offline"]["questions"] == baseline["collection"]["paper_tasks"]
    assert actual["offline"]["historical_summary_fields_compared"] == 20
    assert actual["offline"]["private_state_absent"]
    assert actual["offline"]["figure_counts"] == {"svg": 5, "png": 5}
    assert actual["source_inputs_unchanged"]
    assert actual["original_data_files_verified"] == 200
    assert actual["archive_files_verified"] == 241
    assert actual["model_sessions"] == 0
    assert actual["live"]["status"] == "not_requested"
