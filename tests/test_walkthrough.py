import copy
import json
import logging
import os
import subprocess
import sys

import pytest

from benchmarks.release_v1 import walkthrough
from benchtop import primary, study


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    folder = tmp_path_factory.mktemp("walkthrough") / "recorded"
    source = walkthrough.ROOT / "envs" / walkthrough.ENVIRONMENT
    before = study._snapshot(source)
    state_existed = primary.canonical_root().exists()
    report = walkthrough.run(folder)
    assert study._snapshot(source) == before
    assert primary.canonical_root().exists() == state_existed
    return folder, report


def test_real_loopback_walkthrough_preserves_private_grading_and_the_original_lab(recorded):
    folder, report = recorded
    assert report["status"] == "passed"
    assert all(report["checks"].values())
    assert report["model_sessions"] == report["lab_workers_started"] == 0
    assert report["qualia_verified"] is False
    assert report["human_admission"] == "not_recorded"
    assert report["fresh_calibration_result"] == "not_collected"
    assert report["evaluator_after_submission"]["spent"] <= report["evaluator_after_submission"]["budget"]
    assert report["agent_events"][-1]["actor"] == "post_commit_probe"
    assert report["agent_events"][-2]["response"] == report["agent_events"][-1]["response"] == {"received": True}
    for event in report["agent_events"]:
        walkthrough.public_only(event["response"])
    assert "/e/" not in json.dumps(report)
    assert not (folder / ".incomplete").exists()
    assert (folder / "report.md").read_text() == walkthrough.render(report)
    assert json.loads((folder / "report.json").read_text()) == report


def test_repeated_walkthrough_has_the_same_observations_answer_grade_and_charge(recorded, tmp_path):
    first = recorded[1]
    second = walkthrough.run(tmp_path / "repeat")
    left, right = (copy.deepcopy(report["agent_events"]) for report in (first, second))
    for events in (left, right):
        events[0]["response"].pop("episode")
    assert left == right
    for key in ("policy", "evaluator_after_submission", "source_qualification_files"):
        assert second[key] == first[key]
    assert second["walkthrough_sha256"] != first["walkthrough_sha256"]


def test_cli_starts_no_model_and_never_prints_the_private_transport_url(tmp_path):
    output = tmp_path / "cli"
    result = subprocess.run([sys.executable, "-m", "benchmarks.release_v1.walkthrough", "--output-dir", str(output)],
        cwd=walkthrough.ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, text=True, capture_output=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["status"] == "passed" and summary["model_sessions"] == 0
    assert not result.stderr and "/e/" not in result.stdout


@pytest.mark.parametrize("private_key", ["reward", "score", "seed_ids", "ground_truth", "qualification_seeds"])
def test_nested_private_fields_cannot_enter_the_agent_transcript(private_key):
    with pytest.raises(ValueError, match="private grading"):
        walkthrough.public_only({"results": [{"nested": {private_key: 1}}]})


def test_existing_report_is_not_overwritten_and_public_source_directory_is_refused(recorded):
    with pytest.raises(FileExistsError):
        walkthrough.run(recorded[0])
    with pytest.raises(ValueError, match="artifacts"):
        walkthrough.run(walkthrough.ROOT / "benchmarks/release_v1/cannot-write-demo")


def test_transport_logging_is_restored_after_a_failed_run():
    before = logging.root.manager.disable
    with pytest.raises(ValueError), walkthrough.quiet_transport():
        assert logging.root.manager.disable == logging.CRITICAL
        raise ValueError("fixture failure")
    assert logging.root.manager.disable == before
