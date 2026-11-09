import copy
import json
import os
import shutil
import subprocess
import sys

import pytest

from benchtop import (
    cache,
    calibrate,
    primary,
    provider,
    release,
    release_data,
    runner,
    study,
    study_calibration,
)
from benchtop import confirmation_replay as replay
from benchtop import confirmation_results as results
from tests import test_primary
from tests.test_calibrate import fixture as calibration_fixture
from tests.test_confirmation_analysis import document
from tests.test_confirmation_results import no_models, ready
from tests.test_primary import simulate_provider_transport
from tests.test_study import ready_controls


@pytest.fixture(scope="module")
def packaged(tmp_path_factory):
    temporary = tmp_path_factory.mktemp("public-release")
    with pytest.MonkeyPatch.context() as monkeypatch:
        controls = test_primary.ready_controls

        def frozen_controls(path):
            reviews, foundation = controls(path)
            value = study._json(foundation)
            value["input_sha256"] = release.public_inputs()
            foundation.write_text(json.dumps(value))
            return reviews, foundation

        monkeypatch.setattr(test_primary, "ready_controls", frozen_controls)
        development = calibration_fixture(phase="development")
        selected = calibrate.select_budget(development)
        assert selected["chosen_budget_fraction"] == 0.12
        snapshot = study._seal({"input": development, "selection": selected,
            "scope": "TEST FIXTURE ONLY; development authority stubbed"}, "snapshot_sha256")
        monkeypatch.setattr(study_calibration, "load_closed", lambda **kwargs: copy.deepcopy(snapshot))
        directory, plan = ready(temporary, monkeypatch)
        monkeypatch.setattr(provider, "execute", lambda *args: simulate_provider_transport(*args, answer="paper_better"))
        with runner.Controller() as controller:
            outcome = runner.run_provider(plan["plan_sha256"], plan["slots"][0]["slot_id"], controller=controller)
        assert outcome["status"] == "completed"
        results.collect(plan["plan_sha256"], temporary / "closed.json", close=True)
        play = replay.play_release

        def fail_two_trials(episode, policy, seed, **kwargs):
            if policy == "paired_uniform" and seed == 3:
                raise RuntimeError("synthetic policy failure")
            if policy == "paired_uniform" and seed == 4:
                return {"stop_reason": "partial_reply"}
            return play(episode, policy, seed, **kwargs)

        monkeypatch.setattr(replay, "play_release", fail_two_trials)
        classical = replay.run()
        assert classical["status"] == "complete"
        assert sum(row["state"] != "completed" for group in classical["groups"] for row in group["outcomes"]) == 2
        no_models(monkeypatch)
        before = release._private_inventory(primary.canonical_root())
        original = release.public_inputs()
        output = temporary / "package"
        built = release.build(output)
        assert release._private_inventory(primary.canonical_root()) == before
        assert release.public_inputs() == original
        yield {"output": output, "built": built, "plan": plan, "temporary": temporary}


def test_build_preserves_evidence_and_reproduces_every_completed_cached_trial(packaged):
    output, built = packaged["output"], packaged["built"]
    assert built["status"] == "built" and built["model_sessions_started"] == 0
    assert built["email_ready"] is False
    result = release.reproduce(output)
    assert result["model_grades_rechecked"] == 1
    assert result["budget_selections_rechecked"] == 2
    assert result["cached_policy_trials_replayed"] == 1198
    assert result["recorded_policy_failures_or_missing_preserved"] == 2
    assert result["model_sessions_started"] == result["lab_workers_started"] == 0
    assert result["execution_authorized"] is False and result["bitwise_analysis_match"]
    assert (output / "evaluation/README.md").read_bytes() == (release.ROOT / "README.md").read_bytes()
    assert "TEST FIXTURE ONLY" in (output / "README.md").read_text()
    assert not any((output / name).exists() for name in (".state", ".git", "artifacts", "docs"))
    assert (output / "release-results/figures/confirmation.svg").stat().st_size > 1000
    with pytest.raises(ValueError, match="new directory"):
        release.build(output)


def test_package_runs_from_its_own_code_without_private_state(packaged):
    output = packaged["output"]
    environment = {**os.environ, "PYTHONPATH": str(output), "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run([sys.executable, "-m", "benchtop.cli", "release", "reproduce", "."],
        cwd=output, env=environment, capture_output=True, text=True, timeout=120, check=False)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["cached_policy_trials_replayed"] == 1198
    assert result["recorded_policy_failures_or_missing_preserved"] == 2
    assert result["model_sessions_started"] == result["lab_workers_started"] == 0


@pytest.mark.parametrize("defect", ["modified", "extra", "missing", "symlink", "partial"])
def test_changed_or_partial_package_is_refused(packaged, tmp_path, defect):
    target = tmp_path / "copy"
    shutil.copytree(packaged["output"], target)
    if defect == "modified":
        (target / "README.md").write_text("changed\n")
    elif defect == "extra":
        (target / "unexpected.txt").write_text("extra")
    elif defect == "missing":
        (target / "release-results/analysis.json").unlink()
    elif defect == "symlink":
        (target / "link").symlink_to(target / "README.md")
    else:
        (target / ".release-building").write_text("incomplete")
    with pytest.raises(ValueError, match="changed|symbolic|incomplete"):
        release.reproduce(target, replay_classical=False)


def test_recomputed_truth_detects_resealed_cross_partition_confirmation_data(packaged, tmp_path):
    projection = study._json(packaged["output"] / "release-results/evidence.json")
    shutil.copytree(packaged["output"] / "release-data", tmp_path / "data")
    member = projection["members"][0]
    folder = tmp_path / "data" / member["env_id"] / "confirmation"
    rows = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()]
    rows[0]["seed"] = 0
    wrapped = study._json(folder / "cache.json")
    space = release._environment(tmp_path / "data" / member["env_id"] / "qualification").space
    rows[0] = cache.seal_result(wrapped["execution"], space, rows[0]["config"], 0, rows[0])
    (folder / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="cross-partition"):
        release_data.reproduce(projection, tmp_path / "data", replay_classical=False)


def test_resealed_budget_selection_and_model_reward_tampering_are_regraded(packaged):
    projection = study._json(packaged["output"] / "release-results/evidence.json")
    changed = copy.deepcopy(projection)
    changed["calibration"]["development"]["selection"]["chosen_budget_fraction"] = 0.25
    changed["calibration"]["development"]["selection"] = study._seal(changed["calibration"]["development"]["selection"], "selection_sha256")
    changed = study._seal(changed, "projection_sha256")
    with pytest.raises(ValueError, match="budget_selection"):
        release_data.reproduce(changed, packaged["output"] / "release-data", replay_classical=False)
    changed = copy.deepcopy(projection)
    slot = next(slot for slot in changed["collection"]["slots"] if slot["attempts"])
    slot["details"][-1]["answer"] = "paper_worse"
    changed["collection"] = study._seal(changed["collection"], "snapshot_sha256")
    changed["classical"]["collection_sha256"] = changed["collection"]["snapshot_sha256"]
    changed["classical"] = study._seal(changed["classical"], "summary_sha256")
    changed = study._seal(changed, "projection_sha256")
    with pytest.raises(ValueError, match="model reward"):
        release_data.reproduce(changed, packaged["output"] / "release-data", replay_classical=False)


@pytest.mark.parametrize("text", ["/Users/test/private.json", "file://secret", ".state/run/receipt", "Bearer " + "x" * 20,
    "https://127.0.0.1/e/" + "a" * 12 + "/" + "b" * 43 + "/mcp", "https://user:password@example.com"])
def test_private_text_is_refused(text):
    with pytest.raises(ValueError, match="private path"):
        release_data.safe_text(text)


def test_invalid_raw_answer_is_omitted_from_the_new_public_projection():
    detail = {"attempt_id": "a" * 64, "answer": "private text /Users/test/account", "confidence": None,
        "observed_trace": [{"index": 0, "tool": "describe", "error": False, "response_keys": ["title"], "private_field": "secret"}]}
    value = release_data._detail(detail)
    assert value["answer"] is None and value["confidence"] is None
    assert "private_field" not in value["observed_trace"][0]
    release_data.safe_text(value)
    assert detail["answer"].startswith("private text")


def test_review_projection_keeps_public_citations_and_omits_private_references():
    citation = {"source": "https://arxiv.org/abs/2611.00001v2", "locator": "section 3", "finding": "paper comparison"}
    assert release._review_evidence(citation) == citation
    private = {**citation, "source": ".state/review/source.html"}
    public = release._review_evidence(private)
    assert "source" not in public
    assert public["original_private_source_reference_sha256"] == study.canonical_hash(private["source"])


def test_empty_public_projection_retains_unavailable_comparison_without_policies(packaged, tmp_path):
    projection = study._json(packaged["output"] / "release-results/evidence.json")
    projection["collection"] = document()
    projection["members"] = []
    projection["calibration"]["pilots"] = []
    classical = projection["classical"]
    classical.update(groups=[], collection_sha256=projection["collection"]["snapshot_sha256"])
    projection["classical"] = study._seal(classical, "summary_sha256")
    projection = study._seal(projection, "projection_sha256")
    report, counts = release_data.reproduce(projection, tmp_path / "no-data")
    assert report["primary"]["estimate"] is report["primary"]["full_cohort_bounds"] is None
    assert counts["cached_policy_trials_replayed"] == counts["model_grades_rechecked"] == 0


def test_missing_canonical_state_has_read_only_status_and_cannot_build(tmp_path, monkeypatch):
    monkeypatch.setattr(primary, "STATE_BASE", tmp_path / "state")
    state = release.status()
    assert state["status"] == "not_ready" and state["email_ready"] is False
    assert state["model_sessions_started"] == 0
    with pytest.raises((ValueError, FileNotFoundError)):
        release.build(tmp_path / "unavailable")
    assert not (tmp_path / "state").exists()
    assert not (tmp_path / "unavailable").exists()


def test_recorded_unresolved_review_counts_as_reviewed_without_admitting_the_lab(tmp_path, monkeypatch):
    reviews, _ = ready_controls(tmp_path)
    path = next(reviews.glob("*.json"))
    value = study._json(path)
    value["decision"] = "needs_review"
    for check in value["checks"].values():
        check["status"] = "unresolved"
    path.write_text(json.dumps(value))
    workspace = tmp_path / "workspace"
    (workspace / "artifacts").mkdir(parents=True)
    shutil.copytree(reviews, workspace / "artifacts/reference-human-reviews")
    (workspace / "envs").symlink_to(release.ROOT / "envs", target_is_directory=True)
    monkeypatch.setattr(release, "ROOT", workspace)
    monkeypatch.setattr(primary, "STATE_BASE", tmp_path / "missing-state")
    status = release.status()
    assert status["gates"]["six_recorded_reference_reviews"] is True
    assert status["email_ready"] is False
    inspected = release.admission.inspect_lab(workspace / "envs" / path.stem, review=value)
    assert inspected["human_review"]["status"] == "recorded"
    assert inspected["qualified_claim_admitted"] is False


def test_empty_and_missing_group_figures_are_explicit(packaged, tmp_path):
    report = study._json(packaged["output"] / "release-results/analysis.json")
    report["labs"] = []
    report["denominators"]["complete_paired_labs"] = 0
    release.figures(report, tmp_path / "empty")
    assert "No admitted labs" in (tmp_path / "empty/confirmation.svg").read_text()
