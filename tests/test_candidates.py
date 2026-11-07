import copy
import json
from datetime import timedelta

import pytest

from benchtop import candidate_metadata as metadata
from benchtop import candidates, primary, provider, replay, study, study_calibration
from benchtop.agents.estimator import ESTIMATOR_ID
from benchtop.cli import main
from tests.test_candidate_metadata import AT, DECLARATION, Client, controls, feed
from tests.test_provider import preflight
from tests.test_study import model, ready_controls


def review(tmp_path):
    evidence = tmp_path / "test-review-evidence.md"
    evidence.write_text("TEST FIXTURE ONLY: simulated baseline review and sensitivity evidence.")
    return {"schema_version": 1, "decision": "accepted",
        "reviewer": {"kind": "human", "name": "Simulated reviewer for tests only", "reviewed_at_utc": AT.isoformat()},
        "checks": {name: {"judgment": "accepted", "rationale": "Test control; no actual scientific endorsement.",
                          "evidence": [{"source": study._ref(evidence), "sha256": study.cache.file_hash(evidence)}]}
                   for name in candidates.FREEZE_CHECKS},
        "adaptive_allocation": {"decision": "deferred", "rationale": "Bounded release scope in this test."},
        "replay_directory": study._ref(tmp_path / "classical"), "replay_summary_sha256": "1" * 64}


def setup(tmp_path, monkeypatch):
    controls(monkeypatch)
    monkeypatch.setattr(primary, "STATE_BASE", tmp_path / "private-state")
    monkeypatch.setattr(primary, "BINDINGS", tmp_path / "public/bindings.json")
    root = primary.canonical_root()
    (root / "ledger/plans").mkdir(parents=True)
    plan_sha = "2" * 64
    plan = {"binding": {"reviews_source": study._ref(tmp_path / "test-reviews"),
                        "environments": [{"env_id": f"test-lab-{index}"} for index in range(4)]}}
    study._publish(root / "ledger/plans" / f"{plan_sha}.json", plan)
    study._publish(root / "provider.json", preflight())
    closed = {"plan_sha256": plan_sha, "snapshot_sha256": "3" * 64,
              "selection": {"status": "selected", "closed": True, "chosen_budget_fraction": 0.12,
                            "labs": [{"budgets": [{"missing": 0}]}]}}
    monkeypatch.setattr(study_calibration, "load_closed", lambda: copy.deepcopy(closed))

    def simulated_baseline(freeze, development, at):
        return {"review": freeze, "estimator_id": ESTIMATOR_ID, "policies": ["paired_uniform"],
                "evidence": {str(tmp_path / "test-review-evidence.md"): study.cache.file_hash(tmp_path / "test-review-evidence.md")}}

    monkeypatch.setattr(candidates, "_baselines", simulated_baseline)
    monkeypatch.setattr(provider, "probe", lambda: pytest.fail("candidate lock queried live authentication"))
    monkeypatch.setattr(provider, "execute", lambda *a: pytest.fail("candidate lock started a model"))
    archive = tmp_path / "archive"
    metadata.collect(archive, DECLARATION, client=Client({("2026-11-06", 0): feed("2026-11-06", [{"id": "2611.00003", "version": 2}])}))
    review_path = tmp_path / "baseline-review.json"
    review_path.write_text(json.dumps(review(tmp_path)))
    models = {name: model() for name in ("triage", "synthesis", "repair")}
    return root, archive, review_path, models, closed


def test_candidate_lock_preserves_budget_paper_versions_rules_and_public_history(tmp_path, monkeypatch):
    root, archive, review_path, models, _ = setup(tmp_path, monkeypatch)
    primary.BINDINGS.parent.mkdir()
    earlier = {"id": "development_lock", "record_sha256": "4" * 64}
    primary.BINDINGS.write_text(json.dumps({"schema_version": 1, "protocol_hash": study.canonical_hash(study._json(study.SPEC)), "records": [earlier]}))
    record = candidates.lock_candidates(archive, review_path, models)
    assert record["study_lock"] is True and record["generation_ready"] is False and record["model_sessions_started"] == 0
    assert record["binding"]["global_budget_fraction"] == 0.12
    assert record["binding"]["candidates"][0]["version"] == "2611.00003v2"
    rules = record["binding"]["generation"]
    assert rules["generation_attempts_per_candidate"] == 1 and rules["max_repairs_per_candidate"] == 2
    assert set(rules["prompts"]) == {"triage", "synthesis", "repair"}
    assert len(rules["seed_partitions"]["confirmation"]) == 12
    assert study._json(primary.BINDINGS)["records"][0] == earlier
    assert candidates.verify() == record
    before = (root / "candidate_lock.json").read_bytes()
    with pytest.raises(FileExistsError):
        candidates.lock_candidates(archive, review_path, models)
    assert (root / "candidate_lock.json").read_bytes() == before
    monkeypatch.setattr(metadata, "now", lambda: AT + timedelta(days=1))
    monkeypatch.setattr(metadata, "history", lambda: {"ids": ["2611.00003"], "sources": {"newly_generated": "test"}})
    assert candidates.verify() == record


@pytest.mark.parametrize("defect", ["raw", "review", "selection", "sources", "public", "generation_flag"])
def test_changed_locked_evidence_cannot_authorize_the_next_phase(tmp_path, monkeypatch, defect):
    root, archive, review_path, models, closed = setup(tmp_path, monkeypatch)
    candidates.lock_candidates(archive, review_path, models)
    if defect == "raw":
        path = next((root / "candidate_metadata/requests").glob("*.xml"))
        path.write_bytes(path.read_bytes() + b" ")
    elif defect == "review":
        (root / "candidate_baseline_review.json").write_text("{}")
    elif defect == "selection":
        closed["selection"]["chosen_budget_fraction"] = 0.25
    elif defect == "sources":
        monkeypatch.setattr(study, "_sources", lambda: {"changed": "test"})
    elif defect == "public":
        primary.BINDINGS.unlink()
    else:
        record = study._json(root / "candidate_lock.json")
        record["generation_ready"] = True
        study_path = root / "candidate_lock.json"
        study_path.write_text(json.dumps(study._seal(record, "record_sha256")))
    with pytest.raises((ValueError, FileNotFoundError)):
        candidates.verify()


@pytest.mark.parametrize("defect", ["incomplete", "open", "model", "history", "stale_date"])
def test_candidate_prerequisites_fail_before_any_lock_is_published(tmp_path, monkeypatch, defect):
    root, archive, review_path, models, closed = setup(tmp_path, monkeypatch)
    if defect == "incomplete":
        closed["selection"]["status"] = "incomplete_development"
        closed["selection"]["chosen_budget_fraction"] = None
    elif defect == "open":
        closed["selection"]["closed"] = False
    elif defect == "model":
        models["repair"]["cli_version"] = "different-cli"
    elif defect == "history":
        monkeypatch.setattr(metadata, "history", lambda: {"ids": ["2611.00001"], "sources": {}})
    else:
        monkeypatch.setattr(metadata, "now", lambda: AT + timedelta(days=1))
    with pytest.raises(ValueError):
        candidates.lock_candidates(archive, review_path, models)
    assert not (root / "candidate_lock.json").exists() and not primary.BINDINGS.exists()


def test_real_missing_development_gate_is_read_only_and_cannot_use_a_supplied_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(primary, "STATE_BASE", tmp_path / "absent-private")
    with pytest.raises(FileNotFoundError):
        candidates.lock_candidates(tmp_path / "fake-archive", tmp_path / "fake-review", {})
    assert not (tmp_path / "absent-private").exists()


def test_human_freeze_requires_named_dated_review_and_unchanged_local_evidence(tmp_path):
    freeze = review(tmp_path)
    assert candidates._review(freeze, AT)
    for defect in ("unassigned", "blank", "future", "unresolved", "missing_evidence", "adaptive"):
        value = copy.deepcopy(freeze)
        if defect == "unassigned":
            value["reviewer"]["kind"] = "assistant"
        elif defect == "blank":
            value["reviewer"]["name"] = ""
        elif defect == "future":
            value["reviewer"]["reviewed_at_utc"] = (AT + timedelta(days=1)).isoformat()
        elif defect == "unresolved":
            value["checks"]["estimator_sensitivity"]["judgment"] = "unresolved"
        elif defect == "missing_evidence":
            value["checks"]["policy_correctness"]["evidence"] = []
        else:
            value["adaptive_allocation"]["decision"] = "included"
        with pytest.raises(ValueError):
            candidates._review(value, AT)
    (tmp_path / "test-review-evidence.md").write_text("changed evidence")
    with pytest.raises(ValueError, match="evidence changed"):
        candidates._review(freeze, AT)


def test_baseline_reader_checks_summary_and_raw_journal_under_its_lock(tmp_path, monkeypatch):
    freeze = review(tmp_path)
    directory = tmp_path / "classical"
    (directory / "attempts").mkdir(parents=True)
    (directory / ".replay.lock").touch()
    spec = study._json(study.SPEC)
    labs = [f"test-lab-{index}" for index in range(4)]
    plan = {"plan_id": "5" * 64, "binding": {"mode": "development", "policies": spec["baselines"]["required"] + spec["baselines"]["sanity_controls"],
              "environments": [{"fingerprint": {"env": lab}} for lab in labs], "majority_label": "paper_better"}}
    (directory / "plan.json").write_text(json.dumps(plan))
    summary = {"slots": [{"status": "completed"}], "scope": "SIMULATED replay API boundary only"}
    freeze["replay_summary_sha256"] = study.canonical_hash(summary)
    validated = []
    monkeypatch.setattr(replay, "_validate", lambda plan, **kw: validated.append(plan))
    monkeypatch.setattr(replay, "summarize", lambda *a: copy.deepcopy(summary))
    development = {"binding": {"reviews_source": study._ref(tmp_path / "reviews"), "environments": [{"env_id": lab} for lab in labs]}}
    bound = candidates._baselines(freeze, development, AT)
    assert validated == [plan] and bound["estimator_id"] == ESTIMATOR_ID
    assert bound["development_majority_label"] == "paper_better" and str(directory / "plan.json") in bound["evidence"]
    summary["slots"][0]["status"] = "interrupted"
    freeze["replay_summary_sha256"] = study.canonical_hash(summary)
    with pytest.raises(ValueError, match="full development replay"):
        candidates._baselines(freeze, development, AT)
    summary["slots"][0]["status"] = "completed"
    freeze["replay_summary_sha256"] = study.canonical_hash(summary)
    (directory / "attempts/test.finish.json").write_text(json.dumps({"finished_at_utc": (AT + timedelta(days=1)).isoformat()}))
    with pytest.raises(ValueError, match="review predates"):
        candidates._baselines(freeze, development, AT)
    plan["binding"]["mode"] = "engineering"
    (directory / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="engineering replay"):
        candidates._baselines(freeze, development, AT)


def test_actual_full_development_plan_without_200_seed_outcomes_cannot_be_frozen(tmp_path):
    reviews, _ = ready_controls(tmp_path)
    plan = replay.make_plan(mode="development", reviews_dir=reviews)
    assert len(plan["slots"]) == 14400
    directory = tmp_path / "classical"
    summary = replay.run_plan(plan, directory, reviews_dir=reviews, max_attempts=0)
    summary.pop("attempts_invoked_this_run")
    freeze = review(tmp_path)
    freeze["replay_summary_sha256"] = study.canonical_hash(summary)
    development = {"binding": {"reviews_source": study._ref(reviews), "environments": [
        {"env_id": row["fingerprint"]["env"]} for row in plan["binding"]["environments"]]}}
    with pytest.raises(ValueError, match="full development replay"):
        candidates._baselines(freeze, development, AT)
    assert not (directory / "attempts").exists() or not list((directory / "attempts").iterdir())


def test_candidate_cli_reports_frozen_scope_without_generation(tmp_path, monkeypatch, capsys):
    _, archive, review_path, models, _ = setup(tmp_path, monkeypatch)
    model_path = tmp_path / "generation-models.json"
    model_path.write_text(json.dumps(models))
    main(["study", "lock-candidates", str(archive), "--baseline-review", str(review_path), "--generation-models", str(model_path)])
    result = json.loads(capsys.readouterr().out)
    assert result["id"] == "candidate_lock" and result["generation_ready"] is False
    main(["study", "inspect-candidates"])
    assert json.loads(capsys.readouterr().out) == result


def test_candidate_snapshot_copies_only_the_bound_archive_files(tmp_path, monkeypatch):
    root, archive, review_path, models, _ = setup(tmp_path, monkeypatch)
    (archive / "unrelated-private.txt").write_text("unrelated fixture data")
    candidates.lock_candidates(archive, review_path, models)
    assert not (root / "candidate_metadata/unrelated-private.txt").exists()
    assert candidates.verify()["id"] == "candidate_lock"
