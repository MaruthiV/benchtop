import copy
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest

from benchtop import session_ledger as ledger
from benchtop import study
from benchtop.cli import main
from benchtop.env import ROOT
from tests.test_study import model, plan

EVIDENCE = study.canonical_hash({"synthetic_control_receipt": True})


def fixture(tmp_path, *, phase="development"):
    env, prepared = plan(tmp_path, phase=phase)
    root = tmp_path / "sessions"
    ledger.create(prepared, root)
    return root, prepared, env


def begin(root, prepared, index=0, **kwargs):
    return ledger.begin(root, prepared["plan_sha256"], prepared["slots"][index]["slot_id"], **kwargs)


def finish(root, start, *, state="completed", reward=0, cause=None, submission=None, costs=None, **extra):
    outcome = {"state": state, "reward": reward}
    if cause:
        outcome["failure_category"] = cause
    if costs is not None:
        outcome["costs"] = costs
    evidence = {
        "reported_model_id": model()["reported_model_id"],
        "process_stopped": True,
        "submission_status": submission
        or ("committed" if state == "completed" else "unknown" if state == "unresolved" else "not_committed"),
        "evidence_sha256": EVIDENCE,
        **extra,
    }
    return ledger.finish(root, start["attempt_id"], outcome, **evidence)


def reconcile(root, start, *, state="completed", reward=0, cause=None, submission="committed", costs=None):
    outcome = {"state": state, "reward": reward}
    if cause:
        outcome["failure_category"] = cause
    if costs is not None:
        outcome["costs"] = costs
    return ledger.reconcile(
        root,
        start["attempt_id"],
        outcome,
        reported_model_id=model()["reported_model_id"],
        process_stopped=True,
        submission_status=submission,
        evidence_sha256=EVIDENCE,
    )


def test_initialization_registration_and_empty_snapshot_are_bound(tmp_path):
    root, prepared, _ = fixture(tmp_path)
    result = ledger.inspect(root)
    assert result["scope"] == "engineering_only" and result["registered_plans"] == 1
    assert result["counts"]["total"] == result["counts"]["active"] == 0
    assert result["status_counts"] == {"unattempted": 12}
    assert result["model_sessions_started_by_ledger"] == 0
    assert ledger.register(prepared, root) == prepared["plan_sha256"]
    assert ledger.inspect(root) == result
    assert (root.stat().st_mode & 0o777) == 0o700
    assert result["costs"]["fields"]["input_tokens"]["total"] == 0


def test_completed_wrong_answer_is_immutable_and_never_reinvoked(tmp_path):
    root, prepared, _ = fixture(tmp_path)
    start = begin(root, prepared)
    record = finish(root, start)
    before = (root / "attempts" / f"{start['attempt_id']}.finish.json").read_bytes()
    assert finish(root, start) == record
    with pytest.raises(ValueError, match="immutable completion"):
        finish(root, start, reward=1)
    with pytest.raises(ValueError, match="final or requires reconciliation"):
        begin(root, prepared, retry_technical=True)
    assert (root / "attempts" / f"{start['attempt_id']}.finish.json").read_bytes() == before
    row = ledger.inspect(root)["slots"][0]
    assert row["state"] == "completed" and row["reward"] == 0


def test_pending_processes_consume_two_shared_slots_and_never_expire(tmp_path):
    root, prepared, env = fixture(tmp_path)
    smoke = study.make_plan([env.dir], model=model(), phase="smoke")
    ledger.register(smoke, root)
    first, _ = begin(root, prepared), begin(root, smoke)
    with pytest.raises(ValueError, match="two in-flight"):
        begin(root, prepared, 1)
    with pytest.raises(ValueError, match="requires reconciliation"):
        begin(root, smoke)
    assert ledger.inspect(root)["counts"]["active"] == 2
    with pytest.raises(ValueError, match="process stopped"):
        finish(root, first, process_stopped=False)
    assert ledger.inspect(root)["counts"]["active"] == 2
    finish(root, first, state="unresolved", reward=None, cause="controlled_interruption")
    begin(root, prepared, 1)
    assert ledger.inspect(root)["counts"]["active"] == 2


def test_unknown_submission_blocks_retry_until_explicit_reconciliation(tmp_path):
    root, prepared, _ = fixture(tmp_path)
    start = begin(root, prepared)
    finish(
        root,
        start,
        state="unresolved",
        reward=None,
        cause="controlled_transport_uncertainty",
        costs={"charged_experiment_cost": 2},
    )
    raw = (root / "attempts" / f"{start['attempt_id']}.finish.json").read_bytes()
    with pytest.raises(ValueError, match="requires reconciliation"):
        begin(root, prepared, retry_technical=True)
    resolution = reconcile(root, start, reward=0, costs={"input_tokens": 7})
    assert reconcile(root, start, reward=0, costs={"input_tokens": 7}) == resolution
    assert (root / "attempts" / f"{start['attempt_id']}.finish.json").read_bytes() == raw
    result = ledger.inspect(root)
    assert result["counts"]["total"] == 1 and result["slots"][0]["reward"] == 0
    assert result["costs"]["fields"]["charged_experiment_cost"]["total"] == 2
    assert result["costs"]["fields"]["input_tokens"]["total"] == 7
    with pytest.raises(ValueError, match="final or requires reconciliation"):
        begin(root, prepared, retry_technical=True)


def test_technical_retry_requires_known_no_submission_and_is_bounded(tmp_path):
    root, prepared, _ = fixture(tmp_path)
    start = begin(root, prepared)
    with pytest.raises(ValueError, match="submission reconciliation"):
        finish(
            root,
            start,
            state="technical_failure",
            reward=None,
            cause="transport_failure",
            submission="unknown",
        )
    finish(root, start, state="unresolved", reward=None, cause="unclassified_transport")
    reconcile(
        root,
        start,
        state="technical_failure",
        reward=None,
        cause="transport_failure",
        submission="not_committed",
    )
    with pytest.raises(ValueError, match="explicit technical retry"):
        begin(root, prepared)
    retry = begin(root, prepared, retry_technical=True)
    assert retry["attempt_index"] == 1 and retry["attempt_id"] == prepared["slots"][0]["attempt_ids"][1]
    finish(root, retry, state="technical_failure", reward=None, cause="provider_quota")
    with pytest.raises(ValueError, match="at most one"):
        begin(root, prepared, retry_technical=True)
    result = ledger.inspect(root)
    assert result["counts"]["total"] == 2 and result["counts"]["retry"] == 1
    assert result["slots"][0]["reward"] is None


def test_agent_failure_is_terminal_zero_and_unknown_costs_stay_unknown(tmp_path):
    root, prepared, _ = fixture(tmp_path)
    start = begin(root, prepared)
    finish(root, start, state="agent_failure", reward=0, cause="episode_timeout")
    result = ledger.inspect(root)
    assert result["slots"][0]["state"] == "agent_failure" and result["slots"][0]["reward"] == 0
    assert result["costs"]["fields"]["model_wall_seconds"]["total"] is None
    assert result["costs"]["fields"]["model_wall_seconds"]["known_sum"] == 0
    with pytest.raises(ValueError):
        begin(root, prepared, retry_technical=True)


def test_missing_model_identity_retains_raw_completion_until_reconciled(tmp_path):
    root, prepared, _ = fixture(tmp_path)
    start = begin(root, prepared)
    record = finish(root, start, reward=1, reported_model_id=None)
    assert record["claimed_outcome"]["reward"] == 1 and record["outcome"]["reward"] is None
    assert not ledger.inspect(root)["halted"]
    reconcile(root, start, reward=1)
    assert ledger.inspect(root)["slots"][0]["reward"] == 1


def test_reported_model_change_preserves_diagnostics_and_halts_new_work(tmp_path):
    root, prepared, _ = fixture(tmp_path)
    first, second = begin(root, prepared), begin(root, prepared, 1)
    record = finish(root, first, reward=1, reported_model_id="different-synthetic-model")
    assert (
        record["model_mismatch"]
        and record["claimed_outcome"]["reward"] == 1
        and record["outcome"]["reward"] is None
    )
    with pytest.raises(ValueError, match="halted"):
        begin(root, prepared, 2)
    finish(root, second)
    with pytest.raises(ValueError, match="model mismatch"):
        reconcile(root, first, reward=1)
    result = ledger.inspect(root)
    assert result["halted"] and result["counts"]["active"] == 0


@pytest.mark.parametrize(
    "bucket,phase,index,maximum",
    [
        ("smoke", "smoke", 0, 6),
        ("integration", "integration", 0, 2),
        ("retry", "development", 1, 20),
        ("primary", "development", 0, 272),
        ("phase:development", "development", 0, 48),
        ("phase:fresh_pilot", "fresh_pilot", 0, 96),
        ("phase:confirmation", "confirmation", 0, 128),
        ("total", "development", 0, 300),
    ],
)
def test_all_protocol_allowances_are_inclusive_without_off_by_one(tmp_path, bucket, phase, index, maximum):
    root, _, _ = fixture(tmp_path)
    identity = json.loads((root / "ledger.json").read_text())["identity"]
    counts = Counter({bucket: maximum - 1})
    ledger._capacity(counts, phase, index, identity)
    counts[bucket] += 1
    with pytest.raises(ValueError, match="allowance exhausted"):
        ledger._capacity(counts, phase, index, identity)


def test_concurrent_writers_cannot_claim_more_than_two_slots(tmp_path):
    root, prepared, _ = fixture(tmp_path)

    def attempt(index):
        try:
            return begin(root, prepared, index)["attempt_id"]
        except ValueError as error:
            assert "two in-flight" in str(error)
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(attempt, range(4)))
    assert sum(item is not None for item in result) == 2
    assert ledger.inspect(root)["counts"]["total"] == 2


def test_shared_smoke_limit_cannot_be_reset_by_registering_another_plan(tmp_path):
    root, first, env = fixture(tmp_path, phase="smoke")
    other_path = ROOT / "envs/2610-36935-gaussian-process-bandits-for"
    second = study.make_plan([env.dir, other_path], model=model())
    ledger.register(second, root)
    for prepared, count in ((first, 3), (second, 3)):
        for index in range(count):
            finish(root, begin(root, prepared, index))
    with pytest.raises(ValueError, match="smoke start-intent"):
        begin(root, second, 3)
    assert ledger.inspect(root)["counts"]["smoke"] == 6


def test_retry_reserve_is_shared_across_every_registered_slot(tmp_path):
    root, _, env = fixture(tmp_path)
    reference = ROOT / "envs/2610-36935-gaussian-process-bandits-for"
    prepared = study.make_plan([env.dir, reference], model=model(), phase="development")
    ledger.register(prepared, root)
    for index in range(20):
        first = begin(root, prepared, index)
        finish(root, first, state="technical_failure", reward=None, cause="provider_quota")
        retry = begin(root, prepared, index, retry_technical=True)
        finish(root, retry, state="technical_failure", reward=None, cause="provider_quota")
    first = begin(root, prepared, 20)
    finish(root, first, state="technical_failure", reward=None, cause="provider_quota")
    with pytest.raises(ValueError, match="retry start-intent allowance exhausted"):
        begin(root, prepared, 20, retry_technical=True)
    assert ledger.inspect(root)["counts"]["retry"] == 20


@pytest.mark.parametrize("defect", ["primary", "model", "runtime"])
def test_incompatible_or_primary_plan_registration_is_refused(tmp_path, monkeypatch, defect):
    root, prepared, env = fixture(tmp_path)
    if defect == "primary":
        prepared = copy.deepcopy(prepared)
        prepared["binding"]["mode"] = "study_preparation"
    elif defect == "model":
        changed = model()
        changed["requested_model_id"] = changed["reported_model_id"] = "different-model"
        prepared = study.make_plan([env.dir], model=changed)
    else:
        original = study.cache.runtime_identity
        monkeypatch.setattr(study.cache, "runtime_identity", lambda: {**original(), "python": "changed"})
    with pytest.raises(ValueError):
        ledger.register(prepared, root)
    assert ledger.inspect(root)["registered_plans"] == 1


@pytest.mark.parametrize(
    "defect", ["unknown_cause", "technical_zero", "completed_without_submission", "bad_cost", "no_evidence"]
)
def test_invalid_completion_cannot_release_capacity(tmp_path, defect):
    root, prepared, _ = fixture(tmp_path)
    start = begin(root, prepared)
    options = {"state": "technical_failure", "reward": None, "cause": "provider_quota"}
    if defect == "unknown_cause":
        options["cause"] = "not_allowlisted"
    elif defect == "technical_zero":
        options["reward"] = 0
    elif defect == "completed_without_submission":
        options = {"submission": "not_committed"}
    elif defect == "bad_cost":
        options["costs"] = {"input_tokens": True}
    else:
        options["evidence_sha256"] = "not-a-checksum"
    with pytest.raises(ValueError):
        finish(root, start, **options)
    assert ledger.inspect(root)["counts"]["active"] == 1


def test_reconciliation_cannot_change_known_costs_or_rewrite_final_answer(tmp_path):
    root, prepared, _ = fixture(tmp_path)
    start = begin(root, prepared)
    finish(
        root,
        start,
        state="unresolved",
        reward=None,
        cause="controlled_uncertainty",
        costs={"input_tokens": 7},
    )
    with pytest.raises(ValueError, match="reported cost"):
        reconcile(root, start, costs={"input_tokens": 1})
    reconcile(root, start, costs={"input_tokens": 7})
    with pytest.raises(ValueError, match="immutable completion"):
        reconcile(root, start, reward=1, costs={"input_tokens": 7})


@pytest.mark.parametrize("defect", ["checksum", "orphan", "unknown_file", "limits"])
def test_corrupt_or_orphaned_published_records_fail_visibly(tmp_path, defect):
    root, prepared, _ = fixture(tmp_path)
    start = begin(root, prepared)
    record = finish(root, start)
    path = root / "attempts" / f"{start['attempt_id']}.finish.json"
    if defect == "checksum":
        record["outcome"]["reward"] = 1
        path.write_text(json.dumps(record))
    elif defect == "orphan":
        (root / "attempts" / f"{start['attempt_id']}.start.json").unlink()
    elif defect == "unknown_file":
        (root / "attempts/extra.json").write_text("{}")
    else:
        manifest = json.loads((root / "ledger.json").read_text())
        manifest["identity"]["limits"]["total_max_episode_starts"] = 999
        (root / "ledger.json").write_text(json.dumps(study._seal(manifest, "record_sha256")))
    with pytest.raises((ValueError, KeyError)):
        ledger.inspect(root)


def test_stale_code_blocks_new_work_but_stored_history_is_auditable(tmp_path, monkeypatch):
    root, prepared, _ = fixture(tmp_path)
    start = begin(root, prepared)
    finish(root, start)
    original = study._sources
    monkeypatch.setattr(study, "_sources", lambda: {**original(), "benchtop/study.py": "0" * 64})
    with pytest.raises(ValueError, match="changed"):
        begin(root, prepared, 1)
    assert ledger.inspect(root)["slots"][0]["reward"] == 0


def test_source_drift_does_not_discard_a_late_committed_answer(tmp_path, monkeypatch):
    root, prepared, _ = fixture(tmp_path)
    start = begin(root, prepared)
    original = study._sources
    monkeypatch.setattr(study, "_sources", lambda: {**original(), "benchtop/study.py": "0" * 64})
    with pytest.raises(ValueError, match="changed"):
        begin(root, prepared, 1)
    record = finish(root, start, reward=0)
    assert record["outcome"]["reward"] == 0 and record["recorder"]["source_sha256"]
    assert ledger.inspect(root)["counts"]["active"] == 0


def test_cli_registration_and_exclusive_snapshot_export(tmp_path, capsys):
    _, prepared, env = fixture(tmp_path)
    directory = tmp_path / "preparation"
    study.save_plan(prepared, directory)
    root = tmp_path / "cli-ledger"
    main(["study", "sessions", "init", str(directory), "--output-dir", str(root)])
    assert '"engineering_only"' in capsys.readouterr().out
    smoke = study.make_plan([env.dir], model=model())
    other = tmp_path / "smoke"
    study.save_plan(smoke, other)
    main(["study", "sessions", "register", str(root), str(other)])
    output = tmp_path / "snapshot.json"
    main(["study", "sessions", "inspect", str(root), "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["registered_plans"] == 2 and result["model_sessions_started_by_ledger"] == 0
    before = output.read_bytes()
    with pytest.raises(SystemExit, match="session accounting failed"):
        main(["study", "sessions", "inspect", str(root), "--output", str(output)])
    assert output.read_bytes() == before
    with pytest.raises(ValueError, match="outside the ledger"):
        ledger.export(root, root / "invalid-export.json")
    with pytest.raises(ValueError, match="under artifacts"):
        ledger.create(prepared, ROOT / "results/must-not-exist")
