import asyncio
import json
from datetime import UTC, datetime

import pytest

from benchtop import cohort, confirmation, pilots, primary, provider, runner, study, sweep
from benchtop import confirmation_data as data
from benchtop import pilot_calibration as bridge
from benchtop import session_ledger as ledger
from benchtop.cli import main
from benchtop.env import Env, read_jsonl
from tests.test_cohort import ready as generation_ready
from tests.test_primary import simulate_provider_transport
from tests.test_provider import preflight
from tests.test_session_ledger import finish


def ready(tmp_path, monkeypatch, *, effect=0.2, count=1, lock=True):
    generation_ready(tmp_path, monkeypatch, effect=effect, count=count)
    cohort.lock_cohort()
    pilot = pilots.make_plan()
    pilots.authorize(pilot)
    bridge.export_pilots(pilot["plan_sha256"], tmp_path / "closed.json", close=True)
    plan = confirmation.make_plan()
    if lock:
        confirmation.lock_confirmation(plan)
    return plan, primary.canonical_root() / confirmation.DATA_DIRECTORY


def no_models(monkeypatch):
    monkeypatch.setattr(provider, "execute", lambda *a: pytest.fail("confirmation data invoked a model"))
    monkeypatch.setattr(provider, "execute_text", lambda *a: pytest.fail("confirmation data generated a lab"))
    monkeypatch.setattr(provider, "probe", lambda: pytest.fail("confirmation data queried authentication"))


def no_work(monkeypatch):
    async def forbidden(*args):
        pytest.fail("saved confirmation data was executed again")

    monkeypatch.setattr(data, "_execute", forbidden)


def original_files(plan):
    return {str(study._resolve(member["source"]) / name): sha for member in plan["binding"]["environments"] for name, sha in member["files"].items()}


@pytest.mark.parametrize("count", [1, 2])
def test_real_sandbox_collects_only_confirmation_seeds_and_preserves_qualification_byte_for_byte(tmp_path, monkeypatch, count):
    plan, root = ready(tmp_path, monkeypatch, count=count)
    before = original_files(plan)
    no_models(monkeypatch)
    assert data.status()["status"] == "not_collected" and not root.exists()
    captured = []
    execute = data._execute

    async def observed(folder, jobs, intent, logs, record):
        captured.extend(seed for _, seed in jobs)
        assert intent["workers"] == 2 and intent["run_timeout_seconds"] == 5
        assert (folder / "execution_start.json").exists()
        return await execute(folder, jobs, intent, logs, record)

    monkeypatch.setattr(data, "_execute", observed)
    result = data.collect()
    assert result["status"] == "assessed" and result["full_cohort_size"] == result["gradable_labs"] == count
    assert result["provider_execution_authorized"] is False and result["model_sessions_started"] == 0
    assert set(captured) == set(range(16, 28)) and len(captured) == 48 * count
    for member in result["members"]:
        assert member["status"] == "stable" and member["confirmation_label"] == member["qualification_label"] == "paper_better"
        assert member["task_artifact"]["seed_ids"] == list(range(16, 28))
        assert member["task_artifact"]["interval_method"] == "student_t_95"
        claim = next(row for row in member["task_artifact"]["contrasts"] if row["task_id"] == "claim")
        assert claim["n_seeds"] == 12 and claim["status"] == "settled"
        assert member["observed_results"] == member["recorded_job_intents"] == 48 and member["failed_results"] == 0
        assert member["actual_environment_cpu_seconds"] is None and member["reported_job_seconds"]["unreported_jobs"] == 0
        assert {row["seed"] for row in read_jsonl(root / "labs" / member["env_id"] / "results.jsonl")} == set(range(16, 28))
    assert all(study.cache.file_hash(path) == sha for path, sha in before.items())
    no_work(monkeypatch)
    before_assessment = (root / "assessment.json").read_bytes()
    assert data.collect() == data.load_assessment() == data.status() == result
    assert (root / "assessment.json").read_bytes() == before_assessment
    assert ledger.inspect(primary.canonical_root() / "ledger")["counts"]["total"] == 0
    with pytest.raises(ValueError, match="confirmation provider authorization is missing"):
        ledger.register(plan, primary.canonical_root() / "ledger", authorization=primary.canonical_root())


@pytest.mark.parametrize(("effect", "state"), [
    ("(0.2 if seed < 16 else -0.2)", "unstable"),
    ("(0.2 if seed < 16 else 0.05 + (0.3 if seed % 2 else -0.3))", "ambiguous"),
    ("(0.2 if seed < 16 else 1 / 0)", "incomplete"),
])
def test_unstable_ambiguous_and_failed_confirmation_retain_full_cohort_members(tmp_path, monkeypatch, effect, state):
    plan, root = ready(tmp_path, monkeypatch, effect=effect)
    no_models(monkeypatch)
    result = data.collect()
    member = result["members"][0]
    assert member["status"] == state and member["gradable"] is False
    assert result["full_cohort_size"] == 1 and result["gradable_labs"] == 0 and len(plan["slots"]) == 16
    assert member["observed_results"] == 48
    if state == "unstable":
        assert member["confirmation_label"] == "baseline_better"
    elif state == "ambiguous":
        assert member["confirmation_label"] is None and member["task_artifact"]["status"] == "complete"
    else:
        assert member["failed_results"] == 24 and member["task_artifact"]["status"] == "incomplete"
    no_work(monkeypatch)
    assert data.collect() == result and confirmation.verify()["planned_slots"] == 16
    assert (root / "assessment.json").exists()


def test_interrupted_execution_preserves_partial_rows_and_never_automatically_reruns(tmp_path, monkeypatch):
    plan, root = ready(tmp_path, monkeypatch)
    no_models(monkeypatch)

    async def interrupted(folder, jobs, intent, logs, record):
        config, seed = jobs[0]
        record(config, seed, {"ok": True, "metrics": {"value": 0.5}, "secs": 0.01})
        raise OSError("fixture interruption")

    monkeypatch.setattr(data, "_execute", interrupted)
    with pytest.raises(OSError, match="interruption"):
        data.collect()
    folder = root / "labs" / plan["binding"]["environments"][0]["env_id"]
    assert len(read_jsonl(folder / "results.jsonl")) == 1 and not (folder / "execution_finish.json").exists()
    assert data.status()["members"][0]["status"] == "pending_reconciliation"
    no_work(monkeypatch)
    with pytest.raises(ValueError, match="already started"):
        data.collect()
    assert not (root / "assessment.json").exists()


@pytest.mark.parametrize("durable", [False, True])
def test_finish_acknowledgement_loss_only_resumes_when_durable_stop_evidence_exists(tmp_path, monkeypatch, durable):
    plan, root = ready(tmp_path, monkeypatch)
    publish = study._publish

    def lost(path, value):
        if path.name == "execution_finish.json":
            if durable:
                publish(path, value)
            raise OSError("fixture lost finish acknowledgement")
        return publish(path, value)

    no_models(monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(study, "_publish", lost)
        with pytest.raises(OSError, match="lost finish"):
            data.collect()
    folder = root / "labs" / plan["binding"]["environments"][0]["env_id"]
    assert len(read_jsonl(folder / "results.jsonl")) == 48
    before = (folder / "results.jsonl").read_bytes()
    no_work(monkeypatch)
    if durable:
        assert data.collect()["gradable_labs"] == 1
    else:
        with pytest.raises(ValueError, match="already started"):
            data.collect()
    assert (folder / "results.jsonl").read_bytes() == before


@pytest.mark.parametrize("target", ["source", "copy", "lock_chain"])
def test_mid_execution_input_changes_stop_without_creating_a_finish(tmp_path, monkeypatch, target):
    plan, root = ready(tmp_path, monkeypatch)
    source = study._resolve(plan["binding"]["environments"][0]["source"])

    async def change(folder, jobs, intent, logs, record):
        if target == "lock_chain":
            path = primary.canonical_root() / "candidate_lock.json"
            candidate = study._json(path)
            candidate["scope"] = "changed during confirmation execution"
            path.write_text(json.dumps(study._seal(candidate, "record_sha256")))
        else:
            path = source / "lab.py" if target == "source" else folder / "space.json"
            path.write_text(path.read_text() + "\n")
        record(*jobs[0], {"ok": True, "metrics": {"value": 0.5}, "secs": 0.01})

    no_models(monkeypatch)
    monkeypatch.setattr(data, "_execute", change)
    with pytest.raises(ValueError):
        data.collect()
    assert not list(root.glob("labs/*/execution_finish.json")) and not (root / "assessment.json").exists()


@pytest.mark.parametrize("defect", ["missing_lock", "unrecorded_directory", "active", "halt", "prior_model"])
def test_missing_authority_or_wrong_phase_cannot_collect_truth(tmp_path, monkeypatch, defect):
    plan, root = ready(tmp_path, monkeypatch, lock=defect != "missing_lock")
    directory = primary.canonical_root() / "ledger"
    if defect == "unrecorded_directory":
        root.mkdir()
        (root / "results.jsonl").write_text("not an authorized collection")
    elif defect in ("active", "halt"):
        smoke = study._json(directory.parent / "smoke_plan.json")
        if defect == "active":
            ledger.begin(directory, smoke["plan_sha256"], smoke["slots"][0]["slot_id"])
        else:
            monkeypatch.setattr(provider, "execute", lambda *args: simulate_provider_transport(*args, model_id="changed-model"))
            runner.run_provider(smoke["plan_sha256"], smoke["slots"][0]["slot_id"])
    elif defect == "prior_model":
        study._publish(directory / "plans" / f"{plan['plan_sha256']}.json", plan)
        start = study._seal({**ledger._base(ledger._manifest(directory), plan, plan["slots"][0], 0), "record_kind": "start",
                            "bucket": "primary", "recorded_at_utc": datetime.now(UTC).isoformat()}, "record_sha256")
        study._publish(directory / "attempts" / f"{start['attempt_id']}.start.json", start)
        finish(directory, start)
    no_models(monkeypatch)
    no_work(monkeypatch)
    with pytest.raises((ValueError, FileNotFoundError)):
        data.collect()
    assert not (root / "collection.json").exists() and not (root / "assessment.json").exists()


@pytest.mark.parametrize("field", ["seed_ids", "workers", "confirmation_lock_sha256"])
def test_resealed_collection_rule_changes_are_refused(tmp_path, monkeypatch, field):
    _, root = ready(tmp_path, monkeypatch)
    no_models(monkeypatch)
    data.collect()
    path = root / "collection.json"
    intent = study._json(path)
    intent[field] = [4, 5] if field == "seed_ids" else 8 if field == "workers" else "0" * 64
    path.write_text(json.dumps(study._seal(intent, "collection_sha256")))
    no_work(monkeypatch)
    with pytest.raises(ValueError):
        data.load_assessment()


@pytest.mark.parametrize("defect", ["qualification_seed", "duplicate"])
def test_resealed_cross_partition_or_duplicate_rows_are_refused_before_grading(tmp_path, monkeypatch, defect):
    plan, root = ready(tmp_path, monkeypatch)
    no_models(monkeypatch)
    data.collect()
    folder = root / "labs" / plan["binding"]["environments"][0]["env_id"]
    path = folder / "results.jsonl"
    row = read_jsonl(path)[0]
    if defect == "qualification_seed":
        row = study.cache.seal_result(data._execution(folder), data.Space.load(folder / "space.json"), row["config"], 4, row)
    with path.open("a") as output:
        output.write(json.dumps(row) + "\n")
    finish_path = folder / "execution_finish.json"
    finished = study._json(finish_path)
    finished["files"][str(path.relative_to(root))] = study.cache.file_hash(path)
    finish_path.write_text(json.dumps(study._seal(finished, "record_sha256")))
    monkeypatch.setattr(data, "build_task_set", lambda *a, **kw: pytest.fail("invalid partition or duplicate data reached grading"))
    with pytest.raises(ValueError, match="undeclared seeds|duplicate confirmation cell"):
        data.load_assessment()


@pytest.mark.parametrize("target", ["member", "aggregate", "log"])
def test_changed_truth_or_execution_evidence_is_refused(tmp_path, monkeypatch, target):
    plan, root = ready(tmp_path, monkeypatch)
    no_models(monkeypatch)
    data.collect()
    if target == "member":
        path = root / "labs" / plan["binding"]["environments"][0]["env_id"] / "assessment.json"
        report = study._json(path)
        report["confirmation_label"] = "baseline_better"
        path.write_text(json.dumps(study._seal(report, "record_sha256")))
    elif target == "aggregate":
        path = root / "assessment.json"
        report = study._json(path)
        report["gradable_labs"] = 0
        path.write_text(json.dumps(study._seal(report, "assessment_sha256")))
    else:
        path = next((root / "logs").rglob("*.log"))
        path.write_text(path.read_text() + "changed evidence\n")
    no_work(monkeypatch)
    with pytest.raises(ValueError):
        data.load_assessment()


def test_private_cache_cannot_enter_ordinary_loading_or_sweep_paths(tmp_path, monkeypatch):
    plan, root = ready(tmp_path, monkeypatch)
    no_models(monkeypatch)
    data.collect()
    folder = root / "labs" / plan["binding"]["environments"][0]["env_id"]
    before = data._files(root, folder.iterdir())
    with pytest.raises(ValueError, match="unsupported execution cache schema"):
        Env.load(folder)
    with pytest.raises(ValueError, match="unsupported execution cache schema"):
        Env.load(folder, load_tasks=False)
    for split in (None, "visible", "qualification", "confirmation"):
        with pytest.raises(ValueError, match="cache schema|confirmation"):
            sweep.sweep(folder, split=split)
    assert data._files(root, folder.iterdir()) == before
    assert Env.load(study._resolve(plan["binding"]["environments"][0]["source"])).task("claim")


def test_sandbox_workers_cannot_read_any_original_study_data(tmp_path, monkeypatch):
    _, root = ready(tmp_path, monkeypatch)
    secret = root.parent / "private-boundary.txt"
    secret.write_text("hidden qualification or truth data")
    folder = tmp_path / "worker-boundary"
    folder.mkdir()
    (folder / "lab.py").write_text("def run(config, seed):\n    try:\n        open(" + repr(str(secret)) + ").read()\n"
                                  "    except PermissionError:\n        return {'value': 1.0}\n    return {'value': 0.0}\n")
    observed = []
    asyncio.run(data._execute(folder, [({}, 16)], {"workers": 2, "run_timeout_seconds": 5, "sandbox": "macos"},
                              tmp_path / "boundary-logs", lambda config, seed, result: observed.append(result)))
    assert observed[0]["ok"] and observed[0]["metrics"]["value"] == 1.0


def test_missing_results_and_unreported_worker_timing_stay_incomplete_and_unknown(tmp_path, monkeypatch):
    _, root = ready(tmp_path, monkeypatch)
    no_models(monkeypatch)

    async def missing(folder, jobs, intent, logs, record):
        record(*jobs[0], {"ok": True, "metrics": {"value": float("nan")}})
        for config, seed in jobs[1:-1]:
            record(config, seed, {"ok": True, "metrics": {"value": 0.5}, "secs": 0.01})

    monkeypatch.setattr(data, "_execute", missing)
    member = data.collect()["members"][0]
    assert member["status"] == "incomplete" and member["observed_results"] == 47 and member["failed_results"] == 1
    assert member["reported_job_seconds"]["unreported_jobs"] == 2 and member["reported_job_seconds"]["total"] is None
    assert member["reported_job_seconds"]["known_sum"] == pytest.approx(0.46)
    assert (root / "assessment.json").exists()


def test_lost_aggregate_acknowledgement_and_later_accounting_do_not_change_assessment(tmp_path, monkeypatch):
    _, root = ready(tmp_path, monkeypatch)
    publish = study._publish

    def lost(path, value):
        publish(path, value)
        if path == root / "assessment.json":
            raise OSError("fixture lost aggregate acknowledgement")

    with monkeypatch.context() as patch:
        no_models(patch)
        patch.setattr(study, "_publish", lost)
        with pytest.raises(OSError, match="aggregate acknowledgement"):
            data.collect()
    before = (root / "assessment.json").read_bytes()
    smoke = study._json(root.parent / "smoke_plan.json")
    monkeypatch.setattr(provider, "probe", preflight)
    monkeypatch.setattr(provider, "execute", simulate_provider_transport)
    runner.run_provider(smoke["plan_sha256"], smoke["slots"][0]["slot_id"])
    no_models(monkeypatch)
    no_work(monkeypatch)
    assert data.collect() == data.load_assessment()
    assert (root / "assessment.json").read_bytes() == before
    with ledger._locked(root.parent / "ledger") as context, monkeypatch.context() as patch:
        patch.setattr(ledger, "_locked", lambda *a: pytest.fail("nested lock during assessment verification"))
        assert data.load_assessment(ledger_context=context)["gradable_labs"] == 1


def test_zero_cohort_and_missing_state_cannot_start_workers_or_create_data(tmp_path, monkeypatch):
    generation_ready(tmp_path, monkeypatch, effect="0.05 + (0.3 if seed % 2 else -0.3)")
    cohort.lock_cohort()
    confirmation.lock_confirmation(confirmation.make_plan())
    no_models(monkeypatch)
    no_work(monkeypatch)
    result = data.collect()
    assert result["status"] == "no_primary_comparison" and result["members"] == [] and result["full_cohort_size"] == 0
    assert not (primary.canonical_root() / confirmation.DATA_DIRECTORY).exists()
    monkeypatch.setattr(primary, "STATE_BASE", tmp_path / "missing")
    for operation in (data.collect, data.status, data.load_assessment):
        with pytest.raises(FileNotFoundError):
            operation()
    assert not primary.STATE_BASE.exists()


def test_cli_exposes_status_and_gradability_without_truth_or_raw_metrics(tmp_path, monkeypatch, capsys):
    ready(tmp_path, monkeypatch)
    no_models(monkeypatch)
    main(["study", "inspect-confirmation-data"])
    assert json.loads(capsys.readouterr().out)["status"] == "not_collected"
    main(["study", "collect-confirmation-data"])
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "assessed" and result["gradable_labs"] == 1 and result["model_sessions_started"] == 0
    assert not any(secret in json.dumps(result) for secret in ("qualification_label", "confirmation_label", "metrics", "mcp_url"))
    no_work(monkeypatch)
    main(["study", "inspect-confirmation-data"])
    assert json.loads(capsys.readouterr().out) == result
    with pytest.raises(SystemExit):
        main(["study", "collect-confirmation-data", "--seeds", "4"])
