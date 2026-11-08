import copy
import json
import shutil
from concurrent.futures import ThreadPoolExecutor

import pytest

from benchtop import (
    cohort,
    confirmation,
    confirmation_data,
    episode,
    pilots,
    primary,
    provider,
    runner,
    study,
)
from benchtop import confirmation_execution as execution
from benchtop import fresh_generation as fresh
from benchtop import generation_journal as generation
from benchtop import pilot_calibration as bridge
from benchtop import session_ledger as ledger
from benchtop.cli import main
from tests.test_cohort import ready as generation_ready
from tests.test_confirmation_data import original_files, ready
from tests.test_fresh_generation import Content, accept, lab_reply, process, setup
from tests.test_primary import simulate_provider_transport
from tests.test_session_ledger import finish


def prepared(tmp_path, monkeypatch, **kwargs):
    plan, _ = ready(tmp_path, monkeypatch, **kwargs)
    assessment = confirmation_data.collect()
    record = execution.authorize(plan)
    return plan, assessment, record, primary.canonical_root() / "ledger"


def start(directory, plan, index=0, **kwargs):
    return ledger.begin(directory, plan["plan_sha256"], plan["slots"][index]["slot_id"], **kwargs)


def test_authorization_registers_full_original_plan_without_models_new_allowance_or_protocol_lock(tmp_path, monkeypatch):
    plan, _ = ready(tmp_path, monkeypatch, count=2)
    assessment = confirmation_data.collect()
    before = original_files(plan)
    bindings = primary.BINDINGS.read_bytes()
    directory = primary.canonical_root() / "ledger"
    manifest = ledger._manifest(directory)
    monkeypatch.setattr(provider, "execute", lambda *a: pytest.fail("authorization started a model"))
    record = execution.authorize(plan)
    assert record["status"] == "authorized" and record["gradable_labs"] == record["full_cohort_size"] == 2
    assert record["assessment_sha256"] == assessment["assessment_sha256"]
    assert record["eligible_slots"] == record["planned_slots"] == 32 and record["model_sessions_started"] == 0
    assert record["initial_slot_order"] == [slot["slot_id"] for slot in plan["slots"]]
    assert record["ledger_id"] == manifest["ledger_id"] and record["extra_usage_disabled"] is True
    assert ledger._manifest(directory) == manifest and primary.BINDINGS.read_bytes() == bindings
    assert study._json(directory / "plans" / f"{plan['plan_sha256']}.json") == plan
    assert execution.authorize(plan) == record
    monkeypatch.setattr(provider, "probe", lambda: pytest.fail("inspection queried live authentication"))
    with ledger._locked(directory) as context:
        assert primary.verify(plan, live=False, ledger_context=context) == record
        grading = execution.grading_context(plan, plan["slots"][0]["env_id"], ledger_context=context)
        assert grading["task"]["seed_ids"] == list(range(16, 28))
        assert grading["binding"]["question_task_hash"] == plan["binding"]["environments"][0]["task_hash"]
    assert ledger.inspect(directory)["counts"]["total"] == 0
    assert all(study.cache.file_hash(path) == sha for path, sha in before.items())


def test_lock_and_assessment_alone_do_not_authorize_starts_and_missing_truth_is_refused(tmp_path, monkeypatch):
    plan, _ = ready(tmp_path, monkeypatch)
    directory = primary.canonical_root() / "ledger"
    with pytest.raises(FileNotFoundError):
        execution.authorize(plan)
    assert not (directory.parent / execution.AUTHORIZATION).exists()
    confirmation_data.collect()
    with pytest.raises(ValueError, match="authorization is missing"):
        ledger.register(plan, directory, authorization=directory.parent)
    study._publish(directory / "plans" / f"{plan['plan_sha256']}.json", plan)
    with pytest.raises(ValueError, match="authorization is missing"):
        start(directory, plan)
    assert ledger.inspect(directory)["counts"]["total"] == 0


@pytest.mark.parametrize(("effect", "status"), [
    ("(0.2 if seed < 16 else -0.2)", "unstable"),
    ("(0.2 if seed < 16 else 0.05 + (0.3 if seed % 2 else -0.3))", "ambiguous"),
    ("(0.2 if seed < 16 else 1 / 0)", "incomplete"),
])
def test_unavailable_labs_keep_all_slots_without_zero_rewards_or_replacements(tmp_path, monkeypatch, effect, status):
    plan, assessment, record, directory = prepared(tmp_path, monkeypatch, effect=effect)
    assert record["status"] == "no_gradable_labs" and record["eligible_slots"] == 0
    assert record["planned_slots"] == 16 and record["full_cohort_size"] == 1
    assert record["members"][0]["status"] == assessment["members"][0]["status"] == status
    for index in (0, 1, 15):
        with pytest.raises(ValueError, match="reward is unavailable"):
            start(directory, plan, index)
    with pytest.raises(ValueError, match="reward is unavailable"):
        execution.grading_context(plan, plan["slots"][0]["env_id"])
    with ledger._locked(directory) as (root, manifest):
        state = ledger._read(root, manifest)
        assert len([key for key in state["slots"] if key[0] == plan["plan_sha256"]]) == 16
        assert all(not history for key, history in state["histories"].items() if key[0] == plan["plan_sha256"])
    assert ledger.inspect(directory)["counts"]["total"] == 0


def test_mixed_stability_skips_unavailable_dispatch_but_keeps_original_lab_and_arm_identities(tmp_path, monkeypatch):
    lock, _ = setup(tmp_path, monkeypatch, count=2)
    papers = lock["binding"]["candidates"]

    def execute(work, *args):
        intent = study._json(work / "start.json")
        paper = next(row for row in papers if row["version"] == intent["version"])
        effect = "(0.2 if seed < 16 else -0.2)" if paper == papers[0] else 0.2
        return process(work, lab_reply(paper, effect=effect))

    original = provider.execute_text

    def selected(work, *args):
        result = original(work, *args)
        return result if study._json(work / "start.json")["stage"] == "triage" else execute(work, *args)

    monkeypatch.setattr(provider, "execute_text", selected)
    for paper in papers:
        fresh.step(client=Content())
        fresh.review(accept(generation.directory(), paper["version"], margin=0.05))
        fresh.qualify()
    cohort.lock_cohort()
    pilot = pilots.make_plan()
    pilots.authorize(pilot)
    bridge.export_pilots(pilot["plan_sha256"], tmp_path / "closed.json", close=True)
    plan = confirmation.make_plan()
    confirmation.lock_confirmation(plan)
    confirmation_data.collect()
    record = execution.authorize(plan)
    directory = primary.canonical_root() / "ledger"
    assert record["gradable_labs"] == 1 and record["full_cohort_size"] == 2 and record["planned_slots"] == 32
    assert record["initial_slot_order"] == [slot["slot_id"] for slot in plan["slots"][16:]]
    assert plan["slots"][16]["arm"] == "calibrated_budget"
    with pytest.raises(ValueError, match="reward is unavailable"):
        start(directory, plan, 0)
    with pytest.raises(ValueError, match="interleaved slot order"):
        start(directory, plan, 17)
    assert start(directory, plan, 16)["attempt_index"] == 0


def test_independent_grading_runs_both_arms_with_private_receipts_and_immutable_first_answer(tmp_path, monkeypatch):
    plan, assessment, record, directory = prepared(tmp_path, monkeypatch)
    original = original_files(plan)
    seen = []
    grade = episode.grade

    def observed(env, task, answer):
        seen.append(copy.deepcopy(task))
        return grade(env, task, answer)

    monkeypatch.setattr(episode, "grade", observed)
    monkeypatch.setattr(provider, "execute", lambda *args: simulate_provider_transport(
        *args, answer="paper_better" if len(seen) == 0 else "baseline_better", final_error=True))
    with runner.Controller() as controller:
        for index, expected in enumerate((1, 0)):
            slot = plan["slots"][index]
            result = runner.run_provider(plan["plan_sha256"], slot["slot_id"], controller=controller)
            assert result["status"] == "completed" and result["reward"] == expected
            work = directory / "runner" / result["attempt_id"]
            connection, sealed = (study._json(work / name) for name in ("connection.json", "sealed.json"))
            ep = controller.hub.get(connection["episode"])
            saved = copy.deepcopy(ep.result)
            assert ep.submit("paper_better", 0.99) == {"done": True} and ep.result == saved
            assert sealed["summary"]["private_grading"]["assessment_sha256"] == assessment["assessment_sha256"]
            assert sealed["summary"]["result"]["grading_context_sha256"] == record["members"][0]["grading_context_sha256"]
            assert ep.task["seed_ids"] == list(range(4, 16)) and ep._grading_task["seed_ids"] == list(range(16, 28))
            assert ep.order == plan["binding"]["environments"][0]["confirmation_visible_order"]
            assert all(seed < 16 for cells in ep.env.table.cells.values() for seed in cells)
            assert "private_grading" not in connection["description"]
            for reply in [json.loads(row) for row in (work / "stdout.jsonl").read_text().splitlines() if '"fixture_tool_result"' in row]:
                assert not any(key in json.dumps(reply["response"]) for key in ("truth", "score", "assessment_sha256", "seed_ids"))
            assert runner.resume(directory, result["attempt_id"])["outcome"]["reward"] == expected
    assert len(seen) == 2 and all(task["seed_ids"] == list(range(16, 28)) for task in seen)
    assert all(study.cache.file_hash(path) == sha for path, sha in original.items())
    assert ledger.inspect(directory)["counts"]["total"] == 2
    with pytest.raises(ValueError, match="attempt is final"):
        start(directory, plan, 1, retry_technical=True)


def test_evidence_replay_rejects_qualification_grading_forged_truth_and_missing_context(tmp_path, monkeypatch):
    plan, _, _, directory = prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(provider, "execute", lambda *a: simulate_provider_transport(*a, answer="paper_better"))
    result = runner.run_provider(plan["plan_sha256"], plan["slots"][0]["slot_id"])
    work = directory / "runner" / result["attempt_id"]
    connection, process_record, sealed = (study._json(work / name) for name in ("connection.json", "process.json", "sealed.json"))
    manifest = ledger._manifest(directory)
    for defect in ("connection", "summary", "reward", "truth", "answer", "result_context"):
        connected, changed = copy.deepcopy(connection), copy.deepcopy(sealed)
        if defect == "connection":
            connected["private_grading"]["confirmation_task_sha256"] = plan["binding"]["environments"][0]["task_hash"]
        elif defect == "summary":
            changed["summary"].pop("private_grading")
        elif defect == "reward":
            changed["summary"]["result"]["score"] = 0
        elif defect == "truth":
            changed["summary"]["result"]["graded"]["truth"] = "baseline_better"
        elif defect == "answer":
            next(row for row in changed["summary"]["log"] if row["action"] == "submit")["answer"] = "baseline_better"
        else:
            changed["summary"]["result"].pop("grading_context_sha256")
        outcome = runner._classify(plan, plan["slots"][0], connected, process_record, changed, manifest["identity"]["protocol"])["outcome"]
        assert outcome["state"] == "unresolved" and outcome["reward"] is None
    with ledger._locked(directory):
        assert runner._validated_evidence(manifest, plan, plan["slots"][0], result["attempt_id"], work)["outcome"]["reward"] == 1


def test_initial_starts_are_ordered_atomic_and_share_two_pending_capacity_with_generation(tmp_path, monkeypatch):
    plan, _, record, directory = prepared(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="interleaved slot order"):
        start(directory, plan, 1)

    def launch():
        try:
            return start(directory, plan)
        except ValueError as error:
            return str(error)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: launch(), range(2)))
    assert sum(isinstance(value, dict) for value in results) == 1
    read = generation.read
    monkeypatch.setattr(generation, "read", lambda *args: read(*args) if args else {**read(), "active": 1})
    with pytest.raises(ValueError, match="two in-flight"):
        start(directory, plan, 1)
    monkeypatch.setattr(generation, "read", read)
    start(directory, plan, 1)
    with pytest.raises(ValueError, match="two in-flight"):
        start(directory, plan, 2)
    assert execution.verify(plan, live=False) == record
    copied = tmp_path / "copied-ledger"
    shutil.copytree(directory, copied)
    with pytest.raises(ValueError, match="canonical journal"):
        start(copied, plan, 2)
    counts = ledger.inspect(directory)["counts"]
    assert counts["total"] == counts["primary"] == counts["phase:confirmation"] == counts["active"] == 2


def test_explicit_technical_retry_keeps_attempt_identity_and_does_not_reorder_initial_starts(tmp_path, monkeypatch):
    plan, _, _, directory = prepared(tmp_path, monkeypatch)
    first = start(directory, plan)
    finish(directory, first, state="technical_failure", reward=None, cause="provider_outage")
    with pytest.raises(ValueError, match="explicit technical retry"):
        start(directory, plan)
    retry = start(directory, plan, retry_technical=True)
    assert retry["attempt_id"] == plan["slots"][0]["attempt_ids"][1]
    finish(directory, retry, reward=0)
    with pytest.raises(ValueError, match="attempt is final"):
        start(directory, plan, retry_technical=True)
    start(directory, plan, 1)
    counts = ledger.inspect(directory)["counts"]
    assert counts["retry"] == 1 and counts["primary"] == 2 and counts["total"] == 3


@pytest.mark.parametrize("contract", ["model", "tools"])
def test_provider_violation_halts_later_confirmation_without_replacing_slots(tmp_path, monkeypatch, contract):
    plan, _, _, directory = prepared(tmp_path, monkeypatch)
    options = {"model_id": "changed-model"} if contract == "model" else {"tools": [*provider.TOOLS, "Bash"]}
    monkeypatch.setattr(provider, "execute", lambda *a: simulate_provider_transport(*a, **options))
    result = runner.run_provider(plan["plan_sha256"], plan["slots"][0]["slot_id"])
    assert result["reward"] is None and result["status"] == "unresolved"
    assert ledger.inspect(directory)["halted"]
    with pytest.raises(ValueError, match="ledger halted"):
        start(directory, plan, 1)
    assert len(plan["slots"]) == 16 and ledger.inspect(directory)["counts"]["total"] == 1


@pytest.mark.parametrize("field", ["assessment_sha256", "initial_slot_order", "eligible_slots", "members", "ledger_id", "recorded_at_utc"])
def test_resealed_authorization_changes_do_not_permit_a_start(tmp_path, monkeypatch, field):
    plan, _, record, directory = prepared(tmp_path, monkeypatch)
    changed = copy.deepcopy(record)
    if field == "initial_slot_order":
        changed[field].reverse()
    elif field == "members":
        changed[field][0]["grading_context_sha256"] = "0" * 64
    else:
        changed[field] = 0 if field == "eligible_slots" else "2000-01-01T00:00:00+00:00" if field == "recorded_at_utc" else "0" * 64
    (directory.parent / execution.AUTHORIZATION).write_text(json.dumps(study._seal(changed, "record_sha256")))
    with pytest.raises(ValueError, match="authorization, independent grading"):
        start(directory, plan)
    assert ledger.inspect(directory)["counts"]["total"] == 0


def test_authorization_acknowledgement_loss_registers_same_plan_without_new_permission_or_calls(tmp_path, monkeypatch):
    plan, _ = ready(tmp_path, monkeypatch)
    confirmation_data.collect()
    publish = study._publish

    def lost(path, value):
        publish(path, value)
        if path.name == execution.AUTHORIZATION:
            raise OSError("fixture lost acknowledgement")

    monkeypatch.setattr(study, "_publish", lost)
    with pytest.raises(OSError, match="lost acknowledgement"):
        execution.authorize(plan)
    directory = primary.canonical_root() / "ledger"
    saved = (directory.parent / execution.AUTHORIZATION).read_bytes()
    assert not (directory / "plans" / f"{plan['plan_sha256']}.json").exists()
    monkeypatch.setattr(study, "_publish", publish)
    execution.authorize(plan)
    assert (directory.parent / execution.AUTHORIZATION).read_bytes() == saved
    assert study._json(directory / "plans" / f"{plan['plan_sha256']}.json") == plan
    assert ledger.inspect(directory)["counts"]["total"] == 0


def test_saved_completion_survives_source_drift_but_changed_truth_cannot_be_replayed(tmp_path, monkeypatch):
    plan, _, _, directory = prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(provider, "execute", lambda *a: simulate_provider_transport(*a, answer="paper_better"))
    result = runner.run_provider(plan["plan_sha256"], plan["slots"][0]["slot_id"])
    recorded = runner.resume(directory, result["attempt_id"])
    sources = study._sources()
    monkeypatch.setattr(study, "_sources", lambda: {**sources, "benchtop/runner.py": "0" * 64})
    monkeypatch.setattr(provider, "execute", lambda *a: pytest.fail("saved outcome relaunched a provider"))
    assert runner.resume(directory, result["attempt_id"]) == recorded
    with pytest.raises(ValueError):
        runner.run_provider(plan["plan_sha256"], plan["slots"][1]["slot_id"])
    file = directory.parent / confirmation.DATA_DIRECTORY / "labs" / plan["slots"][0]["env_id"] / "results.jsonl"
    file.write_text(file.read_text() + "\n")
    with pytest.raises(ValueError, match="completion evidence changed"):
        runner.resume(directory, result["attempt_id"])
    assert ledger.inspect(directory)["counts"]["total"] == 1


@pytest.mark.parametrize(("options", "status", "reward"), [
    ({"no_answer": True}, "agent_failure", 0),
    ({"no_answer": True, "final_error": True}, "unresolved", None),
    ({"missing_init": True}, "unresolved", None),
])
def test_no_answer_and_unknown_provider_results_preserve_their_failure_semantics(tmp_path, monkeypatch, options, status, reward):
    plan, _, _, directory = prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(provider, "execute", lambda *a: simulate_provider_transport(*a, **options))
    result = runner.run_provider(plan["plan_sha256"], plan["slots"][0]["slot_id"])
    assert result["status"] == status and result["reward"] == reward
    with pytest.raises(ValueError, match="attempt is final or requires reconciliation"):
        start(directory, plan, retry_technical=True)
    assert ledger.inspect(directory)["counts"]["total"] == 1


@pytest.mark.parametrize("field", ["active", "halted"])
def test_authorization_requires_idle_nonhalted_generation_and_preserves_all_allowances(tmp_path, monkeypatch, field):
    plan, _ = ready(tmp_path, monkeypatch)
    confirmation_data.collect()
    read = generation.read
    monkeypatch.setattr(generation, "read", lambda *args: read(*args) if args else {**read(), field: 1})
    with pytest.raises(ValueError, match="idle shared accounting"):
        execution.authorize(plan)
    assert not (primary.canonical_root() / execution.AUTHORIZATION).exists()
    assert ledger.inspect(primary.canonical_root() / "ledger")["counts"]["total"] == 0


def test_zero_cohort_authorizes_no_model_slots_and_never_creates_confirmation_data(tmp_path, monkeypatch):
    generation_ready(tmp_path, monkeypatch, effect="0.05 + (0.3 if seed % 2 else -0.3)")
    cohort.lock_cohort()
    plan = confirmation.make_plan()
    confirmation.lock_confirmation(plan)
    monkeypatch.setattr(provider, "execute", lambda *a: pytest.fail("empty cohort invoked a model"))
    record = execution.authorize(plan)
    assert record["status"] == "no_primary_comparison" and record["planned_slots"] == record["eligible_slots"] == 0
    assert record["initial_slot_order"] == record["members"] == []
    assert not (primary.canonical_root() / confirmation.DATA_DIRECTORY).exists()
    assert execution.verify(plan, live=False) == record


def test_cli_and_copied_context_do_not_expose_truth_or_probe_live_authentication(tmp_path, monkeypatch, capsys):
    plan, _ = ready(tmp_path, monkeypatch)
    confirmation_data.collect()
    study.save_plan(plan, tmp_path / "preparation")
    main(["study", "authorize-confirmation", str(tmp_path / "preparation")])
    created = json.loads(capsys.readouterr().out)
    assert created["status"] == "authorized" and created["model_sessions_started"] == 0
    monkeypatch.setattr(provider, "probe", lambda: pytest.fail("inspection queried authentication"))
    main(["study", "inspect-confirmation-execution"])
    inspected = capsys.readouterr().out
    assert json.loads(inspected) == created
    assert not any(key in inspected for key in ("confirmation_label", "qualification_label", "truth", "task_artifact", "seed_ids"))
    directory = primary.canonical_root() / "ledger"
    with pytest.raises(ValueError, match="canonical journal"):
        execution.verify(plan, live=False, ledger_context=(tmp_path / "copy", ledger._manifest(directory)))
