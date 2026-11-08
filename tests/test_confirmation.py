import copy
import json
from datetime import UTC, datetime

import pytest

from benchtop import cohort, confirmation, primary, provider, runner, study
from benchtop import pilot_calibration as bridge
from benchtop import session_ledger as ledger
from benchtop.cli import main
from tests.test_cohort import ready as generation_ready
from tests.test_pilot_calibration import no_execution
from tests.test_pilot_calibration import ready as pilot_ready
from tests.test_primary import simulate_provider_transport
from tests.test_session_ledger import finish


def ready(tmp_path, monkeypatch, *, count=1, completed=False):
    directory, pilot = pilot_ready(tmp_path, monkeypatch, count=count)
    if completed:
        runner.run_provider(pilot["plan_sha256"], pilot["slots"][0]["slot_id"])
    collection = bridge.export_pilots(pilot["plan_sha256"], tmp_path / "closed.json", close=True)
    return directory, pilot, confirmation.make_plan(), collection


@pytest.mark.parametrize("count", [1, 2])
def test_plan_has_independent_equal_budget_arms_and_protocol_dispatch_order(tmp_path, monkeypatch, count):
    directory, pilot, plan, collection = ready(tmp_path, monkeypatch, count=count)
    no_execution(monkeypatch)
    assert study.validate_plan(plan) == plan and plan["execution_ready"] is False
    assert plan["model_sessions_started"] == 0 and len(plan["groups"]) == 2 * count and len(plan["slots"]) == 16 * count
    assert len({slot["slot_id"] for slot in plan["slots"]}) == 16 * count
    assert len({attempt for slot in plan["slots"] for attempt in slot["attempt_ids"]}) == 32 * count
    assert plan["binding"]["pilot_snapshot_sha256"] == collection["snapshot_sha256"]
    assert plan["binding"]["pilot_evidence"] == collection["phase_evidence"]
    assert all(decision["fallback_reason"] == "incomplete_pilots" for decision in plan["binding"]["pilot_decisions"])
    for index, member in enumerate(plan["binding"]["environments"]):
        original = pilot["binding"]["environments"][index]
        assert member["visible_order"] == original["confirmation_visible_order"] != original["visible_order"]
        assert member["pilot_visible_order"] == original["visible_order"] and member["task_hash"] == original["task_hash"]
        pair = plan["groups"][2 * index:2 * index + 2]
        assert pair[0]["group_id"] != pair[1]["group_id"]
        assert {key: value for key, value in pair[0]["descriptor"].items() if key != "arm"} == {
            key: value for key, value in pair[1]["descriptor"].items() if key != "arm"}
        slots = plan["slots"][16 * index:16 * index + 16]
        arms = ["global_budget", "calibrated_budget"] if index % 2 == 0 else ["calibrated_budget", "global_budget"]
        assert [slot["arm"] for slot in slots] == arms * 8
        assert [slot["replicate_index"] for slot in slots] == [replicate for replicate in range(8) for _ in range(2)]
    assert not (directory.parent / confirmation.DATA_DIRECTORY).exists()
    assert not (directory.parent / "confirmation_lock.json").exists()
    study.save_plan(plan, tmp_path / "preparation")
    assert study.load_plan(tmp_path / "preparation") == plan


def test_selected_budget_is_bound_from_verified_pilots_without_an_override(tmp_path, monkeypatch):
    directory, pilot = pilot_ready(tmp_path, monkeypatch)
    by_slot = {slot["slot_id"]: slot for slot in pilot["slots"]}

    def controlled(work, timeout, prepared, proof):
        slot = by_slot[study._json(work / "connection.json")["slot_id"]]
        success = slot["replicate_index"] < {0.06: 1, 0.12: 4, 0.25: 4}[slot["budget_fraction"]]
        return simulate_provider_transport(work, timeout, prepared, proof, answer="paper_better" if success else "baseline_better")

    monkeypatch.setattr(provider, "execute", controlled)
    with runner.Controller() as controller:
        for slot in pilot["slots"]:
            runner.run_provider(pilot["plan_sha256"], slot["slot_id"], controller=controller)
    collection = bridge.export_pilots(pilot["plan_sha256"], tmp_path / "closed.json", close=True)
    no_execution(monkeypatch)
    plan = confirmation.make_plan()
    assert collection["labs"][0]["selection"]["status"] == "selected"
    assert [group["descriptor"]["budget_fraction"] for group in plan["groups"]] == [0.12, 0.06]
    assert len(plan["binding"]["pilot_decisions"][0]["pilot_attempt_ids"]) == 12
    assert any(name.endswith("stdout.jsonl") for name in plan["binding"]["pilot_evidence"])
    assert ledger.inspect(directory)["counts"]["total"] == 12


def test_open_or_missing_pilot_collection_cannot_prepare_confirmation(tmp_path, monkeypatch):
    directory, pilot = pilot_ready(tmp_path, monkeypatch)
    no_execution(monkeypatch)
    bridge.export_pilots(pilot["plan_sha256"], tmp_path / "open.json")
    with pytest.raises(FileNotFoundError):
        confirmation.make_plan()
    assert not (directory.parent / "confirmation_plan.json").exists() and not (directory.parent / "confirmation_lock.json").exists()


@pytest.mark.parametrize("kind", ["arm_budget", "slot_removed", "dispatch", "attempt", "question", "pilot_snapshot", "execution_ready"])
def test_resealed_plan_overrides_are_refused(tmp_path, monkeypatch, kind):
    _, _, plan, _ = ready(tmp_path, monkeypatch)
    altered = copy.deepcopy(plan)
    if kind == "arm_budget":
        altered["slots"][1]["budget_fraction"] = 0.25
    elif kind == "slot_removed":
        altered["slots"].pop()
    elif kind == "dispatch":
        altered["slots"][:2] = altered["slots"][:2][::-1]
    elif kind == "attempt":
        altered["slots"][0]["attempt_ids"][0] = "1" * 64
    elif kind == "question":
        altered["binding"]["environments"][0]["task_hash"] = "2" * 64
    elif kind == "pilot_snapshot":
        altered["binding"]["pilot_snapshot_sha256"] = "3" * 64
    else:
        altered["execution_ready"] = True
    altered["bindings_hash"] = study.canonical_hash(altered["binding"])
    altered = study._seal(altered, "plan_sha256")
    with pytest.raises(ValueError, match="confirmation plan"):
        study.validate_plan(altered)


def test_publication_appends_once_and_does_not_register_or_authorize_execution(tmp_path, monkeypatch):
    directory, _, plan, _ = ready(tmp_path, monkeypatch)
    no_execution(monkeypatch)
    before = study._json(primary.BINDINGS)["records"]
    record = confirmation.lock_confirmation(plan)
    assert record["study_lock"] is True and record["execution_ready"] is False and record["model_sessions_started"] == 0
    assert record["planned_slots"] == 16 and record["status"] == "locked_awaiting_confirmation_data"
    assert record["ledger_id"] == ledger._manifest(directory)["ledger_id"]
    assert study._json(primary.BINDINGS)["records"][:-1] == before
    assert study._json(primary.BINDINGS)["records"][-1]["record_sha256"] == record["record_sha256"]
    assert confirmation.verify() == record and ledger.inspect(directory)["counts"]["total"] == 0
    assert not (directory / "plans" / f"{plan['plan_sha256']}.json").exists()
    with pytest.raises(FileExistsError, match="already locked"):
        confirmation.lock_confirmation(plan)
    with pytest.raises(ValueError, match="confirmation provider authorization is missing"):
        ledger.register(plan, directory, authorization=directory.parent)
    study._publish(directory / "plans" / f"{plan['plan_sha256']}.json", plan)
    with pytest.raises(ValueError, match="confirmation provider authorization is missing"):
        runner.run_provider(plan["plan_sha256"], plan["slots"][0]["slot_id"])
    with pytest.raises(ValueError, match="confirmation provider authorization is missing"):
        ledger.begin(directory, plan["plan_sha256"], plan["slots"][0]["slot_id"])
    assert ledger.inspect(directory)["counts"]["total"] == 0


def test_inspection_ignores_post_lock_truth_files_and_later_phase_counts(tmp_path, monkeypatch):
    directory, _, plan, _ = ready(tmp_path, monkeypatch)
    record = confirmation.lock_confirmation(plan)
    folder = directory.parent / confirmation.DATA_DIRECTORY
    folder.mkdir()
    (folder / "truth.json").write_text("must never be parsed by confirmation-lock inspection")
    smoke = study._json(directory.parent / "smoke_plan.json")
    runner.run_provider(smoke["plan_sha256"], smoke["slots"][0]["slot_id"])
    no_execution(monkeypatch)
    assert confirmation.verify() == record and study.validate_plan(plan) == plan
    assert ledger.inspect(directory)["counts"]["total"] == 1


def test_existing_confirmation_data_blocks_publication_without_inspecting_it(tmp_path, monkeypatch):
    directory, _, plan, _ = ready(tmp_path, monkeypatch)
    folder = directory.parent / confirmation.DATA_DIRECTORY
    folder.mkdir()
    (folder / "truth.json").write_text("do not inspect pre-lock outcomes")
    public = primary.BINDINGS.read_bytes()
    no_execution(monkeypatch)
    with pytest.raises(ValueError, match="appeared before the confirmation lock"):
        confirmation.lock_confirmation(plan)
    assert primary.BINDINGS.read_bytes() == public
    assert not (directory.parent / "confirmation_lock.json").exists()


def test_prior_confirmation_intent_cannot_be_hidden_behind_an_operator_finish(tmp_path, monkeypatch):
    directory, _, plan, _ = ready(tmp_path, monkeypatch)
    study._publish(directory / "plans" / f"{plan['plan_sha256']}.json", plan)
    manifest = ledger._manifest(directory)
    # injected test-only history must never be mistaken for authorized model evidence
    start = study._seal({**ledger._base(manifest, plan, plan["slots"][0], 0), "record_kind": "start", "bucket": "primary",
                         "recorded_at_utc": datetime.now(UTC).isoformat()}, "record_sha256")
    study._publish(directory / "attempts" / f"{start['attempt_id']}.start.json", start)
    finish(directory, start)
    no_execution(monkeypatch)
    with pytest.raises(ValueError, match="appeared before the confirmation lock"):
        confirmation.lock_confirmation(plan)


@pytest.mark.parametrize("condition", ["active", "halt"])
def test_active_or_halted_accounting_prevents_locking(tmp_path, monkeypatch, condition):
    directory, _, plan, _ = ready(tmp_path, monkeypatch)
    smoke = study._json(directory.parent / "smoke_plan.json")
    if condition == "active":
        ledger.begin(directory, smoke["plan_sha256"], smoke["slots"][0]["slot_id"])
    else:
        monkeypatch.setattr(provider, "execute", lambda *args: simulate_provider_transport(*args, model_id="changed-model"))
        runner.run_provider(smoke["plan_sha256"], smoke["slots"][0]["slot_id"])
    no_execution(monkeypatch)
    with pytest.raises(ValueError, match="idle canonical accounting"):
        confirmation.lock_confirmation(plan)
    assert not (directory.parent / "confirmation_lock.json").exists()


@pytest.mark.parametrize("kind", ["record", "plan_file", "public_binding", "pilot_raw", "closed_budget"])
def test_changed_saved_lock_or_pilot_evidence_is_refused(tmp_path, monkeypatch, kind):
    directory, pilot, plan, _ = ready(tmp_path, monkeypatch, completed=kind == "pilot_raw")
    record = confirmation.lock_confirmation(plan)
    if kind == "record":
        record["planned_slots"] = 32
        (directory.parent / "confirmation_lock.json").write_text(json.dumps(study._seal(record, "record_sha256")))
    elif kind == "plan_file":
        (directory.parent / "confirmation_plan.json").write_text("{}")
    elif kind == "public_binding":
        public = study._json(primary.BINDINGS)
        public["records"][-1]["plan_sha256"] = "0" * 64
        primary.BINDINGS.write_text(json.dumps(public))
    elif kind == "pilot_raw":
        stdout = directory / "runner" / pilot["slots"][0]["attempt_ids"][0] / "stdout.jsonl"
        stdout.write_text(stdout.read_text() + " ")
    else:
        path = directory.parent / bridge.COLLECTION
        collection = study._json(path)
        collection["labs"][0]["selection"]["chosen_budget_fraction"] = 0.25
        path.write_text(json.dumps(study._seal(collection, "snapshot_sha256")))
    no_execution(monkeypatch)
    with pytest.raises(ValueError):
        confirmation.verify()


def test_zero_cohort_records_no_primary_comparison_and_no_pilot_or_model_work(tmp_path, monkeypatch):
    generation_ready(tmp_path, monkeypatch, effect="0.05 + (0.3 if seed % 2 else -0.3)")
    cohort.lock_cohort()
    no_execution(monkeypatch)
    plan = confirmation.make_plan()
    assert plan["groups"] == plan["slots"] == [] and plan["binding"]["pilot_snapshot_sha256"] is None
    record = confirmation.lock_confirmation(plan)
    assert record["status"] == "no_primary_comparison" and record["planned_slots"] == 0
    assert confirmation.verify() == record and not (primary.canonical_root() / bridge.COLLECTION).exists()


def test_eight_lab_grid_has_128_slots_and_a_larger_cohort_is_refused(tmp_path, monkeypatch):
    _, _, plan, _ = ready(tmp_path, monkeypatch)
    binding = copy.deepcopy(plan["binding"])
    original, decision = binding["environments"][0], binding["pilot_decisions"][0]
    # only the pure grid boundary is stubbed; these are not scientific admissions
    binding["environments"] = [{**original, "env_id": f"fixture-{index}", "lab_hash": study.canonical_hash({"fixture": index})} for index in range(8)]
    binding["pilot_decisions"] = [{**decision, "env_id": env["env_id"]} for env in binding["environments"]]
    groups, slots = confirmation._grid(binding, study.canonical_hash(binding))
    assert len(groups) == 16 and len(slots) == 128 and len({slot["slot_id"] for slot in slots}) == 128
    binding["environments"].append(original)
    with pytest.raises(ValueError, match="protocol ceiling"):
        confirmation._grid(binding, study.canonical_hash(binding))


def test_held_context_and_missing_primary_state_do_not_create_new_state(tmp_path, monkeypatch):
    directory, _, plan, _ = ready(tmp_path, monkeypatch)
    record = confirmation.lock_confirmation(plan)
    no_execution(monkeypatch)
    with ledger._locked(directory) as context:
        monkeypatch.setattr(ledger, "_locked", lambda *a: pytest.fail("nested lock while inspecting confirmation"))
        assert confirmation.verify(ledger_context=context) == record
        with pytest.raises(ValueError, match="context differs"):
            confirmation.verify(ledger_context=(tmp_path / "foreign", context[1]))
    monkeypatch.setattr(primary, "STATE_BASE", tmp_path / "missing")
    for operation in (confirmation.make_plan, confirmation.verify):
        with pytest.raises(FileNotFoundError):
            operation()
    assert not primary.STATE_BASE.exists()


@pytest.mark.parametrize("published", [False, True])
def test_partial_publication_or_lost_acknowledgement_never_replaces_the_lock(tmp_path, monkeypatch, published):
    directory, _, plan, _ = ready(tmp_path, monkeypatch)
    append = primary._append_binding

    def fail(record):
        if published:
            append(record)
        raise OSError("fixture lost public binding acknowledgement")

    with monkeypatch.context() as patch:
        patch.setattr(primary, "_append_binding", fail)
        with pytest.raises(OSError):
            confirmation.lock_confirmation(plan)
    no_execution(monkeypatch)
    before = (directory.parent / "confirmation_lock.json").read_bytes()
    with pytest.raises(FileExistsError, match="already locked"):
        confirmation.lock_confirmation(plan)
    assert (directory.parent / "confirmation_lock.json").read_bytes() == before
    if published:
        assert confirmation.verify()["study_lock"] is True
    else:
        with pytest.raises(ValueError, match="absent from public binding"):
            confirmation.verify()
    assert ledger.inspect(directory)["counts"]["total"] == 0


def test_cli_prepares_locks_and_inspects_without_execution_or_override(tmp_path, monkeypatch, capsys):
    ready(tmp_path, monkeypatch)
    no_execution(monkeypatch)
    output = tmp_path / "preparation"
    main(["study", "plan-confirmation", "--output-dir", str(output)])
    result = json.loads(capsys.readouterr().out)
    assert result["planned_slots"] == 16 and result["execution_ready"] is False
    main(["study", "lock-confirmation", str(output)])
    locked = json.loads(capsys.readouterr().out)
    assert locked["study_lock"] is True and locked["execution_ready"] is False and locked["model_sessions_started"] == 0
    main(["study", "inspect-confirmation"])
    assert json.loads(capsys.readouterr().out) == locked
    assert "mcp_url" not in json.dumps(locked) and "qualification_label" not in json.dumps(locked)
    with pytest.raises(SystemExit):
        main(["study", "plan-confirmation", "--output-dir", str(output), "--budget", "0.25"])
