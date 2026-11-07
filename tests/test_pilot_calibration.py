import copy
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from benchtop import fresh_generation as fresh
from benchtop import pilot_calibration as bridge
from benchtop import pilots, primary, provider, runner, study
from benchtop import session_ledger as ledger
from benchtop.cli import main
from tests.test_pilots import ready as prepared_cohort
from tests.test_primary import simulate_provider_transport
from tests.test_session_ledger import EVIDENCE, finish
from tests.test_study import model


def ready(tmp_path, monkeypatch, *, count=1):
    _, plan, directory = prepared_cohort(tmp_path, monkeypatch, count=count)
    pilots.authorize(plan)
    monkeypatch.setattr(provider, "execute", simulate_provider_transport)
    return directory, plan


def export(plan, tmp_path, *, close=False, name="selection.json"):
    return bridge.export_pilots(plan["plan_sha256"], tmp_path / name, close=close)


def no_execution(monkeypatch):
    monkeypatch.setattr(provider, "probe", lambda: pytest.fail("calibration queried live authentication"))
    monkeypatch.setattr(provider, "execute", lambda *a: pytest.fail("calibration launched a model"))
    monkeypatch.setattr(fresh, "sweep", lambda *a, **kw: pytest.fail("calibration ran a sweep"))


def run(plan, index=0, **kwargs):
    return runner.run_provider(plan["plan_sha256"], plan["slots"][index]["slot_id"], **kwargs)


def test_open_export_keeps_all_labs_slots_pending_intents_and_unknown_costs(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch, count=2)
    start = ledger.begin(directory, plan["plan_sha256"], plan["slots"][0]["slot_id"])
    no_execution(monkeypatch)
    bundle = export(plan, tmp_path)
    assert bundle["closed"] is False and len(bundle["labs"]) == 2 and len(bundle["groups"]) == 6
    assert [lab["env_id"] for lab in bundle["labs"]] == [env["env_id"] for env in plan["binding"]["environments"]]
    assert all(len(lab["input"]["slots"]) == 12 and lab["input"]["lab_ids"] == [lab["env_id"]] for lab in bundle["labs"])
    assert all(lab["selection"]["status"] == "pending" and lab["selection"]["chosen_budget_fraction"] is None for lab in bundle["labs"])
    assert bundle["labs"][0]["input"]["slots"][0]["attempts"] == [{"attempt_id": start["attempt_id"], "attempt_index": 0,
        "state": "unresolved", "reward": None, "failure_category": "pending_reconciliation"}]
    assert bundle["pilot_costs"]["episode_starts"] == 1 and bundle["pilot_costs"]["fields"]["input_tokens"]["total"] is None
    assert not (directory.parent / bridge.COLLECTION).exists()
    with pytest.raises(ValueError, match="in-flight"):
        export(plan, tmp_path, close=True, name="cannot-close.json")
    assert ledger.inspect(directory)["counts"]["active"] == 1


def test_full_two_lab_grid_selects_separate_budgets_and_frozen_exports_are_identical(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch, count=2)
    lab_ids = [member["env_id"] for member in plan["binding"]["environments"]]
    successes = {lab_ids[0]: {0.06: 0, 0.12: 2, 0.25: 4}, lab_ids[1]: {0.06: 1, 0.12: 4, 0.25: 4}}
    by_slot = {slot["slot_id"]: slot for slot in plan["slots"]}

    def controlled(work, timeout, prepared, proof):
        slot = by_slot[study._json(work / "connection.json")["slot_id"]]
        correct = slot["replicate_index"] < successes[slot["env_id"]][slot["budget_fraction"]]
        return simulate_provider_transport(work, timeout, prepared, proof, answer="paper_better" if correct else "baseline_better")

    monkeypatch.setattr(provider, "execute", controlled)
    with runner.Controller() as controller:
        for index in range(24):
            assert run(plan, index, controller=controller)["status"] == "completed"
    no_execution(monkeypatch)
    progress = export(plan, tmp_path)
    assert all(lab["selection"]["chosen_budget_fraction"] is None for lab in progress["labs"])
    bundle = export(plan, tmp_path, close=True, name="closed.json")
    assert bundle["closed"] is True and bundle["study_lock"] is False and bundle["model_sessions_started_by_export"] == 0
    assert [lab["selection"]["chosen_budget_fraction"] for lab in bundle["labs"]] == [0.12, 0.06]
    assert all(lab["selection"]["status"] == "selected" for lab in bundle["labs"])
    assert bundle["pilot_costs"]["episode_starts"] == 24 and bundle["pilot_costs"]["retry_episode_starts"] == 0
    assert all(len(lab["selection"]["pilot_attempt_ids"]) == 12 for lab in bundle["labs"])
    assert any(name.endswith("stdout.jsonl") for name in bundle["phase_evidence"])
    assert all(lab["input"]["global_budget_fraction"] == 0.12 for lab in bundle["labs"])
    assert bridge.load_closed() == bundle
    export(plan, tmp_path, name="copy.json")
    assert (tmp_path / "closed.json").read_bytes() == (tmp_path / "copy.json").read_bytes() == (directory.parent / bridge.COLLECTION).read_bytes()
    with pytest.raises(ValueError, match="collection is closed"):
        ledger.begin(directory, plan["plan_sha256"], plan["slots"][0]["slot_id"])
    assert runner.resume(directory, plan["slots"][0]["attempt_ids"][0])["outcome"]["reward"] == 0


def test_completed_all_wrong_pilots_use_the_fixed_no_mixed_budget_fallback(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch)
    monkeypatch.setattr(provider, "execute", lambda *args: simulate_provider_transport(*args, answer="baseline_better"))
    with runner.Controller() as controller:
        for index in range(12):
            run(plan, index, controller=controller)
    no_execution(monkeypatch)
    bundle = export(plan, tmp_path, close=True)
    selection = bundle["labs"][0]["selection"]
    assert selection["status"] == "fallback" and selection["fallback_reason"] == "no_observed_mixed_budget"
    assert selection["chosen_budget_fraction"] == bundle["global_budget_fraction"] == 0.12
    assert all(group["completed"] == 4 and group["correct"] == 0 and group["missing"] == 0 for group in selection["labs"][0]["budgets"])
    assert ledger.inspect(directory)["counts"]["total"] == 12


def test_incomplete_collection_uses_global_fallback_without_dropping_agent_zero_or_unknown(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch)
    monkeypatch.setattr(provider, "execute", lambda *args: simulate_provider_transport(*args, no_answer=True, final_error=True))
    unknown = run(plan)
    monkeypatch.setattr(provider, "execute", lambda *args: simulate_provider_transport(*args, no_answer=True))
    zero = run(plan, 1)
    assert unknown["status"] == "unresolved" and zero["status"] == "agent_failure"
    no_execution(monkeypatch)
    bundle = export(plan, tmp_path, close=True)
    selection = bundle["labs"][0]["selection"]
    assert selection["fallback_reason"] == "incomplete_pilots" and selection["chosen_budget_fraction"] == 0.12
    group = selection["labs"][0]["budgets"][0]
    assert group["completed"] == 1 and group["correct"] == 0 and group["missing"] == 3
    assert group["status_counts"] == {"unresolved": 1, "agent_failure": 1, "unattempted": 2}
    assert len(bundle["labs"][0]["input"]["slots"]) == 12 and bundle["pilot_costs"]["episode_starts"] == 2
    with pytest.raises(ValueError, match="collection is closed"):
        ledger.reconcile(directory, unknown["attempt_id"], {"state": "agent_failure", "reward": 0,
            "failure_category": "episode_timeout"}, reported_model_id=model()["reported_model_id"], process_stopped=True,
            submission_status="not_committed", evidence_sha256=EVIDENCE)
    assert runner.resume(directory, unknown["attempt_id"])["outcome"]["state"] == "unresolved"


@pytest.mark.parametrize("defect", ["raw", "opaque", "wrong_phase", "authorization", "source", "halt", "confirmation"])
def test_unverified_wrong_phase_or_changed_evidence_cannot_select_budgets(tmp_path, monkeypatch, defect):
    directory, plan = ready(tmp_path, monkeypatch)
    if defect == "raw":
        result = run(plan)
        path = directory / "runner" / result["attempt_id"] / "stdout.jsonl"
        path.write_text(path.read_text() + " ")
    elif defect == "opaque":
        start = ledger.begin(directory, plan["plan_sha256"], plan["slots"][0]["slot_id"])
        finish(directory, start, reward=1)
    elif defect == "wrong_phase":
        plan = study._json(directory.parent / "smoke_plan.json")
    elif defect == "authorization":
        (directory.parent / "pilot_authorization.json").unlink()
    elif defect == "source":
        monkeypatch.setattr(study, "_sources", lambda: {"changed": "fixture"})
    elif defect == "halt":
        monkeypatch.setattr(provider, "execute", lambda *args: simulate_provider_transport(*args, model_id="changed-model"))
        run(plan)
    else:
        (directory.parent / "confirmation_lock.json").write_text('{"fixture": "premature confirmation"}')
    no_execution(monkeypatch)
    with pytest.raises((ValueError, FileNotFoundError)):
        export(plan, tmp_path, close=True)
    assert not (tmp_path / "selection.json").exists() and not (directory.parent / bridge.COLLECTION).exists()


@pytest.mark.parametrize("field", ["budget", "slot", "cohort", "accounting"])
def test_resealed_closed_record_changes_are_refused(tmp_path, monkeypatch, field):
    directory, plan = ready(tmp_path, monkeypatch)
    no_execution(monkeypatch)
    bundle = export(plan, tmp_path, close=True)
    altered = copy.deepcopy(bundle)
    if field == "budget":
        altered["labs"][0]["selection"]["chosen_budget_fraction"] = 0.25
    elif field == "slot":
        altered["labs"][0]["input"]["slots"].pop()
    elif field == "cohort":
        altered["cohort_lock_sha256"] = "0" * 64
    else:
        altered["accounting"]["shared_ledger_counts_at_snapshot"]["total"] = -1
    (directory.parent / bridge.COLLECTION).write_text(json.dumps(study._seal(altered, "snapshot_sha256")))
    with pytest.raises(ValueError):
        bridge.load_closed()


def test_closed_export_stays_identical_as_other_phase_accounting_grows(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch)
    bundle = export(plan, tmp_path, close=True)
    smoke = study._json(directory.parent / "smoke_plan.json")
    runner.run_provider(smoke["plan_sha256"], smoke["slots"][0]["slot_id"])
    no_execution(monkeypatch)
    assert ledger.inspect(directory)["counts"]["total"] == 1
    assert bridge.load_closed() == bundle and bundle["accounting"]["shared_ledger_counts_at_snapshot"]["total"] == 0
    export(plan, tmp_path, name="later.json")
    assert (tmp_path / "selection.json").read_bytes() == (tmp_path / "later.json").read_bytes()


def test_held_context_does_not_acquire_a_nested_lock_and_copies_are_refused(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch)
    no_execution(monkeypatch)
    bundle = export(plan, tmp_path, close=True)
    with ledger._locked(directory) as context:
        monkeypatch.setattr(ledger, "_locked", lambda *a: pytest.fail("nested flock during closed pilot verification"))
        assert bridge.load_closed(ledger_context=context) == bundle
        with pytest.raises(ValueError, match="context differs"):
            bridge.load_closed(ledger_context=(tmp_path / "foreign", context[1]))


def test_retry_cost_normalization_keeps_two_intents_and_one_completed_slot(tmp_path, monkeypatch):
    _, plan = ready(tmp_path, monkeypatch)
    state = {"histories": {(plan["plan_sha256"], slot["slot_id"]): [] for slot in plan["slots"]}}
    slot = plan["slots"][0]
    outcomes = [{"state": "technical_failure", "reward": None, "failure_category": "transport_failure",
                 "costs": {"charged_experiment_cost": 2, "input_tokens": None}},
                {"state": "completed", "reward": 0, "costs": {"charged_experiment_cost": 3, "input_tokens": 7}}]
    for index, outcome in enumerate(outcomes):
        outcome.update(attempt_id=slot["attempt_ids"][index], attempt_index=index)
    state["histories"][(plan["plan_sha256"], slot["slot_id"])] = [{"effective": {"outcome": row}} for row in outcomes]
    selection = bridge._labs(plan, state, closed=True)[0]["selection"]
    assert selection["upfront_cost"]["episode_starts"] == 2 and selection["upfront_cost"]["retry_episode_starts"] == 1
    assert selection["upfront_cost"]["fields"]["charged_experiment_cost"]["total"] == 5
    assert selection["upfront_cost"]["fields"]["input_tokens"]["known_sum"] == 7
    assert selection["upfront_cost"]["fields"]["input_tokens"]["total"] is None
    assert selection["labs"][0]["budgets"][0]["completed"] == 1


def test_closure_and_start_serialize_under_the_shared_journal_lock(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch)
    barrier = Barrier(2)

    def close():
        barrier.wait()
        try:
            export(plan, tmp_path, close=True)
            return "closed"
        except ValueError as error:
            assert "in-flight" in str(error)
            return "pending"

    def start():
        barrier.wait()
        try:
            ledger.begin(directory, plan["plan_sha256"], plan["slots"][0]["slot_id"])
            return "started"
        except ValueError as error:
            assert "collection is closed" in str(error)
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as pool:
        closing, starting = pool.submit(close), pool.submit(start)
        result = closing.result(timeout=20), starting.result(timeout=20)
    assert result in (("closed", "blocked"), ("pending", "started"))


def test_exclusive_output_and_invalid_destinations_do_not_close_collection(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch)
    no_execution(monkeypatch)
    existing = tmp_path / "existing.json"
    existing.write_text("preserve caller data")
    for path, error in ((existing, FileExistsError), (directory.parent / "forbidden.json", ValueError)):
        with pytest.raises(error):
            bridge.export_pilots(plan["plan_sha256"], path, close=True)
    with pytest.raises(ValueError, match="explicit boolean"):
        bridge.export_pilots(plan["plan_sha256"], tmp_path / "bad.json", close="yes")
    with pytest.raises(FileExistsError):
        bridge.export_pilots(plan["plan_sha256"], existing / "child.json", close=True)
    assert existing.read_text() == "preserve caller data" and not (directory.parent / bridge.COLLECTION).exists()


def test_missing_or_copied_primary_state_cannot_export(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch)
    copied = tmp_path / "copy"
    shutil.copytree(directory, copied)
    with ledger._locked(copied) as (_, manifest), pytest.raises(ValueError, match="canonical journal"):
        bridge._context(copied, manifest, plan["plan_sha256"])
    monkeypatch.setattr(primary, "STATE_BASE", tmp_path / "missing")
    no_execution(monkeypatch)
    with pytest.raises(FileNotFoundError):
        export(plan, tmp_path)
    assert not primary.STATE_BASE.exists() and not (tmp_path / "selection.json").exists()


def test_cli_reports_individual_fallbacks_without_private_process_details(tmp_path, monkeypatch, capsys):
    directory, plan = ready(tmp_path, monkeypatch)
    no_execution(monkeypatch)
    main(["study", "calibrate-pilots", plan["plan_sha256"], "--output", str(tmp_path / "cli.json"), "--close"])
    result = json.loads(capsys.readouterr().out)
    assert result["closed"] is True and result["model_sessions"] == 0 and result["cohort_size"] == 1
    assert result["labs"][0]["fallback_reason"] == "incomplete_pilots" and result["labs"][0]["chosen_budget_fraction"] == 0.12
    assert "mcp_url" not in json.dumps(result) and "account_sha256" not in json.dumps(result)
    assert (directory.parent / bridge.COLLECTION).exists()


def test_lost_export_acknowledgement_preserves_closure_and_reexports_identical_bytes(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch)
    no_execution(monkeypatch)
    publish = study._publish

    def lose_output(path, value):
        if path == tmp_path / "selection.json":
            raise OSError("fixture lost output acknowledgement")
        return publish(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(study, "_publish", lose_output)
        with pytest.raises(OSError, match="lost output acknowledgement"):
            export(plan, tmp_path, close=True)
    assert (directory.parent / bridge.COLLECTION).exists() and not (tmp_path / "selection.json").exists()
    with pytest.raises(ValueError, match="collection is closed"):
        ledger.begin(directory, plan["plan_sha256"], plan["slots"][0]["slot_id"])
    recovered = export(plan, tmp_path, name="recovered.json")
    assert recovered == bridge.load_closed()
    assert (tmp_path / "recovered.json").read_bytes() == (directory.parent / bridge.COLLECTION).read_bytes()
    assert ledger.inspect(directory)["counts"]["total"] == 0
