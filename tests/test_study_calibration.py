import copy
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from benchtop import calibrate, primary, provider, runner, study, study_calibration
from benchtop import session_ledger as ledger
from benchtop.cli import main
from tests.test_primary import authorize, preparation, simulate_provider_transport, smoke_runs
from tests.test_session_ledger import EVIDENCE
from tests.test_study import model, ready_controls


def ready(tmp_path, monkeypatch):
    plans = preparation(tmp_path, monkeypatch)
    directory = authorize(plans)
    smoke_runs(plans, monkeypatch)
    primary.lock_development(plans["development"])
    return directory, plans["development"]


def run(plan, index=0, **kwargs):
    return runner.run_provider(plan["plan_sha256"], plan["slots"][index]["slot_id"], **kwargs)


def export(plan, tmp_path, *, close=False, name="selection.json"):
    return study_calibration.export_development(plan["plan_sha256"], tmp_path / name, close=close)


def no_model_calls(monkeypatch):
    monkeypatch.setattr(provider, "probe", lambda: pytest.fail("offline calibration inspected live authentication"))
    monkeypatch.setattr(provider, "execute", lambda *a, **kw: pytest.fail("calibration started a model"))


def test_open_export_retains_all_48_slots_pending_intents_and_unknown_costs(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch)
    start = ledger.begin(directory, plan["plan_sha256"], plan["slots"][0]["slot_id"])
    no_model_calls(monkeypatch)
    bundle = export(plan, tmp_path)
    assert len(bundle["input"]["slots"]) == 48 and len(bundle["groups"]) == 12
    assert bundle["selection"]["status"] == "pending" and bundle["selection"]["chosen_budget_fraction"] is None
    assert bundle["input"]["slots"][0]["attempts"] == [{"attempt_id": start["attempt_id"], "attempt_index": 0,
        "state": "unresolved", "reward": None, "failure_category": "pending_reconciliation"}]
    assert all(slot["attempts"] == [] for slot in bundle["input"]["slots"][1:])
    assert bundle["selection"]["upfront_cost"]["episode_starts"] == 1
    assert bundle["selection"]["upfront_cost"]["fields"]["input_tokens"]["total"] is None
    assert bundle["accounting"]["shared_ledger_counts_at_snapshot"]["total"] == 7
    assert bundle["accounting"]["smoke_included_in_selection"] is False
    assert not (directory.parent / study_calibration.COLLECTION).exists()
    assert bundle["model_sessions_started_by_export"] == 0
    assert (tmp_path / "selection.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="in-flight"):
        export(plan, tmp_path, close=True, name="cannot-close.json")
    assert ledger.inspect(directory)["counts"]["active"] == 1


def test_full_fixture_selects_expected_budget_and_closed_exports_are_identical(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch)
    labels = {row["env"]: row["claim_label"] for row in plan["binding"]["readiness"]["admission"]["references"]}
    slots = {slot["slot_id"]: slot for slot in plan["slots"]}
    correct_per_budget = {0.06: 0, 0.12: 2, 0.25: 4}

    def controlled(work, timeout, prepared, proof):
        connection = study._json(work / "connection.json")
        slot = slots[connection["slot_id"]]
        answer = labels[slot["env_id"]]
        if slot["replicate_index"] >= correct_per_budget[slot["budget_fraction"]]:
            answer = next(label for label in ("paper_better", "baseline_better", "no_meaningful_difference") if label != answer)
        return simulate_provider_transport(work, timeout, prepared, proof, answer=answer)

    monkeypatch.setattr(provider, "execute", controlled)
    with runner.Controller() as controller:
        for index in range(48):
            assert run(plan, index, controller=controller)["status"] == "completed"
    no_model_calls(monkeypatch)
    bundle = export(plan, tmp_path, close=True)
    assert bundle["selection"]["status"] == "selected" and bundle["selection"]["chosen_budget_fraction"] == 0.12
    assert bundle["selection"] == calibrate.select_budget(bundle["input"])
    assert all([group["correct"] for group in lab["budgets"]] == [0, 2, 4] for lab in bundle["selection"]["labs"])
    assert len(bundle["selection"]["pilot_attempt_ids"]) == 48
    assert bundle["selection"]["upfront_cost"]["fields"]["input_tokens"]["total"] == 48 * 12
    assert bundle["accounting"]["shared_ledger_counts_at_snapshot"]["total"] == 54
    assert bundle["accounting"]["shared_ledger_costs_at_snapshot"]["recorded_start_intents"] == 54
    assert export(plan, tmp_path, name="second.json") == bundle == study_calibration.load_closed()
    assert (tmp_path / "selection.json").read_bytes() == (tmp_path / "second.json").read_bytes()
    assert (directory.parent / study_calibration.COLLECTION).read_bytes() == (tmp_path / "selection.json").read_bytes()
    assert bundle["study_lock"] is False


def test_closed_incomplete_development_never_selects_and_blocks_new_starts(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch)
    no_model_calls(monkeypatch)
    bundle = export(plan, tmp_path, close=True)
    assert bundle["selection"]["status"] == "incomplete_development"
    assert bundle["selection"]["chosen_budget_fraction"] is None
    with pytest.raises(ValueError, match="collection is closed"):
        ledger.begin(directory, plan["plan_sha256"], plan["slots"][0]["slot_id"])
    assert ledger.inspect(directory)["counts"]["total"] == 6


def test_verified_agent_zero_and_unknown_provider_outcome_keep_distinct_denominators(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch)
    monkeypatch.setattr(provider, "execute", lambda *args: simulate_provider_transport(*args, no_answer=True, final_error=True))
    missing = run(plan)

    def timeout(*args):
        process = simulate_provider_transport(*args, no_answer=True, final_error=True)
        process["timed_out"] = True
        return process

    monkeypatch.setattr(provider, "execute", timeout)
    zero = run(plan, 1)
    assert missing["status"] == "unresolved" and zero["status"] == "agent_failure"
    no_model_calls(monkeypatch)
    bundle = export(plan, tmp_path, close=True)
    group = bundle["selection"]["labs"][0]["budgets"][0]
    assert group["completed"] == 1 and group["correct"] == 0 and group["missing"] == 3
    assert group["status_counts"] == {"unresolved": 1, "agent_failure": 1, "unattempted": 2}
    assert bundle["selection"]["upfront_cost"]["episode_starts"] == 2
    assert bundle["selection"]["upfront_cost"]["fields"]["input_tokens"]["known_sum"] == 24
    with pytest.raises(ValueError, match="collection is closed"):
        ledger.reconcile(directory, missing["attempt_id"], {"state": "agent_failure", "reward": 0,
                         "failure_category": "episode_timeout"}, reported_model_id=model()["reported_model_id"],
                         process_stopped=True, submission_status="not_committed", evidence_sha256=EVIDENCE)
    assert runner.resume(directory, missing["attempt_id"])["outcome"]["state"] == "unresolved"


def test_reconciled_answer_counts_one_start_and_keeps_original_unknown_record(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch)
    original_finish = ledger.finish

    def ambiguous(directory, attempt, outcome, **evidence):
        return original_finish(directory, attempt, {"state": "unresolved", "reward": None,
            "failure_category": "fixture_uncertain_acknowledgement", "costs": outcome["costs"]}, **evidence)

    with monkeypatch.context() as patch:
        patch.setattr(ledger, "finish", ambiguous)
        result = run(plan)
    path = directory / "attempts" / f"{result['attempt_id']}.finish.json"
    before = path.read_bytes()
    runner.resume(directory, result["attempt_id"])
    no_model_calls(monkeypatch)
    bundle = export(plan, tmp_path)
    attempts = bundle["input"]["slots"][0]["attempts"]
    assert len(attempts) == 1 and attempts[0]["state"] == "completed"
    assert bundle["selection"]["upfront_cost"]["episode_starts"] == 1
    assert path.read_bytes() == before
    assert f"attempts/{result['attempt_id']}.resolve.json" in bundle["phase_evidence"]


@pytest.mark.parametrize("defect", ["raw", "opaque", "smoke_phase", "lock", "source", "halt"])
def test_unverified_or_wrong_phase_records_never_become_selector_inputs(tmp_path, monkeypatch, defect):
    directory, plan = ready(tmp_path, monkeypatch)
    if defect == "raw":
        result = run(plan)
        path = directory / "runner" / result["attempt_id"] / "stdout.jsonl"
        path.write_text(path.read_text() + " ")
    elif defect == "opaque":
        start = ledger.begin(directory, plan["plan_sha256"], plan["slots"][0]["slot_id"])
        ledger.finish(directory, start["attempt_id"], {"state": "completed", "reward": 1},
                      reported_model_id=model()["reported_model_id"], process_stopped=True,
                      submission_status="committed", evidence_sha256=EVIDENCE)
    elif defect == "smoke_phase":
        plan = study._json(directory.parent / "smoke_plan.json")
    elif defect == "lock":
        primary.BINDINGS.unlink()
    elif defect == "source":
        monkeypatch.setattr(study, "_sources", lambda: {"changed": "fixture"})
    else:
        monkeypatch.setattr(provider, "execute", lambda *args: simulate_provider_transport(*args, model_id="different-fixture"))
        run(plan)
    no_model_calls(monkeypatch)
    with pytest.raises((ValueError, FileNotFoundError)):
        export(plan, tmp_path, close=True)
    assert not (tmp_path / "selection.json").exists()
    assert not (directory.parent / study_calibration.COLLECTION).exists()


def test_outputs_are_exclusive_and_destination_errors_do_not_close_collection(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch)
    no_model_calls(monkeypatch)
    output = tmp_path / "existing.json"
    output.write_text("preserve existing result")
    with pytest.raises(FileExistsError):
        study_calibration.export_development(plan["plan_sha256"], output, close=True)
    with pytest.raises(ValueError, match="outside the canonical"):
        study_calibration.export_development(plan["plan_sha256"], directory / "wrong.json", close=True)
    assert output.read_text() == "preserve existing result" and not (directory.parent / study_calibration.COLLECTION).exists()


def test_closed_selector_cannot_be_replaced_by_a_resealed_different_choice(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch)
    no_model_calls(monkeypatch)
    bundle = export(plan, tmp_path, close=True)
    altered = copy.deepcopy(bundle)
    altered["selection"]["chosen_budget_fraction"] = 0.25
    (directory.parent / study_calibration.COLLECTION).write_text(json.dumps(study._seal(altered, "snapshot_sha256")))
    with pytest.raises(ValueError, match="differs from the bound selector"):
        study_calibration.load_closed()


def test_cli_uses_only_canonical_records_and_reports_no_model_calls(tmp_path, monkeypatch, capsys):
    directory, plan = ready(tmp_path, monkeypatch)
    no_model_calls(monkeypatch)
    main(["study", "calibrate-development", plan["plan_sha256"], "--output", str(tmp_path / "cli.json"), "--close"])
    result = json.loads(capsys.readouterr().out)
    assert result["closed"] is True and result["status"] == "incomplete_development" and result["model_sessions"] == 0
    assert "mcp_url" not in json.dumps(result) and "account_sha256" not in json.dumps(result)
    assert (directory.parent / study_calibration.COLLECTION).exists()


def test_retry_normalization_retains_two_cost_records_and_one_final_outcome(tmp_path):
    reviews, foundation = ready_controls(tmp_path)
    plan = study.make_plan(model=model(), mode="study_preparation", phase="development", reviews_dir=reviews, foundation=foundation)
    state = {"histories": {(plan["plan_sha256"], slot["slot_id"]): [] for slot in plan["slots"]}}
    slot = plan["slots"][0]
    outcomes = [{"state": "technical_failure", "reward": None, "failure_category": "transport_failure",
                 "costs": {"charged_experiment_cost": 2, "input_tokens": None}},
                {"state": "completed", "reward": 0, "costs": {"charged_experiment_cost": 3, "input_tokens": 7}}]
    for index, outcome in enumerate(outcomes):
        outcome.update(attempt_id=slot["attempt_ids"][index], attempt_index=index)
    state["histories"][(plan["plan_sha256"], slot["slot_id"])] = [{"effective": {"outcome": row}} for row in outcomes]
    before = copy.deepcopy(state)
    document = study_calibration._input(plan, state, closed=False)
    selection = calibrate.select_budget(document)
    assert selection["upfront_cost"]["episode_starts"] == 2 and selection["upfront_cost"]["retry_episode_starts"] == 1
    assert selection["upfront_cost"]["fields"]["charged_experiment_cost"]["total"] == 5
    assert selection["upfront_cost"]["fields"]["input_tokens"]["known_sum"] == 7
    assert selection["upfront_cost"]["fields"]["input_tokens"]["total"] is None
    assert selection["labs"][0]["budgets"][0]["completed"] == 1 and state == before


def test_close_and_begin_serialize_under_the_same_journal_lock(tmp_path, monkeypatch):
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
        results = closing.result(), starting.result()
    assert results in (("closed", "blocked"), ("pending", "started"))
    assert (directory.parent / study_calibration.COLLECTION).exists() == (ledger.inspect(directory)["counts"]["active"] == 0)


def test_engineering_records_cannot_be_promoted_through_the_calibration_bridge(tmp_path, monkeypatch):
    from tests.test_session_ledger import fixture

    directory, plan, _ = fixture(tmp_path)
    canonical = tmp_path / "private/study/ledger"
    canonical.parent.mkdir(parents=True)
    monkeypatch.setattr(primary, "canonical_root", lambda: canonical.parent)
    directory.rename(canonical)
    no_model_calls(monkeypatch)
    with pytest.raises(ValueError, match="not engineering"):
        export(plan, tmp_path)
    assert not (tmp_path / "selection.json").exists()


def test_operator_only_technical_retry_cannot_hide_an_unverified_first_attempt(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch)
    slot = plan["slots"][0]
    start = ledger.begin(directory, plan["plan_sha256"], slot["slot_id"])
    ledger.finish(directory, start["attempt_id"], {"state": "technical_failure", "reward": None,
                  "failure_category": "transport_failure", "costs": {"input_tokens": 4}},
                  reported_model_id=model()["reported_model_id"], submission_status="not_committed",
                  process_stopped=True, evidence_sha256=EVIDENCE)
    result = run(plan, retry_technical=True)
    assert result["status"] == "completed" and ledger.inspect(directory)["counts"]["retry"] == 1
    no_model_calls(monkeypatch)
    with pytest.raises(ValueError, match="lack replay-verifiable"):
        export(plan, tmp_path, close=True)
    assert ledger.inspect(directory)["costs"]["fields"]["input_tokens"]["known_sum"] == 6 * 12 + 4 + 12
    assert not (directory.parent / study_calibration.COLLECTION).exists()
