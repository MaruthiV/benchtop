import copy
import json
import math
from fractions import Fraction

import pytest

from benchtop import calibrate
from benchtop.cache import file_hash
from benchtop.cli import main
from benchtop.env import ROOT


def fixture(*, phase="fresh_pilot", counts=(0, 2, 4), closed=True):
    spec = json.loads(calibrate.SPEC.read_text())
    labs = [f"synthetic-selector-{index}" for index in range(4 if phase == "development" else 1)]
    document = {"schema_version": 1, "evidence_kind": "synthetic_control", "protocol_hash": calibrate.canonical_hash(spec),
                "phase": phase, "closed": closed, "lab_ids": labs, "slots": []}
    if phase == "fresh_pilot":
        document["global_budget_fraction"] = 0.25
    for lab in labs:
        for budget, correct in zip(spec["budget"]["fractions"], counts, strict=True):
            group_id = calibrate.canonical_hash({"synthetic_lab": lab, "budget": budget, "phase": phase})
            for index in range(4):
                slot_id = calibrate.canonical_hash({"group_id": group_id, "replicate_index": index})
                attempt_id = calibrate.canonical_hash({"slot_id": slot_id, "attempt_index": 0})
                document["slots"].append({"lab_id": lab, "budget_fraction": budget, "group_id": group_id,
                    "slot_id": slot_id, "replicate_index": index, "attempts": [
                        {"attempt_id": attempt_id, "attempt_index": 0, "state": "completed", "reward": int(index < correct)}]})
    return document


@pytest.mark.parametrize("correct,expected", [(0, Fraction(791, 1287)), (1, Fraction(371, 429)),
                                             (2, Fraction(133, 143)), (3, Fraction(371, 429)), (4, Fraction(791, 1287))])
def test_predictive_objective_matches_exact_beta_binomial_mixed_mass(correct, expected):
    summary = calibrate.posterior(correct, 4)
    a, b = correct + 1, 5 - correct
    def beta_integer(x, y):
        return Fraction(math.factorial(x - 1) * math.factorial(y - 1), math.factorial(x + y - 1))
    enumerated = sum(math.comb(8, k) * beta_integer(a + k, b + 8 - k) / beta_integer(a, b) for k in range(1, 8))
    assert enumerated == expected
    assert summary["mixed_group_probability"] == pytest.approx(float(expected), abs=1e-14)


def test_uniform_prior_interval_and_predictive_probability_are_not_plugin_estimates():
    result = calibrate.posterior(0, 0)
    assert result["success_probability_interval"] == pytest.approx([0.025, 0.975])
    assert result["mixed_group_probability"] == pytest.approx(7 / 9)
    assert result["mixed_group_probability"] != pytest.approx(1 - 2 * 0.5**8)
    result = calibrate.posterior(4, 4)
    assert result["success_probability_interval"] == pytest.approx([0.025**0.2, 0.975**0.2])


@pytest.mark.parametrize("args,kwargs", [((-1, 4), {}), ((5, 4), {}), ((True, 4), {}), ((2, 4.0), {}),
                                        ((1, 4), {"group_size": 1}), ((1, 4), {"confidence": float("nan")})])
def test_invalid_posterior_counts_or_settings_are_rejected(args, kwargs):
    with pytest.raises(ValueError):
        calibrate.posterior(*args, **kwargs)


def test_fresh_selection_is_deterministic_bound_and_does_not_mutate_input():
    document = fixture()
    before = copy.deepcopy(document)
    result = calibrate.select_budget(document)
    assert result["status"] == "selected" and result["chosen_budget_fraction"] == 0.12
    assert result["fallback_reason"] is None and result["evidence_kind"] == "synthetic_control"
    assert document == before and result == calibrate.select_budget(document)
    checksum = result.pop("selection_sha256")
    assert checksum == calibrate.canonical_hash(result)
    assert result["binding"]["input_sha256"] == calibrate.canonical_hash(document)
    assert result["binding"]["protocol_file_sha256"] == file_hash(calibrate.SPEC)
    assert result["binding"]["selector_sha256"] == file_hash(calibrate.__file__)


def test_near_best_prefers_smallest_budget_including_boundary():
    rows = [{"budget_fraction": 0.25, "objective": 0.9}, {"budget_fraction": 0.06, "objective": 0.85},
            {"budget_fraction": 0.12, "objective": 0.88}]
    assert calibrate._choose(rows, 0.05) == 0.06
    rows[1]["objective"] = 0.84999
    assert calibrate._choose(rows, 0.05) == 0.12
    assert calibrate.select_budget(fixture(counts=(1, 3, 2)))["chosen_budget_fraction"] == 0.25
    assert calibrate.select_budget(fixture(counts=(1, 3, 1)))["chosen_budget_fraction"] == 0.06


@pytest.mark.parametrize("counts", [(0, 0, 0), (4, 4, 4), (0, 4, 0)])
def test_no_observed_mixed_budget_uses_global_control(counts):
    result = calibrate.select_budget(fixture(counts=counts))
    assert result["status"] == "fallback" and result["chosen_budget_fraction"] == 0.25
    assert result["fallback_reason"] == "no_observed_mixed_budget"
    assert result["diagnostics"]


def test_open_collection_cannot_select_or_prematurely_fallback():
    for counts in ((0, 2, 4), (0, 0, 0)):
        document = fixture(counts=counts, closed=False)
        for incomplete in (False, True):
            if incomplete:
                document["slots"][0]["attempts"] = []
            result = calibrate.select_budget(document)
            assert result["status"] == "pending" and result["chosen_budget_fraction"] is None
            assert result["fallback_reason"] is None


@pytest.mark.parametrize("missing", ["unattempted", "technical_failure", "unresolved"])
def test_missing_outcomes_are_not_zero_and_closed_fresh_input_falls_back(missing):
    document = fixture()
    attempts = document["slots"][0]["attempts"]
    if missing == "unattempted":
        attempts.clear()
    else:
        attempts[0].update(state=missing, reward=None, failure_category="provider_quota" if missing == "technical_failure" else "unclassified_control")
    result = calibrate.select_budget(document)
    assert result["chosen_budget_fraction"] == 0.25 and result["fallback_reason"] == "incomplete_pilots"
    group = result["labs"][0]["budgets"][0]
    assert group["completed"] == 3 and group["missing"] == 1
    assert result["objectives"][0]["objective"] is None


def test_agent_failures_are_completed_zero_rewards():
    document = fixture(counts=(1, 1, 1))
    document["slots"][0]["attempts"][0].update(state="agent_failure", reward=0, failure_category="episode_timeout")
    result = calibrate.select_budget(document)
    group = result["labs"][0]["budgets"][0]
    assert group["completed"] == 4 and group["correct"] == 0 and group["missing"] == 0
    assert group["status_counts"] == {"agent_failure": 1, "completed": 3}


def test_retry_preserves_costs_and_contributes_one_final_outcome():
    document = fixture()
    slot = document["slots"][4]
    slot["attempts"][0].update(state="technical_failure", reward=None, failure_category="transport_failure",
                               costs={"charged_experiment_cost": 2, "input_tokens": None})
    slot["attempts"].append({"attempt_id": calibrate.canonical_hash({"slot_id": slot["slot_id"], "attempt_index": 1}),
        "attempt_index": 1, "state": "completed", "reward": 1, "costs": {"charged_experiment_cost": 3, "input_tokens": 7}})
    result = calibrate.select_budget(document)
    assert result["labs"][0]["budgets"][1]["correct"] == 2
    assert result["upfront_cost"]["episode_starts"] == 13 and result["upfront_cost"]["retry_episode_starts"] == 1
    costs = result["upfront_cost"]["fields"]
    assert costs["charged_experiment_cost"] == {"known_sum": 5.0, "unreported_attempts": 11,
                                               "total": None, "availability": "partial_or_unreported"}
    assert costs["input_tokens"]["known_sum"] == 7 and costs["input_tokens"]["total"] is None


@pytest.mark.parametrize("first_state", ["completed", "agent_failure", "unresolved"])
def test_terminal_or_unclassified_outcomes_cannot_be_replaced_by_retry(first_state):
    document = fixture()
    slot = document["slots"][0]
    first = slot["attempts"][0]
    if first_state == "agent_failure":
        first.update(state=first_state, reward=0, failure_category="episode_timeout")
    elif first_state == "unresolved":
        first.update(state=first_state, reward=None, failure_category="unknown")
    slot["attempts"].append({"attempt_id": calibrate.canonical_hash({"slot_id": slot["slot_id"], "attempt_index": 1}),
                             "attempt_index": 1, "state": "completed", "reward": 1})
    with pytest.raises(ValueError, match="cannot be retried"):
        calibrate.select_budget(document)


def test_more_than_one_technical_retry_is_rejected():
    document = fixture()
    slot = document["slots"][0]
    slot["attempts"] = [{"attempt_id": calibrate.canonical_hash({"slot_id": slot["slot_id"], "attempt_index": index}),
                         "attempt_index": index, "state": "technical_failure", "reward": None,
                         "failure_category": "provider_quota"} for index in range(3)]
    with pytest.raises(ValueError, match="at most one"):
        calibrate.select_budget(document)


def test_input_order_does_not_change_scientific_selection_or_counts():
    document = fixture(phase="development")
    first = calibrate.select_budget(document)
    document["slots"].reverse()
    second = calibrate.select_budget(document)
    assert first["chosen_budget_fraction"] == second["chosen_budget_fraction"]
    assert first["labs"] == second["labs"] and first["upfront_cost"] == second["upfront_cost"]
    assert first["binding"]["input_sha256"] != second["binding"]["input_sha256"]


def test_numeric_overflow_is_a_visible_input_error():
    document = fixture()
    for slot in document["slots"]:
        slot["attempts"][0]["costs"] = {"model_wall_seconds": 1e308}
    with pytest.raises(ValueError, match="overflow"):
        calibrate.select_budget(document)
    document["slots"][0]["attempts"][0]["costs"] = {"input_tokens": 10**1000}
    with pytest.raises(ValueError, match="finite nonnegative"):
        calibrate.select_budget(document)


def test_development_averages_all_four_labs_and_has_no_fresh_fallback():
    document = fixture(phase="development", counts=(0, 0, 0))
    for slot in document["slots"]:
        if slot["budget_fraction"] == 0.12 and slot["lab_id"] != "synthetic-selector-0":
            slot["attempts"][0]["reward"] = int(slot["replicate_index"] < 2)
    result = calibrate.select_budget(document)
    assert result["status"] == "selected" and result["chosen_budget_fraction"] == 0.12
    expected = (3 * float(Fraction(133, 143)) + float(Fraction(791, 1287))) / 4
    assert result["objectives"][1]["objective"] == pytest.approx(expected)
    assert result["upfront_cost"]["episode_starts"] == 48
    assert calibrate.select_budget(fixture(phase="development", counts=(4, 0, 4)))["chosen_budget_fraction"] == 0.06
    document["slots"][0]["attempts"] = []
    result = calibrate.select_budget(document)
    assert result["status"] == "incomplete_development" and result["chosen_budget_fraction"] is None
    assert result["objectives"][0]["complete_labs"] == 3 and result["objectives"][0]["objective"] is None


@pytest.mark.parametrize("defect", ["grid", "duplicate", "slot_id", "attempt_id", "group_reuse", "nonbinary", "boolean_reward",
                                    "technical_zero", "unknown_cause", "agent_success", "confirmation_phase", "protocol", "extra_field",
                                    "unknown_cost", "negative_cost", "nan_cost", "fractional_tokens", "boolean_cost", "lab_count", "global_budget"])
def test_invalid_or_contaminated_normalized_inputs_are_rejected(defect):
    document = fixture()
    slot, attempt = document["slots"][0], document["slots"][0]["attempts"][0]
    if defect == "grid":
        document["slots"].pop()
    elif defect == "duplicate":
        document["slots"][1] = copy.deepcopy(slot)
    elif defect == "slot_id":
        slot["slot_id"] = "0" * 64
    elif defect == "attempt_id":
        attempt["attempt_id"] = "0" * 64
    elif defect == "group_reuse":
        document["slots"][4]["group_id"] = slot["group_id"]
    elif defect in ("nonbinary", "boolean_reward"):
        attempt["reward"] = 0.5 if defect == "nonbinary" else True
    elif defect in ("technical_zero", "unknown_cause"):
        attempt.update(state="technical_failure", reward=0 if defect == "technical_zero" else None,
                       failure_category="provider_quota" if defect == "technical_zero" else "not_allowlisted")
    elif defect == "agent_success":
        attempt.update(state="agent_failure", reward=1, failure_category="episode_timeout")
    elif defect == "confirmation_phase":
        document["phase"] = "confirmation"
    elif defect == "protocol":
        document["protocol_hash"] = "0" * 64
    elif defect == "extra_field":
        document["posthoc_budget"] = 0.06
    elif defect == "lab_count":
        document["lab_ids"].append("extra")
    elif defect == "global_budget":
        document["global_budget_fraction"] = 0.1
    else:
        attempt["costs"] = {"unknown": 1} if defect == "unknown_cost" else {"input_tokens": 1.5} if defect == "fractional_tokens" else {
            "model_wall_seconds": -1 if defect == "negative_cost" else float("nan") if defect == "nan_cost" else True}
    with pytest.raises(ValueError):
        calibrate.select_budget(document)


def test_unknown_costs_are_not_silently_zero_and_do_not_change_the_choice():
    document = fixture()
    first = calibrate.select_budget(document)
    assert first["upfront_cost"]["fields"]["model_wall_seconds"]["total"] is None
    for slot in document["slots"]:
        slot["attempts"][0]["costs"] = {field: 0 for field in calibrate.COST_FIELDS}
    second = calibrate.select_budget(document)
    assert second["chosen_budget_fraction"] == first["chosen_budget_fraction"]
    assert all(field["total"] == 0 and field["unreported_attempts"] == 0 for field in second["upfront_cost"]["fields"].values())


def test_cli_publishes_exclusively_and_binds_exact_input_bytes(tmp_path, capsys):
    source, output = tmp_path / "pilots.json", tmp_path / "selection.json"
    source.write_text(json.dumps(fixture()))
    main(["calibrate", str(source), "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["binding"]["input_file_sha256"] == file_hash(source)
    assert result["chosen_budget_fraction"] == 0.12 and result["model_sessions_started_by_selector"] == 0
    assert '"status": "selected"' in capsys.readouterr().out
    before = output.read_bytes()
    with pytest.raises(SystemExit, match="calibration failed"):
        main(["calibrate", str(source), "--output", str(output)])
    assert output.read_bytes() == before
    with pytest.raises(ValueError, match="under artifacts"):
        calibrate.select_file(source, ROOT / "results" / "must-not-exist.json")


def test_public_engineering_fixture_stays_explicitly_synthetic():
    document = json.loads((ROOT / "benchmarks/release_v1/calibration/engineering_pilots.json").read_text())
    result = calibrate.select_budget(document)
    assert result["evidence_kind"] == "synthetic_control" and result["chosen_budget_fraction"] == 0.12
    assert result["upfront_cost"]["episode_starts"] == 12
    assert result["upfront_cost"]["fields"]["input_tokens"]["total"] is None
