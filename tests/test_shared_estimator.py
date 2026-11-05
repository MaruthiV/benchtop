import copy

import pytest

from benchtop.agents.estimator import estimate_answer
from benchtop.space import Space

RAW = {
    "metric": {"name": "value", "goal": "max", "min_effect": 0.1},
    "factors": [{"name": "method", "role": "treatment", "levels": ["base", "new"]},
                {"name": "context", "role": "context", "levels": ["cheap", "costly"]}],
    "treatment": {"factor": "method", "paper": "new", "baseline": "base"},
}


def row(context, method, replicate, value):
    return {"config": {"context": context, "method": method}, "replicate": replicate, "metrics": {"value": value}}


def paired(context, replicate, effect):
    return [row(context, "new", replicate, effect), row(context, "base", replicate, 0.0)]


def estimate(rows, raw=RAW, question=None):
    return estimate_answer(Space.from_dict(raw), question or {"kind": "claim"}, rows)


def test_mismatched_replicates_cannot_manufacture_a_treatment_effect():
    rows = paired("cheap", 1, 0.4) + paired("costly", 1, 0.0)
    rows += [row("cheap", "new", 2, 1000), row("cheap", "base", 3, -1000)]
    contrast = estimate(rows)["contrasts"]["overall"]
    assert contrast["delta"] == pytest.approx(0.2)
    assert contrast["unmatched_observations"] == 2
    assert contrast["paired_observations"] == 2


def test_repeated_cheap_context_does_not_reweight_the_target():
    rows = sum((paired("cheap", r, 0.2) for r in range(1, 5)), []) + paired("costly", 1, -0.6)
    contrast = estimate(rows)["contrasts"]["overall"]
    assert contrast["delta"] == pytest.approx(-0.2)
    assert contrast["uncertainty"]["status"] == "unavailable"
    assert estimate(rows)["answer"] == "baseline_better"


def test_missing_context_uses_declared_neutral_prior_without_changing_denominator():
    result = estimate(paired("cheap", 1, 0.15))
    contrast = result["contrasts"]["overall"]
    assert contrast["delta"] == pytest.approx(0.075)
    assert contrast["contexts_total"] == 2 and contrast["contexts_observed"] == 1
    assert contrast["status"] == "partial_coverage"
    assert result["answer"] == "no_meaningful_difference"
    empty = estimate([])
    assert empty["contrasts"]["overall"]["status"] == "no_pairs_fallback"
    assert empty["contrasts"]["overall"]["uncertainty"]["status"] == "unavailable"


def test_shared_context_noise_is_one_block_not_independent_replication():
    rows = sum((paired(context, replicate, effect) for context in ("cheap", "costly")
                for replicate, effect in ((1, -1), (2, 1))), [])
    uncertainty = estimate(rows)["contrasts"]["overall"]["uncertainty"]
    assert uncertainty["n_blocks"] == 2
    assert uncertainty["se"] == pytest.approx(1.0)


def test_opposite_context_noise_cancels_within_identity():
    rows = paired("cheap", 1, -0.7) + paired("cheap", 2, 1.3)
    rows += paired("costly", 1, 1.3) + paired("costly", 2, -0.7)
    uncertainty = estimate(rows)["contrasts"]["overall"]["uncertainty"]
    assert uncertainty["delta"] == pytest.approx(0.3)
    assert uncertainty["se"] == 0


def test_complete_contexts_with_different_identities_have_no_block_interval():
    rows = paired("cheap", 1, 0.3) + paired("cheap", 2, 0.3)
    rows += paired("costly", 2, 0.3) + paired("costly", 3, 0.3)
    contrast = estimate(rows)["contrasts"]["overall"]
    assert contrast["status"] == "complete_coverage"
    assert contrast["uncertainty"]["status"] == "unavailable"


def test_duplicates_are_idempotent_and_conflicting_metrics_are_rejected():
    rows = paired("cheap", 1, 0.3)
    duplicate = estimate(rows + [copy.deepcopy(rows[0])])
    assert duplicate["duplicates_ignored"] == 1
    assert duplicate["contrasts"] == estimate(rows)["contrasts"]
    with pytest.raises(ValueError, match="conflicting"):
        estimate(rows + [row("cheap", "new", 1, 100)])


@pytest.mark.parametrize("fault", ["missing_identity", "nan", "partial_config", "bad_identity", "hidden_identity"])
def test_invalid_observations_do_not_silently_become_neutral_answers(fault):
    observation = row("cheap", "new", 1, 0.3)
    if fault == "missing_identity":
        observation.pop("replicate")
    elif fault == "nan":
        observation["metrics"]["value"] = float("nan")
    elif fault == "partial_config":
        observation["config"].pop("context")
    elif fault == "bad_identity":
        observation["replicate"] = True
    else:
        observation["replicate"] = 5
    with pytest.raises(ValueError):
        estimate_answer(Space.from_dict(RAW), {"kind": "claim"}, [observation], allowed_ids=range(1, 5))


def test_metric_direction_and_where_levels_are_explicit():
    raw = copy.deepcopy(RAW)
    raw["metric"]["goal"] = "min"
    rows = paired("cheap", 1, -0.4) + paired("costly", 1, 0.4)
    result = estimate(rows, raw, {"kind": "where", "factor": "context"})
    assert result["answer"] == {"cheap": "paper_better", "costly": "baseline_better"}
    assert set(estimate([], question={"kind": "where", "factor": "context"})["answer"]) == {"cheap", "costly"}


def test_component_comparison_and_boolean_where_keys():
    result = estimate(paired("cheap", 1, 0) + paired("costly", 1, 1), question={"kind": "component", "factor": "context"})
    assert result["answer"] == "helps"
    raw = copy.deepcopy(RAW)
    raw["factors"][1]["levels"] = [False, True]
    assert estimate([], raw, {"kind": "where", "factor": "context"})["answer"] == {
        "false": "no_meaningful_difference", "true": "no_meaningful_difference"}
