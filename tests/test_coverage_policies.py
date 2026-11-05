import copy
import json
from collections import defaultdict

import numpy as np
import pytest

from benchtop.agents.baselines import RELEASE_BASELINES, play_release
from benchtop.agents.estimator import ESTIMATOR_ID
from benchtop.analysis import shared_estimate_from_pairs
from benchtop.cache import initialize_cache, seal_result
from benchtop.env import Env
from benchtop.episode import Episode
from benchtop.space import Space, config_key
from benchtop.tasks import build_task_set

POLICIES = ("question_coverage", "coverage_plus_cost")


def fixture(tmp_path, *, costs=(1.0, 1.0, 1.0), width=False):
    """Analytical synthetic observations and controlled prices, not measured paper results."""
    factors = [{"name": "method", "role": "treatment", "levels": ["base", "new"]},
               {"name": "context", "role": "context", "levels": [0, 1, 2]}]
    if width:
        factors.append({"name": "width", "role": "component", "levels": [0, 1]})
    raw = {"title": "Synthetic coverage control", "metric": {"name": "value", "goal": "max", "min_effect": 0.05},
           "factors": factors, "treatment": {"factor": "method", "paper": "new", "baseline": "base"}}
    source = ("def run(config, seed):\n"
              "    effect = [-0.3, 0.0, 0.6][config['context']] if config['method'] == 'new' else 0.0\n"
              "    return {'value': 0.5 + 0.1 * config.get('width', 0) + effect + 0.001 * seed}\n")
    directory = tmp_path / "coverage"
    directory.mkdir()
    manifest = {"id": "coverage", "seeds": 8, "visible_seeds": 4}
    (directory / "lab.py").write_text(source)
    (directory / "space.json").write_text(json.dumps(raw))
    (directory / "manifest.json").write_text(json.dumps(manifest))
    binding = initialize_cache(directory, manifest)
    space = Space.from_dict(raw)
    rows = []
    for config in space.configs():
        effect = [-0.3, 0.0, 0.6][config["context"]] if config["method"] == "new" else 0.0
        for seed in range(8):
            value = 0.5 + 0.1 * config.get("width", 0) + effect + 0.001 * seed
            rows.append(seal_result(binding, space, config, seed,
                                    {"ok": True, "metrics": {"value": value}, "secs": costs[config["context"]]}))
    (directory / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    env = Env.load(directory)
    (directory / "tasks.json").write_text(json.dumps(build_task_set(env)))
    return Env.load(directory)


@pytest.mark.parametrize("name", POLICIES)
@pytest.mark.parametrize("budget", [0, 0.06, 0.12, 0.25, 1.0])
def test_coverage_policies_respect_budget_and_keep_complete_pairs(tmp_path, name, budget):
    ep = Episode(fixture(tmp_path), "claim", seed=7, budget_frac=budget)
    record = play_release(ep, name, seed=17)
    assert record["spent"] <= record["budget"]
    assert record["calls"] <= min(4, ep.max_calls)
    assert record["pricing_schedule_id"] == record["charging_id"]
    assert record["estimator_id"] == record["estimate"]["estimator_id"] == ESTIMATOR_ID
    paired = defaultdict(set)
    for row in record["observations"]:
        paired[(row["config"]["context"], row["replicate"])].add(row["config"]["method"])
    assert all(methods == {"base", "new"} for methods in paired.values())
    assert record["estimate"]["contrasts"]["overall"]["unmatched_observations"] == 0
    if budget == 0:
        assert record["calls"] == 0 and record["stop_reason"] == "no_affordable_pair"
    if budget == 1.0:
        assert record["score"] == 1 and len(record["observations"]) == 24


@pytest.mark.parametrize("name", POLICIES)
def test_where_levels_are_covered_before_repeating_a_level(tmp_path, name):
    ep = Episode(fixture(tmp_path, width=True), "where:context", seed=2, budget_frac=0.25)
    record = play_release(ep, name, seed=0)
    first = record["allocations"][0]["planned_pairs"][:3]
    assert {row["context"]["context"] for row in first} == {0, 1, 2}
    assert record["estimate"]["contrasts"].keys() == {"0", "1", "2"}


@pytest.mark.parametrize("name", POLICIES)
def test_component_design_spreads_marginal_levels_and_reconstructs_the_same_answer(tmp_path, name):
    ep = Episode(fixture(tmp_path, width=True), "component:width", seed=2, budget_frac=1.0)
    record = play_release(ep, name, seed=4)
    first = record["allocations"][0]["planned_pairs"]
    assert first[0]["context"]["method"] != first[1]["context"]["method"]
    assert {row["context"]["context"] for row in first[:3]} == {0, 1, 2}
    assert record["answer"] == "helps" and record["score"] == 1.0
    assert shared_estimate_from_pairs(ep.env, "component:width", record["ran"])["answer"] == record["answer"]


def test_cost_preference_buys_more_pairs_without_changing_target_weights(tmp_path):
    env = fixture(tmp_path, costs=(1.0, 1.0, 3.0))
    records = {name: play_release(Episode(env, "claim", seed=1, budget_frac=0.2), name, seed=0) for name in POLICIES}
    coverage, cost = records["question_coverage"], records["coverage_plus_cost"]
    assert coverage["allocations"][0]["planned_pairs"][0]["context"]["context"] == 2
    assert cost["allocations"][0]["planned_pairs"][0]["context"]["context"] != 2
    assert coverage["spent"] == cost["spent"] == 8.0
    assert len(coverage["observations"]) == 4 and len(cost["observations"]) == 8
    contrast = cost["estimate"]["contrasts"]["overall"]
    assert contrast["contexts_total"] == 3 and contrast["contexts_observed"] == 2
    assert next(row for row in contrast["contexts"] if row["context"]["context"] == 2)["contribution"] == 0
    # More cheap samples need not produce a better answer: the expensive positive context is missing.
    assert cost["score"] == 0 and coverage["score"] == 1


def test_equal_prices_isolate_coverage_from_cost_preference(tmp_path):
    env = fixture(tmp_path, width=True)
    records = [play_release(Episode(env, "claim", seed=7, budget_frac=0.5), name, seed=19) for name in POLICIES]
    for field in ("allocations", "observations", "estimate", "spent", "calls"):
        assert records[0][field] == records[1][field]


@pytest.mark.parametrize("name", POLICIES)
@pytest.mark.parametrize("calls", [0, 1])
def test_call_caps_are_obeyed_before_and_between_batches(tmp_path, name, calls):
    record = play_release(Episode(fixture(tmp_path), "claim", budget_frac=1.0, max_calls=calls), name, seed=0)
    assert record["calls"] == calls and record["stop_reason"] == "call_limit"


@pytest.mark.parametrize("name", POLICIES)
def test_decimal_quotes_follow_controller_tolerance_without_half_pairs(tmp_path, name):
    ep = Episode(fixture(tmp_path, costs=(0.1, 0.2, 0.3)), "claim", seed=0, budget_frac=0.125)
    record = play_release(ep, name, seed=1)
    assert len(record["observations"]) == 4
    assert record["spent"] == pytest.approx(0.6)
    assert record["spent"] <= record["budget"]
    assert record["estimate"]["contrasts"]["overall"]["unmatched_observations"] == 0


@pytest.mark.parametrize("name", POLICIES)
def test_failures_stop_and_preserve_unmatched_delivered_work(tmp_path, name):
    ep = Episode(fixture(tmp_path), "claim", seed=0, budget_frac=1.0)
    def fail_one(configs, seeds):
        request = ep.begin_run(configs, seeds)
        _, key, seed, _ = request["steps"][1]
        return ep.finish_run(request, {(key, seed): {"code": "fixture_failure", "retryable": True}})

    result = RELEASE_BASELINES[name]({**ep.describe(), "budget": ep.budget_left()}, fail_one, np.random.default_rng(0))
    assert result["policy_calls"] == 1 and result["stop_reason"] == "experiment_error"
    assert result["estimate"]["contrasts"]["overall"]["unmatched_observations"] == 1
    assert ep.spent == 5.0


@pytest.mark.parametrize("name", POLICIES)
@pytest.mark.parametrize("fault", ["schedule", "missing", "zero", "negative", "nan"])
def test_incomplete_or_invalid_public_prices_fail_before_acquisition(tmp_path, name, fault):
    ep = Episode(fixture(tmp_path), "claim", seed=0)
    description = {**ep.describe(), "budget": ep.budget_left()}
    pricing = description["pricing"]
    key = config_key(ep.env.space.configs()[0])
    if fault == "schedule":
        pricing.pop("schedule_id")
    elif fault == "missing":
        pricing["by_config"].pop(key)
    else:
        pricing["by_config"][key] = {"zero": 0.0, "negative": -1.0, "nan": float("nan")}[fault]
    with pytest.raises(ValueError, match="pric"):
        RELEASE_BASELINES[name](description, ep.run, np.random.default_rng(0))
    assert ep.calls == 0


@pytest.mark.parametrize("name", POLICIES)
def test_design_ignores_hidden_outcomes_and_private_timings_after_price_freeze(tmp_path, name):
    env = fixture(tmp_path, width=True)
    def execute():
        ep = Episode(env, "claim", seed=3, budget_frac=0.5)
        result = RELEASE_BASELINES[name]({**ep.describe(), "budget": ep.budget_left()}, ep.run, np.random.default_rng(5))
        return result["allocations"], result["observations"]

    before = execute()
    for cell in env.table.cells.values():
        for seed in env.heldout_seeds:
            cell[seed] = {"value": -1e9}
    for timings in env.table.secs.values():
        for seed in timings:
            timings[seed] = 1e9
    assert execute() == before


@pytest.mark.parametrize("name", POLICIES)
def test_acquisition_uses_coverage_not_visible_effect_magnitudes(tmp_path, name):
    env = fixture(tmp_path, width=True)
    def execute():
        ep = Episode(env, "claim", seed=3, budget_frac=0.5)
        result = RELEASE_BASELINES[name]({**ep.describe(), "budget": ep.budget_left()}, ep.run, np.random.default_rng(5))
        return copy.deepcopy(result["allocations"])

    before = execute()
    for cell in env.table.cells.values():
        for seed in env.visible_seeds:
            cell[seed]["value"] *= -100
    assert execute() == before
