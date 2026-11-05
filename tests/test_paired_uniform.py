import copy

import numpy as np
import pytest

from benchtop.agents.baselines import paired_uniform_public, play_release, uniform_design
from benchtop.analysis import runs_from_pairs, shared_estimate_from_pairs
from benchtop.env import ROOT, Env
from benchtop.episode import Episode
from benchtop.evaluate import corrected_replay
from tests.test_core import toy_env


def ran(ep):
    return [pair for entry in ep.log if entry["action"] == "run" for pair in entry["ran"]]


@pytest.mark.parametrize("budget", [0, 0.06, 0.12, 0.25, 1.0])
def test_paired_uniform_reproduces_original_acquisition_with_corrected_tasks(tmp_path, budget):
    env = toy_env(tmp_path)
    legacy = Episode(env, "claim", seed=7, budget_frac=budget)
    current = Episode(env, "claim", seed=7, budget_frac=budget)
    uniform_design(legacy, np.random.default_rng(17))
    result = play_release(current, seed=17)
    assert ran(legacy) == ran(current) == result["ran"]
    assert legacy.spent == current.spent <= current.budget
    assert result["policy_seed"] == 17 and result["episode_seed"] == 7
    assert result["episode"] == current.id
    assert result["grader_id"] == "seed_block_t_v1"
    assert result["charging_id"] == current.charging_id
    if budget:
        assert result["observations"] and all("replicate" in row for row in result["observations"])
    else:
        assert not result["observations"]
    if budget == 1.0:
        assert result["score"] == 1.0
        assert result["estimate"]["contrasts"]["overall"]["coverage"] == 1.0


def test_policy_choices_do_not_depend_on_hidden_outcomes(tmp_path):
    env = toy_env(tmp_path)
    def run_policy():
        ep = Episode(env, "claim", seed=2, budget_frac=0.25)
        description = {**ep.describe(), "budget": ep.budget_left()}
        return paired_uniform_public(description, ep.run, np.random.default_rng(4))

    before = run_policy()
    for cell in env.table.cells.values():
        for seed in env.heldout_seeds:
            cell[seed] = {"acc": -1e9}
    after = run_policy()
    assert before == after


def test_partial_failure_is_not_a_completed_pair_and_does_not_charge_failed_work(tmp_path):
    ep = Episode(toy_env(tmp_path), "claim", seed=1, budget_frac=1.0)
    def fail_one(configs, seeds):
        request = ep.begin_run(configs, seeds)
        _, key, seed, _ = request["steps"][1]
        return ep.finish_run(request, {(key, seed): {"code": "fixture_failure", "retryable": True}})

    result = paired_uniform_public({**ep.describe(), "budget": ep.budget_left()}, fail_one, np.random.default_rng(0))
    assert result["stop_reason"] == "experiment_error"
    assert result["policy_calls"] == 1
    assert result["estimate"]["contrasts"]["overall"]["unmatched_observations"] == 1
    assert ep.spent == 11.0


def test_trace_reconstruction_preserves_seed_identity_and_public_precision(tmp_path):
    ep = Episode(toy_env(tmp_path), "claim", seed=3, budget_frac=1.0)
    result = play_release(ep, seed=3)
    trace = shared_estimate_from_pairs(ep.env, "claim", result["ran"])
    assert trace["answer"] == result["answer"]
    assert trace["contrasts"]["overall"]["delta"] == pytest.approx(result["estimate"]["contrasts"]["overall"]["delta"])
    assert trace["contrasts"]["overall"]["uncertainty"]["se"] == pytest.approx(result["estimate"]["contrasts"]["overall"]["uncertainty"]["se"])
    assert all("seed" in row for row in runs_from_pairs(ep.env, result["ran"]))
    hidden = copy.deepcopy(result["ran"][:1])
    hidden[0][1] = ep.env.heldout_seeds[0]
    with pytest.raises(ValueError, match="visible seed"):
        shared_estimate_from_pairs(ep.env, "claim", hidden)


def test_historical_tasks_are_not_silently_reinterpreted():
    env = Env.load(ROOT / "envs/2610-36935-gaussian-process-bandits-for", replay=True)
    ep = Episode(env, "claim", seed=0)
    with pytest.raises(ValueError, match="corrected task artifacts"):
        play_release(ep)
    assert ep.calls == 0


def test_corrected_replay_is_separate_read_only_and_keeps_historical_pricing():
    original = Env.load(ROOT / "envs/2610-36935-gaussian-process-bandits-for", replay=True)
    old_prices, old_charging = original.charging_schedule()
    config, held = original.space.configs()[0], original.heldout_seeds[0]
    original_value = original.table.get(config, held)[original.space.metric]
    original.table.get(config, held)[original.space.metric] = 1e9
    view = corrected_replay(original)
    assert original.grader_id == "historical_v0"
    assert view.grader_id == "seed_block_t_v1"
    assert view.table is not original.table
    assert view.table.get(config, held)[view.space.metric] == original_value
    assert view.execution_id == original.execution_id and view.task_version != original.task_version
    assert view.charging_schedule()[1] != old_charging
    assert original.charging_schedule() == (old_prices, old_charging)
    with pytest.raises(ValueError, match="read-only"):
        view.prepare_live()
    record = play_release(Episode(view, "claim", seed=0, budget_frac=0.06), seed=0)
    assert record["grader_id"] == "seed_block_t_v1"
    assert 0 <= record["score"] <= 1
    assert record["spent"] <= record["budget"]


def test_release_adapter_refuses_partly_used_episodes(tmp_path):
    ep = Episode(toy_env(tmp_path), "claim", seed=0, budget_frac=1.0)
    ep.run([ep.env.space.configs()[0]])
    with pytest.raises(ValueError, match="fresh episode"):
        play_release(ep)
    assert ep.calls == 1 and not ep.done


def test_structured_description_contains_only_previously_public_problem_information(tmp_path):
    ep = Episode(toy_env(tmp_path), "claim", seed=0)
    description = ep.describe()
    assert description["question_spec"] == {"kind": "claim"}
    assert description["metric"]["min_effect"] == ep.env.space.min_effect
    assert description["replicate_limit"] == 4
    assert "answer" not in description["question_spec"]
    assert "seed_ids" not in description and "order" not in description
