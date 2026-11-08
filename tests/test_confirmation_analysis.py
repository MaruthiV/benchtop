import copy

import pytest

from benchtop import calibrate, confirmation_replay, confirmation_results, study
from benchtop import confirmation_analysis as analysis


def document(rows=(), *, closed=True, calibrated_budget=0.06):
    spec = study._json(study.SPEC)
    binding = {"phase": "confirmation", "protocol_hash": study.canonical_hash(spec), "model_configuration": {"fixture_only": True},
               "prompt": {"fixture_only": True}, "source_sha256": {"benchtop/tasks.py": "a" * 64, "benchtop/stats.py": "b" * 64}}
    digest = study.canonical_hash({"fixture_only": True})
    members, groups, slots, decisions = [], [], [], []
    for index, (global_rewards, calibrated_rewards, state) in enumerate(rows):
        env = {"env_id": f"fixture_{index}", "lab_hash": study.canonical_hash(index), "task_hash": "c" * 64,
               "charging_hash": "d" * 64, "visible_order": [0, 1, 3, 2]}
        members.append({"env_id": env["env_id"], "stability": state, "gradable": state == "stable", "confirmation_cpu_work": {"actual_environment_cpu_seconds": None}})
        decisions.append({"env_id": env["env_id"], "chosen_budget_fraction": calibrated_budget, "fallback_reason": None})
        for arm, rewards, budget in (("global_budget", global_rewards, 0.12), ("calibrated_budget", calibrated_rewards, calibrated_budget)):
            group = study._group(binding, digest, env, budget, arm)
            groups.append(group)
            for replicate, reward in enumerate(rewards):
                slot = {**study._slot(group, replicate), "arm": arm, "attempts": [], "details": []}
                if reward is not None:
                    slot["attempts"] = [{"attempt_id": slot["attempt_ids"][0], "attempt_index": 0, "state": "completed", "reward": reward}]
                    slot["details"] = [{"answer": "paper_better" if reward else "baseline_better", "confidence": 0.5, "observed_trace": []}]
                slots.append(slot)
    return study._seal({"schema_version": 1, "exporter_id": confirmation_results.EXPORTER_ID, "closed": closed,
        "protocol_hash": binding["protocol_hash"], "bindings_hash": digest, "groups": groups, "members": members, "slots": slots,
        "pilot_decisions": decisions, "global_budget_fraction": 0.12, "candidate_funnel": [],
        "pilot_costs": calibrate._costs([]), "confirmation_costs": calibrate._costs(slots), "generation_and_repair_usage": {},
        "accounting": {"fixture_only": True}, "halted": False, "scope": "Synthetic unit-test observations, not provider evidence."}, "snapshot_sha256")


def reseal(value):
    return study._seal(value, "snapshot_sha256")


def classical(collection, correct):
    spec = study._json(study.SPEC)
    groups = []
    for member, hits in zip(collection["members"], correct, strict=True):
        for policy in spec["baselines"]["required"] + spec["baselines"]["sanity_controls"]:
            outcomes = [{"env_id": member["env_id"], "policy": policy, "policy_seed": index,
                         "slot_id": study.canonical_hash((member["env_id"], policy, index)), "state": "completed", "reward": int(index < hits), "costs": {}}
                        for index in range(200)] if member["gradable"] else []
            groups.append({"env_id": member["env_id"], "policy": policy, "gradable": member["gradable"], "planned": 200,
                           "outcomes": outcomes, "accuracy": hits / 200 if member["gradable"] else None})
    return study._seal({"replay_id": confirmation_replay.REPLAY_ID, "collection_sha256": collection["snapshot_sha256"],
        "protocol_hash": collection["protocol_hash"], "budget_fraction": 0.12, "policy_seeds": list(range(200)),
        "groups": groups, "status": "complete", "costs": {"fixture_only": True}}, "summary_sha256")


def test_paired_whole_lab_effect_and_worst_case_bounds_keep_unstable_members():
    collection = document([([0] * 8, [0, 1] * 4, "stable"), ([0, 1] * 4, [1] * 8, "stable"), ([None] * 8, [None] * 8, "unstable")])
    report = analysis.analyse(collection)
    assert report["denominators"] == {"locked_labs": 3, "stable_labs": 2, "complete_paired_labs": 2, "unavailable_comparisons": 1}
    assert report["primary"]["estimate"] == 0 and report["primary"]["interval"] == [-1, 1]
    assert report["primary"]["full_cohort_bounds"] == [-1 / 3, 1 / 3]
    assert report["primary"]["interval_method"]["resamples"] == 10000
    assert report["primary"]["interval_method"]["rng_seed"] == 1729
    assert report["labs"][2]["arms"]["global_budget"]["mixed"] is None
    assert report["label_agreement"] == {"settled_labs": 3, "agreeing_labs": 2, "fraction_among_settled": 2 / 3}
    assert report == analysis.analyse(collection)


@pytest.mark.parametrize("state", ["unstable", "ambiguous", "incomplete"])
def test_no_gradable_pairs_report_no_effect_and_full_uncertainty(state):
    result = analysis.analyse(document([([None] * 8, [None] * 8, state)]))
    assert result["primary"]["estimate"] is result["primary"]["interval"] is None
    assert result["primary"]["interval_reason"] == "no_complete_pairs"
    assert result["primary"]["full_cohort_bounds"] == [-1, 1]
    assert result["labs"][0]["arms"]["global_budget"]["accuracy_bounds"] is None


def test_one_complete_lab_reports_raw_effect_without_between_lab_interval():
    report = analysis.analyse(document([([1] * 8, [0, 1] * 4, "stable")]))
    assert report["primary"]["estimate"] == 1 and report["primary"]["interval"] is None
    assert report["primary"]["interval_reason"] == "fewer_than_two_complete_pairs"
    assert report["primary"]["full_cohort_bounds"] == [1, 1]


@pytest.mark.parametrize(("global_rewards", "calibrated_rewards", "bounds"), [
    ([0, 1] * 4, [0, 1, None, None, None, None, None, None], [-1, 0]),
    ([0] * 8, [None] * 8, [0, 1]),
    ([None] * 8, [0, 1] * 4, [0, 1]),
    ([None] * 8, [1] * 8, [-1, 0]),
])
def test_unavailable_arms_do_not_shrink_full_cohort_bounds(global_rewards, calibrated_rewards, bounds):
    report = analysis.analyse(document([(global_rewards, calibrated_rewards, "stable")]))
    assert report["primary"]["estimate"] is None and report["primary"]["full_cohort_bounds"] == bounds


def test_open_collection_keeps_raw_observations_without_final_primary_effect():
    report = analysis.analyse(document([([0, 1] * 4, [1] * 8, "stable")], closed=False))
    assert report["status"] == "pending_collection_closure" and report["primary"]["estimate"] is None
    assert report["labs"][0]["arms"]["global_budget"]["observed"] == 8
    assert report["primary"]["full_cohort_bounds"] == [-1, 1]


def test_equal_budget_arms_remain_distinct_and_empty_cohort_is_unavailable():
    collection = document([([0, 1] * 4, [1] * 8, "stable")], calibrated_budget=0.12)
    assert len({group["group_id"] for group in collection["groups"]}) == 2
    assert analysis.analyse(collection)["primary"]["estimate"] == -1
    empty = analysis.analyse(document())
    assert empty["primary"]["estimate"] is empty["primary"]["full_cohort_bounds"] is None
    assert empty["denominators"]["locked_labs"] == 0


def test_agent_failure_zero_stays_usable_and_retry_missing_costs_remain_unknown():
    collection = document([([0, 1] * 4, [1] * 8, "stable")])
    slot = collection["slots"][0]
    slot["attempts"][0].update(state="technical_failure", reward=None, failure_category="provider_outage", costs={"model_wall_seconds": 3})
    slot["attempts"].append({"attempt_id": slot["attempt_ids"][1], "attempt_index": 1, "state": "agent_failure", "reward": 0,
                             "failure_category": "episode_timeout", "costs": {"model_wall_seconds": 2}})
    report = analysis.analyse(reseal(collection))
    arm = report["labs"][0]["arms"]["global_budget"]
    assert arm["mixed"] == 1 and arm["mixed_without_agent_failures"] is None
    assert arm["status_counts"]["agent_failure"] == 1 and arm["costs"]["episode_starts"] == 9
    assert arm["costs"]["retry_episode_starts"] == 1
    assert arm["costs"]["fields"]["model_wall_seconds"] == {"known_sum": 5.0, "unreported_attempts": 7, "total": None, "availability": "partial_or_unreported"}
    assert report["failure_counts"]["attempt_failure_categories"] == {"provider_outage": 1, "episode_timeout": 1}
    assert report["failure_counts"]["final_slot_states"] == {"agent_failure": 1, "completed": 15}


@pytest.mark.parametrize("defect", ["drop", "duplicate", "budget", "phase", "membership", "unavailable_attempt", "gradable"])
def test_resealed_malformed_collections_cannot_improve_the_denominator(defect):
    collection = document([([0, 1] * 4, [1] * 8, "stable")])
    if defect == "drop":
        collection["slots"].pop()
    elif defect == "duplicate":
        collection["slots"][-1] = copy.deepcopy(collection["slots"][0])
    elif defect == "budget":
        collection["slots"][0]["budget_fraction"] = 0.25
    elif defect == "phase":
        collection["groups"][0]["descriptor"]["phase"] = "fresh_pilot"
    elif defect == "membership":
        collection["members"].append(copy.deepcopy(collection["members"][0]))
    elif defect == "unavailable_attempt":
        collection["members"][0].update(stability="unstable", gradable=False)
    else:
        collection["members"][0]["gradable"] = False
    with pytest.raises(ValueError):
        analysis.analyse(reseal(collection))


def test_classical_comparison_uses_global_arm_and_equal_lab_weighting():
    collection = document([([1] + [0] * 7, [1] * 8, "stable"), ([1] * 8, [0] * 8, "stable")])
    replay = classical(collection, [200, 0])
    result = analysis.analyse(collection, replay)
    assert result["classical"]["comparisons"][0]["estimate"] == 0.0625
    assert result["classical"]["comparisons"][0]["labs"] == 2
    assert result["classical"]["comparisons"][0]["interval"] == [-0.875, 1]
    assert result["reporting_status"] == "complete"


@pytest.mark.parametrize("defect", ["seed", "budget", "collection", "mean", "failure", "drop_group"])
def test_mismatched_classical_replays_cannot_enter_primary_comparison(defect):
    collection = document([([0, 1] * 4, [1] * 8, "stable")])
    replay = classical(collection, [100])
    if defect == "seed":
        replay["groups"][0]["outcomes"].pop()
    elif defect == "budget":
        replay["budget_fraction"] = 0.25
    elif defect == "collection":
        replay["collection_sha256"] = "0" * 64
    elif defect == "mean":
        replay["groups"][0]["accuracy"] = 1
    elif defect == "failure":
        replay["groups"][0]["outcomes"][0].update(state="technical_failure", reward=0)
    else:
        replay["groups"].pop()
    with pytest.raises(ValueError):
        analysis.analyse(collection, study._seal(replay, "summary_sha256"))


def test_missing_classical_outcome_makes_only_that_policy_lab_comparison_unavailable():
    collection = document([([0, 1] * 4, [1] * 8, "stable")])
    replay = classical(collection, [100])
    replay["groups"][0]["outcomes"][0].update(state="technical_failure", reward=None)
    replay["groups"][0]["accuracy"] = None
    result = analysis.analyse(collection, study._seal(replay, "summary_sha256"))
    assert result["classical"]["comparisons"][0]["estimate"] is None
    assert result["classical"]["comparisons"][1]["estimate"] == 0
    assert result["primary"]["estimate"] == -1


def test_report_redacts_private_context_and_keeps_unavailable_values_explicit():
    collection = document([([None] * 8, [None] * 8, "stable")])
    collection["phase_evidence"] = {"/private/secret/capability": "do-not-publish"}
    collection["slots"][0]["details"] = [{"answer": "private secret", "confidence": None, "mcp_url": "private secret", "observed_trace": []}]
    result = analysis.analyse(reseal(collection))
    text = analysis.render_markdown(result)
    assert "unavailable" in text and "Whole-lab bootstrap" in text
    assert "private secret" not in str(result) and "do-not-publish" not in str(result)
    assert "break-even" not in text and result["efficiency"]["break_even_groups"] is None


@pytest.mark.parametrize(("values", "expected"), [
    ([1, None, 2], {"known_sum": 3, "unreported_items": 1, "total": None}),
    ([float("nan"), -1, 2], {"known_sum": 2, "unreported_items": 2, "total": None}),
    ([True, 2], {"known_sum": 2, "unreported_items": 1, "total": None}),
    ([], {"known_sum": 0, "unreported_items": 0, "total": 0}),
])
def test_unknown_or_invalid_costs_keep_known_partial_sums_without_invented_zeros(values, expected):
    assert confirmation_results._reported(values) == expected
