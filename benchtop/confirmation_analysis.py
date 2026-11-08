import math
from collections import Counter

import numpy as np

from benchtop import calibrate, confirmation_replay, confirmation_results, study
from benchtop.stats import LABELS
from benchtop.tasks import norm_label

ANALYSIS_ID = "paired_confirmation_analysis_v1"


def _paired(values, settings):
    count = len(values)
    result = {"labs": count, "estimate": math.fsum(values) / count if count else None, "interval": None,
              "interval_reason": "no_complete_pairs" if not count else "fewer_than_two_complete_pairs" if count == 1 else None}
    if count >= 2:
        rng = np.random.default_rng(settings["rng_seed"])
        draws = np.asarray(values)[rng.integers(0, count, size=(settings["resamples"], count))].mean(axis=1)
        tail = (1 - settings["confidence"]) / 2
        result["interval"] = np.quantile(draws, [tail, 1 - tail]).tolist()
    return result


def _validate(collection, spec):
    study._verify(collection, "snapshot_sha256")
    if collection["exporter_id"] != confirmation_results.EXPORTER_ID or collection["protocol_hash"] != study.canonical_hash(spec):
        raise ValueError("confirmation analysis requires the frozen protocol and collection format")
    members, groups, slots = collection["members"], collection["groups"], collection["slots"]
    ids = [member["env_id"] for member in members]
    if len(set(ids)) != len(ids) or len(ids) > spec["fresh_cohort"]["target_labs"]:
        raise ValueError("confirmation membership differs from the full locked cohort")
    expected = []
    for member in members:
        if member["stability"] not in ("stable", "unstable", "ambiguous", "incomplete") or member["gradable"] != (member["stability"] == "stable"):
            raise ValueError("confirmation stability and gradability disagree")
        for arm in spec["confirmation"]["arms"]:
            matching = [group for group in groups if group["env_id"] == member["env_id"] and group["descriptor"]["arm"] == arm]
            if len(matching) != 1:
                raise ValueError("every locked member needs both distinct arm groups")
            group = matching[0]
            descriptor = group["descriptor"]
            if (set(descriptor) != set(spec["identities"]["group_descriptor_fields"]) or study.canonical_hash(descriptor) != group["group_id"]
                    or descriptor["phase"] != "confirmation" or descriptor["bindings_hash"] != collection["bindings_hash"]
                    or descriptor["protocol_hash"] != collection["protocol_hash"]):
                raise ValueError("confirmation group identity changed")
            for index in range(spec["confirmation"]["repeats_per_arm"]):
                expected.append({**study._slot(group, index), "arm": arm})
    if (len(groups) != len(ids) * 2 or len(slots) != len(expected) or len({slot["slot_id"] for slot in slots}) != len(slots)
            or {slot["slot_id"] for slot in slots} != {slot["slot_id"] for slot in expected}):
        raise ValueError("confirmation analysis must retain every original planned slot exactly once")
    by_id = {slot["slot_id"]: slot for slot in slots}
    for slot in expected:
        actual = by_id[slot["slot_id"]]
        if any(actual.get(key) != value for key, value in slot.items()):
            raise ValueError("confirmation slot or budget changed")
        calibrate._attempts(actual, spec)
    decisions = collection["pilot_decisions"]
    if [row["env_id"] for row in decisions] != ids:
        raise ValueError("confirmation decisions must retain cohort order")
    for group in groups:
        budget = collection["global_budget_fraction"] if group["descriptor"]["arm"] == "global_budget" else next(
            row["chosen_budget_fraction"] for row in decisions if row["env_id"] == group["env_id"])
        if group["descriptor"]["budget_fraction"] != budget or budget not in spec["budget"]["fractions"]:
            raise ValueError("confirmation arm budget differs from its frozen choice")


def _arm(slots, *, gradable, closed):
    rows, rewards, counts = [], [], Counter()
    for slot in sorted(slots, key=lambda row: row["replicate_index"]):
        final = slot["attempts"][-1] if slot["attempts"] else None
        state = final["state"] if final else "unattempted"
        reward = final["reward"] if final and gradable and state in ("completed", "agent_failure") else None
        if not gradable and slot["attempts"]:
            raise ValueError("unavailable confirmation tasks cannot contain model attempts")
        if reward is not None:
            rewards.append(reward)
        counts[state] += 1
        detail = slot["details"][-1] if slot.get("details") else {}
        answer = detail.get("answer")
        if isinstance(answer, dict):
            answer = answer.get("answer")
        label = norm_label(answer) if isinstance(answer, str) else None
        confidence = detail.get("confidence")
        rows.append({key: slot[key] for key in ("slot_id", "group_id", "replicate_index", "budget_fraction")} | {
            "state": state, "reward": reward, "answer": label if label in LABELS else None,
            "confidence": confidence if type(confidence) in (int, float) and math.isfinite(confidence) else None,
            "failure_category": final.get("failure_category") if final else None,
            "attempts": slot["attempts"], "observed_trace": [{key: event[key] for key in ("index", "tool", "error", "response_keys")}
                                                            for event in detail.get("observed_trace", [])]})
    complete = gradable and len(rewards) == len(slots)
    mean = math.fsum(rewards) / len(rewards) if rewards else None
    mixed = int(0 < math.fsum(rewards) < len(rewards)) if complete and closed else None
    return {"slots": rows, "planned": len(slots), "observed": len(rewards), "complete": complete,
        "status_counts": dict(counts), "mixed": mixed, "accuracy_observed": mean,
        "reward_variance_observed": math.fsum((value - mean) ** 2 for value in rewards) / len(rewards) if rewards else None,
        "accuracy_bounds": [math.fsum(rewards) / len(slots), (math.fsum(rewards) + len(slots) - len(rewards)) / len(slots)] if gradable else None,
        "mixed_without_agent_failures": mixed if counts["agent_failure"] == 0 else None,
        "costs": calibrate._costs(slots)}


def _classical(collection, labs, replay, spec):
    settings = spec["analysis"]["intervals"]
    policies = spec["baselines"]["required"] + spec["baselines"]["sanity_controls"]
    expected_seeds = list(range(spec["baselines"]["replay_seeds"]["first"], spec["baselines"]["replay_seeds"]["first"] + spec["baselines"]["replay_seeds"]["count"]))
    if replay is not None:
        study._verify(replay, "summary_sha256")
        if (replay["replay_id"] != confirmation_replay.REPLAY_ID or replay["collection_sha256"] != collection["snapshot_sha256"]
                or replay["protocol_hash"] != collection["protocol_hash"] or replay["budget_fraction"] != collection["global_budget_fraction"]
                or replay["policy_seeds"] != expected_seeds):
            raise ValueError("classical replay is not matched to this collection, budget or 200 seeds")
        if sorted((group["env_id"], group["policy"]) for group in replay["groups"]) != sorted((lab["env_id"], policy) for lab in labs for policy in policies):
            raise ValueError("classical replay must retain every lab/policy group")
        for group in replay["groups"]:
            lab = next(lab for lab in labs if lab["env_id"] == group["env_id"])
            if group["gradable"] != lab["gradable"] or group["planned"] != len(expected_seeds):
                raise ValueError("classical membership or planned count changed")
            if [row["policy_seed"] for row in group["outcomes"]] != (expected_seeds if lab["gradable"] else []):
                raise ValueError("classical replay cannot drop or duplicate policy seeds")
            for row in group["outcomes"]:
                if (row["env_id"] != group["env_id"] or row["policy"] != group["policy"]
                        or row["state"] not in ("completed", "policy_failure", "technical_failure", "pending_reconciliation", "unattempted")
                        or row["state"] == "policy_failure" and row["reward"] != 0
                        or row["state"] not in ("completed", "policy_failure") and row["reward"] is not None):
                    raise ValueError("classical failure state or group identity changed")
            usable = [row["reward"] for row in group["outcomes"] if row["state"] in ("completed", "policy_failure")]
            if (any(type(value) not in (int, float) or value not in (0, 1) for value in usable)
                    or group["accuracy"] != (sum(usable) / len(expected_seeds) if len(usable) == len(expected_seeds) else None)):
                raise ValueError("classical accuracy differs from all 200 planned outcomes")
    comparisons = []
    for policy in policies:
        pairs = []
        for lab in labs:
            group = next((group for group in replay["groups"] if group["env_id"] == lab["env_id"] and group["policy"] == policy), None) if replay else None
            if collection["closed"] and lab["gradable"] and lab["arms"]["global_budget"]["complete"] and group and group["accuracy"] is not None:
                pairs.append({"env_id": lab["env_id"], "global_accuracy": lab["arms"]["global_budget"]["accuracy_observed"],
                              "classical_accuracy": group["accuracy"], "difference": lab["arms"]["global_budget"]["accuracy_observed"] - group["accuracy"]})
        comparisons.append({"policy": policy, **_paired([row["difference"] for row in pairs], settings), "per_lab": pairs,
                            "unavailable_labs": len(labs) - len(pairs)})
    return {"status": replay["status"] if replay else "not_run", "summary_sha256": replay["summary_sha256"] if replay else None,
            "comparisons": comparisons, "groups": replay["groups"] if replay else [],
            "interpretation": "Global-arm accuracy minus matched classical accuracy; average equally across available labs. Policy seeds are not independent labs."}


def analyse(collection, replay=None):
    spec = study._json(study.SPEC)
    _validate(collection, spec)
    labs = []
    for member in collection["members"]:
        decision = next(row for row in collection["pilot_decisions"] if row["env_id"] == member["env_id"])
        arms = {arm: _arm([row for row in collection["slots"] if row["env_id"] == member["env_id"] and row["arm"] == arm],
                         gradable=member["gradable"], closed=collection["closed"]) for arm in spec["confirmation"]["arms"]}
        labs.append({"env_id": member["env_id"], "stability": member["stability"], "gradable": member["gradable"],
            "selected_budget": decision["chosen_budget_fraction"], "fallback_reason": decision["fallback_reason"], "arms": arms,
            "confirmation_cpu_work": member["confirmation_cpu_work"]})
    pairs = [lab for lab in labs if all(arm["mixed"] is not None for arm in lab["arms"].values())]
    diffs = [lab["arms"]["calibrated_budget"]["mixed"] - lab["arms"]["global_budget"]["mixed"] for lab in pairs]
    bounds = []
    for lab in labs:
        global_arm, calibrated = (lab["arms"][arm]["mixed"] for arm in spec["confirmation"]["arms"])
        bounds.append([(calibrated if calibrated is not None else 0) - (global_arm if global_arm is not None else 1),
                       (calibrated if calibrated is not None else 1) - (global_arm if global_arm is not None else 0)])
    classical = _classical(collection, labs, replay, spec)
    stability = Counter(lab["stability"] for lab in labs)
    settled = stability["stable"] + stability["unstable"]
    attempts = [attempt for slot in collection["slots"] for attempt in slot["attempts"]]
    return study._seal({"schema_version": 1, "analysis_id": ANALYSIS_ID, "protocol_hash": collection["protocol_hash"],
        "collection_sha256": collection["snapshot_sha256"], "status": "analysed" if collection["closed"] else "pending_collection_closure",
        "reporting_status": "complete" if collection["closed"] and classical["status"] == "complete" else "pending",
        "denominators": {"locked_labs": len(labs), "stable_labs": sum(lab["gradable"] for lab in labs),
                         "complete_paired_labs": len(pairs), "unavailable_comparisons": len(labs) - len(pairs)},
        "primary": {**_paired(diffs, spec["analysis"]["intervals"]), "per_lab": [{"env_id": lab["env_id"], "difference": value} for lab, value in zip(pairs, diffs, strict=True)],
                    "full_cohort_bounds": [math.fsum(row[index] for row in bounds) / len(labs) for index in (0, 1)] if labs else None,
                    "interval_method": spec["analysis"]["intervals"], "interpretation": spec["analysis"]["primary"]},
        "labs": labs, "classical": classical, "candidate_funnel": collection["candidate_funnel"],
        "stability_counts": dict(stability), "fallback_counts": dict(Counter(lab["fallback_reason"] or "no_fallback" for lab in labs)),
        "label_agreement": {"settled_labs": settled, "agreeing_labs": stability["stable"],
                            "fraction_among_settled": stability["stable"] / settled if settled else None},
        "failure_counts": {"final_slot_states": dict(Counter(slot["attempts"][-1]["state"] if slot["attempts"] else "unattempted" for slot in collection["slots"])),
                           "attempt_states": dict(Counter(attempt["state"] for attempt in attempts)),
                           "attempt_failure_categories": dict(Counter(attempt["failure_category"] for attempt in attempts if attempt.get("failure_category")))},
        "costs": {**{key: collection[key] for key in ("pilot_costs", "confirmation_costs", "generation_and_repair_usage")},
                  "classical_replay": replay["costs"] if replay else None, "shared_evaluation_snapshot": collection["accounting"]},
        "efficiency": {"mean_selected_budget_fraction": math.fsum(lab["selected_budget"] for lab in labs) / len(labs) if labs else None,
                       "global_budget_fraction": collection["global_budget_fraction"], "break_even_groups": None,
                       "availability": "No break-even assumption supplied; calibration overhead is reported separately. Mixed rewards alone do not establish an efficiency benefit."},
        "halted": collection["halted"], "model_sessions_started_by_analysis": 0,
        "limitations": [spec["analysis"]["claims"], "Complete-case effects are conditional on stable labels and complete groups. Full-cohort bounds include every unavailable arm.",
                        "Bootstrap resamples whole paired labs, not slots or classical policy seeds. Secondary comparisons are descriptive.",
                        "Reported costs are adapter measurements, not authenticated subscription billing. Unknown usage remains null."]}, "analysis_sha256")


def export(output, *, report=None):
    destination = _destination(output)
    report_path = _destination(report) if report is not None else None
    if report_path == destination:
        raise ValueError("JSON and Markdown need separate destinations")
    collection = confirmation_results.load_closed()
    replay = confirmation_replay.inspect()
    result = analyse(collection, replay if replay["status"] != "not_run" else None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    study._publish(destination, result)
    if report_path is not None:
        from benchtop.candidate_metadata import _bytes

        report_path.parent.mkdir(parents=True, exist_ok=True)
        _bytes(report_path, render_markdown(result).encode())
    return result


def _destination(output):
    from benchtop import primary
    from benchtop.session_ledger import _writable

    path = _writable(output)
    if path.is_relative_to(primary.canonical_root().resolve()):
        raise ValueError("write analysis exports outside the canonical study directory")
    if path.exists():
        raise FileExistsError(path)
    return path


def render_markdown(result):
    study._verify(result, "analysis_sha256")

    def shown(value):
        if value is None:
            return "unavailable"
        if isinstance(value, list):
            return "[" + ", ".join(shown(item) for item in value) + "]"
        return f"{value:.4g}" if isinstance(value, float) else str(value).replace("|", "\\|").replace("\n", " ")

    denominators, primary = result["denominators"], result["primary"]
    lines = ["# Benchtop confirmation analysis", "", f"Protocol: `{result['protocol_hash']}`. Analysis: `{result['analysis_sha256']}`.", "",
        f"Status: {result['status']}; reporting: {result['reporting_status']}.", "",
        f"Locked labs: {denominators['locked_labs']}; stable: {denominators['stable_labs']}; complete pairs: {denominators['complete_paired_labs']}; unavailable comparisons: {denominators['unavailable_comparisons']}.", "",
        "Primary difference in mixed-group frequency, calibrated minus global:", "",
        f"Estimate: {shown(primary['estimate'])}. Whole-lab bootstrap interval: {shown(primary['interval'])}. Full-cohort worst-case bounds: {shown(primary['full_cohort_bounds'])}.", "",
        "| Lab | Stability | Selected budget | Global mixed | Calibrated mixed | Global observations | Calibrated observations | Fallback |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for lab in result["labs"]:
        global_arm, calibrated = (lab["arms"][arm] for arm in ("global_budget", "calibrated_budget"))
        values = [lab["env_id"], lab["stability"], lab["selected_budget"], global_arm["mixed"], calibrated["mixed"],
                  f"{global_arm['observed']}/{global_arm['planned']}", f"{calibrated['observed']}/{calibrated['planned']}", lab["fallback_reason"] or "none"]
        lines.append("| " + " | ".join(shown(value) for value in values) + " |")
    lines += ["", "Classical comparisons use the existing global model arm and 200 policy seeds at its frozen budget.", "",
              "| Classical policy | Complete labs | Model minus classical accuracy | Whole-lab interval |",
              "| --- | --- | --- | --- |"]
    lines.extend("| " + " | ".join(shown(row[key]) for key in ("policy", "labs", "estimate", "interval")) + " |" for row in result["classical"]["comparisons"])
    lines += ["", "| Reported cost | Known sum | Unreported attempts | Total |", "| --- | --- | --- | --- |"]
    for phase in ("pilot_costs", "confirmation_costs"):
        for name in ("charged_experiment_cost", "model_wall_seconds", "input_tokens", "output_tokens"):
            row = result["costs"][phase]["fields"][name]
            lines.append("| " + " | ".join(shown(value) for value in (phase + "/" + name, row["known_sum"], row["unreported_attempts"], row["total"])) + " |")
    lines += ["", "Raw slot outcomes, retries, failure states, stability and generation/CPU costs are retained in the JSON export.", ""]
    lines.extend("- " + limitation for limitation in result["limitations"])
    return "\n".join(lines) + "\n"
