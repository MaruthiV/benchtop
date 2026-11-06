import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

from scipy.stats import beta

from benchtop.cache import file_hash, runtime_identity
from benchtop.env import ROOT

SELECTOR_ID = "beta_mixed_group_v1"
SPEC = ROOT / "benchmarks/release_v1/spec.json"
COST_FIELDS = ("requested_runs", "unique_executed_runs", "charged_experiment_cost",
               "actual_environment_cpu_seconds", "environment_wall_seconds", "model_wall_seconds",
               "input_tokens", "output_tokens")
COUNT_FIELDS = {"requested_runs", "unique_executed_runs", "input_tokens", "output_tokens"}


def canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def posterior(correct: int, completed: int, *, group_size: int = 8, confidence: float = 0.95):
    if type(correct) is not int or type(completed) is not int or not 0 <= correct <= completed:
        raise ValueError("correct/completed must be integer counts with 0 <= correct <= completed")
    if type(group_size) is not int or group_size < 2:
        raise ValueError("group_size must be an integer of at least two")
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be finite and between zero and one")
    a, b = correct + 1, completed - correct + 1
    all_correct = math.prod((a + k) / (a + b + k) for k in range(group_size))
    all_wrong = math.prod((b + k) / (a + b + k) for k in range(group_size))
    tail = (1 - confidence) / 2
    return {"correct": correct, "completed": completed, "alpha": a, "beta": b,
            "success_probability_mean": a / (a + b),
            "success_probability_interval": [float(beta.ppf(tail, a, b)), float(beta.ppf(1 - tail, a, b))],
            "mixed_group_probability": max(0.0, min(1.0, 1 - all_correct - all_wrong))}


def _keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or set(value) - set(required) - set(optional):
        raise ValueError(f"expected fields {sorted(required)} with optional {sorted(optional)}")


def _identifier(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("identities must be lowercase SHA-256 strings")


def _number(value, name, *, integer=False):
    try:
        valid = (type(value) in (int, float) and math.isfinite(value) and value >= 0
                 and (not integer or type(value) is int))
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite nonnegative {'integer' if integer else 'number'}")


def _attempts(slot, spec):
    attempts = slot["attempts"]
    if not isinstance(attempts, list) or len(attempts) > 1 + spec["failure_handling"]["max_technical_retries_per_slot"]:
        raise ValueError("a slot permits an initial attempt and at most one technical retry")
    for index, attempt in enumerate(attempts):
        _keys(attempt, {"attempt_id", "attempt_index", "state", "reward"}, {"failure_category", "costs"})
        expected_id = canonical_hash({"slot_id": slot["slot_id"], "attempt_index": index})
        if type(attempt["attempt_index"]) is not int or attempt["attempt_index"] != index or attempt["attempt_id"] != expected_id:
            raise ValueError("attempt identity/index does not match its slot")
        state, reward = attempt["state"], attempt["reward"]
        if index and attempts[index - 1]["state"] != "technical_failure":
            raise ValueError("completed, agent-failure, and unresolved attempts cannot be retried")
        category = attempt.get("failure_category")
        if state in ("completed", "agent_failure"):
            if type(reward) not in (int, float) or reward not in (0, 1):
                raise ValueError("completed rewards must be binary numbers")
            if state == "completed" and category is not None:
                raise ValueError("completed outcomes must not carry a failure category")
            if state == "agent_failure" and (reward != 0 or category not in spec["failure_handling"]["agent_outcomes_reward_zero"]):
                raise ValueError("agent failures require an allowlisted cause and zero reward")
        elif state in ("technical_failure", "unresolved"):
            if reward is not None:
                raise ValueError("technical/unresolved outcomes require null reward")
            if state == "technical_failure" and category not in spec["failure_handling"]["infrastructure_failures_missing"]:
                raise ValueError("technical failures require an allowlisted cause")
            if state == "unresolved" and (not isinstance(category, str) or not category.strip()):
                raise ValueError("unresolved outcomes require an explicit cause")
        else:
            raise ValueError("unknown normalized outcome state")
        costs = attempt.get("costs", {})
        if not isinstance(costs, dict) or set(costs) - set(COST_FIELDS):
            raise ValueError("unknown cost field or invalid cost object")
        for name, value in costs.items():
            if value is not None:
                _number(value, name, integer=name in COUNT_FIELDS)
    return attempts


def _validate(document, spec):
    _keys(document, {"schema_version", "protocol_hash", "evidence_kind", "phase", "closed", "lab_ids", "slots"}, {"global_budget_fraction"})
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("unsupported calibration input schema")
    if document["protocol_hash"] != canonical_hash(spec):
        raise ValueError("calibration input is bound to another protocol")
    if document["evidence_kind"] not in ("synthetic_control", "operator_supplied"):
        raise ValueError("evidence_kind must identify synthetic_control or operator_supplied outcomes")
    phase = document["phase"]
    if phase not in ("development", "fresh_pilot") or type(document["closed"]) is not bool:
        raise ValueError("expected development/fresh_pilot phase and explicit boolean closed state")
    labs = document["lab_ids"]
    count = spec["development"]["claim_labs"] if phase == "development" else 1
    if (not isinstance(labs, list) or len(labs) != count
            or any(not isinstance(lab, str) or not lab.strip() for lab in labs) or len(set(labs)) != count):
        raise ValueError(f"{phase} requires exactly {count} distinct lab IDs")
    budgets = spec["budget"]["fractions"]
    if phase == "fresh_pilot":
        if (type(document.get("global_budget_fraction")) not in (int, float)
                or document["global_budget_fraction"] not in budgets):
            raise ValueError("fresh pilots require a supplied global budget from the protocol candidates")
    elif "global_budget_fraction" in document:
        raise ValueError("development selection cannot take a prechosen global budget")
    repeats = spec["calibration"]["pilot_repeats_per_budget"]
    if repeats != spec["development"]["budget_selection_repeats"]:
        raise ValueError("development/fresh pilot repeat rules differ; update the selector explicitly")
    if spec["calibration"]["prior"] != {"kind": "beta", "alpha": 1, "beta": 1}:
        raise ValueError("selector requires the protocol Beta(1,1) prior")
    slots = document["slots"]
    if not isinstance(slots, list) or len(slots) != count * len(budgets) * repeats:
        raise ValueError("include the full planned slot grid, retaining missing attempts as empty lists")
    seen, groups = {}, {}
    for slot in slots:
        _keys(slot, {"group_id", "slot_id", "lab_id", "budget_fraction", "replicate_index", "attempts"})
        _identifier(slot["group_id"])
        index, budget = slot["replicate_index"], slot["budget_fraction"]
        if (slot["lab_id"] not in labs or type(budget) not in (int, float) or budget not in budgets
                or type(index) is not int or not 0 <= index < repeats):
            raise ValueError("slot lies outside the planned lab/budget/replicate grid")
        key = slot["lab_id"], budget, index
        if key in seen:
            raise ValueError("duplicate planned slot")
        cell = key[:2]
        if cell in groups and groups[cell] != slot["group_id"]:
            raise ValueError("one lab/budget cell must share one group ID")
        groups[cell] = slot["group_id"]
        if slot["slot_id"] != canonical_hash({"group_id": slot["group_id"], "replicate_index": index}):
            raise ValueError("slot identity does not match its group and replicate")
        _attempts(slot, spec)
        seen[key] = slot
    if len(set(groups.values())) != count * len(budgets):
        raise ValueError("groups cannot be reused across labs or budgets")
    return [seen[(lab, budget, index)] for lab in labs for budget in budgets for index in range(repeats)]


def _costs(slots):
    attempts = [attempt for slot in slots for attempt in slot["attempts"]]
    fields = {}
    for name in COST_FIELDS:
        reported = [attempt.get("costs", {}).get(name) for attempt in attempts]
        values = [value for value in reported if value is not None]
        try:
            known = sum(values) if name in COUNT_FIELDS else math.fsum(values)
        except OverflowError as error:
            raise ValueError("cost aggregation overflow") from error
        _number(known, "aggregate " + name, integer=name in COUNT_FIELDS)
        unknown = len(attempts) - len(values)
        fields[name] = {"known_sum": known, "unreported_attempts": unknown,
                        "total": None if unknown else known,
                        "availability": "partial_or_unreported" if unknown else "reported"}
    return {"episode_starts": len(attempts), "retry_episode_starts": sum(len(slot["attempts"]) == 2 for slot in slots),
            "fields": fields, "source": "operator_supplied_attempt_costs"}


def _group(slots, settings):
    final = [slot["attempts"][-1] if slot["attempts"] else None for slot in slots]
    usable = [attempt for attempt in final if attempt is not None and attempt["state"] in ("completed", "agent_failure")]
    summary = posterior(sum(int(attempt["reward"]) for attempt in usable), len(usable),
                        group_size=settings["target_group_size"], confidence=settings["posterior_interval"]["confidence"])
    return {"group_id": slots[0]["group_id"], "budget_fraction": slots[0]["budget_fraction"], **summary,
            "planned": len(slots), "missing": len(slots) - len(usable),
            "observed_mixed": bool(0 < summary["correct"] < len(usable)),
            "status_counts": dict(Counter(attempt["state"] if attempt else "unattempted" for attempt in final)),
            "pilot_slot_ids": [slot["slot_id"] for slot in slots],
            "pilot_attempt_ids": [attempt["attempt_id"] for slot in slots for attempt in slot["attempts"]]}


def _choose(objectives, tolerance):
    best = max(row["objective"] for row in objectives)
    cutoff = best - tolerance
    eligible = [row["budget_fraction"] for row in objectives if row["objective"] >= cutoff
                or math.isclose(row["objective"], cutoff, rel_tol=0, abs_tol=1e-14)]
    return min(eligible)


def select_budget(document, *, input_file_sha256=None):
    spec = json.loads(SPEC.read_text())
    slots = _validate(document, spec)
    if input_file_sha256 is not None:
        _identifier(input_file_sha256)
    settings, budgets = spec["calibration"], spec["budget"]["fractions"]
    labs = [{"lab_id": lab, "budgets": [_group([slot for slot in slots if slot["lab_id"] == lab
                                               and slot["budget_fraction"] == budget], settings)
                                        for budget in budgets]} for lab in document["lab_ids"]]
    objectives = []
    for index, budget in enumerate(budgets):
        groups = [lab["budgets"][index] for lab in labs]
        complete = all(group["missing"] == 0 for group in groups)
        objectives.append({"budget_fraction": budget, "complete_labs": sum(group["missing"] == 0 for group in groups),
                           "objective": math.fsum(group["mixed_group_probability"] for group in groups) / len(groups)
                           if complete else None})
    incomplete = any(row["objective"] is None for row in objectives)
    diagnostics = ["no_observed_mixed_budget:" + lab["lab_id"] for lab in labs
                   if not any(group["observed_mixed"] for group in lab["budgets"]) and not incomplete]
    status, chosen, fallback = "pending", None, None
    if document["closed"]:
        if incomplete:
            status = "fallback" if document["phase"] == "fresh_pilot" else "incomplete_development"
            fallback = "incomplete_pilots" if document["phase"] == "fresh_pilot" else "incomplete_development"
        elif document["phase"] == "fresh_pilot" and diagnostics:
            status, fallback = "fallback", "no_observed_mixed_budget"
        else:
            status, chosen = "selected", _choose(objectives, settings["near_best_tolerance"])
    if status == "fallback":
        chosen = document["global_budget_fraction"]
    result = {"schema_version": 1, "selector_id": SELECTOR_ID, "phase": document["phase"],
              "evidence_kind": document["evidence_kind"],
              "status": status, "closed": document["closed"], "chosen_budget_fraction": chosen,
              "fallback_reason": fallback, "diagnostics": diagnostics, "labs": labs, "objectives": objectives,
              "global_budget_fraction": document.get("global_budget_fraction"), "upfront_cost": _costs(slots),
              "pilot_slot_ids": [slot["slot_id"] for slot in slots],
              "pilot_attempt_ids": [attempt["attempt_id"] for slot in slots for attempt in slot["attempts"]],
              "parameters": {"budget_fractions": budgets, "pilots_per_budget": settings["pilot_repeats_per_budget"],
                             "target_group_size": settings["target_group_size"], "prior": settings["prior"],
                             "interval": settings["posterior_interval"], "near_best_tolerance": settings["near_best_tolerance"]},
              "binding": {"protocol_hash": canonical_hash(spec), "protocol_file_sha256": file_hash(SPEC),
                          "selector_sha256": file_hash(Path(__file__)), "input_sha256": canonical_hash(document),
                          "input_file_sha256": input_file_sha256, "runtime": json.loads(json.dumps(runtime_identity()))},
              "model_sessions_started_by_selector": 0,
              "scope": "Offline selection from operator-supplied normalized outcomes. Identities and structure checked; phase locks, admission, group descriptors, outcome classifications, global control and measurements are not authenticated. No study lock or execution authorization."}
    return {**result, "selection_sha256": canonical_hash(result)}


def select_file(source, output):
    raw = Path(source).read_bytes()
    result = select_budget(json.loads(raw), input_file_sha256=hashlib.sha256(raw).hexdigest())
    output = Path(output).resolve()
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / "artifacts"):
        raise ValueError("write calibration diagnostics under artifacts/ or outside the repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, prefix=".calibration-", delete=False) as out:
        temporary = Path(out.name)
        try:
            out.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
            os.link(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    return result
