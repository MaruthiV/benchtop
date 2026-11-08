import copy
import time
from datetime import UTC, datetime

from benchtop import calibrate, confirmation_execution, confirmation_results, pilots, primary, study
from benchtop import session_ledger as ledger
from benchtop.agents.baselines import RELEASE_BASELINES, play_release
from benchtop.agents.estimator import ESTIMATOR_ID
from benchtop.cache import CacheError
from benchtop.candidate_metadata import timestamp
from benchtop.episode import Episode
from benchtop.tasks import grade

REPLAY_ID = "confirmation_classical_replay_v1"
DIRECTORY = "confirmation_replay"


def _now():
    return datetime.now(UTC).isoformat()


def _plan():
    directory = primary.canonical_root() / "ledger"
    with ledger._locked(directory) as context:
        collection = confirmation_results.load_closed(ledger_context=context)
        plan = study._json(directory.parent / "confirmation_plan.json")
        baseline = study._json(directory.parent / "cohort_lock.json")["binding"]["baselines"]
        spec = study._json(study.SPEC)
        if (baseline["policies"] != spec["baselines"]["required"] + spec["baselines"]["sanity_controls"]
                or not set(baseline["policies"]) <= RELEASE_BASELINES.keys() or baseline["estimator_id"] != ESTIMATOR_ID):
            raise ValueError("confirmation replay requires the original frozen release policies and shared estimator")
        contexts = {member["env_id"]: confirmation_execution.evidence_context(plan, member["env_id"], collection["confirmation_authorization_sha256"])
                    for member in collection["members"] if member["gradable"]}
        core = {"replay_id": REPLAY_ID, "collection_sha256": collection["snapshot_sha256"], "plan_sha256": plan["plan_sha256"],
            "protocol_hash": collection["protocol_hash"], "source_sha256": study._sources(), "runtime": plan["binding"]["runtime"],
            "baseline_configuration": baseline, "budget_fraction": collection["global_budget_fraction"],
            "policy_seeds": list(range(spec["baselines"]["replay_seeds"]["first"], spec["baselines"]["replay_seeds"]["first"] + spec["baselines"]["replay_seeds"]["count"])),
            "members": collection["members"], "grading_contexts": contexts,
            "environments": copy.deepcopy(plan["binding"]["environments"]), "max_calls": plan["binding"]["limits"]["max_experiment_tool_calls"]}
        digest = study.canonical_hash(core)
        slots = [{"env_id": member["env_id"], "policy": policy, "policy_seed": seed,
                  "slot_id": study.canonical_hash({"binding": digest, "env_id": member["env_id"], "policy": policy, "policy_seed": seed})}
                 for member in collection["members"] if member["gradable"] for policy in baseline["policies"] for seed in core["policy_seeds"]]
        return study._seal({"binding": core, "slots": slots}, "replay_plan_sha256")


def _record(path, key):
    value = study._json(path)
    study._verify(value, key)
    return value


def _finish(root, plan, slot):
    start = _record(root / "attempts" / (slot["slot_id"] + ".start.json"), "record_sha256")
    expected = {"replay_plan_sha256": plan["replay_plan_sha256"], "slot": slot}
    if any(start.get(key) != value for key, value in expected.items()):
        raise ValueError("classical replay start differs from its frozen slot")
    path = root / "attempts" / (slot["slot_id"] + ".finish.json")
    if not path.exists():
        return None
    record = _record(path, "record_sha256")
    context = plan["binding"]["grading_contexts"][slot["env_id"]]
    if (record["start_sha256"] != start["record_sha256"] or record["grading_context_sha256"] != context["context_sha256"]
            or timestamp(record["finished_at_utc"]) < timestamp(start["recorded_at_utc"])):
        raise ValueError("classical replay grading or chronology changed")
    if record["state"] == "completed":
        if record["reward"] != grade(None, context["task"], record["answer"])["score"]:
            raise ValueError("classical reward differs from the independent confirmation claim")
    elif record["state"] not in ("technical_failure", "policy_failure") or record["reward"] != (None if record["state"] == "technical_failure" else 0):
        raise ValueError("invalid classical replay outcome")
    for name in ("charged_experiment_cost", "controller_wall_seconds", "controller_cpu_seconds", "requested_runs"):
        calibrate._number(record[name], name, integer=name == "requested_runs")
    return record


def _summary(root, plan):
    if root.exists() and (any(path.name not in ("plan.json", "attempts", "summary.json") for path in root.iterdir())
                          or any(path.is_symlink() for path in root.rglob("*"))):
        raise ValueError("unplanned or symbolic-link classical replay file")
    allowed = {slot["slot_id"] + suffix for slot in plan["slots"] for suffix in (".start.json", ".finish.json")}
    if (root / "attempts").exists() and any(path.name not in allowed or path.is_symlink() for path in (root / "attempts").iterdir()):
        raise ValueError("unplanned or symbolic-link classical replay record")
    groups = []
    for member in plan["binding"]["members"]:
        for policy in plan["binding"]["baseline_configuration"]["policies"]:
            slots = [slot for slot in plan["slots"] if slot["env_id"] == member["env_id"] and slot["policy"] == policy]
            outcomes = []
            for slot in slots:
                start = root / "attempts" / (slot["slot_id"] + ".start.json")
                finish = root / "attempts" / (slot["slot_id"] + ".finish.json")
                if finish.exists() and not start.exists():
                    raise ValueError("classical completion has no start intent")
                record = _finish(root, plan, slot) if start.exists() else None
                outcomes.append({**slot, "state": record["state"] if record else "pending_reconciliation" if start.exists() else "unattempted",
                    "reward": record["reward"] if record else None,
                    "costs": {key: record[key] for key in ("charged_experiment_cost", "requested_runs", "controller_wall_seconds", "controller_cpu_seconds")} if record else None})
            complete = len(slots) == len(plan["binding"]["policy_seeds"]) and all(row["reward"] is not None for row in outcomes)
            groups.append({"env_id": member["env_id"], "policy": policy, "gradable": member["gradable"],
                "planned": len(plan["binding"]["policy_seeds"]), "outcomes": outcomes,
                "accuracy": sum(row["reward"] for row in outcomes) / len(slots) if complete else None})
    files = {str(path.relative_to(root)): study.cache.file_hash(path) for path in sorted(root.rglob("*")) if path.is_file() and path.name != "summary.json"}
    pending = sum(row["state"] in ("unattempted", "pending_reconciliation") for group in groups for row in group["outcomes"])
    costs = {name: confirmation_results._reported([row["costs"][name] if row["costs"] else None for group in groups for row in group["outcomes"]])
             for name in ("charged_experiment_cost", "requested_runs", "controller_wall_seconds", "controller_cpu_seconds")}
    return {"replay_id": REPLAY_ID, "replay_plan_sha256": plan["replay_plan_sha256"], "collection_sha256": plan["binding"]["collection_sha256"],
        "protocol_hash": plan["binding"]["protocol_hash"], "status": "complete" if not pending else "pending", "groups": groups, "files": files,
        "policy_seeds": plan["binding"]["policy_seeds"], "budget_fraction": plan["binding"]["budget_fraction"], "model_sessions_started": 0,
        "actual_environment_executions": 0, "actual_environment_cpu_seconds": 0, "costs": costs,
        "interpretation": "Cached policy replays reuse the same visible data; 200 policy seeds are not independent labs. Controller CPU/wall time is measured separately."}


def inspect():
    plan = _plan()
    root = primary.canonical_root() / DIRECTORY
    if not plan["slots"]:
        return study._seal(_summary(root, plan), "summary_sha256")
    if not root.exists():
        return {"status": "not_run", "model_sessions_started": 0}
    if root.is_symlink() or study._json(root / "plan.json") != plan:
        raise ValueError("classical replay plan or private directory changed")
    core = _summary(root, plan)
    if (root / "summary.json").exists():
        saved = _record(root / "summary.json", "summary_sha256")
        if {key: value for key, value in saved.items() if key not in ("summary_sha256", "recorded_at_utc")} != core:
            raise ValueError("saved classical summary differs from retained replay evidence")
        return saved
    return study._seal(core, "summary_sha256")


def _run_one(root, plan, slot, env):
    start = study._seal({"replay_plan_sha256": plan["replay_plan_sha256"], "slot": slot, "recorded_at_utc": _now()}, "record_sha256")
    study._publish(root / "attempts" / (slot["slot_id"] + ".start.json"), start)
    context = plan["binding"]["grading_contexts"][env.id]
    wall, cpu = time.perf_counter(), time.process_time()
    ep = Episode(env, "claim", budget_frac=plan["binding"]["budget_fraction"], seed=0, visible_order=next(
        member["visible_order"] for member in plan["binding"]["environments"] if member["env_id"] == env.id),
        max_calls=plan["binding"]["max_calls"], grading_context=context)
    trace, outcome, error = [], None, None
    try:
        outcome = play_release(ep, slot["policy"], slot["policy_seed"], majority_label=plan["binding"]["baseline_configuration"]["development_majority_label"], tool_trace=trace)
        env.assert_current()
    except Exception as exception:
        error = type(exception).__name__
        if isinstance(exception, CacheError):
            raise
    technical = any("exception" in entry or entry.get("response", {}).get("errors") for entry in trace)
    technical |= outcome is not None and outcome["stop_reason"] == "partial_reply"
    state = "technical_failure" if technical else "completed" if ep.done else "policy_failure"
    record = study._seal({"start_sha256": start["record_sha256"], "grading_context_sha256": context["context_sha256"],
        "state": state, "reward": None if technical else ep.result["score"] if ep.done else 0,
        "answer": ep.result["graded"].get("given") if ep.done else None, "error_type": error,
        "charged_experiment_cost": ep.spent, "requested_runs": sum(len(row["configs"]) * min(row["seeds"], len(ep.order)) for row in trace),
        "tool_trace": trace, "controller_wall_seconds": time.perf_counter() - wall, "controller_cpu_seconds": time.process_time() - cpu,
        "finished_at_utc": _now()}, "record_sha256")
    study._publish(root / "attempts" / (slot["slot_id"] + ".finish.json"), record)


def run():
    plan = _plan()
    if not plan["slots"]:
        return inspect()
    root = primary.canonical_root() / DIRECTORY
    with primary._guard(root.parent / ".confirmation-replay.lock"):
        existing = inspect()
        if existing["status"] == "complete":
            if not (root / "summary.json").exists():
                existing = study._seal({**existing, "recorded_at_utc": _now()}, "summary_sha256")
                study._publish(root / "summary.json", existing)
            return existing
        if root.exists() and not (root / "plan.json").exists():
            raise ValueError("unrecorded classical replay directory; preserve it for investigation")
        if not root.exists():
            root.mkdir(mode=0o700)
            study._publish(root / "plan.json", plan)
            (root / "attempts").mkdir(mode=0o700)
        for member in plan["binding"]["environments"]:
            if member["env_id"] not in plan["binding"]["grading_contexts"]:
                continue
            env = pilots.load_environment(member)
            for slot in (row for row in plan["slots"] if row["env_id"] == env.id):
                if (root / "attempts" / (slot["slot_id"] + ".start.json")).exists():
                    if _finish(root, plan, slot) is None:
                        raise ValueError("interrupted classical replay requires reconciliation; do not rerun it automatically")
                    continue
                _run_one(root, plan, slot, env)
            if _plan() != plan:
                raise ValueError("classical replay inputs changed during execution")
        result = study._seal({**_summary(root, plan), "recorded_at_utc": _now()}, "summary_sha256")
        study._publish(root / "summary.json", result)
        return result
