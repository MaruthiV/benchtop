import copy
from datetime import UTC, datetime

from benchtop import cohort, pilot_calibration, pilots, primary, study
from benchtop import session_ledger as ledger
from benchtop.candidate_metadata import timestamp

DATA_DIRECTORY = "confirmation_data"


def _inputs(ledger_context):
    locked = cohort.verify(ledger_context=ledger_context)
    collection = pilot_calibration.load_closed(ledger_context=ledger_context) if locked["binding"]["members"] else None
    if collection is not None and (collection["cohort_lock_sha256"] != locked["record_sha256"]
            or collection["global_budget_fraction"] != locked["binding"]["global_budget_fraction"]
            or [lab["env_id"] for lab in collection["labs"]] != [member["env_id"] for member in locked["binding"]["members"]]):
        raise ValueError("closed pilot collection differs from the frozen cohort")
    return locked, collection


def _grid(binding, digest):
    spec = study._json(study.SPEC)
    if len(binding["environments"]) > spec["fresh_cohort"]["target_labs"]:
        raise ValueError("confirmation cohort exceeds the protocol ceiling")
    groups, slots = [], []
    for lab_index, (env, decision) in enumerate(zip(binding["environments"], binding["pilot_decisions"], strict=True)):
        if env["env_id"] != decision["env_id"] or decision["chosen_budget_fraction"] not in spec["budget"]["fractions"]:
            raise ValueError("confirmation decision does not match its ordered cohort member")
        budgets = {"global_budget": binding["global_budget_fraction"], "calibrated_budget": decision["chosen_budget_fraction"]}
        arms = {arm: study._group(binding, digest, env, budgets[arm], arm) for arm in spec["confirmation"]["arms"]}
        groups.extend(arms.values())
        order = spec["confirmation"]["arms"][::1 if lab_index % 2 == 0 else -1]
        for index in range(spec["confirmation"]["repeats_per_arm"]):
            slots.extend({**study._slot(arms[arm], index), "arm": arm} for arm in order)
    if len({group["group_id"] for group in groups}) != len(groups) or len({slot["slot_id"] for slot in slots}) != len(slots):
        raise ValueError("confirmation group or slot identities collide")
    return groups, slots


def make_plan(*, ledger_context=None):
    locked, collection = _inputs(ledger_context)
    binding, spec = locked["binding"], study._json(study.SPEC)
    environments = copy.deepcopy(binding["members"])
    for env in environments:
        pilots.load_environment(env)
        env["pilot_visible_order"] = env["visible_order"]
        env["visible_order"] = env["confirmation_visible_order"]
    decisions = [{"env_id": lab["env_id"], **{key: lab["selection"][key] for key in (
        "status", "chosen_budget_fraction", "fallback_reason", "selection_sha256", "pilot_slot_ids", "pilot_attempt_ids")}}
        for lab in collection["labs"]] if collection else []
    core = {"planner_id": study.PLANNER_ID, "mode": "study_preparation", "phase": "confirmation",
            "protocol_hash": binding["protocol_hash"], "protocol_file_sha256": study.cache.file_hash(study.SPEC),
            "source_sha256": study._sources(), "runtime": binding["runtime"], "ledger_id": binding["ledger_id"],
            "environments": environments, "model_configuration": binding["model_configuration"], "prompt": binding["prompt"],
            "limits": spec["execution_limits"], "cohort_lock_sha256": locked["record_sha256"], "cohort_size": len(environments),
            "global_budget_fraction": binding["global_budget_fraction"], "pilot_decisions": decisions,
            "pilot_snapshot_sha256": collection["snapshot_sha256"] if collection else None,
            "pilot_snapshot_file_sha256": study.cache.file_hash(primary.canonical_root() / pilot_calibration.COLLECTION) if collection else None,
            "pilot_authorization_sha256": collection["pilot_authorization_sha256"] if collection else None,
            "pilot_evidence": collection["phase_evidence"] if collection else {},
            "prerequisites_recorded_at_utc": collection["recorded_at_utc"] if collection else locked["recorded_at_utc"],
            "confirmation": spec["confirmation"],
            "truth_contract": {"seed_ids": spec["seed_partitions"]["confirmation"], "task_identity": "frozen_qualification_question_and_schema",
                               "stability_rule": spec["grading"]["confirmation_stability"],
                               "execution_requires": "Separate post-lock confirmation data, truth assessment and provider authorization."},
            "scope": "Frozen confirmation arms and question identities only. Qualification snapshots stay immutable; confirmation truth and execution require separate post-lock evidence."}
    digest = study.canonical_hash(core)
    groups, slots = _grid(core, digest)
    if len(slots) != binding["planned_episode_limits"]["confirmation"]:
        raise ValueError("confirmation slots differ from the frozen cohort's reduced episode ceiling")
    return study._seal({"schema_version": 1, "binding": core, "bindings_hash": digest, "groups": groups, "slots": slots,
                        "execution_ready": False, "model_sessions_started": 0}, "plan_sha256")


def _before_data(root, state):
    if (root / DATA_DIRECTORY).exists() or any(history for key, history in state["histories"].items()
            if state["plans"][key[0]]["binding"]["phase"] == "confirmation"):
        raise ValueError("confirmation data or model observations appeared before the confirmation lock")


def lock_confirmation(plan):
    study.validate_plan(plan)
    if plan["binding"]["phase"] != "confirmation":
        raise ValueError("confirmation lock requires the exact confirmation preparation")
    root = primary.canonical_root()
    with primary._guard(root.parent / ".initialization.lock"):
        if (root / "confirmation_lock.json").exists():
            raise FileExistsError("confirmation is already locked; preserve its original arm decisions")
        with ledger._locked(root / "ledger") as context:
            directory, manifest = context
            ledger._current(manifest)
            study.validate_plan(plan, ledger_context=context)
            state = ledger._read(directory, manifest)
            if manifest["identity"]["scope"] != "primary" or state["halted"] or state["counts"]["active"]:
                raise ValueError("confirmation locking requires idle canonical accounting without a model/tool halt")
            _before_data(root, state)
            record = study._seal({"schema_version": 1, "id": "confirmation_lock", "study_lock": True,
                "binding": plan["binding"], "bindings_hash": plan["bindings_hash"], "plan_sha256": plan["plan_sha256"],
                "protocol_hash": plan["binding"]["protocol_hash"], "canonical_root": str(root.resolve()), "ledger_id": manifest["ledger_id"],
                "recorded_at_utc": datetime.now(UTC).isoformat(), "planned_slots": len(plan["slots"]),
                "status": "locked_awaiting_confirmation_data" if plan["slots"] else "no_primary_comparison",
                "execution_ready": False, "model_sessions_started": 0}, "record_sha256")
            study._publish(root / "confirmation_plan.json", plan)
            study._publish(root / "confirmation_lock.json", record)
            primary._append_binding(record)
            return record


def verify(*, ledger_context=None):
    root = primary.canonical_root()
    record = study._json(root / "confirmation_lock.json")
    study._verify(record, "record_sha256")
    plan = make_plan(ledger_context=ledger_context)
    expected = {"schema_version": 1, "id": "confirmation_lock", "study_lock": True, "binding": plan["binding"],
                "bindings_hash": plan["bindings_hash"], "plan_sha256": plan["plan_sha256"], "protocol_hash": plan["binding"]["protocol_hash"],
                "canonical_root": str(root.resolve()), "ledger_id": plan["binding"]["ledger_id"], "planned_slots": len(plan["slots"]),
                "status": "locked_awaiting_confirmation_data" if plan["slots"] else "no_primary_comparison",
                "execution_ready": False, "model_sessions_started": 0}
    if (any(record.get(key) != value for key, value in expected.items()) or study._json(root / "confirmation_plan.json") != plan
            or timestamp(record["recorded_at_utc"]) < timestamp(plan["binding"]["prerequisites_recorded_at_utc"])):
        raise ValueError("confirmation lock, plan or prerequisite bindings changed")
    public = study._json(primary.BINDINGS)
    row = {key: record[key] for key in ("id", "plan_sha256", "bindings_hash", "record_sha256", "recorded_at_utc")}
    if (public.get("schema_version") != 1 or public.get("protocol_hash") != record["protocol_hash"]
            or [entry for entry in public.get("records", []) if entry.get("id") == "confirmation_lock"] != [row]):
        raise ValueError("confirmation lock is absent from public binding history")
    return record
