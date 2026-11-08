import copy
from datetime import UTC, datetime

from benchtop import confirmation, confirmation_data, pilots, primary, study
from benchtop import session_ledger as ledger
from benchtop.candidate_metadata import timestamp

AUTHORIZATION = "confirmation_authorization.json"


def _grading(plan, locked, assessment, report):
    if not report["gradable"]:
        return None
    member = next(env for env in plan["binding"]["environments"] if env["env_id"] == report["env_id"])
    task = next(task for task in report["task_artifact"]["tasks"] if task["id"] == "claim")
    if (report["status"] != "stable" or task["kind"] != "claim" or task["answer"] != report["confirmation_label"]
            or task["seed_ids"] != plan["binding"]["truth_contract"]["seed_ids"]
            or report["question_task_hash"] != member["task_hash"]):
        raise ValueError("confirmation grading differs from the stable independent claim")
    binding = {"role": "independent_confirmation", "env_id": report["env_id"],
        "confirmation_lock_sha256": locked["record_sha256"], "assessment_sha256": assessment["assessment_sha256"],
        "member_assessment_sha256": report["record_sha256"], "task_artifact_sha256": report["task_artifact"]["artifact_sha256"],
        "confirmation_task_sha256": task["task_sha256"], "question_task_hash": member["task_hash"], "seed_ids": task["seed_ids"]}
    return study._seal({"binding": binding, "task": copy.deepcopy(task)}, "context_sha256")


def _bindings(plan, *, live, ledger_context):
    study.validate_plan(plan, ledger_context=ledger_context)
    if plan["binding"]["phase"] != "confirmation":
        raise ValueError("confirmation execution requires the original locked confirmation plan")
    root = primary.canonical_root()
    manifest = ledger._manifest(root / "ledger")
    if ledger_context is not None and (ledger_context[0].resolve() != (root / "ledger").resolve() or ledger_context[1] != manifest):
        raise ValueError("confirmation execution requires the canonical journal context")
    locked = confirmation.verify(ledger_context=ledger_context)
    if study._json(root / "confirmation_plan.json") != plan:
        raise ValueError("confirmation execution plan differs from the arm lock")
    assessment = confirmation_data.load_assessment(ledger_context=ledger_context)
    authority = pilots._authority(plan, live=live)
    contexts = {report["env_id"]: _grading(plan, locked, assessment, report) for report in assessment["members"]}
    members = [{"env_id": report["env_id"], "status": report["status"], "gradable": report["gradable"],
        "member_assessment_sha256": report["record_sha256"],
        "grading_context_sha256": contexts[report["env_id"]]["context_sha256"] if report["gradable"] else None}
        for report in assessment["members"]]
    eligible = [slot["slot_id"] for slot in plan["slots"] if contexts[slot["env_id"]] is not None]
    expected = {"schema_version": 1, "id": "confirmation_authorization", "study_lock": False,
        "plan_sha256": plan["plan_sha256"], "bindings_hash": plan["bindings_hash"], "protocol_hash": locked["protocol_hash"],
        "confirmation_lock_sha256": locked["record_sha256"], "assessment_sha256": assessment["assessment_sha256"],
        "ledger_id": manifest["ledger_id"], "canonical_root": str(root.resolve()),
        "provider_sha256": authority["provider_sha256"], "smoke_authorization_sha256": authority["record_sha256"],
        "extra_usage_disabled": True, "billing_confirmation_source": "inherited_operator_attestation", "legacy_integration_starts": 0,
        "planned_slots": len(plan["slots"]), "full_cohort_size": plan["binding"]["cohort_size"],
        "gradable_labs": assessment["gradable_labs"], "members": members, "initial_slot_order": eligible,
        "eligible_slots": len(eligible), "status": "authorized" if eligible else "no_gradable_labs" if plan["slots"] else "no_primary_comparison",
        "model_sessions_started": 0}
    if manifest["identity"]["scope"] != "primary" or manifest["ledger_id"] != locked["ledger_id"]:
        raise ValueError("confirmation authorization differs from the canonical primary journal")
    earliest = max([timestamp(locked["recorded_at_utc"]), *[timestamp(report["finished_at_utc"]) for report in assessment["members"]]])
    return expected, contexts, earliest


def _verified(plan, *, live=True, ledger_context=None):
    path = primary.canonical_root() / AUTHORIZATION
    if not path.exists():
        raise ValueError("confirmation provider authorization is missing; the arm lock and truth assessment do not permit model starts")
    expected, contexts, earliest = _bindings(plan, live=live, ledger_context=ledger_context)
    record = study._json(path)
    study._verify(record, "record_sha256")
    if (set(record) != {*expected, "recorded_at_utc", "record_sha256"}
            or any(record.get(key) != value for key, value in expected.items()) or timestamp(record["recorded_at_utc"]) < earliest):
        raise ValueError("confirmation provider authorization, independent grading or initial slot order changed")
    return record, contexts


def verify(plan, *, live=True, ledger_context=None):
    return _verified(plan, live=live, ledger_context=ledger_context)[0]


def grading_context(plan, env_id, *, ledger_context=None):
    _, contexts = _verified(plan, live=False, ledger_context=ledger_context)
    if contexts.get(env_id) is None:
        raise ValueError("confirmation reward is unavailable for this frozen cohort member")
    return copy.deepcopy(contexts[env_id])


def evidence_context(plan, env_id, authorization_sha256):
    root = primary.canonical_root()
    if (root / confirmation.DATA_DIRECTORY).is_symlink():
        raise ValueError("confirmation data directory cannot be a symbolic link")
    record = study._json(root / AUTHORIZATION)
    locked = study._json(root / "confirmation_lock.json")
    assessment = study._json(root / confirmation.DATA_DIRECTORY / "assessment.json")
    for value, key in ((record, "record_sha256"), (locked, "record_sha256"), (assessment, "assessment_sha256")):
        study._verify(value, key)
    if (record["record_sha256"] != authorization_sha256 or record["plan_sha256"] != plan["plan_sha256"]
            or record["bindings_hash"] != plan["bindings_hash"] or record["canonical_root"] != str(root.resolve())
            or record["ledger_id"] != ledger._manifest(root / "ledger")["ledger_id"]
            or record["confirmation_lock_sha256"] != locked["record_sha256"] or locked["plan_sha256"] != plan["plan_sha256"]
            or record["assessment_sha256"] != assessment["assessment_sha256"]
            or assessment["confirmation_lock_sha256"] != locked["record_sha256"] or assessment["plan_sha256"] != plan["plan_sha256"]
            or study._json(root / "confirmation_plan.json") != plan):
        raise ValueError("saved confirmation evidence differs from its original authorization or independent assessment")
    confirmation_data._verify_files(root / confirmation.DATA_DIRECTORY, assessment["files"])
    report = next(report for report in assessment["members"] if report["env_id"] == env_id)
    context = _grading(plan, locked, assessment, report)
    member = next(member for member in record["members"] if member["env_id"] == env_id)
    if (context is None or member["grading_context_sha256"] != context["context_sha256"]
            or member["member_assessment_sha256"] != report["record_sha256"] or member["gradable"] is not True):
        raise ValueError("saved confirmation grading is unavailable or changed")
    return context


def authorize(plan):
    study.validate_plan(plan)
    if plan["binding"]["phase"] != "confirmation":
        raise ValueError("confirmation execution requires the original locked confirmation plan")
    root = primary.canonical_root()
    with primary._guard(root.parent / ".initialization.lock"):
        if (root / AUTHORIZATION).exists():
            record = verify(plan)
        else:
            pilots._authority(plan, live=True)
            with ledger._locked(root / "ledger") as context:
                directory, manifest = context
                ledger._current(manifest)
                expected, _, _ = _bindings(plan, live=False, ledger_context=context)
                state = ledger._read(directory, manifest)
                from benchtop.generation_journal import read as generation_state

                generation = generation_state()
                if state["halted"] or generation["halted"] or state["counts"]["active"] or generation["active"]:
                    raise ValueError("confirmation authorization requires idle shared accounting without a model/tool halt")
                if any(history for key, history in state["histories"].items()
                       if state["plans"][key[0]]["binding"]["phase"] == "confirmation"):
                    raise ValueError("confirmation starts appeared before independent truth authorization")
                record = study._seal({**expected, "recorded_at_utc": datetime.now(UTC).isoformat()}, "record_sha256")
                study._publish(root / AUTHORIZATION, record)
        ledger.register(plan, root / "ledger", authorization=root)
        return record


def check_start(plan, slot, state, record):
    if slot["slot_id"] not in record["initial_slot_order"]:
        raise ValueError("confirmation reward is unavailable for this frozen cohort member; preserve its planned slots")
    key = plan["plan_sha256"], slot["slot_id"]
    if not state["histories"][key]:
        next_slot = next((slot_id for slot_id in record["initial_slot_order"] if not state["histories"][(plan["plan_sha256"], slot_id)]), None)
        if slot["slot_id"] != next_slot:
            raise ValueError("confirmation initial starts must follow the frozen interleaved slot order")
