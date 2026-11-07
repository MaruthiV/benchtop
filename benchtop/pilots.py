import copy
from datetime import UTC, datetime

from benchtop import cohort, primary, study
from benchtop import session_ledger as ledger
from benchtop.env import Env, read_jsonl, seed_groups
from benchtop.stats import GRADER_ID


def load_environment(member):
    path = study._resolve(member["source"])
    spec = study._json(study.SPEC)
    groups = {name: spec["seed_partitions"][name] for name in ("visible", "qualification", "confirmation")}
    allowed = groups["visible"] + groups["qualification"]
    if any(row["seed"] not in allowed for row in read_jsonl(path / "results.jsonl")):
        raise ValueError("fresh pilot source contains confirmation or undeclared seed data")
    required = {*study.cache.DATA_FILES, "cache.json"}
    if (set(member["files"]) != required or any(study.cache.file_hash(path / name) != sha for name, sha in member["files"].items())):
        raise ValueError("fresh pilot source differs from frozen cohort files")
    env = Env.load(path)
    if (env.cache_status != "current" or seed_groups(env.manifest) != groups or member["seed_splits"] != groups
            or env.id != member["env_id"] or env.manifest.get("arxiv_version") != member["version"]
            or env.grader_id != GRADER_ID or env.execution_id != member["execution_id"]
            or env.task_version != member["task_version"] or study.canonical_hash(env.task("claim")) != member["task_hash"]
            or env.charging_schedule()[1] != member["charging_hash"] or env.space.min_effect != member["min_effect"]):
        raise ValueError("fresh pilot task, qualification partition, prices or execution identity changed")
    if sorted(member["visible_order"]) != groups["visible"] or len(member["visible_order"]) != len(groups["visible"]):
        raise ValueError("fresh pilot visible order must use the four visible seeds exactly once")
    env.assert_current()
    return env


def make_plan(*, ledger_context=None):
    locked = cohort.verify(ledger_context=ledger_context)
    binding, spec = locked["binding"], study._json(study.SPEC)
    members = copy.deepcopy(binding["members"])
    for member in members:
        load_environment(member)
    core = {"planner_id": study.PLANNER_ID, "mode": "study_preparation", "phase": "fresh_pilot",
            "protocol_hash": binding["protocol_hash"], "protocol_file_sha256": study.cache.file_hash(study.SPEC),
            "source_sha256": study._sources(), "runtime": binding["runtime"], "environments": members,
            "model_configuration": binding["model_configuration"], "model_identity_source": "inherited_frozen_study_configuration",
            "prompt": binding["prompt"], "limits": spec["execution_limits"], "cohort_lock_sha256": locked["record_sha256"],
            "global_budget_fraction": binding["global_budget_fraction"], "cohort_size": len(members),
            "scope": "Immutable fresh pilot preparation only; qualification truth, no confirmation data, model starts or authorization."}
    digest = study.canonical_hash(core)
    groups, slots = study._grid(core, digest)
    if len(slots) != binding["planned_episode_limits"]["fresh_pilot"]:
        raise ValueError("pilot grid differs from the frozen cohort's reduced episode ceiling")
    return study._seal({"schema_version": 1, "binding": core, "bindings_hash": digest, "groups": groups, "slots": slots,
                        "execution_ready": False, "model_sessions_started": 0}, "plan_sha256")


def check_open(directory, plan):
    if plan["binding"]["phase"] == "fresh_pilot" and any((directory.parent / name).exists() for name in ("pilot_collection.json", "confirmation_lock.json")):
        raise ValueError("fresh pilot collection is closed; new starts and outcome changes are forbidden")


def _authority(plan, *, live):
    root = primary.canonical_root()
    smoke = study._json(root / "smoke_plan.json")
    authorization = primary.verify(smoke, live=live)
    if any(smoke["binding"][key] != plan["binding"][key] for key in ("model_configuration", "prompt", "source_sha256", "runtime")):
        raise ValueError("fresh pilot model, prompt or source differs from original smoke authorization")
    return authorization


def authorize(plan):
    study.validate_plan(plan)
    if plan["binding"]["phase"] != "fresh_pilot" or not plan["slots"]:
        raise ValueError("fresh pilot authorization requires a nonempty locked cohort; zero labs imply no primary comparison")
    root = primary.canonical_root()
    with primary._guard(root.parent / ".initialization.lock"):
        if (root / "pilot_authorization.json").exists():
            raise FileExistsError("fresh pilots are already authorized; preserve the original plan and allowance")
        authorization = _authority(plan, live=True)
        with ledger._locked(root / "ledger") as context:
            directory, manifest = context
            ledger._current(manifest)
            study.validate_plan(plan, ledger_context=context)
            check_open(directory, plan)
            state = ledger._read(directory, manifest)
            if manifest["identity"]["scope"] != "primary" or state["halted"] or state["counts"]["active"]:
                raise ValueError("fresh pilot authorization requires an idle canonical journal without a model/tool halt")
            if any(history for key, history in state["histories"].items()
                   if state["plans"][key[0]]["binding"]["phase"] in ("fresh_pilot", "confirmation")):
                raise ValueError("fresh observations appeared before pilot authorization")
            record = study._seal({"schema_version": 1, "id": "fresh_pilot_authorization", "study_lock": False,
                "plan_sha256": plan["plan_sha256"], "bindings_hash": plan["bindings_hash"],
                "protocol_hash": plan["binding"]["protocol_hash"], "cohort_lock_sha256": plan["binding"]["cohort_lock_sha256"],
                "ledger_id": manifest["ledger_id"], "canonical_root": str(root.resolve()),
                "provider_sha256": authorization["provider_sha256"], "smoke_authorization_sha256": authorization["record_sha256"],
                "extra_usage_disabled": True, "billing_confirmation_source": "inherited_operator_attestation", "legacy_integration_starts": 0,
                "planned_slots": len(plan["slots"]), "recorded_at_utc": datetime.now(UTC).isoformat(), "model_sessions_started": 0}, "record_sha256")
            study._publish(root / "pilot_plan.json", plan)
            study._publish(root / "pilot_authorization.json", record)
        ledger.register(plan, root / "ledger", authorization=root)
        return record


def verify(plan, *, live=True, ledger_context=None):
    study.validate_plan(plan, ledger_context=ledger_context)
    root = primary.canonical_root()
    record = study._json(root / "pilot_authorization.json")
    study._verify(record, "record_sha256")
    manifest = ledger._manifest(root / "ledger")
    authority = _authority(plan, live=live)
    expected = {"schema_version": 1, "id": "fresh_pilot_authorization", "study_lock": False,
                "plan_sha256": plan["plan_sha256"], "bindings_hash": plan["bindings_hash"],
                "protocol_hash": plan["binding"]["protocol_hash"], "cohort_lock_sha256": plan["binding"]["cohort_lock_sha256"],
                "ledger_id": manifest["ledger_id"], "canonical_root": str(root.resolve()),
                "provider_sha256": authority["provider_sha256"], "smoke_authorization_sha256": authority["record_sha256"],
                "extra_usage_disabled": True, "billing_confirmation_source": "inherited_operator_attestation", "legacy_integration_starts": 0,
                "planned_slots": len(plan["slots"]), "model_sessions_started": 0}
    if (not plan["slots"] or any(record.get(key) != value for key, value in expected.items())
            or study._json(root / "pilot_plan.json") != plan or manifest["identity"]["scope"] != "primary"):
        raise ValueError("fresh pilot authorization differs from the canonical plan, cohort or shared journal")
    return record
