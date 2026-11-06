import hashlib
import inspect
import itertools
import json
import math
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from benchtop import cache
from benchtop.agents.common import SYSTEM, first_message
from benchtop.calibrate import canonical_hash
from benchtop.env import ROOT, Env
from benchtop.evaluate import corrected_replay
from benchtop.replay import development_gate
from benchtop.stats import GRADER_ID

PLANNER_ID = "study_plan_v1"
SPEC = ROOT / "benchmarks/release_v1/spec.json"


def _json(path):
    return json.loads(Path(path).read_text())


def _seal(value, field):
    return {**value, field: canonical_hash({k: v for k, v in value.items() if k != field})}


def _verify(value, field):
    if not isinstance(value, dict) or value.get(field) != _seal(value, field)[field]:
        raise ValueError(f"invalid {field}")


def _ref(path):
    path = Path(path).resolve()
    return {"base": "repository" if path.is_relative_to(ROOT) else "absolute",
            "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)}


def _resolve(ref):
    if not isinstance(ref, dict) or set(ref) != {"base", "path"} or ref["base"] not in ("repository", "absolute"):
        raise ValueError("invalid source reference")
    path = ((ROOT / ref["path"]) if ref["base"] == "repository" else Path(ref["path"])).resolve()
    if ref["base"] == "repository" and not path.is_relative_to(ROOT):
        raise ValueError("source escapes repository")
    return path


def _sources():
    paths = sorted((ROOT / "benchtop").rglob("*.py")) + [ROOT / "pyproject.toml", ROOT / "uv.lock", SPEC]
    return {str(path.relative_to(ROOT)): cache.file_hash(path) for path in paths}


def _model(config):
    required = {"backend", "requested_model_id", "reported_model_id", "cli_version", "decoding"}
    if not isinstance(config, dict) or set(config) != required or config["backend"] != "claude_cli":
        raise ValueError("model configuration requires exact Claude CLI identity and decoding fields")
    for name in ("requested_model_id", "reported_model_id", "cli_version"):
        value = config[name]
        if not isinstance(value, str) or not value.strip() or value.lower() in ("haiku", "sonnet", "opus", "default", "latest", "not_reported"):
            raise ValueError(f"{name} must be a resolved nonempty identifier, not a family alias")
    if config["requested_model_id"] != config["reported_model_id"]:
        raise ValueError("requested and reported resolved model IDs differ")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", config["requested_model_id"]) is None:
        raise ValueError("model IDs must be identifiers, not URLs, paths, or CLI arguments")
    controls = config["decoding"]
    if not isinstance(controls, dict) or set(controls) != {"temperature", "model_seed"}:
        raise ValueError("record temperature and model_seed availability explicitly")
    for name, control in controls.items():
        if not isinstance(control, dict):
            raise ValueError("decoding controls must be objects")
        if control.get("status") == "unavailable":
            if set(control) != {"status", "reason"} or not isinstance(control["reason"], str) or not control["reason"].strip():
                raise ValueError("unavailable decoding controls require a reason")
        elif control.get("status") == "fixed":
            value = control.get("value")
            try:
                valid = (set(control) == {"status", "value"} and type(value) in (int, float)
                         and math.isfinite(value) and value >= 0 and (name != "model_seed" or type(value) is int))
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError("fixed decoding controls require finite nonnegative values and an integer seed")
        else:
            raise ValueError("decoding control must be fixed or explicitly unavailable")
    return json.loads(json.dumps(config, allow_nan=False))


def _foundation(path):
    if path is None:
        return {"status": "missing", "issue": "A current foundation verification file is required."}
    try:
        proof = _json(path)
        if not isinstance(proof, dict):
            raise ValueError("foundation file must contain an object")
        if (proof.get("scope") != "foundation_engineering_checks" or proof.get("status") != "passed"
                or proof.get("source_inputs_unchanged") is not True or proof.get("model_sessions") != 0
                or proof.get("archive_files_verified") != 241 or proof.get("original_data_files_verified") != 200
                or not isinstance(proof.get("offline"), dict) or proof["offline"].get("status") != "passed"):
            raise ValueError("foundation file is not a passing foundation verification")
        if canonical_hash(proof.get("runtime")) != canonical_hash(cache.runtime_identity()):
            raise ValueError("foundation verification runtime changed")
        inputs = proof["input_sha256"]
        if not isinstance(inputs, dict) or not _sources().keys() <= inputs.keys():
            raise ValueError("foundation proof omits current implementation/dependency inputs")
        for name, expected in inputs.items():
            target = (ROOT / name).resolve()
            if Path(name).is_absolute() or not target.is_relative_to(ROOT) or cache.file_hash(target) != expected:
                raise ValueError("foundation input changed: " + name)
        return {"status": "current", "source": _ref(path), "sha256": cache.file_hash(path),
                "checked_at_utc": proof["checked_at_utc"], "measurement_source": "supplied_file_not_authenticated"}
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {"status": "unavailable", "issue": str(error)}


def gate(*, phase="development", model=None, reviews_dir=None, foundation=None):
    spec = _json(SPEC)
    admission = development_gate(reviews_dir or ROOT / "artifacts/reference-human-reviews")
    proof = _foundation(foundation)
    blockers = list(admission["blockers"])
    if proof["status"] != "current":
        blockers.append(proof["issue"])
    try:
        declared_model = _model(model)
        model_status = {"status": "declared", "configuration": declared_model,
                        "identity_source": "operator_reported_not_authenticated"}
    except (ValueError, TypeError) as error:
        blockers.append(str(error))
        model_status = {"status": "missing_or_invalid"}
    supported = phase in ("smoke", "development")
    if not supported:
        blockers.append("Fresh candidate/cohort/confirmation planning and lock verification are not implemented.")
    execution = ["The reconciled model runner and primary session-ledger integration are not implemented."]
    if phase == "development":
        execution.append("Successful development-lab smoke evidence and the development lock still require verification.")
    elif phase != "smoke":
        execution.append("The required fresh phase locks have not been verified.")
    return {"phase": phase, "planning_status": "blocked" if blockers else "ready",
            "planning_blockers": blockers, "execution_ready": False, "execution_blockers": execution,
            "admission": admission, "foundation": proof, "model": model_status,
            "protocol_locks": spec["lock_schedule"],
            "scope": "Read-only preparation diagnostics. No study lock, model invocation, or quota reservation."}


def _snapshot(path):
    env = Env.load(Path(path), replay=True)
    if env.cache_status == "historical":
        env = corrected_replay(env)
    elif env.grader_id != GRADER_ID or env.paper.get("arxiv_id") or env.manifest.get("seed_splits"):
        raise ValueError("preparation supports frozen historical references or controlled two-part fixtures only")
    env.assert_current()
    visible = _json(SPEC)["seed_partitions"]["visible"]
    if env.visible_seeds != visible:
        raise ValueError("plan requires the protocol's four visible seed IDs")
    files = {name: cache.file_hash(env.dir / name) for name in (*cache.DATA_FILES, "cache.json") if (env.dir / name).is_file()}
    lab_hash = canonical_hash({name: files[name] for name in ("lab.py", "space.json")})
    orders = list(itertools.permutations(visible))
    index = int(hashlib.sha256(("benchtop_release_v1/order/" + lab_hash).encode()).hexdigest(), 16) % len(orders)
    snapshot = {"env_id": env.id, "source": _ref(path), "files": files, "lab_hash": lab_hash,
                "execution_id": env.execution_id, "task_version": env.task_version,
                "task_hash": canonical_hash(env.task("claim")), "grader_id": env.grader_id,
                "charging_hash": env.charging_schedule()[1], "visible_order": list(orders[index]),
                "historical_index_sha256": cache.file_hash(env.dir.parent / "historical_cache.json")
                if env.cache_status == "historical" else None}
    env.assert_current()
    return snapshot


def _grid(binding, bindings_hash):
    spec = _json(SPEC)
    phase = binding["phase"]
    selected = binding["environments"][:2] if phase == "smoke" else binding["environments"]
    repeats = 1 if phase == "smoke" else spec["development"]["budget_selection_repeats"]
    model_hash = canonical_hash(binding["model_configuration"])
    prompt_hash = canonical_hash(binding["prompt"])
    grader_hash = canonical_hash({name: binding["source_sha256"]["benchtop/" + name] for name in ("tasks.py", "stats.py")})
    groups, slots = [], []
    for env in selected:
        for budget in spec["budget"]["fractions"]:
            descriptor = {"protocol_hash": binding["protocol_hash"], "bindings_hash": bindings_hash,
                          "phase": phase, "arm": "smoke" if phase == "smoke" else "budget_candidate",
                          "lab_hash": env["lab_hash"], "task_hash": env["task_hash"], "grader_hash": grader_hash,
                          "model_config_hash": model_hash, "prompt_template_hash": prompt_hash,
                          "charging_hash": env["charging_hash"], "budget_fraction": budget,
                          "visible_order": env["visible_order"], "show_paper": spec["agent"]["show_paper"]}
            if set(descriptor) != set(spec["identities"]["group_descriptor_fields"]):
                raise ValueError("group descriptor schema differs from protocol")
            group_id = canonical_hash(descriptor)
            groups.append({"env_id": env["env_id"], "task": "claim", "group_id": group_id, "descriptor": descriptor})
            for index in range(repeats):
                slot_id = canonical_hash({"group_id": group_id, "replicate_index": index})
                slots.append({"group_id": group_id, "slot_id": slot_id, "env_id": env["env_id"], "task": "claim",
                              "replicate_index": index, "budget_fraction": budget,
                              "attempt_ids": [canonical_hash({"slot_id": slot_id, "attempt_index": attempt}) for attempt in range(2)]})
    if len({group["group_id"] for group in groups}) != len(groups) or len({slot["slot_id"] for slot in slots}) != len(slots):
        raise ValueError("identical lab/task content would collide; use distinct sources")
    return groups, slots


def make_plan(env_paths=None, *, model, phase="smoke", mode="engineering", reviews_dir=None, foundation=None):
    if phase not in ("smoke", "development") or mode not in ("engineering", "study_preparation"):
        raise ValueError("only smoke/development preparation and engineering/study_preparation modes are supported")
    spec, declared_model = _json(SPEC), _model(model)
    readiness = None
    if mode == "study_preparation":
        if env_paths:
            raise ValueError("study development membership comes from the reviewed selection rule")
        readiness = gate(phase=phase, model=model, reviews_dir=reviews_dir, foundation=foundation)
        if readiness["planning_status"] != "ready":
            raise ValueError("study preparation blocked: " + " ".join(readiness["planning_blockers"]))
        paths = [ROOT / "envs" / env for env in readiness["admission"]["selected_env_ids"]]
    else:
        paths = [Path(path) for path in (env_paths or [])]
        if not 1 <= len(paths) <= 2:
            raise ValueError("engineering mode requires one or two explicit sources")
    environments = [_snapshot(path) for path in paths]
    if len({env["env_id"] for env in environments}) != len(environments):
        raise ValueError("environment identities must be distinct")
    core = {"planner_id": PLANNER_ID, "mode": mode, "phase": phase,
            "protocol_hash": canonical_hash(spec), "protocol_file_sha256": cache.file_hash(SPEC),
            "source_sha256": _sources(), "runtime": cache.runtime_identity(), "environments": environments,
            "model_configuration": declared_model, "model_identity_source": "operator_reported_not_authenticated",
            "prompt": {"system_text": SYSTEM, "first_message_renderer": "benchtop.agents.common.first_message",
                       "first_message_source": inspect.getsource(first_message),
                       "description_renderer_sha256": cache.file_hash(ROOT / "benchtop/episode.py")},
            "readiness": readiness, "reviews_source": _ref(reviews_dir or ROOT / "artifacts/reference-human-reviews")
            if mode == "study_preparation" else None,
            "foundation_source": _ref(foundation) if mode == "study_preparation" else None,
            "limits": spec["execution_limits"], "scope": "Immutable preparation only; no phase lock, model starts, or quota reservation."}
    core = json.loads(json.dumps(core, allow_nan=False))
    bindings_hash = canonical_hash(core)
    groups, slots = _grid(core, bindings_hash)
    return _seal({"schema_version": 1, "binding": core, "bindings_hash": bindings_hash, "groups": groups,
                  "slots": slots, "execution_ready": False, "model_sessions_started": 0}, "plan_sha256")


def validate_plan(plan):
    _verify(plan, "plan_sha256")
    binding = plan["binding"]
    if plan["bindings_hash"] != canonical_hash(binding) or binding["planner_id"] != PLANNER_ID:
        raise ValueError("invalid study binding")
    expected = make_plan([_resolve(env["source"]) for env in binding["environments"]]
                         if binding["mode"] == "engineering" else None,
                         model=binding["model_configuration"], mode=binding["mode"], phase=binding["phase"],
                         reviews_dir=_resolve(binding["reviews_source"]) if binding["reviews_source"] else None,
                         foundation=_resolve(binding["foundation_source"]) if binding["foundation_source"] else None)
    if plan != expected:
        raise ValueError("plan or bound implementation/runtime/evidence inputs changed; preserve it and create a new preparation")
    return plan


def _publish(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".study-", delete=False) as out:
        temporary = Path(out.name)
        try:
            out.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def save_plan(plan, directory):
    validate_plan(plan)
    directory = Path(directory).resolve()
    if directory.is_relative_to(ROOT) and not directory.is_relative_to(ROOT / "artifacts"):
        raise ValueError("write study preparations under artifacts/ or outside the repository")
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    _publish(directory / "plan.json", plan)
    receipt = _seal({"schema_version": 1, "record_kind": "preparation_receipt", "study_lock": False,
                     "plan_sha256": plan["plan_sha256"], "bindings_hash": plan["bindings_hash"],
                     "mode": plan["binding"]["mode"], "phase": plan["binding"]["phase"],
                     "created_at_utc": datetime.now(UTC).isoformat(), "model_sessions_started": 0}, "record_sha256")
    _publish(directory / "receipt.json", receipt)
    return receipt


def load_plan(directory):
    directory = Path(directory)
    plan, receipt = _json(directory / "plan.json"), _json(directory / "receipt.json")
    _verify(receipt, "record_sha256")
    if (receipt.get("record_kind") != "preparation_receipt" or receipt.get("study_lock") is not False
            or receipt.get("model_sessions_started") != 0 or receipt.get("plan_sha256") != plan.get("plan_sha256")
            or receipt.get("bindings_hash") != plan.get("bindings_hash") or receipt.get("mode") != plan["binding"]["mode"]
            or receipt.get("phase") != plan["binding"]["phase"]):
        raise ValueError("preparation receipt does not match its plan")
    return validate_plan(plan)
