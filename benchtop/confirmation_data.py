import asyncio
import json
import math
import os
import time
from datetime import UTC, datetime
from types import SimpleNamespace

from benchtop import cache, confirmation, primary, study
from benchtop import session_ledger as ledger
from benchtop.candidate_metadata import _bytes, timestamp
from benchtop.env import Table, read_jsonl
from benchtop.sandbox import Pool, default_deny_reads, sandbox_mode
from benchtop.space import Space, config_key
from benchtop.stats import GRADER_ID
from benchtop.tasks import build_task_set

COLLECTOR_ID = "independent_confirmation_data_v1"


def _now():
    return datetime.now(UTC).isoformat()


def _record(path):
    value = study._json(path)
    study._verify(value, "record_sha256")
    return value


def _put(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("immutable confirmation file changed: " + path.name)
    else:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _bytes(path, raw)


def _files(root, paths):
    result = {}
    for path in sorted(paths):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("confirmation evidence escapes its private directory")
        if path.is_file():
            result[str(path.relative_to(root))] = cache.file_hash(path)
    return result


def _verify_files(root, files):
    if _files(root, [root / name for name in files]) != files:
        raise ValueError("confirmation execution evidence changed")


def _context(ledger_context=None):
    locked = confirmation.verify(ledger_context=ledger_context)
    plan = study._json(primary.canonical_root() / "confirmation_plan.json")
    root = primary.canonical_root() / confirmation.DATA_DIRECTORY
    if root.is_symlink():
        raise ValueError("confirmation data directory cannot be a symbolic link")
    return locked, plan, root


def _configuration(locked, plan):
    candidate = study._json(primary.canonical_root() / "candidate_lock.json")
    rules = candidate["binding"]["generation"]
    return {"collector_id": COLLECTOR_ID, "schema_version": 1, "confirmation_lock_sha256": locked["record_sha256"],
            "plan_sha256": plan["plan_sha256"], "ledger_id": locked["ledger_id"], "protocol_hash": locked["protocol_hash"],
            "source_sha256": plan["binding"]["source_sha256"], "runtime": plan["binding"]["runtime"],
            "members": [member["env_id"] for member in plan["binding"]["environments"]],
            "seed_ids": plan["binding"]["truth_contract"]["seed_ids"], "workers": rules["sweep_workers"],
            "run_timeout_seconds": rules["run_timeout_seconds"], "sandbox": "macos",
            "execution_rules_sha256": study.canonical_hash(rules), "model_sessions_started": 0}


def _intent(locked, plan, root):
    record = study._json(root / "collection.json")
    study._verify(record, "collection_sha256")
    if (any(record.get(key) != value for key, value in _configuration(locked, plan).items())
            or timestamp(record["started_at_utc"]) < timestamp(locked["recorded_at_utc"])):
        raise ValueError("confirmation collection differs from its lock or frozen execution rules")
    allowed = {"collection.json", "labs", "logs", "assessment.json"}
    if any(path.name not in allowed for path in root.iterdir()):
        raise ValueError("unplanned confirmation collection entry")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("confirmation data cannot contain symbolic links")
    if (root / "labs").exists() and any(path.name not in record["members"] for path in (root / "labs").iterdir()):
        raise ValueError("unplanned confirmation lab")
    return record


def _manifest(member):
    return {"id": member["env_id"], "arxiv_version": member["version"], "source": "arxiv",
            "seeds": 28, "visible_seeds": 4, "seed_splits": member["seed_splits"],
            "grader_id": GRADER_ID, "data_role": "confirmation_only"}


def _prepare(root, member):
    folder = root / "labs" / member["env_id"]
    source = study._resolve(member["source"])
    for name in ("lab.py", "space.json"):
        _put(folder / name, (source / name).read_bytes())
    _put(folder / "manifest.json", (json.dumps(_manifest(member), indent=2) + "\n").encode())
    if not (folder / "execution_start.json").exists() and (folder / "results.jsonl").exists():
        raise ValueError("confirmation results appeared without an execution intent")
    if not (folder / "cache.json").exists():
        study._publish(folder / "cache.json", study._seal({"schema_version": "confirmation_private_v1", "data_role": "confirmation_only",
            "execution": cache.execution_binding(folder, _manifest(member))}, "private_cache_sha256"))
    _execution(folder)
    return folder


def _execution(folder):
    wrapped = study._json(folder / "cache.json")
    expected = study._seal({"schema_version": "confirmation_private_v1", "data_role": "confirmation_only",
        "execution": cache.execution_binding(folder, study._json(folder / "manifest.json"))}, "private_cache_sha256")
    if wrapped != expected:
        raise ValueError("private confirmation cache identity, source or runtime changed")
    return wrapped["execution"]


def _rows(folder, intent):
    rows = read_jsonl(folder / "results.jsonl")
    if any(row["seed"] not in intent["seed_ids"] for row in rows):
        raise ValueError("confirmation cache contains visible, qualification or undeclared seeds")
    space = Space.load(folder / "space.json")
    cells = [(config_key(space.normalize(row["config"])), row["seed"]) for row in rows]
    if len(cells) != len(set(cells)):
        raise ValueError("duplicate confirmation cell; do not select a preferred repeated result")
    return rows


async def _execute(folder, jobs, intent, logs, record):
    async with Pool(folder / "lab.py", intent["workers"], intent["run_timeout_seconds"], intent["sandbox"], logs,
                    deny_reads=[*default_deny_reads(), primary.canonical_root()]) as pool:
        async def one(config, seed):
            record(config, seed, await pool.run(config, seed))

        await asyncio.gather(*(one(config, seed) for config, seed in jobs))


def _work_gate(directory, manifest):
    ledger._current(manifest)
    state = ledger._read(directory, manifest)
    if manifest["identity"]["scope"] != "primary" or state["halted"] or state["counts"]["active"]:
        raise ValueError("confirmation data work requires idle primary accounting without a model/tool halt")
    if any(history for key, history in state["histories"].items() if state["plans"][key[0]]["binding"]["phase"] == "confirmation"):
        raise ValueError("confirmation model observations preceded completed independent truth data")


def _collect_lab(locked, plan, root, intent, member):
    folder = _prepare(root, member)
    if (folder / "execution_finish.json").exists():
        return folder
    if (folder / "execution_start.json").exists():
        raise ValueError("confirmation execution already started; preserve partial rows for explicit reconciliation without rerunning")
    sandbox_mode(intent["sandbox"])
    space = Space.load(folder / "space.json")
    jobs = [(config, seed) for config in space.configs() for seed in intent["seed_ids"]]
    inputs = [folder / name for name in ("lab.py", "space.json", "manifest.json", "cache.json")]
    with ledger._locked(primary.canonical_root() / "ledger") as context:
        _work_gate(*context)
        confirmation.verify(ledger_context=context)
        started = study._seal({"collection_sha256": intent["collection_sha256"], "env_id": member["env_id"],
            "files": _files(root, inputs), "jobs_sha256": study.canonical_hash(jobs), "recorded_job_intents": len(jobs),
            "started_at_utc": _now()}, "record_sha256")
        study._publish(folder / "execution_start.json", started)
    protected = [*inputs, root / "collection.json", primary.canonical_root() / "confirmation_lock.json",
                 primary.canonical_root() / "confirmation_plan.json"]
    protected.extend(study._resolve(member["source"]) / name for name in member["files"])
    protected.extend(study.ROOT / name for name in plan["binding"]["source_sha256"])
    signatures = {path: cache.file_signature(path) for path in protected}

    def check():
        if any(cache.file_signature(path) != signature for path, signature in signatures.items()):
            raise ValueError("frozen confirmation inputs changed during execution")

    binding = _execution(folder)
    seen = set()
    _put(folder / "results.jsonl", b"")
    wall = time.perf_counter()
    with (folder / "results.jsonl").open("a") as out:
        def record(config, seed, result):
            check()
            cell = config_key(space.normalize(config)), seed
            if seed not in intent["seed_ids"] or cell in seen:
                raise ValueError("worker returned an undeclared or duplicate confirmation job")
            try:
                row = cache.seal_result(binding, space, config, seed, {key: value for key, value in result.items() if key != "trace"})
            except (AttributeError, KeyError, TypeError, ValueError):
                reported = result.get("secs") if isinstance(result, dict) else None
                available = type(reported) in (int, float) and math.isfinite(reported) and reported >= 0
                row = cache.seal_result(binding, space, config, seed, {"ok": False, "error": "invalid worker result",
                    "secs": reported if available else 0.0, "timing_unreported": not available})
            seen.add(cell)
            out.write(json.dumps(row, allow_nan=False) + "\n")
            out.flush()

        asyncio.run(_execute(folder, jobs, intent, root / "logs" / member["env_id"], record))
        out.flush()
        os.fsync(out.fileno())
    check()
    confirmation.verify()
    _rows(folder, intent)
    finished = study._seal({"start_sha256": started["record_sha256"], "collection_sha256": intent["collection_sha256"],
        "files": _files(root, [*inputs, folder / "execution_start.json", folder / "results.jsonl",
                               *(root / "logs" / member["env_id"]).glob("*")]),
        "collection_wall_seconds": time.perf_counter() - wall, "finished_at_utc": _now()}, "record_sha256")
    study._publish(folder / "execution_finish.json", finished)
    return folder


def _report(root, intent, member):
    folder = root / "labs" / member["env_id"]
    started, finished = _record(folder / "execution_start.json"), _record(folder / "execution_finish.json")
    required = {str((folder / name).relative_to(root)) for name in (
        "lab.py", "space.json", "manifest.json", "cache.json", "results.jsonl", "execution_start.json")}
    if not required <= finished["files"].keys() or started["env_id"] != member["env_id"]:
        raise ValueError("confirmation finish lacks its full execution evidence")
    _verify_files(root, finished["files"])
    if (started["collection_sha256"] != intent["collection_sha256"] or finished["collection_sha256"] != intent["collection_sha256"]
            or finished["start_sha256"] != started["record_sha256"]
            or not timestamp(intent["started_at_utc"]) <= timestamp(started["started_at_utc"]) <= timestamp(finished["finished_at_utc"])):
        raise ValueError("confirmation execution chronology or intent binding changed")
    source = study._resolve(member["source"])
    if (any((folder / name).read_bytes() != (source / name).read_bytes() for name in ("lab.py", "space.json"))
            or study._json(folder / "manifest.json") != _manifest(member)
            or (folder / "tasks.json").exists()):
        raise ValueError("confirmation copy differs from its frozen qualification source or private data role")
    rows = _rows(folder, intent)
    space, binding = Space.load(folder / "space.json"), _execution(folder)
    if any(cache.seal_result(binding, space, row["config"], row["seed"], row) != row for row in rows):
        raise ValueError("private confirmation result checksum or execution identity changed")
    env = SimpleNamespace(space=space, table=Table(space, rows), execution_id=binding["id"], heldout_seeds=intent["seed_ids"])
    jobs = [(config, seed) for config in space.configs() for seed in intent["seed_ids"]]
    if (env.execution_id != member["execution_id"] or space.min_effect != member["min_effect"]
            or started["jobs_sha256"] != study.canonical_hash(jobs) or started["recorded_job_intents"] != len(jobs)
            or started["files"] != _files(root, [folder / name for name in ("lab.py", "space.json", "manifest.json", "cache.json")])):
        raise ValueError("confirmation execution identity, margin or planned seed grid changed")
    artifact = build_task_set(env, seeds=intent["seed_ids"])
    claim = next((item for item in artifact["contrasts"] if item["task_id"] == "claim"), None)
    label = claim["label"] if claim else None
    complete = len(rows) == len(jobs) and all(row["ok"] for row in rows) and artifact["status"] == "complete"
    state = "incomplete" if not complete else "ambiguous" if label is None else "stable" if label == member["qualification_label"] else "unstable"
    timed = [row for row in rows if not row.get("timing_unreported") and not row.get("error", "").startswith("worker crashed")]
    known_seconds = math.fsum(row["secs"] for row in timed)
    unknown = len(jobs) - len(timed)
    _verify_files(root, finished["files"])
    if study._sources() != intent["source_sha256"] or json.loads(json.dumps(cache.runtime_identity())) != intent["runtime"]:
        raise ValueError("confirmation assessment code or runtime changed")
    return study._seal({"env_id": member["env_id"], "status": state, "gradable": state == "stable",
        "qualification_label": member["qualification_label"], "confirmation_label": label, "seed_ids": intent["seed_ids"],
        "question_task_hash": member["task_hash"], "execution_id": env.execution_id, "grader_id": GRADER_ID,
        "min_effect": env.space.min_effect, "task_artifact": artifact, "execution_finish_sha256": finished["record_sha256"],
        "finished_at_utc": finished["finished_at_utc"], "recorded_job_intents": len(jobs), "observed_results": len(rows),
        "failed_results": sum(not row["ok"] for row in rows),
        "reported_job_seconds": {"known_sum": known_seconds, "unreported_jobs": unknown, "total": None if unknown else known_seconds},
        "collection_wall_seconds": finished["collection_wall_seconds"], "actual_environment_cpu_seconds": None,
        "cpu_availability": "Worker timers report elapsed job seconds, not process CPU time.",
        "model_sessions_started": 0}, "record_sha256")


def _assessment(locked, plan, root, intent, reports):
    return study._seal({"schema_version": 1, "collector_id": COLLECTOR_ID, "confirmation_lock_sha256": locked["record_sha256"],
        "plan_sha256": plan["plan_sha256"], "collection_sha256": intent["collection_sha256"] if intent else None,
        "status": "assessed" if reports else "no_primary_comparison", "full_cohort_size": len(plan["binding"]["environments"]),
        "gradable_labs": sum(report["gradable"] for report in reports), "members": reports,
        "files": _files(root, [path for path in root.rglob("*") if path.is_file() and path != root / "assessment.json"]) if intent else {},
        "study_lock": False, "provider_execution_authorized": False, "model_sessions_started": 0,
        "scope": "Private independent confirmation truth/stability only. All frozen cohort members retained; qualification files untouched; no rollout execution authorization."}, "assessment_sha256")


def load_assessment(*, ledger_context=None):
    locked, plan, root = _context(ledger_context)
    if not plan["slots"]:
        return _assessment(locked, plan, root, None, [])
    intent = _intent(locked, plan, root)
    reports = [_report(root, intent, member) for member in plan["binding"]["environments"]]
    for report in reports:
        if _record(root / "labs" / report["env_id"] / "assessment.json") != report:
            raise ValueError("saved confirmation truth or stability assessment changed")
    expected = _assessment(locked, plan, root, intent, reports)
    actual = study._json(root / "assessment.json")
    study._verify(actual, "assessment_sha256")
    if actual != expected:
        raise ValueError("confirmation assessment or collected evidence changed")
    return actual


def collect():
    locked, plan, root = _context()
    if not plan["slots"]:
        return _assessment(locked, plan, root, None, [])
    with primary._guard(root.parent / ".confirmation-data.lock"):
        locked, plan, root = _context()
        if (root / "assessment.json").exists():
            return load_assessment()
        if not (root / "collection.json").exists():
            if root.exists() and any(root.iterdir()):
                raise ValueError("unrecorded confirmation data directory; do not adopt earlier observations")
            with ledger._locked(root.parent / "ledger") as context:
                _work_gate(*context)
                confirmation.verify(ledger_context=context)
                root.mkdir(parents=True, exist_ok=True, mode=0o700)
                study._publish(root / "collection.json", study._seal({**_configuration(locked, plan), "started_at_utc": _now()}, "collection_sha256"))
        intent = _intent(locked, plan, root)
        reports = []
        for member in plan["binding"]["environments"]:
            folder = _collect_lab(locked, plan, root, intent, member)
            report = _report(root, intent, member)
            _put(folder / "assessment.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
            reports.append(report)
        result = _assessment(locked, plan, root, intent, reports)
        study._publish(root / "assessment.json", result)
        return result


def status():
    locked, plan, root = _context()
    if not plan["slots"] or (root / "assessment.json").exists():
        return load_assessment()
    if not root.exists():
        return {"status": "not_collected", "full_cohort_size": len(plan["binding"]["environments"]), "members": [], "model_sessions_started": 0}
    _intent(locked, plan, root)
    members = []
    for member in plan["binding"]["environments"]:
        folder = root / "labs" / member["env_id"]
        state = "pending_finalization" if (folder / "execution_finish.json").exists() else "pending_reconciliation" if (folder / "execution_start.json").exists() else "not_started"
        members.append({"env_id": member["env_id"], "status": state})
    return {"status": "pending", "full_cohort_size": len(members), "members": members, "model_sessions_started": 0}
