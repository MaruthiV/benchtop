from datetime import UTC, datetime
from pathlib import Path

from benchtop import primary, provider, runner, study
from benchtop import session_ledger as ledger


def directory():
    return primary.canonical_root() / "generation"


def initialize(lock):
    root = directory()
    identity = {"schema_version": 1, "id": "fresh_generation_journal_v1",
                "candidate_lock_sha256": lock["record_sha256"], "rules": lock["binding"]["generation"],
                "versions": [row["version"] for row in lock["binding"]["candidates"]]}
    if (root / "journal.json").exists():
        if study._json(root / "journal.json") != study._seal(identity, "journal_sha256"):
            raise ValueError("generation journal belongs to different locked rules")
    else:
        root.mkdir(mode=0o700, exist_ok=False)
        (root / "calls").mkdir(mode=0o700)
        (root / "papers").mkdir(mode=0o700)
        study._publish(root / "journal.json", study._seal(identity, "journal_sha256"))
    return root


def classify(start, process):
    if process["process_stopped"] is not True:
        raise ValueError("generation process stop is unverified; retain its active slot")
    stream = process["stream"]
    models = stream["models"]
    changed = any(model != start["model"]["reported_model_id"] for model in models)
    violation = stream.get("isolation_valid") is not True and (stream.get("init_observed") is True or stream.get("synthetic_control") is True)
    valid = (not changed and stream.get("isolation_valid") is True and models and not process["stdout_truncated"])
    final = stream.get("final") or {}
    success = valid and process["returncode"] == 0 and not process["timed_out"] and final.get("is_error") is False and isinstance(final.get("result"), str)
    return {"state": "completed" if success else "unresolved", "halted": bool(changed or violation),
            "text": final["result"] if success else None,
            "model_usage": final.get("modelUsage"), "process_wall_seconds": process["process_wall_seconds"]}


def _evidence(work, start):
    process = study._json(work / "process.json")
    with (work / "stdout.jsonl").open("rb") as source:
        stream = provider.parse_text_stream(source.read(1024 * 1024).decode(errors="replace"))
    if (process["executor"] != "claude_cli" or process["stream"] != stream
            or process["stdout_sha256"] != runner._digest(work / "stdout.jsonl")
            or process["stderr_sha256"] != runner._digest(work / "stderr.txt")):
        raise ValueError("generation process evidence differs from retained stdout/stderr")
    result = classify(start, process)
    files = {name: runner._digest(work / name) for name in ("start.json", "process.json", "stdout.jsonl", "stderr.txt", "mcp.json", "settings.json")}
    return study._seal({"call_id": start["call_id"], "candidate_lock_sha256": start["candidate_lock_sha256"],
                       "result": result, "files": files}, "record_sha256")


def read(root=None):
    root = directory() if root is None else Path(root)
    if not root.exists():
        return {"active": 0, "halted": False, "starts": 0, "calls": {}}
    manifest = study._json(root / "journal.json")
    study._verify(manifest, "journal_sha256")
    lock = study._json(root.parent / "candidate_lock.json")
    study._verify(lock, "record_sha256")
    if (manifest["candidate_lock_sha256"] != lock["record_sha256"] or manifest["rules"] != lock["binding"]["generation"]
            or manifest["versions"] != [row["version"] for row in lock["binding"]["candidates"]]):
        raise ValueError("generation journal differs from its candidate lock")
    calls, active, halted = {}, 0, False
    for work in sorted((root / "calls").iterdir()):
        start = study._json(work / "start.json")
        study._verify(start, "record_sha256")
        identity = {key: start[key] for key in ("candidate_lock_sha256", "version", "stage")}
        if (start["call_id"] != study.canonical_hash(identity) or work.name != start["call_id"]
                or start["stage"] not in ("triage", "synthesis", "repair_1", "repair_2") or start["version"] not in manifest["versions"]
                or start["candidate_lock_sha256"] != manifest["candidate_lock_sha256"]
                or start["model"] != manifest["rules"]["models"]["repair" if start["stage"].startswith("repair_") else start["stage"]]):
            raise ValueError("invalid generation call identity")
        finish = None
        if (work / "finish.json").exists():
            finish = study._json(work / "finish.json")
            study._verify(finish, "record_sha256")
            if finish != _evidence(work, start):
                raise ValueError("generation completion evidence changed")
            halted |= finish["result"]["halted"]
        else:
            active += 1
        calls[start["call_id"]] = {"start": start, "finish": finish}
    if len(calls) > 64:
        raise ValueError("generation invocation allowance exceeded")
    return {"active": active, "halted": halted, "starts": len(calls), "calls": calls}


def resume(call_id):
    with ledger._locked(primary.canonical_root() / "ledger"):
        state = read()
        if call_id not in state["calls"]:
            raise ValueError("generation call has no registered start")
        entry = state["calls"][call_id]
        if entry["finish"] is not None:
            return entry["finish"]
        work = directory() / "calls" / call_id
        finish = _evidence(work, entry["start"])
        study._publish(work / "finish.json", finish)
        return finish


def invoke(lock, paper, stage, prompt, system):
    if stage not in ("triage", "synthesis", "repair_1", "repair_2") or paper not in lock["binding"]["candidates"]:
        raise ValueError("generation stage or paper lies outside the locked plan")
    identity = {"candidate_lock_sha256": lock["record_sha256"], "version": paper["version"], "stage": stage}
    call_id = study.canonical_hash(identity)
    work = directory() / "calls" / call_id
    model = lock["binding"]["generation"]["models"]["repair" if stage.startswith("repair_") else stage]
    preflight = study._json(primary.canonical_root() / "provider.json")
    with ledger._locked(primary.canonical_root() / "ledger") as (root, manifest):
        if (primary.canonical_root() / "cohort_lock.json").exists():
            raise ValueError("cohort is locked; generation calls are closed")
        if manifest["identity"]["scope"] != "primary":
            raise ValueError("generation requires the canonical primary journal")
        ledger._current(manifest)
        accounting, state = ledger._read(root, manifest), read()
        if call_id in state["calls"]:
            entry = state["calls"][call_id]
            if entry["start"]["prompt"] != prompt or entry["start"]["system"] != system:
                raise ValueError("generation prompt changed for an existing call")
            if entry["finish"] is None:
                raise ValueError("generation call is pending; resume retained evidence without relaunching")
            return entry["finish"]
        if accounting["halted"] or state["halted"]:
            raise ValueError("study halted after a provider/model contract violation")
        if accounting["counts"]["active"] + state["active"] >= 2:
            raise ValueError("two model sessions already occupy the shared study allowance")
        if len(state["calls"]) >= 64:
            raise ValueError("bounded generation invocation allowance exhausted")
        provider.verify(preflight, model)
        work.mkdir(mode=0o700, exist_ok=False)
        study._publish(work / "start.json", study._seal({**identity, "call_id": call_id, "model": model,
            "prompt": prompt, "system": system, "recorded_at_utc": datetime.now(UTC).isoformat()}, "record_sha256"))
    process = provider.execute_text(work, prompt, model, preflight, system, lock["binding"]["generation"]["timeout_seconds"])
    study._publish(work / "process.json", process)
    return resume(call_id)
