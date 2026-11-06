import copy
import fcntl
import json
import re
import secrets
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from benchtop import calibrate, study
from benchtop.calibrate import canonical_hash
from benchtop.env import ROOT

LEDGER_ID = "session_ledger_v1"


def _json(path):
    return json.loads(Path(path).read_text())


def _writable(path):
    path = Path(path).resolve()
    if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / "artifacts"):
        raise ValueError("write engineering session records under artifacts/ or outside the repository")
    return path


def _phase_limits(spec):
    return {"smoke": spec["execution_limits"]["reserve"]["smoke"],
            "integration": spec["execution_limits"]["reserve"]["integration"],
            "development": spec["episode_accounting"]["development"]["episodes"],
            "fresh_pilot": spec["episode_accounting"]["fresh_pilots"]["episodes_max"],
            "confirmation": spec["episode_accounting"]["fresh_confirmation"]["episodes_max"]}


def _identity(plan):
    core = plan["binding"]
    if core["mode"] != "engineering":
        raise ValueError("this ledger accepts engineering preparations only; primary execution remains blocked")
    spec = _json(study.SPEC)
    return {"scope": "engineering_only", "protocol": spec, "protocol_hash": canonical_hash(spec),
            "source_sha256": core["source_sha256"], "runtime": core["runtime"],
            "model_configuration": core["model_configuration"], "prompt": core["prompt"],
            "limits": spec["execution_limits"], "phase_limits": _phase_limits(spec)}


def create(plan, directory):
    _identity(plan)
    study.validate_plan(plan)
    identity = _identity(plan)
    directory = _writable(directory)
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    (directory / "plans").mkdir(mode=0o700)
    (directory / "attempts").mkdir(mode=0o700)
    (directory / ".ledger.lock").touch(mode=0o600, exist_ok=False)
    core = {"version": LEDGER_ID, "instance_id": secrets.token_hex(16), "identity": identity}
    manifest = study._seal({**core, "ledger_id": canonical_hash(core),
                            "created_at_utc": datetime.now(UTC).isoformat()}, "record_sha256")
    study._publish(directory / "ledger.json", manifest)
    register(plan, directory)
    return inspect(directory)


def _manifest(directory):
    manifest = _json(directory / "ledger.json")
    study._verify(manifest, "record_sha256")
    core = {key: manifest[key] for key in ("version", "instance_id", "identity")}
    identity = core["identity"]
    if (core["version"] != LEDGER_ID or manifest["ledger_id"] != canonical_hash(core)
            or identity["scope"] != "engineering_only" or identity["protocol_hash"] != canonical_hash(identity["protocol"])
            or identity["limits"] != identity["protocol"]["execution_limits"]
            or identity["phase_limits"] != _phase_limits(identity["protocol"])):
        raise ValueError("invalid ledger identity or limits")
    return manifest


@contextmanager
def _locked(directory):
    directory = Path(directory).resolve()
    with (directory / ".ledger.lock").open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield directory, _manifest(directory)


def _current(manifest):
    identity = manifest["identity"]
    runtime = json.loads(json.dumps(study.cache.runtime_identity()))
    if (identity["source_sha256"] != study._sources() or identity["runtime"] != runtime
            or identity["protocol_hash"] != canonical_hash(_json(study.SPEC))):
        raise ValueError("ledger implementation/runtime/protocol changed; preserve it for audit")


def register(plan, directory):
    identity = _identity(plan)
    study.validate_plan(plan)
    with _locked(_writable(directory)) as (root, manifest):
        _current(manifest)
        _read(root, manifest)
        if identity != manifest["identity"]:
            raise ValueError("plan model/prompt/runtime/implementation does not match the shared ledger")
        path = root / "plans" / f"{plan['plan_sha256']}.json"
        if path.exists():
            if _json(path) != plan:
                raise ValueError("registered plan content changed")
        else:
            study._publish(path, plan)
        return plan["plan_sha256"]


def _bucket(phase, index):
    return "retry" if index else phase if phase in ("smoke", "integration") else "primary"


def _capacity(counts, phase, index, identity):
    limits = identity["limits"]
    if counts["total"] >= limits["total_max_episode_starts"]:
        raise ValueError("total start-intent allowance exhausted")
    if counts["active"] >= limits["max_concurrent_model_sessions"]:
        raise ValueError("two in-flight or unreconciled attempts already occupy the shared ledger")
    bucket = _bucket(phase, index)
    maximum = limits["primary_max_episodes"] if bucket == "primary" else limits["reserve"]["infrastructure_retries" if bucket == "retry" else bucket]
    if counts[bucket] >= maximum:
        raise ValueError(bucket + " start-intent allowance exhausted")
    if not index and counts["phase:" + phase] >= identity["phase_limits"][phase]:
        raise ValueError(phase + " phase allowance exhausted")


def _base(manifest, plan, slot, index):
    return {"ledger_id": manifest["ledger_id"], "protocol_hash": manifest["identity"]["protocol_hash"],
            "plan_sha256": plan["plan_sha256"], "slot_id": slot["slot_id"], "attempt_index": index,
            "attempt_id": slot["attempt_ids"][index], "phase": plan["binding"]["phase"]}


def _normalized(slot, index, outcome):
    calibrate._keys(outcome, {"state", "reward"}, {"failure_category", "costs"})
    return {**copy.deepcopy(outcome), "attempt_id": slot["attempt_ids"][index], "attempt_index": index}


def _completion(manifest, slot, index, claimed, prior, *, reported_model_id, submission_status,
                process_stopped, evidence_sha256, previous=None):
    if process_stopped is not True:
        raise ValueError("verify that the old process stopped before releasing its in-flight slot")
    if not isinstance(evidence_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", evidence_sha256) is None:
        raise ValueError("retain a SHA-256 reference to adapter/server reconciliation evidence")
    if reported_model_id is not None and (not isinstance(reported_model_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", reported_model_id) is None):
        raise ValueError("reported model must be an identifier or null")
    if submission_status not in ("committed", "not_committed", "unknown"):
        raise ValueError("submission reconciliation must be committed, not_committed, or unknown")
    claimed = copy.deepcopy(claimed)
    if previous is not None:
        costs = claimed.setdefault("costs", {})
        if not isinstance(costs, dict):
            raise ValueError("costs must be an object")
        for name, value in previous.get("costs", {}).items():
            if value is not None:
                if costs.get(name) is not None and costs[name] != value:
                    raise ValueError("resolution cannot replace an already reported cost")
                costs[name] = value
    calibrate._attempts({"slot_id": slot["slot_id"], "attempts": [*prior, claimed]}, manifest["identity"]["protocol"])
    if claimed["attempt_id"] != slot["attempt_ids"][index] or claimed["attempt_index"] != index:
        raise ValueError("completion belongs to another attempt")
    state = claimed["state"]
    if (state == "completed" and submission_status != "committed"
            or state == "technical_failure" and submission_status != "not_committed"
            or state != "unresolved" and submission_status == "unknown"):
        raise ValueError("outcome conflicts with submission reconciliation; retain uncertainty as unresolved")
    expected_model = manifest["identity"]["model_configuration"]["reported_model_id"]
    mismatch = reported_model_id is not None and reported_model_id != expected_model
    unverified = reported_model_id is None and state in ("completed", "agent_failure")
    effective = claimed
    if mismatch or unverified:
        effective = {**claimed, "state": "unresolved", "reward": None,
                     "failure_category": "reported_model_mismatch" if mismatch else "unverified_reported_model"}
    return {"claimed_outcome": claimed, "outcome": effective, "reported_model_id": reported_model_id,
            "submission_status": submission_status, "process_stopped": True, "evidence_sha256": evidence_sha256,
            "model_mismatch": mismatch, "identity_source": "adapter_supplied_not_authenticated"}


def _files(directory):
    for path in sorted(directory.iterdir()):
        if path.name.startswith(".study-"):
            continue
        if not path.is_file() or path.suffix != ".json":
            raise ValueError("unexpected published ledger entry: " + path.name)
        yield path


def _read(root, manifest):
    plans, slots, records = {}, {}, defaultdict(dict)
    for path in _files(root / "plans"):
        plan = _json(path)
        study._verify(plan, "plan_sha256")
        if (path.name != plan["plan_sha256"] + ".json" or plan["bindings_hash"] != canonical_hash(plan["binding"])
                or _identity_recorded(plan, manifest) is False):
            raise ValueError("invalid registered plan")
        plans[plan["plan_sha256"]] = plan
        for slot in plan["slots"]:
            key = plan["plan_sha256"], slot["slot_id"]
            if key in slots:
                raise ValueError("duplicate registered slot")
            slots[key] = slot
    for path in _files(root / "attempts"):
        record = _json(path)
        study._verify(record, "record_sha256")
        kind = record["record_kind"]
        if kind not in ("start", "finish", "resolve") or path.name != f"{record['attempt_id']}.{kind}.json":
            raise ValueError("unexpected attempt record name/kind")
        key = record["plan_sha256"], record["slot_id"]
        if key not in slots or type(record["attempt_index"]) is not int or record["attempt_index"] not in (0, 1):
            raise ValueError("orphaned attempt or invalid index")
        base = _base(manifest, plans[key[0]], slots[key], record["attempt_index"])
        if any(record.get(name) != value for name, value in base.items()):
            raise ValueError("attempt identity differs from its plan")
        records[(key, record["attempt_index"])][kind] = record
    histories, counts, halted = {}, Counter({name: 0 for name in ("total", "active", "primary", "retry", "smoke", "integration")}), False
    for key, slot in slots.items():
        history, prior = [], []
        for index in (0, 1):
            events = records.get((key, index), {})
            if not events:
                continue
            if "start" not in events or index != len(history):
                raise ValueError("orphaned finish/resolution or noncontiguous attempt history")
            if index and (history[-1]["effective"] is None or history[-1]["effective"]["outcome"]["state"] != "technical_failure"):
                raise ValueError("only reconciled technical failures can precede a retry")
            start = events["start"]
            bucket = _bucket(start["phase"], index)
            if start.get("bucket") != bucket:
                raise ValueError("start charged to the wrong allowance")
            counts["total"] += 1
            counts[bucket] += 1
            if not index:
                counts["phase:" + start["phase"]] += 1
            effective = None
            if "finish" not in events:
                if "resolve" in events:
                    raise ValueError("resolution has no original finish")
                counts["active"] += 1
            else:
                finish = events["finish"]
                _verify_completion(manifest, slot, index, finish, prior)
                effective = finish
                halted |= finish["model_mismatch"]
                if "resolve" in events:
                    resolution = events["resolve"]
                    if (finish["outcome"]["state"] != "unresolved" or finish["model_mismatch"]
                            or resolution.get("original_finish_sha256") != finish["record_sha256"]
                            or resolution["outcome"]["state"] == "unresolved"):
                        raise ValueError("invalid or nonterminal resolution")
                    _verify_completion(manifest, slot, index, resolution, prior, previous=finish["outcome"])
                    effective = resolution
                    halted |= resolution["model_mismatch"]
                prior.append(effective["outcome"])
            history.append({**events, "effective": effective})
        histories[key] = history
    identity = manifest["identity"]
    if (counts["total"] > identity["limits"]["total_max_episode_starts"]
            or counts["active"] > identity["limits"]["max_concurrent_model_sessions"]
            or counts["primary"] > identity["limits"]["primary_max_episodes"]
            or counts["retry"] > identity["limits"]["reserve"]["infrastructure_retries"]
            or any(counts["phase:" + phase] > maximum for phase, maximum in identity["phase_limits"].items())):
        raise ValueError("ledger records exceed the protocol allowances")
    return {"plans": plans, "slots": slots, "histories": histories, "counts": counts, "halted": halted}


def _identity_recorded(plan, manifest):
    core, identity = plan["binding"], manifest["identity"]
    return (core["mode"] == "engineering" and plan["execution_ready"] is False
            and core["protocol_hash"] == identity["protocol_hash"]
            and all(core[name] == identity[name] for name in ("source_sha256", "runtime", "model_configuration", "prompt")))


def _verify_completion(manifest, slot, index, record, prior, previous=None):
    expected = _completion(manifest, slot, index, record["claimed_outcome"], prior,
        reported_model_id=record["reported_model_id"], submission_status=record["submission_status"],
        process_stopped=record["process_stopped"], evidence_sha256=record["evidence_sha256"], previous=previous)
    if any(record.get(name) != value for name, value in expected.items()):
        raise ValueError("completion payload differs from its normalized evidence")
    if (not isinstance(record.get("recorder"), dict) or set(record["recorder"]) != {"source_sha256", "runtime_sha256"}
            or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in record["recorder"].values())):
        raise ValueError("completion recorder identity is invalid")


def begin(directory, plan_sha256, slot_id, *, retry_technical=False):
    with _locked(_writable(directory)) as (root, manifest):
        _current(manifest)
        state = _read(root, manifest)
        key = plan_sha256, slot_id
        if key not in state["slots"]:
            raise ValueError("slot is not registered in this ledger")
        plan, slot = state["plans"][plan_sha256], state["slots"][key]
        study.validate_plan(plan)
        history = state["histories"][key]
        if state["halted"]:
            raise ValueError("ledger halted after a reported model change")
        if history and (history[-1]["effective"] is None or history[-1]["effective"]["outcome"]["state"] != "technical_failure"):
            raise ValueError("attempt is final or requires reconciliation; never relaunch it automatically")
        index = len(history)
        if index > 1 or index and not retry_technical:
            raise ValueError("at most one explicit technical retry is permitted")
        _capacity(state["counts"], plan["binding"]["phase"], index, manifest["identity"])
        record = study._seal({**_base(manifest, plan, slot, index), "record_kind": "start",
                              "recorded_at_utc": datetime.now(UTC).isoformat(),
                              "bucket": _bucket(plan["binding"]["phase"], index)}, "record_sha256")
        study._publish(root / "attempts" / f"{record['attempt_id']}.start.json", record)
        return record


def _complete(directory, attempt_id, outcome, *, reported_model_id, submission_status,
              process_stopped, evidence_sha256, resolve):
    with _locked(_writable(directory)) as (root, manifest):
        state = _read(root, manifest)
        matches = [(key, index, entry) for key, history in state["histories"].items()
                   for index, entry in enumerate(history) if entry["start"]["attempt_id"] == attempt_id]
        if len(matches) != 1:
            raise ValueError("attempt has no unique registered start")
        key, index, entry = matches[0]
        slot, plan = state["slots"][key], state["plans"][key[0]]
        prior = [item["effective"]["outcome"] for item in state["histories"][key][:index]]
        previous = None
        if resolve:
            if "finish" not in entry or entry["finish"]["outcome"]["state"] != "unresolved" or entry["finish"]["model_mismatch"]:
                raise ValueError("only an unresolved finish can be reconciled, without a model mismatch")
            previous = entry["finish"]["outcome"]
        payload = _completion(manifest, slot, index, _normalized(slot, index, outcome), prior,
            reported_model_id=reported_model_id, submission_status=submission_status,
            process_stopped=process_stopped, evidence_sha256=evidence_sha256, previous=previous)
        if resolve and payload["outcome"]["state"] == "unresolved":
            raise ValueError("resolution must provide a definite outcome; keep unknown work unresolved")
        kind = "resolve" if resolve else "finish"
        payload.update(_base(manifest, plan, slot, index), record_kind=kind)
        if resolve:
            payload["original_finish_sha256"] = entry["finish"]["record_sha256"]
        if kind in entry:
            existing = entry[kind]
            if all(existing.get(name) == value for name, value in payload.items()):
                return existing
            raise ValueError("an immutable completion already exists; preserve the original outcome")
        recorder = {"source_sha256": study.cache.file_hash(Path(__file__)),
                    "runtime_sha256": canonical_hash(study.cache.runtime_identity())}
        record = study._seal({**payload, "recorded_at_utc": datetime.now(UTC).isoformat(),
                              "recorder": recorder}, "record_sha256")
        study._publish(root / "attempts" / f"{attempt_id}.{kind}.json", record)
        return record


def finish(directory, attempt_id, outcome, **evidence):
    return _complete(directory, attempt_id, outcome, resolve=False, **evidence)


def reconcile(directory, attempt_id, outcome, **evidence):
    return _complete(directory, attempt_id, outcome, resolve=True, **evidence)


def inspect(directory):
    with _locked(directory) as (root, manifest):
        state = _read(root, manifest)
        rows, normalized = [], []
        for key, slot in state["slots"].items():
            history = state["histories"][key]
            attempts = []
            for index, entry in enumerate(history):
                attempts.append(entry["effective"]["outcome"] if entry["effective"] is not None else {
                    "attempt_id": entry["start"]["attempt_id"], "attempt_index": index,
                    "state": "unresolved", "reward": None, "failure_category": "pending_reconciliation"})
            normalized.append({"attempts": attempts})
            latest = history[-1]["effective"] if history else None
            rows.append({"plan_sha256": key[0], "slot_id": slot["slot_id"],
                         "phase": state["plans"][key[0]]["binding"]["phase"],
                         "state": latest["outcome"]["state"] if latest else "in_flight_or_interrupted" if history else "unattempted",
                         "reward": latest["outcome"]["reward"] if latest else None,
                         "attempt_ids": [entry["start"]["attempt_id"] for entry in history]})
        costs = calibrate._costs(normalized)
        costs["recorded_start_intents"] = costs.pop("episode_starts")
        costs["retry_start_intents"] = costs.pop("retry_episode_starts")
        return {"schema_version": 1, "ledger_id": manifest["ledger_id"], "scope": "engineering_only",
                "registered_plans": len(state["plans"]), "slots": rows, "counts": dict(state["counts"]),
                "status_counts": dict(Counter(row["state"] for row in rows)), "halted": state["halted"],
                "costs": costs, "limits": manifest["identity"]["limits"], "phase_limits": manifest["identity"]["phase_limits"],
                "model_sessions_started_by_ledger": 0,
                "accounting": "Start intents conservatively consume allowances and are never refunded. Pending processes remain active until adapter-supplied reconciliation. No models invoked; no primary-study authorization."}


def export(directory, output):
    output = _writable(output)
    if output.is_relative_to(Path(directory).resolve()):
        raise ValueError("export snapshots outside the ledger directory")
    result = study._seal(inspect(directory), "snapshot_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    study._publish(output, result)
    return result
