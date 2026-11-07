import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from benchtop import provider, study
from benchtop.calibrate import canonical_hash
from benchtop.env import ROOT

STATE_BASE = ROOT / ".state"
BINDINGS = ROOT / "benchmarks/release_v1/bindings.json"


def canonical_root():
    return STATE_BASE / "study" / canonical_hash(study._json(study.SPEC))


def legacy_directory():
    return STATE_BASE / "integration"


def block_legacy(directory):
    if Path(directory).resolve() == legacy_directory().resolve() and (canonical_root() / "smoke_authorization.json").exists():
        raise ValueError("primary accounting is active; legacy integration starts cannot use a separate allowance")


@contextmanager
def _guard(path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield lock


def _plan(plan, phase):
    if plan["binding"]["mode"] != "study_preparation" or plan["binding"]["phase"] != phase:
        raise ValueError("primary authorization requires a reviewed study preparation for the exact phase")
    study.validate_plan(plan)


def authorize_smoke(plan, preflight, *, extra_usage_disabled=False):
    from benchtop import session_ledger as ledger

    _plan(plan, "smoke")
    provider.verify(preflight, plan["binding"]["model_configuration"])
    if extra_usage_disabled is not True:
        raise ValueError("confirm subscription extra usage is disabled to preserve the zero-new-spend constraint")
    root = canonical_root()
    with _guard(root.parent / ".initialization.lock"), _guard(legacy_directory() / "starts.jsonl") as legacy:
        if legacy.read().strip():
            raise ValueError("existing integration starts require explicit reconciliation before primary authorization; none are discarded")
        root.mkdir(mode=0o700, exist_ok=False)
        study._publish(root / "provider.json", preflight)
        study._publish(root / "smoke_plan.json", plan)
        record = study._seal({"schema_version": 1, "id": "smoke_authorization", "study_lock": False,
            "plan_sha256": plan["plan_sha256"], "bindings_hash": plan["bindings_hash"],
            "protocol_hash": plan["binding"]["protocol_hash"], "canonical_root": str(root.resolve()),
            "provider_sha256": preflight["preflight_sha256"], "extra_usage_disabled": True,
            "billing_confirmation_source": "operator_attestation_not_authenticated", "legacy_integration_starts": 0,
            "recorded_at_utc": datetime.now(UTC).isoformat()}, "record_sha256")
        study._publish(root / "smoke_authorization.json", record)
        ledger.create(plan, root / "ledger", authorization=root)
        return record


def verify(plan, root=None, *, allow_missing_ledger=False, live=True, ledger_context=None):
    root = canonical_root() if root is None else Path(root).resolve()
    if root.resolve() != canonical_root().resolve():
        raise ValueError("primary execution must use the single canonical study directory")
    phase = plan["binding"]["phase"]
    if phase == "fresh_pilot":
        from benchtop import pilots

        return pilots.verify(plan, live=live, ledger_context=ledger_context)
    if phase not in ("smoke", "development"):
        raise ValueError("fresh phase authorization is not implemented")
    _plan(plan, phase)
    record = study._json(root / ("smoke_authorization.json" if phase == "smoke" else "development_lock.json"))
    study._verify(record, "record_sha256")
    expected_id = "smoke_authorization" if phase == "smoke" else "development_lock"
    if (record.get("id") != expected_id or record.get("plan_sha256") != plan["plan_sha256"]
            or record.get("bindings_hash") != plan["bindings_hash"] or record.get("protocol_hash") != plan["binding"]["protocol_hash"]
            or record.get("canonical_root") != str(root.resolve()) or record.get("extra_usage_disabled") is not True
            or record.get("legacy_integration_starts") != 0):
        raise ValueError("phase authorization differs from the current plan or accounting contract")
    preflight = study._json(root / "provider.json")
    if record["provider_sha256"] != preflight["preflight_sha256"]:
        raise ValueError("authorized provider binding changed")
    provider.verify(preflight, plan["binding"]["model_configuration"], live=live)
    legacy = legacy_directory() / "starts.jsonl"
    if legacy.exists() and legacy.read_text().strip():
        raise ValueError("unreconciled legacy integration usage appeared after authorization")
    if not allow_missing_ledger and not (root / "ledger/ledger.json").exists():
        raise ValueError("primary journal is incomplete")
    if phase == "development":
        from benchtop.runner import _digest

        for name, sha in record["evidence"].items():
            path = (root / name).resolve()
            if Path(name).is_absolute() or not path.is_relative_to(root) or _digest(path) != sha:
                raise ValueError("locked smoke evidence changed")
        public = study._json(BINDINGS)
        if public.get("protocol_hash") != record["protocol_hash"] or not any(
            row.get("id") == "development_lock" and row.get("record_sha256") == record["record_sha256"]
            for row in public.get("records", [])):
            raise ValueError("development lock is not present in the append-only public bindings")
    return record


def _smokes(root, plan):
    from benchtop import runner
    from benchtop import session_ledger as ledger

    inventory = {}
    with ledger._locked(root / "ledger") as (journal, manifest):
        state = ledger._read(journal, manifest)
        if manifest["identity"]["scope"] != "primary" or state["halted"]:
            raise ValueError("smoke evidence needs a nonhalted primary journal")
        if len(plan["slots"]) != 6:
            raise ValueError("complete the six planned smoke slots before development lock")
        for slot in plan["slots"]:
            history = state["histories"].get((plan["plan_sha256"], slot["slot_id"]), [])
            final = history[-1]["effective"] if history else None
            if final is None or final["outcome"]["state"] != "completed" or final["submission_status"] != "committed":
                raise ValueError("all six smoke slots need committed, model-verified outcomes; wrong answers still count")
            work = journal / "runner" / final["attempt_id"]
            completion = runner._validated_evidence(manifest, plan, slot, final["attempt_id"], work)
            if (ledger._normalized(slot, final["attempt_index"], completion["outcome"]) != final["claimed_outcome"]
                    or any(final.get(key) != value for key, value in completion.items() if key != "outcome")):
                raise ValueError("smoke journal completion differs from its provider evidence")
            proof = study._json(work / "evidence.json")
            study._verify(proof, "artifact_sha256")
            if proof.get("runner_id") != runner.PROVIDER_RUNNER_ID or runner._digest(work / "evidence.json") != final["evidence_sha256"]:
                raise ValueError("scripted or unbound records cannot satisfy primary smoke requirements")
            process = study._json(work / "process.json")
            sealed = study._json(work / "sealed.json")
            if (process.get("executor") != "claude_cli" or process.get("process_stopped") is not True
                    or process["stream"].get("isolation_valid") is not True or process["stream"].get("synthetic_control") is not False
                    or sealed.get("sealed") is not True or sealed["summary"]["result"]["score"] != final["outcome"]["reward"]
                    or sealed["trace"].get("submission_source") != "mcp"):
                raise ValueError("smoke provider/process/transport evidence is incomplete")
            successful = {event["tool"] for event in sealed["trace"]["events"] if not event["error"]}
            if not {"describe_lab", "run_experiments", "submit_answer"} <= successful:
                raise ValueError("smoke must exercise all three tools without grade leakage")
            for path in work.iterdir():
                if path.is_file():
                    inventory[str(path.relative_to(root))] = runner._digest(path)
            for entry in history:
                for kind in ("start", "finish", "resolve"):
                    if kind in entry:
                        path = journal / "attempts" / f"{entry[kind]['attempt_id']}.{kind}.json"
                        inventory[str(path.relative_to(root))] = runner._digest(path)
    return inventory


def _append_binding(record):
    if BINDINGS.exists():
        value = study._json(BINDINGS)
        if (value.get("schema_version") != 1 or value.get("protocol_hash") != record["protocol_hash"]
                or not isinstance(value.get("records"), list)):
            raise ValueError("existing binding history is incompatible; preserve it")
    else:
        value = {"schema_version": 1, "protocol_hash": record["protocol_hash"], "records": []}
    if any(row.get("id") == record["id"] for row in value["records"]):
        raise ValueError("this protocol lock was already recorded")
    row = {key: record[key] for key in ("id", "plan_sha256", "bindings_hash", "record_sha256", "recorded_at_utc")}
    result = {**value, "records": [*value["records"], row]}
    BINDINGS.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=BINDINGS.parent, delete=False) as out:
        temporary = Path(out.name)
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    try:
        os.replace(temporary, BINDINGS)
    finally:
        temporary.unlink(missing_ok=True)


def lock_development(plan):
    from benchtop import session_ledger as ledger

    _plan(plan, "development")
    root = canonical_root()
    with _guard(root.parent / ".initialization.lock"):
        smoke = study._json(root / "smoke_plan.json")
        authorization = verify(smoke)
        for name in ("model_configuration", "prompt", "source_sha256", "runtime", "environments"):
            if plan["binding"][name] != smoke["binding"][name]:
                raise ValueError("development configuration differs from successful smoke preparation")
        evidence = _smokes(root, smoke)
        record = study._seal({"schema_version": 1, "id": "development_lock", "study_lock": True,
            "plan_sha256": plan["plan_sha256"], "bindings_hash": plan["bindings_hash"],
            "protocol_hash": plan["binding"]["protocol_hash"], "canonical_root": str(root.resolve()),
            "provider_sha256": authorization["provider_sha256"], "extra_usage_disabled": True,
            "billing_confirmation_source": "inherited_operator_attestation", "legacy_integration_starts": 0,
            "smoke_plan_sha256": smoke["plan_sha256"], "evidence": evidence,
            "recorded_at_utc": datetime.now(UTC).isoformat()}, "record_sha256")
        study._publish(root / "development_lock.json", record)
        _append_binding(record)
        ledger.register(plan, root / "ledger", authorization=root)
        return record
