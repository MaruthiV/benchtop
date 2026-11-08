import copy
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from benchtop import calibrate, confirmation_data, pilot_calibration, primary, study, study_calibration
from benchtop import session_ledger as ledger
from benchtop.candidate_metadata import timestamp
from benchtop.generation_journal import read as generation_state

EXPORTER_ID = "primary_confirmation_collection_v1"
COLLECTION = "confirmation_collection.json"


def check_open(directory, plan):
    if plan["binding"]["phase"] == "confirmation" and (directory.parent / COLLECTION).exists():
        closed = study._json(directory.parent / COLLECTION)
        study._verify(closed, "snapshot_sha256")
        if closed.get("exporter_id") != EXPORTER_ID or closed.get("closed") is not True:
            raise ValueError("invalid confirmation collection record; preserve it for investigation")
        raise ValueError("confirmation collection is closed; new starts and outcome changes are forbidden")


def _context(directory, manifest, plan_sha256):
    ledger._directory(directory, "primary")
    ledger._current(manifest)
    state = ledger._read(directory, manifest)
    if manifest["identity"]["scope"] != "primary" or plan_sha256 not in state["plans"]:
        raise ValueError("confirmation collection requires a registered canonical primary plan")
    plan = state["plans"][plan_sha256]
    if plan["binding"]["phase"] != "confirmation":
        raise ValueError("only confirmation outcomes belong in this collection")
    authorization = primary.verify(plan, live=False, ledger_context=(directory, manifest))
    assessment = confirmation_data.load_assessment(ledger_context=(directory, manifest))
    return state, plan, authorization, assessment


def _slots(directory, plan, state):
    rows = []
    for slot in plan["slots"]:
        history = state["histories"][(plan["plan_sha256"], slot["slot_id"])]
        details = []
        for entry in history:
            final = entry["effective"]
            detail = {"attempt_id": entry["start"]["attempt_id"], "answer": None, "confidence": None, "observed_trace": []}
            if final is not None:
                sealed = study._json(directory / "runner" / detail["attempt_id"] / "sealed.json")
                submission = next((row for row in sealed["summary"]["log"] if row["action"] == "submit"), None)
                if submission is not None:
                    detail.update(answer=submission["answer"], confidence=submission["confidence"])
                detail["observed_trace"] = sealed["trace"]["events"]
            details.append(detail)
        rows.append({**copy.deepcopy(slot), "attempts": ledger._normalized_history(history), "details": details})
    return rows


def _reported(values):
    values = [value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None for value in values]
    present = [value for value in values if value is not None]
    known = sum(present) if all(type(value) is int for value in present) else math.fsum(present)
    missing = sum(value is None for value in values)
    return {"known_sum": known, "unreported_items": missing, "total": None if missing else known}


def _generation():
    state = generation_state()
    calls = list(state["calls"].values())
    fields = {"model_wall_seconds": _reported([entry["finish"]["result"]["process_wall_seconds"] if entry["finish"] else None for entry in calls])}
    for name, key in (("input_tokens", "inputTokens"), ("output_tokens", "outputTokens")):
        values = []
        for entry in calls:
            usage = entry["finish"]["result"]["model_usage"] if entry["finish"] else None
            parts = [row.get(key) if isinstance(row, dict) else None for row in usage.values()] if isinstance(usage, dict) and usage else [None]
            values.append(sum(parts) if all(type(value) is int and value >= 0 for value in parts) else None)
        fields[name] = _reported(values)
    return {"recorded_invocation_intents": state["starts"], "active": state["active"], "halted": state["halted"], "fields": fields}


def _accounting(state, at):
    counts = Counter({name: 0 for name in ("total", "active", "primary", "retry", "smoke", "integration")})
    histories = {}
    for key, history in state["histories"].items():
        kept = []
        for entry in history:
            start = entry["start"]
            if timestamp(start["recorded_at_utc"]) > at:
                continue
            final = next((entry[kind] for kind in ("resolve", "finish") if kind in entry and timestamp(entry[kind]["recorded_at_utc"]) <= at), None)
            kept.append({"start": start, "effective": final})
            counts["total"] += 1
            counts[start["bucket"]] += 1
            counts["active"] += final is None
            if start["attempt_index"] == 0:
                counts["phase:" + start["phase"]] += 1
        histories[key] = kept
    return pilot_calibration._accounting({"counts": counts, "histories": histories})


def _core(directory, manifest, state, plan, authorization, assessment, *, closed):
    evidence = study_calibration._inventory(directory, manifest, plan, state)
    pilots = pilot_calibration.load_closed(ledger_context=(directory, manifest)) if plan["slots"] else None
    cohort = study._json(directory.parent / "cohort_lock.json")
    slots = _slots(directory, plan, state)
    costs = calibrate._costs(slots)
    costs["source"] = "replay_verified_primary_adapter_reports"
    return {"schema_version": 1, "exporter_id": EXPORTER_ID, "closed": closed, "ledger_id": manifest["ledger_id"],
        "protocol_hash": plan["binding"]["protocol_hash"], "plan_sha256": plan["plan_sha256"], "bindings_hash": plan["bindings_hash"],
        "confirmation_authorization_sha256": authorization["record_sha256"], "confirmation_lock_sha256": assessment["confirmation_lock_sha256"],
        "assessment_sha256": assessment["assessment_sha256"], "groups": copy.deepcopy(plan["groups"]), "slots": slots,
        "members": [{"env_id": member["env_id"], "stability": member["status"], "gradable": member["gradable"],
                     "member_assessment_sha256": member["record_sha256"], "confirmation_cpu_work": {key: member[key] for key in (
                         "recorded_job_intents", "observed_results", "failed_results", "reported_job_seconds", "collection_wall_seconds", "actual_environment_cpu_seconds")}}
                    for member in assessment["members"]],
        "pilot_decisions": copy.deepcopy(plan["binding"]["pilot_decisions"]), "global_budget_fraction": plan["binding"]["global_budget_fraction"],
        "candidate_funnel": [{key: row[key] for key in ("id", "version", "status", "basis")} for row in cohort["binding"]["funnel"]],
        "phase_evidence": evidence, "confirmation_costs": costs, "pilot_costs": pilots["pilot_costs"] if pilots else calibrate._costs([]),
        "pilot_snapshot_sha256": pilots["snapshot_sha256"] if pilots else None, "generation_and_repair_usage": _generation(),
        "halted": any(entry["effective"] and (entry["effective"]["model_mismatch"] or entry["effective"].get("integrity_failure", False))
                      for slot in plan["slots"] for entry in state["histories"][(plan["plan_sha256"], slot["slot_id"])]),
        "study_lock": False, "model_sessions_started_by_export": 0,
        "scope": "Private replay-verified confirmation evidence. Full cohort and all attempts retained; no inference or final analysis in this collection."}


def _closed(directory, manifest, state, plan, authorization, assessment):
    frozen = study._json(directory.parent / COLLECTION)
    study._verify(frozen, "snapshot_sha256")
    expected = _core(directory, manifest, state, plan, authorization, assessment, closed=True)
    if (set(frozen) != {*expected, "recorded_at_utc", "snapshot_sha256", "accounting"}
            or any(frozen.get(key) != value for key, value in expected.items())):
        raise ValueError("closed confirmation collection differs from its canonical evidence")
    at = timestamp(frozen["recorded_at_utc"])
    if at < timestamp(authorization["recorded_at_utc"]) or frozen["accounting"] != _accounting(state, at):
        raise ValueError("closed confirmation accounting or authorization chronology changed")
    for slot in plan["slots"]:
        for entry in state["histories"][(plan["plan_sha256"], slot["slot_id"])]:
            if any(timestamp(entry[kind]["recorded_at_utc"]) > at for kind in ("start", "finish", "resolve") if kind in entry):
                raise ValueError("confirmation observation postdates collection closure")
    counts = frozen["accounting"]["shared_ledger_counts_at_snapshot"]
    costs = frozen["accounting"]["shared_ledger_costs_at_snapshot"]
    if (counts.get("active") != 0 or counts.get("total", -1) < expected["confirmation_costs"]["episode_starts"]
            or counts.get("phase:confirmation", 0) != expected["confirmation_costs"]["episode_starts"] - expected["confirmation_costs"]["retry_episode_starts"]
            or costs.get("recorded_start_intents") != counts.get("total") or costs.get("retry_start_intents") != counts.get("retry")):
        raise ValueError("closed confirmation accounting is inconsistent")
    return frozen


def load_closed(*, ledger_context=None):
    directory = primary.canonical_root() / "ledger"
    frozen = study._json(directory.parent / COLLECTION)
    calibrate._identifier(frozen["plan_sha256"])
    if ledger_context is None:
        with ledger._locked(directory) as context:
            return load_closed(ledger_context=context)
    root, manifest = ledger_context
    if Path(root).resolve() != directory.resolve() or manifest != ledger._manifest(directory):
        raise ValueError("closed confirmation context differs from the canonical journal")
    return _closed(root, manifest, *_context(root, manifest, frozen["plan_sha256"]))


def collect(plan_sha256, output, *, close=False):
    calibrate._identifier(plan_sha256)
    if type(close) is not bool:
        raise ValueError("close must be an explicit boolean")
    output = ledger._writable(output)
    directory = primary.canonical_root() / "ledger"
    if output.is_relative_to(directory.parent.resolve()):
        raise ValueError("write confirmation exports outside the canonical study directory")
    if output.exists():
        raise FileExistsError(output)
    with ledger._locked(directory) as (root, manifest):
        context = _context(root, manifest, plan_sha256)
        state, plan, authorization, assessment = context
        output.parent.mkdir(parents=True, exist_ok=True)
        if (root.parent / COLLECTION).exists():
            result = _closed(root, manifest, *context)
        else:
            if close and (state["counts"]["active"] or generation_state()["active"]):
                raise ValueError("reconcile all in-flight evaluation/generation before closing confirmation collection")
            at = datetime.now(UTC).isoformat()
            result = study._seal({**_core(root, manifest, *context, closed=close),
                "accounting": _accounting(state, timestamp(at)), "recorded_at_utc": at}, "snapshot_sha256")
            if close:
                study._publish(root.parent / COLLECTION, result)
        output.parent.mkdir(parents=True, exist_ok=True)
        study._publish(output, result)
        return result
