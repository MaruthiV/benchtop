import copy
from datetime import UTC, datetime
from pathlib import Path

from benchtop import calibrate, primary, study, study_calibration
from benchtop import session_ledger as ledger
from benchtop.candidate_metadata import timestamp

EXPORTER_ID = "primary_pilot_calibration_v1"
COLLECTION = "pilot_collection.json"


def _context(directory, manifest, plan_sha256):
    ledger._directory(directory, "primary")
    if manifest["identity"]["scope"] != "primary":
        raise ValueError("pilot calibration requires the canonical primary journal")
    ledger._current(manifest)
    state = ledger._read(directory, manifest)
    if plan_sha256 not in state["plans"]:
        raise ValueError("pilot calibration plan is not registered")
    plan = state["plans"][plan_sha256]
    if plan["binding"]["phase"] != "fresh_pilot":
        raise ValueError("only fresh pilot slots can calibrate a cohort member")
    authorization = primary.verify(plan, live=False, ledger_context=(directory, manifest))
    return state, plan, authorization


def _labs(plan, state, *, closed):
    slots = study_calibration._slots(plan, state)
    labs = []
    for member in plan["binding"]["environments"]:
        document = {"schema_version": 1, "protocol_hash": plan["binding"]["protocol_hash"], "evidence_kind": "operator_supplied",
                    "phase": "fresh_pilot", "closed": closed, "lab_ids": [member["env_id"]],
                    "global_budget_fraction": plan["binding"]["global_budget_fraction"],
                    "slots": [slot for slot in slots if slot["lab_id"] == member["env_id"]]}
        labs.append({"env_id": member["env_id"], "input": document, "selection": calibrate.select_budget(document)})
    return labs


def _core(directory, manifest, state, plan, authorization, *, closed):
    evidence = study_calibration._inventory(directory, manifest, plan, state)
    return {"schema_version": 1, "exporter_id": EXPORTER_ID, "ledger_id": manifest["ledger_id"],
            "protocol_hash": plan["binding"]["protocol_hash"], "plan_sha256": plan["plan_sha256"], "bindings_hash": plan["bindings_hash"],
            "pilot_authorization_sha256": authorization["record_sha256"], "cohort_lock_sha256": plan["binding"]["cohort_lock_sha256"],
            "global_budget_fraction": plan["binding"]["global_budget_fraction"], "cohort_size": plan["binding"]["cohort_size"],
            "closed": closed, "groups": copy.deepcopy(plan["groups"]), "labs": _labs(plan, state, closed=closed),
            "phase_evidence": evidence,
            "pilot_costs": calibrate._costs(study_calibration._slots(plan, state)),
            "selection_scope": "Each member's fresh pilots only; smoke, development and confirmation excluded.",
            "study_lock": False, "model_sessions_started_by_export": 0,
            "scope": "Offline per-lab calibration from canonical pilot records and replay-verified provider/MCP evidence. No inference, independent answer regrading, authenticated billing, confirmation lock or execution authorization."}


def _accounting(state):
    from benchtop import continuation

    slots = [{"attempts": ledger._normalized_history(history)} for history in state["histories"].values()]
    costs = calibrate._costs(slots)
    costs["recorded_start_intents"] = costs.pop("episode_starts")
    costs["retry_start_intents"] = costs.pop("retry_episode_starts")
    prior = continuation.usage()
    return {"shared_ledger_counts_at_snapshot": dict(state["counts"]), "shared_ledger_costs_at_snapshot": costs,
            "interpretation": "Frozen snapshot of recorded intents and adapter-reported costs, not authenticated provider calls or billing. Generation is accounted separately.",
            **({"prior_protocol_usage": prior} if prior else {})}


def _closed(directory, manifest, state, plan, authorization):
    frozen = study._json(directory.parent / COLLECTION)
    study._verify(frozen, "snapshot_sha256")
    expected = _core(directory, manifest, state, plan, authorization, closed=True)
    if (set(frozen) != set(expected) | {"snapshot_sha256", "recorded_at_utc", "accounting"}
            or any(frozen.get(key) != value for key, value in expected.items())):
        raise ValueError("closed pilot evidence or per-lab selection differs from the canonical collection")
    accounting = frozen["accounting"]
    calibrate._keys(accounting, {"shared_ledger_counts_at_snapshot", "shared_ledger_costs_at_snapshot", "interpretation"}, {"prior_protocol_usage"})
    from benchtop import continuation

    if accounting.get("prior_protocol_usage") != continuation.usage():
        raise ValueError("closed pilot prior-protocol usage changed")
    counts, costs = accounting["shared_ledger_counts_at_snapshot"], accounting["shared_ledger_costs_at_snapshot"]
    if (not isinstance(counts, dict) or not isinstance(costs, dict) or any(type(value) is not int or value < 0 for value in counts.values())
            or counts.get("active") != 0 or counts.get("total", -1) < expected["pilot_costs"]["episode_starts"]
            or counts.get("phase:fresh_pilot", 0) != expected["pilot_costs"]["episode_starts"] - expected["pilot_costs"]["retry_episode_starts"]
            or costs.get("recorded_start_intents") != counts.get("total") or costs.get("retry_start_intents") != counts.get("retry")):
        raise ValueError("closed pilot accounting snapshot is inconsistent")
    at = timestamp(frozen["recorded_at_utc"])
    for slot in plan["slots"]:
        for entry in state["histories"][(plan["plan_sha256"], slot["slot_id"])]:
            if any(timestamp(entry[kind]["recorded_at_utc"]) > at for kind in ("start", "finish", "resolve") if kind in entry):
                raise ValueError("pilot observation postdates collection closure")
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
        raise ValueError("closed pilot context differs from the canonical journal")
    state, plan, authorization = _context(root, manifest, frozen["plan_sha256"])
    return _closed(root, manifest, state, plan, authorization)


def export_pilots(plan_sha256, output, *, close=False):
    calibrate._identifier(plan_sha256)
    if type(close) is not bool:
        raise ValueError("close must be an explicit boolean")
    output = ledger._writable(output)
    directory = primary.canonical_root() / "ledger"
    if output.is_relative_to(directory.parent.resolve()):
        raise ValueError("write pilot exports outside the canonical study directory")
    if output.exists():
        raise FileExistsError(output)
    with ledger._locked(directory) as (root, manifest):
        state, plan, authorization = _context(root, manifest, plan_sha256)
        path = root.parent / COLLECTION
        if path.exists():
            result = _closed(root, manifest, state, plan, authorization)
        else:
            if state["halted"]:
                raise ValueError("journal halted after a model or provider contract violation; no pilot budget selection is permitted")
            if close and state["counts"]["active"]:
                raise ValueError("reconcile all in-flight attempts before closing pilot collection")
            if (root.parent / "confirmation_lock.json").exists():
                raise ValueError("confirmation lock appeared before pilot collection closure")
            result = study._seal({**_core(root, manifest, state, plan, authorization, closed=close),
                "accounting": _accounting(state), "recorded_at_utc": datetime.now(UTC).isoformat()}, "snapshot_sha256")
            output.parent.mkdir(parents=True, exist_ok=True)
            if close:
                study._publish(path, result)
        output.parent.mkdir(parents=True, exist_ok=True)
        study._publish(output, result)
        return result
