import copy
from datetime import UTC, datetime
from pathlib import Path

from benchtop import calibrate, primary, runner, study
from benchtop import session_ledger as ledger

EXPORTER_ID = "primary_calibration_v1"
COLLECTION = "development_collection.json"


def check_open(directory, plan):
    from benchtop.pilots import check_open as check_pilots_open

    check_pilots_open(directory, plan)
    path = Path(directory).parent / COLLECTION
    if path.exists():
        closed = study._json(path)
        study._verify(closed, "snapshot_sha256")
        if closed.get("exporter_id") != EXPORTER_ID or closed.get("input", {}).get("closed") is not True:
            raise ValueError("invalid development collection record; preserve it for investigation")
        if plan["binding"]["phase"] == "development":
            if closed["plan_sha256"] != plan["plan_sha256"]:
                raise ValueError("closed collection belongs to another development plan")
            raise ValueError("development collection is closed; new starts and outcome changes are forbidden")


def _inventory(directory, manifest, plan, state):
    files = {"ledger.json", f"plans/{plan['plan_sha256']}.json"}
    for slot in plan["slots"]:
        history = state["histories"][(plan["plan_sha256"], slot["slot_id"])]
        for index, entry in enumerate(history):
            for kind in ("start", "finish", "resolve"):
                if kind in entry:
                    files.add(f"attempts/{entry[kind]['attempt_id']}.{kind}.json")
            final = entry["effective"]
            if final is None:
                continue
            attempt_id = final["attempt_id"]
            work = directory / "runner" / attempt_id
            if not (work / "evidence.json").is_file():
                raise ValueError("finished primary attempts lack replay-verifiable provider evidence; preserve the journal for reconciliation")
            completion = runner._validated_evidence(manifest, plan, slot, attempt_id, work)
            if (runner._digest(work / "evidence.json") != final["evidence_sha256"]
                    or ledger._normalized(slot, index, completion["outcome"]) != final["claimed_outcome"]
                    or any(final.get(key) != value for key, value in completion.items() if key != "outcome")):
                raise ValueError("journal outcome differs from its replay-verified provider evidence")
            for name in (*runner.FILES, "evidence.json", "mcp.json", "settings.json"):
                if (work / name).is_file():
                    files.add(f"runner/{attempt_id}/{name}")
    return {name: runner._digest(directory / name) for name in sorted(files)}


def _slots(plan, state):
    return [{"group_id": slot["group_id"], "slot_id": slot["slot_id"], "lab_id": slot["env_id"],
             "budget_fraction": slot["budget_fraction"], "replicate_index": slot["replicate_index"],
             "attempts": ledger._normalized_history(state["histories"][(plan["plan_sha256"], slot["slot_id"])])}
            for slot in plan["slots"]]


def _input(plan, state, *, closed):
    return {"schema_version": 1, "protocol_hash": plan["binding"]["protocol_hash"],
            "evidence_kind": "operator_supplied", "phase": "development", "closed": closed,
            "lab_ids": [env["env_id"] for env in plan["binding"]["environments"]], "slots": _slots(plan, state)}


def _context(directory, manifest, plan_sha256):
    ledger._directory(directory, "primary")
    if manifest["identity"]["scope"] != "primary":
        raise ValueError("calibration requires a canonical primary journal, not engineering records")
    ledger._current(manifest)
    state = ledger._read(directory, manifest)
    if plan_sha256 not in state["plans"]:
        raise ValueError("calibration plan is not registered")
    plan = state["plans"][plan_sha256]
    if plan["binding"]["phase"] != "development":
        raise ValueError("only development outcomes can select the global budget; smoke and fresh phases are excluded")
    lock = primary.verify(plan, live=False)
    return state, plan, lock


def _closed(directory, manifest, state, plan, lock):
    frozen = study._json(directory.parent / COLLECTION)
    study._verify(frozen, "snapshot_sha256")
    if (frozen.get("exporter_id") != EXPORTER_ID or frozen.get("ledger_id") != manifest["ledger_id"]
            or frozen.get("schema_version") != 1 or frozen.get("groups") != plan["groups"]
            or frozen.get("plan_sha256") != plan["plan_sha256"] or frozen.get("study_lock") is not False
            or frozen.get("development_lock_sha256") != lock["record_sha256"]
            or frozen.get("input") != _input(plan, state, closed=True)
            or frozen.get("phase_evidence") != _inventory(directory, manifest, plan, state)):
        raise ValueError("closed development evidence differs from its canonical collection")
    expected = calibrate.select_budget(frozen["input"])
    if frozen.get("selection") != expected:
        raise ValueError("closed development selection differs from the bound selector")
    return frozen


def load_closed(*, ledger_context=None):
    directory = primary.canonical_root() / "ledger"
    frozen = study._json(directory.parent / COLLECTION)
    calibrate._identifier(frozen["plan_sha256"])
    if ledger_context is not None:
        root, manifest = ledger_context
        if Path(root).resolve() != directory.resolve() or manifest != ledger._manifest(directory):
            raise ValueError("closed development context differs from the canonical journal")
        state, plan, lock = _context(root, manifest, frozen["plan_sha256"])
        return _closed(root, manifest, state, plan, lock)
    with ledger._locked(directory) as (root, manifest):
        state, plan, lock = _context(root, manifest, frozen["plan_sha256"])
        return _closed(root, manifest, state, plan, lock)


def export_development(plan_sha256, output, *, close=False):
    calibrate._identifier(plan_sha256)
    if type(close) is not bool:
        raise ValueError("close must be an explicit boolean")
    output = ledger._writable(output)
    directory = primary.canonical_root() / "ledger"
    if output.is_relative_to(directory.parent.resolve()):
        raise ValueError("write calibration exports outside the canonical study directory")
    if output.exists():
        raise FileExistsError(output)
    with ledger._locked(directory) as (root, manifest):
        state, plan, lock = _context(root, manifest, plan_sha256)
        path = root.parent / COLLECTION
        if path.exists():
            result = _closed(root, manifest, state, plan, lock)
        else:
            if state["halted"]:
                raise ValueError("journal halted after a model or provider contract violation; no budget selection is permitted")
            if close and state["counts"]["active"]:
                raise ValueError("reconcile all in-flight attempts before closing development collection")
            evidence = _inventory(root, manifest, plan, state)
            document = _input(plan, state, closed=close)
            all_slots = [{"attempts": ledger._normalized_history(history)} for history in state["histories"].values()]
            costs = calibrate._costs(all_slots)
            costs["recorded_start_intents"] = costs.pop("episode_starts")
            costs["retry_start_intents"] = costs.pop("retry_episode_starts")
            result = study._seal({"schema_version": 1, "exporter_id": EXPORTER_ID,
                "ledger_id": manifest["ledger_id"], "plan_sha256": plan_sha256,
                "development_lock_sha256": lock["record_sha256"], "groups": copy.deepcopy(plan["groups"]),
                "input": document, "selection": calibrate.select_budget(document), "phase_evidence": evidence,
                "accounting": {"shared_ledger_counts_at_snapshot": dict(state["counts"]),
                               "shared_ledger_costs_at_snapshot": costs, "smoke_included_in_selection": False,
                               "interpretation": "Recorded intents conservatively count starts; costs are adapter reports, not authenticated billing."},
                "recorded_at_utc": datetime.now(UTC).isoformat(), "study_lock": False,
                "model_sessions_started_by_export": 0,
                "scope": "Offline calibration from current canonical development records with replay-verified local provider/MCP evidence. No inference, independent regrading, authenticated measurements, candidate lock or fresh-phase authorization."},
                "snapshot_sha256")
            if close:
                study._publish(path, result)
        output.parent.mkdir(parents=True, exist_ok=True)
        study._publish(output, result)
        return result
