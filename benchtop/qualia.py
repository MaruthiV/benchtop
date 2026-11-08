import copy
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from benchtop import integration, primary, study
from benchtop import session_ledger as ledger
from benchtop.access import redact
from benchtop.candidate_metadata import timestamp
from benchtop.env import Env
from benchtop.evaluate import corrected_replay

DIRECTORY = "qualia"
AUTHORIZATION = "qualia_authorization.json"
PROMPT = {"system_prefix": study.SYSTEM, "first_message_source": "inherited_frozen_first_message",
          "client_instructions": integration.PROMPT,
          "scope": "Operator-assisted connectivity only. Qualia system prompt and unsupported controls are not assumed identical to the CLI study."}


def _now():
    return datetime.now(UTC).isoformat()


def _root():
    return primary.canonical_root() / DIRECTORY


def _authority():
    smoke = study._json(primary.canonical_root() / "smoke_plan.json")
    return smoke, primary.verify(smoke, live=False)


def make_plan(*, ledger_context=None):
    smoke, authority = _authority()
    binding = smoke["binding"]
    environments = copy.deepcopy(binding["environments"][:2])
    if len(environments) != 2:
        raise ValueError("Qualia accounting requires the original two smoke references")
    core = {"planner_id": study.PLANNER_ID, "mode": "study_preparation", "phase": "integration",
        **{key: copy.deepcopy(binding[key]) for key in ("protocol_hash", "protocol_file_sha256", "source_sha256", "runtime", "model_configuration", "prompt", "limits")},
        "environments": environments, "client_prompt": copy.deepcopy(PROMPT), "smoke_authorization_sha256": authority["record_sha256"],
        "scope": "Qualia connectivity reserve only; never smoke, development selection, fresh calibration or primary model evidence."}
    for member in environments:
        if study._snapshot(study._resolve(member["source"])) != member:
            raise ValueError("Qualia reference differs from the original smoke snapshot")
    digest = study.canonical_hash(core)
    groups, slots = [], []
    for member in environments:
        group = study._group(core, digest, member, 0.25, "qualia_connectivity")
        group["descriptor"]["prompt_template_hash"] = study.canonical_hash(PROMPT)
        group["group_id"] = study.canonical_hash(group["descriptor"])
        groups.append(group)
        slots.append(study._slot(group, 0))
    return study._seal({"schema_version": 1, "binding": core, "bindings_hash": digest, "groups": groups, "slots": slots,
        "execution_ready": False, "model_sessions_started": 0}, "plan_sha256")


def _expected(plan, manifest):
    _, authority = _authority()
    return {"schema_version": 1, "id": "qualia_authorization", "study_lock": False, "plan_sha256": plan["plan_sha256"],
        "bindings_hash": plan["bindings_hash"], "protocol_hash": plan["binding"]["protocol_hash"], "ledger_id": manifest["ledger_id"],
        "canonical_root": str(primary.canonical_root().resolve()), "smoke_authorization_sha256": authority["record_sha256"],
        "planned_slots": len(plan["slots"]), "scope": "Operator-assisted Qualia connectivity, excluded from all scientific selection/comparisons.",
        "model_sessions_started": 0}


def verify(plan, *, live=False, ledger_context=None):
    study.validate_plan(plan, ledger_context=ledger_context)
    root = primary.canonical_root()
    record = study._json(root / AUTHORIZATION)
    study._verify(record, "record_sha256")
    manifest = ledger._manifest(root / "ledger")
    expected = _expected(plan, manifest)
    if (manifest["identity"]["scope"] != "primary" or study._json(root / "qualia_plan.json") != plan
            or set(record) != {*expected, "recorded_at_utc", "record_sha256"}
            or any(record.get(key) != value for key, value in expected.items())
            or timestamp(record["recorded_at_utc"]) < timestamp(study._json(root / "smoke_authorization.json")["recorded_at_utc"])):
        raise ValueError("Qualia authorization differs from the original canonical journal or plan")
    if ledger_context is not None and (Path(ledger_context[0]).resolve() != (root / "ledger").resolve() or ledger_context[1] != manifest):
        raise ValueError("Qualia authorization requires the canonical journal context")
    return record


def authorize():
    plan = make_plan()
    root = primary.canonical_root()
    with primary._guard(root.parent / ".initialization.lock"):
        if (root / AUTHORIZATION).exists():
            record = verify(plan)
        else:
            with ledger._locked(root / "ledger") as context:
                directory, manifest = context
                ledger._current(manifest)
                state = ledger._read(directory, manifest)
                if state["halted"] or manifest["identity"]["scope"] != "primary":
                    raise ValueError("Qualia authorization requires a nonhalted primary journal")
                record = study._seal({**_expected(plan, manifest), "recorded_at_utc": _now()}, "record_sha256")
                if (root / "qualia_plan.json").exists():
                    if study._json(root / "qualia_plan.json") != plan:
                        raise ValueError("partial Qualia authorization has a different saved plan")
                else:
                    study._publish(root / "qualia_plan.json", plan)
                study._publish(root / AUTHORIZATION, record)
        ledger.register(plan, root / "ledger", authorization=root)
    return record


def _registered(slot_id, *, current=True):
    plan = study._json(primary.canonical_root() / "qualia_plan.json")
    if current:
        verify(plan)
    else:
        study._verify(plan, "plan_sha256")
    slot = next((slot for slot in plan["slots"] if slot["slot_id"] == slot_id), None)
    if slot is None:
        raise ValueError("unknown Qualia integration slot")
    return plan, slot


def create_episode(hub, plan_sha256, slot_id):
    plan, slot = _registered(slot_id)
    if plan_sha256 != plan["plan_sha256"] or hub.live:
        raise ValueError("Qualia primary preparation requires the exact plan and cached server")
    with ledger._locked(primary.canonical_root() / "ledger") as (directory, manifest):
        state = ledger._read(directory, manifest)
        if state["histories"].get((plan_sha256, slot_id)) or state["halted"]:
            raise ValueError("this Qualia slot was already reserved or the journal halted")
    member = next(member for member in plan["binding"]["environments"] if member["env_id"] == slot["env_id"])
    env = _reference(member)
    hub.envs[env.id] = env
    ep = hub.create(env.id, "claim", seed=0, budget_frac=slot["budget_fraction"], show_paper=True,
                    visible_order=member["visible_order"], max_calls=plan["binding"]["limits"]["max_experiment_tool_calls"])
    hub.integrations[ep.id] = {"binding": {"server_instance_id": hub.instance_id, "plan_sha256": plan_sha256, "slot_id": slot_id}, "activation": None}
    return ep


def _reference(member):
    env = Env.load(study._resolve(member["source"]), replay=True)
    return corrected_replay(env) if env.cache_status == "historical" else env


def _versions(plan, slot):
    member = next(member for member in plan["binding"]["environments"] if member["env_id"] == slot["env_id"])
    return {"env": member["env_id"], "task": "claim", "grader_id": member["grader_id"], "execution_id": member["execution_id"],
        "task_version": member["task_version"], "charging_id": member["charging_hash"], "visible_order": member["visible_order"],
        "visible_order_source": "explicit", "show_paper": True, "seed": 0}


def _connection(episode):
    if not isinstance(episode, str) or re.fullmatch(r"[0-9a-f]{12}", episode) is None:
        raise ValueError("invalid Qualia episode ID")
    path = _root() / "connections" / (episode + ".json")
    value = study._json(path)
    study._verify(value, "connection_sha256")
    if value["episode"] != episode:
        raise ValueError("Qualia connection belongs to another episode")
    return path, value


def prepare(client, slot_id, *, lab=None):
    plan, slot = _registered(slot_id)
    with ledger._locked(primary.canonical_root() / "ledger") as (directory, manifest):
        state = ledger._read(directory, manifest)
        if state["histories"][(plan["plan_sha256"], slot_id)] or state["halted"]:
            raise ValueError("this Qualia slot was already reserved or the journal halted")
    response = client.http.post("/episodes", json={"qualia": {"plan_sha256": plan["plan_sha256"], "slot_id": slot_id}})
    response.raise_for_status()
    created = response.json()
    checked = integration.preflight(created["mcp_url"], created["id"], lab=lab)
    summary = client.summary(created["id"])
    binding = summary["qualia"]["binding"]
    versions = _versions(plan, slot)
    env = _reference(next(member for member in plan["binding"]["environments"] if member["env_id"] == slot["env_id"]))
    expected_budget = sum(env.charging_schedule()[0].values()) * len(versions["visible_order"]) * slot["budget_fraction"]
    if (summary["calls"] or summary["result"] is not None or summary["qualia"]["activation"] is not None
            or any(summary.get(key) != value for key, value in versions.items())
            or summary["budget"] != expected_budget
            or binding["plan_sha256"] != plan["plan_sha256"] or binding["slot_id"] != slot_id):
        raise ValueError("Qualia preparation must leave an inactive untouched episode")
    prompt = study.SYSTEM + "\n\n" + study.first_message(created["describe"], created["budget"]) + "\n\n" + integration.PROMPT
    record = study._seal({"schema_version": 1, "episode": created["id"], "admin_base": client.base, "binding": binding,
        "mcp_url": created["mcp_url"], "description": created["describe"], "budget": created["budget"], "expected_budget": expected_budget,
        "versions": versions,
        "prompt": prompt, "prompt_sha256": study.canonical_hash(prompt), "preflight": checked,
        "recorded_at_utc": _now(), "model_sessions_started": 0}, "connection_sha256")
    directory = primary.canonical_root() / "ledger"
    with ledger._locked(directory) as context:
        state = ledger._read(*context)
        verify(plan, ledger_context=context)
        if state["histories"][(plan["plan_sha256"], slot["slot_id"])]:
            raise ValueError("Qualia slot was reserved during preparation; this inactive episode cannot be used")
        path = _root() / "connections" / (record["episode"] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        study._publish(path, record)
    return {"episode": record["episode"], "slot_id": slot_id, "private_connection_file": str(path),
            "qualia_verified": False, "model_sessions_started": 0, "status": "prepared_inactive"}


def validate_request(plan, slot, request):
    if not isinstance(request, dict):
        raise ValueError("Qualia starts require a prepared connection and explicit UI accounting request")
    study._verify(request, "request_sha256")
    _, connection = _connection(request["episode"])
    if (set(request) != {"episode", "connection_sha256", "client", "selected_model_label", "reported_model_id", "restrictions", "no_new_spend", "request_sha256"}
            or request["connection_sha256"] != connection["connection_sha256"]
            or connection["binding"]["plan_sha256"] != plan["plan_sha256"] or connection["binding"]["slot_id"] != slot["slot_id"]
            or request["client"] not in ("qualia-cloud", "qualia-desktop") or request["no_new_spend"] is not True):
        raise ValueError("Qualia request differs from its prepared connection, slot or zero-new-spend attestation")
    for key in ("selected_model_label", "restrictions"):
        if not isinstance(request[key], str) or not request[key].strip() or len(request[key]) > 2000:
            raise ValueError("record the actual UI model label and supported or unavailable restrictions")
    reported = request["reported_model_id"]
    if reported is not None and (not isinstance(reported, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", reported) is None):
        raise ValueError("record an exact reported model ID or leave it unavailable")
    if reported is not None and reported != plan["binding"]["model_configuration"]["reported_model_id"]:
        raise ValueError("Qualia's reported model differs from the frozen study model; do not invoke it")
    origin = urlsplit(connection["mcp_url"])
    if request["client"] == "qualia-cloud" and (origin.scheme != "https" or origin.hostname in ("127.0.0.1", "localhost", "::1")):
        raise ValueError("Qualia Cloud requires a public HTTPS agent URL")
    return connection


def _client_matches(client, connection):
    if client.base != connection["admin_base"]:
        raise ValueError("Qualia connection belongs to another administrative server")
    summary = client.summary(connection["episode"])
    if summary.get("qualia", {}).get("binding") != connection["binding"] or any(summary.get(key) != value for key, value in connection["versions"].items()):
        raise ValueError("Qualia server restarted or prepared episode identity changed")
    return summary


def begin(client, episode, *, client_name, selected_model_label, reported_model=None, restrictions="not_verified", no_new_spend=False):
    _, connection = _connection(episode)
    plan, slot = _registered(connection["binding"]["slot_id"])
    request = study._seal({"episode": episode, "connection_sha256": connection["connection_sha256"], "client": client_name,
        "selected_model_label": selected_model_label, "reported_model_id": reported_model, "restrictions": restrictions,
        "no_new_spend": no_new_spend}, "request_sha256")
    validate_request(plan, slot, request)
    summary = _client_matches(client, connection)
    if summary["calls"] or summary["result"] is not None or summary["qualia"]["activation"] is not None:
        raise ValueError("Qualia begin requires an untouched inactive episode; never resend a conversation")
    start = ledger.begin(primary.canonical_root() / "ledger", plan["plan_sha256"], slot["slot_id"], integration_request=request)
    activation = activate(client, episode)
    return {**activation, "attempt_id": start["attempt_id"], "next_step": "Send the saved prompt in one Qualia conversation exactly once. Every additional conversation needs a different unused slot."}


def _intent(plan, slot, episode, connection_sha256, *, current):
    directory = primary.canonical_root() / "ledger"
    with ledger._locked(directory) as context:
        root, manifest = context
        if current:
            ledger._current(manifest)
            verify(plan, ledger_context=context)
        state = ledger._read(root, manifest)
        if current:
            from benchtop.generation_journal import read as generation_state

            if generation_state()["halted"]:
                raise ValueError("generation halted after a model/provider contract violation")
        history = state["histories"][(plan["plan_sha256"], slot["slot_id"])]
        if len(history) != 1 or history[0]["effective"] is not None or state["halted"]:
            raise ValueError("Qualia requires one active canonical start without a model/tool halt")
        start = history[0]["start"]
        if start["integration_request"]["episode"] != episode or start["integration_request"]["connection_sha256"] != connection_sha256:
            raise ValueError("Qualia start belongs to another prepared episode")
        return start


def arm(hub, ep):
    guard = hub.integrations.get(ep.id)
    if guard is None or ep.id in hub.sealed:
        raise ValueError("only an unsealed prepared Qualia episode can be activated")
    _, connection = _connection(ep.id)
    plan, slot = _registered(guard["binding"]["slot_id"])
    if guard["binding"] != connection["binding"] or guard["binding"]["server_instance_id"] != hub.instance_id:
        raise ValueError("Qualia connection or server instance changed")
    start = _intent(plan, slot, ep.id, connection["connection_sha256"], current=True)
    if guard["activation"] is None:
        if ep.calls or ep.result is not None:
            raise ValueError("Qualia activation requires an untouched episode")
        guard["activation"] = {"attempt_id": start["attempt_id"], "start_sha256": start["record_sha256"],
            "connection_sha256": connection["connection_sha256"], "trace_boundary": hub.mcp_trace(ep)["total"], "activated_at_utc": _now()}
        guard["deadline"] = time.monotonic() + plan["binding"]["limits"]["max_episode_seconds"]
    return copy.deepcopy(guard["activation"])


def check_lease(hub, ep):
    guard = hub.integrations.get(ep.id)
    if guard is None:
        return
    if guard["activation"] is None or time.monotonic() >= guard["deadline"]:
        raise ValueError("Qualia tools are inactive or the reserved episode timed out; retain the start for reconciliation")
    plan, slot = _registered(guard["binding"]["slot_id"])
    _intent(plan, slot, ep.id, guard["activation"]["connection_sha256"], current=True)


def activate(client, episode):
    _, connection = _connection(episode)
    _client_matches(client, connection)
    response = client.http.post(f"/episodes/{episode}/qualia-activate")
    response.raise_for_status()
    value = response.json()
    return {"episode": episode, "attempt_id": value["activation"]["attempt_id"], "status": "reserved_active",
            "model_sessions_started_by_helper": 0, "instruction": "This acknowledges the same reserved intent; never resend a Qualia conversation."}


def _attempt(episode):
    path, connection = _connection(episode)
    directory = primary.canonical_root() / "ledger"
    with ledger._locked(directory) as (root, manifest):
        state = ledger._read(root, manifest)
        plan_sha256, slot_id = (connection["binding"][key] for key in ("plan_sha256", "slot_id"))
        history = state["histories"].get((plan_sha256, slot_id), [])
        if len(history) != 1 or history[0]["start"]["integration_request"]["episode"] != episode:
            raise ValueError("Qualia episode has no unique recorded canonical invocation")
        return manifest, state["plans"][plan_sha256], state["slots"][(plan_sha256, slot_id)], path, connection, history[0]


def _capture_path(attempt_id):
    return _root() / "captures" / (attempt_id + ".json")


def _classify(plan, slot, connection, start, capture):
    sealed = capture["sealed"]
    result, events = None, []
    valid, transport = False, False
    costs = {}
    if sealed is not None:
        summary, trace = sealed["summary"], sealed["trace"]
        guard = summary.get("qualia", {})
        activation = guard.get("activation")
        valid = (sealed.get("sealed") is True and summary.get("episode") == connection["episode"]
            and all(summary.get(key) == value for key, value in _versions(plan, slot).items())
            and guard.get("binding") == connection["binding"] and sealed.get("max_calls") == plan["binding"]["limits"]["max_experiment_tool_calls"]
            and summary.get("budget") == connection["expected_budget"] and activation is not None
            and activation.get("start_sha256") == start["record_sha256"] and activation.get("attempt_id") == start["attempt_id"]
            and activation.get("connection_sha256") == connection["connection_sha256"])
        boundary = activation.get("trace_boundary") if activation else None
        if type(boundary) is int and 0 <= boundary <= trace["total"]:
            events = [event for event in trace["events"] if event["index"] >= boundary]
            complete = [event["index"] for event in events] == list(range(boundary, trace["total"]))
            successful = {event["tool"] for event in events if not event["error"]}
            runs = sum(event["tool"] == "run_experiments" and "results" in event["response_keys"] for event in events)
            receipt = any(event["tool"] == "submit_answer" and event["receipt_only"] for event in events)
            result = summary.get("result")
            transport = bool(valid and complete and successful >= integration.TOOLS and receipt and result is not None
                and runs == summary["calls"] and runs > 0 and trace.get("submission_source") == "mcp")
        costs = {"charged_experiment_cost": summary["spent"], "unique_executed_runs": 0, "actual_environment_cpu_seconds": 0,
                 "requested_runs": sum(len(row["configs"]) * min(row["seeds"], len(summary["visible_order"])) for row in summary["log"] if row["action"] == "run")}
    outcome = {"state": "unresolved", "reward": None, "failure_category": "qualia_transport_incomplete" if sealed is not None else "qualia_server_receipt_unavailable", "costs": costs}
    if transport and type(result.get("score")) in (int, float) and result["score"] in (0, 1):
        outcome = {"state": "completed", "reward": result["score"], "costs": costs}
    elif sealed is not None and not valid:
        outcome["failure_category"] = "qualia_episode_binding_mismatch"
    observed = capture["client_observed"]
    reported = capture["reported_model_id"]
    matches = reported is None or reported == plan["binding"]["model_configuration"]["reported_model_id"]
    return {"outcome": outcome, "reported_model_id": reported, "submission_status": "committed" if result is not None else "not_committed" if valid else "unknown",
        "process_stopped": True, "transport_complete": transport, "qualia_verified": bool(observed and transport and matches), "events": events}


def _validated(episode):
    manifest, plan, slot, path, connection, entry = _attempt(episode)
    start = entry["start"]
    capture = study._json(_capture_path(start["attempt_id"]))
    study._verify(capture, "artifact_sha256")
    authority = study._json(primary.canonical_root() / AUTHORIZATION)
    study._verify(authority, "record_sha256")
    if (capture["start_sha256"] != start["record_sha256"] or capture["connection_file_sha256"] != study.cache.file_hash(path)
            or capture["authorization_sha256"] != authority["record_sha256"] or authority["plan_sha256"] != plan["plan_sha256"]
            or authority["ledger_id"] != manifest["ledger_id"] or authority["canonical_root"] != str(primary.canonical_root().resolve())
            or capture["client_stopped"] is not True or type(capture["client_observed"]) is not bool
            or timestamp(capture["recorded_at_utc"]) < timestamp(start["recorded_at_utc"])):
        raise ValueError("Qualia capture, stop attestation or original journal binding changed")
    expected = _classify(plan, slot, connection, start, capture)
    if capture["completion"] != expected:
        raise ValueError("Qualia capture classification differs from its sealed server evidence")
    if entry["effective"] is not None:
        final = entry["effective"]
        if (final["evidence_sha256"] != study.cache.file_hash(_capture_path(start["attempt_id"]))
                or final["claimed_outcome"] != ledger._normalized(slot, 0, expected["outcome"])
                or any(final[key] != expected[key] for key in ("reported_model_id", "submission_status", "process_stopped"))):
            raise ValueError("Qualia journal completion differs from its retained capture")
    return plan, slot, start, entry, capture, expected


def resume(episode):
    _, _, start, _, _, completion = _validated(episode)
    fields = {key: completion[key] for key in ("reported_model_id", "submission_status", "process_stopped")}
    return ledger.finish(primary.canonical_root() / "ledger", start["attempt_id"], completion["outcome"],
        evidence_sha256=study.cache.file_hash(_capture_path(start["attempt_id"])), **fields)


def _export(episode, output=None):
    _, _, start, entry, capture, completion = _validated(episode)
    final = entry["effective"]
    verified = final is not None and completion["qualia_verified"]
    result = redact({"schema_version": 1, "scope": "qualia_connectivity_only_excluded_from_scientific_results", "episode": episode,
        "attempt_id": start["attempt_id"], "plan_sha256": start["plan_sha256"], "qualia_verified": verified,
        "status": "passed" if verified else "incomplete" if final else "capture_pending_journal",
        "outcome": final["outcome"] if final else None, "transport_complete": completion["transport_complete"],
        "client_identity_source": "operator_attestation_not_authenticated", "client_observed": capture["client_observed"],
        "client_stopped": True, "stop_source": "operator_attestation_not_process_inspection", "reported_model_id": completion["reported_model_id"],
        "reported_model_source": "operator_observation_not_authenticated" if completion["reported_model_id"] else "unavailable",
        "mcp_events": completion["events"], "model_sessions_started_by_helper": 0,
        "limits": ["The helper records conservative intents; it cannot count Qualia conversations or provider sessions launched outside this workflow.",
                   "UI model identity, restrictions, zero-new-spend eligibility and stopped-client state are operator reports, not provider authentication.",
                   "Cached connectivity grades never enter scientific selection or primary comparisons."]})
    if output is not None:
        target = ledger._writable(output)
        if target.is_relative_to(primary.canonical_root().resolve()):
            raise ValueError("write Qualia exports outside canonical study state")
        target.parent.mkdir(parents=True, exist_ok=True)
        study._publish(target, result)
    return result


def collect(client, episode, *, client_observed=False, client_stopped=False, reported_model=None, output=None):
    if client_stopped is not True:
        raise ValueError("confirm the Qualia conversation stopped before releasing its reserved session")
    if _capture_for_episode_exists(episode):
        resume(episode)
        return _export(episode, output)
    _, plan, slot, path, connection, entry = _attempt(episode)
    if entry["effective"] is not None:
        raise ValueError("Qualia completion lacks its original retained capture; preserve the journal")
    _client_matches(client, connection)
    response = client.http.post(f"/episodes/{episode}/seal")
    response.raise_for_status()
    return _save_capture(plan, slot, path, connection, entry["start"], response.json(), client_observed, reported_model, output)


def _capture_for_episode_exists(episode):
    return _capture_path(_attempt(episode)[-1]["start"]["attempt_id"]).exists()


def _save_capture(plan, slot, path, connection, start, sealed, observed, reported, output, *, reason=None):
    if type(observed) is not bool or reported is not None and (not isinstance(reported, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", reported) is None):
        raise ValueError("record a factual UI observation and exact reported model ID or leave it unavailable")
    initial = start["integration_request"]["reported_model_id"]
    if initial is not None and reported is None:
        reported = initial
    authority = study._json(primary.canonical_root() / AUTHORIZATION)
    capture = {"start_sha256": start["record_sha256"], "connection_file_sha256": study.cache.file_hash(path),
        "authorization_sha256": authority["record_sha256"], "sealed": sealed, "client_observed": observed, "client_stopped": True,
        "reported_model_id": reported, "abandonment_reason": reason, "recorded_at_utc": _now()}
    capture["completion"] = _classify(plan, slot, connection, start, capture)
    target = _capture_path(start["attempt_id"])
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    study._publish(target, study._seal(capture, "artifact_sha256"))
    resume(connection["episode"])
    return _export(connection["episode"], output)


def abandon(episode, *, client_stopped=False, reason, output=None):
    if client_stopped is not True or not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
        raise ValueError("explicitly attest the client stopped and record why server evidence is unavailable")
    if _capture_for_episode_exists(episode):
        resume(episode)
        return _export(episode, output)
    _, plan, slot, path, connection, entry = _attempt(episode)
    return _save_capture(plan, slot, path, connection, entry["start"], None, False, None, output, reason=reason)


def inspect():
    plan = study._json(primary.canonical_root() / "qualia_plan.json")
    verify(plan)
    state = ledger.inspect(primary.canonical_root() / "ledger")
    rows = [row for row in state["slots"] if row["plan_sha256"] == plan["plan_sha256"]]
    return {"status": "authorized", "plan_sha256": plan["plan_sha256"], "slots": rows, "counts": state["counts"],
        "combined_active_model_intents": state["combined_active_model_intents"], "model_sessions_started": 0,
        "qualia_verified": False, "interpretation": "Accounting inspection alone never establishes an observed Qualia pass."}
