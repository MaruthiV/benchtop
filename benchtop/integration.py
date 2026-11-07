"""Prepare and record operator-assisted MCP connectivity checks, separate from the study."""

import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from benchtop.access import redact
from benchtop.agents.common import Client, LabClient
from benchtop.env import ROOT, STATE
from benchtop.space import canon

TOOLS = {"describe_lab", "run_experiments", "submit_answer"}
DEFAULT_ENV = "2610-36983-advancement-or-acceleration-how"
DEFAULT_TASK = "where:gap_deg"
PROMPT = (
    "Use only the Benchtop MCP tools for this investigation. Call describe_lab first, choose experiments "
    "with run_experiments within the reported budget, then call submit_answer once with your answer "
    "and confidence. Do not use shell, files, browser, other connectors, or parallel agents. "
    "Stop after the submission receipt; the operator will read the grade separately."
)


def _now():
    return datetime.now(UTC).isoformat()


def _private_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
        temporary = Path(out.name)
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bundle_path(root, episode):
    if not re.fullmatch(r"[0-9a-f]{12}", episode):
        raise ValueError("invalid episode ID")
    return Path(root) / "episodes" / f"{episode}.json"


def _load(root, episode):
    path = _bundle_path(root, episode)
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1 or value.get("episode") != episode:
        raise ValueError("invalid integration connection file")
    return path, value


def _protocol():
    value = json.loads((ROOT / "benchmarks/release_v1/spec.json").read_text())
    return hashlib.sha256(canon(value).encode()).hexdigest(), value["execution_limits"]["reserve"]["integration"]


def preflight(url, episode, *, lab=None):
    owned = lab is None
    lab = lab or LabClient(url)
    lab.url = url
    try:
        initialized = lab.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                             "clientInfo": {"name": "benchtop-preflight", "version": "1"}})
        lab.initialized()
        names = {tool["name"] for tool in lab.rpc("tools/list")["tools"]}
        if names != TOOLS:
            raise ValueError("agent endpoint does not expose exactly the three lab tools")
        description = lab.call("describe_lab")
        if description.get("episode") != episode:
            raise ValueError("MCP endpoint describes a different episode")
        origin = urlsplit(url)
        base = f"{origin.scheme}://{origin.netloc}"
        paths = ["/health", "/episodes", f"/episodes/{episode}/mcp-trace", "/mcp"]
        statuses = {path: lab.http.get(base + path).status_code for path in paths}
        if any(status != 404 for status in statuses.values()):
            raise ValueError("agent origin unexpectedly exposes an administrative or unscoped route")
        return {"status": "passed", "client": "benchtop-preflight", "qualia_verified": False,
                "protocol_version": initialized["protocolVersion"], "server_info": initialized["serverInfo"],
                "tools": sorted(names), "admin_routes_absent": True, "model_sessions": 0}
    finally:
        if owned:
            lab.http.close()


def prepare(client: Client, *, env=DEFAULT_ENV, task=DEFAULT_TASK, root=STATE / "integration", lab=None):
    created = client.create(env, task, seed=777, budget_frac=0.25, show_paper=True)
    episode = created["id"]
    checked = preflight(created["mcp_url"], episode, lab=lab)
    summary = client.summary(episode)
    if summary["calls"] or summary["result"] is not None:
        raise ValueError("preflight unexpectedly consumed an experiment call or submitted")
    versions = {key: summary[key] for key in ("execution_id", "task_version", "grader_id", "charging_id")}
    bundle = {
        "schema_version": 1, "episode": episode, "created_at_utc": _now(), "admin_base": client.base,
        "env": env, "task": task, "versions": versions, "mcp_url": created["mcp_url"],
        "prompt": PROMPT, "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(), "preflight": checked,
        "mcpServers": {"benchtop": {"url": created["mcp_url"], "enabled": True,
                                   "tool_permissions": {tool: "always" for tool in sorted(TOOLS)}}},
    }
    path = _bundle_path(root, episode)
    _private_write(path, bundle)
    return {"episode": episode, "private_connection_file": str(path), "preflight": checked,
            "qualia_verified": False, "model_sessions_started_by_preparation": 0,
            "integration_starts_recorded": len(_starts(root))}


def _starts(root):
    path = Path(root) / "starts.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def begin(client: Client, episode, *, client_name, requested_model, reported_model="not_reported",
          restrictions="not_verified", root=STATE / "integration"):
    import fcntl

    path, bundle = _load(root, episode)
    if bundle["admin_base"] != client.base:
        raise ValueError("integration belongs to another admin server")
    if not requested_model.strip() or not reported_model.strip():
        raise ValueError("record a model label or explicit not_reported value")
    parts = urlsplit(bundle["mcp_url"])
    if client_name == "qualia-cloud" and (parts.scheme != "https" or parts.hostname in ("localhost", "127.0.0.1", "::1")):
        raise ValueError("Qualia Cloud needs a publicly reachable HTTPS agent URL")
    summary = client.summary(episode)
    if summary["calls"] or summary["result"] is not None:
        raise ValueError("start accounting before invoking the client; prepare a new untouched episode")
    protocol_hash, limit = _protocol()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(root / "starts.jsonl", os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a+") as ledger:
        fcntl.flock(ledger, fcntl.LOCK_EX)
        from benchtop.primary import block_legacy

        block_legacy(root)
        ledger.seek(0)
        starts = [json.loads(line) for line in ledger if line.strip()]
        if any(item["protocol_hash"] != protocol_hash for item in starts):
            raise ValueError("protocol changed; reconcile the existing integration ledger")
        if any(item["episode"] == episode for item in starts):
            raise ValueError("this episode already has a recorded client invocation")
        if len(starts) >= limit:
            raise ValueError(f"the protocol's {limit}-session integration reserve is exhausted")
        trace = client.mcp_trace(episode)
        entry = {
            "protocol_hash": protocol_hash, "integration_attempt": len(starts) + 1, "episode": episode,
            "recorded_before_invocation_utc": _now(), "client_reported_by_operator": client_name,
            "requested_model": requested_model, "reported_model": reported_model, "restrictions": restrictions,
            "prompt_sha256": bundle["prompt_sha256"], "versions": bundle["versions"],
            "mcp_events_before_invocation": trace["total"],
        }
        ledger.write(json.dumps(redact(entry), allow_nan=False) + "\n")
        ledger.flush()
        os.fsync(ledger.fileno())
    return {"episode": episode, "integration_attempt": entry["integration_attempt"],
            "sessions_remaining": limit - len(starts) - 1,
            "next_step": "Invoke one Qualia conversation with the private connection and prompt. Do not start a second conversation without a new recorded attempt."}


def collect(client: Client, episode, *, client_observed=False, output=None, root=STATE / "integration"):
    _, bundle = _load(root, episode)
    if bundle["admin_base"] != client.base:
        raise ValueError("integration belongs to another admin server")
    entries = [entry for entry in _starts(root) if entry["episode"] == episode]
    if len(entries) != 1:
        raise ValueError("record the integration start before invoking the client")
    entry = entries[0]
    summary, trace = client.summary(episode), client.mcp_trace(episode)
    if any(summary[key] != value for key, value in bundle["versions"].items()):
        raise ValueError("integration episode versions changed")
    start = entry["mcp_events_before_invocation"]
    events = [event for event in trace["events"] if event["index"] >= start]
    successful = {event["tool"] for event in events if not event["error"]}
    missing = sorted(TOOLS - successful)
    complete_trace = len(events) == trace["total"] - start
    received = any(event["tool"] == "submit_answer" and event["receipt_only"] for event in events)
    mcp_runs = sum(event["tool"] == "run_experiments" and "results" in event["response_keys"] for event in events)
    exclusive_mcp = mcp_runs == summary["calls"] and trace.get("submission_source") == "mcp"
    transport_complete = bool(summary["result"] is not None and summary["calls"] > 0
                              and not missing and complete_trace and received and exclusive_mcp)
    qualia = entry["client_reported_by_operator"] in ("qualia-cloud", "qualia-desktop")
    passed = transport_complete and client_observed and qualia
    result = redact({
        "schema_version": 1, "scope": "integration_connectivity_only", "recorded_at_utc": _now(),
        "status": "passed" if passed else "incomplete", "qualia_verified": passed,
        "client_evidence": {"source": "operator_attestation", "client_observed": client_observed, **entry},
        "transport_complete": transport_complete, "missing_successful_tools": missing,
        "all_experiments_and_final_submission_via_mcp": exclusive_mcp,
        "complete_trace": complete_trace, "mcp_events": events, "env": bundle["env"], "task": bundle["task"],
        "versions": bundle["versions"], "budget": summary["budget"], "spent": summary["spent"],
        "experiment_calls": summary["calls"], "result": summary["result"],
        "observed_runs": [event for event in summary["log"] if event["action"] == "run"],
        "integration_starts_recorded": len(_starts(root)),
        "limits": ["Client identity and model labels are operator-reported, not authenticated by MCP.",
                   "This historical connectivity run is excluded from the release study.",
                   "Start accounting requires the operator to record every UI invocation."],
    })
    target = Path(output) if output is not None else ROOT / "artifacts" / f"integration-{episode}.json"
    target = target.resolve()
    if target.is_relative_to(ROOT) and not target.is_relative_to(ROOT / "artifacts"):
        raise ValueError("write integration evidence under artifacts/ or outside the repository")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
