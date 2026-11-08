import asyncio
import copy
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path

import uvicorn

from benchtop import session_ledger as ledger
from benchtop import study
from benchtop.env import Env
from benchtop.evaluate import corrected_replay
from benchtop.server import Hub, build_app
from benchtop.stats import LABELS

RUNNER_ID = "scripted_mcp_runner_v1"
PROVIDER_RUNNER_ID = "claude_mcp_runner_v1"
FILES = ("connection.json", "process.json", "sealed.json", "stdout.jsonl", "stderr.txt")
SCENARIOS = ("submit", "no_answer", "quota", "unknown_error", "post_submit_error", "timeout", "post_submit_timeout", "model_change")


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while data := source.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def _private(path, value):
    study._publish(Path(path), value)


def parse_stream(text):
    models, final = [], None
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "system" and event.get("subtype") == "init" and isinstance(event.get("model"), str):
            models.append(event["model"])
        if event.get("type") == "result" and event.get("synthetic_control") is True:
            final = event
    return {"models": list(dict.fromkeys(models)), "final": final}


def _group_stopped(pid):
    try:
        os.killpg(pid, 0)
        return False
    except ProcessLookupError:
        return True
    except PermissionError:
        return False


def execute_script(work, timeout):
    return execute_command(work, timeout, [sys.executable, "-I", str(Path(__file__).with_name("_scripted_agent.py")),
                                          str(work / "connection.json")],
                           executor="scripted_mcp_subprocess", parser=parse_stream)


def execute_command(work, timeout, command, *, executor, parser, env=None):
    files = [os.fdopen(os.open(work / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")
             for name in ("stdout.jsonl", "stderr.txt")]
    process, timed_out = None, False
    started = time.monotonic()
    try:
        process = subprocess.Popen(command, cwd=work, stdin=subprocess.DEVNULL, env=env,
                                   stdout=files[0], stderr=files[1], start_new_session=True)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
    finally:
        if process is not None and process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
        for stream in files:
            stream.close()
    stopped = process is not None and process.returncode is not None and _group_stopped(process.pid)
    with (work / "stdout.jsonl").open("rb") as source:
        stdout = source.read(1024 * 1024).decode(errors="replace")
    return {"executor": executor, "process_stopped": stopped,
            "returncode": process.returncode if process else None, "timed_out": timed_out,
            "process_wall_seconds": time.monotonic() - started,
            "stdout_sha256": _digest(work / "stdout.jsonl"), "stderr_sha256": _digest(work / "stderr.txt"),
            "stdout_truncated": (work / "stdout.jsonl").stat().st_size > 1024 * 1024,
            "stream": parser(stdout), "model_sessions": int(executor == "claude_cli")}


class Controller:
    def __init__(self):
        self.identity = os.urandom(16).hex()
        self.hub = Hub(envs_dir=Path("/nonexistent-engineering-source"), live=False)
        self.loop = None
        self.server = None
        self.thread = None

    def __enter__(self):
        self.socket = socket.socket()
        try:
            self.socket.bind(("127.0.0.1", 0))
            self.socket.listen(128)
        except BaseException:
            self.socket.close()
            raise
        self.hub.agent_base = f"http://127.0.0.1:{self.socket.getsockname()[1]}"
        app, _ = build_app(self.hub)

        async def owned_app(scope, receive, send):
            if scope["type"] == "lifespan":
                self.loop = asyncio.get_running_loop()
            await app(scope, receive, send)

        self.server = uvicorn.Server(uvicorn.Config(owned_app, log_level="error", access_log=False, lifespan="on"))
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [self.socket]}, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 5
        while not self.server.started and self.thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not self.server.started:
            self.__exit__(None, None, None)
            raise RuntimeError("owned engineering MCP server did not start")
        return self

    def __exit__(self, *args):
        if self.server:
            self.server.should_exit = True
        if self.thread:
            self.thread.join(timeout=10)
        self.socket.close()
        if self.thread and self.thread.is_alive():
            raise RuntimeError("owned engineering server has not stopped")

    def prepare(self, plan, slot):
        snapshot = next(env for env in plan["binding"]["environments"] if env["env_id"] == slot["env_id"])
        path = study._resolve(snapshot["source"])
        grading_context = None
        confirmation_authorization = None
        if plan["binding"]["phase"] in ("fresh_pilot", "confirmation"):
            from benchtop.pilots import load_environment

            env = load_environment(snapshot)
            if plan["binding"]["phase"] == "confirmation":
                from benchtop.confirmation_execution import AUTHORIZATION
                from benchtop.confirmation_execution import grading_context as independent_grading
                from benchtop.primary import canonical_root

                grading_context = independent_grading(plan, slot["env_id"])
                confirmation_authorization = study._json(canonical_root() / AUTHORIZATION)["record_sha256"]
        else:
            if study._snapshot(path) != snapshot:
                raise ValueError("episode source differs from bound preparation")
            env = Env.load(path, replay=True)
            if env.cache_status == "historical":
                env = corrected_replay(env)
        prices, _ = env.charging_schedule()
        expected_budget = sum(prices.values()) * len(snapshot["visible_order"]) * slot["budget_fraction"]

        async def prepare():
            self.hub.envs[env.id] = env
            ep = self.hub.create(env.id, "claim", budget_frac=slot["budget_fraction"], seed=0,
                                 visible_order=snapshot["visible_order"], show_paper=True,
                                 grading_context=grading_context,
                                 max_calls=plan["binding"]["limits"]["max_experiment_tool_calls"])
            return {"controller_id": self.identity, "episode": ep.id, "mcp_url": self.hub.agent_url(ep),
                    **({"private_grading": {**grading_context["binding"], "context_sha256": grading_context["context_sha256"]}}
                       if grading_context is not None else {}),
                    **({"grading_context": grading_context, "confirmation_authorization_sha256": confirmation_authorization}
                       if grading_context is not None else {}),
                    "plan_sha256": plan["plan_sha256"], "slot_id": slot["slot_id"], "expected_budget": expected_budget,
                    "description": ep.describe(), "budget": ep.budget_left()}

        return asyncio.run_coroutine_threadsafe(prepare(), self.loop).result(timeout=10)

    def seal(self, connection):
        if connection["controller_id"] != self.identity:
            raise ValueError("connection belongs to another controller")
        ep = self.hub.get(connection["episode"])
        return asyncio.run_coroutine_threadsafe(self.hub.seal(ep), self.loop).result(timeout=10)


def _lookup(directory, attempt_id):
    with ledger._locked(directory) as (root, manifest):
        state = ledger._read(root, manifest)
        for key, history in state["histories"].items():
            for index, entry in enumerate(history):
                if entry["start"]["attempt_id"] == attempt_id:
                    return manifest, state["plans"][key[0]], state["slots"][key], index, entry
    raise ValueError("attempt is not registered")


def _classify(plan, slot, connection, process, sealed, spec):
    snapshot = next(env for env in plan["binding"]["environments"] if env["env_id"] == slot["env_id"])
    summary, trace = sealed["summary"], sealed["trace"]
    expected = {"episode": connection["episode"], "env": slot["env_id"], "task": "claim",
                "grader_id": snapshot["grader_id"], "execution_id": snapshot["execution_id"],
                "task_version": snapshot["task_version"], "charging_id": snapshot["charging_hash"],
                "visible_order": snapshot["visible_order"], "visible_order_source": "explicit", "show_paper": True}
    valid = (sealed.get("sealed") is True and all(summary.get(k) == v for k, v in expected.items())
             and sealed.get("max_calls") == plan["binding"]["limits"]["max_experiment_tool_calls"]
             and summary.get("budget") == connection["expected_budget"])
    result = summary.get("result")
    if plan["binding"]["phase"] == "confirmation":
        from benchtop.confirmation_execution import evidence_context
        from benchtop.tasks import grade

        try:
            context = evidence_context(plan, slot["env_id"], connection.get("confirmation_authorization_sha256"))
        except (KeyError, ValueError, FileNotFoundError):
            context = None
        binding = {**context["binding"], "context_sha256": context["context_sha256"]} if context is not None else None
        valid = (valid and context is not None and connection.get("grading_context") == context
                 and connection.get("private_grading") == summary.get("private_grading") == binding)
        if result is not None and context is not None:
            submissions = [row for row in summary["log"] if row["action"] == "submit"]
            graded = grade(None, context["task"], submissions[0]["answer"]) if len(submissions) == 1 else None
            valid = (valid and result.get("grading_context_sha256") == context["context_sha256"]
                     and graded is not None and result.get("graded") == graded and result.get("score") == graded["score"])
    submission = "committed" if result is not None else "not_committed" if valid else "unknown"
    models = process["stream"]["models"]
    expected_model = plan["binding"]["model_configuration"]["reported_model_id"]
    reported = next((value for value in models if value != expected_model), models[0] if models else None)
    outcome = {"state": "unresolved", "reward": None, "failure_category": "unclassified_process_result"}
    unverified_provider = process["executor"] == "claude_cli" and process["stream"].get("isolation_valid") is not True
    contract_failure = unverified_provider and (process["stream"].get("init_observed") is True
                                                or process["stream"].get("synthetic_control") is True)
    if not valid:
        outcome["failure_category"] = "episode_binding_mismatch"
    elif process["stdout_truncated"]:
        outcome["failure_category"] = "truncated_process_metadata"
    elif contract_failure:
        outcome["failure_category"] = "provider_contract_mismatch"
    elif unverified_provider:
        outcome["failure_category"] = "unverified_provider_metadata"
    elif result is not None:
        private_receipt = any(event["tool"] == "submit_answer" and event.get("receipt_only") is True for event in trace["events"])
        if (trace.get("submission_source") != "mcp" or not private_receipt
                or type(result.get("score")) not in (int, float) or result["score"] not in (0, 1)):
            outcome["failure_category"] = "unverified_submission_source_or_reward"
        else:
            outcome = {"state": "completed", "reward": result["score"]}
    elif process["timed_out"]:
        outcome = {"state": "agent_failure", "reward": 0, "failure_category": "episode_timeout"}
    else:
        final = process["stream"]["final"] or {}
        category = final.get("failure_category")
        allowed = spec["failure_handling"]["infrastructure_failures_missing"]
        if process["executor"] == "scripted_mcp_subprocess" and final.get("is_error") is True and category in allowed:
            outcome = {"state": "technical_failure", "reward": None, "failure_category": category}
        elif final.get("is_error") is False and process["returncode"] == 0:
            outcome = {"state": "agent_failure", "reward": 0, "failure_category": "exited_without_answer"}
    outcome["costs"] = {"charged_experiment_cost": summary["spent"], "unique_executed_runs": 0,
                        "requested_runs": sum(len(row["configs"]) * max(1, min(row["seeds"], len(summary["visible_order"])))
                                              for row in summary["log"] if row["action"] == "run")}
    if process["executor"] == "claude_cli":
        outcome["costs"]["model_wall_seconds"] = process["process_wall_seconds"]
        usage = (process["stream"]["final"] or {}).get("modelUsage")
        if isinstance(usage, dict) and usage:
            for field, key in (("input_tokens", "inputTokens"), ("output_tokens", "outputTokens")):
                values = [row.get(key) if isinstance(row, dict) else None for row in usage.values()]
                if all(type(value) is int and value >= 0 for value in values):
                    outcome["costs"][field] = sum(values)
    completion = {"outcome": outcome, "reported_model_id": reported, "submission_status": submission,
                  "process_stopped": process["process_stopped"]}
    if process["executor"] == "claude_cli":
        completion["integrity_failure"] = contract_failure
    return completion


def resume(directory, attempt_id):
    manifest, plan, slot, _, entry = _lookup(directory, attempt_id)
    work = Path(directory) / "runner" / attempt_id
    completion = _validated_evidence(manifest, plan, slot, attempt_id, work)
    outcome = completion.pop("outcome")
    completion["evidence_sha256"] = _digest(work / "evidence.json")
    if "finish" in entry and entry["finish"]["outcome"]["state"] == "unresolved" and outcome["state"] != "unresolved":
        return ledger.reconcile(directory, attempt_id, outcome, **completion)
    return ledger.finish(directory, attempt_id, outcome, **completion)


def _validated_evidence(manifest, plan, slot, attempt_id, work):
    evidence = study._json(work / "evidence.json")
    study._verify(evidence, "artifact_sha256")
    runner_id = PROVIDER_RUNNER_ID if manifest["identity"]["scope"] == "primary" else RUNNER_ID
    if (evidence.get("runner_id") != runner_id or evidence.get("ledger_id") != manifest["ledger_id"]
            or evidence.get("attempt_id") != attempt_id or evidence.get("plan_sha256") != plan["plan_sha256"]):
        raise ValueError("runner evidence belongs to another attempt")
    for name, sha in evidence["files"].items():
        if name not in FILES or _digest(work / name) != sha:
            raise ValueError("runner evidence files changed")
    if set(evidence["files"]) != set(FILES):
        raise ValueError("runner evidence is incomplete")
    connection, process, sealed = (study._json(work / name) for name in ("connection.json", "process.json", "sealed.json"))
    executor = "claude_cli" if runner_id == PROVIDER_RUNNER_ID else "scripted_mcp_subprocess"
    if (process.get("executor") != executor or connection["plan_sha256"] != plan["plan_sha256"] or connection["slot_id"] != slot["slot_id"]
            or process["stdout_sha256"] != _digest(work / "stdout.jsonl") or process["stderr_sha256"] != _digest(work / "stderr.txt")):
        raise ValueError("private process/connection binding differs")
    if executor == "claude_cli":
        from benchtop import provider

        with (work / "stdout.jsonl").open("rb") as source:
            stream = provider.parse_stream(source.read(1024 * 1024).decode(errors="replace"))
        if process["stream"] != stream:
            raise ValueError("provider process metadata differs from retained stdout")
    expected = _classify(plan, slot, connection, process, sealed, manifest["identity"]["protocol"])
    if evidence["completion"] != expected or not expected["process_stopped"]:
        raise ValueError("process is not confirmed stopped or completion evidence changed")
    return copy.deepcopy(expected)


def run_one(directory, plan_sha256, slot_id, *, answer="paper_better", scenario="submit", timeout=20,
            retry_technical=False, controller=None):
    if answer not in LABELS or scenario not in SCENARIOS or type(timeout) not in (int, float) or not 0 < timeout <= 1200:
        raise ValueError("invalid controlled answer, scenario, or timeout")
    plan, slot = _registered(directory, plan_sha256, slot_id)
    if plan["binding"]["mode"] != "engineering":
        raise ValueError("only scripted engineering execution is enabled")
    extra = {"answer": answer, "scenario": scenario,
             "reported_model_id": plan["binding"]["model_configuration"]["reported_model_id"]}
    return _run(directory, plan, slot, timeout, retry_technical, controller, execute_script, RUNNER_ID, extra)


def run_provider(plan_sha256, slot_id, *, timeout=1200, retry_technical=False, controller=None):
    from benchtop import primary, provider

    if type(timeout) not in (int, float) or not 0 < timeout <= 1200:
        raise ValueError("timeout must be positive and at most 1200 seconds")
    root = primary.canonical_root()
    directory = root / "ledger"
    plan, slot = _registered(directory, plan_sha256, slot_id)
    if plan["binding"]["phase"] == "integration":
        raise ValueError("Qualia integration uses the operator-assisted UI path, not the CLI provider runner")
    primary.verify(plan)
    preflight = study._json(root / "provider.json")
    return _run(directory, plan, slot, timeout, retry_technical, controller,
                lambda work, seconds: provider.execute(work, seconds, plan, preflight), PROVIDER_RUNNER_ID, {})


def _registered(directory, plan_sha256, slot_id):
    if re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None:
        raise ValueError("invalid plan identity")
    plan = study._json(Path(directory) / "plans" / f"{plan_sha256}.json")
    study.validate_plan(plan)
    slot = next((row for row in plan["slots"] if row["slot_id"] == slot_id), None)
    if slot is None:
        raise ValueError("unknown planned slot")
    return plan, slot


def _run(directory, plan, slot, timeout, retry_technical, controller, execute, runner_id, extra):
    real = runner_id == PROVIDER_RUNNER_ID
    plan_sha256, slot_id = plan["plan_sha256"], slot["slot_id"]
    with _controller(controller) as owned:
        start = ledger.begin(directory, plan_sha256, slot_id, retry_technical=retry_technical)
        work = Path(directory).resolve() / "runner" / start["attempt_id"]
        work.mkdir(parents=True, exist_ok=False, mode=0o700)
        connection = owned.prepare(plan, slot)
        connection.update(extra)
        _private(work / "connection.json", connection)
        process = execute(work, timeout)
        _private(work / "process.json", process)
        sealed = owned.seal(connection)
        _private(work / "sealed.json", sealed)
        manifest = ledger._json(Path(directory) / "ledger.json")
        completion = _classify(plan, slot, connection, process, sealed, manifest["identity"]["protocol"])
        evidence = study._seal({"runner_id": runner_id, "ledger_id": manifest["ledger_id"],
            "plan_sha256": plan_sha256, "attempt_id": start["attempt_id"], "completion": completion,
            "files": {name: _digest(work / name) for name in FILES},
            "scope": "Authorized subscription CLI attempt with private MCP grading." if real else
                     "Scripted subprocess/MCP engineering control; no model or primary-study authorization."}, "artifact_sha256")
        _private(work / "evidence.json", evidence)
        if not process["process_stopped"]:
            return {"attempt_id": start["attempt_id"], "status": "pending_reconciliation", "model_sessions": int(real)}
        result = resume(directory, start["attempt_id"])
        return {"attempt_id": start["attempt_id"], "status": result["outcome"]["state"],
                "reward": result["outcome"]["reward"], "model_sessions": int(real), "scripted_processes": int(not real)}


@contextmanager
def _controller(existing):
    if existing is not None:
        yield existing
    else:
        with Controller() as controller:
            yield controller
