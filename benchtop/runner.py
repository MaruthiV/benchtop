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
    files = [os.fdopen(os.open(work / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")
             for name in ("stdout.jsonl", "stderr.txt")]
    process, timed_out = None, False
    started = time.monotonic()
    try:
        process = subprocess.Popen([sys.executable, "-I", str(Path(__file__).with_name("_scripted_agent.py")),
                                    str(work / "connection.json")], cwd=work, stdin=subprocess.DEVNULL,
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
    stdout = (work / "stdout.jsonl").read_bytes()[:1024 * 1024].decode(errors="replace")
    return {"executor": "scripted_mcp_subprocess", "process_stopped": stopped,
            "returncode": process.returncode if process else None, "timed_out": timed_out,
            "process_wall_seconds": time.monotonic() - started,
            "stdout_sha256": _digest(work / "stdout.jsonl"), "stderr_sha256": _digest(work / "stderr.txt"),
            "stdout_truncated": (work / "stdout.jsonl").stat().st_size > 1024 * 1024,
            "stream": parse_stream(stdout), "model_sessions": 0}


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
                                 max_calls=plan["binding"]["limits"]["max_experiment_tool_calls"])
            return {"controller_id": self.identity, "episode": ep.id, "mcp_url": self.hub.agent_url(ep),
                    "plan_sha256": plan["plan_sha256"], "slot_id": slot["slot_id"], "expected_budget": expected_budget}

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
    submission = "committed" if result is not None else "not_committed" if valid else "unknown"
    models = process["stream"]["models"]
    expected_model = plan["binding"]["model_configuration"]["reported_model_id"]
    reported = next((value for value in models if value != expected_model), models[0] if models else None)
    outcome = {"state": "unresolved", "reward": None, "failure_category": "unclassified_process_result"}
    if not valid:
        outcome["failure_category"] = "episode_binding_mismatch"
    elif process["stdout_truncated"]:
        outcome["failure_category"] = "truncated_process_metadata"
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
        if final.get("is_error") is True and category in allowed:
            outcome = {"state": "technical_failure", "reward": None, "failure_category": category}
        elif final.get("is_error") is False and process["returncode"] == 0:
            outcome = {"state": "agent_failure", "reward": 0, "failure_category": "exited_without_answer"}
    outcome["costs"] = {"charged_experiment_cost": summary["spent"], "unique_executed_runs": 0,
                        "requested_runs": sum(len(row["configs"]) * max(1, min(row["seeds"], len(summary["visible_order"])))
                                              for row in summary["log"] if row["action"] == "run")}
    return {"outcome": outcome, "reported_model_id": reported, "submission_status": submission,
            "process_stopped": process["process_stopped"]}


def resume(directory, attempt_id):
    manifest, plan, slot, _, entry = _lookup(directory, attempt_id)
    work = Path(directory) / "runner" / attempt_id
    evidence = study._json(work / "evidence.json")
    study._verify(evidence, "artifact_sha256")
    if (evidence.get("runner_id") != RUNNER_ID or evidence.get("ledger_id") != manifest["ledger_id"]
            or evidence.get("attempt_id") != attempt_id or evidence.get("plan_sha256") != plan["plan_sha256"]):
        raise ValueError("runner evidence belongs to another attempt")
    for name, sha in evidence["files"].items():
        if name not in ("connection.json", "process.json", "sealed.json", "stdout.jsonl", "stderr.txt") or _digest(work / name) != sha:
            raise ValueError("runner evidence files changed")
    if set(evidence["files"]) != {"connection.json", "process.json", "sealed.json", "stdout.jsonl", "stderr.txt"}:
        raise ValueError("runner evidence is incomplete")
    connection, process, sealed = (study._json(work / name) for name in ("connection.json", "process.json", "sealed.json"))
    if (connection["plan_sha256"] != plan["plan_sha256"] or connection["slot_id"] != slot["slot_id"]
            or process["stdout_sha256"] != _digest(work / "stdout.jsonl") or process["stderr_sha256"] != _digest(work / "stderr.txt")):
        raise ValueError("private process/connection binding differs")
    expected = _classify(plan, slot, connection, process, sealed, manifest["identity"]["protocol"])
    if evidence["completion"] != expected or not expected["process_stopped"]:
        raise ValueError("process is not confirmed stopped or completion evidence changed")
    completion = copy.deepcopy(expected)
    outcome = completion.pop("outcome")
    completion["evidence_sha256"] = _digest(work / "evidence.json")
    if "finish" in entry and entry["finish"]["outcome"]["state"] == "unresolved" and outcome["state"] != "unresolved":
        return ledger.reconcile(directory, attempt_id, outcome, **completion)
    return ledger.finish(directory, attempt_id, outcome, **completion)


def run_one(directory, plan_sha256, slot_id, *, answer="paper_better", scenario="submit", timeout=20,
            retry_technical=False, controller=None):
    if answer not in LABELS or scenario not in SCENARIOS or type(timeout) not in (int, float) or not 0 < timeout <= 1200:
        raise ValueError("invalid controlled answer, scenario, or timeout")
    if re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None:
        raise ValueError("invalid plan identity")
    plan = study._json(Path(directory) / "plans" / f"{plan_sha256}.json")
    study.validate_plan(plan)
    if plan["binding"]["mode"] != "engineering":
        raise ValueError("only scripted engineering execution is enabled")
    slot = next((row for row in plan["slots"] if row["slot_id"] == slot_id), None)
    if slot is None:
        raise ValueError("unknown planned slot")
    with _controller(controller) as owned:
        start = ledger.begin(directory, plan_sha256, slot_id, retry_technical=retry_technical)
        work = Path(directory).resolve() / "runner" / start["attempt_id"]
        work.mkdir(parents=True, exist_ok=False, mode=0o700)
        connection = owned.prepare(plan, slot)
        connection.update(answer=answer, scenario=scenario,
                          reported_model_id=plan["binding"]["model_configuration"]["reported_model_id"])
        _private(work / "connection.json", connection)
        process = execute_script(work, timeout)
        _private(work / "process.json", process)
        sealed = owned.seal(connection)
        _private(work / "sealed.json", sealed)
        manifest = ledger._json(Path(directory) / "ledger.json")
        completion = _classify(plan, slot, connection, process, sealed, manifest["identity"]["protocol"])
        evidence = study._seal({"runner_id": RUNNER_ID, "ledger_id": manifest["ledger_id"],
            "plan_sha256": plan_sha256, "attempt_id": start["attempt_id"], "completion": completion,
            "files": {name: _digest(work / name) for name in ("connection.json", "process.json", "sealed.json", "stdout.jsonl", "stderr.txt")},
            "scope": "Scripted subprocess/MCP engineering control; no model or primary-study authorization."}, "artifact_sha256")
        _private(work / "evidence.json", evidence)
        if not process["process_stopped"]:
            return {"attempt_id": start["attempt_id"], "status": "pending_reconciliation", "model_sessions": 0}
        result = resume(directory, start["attempt_id"])
        return {"attempt_id": start["attempt_id"], "status": result["outcome"]["state"],
                "reward": result["outcome"]["reward"], "model_sessions": 0, "scripted_processes": 1}


@contextmanager
def _controller(existing):
    if existing is not None:
        yield existing
    else:
        with Controller() as controller:
            yield controller
