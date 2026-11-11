import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from benchtop.access import admin_headers, redact
from benchtop.env import ROOT, STATE

RESULTS = ROOT / "results"

SYSTEM = (
    "You are an experimental researcher working in a small computational lab through tools. "
    "Your goal is to answer the lab's question correctly. Decide which experiments are most "
    "informative, replicate when results are noisy, and stop when you are confident. Your answer is "
    "graded on fresh seeds you never see, so judge differences against run-to-run noise. When you are "
    "done, call submit_answer exactly once in the requested format with your confidence."
)


def first_message(describe: dict, budget: dict) -> str:
    lab = json.dumps({**describe, "budget": budget}, indent=1)
    return (
        "Here is your initial lab information. You must call describe_lab first to verify the live episode and budget:\n\n"
        + lab
        + "\n\nCall describe_lab before any experiments. Then investigate with run_experiments and call submit_answer once."
    )


class Client:
    def __init__(self, base: str = "http://127.0.0.1:8787", token: str | None = None):
        self.base = base.rstrip("/")
        self.http = httpx.Client(base_url=self.base, timeout=60, headers=admin_headers(self.base, token))

    def create(self, env: str, task: str, **kw) -> dict:
        r = self.http.post("/episodes", json={"env": env, "task": task, **kw})
        r.raise_for_status()
        return r.json()

    def summary(self, eid: str) -> dict:
        response = self.http.get(f"/episodes/{eid}")
        response.raise_for_status()
        return response.json()

    def mcp_trace(self, eid: str) -> dict:
        response = self.http.get(f"/episodes/{eid}/mcp-trace")
        response.raise_for_status()
        return response.json()

    def run(self, eid: str, configs: list[dict], seeds: int = 1) -> dict:
        return self.http.post(f"/episodes/{eid}/run", json={"configs": configs, "seeds": seeds}).json()

    def submit(self, eid: str, answer, confidence=None) -> dict:
        return self.http.post(
            f"/episodes/{eid}/submit", json={"answer": answer, "confidence": confidence}
        ).json()


def record(row: dict, name: str = "rollouts.jsonl") -> None:
    RESULTS.mkdir(exist_ok=True)
    row = redact({"at": datetime.now(UTC).isoformat(timespec="seconds"), **row})
    with (RESULTS / name).open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def transcript_dir(agent: str) -> Path:
    d = STATE / "transcripts" / agent.replace(":", "-") / time.strftime("%Y%m%d")
    d.mkdir(parents=True, exist_ok=True)
    return d


def outcome(summary: dict, agent: str, extra: dict) -> dict:
    result = summary.get("result") or {}
    return redact({
        "agent": agent,
        "env": summary["env"],
        "task": summary["task"],
        "grader_id": summary["grader_id"],
        "execution_id": summary["execution_id"],
        "task_version": summary["task_version"],
        "charging_id": summary["charging_id"],
        "episode": summary["episode"],
        "show_paper": summary["show_paper"],
        "score": result.get("score"),
        "confidence": result.get("confidence"),
        "spent_frac": result.get("spent_frac"),
        "calls": summary["calls"],
        "submitted": bool(result),
        "answer": next((e["answer"] for e in summary["log"] if e["action"] == "submit"), None),
        "runs": sum(len(e["ran"]) for e in summary["log"] if e["action"] == "run"),
        "requested_runs": sum(len(e.get("attempted", e["ran"])) for e in summary["log"] if e["action"] == "run"),
        "ran": [pair for e in summary["log"] if e["action"] == "run" for pair in e["ran"]],
        "episode_seed": summary["seed"],
        **extra,
    })


class LabClient:
    def __init__(self, url: str):
        self.url = url
        self.http = httpx.Client(timeout=60)

    def rpc(self, method: str, params: dict | None = None) -> dict:
        response = self.http.post(
            self.url,
            headers={"accept": "application/json, text/event-stream", "mcp-protocol-version": "2025-06-18"},
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}},
        )
        response.raise_for_status()
        body = response.json()
        if "error" in body:
            raise RuntimeError(body["error"]["message"])
        return body["result"]

    def initialized(self):
        response = self.http.post(
            self.url, headers={"accept": "application/json, text/event-stream", "mcp-protocol-version": "2025-06-18"},
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        response.raise_for_status()

    def call(self, name: str, arguments: dict | None = None) -> dict:
        result = self.rpc("tools/call", {"name": name, "arguments": arguments or {}})
        if result.get("isError"):
            return {"error": result["content"][0]["text"]}
        return json.loads(result["content"][0]["text"])


def run_cmd(cmd: list[str], cwd: str, timeout: float) -> tuple[str, str, bool]:
    # own process group, so a timeout kills the cli and anything it spawned
    import os
    import signal
    import subprocess

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return out, err, False
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        return out or "", err or "", True
