import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

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
        "Here is your lab (describe_lab returns the same thing):\n\n"
        + lab
        + "\n\nInvestigate with run_experiments, then call submit_answer once."
    )


class Client:
    def __init__(self, base: str = "http://127.0.0.1:8787"):
        self.base = base.rstrip("/")
        self.http = httpx.Client(base_url=self.base, timeout=60)

    def create(self, env: str, task: str, **kw) -> dict:
        r = self.http.post("/episodes", json={"env": env, "task": task, **kw})
        r.raise_for_status()
        return r.json()

    def summary(self, eid: str) -> dict:
        return self.http.get(f"/episodes/{eid}").json()

    def run(self, eid: str, configs: list[dict], seeds: int = 1) -> dict:
        return self.http.post(f"/episodes/{eid}/run", json={"configs": configs, "seeds": seeds}).json()

    def submit(self, eid: str, answer, confidence=None) -> dict:
        return self.http.post(
            f"/episodes/{eid}/submit", json={"answer": answer, "confidence": confidence}
        ).json()


def record(row: dict, name: str = "rollouts.jsonl") -> None:
    RESULTS.mkdir(exist_ok=True)
    row = {"at": datetime.now(UTC).isoformat(timespec="seconds"), **row}
    with (RESULTS / name).open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def transcript_dir(agent: str) -> Path:
    d = STATE / "transcripts" / agent.replace(":", "-") / time.strftime("%Y%m%d")
    d.mkdir(parents=True, exist_ok=True)
    return d


def outcome(summary: dict, agent: str, extra: dict) -> dict:
    result = summary.get("result") or {}
    return {
        "agent": agent,
        "env": summary["env"],
        "task": summary["task"],
        "episode": summary["episode"],
        "show_paper": summary["show_paper"],
        "score": result.get("score"),
        "confidence": result.get("confidence"),
        "spent_frac": result.get("spent_frac"),
        "calls": summary["calls"],
        "submitted": bool(result),
        "answer": next((e["answer"] for e in summary["log"] if e["action"] == "submit"), None),
        "runs": sum(len(e["ran"]) for e in summary["log"] if e["action"] == "run"),
        "ran": [pair for e in summary["log"] if e["action"] == "run" for pair in e["ran"]],
        "episode_seed": summary["seed"],
        **extra,
    }


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
