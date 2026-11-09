import argparse
import asyncio
import copy
import json
import logging
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from benchtop import release_data, study
from benchtop.access import redact
from benchtop.agents.baselines import paired_uniform_public
from benchtop.agents.common import LabClient
from benchtop.cache import file_hash
from benchtop.env import ROOT, Env
from benchtop.evaluate import corrected_replay
from benchtop.runner import Controller

ENVIRONMENT = "2610-36983-advancement-or-acceleration-how"
FORBIDDEN = {"score", "reward", "graded", "ground_truth", "heldout_label", "seed_ids", "qualification_seeds", "confirmation_seeds"}


def public_only(value):
    if isinstance(value, dict):
        if FORBIDDEN & value.keys():
            raise ValueError("private grading or seed identities appeared in an agent response")
        for item in value.values():
            public_only(item)
    elif isinstance(value, list):
        for item in value:
            public_only(item)


def prepare(controller, env):
    async def create():
        controller.hub.envs[env.id] = env
        episode = controller.hub.create(env.id, "claim", budget_frac=0.12, seed=0, max_calls=64, show_paper=True)
        return {"controller_id": controller.identity, "episode": episode.id, "mcp_url": controller.hub.agent_url(episode)}

    return asyncio.run_coroutine_threadsafe(create(), controller.loop).result(timeout=10)


@contextmanager
def quiet_transport():
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous)


def collect():
    with quiet_transport():
        return _collect()


def _collect():
    source = ROOT / "envs" / ENVIRONMENT
    snapshot = study._snapshot(source)
    env = corrected_replay(Env.load(source, replay=True))
    events = []
    started = time.perf_counter()
    with Controller() as controller:
        connection = prepare(controller, env)
        client = LabClient(connection["mcp_url"])
        try:
            client.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "benchtop-scripted-walkthrough", "version": "1"}})
            client.initialized()
            tools = client.rpc("tools/list")
            if {row["name"] for row in tools["tools"]} != {"describe_lab", "run_experiments", "submit_answer"}:
                raise ValueError("walkthrough needs exactly the three scoped lab tools")

            def call(name, arguments=None, *, actor="scripted_policy"):
                response = client.call(name, arguments)
                public_only(response)
                if "error" in response or response.get("errors"):
                    raise ValueError("walkthrough tool failed: " + name)
                events.append({"actor": actor, "tool": name, "arguments": copy.deepcopy(arguments or {}), "response": response})
                return response

            description = call("describe_lab")
            outcome = paired_uniform_public(description, lambda configs, seeds=1: call("run_experiments", {"configs": configs, "seeds": seeds}), np.random.default_rng(0))
            receipt = call("submit_answer", {"answer": outcome["answer"]})
            repeated = call("submit_answer", {"answer": outcome["answer"]}, actor="post_commit_probe")
            if receipt != {"received": True} or repeated != receipt:
                raise ValueError("submission must return only an idempotent receipt")
            sealed = controller.seal(connection)
            revoked = client.http.post(connection["mcp_url"], headers={"accept": "application/json, text/event-stream"},
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
            if revoked.status_code != 401:
                raise ValueError("sealing did not revoke the episode capability")
        finally:
            client.http.close()
    if study._snapshot(source) != snapshot:
        raise ValueError("source lab changed during the walkthrough")
    summary, trace = sealed["summary"], sealed["trace"]
    if not summary["result"] or trace["submission_source"] != "mcp" or summary["spent"] > summary["budget"]:
        raise ValueError("walkthrough did not commit one budgeted MCP answer")
    report = {"schema_version": 1, "walkthrough_id": "cached_mcp_walkthrough_v1", "recorded_at_utc": datetime.now(UTC).isoformat(),
        "status": "passed", "executor": "deterministic_paired_uniform_policy_over_real_loopback_http_mcp",
        "env_id": env.id, "task": "claim", "budget_fraction": 0.12, "policy_seed": 0, "episode_seed": 0,
        "source_qualification_files": snapshot["files"], "code_sha256": study._sources(), "walkthrough_source_sha256": file_hash(Path(__file__)),
        "agent_events": events, "policy": {"policy_id": outcome["policy_id"], "answer": outcome["answer"], "stop_reason": outcome["stop_reason"]},
        "checks": {"three_tools_discovered": True, "agent_responses_grade_free": True, "receipt_only": True,
            "repeated_submission_same_receipt": True, "capability_revoked_after_seal": True, "original_lab_unchanged": True},
        "evaluator_after_submission": {key: summary[key] for key in ("grader_id", "execution_id", "task_version", "charging_id", "budget", "spent", "calls")}
            | {"score": summary["result"]["score"], "submitted_answer": outcome["answer"], "heldout_label": env.task("claim")["answer"]},
        "controller_wall_seconds": time.perf_counter() - started, "model_sessions": 0, "lab_workers_started": 0, "qualia_verified": False,
        "human_admission": "not_recorded", "fresh_calibration_result": "not_collected",
        "scope": "Scripted engineering demonstration on a corrected cached historical question. The policy receives only public MCP responses; evaluator grading is read after submission. No model, actual Qualia client, fresh study, scientific admission or inference billing is implied."}
    release_data.safe_text(report)
    return study._seal(report, "walkthrough_sha256")


def render(report):
    evaluation = report["evaluator_after_submission"]
    lines = ["# A budgeted experiment over MCP", "", "This run uses a deterministic classical policy, not a language model or Qualia client.", "",
        f"Lab: `{report['env_id']}`. Task: `claim`. Frozen budget: 12% of the visible grid cost.", "",
        "## Agent view", "", "The public transcript is in report.json. It contains description, budgeted experiments and submission receipts, with no grading fields.", ""]
    for index, event in enumerate(report["agent_events"], 1):
        response = event["response"]
        detail = (str(len(response["results"])) + " visible observations") if event["tool"] == "run_experiments" else "receipt only" if event["tool"] == "submit_answer" else "question, search space and budget"
        lines.append(f"{index}. `{event['tool']}`: {detail} ({event['actor']}).")
    lines += ["", "## Evaluator view after submission", "", f"Submitted: `{evaluation['submitted_answer']}`. Corrected cached held-out label: `{evaluation['heldout_label']}`. Reward: **{evaluation['score']}**.", "",
        f"Charged experiment cost: {evaluation['spent']:.6g} / {evaluation['budget']:.6g} compute seconds. Experiment calls: {evaluation['calls']}. The charge uses the frozen historical prices, not a new CPU measurement.", "",
        "The evaluator sealed the episode and revoked its capability. A repeated submission returned the same receipt without exposing a reward.", "",
        "## Scope", "", report["scope"], "", "Human admission, the fresh calibration study and the actual Qualia demonstration remain separate work. An individual scripted answer does not measure model performance.", "",
        "Reproduce in a new directory:", "", "```sh", "uv run python -m benchmarks.release_v1.walkthrough --output-dir artifacts/walkthrough-repeat", "```", ""]
    return "\n".join(lines)


def run(output):
    target = Path(output).resolve()
    if target.is_relative_to(ROOT) and not target.is_relative_to(ROOT / "artifacts"):
        raise ValueError("choose a new output directory under artifacts/ or outside the checkout")
    target.mkdir(parents=True, exist_ok=False)
    marker = target / ".incomplete"
    marker.write_text("walkthrough incomplete; retain this output and use a new directory\n")
    report = collect()
    study._publish(target / "report.json", report)
    (target / "report.md").write_text(render(report))
    marker.unlink()
    return report


def main():
    parser = argparse.ArgumentParser(description="Run a scripted cached MCP walkthrough; starts no model or lab worker.")
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()
    try:
        report = run(arguments.output_dir)
    except Exception as error:
        raise SystemExit("walkthrough failed: " + redact(str(error))) from None
    print(json.dumps({"status": report["status"], "report": str(Path(arguments.output_dir) / "report.md"),
        "model_sessions": 0, "lab_workers_started": 0, "qualia_verified": False}, indent=2))


if __name__ == "__main__":
    main()
