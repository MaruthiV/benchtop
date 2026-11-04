import json
import tempfile
import time
from pathlib import Path

from benchtop.agents.common import SYSTEM, Client, first_message, outcome, run_cmd, transcript_dir

TOOLS = ["mcp__benchtop__describe_lab", "mcp__benchtop__run_experiments", "mcp__benchtop__submit_answer"]


def rollout(
    client: Client,
    env: str,
    task: str,
    model: str = "haiku",
    show_paper: bool = True,
    seed: int | None = None,
    budget_frac: float = 0.25,
    timeout: float = 1200,
) -> dict:
    ep = client.create(env, task, show_paper=show_paper, seed=seed, budget_frac=budget_frac)
    eid = ep["id"]
    config = {"mcpServers": {"benchtop": {"type": "http", "url": f"{client.base}/e/{eid}/mcp"}}}
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="benchtop-agent-") as cwd:
        cfg = Path(cwd) / "mcp.json"
        cfg.write_text(json.dumps(config))
        cmd = [
            "claude",
            "-p",
            first_message(ep["describe"], ep["budget"]),
            "--model",
            model,
            "--system-prompt",
            SYSTEM,
            "--tools",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            str(cfg),
            "--allowedTools",
            *TOOLS,
            "--permission-prompts",
            "none",
            "--output-format",
            "stream-json",
            "--verbose",
            "--no-session-persistence",
        ]
        stream, err, timed_out = run_cmd(cmd, cwd, timeout)
        err = "timeout" if timed_out else err[-2000:]

    (transcript_dir(f"claude-{model}") / f"{eid}.jsonl").write_text(stream)
    final, tool_uses = {}, 0
    for line in stream.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("type") == "assistant":
            tool_uses += sum(1 for c in msg["message"].get("content", []) if c.get("type") == "tool_use")
        if msg.get("type") == "result":
            final = msg
    extra = {
        "model": model,
        "tool_uses": tool_uses,
        "secs": round(time.time() - started, 1),
        "api_equiv_usd": final.get("total_cost_usd"),
        "turns": final.get("num_turns"),
        "error": None if final and not final.get("is_error") else (err or final.get("subtype")),
    }
    return outcome(client.summary(eid), f"claude:{model}", extra)
