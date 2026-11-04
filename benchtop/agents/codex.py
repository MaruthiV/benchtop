import json
import tempfile
import time

from benchtop.agents.common import SYSTEM, Client, first_message, outcome, run_cmd, transcript_dir

OFF = ["shell_tool", "unified_exec", "browser_use", "computer_use", "apps", "view_image", "goals"]


def rollout(
    client: Client,
    env: str,
    task: str,
    model: str = "default",
    show_paper: bool = True,
    seed: int | None = None,
    budget_frac: float = 0.25,
    timeout: float = 900,
) -> dict:
    ep = client.create(env, task, show_paper=show_paper, seed=seed, budget_frac=budget_frac)
    eid = ep["id"]
    prompt = SYSTEM + "\n\n" + first_message(ep["describe"], ep["budget"])
    cmd = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--json",
        "-c",
        f'mcp_servers.benchtop.url={json.dumps(ep["mcp_url"])}',
        "-c",
        'web_search="disabled"',
        "-c",
        "project_doc_max_bytes=0",
    ]
    for feature in OFF:
        cmd += ["--disable", feature]
    if model != "default":
        cmd += ["-m", model]
    cmd.append(prompt)
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="benchtop-agent-") as cwd:
        stream, _, timed_out = run_cmd(cmd, cwd, timeout)
    (transcript_dir(f"codex-{model}") / f"{eid}.jsonl").write_text(stream)
    tool_uses, error, usage = 0, None, {}
    for line in stream.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("item", {}).get("type") == "mcp_tool_call" and msg.get("type") == "item.completed":
            tool_uses += 1
        if msg.get("type") in ("error", "turn.failed"):
            error = msg.get("message") or msg.get("error", {}).get("message")
        if msg.get("type") == "turn.completed":
            usage = msg.get("usage", {})
    if timed_out:
        error = f"timeout after {timeout:.0f}s" + (f" ({error})" if error else "")
    extra = {
        "model": model,
        "tool_uses": tool_uses,
        "secs": round(time.time() - started, 1),
        "api_equiv_usd": None,
        "usage": usage,
        "error": error,
    }
    return outcome(client.summary(eid), f"codex:{model}", extra)
