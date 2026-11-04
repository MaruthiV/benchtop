import json
import tempfile
import time
from pathlib import Path

import httpx

from benchtop.agents.common import run_cmd
from benchtop.env import STATE

LOG = STATE / "llm"


class LLMError(RuntimeError):
    pass


def _log(backend: str, model: str, prompt: str, out: dict) -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    row = {"t": time.time(), "backend": backend, "model": model, "prompt_chars": len(prompt), **out}
    with (LOG / "calls.jsonl").open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def claude(
    prompt: str,
    model: str = "sonnet",
    system: str = "You are a precise research engineer.",
    schema: dict | None = None,
    timeout: float = 900,
) -> dict:
    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--system-prompt",
        system,
        "--tools",
        "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--output-format",
        "json",
    ]
    if schema:
        cmd += ["--json-schema", json.dumps(schema)]
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="benchtop-llm-") as cwd:
        stdout, stderr, timed_out = run_cmd(cmd, cwd, timeout)
    if timed_out:
        raise LLMError(f"claude timed out after {timeout}s")
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise LLMError(f"claude returned no json: {stdout[-300:]} {stderr[-300:]}") from e
    out = {
        "text": data.get("result", ""),
        "structured": data.get("structured_output"),
        "api_equiv_usd": data.get("total_cost_usd"),
        "secs": round(time.time() - started, 1),
        "error": data.get("subtype") if data.get("is_error") else None,
    }
    _log(
        "claude",
        model,
        prompt,
        {k: v for k, v in out.items() if k != "structured"} | {"text": out["text"][:200]},
    )
    if out["error"] or (schema and out["structured"] is None):
        raise LLMError(f"claude error: {out['error'] or 'no structured output'} {out['text'][:300]}")
    return out


def codex(prompt: str, model: str | None = None, timeout: float = 900) -> dict:
    off = ["shell_tool", "unified_exec", "browser_use", "computer_use", "apps", "view_image", "goals"]
    cmd = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--json",
        "-c",
        'web_search="disabled"',
        "-c",
        "project_doc_max_bytes=0",
    ]
    for feature in off:
        cmd += ["--disable", feature]
    if model:
        cmd += ["-m", model]
    cmd.append(prompt)
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="benchtop-llm-") as cwd:
        stdout, _, timed_out = run_cmd(cmd, cwd, timeout)
    if timed_out:
        raise LLMError(f"codex timed out after {timeout}s")
    text, error = "", None
    for line in stdout.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = msg.get("item", {})
        if item.get("type") == "agent_message":
            text = item.get("text", text)
        if msg.get("type") in ("error", "turn.failed"):
            error = msg.get("message") or msg.get("error", {}).get("message")
    out = {
        "text": text,
        "structured": None,
        "api_equiv_usd": None,
        "secs": round(time.time() - started, 1),
        "error": error,
    }
    _log("codex", model or "default", prompt, out | {"text": text[:200]})
    if error or not text:
        raise LLMError(f"codex error: {error or 'empty reply'}")
    return out


def ollama(prompt: str, model: str = "qwen3:4b-instruct", system: str = "", timeout: float = 900) -> dict:
    started = time.time()
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    r = httpx.post(
        "http://127.0.0.1:11434/api/chat",
        timeout=timeout,
        json={"model": model, "messages": messages, "stream": False},
    )
    if r.status_code != 200:
        raise LLMError(f"ollama {r.status_code}")
    text = r.json()["message"]["content"]
    out = {
        "text": text,
        "structured": None,
        "api_equiv_usd": 0.0,
        "secs": round(time.time() - started, 1),
        "error": None,
    }
    _log("ollama", model, prompt, out | {"text": text[:200]})
    return out


def complete(prompt: str, backend: str = "claude", model: str | None = None, **kw) -> dict:
    if backend == "claude":
        return claude(prompt, model=model or "sonnet", **kw)
    if backend == "codex":
        return codex(prompt, model=model, **{k: v for k, v in kw.items() if k == "timeout"})
    if backend == "ollama":
        return ollama(
            prompt, model=model or "qwen3:4b-instruct", **{k: v for k, v in kw.items() if k != "schema"}
        )
    raise ValueError(f"unknown backend {backend}")


def fenced(text: str, lang: str) -> str | None:
    marker = f"```{lang}"
    start = text.find(marker)
    if start < 0:
        return None
    start = text.index("\n", start) + 1
    end = text.find("```", start)
    return text[start : end if end >= 0 else None].strip() + "\n"


def save(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
