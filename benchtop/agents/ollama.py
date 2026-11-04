import json
import time

import httpx

from benchtop.agents.common import SYSTEM, Client, LabClient, first_message, outcome, transcript_dir

OLLAMA = "http://127.0.0.1:11434"

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "describe_lab",
            "description": "Read the lab, budget, rules, and question.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_experiments",
            "description": "Run configs (list of objects mapping factor names to levels), each with `seeds` fresh seeds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "configs": {"type": "array", "items": {"type": "object"}},
                    "seeds": {"type": "integer", "default": 1},
                },
                "required": ["configs"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final answer once, in the format the lab asks for, with confidence 0 to 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"description": "string label, or an object for where/best questions"},
                    "confidence": {"type": "number"},
                },
                "required": ["answer"],
            },
        },
    },
]


def _args(raw) -> dict:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def rollout(
    client: Client,
    env: str,
    task: str,
    model: str = "qwen3:4b-instruct",
    show_paper: bool = True,
    seed: int | None = None,
    budget_frac: float = 0.25,
    max_turns: int = 30,
    deadline: float = 480,
) -> dict:
    ep = client.create(env, task, show_paper=show_paper, seed=seed, budget_frac=budget_frac)
    eid = ep["id"]
    lab = LabClient(ep["mcp_url"])
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": first_message(ep["describe"], ep["budget"])},
    ]
    started, tool_uses, nudges, done, error = time.time(), 0, 0, False, None
    http = httpx.Client(base_url=OLLAMA, timeout=600)
    for _ in range(max_turns):
        if time.time() - started > deadline:
            error = f"deadline of {deadline:.0f}s"
            break
        r = http.post(
            "/api/chat",
            json={
                "model": model,
                "messages": messages,
                "tools": TOOL_SPECS,
                "stream": False,
                "options": {"temperature": 0.7, "num_ctx": 16384, "num_predict": 1536},
            },
        )
        if r.status_code != 200:
            error = f"ollama {r.status_code}: {r.text[:200]}"
            break
        msg = r.json()["message"]
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            if nudges >= 2:
                break
            nudges += 1
            messages.append(
                {"role": "user", "content": "Use the tools. When you are confident, call submit_answer."}
            )
            continue
        for call in calls:
            name, args = call["function"]["name"], _args(call["function"].get("arguments"))
            tool_uses += 1
            if name == "describe_lab":
                result = lab.call("describe_lab")
            elif name == "run_experiments":
                result = lab.call(name, {"configs": args.get("configs") or [], "seeds": args.get("seeds", 1)})
            elif name == "submit_answer":
                result = lab.call(name, {"answer": args.get("answer"), "confidence": args.get("confidence")})
                done = bool(result.get("received"))
            else:
                result = {"error": f"unknown tool {name}"}
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
        if done:
            break
    (transcript_dir(f"ollama-{model}") / f"{eid}.json").write_text(json.dumps(messages, indent=1))
    extra = {
        "model": model,
        "tool_uses": tool_uses,
        "secs": round(time.time() - started, 1),
        "api_equiv_usd": 0.0,
        "error": error if not done else None,
    }
    lab.http.close()
    http.close()
    return outcome(client.summary(eid), f"ollama:{model}", extra)
