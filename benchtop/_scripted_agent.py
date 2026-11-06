import json
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    url = config["mcp_url"]
    if urlsplit(url).hostname != "127.0.0.1" or urlsplit(url).scheme != "http":
        raise ValueError("scripted engineering client requires its owned loopback endpoint")
    scenario = config["scenario"]
    model = "different-synthetic-model" if scenario == "model_change" else config["reported_model_id"]
    print(json.dumps({"type": "system", "subtype": "init", "model": model, "synthetic_control": True}), flush=True)
    if scenario == "timeout":
        time.sleep(60)
    if scenario in ("quota", "unknown_error"):
        print(json.dumps({"type": "result", "is_error": True, "synthetic_control": True,
                          "failure_category": "provider_quota" if scenario == "quota" else "unclassified_control"}), flush=True)
        return
    counter = 0

    def rpc(method, params, notification=False):
        nonlocal counter
        counter += 1
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        if not notification:
            payload["id"] = counter
        request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={
            "content-type": "application/json", "accept": "application/json, text/event-stream",
            "mcp-protocol-version": "2025-06-18"})
        with urllib.request.urlopen(request, timeout=5) as response:
            raw = response.read()
        if notification:
            return None
        result = json.loads(raw)
        if "error" in result:
            raise ValueError(result["error"])
        return result["result"]

    def call(name, arguments=None):
        response = rpc("tools/call", {"name": name, "arguments": arguments or {}})
        if response.get("isError"):
            raise ValueError("scripted tool returned a transport error")
        decoded = json.loads(response["content"][0]["text"])
        print(json.dumps({"type": "tool_result", "tool": name, "response": decoded,
                          "synthetic_control": True}), flush=True)
        return decoded

    rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "benchtop-scripted-control", "version": "1"}})
    rpc("notifications/initialized", {}, notification=True)
    if {tool["name"] for tool in rpc("tools/list", {})["tools"]} != {"describe_lab", "run_experiments", "submit_answer"}:
        raise ValueError("unexpected agent tools")
    description = call("describe_lab")
    defaults = {factor["name"]: factor["levels"][0] for factor in description["factors"]}
    treatment = description["treatment"]
    call("run_experiments", {"configs": [{**defaults, treatment["factor"]: treatment[arm]}
                                         for arm in ("baseline", "paper_method")], "seeds": 1})
    if scenario != "no_answer":
        receipt = call("submit_answer", {"answer": config["answer"], "confidence": 0.5})
        if receipt != {"received": True}:
            raise ValueError("expected a grade-free completion receipt")
    if scenario == "post_submit_timeout":
        time.sleep(60)
    print(json.dumps({"type": "result", "is_error": scenario == "post_submit_error", "synthetic_control": True,
                      "failure_category": "transport_failure" if scenario == "post_submit_error" else None}), flush=True)


if __name__ == "__main__":
    main()
