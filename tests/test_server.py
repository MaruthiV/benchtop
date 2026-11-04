import json
from urllib.parse import urlsplit

import pytest
from starlette.testclient import TestClient

from benchtop.server import Hub, build_app
from tests.test_core import toy_env

MCP_HEADERS = {"accept": "application/json, text/event-stream", "mcp-protocol-version": "2025-06-18"}


def rpc(client, url, method, params, i=1, headers=None):
    r = client.post(
        url,
        headers={**MCP_HEADERS, **(headers or {})},
        json={"jsonrpc": "2.0", "id": i, "method": method, "params": params},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_mcp_episode_roundtrip(server_pair):
    control, agent, _ = server_pair
    created = control.post("/episodes", json={"env": "toy", "task": "claim", "seed": 4}).json()
    eid, url = created["id"], created["mcp_url"]
    rpc(
        agent,
        url,
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        },
    )
    tools = rpc(agent, url, "tools/list", {}, 2)["result"]["tools"]
    assert {t["name"] for t in tools} == {"describe_lab", "run_experiments", "submit_answer"}

    out = rpc(
        agent,
        url,
        "tools/call",
        {
            "name": "run_experiments",
            "arguments": {
                "configs": [{"method": "new", "ctx": 2}, {"method": "base", "ctx": 2}],
                "seeds": 2,
            },
        },
        3,
    )
    payload = json.loads(out["result"]["content"][0]["text"])
    assert len(payload["results"]) == 4

    out = rpc(
        agent,
        url,
        "tools/call",
        {"name": "submit_answer", "arguments": {"answer": "no_meaningful_difference", "confidence": 0.7}},
        4,
    )
    assert json.loads(out["result"]["content"][0]["text"]) == {"received": True}

    summary = control.get(f"/episodes/{eid}").json()
    assert summary["result"]["score"] == 1.0
    assert summary["calls"] == 1


@pytest.mark.parametrize("transport", ["mcp", "http"])
@pytest.mark.parametrize(
    ("answers", "score"),
    [
        (["paper_better"], 0.0),
        (["no_meaningful_difference"], 1.0),
        (["maybe", "paper_better"], 0.0),
        (["maybe", "no_meaningful_difference"], 1.0),
        (["maybe", "still maybe"], 0.0),
    ],
)
def test_submission_receipts_keep_grades_private(server_pair, transport, answers, score):
    control, agent, _ = server_pair
    created = control.post("/episodes", json={"env": "toy", "task": "claim", "seed": 4}).json()
    eid, url = created["id"], created["mcp_url"]
    receipt = {"received": True} if transport == "mcp" else {"done": True}

    def submit(answer, confidence):
        arguments = {"answer": answer, "confidence": confidence}
        if transport == "http":
            response = control.post(f"/episodes/{eid}/submit", json=arguments)
            assert response.status_code == 200
            return response.json()
        result = rpc(agent, url, "tools/call", {"name": "submit_answer", "arguments": arguments})["result"]
        payload = json.loads(result["content"][0]["text"])
        if "structuredContent" in result:
            assert result["structuredContent"] == payload
        return payload

    for index, answer in enumerate(answers):
        payload = submit(answer, 0.8)
        if index < len(answers) - 1:
            assert set(payload) == {"error", "hint"}
            assert control.get(f"/episodes/{eid}").json()["result"] is None
        else:
            assert payload == receipt

    saved = control.get(f"/episodes/{eid}").json()
    assert saved["result"]["score"] == score
    assert saved["result"]["confidence"] == 0.8
    submissions = [entry for entry in saved["log"] if entry["action"] == "submit"]
    assert len(submissions) == 1
    assert submissions[0]["answer"] == answers[-1]

    for answer in ["paper_better", "no_meaningful_difference", "maybe"]:
        assert submit(answer, 0.01) == receipt
        assert control.get(f"/episodes/{eid}").json() == saved


def test_hosts_and_origins_require_explicit_allowance(tmp_path):
    toy_env(tmp_path)
    hub = Hub(tmp_path)
    ep = hub.create("toy", "claim")
    path = urlsplit(hub.agent_url(ep)).path
    init = {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    }
    cases = [
        ("https://demo.trycloudflare.com", {}, ["demo.trycloudflare.com"], [], 421),
        ("http://127.0.0.1:8788", {"origin": "app://qualia"}, [], ["app://qualia"], 403),
    ]
    for base_url, extra, hosts, origins, blocked in cases:
        for allowed, status in [(False, blocked), (True, 200)]:
            app, _ = build_app(hub, allowed_hosts=hosts if allowed else [], allowed_origins=origins if allowed else [])
            with TestClient(app, base_url=base_url) as client:
                response = client.post(
                    path,
                    headers={**MCP_HEADERS, **extra},
                    json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": init},
                )
                assert response.status_code == status
