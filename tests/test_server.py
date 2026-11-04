import json

import pytest
from starlette.testclient import TestClient

from benchtop.server import Hub, build_app
from tests.test_core import toy_env

MCP_HEADERS = {"accept": "application/json, text/event-stream", "mcp-protocol-version": "2025-06-18"}


def rpc(client, eid, method, params, i=1):
    r = client.post(
        f"/e/{eid}/mcp",
        headers=MCP_HEADERS,
        json={"jsonrpc": "2.0", "id": i, "method": method, "params": params},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_mcp_episode_roundtrip(tmp_path):
    toy_env(tmp_path)
    app, hub = build_app(Hub(tmp_path))
    with TestClient(app, base_url="http://127.0.0.1:8787") as client:
        created = client.post("/episodes", json={"env": "toy", "task": "claim", "seed": 4}).json()
        eid = created["id"]
        rpc(
            client,
            eid,
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        )
        tools = rpc(client, eid, "tools/list", {}, 2)["result"]["tools"]
        assert {t["name"] for t in tools} == {"describe_lab", "run_experiments", "submit_answer"}

        out = rpc(
            client,
            eid,
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
            client,
            eid,
            "tools/call",
            {"name": "submit_answer", "arguments": {"answer": "no_meaningful_difference", "confidence": 0.7}},
            4,
        )
        assert json.loads(out["result"]["content"][0]["text"]) == {"received": True}

        summary = client.get(f"/episodes/{eid}").json()
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
def test_submission_receipts_keep_grades_private(tmp_path, transport, answers, score):
    toy_env(tmp_path)
    app, _ = build_app(Hub(tmp_path))
    with TestClient(app, base_url="http://127.0.0.1:8787") as client:
        eid = client.post("/episodes", json={"env": "toy", "task": "claim", "seed": 4}).json()["id"]
        receipt = {"received": True} if transport == "mcp" else {"done": True}

        def submit(answer, confidence):
            arguments = {"answer": answer, "confidence": confidence}
            if transport == "http":
                response = client.post(f"/episodes/{eid}/submit", json=arguments)
                assert response.status_code == 200
                return response.json()
            result = rpc(client, eid, "tools/call", {"name": "submit_answer", "arguments": arguments})["result"]
            payload = json.loads(result["content"][0]["text"])
            if "structuredContent" in result:
                assert result["structuredContent"] == payload
            return payload

        for index, answer in enumerate(answers):
            payload = submit(answer, 0.8)
            if index < len(answers) - 1:
                assert set(payload) == {"error", "hint"}
                assert client.get(f"/episodes/{eid}").json()["result"] is None
            else:
                assert payload == receipt

        saved = client.get(f"/episodes/{eid}").json()
        assert saved["result"]["score"] == score
        assert saved["result"]["confidence"] == 0.8
        submissions = [entry for entry in saved["log"] if entry["action"] == "submit"]
        assert len(submissions) == 1
        assert submissions[0]["answer"] == answers[-1]

        for answer in ["paper_better", "no_meaningful_difference", "maybe"]:
            assert submit(answer, 0.01) == receipt
            assert client.get(f"/episodes/{eid}").json() == saved


def test_any_host_lets_tunnels_and_app_origins_in(tmp_path):
    toy_env(tmp_path)
    init = {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    }
    # a tunnel's host header, then a desktop app's origin header
    cases = [
        ("https://demo.trycloudflare.com", {}, 421),
        ("http://127.0.0.1:8787", {"origin": "app://qualia"}, 403),
    ]
    for base_url, extra, blocked in cases:
        for any_host, status in [(False, blocked), (True, 200)]:
            app, _ = build_app(Hub(tmp_path), any_host=any_host)
            with TestClient(app, base_url=base_url) as client:
                eid = client.post("/episodes", json={"env": "toy", "task": "claim", "seed": 4}).json()["id"]
                body = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": init}
                r = client.post(f"/e/{eid}/mcp", headers={**MCP_HEADERS, **extra}, json=body)
                assert r.status_code == status, (base_url, any_host, r.text)
