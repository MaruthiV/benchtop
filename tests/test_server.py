import json

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
