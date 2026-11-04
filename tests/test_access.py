import json
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest
from starlette.testclient import TestClient

from benchtop import access
from benchtop.agents import common
from benchtop.agents.common import Client, LabClient
from benchtop.server import HEADER, Hub, build_app
from tests.test_core import toy_env
from tests.test_server import MCP_HEADERS, rpc


def test_agent_interface_has_no_administrative_routes(tmp_path):
    toy_env(tmp_path)
    app, hub = build_app(Hub(tmp_path))
    episode = hub.create("toy", "claim")
    with TestClient(app, base_url="http://127.0.0.1:8788") as client:
        for path in ["/episodes", "/envs", "/health", f"/episodes/{episode.id}"]:
            assert client.get(path).status_code == 404
        assert client.post("/episodes", json={"env": "toy", "task": "claim"}).status_code == 404


def test_episode_id_alone_is_not_agent_authorization(tmp_path):
    toy_env(tmp_path)
    app, hub = build_app(Hub(tmp_path))
    episode = hub.create("toy", "claim")
    with TestClient(app, base_url="http://127.0.0.1:8788") as client:
        response = client.post(
            f"/e/{episode.id}/mcp",
            headers=MCP_HEADERS,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        assert response.status_code in (401, 404)


def test_unscoped_mcp_rejects_a_forged_episode_header(tmp_path):
    toy_env(tmp_path)
    app, hub = build_app(Hub(tmp_path))
    episode = hub.create("toy", "claim")
    with TestClient(app, base_url="http://127.0.0.1:8788") as client:
        response = client.post(
            "/mcp",
            headers={**MCP_HEADERS, HEADER: episode.id},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "describe_lab", "arguments": {}}},
        )
        assert response.status_code in (401, 404)


def test_capabilities_are_scoped_to_one_episode(server_pair):
    control, agent, hub = server_pair
    first = control.post("/episodes", json={"env": "toy", "task": "claim"}).json()
    second = control.post("/episodes", json={"env": "toy", "task": "claim"}).json()
    first_path = urlsplit(first["mcp_url"]).path
    second_token = hub.capabilities[second["id"]]
    wrong_paths = [
        first_path.replace(hub.capabilities[first["id"]], second_token),
        first_path.replace(hub.capabilities[first["id"]], "x" * 43),
        first_path.replace(first["id"], "0" * 12 if first["id"] != "0" * 12 else "1" * 12),
    ]
    for path in wrong_paths:
        response = agent.post(path, headers=MCP_HEADERS, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        assert response.status_code == 401
        assert response.json() == {"error": "unauthorized"}
    assert len(hub.episodes) == 2
    assert all(ep.calls == 0 and ep.spent == 0 for ep in hub.episodes.values())


def test_forged_header_cannot_retarget_a_valid_capability(server_pair):
    control, agent, hub = server_pair
    first = control.post("/episodes", json={"env": "toy", "task": "claim"}).json()
    second = control.post("/episodes", json={"env": "toy", "task": "claim"}).json()
    headers = {HEADER: second["id"]}
    result = rpc(agent, first["mcp_url"], "tools/call", {"name": "describe_lab", "arguments": {}}, headers=headers)
    assert json.loads(result["result"]["content"][0]["text"])["episode"] == first["id"]
    rpc(agent, first["mcp_url"], "tools/call", {"name": "run_experiments", "arguments": {"configs": [{"ctx": 2}]}}, headers=headers)
    assert hub.get(first["id"]).calls == 1
    assert hub.get(second["id"]).calls == 0
    assert not hub.get(second["id"]).done


def test_admin_requires_separate_credentials_and_local_request_context(server_pair):
    control, agent, hub = server_pair
    created = control.post("/episodes", json={"env": "toy", "task": "claim"}).json()
    path = f"/episodes/{created['id']}"
    for token in ["", "wrong", hub.capabilities[created["id"]]]:
        response = control.get(path, headers={"authorization": f"Bearer {token}"})
        assert response.status_code == 401
        assert response.json() == {"error": "unauthorized"}
    assert control.get(path, headers={"origin": "https://untrusted.example"}).status_code == 403
    assert control.get(path, headers={"host": "untrusted.example"}).status_code == 421
    assert control.get(path, headers={"host": "["}).status_code == 421
    assert control.get(path, headers=[(b"authorization", b"Bearer \xff")]).status_code == 401
    assert control.get(urlsplit(created["mcp_url"]).path).status_code == 404
    assert agent.get(path, headers={"authorization": f"Bearer {hub.admin_token}"}).status_code == 404
    assert control.get(path).status_code == 200


def test_eviction_revokes_the_old_episode_capability(server_pair, monkeypatch):
    from benchtop import server

    control, agent, hub = server_pair
    monkeypatch.setattr(server, "MAX_EPISODES", 1)
    first = control.post("/episodes", json={"env": "toy", "task": "claim"}).json()
    second = control.post("/episodes", json={"env": "toy", "task": "claim"}).json()
    assert first["id"] not in hub.capabilities
    assert agent.post(first["mcp_url"], headers=MCP_HEADERS, json={}).status_code == 401
    assert rpc(agent, second["mcp_url"], "tools/list", {})["result"]["tools"]


def test_new_episode_cannot_overwrite_an_existing_capability(server_pair, monkeypatch):
    from benchtop import server

    _, agent, hub = server_pair
    first = hub.create("toy", "claim")
    original_url = hub.agent_url(first)
    episode_class = server.Episode

    def collision(*args, **kwargs):
        episode = episode_class(*args, **kwargs)
        episode.id = first.id
        return episode

    monkeypatch.setattr(server, "Episode", collision)
    second = hub.create("toy", "claim")
    assert second.id != first.id
    assert hub.get(first.id) is first
    assert rpc(agent, original_url, "tools/list", {})["result"]["tools"]


def test_custom_transport_rules_cannot_disable_host_protection(tmp_path):
    for options in [{"allowed_hosts": ["*"]}, {"allowed_origins": ["*"]}]:
        with pytest.raises(ValueError, match="explicit"):
            build_app(Hub(tmp_path), **options)


def test_published_records_redact_capabilities_and_admin_credentials(server_pair, tmp_path, monkeypatch):
    control, _, hub = server_pair
    created = control.post("/episodes", json={"env": "toy", "task": "claim"}).json()
    monkeypatch.setattr(common, "RESULTS", tmp_path / "results")
    common.record({
        "error": f"failed at {created['mcp_url']}",
        "nested": [{"mcp_url": created["mcp_url"], "admin_token": hub.admin_token}],
        "headers": {"Authorization": f"Bearer {hub.admin_token}"},
    })
    saved = (common.RESULTS / "rollouts.jsonl").read_text()
    assert hub.capabilities[created["id"]] not in saved
    assert hub.admin_token not in saved
    assert "[redacted]" in saved


def test_local_profile_is_private_and_never_sent_to_another_origin(tmp_path, monkeypatch):
    monkeypatch.setattr(access, "STATE", tmp_path)
    token = secrets.token_urlsafe(32)
    path = access.write_profile("http://127.0.0.1:8787", "http://127.0.0.1:8788", token)
    assert path.stat().st_mode & 0o777 == 0o600
    assert access.admin_headers("http://127.0.0.1:8787")["authorization"] == f"Bearer {token}"
    with pytest.raises(ValueError, match="origin"):
        access.admin_headers("http://localhost:8787")
    with pytest.raises(ValueError, match="loopback"):
        access.admin_headers("https://untrusted.example:8787")


@pytest.mark.parametrize("backend", ["claude", "codex"])
def test_model_adapters_receive_only_the_episode_capability(server_pair, tmp_path, monkeypatch, backend):
    import importlib

    control, agent, hub = server_pair
    module = importlib.import_module(f"benchtop.agents.{backend}")
    created = []

    class Evaluator:
        def create(self, env, task, **options):
            ep = control.post("/episodes", json={"env": env, "task": task, **options}).json()
            created.append(ep)
            return ep

        def summary(self, eid):
            return control.get(f"/episodes/{eid}").json()

    def fake_model(cmd, cwd, timeout):
        ep = created[-1]
        if backend == "claude":
            cfg = json.loads(Path(cmd[cmd.index("--mcp-config") + 1]).read_text())
            assert cfg["mcpServers"]["benchtop"]["url"] == ep["mcp_url"]
            assert hub.admin_token not in json.dumps(cfg)
        else:
            assert f'mcp_servers.benchtop.url={json.dumps(ep["mcp_url"])}' in cmd
        assert hub.admin_token not in json.dumps(cmd)
        rpc(agent, ep["mcp_url"], "tools/call", {"name": "submit_answer", "arguments": {"answer": "no_meaningful_difference"}})
        return json.dumps({"type": "result", "is_error": False}), "", False

    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    monkeypatch.setattr(module, "run_cmd", fake_model)
    monkeypatch.setattr(module, "transcript_dir", lambda _: transcripts)
    row = module.rollout(Evaluator(), "toy", "claim")
    assert row["submitted"] and row["score"] == 1.0


def test_real_server_separates_ports_and_serves_scoped_mcp(tmp_path):
    toy_env(tmp_path)
    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        admin_port, agent_port = first.getsockname()[1], second.getsockname()[1]
    base = f"http://127.0.0.1:{admin_port}"
    agent_base = f"http://127.0.0.1:{agent_port}"
    profile = access.profile_path(base)
    process = subprocess.Popen(
        [sys.executable, "-m", "benchtop.cli", "serve", "--port", str(admin_port), "--agent-port", str(agent_port), "--envs-dir", str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError("server exited during startup: " + process.communicate()[1])
            if profile.exists():
                try:
                    if httpx.get(base + "/health", headers=access.admin_headers(base), timeout=0.5).status_code == 200:
                        break
                except httpx.TransportError:
                    pass
            time.sleep(0.05)
        else:
            raise AssertionError("server did not become ready")
        assert profile.stat().st_mode & 0o777 == 0o600
        control = Client(base)
        ep = control.create("toy", "claim")
        assert ep["mcp_url"].startswith(agent_base + "/e/")
        lab = LabClient(ep["mcp_url"])
        assert lab.call("describe_lab")["episode"] == ep["id"]
        assert "error" in lab.call("run_experiments", {"configs": "not a list"})
        assert lab.call("run_experiments", {"configs": [{"ctx": 2}], "seeds": 1})["results"]
        assert httpx.get(agent_base + f"/episodes/{ep['id']}", headers={"host": f"127.0.0.1:{admin_port}"}).status_code == 404
        assert httpx.get(base + f"/episodes/{ep['id']}").status_code == 401
        assert lab.call("submit_answer", {"answer": "no_meaningful_difference"}) == {"received": True}
        assert control.summary(ep["id"])["result"]["score"] == 1.0
        lab.http.close()
        control.http.close()
    finally:
        process.terminate()
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
    assert not profile.exists()
