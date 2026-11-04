import json
import secrets
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from benchtop.agents.common import Client, LabClient
from benchtop.cli import main
from benchtop.env import ROOT
from benchtop.integration import begin, collect, prepare
from tests.test_server import rpc


def connections(server_pair):
    control, agent, hub = server_pair
    client = Client.__new__(Client)
    client.base = "http://127.0.0.1:8787"
    client.http = control
    lab = LabClient.__new__(LabClient)
    lab.http = agent
    return client, lab, hub


def setup_episode(server_pair, tmp_path):
    client, lab, hub = connections(server_pair)
    result = prepare(client, env="toy", task="claim", root=tmp_path, lab=lab)
    return result, client, lab, hub


def test_preparation_is_private_and_does_not_start_or_spend(server_pair, tmp_path):
    result, client, lab, hub = setup_episode(server_pair, tmp_path)
    saved = json.loads((tmp_path / "episodes" / f"{result['episode']}.json").read_text())
    assert saved["mcpServers"]["benchtop"]["url"] == lab.url
    assert result["preflight"]["status"] == "passed"
    assert result["preflight"]["tools"] == ["describe_lab", "run_experiments", "submit_answer"]
    assert result["model_sessions_started_by_preparation"] == result["integration_starts_recorded"] == 0
    assert not result["qualia_verified"]
    assert not (tmp_path / "starts.jsonl").exists()
    assert (tmp_path / "episodes" / f"{result['episode']}.json").stat().st_mode & 0o777 == 0o600
    assert lab.url not in json.dumps(result)
    assert hub.admin_token not in json.dumps(saved)
    summary = client.summary(result["episode"])
    assert summary["calls"] == summary["spent"] == 0
    assert summary["result"] is None
    assert client.mcp_trace(result["episode"])["total"] == 1


def test_preflight_and_admin_calls_cannot_pass_as_qualia_evidence(server_pair, tmp_path):
    result, client, _, _ = setup_episode(server_pair, tmp_path)
    episode = result["episode"]
    with pytest.raises(ValueError, match="record the integration start"):
        collect(client, episode, root=tmp_path, output=tmp_path / "proof.json")
    begin(client, episode, root=tmp_path, client_name="qualia-desktop", requested_model="fixture")
    first = collect(client, episode, client_observed=True, root=tmp_path, output=tmp_path / "proof.json")
    assert not first["qualia_verified"]
    client.run(episode, [{"ctx": 2}])
    client.submit(episode, "no_meaningful_difference")
    proof = collect(client, episode, client_observed=True, root=tmp_path, output=tmp_path / "proof.json")
    assert not proof["transport_complete"]
    assert proof["missing_successful_tools"] == ["describe_lab", "run_experiments", "submit_answer"]
    assert proof["status"] == "incomplete"


@pytest.mark.parametrize("truncated", [False, True])
def test_capture_uses_mcp_events_and_explicit_operator_attestation(server_pair, tmp_path, truncated):
    result, client, lab, hub = setup_episode(server_pair, tmp_path)
    episode = result["episode"]
    begin(client, episode, root=tmp_path, client_name="qualia-desktop", requested_model="fixture")
    if truncated:
        for _ in range(280):
            hub.record_mcp(hub.get(episode), "describe_lab", {})
    for name, arguments in [
        ("describe_lab", {}), ("run_experiments", {"configs": [{"ctx": 2}]}),
        ("submit_answer", {"answer": "no_meaningful_difference"}),
    ]:
        assert "result" in rpc(lab.http, lab.url, "tools/call", {"name": name, "arguments": arguments})
    pending = collect(client, episode, root=tmp_path, output=tmp_path / "proof.json")
    assert pending["transport_complete"] != truncated
    assert not pending["qualia_verified"]
    proof = collect(client, episode, client_observed=True, root=tmp_path, output=tmp_path / "proof.json")
    assert proof["status"] == ("incomplete" if truncated else "passed")
    assert proof["client_evidence"]["source"] == "operator_attestation"
    assert proof["client_evidence"]["reported_model"] == "not_reported"
    assert proof["result"]["score"] == 1.0
    assert len(proof["mcp_events"]) == (256 if truncated else 3)
    assert proof["mcp_events"][-1]["receipt_only"]
    encoded = json.dumps(proof)
    assert lab.url not in encoded and hub.admin_token not in encoded
    assert "mcp_url" not in encoded and "admin_token" not in encoded
    with pytest.raises(ValueError, match="under artifacts"):
        collect(client, episode, root=tmp_path, output=ROOT / "results/summary.json")


def test_start_ledger_enforces_duplicate_and_concurrent_reserve_limits(server_pair, tmp_path):
    client, lab, _ = connections(server_pair)
    episodes = [prepare(client, env="toy", task="claim", root=tmp_path, lab=lab)["episode"] for _ in range(3)]

    def start(episode):
        try:
            return begin(client, episode, root=tmp_path, client_name="qualia-desktop", requested_model="fixture")
        except ValueError as error:
            return str(error)

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(start, episodes))
    assert sum(isinstance(result, dict) for result in results) == 2
    assert any("reserve is exhausted" in result for result in results if isinstance(result, str))
    starts = [json.loads(line) for line in (tmp_path / "starts.jsonl").read_text().splitlines()]
    assert len(starts) == 2
    assert (tmp_path / "starts.jsonl").stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="already has a recorded"):
        begin(client, starts[0]["episode"], root=tmp_path, client_name="qualia-desktop", requested_model="fixture")


@pytest.mark.parametrize("admin_action", ["run", "submit"])
def test_admin_activity_cannot_be_misattributed_to_the_client(server_pair, tmp_path, admin_action):
    result, client, lab, _ = setup_episode(server_pair, tmp_path)
    episode = result["episode"]
    begin(client, episode, root=tmp_path, client_name="qualia-desktop", requested_model="fixture")
    lab.call("describe_lab")
    lab.call("run_experiments", {"configs": [{"ctx": 2}]})
    if admin_action == "run":
        client.run(episode, [{"ctx": 2}])
    else:
        client.submit(episode, "no_meaningful_difference")
    lab.call("submit_answer", {"answer": "no_meaningful_difference"})
    proof = collect(client, episode, client_observed=True, root=tmp_path, output=tmp_path / "proof.json")
    assert not proof["all_experiments_and_final_submission_via_mcp"]
    assert not proof["qualia_verified"]


def test_cloud_start_rejects_loopback_and_used_episodes(server_pair, tmp_path):
    result, client, _, _ = setup_episode(server_pair, tmp_path)
    with pytest.raises(ValueError, match="publicly reachable HTTPS"):
        begin(client, result["episode"], root=tmp_path, client_name="qualia-cloud", requested_model="fixture")
    client.run(result["episode"], [{"ctx": 2}])
    with pytest.raises(ValueError, match="before invoking"):
        begin(client, result["episode"], root=tmp_path, client_name="qualia-desktop", requested_model="fixture")
    assert not (tmp_path / "starts.jsonl").exists()


def test_mcp_trace_is_bounded_admin_only_and_evicted_with_episode(server_pair, tmp_path, monkeypatch):
    result, client, _, hub = setup_episode(server_pair, tmp_path)
    episode = hub.get(result["episode"])
    for _ in range(300):
        hub.record_mcp(episode, "describe_lab", {})
    trace = client.mcp_trace(episode.id)
    assert trace["total"] == 301 and len(trace["events"]) == 256
    control, agent, _ = server_pair
    path = f"/episodes/{episode.id}/mcp-trace"
    assert agent.get(path).status_code == 404
    assert control.get(path, headers={"authorization": "invalid"}).status_code == 401
    monkeypatch.setattr("benchtop.server.MAX_EPISODES", 1)
    client.create("toy", "claim")
    assert episode.id not in hub.mcp_traces


def test_cli_errors_redact_the_capability(monkeypatch):
    token = secrets.token_urlsafe(32)
    url = f"https://example.test/e/123456abcdef/{token}/mcp"
    monkeypatch.setattr("benchtop.agents.common.Client", lambda *_: SimpleNamespace(http=SimpleNamespace(close=lambda: None)))

    def failed(*args, **kwargs):
        raise RuntimeError(f"cannot connect to {url}")

    monkeypatch.setattr("benchtop.integration.prepare", failed)
    with pytest.raises(SystemExit) as error:
        main(["integration", "prepare"])
    assert token not in str(error.value)
    assert "[redacted]" in str(error.value)
