import copy
import json
import subprocess
from types import SimpleNamespace

import pytest

from benchtop import provider, runner, study
from benchtop.cli import main
from tests.test_study import model


def preflight():
    return study._seal({"schema_version": 1, "kind": "claude_cli_preflight", "binary": "/fixture/claude",
        "binary_sha256": "0" * 64, "cli_version": model()["cli_version"], "help_sha256": "1" * 64,
        "flags": dict.fromkeys(provider.FLAGS, True),
        "auth": {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty", "subscriptionType": "pro"},
        "account_sha256": "2" * 64, "overrides": [], "managed_policy_files": [], "model_sessions": 0,
        "scope": "TEST FIXTURE ONLY; no CLI login or model invocation."}, "preflight_sha256")


def stream_events(*, model_id=None, tools=None, final_error=False):
    model_id = model_id or model()["reported_model_id"]
    return [{"type": "system", "subtype": "init", "model": model_id,
             "tools": provider.TOOLS if tools is None else tools,
             "mcp_servers": [{"name": "benchtop", "status": "connected"}], "fixture_only": True},
            {"type": "assistant", "message": {"model": model_id}},
            {"type": "result", "is_error": final_error,
             "modelUsage": {model_id: {"inputTokens": 12, "outputTokens": 5}}, "fixture_only": True}]


def encode(events):
    return "\n".join(json.dumps(row) for row in events) + "\n"


def test_metadata_probe_never_invokes_inference_or_exposes_account_fields(tmp_path, monkeypatch):
    binary = tmp_path / "claude"
    binary.write_text("fixture binary")
    calls = []

    def metadata(command, **kwargs):
        calls.append(command[1:])
        values = {("--version",): model()["cli_version"], ("--help",): " ".join(provider.FLAGS),
                  ("auth", "status", "--json"): json.dumps({"loggedIn": True, "authMethod": "claude.ai",
                      "apiProvider": "firstParty", "subscriptionType": "pro", "email": "fixture@example.test",
                      "orgId": "fixture-org", "secret_extra_field": "never-save-this"})}
        return SimpleNamespace(returncode=0, stdout=values[tuple(command[1:])])

    monkeypatch.setattr(provider.shutil, "which", lambda name: str(binary))
    monkeypatch.setattr(provider.subprocess, "run", metadata)
    proof = provider.probe()
    assert calls == [["--version"], ["--help"], ["auth", "status", "--json"]]
    assert proof["model_sessions"] == 0 and len(proof["account_sha256"]) == 64
    assert "fixture@example.test" not in json.dumps(proof) and "never-save-this" not in json.dumps(proof)


@pytest.mark.parametrize("defect", ["api", "logged_out", "cli", "flag", "overrides", "managed", "account"])
def test_provider_refuses_unverified_or_potentially_billed_configuration(defect):
    proof = preflight()
    if defect == "api":
        proof["auth"]["authMethod"] = "api_key"
    elif defect == "logged_out":
        proof["auth"]["loggedIn"] = False
    elif defect == "cli":
        proof["cli_version"] = "different"
    elif defect == "flag":
        proof["flags"]["--restricted"] = False
    elif defect == "account":
        proof["account_sha256"] = None
    else:
        proof["managed_policy_files" if defect == "managed" else "overrides"] = ["fixture"]
    proof = study._seal(proof, "preflight_sha256")
    with pytest.raises(ValueError, match="provider requires"):
        provider.verify(proof, model(), live=False)


def test_provider_checks_live_account_binding_and_unsupported_decoding(monkeypatch):
    proof = preflight()
    changed = study._seal({**proof, "account_sha256": "3" * 64}, "preflight_sha256")
    monkeypatch.setattr(provider, "probe", lambda: changed)
    with pytest.raises(ValueError, match="changed since authorization"):
        provider.verify(proof, model())
    config = model()
    config["decoding"]["temperature"] = {"status": "fixed", "value": 0.7}
    with pytest.raises(ValueError, match="cannot apply fixed"):
        provider.verify(proof, config, live=False)


@pytest.mark.parametrize("defect", ["extra_tool", "extra_server", "synthetic", "duplicate_init", "no_init", "malformed"])
def test_stream_isolation_refuses_extra_or_unverified_capabilities(defect):
    events = stream_events()
    if defect == "extra_tool":
        events[0]["tools"] = [*provider.TOOLS, "Bash"]
    elif defect == "extra_server":
        events[0]["mcp_servers"].append({"name": "other", "status": "connected"})
    elif defect == "synthetic":
        events[-1]["synthetic_control"] = True
    elif defect == "duplicate_init":
        events.insert(0, copy.deepcopy(events[0]))
    elif defect == "no_init":
        events.pop(0)
    else:
        events[0]["tools"] = [{}]
        events[1]["message"] = None
    assert provider.parse_stream(encode(events))["isolation_valid"] is False


def test_command_uses_only_explicit_mcp_and_disables_discovered_customizations(tmp_path, monkeypatch):
    proof = preflight()
    monkeypatch.setattr(provider, "probe", lambda: proof)
    connection = {"mcp_url": "http://127.0.0.1:1/capability", "description": {"question": "fixture"}, "budget": {"total": 1}}
    study._publish(tmp_path / "connection.json", connection)
    argv = provider.command(tmp_path, {"binding": {"model_configuration": model(), "prompt": {"system_text": "fixture system"}}}, proof)
    assert argv[0] == proof["binary"] and argv[argv.index("--tools") + 1] == ""
    assert all(flag in argv for flag in ("--restricted", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"))
    assert argv[argv.index("--setting-sources") + 1] == "" and "--bare" not in argv and "--safe-mode" not in argv
    assert json.loads((tmp_path / "mcp.json").read_text()) == {"mcpServers": {"benchtop": {
        "type": "http", "url": connection["mcp_url"], "alwaysLoad": True}}}
    assert json.loads((tmp_path / "settings.json").read_text()) == {"disableAllHooks": True}
    monkeypatch.setenv("CLAUDE_CODE_SAFE_MODE", "1")
    env = provider.environment()
    assert "CLAUDE_CODE_SAFE_MODE" not in env and "CLAUDE_CODE_SIMPLE" not in env
    assert env["ENABLE_TOOL_SEARCH"] == "false" and env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1" and env["DISABLE_AUTOUPDATER"] == "1"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-never-use")
    with pytest.raises(ValueError, match="environment overrides"):
        provider.environment()


def test_shared_executor_owns_timeout_and_passes_the_requested_environment(tmp_path):
    result = runner.execute_command(tmp_path, 0.05, ["/bin/sleep", "30"], executor="fixture_only",
                                    parser=provider.parse_stream, env={"PATH": "/bin"})
    assert result["timed_out"] and result["process_stopped"] and result["model_sessions"] == 0
    assert result["returncode"] != 0


def test_preflight_cli_is_metadata_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(provider, "probe", preflight)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("preflight launched a model"))
    output = tmp_path / "metadata.json"
    main(["study", "provider-preflight", "--output", str(output)])
    assert json.loads(capsys.readouterr().out)["model_sessions"] == 0
    assert study._json(output) == preflight()
