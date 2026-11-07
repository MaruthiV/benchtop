import json
import os
import shutil
import subprocess
from pathlib import Path

from benchtop import study
from benchtop.calibrate import canonical_hash

TOOLS = ["mcp__benchtop__describe_lab", "mcp__benchtop__run_experiments", "mcp__benchtop__submit_answer"]
FLAGS = ("--print", "--model", "--system-prompt", "--tools", "--allowedTools", "--strict-mcp-config",
         "--mcp-config", "--permission-prompts", "--output-format", "--verbose", "--no-session-persistence",
         "--restricted", "--setting-sources", "--disable-slash-commands", "--settings")
OVERRIDES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_OAUTH_TOKEN",
             "CLAUDE_CONFIG_DIR", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY", "ANTHROPIC_MODEL")


def probe():
    from benchtop.runner import _digest

    binary = shutil.which("claude")
    if binary is None:
        raise ValueError("Claude CLI is unavailable")
    binary = str(Path(binary).resolve())
    def read(arguments):
        result = subprocess.run([binary, *arguments], capture_output=True, text=True, timeout=20)
        if result.returncode:
            raise ValueError("Claude CLI metadata command failed: " + arguments[0])
        return result.stdout
    version, help_text = read(["--version"]).strip(), read(["--help"])
    auth = json.loads(read(["auth", "status", "--json"]))
    safe_auth = {key: auth.get(key) for key in ("loggedIn", "authMethod", "apiProvider", "subscriptionType")}
    managed = [str(path) for path in (
        Path("/Library/Application Support/ClaudeCode/managed-settings.json"),
        Path("/Library/Application Support/ClaudeCode/managed-mcp.json"),
        Path("/Library/Managed Preferences/com.anthropic.claudecode.plist"),
        Path("/etc/claude-code/managed-settings.json"), Path("/etc/claude-code/managed-mcp.json")) if path.exists()]
    data = {"schema_version": 1, "kind": "claude_cli_preflight", "binary": binary,
            "binary_sha256": _digest(binary), "cli_version": version, "help_sha256": canonical_hash(help_text),
            "flags": {flag: flag in help_text for flag in FLAGS}, "auth": safe_auth,
            "account_sha256": canonical_hash({key: auth[key] for key in ("email", "orgId")})
            if all(isinstance(auth.get(key), str) and auth[key] for key in ("email", "orgId")) else None,
            "overrides": [name for name in OVERRIDES if os.environ.get(name)], "managed_policy_files": managed,
            "model_sessions": 0, "scope": "CLI/auth metadata only; no model availability, billing-limit or successful-smoke claim."}
    return study._seal(data, "preflight_sha256")


def verify(preflight, model, *, live=True):
    study._verify(preflight, "preflight_sha256")
    study._model(model)
    if (preflight.get("kind") != "claude_cli_preflight" or preflight.get("model_sessions") != 0
            or preflight.get("cli_version") != model["cli_version"] or not all(preflight["flags"].get(flag) is True for flag in FLAGS)
            or preflight["auth"].get("loggedIn") is not True or preflight["auth"].get("authMethod") != "claude.ai"
            or preflight["auth"].get("apiProvider") != "firstParty" or preflight["auth"].get("subscriptionType") not in ("pro", "max")
            or not isinstance(preflight.get("account_sha256"), str)
            or len(preflight["account_sha256"]) != 64 or preflight.get("overrides") or preflight.get("managed_policy_files")):
        raise ValueError("provider requires pinned CLI flags, personal subscription auth, and no provider or managed-policy overrides")
    if any(control["status"] != "unavailable" for control in model["decoding"].values()):
        raise ValueError("this CLI adapter cannot apply fixed temperature/model-seed controls; declare them unavailable")
    if live and probe() != preflight:
        raise ValueError("CLI binary/version/authentication/environment changed since authorization")
    return preflight


def environment():
    if any(os.environ.get(name) for name in OVERRIDES):
        raise ValueError("provider environment overrides are not permitted")
    clean = {key: value for key, value in os.environ.items() if key in {
        "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "LANG", "LC_ALL", "TERM", "NO_COLOR"}}
    return {**clean, "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1", "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "ENABLE_TOOL_SEARCH": "false", "DISABLE_AUTOUPDATER": "1"}


def command(work, plan, preflight):
    from benchtop.agents.common import first_message

    verify(preflight, plan["binding"]["model_configuration"])
    connection = study._json(work / "connection.json")
    config = work / "mcp.json"
    study._publish(config, {"mcpServers": {"benchtop": {"type": "http", "url": connection["mcp_url"], "alwaysLoad": True}}})
    settings = work / "settings.json"
    study._publish(settings, {"disableAllHooks": True})
    return [preflight["binary"], "--print", first_message(connection["description"], connection["budget"]),
            "--model", plan["binding"]["model_configuration"]["requested_model_id"],
            "--system-prompt", plan["binding"]["prompt"]["system_text"], "--tools", "",
            "--strict-mcp-config", "--mcp-config", str(config), "--allowedTools", *TOOLS,
            "--permission-prompts", "none", "--output-format", "stream-json", "--verbose",
            "--no-session-persistence", "--restricted", "--setting-sources", "", "--disable-slash-commands",
            "--settings", str(settings)]


def parse_stream(text):
    models, final, init, synthetic, init_count = [], None, None, False, 0
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        synthetic |= event.get("synthetic_control") is True
        if event.get("type") == "system" and event.get("subtype") == "init":
            init_count += 1
            init = event
            if isinstance(event.get("model"), str):
                models.append(event["model"])
        if event.get("type") == "assistant" and isinstance(event.get("message"), dict) and isinstance(event["message"].get("model"), str):
            models.append(event["message"]["model"])
        if event.get("type") == "result":
            final = event
            if isinstance(event.get("modelUsage"), dict):
                models.extend(event["modelUsage"])
    init = init or {}
    servers = init.get("mcp_servers")
    isolation = (not synthetic and init_count == 1 and isinstance(init.get("model"), str)
                 and isinstance(init.get("tools"), list) and all(isinstance(tool, str) for tool in init["tools"])
                 and len(init["tools"]) == len(TOOLS) and set(init["tools"]) == set(TOOLS)
                 and isinstance(servers, list) and len(servers) == 1 and isinstance(servers[0], dict) and servers[0].get("name") == "benchtop"
                 and servers[0].get("status") == "connected")
    return {"models": list(dict.fromkeys(models)), "final": final, "isolation_valid": isolation,
            "synthetic_control": synthetic, "init_observed": init_count > 0,
            "init_tools": init.get("tools"), "mcp_servers": servers}


def execute(work, timeout, plan, preflight):
    from benchtop.runner import execute_command

    return execute_command(work, timeout, command(work, plan, preflight), executor="claude_cli",
                           parser=parse_stream, env=environment())
