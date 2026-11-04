import json
import os
import re
import tempfile
from urllib.parse import urlsplit

from benchtop.env import STATE

CAPABILITY = re.compile(r"(/e/[0-9a-f]{12}/)[A-Za-z0-9_-]{43}(/mcp)")


def url_base(base: str) -> str:
    base = base.rstrip("/")
    parts = urlsplit(base)
    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.path
        or parts.query
        or parts.fragment
    ):
        raise ValueError("server base must be an HTTP origin without credentials or a path")
    return base


def profile_path(base: str):
    parts = urlsplit(url_base(base))
    if parts.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("admin credentials are available only for a loopback server")
    return STATE / "servers" / f"{parts.port or (443 if parts.scheme == 'https' else 80)}.json"


def admin_headers(base: str, token: str | None = None) -> dict:
    base = url_base(base)
    if token is None:
        profile = json.loads(profile_path(base).read_text())
        if profile["admin_base"] != base:
            raise ValueError("admin credential profile belongs to a different server origin")
        token = profile["admin_token"]
    if not token:
        raise ValueError("an admin token is required")
    return {"authorization": f"Bearer {token}"}


def write_profile(base: str, agent_base: str, token: str):
    path = profile_path(base)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as target:
        temporary = target.name
        json.dump({"admin_base": base, "agent_base": agent_base, "admin_token": token}, target)
    try:
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def redact(value):
    if isinstance(value, dict):
        private = ("admin_token", "capability", "mcp_url", "authorization")
        return {k: "[redacted]" if str(k).lower() in private else redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return CAPABILITY.sub(r"\1[redacted]\2", value)
    return value
