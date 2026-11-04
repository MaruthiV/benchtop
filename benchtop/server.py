import asyncio
import json
import re
import secrets
import socket
import time
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from benchtop.access import url_base, write_profile
from benchtop.cache import CacheError
from benchtop.env import ENVS, Env
from benchtop.episode import Episode
from benchtop.live import LiveStore
from benchtop.sandbox import sandbox_mode

HEADER = "x-benchtop-episode"
MAX_EPISODES = 200_000

INSTRUCTIONS = (
    "You are a researcher with a small lab. Call describe_lab first. Spend the compute budget on "
    "run_experiments, think about noise, then call submit_answer exactly once."
)


class Hub:
    def __init__(
        self, envs_dir: Path = ENVS, live: bool = False, live_workers: int = 4,
        agent_base: str = "http://127.0.0.1:8788",
    ):
        if live:
            sandbox_mode()
        self.envs_dir = Path(envs_dir)
        self.envs: dict[str, Env] = {}
        self.episodes: OrderedDict[str, Episode] = OrderedDict()
        self.live = live
        self.live_workers = live_workers
        self.stores: dict[str, LiveStore] = {}
        self.agent_base = url_base(agent_base)
        self.admin_token = secrets.token_urlsafe(32)
        self.capabilities: dict[str, str] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    async def run(self, ep: Episode, configs, seeds=1) -> dict:
        async with self.locks[ep.id]:
            request = ep.begin_run(configs, seeds)
            if "error" in request:
                return request
            try:
                failures = {}
                if self.live and request["steps"]:
                    store = self.stores.get(ep.env.id)
                    if store is None:
                        store = LiveStore(ep.env, workers=self.live_workers)
                        self.stores[ep.env.id] = store
                    pairs = [(config, seed) for config, _, seed, _ in request["steps"]]
                    outcomes = await store.ensure(pairs)
                    failures = {
                        (key, seed): outcome["error"]
                        for (_, key, seed, _), outcome in zip(request["steps"], outcomes, strict=True)
                        if not outcome["ok"]
                    }
                ep.env.assert_current()
                return ep.finish_run(request, failures)
            except asyncio.CancelledError:
                ep.abort_run(request, "cancelled")
                raise
            except CacheError:
                ep.abort_run(request, "failed")
                return {"error": "environment cache inputs changed; reload or rebuild", "code": "stale_cache"}
            except Exception:
                ep.abort_run(request, "failed")
                raise

    async def submit(self, ep: Episode, answer, confidence=None) -> dict:
        async with self.locks[ep.id]:
            return ep.submit(answer, confidence)

    async def describe(self, ep: Episode) -> dict:
        async with self.locks[ep.id]:
            return {**ep.describe(), "budget": ep.budget_left()}

    async def summary(self, ep: Episode) -> dict:
        async with self.locks[ep.id]:
            return ep.summary()

    async def close(self):
        await asyncio.gather(*(store.close() for store in self.stores.values()))

    def env(self, env_id: str) -> Env:
        if env_id not in self.envs:
            path = self.envs_dir / env_id
            if not (path / "tasks.json").exists():
                raise KeyError(f"unknown env {env_id!r}")
            self.envs[env_id] = Env.load(path, replay=not self.live)
        return self.envs[env_id]

    def list_envs(self) -> list[dict]:
        out = []
        for path in sorted(self.envs_dir.glob("*/tasks.json")):
            env = self.env(path.parent.name)
            out.append(
                {
                    "id": env.id,
                    "title": env.space.title,
                    "tasks": [t["id"] for t in env.tasks],
                    "configs": env.space.size(),
                }
            )
        return out

    def create(self, env_id: str, task_id: str, **kw) -> Episode:
        env = self.env(env_id)
        if self.live:
            env.prepare_live()
        ep = Episode(env, task_id, **kw)
        while ep.id in self.episodes:
            ep.id = secrets.token_hex(6)
        self.episodes[ep.id] = ep
        self.capabilities[ep.id] = secrets.token_urlsafe(32)
        self.locks[ep.id] = asyncio.Lock()
        while len(self.episodes) > MAX_EPISODES:
            expired, _ = self.episodes.popitem(last=False)
            self.capabilities.pop(expired)
            self.locks.pop(expired)
        return ep

    def agent_url(self, ep: Episode) -> str:
        return f"{self.agent_base}/e/{ep.id}/{self.capabilities[ep.id]}/mcp"

    def get(self, episode_id: str) -> Episode:
        try:
            return self.episodes[episode_id]
        except KeyError:
            raise KeyError(f"unknown episode {episode_id!r}") from None


def build_app(
    hub: Hub | None = None, allowed_hosts: list[str] | None = None, allowed_origins: list[str] | None = None,
):
    hub = hub or Hub()
    mcp = MCPServer(name="benchtop", instructions=INSTRUCTIONS)

    def episode_of(ctx: Context) -> Episode:
        headers = ctx.headers or {}
        return hub.get(headers.get(HEADER, ""))

    @mcp.tool(
        description="Read the lab: setting, factors and levels, metric, budget, rules, and the question you must answer."
    )
    async def describe_lab(ctx: Context) -> dict[str, Any]:
        ep = episode_of(ctx)
        return await hub.describe(ep)

    @mcp.tool(
        description=(
            "Run experiments. configs is a list of objects mapping factor names to levels (missing factors use "
            "the default). Each config runs with `seeds` fresh seeds. Costs budget. Returns per-run metrics."
        )
    )
    async def run_experiments(configs: list[dict[str, Any]], ctx: Context, seeds: int = 1) -> dict[str, Any]:
        ep = episode_of(ctx)
        return await hub.run(ep, configs, seeds)

    @mcp.tool(
        description=(
            "Submit your final answer once, in the format describe_lab asks for. confidence is your probability "
            "(0 to 1) that the answer is right."
        )
    )
    async def submit_answer(
        answer: str | dict[str, Any], ctx: Context, confidence: float | None = None
    ) -> dict[str, Any]:
        out = await hub.submit(episode_of(ctx), answer, confidence)
        # don't leak the grade to the agent mid-run
        return {"received": True} if out.get("done") else out

    hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*", urlsplit(hub.agent_base).netloc, *(allowed_hosts or [])]
    origins = ["http://127.0.0.1:*", "http://localhost:*", *(allowed_origins or [])]
    if any("*" in value for value in [*(allowed_hosts or []), *(allowed_origins or [])]):
        raise ValueError("custom hosts and origins must be explicit, without wildcards")
    security = TransportSecuritySettings(allowed_hosts=hosts, allowed_origins=origins)
    mcp_app = mcp.streamable_http_app(
        streamable_http_path="/mcp", stateless_http=True, json_response=True, transport_security=security
    )
    return EpisodePaths(mcp_app, hub), hub


def build_admin_app(hub: Hub):
    async def envs(request: Request):
        return JSONResponse(hub.list_envs())

    async def create(request: Request):
        body = await request.json()
        try:
            ep = hub.create(
                body["env"],
                body["task"],
                budget_frac=body.get("budget_frac", 0.25),
                seed=body.get("seed"),
                show_paper=body.get("show_paper", True),
            )
        except KeyError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        except (TypeError, ValueError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse({"id": ep.id, "describe": ep.describe(), "budget": ep.budget_left(), "mcp_url": hub.agent_url(ep)})

    async def show(request: Request):
        try:
            ep = hub.get(request.path_params["eid"])
        except KeyError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse(await hub.summary(ep))

    async def run(request: Request):
        body = await request.json()
        try:
            ep = hub.get(request.path_params["eid"])
        except KeyError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        configs, seeds = body.get("configs", []), body.get("seeds", 1)
        return JSONResponse(await hub.run(ep, configs, seeds))

    async def submit(request: Request):
        body = await request.json()
        try:
            ep = hub.get(request.path_params["eid"])
        except KeyError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse(await hub.submit(ep, body.get("answer"), body.get("confidence")))

    async def episodes(request: Request):
        rows = [
            {
                "id": e.id,
                "env": e.env.id,
                "task": e.task["id"],
                "calls": e.calls,
                "done": e.done,
                "age_s": round(time.time() - e.started, 1),
            }
            for e in list(hub.episodes.values())[-200:]
        ]
        return JSONResponse(rows)

    async def health(request: Request):
        live = {k: v.stats for k, v in hub.stores.items()}
        return JSONResponse({"ok": True, "episodes": len(hub.episodes), "live": live})

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/envs", envs, methods=["GET"]),
        Route("/episodes", create, methods=["POST"]),
        Route("/episodes", episodes, methods=["GET"]),
        Route("/episodes/{eid}", show, methods=["GET"]),
        Route("/episodes/{eid}/run", run, methods=["POST"]),
        Route("/episodes/{eid}/submit", submit, methods=["POST"]),
    ]
    return AdminAccess(Starlette(routes=routes), hub.admin_token)


class AdminAccess:
    def __init__(self, app, token):
        self.app, self.token = app, token

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = Headers(scope=scope)
            supplied = headers.get("authorization", "")
            if not secrets.compare_digest(supplied.encode(), f"Bearer {self.token}".encode()):
                return await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
            if headers.get("origin") is not None:
                return await JSONResponse({"error": "origin not allowed"}, status_code=403)(scope, receive, send)
            try:
                host = urlsplit("http://" + headers.get("host", "")).hostname
            except ValueError:
                host = None
            if host not in ("127.0.0.1", "localhost", "::1"):
                return await JSONResponse({"error": "host not allowed"}, status_code=421)(scope, receive, send)
        await self.app(scope, receive, send)


class EpisodePaths:
    pattern = re.compile(r"^/e/([0-9a-f]{12})/([A-Za-z0-9_-]{43})/mcp/?$")

    def __init__(self, app, hub):
        self.app, self.hub = app, hub

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            m = self.pattern.fullmatch(scope["path"])
            if m is None:
                return await JSONResponse({"error": "not found"}, status_code=404)(scope, receive, send)
            expected = self.hub.capabilities.get(m.group(1), "")
            if not expected or not secrets.compare_digest(expected, m.group(2)):
                return await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
            scope = dict(scope)
            scope["path"], scope["raw_path"] = "/mcp", b"/mcp"
            headers = [(key, value) for key, value in scope["headers"] if key.lower() != HEADER.encode()]
            scope["headers"] = [*headers, (HEADER.encode(), m.group(1).encode())]
        await self.app(scope, receive, send)


class PortApps:
    def __init__(self, admin, agent, admin_port, cleanup, hub):
        self.admin, self.agent, self.admin_port = admin, agent, admin_port
        self.cleanup = cleanup
        self.hub = hub

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            try:
                await self.agent(scope, receive, send)
            finally:
                try:
                    await self.hub.close()
                finally:
                    self.cleanup()
            return
        app = self.admin if scope["server"][1] == self.admin_port else self.agent
        await app(scope, receive, send)


def serve(
    port: int = 8787,
    envs_dir: Path = ENVS,
    log_level: str = "warning",
    live: bool = False,
    agent_port: int | None = None,
    agent_base: str | None = None,
    allowed_hosts: list[str] | None = None,
    allowed_origins: list[str] | None = None,
):
    import uvicorn

    agent_port = port + 1 if agent_port is None else agent_port
    if port == agent_port or not all(1 <= p <= 65535 for p in (port, agent_port)):
        raise ValueError("admin and agent ports must be distinct valid ports")
    hub = Hub(envs_dir, live=live, agent_base=agent_base or f"http://127.0.0.1:{agent_port}")
    agent, _ = build_app(hub, allowed_hosts, allowed_origins)
    admin = build_admin_app(hub)
    base = f"http://127.0.0.1:{port}"
    with ExitStack() as stack:
        sockets = []
        for number in (port, agent_port):
            sock = stack.enter_context(socket.socket())
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", number))
            sock.listen(4096)
            sockets.append(sock)
        profile = write_profile(base, hub.agent_base, hub.admin_token)

        def cleanup():
            try:
                if json.loads(profile.read_text())["admin_token"] == hub.admin_token:
                    profile.unlink()
            except FileNotFoundError:
                pass

        try:
            app = PortApps(admin, agent, port, cleanup, hub)
            config = uvicorn.Config(app, log_level=log_level, access_log=False, proxy_headers=False, timeout_keep_alive=75)
            uvicorn.Server(config).run(sockets=sockets)
        finally:
            cleanup()
