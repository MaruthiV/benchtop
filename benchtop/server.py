import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from benchtop.env import ENVS, Env
from benchtop.episode import Episode
from benchtop.live import LiveStore

HEADER = "x-benchtop-episode"
MAX_EPISODES = 200_000

INSTRUCTIONS = (
    "You are a researcher with a small lab. Call describe_lab first. Spend the compute budget on "
    "run_experiments, think about noise, then call submit_answer exactly once."
)


class Hub:
    def __init__(self, envs_dir: Path = ENVS, live: bool = False, live_workers: int = 4):
        self.envs_dir = Path(envs_dir)
        self.envs: dict[str, Env] = {}
        self.episodes: OrderedDict[str, Episode] = OrderedDict()
        self.live = live
        self.live_workers = live_workers
        self.stores: dict[str, LiveStore] = {}

    async def prepare(self, ep: Episode, configs, seeds) -> None:
        # live mode runs whatever the table is missing before the episode reads it
        if not self.live:
            return
        pairs = ep.pending(configs, seeds)
        if pairs:
            store = self.stores.setdefault(ep.env.id, LiveStore(ep.env, workers=self.live_workers))
            await store.ensure(pairs)

    def env(self, env_id: str) -> Env:
        if env_id not in self.envs:
            path = self.envs_dir / env_id
            if not (path / "tasks.json").exists():
                raise KeyError(f"unknown env {env_id!r}")
            self.envs[env_id] = Env.load(path)
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
        ep = Episode(self.env(env_id), task_id, **kw)
        self.episodes[ep.id] = ep
        while len(self.episodes) > MAX_EPISODES:
            self.episodes.popitem(last=False)
        return ep

    def get(self, episode_id: str) -> Episode:
        try:
            return self.episodes[episode_id]
        except KeyError:
            raise KeyError(f"unknown episode {episode_id!r}") from None


def build_app(hub: Hub | None = None, any_host: bool = False):
    hub = hub or Hub()
    mcp = MCPServer(name="benchtop", instructions=INSTRUCTIONS)

    def episode_of(ctx: Context) -> Episode:
        headers = ctx.headers or {}
        return hub.get(headers.get(HEADER, ""))

    @mcp.tool(
        description="Read the lab: setting, factors and levels, metric, budget, rules, and the question you must answer."
    )
    def describe_lab(ctx: Context) -> dict[str, Any]:
        ep = episode_of(ctx)
        return {**ep.describe(), "budget": ep.budget_left()}

    @mcp.tool(
        description=(
            "Run experiments. configs is a list of objects mapping factor names to levels (missing factors use "
            "the default). Each config runs with `seeds` fresh seeds. Costs budget. Returns per-run metrics."
        )
    )
    async def run_experiments(configs: list[dict[str, Any]], ctx: Context, seeds: int = 1) -> dict[str, Any]:
        ep = episode_of(ctx)
        await hub.prepare(ep, configs, seeds)
        return ep.run(configs, seeds)

    @mcp.tool(
        description=(
            "Submit your final answer once, in the format describe_lab asks for. confidence is your probability "
            "(0 to 1) that the answer is right."
        )
    )
    def submit_answer(
        answer: str | dict[str, Any], ctx: Context, confidence: float | None = None
    ) -> dict[str, Any]:
        out = episode_of(ctx).submit(answer, confidence)
        # don't leak the grade to the agent mid-run
        return {"received": True} if out.get("done") else out

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
        return JSONResponse({"id": ep.id, "describe": ep.describe(), "budget": ep.budget_left()})

    async def show(request: Request):
        try:
            ep = hub.get(request.path_params["eid"])
        except KeyError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse(ep.summary())

    async def run(request: Request):
        body = await request.json()
        try:
            ep = hub.get(request.path_params["eid"])
        except KeyError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        configs, seeds = body.get("configs", []), body.get("seeds", 1)
        await hub.prepare(ep, configs, seeds)
        return JSONResponse(ep.run(configs, seeds))

    async def submit(request: Request):
        body = await request.json()
        try:
            ep = hub.get(request.path_params["eid"])
        except KeyError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse(ep.submit(body.get("answer"), body.get("confidence")))

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
    # the sdk only lets localhost host/origin headers through; any_host is for tunnels and app clients
    security = TransportSecuritySettings(enable_dns_rebinding_protection=False) if any_host else None
    mcp_app = mcp.streamable_http_app(
        streamable_http_path="/mcp", stateless_http=True, json_response=True, transport_security=security
    )
    # plain routes skip the mcp middleware stack; only /mcp goes through it
    app = Starlette(
        routes=[*routes, Mount("/", app=mcp_app)], lifespan=lambda _: mcp_app.router.lifespan_context(mcp_app)
    )
    return EpisodePaths(app), hub


class EpisodePaths:
    # /e/<id>/mcp becomes /mcp plus a header, so url-only mcp clients get their own episode
    pattern = re.compile(r"^/e/([0-9a-f]{12})/mcp/?$")

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and (m := self.pattern.match(scope["path"])):
            scope = dict(scope)
            scope["path"], scope["raw_path"] = "/mcp", b"/mcp"
            scope["headers"] = [*scope["headers"], (HEADER.encode(), m.group(1).encode())]
        await self.app(scope, receive, send)


def serve(
    host: str = "127.0.0.1",
    port: int = 8787,
    envs_dir: Path = ENVS,
    log_level: str = "warning",
    live: bool = False,
    any_host: bool = False,
):
    import uvicorn

    app, _ = build_app(Hub(envs_dir, live=live), any_host=any_host)
    uvicorn.run(app, host=host, port=port, log_level=log_level, timeout_keep_alive=75, backlog=4096)
