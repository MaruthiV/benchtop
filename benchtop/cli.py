import argparse
import json
import sys
from pathlib import Path


def _listing(name: str) -> list[dict]:
    from benchtop.arxiv import CACHE

    return json.loads((CACHE / f"listing-{name}.json").read_text())


def cmd_fetch(a):
    from benchtop import arxiv

    papers = []
    for start in range(0, a.n, 100):
        papers += arxiv.recent(tuple(a.cats), max_results=min(100, a.n - start), start=start)
    arxiv.save_listing(papers, a.name)
    print(f"{len(papers)} papers, {papers[-1]['published']} to {papers[0]['published']}")


def cmd_triage(a):
    from collections import Counter

    from benchtop.triage import selected, triage_many

    papers = [p for p in _listing(a.listing) if p["primary"] in a.primary]
    cards = triage_many(papers[: a.limit], jobs=a.jobs, model=a.model)
    print(len(cards), "triaged;", sum(map(selected, cards)), "selected")
    print(Counter(c.get("blocker") for c in cards))


def cmd_build(a):
    from benchtop.pipeline import FUNNEL, build_many
    from benchtop.triage import load_cards, selected

    papers = {p["id"]: p for p in _listing(a.listing)}
    done = {json.loads(line)["id"] for line in FUNNEL.read_text().splitlines()} if FUNNEL.exists() else set()
    cards = [c for c in load_cards(a.ids or None) if c["id"] in papers and selected(c)]
    cards = [c for c in cards if a.ids or c["id"] not in done][: a.limit]
    print(f"building {len(cards)} labs", flush=True)
    for row in build_many(
        [(papers[c["id"]], c) for c in cards], jobs=a.jobs, backend=a.backend, model=a.model
    ):
        print(row["outcome"], row["id"], row.get("env", ""), row["detail"][:120], flush=True)


def cmd_sweep(a):
    from benchtop.sweep import sweep

    sweep(Path(a.env), workers=a.workers)


def cmd_serve(a):
    from benchtop.env import ENVS
    from benchtop.server import serve

    serve(
        port=a.port, envs_dir=Path(a.envs_dir) if a.envs_dir else ENVS, live=a.live,
        agent_port=a.agent_port, agent_base=a.agent_base,
        allowed_hosts=a.allowed_host, allowed_origins=a.allowed_origin,
    )


def cmd_baselines(a):
    from benchtop.evaluate import run_baselines

    rows = run_baselines(episodes=a.episodes)
    for r in rows:
        print(f"{r['env'][:45]:45s} {r['task']:20s} {r['agent']:8s} {r['mean']:.2f}")


def cmd_rollouts(a):
    from benchtop.evaluate import episode_list, run_rollouts

    pairs = episode_list(a.envs or None, tuple(a.kinds))
    show = {"yes": (True,), "no": (False,), "both": (True, False)}[a.paper]
    print(f"{len(pairs)} episodes x {a.repeats} repeats x {len(show)} paper settings", flush=True)
    run_rollouts(
        a.agent, pairs[: a.limit], repeats=a.repeats, show_paper=show, jobs=a.jobs, tag=a.tag, seed0=a.seed0
    )


def cmd_episode(a):
    from benchtop.agents.common import Client

    c = Client(a.base)
    ep = c.create(a.env, a.task, show_paper=not a.no_paper)
    print(f"episode {ep['id']}")
    print(f"mcp url: {ep['mcp_url']}")
    print(f"question: {ep['describe']['question']}")
    print(f"grade later with the authenticated admin Client at {c.base}, episode {ep['id']}")


def cmd_report(a):
    from benchtop.report import build_report

    out = build_report(tag=a.tag or None)
    print(
        json.dumps(
            {k: out[k] for k in ("envs", "episodes", "claim_agrees_with_paper", "claim_tasks", "funnel")},
            indent=1,
        )
    )


def main(argv=None):
    p = argparse.ArgumentParser(prog="benchtop")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("fetch")
    s.add_argument("--cats", nargs="+", default=["cs.LG", "stat.ML"])
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--name", default="latest")
    s.set_defaults(fn=cmd_fetch)

    s = sub.add_parser("triage")
    s.add_argument("--listing", default="latest")
    s.add_argument("--primary", nargs="+", default=["cs.LG", "stat.ML"])
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--jobs", type=int, default=4)
    s.add_argument("--model", default="haiku")
    s.set_defaults(fn=cmd_triage)

    s = sub.add_parser("build")
    s.add_argument("--listing", default="latest")
    s.add_argument("--ids", nargs="*")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--jobs", type=int, default=2)
    s.add_argument("--backend", default="claude")
    s.add_argument("--model", default="sonnet")
    s.set_defaults(fn=cmd_build)

    s = sub.add_parser("sweep")
    s.add_argument("env")
    s.add_argument("--workers", type=int, default=8)
    s.set_defaults(fn=cmd_sweep)

    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8787)
    s.add_argument("--envs-dir", default=None)
    s.add_argument("--live", action="store_true")
    s.add_argument("--agent-port", type=int, default=None)
    s.add_argument("--agent-base", default=None)
    s.add_argument("--allowed-host", action="append", default=[])
    s.add_argument("--allowed-origin", action="append", default=[])
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("baselines")
    s.add_argument("--episodes", type=int, default=50)
    s.set_defaults(fn=cmd_baselines)

    s = sub.add_parser("rollouts")
    s.add_argument("--agent", default="claude:haiku")
    s.add_argument("--envs", nargs="*")
    s.add_argument("--kinds", nargs="+", default=["claim", "where", "component", "best"])
    s.add_argument("--repeats", type=int, default=1)
    s.add_argument("--paper", choices=["yes", "no", "both"], default="yes")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--jobs", type=int, default=2)
    s.add_argument("--tag", default="")
    s.add_argument("--seed0", type=int, default=0)
    s.set_defaults(fn=cmd_rollouts)

    s = sub.add_parser("episode")
    s.add_argument("env")
    s.add_argument("task")
    s.add_argument("--no-paper", action="store_true")
    s.add_argument("--base", default="http://127.0.0.1:8787")
    s.set_defaults(fn=cmd_episode)

    s = sub.add_parser("report")
    s.add_argument("--tag", default="main")
    s.set_defaults(fn=cmd_report)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
