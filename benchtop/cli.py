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

    sweep(Path(a.env), workers=a.workers, seeds=a.seeds)


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


def cmd_verify(a):
    from benchtop.verify import verify_foundation

    try:
        result = verify_foundation(live=a.live, output=Path(a.output))
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"verification failed: {error}") from error
    print(json.dumps({key: value for key, value in result.items() if key not in ("input_sha256", "runtime")},
                     indent=2, allow_nan=False))


def cmd_integration(a):
    from benchtop.access import redact
    from benchtop.agents.common import Client
    from benchtop.integration import begin, collect, prepare

    client = Client(a.base)
    try:
        if a.action == "prepare":
            result = prepare(client, env=a.env, task=a.task)
        elif a.action == "begin":
            result = begin(client, a.episode, client_name=a.client, requested_model=a.requested_model,
                           reported_model=a.reported_model, restrictions=a.restrictions)
        else:
            result = collect(client, a.episode, client_observed=a.client_observed, output=a.output)
        print(json.dumps(redact(result), indent=2, allow_nan=False))
    except Exception as error:
        raise SystemExit(f"integration failed: {redact(str(error))}") from None
    finally:
        client.http.close()


def cmd_admission(a):
    from benchtop.admission import draft_review, inspect_lab, screen_catalog, write_json
    from benchtop.env import ENVS

    try:
        if a.action == "inspect":
            review = json.loads(Path(a.review).read_text()) if a.review else None
            result = inspect_lab(Path(a.env), review=review)
            if a.draft_review:
                write_json(a.draft_review, draft_review(result), draft=True)
            summary = {key: result[key] for key in ("env", "decision", "qualified_claim_admitted")}
            summary["mechanical"] = result["mechanical"]["status"]
            summary["human_review"] = result["human_review"]["status"]
        else:
            result = screen_catalog(Path(a.envs_dir) if a.envs_dir else ENVS, reviews_dir=a.reviews_dir)
            summary = result["summary"]
        write_json(a.output, result)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise SystemExit(f"admission screen failed: {error}") from error
    print(json.dumps({**summary, "output": a.output}, indent=2, allow_nan=False))


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
    s.add_argument("--seeds", type=int, default=None, help="total seeds; defaults to the manifest, or 16 for a new lab")
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

    s = sub.add_parser("verify", help="reproduce foundation checks in a temporary copy without private state")
    s.add_argument("--live", action="store_true", help="also sweep a small synthetic fixture in the macOS sandbox")
    s.add_argument("--output", default="artifacts/foundation_check.json")
    s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("integration", help="prepare and record an operator-assisted Qualia connectivity check")
    actions = s.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("prepare")
    prepare.add_argument("--env", default="2610-36983-advancement-or-acceleration-how")
    prepare.add_argument("--task", default="where:gap_deg")
    begin = actions.add_parser("begin")
    begin.add_argument("episode")
    begin.add_argument("--client", choices=["qualia-cloud", "qualia-desktop"], required=True)
    begin.add_argument("--requested-model", required=True)
    begin.add_argument("--reported-model", default="not_reported")
    begin.add_argument("--restrictions", default="not_verified")
    collect = actions.add_parser("collect")
    collect.add_argument("episode")
    collect.add_argument("--client-observed", action="store_true", help="attest that the calls were observed in the actual Qualia UI")
    collect.add_argument("--output", default=None)
    for action in (prepare, begin, collect):
        action.add_argument("--base", default="http://127.0.0.1:8787")
        action.set_defaults(fn=cmd_integration)

    s = sub.add_parser("admission", help="screen cached lab evidence separately from human scientific review")
    actions = s.add_subparsers(dest="action", required=True)
    inspect = actions.add_parser("inspect")
    inspect.add_argument("env")
    inspect.add_argument("--review", default=None)
    inspect.add_argument("--draft-review", default=None)
    inspect.add_argument("--output", default="artifacts/admission.json")
    screen = actions.add_parser("screen")
    screen.add_argument("--envs-dir", default=None)
    screen.add_argument("--reviews-dir", default=None)
    screen.add_argument("--output", default="artifacts/admission_screen.json")
    for action in (inspect, screen):
        action.set_defaults(fn=cmd_admission)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
