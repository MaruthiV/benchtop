import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchtop.agents.common import RESULTS
from benchtop.env import ENVS, ROOT, STATE, Env, load_catalog
from benchtop.evaluate import all_envs

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(STATE / "mpl"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ASSETS = ROOT / "assets"
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b9b8b0"
KIND_NAMES = {"claim": "Does the method help?", "where": "Where does it help?", "best": "Best config"}


def style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.edgecolor": BASE,
            "axes.labelcolor": INK2,
            "xtick.color": MUTED,
            "ytick.color": INK2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "svg.fonttype": "none",
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
        }
    )


def save(fig, path: Path) -> None:
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def env_index() -> dict[str, Env]:
    return {env.id: env for env in all_envs(envs_dir=ENVS)}


def task_kind(envs, env_id, task_id) -> str | None:
    env = envs.get(env_id)
    if not env:
        return None
    try:
        return env.task(task_id)["kind"]
    except KeyError:
        return None


def claim_truth(envs, env_id) -> str | None:
    env = envs.get(env_id)
    for t in env.tasks if env else []:
        if t["kind"] == "claim":
            return t["answer"]
    return None


def bootstrap(xs, n=2000, seed=0) -> tuple[float, float, float]:
    xs = np.asarray(xs, float)
    if len(xs) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(xs, (n, len(xs))).mean(1)
    return float(xs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


INFRA = ("usage limit", "hit your", "5-hour limit", "Request timed out", "stream disconnected")


def infra_failure(row: dict) -> bool:
    # quota and network failures say nothing about the agent, so they're dropped instead of scored 0
    err = row.get("error") or ""
    return not row.get("submitted") and (err == "success" or any(k in err for k in INFRA))


def rollout_rows(tag: str | None = None, dedupe: bool = True) -> list[dict]:
    rows = [
        r
        for r in read_jsonl(RESULTS / "rollouts.jsonl")
        if r.get("score") is not None or r.get("submitted") is False
    ]
    rows = [r for r in rows if not infra_failure(r)]
    rows = [r for r in rows if tag is None or r.get("tag") == tag]
    if not dedupe:
        return rows
    # a rerun of the same rollout replaces the earlier row
    latest = {}
    for r in rows:
        latest[(r["agent"], r["env"], r["task"], r.get("show_paper", True), r.get("seed"), r.get("tag"))] = r
    rows = list(latest.values())
    for r in rows:
        r["score"] = r.get("score") or 0.0
    return rows


def score_table(envs, rows, base_rows) -> dict:
    table = defaultdict(lambda: defaultdict(list))
    for r in base_rows:
        kind = task_kind(envs, r["env"], r["task"])
        if kind:
            table[r["agent"]][kind].append(r["mean"])
    for r in rows:
        if not r.get("show_paper", True):
            continue
        kind = task_kind(envs, r["env"], r["task"])
        if kind:
            table[r["agent"]][kind].append(r["score"])
    return {a: {k: bootstrap(v) + (len(v),) for k, v in kinds.items()} for a, kinds in table.items()}


def fig_scores(table: dict, agents: list[str], labels: dict[str, str], path: Path) -> None:
    style()
    kinds = [k for k in ("claim", "where", "best") if any(k in table.get(a, {}) for a in agents)]
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.3 * len(kinds), 0.42 * len(agents) + 1.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, kind in zip(axes, kinds, strict=True):
        ys = np.arange(len(agents))[::-1]
        for y, a in zip(ys, agents, strict=True):
            if kind not in table.get(a, {}):
                continue
            mean, lo, hi, n = table[a][kind]
            color = GRAY if ":" not in a else BLUE
            ax.barh(y, mean, height=0.56, color=color, edgecolor=SURFACE, linewidth=2)
            ax.plot([lo, hi], [y, y], color=INK2, linewidth=1.2)
            ax.text(min(hi, 1.0) + 0.03, y, f"{mean:.2f}", va="center", color=INK2, fontsize=9)
        ax.set_xlim(0, 1.18)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_title(KIND_NAMES[kind], loc="left")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_yticks(np.arange(len(agents))[::-1])
    axes[0].set_yticklabels([labels.get(a, a) for a in agents])
    fig.text(
        0.01,
        0.01,
        "mean score on held-out seeds, bars show 95% bootstrap intervals over episodes; gray = no LLM",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, path)
    plt.close(fig)


def fig_funnel(stages: list[tuple[str, int]], path: Path) -> None:
    style()
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(stages) + 0.6))
    ys = np.arange(len(stages))[::-1]
    top = stages[0][1]
    for y, (_, n) in zip(ys, stages, strict=True):
        ax.barh(y, n, height=0.6, color=BLUE, edgecolor=SURFACE, linewidth=2)
        ax.text(n + top * 0.015, y, f"{n}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in stages])
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, top * 1.12)
    fig.tight_layout()
    save(fig, path)
    plt.close(fig)


def fig_trust(rows_by_agent: dict[str, tuple[list, list]], labels: dict[str, str], path: Path) -> None:
    style()
    agents = list(rows_by_agent)
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(agents) + 1.3))
    ys = np.arange(len(agents))[::-1]
    for y, a in zip(ys, agents, strict=True):
        holds, fails = rows_by_agent[a]
        for off, vals, color in ((0.17, holds, BLUE), (-0.17, fails, ORANGE)):
            if not vals:
                continue
            m, lo, hi = bootstrap(vals)
            ax.barh(y + off, m, height=0.32, color=color, edgecolor=SURFACE, linewidth=2)
            ax.plot([lo, hi], [y + off, y + off], color=INK2, linewidth=1)
            ax.text(min(hi, 1) + 0.02, y + off, f"{m:.2f}", va="center", color=INK2, fontsize=8.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([labels.get(a, a) for a in agents])
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 0.5, 1.0])
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=BLUE, label="lab agrees with the paper"),
            Patch(color=ORANGE, label="lab contradicts the paper"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    ax.set_xlabel("accuracy on 'does the method help?'", color=INK2)
    fig.tight_layout()
    save(fig, path)
    plt.close(fig)


def fig_scale(points: list[dict], path: Path) -> None:
    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    for servers, color in ((1, BLUE), (4, ORANGE)):
        pts = sorted(
            (p for p in points if p.get("servers") == servers and p.get("think_mean_s", 0) > 0),
            key=lambda p: p["agents"],
        )
        if not pts:
            continue
        xs, ys = [p["agents"] for p in pts], [p["p99_ms"] for p in pts]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE)
        name = "1 process" if servers == 1 else f"{servers} processes"
        ax.text(xs[-1] * 1.08, ys[-1], name, color=INK2, va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    xt = [5000, 10000, 20000, 40000, 80000]
    ax.set_xticks(xt, [f"{x // 1000}k" for x in xt])
    ax.minorticks_off()
    ax.set_yticks([10, 100, 1000, 10000], ["10 ms", "100 ms", "1 s", "10 s"])
    worst = max(points, key=lambda p: p["p99_ms"])
    share = worst["errors"] / worst["agents"]
    if share > 0.1:
        ax.annotate(
            f"{share:.0%} of agents hit\nconnection errors",
            (worst["agents"], worst["p99_ms"]),
            xytext=(-40, -2),
            textcoords="offset points",
            ha="right",
            va="top",
            color=MUTED,
            fontsize=8,
        )
    ax.set_xlabel("concurrent agents, each acting every ~10 s")
    ax.set_ylabel("p99 latency per call")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, path)
    plt.close(fig)


AGENT_LABELS = {
    "prior": "trust the paper",
    "random": "random design",
    "uniform": "uniform statistician",
    "ollama:qwen3:4b-instruct": "Qwen3 4B (local)",
    "claude:haiku": "Claude Haiku 5.5",
    "claude:sonnet": "Claude Sonnet 5.5",
    "claude:opus": "Claude Opus 5.5",
    "claude:fable": "Claude Fable 5.1",
    "codex:default": "GPT (Codex CLI)",
}
ORDER = [
    "prior",
    "random",
    "uniform",
    "ollama:qwen3:4b-instruct",
    "claude:haiku",
    "claude:sonnet",
    "claude:opus",
    "claude:fable",
    "codex:default",
]


def audit_of(env: Env) -> dict:
    path = env.dir / "audit.json"
    return json.loads(path.read_text()) if path.exists() else {}


def clean_envs(envs: dict[str, Env]) -> set[str]:
    # labs an independent auditor passed, minus claims driven by one extreme setting
    out = set()
    for env_id, env in envs.items():
        a = audit_of(env)
        claim = next((t for t in env.tasks if t["kind"] == "claim"), {})
        if (
            a.get("ok")
            and a.get("tests_core_claim")
            and a.get("baseline_fair")
            and a.get("fidelity") in ("high", "medium")
            and claim.get("top_share", 0) <= 0.5
        ):
            out.add(env_id)
    return out


def funnel_stats() -> dict:
    return load_catalog(ENVS)["funnel"]


def build_report(tag: str | None = "main") -> dict:
    from benchtop.analysis import calibration, decompose

    envs = env_index()
    base_rows = (
        [r for r in json.loads((RESULTS / "baselines.json").read_text()) if r["env"] in envs]
        if (RESULTS / "baselines.json").exists()
        else []
    )
    rows = [r for r in rollout_rows(tag) if r["env"] in envs]
    table = score_table(envs, rows, base_rows)
    agents = [a for a in ORDER if a in table]

    clean = clean_envs(envs)
    trust = {}
    for a in agents:
        holds, fails = [], []
        if ":" in a:
            src = [
                (r["env"], float(r["score"] == 1.0))
                for r in rows
                if r["agent"] == a
                and r.get("show_paper", True)
                and task_kind(envs, r["env"], r["task"]) == "claim"
            ]
        else:
            src = [(r["env"], r["mean"]) for r in base_rows if r["agent"] == a and r["kind"] == "claim"]
        for env_id, s in src:
            if env_id in clean:
                (holds if claim_truth(envs, env_id) == "paper_better" else fails).append(s)
        trust[a] = (holds, fails)

    decomp, calib, ablation = {}, {}, {}
    for a in agents:
        if ":" not in a:
            continue
        mine = [r for r in rows if r["agent"] == a]
        cats = defaultdict(int)
        for r in mine:
            if r.get("show_paper", True) and r["env"] in clean and (c := decompose(envs[r["env"]], r)):
                cats[c] += 1
        decomp[a] = dict(cats)
        calib[a] = calibration(
            [r for r in mine if r.get("show_paper", True) and task_kind(envs, r["env"], r["task"]) == "claim"]
        )
        with_p = {(r["env"], r["task"]): r["score"] for r in mine if r.get("show_paper", True)}
        without = {(r["env"], r["task"]): r["score"] for r in mine if r.get("show_paper") is False}
        both = sorted(set(with_p) & set(without))
        if both:
            contra = [k for k in both if claim_truth(envs, k[0]) != "paper_better" and k[1] == "claim"]
            ablation[a] = {
                "pairs": len(both),
                "with_paper": float(np.mean([with_p[k] for k in both])),
                "without_paper": float(np.mean([without[k] for k in both])),
                "contradicting_claims": len(contra),
                "with_paper_on_contradicting": float(np.mean([with_p[k] for k in contra]))
                if contra
                else None,
                "without_paper_on_contradicting": float(np.mean([without[k] for k in contra]))
                if contra
                else None,
            }

    budget = {
        a: {
            "spent_frac": float(np.mean([r["spent_frac"] or 0 for r in rows if r["agent"] == a])),
            "runs": float(np.mean([r["runs"] for r in rows if r["agent"] == a])),
            "secs": float(np.mean([r.get("secs") or 0 for r in rows if r["agent"] == a])),
            "api_equiv_usd": float(np.sum([r.get("api_equiv_usd") or 0 for r in rows if r["agent"] == a])),
            "n": sum(1 for r in rows if r["agent"] == a),
        }
        for a in agents
        if ":" in a
    }

    audits = [audit_of(e) for e in envs.values()]
    summary = {
        "envs": len(envs),
        "episodes": sum(len(e.tasks) for e in envs.values()),
        "clean_envs": len(clean),
        "audit": {
            "fidelity": {k: sum(a.get("fidelity") == k for a in audits) for k in ("high", "medium", "low")},
            "unfair_baseline": sum(a.get("baseline_fair") is False for a in audits),
            "not_core_claim": sum(a.get("tests_core_claim") is False for a in audits),
        },
        "clean_claims_agree": sum(claim_truth(envs, e) == "paper_better" for e in clean),
        "clean_claim_tasks": sum(claim_truth(envs, e) is not None for e in clean),
        "claim_agrees_with_paper": sum(claim_truth(envs, e) == "paper_better" for e in envs),
        "claim_tasks": sum(claim_truth(envs, e) is not None for e in envs),
        "funnel": funnel_stats(),
        "scores": {a: table[a] for a in agents},
        "decomposition": decomp,
        "calibration": calib,
        "paper_ablation": ablation,
        "budget": budget,
        "trust": {
            a: {"holds": bootstrap(h), "fails": bootstrap(f), "n_holds": len(h), "n_fails": len(f)}
            for a, (h, f) in trust.items()
        },
    }
    summary["paper_effect"] = {
        a: e for a in ("claude:haiku", "ollama:qwen3:4b-instruct") if (e := paper_effect(envs, a))
    }
    summary["group_reuse"] = group_reuse()
    summary["group_reuse_tight"] = group_reuse("group_tight")
    summary["tight"] = tight_budget(envs)
    summary["tight_vs_uniform"] = tight_vs_uniform(envs)
    if summary["tight"]:
        fig_budget(table, summary["tight"], AGENT_LABELS, ASSETS / "budget.svg")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    ASSETS.mkdir(exist_ok=True)
    fig_scores(table, agents, AGENT_LABELS, ASSETS / "scores.svg")
    fig_trust(trust, AGENT_LABELS, ASSETS / "trust.svg")
    f = summary["funnel"]
    fig_funnel(
        [
            ("papers screened", f["screened"]),
            ("judged testable at miniature scale", f["selected"]),
            ("lab written and validated", f["lab_written"]),
            ("question settled on held-out seeds", f["accepted"]),
        ],
        ASSETS / "funnel.svg",
    )
    scale = RESULTS / "scale.json"
    if scale.exists():
        pts = [p for p in json.loads(scale.read_text()) if p.get("mode") == "http"]
        if pts:
            fig_scale(pts, ASSETS / "scale.svg")
    return summary


def group_reuse(tag: str = "group") -> dict:
    # k rollouts of one episode seed: how many runs could a shared cache skip
    envs = env_index()
    groups = defaultdict(list)
    for r in rollout_rows(tag, dedupe=False):
        if r.get("submitted"):
            groups[(r["env"], r["task"], r.get("seed"))].append(r)
    out = []
    for (env_id, task, _), rs in groups.items():
        if len(rs) < 2 or env_id not in envs:
            continue
        env = envs[env_id]
        requested = sum(len(r.get("ran", [])) for r in rs)
        unique = len({tuple(p) for r in rs for p in r.get("ran", [])})
        visible_grid = env.space.size() * len(env.visible_seeds)
        out.append(
            {
                "env": env_id,
                "task": task,
                "k": len(rs),
                "requested": requested,
                "unique": unique,
                "saved_frac": 1 - unique / requested if requested else 0.0,
                "visible_grid": visible_grid,
                "unique_frac_of_grid": unique / visible_grid,
                "score_mean": float(np.mean([r["score"] for r in rs])),
                "score_std": float(np.std([r["score"] for r in rs])),
            }
        )
    if not out:
        return {}
    req, uniq = sum(g["requested"] for g in out), sum(g["unique"] for g in out)
    return {
        "groups": out,
        "requested": req,
        "unique": uniq,
        "saved_frac": 1 - uniq / req,
        "mean_unique_frac_of_grid": float(np.mean([g["unique_frac_of_grid"] for g in out])),
        "groups_with_reward_variance": sum(g["score_std"] > 0 for g in out),
    }


def fig_budget(loose: dict, tight: dict, labels: dict, path: Path, kinds=("claim", "where")) -> None:
    style()
    # trusting the paper doesn't depend on budget, so it adds nothing here
    agents = [a for a in ORDER if a in tight and a in loose and a != "prior"]
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(agents) + 1.2))
    ys = np.arange(len(agents))[::-1]

    def avg(table, a):
        vals = [(table[a][k][0], table[a][k][3]) for k in kinds if k in table[a]]
        n = sum(w for _, w in vals)
        return sum(m * w for m, w in vals) / n if n else float("nan")

    for y, a in zip(ys, agents, strict=True):
        lo, hi = avg(tight, a), avg(loose, a)
        ax.plot([lo, hi], [y, y], color=BASE, linewidth=2, zorder=1)
        ax.scatter([hi], [y], s=60, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=2)
        ax.scatter([lo], [y], s=60, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=2)
        ax.text(max(lo, hi) + 0.03, y, f"{hi:.2f} / {lo:.2f}", va="center", color=INK2, fontsize=8.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([labels.get(a, a) for a in agents])
    ax.set_xlim(0, 1.2)
    ax.set_xticks([0, 0.5, 1.0])
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D(
                [], [], marker="o", color=SURFACE, markerfacecolor=BLUE, markersize=8, label="25% of grid"
            ),
            Line2D(
                [], [], marker="o", color=SURFACE, markerfacecolor=ORANGE, markersize=8, label="6% of grid"
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    ax.set_xlabel("mean score on 'does it help?' and 'where does it help?'", color=INK2)
    fig.tight_layout()
    save(fig, path)
    plt.close(fig)


def tight_budget(envs: dict) -> dict:
    path = RESULTS / "baselines_tight.json"
    base = [r for r in json.loads(path.read_text()) if r["env"] in envs] if path.exists() else []
    rows = [r for r in rollout_rows("tight") if r["env"] in envs]
    table = score_table(envs, rows, base)
    return {a: table[a] for a in ORDER if a in table}


def tight_vs_uniform(envs: dict) -> dict:
    # paired gap over the uniform design on the same tight-budget episodes
    path = RESULTS / "baselines_tight.json"
    if not path.exists():
        return {}
    uniform = {
        (r["env"], r["task"]): r["mean"]
        for r in json.loads(path.read_text())
        if r["agent"] == "uniform" and r["env"] in envs
    }
    gaps = defaultdict(lambda: defaultdict(list))
    for r in rollout_rows("tight"):
        if r["env"] not in envs or not r.get("show_paper", True) or (r["env"], r["task"]) not in uniform:
            continue
        if kind := task_kind(envs, r["env"], r["task"]):
            gaps[r["agent"]][kind].append(r["score"] - uniform[(r["env"], r["task"])])
    out = {}
    for a in ORDER:
        for kind, d in gaps.get(a, {}).items():
            d = np.array(d)
            rng = np.random.default_rng(0)
            boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
            out.setdefault(a, {})[kind] = {
                "n": len(d),
                "diff": float(d.mean()),
                "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            }
    return out


def paper_effect(envs: dict, agent: str, tags=("main", "ablation")) -> dict:
    # same episode seed with and without the paper's claim, on labs that contradict it
    rows = [
        r
        for t in tags
        for r in rollout_rows(t)
        if r["agent"] == agent
        and task_kind(envs, r["env"], r["task"]) == "claim"
        and claim_truth(envs, r["env"]) not in (None, "paper_better")
    ]
    pairs = defaultdict(dict)
    for r in rows:
        pairs[(r["env"], r.get("seed"))][r.get("show_paper", True)] = float(r["score"] == 1.0)
    both = [v for v in pairs.values() if True in v and False in v]
    if not both:
        return {}
    w, wo = np.array([v[True] for v in both]), np.array([v[False] for v in both])
    d = wo - w
    rng = np.random.default_rng(0)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    return {
        "pairs": len(both),
        "labs": len({k[0] for k, v in pairs.items() if True in v and False in v}),
        "with_paper": float(w.mean()),
        "without_paper": float(wo.mean()),
        "diff": float(d.mean()),
        "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }
