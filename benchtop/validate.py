import ast
import asyncio
import json
import statistics
from pathlib import Path

from benchtop.sandbox import run_jobs
from benchtop.space import Space

ALLOWED = {
    "numpy",
    "scipy",
    "sklearn",
    "torch",
    "math",
    "random",
    "itertools",
    "functools",
    "collections",
    "dataclasses",
    "typing",
    "statistics",
    "heapq",
    "bisect",
    "copy",
    "warnings",
    "operator",
    "enum",
    "abc",
    "numbers",
    "fractions",
    "string",
    "contextlib",
}
BANNED_CALLS = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint"}
BANNED_ATTRS = {"save", "savez", "savetxt", "tofile", "hub", "system", "popen"}
MAX_SECS = 2.5


def static_errors(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"lab.py does not parse: {e}"]
    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                if name.split(".")[0] not in ALLOWED:
                    errors.append(f"import of {name!r} is not allowed (allowed: {sorted(ALLOWED)})")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            errors.append(f"call to {node.func.id}() is not allowed")
        elif isinstance(node, ast.Attribute):
            if (
                node.attr in BANNED_ATTRS
                or node.attr.startswith("fetch_")
                or node.attr.startswith("load_svm")
            ):
                errors.append(f"use of .{node.attr} is not allowed (no files, downloads, or shell)")
    runs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"]
    if not runs:
        errors.append("lab.py must define a top-level function run(config, seed)")
    elif len(runs[0].args.args) != 2:
        errors.append("run must take exactly two arguments: config and seed")
    return sorted(set(errors))


def space_errors(env_dir: Path) -> tuple[Space | None, list[str]]:
    try:
        raw = json.loads((env_dir / "space.json").read_text())
    except json.JSONDecodeError as e:
        return None, [f"space.json is not valid json: {e}"]
    try:
        space = Space.from_dict(raw)
    except (KeyError, ValueError, TypeError) as e:
        return None, [f"space.json is invalid: {type(e).__name__}: {e}"]
    errors = []
    if space.size() > 48:
        errors.append(f"grid has {space.size()} configs; keep it at 48 or fewer")
    if not space.context_factors():
        errors.append("add at least one context factor")
    if "paper" not in raw or "claim" not in raw.get("paper", {}):
        errors.append("space.json needs paper.claim")
    return space, errors


def smoke_errors(env_dir: Path, space: Space) -> tuple[list[str], dict]:
    t = space.treatment
    probes = [
        space.default,
        {**space.default, t: space.paper_level},
        {**space.default, t: space.baseline_level},
    ]
    biggest = {f.name: f.levels[-1] for f in space.factors}
    probes.append(biggest)
    # the method may only matter in some contexts, so also compare the arms at the far corner
    probes += [{**biggest, t: space.paper_level}, {**biggest, t: space.baseline_level}]
    jobs = [(c, s) for c in probes for s in (0, 1)]
    results = asyncio.run(run_jobs(env_dir / "lab.py", jobs, workers=4, timeout=30))
    errors = []
    for (config, seed), r in zip(jobs, results, strict=True):
        if not r["ok"]:
            errors.append(f"run({config}, {seed}) failed: {r['error']}")
        elif space.metric not in r["metrics"]:
            errors.append(f"run() must return the metric {space.metric!r}; got {sorted(r['metrics'])}")
    if errors:
        return sorted(set(errors))[:6], {}

    again = asyncio.run(run_jobs(env_dir / "lab.py", [(probes[1], 0)], workers=1, timeout=30))[0]
    if again["ok"] and again["metrics"] != results[2]["metrics"]:
        errors.append(
            "run() is not deterministic: same config and seed gave different metrics in a fresh process"
        )
    secs = [r["secs"] for r in results]
    median = statistics.median(secs)
    if median > MAX_SECS:
        errors.append(
            f"too slow: median {median:.2f}s per run (aim for under 1.5s; shrink data, model, or steps)"
        )
    paper = [r["metrics"][space.metric] for r in results[2:4] + results[8:10]]
    base = [r["metrics"][space.metric] for r in results[4:6] + results[10:12]]
    info = {
        "median_secs": median,
        "max_secs": max(secs),
        "seed_varies": paper[0] != paper[1] or base[0] != base[1],
        "arms_differ": paper != base,
    }
    if not info["arms_differ"]:
        errors.append(
            "paper and baseline arms return identical metrics at both the default and the largest "
            "config; the treatment factor seems to have no effect in code"
        )
    return errors, info


def preflight(env_dir: Path) -> tuple[list[str], dict]:
    env_dir = Path(env_dir)
    errors = static_errors((env_dir / "lab.py").read_text())
    space, more = space_errors(env_dir)
    errors += more
    if errors or space is None:
        return errors, {}
    return smoke_errors(env_dir, space)
