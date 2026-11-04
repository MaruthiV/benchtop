import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchtop import arxiv
from benchtop.env import ENVS
from benchtop.llm import LLMError, complete

SCHEMA = {
    "type": "object",
    "properties": {
        "tests_core_claim": {"type": "boolean"},
        "fidelity": {"type": "string", "enum": ["high", "medium", "low"]},
        "baseline_fair": {"type": "boolean"},
        "same_data_across_arms": {"type": "boolean"},
        "metric_on_heldout": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": [
        "tests_core_claim",
        "fidelity",
        "baseline_fair",
        "same_data_across_arms",
        "metric_on_heldout",
        "issues",
        "summary",
    ],
}

PROMPT = """You are auditing a "benchtop lab": a miniature, runnable re-test of a paper's central claim that another model wrote. Be skeptical and specific. You did not write it.

Check:
1. tests_core_claim: does the lab test the paper's actual contribution, or only a generic stand-in that happens to share vocabulary?
2. fidelity: how faithfully the miniature captures the paper's method and comparison (high, medium, low).
3. baseline_fair: is the baseline implemented with comparable care and budget, not sabotaged?
4. same_data_across_arms: for a fixed seed, is the data/split/init identical across levels of the treatment factor?
5. metric_on_heldout: is the primary metric computed on held-out data or fresh samples?
6. issues: concrete bugs, leaks, or rigging (quote the line or describe it). Empty list if none.

## Paper
Title: {title}
Abstract: {abstract}

Excerpt:
{excerpt}

## space.json
{space}

## lab.py
{lab}
"""


def audit_one(env_dir: Path, backend: str = "claude", model: str = "opus", force: bool = False) -> dict:
    out_path = env_dir / "audit.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())
    manifest = json.loads((env_dir / "manifest.json").read_text())
    pid = manifest.get("arxiv_id")
    paper = arxiv.lookup(pid) if pid else {"title": manifest.get("paper_title", ""), "abstract": ""}
    excerpt = (arxiv.fulltext(pid) or "")[:16_000] if pid else ""
    prompt = PROMPT.format(
        title=paper["title"],
        abstract=paper["abstract"],
        excerpt=excerpt,
        space=(env_dir / "space.json").read_text(),
        lab=(env_dir / "lab.py").read_text(),
    )
    try:
        res = complete(prompt, backend=backend, model=model, schema=SCHEMA, timeout=600)
        verdict = {"ok": True, "auditor": f"{backend}:{model}", **res["structured"]}
    except LLMError as e:
        verdict = {"ok": False, "auditor": f"{backend}:{model}", "error": str(e)[:300]}
    out_path.write_text(json.dumps(verdict, indent=1) + "\n")
    return verdict


def audit_all(jobs: int = 2, **kw) -> list[dict]:
    dirs = [p.parent for p in sorted(ENVS.glob("*/manifest.json")) if (p.parent / "tasks.json").exists()]
    dirs = [d for d in dirs if json.loads((d / "manifest.json").read_text()).get("source") == "arxiv"]
    with ThreadPoolExecutor(jobs) as pool:
        return list(pool.map(lambda d: {"env": d.name, **audit_one(d, **kw)}, dirs))
