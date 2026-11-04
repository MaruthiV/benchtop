import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchtop.env import STATE
from benchtop.llm import LLMError, complete

OUT = STATE / "pipeline" / "triage"

BLOCKERS = [
    "none",
    "needs_pretrained_or_large_model",
    "needs_external_data",
    "needs_large_compute",
    "not_empirical",
    "no_clear_comparison",
    "other",
]

SCHEMA = {
    "type": "object",
    "properties": {
        "testable": {"type": "boolean"},
        "blocker": {"type": "string", "enum": BLOCKERS},
        "reason": {"type": "string"},
        "fidelity": {"type": "string", "enum": ["high", "medium", "low"]},
        "claim": {"type": "string"},
        "method": {"type": "string"},
        "baseline": {"type": "string"},
        "metric": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "goal": {"type": "string", "enum": ["max", "min"]},
                "description": {"type": "string"},
            },
            "required": ["name", "goal", "description"],
        },
        "contexts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "levels": {"type": "array"},
                    "why": {"type": "string"},
                },
                "required": ["name", "levels", "why"],
            },
        },
        "miniature": {"type": "string"},
    },
    "required": [
        "testable",
        "blocker",
        "reason",
        "fidelity",
        "claim",
        "method",
        "baseline",
        "metric",
        "contexts",
        "miniature",
    ],
}

PROMPT = """You are screening new machine learning papers for a project that turns papers into small runnable experiments called benchtop labs.

A benchtop lab re-tests a paper's central comparison at miniature scale:
- pure Python with numpy, scipy, scikit-learn, or PyTorch, on CPU
- synthetic data, or datasets bundled inside scikit-learn (load_digits, load_iris, load_wine, load_breast_cancer, load_diabetes, make_* generators). No downloads, no pretrained weights.
- one run takes about one second on a single CPU core
- it compares the paper's proposed method against a fair baseline, and varies 2 or 3 context factors where the effect might plausibly change

Decide whether this paper's core idea can be captured faithfully by such a lab. Many papers can't: they need pretrained LLMs, large datasets, GPUs, or they are pure theory or surveys. Be strict. A lab that only tests something generic and unrelated to the paper's actual contribution has low fidelity and is worthless. If testable is false, still fill every field with your best short attempt or "n/a".

Title: {title}
Categories: {categories}
Abstract: {abstract}
"""


def triage_one(paper: dict, backend: str = "claude", model: str = "haiku", force: bool = False) -> dict:
    path = OUT / f"{paper['id']}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    prompt = PROMPT.format(
        title=paper["title"], categories=", ".join(paper["categories"]), abstract=paper["abstract"]
    )
    try:
        out = complete(prompt, backend=backend, model=model, schema=SCHEMA, timeout=300)
        card = {"id": paper["id"], "title": paper["title"], "ok": True, **out["structured"]}
    except LLMError as e:
        card = {"id": paper["id"], "title": paper["title"], "ok": False, "error": str(e)[:500]}
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=1))
    return card


def selected(card: dict) -> bool:
    return bool(
        card.get("ok")
        and card.get("testable")
        and card.get("blocker") == "none"
        and card.get("fidelity") in ("high", "medium")
    )


def triage_many(papers: list[dict], jobs: int = 4, **kw) -> list[dict]:
    with ThreadPoolExecutor(jobs) as pool:
        return list(pool.map(lambda p: triage_one(p, **kw), papers))


def load_cards(ids: list[str] | None = None) -> list[dict]:
    paths = sorted(OUT.glob("*.json")) if ids is None else [OUT / f"{i}.json" for i in ids]
    return [json.loads(p.read_text()) for p in paths if Path(p).exists()]
