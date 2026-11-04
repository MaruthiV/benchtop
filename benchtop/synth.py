import json
import re
import shutil
import time
from pathlib import Path

from benchtop import arxiv
from benchtop.env import ENVS, STATE
from benchtop.llm import LLMError, complete, fenced, save
from benchtop.validate import preflight

DRAFTS = STATE / "pipeline" / "drafts"

SYNTH = """You are turning a new arXiv paper into a benchtop lab: a small, fast, faithful miniature of the paper's central experiment. Researchers and RL agents will run it many times to test the paper's claim under different conditions.

Write two files.

1. lab.py
- Defines run(config: dict, seed: int) -> dict of floats: the primary metric plus up to 4 diagnostics.
- Deterministic given (config, seed). Derive all randomness from seed (np.random.default_rng(seed), torch.manual_seed(seed), random_state=seed).
- Common random numbers: for a given seed, data, splits, and initializations must be identical across levels of the treatment factor, so the arms are compared on the same draws. Use separate rngs for data and for model init so that changing the method does not shift the data.
- Allowed imports only: numpy, scipy, sklearn, torch, math, random, itertools, functools, collections, dataclasses, typing, statistics, heapq, copy, warnings. No file or network access and no downloads. sklearn load_* bundled datasets and make_* generators are fine; fetch_* is not.
- Fast: about 0.2 to 1.5 seconds per run on one CPU thread, even at the largest factor levels. Keep data and models small and use a fixed number of steps.
- Implement the paper's method faithfully at small scale, and the baseline with equal care (same budget and tuning effort). Don't rig the comparison either way.
- Measure the primary metric on held-out data or fresh samples, never on training data.
- Module-level code may load a bundled dataset or set constants, but must not use randomness.

2. space.json
{{
  "title": "short lab title",
  "setting": "2 or 3 sentences for a researcher who has not read the code: data, model, training, what the metric measures",
  "paper": {{"title": "...", "arxiv_id": "{arxiv_id}", "claim": "one sentence: the paper's claim this lab tests"}},
  "metric": {{"name": "key returned by run()", "goal": "max or min", "min_effect": 0.01, "description": "..."}},
  "factors": [
    {{"name": "method", "role": "treatment", "levels": ["<baseline>", "<paper method>"], "description": "..."}},
    {{"name": "...", "role": "context", "levels": [2 or 3 values], "description": "..."}}
  ],
  "treatment": {{"factor": "method", "paper": "<paper method level>", "baseline": "<baseline level>"}}
}}
- The treatment factor may have a third level (another reasonable alternative). Add 2 or 3 context factors where the paper's effect might plausibly change (data size, noise, dimension, shift strength, model size, imbalance...).
- Levels are short strings, ints, floats, or booleans. The full grid (product of all level counts) must be 48 configs or fewer.
- Pick levels so the question is open: include settings where the claim might fail as well as where it should hold.
- min_effect is the smallest difference in metric units a researcher would care about (e.g. 0.01 for accuracy, about 5% of a typical loss).

Return exactly two fenced blocks, ```python for lab.py then ```json for space.json, and nothing else.

## Paper
Title: {title}
arXiv: {arxiv_id}
Abstract: {abstract}

## Screening notes (from a quick read of the abstract; improve on them if the paper text says otherwise)
{card}

## Paper text excerpt
{excerpt}
"""

REPAIR = """Your benchtop lab failed validation. Fix every problem and return both files in full, in the same format (```python then ```json, nothing else).

Validation errors:
{errors}

Your lab.py:
```python
{lab}
```

Your space.json:
```json
{space}
```

Reminder of the original task:
{task}
"""


def slug(arxiv_id: str, title: str) -> str:
    words = re.sub(r"[^a-z0-9 ]", "", title.lower()).split()[:4]
    return f"{arxiv_id.replace('.', '-')}-" + "-".join(words)


def card_text(card: dict) -> str:
    keep = ("claim", "method", "baseline", "metric", "contexts", "miniature", "fidelity")
    return json.dumps({k: card[k] for k in keep if k in card}, indent=1)


def synthesize(
    paper: dict,
    card: dict,
    backend: str = "claude",
    model: str = "sonnet",
    rounds: int = 3,
    excerpt_chars: int = 24_000,
) -> dict:
    draft = DRAFTS / slug(paper["id"], paper["title"])
    if draft.exists():
        shutil.rmtree(draft)
    draft.mkdir(parents=True)
    text = arxiv.fulltext(paper["id"]) or "(no html full text available; work from the abstract)"
    task = SYNTH.format(
        title=paper["title"],
        arxiv_id=paper["id"],
        abstract=paper["abstract"],
        card=card_text(card),
        excerpt=text[:excerpt_chars],
    )
    prompt, history = task, []
    for attempt in range(rounds + 1):
        started = time.time()
        try:
            out = complete(prompt, backend=backend, model=model, timeout=900)
        except LLMError as e:
            history.append({"attempt": attempt, "errors": [str(e)[:300]], "llm_error": True})
            break
        lab, space = fenced(out["text"], "python"), fenced(out["text"], "json")
        if not lab or not space:
            errors, info = ["reply must contain a ```python block and a ```json block"], {}
            lab, space = lab or "", space or ""
        else:
            save(draft / "lab.py", lab)
            save(draft / "space.json", space)
            errors, info = preflight(draft)
        history.append(
            {
                "attempt": attempt,
                "errors": errors,
                "info": info,
                "llm_secs": out["secs"],
                "api_equiv_usd": out.get("api_equiv_usd"),
                "secs": round(time.time() - started, 1),
            }
        )
        save(draft / f"attempt{attempt}.txt", out["text"])
        if not errors:
            return {"ok": True, "dir": str(draft), "attempts": attempt + 1, "history": history}
        prompt = REPAIR.format(
            errors="\n".join(f"- {e}" for e in errors), lab=lab.strip(), space=space.strip(), task=task[:6000]
        )
    return {"ok": False, "dir": str(draft), "attempts": len(history), "history": history}


def promote(draft: Path, manifest: dict) -> Path:
    target = ENVS / Path(draft).name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in ("lab.py", "space.json"):
        shutil.copyfile(Path(draft) / name, target / name)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return target
