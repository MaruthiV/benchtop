"""Check the six static review drafts against public, cached repository inputs.

Run from the checkout: python -m benchmarks.release_v1.admission.check_references
This does not fetch papers, execute lab code, or certify scientific judgments.
"""

import json
import math
import re
from pathlib import Path

import numpy as np

from benchtop.admission import REVIEW_CHECKS, inspect_lab
from benchtop.cache import file_hash
from benchtop.env import ROOT, Env, load_catalog
from benchtop.tasks import contrast, contrast_pairs, top_share

REFERENCES = Path(__file__).with_name("references")
SPEC = ROOT / "benchmarks/release_v1/spec.json"


def diagnostics(env, report):
    """Retrospective qualification diagnostics, with positive effects favoring treatment."""
    seeds = report["qualification"]["seed_ids"]
    space = env.space
    pairs = contrast_pairs(env, space.treatment, space.paper_level, space.baseline_level, seeds)
    differences = np.array([[a[s] - b[s] for s in seeds] for a, b in pairs])
    block_se = float(differences.mean(axis=0).std(ddof=1) / math.sqrt(len(seeds)))
    independent_se = float(np.sqrt(differences.var(axis=1, ddof=1).sum() / len(seeds)) / len(pairs))
    costs, _ = env.charging_schedule()
    return {
        "seed_ids": seeds,
        "claim": report["qualification"]["claim"],
        "min_effect": space.min_effect,
        "contexts": len(pairs),
        "cost_max_over_min": max(costs.values()) / min(costs.values()),
        "seed_block_se": block_se,
        "context_independence_se": independent_se,
        "se_ratio": block_se / independent_se if independent_se > 0 else None,
        "context_top_share": top_share(env, space.treatment, space.paper_level, space.baseline_level, seeds),
        "effects_by_context_factor": {
            f.name: [{"level": level, "delta": contrast(
                env, space.treatment, space.paper_level, space.baseline_level, seeds,
                fixed={f.name: level},
            )[0]} for level in f.levels]
            for f in space.context_factors()
        },
    }


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same_numbers(actual, expected):
    if type(actual) is float and type(expected) in (float, int):
        return math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_same_numbers(actual[k], expected[k]) for k in actual)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(_same_numbers(a, b) for a, b in zip(actual, expected, strict=True))
    return actual == expected


def check(directory=REFERENCES):
    directory = Path(directory)
    selection = json.loads((directory / "selection.json").read_text())
    spec = json.loads(SPEC.read_text())
    _require(selection["protocol_sha256"] == file_hash(SPEC), "reference protocol binding changed")
    _require(selection["diagnostics_generator_sha256"] == file_hash(Path(__file__)), "diagnostics checker changed")
    rows = selection["cases"]
    ids = [row["env"] for row in rows]
    _require(len(ids) == spec["development"]["reference_labs"] and len(set(ids)) == len(ids), "expected six unique references")
    _require(ids == sorted(ids), "reference cases must be sorted by ID")
    _require(set(ids) <= set(load_catalog(ROOT / "envs")["environments"]), "reference absent from catalog")
    coverage = set().union(*(set(row["coverage"]) for row in rows))
    _require(coverage == set(spec["development"]["coverage"]), "reference coverage does not match protocol")
    for row in rows:
        env = Env.load(ROOT / "envs" / row["env"], replay=True)
        path = directory / "drafts" / f"{env.id}.json"
        _require(row["draft_sha256"] == file_hash(path), f"{env.id}: draft content changed")
        review = json.loads(path.read_text())
        _require(review["reviewer"]["kind"] == "assistant" and review["decision"] == "needs_review", f"{env.id}: expected an assistant draft")
        report = inspect_lab(env.dir, review=review)
        _require(report["mechanical"]["status"] == "passed", f"{env.id}: mechanical screening failed")
        _require(report["human_review"]["status"] == "draft" and not report["qualified_claim_admitted"], f"{env.id}: draft became stale or admitted")
        _require(_same_numbers(diagnostics(env, report), row["diagnostics"]), f"{env.id}: cached diagnostics changed")
        paper = review["paper"]
        version = f"{env.raw_space['paper']['arxiv_id']}v1"
        url = f"https://arxiv.org/html/{version}"
        _require(paper["version"] == version and paper["url"] == url, f"{env.id}: paper identity changed")
        snapshot = row["paper_snapshot"]
        _require(snapshot["url"] == url and snapshot["version"] == version and snapshot["version_marker_verified"], f"{env.id}: unpinned source snapshot")
        for key in ("html_sha256", "text_sha256"):
            _require(bool(re.fullmatch(r"[a-f0-9]{64}", snapshot[key])), f"{env.id}: invalid snapshot hash")
        _require(set(review["checks"]) == set(REVIEW_CHECKS), f"{env.id}: incomplete scientific criteria")
        for name, item in review["checks"].items():
            _require(item["status"] in ("pass", "fail", "unresolved") and item["evidence"], f"{env.id}: missing {name} evidence")
            for evidence in item["evidence"]:
                _require(all(isinstance(evidence.get(k), str) and evidence[k].strip() for k in ("source", "locator", "finding")), f"{env.id}: incomplete citation")
                source = evidence["source"]
                if source.startswith("https://"):
                    _require(source.split("#")[0] == url, f"{env.id}: citation is not the pinned primary paper")
                else:
                    target = (ROOT / source).resolve()
                    _require(target.is_relative_to(ROOT) and target.is_file(), f"{env.id}: absent repository citation")
                    if target.suffix == ".py":
                        match = re.fullmatch(r"L([1-9][0-9]*)(?:-L([1-9][0-9]*))?", evidence["locator"])
                        _require(match is not None, f"{env.id}: invalid code line locator")
                        start, end = int(match[1]), int(match[2] or match[1])
                        _require(start <= end <= len(target.read_text().splitlines()), f"{env.id}: code locator outside file")
    return {"references": len(rows), "coverage_roles": len(coverage), "assistant_drafts": len(rows),
            "human_admissions": 0, "live_probes": "not_run", "model_calls": 0,
            "limits": "Bindings, citations and cached diagnostics checked; scientific judgments and remote snapshot bytes not authenticated."}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
