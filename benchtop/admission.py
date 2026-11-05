"""Read-only lab screening with separate technical, question, and human-review evidence."""

import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

from benchtop.cache import digest, file_hash, runtime_identity
from benchtop.env import ENVS, ROOT, Env, load_catalog
from benchtop.stats import GRADER_ID
from benchtop.tasks import build_task_set
from benchtop.validate import static_errors

ADMISSION_ID = "cached_admission_v1"
REVIEW_CHECKS = (
    "mechanism_fidelity", "baseline_fairness", "shared_randomness", "heldout_metric",
    "meaningful_margin", "execution_behavior", "shortcut_audit",
)
SCOPE_TEXT = ("paper_mechanism", "miniature_comparison")
SCOPE_LISTS = ("preserved_assumptions", "simplifications", "unsupported_claims")
INPUTS = ("lab.py", "space.json", "manifest.json", "results.jsonl", "tasks.json", "audit.json")


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _messages(error, env_dir):
    return str(error).replace(str(env_dir.resolve()), env_dir.name).replace(str(ROOT), ".")


def _qualification_seeds(env):
    stored = json.loads((env.dir / "tasks.json").read_text())
    declared = stored.get("seed_ids") if isinstance(stored, dict) else None
    task_sources = [task["seed_ids"] for task in env.tasks if "seed_ids" in task]
    if task_sources:
        declared = task_sources[0] if declared is None else declared
        if any(seeds != declared for seeds in task_sources):
            raise ValueError("task artifacts use inconsistent qualification seed sources")
    splits = env.manifest.get("seed_splits")
    if splits is not None:
        if not isinstance(splits, dict) or set(splits) != {"visible", "qualification", "confirmation"}:
            raise ValueError("seed_splits must name visible, qualification, and confirmation")
        groups = list(splits.values())
        if any(not isinstance(group, list) for group in groups):
            raise ValueError("seed groups must be lists")
        flat = [seed for group in groups for seed in group]
        if any(type(seed) is not int for seed in flat) or len(set(flat)) != len(flat) or set(flat) != set(range(env.seeds)):
            raise ValueError("seed groups must be disjoint and cover the declared seeds")
        if splits["visible"] != env.visible_seeds or len(splits["confirmation"]) < 2:
            raise ValueError("visible or confirmation seed group is invalid")
        seeds, source = splits["qualification"], "explicit_qualification_partition"
        if declared is not None and declared != seeds:
            raise ValueError("task seed source differs from the qualification partition")
    else:
        seeds = env.heldout_seeds if declared is None else declared
        source = "historical_heldout" if declared is None else "task_qualification_source"
    if (not isinstance(seeds, list) or len(seeds) < 2 or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds) or not set(seeds) <= set(env.heldout_seeds)):
        raise ValueError("qualification seeds must be unique held-out integers, with at least two seeds")
    return seeds, source


def _review(review, subject):
    result = {"status": "missing", "decision": "needs_review", "issues": [], "identity_source": "operator_reported"}
    if review is None:
        result["issues"] = ["A human scientific review is required; an LLM audit is not admission."]
        return result
    if not isinstance(review, dict) or type(review.get("schema_version")) is not int or review["schema_version"] != 1:
        return {**result, "status": "invalid", "issues": ["unsupported human-review schema"]}
    try:
        result["record_sha256"] = digest(review)
    except (TypeError, ValueError):
        return {**result, "status": "invalid", "issues": ["review must contain finite JSON values"]}
    if review.get("subject") != subject:
        return {**result, "status": "stale", "issues": ["review is bound to different lab, qualification, or charging artifacts"]}
    reviewer = review.get("reviewer", {})
    if not isinstance(reviewer, dict) or reviewer.get("kind") != "human":
        return {**result, "status": "draft", "issues": ["review is not attributed to a human reviewer"]}
    issues = []
    if not _text(reviewer.get("name")):
        issues.append("reviewer name is missing")
    try:
        date = datetime.fromisoformat(reviewer.get("reviewed_at_utc", ""))
        if date.tzinfo is None or date.utcoffset().total_seconds() != 0:
            raise ValueError("UTC timestamp required")
    except (TypeError, ValueError):
        issues.append("reviewer needs a UTC review timestamp")
    decision = review.get("decision")
    if decision not in ("accepted", "rejected", "needs_review"):
        issues.append("unknown review decision")
    if not _text(review.get("rationale")):
        issues.append("decision rationale is missing")
    paper = review.get("paper", {})
    if not isinstance(paper, dict) or not _text(paper.get("url")) or not _text(paper.get("version")):
        issues.append("paper URL and exact version are required")
    if not isinstance(paper, dict) or not isinstance(paper.get("sections"), list) or not paper["sections"] or not all(_text(x) for x in paper["sections"]):
        issues.append("paper section references are required")
    scope = review.get("scope", {})
    if not isinstance(scope, dict):
        issues.append("scope is missing")
    elif decision == "accepted":
        if any(not _text(scope.get(key)) for key in SCOPE_TEXT):
            issues.append("accepted review needs the paper mechanism and miniature comparison")
        if any(not isinstance(scope.get(key), list) or not scope[key] or not all(_text(x) for x in scope[key]) for key in SCOPE_LISTS):
            issues.append("accepted review needs preserved assumptions, simplifications, and unsupported claims")
    checks = review.get("checks", {})
    failed = []
    if not isinstance(checks, dict) or set(checks) != set(REVIEW_CHECKS):
        issues.append("review must contain all named scientific checks")
    else:
        for key, check in checks.items():
            if not isinstance(check, dict) or check.get("status") not in ("pass", "fail", "unresolved"):
                issues.append(f"{key}: invalid status")
                continue
            evidence = check.get("evidence")
            valid = isinstance(evidence, list) and bool(evidence) and all(
                isinstance(item, dict) and all(_text(item.get(field)) for field in ("source", "locator", "finding"))
                for item in evidence
            )
            if check["status"] in ("pass", "fail") and not valid:
                issues.append(f"{key}: paper/code/measurement evidence is required")
            if check["status"] == "fail" and valid:
                failed.append(key)
            if decision == "accepted" and check["status"] != "pass":
                issues.append(f"{key}: accepted review cannot leave a failed or unresolved check")
    if decision == "rejected" and not failed:
        issues.append("rejection needs at least one evidenced failed scientific check")
    if issues:
        return {**result, "status": "incomplete", "issues": issues}
    return {**result, "status": "recorded", "decision": decision, "reviewer": reviewer,
            "paper": paper, "scope": scope, "checks": checks, "rationale": review["rationale"]}


def inspect_lab(env_dir: Path, *, review=None):
    env_dir = Path(env_dir)
    report = {
        "schema_version": 1, "admission_id": ADMISSION_ID, "env": env_dir.name,
        "scope": "cached_reference_screen", "decision": "needs_review", "qualified_claim_admitted": False,
        "analysis_runtime": runtime_identity(),
        "input_sha256": {},
        "generator_sha256": {f"benchtop/{name}": file_hash(ROOT / "benchtop" / name) for name in
                             ("admission.py", "cache.py", "env.py", "space.py", "stats.py", "tasks.py", "validate.py")},
        "mechanical": {"status": "not_checked", "checks": []},
        "qualification": {"status": "not_checked", "grader_id": GRADER_ID},
        "human_review": {"status": "not_checked", "decision": "needs_review"}, "review_subject": None,
        "runtime_probes": {"status": "not_run", "reason": "This screen reads caches; it does not execute lab code."},
        "model_difficulty": {"status": "not_measured", "reason": "Difficulty requires the later model calibration study."},
        "advisories": [],
        "limits": ["Static checks do not establish scientific fidelity or security.",
                   "Matching seed keys do not prove the arms use identical data or initialization.",
                   "Human identity and evidence are operator-reported, not independently authenticated.",
                   "This report does not change catalog membership or satisfy release lock gates."],
    }

    def check(name, passed, **evidence):
        report["mechanical"]["checks"].append({"id": name, "status": "passed" if passed else "failed", **evidence})

    def rejected():
        report["mechanical"]["status"] = "failed"
        report["decision"] = "rejected"
        return report

    try:
        for name in INPUTS[:5]:
            if not (env_dir / name).is_file():
                raise FileNotFoundError(f"missing required artifact: {name}")
        report["input_sha256"] = {name: file_hash(env_dir / name) for name in INPUTS if (env_dir / name).is_file()}
        env = Env.load(env_dir, replay=True)
    except (OSError, ValueError, TypeError, KeyError) as error:
        check("input_integrity", False, issue=_messages(error, env_dir))
        return rejected()
    report["env"] = env.id
    report["execution_id"], report["source_grader_id"] = env.execution_id, env.grader_id
    report["cache_mode"] = env.cache_status
    check("input_integrity", True, execution_id=env.execution_id)
    errors = static_errors((env_dir / "lab.py").read_text())
    check("static_contract", not errors, issues=errors)
    errors = []
    if env.space.size() > 48 or not env.space.context_factors():
        errors.append("paper labs require context factors and a grid of at most 48 configurations")
    if not math.isfinite(env.space.min_effect) or env.space.min_effect <= 0:
        errors.append("meaningful-effect margin must be finite and positive")
    paper = env.raw_space.get("paper", {})
    if not isinstance(paper, dict) or not _text(paper.get("claim")):
        errors.append("paper.claim is required as a claim to review, not as verified truth")
    check("lab_contract", not errors, issues=errors, configurations=env.space.size())
    if any(item["status"] == "failed" for item in report["mechanical"]["checks"]):
        return rejected()
    try:
        seeds, source = _qualification_seeds(env)
        artifact = build_task_set(env, seeds=seeds)
        claim = next((row for row in artifact["contrasts"] if row["task_id"] == "claim"), None)
        report["qualification"] = {
            "status": artifact["status"], "grader_id": GRADER_ID, "seed_ids": seeds, "seed_source": source,
            "artifact_sha256": artifact["artifact_sha256"], "issues": artifact["issues"],
            "eligible_task_ids": [task["id"] for task in artifact["tasks"]],
            "claim": claim, "claim_eligible": claim is not None and claim["label"] is not None,
            "interpretation": "retrospective development only" if env.cache_status == "historical" else "qualification only",
        }
        check("qualification_data", artifact["status"] == "complete", data_status=artifact["status"],
              issues=artifact["issues"], expected_cells=len(seeds) * env.space.size())
        costs, charging_id = env.charging_schedule()
        check("charging_schedule", True, schedule_id=charging_id, minimum=min(costs.values()), maximum=max(costs.values()))
    except (OSError, ValueError, TypeError, KeyError) as error:
        check("qualification_inputs", False, issue=_messages(error, env_dir))
        return rejected()
    if any(item["status"] == "failed" for item in report["mechanical"]["checks"]):
        return rejected()
    report["mechanical"]["status"] = "passed"
    report["review_subject"] = {"contract": ADMISSION_ID, "env": env.id, "execution_id": env.execution_id,
                                "lab_sha256": report["input_sha256"]["lab.py"],
                                "space_sha256": report["input_sha256"]["space.json"],
                                "qualification_artifact_sha256": artifact["artifact_sha256"], "charging_id": charging_id}
    audit_path = env_dir / "audit.json"
    if audit_path.exists():
        try:
            audit = json.loads(audit_path.read_text())
            digest(audit)
            flags = {key: audit.get(key) for key in ("tests_core_claim", "baseline_fair", "same_data_across_arms", "metric_on_heldout", "fidelity")}
            report["advisories"].append({"id": "prior_llm_audit", "source": "audit.json", "auditor": audit.get("auditor"),
                                         "flags": flags, "issues": audit.get("issues", []), "certifies_admission": False})
        except (ValueError, TypeError, AttributeError):
            report["advisories"].append({"id": "prior_llm_audit_unreadable", "certifies_admission": False})
    claim_task = next((task for task in artifact["tasks"] if task["id"] == "claim"), None)
    if claim_task is not None and claim_task["top_share"] >= 0.8:
        report["advisories"].append({"id": "context_dominance", "top_share": claim_task["top_share"],
                                     "interpretation": "Review signal only; legitimate interactions can dominate an average."})
    if claim is not None and claim["label"] is None:
        report["advisories"].append({"id": "ambiguous_claim", "interpretation": "No claim reward is available; this alone does not prove a lab is invalid."})
    report["human_review"] = _review(review, report["review_subject"])
    report["decision"] = report["human_review"]["decision"]
    report["qualified_claim_admitted"] = report["decision"] == "accepted" and report["qualification"]["claim_eligible"]
    return report


def draft_review(report):
    if report["review_subject"] is None:
        raise ValueError("resolve mechanical failures before creating a scientific review")
    return {
        "schema_version": 1, "subject": report["review_subject"], "decision": "needs_review",
        "reviewer": {"kind": "unassigned", "name": "", "reviewed_at_utc": ""},
        "paper": {"url": "", "version": "", "sections": []},
        "scope": {**dict.fromkeys(SCOPE_TEXT, ""), **{key: [] for key in SCOPE_LISTS}},
        "checks": {key: {"status": "unresolved", "evidence": []} for key in REVIEW_CHECKS}, "rationale": "",
    }


def screen_catalog(envs_dir=ENVS, *, reviews_dir=None):
    envs_dir = Path(envs_dir)
    catalog = load_catalog(envs_dir)
    rows = []
    for name in catalog["environments"]:
        path = Path(reviews_dir) / f"{name}.json" if reviews_dir is not None else None
        review = None
        if path is not None and path.exists():
            try:
                review = json.loads(path.read_text())
            except (OSError, ValueError):
                review = {"schema_version": 0}
        row = inspect_lab(envs_dir / name, review=review)
        if path is not None and path.is_file():
            try:
                row["human_review"]["input_sha256"] = file_hash(path)
            except OSError:
                row["human_review"].update(status="invalid", decision="needs_review", issues=["review file could not be read"])
                row["decision"], row["qualified_claim_admitted"] = "needs_review", False
        if row["env"] != name:
            row["mechanical"]["checks"].append({"id": "catalog_identity", "status": "failed", "issue": "manifest ID differs from catalog directory"})
            row["mechanical"]["status"], row["decision"], row["qualified_claim_admitted"] = "failed", "rejected", False
        if row.get("source_grader_id") is not None and row["source_grader_id"] != catalog["grader_id"]:
            row["mechanical"]["checks"].append({"id": "catalog_grader", "status": "failed", "issue": "catalog and source task graders differ"})
            row["mechanical"]["status"], row["decision"], row["qualified_claim_admitted"] = "failed", "rejected", False
        row["analysis_runtime_sha256"] = digest(row.pop("analysis_runtime"))
        rows.append(row)
    decisions = Counter(row["decision"] for row in rows)
    return {"schema_version": 1, "admission_id": ADMISSION_ID, "scope": "reference_admission_screen",
            "analysis_runtime": runtime_identity(), "reviews_supplied": reviews_dir is not None,
            "catalog_sha256": file_hash(envs_dir / "catalog.json"), "catalog_id": catalog["id"],
            "summary": {"labs": len(rows), "mechanically_passed": sum(row["mechanical"]["status"] == "passed" for row in rows),
                        "decisions": {key: decisions[key] for key in ("accepted", "rejected", "needs_review")},
                        "qualified_claims_before_review": sum(row["qualification"].get("claim_eligible", False) for row in rows),
                        "qualified_claims_admitted": sum(row["qualified_claim_admitted"] for row in rows)}, "labs": rows}


def write_json(path, value, *, draft=False):
    path = Path(path).resolve()
    allowed = (ROOT / "artifacts", ROOT / "benchmarks/release_v1/admission")
    if path.is_relative_to(ROOT) and not any(path.is_relative_to(directory) for directory in allowed):
        raise ValueError("write admission outputs under artifacts/ or benchmarks/release_v1/admission/, or outside the repository")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if draft else "w") as output:
        output.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
