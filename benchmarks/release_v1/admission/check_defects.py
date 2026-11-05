"""Run explicitly scoped semantic fixtures without changing historical lab artifacts."""

import argparse
import asyncio
import json
import math
import shutil
import tempfile
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from benchtop.cache import CacheError, digest, file_hash, runtime_identity
from benchtop.env import ROOT, Env, Table, read_jsonl
from benchtop.sandbox import Pool, sandbox_mode
from benchtop.tasks import build_task_set

DEFECTS = Path(__file__).resolve().with_name("defects")
MUON = "2610-37225-sharpnessaware-muon-matrixaware-optimization"
CURATION = "2610-36983-advancement-or-acceleration-how"
FINGERPRINTS = ("train", "test", "batches", "initial_parameters")
USER_METRICS = ("test_acc", "train_acc", "gap", "test_loss", "sharpness")


def mutate_once(source, before, after):
    if not before or before == after or source.count(before) != 1:
        raise ValueError("fixture mutation needs exactly one source match")
    return source.replace(before, after, 1)


def load_manifest():
    manifest = json.loads((DEFECTS / "manifest.json").read_text())
    if manifest["schema_version"] != 1 or set(manifest["sources"]) != {MUON, CURATION}:
        raise ValueError("unsupported fixture manifest")
    if (manifest["muon"]["env"] != MUON or manifest["cached"]["env"] != CURATION
            or manifest["muon"]["seeds"] != [0, 1]
            or manifest["muon"]["config"] != {"batch_size": 16, "rho": 0.3, "train_n": 200}):
        raise ValueError("unsupported fixture probe profile; version the suite before changing its scope")
    mutations = manifest["muon"]["mutations"]
    expected = {"disabled_treatment": "muon.momentum_perturbation",
                "advantaged_baseline": "muon.heldout_disjoint",
                "inconsistent_shared_randomness": "muon.common_inputs"}
    if (len(mutations) != len(expected)
            or {m["id"]: m["guard"] for m in mutations} != expected
            or any(m["expected"] != "fail" for m in mutations)):
        raise ValueError("fixture manifest must retain all three named runtime mutations")
    for env, files in manifest["sources"].items():
        if set(files) != {"lab.py", "space.json", "manifest.json", "results.jsonl", "tasks.json"}:
            raise ValueError("fixture manifest must bind all five historical source files")
        for name, sha in files.items():
            if file_hash(ROOT / "envs" / env / name) != sha:
                raise ValueError(f"fixture source changed: {env}/{name}")
    return manifest


def case(name, guard, expected, observed, evidence, *, kind="control"):
    return {"id": name, "kind": kind, "guard": guard, "expected": expected, "observed": observed,
            "result": "pass" if expected == observed else "fail", "evidence": evidence}


def accuracy_direction(raw):
    valid = raw["metric"]["name"] == "test_acc" and raw["metric"]["goal"] == "max"
    return "pass" if valid else "fail"


def claim_guard(env):
    artifact = build_task_set(env)
    claim = next(c for c in artifact["contrasts"] if c["task_id"] == "claim")
    eligible = "claim" in {task["id"] for task in artifact["tasks"]}
    return "eligible" if eligible else "unavailable", {
        "data_status": artifact["status"], "claim": claim, "seed_ids": artifact["seed_ids"],
        "scientific_admission": "needs_review",
    }


def cached_cases(manifest, work):
    """Use validated historical inputs; counterfactual metrics remain in memory only."""
    cases = []
    raw = json.loads((ROOT / "envs" / MUON / "space.json").read_text())
    metric_recipe = manifest["cached"]["reversed_metric"]
    for name, goal, expected in (("valid_metric", metric_recipe["from_goal"], "pass"),
                                 ("reversed_metric", metric_recipe["to_goal"], "fail")):
        raw["metric"]["goal"] = goal
        cases.append(case(name, "muon.accuracy_direction", expected, accuracy_direction(raw),
                          {"metric": "test_acc", "declared_goal": goal,
                           "basis": f"envs/{MUON}/lab.py:L106-L108 defines fraction correctly classified; larger is better."},
                          kind="defect" if expected == "fail" else "control"))

    path = work / "cache" / CURATION
    path.mkdir(parents=True)
    for name in manifest["sources"][CURATION]:
        shutil.copyfile(ROOT / "envs" / CURATION / name, path / name)
    shutil.copyfile(ROOT / "envs/historical_cache.json", path.parent / "historical_cache.json")
    Env.load(path, replay=True)
    cases.append(case("valid_cache", "cache.source_binding", "pass", "pass",
                      {"cache_mode": "verified_historical_copy"}))
    recipe = manifest["cached"]["stale_cache"]
    source = (path / "lab.py").read_text()
    (path / "lab.py").write_text(mutate_once(source, recipe["before"], recipe["after"]))
    try:
        Env.load(path, replay=True)
    except CacheError as error:
        outcome, reason = "fail", str(error)
    else:
        outcome, reason = "pass", "stale observations were unexpectedly accepted"
    cases.append(case("stale_cache", recipe["guard"], "fail", outcome,
                      {"mutation": "Change target-label count 40 to 41 while retaining the old cache.", "message": reason}, kind="defect"))

    original = Env.load(ROOT / "envs" / CURATION, replay=True)
    for name, env in (("valid_null_claim", original), ("valid_negative_claim", Env.load(ROOT / "envs" / MUON, replay=True))):
        outcome, evidence = claim_guard(env)
        expected_label = "no_meaningful_difference" if name == "valid_null_claim" else "baseline_better"
        evidence["expected_control_label"] = expected_label
        if evidence["claim"]["label"] != expected_label:
            outcome = "unexpected_control_label"
        cases.append(case(name, "qualification.claim_interval", "eligible", outcome, evidence))
    rows = read_jsonl(original.dir / "results.jsonl")
    offset = manifest["cached"]["ambiguous_reward"]["treatment_seed_offset"]
    for row in rows:
        if row["seed"] in original.heldout_seeds and row["config"]["method"] == "target_select":
            value = row["metrics"]["target_acc"] + offset * (1 if row["seed"] % 2 else -1)
            row["metrics"]["target_acc"] = min(1.0, max(0.0, value))
    counterfactual = replace(original, table=Table(original.space, rows), tasks=[])
    outcome, evidence = claim_guard(counterfactual)
    evidence.update(counterfactual=True, transform=manifest["cached"]["ambiguous_reward"]["transform"],
                    interpretation="Question unavailable, not a scientific rejection; no transformed row is persisted as lab data.")
    cases.append(case("ambiguous_reward", "qualification.claim_interval", "unavailable", outcome, evidence, kind="defect"))
    return cases


def unpack_metrics(metrics):
    traces = {}
    for name in FINGERPRINTS:
        traces[name] = "".join(f"{int(metrics[f'fixture.{name}.{i}']):08x}" for i in range(8))
    traces.update({key.removeprefix("fixture."): value for key, value in metrics.items()
                   if key.startswith("fixture.") and key.split(".")[1] not in FINGERPRINTS})
    return {key: metrics[key] for key in USER_METRICS}, traces


async def execute_source(work, name, source, jobs, *, instrumented):
    directory = work / name
    directory.mkdir()
    lab = directory / "lab.py"
    lab.write_text(source)
    rows = []
    async with Pool(lab, workers=1, timeout=60) as pool:
        for config, seed in jobs:
            result = await pool.run(config, seed)
            if not result.get("ok"):
                raise RuntimeError(f"{name}: sandbox probe failed: {result.get('error')}")
            metrics, traces = unpack_metrics(result["metrics"]) if instrumented else (result["metrics"], {})
            rows.append({"config": config, "seed": seed, "metrics": metrics, "traces": traces})
    return {"variant": name, "executed_source_sha256": file_hash(lab), "observations": rows}


def muon_guard(guard, observations):
    if not observations:
        raise ValueError("a runtime guard needs observations")
    if guard == "muon.momentum_perturbation":
        rows = [r for r in observations if r["config"]["method"] == "momusam" and r["config"]["rho"] > 0]
        if not rows:
            raise ValueError("momentum guard needs nonzero-radius treatment observations")
        evidence = [{"seed": r["seed"], **{k: r["traces"][k] for k in (
            "gradient_calls", "training_perturbation_calls", "first_training_perturbation_norm", "later_max_perturbation_norm")}} for r in rows]
        valid = all(r["gradient_calls"] == 151 and r["training_perturbation_calls"] == 150
                    and r["first_training_perturbation_norm"] == 0 and r["later_max_perturbation_norm"] > 1e-8 for r in evidence)
    elif guard == "muon.heldout_disjoint":
        rows = [r for r in observations if r["config"]["method"] == "muon"]
        if not rows:
            raise ValueError("held-out guard needs baseline observations")
        evidence = [{"seed": r["seed"], "train_test_overlap": r["traces"]["train_test_overlap"]} for r in rows]
        valid = all(r["train_test_overlap"] == 0 for r in evidence)
    elif guard == "muon.common_inputs":
        groups = defaultdict(dict)
        for row in observations:
            config = {k: v for k, v in row["config"].items() if k != "method"}
            group = groups[(row["seed"], digest(config))]
            if row["config"]["method"] in group:
                raise ValueError("common-input guard received duplicate method/seed/context observations")
            group[row["config"]["method"]] = row["traces"]
        evidence = []
        for (seed, context), methods in groups.items():
            if not {"muon", "momusam"} <= methods.keys():
                raise ValueError("common-input guard needs paired baseline and treatment")
            mismatches = [k for k in FINGERPRINTS if methods["muon"][k] != methods["momusam"][k]]
            evidence.append({"seed": seed, "context_sha256": context, "mismatched_streams": mismatches})
        valid = all(not row["mismatched_streams"] for row in evidence)
    else:
        raise ValueError(f"unknown Muon runtime guard: {guard}")
    return "pass" if valid else "fail", evidence


def metrics_equal(a, b):
    return all(math.isclose(a[k], b[k], abs_tol=1e-12, rel_tol=1e-12) for k in USER_METRICS)


async def live_cases(manifest, work):
    profile = manifest["muon"]
    original = (ROOT / "envs" / MUON / "lab.py").read_text()
    source = mutate_once(original, profile["instrumentation_anchor"],
                         profile["instrumentation_anchor"] + profile["instrumentation_insert"])
    source += "\n\n" + (DEFECTS / "muon_observer.py").read_text() + "\ninstall_probe(globals())\n"
    jobs = [({**profile["config"], "method": method}, seed)
            for seed in profile["seeds"] for method in ("muon", "momusam")]
    jobs.append(({**profile["config"], "method": "musam"}, profile["seeds"][0]))
    zero_jobs = [({**profile["config"], "rho": 0.0, "method": method}, profile["seeds"][0]) for method in ("muon", "momusam")]
    control = await execute_source(work, "valid_muon", source, jobs + zero_jobs, instrumented=True)
    repeat = await execute_source(work, "fresh_worker", source, jobs[:2], instrumented=True)
    pristine_jobs = [job for job in jobs if job[1] == profile["seeds"][0]]
    pristine = await execute_source(work, "pristine_muon", original, pristine_jobs, instrumented=False)
    cases = []
    for name, guard in (("valid_treatment", "muon.momentum_perturbation"),
                        ("valid_holdout", "muon.heldout_disjoint"), ("valid_shared_inputs", "muon.common_inputs")):
        outcome, evidence = muon_guard(guard, control["observations"][:4])
        cases.append(case(name, guard, "pass", outcome, evidence))
    corresponding = {digest([r["config"], r["seed"]]): r for r in control["observations"]}
    faithful = all(metrics_equal(r["metrics"], corresponding[digest([r["config"], r["seed"]])]["metrics"])
                   for r in pristine["observations"])
    cases.append(case("observer_transparency", "fixture.observer_transparency", "pass", "pass" if faithful else "fail",
                      {"compared_methods": [r["config"]["method"] for r in pristine["observations"]], "metrics": list(USER_METRICS)}))
    identical = all(r == corresponding[digest([r["config"], r["seed"]])] for r in repeat["observations"])
    cases.append(case("fresh_worker_repeatability", "fixture.fresh_worker_repeatability", "pass", "pass" if identical else "fail",
                      {"repeated_jobs": len(repeat["observations"]), "comparison": "all metrics and complete input/perturbation traces"}))
    zero = control["observations"][-2:]
    identical = metrics_equal(zero[0]["metrics"], zero[1]["metrics"])
    cases.append(case("zero_radius_identity", "muon.zero_radius_identity", "pass", "pass" if identical else "fail",
                      {"radius": 0.0, "methods": [r["config"]["method"] for r in zero], "metrics": list(USER_METRICS)}))
    runs = [control, repeat, pristine]
    for recipe in profile["mutations"]:
        mutant = mutate_once(source, recipe["before"], recipe["after"])
        run = await execute_source(work, recipe["id"], mutant, jobs[:4], instrumented=True)
        outcome, evidence = muon_guard(recipe["guard"], run["observations"])
        cases.append(case(recipe["id"], recipe["guard"], recipe["expected"], outcome, evidence, kind="defect"))
        runs.append(run)
    return cases, runs


def run_suite(*, live=True, output=None):
    target = Path(output).resolve() if output is not None else None
    if (target is not None and target.is_relative_to(ROOT) and target != DEFECTS / "report.json"
            and not target.is_relative_to(ROOT / "artifacts")):
        raise ValueError("write under artifacts/, to defects/report.json, or outside the repository")
    manifest = load_manifest()
    if live:
        sandbox_mode()  # Fail closed; never fall back to importing/executing a lab in the host process.
    inputs = {f"envs/{env}/{name}": sha for env, files in manifest["sources"].items() for name, sha in files.items()}
    for path in (Path(__file__).resolve(), DEFECTS / "manifest.json", DEFECTS / "muon_observer.py",
                 DEFECTS.parent / "references/selection.json", ROOT / "envs/historical_cache.json"):
        inputs[str(path.relative_to(ROOT))] = file_hash(path)
    for name in ("admission.py", "cache.py", "env.py", "tasks.py", "stats.py", "space.py", "sandbox.py", "_worker.py"):
        inputs[f"benchtop/{name}"] = file_hash(ROOT / "benchtop" / name)
    with tempfile.TemporaryDirectory(prefix="benchtop-semantic-fixtures-") as tmp:
        work = Path(tmp)
        cases = cached_cases(manifest, work)
        if live:
            observed, runs = asyncio.run(live_cases(manifest, work))
            cases.extend(observed)
        else:
            runs = []
    unchanged = all(file_hash(ROOT / name) == sha for name, sha in inputs.items())
    defects = [c for c in cases if c["kind"] == "defect"]
    failed = [c["id"] for c in cases if c["result"] != "pass"]
    result = {
        "schema_version": 1, "suite_id": manifest["suite_id"], "checked_at_utc": datetime.now(UTC).isoformat(),
        "status": "failed" if failed or not unchanged else "passed" if live else "partial",
        "scope": manifest["scope"], "analysis_runtime": runtime_identity(), "input_sha256": inputs,
        "summary": {"cases": len(cases), "expectations_met": len(cases) - len(failed), "failed_cases": failed,
                    "controlled_defects_observed": len(defects), "controlled_defects_planned": 6,
                    "valid_controls": sum(c["kind"] == "control" for c in cases),
                    "sandbox_jobs": sum(len(r["observations"]) for r in runs), "source_inputs_unchanged": unchanged},
        "cases": cases, "live_runs": runs, "live_probes": "completed_bounded_muon_profile" if live else "not_run",
        "human_admissions": 0, "model_sessions": 0,
        "limits": ["Guard outcomes cover these deliberately constructed cases only; no estimate of general scientific-defect recall.",
                   "Passing controls does not admit a lab or establish paper fidelity, fair baseline capacity, meaningful margins, or fresh reward stability.",
                   "Ambiguity removes question eligibility without a scientific rejection.",
                   "Runtime probes use two visible seeds and one Muon context, with a zero-radius extension; no qualification or confirmation sweep.",
                   "Instrumentation is validated by pristine metric agreement for three methods at one seed; it is not an adversarial security boundary."],
    }
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-only", action="store_true", help="run only cached cases; report partial coverage")
    parser.add_argument("--output", default="artifacts/admission_defects.json")
    args = parser.parse_args()
    result = run_suite(live=not args.cached_only, output=args.output)
    print(json.dumps({"status": result["status"], **result["summary"], "output": args.output}, indent=2))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
