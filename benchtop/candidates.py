import fcntl
import hashlib
import json
from pathlib import Path

from benchtop import calibrate, primary, provider, replay, study, study_calibration, synth, triage
from benchtop import candidate_metadata as metadata
from benchtop.agents.baselines import RELEASE_BASELINES
from benchtop.agents.estimator import ESTIMATOR_ID

FREEZE_CHECKS = ("policy_correctness", "estimator_sensitivity")


def _review(freeze, at):
    calibrate._keys(freeze, {"schema_version", "decision", "reviewer", "checks", "adaptive_allocation", "replay_directory", "replay_summary_sha256"})
    calibrate._keys(freeze["reviewer"], {"kind", "name", "reviewed_at_utc"})
    if (freeze.get("schema_version") != 1 or freeze.get("decision") != "accepted"
            or freeze.get("reviewer", {}).get("kind") != "human"
            or not isinstance(freeze["reviewer"].get("name"), str) or not freeze["reviewer"]["name"].strip()
            or metadata.timestamp(freeze["reviewer"]["reviewed_at_utc"]) > at
            or set(freeze.get("checks", {})) != set(FREEZE_CHECKS)):
        raise ValueError("baseline freeze requires a dated human correctness and estimator-sensitivity review")
    evidence = {}
    for name in FREEZE_CHECKS:
        check = freeze["checks"][name]
        calibrate._keys(check, {"judgment", "rationale", "evidence"})
        if (check.get("judgment") != "accepted" or not isinstance(check.get("rationale"), str)
                or not check["rationale"].strip() or not check.get("evidence")):
            raise ValueError("baseline freeze checks need accepted judgments, rationale and local evidence")
        for row in check["evidence"]:
            calibrate._keys(row, {"source", "sha256"})
            path = study._resolve(row["source"])
            if study.cache.file_hash(path) != row["sha256"]:
                raise ValueError("baseline review evidence changed")
            evidence[str(path)] = row["sha256"]
    optional = freeze.get("adaptive_allocation", {})
    calibrate._keys(optional, {"decision", "rationale"})
    if optional.get("decision") != "deferred" or not isinstance(optional.get("rationale"), str) or not optional["rationale"].strip():
        raise ValueError("record an explicit adaptive-allocation deferral; no adaptive release policy is implemented")
    return evidence


def _baselines(freeze, development_plan, at):
    evidence = _review(freeze, at)
    directory = study._resolve(freeze["replay_directory"])
    with (directory / ".replay.lock").open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan = study._json(directory / "plan.json")
        if plan["binding"]["mode"] != "development":
            raise ValueError("engineering replay cannot justify the candidate baseline freeze")
        replay._validate(plan, reviews_dir=study._resolve(development_plan["binding"]["reviews_source"]))
        spec = study._json(study.SPEC)
        expected_labs = [row["env_id"] for row in development_plan["binding"]["environments"]]
        if ([row["fingerprint"]["env"] for row in plan["binding"]["environments"]] != expected_labs
                or plan["binding"]["policies"] != spec["baselines"]["required"] + spec["baselines"]["sanity_controls"]
                or not set(plan["binding"]["policies"]) <= RELEASE_BASELINES.keys()):
            raise ValueError("baseline replay membership or policies differ from the reviewed development selection")
        summary = replay.summarize(plan, directory)
        if (any(row["status"] in ("unattempted", "interrupted") for row in summary["slots"])
                or study.canonical_hash(summary) != freeze["replay_summary_sha256"]):
            raise ValueError("complete the full development replay and review its current summary before freezing")
        for path in [directory / "plan.json", *sorted((directory / "attempts").glob("*.json"))]:
            if path.name.endswith(".finish.json") and metadata.timestamp(study._json(path)["finished_at_utc"]) > metadata.timestamp(freeze["reviewer"]["reviewed_at_utc"]):
                raise ValueError("human baseline review predates the recorded replay outcomes")
            evidence[str(path)] = study.cache.file_hash(path)
        return {"policies": plan["binding"]["policies"], "estimator_id": ESTIMATOR_ID,
                "development_majority_label": plan["binding"]["majority_label"],
                "replay_plan_id": plan["plan_id"], "summary_sha256": study.canonical_hash(summary),
                "adaptive_allocation": freeze["adaptive_allocation"], "review": freeze,
                "evidence": evidence}


def generation_rules(models, preflight):
    if not isinstance(models, dict) or set(models) != {"triage", "synthesis", "repair"}:
        raise ValueError("declare resolved model configurations for triage, synthesis and repair")
    for model in models.values():
        provider.verify(preflight, model, live=False)
    models = {name: study._model(model) for name, model in models.items()}
    spec = study._json(study.SPEC)
    return {"rules_id": "fresh_generation_rules_v1", "models": models,
            "generation_attempts_per_candidate": spec["fresh_cohort"]["generation_attempts_per_candidate"],
            "max_repairs_per_candidate": spec["fresh_cohort"]["max_repairs_per_candidate"],
            "seed_partitions": spec["seed_partitions"], "eligibility": spec["fresh_cohort"]["eligibility"],
            "effect_margin": spec["fresh_cohort"]["effect_margin"], "membership": spec["fresh_cohort"]["membership"],
            "max_concurrent_model_sessions": spec["execution_limits"]["max_concurrent_model_sessions"],
            "excerpt_chars": 24000, "timeout_seconds": 900,
            "system_prompt": "You are a precise research engineer.", "triage_schema": triage.SCHEMA,
            "content": "Exact locked version HTML without redirects; abstract fallback only on HTTP 404.",
            "validation_seeds": [0, 1], "sweep_workers": 2, "run_timeout_seconds": 5, "max_recorded_sweep_cpu_seconds": 1200,
            "prompts": {"triage": triage.PROMPT, "synthesis": synth.SYNTH, "repair": synth.REPAIR},
            "admission_code_sha256": {name: study.cache.file_hash(study.ROOT / "benchtop" / name)
                                      for name in ("admission.py", "validate.py", "sweep.py", "tasks.py", "stats.py")},
            "scope": "Frozen rules and prompts for the bounded fresh driver; legacy build/synthesize is not authorized by this record."}


def _development(*, ledger_context=None):
    frozen = study_calibration.load_closed(ledger_context=ledger_context)
    selection = frozen["selection"]
    if (selection["status"] != "selected" or selection["closed"] is not True
            or selection["chosen_budget_fraction"] not in study._json(study.SPEC)["budget"]["fractions"]
            or any(group["missing"] for lab in selection["labs"] for group in lab["budgets"])):
        raise ValueError("candidate lock requires a complete, selected canonical development collection")
    plan = study._json(primary.canonical_root() / "ledger/plans" / f"{frozen['plan_sha256']}.json")
    return frozen, plan


def lock_candidates(archive_directory, baseline_review, models):
    _development()
    root = primary.canonical_root()
    at = metadata.now()
    with primary._guard(root.parent / ".initialization.lock"):
        if (root / "candidate_lock.json").exists():
            raise FileExistsError("candidate lock already exists; retain its original date and candidate order")
        development, plan = _development()
        raw_review = Path(baseline_review).read_bytes()
        freeze = json.loads(raw_review)
        baseline = _baselines(freeze, plan, at)
        archive_directory = Path(archive_directory).resolve()
        archive = metadata.load(archive_directory, at=at)
        configuration = study._json(archive_directory / "configuration.json")
        if configuration["history"] != metadata.history():
            raise ValueError("project generation/review history changed before candidate lock; refresh the exclusions archive")
        rules = generation_rules(models, study._json(root / "provider.json"))
        binding = {"phase": "candidate", "protocol_hash": study.canonical_hash(study._json(study.SPEC)),
            "development_snapshot_sha256": development["snapshot_sha256"],
            "global_budget_fraction": development["selection"]["chosen_budget_fraction"],
            "source_sha256": study._sources(), "runtime": json.loads(json.dumps(study.cache.runtime_identity())),
            "metadata_archive_sha256": archive["archive_sha256"], "selection_date": archive["selection_date"],
            "candidates": archive["candidates"], "baselines": baseline, "generation": rules}
        record = study._seal({"schema_version": 1, "id": "candidate_lock", "study_lock": True,
            "binding": binding, "bindings_hash": study.canonical_hash(binding),
            "plan_sha256": study.canonical_hash({"phase": "candidate", "binding": binding}),
            "protocol_hash": binding["protocol_hash"], "canonical_root": str(root.resolve()),
            "baseline_review_source": study._ref(baseline_review), "baseline_review_sha256": hashlib.sha256(raw_review).hexdigest(),
            "recorded_at_utc": at.isoformat(), "generation_ready": True, "model_sessions_started": 0}, "record_sha256")
        destination = root / "candidate_metadata"
        with primary._guard(archive_directory / ".archive.lock"):
            if metadata.load(archive_directory, at=at) != archive:
                raise ValueError("metadata archive changed during candidate lock")
            destination.mkdir(mode=0o700, exist_ok=False)
            (destination / "requests").mkdir(mode=0o700)
            for name in ("configuration.json", "archive.json", *archive["request_evidence"]):
                metadata._bytes(destination / name, (archive_directory / name).read_bytes())
            if metadata.load(destination, at=at) != archive:
                raise ValueError("copied metadata archive differs from its bound response")
        metadata._bytes(root / "candidate_baseline_review.json", raw_review)
        study._publish(root / "candidate_lock.json", record)
        primary._append_binding(record)
        return record


def verify(*, ledger_context=None):
    root = primary.canonical_root()
    record = study._json(root / "candidate_lock.json")
    study._verify(record, "record_sha256")
    binding = record["binding"]
    if (record.get("schema_version") != 1 or record.get("id") != "candidate_lock" or record.get("study_lock") is not True
            or record.get("generation_ready") is not True or record.get("model_sessions_started") != 0 or binding.get("phase") != "candidate"
            or record["canonical_root"] != str(root.resolve()) or binding["source_sha256"] != study._sources()
            or binding["runtime"] != json.loads(json.dumps(study.cache.runtime_identity()))
            or record["bindings_hash"] != study.canonical_hash(binding)
            or record["plan_sha256"] != study.canonical_hash({"phase": "candidate", "binding": binding})
            or record["protocol_hash"] != study.canonical_hash(study._json(study.SPEC))):
        raise ValueError("candidate lock configuration changed")
    development, plan = _development(ledger_context=ledger_context)
    if study.cache.file_hash(root / "candidate_baseline_review.json") != record["baseline_review_sha256"]:
        raise ValueError("archived baseline review changed")
    at = metadata.timestamp(record["recorded_at_utc"])
    archive = metadata.load(root / "candidate_metadata", at=at)
    if (development["snapshot_sha256"] != binding["development_snapshot_sha256"]
            or development["selection"]["chosen_budget_fraction"] != binding["global_budget_fraction"]
            or archive["archive_sha256"] != binding["metadata_archive_sha256"]
            or archive["selection_date"] != binding["selection_date"] or archive["candidates"] != binding["candidates"]
            or _baselines(study._json(root / "candidate_baseline_review.json"), plan, at) != binding["baselines"]
            or generation_rules(binding["generation"]["models"], study._json(root / "provider.json")) != binding["generation"]):
        raise ValueError("candidate lock evidence or rules changed")
    public = study._json(primary.BINDINGS)
    if public.get("protocol_hash") != record["protocol_hash"] or not any(
        row.get("id") == "candidate_lock" and row.get("record_sha256") == record["record_sha256"] for row in public.get("records", [])):
        raise ValueError("candidate lock is absent from public binding history")
    return record
