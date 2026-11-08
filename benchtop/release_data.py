import copy
import json
import math
import re
from types import SimpleNamespace

from benchtop import cache, calibrate, confirmation_analysis, study
from benchtop.agents.baselines import play_release
from benchtop.env import Env, Table, read_jsonl
from benchtop.episode import Episode
from benchtop.space import config_key
from benchtop.stats import LABELS
from benchtop.tasks import build_task_set, grade, norm_label

FORMAT = "public_confirmation_projection_v1"
UNSAFE = re.compile(r"/e/[0-9a-f]{12}/[A-Za-z0-9_-]{43}/mcp|(?:/Users/|/home/|/private/var/|/var/folders/|/tmp/|file://|\.state/)|\bBearer\s+[A-Za-z0-9._-]{12,}|\bsk-[A-Za-z0-9_-]{16,}|https?://[^\s/]+@")


def safe_text(value):
    text = value if isinstance(value, str) else json.dumps(value, allow_nan=False)
    if UNSAFE.search(text):
        raise ValueError("publication input contains a capability, credential or private path; supply publishable evidence before exporting")
    return value


def compare(expected, actual, *, path="result"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise ValueError("reproduction schema differs: " + path)
        for key in expected:
            if key not in ("artifact_sha256", "task_sha256"):
                compare(expected[key], actual[key], path=path + "/" + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError("reproduction length differs: " + path)
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            compare(left, right, path=path + "/" + str(index))
    elif type(expected) in (float, int):
        if type(actual) not in (float, int) or not math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("reproduction number differs: " + path)
    elif expected != actual:
        raise ValueError("reproduction value differs: " + path)


def _detail(value):
    answer = value.get("answer")
    if isinstance(answer, dict):
        answer = answer.get("answer")
    answer = norm_label(answer) if isinstance(answer, str) else None
    confidence = value.get("confidence")
    return {"attempt_id": value["attempt_id"], "answer": answer if answer in LABELS else None,
        "confidence": confidence if type(confidence) in (int, float) and math.isfinite(confidence) else None,
        "observed_trace": [{key: event[key] for key in ("index", "tool", "error", "response_keys")} for event in value["observed_trace"]]}


def project(collection, classical, plan, assessment, baselines, calibration):
    public = {key: copy.deepcopy(collection[key]) for key in (
        "schema_version", "exporter_id", "closed", "protocol_hash", "bindings_hash", "groups", "members", "pilot_decisions",
        "global_budget_fraction", "candidate_funnel", "pilot_costs", "confirmation_costs", "generation_and_repair_usage", "accounting", "halted")}
    public["slots"] = [{key: copy.deepcopy(slot[key]) for key in (
        "group_id", "slot_id", "env_id", "task", "replicate_index", "budget_fraction", "attempt_ids", "arm", "attempts")} | {
        "details": [_detail(detail) for detail in slot["details"]]} for slot in collection["slots"]]
    public = study._seal(public, "snapshot_sha256")
    replay = {key: copy.deepcopy(classical[key]) for key in (
        "replay_id", "replay_plan_sha256", "protocol_hash", "status", "groups", "policy_seeds", "budget_fraction",
        "model_sessions_started", "actual_environment_executions", "actual_environment_cpu_seconds", "costs", "interpretation")}
    replay["collection_sha256"] = public["snapshot_sha256"]
    replay = study._seal(replay, "summary_sha256")
    original = confirmation_analysis.analyse(collection, classical)
    derived = confirmation_analysis.analyse(public, replay)
    left, right = copy.deepcopy(original), copy.deepcopy(derived)
    for value in (left, right):
        for key in ("analysis_sha256", "collection_sha256"):
            value.pop(key)
        value["classical"].pop("summary_sha256")
    compare(left, right)
    members = []
    for member, report in zip(plan["binding"]["environments"], assessment["members"], strict=True):
        if member["env_id"] != report["env_id"] or re.fullmatch(r"[0-9]{4}\.[0-9]{4,5}v[1-9][0-9]*", member["env_id"]) is None:
            raise ValueError("publication requires ordered versioned cohort identities")
        members.append({key: copy.deepcopy(member[key]) for key in (
            "env_id", "version", "lab_hash", "execution_id", "task_version", "task_hash", "grader_id", "charging_hash",
            "qualification_label", "min_effect", "visible_order", "confirmation_visible_order", "seed_splits")} | {
            "status": report["status"], "gradable": report["gradable"], "confirmation_task_artifact": report["task_artifact"],
            "source_qualification_files": member["files"], "source_member_assessment_sha256": report["record_sha256"]})
    projection = study._seal({"schema_version": 1, "format": FORMAT, "collection": public, "classical": replay, "members": members,
        "model_configuration": plan["binding"]["model_configuration"], "max_calls": plan["binding"]["limits"]["max_experiment_tool_calls"],
        "calibration": copy.deepcopy(calibration),
        "baselines": {key: copy.deepcopy(baselines[key]) for key in ("policies", "estimator_id", "development_majority_label", "adaptive_allocation")},
        "origin": {"private_collection_sha256": collection["snapshot_sha256"], "private_classical_summary_sha256": classical["summary_sha256"],
                   "private_analysis_sha256": original["analysis_sha256"], "confirmation_plan_sha256": plan["plan_sha256"]},
        "scope": "New sanitized projections of verified canonical records. Projection hashes differ from original records; no original evidence is rebound or rewritten. Raw provider transcripts remain private."}, "projection_sha256")
    safe_text(projection)
    return projection, derived


def _calibration(projection):
    evidence = projection["calibration"]
    rows = [evidence["development"], *evidence["pilots"]]
    selections = []
    for row in rows:
        study._verify(row["selection"], "selection_sha256")
        expected = copy.deepcopy(row["selection"])
        actual = calibrate.select_budget(row["input"])
        for value in (expected, actual):
            value.pop("selection_sha256")
            value["binding"].pop("runtime")
        compare(expected, actual, path="budget_selection")
        selections.append(actual)
    global_budget = projection["collection"]["global_budget_fraction"]
    if selections[0]["status"] != "selected" or selections[0]["chosen_budget_fraction"] != global_budget:
        raise ValueError("published global budget differs from closed development selection")
    decisions = projection["collection"]["pilot_decisions"]
    if len(decisions) != len(evidence["pilots"]):
        raise ValueError("published per-lab calibration membership differs")
    for row, selection, decision in zip(evidence["pilots"], selections[1:], decisions, strict=True):
        if (row["env_id"] != decision["env_id"] or selection["global_budget_fraction"] != global_budget
                or row["selection"]["selection_sha256"] != decision["selection_sha256"]
                or any(selection[key] != decision[key] for key in ("status", "chosen_budget_fraction", "fallback_reason"))):
            raise ValueError("published per-lab budget differs from its closed pilot selection")
    return len(rows)


def manifest_projection(manifest):
    keys = ("id", "source", "arxiv_id", "arxiv_version", "paper_title", "published", "seeds", "visible_seeds", "seed_splits", "grader_id", "data_role")
    return safe_text({key: manifest[key] for key in keys if key in manifest})


def rows_projection(folder, execution, space):
    rows = []
    for row in read_jsonl(folder / "results.jsonl"):
        result = {key: copy.deepcopy(row[key]) for key in ("ok", "secs", "metrics", "timing_unreported") if key in row}
        if not row["ok"]:
            result["error"] = "original_result_failed"
        result["source_row_sha256"] = row["row_sha256"]
        rows.append(cache.seal_result(execution, space, row["config"], row["seed"], result))
    safe_text(rows)
    return rows


def _confirmation(folder, env, member):
    wrapped = study._json(folder / "cache.json")
    study._verify(wrapped, "private_cache_sha256")
    execution = wrapped["execution"]
    if (wrapped["schema_version"] != "confirmation_private_v1" or wrapped["data_role"] != "confirmation_only"
            or execution["id"] != cache.digest(execution["inputs"]) or execution["id"] != env.execution_id
            or execution["inputs"]["lab_sha256"] != cache.file_hash(env.dir / "lab.py")
            or execution["inputs"]["space_sha256"] != cache.file_hash(env.dir / "space.json")
            or execution["inputs"]["seed_partition"] != cache.seed_partition(study._json(folder / "manifest.json"))):
        raise ValueError("published confirmation cache differs from its recorded execution identity")
    rows, seen = read_jsonl(folder / "results.jsonl"), set()
    seeds = member["seed_splits"]["confirmation"]
    for row in rows:
        key = config_key(env.space.normalize(row["config"])), row["seed"]
        if key in seen or row["seed"] not in seeds or cache.seal_result(execution, env.space, row["config"], row["seed"], row) != row:
            raise ValueError("published confirmation rows are duplicated, cross-partition or corrupt")
        seen.add(key)
    table = SimpleNamespace(space=env.space, table=Table(env.space, rows), execution_id=execution["id"], heldout_seeds=seeds)
    artifact = build_task_set(table, seeds=seeds)
    compare(member["confirmation_task_artifact"], artifact, path="independent_truth")
    claim = next((task for task in artifact["tasks"] if task["id"] == "claim"), None)
    label = claim["answer"] if claim else None
    status = "incomplete" if artifact["status"] != "complete" else "ambiguous" if label is None else "stable" if label == member["qualification_label"] else "unstable"
    if status != member["status"] or member["gradable"] != (status == "stable"):
        raise ValueError("published stability differs from independently recomputed confirmation truth")
    return claim


def reproduce(projection, directory, *, replay_classical=True):
    study._verify(projection, "projection_sha256")
    if projection["format"] != FORMAT or projection["classical"]["status"] != "complete":
        raise ValueError("publication needs the supported projection and completed classical collection")
    selections = _calibration(projection)
    report = confirmation_analysis.analyse(projection["collection"], projection["classical"])
    model_grades, policies, preserved = 0, 0, 0
    for member in projection["members"]:
        folder = directory / member["env_id"]
        env = Env.load(folder / "qualification", replay=True)
        if (env.id != member["env_id"] or env.execution_id != member["execution_id"] or env.task_version != member["task_version"]
                or study.canonical_hash(env.task("claim")) != member["task_hash"] or env.charging_schedule()[1] != member["charging_hash"]
                or env.visible_seeds != member["seed_splits"]["visible"] or env.qualification_seeds != member["seed_splits"]["qualification"]):
            raise ValueError("published qualification question, seed partition or frozen prices changed")
        if any(seed not in member["seed_splits"]["visible"] + member["seed_splits"]["qualification"] for cells in env.table.cells.values() for seed in cells):
            raise ValueError("published qualification contains confirmation data")
        compare(study._json(folder / "qualification/tasks.json"), build_task_set(env, seeds=env.qualification_seeds), path="qualification_truth")
        claim = _confirmation(folder / "confirmation", env, member)
        for slot in (slot for slot in projection["collection"]["slots"] if slot["env_id"] == env.id and slot["attempts"]):
            final = slot["attempts"][-1]
            if final["state"] == "completed":
                if not member["gradable"] or grade(None, claim, slot["details"][-1]["answer"])["score"] != final["reward"]:
                    raise ValueError("published model reward differs from independent confirmation grading")
                model_grades += 1
        if not member["gradable"]:
            continue
        context = study._seal({"binding": {"role": "independent_confirmation", "env_id": env.id, "question_task_hash": member["task_hash"]}, "task": claim}, "context_sha256")
        for group in (group for group in projection["classical"]["groups"] if group["env_id"] == env.id):
            for row in group["outcomes"]:
                if row["state"] != "completed":
                    preserved += 1
                    continue
                if not replay_classical:
                    continue
                ep = Episode(env, "claim", seed=0, budget_frac=projection["classical"]["budget_fraction"],
                    visible_order=member["confirmation_visible_order"], max_calls=projection["max_calls"], grading_context=context)
                result = play_release(ep, group["policy"], row["policy_seed"], majority_label=projection["baselines"]["development_majority_label"])
                if result["score"] != row["reward"] or not math.isclose(result["spent"], row["costs"]["charged_experiment_cost"], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("published classical replay differs from cached policy reward or charge")
                policies += 1
    return report, {"budget_selections_rechecked": selections, "model_grades_rechecked": model_grades, "cached_policy_trials_replayed": policies,
        "recorded_policy_failures_or_missing_preserved": preserved, "model_sessions_started": 0, "lab_workers_started": 0,
        "scope": "Cached numerical reproduction; no inference, lab-code execution, new admission or authorization. Missing outcomes are preserved, never filled by replay."}
