"""Bounded classical replay with explicit review gates and immutable attempt records."""

import argparse
import fcntl
import hashlib
import itertools
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from benchtop import cache, protocol
from benchtop.admission import inspect_lab
from benchtop.agents.baselines import RELEASE_BASELINES, play_release
from benchtop.env import ROOT, Env, load_catalog
from benchtop.episode import Episode
from benchtop.evaluate import corrected_replay
from benchtop.stats import GRADER_ID

SPEC = protocol.SPEC
REFERENCES = protocol.REFERENCES
REPLAY_ID = "classical_replay_v1"
SOURCE_FILES = ("replay.py", "agents/baselines.py", "agents/estimator.py", "episode.py", "evaluate.py",
                "analysis.py", "admission.py", "validate.py", "cache.py", "env.py", "space.py", "stats.py", "tasks.py")


def _json(path):
    return json.loads(Path(path).read_text())


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _sealed(value, key):
    return {**value, key: _hash({k: v for k, v in value.items() if k != key})}


def _verify(value, key):
    if value.get(key) != _sealed(value, key)[key]:
        raise ValueError(f"invalid {key}")


def _now():
    return datetime.now(UTC).isoformat()


def _output(path):
    path = Path(path).resolve()
    if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / "artifacts"):
        raise ValueError("write replay outputs under artifacts/ or outside the repository")
    return path


def _publish(path, value):
    """Publish a complete JSON file atomically and exclusively; never replace an outcome."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".replay-", delete=False) as out:
        temporary = Path(out.name)
        try:
            out.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _path_ref(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "base": "repository" if path.is_relative_to(ROOT) else "absolute"}


def _resolve(ref):
    if ref["base"] not in ("repository", "absolute"):
        raise ValueError("unsupported source path base")
    path = (ROOT / ref["path"]).resolve() if ref["base"] == "repository" else Path(ref["path"]).resolve()
    if ref["base"] == "repository" and not path.is_relative_to(ROOT):
        raise ValueError("source path escapes repository")
    return path


def _view(path, *, engineering=False):
    source = Env.load(Path(path), replay=True)
    if source.cache_status == "historical":
        return corrected_replay(source)
    if source.grader_id != GRADER_ID:
        raise ValueError("replay requires corrected tasks or a frozen historical snapshot")
    if engineering and (source.paper.get("arxiv_id") or source.manifest.get("seed_splits")):
        raise ValueError("engineering mode supports historical snapshots or controlled two-part fixtures, not fresh paper cohorts")
    return source


def _fingerprint(env):
    env.assert_current()
    if not 1 <= len(env.visible_seeds) <= 4:
        raise ValueError("classical replay v1 supports at most four visible seeds")
    files = {name: cache.file_hash(env.dir / name) for name in (*cache.DATA_FILES, "cache.json") if (env.dir / name).is_file()}
    lab_hash = _hash({name: files[name] for name in ("lab.py", "space.json")})
    orders = list(itertools.permutations(env.visible_seeds))
    index = int(hashlib.sha256(("benchtop_release_v1/order/" + lab_hash).encode()).hexdigest(), 16) % len(orders)
    return {"env": env.id, "files": files, "lab_hash": lab_hash, "visible_order": list(orders[index]),
            "execution_id": env.execution_id, "task_version": env.task_version,
            "task_hash": _hash(env.task("claim")), "charging_id": env.charging_schedule()[1],
            "historical_index_sha256": cache.file_hash(env.dir.parent / "historical_cache.json") if env.cache_status == "historical" else None}


def _majority(labels, order):
    counts = Counter(labels)
    return max(order, key=lambda label: counts[label])


def development_gate(reviews_dir):
    spec, references = _json(SPEC), _json(REFERENCES)
    if references["protocol_sha256"] != cache.file_hash(SPEC):
        raise ValueError("reference selection is bound to another protocol")
    ids = [row["env"] for row in references["cases"]]
    if (len(ids) != spec["development"]["reference_labs"] or ids != sorted(set(ids))
            or not set(ids) <= set(load_catalog(ROOT / "envs")["environments"])):
        raise ValueError("expected the six ordered reference cases")
    rows, review_hashes = [], {}
    for env in ids:
        path = Path(reviews_dir) / f"{env}.json"
        review = None
        if path.is_file():
            review_hashes[env] = cache.file_hash(path)
            try:
                review = _json(path)
            except (OSError, ValueError):
                review = {"schema_version": 0}
        inspected = inspect_lab(ROOT / "envs" / env, review=review)
        rows.append({"env": env, "review_status": inspected["human_review"]["status"],
                     "decision": inspected["decision"], "subject": inspected["review_subject"],
                     "qualified_claim_admitted": inspected["qualified_claim_admitted"],
                     "claim_label": (inspected["qualification"].get("claim") or {}).get("label")})
    eligible = [row for row in rows if row["qualified_claim_admitted"]]
    needed = spec["development"]["claim_labs"]
    blockers = []
    if any(row["review_status"] != "recorded" for row in rows):
        blockers.append("All six source-bound human review records must be completed; drafts do not count.")
    if len(eligible) < needed:
        blockers.append(f"Only {len(eligible)} accepted qualified references; {needed} required. Amend before model budget selection if fewer qualify.")
    selected = eligible[:needed] if not blockers else []
    order = spec["grading"]["primary_labels"]
    majority = _majority([row["claim_label"] for row in selected], order) if selected else None
    return {"status": "blocked" if blockers else "ready", "blockers": blockers, "references": rows,
            "selected_env_ids": [row["env"] for row in selected], "majority_label": majority,
            "review_sha256": review_hashes, "selection_sha256": cache.file_hash(REFERENCES),
            "scope": "Scientific review of historical source snapshots; corrected replay task/pricing versions are bound separately. Not a model development lock.",
            "identity_source": "operator_reported_not_authenticated"}


def _slots(binding, binding_id):
    slots = []
    for env in binding["environments"]:
        for policy in binding["policies"]:
            for budget in binding["budgets"]:
                descriptor = {"binding_id": binding_id, "env": env["fingerprint"]["env"], "task": "claim",
                              "policy": policy, "budget_fraction": budget}
                group_id = _hash(descriptor)
                for index, seed in enumerate(binding["policy_seeds"]):
                    slots.append({**descriptor, "group_id": group_id, "replicate_index": index, "policy_seed": seed,
                                  "slot_id": _hash({"group_id": group_id, "replicate_index": index})})
    return slots


def make_plan(env_paths=None, *, reviews_dir=None, mode="engineering", repeats=2):
    spec = _json(SPEC)
    if mode not in ("engineering", "development"):
        raise ValueError("unknown replay mode")
    gate = None
    if mode == "development":
        if env_paths:
            raise ValueError("development membership is selected from reviews, not explicit paths")
        gate = development_gate(reviews_dir or ROOT / "artifacts/reference-human-reviews")
        if gate["status"] != "ready":
            raise ValueError("development replay blocked: " + " ".join(gate["blockers"]))
        paths = [ROOT / "envs" / env for env in gate["selected_env_ids"]]
        seeds = list(range(spec["baselines"]["replay_seeds"]["first"],
                           spec["baselines"]["replay_seeds"]["first"] + spec["baselines"]["replay_seeds"]["count"]))
        policies = spec["baselines"]["required"] + spec["baselines"]["sanity_controls"]
    else:
        paths = [Path(path) for path in (env_paths or [])]
        if not 1 <= len(paths) <= 2 or type(repeats) is not int or not 1 <= repeats <= 2:
            raise ValueError("engineering mode requires one or two sources and one or two policy seeds")
        seeds = list(range(repeats))
        policies = spec["baselines"]["required"] + ["paper_prior_only", "random_experiment_design"]
    if not set(policies) <= RELEASE_BASELINES.keys():
        raise ValueError("plan requests an unimplemented policy")
    environments = []
    for path in paths:
        env = _view(path, engineering=mode == "engineering")
        if mode == "development" and env.visible_seeds != spec["seed_partitions"]["visible"]:
            raise ValueError("development visible partition differs from protocol")
        environments.append({"source": _path_ref(path), "fingerprint": _fingerprint(env)})
    if len({row["fingerprint"]["env"] for row in environments}) != len(environments):
        raise ValueError("source identities must be unique")
    binding = {"replay_id": REPLAY_ID, "mode": mode, "protocol_file_sha256": cache.file_hash(SPEC), "protocol_hash": _hash(spec),
               "source_sha256": {f"benchtop/{name}": cache.file_hash(ROOT / "benchtop" / name) for name in SOURCE_FILES},
               "runtime": cache.runtime_identity(), "environments": environments, "policies": policies,
               "budgets": spec["baselines"]["budget_fractions"], "policy_seeds": seeds, "attempt_limit": 2,
               "development_gate": gate, "majority_label": gate["majority_label"] if gate else None,
               "visible_order_rule": "Protocol prefix plus canonical lab.py/space.json SHA-256 object hash; lexicographic visible permutations.",
               "scope": "Classical cached replay only; engineering results are excluded from development comparisons. No model sessions or fresh cohort lock."}
    binding = json.loads(json.dumps(binding, allow_nan=False))
    binding_id = _hash(binding)
    return _sealed({"schema_version": 1, "binding": binding, "binding_id": binding_id,
                    "slots": _slots(binding, binding_id)}, "plan_id")


def _validate(plan, *, reviews_dir=None):
    _verify(plan, "plan_id")
    binding = plan["binding"]
    if binding["replay_id"] != REPLAY_ID or plan["binding_id"] != _hash(binding):
        raise ValueError("unsupported or changed replay binding")
    if plan["slots"] != _slots(binding, plan["binding_id"]):
        raise ValueError("planned slots do not match the full bound grid")
    if (binding["protocol_file_sha256"] != cache.file_hash(SPEC) or binding["protocol_hash"] != _hash(_json(SPEC))
            or _hash(binding["runtime"]) != _hash(cache.runtime_identity())):
        raise ValueError("protocol or analysis runtime changed; preserve this plan and create a new one")
    for name, sha in binding["source_sha256"].items():
        if name not in {f"benchtop/{file}" for file in SOURCE_FILES} or cache.file_hash(ROOT / name) != sha:
            raise ValueError("replay implementation changed; do not resume under another version")
    if set(binding["source_sha256"]) != {f"benchtop/{name}" for name in SOURCE_FILES}:
        raise ValueError("missing implementation bindings")
    spec = _json(SPEC)
    if binding["budgets"] != spec["baselines"]["budget_fractions"] or binding["attempt_limit"] != 2:
        raise ValueError("replay budget/retry rules changed")
    if binding["mode"] == "development":
        gate = development_gate(reviews_dir or ROOT / "artifacts/reference-human-reviews")
        if gate != binding["development_gate"] or gate["status"] != "ready":
            raise ValueError("development reviews or source admission changed")
        expected = make_plan(reviews_dir=reviews_dir, mode="development")
        if plan["plan_id"] != expected["plan_id"]:
            raise ValueError("development grid differs from the reviewed protocol plan")
    elif binding["mode"] == "engineering":
        expected = make_plan([_resolve(row["source"]) for row in binding["environments"]],
                             repeats=len(binding["policy_seeds"]))
        if plan["plan_id"] != expected["plan_id"]:
            raise ValueError("engineering grid differs from its bounded plan")
    else:
        raise ValueError("unknown replay mode")
    views = {}
    for row in binding["environments"]:
        env = _view(_resolve(row["source"]), engineering=binding["mode"] == "engineering")
        if _fingerprint(env) != row["fingerprint"]:
            raise ValueError("environment/task/pricing inputs changed")
        views[env.id] = env
    return views


def _attempt_id(slot, index):
    return _hash({"slot_id": slot["slot_id"], "attempt_index": index})


def _history(directory, plan, slot):
    history = []
    for index in range(plan["binding"]["attempt_limit"]):
        aid = _attempt_id(slot, index)
        start_path, finish_path = (directory / "attempts" / f"{aid}.{kind}.json" for kind in ("start", "finish"))
        if not start_path.exists():
            if finish_path.exists():
                raise ValueError("orphan replay finish record")
            continue
        start = _json(start_path)
        if len(history) != index:
            raise ValueError("attempt sequence has a gap")
        _verify(start, "record_sha256")
        for key, expected in (("plan_id", plan["plan_id"]), ("slot_id", slot["slot_id"]), ("attempt_id", aid), ("attempt_index", index)):
            if start[key] != expected:
                raise ValueError("attempt identity mismatch")
        finish = _json(finish_path) if finish_path.exists() else None
        if finish is not None:
            _verify(finish, "record_sha256")
            for key in ("plan_id", "slot_id", "attempt_id", "attempt_index"):
                if finish[key] != start[key]:
                    raise ValueError("finish identity mismatch")
            if finish["status"] not in ("completed", "policy_failure", "technical_failure"):
                raise ValueError("unknown replay outcome")
            if (finish["status"] == "technical_failure" and finish["reward"] is not None
                    or finish["status"] != "technical_failure" and finish["reward"] not in (0.0, 1.0)):
                raise ValueError("invalid replay reward state")
        history.append((start, finish))
    if len(history) > 1 and (history[0][1] is None or history[0][1]["status"] != "technical_failure"):
        raise ValueError("retry replaced a terminal or unresolved outcome")
    return history


def summarize(plan, directory):
    """Read evidence without executing policies or rewriting missing/incorrect outcomes."""
    directory = Path(directory)
    _verify(plan, "plan_id")
    expected = {f"{_attempt_id(slot, index)}.{kind}.json" for slot in plan["slots"]
                for index in range(plan["binding"]["attempt_limit"]) for kind in ("start", "finish")}
    if any(path.name not in expected for path in (directory / "attempts").glob("*.json")):
        raise ValueError("attempt directory contains records outside this plan")
    groups = defaultdict(lambda: {"planned": 0, "completed": 0, "policy_failures": 0, "technical_missing": 0,
                                  "unattempted": 0, "correct": 0, "attempts": 0, "spent": 0.0, "unknown_spend_attempts": 0})
    slots = []
    for slot in plan["slots"]:
        group = groups[slot["group_id"]]
        group.update({key: slot[key] for key in ("env", "policy", "budget_fraction")})
        group["planned"] += 1
        history = _history(directory, plan, slot)
        group["attempts"] += len(history)
        for _, record in history:
            if record is not None:
                if record.get("spent") is None:
                    group["unknown_spend_attempts"] += 1
                else:
                    group["spent"] += record["spent"]
            else:
                group["unknown_spend_attempts"] += 1
        if not history:
            state, reward = "unattempted", None
            group["unattempted"] += 1
        else:
            record = history[-1][1]
            state, reward = (record["status"], record["reward"]) if record is not None else ("interrupted", None)
            if state in ("technical_failure", "interrupted"):
                group["technical_missing"] += 1
            else:
                group["completed"] += 1
                group["policy_failures"] += state == "policy_failure"
                group["correct"] += reward == 1.0
        slots.append({"slot_id": slot["slot_id"], "group_id": slot["group_id"], "status": state, "reward": reward})
    for group in groups.values():
        group["complete_case_accuracy"] = group["correct"] / group["completed"] if group["completed"] else None
        group["accuracy_bounds"] = [group["correct"] / group["planned"],
                                    (group["correct"] + group["technical_missing"] + group["unattempted"]) / group["planned"]]
    return {"replay_id": REPLAY_ID, "plan_id": plan["plan_id"], "mode": plan["binding"]["mode"],
            "groups": list(groups.values()), "slots": slots, "model_sessions": 0,
            "limits": ["Policy seeds reuse cached visible data; repetitions are not independent new labs or model sessions.",
                       "Engineering results are excluded from the development comparison; no fresh-cohort or model lock is created.",
                       "Missing bounds include technical failures and unattempted slots; completed wrong answers remain zero."]}


def _finish_record(plan, slot, index, **fields):
    return _sealed({"plan_id": plan["plan_id"], "slot_id": slot["slot_id"], "attempt_id": _attempt_id(slot, index),
                    "attempt_index": index, "finished_at_utc": _now(), **fields}, "record_sha256")


def _run_one(plan, slot, index, env, guard):
    trace, ep, outcome, error = [], None, None, None
    try:
        fingerprint = next(row["fingerprint"] for row in plan["binding"]["environments"] if row["fingerprint"]["env"] == env.id)
        ep = Episode(env, "claim", seed=0, budget_frac=slot["budget_fraction"], visible_order=fingerprint["visible_order"])
        outcome = play_release(ep, slot["policy"], slot["policy_seed"],
                               majority_label=plan["binding"]["majority_label"], tool_trace=trace)
        guard()
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    technical = (error is not None and error["type"] == "CacheError") or any(
        "exception" in item or item.get("response", {}).get("code") in ("stale_cache", "execution_failed", "service_unavailable")
        or item.get("response", {}).get("errors") for item in trace)
    technical = technical or (outcome is not None and outcome["stop_reason"] == "partial_reply")
    policy_fault = error is not None or any("error" in item.get("response", {}) for item in trace)
    status = "technical_failure" if technical else "policy_failure" if policy_fault else "completed"
    reward = None if technical else 0.0 if policy_fault else outcome["score"]
    if outcome is not None:
        outcome = {key: value for key, value in outcome.items() if key != "tool_trace"}
    return _finish_record(plan, slot, index, status=status, reward=reward, error=error,
                          spent=ep.spent if ep else 0.0, outcome=outcome, tool_trace=trace,
                          evaluator_log=ep.log if ep else [],
                          diagnostic_result=ep.result if ep else None)


def run_plan(plan, directory, *, reviews_dir=None, retry_technical=False, max_attempts=None):
    directory = _output(directory)
    if max_attempts is not None and (type(max_attempts) is not int or max_attempts < 0):
        raise ValueError("max_attempts must be nonnegative")
    views = _validate(plan, reviews_dir=reviews_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".replay.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("replay directory is already running") from error
        snapshot = directory / "plan.json"
        if snapshot.exists():
            if _json(snapshot) != plan:
                raise ValueError("replay directory belongs to another plan")
        else:
            if any(path.name != ".replay.lock" for path in directory.iterdir()):
                raise ValueError("refusing an unrelated nonempty replay directory")
            _publish(snapshot, plan)
        tracked = {ROOT / name: cache.file_signature(ROOT / name) for name in plan["binding"]["source_sha256"]}

        def guard():
            if any(cache.file_signature(path) != sig for path, sig in tracked.items()):
                raise cache.CacheError("replay code changed during execution")
            for view in views.values():
                view.assert_current()

        invoked = 0
        for slot in plan["slots"]:
            guard()
            history = _history(directory, plan, slot)
            if history and history[-1][1] is None:
                start, _ = history[-1]
                record = _finish_record(plan, slot, start["attempt_index"], status="technical_failure", reward=None,
                                        spent=None, error={"type": "Interrupted", "message": "Prior invocation started without a published finish; actual spend unknown."},
                                        outcome=None, tool_trace=[], evaluator_log=[], diagnostic_result=None)
                _publish(directory / "attempts" / f"{record['attempt_id']}.finish.json", record)
                history = _history(directory, plan, slot)
            if history and (history[-1][1]["status"] != "technical_failure" or not retry_technical or len(history) >= 2):
                continue
            if max_attempts is not None and invoked >= max_attempts:
                break
            index = len(history)
            start = _sealed({"plan_id": plan["plan_id"], "slot_id": slot["slot_id"],
                             "attempt_id": _attempt_id(slot, index), "attempt_index": index,
                             "started_at_utc": _now()}, "record_sha256")
            _publish(directory / "attempts" / f"{start['attempt_id']}.start.json", start)
            record = _run_one(plan, slot, index, views[slot["env"]], guard)
            _publish(directory / "attempts" / f"{record['attempt_id']}.finish.json", record)
            invoked += 1
        result = summarize(plan, directory)
        result["attempts_invoked_this_run"] = invoked
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gate = commands.add_parser("gate")
    gate.add_argument("--reviews-dir", default="artifacts/reference-human-reviews")
    gate.add_argument("--output")
    plan = commands.add_parser("plan")
    plan.add_argument("--mode", choices=("engineering", "development"), default="engineering")
    plan.add_argument("--env", action="append")
    plan.add_argument("--reviews-dir")
    plan.add_argument("--repeats", type=int, default=2)
    plan.add_argument("--output", required=True)
    run = commands.add_parser("run")
    run.add_argument("plan")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--reviews-dir")
    run.add_argument("--retry-technical", action="store_true")
    run.add_argument("--max-attempts", type=int)
    summary = commands.add_parser("summary")
    summary.add_argument("directory")
    summary.add_argument("--output", help="publish a new immutable full-summary snapshot")
    args = parser.parse_args()
    if args.command == "gate":
        value = development_gate(args.reviews_dir)
        if args.output:
            _publish(_output(args.output), value)
        print(json.dumps({"status": value["status"], "blockers": value["blockers"], "output": args.output}, indent=2))
    elif args.command == "plan":
        value = make_plan(args.env, reviews_dir=args.reviews_dir, mode=args.mode, repeats=args.repeats)
        _publish(_output(args.output), value)
        print(json.dumps({"plan_id": value["plan_id"], "mode": value["binding"]["mode"], "slots": len(value["slots"]), "output": args.output}, indent=2))
    else:
        if args.command == "run":
            value = run_plan(_json(args.plan), args.output_dir, reviews_dir=args.reviews_dir,
                             retry_technical=args.retry_technical, max_attempts=args.max_attempts)
        else:
            directory = Path(args.directory)
            plan = _json(directory / "plan.json")
            _verify(plan, "plan_id")
            value = summarize(plan, directory)
            if args.output:
                _publish(_output(args.output), value)
        print(json.dumps({"mode": value["mode"], "planned_slots": len(value["slots"]),
                          "status_counts": dict(Counter(row["status"] for row in value["slots"])),
                          "attempts_invoked_this_run": value.get("attempts_invoked_this_run", 0), "model_sessions": 0}, indent=2))


if __name__ == "__main__":
    main()
