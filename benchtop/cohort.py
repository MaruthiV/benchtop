import hashlib
import itertools
import json
import math
from datetime import UTC, datetime

from benchtop import admission, candidates, primary, study
from benchtop import fresh_generation as fresh
from benchtop import generation_journal as journal
from benchtop import session_ledger as ledger
from benchtop.candidate_metadata import timestamp
from benchtop.env import Env, read_jsonl, seed_groups
from benchtop.llm import fenced
from benchtop.space import config_key


def _record(path):
    value = study._json(path)
    study._verify(value, "record_sha256")
    return value


def _expect(condition, message):
    if not condition:
        raise ValueError(message)


def _commands(root, lock, paper, calls):
    rules = lock["binding"]["generation"]
    _expect("triage" in calls, "processed candidate has no recorded triage call")
    for entry in calls.values():
        _expect(entry["finish"] is not None and entry["finish"]["result"]["state"] == "completed",
                "resolve every generation call before freezing candidate dispositions")
    triage = calls["triage"]
    prompt = rules["prompts"]["triage"].format(title=paper["title"], categories=", ".join(paper["categories"]), abstract=paper["abstract"])
    system = rules["system_prompt"] + "\nReturn only one JSON object matching this schema:\n" + json.dumps(rules["triage_schema"], sort_keys=True)
    _expect(triage["start"]["prompt"] == prompt and triage["start"]["system"] == system, "triage prompt differs from the candidate lock")
    try:
        card = fresh._card(triage["finish"]["result"]["text"])
    except (ValueError, TypeError):
        _expect(set(calls) == {"triage"}, "invalid triage was followed by unplanned model calls")
        return "generation_failed", None
    from benchtop.triage import selected

    if not selected({**card, "ok": True}):
        _expect(set(calls) == {"triage"}, "triage rejection was followed by synthesis")
        return "rejected", None
    folder = fresh._folder(root, paper)
    excerpt = fresh.saved_content(folder, paper)
    from benchtop.synth import card_text

    task = rules["prompts"]["synthesis"].format(title=paper["title"], arxiv_id=paper["id"], abstract=paper["abstract"], card=card_text(card), excerpt=excerpt)
    prompt, consumed = task, {"triage"}
    for index in range(1 + rules["max_repairs_per_candidate"]):
        stage = "synthesis" if not index else f"repair_{index}"
        _expect(stage in calls, "generation disposition lacks its bounded synthesis/repair history")
        entry = calls[stage]
        _expect(entry["start"]["prompt"] == prompt and entry["start"]["system"] == rules["system_prompt"], "synthesis/repair used a different prompt or hidden feedback")
        consumed.add(stage)
        text = entry["finish"]["result"]["text"]
        work = folder / "drafts" / stage
        _expect((work / "lab.py").read_text() == (fenced(text, "python") or "")
                and (work / "space.json").read_text() == (fenced(text, "json") or ""), "draft differs from its recorded model response")
        proof = _record(work / "validation.json")
        fresh._verify_files(folder, proof["files"])
        _expect(proof["stage"] == stage and timestamp(proof["finished_at_utc"]) >= timestamp(entry["start"]["recorded_at_utc"]), "invalid mechanical validation lineage")
        if not proof["errors"]:
            _expect(set(calls) == consumed, "model calls continued after a valid lab was generated")
            return "human_review", (work, proof)
        prompt = rules["prompts"]["repair"].format(errors="\n".join(proof["errors"]), lab=(work / "lab.py").read_text().strip(),
                                                  space=(work / "space.json").read_text().strip(), task=task[:6000])
    _expect(set(calls) == consumed, "generation exceeded the locked repair allowance")
    return "generation_failed", None


def _human(root, lock, paper, draft, row):
    work, validation = draft
    folder = fresh._folder(root, paper)
    subject = {"kind": "fresh_prequalification_v1", "candidate_lock_sha256": lock["record_sha256"], "version": paper["version"],
               "draft": str(work.relative_to(root)), "lab_sha256": study.cache.file_hash(work / "lab.py"),
               "space_sha256": study.cache.file_hash(work / "space.json"), "validation_sha256": validation["record_sha256"]}
    _expect(_record(folder / "review_subject.json") == study._seal(subject, "record_sha256"), "scientific review subject is stale or belongs to another draft")
    document = study._json(folder / "human_review.json")
    reviewed = admission._review(document, subject)
    _expect(reviewed["status"] == "recorded" and reviewed["decision"] in ("accepted", "rejected"), "a completed source-bound human review is required")
    _expect(document["paper"]["version"] == paper["version"] and document["paper"]["url"] == "https://arxiv.org/abs/" + paper["version"], "scientific review cites a different paper version")
    _expect(timestamp(validation["finished_at_utc"]) <= timestamp(document["reviewer"]["reviewed_at_utc"]) <= timestamp(row["recorded_at_utc"]), "scientific review falls outside its mechanical evidence and candidate disposition")
    if reviewed["decision"] == "rejected":
        _expect(not any((folder / name).exists() for name in ("qualification_permit.json", "qualification_start.json", "qualification.json", "lab/results.jsonl")), "human-rejected lab was qualified")
        return document, None
    margin = document.get("margin", {})
    value = margin.get("value")
    _expect(type(value) in (int, float) and math.isfinite(value) and value > 0
            and isinstance(margin.get("rationale"), str) and margin["rationale"].strip(), "effect margin lacks a justified pre-qualification decision")
    lab = folder / "lab"
    space = study._json(work / "space.json")
    space["metric"]["min_effect"] = value
    _expect((lab / "lab.py").read_bytes() == (work / "lab.py").read_bytes() and study._json(lab / "space.json") == space,
            "qualified lab changed beyond its human-approved effect margin")
    initial = fresh.initial_manifest(paper, lock)
    manifest = study._json(lab / "manifest.json")
    _expect(all(manifest.get(key) == expected for key, expected in initial.items()), "qualified manifest or seed partition differs from the locked paper")
    permit = _record(folder / "qualification_permit.json")
    expected_files = fresh._files(root, [lab / "lab.py", lab / "space.json"])
    expected_files[str((lab / "manifest.json").relative_to(root))] = hashlib.sha256((json.dumps(initial, indent=2) + "\n").encode()).hexdigest()
    _expect(permit["subject"] == subject and permit["review_sha256"] == study.cache.file_hash(folder / "human_review.json")
            and permit["files"] == expected_files, "qualification permit differs from the original review, lab or seed manifest")
    started = _record(folder / "qualification_start.json")
    _expect(started["permit_sha256"] == permit["record_sha256"]
            and timestamp(document["reviewer"]["reviewed_at_utc"]) <= timestamp(permit["recorded_at_utc"]) <= timestamp(started["started_at_utc"]),
            "qualification preceded the frozen human review/margin")
    return document, permit


def _qualified(root, lock, paper, row, permit):
    folder, lab = fresh._folder(root, paper), fresh._folder(root, paper) / "lab"
    rows = read_jsonl(lab / "results.jsonl")
    spec = study._json(study.SPEC)
    allowed = spec["seed_partitions"]["visible"] + spec["seed_partitions"]["qualification"]
    _expect(all(row["seed"] in allowed for row in rows), "confirmation or undeclared seeds appeared before the cohort lock")
    summary = study._json(lab / "manifest.json")["sweep"]
    _expect(summary["cpu_secs"] == round(sum(item.get("secs", 0) for item in rows), 1)
            and summary["confirmation_evaluated"] is False, "sweep accounting differs from its cached rows")
    if not (folder / "qualification.json").exists():
        _expect(row["status"] == "generation_failed" and summary["seed_stage"] == "visible"
                and summary["swept_seed_ids"] == spec["seed_partitions"]["visible"]
                and all(item["seed"] in spec["seed_partitions"]["visible"] for item in rows)
                and (any(not item["ok"] for item in rows) or summary["cpu_secs"] > lock["binding"]["generation"]["max_recorded_sweep_cpu_seconds"]),
                "candidate has no completed qualification proof or verified visible failure")
        return None
    proof = _record(folder / "qualification.json")
    fresh._verify_files(root, proof["files"])
    _expect(proof["permit_sha256"] == permit["record_sha256"] and proof["confirmation_evaluated"] is False, "qualification proof is not bound to its pre-data permit")
    _expect(proof["qualification"] == summary and all(proof[key]["seed_stage"] == key
            and proof[key]["swept_seed_ids"] == spec["seed_partitions"][key]
            and proof[key]["confirmation_evaluated"] is False for key in ("visible", "qualification"))
            and proof["visible"]["cpu_secs"] == round(sum(item.get("secs", 0) for item in rows if item["seed"] in spec["seed_partitions"]["visible"]), 1),
            "qualification sweep stages or CPU acceptance evidence changed")
    report = admission.inspect_lab(lab)
    _expect(study.canonical_hash(proof["screen"]) == study.canonical_hash(report), "qualification screen changed or disagrees with cached source data")
    state, _ = fresh.qualification_decision(proof["visible"], proof["qualification"], report, lock["binding"]["generation"])
    _expect(state == row["status"] == proof["status"], "candidate disposition hides or promotes a qualification result")
    _expect(timestamp(_record(folder / "qualification_start.json")["started_at_utc"]) <= timestamp(proof["finished_at_utc"]) <= timestamp(row["recorded_at_utc"]), "qualification outcome chronology is inconsistent")
    if state == "generation_failed":
        return None
    _expect(row.get("lab") == str(lab.relative_to(root)), "qualification outcome points to a different lab")
    env = Env.load(lab, replay=True)
    cells = [(config_key(env.space.normalize(item["config"])), item["seed"]) for item in rows]
    expected = {(config_key(config), seed) for config in env.space.configs() for seed in allowed}
    _expect(len(cells) == len(set(cells)) and set(cells) == expected and all(item["ok"] for item in rows), "qualified lab lacks a complete paired visible/qualification grid")
    groups = {name: spec["seed_partitions"][name] for name in ("visible", "qualification", "confirmation")}
    _expect(seed_groups(env.manifest) == groups, "qualification seed partitions differ from the protocol")
    _expect(report["qualification"]["seed_ids"] == groups["qualification"], "claim was derived from a different seed partition")
    label = (report["qualification"].get("claim") or {}).get("label")
    _expect(row["qualification_label"] == label, "funnel qualification label differs from the corrected grader")
    if state != "admitted":
        return None
    hashes = {name: study.cache.file_hash(lab / name) for name in (*study.cache.DATA_FILES, "cache.json") if (lab / name).is_file()}
    lab_hash = study.canonical_hash({name: hashes[name] for name in ("lab.py", "space.json")})
    orders = list(itertools.permutations(groups["visible"]))
    index = int(hashlib.sha256(("benchtop_release_v1/order/" + lab_hash).encode()).hexdigest(), 16) % len(orders)
    return {"env_id": env.id, "version": paper["version"], "source": study._ref(lab), "files": hashes, "lab_hash": lab_hash,
            "seed_splits": groups, "visible_order": list(orders[index]), "confirmation_visible_order": list(orders[(index + 1) % len(orders)]),
            "execution_id": env.execution_id, "task_version": env.task_version, "task_hash": study.canonical_hash(env.task("claim")),
            "grader_id": env.grader_id, "charging_hash": env.charging_schedule()[1], "qualification_label": label,
            "min_effect": env.space.min_effect, "review_sha256": permit["review_sha256"], "permit_sha256": permit["record_sha256"],
            "qualification_sha256": proof["record_sha256"]}


def _binding(root, lock, manifest, at):
    accounting = journal.read(root)
    _expect(not accounting["active"] and not accounting["halted"], "cohort lock requires stopped, reconciled generation without a model/tool halt")
    rows = fresh._outcomes(root, lock)
    _expect(all(row["status"] != "unprocessed" for row in rows), "finish the complete ordered candidate funnel before locking the cohort")
    _expect(all(timestamp(row["recorded_at_utc"]) <= timestamp(at) for row in rows), "candidate disposition postdates the cohort lock")
    versions = {paper["version"] for paper in lock["binding"]["candidates"]}
    _expect({path.name for path in (root / "papers").iterdir()} <= versions, "generation contains unplanned candidate directories")
    members, funnel, target = [], [], study._json(study.SPEC)["fresh_cohort"]["target_labs"]
    for paper, row in zip(lock["binding"]["candidates"], rows, strict=True):
        folder = fresh._folder(root, paper)
        calls = {entry["start"]["stage"]: entry for entry in accounting["calls"].values() if entry["start"]["version"] == paper["version"]}
        _expect(all(timestamp(entry["start"]["recorded_at_utc"]) <= timestamp(row["recorded_at_utc"]) for entry in calls.values()), "model call postdates the candidate disposition")
        if len(members) >= target:
            _expect(row["status"] == "not_processed_after_target" and not calls
                    and {path.name for path in folder.iterdir()} == {"outcome.json"}, "candidate was processed or replaced after the cohort target was reached")
            basis = "target_reached"
        else:
            _expect(row["status"] != "not_processed_after_target", "candidate was skipped before eight qualified admissions")
            disposition, draft = _commands(root, lock, paper, calls)
            if draft is None:
                _expect(row["status"] == disposition and not (folder / "review_subject.json").exists()
                        and not (folder / "lab").exists() and not any((folder / name).exists() for name in (
                            "human_review.json", "qualification_permit.json", "qualification_start.json", "qualification.json")),
                        "early generation disposition has hidden review or qualification data")
                basis = "triage_or_mechanical_failure"
            else:
                _, permit = _human(root, lock, paper, draft, row)
                if permit is None:
                    _expect(row["status"] == "rejected", "human rejection was promoted to admission")
                    basis = "human_rejection"
                else:
                    member = _qualified(root, lock, paper, row, permit)
                    if member is not None:
                        members.append(member)
                    basis = "prequalification_review_and_cached_qualification"
        funnel.append({"id": paper["id"], "version": paper["version"], "status": row["status"], "reason": row["reason"],
                       "basis": basis, "outcome_sha256": row["record_sha256"]})
    evidence = fresh._files(root, [path for path in root.rglob("*") if path.is_file() and path.name != ".generation.lock"])
    identity = manifest["identity"]
    return {"phase": "cohort", "protocol_hash": study.canonical_hash(study._json(study.SPEC)), "source_sha256": study._sources(),
            "runtime": json.loads(json.dumps(study.cache.runtime_identity())), "candidate_lock_sha256": lock["record_sha256"],
            "ledger_id": manifest["ledger_id"], "model_configuration": identity["model_configuration"], "prompt": identity["prompt"],
            "global_budget_fraction": lock["binding"]["global_budget_fraction"],
            "baselines": {key: value for key, value in lock["binding"]["baselines"].items() if key not in ("review", "evidence")},
            "baseline_binding_sha256": study.canonical_hash(lock["binding"]["baselines"]),
            "funnel": funnel, "members": members, "cohort_size": len(members), "generation_evidence": evidence,
            "planned_episode_limits": {"fresh_pilot": 12 * len(members), "confirmation": 16 * len(members)},
            "scope": "Frozen qualification-only cohort. Confirmation data must be collected separately without modifying these lab snapshots."}


def lock_cohort():
    candidate = candidates.verify()
    root, generation = primary.canonical_root(), journal.directory()
    _expect((generation / "journal.json").is_file(), "complete generation before locking a cohort")
    with primary._guard(root.parent / ".initialization.lock"), primary._guard(generation / ".generation.lock"):
        if (root / "cohort_lock.json").exists():
            raise FileExistsError("cohort is already locked; keep its original membership")
        candidate = candidates.verify()
        with ledger._locked(root / "ledger") as (_, manifest):
            ledger._current(manifest)
            state = ledger._read(root / "ledger", manifest)
            _expect(manifest["identity"]["scope"] == "primary" and not state["halted"] and not state["counts"]["active"], "close in-flight or halted evaluation before cohort publication")
            _expect(not any(entry for key, history in state["histories"].items()
                            if state["plans"][key[0]]["binding"]["phase"] in ("fresh_pilot", "confirmation") for entry in history),
                    "fresh model observations appeared before the cohort lock")
            at = datetime.now(UTC).isoformat()
            binding = _binding(generation, candidate, manifest, at)
            record = study._seal({"schema_version": 1, "id": "cohort_lock", "study_lock": True, "binding": binding,
                "bindings_hash": study.canonical_hash(binding), "plan_sha256": study.canonical_hash({"phase": "cohort", "binding": binding}),
                "protocol_hash": binding["protocol_hash"], "canonical_root": str(root.resolve()), "recorded_at_utc": at,
                "status": "qualified_cohort" if binding["members"] else "no_gradable_labs", "model_sessions_started": 0}, "record_sha256")
            study._publish(root / "cohort_lock.json", record)
            primary._append_binding(record)
            return record


def verify(*, ledger_context=None):
    root = primary.canonical_root()
    record = _record(root / "cohort_lock.json")
    candidate = candidates.verify(ledger_context=ledger_context)
    manifest = ledger._manifest(root / "ledger")
    if ledger_context is not None:
        _expect(ledger_context[0].resolve() == (root / "ledger").resolve() and ledger_context[1] == manifest,
                "cohort context differs from the canonical journal")
    ledger._current(manifest)
    binding = _binding(journal.directory(), candidate, manifest, record["recorded_at_utc"])
    _expect(record.get("schema_version") == 1 and record.get("id") == "cohort_lock" and record.get("study_lock") is True
            and record["canonical_root"] == str(root.resolve()) and record.get("model_sessions_started") == 0
            and record["binding"] == binding and record["bindings_hash"] == study.canonical_hash(binding)
            and record["plan_sha256"] == study.canonical_hash({"phase": "cohort", "binding": binding})
            and record["protocol_hash"] == binding["protocol_hash"]
            and record["status"] == ("qualified_cohort" if binding["members"] else "no_gradable_labs"), "cohort membership, data or rule bindings changed")
    public = study._json(primary.BINDINGS)
    expected = {key: record[key] for key in ("id", "plan_sha256", "bindings_hash", "record_sha256", "recorded_at_utc")}
    _expect(public.get("schema_version") == 1 and public.get("protocol_hash") == record["protocol_hash"]
            and [row for row in public.get("records", []) if row.get("id") == "cohort_lock"] == [expected],
        "cohort lock is absent from public binding history")
    return record
