import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from benchtop import admission, arxiv, candidates, primary, study, synth, triage, validate
from benchtop import generation_journal as journal
from benchtop.env import Env, read_jsonl
from benchtop.llm import fenced
from benchtop.sweep import sweep


def _now():
    return datetime.now(UTC).isoformat()


def _put(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("immutable generation artifact changed: " + path.name)
    else:
        arxiv_dir = path.parent
        arxiv_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        from benchtop.candidate_metadata import _bytes

        _bytes(path, raw)


def _sealed(path, value):
    result = study._seal(value, "record_sha256")
    if path.exists():
        if study._json(path) != result:
            raise ValueError("immutable generation record changed")
    else:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        study._publish(path, result)
    return result


def _verify_files(root, files):
    for name, sha in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or study.cache.file_hash(path) != sha:
            raise ValueError("fresh candidate evidence changed")


def _files(root, paths):
    return {str(path.relative_to(root)): study.cache.file_hash(path) for path in paths if path.is_file()}


def _folder(root, paper):
    return root / "papers" / paper["version"]


def _outcomes(root, lock):
    rows, missing = [], False
    for paper in lock["binding"]["candidates"]:
        path = _folder(root, paper) / "outcome.json"
        if not path.exists():
            missing = True
            rows.append({"version": paper["version"], "status": "unprocessed"})
            continue
        if missing:
            raise ValueError("candidate outcome skips an earlier locked paper")
        row = study._json(path)
        study._verify(row, "record_sha256")
        if row["version"] != paper["version"] or row["candidate_lock_sha256"] != lock["record_sha256"]:
            raise ValueError("candidate outcome belongs to another paper or lock")
        if row["status"] not in ("admitted", "rejected", "generation_failed", "qualification_ambiguous", "not_processed_after_target"):
            raise ValueError("unknown candidate disposition")
        if row["status"] in ("admitted", "qualification_ambiguous"):
            proof = study._json(path.parent / "qualification.json")
            study._verify(proof, "record_sha256")
            if proof["status"] != row["status"] or proof["outcome_fields"]["qualification_label"] != row["qualification_label"]:
                raise ValueError("candidate disposition differs from qualification evidence")
        _verify_files(root, row["files"])
        rows.append(row)
    return rows


def _finish(root, lock, paper, status, reason, *, extra=None):
    folder = _folder(root, paper)
    existing = folder / "outcome.json"
    if existing.exists():
        return study._json(existing)
    paths = [path for path in folder.rglob("*") if path.is_file() and path != existing]
    for entry in journal.read()["calls"].values():
        if entry["start"]["version"] == paper["version"]:
            paths.extend(path for path in (root / "calls" / entry["start"]["call_id"]).iterdir() if path.is_file())
    return _sealed(existing, {"version": paper["version"], "candidate_lock_sha256": lock["record_sha256"],
        "status": status, "reason": reason, "files": _files(root, paths), "recorded_at_utc": _now(), **(extra or {})})


def status():
    lock = candidates.verify()
    root = journal.directory()
    rows = _outcomes(root, lock) if root.exists() else [{"version": row["version"], "status": "unprocessed"} for row in lock["binding"]["candidates"]]
    accounting = journal.read()
    for row in rows:
        if row["status"] != "unprocessed":
            continue
        folder = root / "papers" / row["version"]
        if (folder / "qualification_start.json").exists():
            row["status"] = "qualification_pending_reconciliation"
        elif (folder / "qualification_permit.json").exists():
            row["status"] = "awaiting_qualification"
        elif (folder / "review_subject.json").exists():
            row["status"] = "awaiting_human_review"
    return {"candidate_lock_sha256": lock["record_sha256"], "candidates": rows,
            "generation_start_intents": accounting["starts"], "active_generation": accounting["active"],
            "halted": accounting["halted"], "admitted": sum(row["status"] == "admitted" for row in rows),
            "model_sessions_started_by_inspection": 0,
            "scope": "Generation starts are separate from evaluation episode allowances; unknown usage is not authenticated billing."}


def _current(root, lock):
    rows = _outcomes(root, lock)
    target = study._json(study.SPEC)["fresh_cohort"]["target_labs"]
    if sum(row["status"] == "admitted" for row in rows) >= target:
        for paper, row in zip(lock["binding"]["candidates"], rows, strict=True):
            if row["status"] == "unprocessed":
                _finish(root, lock, paper, "not_processed_after_target", "target reached in archived candidate order")
        return None
    return next((paper for paper, row in zip(lock["binding"]["candidates"], rows, strict=True) if row["status"] == "unprocessed"), None)


def _content(folder, paper, client=None):
    path = folder / "content.json"
    if path.exists():
        record = study._json(path)
        study._verify(record, "record_sha256")
        if record["version"] != paper["version"]:
            raise ValueError("paper content version changed")
        _verify_files(folder, record["files"])
        receipt = study._json(folder / record["receipt_file"])
        study._verify(receipt, "record_sha256")
        if receipt["record_sha256"] != record["receipt_sha256"]:
            raise ValueError("paper retrieval receipt changed")
        raw_path = folder / next(iter(record["files"]))
        if record["http_status"] == 404:
            expected = "(no html full text available; work from the abstract)"
        else:
            parser = arxiv._Text()
            parser.feed(raw_path.read_bytes().decode(record["encoding"], errors="replace"))
            expected = parser.text()[:24000]
        if record["excerpt"] != expected:
            raise ValueError("paper excerpt differs from its archived versioned response")
        return record["excerpt"]
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    fetches = folder / "fetches"
    fetches.mkdir(exist_ok=True, mode=0o700)
    index = max((int(path.stem) for path in fetches.iterdir()), default=-1) + 1
    url = "https://arxiv.org/html/" + paper["version"]
    owned, client = client is None, client or httpx.Client()
    record = {"version": paper["version"], "url": url, "started_at_utc": _now(), "status": "incomplete"}
    try:
        arxiv._polite()
        response = client.get(url, headers=arxiv.UA, timeout=60, follow_redirects=False)
        final = urlsplit(str(response.url))
        if final.hostname != "arxiv.org" or final.path.rstrip("/") != "/html/" + paper["version"]:
            raise ValueError("paper retrieval changed the locked arXiv version")
        if len(response.content) > 20 * 1024 * 1024:
            raise ValueError("paper response exceeds the bounded content size")
        raw_path = fetches / f"{index:04d}.html"
        _put(raw_path, response.content)
        record.update(http_status=response.status_code, files=_files(folder, [raw_path]), encoding=response.encoding or "utf-8")
        if response.status_code == 404:
            excerpt = "(no html full text available; work from the abstract)"
        else:
            response.raise_for_status()
            if "ltx_document" not in response.text:
                raise ValueError("successful paper response is not the expected versioned article")
            parser = arxiv._Text()
            parser.feed(response.text)
            excerpt = parser.text()[:24000]
        record.update(status="passed", excerpt=excerpt, finished_at_utc=_now())
        receipt = _sealed(fetches / f"{index:04d}.json", record)
        return _sealed(path, {**record, "receipt_sha256": receipt["record_sha256"], "receipt_file": f"fetches/{index:04d}.json"})["excerpt"]
    except (httpx.HTTPError, ValueError) as error:
        record.update(status="failed", error_type=type(error).__name__, finished_at_utc=_now())
        _sealed(fetches / f"{index:04d}.json", record)
        raise ValueError("paper retrieval failed; retry this same locked candidate before generation") from error
    finally:
        if owned:
            client.close()


def _call(lock, paper, stage, prompt, system):
    finish = journal.invoke(lock, paper, stage, prompt, system)
    if finish["result"]["state"] != "completed":
        raise ValueError("generation provider result is unresolved; preserve it and reconcile without relaunching")
    return finish["result"]["text"]


def _card(text):
    card = json.loads(text)
    if not isinstance(card, dict) or not set(triage.SCHEMA["required"]) <= card.keys() or type(card["testable"]) is not bool:
        raise ValueError("invalid triage schema")
    for key in ("blocker", "reason", "fidelity", "claim", "method", "baseline", "miniature"):
        if not isinstance(card[key], str) or not card[key].strip():
            raise ValueError("invalid triage text")
    if card["blocker"] not in triage.BLOCKERS or card["fidelity"] not in ("high", "medium", "low") or not isinstance(card["contexts"], list):
        raise ValueError("invalid triage enum/contexts")
    metric = card["metric"]
    if not isinstance(metric, dict) or metric.get("goal") not in ("max", "min") or not all(isinstance(metric.get(key), str) and metric[key].strip() for key in ("name", "description")):
        raise ValueError("invalid triage metric")
    for context in card["contexts"]:
        if not isinstance(context, dict) or not isinstance(context.get("levels"), list) or not all(isinstance(context.get(key), str) and context[key].strip() for key in ("name", "why")):
            raise ValueError("invalid triage context")
    return card


def _draft(folder, stage, text, paper):
    work = folder / "drafts" / stage
    code, raw = fenced(text, "python"), fenced(text, "json")
    _put(work / "lab.py", (code or "").encode())
    _put(work / "space.json", (raw or "").encode())
    proof_path = work / "validation.json"
    if proof_path.exists():
        proof = study._json(proof_path)
        study._verify(proof, "record_sha256")
        _verify_files(folder, proof["files"])
        return work, proof
    errors, info = [], {}
    if not code or not raw:
        errors = ["reply must contain both python and json fenced blocks"]
    else:
        try:
            space = json.loads(raw)
            if space.get("paper", {}).get("arxiv_id") != paper["id"]:
                errors.append("space.paper.arxiv_id must match the locked paper")
            if not errors:
                errors, info = validate.preflight(work)
        except (ValueError, TypeError, KeyError) as error:
            errors = ["invalid lab/space contract: " + type(error).__name__]
    return work, _sealed(proof_path, {"stage": stage, "errors": errors, "info": info,
        "files": _files(folder, [work / "lab.py", work / "space.json"]), "finished_at_utc": _now(),
        "seed_scope": "mechanical probes on visible seeds 0 and 1 only; no qualification or confirmation"})


def _ready(lock):
    if lock.get("generation_ready") is not True:
        raise ValueError("candidate lock does not authorize the implemented fresh driver")
    if (primary.canonical_root() / "cohort_lock.json").exists():
        raise ValueError("cohort is locked; generation and admission mutations are closed")
    if journal.read()["halted"]:
        raise ValueError("generation halted after a model/provider contract violation")


def step(*, client=None):
    lock = candidates.verify()
    _ready(lock)
    root = journal.initialize(lock)
    with primary._guard(root / ".generation.lock"):
        paper = _current(root, lock)
        if paper is None:
            return {"status": "collection_complete", "model_sessions_started_by_step": 0}
        folder = _folder(root, paper)
        if (folder / "review_subject.json").exists():
            return {"status": "awaiting_qualification" if (folder / "human_review.json").exists() else "awaiting_human_review", "version": paper["version"]}
        rules = lock["binding"]["generation"]
        prompt = rules["prompts"]["triage"].format(title=paper["title"], categories=", ".join(paper["categories"]), abstract=paper["abstract"])
        system = rules["system_prompt"] + "\nReturn only one JSON object matching this schema:\n" + json.dumps(rules["triage_schema"], sort_keys=True)
        text = _call(lock, paper, "triage", prompt, system)
        try:
            card = _card(text)
        except (ValueError, TypeError):
            return _finish(root, lock, paper, "generation_failed", "triage reply has invalid JSON/schema")
        if not triage.selected({**card, "ok": True}):
            return _finish(root, lock, paper, "rejected", "frozen abstract triage: " + str(card["reason"]))
        excerpt = _content(folder, paper, client)
        task = rules["prompts"]["synthesis"].format(title=paper["title"], arxiv_id=paper["id"], abstract=paper["abstract"], card=synth.card_text(card), excerpt=excerpt)
        prompt = task
        for index in range(1 + rules["max_repairs_per_candidate"]):
            stage = "synthesis" if index == 0 else f"repair_{index}"
            text = _call(lock, paper, stage, prompt, rules["system_prompt"])
            work, proof = _draft(folder, stage, text, paper)
            if not proof["errors"]:
                subject = {"kind": "fresh_prequalification_v1", "candidate_lock_sha256": lock["record_sha256"], "version": paper["version"],
                           "draft": str(work.relative_to(root)), "lab_sha256": study.cache.file_hash(work / "lab.py"),
                           "space_sha256": study.cache.file_hash(work / "space.json"), "validation_sha256": proof["record_sha256"]}
                _sealed(folder / "review_subject.json", subject)
                template = admission.draft_review({"review_subject": subject})
                template["paper"].update(url="https://arxiv.org/abs/" + paper["version"], version=paper["version"])
                template["margin"] = {"value": study._json(work / "space.json")["metric"].get("min_effect"), "rationale": ""}
                study._publish(folder / "review_template.json", template)
                return {"status": "awaiting_human_review", "version": paper["version"]}
            prompt = rules["prompts"]["repair"].format(errors="\n".join(proof["errors"]), lab=(work / "lab.py").read_text().strip(),
                space=(work / "space.json").read_text().strip(), task=task[:6000])
        return _finish(root, lock, paper, "generation_failed", "initial generation and two bounded repairs failed mechanical validation")


def review(source):
    lock = candidates.verify()
    _ready(lock)
    root = journal.directory()
    if not (root / "journal.json").exists():
        raise ValueError("generate a candidate before submitting its human review")
    with primary._guard(root / ".generation.lock"):
        paper = _current(root, lock)
        if paper is None:
            raise ValueError("candidate collection is already complete")
        folder = _folder(root, paper)
        if (folder / "qualification_permit.json").exists():
            raise ValueError("human review and margin are already frozen; continue qualification")
        saved = study._json(folder / "review_subject.json")
        study._verify(saved, "record_sha256")
        subject = {key: value for key, value in saved.items() if key != "record_sha256"}
        raw = Path(source).read_bytes()
        document = json.loads(raw)
        assessed = admission._review(document, subject)
        if assessed["status"] != "recorded" or assessed["decision"] == "needs_review":
            raise ValueError("complete the source-bound human scientific review before qualification")
        work = root / subject["draft"]
        proof = study._json(work / "validation.json")
        if (proof["record_sha256"] != subject["validation_sha256"] or study.cache.file_hash(work / "lab.py") != subject["lab_sha256"]
                or study.cache.file_hash(work / "space.json") != subject["space_sha256"] or proof["errors"]):
            raise ValueError("reviewed lab/space or mechanical evidence changed")
        from benchtop.candidate_metadata import timestamp

        if not timestamp(proof["finished_at_utc"]) <= timestamp(document["reviewer"]["reviewed_at_utc"]) <= timestamp(_now()):
            raise ValueError("human review must follow mechanical validation and precede qualification")
        if document["paper"]["version"] != paper["version"] or document["paper"]["url"] != "https://arxiv.org/abs/" + paper["version"]:
            raise ValueError("human review cites a different paper version")
        if (folder / "qualification_start.json").exists():
            raise ValueError("qualification already started; the review and effect margin are frozen")
        if assessed["decision"] == "rejected":
            _put(folder / "human_review.json", raw)
            return _finish(root, lock, paper, "rejected", "source-bound human rejection before qualification")
        margin = document.get("margin", {})
        value = margin.get("value")
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0 or not isinstance(margin.get("rationale"), str) or not margin["rationale"].strip():
            raise ValueError("justify a finite positive effect margin before qualification")
        lab = folder / "lab"
        space = study._json(work / "space.json")
        space["metric"]["min_effect"] = value
        _put(lab / "lab.py", (work / "lab.py").read_bytes())
        _put(lab / "space.json", (json.dumps(space, indent=2, allow_nan=False) + "\n").encode())
        splits = {name: lock["binding"]["generation"]["seed_partitions"][name] for name in ("visible", "qualification", "confirmation")}
        manifest = {"id": paper["version"], "source": "arxiv", "arxiv_id": paper["id"], "arxiv_version": paper["version"],
                    "paper_title": paper["title"], "published": paper["published"], "seeds": 28, "visible_seeds": 4, "seed_splits": splits}
        _put(lab / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
        _put(folder / "human_review.json", raw)
        permit = _sealed(folder / "qualification_permit.json", {"subject": subject, "review_sha256": hashlib.sha256(raw).hexdigest(),
            "files": _files(root, [lab / "lab.py", lab / "space.json", lab / "manifest.json"]), "recorded_at_utc": _now()})
        return {"status": "awaiting_qualification", "version": paper["version"], "permit_sha256": permit["record_sha256"]}


def qualify():
    lock = candidates.verify()
    _ready(lock)
    root = journal.directory()
    if not (root / "journal.json").exists():
        raise ValueError("generate and review a candidate before qualification")
    with primary._guard(root / ".generation.lock"):
        paper = _current(root, lock)
        if paper is None:
            return {"status": "collection_complete"}
        folder = _folder(root, paper)
        permit = study._json(folder / "qualification_permit.json")
        study._verify(permit, "record_sha256")
        if study.cache.file_hash(folder / "human_review.json") != permit["review_sha256"]:
            raise ValueError("human review changed after effect-margin authorization")
        finished = folder / "qualification.json"
        if finished.exists():
            proof = study._json(finished)
            study._verify(proof, "record_sha256")
            if proof["permit_sha256"] != permit["record_sha256"]:
                raise ValueError("qualification proof belongs to another pre-qualification review")
            _verify_files(root, proof["files"])
            return _finish(root, lock, paper, proof["status"], proof["reason"], extra=proof["outcome_fields"])
        _verify_files(root, permit["files"])
        if (folder / "qualification_start.json").exists():
            raise ValueError("qualification already started; retain unfinished evidence for explicit reconciliation")
        lab = folder / "lab"
        if (lab / "results.jsonl").exists():
            raise ValueError("qualification data appeared before its authorized start")
        _sealed(folder / "qualification_start.json", {"permit_sha256": permit["record_sha256"], "started_at_utc": _now()})
        rules = lock["binding"]["generation"]
        visible = sweep(lab, split="visible", workers=rules["sweep_workers"], timeout=rules["run_timeout_seconds"], log=lambda *_: None)
        if visible["runs_failed"] or visible["cpu_secs"] > rules["max_recorded_sweep_cpu_seconds"]:
            return _finish(root, lock, paper, "generation_failed", "visible sweep failed before qualification; no code repair uses qualification outcomes",
                           extra={"confirmation_evaluated": False})
        qualified = sweep(lab, split="qualification", workers=rules["sweep_workers"], timeout=rules["run_timeout_seconds"], log=lambda *_: None)
        env = Env.load(lab)
        if any(row["seed"] in env.confirmation_seeds for row in read_jsonl(lab / "results.jsonl")):
            raise ValueError("confirmation seeds were evaluated before the confirmation lock")
        report = admission.inspect_lab(lab)
        if report["mechanical"]["status"] != "passed" or visible["cpu_secs"] + qualified["cpu_secs"] > rules["max_recorded_sweep_cpu_seconds"]:
            state, reason = "generation_failed", "qualification/sandbox/completeness or CPU acceptance checks failed"
        elif report["qualification"]["claim_eligible"]:
            state, reason = "admitted", "pre-qualification human acceptance and complete settled qualification claim"
        else:
            state, reason = "qualification_ambiguous", "no settled qualification claim; retain this candidate without replacement"
        fields = {"lab": str(lab.relative_to(root)), "qualification_label": (report["qualification"].get("claim") or {}).get("label"), "confirmation_evaluated": False}
        _sealed(finished, {"permit_sha256": permit["record_sha256"], "visible": visible, "qualification": qualified,
            "screen": report, "files": _files(root, [path for path in lab.iterdir() if path.is_file()]), "finished_at_utc": _now(),
            "confirmation_evaluated": False, "status": state, "reason": reason, "outcome_fields": fields})
        return _finish(root, lock, paper, state, reason, extra=fields)
