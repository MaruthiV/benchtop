import copy
import json
from datetime import UTC, datetime

import httpx
import pytest

from benchtop import candidates, primary, provider, study
from benchtop import fresh_generation as fresh
from benchtop import generation_journal as journal
from benchtop import session_ledger as ledger
from benchtop.cli import main
from benchtop.env import read_jsonl
from tests.test_admission import human_control
from tests.test_primary import authorize, preparation
from tests.test_provider import encode, preflight
from tests.test_study import model


def card():
    return {"testable": True, "blocker": "none", "reason": "Synthetic fixture only", "fidelity": "high",
            "claim": "A controlled shift", "method": "new", "baseline": "base",
            "metric": {"name": "value", "goal": "max", "description": "Synthetic metric"},
            "contexts": [{"name": "context", "levels": [0, 1], "why": "Synthetic controlled context"}], "miniature": "Test fixture"}


def lab_reply(paper, *, bad=False, effect=0.2):
    code = "import os\ndef run(config, seed): return {'value': 0.0}" if bad else (
        "def run(config, seed):\n"
        f"    return {{'value': 0.5 + 0.001 * (seed % 3) + ({effect} if config['method'] == 'new' else 0)}}\n")
    space = {"title": "Synthetic generation control", "setting": "Test data only", "paper": {"arxiv_id": paper["id"], "claim": "Controlled treatment"},
             "metric": {"name": "value", "goal": "max", "min_effect": 0.02},
             "factors": [{"name": "method", "role": "treatment", "levels": ["base", "new"]},
                         {"name": "context", "role": "context", "levels": [0, 1]}],
             "treatment": {"factor": "method", "baseline": "base", "paper": "new"}}
    return f"```python\n{code}\n```\n```json\n{json.dumps(space)}\n```"


class Content:
    def __init__(self, status=404, suffix="", text=None):
        self.status, self.suffix, self.text, self.calls = status, suffix, text, []

    def get(self, url, **kwargs):
        self.calls.append(url)
        return httpx.Response(self.status, text=self.text if self.text is not None else "not converted to HTML",
                              request=httpx.Request("GET", url + self.suffix))


def setup(tmp_path, monkeypatch, *, count=2):
    plans = preparation(tmp_path, monkeypatch)
    authorize(plans)
    papers = [{"id": f"2611.{index:05d}", "version": f"2611.{index:05d}v2", "title": "Generation fixture",
               "categories": ["cs.LG"], "primary": "cs.LG", "abstract": "Synthetic test only", "published": "2026-11-06T01:00:00Z"}
              for index in range(1, count + 1)]
    rules = candidates.generation_rules({stage: model() for stage in ("triage", "synthesis", "repair")}, preflight())
    lock = study._seal({"id": "candidate_lock", "study_lock": True, "generation_ready": True,
        "binding": {"generation": rules, "candidates": papers, "global_budget_fraction": 0.12,
                    "baselines": {"policies": ["random"], "estimator_id": "fixture_only", "development_majority_label": "paper_better"}},
        "scope": "TEST FIXTURE ONLY; candidate-lock authority stubbed"}, "record_sha256")
    study._publish(primary.canonical_root() / "candidate_lock.json", lock)
    monkeypatch.setattr(candidates, "verify", lambda: copy.deepcopy(lock))
    monkeypatch.setattr(fresh.arxiv, "_polite", lambda: None)
    calls = []

    def execute(work, prompt, config, checked, system, timeout):
        provider.text_command(work, prompt, config, checked, system)
        start = study._json(work / "start.json")
        calls.append(start)
        paper = next(row for row in papers if row["version"] == start["version"])
        result = json.dumps(card()) if start["stage"] == "triage" else lab_reply(paper)
        return process(work, result)

    monkeypatch.setattr(provider, "execute_text", execute)
    return lock, calls


def process(work, text, *, models=None, tools=None, error=False):
    model_id = model()["reported_model_id"] if models is None else models
    stdout = encode([{"type": "system", "subtype": "init", "model": model_id, "tools": [] if tools is None else tools,
                      "mcp_servers": [], "fixture_only": True},
                     {"type": "result", "result": text, "is_error": error, "modelUsage": {model_id: {"inputTokens": 10, "outputTokens": 5}},
                      "fixture_only": True}])
    (work / "stdout.jsonl").write_text(stdout)
    (work / "stderr.txt").write_text("TEST FIXTURE ONLY; no Claude process\n")
    return {"executor": "claude_cli", "process_stopped": True, "returncode": int(error), "timed_out": False,
            "stdout_truncated": False, "process_wall_seconds": 0.01, "stdout_sha256": study.cache.file_hash(work / "stdout.jsonl"),
            "stderr_sha256": study.cache.file_hash(work / "stderr.txt"), "stream": provider.parse_text_stream(stdout), "model_sessions": 1}


def accept(root, version, *, margin=0.02):
    template = study._json(root / "papers" / version / "review_template.json")
    document = human_control({"review_subject": template["subject"]})
    document["paper"].update(url=template["paper"]["url"], version=version)
    document["reviewer"]["reviewed_at_utc"] = datetime.now(UTC).isoformat()
    document["margin"] = {"value": margin, "rationale": "Fixed from the meaning of the synthetic metric before qualification."}
    path = root.parent.parent / "human-input.json"
    path.write_text(json.dumps(document))
    return path


def test_real_sandbox_generation_review_and_staged_sweeps_never_evaluate_confirmation(tmp_path, monkeypatch):
    lock, calls = setup(tmp_path, monkeypatch)
    fetched = Content()
    first = fresh.step(client=fetched)
    root = journal.directory()
    paper = lock["binding"]["candidates"][0]
    folder = root / "papers" / paper["version"]
    assert first["status"] == "awaiting_human_review" and len(calls) == 2
    assert fetched.calls == ["https://arxiv.org/html/" + paper["version"]]
    assert fresh.step(client=Content())["status"] == "awaiting_human_review" and len(calls) == 2
    assert not (folder / "lab/results.jsonl").exists()
    with pytest.raises(ValueError, match="complete.*human"):
        fresh.review(folder / "review_template.json")
    fresh.review(accept(root, paper["version"], margin=0.05))
    assert not (folder / "lab/results.jsonl").exists()
    result = fresh.qualify()
    assert result["status"] == "admitted" and result["confirmation_evaluated"] is False
    rows = read_jsonl(folder / "lab/results.jsonl")
    assert len(rows) == 64 and {row["seed"] for row in rows} == set(range(16))
    assert study._json(folder / "lab/space.json")["metric"]["min_effect"] == 0.05
    assert len(calls) == 2 and fresh.status()["admitted"] == 1
    assert ledger.inspect(primary.canonical_root() / "ledger")["counts"]["total"] == 0
    (folder / "lab/space.json").write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        fresh.status()


def test_two_repairs_are_bounded_and_do_not_use_qualification_feedback(tmp_path, monkeypatch):
    lock, calls = setup(tmp_path, monkeypatch)
    original = provider.execute_text

    def invalid(work, prompt, config, checked, system, timeout):
        result = original(work, prompt, config, checked, system, timeout)
        start = study._json(work / "start.json")
        if start["stage"] != "triage":
            result = process(work, lab_reply(lock["binding"]["candidates"][0], bad=True))
        return result

    monkeypatch.setattr(provider, "execute_text", invalid)
    result = fresh.step(client=Content())
    assert result["status"] == "generation_failed"
    assert [row["stage"] for row in calls] == ["triage", "synthesis", "repair_1", "repair_2"]
    assert not list(journal.directory().glob("papers/*/lab/results.jsonl"))
    assert all("qualification" not in row["prompt"].lower() for row in calls[1:])
    assert len(fresh.status()["candidates"]) == 2


@pytest.mark.parametrize("defect", ["model", "tools", "provider_error"])
def test_provider_contract_or_unknown_errors_never_launch_a_retry(tmp_path, monkeypatch, defect):
    _, calls = setup(tmp_path, monkeypatch)
    def bad(work, *args):
        provider.text_command(work, args[0], args[1], args[2], args[3])
        return process(work, json.dumps(card()), models="different-model" if defect == "model" else None,
                       tools=["Bash"] if defect == "tools" else None, error=defect == "provider_error")
    monkeypatch.setattr(provider, "execute_text", bad)
    with pytest.raises(ValueError, match="unresolved"):
        fresh.step(client=Content())
    assert journal.read()["starts"] == 1 and journal.read()["active"] == 0
    assert journal.read()["halted"] == (defect != "provider_error")
    with pytest.raises(ValueError):
        fresh.step(client=Content())
    assert journal.read()["starts"] == 1 and calls == []


def test_lost_generation_acknowledgement_resumes_without_a_second_provider_call(tmp_path, monkeypatch):
    _, calls = setup(tmp_path, monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(journal, "resume", lambda *a: (_ for _ in ()).throw(OSError("fixture lost acknowledgement")))
        with pytest.raises(OSError):
            fresh.step(client=Content())
    state = journal.read()
    assert state["starts"] == state["active"] == 1 and len(calls) == 1
    call_id = next(iter(state["calls"]))
    journal.resume(call_id)
    assert journal.read()["active"] == 0
    assert fresh.step(client=Content())["status"] == "awaiting_human_review"
    assert len(calls) == 2


def test_pending_generation_and_evaluation_share_two_active_slots(tmp_path, monkeypatch):
    _, _ = setup(tmp_path, monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(provider, "execute_text", lambda *a: (_ for _ in ()).throw(OSError("fixture startup interruption")))
        with pytest.raises(OSError):
            fresh.step(client=Content())
    root = primary.canonical_root() / "ledger"
    plan = next((root / "plans").glob("*.json"))
    prepared = study._json(plan)
    ledger.begin(root, prepared["plan_sha256"], prepared["slots"][0]["slot_id"])
    with pytest.raises(ValueError, match="two in-flight"):
        ledger.begin(root, prepared["plan_sha256"], prepared["slots"][1]["slot_id"])
    snapshot = ledger.inspect(root)
    assert snapshot["active_generation"] == 1 and snapshot["combined_active_model_intents"] == 2
    assert snapshot["counts"]["total"] == 1


@pytest.mark.parametrize("status,suffix", [(503, ""), (200, "v3")])
def test_content_failures_pause_the_same_candidate_and_keep_version_pin(tmp_path, monkeypatch, status, suffix):
    lock, calls = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="same locked candidate"):
        fresh.step(client=Content(status, suffix))
    assert len(calls) == 1 and fresh.status()["candidates"][0]["status"] == "unprocessed"
    success = Content()
    assert fresh.step(client=success)["status"] == "awaiting_human_review"
    assert len(calls) == 2 and lock["binding"]["candidates"][0]["version"] in success.calls[0]


def test_human_margin_and_draft_are_frozen_before_qualification(tmp_path, monkeypatch):
    lock, _ = setup(tmp_path, monkeypatch)
    fresh.step(client=Content())
    version = lock["binding"]["candidates"][0]["version"]
    root = journal.directory()
    path = accept(root, version, margin=-0.1)
    with pytest.raises(ValueError, match="positive effect margin"):
        fresh.review(path)
    path = accept(root, version)
    fresh.review(path)
    with pytest.raises(ValueError, match="already frozen"):
        fresh.review(path)
    source = root / "papers" / version / "lab/lab.py"
    source.write_text(source.read_text() + "\n# changed fixture\n")
    with pytest.raises(ValueError, match="evidence changed"):
        fresh.qualify()
    assert not (source.parent / "results.jsonl").exists()


def test_premature_review_and_qualification_do_not_poison_generation_directory(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    for action in (lambda: fresh.review(tmp_path / "absent.json"), fresh.qualify):
        with pytest.raises(ValueError, match="before"):
            action()
    assert not journal.directory().exists()
    assert fresh.step(client=Content())["status"] == "awaiting_human_review"


def test_candidate_gate_prevents_any_real_or_scripted_provider_invocation(tmp_path, monkeypatch):
    monkeypatch.setattr(primary, "STATE_BASE", tmp_path / "absent")
    monkeypatch.setattr(provider, "execute_text", lambda *a: pytest.fail("unauthorized model invocation"))
    with pytest.raises(FileNotFoundError):
        fresh.step()
    assert not (tmp_path / "absent").exists()


def test_cli_status_has_no_prompt_or_hidden_data(tmp_path, monkeypatch, capsys):
    setup(tmp_path, monkeypatch)
    fresh.step(client=Content())
    main(["study", "generation-status"])
    result = json.loads(capsys.readouterr().out)
    assert result["generation_start_intents"] == 2 and result["model_sessions_started_by_inspection"] == 0
    assert result["candidates"][0]["status"] == "awaiting_human_review"
    assert "prompt" not in json.dumps(result) and "qualification_label" not in json.dumps(result)


def test_ambiguous_qualification_never_triggers_outcome_based_code_repair(tmp_path, monkeypatch):
    lock, calls = setup(tmp_path, monkeypatch)
    original = provider.execute_text
    def mixed(work, *args):
        result = original(work, *args)
        start = study._json(work / "start.json")
        if start["stage"] != "triage":
            paper = next(paper for paper in lock["binding"]["candidates"] if paper["version"] == start["version"])
            result = process(work, lab_reply(paper, effect="0.02 + (0.3 if seed % 2 else -0.3)"))
        return result
    monkeypatch.setattr(provider, "execute_text", mixed)
    fresh.step(client=Content())
    version = lock["binding"]["candidates"][0]["version"]
    fresh.review(accept(journal.directory(), version))
    assert fresh.qualify()["status"] == "qualification_ambiguous"
    fresh.step(client=Content())
    assert len(calls) == 4 and all(row["stage"] in ("triage", "synthesis") for row in calls)
    assert calls[-1]["version"] == lock["binding"]["candidates"][1]["version"]


def test_qualification_interruption_is_not_silently_reexecuted(tmp_path, monkeypatch):
    lock, _ = setup(tmp_path, monkeypatch)
    fresh.step(client=Content())
    version = lock["binding"]["candidates"][0]["version"]
    fresh.review(accept(journal.directory(), version))
    counter = []
    def interrupted(*args, **kwargs):
        counter.append(kwargs["split"])
        raise OSError("fixture CPU interruption")
    monkeypatch.setattr(fresh, "sweep", interrupted)
    with pytest.raises(OSError):
        fresh.qualify()
    with pytest.raises(ValueError, match="already started"):
        fresh.qualify()
    assert counter == ["visible"]


def test_qualification_saved_proof_can_finish_after_lost_ack_without_more_cpu(tmp_path, monkeypatch):
    lock, _ = setup(tmp_path, monkeypatch)
    fresh.step(client=Content())
    fresh.review(accept(journal.directory(), lock["binding"]["candidates"][0]["version"]))
    with monkeypatch.context() as patch:
        patch.setattr(fresh, "_finish", lambda *a, **kw: (_ for _ in ()).throw(OSError("fixture lost qualification acknowledgement")))
        with pytest.raises(OSError):
            fresh.qualify()
    monkeypatch.setattr(fresh, "sweep", lambda *a, **kw: pytest.fail("qualification CPU repeated after durable proof"))
    assert fresh.qualify()["status"] == "admitted"


def test_eight_admitted_gate_results_stop_processing_remaining_candidates(tmp_path, monkeypatch):
    lock, calls = setup(tmp_path, monkeypatch, count=10)
    root = journal.initialize(lock)
    rows = [{"version": paper["version"], "status": "admitted" if index < 8 else "unprocessed"}
            for index, paper in enumerate(lock["binding"]["candidates"])]
    monkeypatch.setattr(fresh, "_outcomes", lambda *a: rows)
    stopped = []
    monkeypatch.setattr(fresh, "_finish", lambda root, locked, paper, status, reason: stopped.append((paper["version"], status)))
    assert fresh._current(root, lock) is None and len(stopped) == 2 and calls == []
    assert all(status == "not_processed_after_target" for _, status in stopped)


def test_invalid_triage_and_locked_cohort_never_start_synthesis(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    def invalid(work, prompt, config, checked, system, timeout):
        provider.text_command(work, prompt, config, checked, system)
        return process(work, "{}")
    monkeypatch.setattr(provider, "execute_text", invalid)
    result = fresh.step(client=Content())
    assert result["status"] == "generation_failed" and journal.read()["starts"] == 1
    study._publish(primary.canonical_root() / "cohort_lock.json", {"scope": "fixture closed-cohort marker"})
    with pytest.raises(ValueError, match="cohort is locked"):
        fresh.step(client=Content())
    assert journal.read()["starts"] == 1
