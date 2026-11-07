import hashlib
import itertools
import json
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import pytest

from benchtop import admission, cohort, primary, provider, study
from benchtop import fresh_generation as fresh
from benchtop import generation_journal as journal
from benchtop import session_ledger as ledger
from benchtop.cli import main
from tests.test_fresh_generation import Content, accept, card, lab_reply, process, setup


def ready(tmp_path, monkeypatch, *, count=1, effect=0.2):
    lock, calls = setup(tmp_path, monkeypatch, count=count)
    if effect != 0.2:
        original = provider.execute_text

        def execute(work, *args):
            result = original(work, *args)
            start = study._json(work / "start.json")
            paper = next(p for p in lock["binding"]["candidates"] if p["version"] == start["version"])
            return result if start["stage"] == "triage" else process(work, lab_reply(paper, effect=effect))

        monkeypatch.setattr(provider, "execute_text", execute)
    for paper in lock["binding"]["candidates"]:
        assert fresh.step(client=Content())["status"] == "awaiting_human_review"
        fresh.review(accept(journal.directory(), paper["version"], margin=0.05))
        fresh.qualify()
    return lock, calls


def no_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("cohort publication/inspection must only read existing evidence")

    monkeypatch.setattr(provider, "execute_text", forbidden)
    monkeypatch.setattr(provider, "probe", forbidden)
    monkeypatch.setattr(fresh, "sweep", forbidden)


def reseal_outcome(folder):
    path = folder / "outcome.json"
    row = study._json(path)
    row["files"] = fresh._files(journal.directory(), [journal.directory() / name for name in row["files"]])
    path.write_text(json.dumps(study._seal(row, "record_sha256")))


def test_freeze_smaller_cohort_binds_order_truth_seed_orders_and_all_evidence_without_execution(tmp_path, monkeypatch):
    candidate, calls = ready(tmp_path, monkeypatch, count=2)
    no_execution(monkeypatch)
    before = ledger.inspect(primary.canonical_root() / "ledger")["counts"]
    record = cohort.lock_cohort()
    binding = record["binding"]
    assert record["study_lock"] and record["status"] == "qualified_cohort" and record["model_sessions_started"] == 0
    assert binding["cohort_size"] == 2 and binding["planned_episode_limits"] == {"fresh_pilot": 24, "confirmation": 32}
    assert [member["version"] for member in binding["members"]] == [paper["version"] for paper in candidate["binding"]["candidates"]]
    assert binding["global_budget_fraction"] == 0.12
    assert binding["baseline_binding_sha256"] == study.canonical_hash(candidate["binding"]["baselines"])
    assert len(calls) == 4 and ledger.inspect(primary.canonical_root() / "ledger")["counts"] == before
    for member in binding["members"]:
        assert member["qualification_label"] == "paper_better" and member["min_effect"] == 0.05
        assert member["seed_splits"] == {"visible": list(range(4)), "qualification": list(range(4, 16)), "confirmation": list(range(16, 28))}
        orders = list(itertools.permutations(range(4)))
        index = int(hashlib.sha256(("benchtop_release_v1/order/" + member["lab_hash"]).encode()).hexdigest(), 16) % 24
        assert member["visible_order"] == list(orders[index])
        assert member["confirmation_visible_order"] == list(orders[(index + 1) % 24])
        assert all(name in member["files"] for name in ("lab.py", "space.json", "manifest.json", "results.jsonl", "tasks.json", "cache.json"))
    assert any(name.endswith("stdout.jsonl") for name in binding["generation_evidence"])
    assert any(name.endswith("human_review.json") for name in binding["generation_evidence"])
    assert cohort.verify() == record
    public = study._json(primary.BINDINGS)
    assert [row["id"] for row in public["records"]] == ["cohort_lock"]


@pytest.mark.parametrize("case", ["triage_declined", "invalid_triage", "exhausted_repairs"])
def test_zero_eligible_cohort_preserves_failures_and_reports_no_comparison(tmp_path, monkeypatch, case):
    candidate, _ = setup(tmp_path, monkeypatch, count=1)
    original = provider.execute_text

    def execute(work, *args):
        result = original(work, *args)
        stage = study._json(work / "start.json")["stage"]
        if case == "invalid_triage":
            return process(work, "malformed fixture output")
        if case == "triage_declined":
            return process(work, json.dumps({**card(), "testable": False}))
        return result if stage == "triage" else process(work, lab_reply(candidate["binding"]["candidates"][0], bad=True))

    monkeypatch.setattr(provider, "execute_text", execute)
    fresh.step(client=Content())
    no_execution(monkeypatch)
    record = cohort.lock_cohort()
    assert record["status"] == "no_gradable_labs" and record["binding"]["members"] == []
    assert record["binding"]["planned_episode_limits"] == {"fresh_pilot": 0, "confirmation": 0}
    assert record["binding"]["funnel"][0]["status"] == ("rejected" if case == "triage_declined" else "generation_failed")
    assert cohort.verify() == record


def test_ambiguous_qualification_is_retained_without_substitution(tmp_path, monkeypatch):
    ready(tmp_path, monkeypatch, effect="0.05 + (0.3 if seed % 2 else -0.3)")
    no_execution(monkeypatch)
    record = cohort.lock_cohort()
    assert record["status"] == "no_gradable_labs"
    assert record["binding"]["funnel"][0]["status"] == "qualification_ambiguous"


def test_incomplete_human_review_cannot_publish_a_cohort(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch, count=1)
    fresh.step(client=Content())
    before = primary.BINDINGS.read_bytes() if primary.BINDINGS.exists() else None
    with pytest.raises(ValueError, match="complete ordered candidate funnel"):
        cohort.lock_cohort()
    assert not (primary.canonical_root() / "cohort_lock.json").exists()
    assert (primary.BINDINGS.read_bytes() if primary.BINDINGS.exists() else None) == before


def test_human_rejection_is_retained_without_qualification(tmp_path, monkeypatch):
    candidate, _ = setup(tmp_path, monkeypatch, count=1)
    fresh.step(client=Content())
    path = accept(journal.directory(), candidate["binding"]["candidates"][0]["version"])
    review = study._json(path)
    review["decision"] = "rejected"
    review["checks"][next(iter(review["checks"]))]["status"] = "fail"
    path.write_text(json.dumps(review))
    fresh.review(path)
    no_execution(monkeypatch)
    record = cohort.lock_cohort()
    assert record["status"] == "no_gradable_labs" and record["binding"]["funnel"][0]["basis"] == "human_rejection"
    assert not list(journal.directory().glob("papers/*/lab/results.jsonl"))


def test_lock_is_immutable_and_closes_all_generation_mutations(tmp_path, monkeypatch):
    candidate, calls = ready(tmp_path, monkeypatch)
    no_execution(monkeypatch)
    record = cohort.lock_cohort()
    before = (primary.canonical_root() / "cohort_lock.json").read_bytes()
    with pytest.raises(FileExistsError, match="already locked"):
        cohort.lock_cohort()
    for operation in (fresh.step, fresh.qualify, lambda: fresh.review(tmp_path / "missing"),
                      lambda: journal.invoke(candidate, candidate["binding"]["candidates"][0], "triage", "different", "different")):
        with pytest.raises(ValueError, match="cohort is locked"):
            operation()
    assert len(calls) == 2 and (primary.canonical_root() / "cohort_lock.json").read_bytes() == before
    assert cohort.verify() == record


@pytest.mark.parametrize("kind", ["review", "margin", "results", "tasks", "public_binding"])
def test_post_lock_evidence_edits_are_refused(tmp_path, monkeypatch, kind):
    ready(tmp_path, monkeypatch)
    no_execution(monkeypatch)
    cohort.lock_cohort()
    folder = next((journal.directory() / "papers").iterdir())
    if kind == "public_binding":
        public = study._json(primary.BINDINGS)
        public["records"] = [row for row in public["records"] if row["id"] != "cohort_lock"]
        primary.BINDINGS.write_text(json.dumps(public))
    else:
        path = folder / {"review": "human_review.json", "margin": "lab/space.json", "results": "lab/results.jsonl", "tasks": "lab/tasks.json"}[kind]
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        cohort.verify()


def test_confirmation_rows_are_rejected_before_any_cached_grading(tmp_path, monkeypatch):
    ready(tmp_path, monkeypatch)
    folder = next((journal.directory() / "papers").iterdir())
    result = folder / "lab/results.jsonl"
    row = json.loads(result.read_text().splitlines()[0])
    row["seed"] = 16
    with result.open("a") as output:
        output.write(json.dumps(row) + "\n")
    reseal_outcome(folder)
    no_execution(monkeypatch)
    monkeypatch.setattr(admission, "inspect_lab", lambda *a: pytest.fail("confirmation data must be refused before grading"))
    with pytest.raises(ValueError, match="confirmation or undeclared seeds"):
        cohort.lock_cohort()


@pytest.mark.parametrize("kind", ["wrong_version", "after_qualification", "margin", "lab_pointer", "early_skip", "sweep_accounting"])
def test_resealed_dispositions_still_require_real_lineage(tmp_path, monkeypatch, kind):
    ready(tmp_path, monkeypatch)
    folder = next((journal.directory() / "papers").iterdir())
    path = folder / "human_review.json"
    review = study._json(path)
    if kind == "wrong_version":
        review["paper"]["version"] = "2611.99999v1"
    elif kind == "after_qualification":
        review["reviewer"]["reviewed_at_utc"] = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    elif kind == "margin":
        review["margin"]["value"] = 2
    if kind in ("wrong_version", "after_qualification", "margin"):
        path.write_text(json.dumps(review))
    elif kind == "sweep_accounting":
        path = folder / "qualification.json"
        proof = study._json(path)
        proof["visible"]["cpu_secs"] += 1
        path.write_text(json.dumps(study._seal(proof, "record_sha256")))
    else:
        path = folder / "outcome.json"
        row = study._json(path)
        row.update({"lab": "other/lab"} if kind == "lab_pointer" else {"status": "not_processed_after_target"})
        path.write_text(json.dumps(study._seal(row, "record_sha256")))
    reseal_outcome(folder)
    no_execution(monkeypatch)
    with pytest.raises(ValueError):
        cohort.lock_cohort()
    assert not (primary.canonical_root() / "cohort_lock.json").exists()


def test_resealed_membership_cannot_reorder_the_frozen_cohort(tmp_path, monkeypatch):
    ready(tmp_path, monkeypatch)
    no_execution(monkeypatch)
    record = cohort.lock_cohort()
    record["binding"]["members"][0]["visible_order"].reverse()
    record["bindings_hash"] = study.canonical_hash(record["binding"])
    record["plan_sha256"] = study.canonical_hash({"phase": "cohort", "binding": record["binding"]})
    (primary.canonical_root() / "cohort_lock.json").write_text(json.dumps(study._seal(record, "record_sha256")))
    with pytest.raises(ValueError, match="membership, data or rule bindings"):
        cohort.verify()


def test_no_canonical_state_does_not_create_any_when_locking_or_inspecting(tmp_path, monkeypatch):
    monkeypatch.setattr(primary, "STATE_BASE", tmp_path / "private")
    monkeypatch.setattr(primary, "BINDINGS", tmp_path / "public.json")
    no_execution(monkeypatch)
    for operation in (cohort.lock_cohort, cohort.verify):
        with pytest.raises(FileNotFoundError):
            operation()
    assert not primary.STATE_BASE.exists() and not primary.BINDINGS.exists()


def test_cli_has_no_membership_override_and_redacts_qualification_labels(tmp_path, monkeypatch, capsys):
    ready(tmp_path, monkeypatch)
    no_execution(monkeypatch)
    main(["study", "lock-cohort"])
    result = json.loads(capsys.readouterr().out)
    assert result["cohort_size"] == 1 and result["model_sessions_started"] == 0
    assert "members" not in result and "qualification_label" not in result
    main(["study", "inspect-cohort"])
    assert json.loads(capsys.readouterr().out) == result
    with pytest.raises(SystemExit):
        main(["study", "lock-cohort", "--lab", "cherry-picked"])


def test_generation_rechecks_closure_after_waiting_for_its_lock(tmp_path, monkeypatch):
    _, calls = setup(tmp_path, monkeypatch, count=1)
    guard = primary._guard

    @contextmanager
    def close_before_entry(path):
        with guard(path):
            if path.name == ".generation.lock":
                study._publish(primary.canonical_root() / "cohort_lock.json", {"fixture": "concurrent closure"})
            yield

    monkeypatch.setattr(primary, "_guard", close_before_entry)
    with pytest.raises(ValueError, match="cohort is locked"):
        fresh.step(client=Content())
    assert calls == [] and journal.read()["starts"] == 0


@pytest.mark.parametrize("kind", ["pending", "halted"])
def test_unreconciled_or_halted_generation_cannot_lock(tmp_path, monkeypatch, kind):
    ready(tmp_path, monkeypatch)
    saved = journal.read

    def accounting(root=None):
        return {**saved(root), "active": int(kind == "pending"), "halted": kind == "halted"}

    monkeypatch.setattr(journal, "read", accounting)
    no_execution(monkeypatch)
    with pytest.raises(ValueError, match="stopped, reconciled generation"):
        cohort.lock_cohort()
    assert not (primary.canonical_root() / "cohort_lock.json").exists()


def test_target_stops_at_first_eight_and_requires_explicit_unprocessed_tail(tmp_path, monkeypatch):
    lock, _ = setup(tmp_path, monkeypatch, count=10)
    root = journal.initialize(lock)
    at = datetime.now(UTC).isoformat()
    rows = []
    for index, paper in enumerate(lock["binding"]["candidates"]):
        folder = fresh._folder(root, paper)
        folder.mkdir()
        (folder / "outcome.json").write_text("fixture-only membership boundary\n")
        rows.append({"id": paper["id"], "version": paper["version"], "status": "admitted" if index < 8 else "not_processed_after_target",
                     "reason": "TEST BOUNDARY ONLY; individual evidence validators stubbed", "record_sha256": str(index), "recorded_at_utc": at})
    monkeypatch.setattr(fresh, "_outcomes", lambda *a: rows)
    monkeypatch.setattr(cohort, "_commands", lambda *a: ("human_review", "fixture-draft"))
    monkeypatch.setattr(cohort, "_human", lambda *a: ({}, "fixture-permit"))
    monkeypatch.setattr(cohort, "_qualified", lambda root, lock, paper, row, permit: {"version": paper["version"]})
    manifest = ledger._manifest(primary.canonical_root() / "ledger")
    binding = cohort._binding(root, lock, manifest, at)
    assert [member["version"] for member in binding["members"]] == [paper["version"] for paper in lock["binding"]["candidates"][:8]]
    assert binding["planned_episode_limits"] == {"fresh_pilot": 96, "confirmation": 128}
    rows[-1]["status"] = "admitted"
    with pytest.raises(ValueError, match="after the cohort target"):
        cohort._binding(root, lock, manifest, at)
    rows[-1]["status"] = "not_processed_after_target"
    rows[0]["status"] = "not_processed_after_target"
    with pytest.raises(ValueError, match="skipped before eight"):
        cohort._binding(root, lock, manifest, at)
