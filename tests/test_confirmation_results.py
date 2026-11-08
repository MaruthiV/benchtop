import copy
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from benchtop import (
    candidates,
    cohort,
    confirmation,
    confirmation_data,
    confirmation_execution,
    pilots,
    primary,
    provider,
    runner,
    study,
)
from benchtop import confirmation_analysis as analysis
from benchtop import confirmation_replay as replay
from benchtop import confirmation_results as results
from benchtop import fresh_generation as fresh
from benchtop import generation_journal as generation
from benchtop import pilot_calibration as bridge
from benchtop import session_ledger as ledger
from benchtop.agents.estimator import ESTIMATOR_ID
from benchtop.cli import main
from tests.test_fresh_generation import Content, accept, lab_reply, process, setup
from tests.test_primary import simulate_provider_transport
from tests.test_session_ledger import finish


def ready(tmp_path, monkeypatch, *, effects=(0.2,)):
    candidate, _ = setup(tmp_path, monkeypatch, count=len(effects))
    spec = study._json(study.SPEC)
    candidate["binding"]["baselines"] = {"policies": spec["baselines"]["required"] + spec["baselines"]["sanity_controls"],
        "estimator_id": ESTIMATOR_ID, "development_majority_label": "paper_better", "adaptive_allocation": "deferred"}
    candidate = study._seal(candidate, "record_sha256")
    (primary.canonical_root() / "candidate_lock.json").write_text(json.dumps(candidate))
    monkeypatch.setattr(candidates, "verify", lambda **kwargs: copy.deepcopy(candidate))
    original = provider.execute_text

    def execute(work, *args):
        value = original(work, *args)
        start = study._json(work / "start.json")
        index = next(index for index, paper in enumerate(candidate["binding"]["candidates"]) if paper["version"] == start["version"])
        return value if start["stage"] == "triage" else process(work, lab_reply(candidate["binding"]["candidates"][index], effect=effects[index]))

    monkeypatch.setattr(provider, "execute_text", execute)
    for paper in candidate["binding"]["candidates"]:
        fresh.step(client=Content())
        fresh.review(accept(generation.directory(), paper["version"], margin=0.05))
        fresh.qualify()
    locked = cohort.lock_cohort()
    if locked["binding"]["members"]:
        pilot = pilots.make_plan()
        pilots.authorize(pilot)
        bridge.export_pilots(pilot["plan_sha256"], tmp_path / "pilot.json", close=True)
    plan = confirmation.make_plan()
    confirmation.lock_confirmation(plan)
    confirmation_data.collect()
    confirmation_execution.authorize(plan)
    return primary.canonical_root() / "ledger", plan


def collect(directory, plan, tmp_path, *, close=False, name="confirmation.json"):
    return results.collect(plan["plan_sha256"], tmp_path / name, close=close)


def no_models(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline confirmation work invoked a model or live authentication")

    for name in ("execute", "execute_text", "probe"):
        monkeypatch.setattr(provider, name, forbidden)


def test_open_collection_retains_all_slots_pending_work_and_unknown_costs_without_models(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch)
    first = ledger.begin(directory, plan["plan_sha256"], plan["slots"][0]["slot_id"])
    no_models(monkeypatch)
    record = collect(directory, plan, tmp_path)
    assert record["closed"] is False and len(record["slots"]) == 16
    assert record["slots"][0]["attempts"][0]["attempt_id"] == first["attempt_id"]
    assert record["slots"][0]["attempts"][0]["state"] == "unresolved"
    assert record["confirmation_costs"]["fields"]["input_tokens"]["total"] is None
    assert record["model_sessions_started_by_export"] == 0
    assert analysis.analyse(record)["primary"]["estimate"] is None
    with pytest.raises(ValueError, match="in-flight"):
        collect(directory, plan, tmp_path, close=True, name="cannot-close.json")
    assert not (directory.parent / results.COLLECTION).exists()


def test_closed_empty_collection_freezes_missing_slots_and_copies_are_stable_after_other_phase_growth(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch)
    probe = provider.probe
    no_models(monkeypatch)
    record = collect(directory, plan, tmp_path, close=True)
    assert record["closed"] and len(record["slots"]) == 16
    before = (directory.parent / results.COLLECTION).read_bytes()
    with ledger._locked(directory) as context:
        assert results.load_closed(ledger_context=context) == record
    with pytest.raises(ValueError, match="collection is closed"):
        ledger.begin(directory, plan["plan_sha256"], plan["slots"][0]["slot_id"])
    monkeypatch.setattr(provider, "probe", probe)
    smoke = study._json(directory.parent / "smoke_plan.json")
    started = ledger.begin(directory, smoke["plan_sha256"], smoke["slots"][0]["slot_id"])
    finish(directory, started, reward=0)
    no_models(monkeypatch)
    assert results.load_closed() == record
    assert collect(directory, plan, tmp_path, name="again.json") == record
    assert (directory.parent / results.COLLECTION).read_bytes() == before
    assert (tmp_path / "again.json").read_bytes() == before


def test_opaque_finished_attempt_is_not_accepted_as_provider_evidence(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch)
    start = ledger.begin(directory, plan["plan_sha256"], plan["slots"][0]["slot_id"])
    finish(directory, start, reward=0)
    no_models(monkeypatch)
    with pytest.raises(ValueError, match="replay-verifiable"):
        collect(directory, plan, tmp_path, close=True)
    assert not (directory.parent / results.COLLECTION).exists()


def test_full_two_lab_pipeline_replays_200_seeds_per_policy_once_and_exports_private_free_reports(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch, effects=(0.2, 0.2))
    ids = [member["env_id"] for member in plan["binding"]["environments"]]
    slots = {slot["slot_id"]: slot for slot in plan["slots"]}

    def simulated(work, *args):
        slot = slots[study._json(work / "connection.json")["slot_id"]]
        alternating = slot["arm"] == ("calibrated_budget" if slot["env_id"] == ids[0] else "global_budget")
        correct = not alternating or slot["replicate_index"] % 2 == 0
        return simulate_provider_transport(work, *args, answer="paper_better" if correct else "baseline_better")

    monkeypatch.setattr(provider, "execute", simulated)
    with runner.Controller() as controller:
        for slot in plan["slots"]:
            assert runner.run_provider(plan["plan_sha256"], slot["slot_id"], controller=controller)["status"] == "completed"
    no_models(monkeypatch)
    record = collect(directory, plan, tmp_path, close=True)
    assert record["confirmation_costs"]["episode_starts"] == 32
    assert len(record["slots"]) == 32 and len(record["members"]) == 2
    seen, grading = [], []
    play = replay.play_release

    def observed(ep, policy, seed, **kwargs):
        seen.append((ep.env.id, policy, seed))
        grading.append(ep._grading_task["seed_ids"])
        assert ep.order == next(member["confirmation_visible_order"] for member in plan["binding"]["environments"] if member["env_id"] == ep.env.id)
        assert ep.task["seed_ids"] == list(range(4, 16))
        return play(ep, policy, seed, **kwargs)

    monkeypatch.setattr(replay, "play_release", observed)
    publish = study._publish
    failures = {"finish", "summary"}

    def lose_ack(path, value):
        publish(path, value)
        kind = "finish" if path.name.endswith(".finish.json") and path.parent.name == "attempts" else "summary" if path.name == "summary.json" else None
        if kind in failures:
            failures.remove(kind)
            raise OSError("fixture lost " + kind + " acknowledgement")

    monkeypatch.setattr(study, "_publish", lose_ack)
    with pytest.raises(OSError, match="lost finish"):
        replay.run()
    assert len(seen) == 1 and replay.inspect()["status"] == "pending"
    with pytest.raises(OSError, match="lost summary"):
        replay.run()
    result = replay.run()
    assert result == replay.inspect() and result["status"] == "complete"
    assert len(seen) == len(set(seen)) == 2 * 6 * 200
    assert all(seeds == list(range(16, 28)) for seeds in grading)
    assert result["model_sessions_started"] == result["actual_environment_executions"] == result["actual_environment_cpu_seconds"] == 0
    assert all(group["planned"] == len(group["outcomes"]) == 200 for group in result["groups"])
    assert all(group["accuracy"] == 1 for group in result["groups"] if group["policy"] in ("paper_prior_only", "development_majority_label"))
    report = analysis.export(tmp_path / "analysis.json", report=tmp_path / "report.md")
    assert report["primary"]["estimate"] == 0 and report["primary"]["interval"] == [-1, 1]
    prior = next(row for row in report["classical"]["comparisons"] if row["policy"] == "paper_prior_only")
    assert prior["estimate"] == -0.25 and prior["labs"] == 2
    assert report["reporting_status"] == "complete"
    assert "Benchtop confirmation analysis" in (tmp_path / "report.md").read_text()
    public = json.dumps(report)
    assert str(tmp_path) not in public and "mcp_url" not in public and "grading_contexts" not in public
    assert ledger.inspect(directory)["counts"]["total"] == 32
    assert results.load_closed() == record
    attempt = next(row for row in record["slots"] if row["attempts"])["attempts"][0]["attempt_id"]
    assert runner.resume(directory, attempt)["outcome"]["state"] == "completed"


def test_interrupted_classical_start_never_automatically_reexecutes_or_discards_intent(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch)
    collect(directory, plan, tmp_path, close=True)
    no_models(monkeypatch)
    publish = study._publish

    def interrupt(path, value):
        publish(path, value)
        if path.name.endswith(".start.json"):
            raise OSError("fixture interrupted after intent")

    monkeypatch.setattr(study, "_publish", interrupt)
    with pytest.raises(OSError, match="after intent"):
        replay.run()
    snapshot = replay.inspect()
    assert snapshot["status"] == "pending"
    assert snapshot["groups"][0]["outcomes"][0]["state"] == "pending_reconciliation"
    monkeypatch.setattr(study, "_publish", publish)
    monkeypatch.setattr(replay, "play_release", lambda *a, **kw: pytest.fail("interrupted replay reran"))
    with pytest.raises(ValueError, match="requires reconciliation"):
        replay.run()
    assert ledger.inspect(directory)["counts"]["total"] == 0


@pytest.mark.parametrize("field", ["accounting", "slot", "evidence", "stability", "time"])
def test_resealed_closed_collection_changes_fail_verification(tmp_path, monkeypatch, field):
    directory, plan = ready(tmp_path, monkeypatch)
    record = collect(directory, plan, tmp_path, close=True)
    if field == "accounting":
        record["accounting"]["shared_ledger_costs_at_snapshot"]["fields"]["input_tokens"]["known_sum"] += 1
    elif field == "slot":
        record["slots"].pop()
    elif field == "evidence":
        record["phase_evidence"]["ledger.json"] = "0" * 64
    elif field == "stability":
        record["members"][0]["gradable"] = False
    else:
        record["recorded_at_utc"] = "2000-01-01T00:00:00+00:00"
    (directory.parent / results.COLLECTION).write_text(json.dumps(study._seal(record, "snapshot_sha256")))
    no_models(monkeypatch)
    with pytest.raises(ValueError):
        results.load_closed()


def test_lost_collection_output_acknowledgement_recovers_identical_closed_snapshot(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch)
    publish = study._publish
    output = tmp_path / "lost.json"

    def lost(path, value):
        if path == output:
            raise OSError("fixture missing output acknowledgement")
        return publish(path, value)

    monkeypatch.setattr(study, "_publish", lost)
    with pytest.raises(OSError, match="missing output"):
        collect(directory, plan, tmp_path, close=True, name="lost.json")
    before = (directory.parent / results.COLLECTION).read_bytes()
    monkeypatch.setattr(study, "_publish", publish)
    no_models(monkeypatch)
    collect(directory, plan, tmp_path, name="recovered.json")
    assert (tmp_path / "recovered.json").read_bytes() == before


def test_closure_and_start_serialize_under_one_journal_lock(tmp_path, monkeypatch):
    directory, plan = ready(tmp_path, monkeypatch)
    barrier = Barrier(2)

    def closing():
        barrier.wait()
        try:
            collect(directory, plan, tmp_path, close=True)
            return "closed"
        except ValueError as error:
            assert "in-flight" in str(error)
            return "pending"

    def starting():
        barrier.wait()
        try:
            ledger.begin(directory, plan["plan_sha256"], plan["slots"][0]["slot_id"])
            return "started"
        except ValueError as error:
            assert "collection is closed" in str(error)
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as pool:
        closed, started = pool.submit(closing), pool.submit(starting)
        assert (closed.result(timeout=30), started.result(timeout=30)) in (("closed", "blocked"), ("pending", "started"))


@pytest.mark.parametrize("effect", ["(0.2 if seed < 16 else -0.2)", "0.05 + (0.3 if seed % 2 else -0.3)"])
def test_unstable_and_zero_cohorts_do_not_run_classical_policies_or_remove_members(tmp_path, monkeypatch, effect):
    directory, plan = ready(tmp_path, monkeypatch, effects=(effect,))
    record = collect(directory, plan, tmp_path, close=True)
    no_models(monkeypatch)
    monkeypatch.setattr(replay, "play_release", lambda *a, **kw: pytest.fail("unavailable task replayed"))
    summary = replay.run()
    report = analysis.analyse(record, summary)
    assert report["primary"]["estimate"] is None
    assert report["denominators"]["locked_labs"] == len(plan["binding"]["environments"])
    assert not (directory.parent / replay.DIRECTORY).exists()
    assert ledger.inspect(directory)["counts"]["total"] == 0


def test_missing_or_wrong_phase_and_invalid_destinations_cannot_close_collection(tmp_path, monkeypatch):
    monkeypatch.setattr(primary, "STATE_BASE", tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        results.collect("0" * 64, tmp_path / "no.json", close=True)
    assert not primary.STATE_BASE.exists()
    directory, plan = ready(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="only confirmation"):
        results.collect(study._json(directory.parent / "smoke_plan.json")["plan_sha256"], tmp_path / "wrong.json", close=True)
    with pytest.raises(ValueError, match="outside the canonical"):
        results.collect(plan["plan_sha256"], directory.parent / "forbidden.json", close=True)
    target = tmp_path / "existing.json"
    target.write_text("keep")
    with pytest.raises(FileExistsError):
        results.collect(plan["plan_sha256"], target, close=True)
    assert target.read_text() == "keep" and not (directory.parent / results.COLLECTION).exists()


def test_cli_collection_and_analysis_print_counts_without_truth_or_private_paths(tmp_path, monkeypatch, capsys):
    directory, plan = ready(tmp_path, monkeypatch)
    no_models(monkeypatch)
    main(["study", "collect-confirmation", plan["plan_sha256"], "--output", str(tmp_path / "collection.json"), "--close"])
    assert json.loads(capsys.readouterr().out)["closed"]
    main(["study", "inspect-confirmation-replay"])
    assert json.loads(capsys.readouterr().out)["status"] == "not_run"
    main(["study", "analyse-confirmation", "--output", str(tmp_path / "analysis.json"), "--report", str(tmp_path / "report.md")])
    printed = capsys.readouterr().out
    assert json.loads(printed)["denominators"]["locked_labs"] == 1
    assert str(directory) not in printed and "confirmation_label" not in printed and "qualification_label" not in printed
    assert (tmp_path / "report.md").exists()
