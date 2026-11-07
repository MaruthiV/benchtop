import copy
import json
import shutil

import pytest

from benchtop import cohort, pilots, primary, provider, runner, study, study_calibration
from benchtop import fresh_generation as fresh
from benchtop import session_ledger as ledger
from benchtop.cli import main
from tests.test_cohort import ready as generation_ready
from tests.test_primary import simulate_provider_transport
from tests.test_session_ledger import finish
from tests.test_study_calibration import export
from tests.test_study_calibration import ready as development_ready


def ready(tmp_path, monkeypatch, *, count=1):
    generation_ready(tmp_path, monkeypatch, count=count)
    locked = cohort.lock_cohort()
    return locked, pilots.make_plan(), primary.canonical_root() / "ledger"


@pytest.mark.parametrize("count", [1, 2])
def test_plan_uses_frozen_cohort_budget_grid_and_independent_replicate_identities(tmp_path, monkeypatch, count):
    locked, plan, directory = ready(tmp_path, monkeypatch, count=count)
    monkeypatch.setattr(provider, "probe", lambda: pytest.fail("planning must not query authentication"))
    assert study.validate_plan(plan) == plan
    assert plan["execution_ready"] is False and plan["model_sessions_started"] == 0
    assert plan["binding"]["cohort_lock_sha256"] == locked["record_sha256"]
    assert plan["binding"]["environments"] == locked["binding"]["members"]
    assert len(plan["groups"]) == 3 * count and len(plan["slots"]) == 12 * count
    assert {group["descriptor"]["budget_fraction"] for group in plan["groups"]} == {0.06, 0.12, 0.25}
    orders = {member["env_id"]: member["visible_order"] for member in locked["binding"]["members"]}
    assert all(group["descriptor"]["visible_order"] == orders[group["env_id"]] for group in plan["groups"])
    assert all(group["descriptor"]["phase"] == "fresh_pilot" for group in plan["groups"])
    assert len({attempt for slot in plan["slots"] for attempt in slot["attempt_ids"]}) == 24 * count
    assert {slot["replicate_index"] for slot in plan["slots"]} == set(range(4))
    assert ledger.inspect(directory)["counts"]["total"] == 0
    assert not (directory.parent / "pilot_authorization.json").exists()
    study.save_plan(plan, tmp_path / "preparation")
    assert study.load_plan(tmp_path / "preparation") == plan


@pytest.mark.parametrize("kind", ["model", "budget", "visible_order", "member", "attempt", "slot_removed", "show_paper"])
def test_resealed_overrides_cannot_change_pilot_plan(tmp_path, monkeypatch, kind):
    _, plan, _ = ready(tmp_path, monkeypatch)
    altered = copy.deepcopy(plan)
    if kind == "model":
        altered["binding"]["model_configuration"]["requested_model_id"] = "different-pinned-model"
    elif kind == "budget":
        altered["slots"][0]["budget_fraction"] = 0.9
    elif kind == "visible_order":
        altered["binding"]["environments"][0]["visible_order"].reverse()
    elif kind == "member":
        altered["binding"]["environments"][0]["source"] = study._ref(tmp_path / "substitute")
    elif kind == "attempt":
        altered["slots"][0]["attempt_ids"][0] = "1" * 64
    elif kind == "slot_removed":
        altered["slots"].pop()
    else:
        altered["groups"][0]["descriptor"]["show_paper"] = False
    altered["bindings_hash"] = study.canonical_hash(altered["binding"])
    altered = study._seal(altered, "plan_sha256")
    with pytest.raises(ValueError, match="fresh pilot plan"):
        study.validate_plan(altered)


def test_authorization_preserves_canonical_allowance_and_does_not_append_a_protocol_lock(tmp_path, monkeypatch):
    locked, plan, directory = ready(tmp_path, monkeypatch)
    before = primary.BINDINGS.read_bytes()
    identity = ledger._manifest(directory)["ledger_id"]
    monkeypatch.setattr(provider, "execute", lambda *a: pytest.fail("authorization launched a model"))
    record = pilots.authorize(plan)
    assert record["study_lock"] is False and record["planned_slots"] == 12 and record["model_sessions_started"] == 0
    assert record["cohort_lock_sha256"] == locked["record_sha256"] and record["ledger_id"] == identity
    assert ledger.inspect(directory)["counts"]["total"] == 0 and primary.BINDINGS.read_bytes() == before
    assert plan["plan_sha256"] in ledger._read(directory, ledger._manifest(directory))["plans"]
    monkeypatch.setattr(provider, "probe", lambda: pytest.fail("inspection queried live authentication"))
    assert primary.verify(plan, live=False) == record and cohort.verify() == locked
    with pytest.raises(FileExistsError, match="already authorized"):
        pilots.authorize(plan)


def test_real_mcp_pilots_keep_qualification_truth_private_and_retain_wrong_zero(tmp_path, monkeypatch):
    locked, plan, directory = ready(tmp_path, monkeypatch)
    pilots.authorize(plan)
    before = {path: study.cache.file_hash(study._resolve(locked["binding"]["members"][0]["source"]) / path)
              for path in locked["binding"]["members"][0]["files"]}
    monkeypatch.setattr(provider, "execute", lambda *args: simulate_provider_transport(*args, answer="baseline_better"))
    with runner.Controller() as controller:
        first = runner.run_provider(plan["plan_sha256"], plan["slots"][0]["slot_id"], controller=controller)
        assert first["status"] == "completed" and first["reward"] == 0
        monkeypatch.setattr(provider, "execute", lambda *args: simulate_provider_transport(*args, answer="paper_better"))
        second = runner.run_provider(plan["plan_sha256"], plan["slots"][1]["slot_id"], controller=controller)
        assert second["status"] == "completed" and second["reward"] == 1
    for result in (first, second):
        work = directory / "runner" / result["attempt_id"]
        connection = study._json(work / "connection.json")
        public = json.dumps(connection["description"])
        assert not any(key in public for key in ("qualification_label", "seed_splits", "seed_ids", "results.jsonl"))
        stdout = (work / "stdout.jsonl").read_text()
        assert '"score"' not in stdout and '"correct"' not in stdout
        sealed = study._json(work / "sealed.json")
        assert sealed["summary"]["visible_order"] == locked["binding"]["members"][0]["visible_order"]
        assert sealed["summary"]["task_version"] == locked["binding"]["members"][0]["task_version"]
        assert sealed["summary"]["charging_id"] == locked["binding"]["members"][0]["charging_hash"]
        assert sealed["trace"]["submission_source"] == "mcp"
    assert cohort.verify() == locked
    assert before == {path: study.cache.file_hash(study._resolve(locked["binding"]["members"][0]["source"]) / path) for path in before}
    state = ledger.inspect(directory)
    assert state["counts"]["total"] == state["counts"]["primary"] == state["counts"]["phase:fresh_pilot"] == 2
    monkeypatch.setattr(provider, "execute", lambda *a: pytest.fail("resume relaunched a model"))
    assert runner.resume(directory, first["attempt_id"])["outcome"]["reward"] == 0
    assert ledger.inspect(directory)["counts"]["total"] == 2


def test_exiting_without_an_answer_is_a_fresh_pilot_zero(tmp_path, monkeypatch):
    _, plan, directory = ready(tmp_path, monkeypatch)
    pilots.authorize(plan)
    monkeypatch.setattr(provider, "execute", lambda *args: simulate_provider_transport(*args, no_answer=True))
    result = runner.run_provider(plan["plan_sha256"], plan["slots"][0]["slot_id"])
    assert result["status"] == "agent_failure" and result["reward"] == 0
    assert ledger.inspect(directory)["counts"]["active"] == 0


def test_primary_pilot_starts_require_separate_authorization_even_for_manually_published_plan(tmp_path, monkeypatch):
    _, plan, directory = ready(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        ledger.register(plan, directory, authorization=directory.parent)
    study._publish(directory / "plans" / f"{plan['plan_sha256']}.json", plan)
    with pytest.raises(FileNotFoundError):
        ledger.begin(directory, plan["plan_sha256"], plan["slots"][0]["slot_id"])
    assert ledger.inspect(directory)["counts"]["total"] == 0
    with pytest.raises(ValueError, match="only scripted"):
        runner.run_one(directory, plan["plan_sha256"], plan["slots"][0]["slot_id"])


def test_two_pending_pilots_occupy_shared_capacity_and_copies_cannot_reset_it(tmp_path, monkeypatch):
    _, plan, directory = ready(tmp_path, monkeypatch)
    pilots.authorize(plan)
    for slot in plan["slots"][:2]:
        ledger.begin(directory, plan["plan_sha256"], slot["slot_id"])
    with pytest.raises(ValueError, match="two in-flight"):
        ledger.begin(directory, plan["plan_sha256"], plan["slots"][2]["slot_id"])
    copied = tmp_path / "copy"
    shutil.copytree(directory, copied)
    with pytest.raises(ValueError, match="canonical journal"):
        ledger.begin(copied, plan["plan_sha256"], plan["slots"][2]["slot_id"])
    assert ledger.inspect(directory)["counts"]["total"] == 2


@pytest.mark.parametrize("marker", ["pilot_collection.json", "confirmation_lock.json"])
def test_later_phase_markers_block_new_pilot_starts_and_outcome_changes(tmp_path, monkeypatch, marker):
    _, plan, directory = ready(tmp_path, monkeypatch)
    pilots.authorize(plan)
    start = ledger.begin(directory, plan["plan_sha256"], plan["slots"][0]["slot_id"])
    (directory.parent / marker).write_text('{"fixture": "future collection closure"}')
    with pytest.raises(ValueError, match="pilot collection is closed"):
        ledger.begin(directory, plan["plan_sha256"], plan["slots"][1]["slot_id"])
    with pytest.raises(ValueError, match="pilot collection is closed"):
        finish(directory, start)
    assert ledger.inspect(directory)["counts"]["active"] == 1


def test_zero_cohort_has_no_pilot_authorization_or_primary_comparison(tmp_path, monkeypatch):
    generation_ready(tmp_path, monkeypatch, effect="0.05 + (0.3 if seed % 2 else -0.3)")
    cohort.lock_cohort()
    plan = pilots.make_plan()
    assert plan["slots"] == plan["groups"] == [] and study.validate_plan(plan) == plan
    with pytest.raises(ValueError, match="zero labs imply no primary comparison"):
        pilots.authorize(plan)
    assert not (primary.canonical_root() / "pilot_authorization.json").exists()


def test_generic_preparation_still_refuses_fresh_sources(tmp_path, monkeypatch):
    locked, _, _ = ready(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="historical references or controlled two-part"):
        study._snapshot(study._resolve(locked["binding"]["members"][0]["source"]))


def test_closed_development_verification_reuses_the_callers_lock_and_rejects_foreign_context(tmp_path, monkeypatch):
    directory, development = development_ready(tmp_path, monkeypatch)
    frozen = export(development, tmp_path, close=True)
    with ledger._locked(directory) as context:
        monkeypatch.setattr(ledger, "_locked", lambda *a: pytest.fail("attempted nested flock instead of using held context"))
        assert study_calibration.load_closed(ledger_context=context) == frozen
        with pytest.raises(ValueError, match="context differs"):
            study_calibration.load_closed(ledger_context=(tmp_path / "foreign", context[1]))


def test_missing_cohort_creates_no_state_or_model_activity(tmp_path, monkeypatch):
    monkeypatch.setattr(primary, "STATE_BASE", tmp_path / "private")
    monkeypatch.setattr(primary, "BINDINGS", tmp_path / "public.json")
    monkeypatch.setattr(provider, "execute", lambda *a: pytest.fail("missing cohort launched a model"))
    with pytest.raises(FileNotFoundError):
        pilots.make_plan()
    assert not primary.STATE_BASE.exists() and not primary.BINDINGS.exists()


def test_cli_prepares_and_authorizes_without_model_or_membership_override(tmp_path, monkeypatch, capsys):
    ready(tmp_path, monkeypatch)
    monkeypatch.setattr(provider, "execute", lambda *a: pytest.fail("CLI setup invoked a model"))
    output = tmp_path / "preparation"
    main(["study", "plan-pilots", "--output-dir", str(output)])
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["planned_slots"] == 12 and prepared["execution_ready"] is False
    main(["study", "authorize-pilots", str(output)])
    authorized = json.loads(capsys.readouterr().out)
    assert authorized["planned_slots"] == 12 and authorized["model_sessions_started"] == 0
    monkeypatch.setattr(provider, "probe", lambda: pytest.fail("inspection queried authentication"))
    main(["study", "inspect-pilots"])
    assert json.loads(capsys.readouterr().out) == authorized
    with pytest.raises(SystemExit):
        main(["study", "plan-pilots", "--output-dir", str(output), "--model", "override"])


@pytest.mark.parametrize("contract", ["model", "tools"])
def test_changed_provider_contract_halts_fresh_starts_without_replacing_the_lab(tmp_path, monkeypatch, contract):
    locked, plan, directory = ready(tmp_path, monkeypatch)
    pilots.authorize(plan)
    options = {"model_id": "changed-resolved-model"} if contract == "model" else {"tools": ["Bash"]}
    monkeypatch.setattr(provider, "execute", lambda *args: simulate_provider_transport(*args, **options))
    result = runner.run_provider(plan["plan_sha256"], plan["slots"][0]["slot_id"])
    assert result["status"] == "unresolved" and result["reward"] is None
    assert ledger.inspect(directory)["halted"] is True and cohort.verify() == locked
    with pytest.raises(ValueError, match="ledger halted"):
        ledger.begin(directory, plan["plan_sha256"], plan["slots"][1]["slot_id"])
    assert ledger.inspect(directory)["counts"]["total"] == 1


def test_changed_account_blocks_a_provider_start_before_consuming_allowance(tmp_path, monkeypatch):
    _, plan, directory = ready(tmp_path, monkeypatch)
    pilots.authorize(plan)
    changed = study._json(directory.parent / "provider.json")
    changed["account_sha256"] = "3" * 64
    changed = study._seal(changed, "preflight_sha256")
    monkeypatch.setattr(provider, "probe", lambda: changed)
    monkeypatch.setattr(provider, "execute", lambda *a: pytest.fail("changed account launched a model"))
    with pytest.raises(ValueError, match="changed since authorization"):
        runner.run_provider(plan["plan_sha256"], plan["slots"][0]["slot_id"])
    assert ledger.inspect(directory)["counts"]["total"] == 0


def test_controller_loader_rejects_confirmation_rows_before_loading_a_task(tmp_path, monkeypatch):
    locked, _, _ = ready(tmp_path, monkeypatch)
    member = locked["binding"]["members"][0]
    result = study._resolve(member["source"]) / "results.jsonl"
    row = json.loads(result.read_text().splitlines()[0])
    row["seed"] = 16
    with result.open("a") as output:
        output.write(json.dumps(row) + "\n")
    monkeypatch.setattr(pilots.Env, "load", lambda *a, **kw: pytest.fail("confirmation data reached task loading"))
    monkeypatch.setattr(fresh, "sweep", lambda *a, **kw: pytest.fail("pilot loader ran a sweep"))
    with pytest.raises(ValueError, match="confirmation or undeclared seed"):
        pilots.load_environment(member)
