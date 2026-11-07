import copy
import json
import shutil

import pytest

from benchtop import primary, provider, runner, study
from benchtop import session_ledger as ledger
from benchtop.agents.common import LabClient
from benchtop.cli import main
from tests.test_provider import encode, preflight, stream_events
from tests.test_study import model, ready_controls


def preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(primary, "STATE_BASE", tmp_path / "private-state")
    monkeypatch.setattr(primary, "BINDINGS", tmp_path / "public/bindings.json")
    monkeypatch.setattr(provider, "probe", preflight)
    reviews, foundation = ready_controls(tmp_path)
    plans = {phase: study.make_plan(model=model(), phase=phase, mode="study_preparation",
                                   reviews_dir=reviews, foundation=foundation) for phase in ("smoke", "development")}
    return plans


def authorize(plans):
    primary.authorize_smoke(plans["smoke"], preflight(), extra_usage_disabled=True)
    return primary.canonical_root() / "ledger"


def simulate_provider_transport(work, timeout, plan, proof, *, tools=None, model_id=None, final_error=False,
                                missing_init=False, no_answer=False, answer="no_meaningful_difference"):
    # test-only metadata fixture with real MCP calls, never a language model
    provider.command(work, plan, proof)
    connection = study._json(work / "connection.json")
    lab = LabClient(connection["mcp_url"])
    replies = []
    try:
        lab.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                              "clientInfo": {"name": "primary-test-fixture", "version": "1"}})
        lab.initialized()
        description = lab.call("describe_lab")
        config = {factor["name"]: factor["levels"][0] for factor in description["factors"]}
        actions = [("run_experiments", {"configs": [config], "seeds": 1})]
        if not no_answer:
            actions.append(("submit_answer", {"answer": answer, "confidence": 0.5}))
        for tool, arguments in actions:
            reply = lab.call(tool, arguments)
            assert "error" not in reply
            replies.append({"type": "fixture_tool_result", "tool": tool, "response": reply})
    finally:
        lab.http.close()
    events = stream_events(tools=tools, model_id=model_id, final_error=final_error)
    if missing_init:
        events.pop(0)
    if final_error:
        events[-1]["failure_category"] = "provider_quota"
    stdout = encode([*events, *replies])
    (work / "stdout.jsonl").write_text(stdout)
    (work / "stderr.txt").write_text("TEST FIXTURE ONLY; no Claude process.\n")
    return {"executor": "claude_cli", "process_stopped": True, "returncode": 1 if final_error else 0,
            "timed_out": False, "process_wall_seconds": 0.01, "stdout_sha256": runner._digest(work / "stdout.jsonl"),
            "stderr_sha256": runner._digest(work / "stderr.txt"), "stdout_truncated": False,
            "stream": provider.parse_stream(stdout), "model_sessions": 1}


def smoke_runs(plans, monkeypatch, **options):
    monkeypatch.setattr(provider, "execute", lambda *args: simulate_provider_transport(*args, **options))
    results = []
    with runner.Controller() as controller:
        for slot in plans["smoke"]["slots"]:
            results.append(runner.run_provider(plans["smoke"]["plan_sha256"], slot["slot_id"], controller=controller))
    return results


def test_canonical_primary_authorization_cannot_be_reinitialized_or_copied(tmp_path, monkeypatch):
    plans = preparation(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="extra usage"):
        primary.authorize_smoke(plans["smoke"], preflight())
    assert not primary.canonical_root().exists()
    root = authorize(plans)
    assert ledger.inspect(root)["scope"] == "primary" and ledger.inspect(root)["counts"]["total"] == 0
    with pytest.raises(FileExistsError):
        authorize(plans)
    copied = tmp_path / "copied-ledger"
    shutil.copytree(root, copied)
    with pytest.raises(ValueError, match="canonical journal"):
        ledger.begin(copied, plans["smoke"]["plan_sha256"], plans["smoke"]["slots"][0]["slot_id"])
    assert ledger.inspect(copied)["counts"]["total"] == 0
    with pytest.raises(ValueError, match="only; primary execution"):
        ledger.create(plans["smoke"], tmp_path / "unapproved")


def test_existing_legacy_usage_blocks_primary_and_late_legacy_start_is_refused(tmp_path, monkeypatch):
    plans = preparation(tmp_path, monkeypatch)
    legacy = primary.legacy_directory() / "starts.jsonl"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"fixture_prior_usage":true}\n')
    with pytest.raises(ValueError, match="explicit reconciliation"):
        authorize(plans)
    assert not primary.canonical_root().exists() and "fixture_prior_usage" in legacy.read_text()
    legacy.write_text("")
    authorize(plans)
    with pytest.raises(ValueError, match="separate allowance"):
        primary.block_legacy(legacy.parent)
    legacy.write_text('{"fixture_late_usage":true}\n')
    with pytest.raises(ValueError, match="unreconciled legacy"):
        primary.verify(plans["smoke"])


def test_development_lock_requires_provider_smokes_and_completed_zero_counts(tmp_path, monkeypatch):
    plans = preparation(tmp_path, monkeypatch)
    root = authorize(plans)
    with pytest.raises(ValueError, match="all six smoke"):
        primary.lock_development(plans["development"])
    assert not primary.BINDINGS.exists()
    results = smoke_runs(plans, monkeypatch, final_error=True)
    assert all(row["status"] == "completed" for row in results)
    assert any(row["reward"] == 0 for row in results)
    assert ledger.inspect(root)["counts"]["smoke"] == 6
    first_path = root / "runner" / results[0]["attempt_id"]
    first_proof = study._json(first_path / "evidence.json")
    assert first_proof["completion"]["outcome"]["costs"]["input_tokens"] == 12
    lock = primary.lock_development(plans["development"])
    assert lock["study_lock"] and lock["id"] == "development_lock"
    assert len(primary.BINDINGS.read_text()) < 2000
    assert primary.verify(plans["development"])["record_sha256"] == lock["record_sha256"]
    assert ledger.inspect(root)["registered_plans"] == 2
    assert (primary.canonical_root() / "smoke_authorization.json").exists()
    slot = plans["development"]["slots"][0]
    with runner.Controller() as controller:
        result = runner.run_provider(plans["development"]["plan_sha256"], slot["slot_id"], controller=controller)
    assert result["status"] == "completed" and ledger.inspect(root)["counts"]["primary"] == 1
    with pytest.raises(FileExistsError):
        primary.lock_development(plans["development"])
    with (first_path / "stdout.jsonl").open("a") as out:
        out.write(" ")
    with pytest.raises(ValueError, match="smoke evidence changed"):
        primary.verify(plans["development"])


@pytest.mark.parametrize("defect", ["tools", "model"])
def test_provider_contract_change_halts_canonical_journal(tmp_path, monkeypatch, defect):
    plans = preparation(tmp_path, monkeypatch)
    root = authorize(plans)
    options = {"tools": [*provider.TOOLS, "Bash"]} if defect == "tools" else {"model_id": "different-model-fixture"}
    monkeypatch.setattr(provider, "execute", lambda *args: simulate_provider_transport(*args, **options))
    with runner.Controller() as controller:
        result = runner.run_provider(plans["smoke"]["plan_sha256"], plans["smoke"]["slots"][0]["slot_id"], controller=controller)
        assert result["status"] == "unresolved" and result["reward"] is None
        assert ledger.inspect(root)["halted"] is True
        with pytest.raises(ValueError, match="ledger halted"):
            runner.run_provider(plans["smoke"]["plan_sha256"], plans["smoke"]["slots"][1]["slot_id"], controller=controller)
    assert ledger.inspect(root)["counts"]["total"] == 1


@pytest.mark.parametrize("missing_init", [False, True])
def test_unknown_real_provider_errors_never_inherit_synthetic_retry_mapping(tmp_path, monkeypatch, missing_init):
    plans = preparation(tmp_path, monkeypatch)
    root = authorize(plans)
    monkeypatch.setattr(provider, "execute", lambda *args: simulate_provider_transport(
        *args, final_error=True, no_answer=True, missing_init=missing_init))
    slot = plans["smoke"]["slots"][0]
    result = runner.run_provider(plans["smoke"]["plan_sha256"], slot["slot_id"])
    assert result["status"] == "unresolved" and result["reward"] is None
    assert ledger.inspect(root)["halted"] is False
    with pytest.raises(ValueError, match="requires reconciliation"):
        ledger.begin(root, plans["smoke"]["plan_sha256"], slot["slot_id"], retry_technical=True)


def test_legacy_integration_begin_cannot_bypass_the_primary_accounting_barrier(tmp_path, monkeypatch, server_pair):
    from benchtop import integration
    from tests.test_integration import setup_episode

    plans = preparation(tmp_path, monkeypatch)
    root = primary.legacy_directory()
    result, client, _, _ = setup_episode(server_pair, root)
    authorize(plans)
    with pytest.raises(ValueError, match="separate allowance"):
        integration.begin(client, result["episode"], client_name="fixture-ui", requested_model="fixture-model", root=root)
    assert (root / "starts.jsonl").read_text() == ""


def test_saved_primary_evidence_resumes_after_source_drift_without_new_provider_call(tmp_path, monkeypatch):
    plans = preparation(tmp_path, monkeypatch)
    root = authorize(plans)
    monkeypatch.setattr(provider, "execute", simulate_provider_transport)
    with monkeypatch.context() as patch:
        patch.setattr(ledger, "finish", lambda *a, **kw: (_ for _ in ()).throw(OSError("fixture lost acknowledgement")))
        with pytest.raises(OSError, match="lost acknowledgement"):
            runner.run_provider(plans["smoke"]["plan_sha256"], plans["smoke"]["slots"][0]["slot_id"])
    attempt = plans["smoke"]["slots"][0]["attempt_ids"][0]
    monkeypatch.setattr(provider, "execute", lambda *a: pytest.fail("resume invoked a model"))
    monkeypatch.setattr(study, "_sources", lambda: {"changed": "fixture"})
    assert runner.resume(root, attempt)["outcome"]["state"] == "completed"
    assert ledger.inspect(root)["counts"]["total"] == 1


def test_scripted_evidence_and_changed_usage_metadata_cannot_pass_primary_smoke(tmp_path, monkeypatch):
    plans = preparation(tmp_path, monkeypatch)
    authorize(plans)
    results = smoke_runs(plans, monkeypatch)
    work = primary.canonical_root() / "ledger/runner" / results[0]["attempt_id"]
    proof = study._json(work / "evidence.json")
    altered = study._seal({**proof, "runner_id": runner.RUNNER_ID}, "artifact_sha256")
    (work / "evidence.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="another attempt"):
        primary.lock_development(plans["development"])
    assert not primary.BINDINGS.exists()
    (work / "evidence.json").write_text(json.dumps(proof))
    process = study._json(work / "process.json")
    process["stream"]["final"]["modelUsage"][model()["reported_model_id"]]["inputTokens"] = 999
    (work / "process.json").write_text(json.dumps(process))
    proof["files"]["process.json"] = runner._digest(work / "process.json")
    (work / "evidence.json").write_text(json.dumps(study._seal(proof, "artifact_sha256")))
    with pytest.raises(ValueError, match="differs from retained stdout"):
        primary.lock_development(plans["development"])


def test_missing_or_stale_authorization_never_reaches_provider(tmp_path, monkeypatch):
    plans = preparation(tmp_path, monkeypatch)
    root = authorize(plans)
    monkeypatch.setattr(provider, "execute", lambda *a: pytest.fail("unauthorized inference"))
    changed = study._seal({**preflight(), "binary_sha256": "f" * 64}, "preflight_sha256")
    monkeypatch.setattr(provider, "probe", lambda: changed)
    with pytest.raises(ValueError, match="changed since authorization"):
        runner.run_provider(plans["smoke"]["plan_sha256"], plans["smoke"]["slots"][0]["slot_id"])
    assert ledger.inspect(root)["counts"]["total"] == 0
    monkeypatch.setattr(provider, "probe", preflight)
    (primary.canonical_root() / "smoke_authorization.json").unlink()
    with pytest.raises(FileNotFoundError):
        ledger.begin(root, plans["smoke"]["plan_sha256"], plans["smoke"]["slots"][0]["slot_id"])
    assert ledger.inspect(root)["counts"]["total"] == 0


def test_authorization_cli_and_missing_public_lock_fail_closed(tmp_path, monkeypatch, capsys):
    plans = preparation(tmp_path, monkeypatch)
    directory = tmp_path / "plan"
    study.save_plan(plans["smoke"], directory)
    proof = tmp_path / "preflight.json"
    proof.write_text(json.dumps(preflight()))
    main(["study", "authorize-smoke", str(directory), "--preflight", str(proof), "--confirm-extra-usage-disabled"])
    assert json.loads(capsys.readouterr().out)["study_lock"] is False
    smoke_runs(plans, monkeypatch)
    primary.lock_development(plans["development"])
    primary.BINDINGS.unlink()
    with pytest.raises(FileNotFoundError):
        primary.verify(plans["development"])


def test_unreviewed_records_cannot_create_primary_authorization(tmp_path, monkeypatch):
    plans = preparation(tmp_path, monkeypatch)
    review = next((tmp_path / "simulated-reviews").glob("*.json"))
    review.write_text("{}")
    monkeypatch.setattr(provider, "probe", lambda: pytest.fail("unreviewed plan reached provider metadata"))
    with pytest.raises(ValueError, match="study preparation blocked"):
        authorize(plans)
    assert not primary.canonical_root().exists()


def test_fresh_execution_remains_unavailable(tmp_path, monkeypatch):
    plans = preparation(tmp_path, monkeypatch)
    altered = copy.deepcopy(plans["smoke"])
    altered["binding"]["phase"] = "fresh_pilot"
    with pytest.raises(ValueError, match="fresh phase authorization"):
        primary.verify(altered)
