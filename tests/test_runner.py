import asyncio
import copy
import json

import pytest

from benchtop import runner
from benchtop import session_ledger as ledger
from benchtop.cli import main
from tests.test_session_ledger import fixture


def run(root, prepared, index=0, **kwargs):
    return runner.run_one(root, prepared["plan_sha256"], prepared["slots"][index]["slot_id"], **kwargs)


def evidence(root, result):
    path = root / "runner" / result["attempt_id"]
    return path, json.loads((path / "evidence.json").read_text())


def test_scripted_process_calls_real_mcp_and_retains_wrong_answer(tmp_path):
    root, prepared, _ = fixture(tmp_path)
    result = run(root, prepared, answer="baseline_better")
    assert result["status"] == "completed" and result["reward"] == 0 and result["model_sessions"] == 0
    path, proof = evidence(root, result)
    sealed = json.loads((path / "sealed.json").read_text())
    assert sealed["sealed"] and sealed["trace"]["submission_source"] == "mcp"
    assert {row["tool"] for row in sealed["trace"]["events"]} == {
        "describe_lab",
        "run_experiments",
        "submit_answer",
    }
    assert sealed["summary"]["visible_order"] == prepared["binding"]["environments"][0]["visible_order"]
    assert proof["completion"]["outcome"]["costs"]["unique_executed_runs"] == 0
    replies = [
        event
        for line in (path / "stdout.jsonl").read_text().splitlines()
        if (event := json.loads(line)).get("type") == "tool_result"
    ]
    assert len(replies) == 3 and replies[-1]["response"] == {"received": True}
    assert all("seed" not in row for row in replies[1]["response"]["results"])
    for name in (
        "connection.json",
        "stdout.jsonl",
        "stderr.txt",
        "process.json",
        "sealed.json",
        "evidence.json",
    ):
        assert (path / name).stat().st_mode & 0o777 == 0o600
    assert runner.resume(root, result["attempt_id"])["outcome"]["reward"] == 0
    with pytest.raises(ValueError, match="final or requires reconciliation"):
        run(root, prepared)
    assert ledger.inspect(root)["counts"]["total"] == 1


@pytest.mark.parametrize(
    "scenario,state,reward",
    [
        ("submit", "completed", 1),
        ("no_answer", "agent_failure", 0),
        ("quota", "technical_failure", None),
        ("unknown_error", "unresolved", None),
        ("model_change", "unresolved", None),
        ("post_submit_error", "completed", 1),
    ],
)
def test_controlled_process_outcomes_use_sealed_server_result(tmp_path, scenario, state, reward):
    root, prepared, _ = fixture(tmp_path)
    result = run(root, prepared, scenario=scenario)
    assert result["status"] == state and result["reward"] == reward
    assert ledger.inspect(root)["counts"]["active"] == 0
    assert ledger.inspect(root)["halted"] == (scenario == "model_change")


@pytest.mark.parametrize(
    "scenario,state,reward", [("timeout", "agent_failure", 0), ("post_submit_timeout", "completed", 0)]
)
def test_owned_process_timeout_cannot_erase_a_committed_submission(tmp_path, scenario, state, reward):
    root, prepared, _ = fixture(tmp_path)
    result = run(root, prepared, answer="baseline_better", scenario=scenario, timeout=1)
    assert result["status"] == state and result["reward"] == reward
    path, _ = evidence(root, result)
    process = json.loads((path / "process.json").read_text())
    assert process["timed_out"] and process["process_stopped"] and process["returncode"] != 0


def test_retry_is_explicit_and_uses_a_new_episode(tmp_path):
    root, prepared, _ = fixture(tmp_path)
    first = run(root, prepared, scenario="quota")
    with pytest.raises(ValueError, match="explicit technical retry"):
        run(root, prepared)
    second = run(root, prepared, retry_technical=True)
    first_path, _ = evidence(root, first)
    second_path, _ = evidence(root, second)
    assert (
        json.loads((first_path / "connection.json").read_text())["episode"]
        != json.loads((second_path / "connection.json").read_text())["episode"]
    )
    assert ledger.inspect(root)["counts"]["retry"] == 1


def test_crash_after_evidence_publication_resumes_without_another_process(tmp_path, monkeypatch):
    root, prepared, _ = fixture(tmp_path)
    original = ledger.finish
    with monkeypatch.context() as patch:
        patch.setattr(
            ledger,
            "finish",
            lambda *a, **kw: (_ for _ in ()).throw(OSError("controlled interrupted ledger acknowledgement")),
        )
        with pytest.raises(OSError):
            run(root, prepared, answer="baseline_better")
    attempt = prepared["slots"][0]["attempt_ids"][0]
    with monkeypatch.context() as patch:
        patch.setattr(
            runner, "execute_script", lambda *a, **kw: pytest.fail("Resume spawned a second process")
        )
        result = runner.resume(root, attempt)
    assert result["outcome"]["reward"] == 0 and ledger.finish is original
    assert ledger.inspect(root)["counts"]["total"] == 1


def test_unknown_process_stop_is_not_released_or_automatically_retried(tmp_path, monkeypatch):
    root, prepared, _ = fixture(tmp_path)
    original = runner.execute_script

    def unverified(*args):
        result = original(*args)
        result["process_stopped"] = False
        return result

    monkeypatch.setattr(runner, "execute_script", unverified)
    result = run(root, prepared)
    assert result["status"] == "pending_reconciliation" and ledger.inspect(root)["counts"]["active"] == 1
    with pytest.raises(ValueError, match="not confirmed stopped"):
        runner.resume(root, result["attempt_id"])


@pytest.mark.parametrize("file", ["stdout.jsonl", "sealed.json", "connection.json"])
def test_saved_evidence_tampering_is_rejected(tmp_path, file):
    root, prepared, _ = fixture(tmp_path)
    result = run(root, prepared)
    path, _ = evidence(root, result)
    with (path / file).open("a") as out:
        out.write(" ")
    with pytest.raises(ValueError, match="evidence files changed"):
        runner.resume(root, result["attempt_id"])


def test_sealed_episode_revokes_capability_and_refuses_queued_mutations(tmp_path):
    root, prepared, _ = fixture(tmp_path)
    with runner.Controller() as controller:
        connection = controller.prepare(prepared, prepared["slots"][0])
        ep = controller.hub.get(connection["episode"])
        sealed = controller.seal(connection)
        assert sealed["summary"]["result"] is None
        assert ep.id not in controller.hub.capabilities

        async def late():
            return await controller.hub.submit(ep, "paper_better", source="mcp"), await controller.hub.run(
                ep, [{}]
            )

        values = asyncio.run_coroutine_threadsafe(late(), controller.loop).result(timeout=5)
        assert all(value["code"] == "episode_sealed" for value in values)
        assert controller.seal(connection)["summary"]["result"] is None


def test_sealing_waits_for_existing_submission_lock(tmp_path):
    _, prepared, _ = fixture(tmp_path)
    with runner.Controller() as controller:
        connection = controller.prepare(prepared, prepared["slots"][0])
        ep = controller.hub.get(connection["episode"])

        async def ordered():
            async with controller.hub.locks[ep.id]:
                submission = asyncio.create_task(controller.hub.submit(ep, "baseline_better", source="mcp"))
                seal = asyncio.create_task(controller.hub.seal(ep))
                await asyncio.sleep(0)
                assert not seal.done()
            await submission
            return await seal

        receipt = asyncio.run_coroutine_threadsafe(ordered(), controller.loop).result(timeout=5)
        assert receipt["summary"]["result"]["score"] == 0


def test_parse_stream_is_strict_about_synthetic_result_metadata():
    result = runner.parse_stream(
        "not json\n[]\n"
        + json.dumps({"type": "system", "subtype": "init", "model": "synthetic"})
        + "\n"
        + json.dumps({"type": "result", "is_error": False})
    )
    assert result == {"models": ["synthetic"], "final": None}


@pytest.mark.parametrize("defect", ["budget", "source", "truncated", "receipt"])
def test_unverified_bound_result_remains_unresolved(tmp_path, defect):
    root, prepared, _ = fixture(tmp_path)
    result = run(root, prepared)
    path, _ = evidence(root, result)
    connection, process, sealed = (
        json.loads((path / name).read_text()) for name in ("connection.json", "process.json", "sealed.json")
    )
    if defect == "budget":
        sealed["summary"]["budget"] += 1
    elif defect == "source":
        sealed["trace"]["submission_source"] = "admin"
    elif defect == "receipt":
        for event in sealed["trace"]["events"]:
            event["receipt_only"] = False
    else:
        process["stdout_truncated"] = True
    manifest = json.loads((root / "ledger.json").read_text())
    completion = runner._classify(
        prepared, prepared["slots"][0], connection, process, sealed, manifest["identity"]["protocol"]
    )
    assert completion["outcome"]["state"] == "unresolved" and completion["outcome"]["reward"] is None


def test_missing_durable_evidence_never_restarts_a_pending_attempt(tmp_path, monkeypatch):
    root, prepared, _ = fixture(tmp_path)
    start = ledger.begin(root, prepared["plan_sha256"], prepared["slots"][0]["slot_id"])
    monkeypatch.setattr(runner, "execute_script", lambda *args: pytest.fail("Pending work was relaunched"))
    with pytest.raises(FileNotFoundError):
        runner.resume(root, start["attempt_id"])
    assert ledger.inspect(root)["counts"]["active"] == 1


def test_primary_mode_cannot_reach_the_scripted_executor(tmp_path, monkeypatch):
    root, prepared, _ = fixture(tmp_path)
    altered = copy.deepcopy(prepared)
    altered["binding"]["mode"] = "study_preparation"
    (root / "plans" / f"{prepared['plan_sha256']}.json").write_text(json.dumps(altered))
    monkeypatch.setattr(runner.study, "validate_plan", lambda value: value)
    monkeypatch.setattr(runner, "execute_script", lambda *args: pytest.fail("Primary execution was reached"))
    with pytest.raises(ValueError, match="only scripted"):
        run(root, prepared)


def test_admin_seal_is_private_and_revokes_further_agent_access(server_pair):
    from tests.test_server import rpc

    control, agent, _ = server_pair
    created = control.post("/episodes", json={"env": "toy", "task": "claim"}).json()
    response = rpc(
        agent,
        created["mcp_url"],
        "tools/call",
        {"name": "submit_answer", "arguments": {"answer": "baseline_better"}},
    )
    assert json.loads(response["result"]["content"][0]["text"]) == {"received": True}
    sealed = control.post(f"/episodes/{created['id']}/seal").json()
    assert sealed["sealed"] and sealed["summary"]["result"]["score"] == 0
    assert agent.post(f"/episodes/{created['id']}/seal").status_code == 404
    assert agent.post(created["mcp_url"], json={}).status_code == 401
    refused = control.post(f"/episodes/{created['id']}/submit", json={"answer": "paper_better"}).json()
    assert refused["code"] == "episode_sealed"
    assert (
        control.post(f"/episodes/{created['id']}/seal").json()["summary"]["result"]
        == sealed["summary"]["result"]
    )


def test_cli_runs_and_resumes_without_exposing_connection(tmp_path, capsys):
    root, prepared, _ = fixture(tmp_path)
    main(
        [
            "study",
            "run-engineering",
            str(root),
            prepared["plan_sha256"],
            "--slot",
            prepared["slots"][0]["slot_id"],
            "--answer",
            "baseline_better",
        ]
    )
    text = capsys.readouterr().out
    assert '"reward": 0.0' in text or '"reward": 0' in text
    assert "mcp_url" not in text and "/mcp" not in text
    main(["study", "resume-engineering", str(root), prepared["slots"][0]["attempt_ids"][0]])
    assert '"model_sessions": 0' in capsys.readouterr().out
