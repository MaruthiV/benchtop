import copy
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from benchtop import generation_journal as generation
from benchtop import integration, primary, provider, qualia, runner, study
from benchtop import session_ledger as ledger
from benchtop.cli import main
from tests.test_integration import connections
from tests.test_primary import authorize, preparation
from tests.test_server import rpc
from tests.test_session_ledger import finish
from tests.test_study import model


def ready(tmp_path, monkeypatch, server_pair, *, prepare=True):
    plans = preparation(tmp_path, monkeypatch)
    directory = authorize(plans)
    qualia.authorize()
    plan = study._json(directory.parent / "qualia_plan.json")
    client, lab, hub = connections(server_pair)
    prepared = qualia.prepare(client, plan["slots"][0]["slot_id"], lab=lab) if prepare else None
    return directory, plan, client, lab, hub, prepared


def begin(client, episode, **kwargs):
    return qualia.begin(client, episode, client_name="qualia-desktop", selected_model_label="synthetic UI label",
                        restrictions="test fixture only; no actual UI observation", no_new_spend=True, **kwargs)


def run_tools(lab, hub, episode, *, wrong=False):
    ep = hub.get(episode)
    config = ep.env.space.configs()[0]
    truth = ep.task["answer"]
    answer = next(label for label in ("paper_better", "baseline_better", "no_meaningful_difference") if label != truth) if wrong else truth
    for name, arguments in (("describe_lab", {}), ("run_experiments", {"configs": [config], "seeds": 1}),
                            ("submit_answer", {"answer": answer, "confidence": 0.5})):
        response = rpc(lab.http, hub.agent_url(ep), "tools/call", {"name": name, "arguments": arguments})
        value = json.loads(response["result"]["content"][0]["text"])
        assert "error" not in value and not any(key in value for key in ("truth", "score", "graded"))


def no_provider(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Qualia accounting invoked a model or CLI authentication probe")

    for name in ("probe", "execute", "execute_text"):
        monkeypatch.setattr(provider, name, forbidden)


def test_authorization_uses_original_journal_and_two_slots_without_models_or_new_lock(tmp_path, monkeypatch, server_pair):
    directory, plan, _, _, _, _ = ready(tmp_path, monkeypatch, server_pair, prepare=False)
    no_provider(monkeypatch)
    assert qualia.authorize()["plan_sha256"] == plan["plan_sha256"]
    assert len(plan["slots"]) == len(plan["groups"]) == 2
    assert all(slot["budget_fraction"] == 0.25 for slot in plan["slots"])
    smoke = study._json(directory.parent / "smoke_plan.json")
    assert all(plan["binding"][key] == smoke["binding"][key] for key in ("model_configuration", "prompt", "source_sha256", "runtime"))
    assert plan["binding"]["client_prompt"]["scope"].startswith("Operator-assisted connectivity only")
    assert qualia.inspect()["counts"]["total"] == 0
    assert not primary.BINDINGS.exists()
    with ledger._locked(directory) as context:
        assert primary.verify(plan, live=False, ledger_context=context)["planned_slots"] == 2
    with pytest.raises(ValueError, match="operator-assisted UI path"):
        runner.run_provider(plan["plan_sha256"], plan["slots"][0]["slot_id"])


def test_preparation_is_private_inactive_and_admin_or_mcp_cannot_spend_before_intent(tmp_path, monkeypatch, server_pair):
    directory, plan, client, lab, hub, prepared = ready(tmp_path, monkeypatch, server_pair)
    no_provider(monkeypatch)
    episode = prepared["episode"]
    saved = study._json(prepared["private_connection_file"])
    assert prepared["status"] == "prepared_inactive" and prepared["model_sessions_started"] == 0
    assert saved["mcp_url"] not in json.dumps(prepared) and hub.admin_token not in json.dumps(saved)
    assert (qualia._root() / "connections" / (episode + ".json")).stat().st_mode & 0o777 == 0o600
    assert client.run(episode, [hub.get(episode).env.space.configs()[0]])["code"] == "integration_inactive"
    assert client.submit(episode, "paper_better")["code"] == "integration_inactive"
    reply = rpc(lab.http, saved["mcp_url"], "tools/call", {"name": "run_experiments", "arguments": {"configs": [hub.get(episode).env.space.configs()[0]]}})
    assert "integration_inactive" in str(reply)
    assert client.http.post(f"/episodes/{episode}/qualia-activate").status_code == 400
    summary = client.summary(episode)
    assert summary["spent"] == summary["calls"] == 0 and summary["result"] is None
    assert ledger.inspect(directory)["counts"]["active"] == 0
    with pytest.raises(ValueError, match="prepared connection"):
        ledger.begin(directory, plan["plan_sha256"], plan["slots"][0]["slot_id"])


@pytest.mark.parametrize("reported", [None, "matched"])
def test_observed_mcp_capture_is_connectivity_only_and_unknown_model_never_becomes_usable_reward(tmp_path, monkeypatch, server_pair, reported):
    directory, plan, client, lab, hub, prepared = ready(tmp_path, monkeypatch, server_pair)
    no_provider(monkeypatch)
    episode = prepared["episode"]
    model_id = model()["reported_model_id"] if reported else None
    started = begin(client, episode, reported_model=model_id)
    assert started["status"] == "reserved_active"
    assert ledger.inspect(directory)["counts"]["integration"] == ledger.inspect(directory)["counts"]["active"] == 1
    run_tools(lab, hub, episode, wrong=True)
    with pytest.raises(ValueError, match="conversation stopped"):
        qualia.collect(client, episode, client_observed=True)
    result = qualia.collect(client, episode, client_observed=True, client_stopped=True, reported_model=model_id, output=tmp_path / "observed.json")
    assert result["qualia_verified"] and result["transport_complete"]
    assert result["scope"] == "qualia_connectivity_only_excluded_from_scientific_results"
    assert result["outcome"]["reward"] == (0 if model_id else None)
    assert result["outcome"]["state"] == ("completed" if model_id else "unresolved")
    assert result["reported_model_source"] == ("operator_observation_not_authenticated" if model_id else "unavailable")
    assert result["stop_source"] == "operator_attestation_not_process_inspection"
    assert hub.get(episode).result["confidence"] == 0.5
    assert episode not in hub.capabilities
    assert ledger.inspect(directory)["counts"]["total"] == ledger.inspect(directory)["counts"]["integration"] == 1
    assert ledger.inspect(directory)["counts"]["active"] == ledger.inspect(directory)["counts"]["primary"] == 0
    assert qualia.resume(episode)["outcome"] == result["outcome"]
    assert "mcp_url" not in (tmp_path / "observed.json").read_text()
    assert str(directory) not in (tmp_path / "observed.json").read_text()
    with pytest.raises(ValueError, match="never resend"):
        begin(client, episode, reported_model=model_id)


@pytest.mark.parametrize("defect", ["admin", "truncated", "missing_tool", "unobserved"])
def test_preflight_admin_activity_truncated_trace_and_unobserved_client_do_not_pass(tmp_path, monkeypatch, server_pair, defect):
    _, _, client, lab, hub, prepared = ready(tmp_path, monkeypatch, server_pair)
    episode = prepared["episode"]
    begin(client, episode)
    if defect == "admin":
        client.run(episode, [hub.get(episode).env.space.configs()[0]])
        client.submit(episode, "paper_better")
    elif defect == "missing_tool":
        rpc(lab.http, hub.agent_url(hub.get(episode)), "tools/call", {"name": "submit_answer", "arguments": {"answer": "paper_better"}})
    else:
        run_tools(lab, hub, episode)
        if defect == "truncated":
            for _ in range(280):
                hub.record_mcp(hub.get(episode), "describe_lab", {})
    result = qualia.collect(client, episode, client_observed=defect != "unobserved", client_stopped=True)
    assert not result["qualia_verified"]
    assert result["transport_complete"] == (defect == "unobserved")


@pytest.mark.parametrize("defect", ["billing", "cloud", "model", "empty_controls"])
def test_invalid_ui_request_is_refused_before_consuming_an_intent(tmp_path, monkeypatch, server_pair, defect):
    directory, _, client, _, _, prepared = ready(tmp_path, monkeypatch, server_pair)
    options = {"client_name": "qualia-desktop", "selected_model_label": "fixture", "no_new_spend": True, "restrictions": "unavailable in this fixture"}
    if defect == "billing":
        options["no_new_spend"] = False
    elif defect == "cloud":
        options["client_name"] = "qualia-cloud"
    elif defect == "model":
        options["reported_model"] = "changed-model"
    else:
        options["restrictions"] = ""
    no_provider(monkeypatch)
    with pytest.raises(ValueError):
        qualia.begin(client, prepared["episode"], **options)
    assert ledger.inspect(directory)["counts"]["total"] == 0


def test_ui_and_cli_pending_intents_share_capacity_and_two_ui_starts_are_never_refunded(tmp_path, monkeypatch, server_pair):
    directory, plan, client, lab, _, first = ready(tmp_path, monkeypatch, server_pair)
    second = qualia.prepare(client, plan["slots"][1]["slot_id"], lab=lab)
    smoke = study._json(directory.parent / "smoke_plan.json")
    pending = ledger.begin(directory, smoke["plan_sha256"], smoke["slots"][0]["slot_id"])
    begin(client, first["episode"])
    with pytest.raises(ValueError, match="two in-flight"):
        begin(client, second["episode"])
    finish(directory, pending, reward=0)
    begin(client, second["episode"])
    for episode in (first["episode"], second["episode"]):
        result = qualia.abandon(episode, client_stopped=True, reason="TEST FIXTURE ONLY; no UI was invoked")
        assert not result["qualia_verified"] and result["outcome"]["reward"] is None
    counts = ledger.inspect(directory)["counts"]
    assert counts["integration"] == 2 and counts["total"] == 3 and counts["active"] == 0
    with pytest.raises(ValueError, match="already reserved"):
        qualia.prepare(client, plan["slots"][0]["slot_id"], lab=lab)
    copied = tmp_path / "copied-ledger"
    shutil.copytree(directory, copied)
    with pytest.raises(ValueError, match="canonical journal"):
        ledger.begin(copied, plan["plan_sha256"], plan["slots"][0]["slot_id"])


@pytest.mark.parametrize("field", ["active", "halted"])
def test_generation_accounting_blocks_a_new_ui_reservation(tmp_path, monkeypatch, server_pair, field):
    directory, _, client, _, _, prepared = ready(tmp_path, monkeypatch, server_pair)
    original = generation.read
    monkeypatch.setattr(generation, "read", lambda *args: {**original(*args), field: 2 if field == "active" else True})
    with pytest.raises(ValueError):
        begin(client, prepared["episode"])
    assert ledger.inspect(directory)["counts"]["total"] == 0


def test_duplicate_concurrent_ui_starts_consume_one_intent(tmp_path, monkeypatch, server_pair):
    directory, plan, client, lab, _, first = ready(tmp_path, monkeypatch, server_pair)
    second = qualia.prepare(client, plan["slots"][0]["slot_id"], lab=lab)

    def reserve(episode):
        try:
            begin(client, episode)
            return "started"
        except ValueError:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(reserve, [first["episode"], second["episode"]]))
    assert sorted(values) == ["refused", "started"]
    assert ledger.inspect(directory)["counts"]["total"] == ledger.inspect(directory)["counts"]["active"] == 1


def test_lost_activation_acknowledgement_reuses_reserved_intent_without_launch_or_resend(tmp_path, monkeypatch, server_pair):
    directory, _, client, _, hub, prepared = ready(tmp_path, monkeypatch, server_pair)
    episode = prepared["episode"]
    post = client.http.post

    def lost(path, *args, **kwargs):
        response = post(path, *args, **kwargs)
        if path.endswith("qualia-activate"):
            raise OSError("fixture lost activation acknowledgement")
        return response

    monkeypatch.setattr(client.http, "post", lost)
    with pytest.raises(OSError, match="activation acknowledgement"):
        begin(client, episode)
    original = copy.deepcopy(hub.integrations[episode]["activation"])
    monkeypatch.setattr(client.http, "post", post)
    assert "never resend" in qualia.activate(client, episode)["instruction"]
    assert hub.integrations[episode]["activation"] == original
    assert ledger.inspect(directory)["counts"]["total"] == 1


def test_capture_acknowledgement_and_source_drift_recover_without_network_or_more_models(tmp_path, monkeypatch, server_pair):
    directory, plan, client, lab, hub, prepared = ready(tmp_path, monkeypatch, server_pair)
    episode = prepared["episode"]
    begin(client, episode, reported_model=model()["reported_model_id"])
    run_tools(lab, hub, episode)
    publish = study._publish

    def lost(path, value):
        publish(path, value)
        if path.parent.name == "captures":
            raise OSError("fixture lost capture acknowledgement")

    monkeypatch.setattr(study, "_publish", lost)
    with pytest.raises(OSError, match="capture acknowledgement"):
        qualia.collect(client, episode, client_observed=True, client_stopped=True)
    assert ledger.inspect(directory)["counts"]["active"] == 1
    monkeypatch.setattr(study, "_publish", publish)
    no_provider(monkeypatch)
    monkeypatch.setattr(client, "summary", lambda *a: pytest.fail("resume queried the live server"))
    sources = study._sources()
    monkeypatch.setattr(study, "_sources", lambda: {**sources, "benchtop/qualia.py": "0" * 64})
    saved = qualia.resume(episode)
    assert saved["outcome"]["state"] == "completed" and qualia.resume(episode) == saved
    with pytest.raises(ValueError):
        ledger.begin(directory, plan["plan_sha256"], plan["slots"][1]["slot_id"])
    assert ledger.inspect(directory)["counts"]["total"] == 1


def test_model_change_halts_new_starts_and_expired_lease_never_releases_itself(tmp_path, monkeypatch, server_pair):
    directory, _, client, lab, hub, prepared = ready(tmp_path, monkeypatch, server_pair)
    episode = prepared["episode"]
    begin(client, episode)
    deadline = hub.integrations[episode]["deadline"]
    hub.integrations[episode]["deadline"] = 0
    assert client.run(episode, [hub.get(episode).env.space.configs()[0]])["code"] == "integration_inactive"
    assert ledger.inspect(directory)["counts"]["active"] == 1
    hub.integrations[episode]["deadline"] = deadline
    run_tools(lab, hub, episode)
    result = qualia.collect(client, episode, client_stopped=True, client_observed=True, reported_model="changed-model")
    assert not result["qualia_verified"] and result["outcome"]["reward"] is None
    assert ledger.inspect(directory)["halted"]
    smoke = study._json(directory.parent / "smoke_plan.json")
    with pytest.raises(ValueError, match="halted"):
        ledger.begin(directory, smoke["plan_sha256"], smoke["slots"][0]["slot_id"])


def test_stale_server_and_task_overrides_fail_before_reservation_and_legacy_route_is_blocked(tmp_path, monkeypatch, server_pair):
    directory, plan, client, _, hub, prepared = ready(tmp_path, monkeypatch, server_pair)
    episode = prepared["episode"]
    assert client.http.post("/episodes", json={"qualia": {"plan_sha256": plan["plan_sha256"], "slot_id": plan["slots"][1]["slot_id"]}, "budget_frac": 1}).status_code == 400
    hub.integrations[episode]["binding"]["server_instance_id"] = "different"
    with pytest.raises(ValueError, match="restarted"):
        begin(client, episode)
    with pytest.raises(ValueError, match="separate allowance"):
        integration.prepare(client, root=primary.legacy_directory())
    assert ledger.inspect(directory)["counts"]["total"] == 0


def test_no_ui_retry_authority_even_for_an_operator_supplied_technical_outcome(tmp_path, monkeypatch, server_pair):
    directory, plan, client, _, _, prepared = ready(tmp_path, monkeypatch, server_pair)
    episode = prepared["episode"]
    begin(client, episode)
    with ledger._locked(directory) as (root, manifest):
        entry = ledger._read(root, manifest)["histories"][(plan["plan_sha256"], plan["slots"][0]["slot_id"])][0]
    finish(directory, entry["start"], state="technical_failure", reward=None, cause="provider_outage")
    with pytest.raises(ValueError, match="no authorized UI retry"):
        ledger.begin(directory, plan["plan_sha256"], plan["slots"][0]["slot_id"], retry_technical=True, integration_request=entry["start"]["integration_request"])
    assert ledger.inspect(directory)["counts"]["retry"] == 0


@pytest.mark.parametrize("field", ["connection", "authorization", "capture"])
def test_changed_private_bindings_cannot_be_promoted_to_a_qualia_pass(tmp_path, monkeypatch, server_pair, field):
    _, _, client, lab, hub, prepared = ready(tmp_path, monkeypatch, server_pair)
    episode = prepared["episode"]
    started = begin(client, episode)
    run_tools(lab, hub, episode)
    qualia.collect(client, episode, client_stopped=True, client_observed=True)
    if field == "connection":
        path = qualia._root() / "connections" / (episode + ".json")
        record = study._json(path)
        record["expected_budget"] += 1
        path.write_text(json.dumps(study._seal(record, "connection_sha256")))
    elif field == "authorization":
        path = primary.canonical_root() / qualia.AUTHORIZATION
        record = study._json(path)
        record["ledger_id"] = "0" * 64
        path.write_text(json.dumps(study._seal(record, "record_sha256")))
    else:
        path = qualia._capture_path(started["attempt_id"])
        record = study._json(path)
        record["sealed"]["summary"]["result"]["score"] = 0.25
        path.write_text(json.dumps(study._seal(record, "artifact_sha256")))
    with pytest.raises(ValueError):
        qualia.resume(episode)


def test_missing_authority_and_cli_inspection_create_no_models_or_ui_pass(tmp_path, monkeypatch, server_pair, capsys):
    monkeypatch.setattr(primary, "STATE_BASE", tmp_path / "missing")
    no_provider(monkeypatch)
    with pytest.raises(FileNotFoundError):
        qualia.authorize()
    assert not primary.STATE_BASE.exists()
    directory, _, _, _, _, _ = ready(tmp_path, monkeypatch, server_pair, prepare=False)
    no_provider(monkeypatch)
    main(["study", "authorize-qualia"])
    assert json.loads(capsys.readouterr().out)["planned_slots"] == 2
    main(["study", "inspect-qualia"])
    printed = capsys.readouterr().out
    assert not json.loads(printed)["qualia_verified"] and str(directory) not in printed


def test_ui_prompt_tampering_cannot_mutate_the_authorized_template(tmp_path, monkeypatch, server_pair):
    ready(tmp_path, monkeypatch, server_pair, prepare=False)
    plan = qualia.make_plan()
    original = copy.deepcopy(qualia.PROMPT)
    plan["binding"]["client_prompt"]["client_instructions"] = "changed prompt"
    plan["bindings_hash"] = study.canonical_hash(plan["binding"])
    assert original == qualia.PROMPT
    with pytest.raises(ValueError, match="Qualia integration plan"):
        study.validate_plan(study._seal(plan, "plan_sha256"))


def test_primary_cli_errors_redact_private_capabilities(monkeypatch):
    capability = "s" * 43
    url = "https://example.invalid/e/" + "a" * 12 + "/" + capability + "/mcp"
    monkeypatch.setattr("benchtop.agents.common.Client", lambda *args: SimpleNamespace(http=SimpleNamespace(close=lambda: None)))

    def failed(*args):
        raise ValueError("fixture failed " + url)

    monkeypatch.setattr(qualia, "prepare", failed)
    with pytest.raises(SystemExit) as error:
        main(["study", "prepare-qualia", "unused"])
    assert capability not in str(error.value) and "[redacted]" in str(error.value)
