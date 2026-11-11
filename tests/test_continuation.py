import copy
import json
import os
import subprocess
import sys
from collections import Counter

import pytest

from benchtop import calibrate, continuation, primary, provider, replay, runner, study
from benchtop import session_ledger as ledger
from benchtop.cache import file_hash
from benchtop.env import ROOT
from tests.test_primary import authorize, preparation, simulate_provider_transport
from tests.test_study import model, ready_controls


@pytest.fixture
def parent(tmp_path, monkeypatch):
    plans = preparation(tmp_path, monkeypatch)
    directory = authorize(plans)
    monkeypatch.setattr(provider, "execute", simulate_provider_transport)
    with runner.Controller() as controller:
        result = runner.run_provider(plans["smoke"]["plan_sha256"], plans["smoke"]["slots"][0]["slot_id"], controller=controller)
    assert result["status"] == "completed"
    manifest = ledger._manifest(directory)
    inventory = continuation._inventory(directory.parent)
    old_spec = study.SPEC
    sources = study._sources()
    spec = study._json(ROOT / "benchmarks/release_v2/spec.json")
    spec["continuation"].update(parent_ledger_id=manifest["ledger_id"], parent_manifest_sha256=file_hash(directory / "ledger.json"),
        parent_inventory_sha256=study.canonical_hash(inventory))
    target = tmp_path / "protocol/spec.json"
    target.parent.mkdir()
    target.write_text(json.dumps(spec))
    references = study._json(ROOT / "benchmarks/release_v2/references.json")
    references["protocol_sha256"] = file_hash(target)
    (target.parent / "references.json").write_text(json.dumps(references))
    monkeypatch.setattr(study, "SPEC", target)
    return {"directory": directory, "manifest": manifest, "inventory": inventory, "spec": spec, "old_spec": old_spec, "plans": plans, "sources": sources}


def save_spec(value, path):
    path.write_text(json.dumps(value))
    refs = study._json(path.parent / "references.json")
    refs["protocol_sha256"] = file_hash(path)
    (path.parent / "references.json").write_text(json.dumps(refs))


def test_prior_usage_is_consumed_without_rebinding_parent_or_increasing_global_budget(parent):
    record = continuation.prepare()
    assert record["prior_counts"]["total"] == 1
    assert record["remaining_start_intents"] == 299
    assert record["prior_counts"]["total"] + record["remaining_start_intents"] == record["global_total_max_episode_starts"] == 300
    assert record["prior_costs"]["episode_starts"] == 1
    assert continuation.prepare() == continuation.verify() == record
    assert continuation._inventory(parent["directory"].parent) == parent["inventory"]
    assert ledger._manifest(parent["directory"]) == parent["manifest"]
    assert continuation.usage()["prior_counts"]["total"] == 1
    identity = {"limits": parent["spec"]["execution_limits"], "phase_limits": ledger._phase_limits(parent["spec"])}
    with pytest.raises(ValueError, match="total start"):
        ledger._capacity(Counter(total=299), "confirmation", 0, identity)
    ledger._capacity(Counter(total=298), "confirmation", 0, identity)


@pytest.mark.parametrize("change", ["extra_file", "active_parent", "manifest", "missing_authority", "expanded_cap", "changed_science", "reference_order"])
def test_changed_parent_or_expanded_or_unbound_successor_cannot_start(parent, change):
    if change != "missing_authority":
        continuation.prepare()
    directory = parent["directory"]
    if change == "extra_file":
        (directory.parent / "unexpected.json").write_text("{}")
    elif change == "active_parent":
        next((directory / "attempts").glob("*.finish.json")).unlink()
    elif change == "manifest":
        path = directory / "ledger.json"
        path.write_text(path.read_text() + " ")
    elif change in ("expanded_cap", "changed_science"):
        spec = copy.deepcopy(parent["spec"])
        if change == "expanded_cap":
            spec["execution_limits"]["total_max_episode_starts"] = 300
        else:
            spec["seed_partitions"]["confirmation"] = [999]
        save_spec(spec, study.SPEC)
    elif change == "reference_order":
        path = study.SPEC.parent / "references.json"
        value = study._json(path)
        value["cases"].reverse()
        path.write_text(json.dumps(value))
    with pytest.raises((ValueError, FileNotFoundError)):
        continuation.verify()


def test_retired_parent_is_inspectable_but_cannot_start_again(parent, monkeypatch):
    continuation.prepare()
    monkeypatch.setattr(study, "SPEC", parent["old_spec"])
    assert ledger.inspect(parent["directory"])["counts"]["total"] == 1
    with pytest.raises(ValueError, match="retired"):
        ledger.begin(parent["directory"], parent["plans"]["smoke"]["plan_sha256"], parent["plans"]["smoke"]["slots"][1]["slot_id"])
    assert ledger.inspect(parent["directory"])["counts"]["total"] == 1


def test_successor_smokes_and_development_lock_retain_parent_usage(parent, tmp_path, monkeypatch):
    # temporary protocol fixture; real local MCP calls, no language model
    monkeypatch.setattr(study, "_sources", lambda: parent["sources"])
    monkeypatch.setattr(calibrate, "SPEC", study.SPEC)
    monkeypatch.setattr(replay, "SPEC", study.SPEC)
    monkeypatch.setattr(replay, "REFERENCES", study.SPEC.parent / "references.json")
    controls = tmp_path / "successor-controls"
    controls.mkdir()
    reviews, foundation = ready_controls(controls)
    plans = {phase: study.make_plan(model=model(), phase=phase, mode="study_preparation",
                                   reviews_dir=reviews, foundation=foundation) for phase in ("smoke", "development")}
    directory = authorize(plans)
    with runner.Controller() as controller:
        for slot in plans["smoke"]["slots"]:
            result = runner.run_provider(plans["smoke"]["plan_sha256"], slot["slot_id"], controller=controller)
            assert result["status"] == "completed"
    record = primary.lock_development(plans["development"])
    assert record["continuation_sha256"] == continuation.verify()["record_sha256"]
    assert primary.verify(plans["development"])["record_sha256"] == record["record_sha256"]
    summary = ledger.inspect(directory)
    assert summary["counts"]["total"] == 6
    assert summary["cumulative_start_intents"] == 7
    assert summary["prior_protocol_usage"]["prior_counts"]["total"] == 1
    assert continuation._inventory(parent["directory"].parent) == parent["inventory"]


def test_protocol_selection_is_consistent_across_modules_and_unknown_profiles_are_refused():
    script = "from benchtop import study,calibrate,replay,primary; assert study.SPEC==calibrate.SPEC==replay.SPEC; assert study.SPEC.parent.name=='release_v2'; assert primary.BINDINGS.parent==study.SPEC.parent; assert replay.REFERENCES.parent==study.SPEC.parent"
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env={**os.environ, "BENCHTOP_PROTOCOL": "release_v2"}, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env={**os.environ, "BENCHTOP_PROTOCOL": "../../arbitrary"}, capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "must be release_v1 or release_v2" in result.stderr


def test_v2_offline_analysis_retains_excluded_usage_without_private_authority():
    script = """
from pathlib import Path
from unittest.mock import patch
from benchtop import calibrate, confirmation_analysis, primary, release, study
from tests.test_confirmation_analysis import document
primary.STATE_BASE = Path('/nonexistent-continuation-fixture-state')
collection = document()
collection['accounting']['prior_protocol_usage'] = {
    'prior_counts': {'total': 1}, 'global_total_max_episode_starts': 300,
    'prior_costs': calibrate._costs([{'attempts': [{'costs': {}}]}])}
collection = study._seal(collection, 'snapshot_sha256')
with patch('benchtop.continuation.verify', side_effect=AssertionError('offline analysis needs private authority')):
    result = confirmation_analysis.analyse(collection)
    report = confirmation_analysis.render_markdown(result)
assert '1 consumed compatibility start(s)' in report
assert 'Earlier protocol input_tokens: known 0; total unavailable; unreported 1.' in report
assert release._binding_name() == 'benchmarks/release_v2/bindings.json'
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env={**os.environ, "BENCHTOP_PROTOCOL": "release_v2"}, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
