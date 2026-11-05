import importlib.util
import json
import sys

import pytest

from benchmarks.release_v1.admission import check_defects as fixtures
from benchtop.cache import file_hash
from benchtop.env import ROOT


def test_cached_cases_separate_valid_null_negative_and_unavailable_rewards(tmp_path):
    result = fixtures.run_suite(live=False, output=tmp_path / "report.json")
    rows = {case["id"]: case for case in result["cases"]}
    assert result["status"] == "partial"
    assert result["summary"]["controlled_defects_observed"] == 3
    assert result["summary"]["sandbox_jobs"] == 0
    assert result["summary"]["source_inputs_unchanged"]
    assert rows["valid_null_claim"]["evidence"]["claim"]["label"] == "no_meaningful_difference"
    assert rows["valid_negative_claim"]["evidence"]["claim"]["label"] == "baseline_better"
    ambiguous = rows["ambiguous_reward"]
    assert ambiguous["observed"] == "unavailable"
    assert ambiguous["evidence"]["claim"]["label"] is None
    assert ambiguous["evidence"]["data_status"] == "complete"
    assert ambiguous["evidence"]["scientific_admission"] == "needs_review"
    assert ambiguous["evidence"]["counterfactual"]
    assert rows["stale_cache"]["observed"] == "fail"
    assert rows["valid_cache"]["observed"] == "pass"
    assert rows["reversed_metric"]["observed"] == "fail"
    assert rows["valid_metric"]["observed"] == "pass"
    assert json.loads((tmp_path / "report.json").read_text())["human_admissions"] == 0


@pytest.mark.parametrize("source", ["no match", "anchor anchor"])
def test_mutations_cannot_silently_miss_or_modify_multiple_locations(source):
    with pytest.raises(ValueError, match="exactly one"):
        fixtures.mutate_once(source, "anchor", "mutant")


@pytest.mark.parametrize("guard", ["muon.momentum_perturbation", "muon.heldout_disjoint", "muon.common_inputs"])
def test_missing_runtime_observations_never_pass(guard):
    with pytest.raises(ValueError, match="observations"):
        fixtures.muon_guard(guard, [])


def test_common_input_guard_requires_actual_pairing():
    with pytest.raises(ValueError, match="paired baseline and treatment"):
        fixtures.muon_guard("muon.common_inputs", [{"config": {"method": "muon", "rho": 0.3}, "seed": 0, "traces": {}}])


def test_reproduction_fails_when_reference_source_is_unbound(monkeypatch):
    monkeypatch.setattr(fixtures, "file_hash", lambda path: "0" * 64)
    with pytest.raises(ValueError, match="fixture source changed"):
        fixtures.run_suite(live=False)


def test_manifest_cannot_silently_drop_a_runtime_defect(tmp_path, monkeypatch):
    manifest = fixtures.load_manifest()
    manifest["muon"]["mutations"].pop()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(fixtures, "DEFECTS", tmp_path)
    with pytest.raises(ValueError, match="retain all three"):
        fixtures.load_manifest()


def test_fixture_checker_handles_a_symlinked_checkout_path(tmp_path):
    alias = tmp_path / "checkout-alias"
    alias.symlink_to(ROOT, target_is_directory=True)
    source = alias / "benchmarks/release_v1/admission/check_defects.py"
    spec = importlib.util.spec_from_file_location("fixture_checker_alias", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run_suite(live=False)
    assert result["status"] == "partial"
    assert result["summary"]["source_inputs_unchanged"]
    assert "benchmarks/release_v1/admission/check_defects.py" in result["input_sha256"]


def test_live_mode_does_not_fall_back_to_host_execution(monkeypatch):
    def unsupported():
        raise RuntimeError("unsupported fixture sandbox")

    monkeypatch.setattr(fixtures, "sandbox_mode", unsupported)
    with pytest.raises(RuntimeError, match="unsupported fixture sandbox"):
        fixtures.run_suite(live=True)


def test_output_cannot_overwrite_reference_inputs_or_review_drafts(monkeypatch):
    protected = ROOT / "envs" / fixtures.MUON / "lab.py"
    before = file_hash(protected)

    def should_not_run():
        raise AssertionError("invalid output should fail before input loading or worker startup")

    monkeypatch.setattr(fixtures, "load_manifest", should_not_run)
    with pytest.raises(ValueError, match="write under artifacts"):
        fixtures.run_suite(output=protected)
    assert file_hash(protected) == before


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS only")
def test_source_derived_runtime_mutations_and_valid_controls():
    result = fixtures.run_suite()
    assert result["status"] == "passed"
    assert result["summary"]["controlled_defects_observed"] == 6
    assert result["summary"]["valid_controls"] == 10
    assert result["summary"]["sandbox_jobs"] == 24
    assert result["summary"]["source_inputs_unchanged"]
    assert result["human_admissions"] == result["model_sessions"] == 0
    rows = {case["id"]: case for case in result["cases"]}
    disabled = rows["disabled_treatment"]
    assert disabled["observed"] == "fail"
    assert all(r["training_perturbation_calls"] == 0 for r in disabled["evidence"])
    overlap = rows["advantaged_baseline"]
    assert overlap["observed"] == "fail"
    assert all(r["train_test_overlap"] == 400 for r in overlap["evidence"])
    different = rows["inconsistent_shared_randomness"]
    assert different["observed"] == "fail"
    assert all(set(r["mismatched_streams"]) == {"train", "test"} for r in different["evidence"])
    for name in ("valid_treatment", "valid_holdout", "valid_shared_inputs", "observer_transparency",
                 "fresh_worker_repeatability", "zero_radius_identity"):
        assert rows[name]["observed"] == "pass"
    runs = {run["variant"]: run for run in result["live_runs"]}
    assert all(len(run["executed_source_sha256"]) == 64 for run in runs.values())
    original = runs["valid_muon"]["observations"]
    musam = next(r for r in original if r["config"]["method"] == "musam")
    assert musam["traces"]["gradient_calls"] == 301
    assert musam["traces"]["training_perturbation_calls"] == 150
