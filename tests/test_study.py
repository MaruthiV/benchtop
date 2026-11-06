import copy
import json
import shutil
from concurrent.futures import ThreadPoolExecutor

import pytest

from benchtop import study
from benchtop.admission import inspect_lab
from benchtop.cli import main
from benchtop.env import ROOT
from tests.test_admission import human_control
from tests.test_coverage_policies import fixture


def model():
    return json.loads((ROOT / "benchmarks/release_v1/study/engineering_model.json").read_text())


def plan(tmp_path, *, phase="smoke"):
    env = fixture(tmp_path)
    return env, study.make_plan([env.dir], model=model(), phase=phase)


def ready_controls(tmp_path):
    reviews = tmp_path / "simulated-reviews"
    reviews.mkdir()
    for row in study._json(ROOT / "benchmarks/release_v1/admission/references/selection.json")["cases"]:
        record = human_control(inspect_lab(ROOT / "envs" / row["env"]))
        (reviews / f"{row['env']}.json").write_text(json.dumps(record))
    foundation = tmp_path / "simulated-foundation.json"
    foundation.write_text(json.dumps({"scope": "foundation_engineering_checks", "status": "passed",
        "source_inputs_unchanged": True, "model_sessions": 0, "archive_files_verified": 241,
        "original_data_files_verified": 200, "offline": {"status": "passed"},
        "runtime": study.cache.runtime_identity(),
        "input_sha256": study._sources(), "checked_at_utc": "2026-11-06T00:00:00+00:00"}))
    return reviews, foundation


def test_canonical_group_slot_attempt_identities_have_no_circular_hash(tmp_path):
    _, prepared = plan(tmp_path, phase="development")
    binding = prepared["binding"]
    assert prepared["bindings_hash"] == study.canonical_hash(binding)
    assert "slots" not in binding and "bindings_hash" not in binding
    assert len(prepared["groups"]) == 3 and len(prepared["slots"]) == 12
    fields = study._json(study.SPEC)["identities"]["group_descriptor_fields"]
    for group in prepared["groups"]:
        assert set(group["descriptor"]) == set(fields)
        assert group["group_id"] == study.canonical_hash(group["descriptor"])
        assert group["descriptor"]["bindings_hash"] == prepared["bindings_hash"]
    for slot in prepared["slots"]:
        assert slot["slot_id"] == study.canonical_hash({"group_id": slot["group_id"], "replicate_index": slot["replicate_index"]})
        assert slot["attempt_ids"] == [study.canonical_hash({"slot_id": slot["slot_id"], "attempt_index": i}) for i in (0, 1)]
    assert len({slot["slot_id"] for slot in prepared["slots"]}) == 12
    assert len({attempt for slot in prepared["slots"] for attempt in slot["attempt_ids"]}) == 24
    assert study.validate_plan(json.loads(json.dumps(prepared))) == prepared


def test_prompt_model_runtime_and_visible_order_are_bound_without_execution(tmp_path):
    env, prepared = plan(tmp_path)
    binding = prepared["binding"]
    assert binding["prompt"]["system_text"] == study.SYSTEM
    assert "def first_message" in binding["prompt"]["first_message_source"]
    assert binding["model_configuration"] == model()
    assert binding["runtime"]["python"]
    assert binding["source_sha256"]["benchtop/study.py"] == study.cache.file_hash(study.__file__)
    assert set(binding["environments"][0]["visible_order"]) == {0, 1, 2, 3}
    assert len({tuple(group["descriptor"]["visible_order"]) for group in prepared["groups"]}) == 1
    assert binding["environments"][0]["charging_hash"] == env.charging_schedule()[1]
    assert prepared["model_sessions_started"] == 0 and not prepared["execution_ready"]


def test_smoke_is_bounded_and_phase_separates_otherwise_identical_rollouts(tmp_path):
    env, smoke = plan(tmp_path)
    development = study.make_plan([env.dir], model=model(), phase="development")
    assert len(smoke["slots"]) == 3 and {s["budget_fraction"] for s in smoke["slots"]} == {0.06, 0.12, 0.25}
    assert not {s["slot_id"] for s in smoke["slots"]} & {s["slot_id"] for s in development["slots"]}
    with pytest.raises(ValueError, match="one or two"):
        study.make_plan([env.dir] * 3, model=model())


def test_historical_plan_does_not_upgrade_or_change_original_inputs():
    path = ROOT / "envs/2610-36935-gaussian-process-bandits-for"
    before = {name: study.cache.file_hash(path / name) for name in study.cache.DATA_FILES}
    prepared = study.make_plan([path], model=model())
    assert prepared["binding"]["environments"][0]["execution_id"].startswith("historical_v0:")
    assert {name: study.cache.file_hash(path / name) for name in study.cache.DATA_FILES} == before


@pytest.mark.parametrize("defect", ["alias", "mismatch", "missing_cli", "backend", "extra_secret", "implicit_decoding", "empty_reason", "boolean_seed", "negative_temperature", "url"])
def test_ambiguous_model_or_decoding_configuration_is_rejected(tmp_path, defect):
    config = model()
    if defect == "alias":
        config["requested_model_id"] = "haiku"
    elif defect == "mismatch":
        config["reported_model_id"] = "different-model"
    elif defect == "missing_cli":
        config["cli_version"] = ""
    elif defect == "backend":
        config["backend"] = "paid-api"
    elif defect == "extra_secret":
        config["api_key"] = "synthetic-invalid-key"
    elif defect == "implicit_decoding":
        config["decoding"].pop("temperature")
    elif defect == "empty_reason":
        config["decoding"]["temperature"]["reason"] = ""
    elif defect == "url":
        config["requested_model_id"] = config["reported_model_id"] = "https://example.invalid/private/capability"
    else:
        config["decoding"]["model_seed" if defect == "boolean_seed" else "temperature"] = {
            "status": "fixed", "value": True if defect == "boolean_seed" else -1}
    with pytest.raises(ValueError):
        study.make_plan([tmp_path / "not-read"], model=config)


def test_real_unassigned_drafts_cannot_create_a_study_preparation(tmp_path):
    reviews = ROOT / "benchmarks/release_v1/admission/references/drafts"
    result = study.gate(model=model(), reviews_dir=reviews)
    assert result["planning_status"] == "blocked" and not result["execution_ready"]
    with pytest.raises(ValueError, match="study preparation blocked"):
        study.make_plan(model=model(), mode="study_preparation", reviews_dir=reviews)


@pytest.mark.parametrize("phase,slots", [("smoke", 6), ("development", 48)])
def test_simulated_prerequisites_allow_only_preparation_never_primary_execution(tmp_path, phase, slots):
    reviews, foundation = ready_controls(tmp_path)
    result = study.gate(model=model(), phase=phase, reviews_dir=reviews, foundation=foundation)
    assert result["planning_status"] == "ready" and not result["execution_ready"]
    assert result["execution_blockers"]
    prepared = study.make_plan(model=model(), phase=phase, mode="study_preparation", reviews_dir=reviews, foundation=foundation)
    assert len(prepared["slots"]) == slots and len(prepared["binding"]["environments"]) == 4
    assert not prepared["execution_ready"] and study.validate_plan(prepared) == prepared
    directory = tmp_path / "preparation"
    receipt = study.save_plan(prepared, directory)
    assert receipt["study_lock"] is False and study.load_plan(directory) == prepared


@pytest.mark.parametrize("defect", ["missing", "wrong_scope", "stale_code", "missing_inputs", "outside_path", "malformed", "runtime"])
def test_foundation_gates_reject_stale_partial_or_unrelated_evidence(tmp_path, defect):
    reviews, foundation = ready_controls(tmp_path)
    value = json.loads(foundation.read_text())
    if defect == "missing":
        foundation.unlink()
    else:
        if defect == "wrong_scope":
            value["scope"] = "unrelated"
        elif defect == "stale_code":
            value["input_sha256"]["benchtop/study.py"] = "0" * 64
        elif defect == "missing_inputs":
            value["input_sha256"].pop("uv.lock")
        elif defect == "outside_path":
            value["input_sha256"]["../outside"] = "0" * 64
        elif defect == "runtime":
            value["runtime"] = {"python": "different-runtime"}
        else:
            value = []
        foundation.write_text(json.dumps(value))
    result = study.gate(model=model(), reviews_dir=reviews, foundation=foundation)
    assert result["planning_status"] == "blocked" and result["foundation"]["status"] != "current"


@pytest.mark.parametrize("phase", ["fresh_pilot", "confirmation"])
def test_future_phase_gates_cannot_be_unlocked_by_passing_preparation_inputs(tmp_path, phase):
    reviews, foundation = ready_controls(tmp_path)
    result = study.gate(model=model(), phase=phase, reviews_dir=reviews, foundation=foundation)
    assert result["planning_status"] == "blocked" and not result["execution_ready"]
    with pytest.raises(ValueError, match="only smoke/development"):
        study.make_plan(model=model(), phase=phase)


def test_saved_plan_is_exclusive_and_partial_publication_is_not_complete(tmp_path):
    _, prepared = plan(tmp_path)
    directory = tmp_path / "saved"
    study.save_plan(prepared, directory)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(FileExistsError):
        study.save_plan(prepared, directory)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    (directory / "receipt.json").unlink()
    with pytest.raises(FileNotFoundError):
        study.load_plan(directory)
    with pytest.raises(ValueError, match="under artifacts"):
        study.save_plan(prepared, ROOT / "benchmarks/release_v1/must-not-exist")


@pytest.mark.parametrize("defect", ["slots", "execution", "mode", "receipt"])
def test_resealed_tampering_cannot_promote_or_modify_preparations(tmp_path, defect):
    _, prepared = plan(tmp_path)
    directory = tmp_path / "saved"
    study.save_plan(prepared, directory)
    altered = copy.deepcopy(prepared)
    if defect == "slots":
        altered["slots"].pop()
    elif defect == "execution":
        altered["execution_ready"] = True
    elif defect == "mode":
        altered["binding"]["mode"] = "study_preparation"
        altered["bindings_hash"] = study.canonical_hash(altered["binding"])
    else:
        receipt = json.loads((directory / "receipt.json").read_text())
        receipt["study_lock"] = True
        (directory / "receipt.json").write_text(json.dumps(study._seal(receipt, "record_sha256")))
        with pytest.raises(ValueError, match="does not match"):
            study.load_plan(directory)
        return
    altered = study._seal(altered, "plan_sha256")
    with pytest.raises((ValueError, TypeError)):
        study.validate_plan(altered)


def test_stale_sources_runtime_and_reviews_cannot_revalidate(tmp_path, monkeypatch):
    engineering = tmp_path / "engineering"
    engineering.mkdir()
    env, prepared = plan(engineering)
    with monkeypatch.context() as patch:
        original = study.cache.runtime_identity
        patch.setattr(study.cache, "runtime_identity", lambda: {**original(), "python": "changed"})
        with pytest.raises(ValueError, match="changed"):
            study.validate_plan(prepared)
    with monkeypatch.context() as patch:
        original = study._sources
        patch.setattr(study, "_sources", lambda: {**original(), "benchtop/study.py": "0" * 64})
        with pytest.raises(ValueError, match="changed"):
            study.validate_plan(prepared)
    with (env.dir / "lab.py").open("a") as out:
        out.write("\n# controlled source change\n")
    with pytest.raises(ValueError):
        study.validate_plan(prepared)
    reviews, foundation = ready_controls(tmp_path)
    primary = study.make_plan(model=model(), mode="study_preparation", reviews_dir=reviews, foundation=foundation)
    review = next(reviews.glob("*.json"))
    review.write_text("{}")
    with pytest.raises(ValueError, match="blocked"):
        study.validate_plan(primary)


def test_duplicate_content_cannot_create_colliding_slots(tmp_path):
    env, _ = plan(tmp_path)
    clone = tmp_path / "clone"
    shutil.copytree(env.dir, clone)
    manifest = json.loads((clone / "manifest.json").read_text())
    manifest["id"] = "clone"
    (clone / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="collide"):
        study.make_plan([env.dir, clone], model=model())


def test_fresh_partition_cannot_be_read_as_a_development_preparation(tmp_path):
    from tests.test_admission import fixture as admission_fixture

    groups = {"visible": [0, 1], "qualification": [2, 3, 4], "confirmation": [5, 6, 7]}
    directory = admission_fixture(tmp_path, splits=groups)
    with pytest.raises(ValueError, match="two-part"):
        study.make_plan([directory], model=model())


def test_concurrent_publication_preserves_one_complete_preparation(tmp_path):
    _, prepared = plan(tmp_path)
    directory = tmp_path / "concurrent"

    def publish():
        try:
            study.save_plan(prepared, directory)
            return "published"
        except FileExistsError:
            return "already_exists"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: publish(), range(2)))
    assert sorted(results) == ["already_exists", "published"]
    assert study.load_plan(directory) == prepared


def test_cli_prepares_inspects_and_reports_primary_gate_without_running_models(tmp_path, capsys):
    env = fixture(tmp_path)
    target = tmp_path / "cli-preparation"
    config = ROOT / "benchmarks/release_v1/study/engineering_model.json"
    main(["study", "plan", "--env", str(env.dir), "--model-config", str(config), "--output-dir", str(target)])
    assert '"model_sessions_started": 0' in capsys.readouterr().out
    main(["study", "inspect", str(target)])
    assert '"execution_ready": false' in capsys.readouterr().out
    main(["study", "gate", "--model-config", str(config)])
    assert '"planning_status": "blocked"' in capsys.readouterr().out
