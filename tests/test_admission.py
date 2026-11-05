import json

import pytest

from benchtop.admission import REVIEW_CHECKS, draft_review, inspect_lab, screen_catalog, write_json
from benchtop.cache import initialize_cache, seal_result
from benchtop.cli import main
from benchtop.env import ROOT, Env
from benchtop.space import Space
from benchtop.stats import GRADER_ID
from benchtop.tasks import build_task_set


def fixture(root, *, effect="positive", splits=None, missing=None, source=None):
    """Synthetic cached rows and review attestations are test controls, not scientific observations."""
    env_dir = root / "fixture"
    env_dir.mkdir(parents=True)
    expression = {"positive": "0.2", "null": "0.0", "ambiguous": "0.02 + (0.15 if seed % 2 else -0.15)"}[effect]
    source = source or (
        "def run(config, seed):\n"
        f"    gain = ({expression}) if config['method'] == 'new' else 0.0\n"
        "    return {'value': 0.5 + gain}\n"
    )
    raw = {
        "title": "Synthetic admission control", "paper": {"claim": "A synthetic treatment changes the metric."},
        "metric": {"name": "value", "goal": "max", "min_effect": 0.02},
        "factors": [{"name": "method", "role": "treatment", "levels": ["base", "new"]},
                    {"name": "context", "role": "context", "levels": [0, 1]}],
        "treatment": {"factor": "method", "paper": "new", "baseline": "base"},
    }
    manifest = {"id": "fixture", "seeds": 8, "visible_seeds": 2}
    if splits is not None:
        manifest["seed_splits"] = splits
    (env_dir / "lab.py").write_text(source)
    (env_dir / "space.json").write_text(json.dumps(raw))
    (env_dir / "manifest.json").write_text(json.dumps(manifest))
    binding = initialize_cache(env_dir, manifest)
    space = Space.from_dict(raw)
    rows = []
    for index, config in enumerate(space.configs()):
        for seed in range(8):
            if missing == (index, seed):
                continue
            gain = 0.2 if effect == "positive" else 0.0 if effect == "null" else 0.02 + (0.15 if seed % 2 else -0.15)
            value = 0.5 + (gain if config["method"] == "new" else 0)
            rows.append(seal_result(binding, space, config, seed, {"ok": True, "metrics": {"value": value}, "secs": 1.0}))
    (env_dir / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    env = Env.load(env_dir)
    artifact = build_task_set(env, seeds=splits["qualification"] if splits else None)
    (env_dir / "tasks.json").write_text(json.dumps(artifact))
    return env_dir


def human_control(report):
    """Simulated human metadata exercises schema logic; it is never saved as a real reference review."""
    review = draft_review(report)
    review.update(decision="accepted", rationale="This controlled synthetic case meets its stated fixture contract.")
    review["reviewer"] = {"kind": "human", "name": "Synthetic fixture reviewer", "reviewed_at_utc": "2026-11-04T00:00:00+00:00"}
    review["paper"] = {"url": "https://example.invalid/fixture", "version": "fixture-v1", "sections": ["Synthetic specification"]}
    review["scope"] = {
        "paper_mechanism": "A controlled additive effect, used only in this test.",
        "miniature_comparison": "Paired base/new fixture metrics.",
        "preserved_assumptions": ["Equal contexts and seed pairing."],
        "simplifications": ["No real model training."],
        "unsupported_claims": ["No claims about a real paper or model performance."],
    }
    for check in review["checks"].values():
        check.update(status="pass", evidence=[{"source": "synthetic-specification", "locator": "valid-control",
                                                "finding": "This simulated review affirms the controlled fixture contract."}])
    return review


def test_mechanical_success_and_llm_audit_cannot_admit_a_lab(tmp_path):
    env = fixture(tmp_path)
    (env / "audit.json").write_text(json.dumps({"ok": True, "auditor": "synthetic-model", "fidelity": "high",
                                                "baseline_fair": True, "issues": []}))
    before = {path.name: path.read_bytes() for path in env.iterdir()}
    report = inspect_lab(env)
    assert report["mechanical"]["status"] == "passed"
    assert report["qualification"]["claim_eligible"]
    assert report["decision"] == "needs_review"
    assert not report["qualified_claim_admitted"]
    assert report["runtime_probes"]["status"] == "not_run"
    assert report["model_difficulty"]["status"] == "not_measured"
    assert not report["advisories"][0]["certifies_admission"]
    assert {path.name: path.read_bytes() for path in env.iterdir()} == before


@pytest.mark.parametrize("effect", ["positive", "null"])
def test_valid_controls_can_be_admitted_with_an_explicit_bound_human_review(tmp_path, effect):
    env = fixture(tmp_path, effect=effect)
    original = inspect_lab(env)
    report = inspect_lab(env, review=human_control(original))
    assert report["decision"] == "accepted"
    assert report["qualified_claim_admitted"]
    assert report["human_review"]["identity_source"] == "operator_reported"
    expected = "paper_better" if effect == "positive" else "no_meaningful_difference"
    assert report["qualification"]["claim"]["label"] == expected


def test_ambiguous_question_is_separate_from_scientific_admission(tmp_path):
    env = fixture(tmp_path, effect="ambiguous")
    first = inspect_lab(env)
    report = inspect_lab(env, review=human_control(first))
    assert report["mechanical"]["status"] == "passed"
    assert report["qualification"]["status"] == "complete"
    assert report["qualification"]["claim"]["status"] == "ambiguous"
    assert report["decision"] == "accepted"
    assert not report["qualified_claim_admitted"]


@pytest.mark.parametrize("defect", ["assistant", "missing_evidence", "unresolved_check", "missing_scope", "missing_version", "missing_time"])
def test_incomplete_or_model_authored_reviews_cannot_admit(tmp_path, defect):
    env = fixture(tmp_path)
    review = human_control(inspect_lab(env))
    if defect == "assistant":
        review["reviewer"]["kind"] = "assistant"
    elif defect == "missing_evidence":
        review["checks"]["baseline_fairness"]["evidence"] = []
    elif defect == "unresolved_check":
        review["checks"]["shared_randomness"]["status"] = "unresolved"
    elif defect == "missing_scope":
        review["scope"]["unsupported_claims"] = []
    elif defect == "missing_version":
        review["paper"]["version"] = ""
    else:
        review["reviewer"]["reviewed_at_utc"] = ""
    report = inspect_lab(env, review=review)
    assert report["mechanical"]["status"] == "passed"
    assert report["decision"] == "needs_review"
    assert not report["qualified_claim_admitted"]


def test_human_rejection_needs_evidence_and_does_not_depend_on_a_large_effect(tmp_path):
    env = fixture(tmp_path)
    first = inspect_lab(env)
    review = human_control(first)
    review["decision"] = "rejected"
    report = inspect_lab(env, review=review)
    assert report["decision"] == "needs_review"
    review["checks"]["baseline_fairness"]["status"] = "fail"
    report = inspect_lab(env, review=review)
    assert report["qualification"]["claim"]["label"] == "paper_better"
    assert report["decision"] == "rejected"
    assert not report["qualified_claim_admitted"]


def test_review_becomes_stale_on_a_different_bound_lab(tmp_path):
    first = fixture(tmp_path / "first")
    second = fixture(tmp_path / "second", effect="null")
    review = human_control(inspect_lab(first))
    report = inspect_lab(second, review=review)
    assert report["human_review"]["status"] == "stale"
    assert report["decision"] == "needs_review"


def test_visible_append_does_not_invalidate_a_qualification_bound_review(tmp_path):
    env_dir = fixture(tmp_path, missing=(0, 0))
    first = inspect_lab(env_dir)
    review = human_control(first)
    env = Env.load(env_dir)
    row = seal_result(env.execution, env.space, env.space.configs()[0], 0,
                      {"ok": True, "metrics": {"value": 0.5}, "secs": 999.0})
    with (env_dir / "results.jsonl").open("a") as result:
        result.write(json.dumps(row) + "\n")
    report = inspect_lab(env_dir, review=review)
    assert report["review_subject"] == first["review_subject"]
    assert report["decision"] == "accepted"


def test_incomplete_qualification_and_corrupt_cache_remain_visible(tmp_path):
    incomplete = fixture(tmp_path / "incomplete", missing=(0, 3))
    report = inspect_lab(incomplete)
    assert report["qualification"]["status"] == "incomplete"
    assert report["decision"] == "rejected"
    corrupt = fixture(tmp_path / "corrupt")
    path = corrupt / "results.jsonl"
    path.write_text('{"broken":\n')
    report = inspect_lab(corrupt)
    assert report["decision"] == "rejected"
    assert report["mechanical"]["checks"][0]["id"] == "input_integrity"
    assert path.read_text() == '{"broken":\n'


def test_disallowed_source_is_rejected_even_when_cached_numbers_are_settled(tmp_path):
    env = fixture(tmp_path, source="import os\ndef run(config, seed):\n    return {'value': 0.5}\n")
    report = inspect_lab(env)
    assert report["decision"] == "rejected"
    assert any(check["id"] == "static_contract" and check["status"] == "failed" for check in report["mechanical"]["checks"])


def test_explicit_qualification_ignores_confirmation_outcomes(tmp_path):
    splits = {"visible": [0, 1], "qualification": [2, 3, 4], "confirmation": [5, 6, 7]}
    env_dir = fixture(tmp_path, splits=splits)
    env = Env.load(env_dir)
    rows = [json.loads(line) for line in (env_dir / "results.jsonl").read_text().splitlines()]
    for row in rows:
        if row["seed"] in splits["confirmation"] and row["config"]["method"] == "new":
            row["metrics"]["value"] = -999.0
            row.update(seal_result(env.execution, env.space, row["config"], row["seed"], row))
    (env_dir / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = inspect_lab(env_dir)
    assert report["qualification"]["seed_ids"] == [2, 3, 4]
    assert report["qualification"]["claim"]["label"] == "paper_better"


def test_overlapping_seed_groups_cannot_enter_admission(tmp_path):
    splits = {"visible": [0, 1], "qualification": [1, 2, 3], "confirmation": [5, 6, 7]}
    env = fixture(tmp_path)
    path = env / "manifest.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "seed_splits": splits}))
    report = inspect_lab(env)
    assert report["decision"] == "rejected"
    assert report["qualification"]["status"] == "not_checked"


def test_catalog_retains_invalid_candidates_and_malformed_reviews(tmp_path):
    root = tmp_path / "envs"
    fixture(root)
    (root / "broken").mkdir()
    fixture(root / "unlisted")
    catalog = {"schema_version": 1, "id": "controls", "grader_id": GRADER_ID,
               "environments": ["fixture", "broken"],
               "funnel": {key: 2 for key in ("screened", "selected", "built", "lab_written", "accepted")}}
    (root / "catalog.json").write_text(json.dumps(catalog))
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    (reviews / "fixture.json").write_text('{"invalid":')
    report = screen_catalog(root, reviews_dir=reviews)
    assert [row["env"] for row in report["labs"]] == ["fixture", "broken"]
    assert report["summary"]["decisions"] == {"accepted": 0, "rejected": 1, "needs_review": 1}
    assert report["labs"][0]["human_review"]["status"] == "invalid"


def test_cli_writes_pending_review_without_overwriting_inputs_or_existing_reviews(tmp_path, capsys):
    env = fixture(tmp_path / "envs")
    output, draft = tmp_path / "admission.json", tmp_path / "review.json"
    main(["admission", "inspect", str(env), "--output", str(output), "--draft-review", str(draft)])
    report = json.loads(output.read_text())
    assert json.loads(capsys.readouterr().out)["decision"] == "needs_review"
    pending = json.loads(draft.read_text())
    assert pending["reviewer"]["kind"] == "unassigned"
    assert set(pending["checks"]) == set(REVIEW_CHECKS)
    assert inspect_lab(env, review=pending)["decision"] == "needs_review"
    with pytest.raises(SystemExit, match="File exists"):
        main(["admission", "inspect", str(env), "--draft-review", str(draft), "--output", str(output)])
    assert json.loads(draft.read_text()) == pending
    with pytest.raises(ValueError, match="write admission outputs"):
        write_json(ROOT / "results/summary.json", report)


def test_llm_negative_flags_are_evidence_to_review_not_an_automatic_verdict(tmp_path):
    env = fixture(tmp_path)
    (env / "audit.json").write_text(json.dumps({"ok": True, "auditor": "synthetic-model", "baseline_fair": False,
                                                "fidelity": "low", "issues": ["A synthetic concern requiring human examination."]}))
    first = inspect_lab(env)
    assert first["decision"] == "needs_review"
    review = human_control(first)
    review["checks"]["baseline_fairness"]["evidence"][0]["finding"] = "In this synthetic control the concern is a deliberate false positive."
    assert inspect_lab(env, review=review)["decision"] == "accepted"
