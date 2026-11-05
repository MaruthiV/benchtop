import copy
import json
import re

import pytest

from benchmarks.release_v1.admission import build_report as reports
from benchtop.admission import REVIEW_CHECKS, inspect_lab
from benchtop.cache import file_hash
from benchtop.env import ROOT
from tests.test_admission import human_control


@pytest.fixture(scope="module")
def packet():
    return reports.build_report()


def test_report_separates_science_qualification_assistant_and_runtime_evidence(packet):
    assert packet["catalog_summary"]["labs"] == 30
    assert packet["catalog_summary"]["qualified_claims_before_review"] == 28
    assert packet["reference_summary"]["decisions"] == {"accepted": 0, "rejected": 0, "needs_review": 6}
    assert packet["reference_summary"]["recorded_human_reviews"] == 0
    rows = {row["env"]: row for row in packet["references"]}
    callm = next(row for row in rows.values() if "callm-" in row["env"])
    assert callm["qualification"]["claim"]["label"] == "paper_better"
    assert callm["decision"] == "needs_review"
    assert callm["assistant_draft"]["assistant_assessment"]["recommendation"] == "reject_current_comparison"
    assert callm["runtime_evidence"]["status"] == "not_run_for_this_lab"
    assert rows[reports.MUON]["runtime_evidence"]["status"] == "bounded"
    assert all(row["assistant_draft"]["checks"]["execution_behavior"]["status"] == "unresolved" for row in rows.values())
    assert packet["development"]["eligible_reference_ids"] == packet["development"]["selected_lab_ids"] == []
    assert packet["new_lab_executions"] == packet["new_model_sessions"] == 0


@pytest.mark.parametrize("change", ["stale", "missing_bindings", "accounting", "escape"])
def test_fixture_evidence_cannot_be_rebound_or_falsely_counted(tmp_path, change):
    fixture = json.loads((reports.DIRECTORY / "defects/report.json").read_text())
    if change == "stale":
        fixture["input_sha256"]["benchtop/cache.py"] = "0" * 64
    elif change == "missing_bindings":
        fixture["input_sha256"] = {}
    elif change == "accounting":
        fixture["summary"]["expectations_met"] += 1
    else:
        fixture["input_sha256"]["../outside-repository"] = "0" * 64
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture))
    with pytest.raises(ValueError):
        reports.fixture_snapshot(path)


@pytest.mark.parametrize("status", ["partial", "failed"])
def test_nonpassing_fixture_report_never_supplies_bounded_runtime_pass(tmp_path, status):
    fixture = json.loads((reports.DIRECTORY / "defects/report.json").read_text())
    fixture["status"] = status
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture))
    snapshot, _ = reports.fixture_snapshot(path)
    assert snapshot["status"] == status
    assert not snapshot["bounded_muon_evidence"]


def test_unassigned_packet_never_impersonates_a_reviewer_or_overwrites_records(tmp_path, packet):
    destination = tmp_path / "reviews"
    paths = reports.prepare_reviews(packet, destination)
    assert len(paths) == 6
    for path in paths:
        record = json.loads(path.read_text())
        assert record["reviewer"] == {"kind": "unassigned", "name": "", "reviewed_at_utc": ""}
        assert record["decision"] == "needs_review"
        assert all(record["checks"][name]["status"] == "unresolved" for name in REVIEW_CHECKS)
        inspected = inspect_lab(ROOT / "envs" / path.stem, review=record)
        assert inspected["human_review"]["status"] == "draft"
        assert not inspected["qualified_claim_admitted"]
    before = {path.name: path.read_bytes() for path in paths}
    with pytest.raises(FileExistsError):
        reports.prepare_reviews(packet, destination)
    assert {path.name: path.read_bytes() for path in paths} == before
    assert json.loads((destination / f"{reports.MUON}.json").read_text())["checks"]["execution_behavior"]["evidence"][-1]["source"].endswith("admission/report.json")


def test_explicit_simulated_human_decisions_are_reported_without_creating_a_lock(tmp_path, packet):
    """Temporary fixture identities test aggregation; no actual human review is asserted."""
    accepted, rejected, stale, malformed = [row["env"] for row in packet["references"][:4]]
    for env in (accepted, rejected, stale):
        record = human_control(inspect_lab(ROOT / "envs" / env))
        if env == rejected:
            record["decision"] = "rejected"
            record["checks"]["baseline_fairness"]["status"] = "fail"
            record["rationale"] = "Simulated rejection: test evidence only."
        if env == stale:
            record["subject"]["lab_sha256"] = "0" * 64
        (tmp_path / f"{env}.json").write_text(json.dumps(record))
    (tmp_path / f"{malformed}.json").write_text("not JSON")
    report = reports.build_report(reviews_dir=tmp_path)
    rows = {row["env"]: row for row in report["references"]}
    assert rows[accepted]["decision"] == "accepted"
    assert rows[rejected]["decision"] == "rejected"
    assert rows[stale]["human_review"]["status"] == "stale"
    assert rows[malformed]["human_review"]["status"] == "invalid"
    assert report["reference_summary"]["recorded_human_reviews"] == 2
    assert report["development"]["eligible_reference_ids"] == [accepted]
    assert report["development"]["selected_lab_ids"] == []
    assert reports.render_markdown(report).count("**Human rationale:**") == 2
    assert "Simulated rejection: test evidence only." in reports.render_markdown(report)


def test_generated_scratch_links_resolve_and_input_locations_are_protected(tmp_path, packet):
    output = tmp_path / "report"
    reports.write_report(packet, output)
    for source in re.findall(r"\]\(([^)]+)\)", (output / "REPORT.md").read_text()):
        if not source.startswith(("https://", "#")):
            assert (output / source.split("#")[0]).resolve().exists(), source
    before = file_hash(reports.DIRECTORY / "references/selection.json")
    with pytest.raises(ValueError, match="choose artifacts"):
        reports.write_report(packet, reports.DIRECTORY / "references")
    with pytest.raises(ValueError, match="choose artifacts"):
        reports.prepare_reviews(packet, reports.DIRECTORY / "references/drafts")
    assert file_hash(reports.DIRECTORY / "references/selection.json") == before


def test_pending_owner_record_can_keep_unstructured_evidence_unresolved(tmp_path, packet):
    env = packet["references"][0]["env"]
    record = human_control(inspect_lab(ROOT / "envs" / env))
    record["decision"] = "needs_review"
    record["checks"]["execution_behavior"] = {"status": "unresolved", "evidence": None}
    record["scope"] = {"pending_notes": [1, 2]}
    (tmp_path / f"{env}.json").write_text(json.dumps(record))
    report = reports.build_report(reviews_dir=tmp_path)
    assert report["references"][0]["decision"] == "needs_review"
    assert report["development"]["eligible_reference_ids"] == []
    markdown = reports.render_markdown(report)
    assert "No structured evidence supplied." in markdown
    assert "execution_behavior | unresolved" in markdown


def test_report_rendering_does_not_mutate_the_evidence(packet):
    before = copy.deepcopy(packet)
    reports.render_markdown(packet)
    assert packet == before
