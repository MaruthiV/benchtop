"""Consolidate public admission evidence; do not execute labs or create human decisions."""

import argparse
import copy
import json
import os
import re
from pathlib import Path

from benchmarks.release_v1.admission.check_defects import MUON, load_manifest
from benchmarks.release_v1.admission.check_references import check as check_references
from benchtop.admission import REVIEW_CHECKS, draft_review, screen_catalog
from benchtop.cache import digest, file_hash, runtime_identity
from benchtop.env import ROOT

DIRECTORY = Path(__file__).resolve().parent
RUNTIME_CONTROLS = ("valid_treatment", "valid_holdout", "valid_shared_inputs", "observer_transparency",
                    "fresh_worker_repeatability", "zero_radius_identity")


def _json(path):
    return json.loads(Path(path).read_text())


def _core(screen):
    keys = ("env", "input_sha256", "generator_sha256", "mechanical", "qualification", "review_subject")
    return {"catalog_sha256": screen["catalog_sha256"], "catalog_id": screen["catalog_id"],
            "labs": [{key: row.get(key) for key in keys} for row in screen["labs"]]}


def fixture_snapshot(path):
    """Verify saved input bindings and accounting; do not rerun or authenticate measurements."""
    report = _json(path)
    manifest = load_manifest()
    required = {f"envs/{env}/{name}" for env, files in manifest["sources"].items() for name in files}
    required.update(f"benchtop/{name}" for name in
                    ("admission.py", "cache.py", "env.py", "tasks.py", "stats.py", "space.py", "sandbox.py", "_worker.py"))
    required.update(f"benchmarks/release_v1/admission/{name}" for name in
                    ("check_defects.py", "defects/manifest.json", "defects/muon_observer.py", "references/selection.json"))
    required.add("envs/historical_cache.json")
    if (report.get("schema_version") != 1 or report.get("suite_id") != manifest["suite_id"]
            or report.get("status") not in ("passed", "partial", "failed")
            or not required <= report["input_sha256"].keys()):
        raise ValueError("unsupported fixture evidence or missing input bindings")
    for name, expected in report["input_sha256"].items():
        target = (ROOT / name).resolve()
        if Path(name).is_absolute() or not target.is_relative_to(ROOT) or file_hash(target) != expected:
            raise ValueError(f"stale or invalid fixture evidence input: {name}")
    cases = report["cases"]
    if len({row["id"] for row in cases}) != len(cases):
        raise ValueError("duplicate fixture evidence case")
    failures = [row["id"] for row in cases if row["expected"] != row["observed"] or row["result"] != "pass"]
    summary = report["summary"]
    observed = {
        "cases": len(cases), "expectations_met": len(cases) - len(failures), "failed_cases": failures,
        "controlled_defects_observed": sum(row["kind"] == "defect" for row in cases),
        "valid_controls": sum(row["kind"] == "control" for row in cases),
        "sandbox_jobs": sum(len(run["observations"]) for run in report["live_runs"]),
    }
    if any(summary.get(key) != value for key, value in observed.items()):
        raise ValueError("fixture evidence accounting does not match its cases/observations")
    if report["status"] == "passed" and (failures or not summary["source_inputs_unchanged"]
                                         or observed["controlled_defects_observed"] != 6
                                         or observed["valid_controls"] != 10 or observed["sandbox_jobs"] != 24):
        raise ValueError("fixture evidence claims a full pass without the complete covered suite")
    controls = {row["id"]: row for row in cases if row["id"] in RUNTIME_CONTROLS}
    bounded = report["status"] == "passed" and set(controls) == set(RUNTIME_CONTROLS) and all(
        row["result"] == "pass" and row["observed"] == "pass" for row in controls.values())
    return {"source": "defects/report.json", "sha256": file_hash(path), "checked_at_utc": report["checked_at_utc"],
            "status": report["status"], "summary": summary, "cases": cases, "bounded_muon_evidence": bounded,
            "runtime_controls": list(controls.values()),
            "limits": report["limits"] + ["This builder checks stored bindings/accounting, not the authenticity of measurements."]}, report


def build_report(*, reviews_dir=None):
    check_references()
    base = screen_catalog()
    stored = _json(DIRECTORY / "screen.json")
    if digest(_core(base)) != digest(_core(stored)):
        raise ValueError("stored catalog screen is stale; review and regenerate it before consolidating")
    if reviews_dir is not None and not Path(reviews_dir).is_dir():
        raise ValueError("reviews-dir must be an existing directory")
    screen = screen_catalog(reviews_dir=reviews_dir) if reviews_dir is not None else base
    by_id = {row["env"]: row for row in screen["labs"]}
    fixtures, raw_fixtures = fixture_snapshot(DIRECTORY / "defects/report.json")
    selection = _json(DIRECTORY / "references/selection.json")
    spec = _json(DIRECTORY.parent / "spec.json")
    inputs = {}
    for path in (Path(__file__).resolve(), DIRECTORY / "screen.json", DIRECTORY / "check_references.py",
                 DIRECTORY / "references/selection.json", DIRECTORY / "defects/report.json",
                 DIRECTORY / "REVIEW.md", ROOT / "envs/catalog.json", DIRECTORY.parent / "spec.json"):
        inputs[str(path.relative_to(ROOT))] = file_hash(path)
    inputs.update(raw_fixtures["input_sha256"])
    for row in base["labs"]:
        inputs.update({f"envs/{row['env']}/{name}": sha for name, sha in row["input_sha256"].items()})
        inputs.update(row["generator_sha256"])
        if reviews_dir is not None:
            review_path = Path(reviews_dir) / f"{row['env']}.json"
            if review_path.is_file():
                inputs[f"supplied_review:{row['env']}"] = file_hash(review_path)
    references = []
    for selected in selection["cases"]:
        env = selected["env"]
        path = DIRECTORY / "references/drafts" / f"{env}.json"
        draft = _json(path)
        inputs[str(path.relative_to(ROOT))] = file_hash(path)
        row = by_id[env]
        if env == MUON:
            runtime = {"status": "bounded" if fixtures["bounded_muon_evidence"] else "insufficient",
                       "source": "defects/report.json", "checked_at_utc": fixtures["checked_at_utc"],
                       "profile": _json(DIRECTORY / "defects/manifest.json")["muon"],
                       "controls": fixtures["runtime_controls"],
                       "limits": ["Two visible seeds and one context, plus zero radius; no independent reward-stability sweep.",
                                  "Controls concern the original Muon variant; intentional mutants are demonstration cases.",
                                  "Numerical, fairness, margin, and paper-scope questions still need review."]}
            runtime["profile"] = {k: runtime["profile"][k] for k in ("env", "seeds", "config")}
        else:
            runtime = {"status": "not_run_for_this_lab", "controls": [],
                       "limits": ["Cached diagnostics and controlled cache/counterfactual cases are not new executions of this lab."]}
        references.append({
            "env": env, "title": selected["title"], "coverage": selected["coverage"],
            "metric": _json(ROOT / "envs" / env / "space.json")["metric"],
            "subject": row["review_subject"], "mechanical": row["mechanical"]["status"],
            "qualification": row["qualification"], "diagnostics": selected["diagnostics"],
            "decision": row["decision"], "human_review": row["human_review"],
            "qualified_claim_admitted": row["qualified_claim_admitted"],
            "assistant_draft": {"source": f"references/drafts/{env}.json", **draft},
            "runtime_evidence": runtime,
            "remaining_questions": draft["assistant_assessment"]["owner_actions"] + [
                "Give an independent practical rationale for the declared effect margin; do not choose it from the observed interval.",
                "Record your own criterion judgments and decision; keep missing runtime or reward-stability evidence unresolved."],
        })
    catalog_rows = [{"env": row["env"], "mechanical": row["mechanical"]["status"],
                     "claim_eligible": row["qualification"].get("claim_eligible", False), "decision": row["decision"],
                     "human_review_status": row["human_review"]["status"],
                     "qualified_claim_admitted": row["qualified_claim_admitted"]} for row in screen["labs"]]
    eligible = [row["env"] for row in references if row["qualified_claim_admitted"]]
    return {
        "schema_version": 1, "report_id": "admission_reference_report_v1",
        "scope": "Historical reference evidence and review preparation; no fresh cohort or development lock.",
        "analysis_runtime": runtime_identity(), "input_sha256": inputs,
        "protocol_id": spec["protocol_id"], "catalog_summary": screen["summary"], "catalog": catalog_rows,
        "reviews_supplied": reviews_dir is not None,
        "reference_summary": {"labs": len(references),
                              "decisions": {name: sum(row["decision"] == name for row in references)
                                            for name in ("accepted", "rejected", "needs_review")},
                              "recorded_human_reviews": sum(row["human_review"]["status"] == "recorded" for row in references)},
        "references": references, "fixtures": fixtures,
        "development": {"required_labs": spec["development"]["claim_labs"], "eligible_reference_ids": eligible,
                        "selected_lab_ids": [], "lock": "not_created_by_this_report",
                        "selection_rule": spec["development"]["claim_selection"],
                        "next_gate": "Human admission and qualified claims, then explicit development/model/prompt/version bindings."},
        "model_difficulty": "not_measured", "new_model_sessions": 0, "new_lab_executions": 0,
        "limits": ["Assistant suggestions and recorded fixture outcomes cannot substitute for a human scientific decision.",
                   "Reviewer identity and citation contents are operator-reported, not authenticated.",
                   "The CaLLM comparison concern is from its static draft; the controlled test-set leak belongs to a separate Muon mutant.",
                   "Historical qualification is retrospective development evidence, not fresh confirmation.",
                   "Qualia UI use, broader runtime evidence, practical margins, and the remaining release gates are not established by this report."],
    }


def _link(source):
    if source.startswith(("https://", "http://")):
        return source
    return os.path.relpath(ROOT / source, DIRECTORY)


def _citations(evidence):
    if not isinstance(evidence, list):
        return "No structured evidence supplied."
    citations = []
    for item in evidence:
        if not isinstance(item, dict) or any(not isinstance(item.get(key), str) for key in ("source", "locator", "finding")):
            citations.append("Incomplete evidence record.")
            continue
        finding = item["finding"].replace("|", "\\|").replace("\n", " ")
        source = item["source"]
        label = item["locator"].replace("|", "\\|")
        if source.startswith(("https://", "http://")) or (ROOT / source).is_file():
            citations.append(f"[{label}]({_link(source)}): {finding}")
        else:
            citations.append(f"{source} ({label}): {finding}")
    return "<br>".join(citations)


def render_markdown(report, directory=DIRECTORY):
    summary = report["catalog_summary"]
    lines = ["# Admission evidence report", "", "Generated by `build_report.py` from public inputs. This report creates no human decisions or development lock.", "",
             f"The catalog has {summary['labs']} labs, {summary['mechanically_passed']} mechanical passes, "
             f"{summary['qualified_claims_before_review']} corrected qualified claims, and {summary['qualified_claims_admitted']} admitted qualified claims.", "",
             "Read [the owner review guide](REVIEW.md) to create separate, initially unassigned records. The original assistant drafts stay intact.", "",
             "## Reference decisions", "",
             "Positive effects favor the designated paper arm. Each effect is in its lab's own metric units; effect sizes are not comparable across labs. These are corrected historical qualification results, not paper replications or fresh confirmation.", "",
             "| Reference | Cached label | Effect [95% interval] | Actual decision | Human record | Newer runtime evidence |",
             "|---|---|---|---|---|---|"]
    for row in report["references"]:
        claim = row["qualification"]["claim"]
        lo, hi = claim["interval"]
        lines.append(f"| [{row['title']}](#{row['env']}) | {claim['label']} | {claim['delta']:.6g} [{lo:.6g}, {hi:.6g}] | "
                     f"{row['decision']} | {row['human_review']['status']} | {row['runtime_evidence']['status']} |")
    lines += ["", "Static check statuses below are the assistant's suggestions, not human passes. Runtime evidence is additive and does not update those suggestions automatically.", "",
              "## Controlled cases", "",
              f"The saved [fixture report](defects/README.md) is **{report['fixtures']['status']}**: "
              f"{report['fixtures']['summary']['expectations_met']}/{report['fixtures']['summary']['cases']} expected outcomes, "
              f"{report['fixtures']['summary']['sandbox_jobs']} sandbox jobs in that recorded run. The builder executes zero jobs.", "",
              "| Case | Kind | Guard | Expected | Observed | Check |", "|---|---|---|---|---|---|"]
    for row in report["fixtures"]["cases"]:
        lines.append(f"| {row['id']} | {row['kind']} | `{row['guard']}` | {row['expected']} | {row['observed']} | {row['result']} |")
    lines += ["", "These are deliberately covered cases, not a general scientific-defect detection rate. An ambiguous interval makes a question unavailable without rejecting the science."]
    for row in report["references"]:
        draft = row["assistant_draft"]
        lines += ["", f"<a id=\"{row['env']}\"></a>", f"## {row['title']}", "",
                  f"Actual admission: **{row['decision']}**. Human record: **{row['human_review']['status']}**. "
                  f"Assistant recommendation: **{draft['assistant_assessment']['recommendation']}**.", "",
                  f"[Pinned paper]({draft['paper']['url']}) — {draft['paper']['version']}; sections: " + "; ".join(draft["paper"]["sections"]) + ".", "",
                  f"[Original assistant draft]({draft['source']}); [implementation](../../../envs/{row['env']}/lab.py).", "",
                  f"Metric: `{row['metric']['name']}`; raw goal: `{row['metric']['goal']}`; declared meaningful-effect margin: ±{row['metric']['min_effect']}. " + row['metric']['description'], "",
                  f"Paper mechanism (assistant summary): {draft['scope']['paper_mechanism']}", "",
                  f"Miniature comparison: {draft['scope']['miniature_comparison']}"]
        if row["human_review"]["status"] == "recorded":
            human = row["human_review"]
            lines += ["", f"**Human record:** {human['reviewer']['name']}, {human['reviewer']['reviewed_at_utc']} (operator-reported identity).", "",
                      "**Human rationale:** " + human["rationale"], "",
                      f"**Human paper reference:** [{human['paper']['version']}]({human['paper']['url']}); " + "; ".join(human["paper"]["sections"]) + "."]
            if isinstance(human["scope"], dict):
                for key, value in human["scope"].items():
                    if value:
                        lines += ["", f"Human scope — {key}: " + (" ".join(map(str, value)) if isinstance(value, list) else str(value))]
            lines += ["", "| Criterion | Human status | Human evidence |", "|---|---|---|"]
            for name in REVIEW_CHECKS:
                check = human["checks"][name]
                lines.append(f"| {name} | {check['status']} | {_citations(check.get('evidence', []))} |")
        for title, key in (("Preserved assumptions", "preserved_assumptions"), ("Simplifications", "simplifications"), ("Unsupported claims", "unsupported_claims")):
            lines += ["", f"**{title}:** " + " ".join(draft["scope"][key])]
        lines += ["", "| Criterion | Assistant status | Evidence |", "|---|---|---|"]
        for name in REVIEW_CHECKS:
            check = draft["checks"][name]
            lines.append(f"| {name} | {check['status']} | {_citations(check['evidence'])} |")
        lines += ["", "**Newer runtime evidence:** " + row["runtime_evidence"]["status"] + "."]
        if row["runtime_evidence"]["controls"]:
            profile = row["runtime_evidence"]["profile"]
            lines += ["", f"[Saved Muon controls](defects/report.json), visible seeds {profile['seeds']}, context `{json.dumps(profile['config'], sort_keys=True)}` plus zero radius. "
                      "Pristine-observer agreement and fresh-worker repetition are bounded to this profile."]
            lines += ["", "Controls: " + ", ".join(f"`{item['id']}` ({item['observed']})" for item in row["runtime_evidence"]["controls"]) + "."]
        lines += ["", " ".join(row["runtime_evidence"]["limits"]), "", "**Owner questions:**", ""]
        lines += [f"- [ ] {question}" for question in row["remaining_questions"]]
        if row["human_review"].get("issues"):
            lines += ["", "Current review issues: " + "; ".join(row["human_review"]["issues"]) + "."]
    lines += ["", "## Development and remaining gates", "",
              "Qualified, human-admitted reference IDs: " + (", ".join(report["development"]["eligible_reference_ids"]) or "none") + ". No development IDs or locks are created here.", "",
              report["development"]["selection_rule"], "",
              "Human reviews and practical margins remain separate from fixture results. Stronger classical policies can be developed independently before locked model budget-selection runs. Qualia UI use, the fresh study, packaging, and outreach readiness remain later gates.", "",
              "## Limits", ""]
    lines += [f"- {item}" for item in report["limits"]]
    lines += ["", "Reproduce offline from the repository root:", "", "```sh",
              ".venv/bin/python -m benchmarks.release_v1.admission.build_report \\",
              "  --output-dir artifacts/admission_report", "```", ""]
    markdown = "\n".join(lines)

    def relocate(match):
        source = match[1]
        if source.startswith(("https://", "http://", "#")):
            return match[0]
        path, marker, fragment = source.partition("#")
        target = os.path.relpath((DIRECTORY / path).resolve(), Path(directory).resolve())
        return f"]({target}{marker}{fragment})"

    return re.sub(r"\]\(([^)]+)\)", relocate, markdown)


def _output_directory(directory, *, public=False):
    target = Path(directory).resolve()
    if target.is_relative_to(ROOT) and not target.is_relative_to(ROOT / "artifacts") and not (public and target == DIRECTORY):
        raise ValueError("choose artifacts/, the public admission report directory, or a directory outside the repository")
    return target


def write_report(report, directory):
    target = _output_directory(directory, public=True)
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (target / "REPORT.md").write_text(render_markdown(report, target))


def prepare_reviews(report, directory):
    target = _output_directory(directory)
    paths = [target / f"{row['env']}.json" for row in report["references"]]
    if any(path.exists() for path in paths):
        raise FileExistsError("review preparation never overwrites an existing record; choose an empty destination")
    target.mkdir(parents=True, exist_ok=True)
    for index, (row, path) in enumerate(zip(report["references"], paths, strict=True)):
        review = draft_review({"review_subject": row["subject"]})
        review["paper"] = copy.deepcopy(row["assistant_draft"]["paper"])
        review["scope"] = copy.deepcopy(row["assistant_draft"]["scope"])
        for name in REVIEW_CHECKS:
            review["checks"][name]["evidence"] = copy.deepcopy(row["assistant_draft"]["checks"][name]["evidence"])
        if row["runtime_evidence"]["status"] == "bounded":
            review["checks"]["execution_behavior"]["evidence"].append({
                "source": "benchmarks/release_v1/admission/report.json", "locator": f"/references/{index}/runtime_evidence",
                "finding": "Bounded Muon observations are available; numerical, broader-context, and reward-stability questions remain for review."})
        review["assistant_context"] = {"draft_source": row["assistant_draft"]["source"], "report_id": report["report_id"],
                                       "warning": "Prefilled scope and citations are assistant suggestions. All criterion judgments are unresolved; record your own review."}
        with path.open("x") as output:
            output.write(json.dumps(review, indent=2, allow_nan=False) + "\n")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/admission_report")
    parser.add_argument("--reviews-dir", help="separate completed or in-progress owner records; invalid ones remain visible")
    parser.add_argument("--prepare-reviews", help="create six unassigned records in an empty scratch destination")
    args = parser.parse_args()
    _output_directory(args.output_dir, public=True)
    report = build_report(reviews_dir=args.reviews_dir)
    if args.prepare_reviews:
        prepare_reviews(report, args.prepare_reviews)
    write_report(report, args.output_dir)
    print(json.dumps({"catalog": report["catalog_summary"], "references": report["reference_summary"],
                      "new_lab_executions": 0, "new_model_sessions": 0, "output_dir": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()
