import copy
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from benchtop import (
    admission,
    cache,
    confirmation_analysis,
    confirmation_data,
    confirmation_replay,
    confirmation_results,
    pilot_calibration,
    primary,
    protocol,
    release_data,
    study,
    study_calibration,
)
from benchtop import session_ledger as ledger
from benchtop.env import ROOT
from benchtop.generation_journal import read as generation_state
from benchtop.verify import PUBLIC_DIRS, PUBLIC_FILES

FORMAT = "benchtop_public_release_v1"
MANIFEST = "release-manifest.json"


def _binding_name():
    return str(protocol.BINDINGS.relative_to(ROOT))


def _files(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError("publication files cannot be symbolic links")
        if path.is_file():
            result[str(path.relative_to(directory))] = cache.file_hash(path)
    return result


def public_inputs():
    result = {}
    for name in PUBLIC_FILES:
        path = ROOT / name
        if path.is_symlink():
            raise ValueError("frozen public inputs cannot be symbolic links")
        result[name] = cache.file_hash(path)
    for name in PUBLIC_DIRS:
        for path, sha in _files(ROOT / name).items():
            relative = str(Path(name) / path)
            if any(part in (".state", ".git", "artifacts") or part.startswith(".env") for part in Path(relative).parts):
                raise ValueError("private material appeared inside frozen public inputs")
            result[relative] = sha
    if primary.BINDINGS.exists():
        result[_binding_name()] = cache.file_hash(primary.BINDINGS)
    return result


def _private_inventory(root):
    return study.canonical_hash({str(path.relative_to(root)): cache.file_hash(path) for path in sorted(root.rglob("*"))
                                if path.is_file() and not path.name.startswith(".")})


def status():
    root = primary.canonical_root()
    reviews = ROOT / "artifacts/reference-human-reviews"
    reference = []
    for path in sorted(reviews.glob("*.json")):
        value = study._json(path)
        checked = admission.inspect_lab(ROOT / "envs" / path.stem, review=value)["human_review"]
        if checked["status"] == "recorded":
            reference.append(path.name)
    gates = {"six_recorded_reference_reviews": len(reference) == 6, "canonical_primary_journal": (root / "ledger/ledger.json").exists(),
        "confirmation_closed": (root / confirmation_results.COLLECTION).exists(), "classical_collection_complete": False,
        "observed_qualia_connectivity": False}
    issues = []
    if gates["canonical_primary_journal"]:
        try:
            ledger.inspect(root / "ledger")
            ledger._current(ledger._manifest(root / "ledger"))
            if gates["confirmation_closed"]:
                confirmation_results.load_closed()
                gates["classical_collection_complete"] = confirmation_replay.inspect()["status"] == "complete"
            if (root / "qualia_authorization.json").exists():
                from benchtop import qualia

                for path in (root / "qualia/connections").glob("*.json"):
                    episode = path.stem
                    if qualia._capture_for_episode_exists(episode):
                        gates["observed_qualia_connectivity"] |= qualia._export(episode)["qualia_verified"]
        except (ValueError, OSError, KeyError) as error:
            issues.append(str(error))
    return {"status": "scoring_ready_for_package" if gates["confirmation_closed"] and gates["classical_collection_complete"] and not issues else "not_ready",
        "gates": gates, "issues": issues, "actual_demo_observed_by_packager": False, "email_ready": False,
        "next_step": "Record the completed source-bound reference reviews." if not gates["six_recorded_reference_reviews"] else
                     "Follow EXECUTION.md from the first missing gate; preserve the single canonical journal.",
        "model_sessions_started": 0, "scope": "Read-only recorded-state inspection; no human decisions, UI pass or outreach approval inferred."}


def _review_evidence(row):
    result = {key: row[key] for key in ("locator", "finding", "sha256") if key in row}
    source = row.get("source", row.get("path"))
    if source is not None:
        try:
            result["source"] = release_data.safe_text(source)
        except ValueError:
            result["original_private_source_reference_sha256"] = study.canonical_hash(source)
    return result


def _review(value):
    if value is None:
        return None
    result = {key: copy.deepcopy(value[key]) for key in ("schema_version", "decision", "paper", "scope", "rationale", "margin", "adaptive_allocation") if key in value}
    result["reviewer"] = {key: value.get("reviewer", {}).get(key) for key in ("kind", "name", "reviewed_at_utc")}
    result["checks"] = {}
    for name, check in value.get("checks", {}).items():
        item = {key: check[key] for key in ("status", "judgment", "rationale") if key in check}
        item["evidence"] = [_review_evidence(row) for row in check.get("evidence", [])]
        result["checks"][name] = item
    result["original_record_sha256"] = study.canonical_hash(value)
    result["projection_note"] = "Human assertions retained; private subjects/paths and contact fields omitted. This projection is not the original signed or authenticated review."
    return release_data.safe_text(result)


def _gather():
    root = primary.canonical_root()
    with ledger._locked(root / "ledger") as context:
        directory, manifest = context
        ledger._current(manifest)
        state = ledger._read(directory, manifest)
        if state["counts"]["active"] or generation_state()["active"]:
            raise ValueError("stop and reconcile active evaluation/generation before building a publication package")
        collection = confirmation_results.load_closed(ledger_context=context)
        plan = state["plans"][collection["plan_sha256"]]
        assessment = confirmation_data.load_assessment(ledger_context=context)
        development = study_calibration.load_closed(ledger_context=context)
        calibration = {"development": {key: copy.deepcopy(development[key]) for key in ("input", "selection")},
            "development_snapshot_sha256": development["snapshot_sha256"], "pilots": []}
        if plan["binding"]["environments"]:
            pilots = pilot_calibration.load_closed(ledger_context=context)
            calibration.update(pilots=copy.deepcopy(pilots["labs"]), pilot_snapshot_sha256=pilots["snapshot_sha256"])
        cohort = study._json(root / "cohort_lock.json")
        candidate = study._json(root / "candidate_lock.json")
        smoke = study._json(root / "smoke_plan.json")
        foundation = study._json(study._resolve(smoke["binding"]["foundation_source"]))
        if study._foundation(study._resolve(smoke["binding"]["foundation_source"]))["status"] != "current":
            raise ValueError("publication requires the original current foundation inputs")
        inputs = public_inputs()
        allowed = set(foundation["input_sha256"]) | {_binding_name()}
        if set(inputs) != allowed or any(inputs[name] != sha for name, sha in foundation["input_sha256"].items()):
            raise ValueError("public input set changed after the foundation snapshot; preserve the frozen evaluation workspace")
        baseline = cohort["binding"]["baselines"]
        reviews = {"fresh": {}, "references": {}, "baseline": _review(candidate["binding"]["baselines"].get("review"))}
        for member in plan["binding"]["environments"]:
            path = study._resolve(member["source"]).parent / "human_review.json"
            reviews["fresh"][member["env_id"]] = _review(study._json(path))
        for path in sorted(study._resolve(smoke["binding"]["reviews_source"]).glob("*.json")):
            reviews["references"][path.stem] = _review(study._json(path))
        bindings = study._json(primary.BINDINGS)
        identity = {"model_configuration": plan["binding"]["model_configuration"], "runtime": plan["binding"]["runtime"],
            "source_sha256": plan["binding"]["source_sha256"], "foundation_file_sha256": cache.file_hash(study._resolve(smoke["binding"]["foundation_source"])),
            "bindings": bindings, "ordered_candidates": [{key: row[key] for key in ("id", "version", "title", "categories", "published")}
                for row in candidate["binding"]["candidates"]], "source_scope": candidate.get("scope", "Canonical operator records; identities are not independently authenticated."),
            "private_inventory_sha256": _private_inventory(root)}
    classical = confirmation_replay.inspect()
    if classical["status"] != "complete":
        raise ValueError("complete the fixed classical collection before publication; no trials are run by the builder")
    return collection, classical, plan, assessment, baseline, reviews, inputs, identity, calibration


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    study._publish(path, value)


def _copy_labs(directory, plan):
    provenance = {}
    for member in plan["binding"]["environments"]:
        source = study._resolve(member["source"])
        target = directory / member["env_id"]
        env = _environment(source)
        for name in ("lab.py", "space.json", "cache.json", "tasks.json"):
            path = target / "qualification" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, path)
            release_data.safe_text(path.read_text())
        _write(target / "qualification/manifest.json", release_data.manifest_projection(study._json(source / "manifest.json")))
        rows = release_data.rows_projection(source, env.execution, env.space)
        (target / "qualification/results.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
        confirmed = primary.canonical_root() / "confirmation_data/labs" / member["env_id"]
        wrapped = study._json(confirmed / "cache.json")
        _write(target / "confirmation/cache.json", wrapped)
        _write(target / "confirmation/manifest.json", release_data.manifest_projection(study._json(confirmed / "manifest.json")))
        rows = release_data.rows_projection(confirmed, wrapped["execution"], env.space)
        (target / "confirmation/results.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
        provenance[member["env_id"]] = {"original_qualification_files": member["files"],
            "original_confirmation_results_sha256": cache.file_hash(confirmed / "results.jsonl"),
            "derived_data_note": "Manifests and result rows are sanitized projections, resealed as new rows with original source-row hashes. Lab code, space, task artifacts and execution bindings are unchanged."}
    return provenance


def _environment(path):
    from benchtop.env import Env

    return Env.load(path, replay=True)


def figures(report, directory):
    directory.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="benchtop-public-plot-") as temporary:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = temporary
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            fig = Figure(figsize=(10, max(3.6, 0.48 * len(report["labs"]) + 2.3)), layout="constrained", facecolor="white")
            FigureCanvasAgg(fig)
            axes = fig.subplots(1, 2)
            labs = report["labs"]
            for axis in axes:
                axis.spines[["top", "right"]].set_visible(False)
                axis.tick_params(labelsize=9)
            axes[0].set_title("Complete mixed groups", loc="left", fontsize=12)
            axes[0].set_xlim(-0.15, 1.15)
            axes[0].set_xticks([0, 1], ["all rewards equal", "both 0 and 1"])
            axes[0].set_yticks(range(len(labs)), [lab["env_id"] for lab in labs])
            axes[0].set_ylim(-0.6, max(0.6, len(labs) - 0.4))
            for index, lab in enumerate(labs):
                for offset, arm, color in ((-0.17, "global_budget", "#2767b0"), (0.17, "calibrated_budget", "#d16629")):
                    value = lab["arms"][arm]["mixed"]
                    if value is None:
                        axes[0].text(0.5, index + offset, ("G" if arm == "global_budget" else "C") + ": unavailable", ha="center", va="center", color=color, fontsize=8)
                    else:
                        axes[0].scatter([value], [index + offset], color=color, s=45)
            axes[0].scatter([], [], color="#2767b0", label="global")
            axes[0].scatter([], [], color="#d16629", label="calibrated")
            axes[0].legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
            axes[1].set_title("Frozen experimental budgets", loc="left", fontsize=12)
            axes[1].barh(range(len(labs)), [lab["selected_budget"] for lab in labs], height=0.55, color="#d16629")
            axes[1].axvline(report["efficiency"]["global_budget_fraction"], color="#2767b0", linewidth=1.5, label="global budget")
            axes[1].set_yticks(range(len(labs)), [lab["env_id"] for lab in labs])
            axes[1].set_xlim(0, 0.28)
            axes[1].set_xlabel("fraction of the frozen full-grid cost", fontsize=9)
            axes[1].legend(frameon=False, fontsize=9)
            if not labs:
                for axis in axes:
                    axis.text(0.5, 0.5, "No admitted labs; comparison unavailable", transform=axis.transAxes, ha="center", va="center", fontsize=10)
            fig.suptitle("Benchtop confirmation • " + str(report["denominators"]["complete_paired_labs"]) + " complete paired labs", fontsize=14, fontweight="bold")
            fig.savefig(directory / "confirmation.png", dpi=160)
            fig.savefig(directory / "confirmation.svg", metadata={"Date": None})
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def _readme(report, identity):
    primary_result = report["primary"]
    estimate = "unavailable" if primary_result["estimate"] is None else f"{primary_result['estimate']:.3f}"
    command = ("BENCHTOP_PROTOCOL=" + protocol.NAME + " ") if protocol.NAME != "release_v1" else ""
    return "\n".join(["# Benchtop", "", "Paper-inspired computational labs, independent reward checks, and model-specific budget calibration.", "",
        "This release preserves the full cohort, missing outcomes and measured overhead. It contains a sanitized projection of the recorded study, cached lab data and an offline reproducer.", "",
        "## Recorded result", "", f"Locked labs: **{report['denominators']['locked_labs']}**. Stable: **{report['denominators']['stable_labs']}**. Complete pairs: **{report['denominators']['complete_paired_labs']}**.", "",
        f"Calibrated minus global mixed-group frequency: **{estimate}**. See [the report](release-results/report.md) for the interval, full-cohort bounds, controls and limitations.", "",
        "![Confirmation groups and budgets](release-results/figures/confirmation.png)", "", "## Reproduce without models", "",
        "```sh", "uv sync --locked", command + "uv run python -m benchtop.cli release reproduce .", "```", "",
        "Reproduction checks file hashes, seed partitions, independent grading, recorded model rewards, completed classical policy trials and the paired analysis. It starts no model, network request or lab worker; recorded failures remain missing or failed.", "",
        "[Machine-readable analysis](release-results/analysis.json) · [projected evidence](release-results/evidence.json) · [execution runbook](benchmarks/release_v1/EXECUTION.md)", "",
        "## Evidence and scope", "", identity["source_scope"], "",
        "Raw provider transcripts, capabilities, account data and the private journal stay in the original evaluation workspace. Integrity hashes do not authenticate provider or human identities. The public evidence has new projection hashes and retains the original record hashes as provenance.", "",
        "Original evaluation inputs remain intact. The pre-publication README is preserved at [evaluation/README.md](evaluation/README.md); the manifest maps every frozen input to its unchanged bytes.", "",
        "This is an exploratory pilot on miniature labs, with at most eight labs. Mixed rewards do not establish learning or efficiency. Complete-case results are conditional on stable labels and complete groups.", "",
        "Actual Qualia observation, a public demonstration and the final factual outreach review remain separate from this reproducible data package.", ""])


def build(output):
    target = ledger._writable(output)
    if target.exists() or target.is_relative_to(primary.canonical_root().resolve()):
        raise ValueError("publication needs a new directory outside canonical study state")
    collection, classical, plan, assessment, baseline, reviews, inputs, identity, calibration = _gather()
    projection, report = release_data.project(collection, classical, plan, assessment, baseline, calibration)
    target.mkdir(parents=True, mode=0o700)
    marker = target / ".release-building"
    marker.write_text("incomplete publication; do not publish\n")
    for name in inputs:
        destination = target / ("evaluation/README.md" if name == "README.md" else name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = primary.BINDINGS if name == _binding_name() else ROOT / name
        shutil.copyfile(source, destination)
    provenance = _copy_labs(target / "release-data", plan)
    _write(target / "release-results/evidence.json", projection)
    _write(target / "release-results/analysis.json", report)
    _write(target / "release-results/reviews.json", reviews)
    (target / "release-results/report.md").write_text(confirmation_analysis.render_markdown(report))
    figures(report, target / "release-results/figures")
    (target / "README.md").write_text(release_data.safe_text(_readme(report, identity)))
    reproduced, checks = release_data.reproduce(projection, target / "release-data", replay_classical=False)
    release_data.compare(report, reproduced)
    if public_inputs() != inputs or _private_inventory(primary.canonical_root()) != identity["private_inventory_sha256"]:
        raise ValueError("evaluation inputs or private journal changed during publication; preserve the incomplete output")
    files = _files(target)
    files.pop(".release-building")
    manifest = study._seal({"schema_version": 1, "format": FORMAT, "recorded_at_utc": datetime.now(UTC).isoformat(),
        "protocol_hash": plan["binding"]["protocol_hash"], "protocol_profile": protocol.NAME, "source_sha256": identity["source_sha256"], "runtime": identity["runtime"],
        "original_public_inputs": inputs, "original_private_inventory_sha256": identity["private_inventory_sha256"],
        "foundation_file_sha256": identity["foundation_file_sha256"], "binding_records": identity["bindings"],
        "ordered_candidates": identity["ordered_candidates"], "membership_decisions": collection["candidate_funnel"],
        "environment_versions": [{key: member[key] for key in ("env_id", "version", "execution_id", "task_version", "seed_splits")} for member in plan["binding"]["environments"]],
        "data_projections": provenance, "projection_sha256": projection["projection_sha256"], "analysis_sha256": report["analysis_sha256"],
        "files": files, "build_checks": checks, "model_sessions_started_by_builder": 0,
        "scope": "Separate public data package. Original workspace and records unchanged; no private authority, tokens or raw transcripts included. This package does not establish an observed UI/demo or outreach readiness."}, "manifest_sha256")
    _write(target / MANIFEST, manifest)
    marker.unlink()
    return {"status": "built", "directory": str(target), "manifest_sha256": manifest["manifest_sha256"], "files": len(files),
            "model_sessions_started": 0, "email_ready": False}


def reproduce(directory, *, replay_classical=True):
    root = Path(directory).resolve()
    manifest = study._json(root / MANIFEST)
    study._verify(manifest, "manifest_sha256")
    if manifest["format"] != FORMAT or (root / ".release-building").exists():
        raise ValueError("publication is incomplete or uses an unsupported format")
    actual = _files(root)
    actual.pop(MANIFEST)
    if actual != manifest["files"]:
        raise ValueError("publication files changed, are missing or contain untracked material")
    if any(part in (".state", ".git", "artifacts") or part.startswith(".env") for name in actual for part in Path(name).parts):
        raise ValueError("private files cannot be included in a publication package")
    for name in actual:
        if name.startswith(("release-data/", "release-results/")) or name == "README.md":
            release_data.safe_text((root / name).read_text() if not name.endswith((".png", ".svg")) else "")
    for name, sha in manifest["original_public_inputs"].items():
        path = root / ("evaluation/README.md" if name == "README.md" else name)
        if Path(name).is_absolute() or not path.resolve().is_relative_to(root) or cache.file_hash(path) != sha:
            raise ValueError("frozen evaluation input differs from its original bytes")
    if manifest["source_sha256"] != study._sources():
        raise ValueError("run reproduction with the package's frozen code and dependency lock")
    projection = study._json(root / "release-results/evidence.json")
    if projection["projection_sha256"] != manifest["projection_sha256"]:
        raise ValueError("public evidence projection identity changed")
    report, checks = release_data.reproduce(projection, root / "release-data", replay_classical=replay_classical)
    expected = study._json(root / "release-results/analysis.json")
    if expected["analysis_sha256"] != manifest["analysis_sha256"]:
        raise ValueError("published analysis identity changed")
    release_data.compare({key: value for key, value in expected.items() if key != "analysis_sha256"},
                         {key: value for key, value in report.items() if key != "analysis_sha256"})
    if (root / "release-results/report.md").read_text() != confirmation_analysis.render_markdown(expected):
        raise ValueError("readable report differs from its recorded analysis")
    return {"status": "reproduced", "manifest_sha256": manifest["manifest_sha256"], "files_verified": len(actual), **checks,
        "classical_replay_requested": replay_classical, "runtime_matches_recorded": manifest["runtime"] == json.loads(json.dumps(cache.runtime_identity())),
        "bitwise_analysis_match": expected["analysis_sha256"] == report["analysis_sha256"], "numeric_tolerance": 1e-12,
        "execution_authorized": False}
