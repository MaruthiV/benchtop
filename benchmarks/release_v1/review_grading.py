import argparse
import hashlib
import json
from pathlib import Path

from benchtop.env import ROOT
from benchtop.evaluate import all_envs
from benchtop.stats import GRADER_ID, LEGACY_GRADER_ID
from benchtop.tasks import COMPONENT_LABELS, build_task_set


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review():
    labs = []
    totals = {
        "labs": 0,
        "historical_tasks": 0,
        "corrected_tasks": 0,
        "historical_settled_contrasts": 0,
        "unchanged_labels": 0,
        "became_ambiguous": 0,
        "changed_settled_labels": 0,
        "unavailable_contrasts": 0,
        "invalid_or_incomplete_labs": 0,
    }
    component_labels = {value: key for key, value in COMPONENT_LABELS.items()}
    for env in all_envs():
        if env.grader_id != LEGACY_GRADER_ID:
            raise ValueError("this review requires a historical_v0 source cohort")
        corrected = build_task_set(env)
        diagnostics = {(d["task_id"], d.get("level")): d for d in corrected["contrasts"]}
        changes = []
        for task in env.tasks:
            if task["kind"] == "best":
                continue
            answers = task["answer"].items() if task["kind"] == "where" else [(None, task["answer"])]
            for level, before in answers:
                if task["kind"] == "component":
                    before = component_labels[before]
                detail = diagnostics.get((task["id"], level))
                after = detail.get("label") if detail is not None else None
                totals["historical_settled_contrasts"] += 1
                if after == before:
                    totals["unchanged_labels"] += 1
                    continue
                if detail is None:
                    outcome = "unavailable_contrasts"
                else:
                    outcome = "became_ambiguous" if after is None else "changed_settled_labels"
                totals[outcome] += 1
                changes.append({"task_id": task["id"], "level": level, "before": before, "after": after})
        totals["labs"] += 1
        totals["historical_tasks"] += len(env.tasks)
        totals["corrected_tasks"] += len(corrected["tasks"])
        totals["invalid_or_incomplete_labs"] += corrected["status"] != "complete"
        labs.append(
            {
                "env": env.id,
                "input_sha256": {
                    name: checksum(env.dir / name)
                    for name in ["lab.py", "space.json", "manifest.json", "results.jsonl", "tasks.json"]
                },
                "historical_task_count": len(env.tasks),
                "corrected_task_count": len(corrected["tasks"]),
                "label_changes": changes,
                "corrected_task_set": corrected,
            }
        )
    sources = ["benchtop/stats.py", "benchtop/tasks.py", "benchtop/env.py", "benchtop/cache.py", "benchmarks/release_v1/review_grading.py"]
    return {
        "schema_version": 1,
        "source_grader": LEGACY_GRADER_ID,
        "corrected_grader": GRADER_ID,
        "scope": "Retrospective task construction on existing historical held-out seeds; not fresh confirmation or agent evaluation.",
        "catalog_sha256": checksum(ROOT / "envs/catalog.json"),
        "generator_sha256": {name: checksum(ROOT / name) for name in sources},
        "summary": totals,
        "labs": labs,
    }


def main():
    parser = argparse.ArgumentParser(description="Review corrected grading without changing historical lab files.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(review(), indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == "__main__":
    main()
