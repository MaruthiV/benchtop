# Evidence for lab admission

The admission screen separates technical validity, question eligibility, human scientific review, and model difficulty. It reads cached artifacts without executing lab code or calling a model. A settled treatment effect and a favorable LLM audit do not approve a lab.

Start with the [consolidated evidence report](admission/REPORT.md) for the six reference cases, the newer bounded Muon evidence, and all controlled fixture outcomes. The [owner review guide](admission/REVIEW.md) creates separate unassigned records and explains how to complete them. Assistant suggestions, actual human decisions, question eligibility, and runtime scope are displayed separately.

## Current historical screen

[admission/screen.json](admission/screen.json) records the 30-lab historical catalog under `cached_admission_v1`:

| Evidence | Result |
|---|---:|
| Labs screened | 30 |
| Mechanical screens passed | 30 |
| Corrected, qualified claim questions before review | 28 |
| Human-reviewed admissions | 0 |
| Labs needing scientific review | 30 |

The source task grader remains `historical_v0`. The screen independently reconstructs qualification diagnostics with `seed_block_t_v1`, using the historical held-out seeds for retrospective development evidence. It does not reinterpret saved model scores as corrected evaluations or admit these labs to the fresh cohort.

The generation pipeline's historical acceptance heuristic is unchanged. This screen is a separate evidence step; integration with fresh generation and the candidate lock remains later work.

For example, the CALLM lab has a settled positive cached contrast, while its prior model audit flags baseline fairness. The report preserves that concern and leaves admission pending. The Para-Gaussian lab has a settled contrast and a context-dominance diagnostic of about 0.947. That diagnostic is a review signal: a legitimate interaction can also dominate an average. Neither example has a human admission decision yet. Existing model audit text is advisory evidence with a recorded source hash, not a verified paper-fidelity finding.

## Generate evidence and an unassigned review

```sh
.venv/bin/python -m benchtop.cli admission inspect \
  envs/2610-36983-advancement-or-acceleration-how \
  --output artifacts/data-curation-admission.json \
  --draft-review artifacts/data-curation-review.json
```

The review draft is bound to the inspected lab, qualified data/artifact, and charging schedule. Its reviewer is `unassigned`, checks are unresolved, and decision is `needs_review`. An existing draft is never overwritten. Read the paper and implementation, then complete the record with an explicit reviewer identity and UTC timestamp, exact paper version and section references, the miniature's scope, and an evidenced decision.

Each of seven scientific checks needs a status and paper/code/measurement evidence with a source, locator, and finding:

- Mechanism fidelity: what the source paper contributes and what the miniature implements.
- Baseline fairness: comparable data, information, tuning, and compute assumptions.
- Shared randomness: actual data/split/initialization relationships across treatment arms.
- Held-out metric: what was held out and whether metric direction is appropriate.
- Meaningful margin: why the effect threshold has meaning in this miniature.
- Execution behavior: determinism, parameter sensitivity, and supporting runtime/probe evidence.
- Shortcut audit: leaks, disabled mechanisms, information advantages, or predetermined outcomes.

An accepted human review requires every check to pass with nonempty evidence and explicit preserved assumptions, simplifications, and unsupported claims. A rejection requires an evidenced failed check. Unresolved cases remain `needs_review`. Model-authored or unassigned drafts cannot admit a lab. Reviewer identity and the content of citations are operator-reported; the checker validates structure and bindings, not the authenticity of an identity or a scientific judgment.

Evaluate a completed review without modifying the lab:

```sh
.venv/bin/python -m benchtop.cli admission inspect \
  envs/2610-36983-advancement-or-acceleration-how \
  --review artifacts/data-curation-review.json \
  --output artifacts/data-curation-admission.json
```

Changed lab, qualification, grader/generator, or charging artifacts invalidate an earlier review. Visible cache appends alone do not invalidate a qualification-bound review. The completed record should be retained under a deliberate version, rather than silently updated to approve changed artifacts.

## Mechanical scope and seed handling

The screen checks required artifacts and cache provenance, the existing static source contract, context/grid requirements, a finite positive effect margin, the paper claim to be reviewed, complete paired qualification data, and valid charging costs. Invalid artifacts remain visible as rejections; they are not silently omitted from a catalog screen. Malformed and stale review files remain visible without approving a lab.

Qualification uses explicit `visible`/`qualification`/`confirmation` seed groups when present. They must be disjoint and cover the declared seeds; visible IDs must match the visible partition. Otherwise the screen uses the task artifact's qualification source or the historical held-out set. It rejects visible-seed leakage and inconsistent task seed sources. Confirmation metric values do not enter its contrasts or qualification digest. This selection support does not implement the complete fresh-cohort harness or its lock gates.

A complete but ambiguous interval leaves the claim ineligible without declaring the underlying science invalid. A reviewed, valid null-effect lab can have a usable equivalence reward. The report's `qualified_claim_admitted` field requires a technically usable artifact, an accepted human review, and a settled corrected claim label. Model difficulty remains `not_measured` and live runtime probes remain `not_run`; cached numbers cannot establish runtime repeatability or prove identical data merely because seed keys align.

## Reproduce the catalog screen

```sh
.venv/bin/python -m benchtop.cli admission screen \
  --output benchmarks/release_v1/admission/screen.json
```

To include completed records, provide `--reviews-dir PATH`, containing `ENVIRONMENT_ID.json` files. The catalog controls membership and source grading; unlisted directories are excluded. The output records input/checker hashes and the actual analysis runtime. It changes no catalog, lab, saved score, or model-difficulty record.

Reports and draft files may be written under `artifacts/`, under `benchmarks/release_v1/admission/`, or outside the repository. Original code, environment, score, and figure locations are protected from output overwrites. Reports are deterministic for matching inputs and analysis runtime.

The tests include positive and valid-null controls, simulated human accept/reject records, incomplete/model-authored/stale reviews, visible appends, malformed caches, missing qualification cells, invalid source, seed overlap, and confirmation-data perturbation. Simulated reviewer records in tests are not real human reviews.

The [six reference cases and cited assistant drafts](admission/references/README.md) now cover the planned engineering roles. Their pinned paper sources, code findings, frozen charges, and corrected cached diagnostics are recorded and checked offline. All six remain `needs_review`; the CaLLM draft recommends rejecting its confounded primary comparison, without recording a human verdict. No development labs have been locked.

The [controlled semantic fixture suite](admission/defects/README.md) now exercises six named changes with ten valid controls: reversed accuracy direction, a disabled momentum perturbation, baseline access to test examples, mismatched paired inputs, stale cached source, and an ambiguous question. Twenty-four sandbox jobs observe a bounded Muon profile, with pristine-observer agreement, fresh-worker repetition, and zero-radius identity controls. Cached transformations stay separate from new executions. These lab-specific guards do not establish general scientific-defect recall or complete any human admission; ambiguous rewards make a question unavailable rather than rejecting the science. The cached screen above still runs no live probes.

The consolidated report can be regenerated offline with `python -m benchmarks.release_v1.admission.build_report --output-dir artifacts/admission_report`. It validates the stored catalog screen, reference bindings, and saved fixture input/accounting evidence without executing lab code or models. Provide `--reviews-dir PATH` to include separate owner records; malformed, stale, incomplete, and missing records remain visible. It creates no development lock. The original static drafts and recorded measurements are preserved.

Human decisions, practical margin justifications, and remaining runtime/reward-stability evidence remain pending. The actual Qualia UI check also remains open; preparation and generic MCP preflight do not complete it.
