# Owner review guide

Start with [the consolidated report](REPORT.md). It shows the six reference cases, the original assistant suggestions, corrected cached effects, the newer bounded Muon measurements, and the remaining questions. [report.json](report.json) retains machine-readable bindings and evidence. It is a review packet, not a human approval record.

The generated labs currently have zero human admissions. CaLLM's favorable cached effect accompanies a concrete architecture/training confound and an assistant rejection recommendation. Decide its scientific status after inspecting the cited code and paper. The controlled Muon baseline leak is a separate demonstration, not evidence that CaLLM trained on test data.

## Prepare separate records

From the repository root:

```sh
.venv/bin/python -m benchmarks.release_v1.admission.build_report \
  --output-dir artifacts/admission_report \
  --prepare-reviews artifacts/reference-human-reviews
```

This creates six source-bound JSON files. Reviewer kind stays `unassigned`; name and UTC time stay blank; all seven criterion judgments stay `unresolved`; decision stays `needs_review`. Scope and citations are prefilled assistant suggestions. The original drafts are preserved. Preparation refuses to overwrite any existing record; rerun the report without `--prepare-reviews` when records already exist.

The scratch records and report are ignored by Git. Completed records should later be copied to a deliberate public version after review; the builder does not publish or commit them.

To reproduce the public bundle deliberately, use `--output-dir benchmarks/release_v1/admission` without `--prepare-reviews`. The default builder includes no private owner records; `--reviews-dir` must be explicit. Supplied record hashes are labeled `supplied_review:ENV_ID` in the JSON input inventory, without exposing absolute private paths. Generated links are adjusted for the report's actual output directory.

## Perform the review

For each case, use the linked exact paper version and implementation, then work through its owner questions in the report. Review the following criteria with your own rationale and evidence:

| Criterion | What to decide |
|---|---|
| Mechanism fidelity | Does the implemented comparison preserve the declared paper mechanism under an honest, sufficiently narrow scope? |
| Baseline fairness | Are differences in data, target information, model/head capacity, tuning, and compute appropriate for the specific claim? |
| Shared randomness | Does the implementation share the intended exogenous samples, splits, batches, and initialization? Which trajectories should differ? |
| Held-out metric | What is held out, how is the metric defined, and is its direction correct? Distinguish test metrics from training diagnostics and Monte Carlo references. |
| Meaningful margin | Why does this tolerance have practical meaning in the miniature's units? Give an independent rationale rather than choosing a margin to fit the observed interval. |
| Execution behavior | What has actually been checked for determinism, update behavior, parameter sensitivity, numerical reliability, and reward stability? Keep missing evidence unresolved. |
| Shortcut audit | Could information leakage, an omitted mechanism, confounded comparisons, or a predetermined objective explain the result? |

The newer Muon probes cover two visible seeds and one context, with a zero-radius extension. They support specific input-sharing and execution observations. They do not validate all contexts, practical margins, numerical convergence, or independent reward stability. Data-curation cache mutations and counterfactual reward changes are fixture evidence, not new executions of that lab.

Only after you have performed the review, enter your name, actual UTC review time, and `reviewer.kind = "human"`. Set each criterion to `pass`, `fail`, or `unresolved` and retain source/locator/finding evidence. Replace the pending rationale with your own explanation. The checker validates structure and bindings; it cannot authenticate your identity or the scientific content of a citation.

- `accepted` requires all seven criteria to pass with evidence and an explicit scope, including preserved assumptions, simplifications, and unsupported claims.
- `rejected` requires at least one evidenced failed criterion. A favorable cached score does not override a scientific rejection.
- `needs_review` preserves unresolved evidence. Do not set missing checks to pass to meet a quota.

Keep each decision about scientific validity separate from question eligibility and model difficulty. A reviewed valid null experiment can yield a useful reward. An ambiguous qualification interval can remove a claim reward without establishing invalid science. No model difficulty has been measured by this packet.

## Validate and consolidate your records

For example:

```sh
.venv/bin/python -m benchtop.cli admission inspect \
  envs/2610-36994-callm-continual-adaptation-of \
  --review artifacts/reference-human-reviews/2610-36994-callm-continual-adaptation-of.json \
  --output artifacts/callm-owner-admission.json

.venv/bin/python -m benchmarks.release_v1.admission.build_report \
  --reviews-dir artifacts/reference-human-reviews \
  --output-dir artifacts/admission_report
```

Malformed, stale, incomplete, or unassigned records remain visible in the report and cannot admit a lab. A missing review file also remains pending. The report does not silently adopt an assistant recommendation as a human decision.

The protocol later chooses the first four human-admitted, corrected-qualified reference claims in lexicographic ID order. This report can list eligible references; it creates no development lock or selected lab IDs. If fewer than four qualify, preserve the protocol and record an amendment before budget-selection model runs. Keep rejected and unresolved cases rather than replacing them based on model scores.

Stronger classical baseline policy implementation can proceed before this lock. Actual Qualia UI use, the fresh-cohort study, demonstration, and outreach readiness remain separate gates.
