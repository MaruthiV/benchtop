# Controlled reference defects and valid controls

This suite runs deliberately changed versions of reviewed reference code and data. It tests **six specific failure cases and ten valid controls**, with a named guard and recorded evidence for each. Passing this suite does not admit a lab or estimate recall on arbitrary scientific defects.

[manifest.json](manifest.json) binds the historical source files, exact mutations, probe configuration, and counterfactual transformation. [check_defects.py](../check_defects.py) builds disposable variants and applies the guards. [report.json](report.json) records the observed results, input hashes, runtime, complete probe observations, and executed source hashes. Historical lab code, cached results, tasks, figures, and source-bound review drafts are unchanged.

## Covered changes

| Controlled change | Reference and mutation | Guard | Expected observation | Valid control |
|---|---|---|---|---|
| Reversed metric | Muon: declare accuracy lower-better | `muon.accuracy_direction` | Fail: the declared direction contradicts the implemented fraction correct | Original higher-better accuracy |
| Disabled treatment | Muon: remove the momentum perturbation but retain the `momusam` method label | `muon.momentum_perturbation` | Fail: zero training perturbation calls and zero perturbation norm | Original update: 150 perturbation calls, initial zero, later nonzero |
| Advantaged baseline | Muon: let only the baseline train on the test indices | `muon.heldout_disjoint` | Fail: 400 training/test indices overlap | Original disjoint train/test split |
| Inconsistent randomness | Muon: offset the treatment's data seed without changing its reported seed key | `muon.common_inputs` | Fail: training and test fingerprints differ within the intended pair | Original paired inputs, batches, and initialization agree |
| Stale cache | Data curation: change target-label count from 40 to 41 while retaining the historical cache | `cache.source_binding` | Fail: historical input verification rejects the copy | Unchanged copied cache loads successfully |
| Ambiguous reward | Data curation: add alternating ±0.04 to held-out treatment accuracies, clipped to [0,1], in memory | `qualification.claim_interval` | Question unavailable: complete data, but an interval crossing the effect thresholds | Original qualified near-null claim and qualified negative Muon claim |

The baseline leak is detected through overlap, regardless of whether it happens to improve accuracy. The treatment guard observes the update path, rather than requiring a favorable treatment score. The metric guard is specific to this implementation's definition of accuracy; it is not a rule that all metrics should be maximized.

The ambiguous counterfactual keeps the original near-zero mean but widens its 95% interval to approximately [−0.03170, +0.03114], exceeding the declared ±0.01 equivalence band. The transformed rows are never saved as lab observations or represented as new executions. The source-derived artifact is fixture evidence only. An ambiguous question does not establish invalid science.

## Runtime controls and instrumentation

Live probes use the unchanged 150-step Muon implementation, two existing visible seeds (0 and 1), batch size 16, 200 training examples, and radius 0.3. A zero-radius extension checks an identity case. This is one bounded context, not a qualification sweep or an independent confirmation experiment.

[muon_observer.py](muon_observer.py) is appended only to temporary copies, with a single declared capture call inserted after parameter initialization. It observes actual input arrays/indices, batch indices, initial parameters, gradient calls, and applied perturbation norms. SHA-256 fingerprints are transported as eight exact 32-bit numeric limbs through the worker's numeric-metric protocol, then reassembled by the evaluator. None of this instrumentation enters the agent endpoint or original lab.

Three further controls check the measuring process:

- **Observer transparency:** all five ordinary output metrics agree with a pristine worker for Muon, MoMuSAM, and MuSAM at seed 0, within a 1e-12 tolerance. MuSAM also exhibits its two-gradient training path; the stored trace includes 301 gradient calls, including the final diagnostic gradient.
- **Fresh-worker repetition:** baseline and treatment at seed 0 repeat all metrics and traces exactly in a second worker.
- **Zero-radius identity:** Muon and MoMuSAM return equal ordinary metrics when the treatment perturbation radius is zero. A valid identity case is not rejected for lacking an outcome difference.

One full run executes 24 jobs in six isolated, single-worker pools. The existing macOS sandbox denies network and repository writes. Lab execution never falls back to the host interpreter when that backend is unavailable. Temporary sources and caches are removed after the run. No model sessions, generation calls, or paid services are involved.

The common-input guard checks the exact profile's intended streams. It is not a universal requirement that every method have identical internal randomness: policy-dependent trajectories or unequal architectures can make that inappropriate. The selected profile uses shared architecture and intended common inputs. The baseline-leak guard does not establish equal capacity, tuning effort, compute, or supervision in other comparisons.

## Reproduce

On the supported macOS host, from the repository root:

```sh
.venv/bin/python -m benchmarks.release_v1.admission.check_defects \
  --output artifacts/admission_defects.json
```

To deliberately refresh the versioned public report after checking the inputs:

```sh
.venv/bin/python -m benchmarks.release_v1.admission.check_defects \
  --output benchmarks/release_v1/admission/defects/report.json
```

The report is a measured run, with its actual UTC time and runtime identity. New runs may change that metadata; input bindings, guard expectations, and raw observations make the result assessable. It is not promised to be byte-identical across machines or library versions. If a source changes, the suite refuses it; update the reviewed fixture version deliberately rather than silently altering the historical input.

For cached cases on a host without the supported live sandbox:

```sh
.venv/bin/python -m benchmarks.release_v1.admission.check_defects \
  --cached-only --output artifacts/admission_defects_cached.json
```

That mode reports **partial**, three of six controlled changes, and zero sandbox jobs. It cannot establish the runtime guards. Both modes use public files only and need no `.state/`, downloaded paper text, or model credentials. Outputs inside the repository are restricted to `artifacts/` or this report path.

Checkpoint validation: all 16 expected outcomes passed, including all six controlled changes and ten controls. All 189 repository tests and Ruff passed. The full 24-job fixture suite also passed in a disposable public-file copy without private state, paper snapshots, or model credentials. A regression test covers symlinked checkout paths, including the macOS `/var` versus `/private/var` alias encountered during that check. The saved report's input and executed-source hashes match the public files and mutation recipes.

## What remains unresolved

The report contains zero human admissions. The six [reference drafts](../references/README.md) retain their original evidence scope and timestamps; these new bounded Muon measurements can inform a separate owner review. They do not automatically change any draft status or cached admission screen.

CaLLM's architecture/training confound remains a real unresolved generated-lab example in that review package. The controlled baseline leak here is a separate fixture, not a claim that the CaLLM lab trained on its test examples. Paper fidelity, baseline capacity and information fairness, practical effect margins, optimizer numerics beyond the bounded probes, and reward stability still need the planned review. The suite is not wired into fresh generation or the development lock yet.

The next admission step is a consolidated evidence report and explicit human decisions, preserving accepted, rejected, and unresolved cases. Four development labs may be selected only under the existing human-admission and corrected-qualification rule. Actual Qualia UI use remains independently pending.
