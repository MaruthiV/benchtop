# Seed block grading

The corrected grader is `seed_block_t_v1`. It preserves the dependence created when every experimental context reuses the same random seeds.

For each seed, first average the treatment-minus-baseline differences across contexts, weighting each context equally and orienting the metric so higher is better. Then estimate uncertainty across those seed averages:

```text
D[s] = mean over contexts of oriented treatment-minus-baseline difference
delta = mean(D)
se = sample_standard_deviation(D) / sqrt(number_of_seeds)
interval = delta +/- t_0.975(number_of_seeds - 1) * se
```

The historical calculation combined context variances as if contexts were independent. That can underestimate uncertainty when context effects move together and overestimate it when they cancel. The regressions cover both cases. Small samples use the Student t multiplier rather than a fixed normal multiplier of 1.96.

Improvement requires the entire interval above the meaningful-effect margin; harm requires it below the negative margin. Equivalence requires the interval entirely inside the margin. Other complete-data cases are ambiguous. Equality at a decision boundary remains ambiguous.

A deterministic 2,000-resample bootstrap samples whole seed blocks, using RNG seed 1729. Its percentile interval and label are diagnostics only; they do not replace the primary t decision or change cohort membership. These intervals quantify uncertainty within the miniature under their sampling assumptions; they do not certify that it faithfully represents its source paper. Supporting `where` intervals are pointwise, not simultaneous coverage guarantees.

## Task artifacts and compatibility

New sweeps persist a task object with schema version 1, grader ID, qualification seed IDs, data status, eligible tasks, and all contrast diagnostics. Each eligible task also carries its grader and seeds. Complete data may still contain ambiguous contrasts. Missing pairs produce an `incomplete` artifact with no eligible tasks; nonfinite or invalid metrics produce an `invalid` artifact. Pipeline acceptance rejects those data states.

New artifacts also carry [execution and qualification-data bindings](CACHE.md), generator source hashes, and payload checksums. Loading rejects stale generators, changed qualification observations, and altered task payloads. Visible appends do not change the qualification digest.

The eligible-only `build_tasks` helper remains available and raises on incomplete or invalid data. `build_task_set` returns the full diagnostic artifact and accepts explicit seed IDs. The generation pipeline still defaults to its existing held-out split; the fresh three-part split and execution lock gates must be wired into the release harness before that study can run.

Historical unversioned task lists load as `historical_v0` without modifying their files. Mixed task graders and contradictions between the task artifact and environment manifest are errors. Catalogs must declare a supported grader; selected labs must match it and contain eligible tasks. Changing a catalog label alone cannot upgrade historical task labels.

New baseline and agent records carry `grader_id`, and generated reports identify their task graders. Untagged saved results are explicitly interpreted as historical. Report generation rejects selected-environment score records whose grader differs from the loaded tasks, before writing output. This prevents reusing historical scores as if they were measured against corrected labels.

Historical directories are verified read-only snapshots. Rebuild reviewed code in a separate versioned directory to collect new data and corrected tasks. Old catalog and result files must not be silently reused; use an intentionally versioned cohort and matching evaluation artifacts. Best-configuration questions still use the observed held-out best mean and are not confidence-qualified optima.

## Review of the existing collection

[grading_review.json](grading_review.json) applies corrected task construction to the existing cached qualification data. It preserves source-file and generator hashes and the complete corrected task diagnostics. This is retrospective development analysis of the original 12 held-out seeds, not independent confirmation or a new agent evaluation.

| Measurement | Result |
|---|---:|
| Historical paper-derived labs | 30 |
| Historical eligible questions | 78 |
| Corrected eligible questions | 75 |
| Previously settled claim or per-level contrasts | 77 |
| Labels unchanged | 74 |
| Labels becoming ambiguous | 3 |
| Labels changing to another settled label | 0 |

The three ambiguous cases are `where:model` at TransE in `2610-37389-how-many-and-how`, `where:noise_sd` at 0.25 in `2610-37456-an-input-skip-can`, and the claim in `2610-37553-collective-dynamics-of-selfreinforcement`. The first two questions no longer have enough qualified levels and the third has no settled claim, reducing the eligible question count by three.

The active historical catalog, original task files, tables, recorded model scores, and figures remain unchanged. Regenerate the separate review without model calls:

```sh
.venv/bin/python benchmarks/release_v1/review_grading.py --output benchmarks/release_v1/grading_review.json
```
