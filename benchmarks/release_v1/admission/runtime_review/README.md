# Bounded visible-seed runtime review

The [recorded run](report.json) executes 66 sandbox jobs on existing visible seeds 0 and 1. It checks three reference miniatures without modifying their original code, caches, qualification data or confirmation data. This is runtime evidence for admission review, not human admission or new model performance.

| Miniature | Bounded contexts | What was observed |
|---|---|---|
| GP bandits | Stationary/low-noise/no delay; drift/high-noise/delay; reactive/high-noise/delay | Identical numerical metrics for seed 0 in fresh workers for both designated arms. Each method still has its own computation and tuning; no equal wall-clock claim. |
| Algorithmic monoculture | Low-correlation/small-gap/tight-cap; high-correlation/large-gap/wide-cap | Fresh-worker repetition, unchanged ordinary metrics under the observer, and no reported optimizer failures/nonfinite optimizer outputs in the observed calls. |
| Data curation | Zero-gap/small clean pool; shifted-gap/large noisy pool | Fresh-worker repetition, unchanged ordinary metrics under the observer, and no reported optimizer failures/nonfinite optimizer outputs in the observed calls. |

The pristine pass runs each designated configuration at both visible seeds. A new worker repeats seed 0. Separate observer copies wrap the existing SciPy `minimize`, return its result unchanged, and collect the reported success flag, iteration count and finite result/objective checks. Observer transparency is checked against pristine ordinary metrics to relative/absolute tolerance `1e-12`.

The GP probe does not instrument linear algebra internals. Optimizer success does not prove a global optimum, full-grid convergence, scientific fidelity or an appropriate practical effect margin. Frozen historical charges are not changed and these timings do not establish an end-to-end compute advantage. The estimated-risk cap in the monoculture miniature is not a guaranteed cap on realized true risk. The data-curation method still has extra target supervision.

The separate [Muon fixture suite](../defects/README.md) covers the stated perturbation/input-sharing/repeatability controls. It does not gain broader coverage from this run.

## Reproduce

In a clean supported macOS checkout with dependencies installed:

```sh
uv run python -m benchmarks.release_v1.admission.runtime_review.run_visible --output artifacts/reference_runtime_repeat.json
```

The generator's SHA-256 must match `source_script_sha256` in the report. It writes a new JSON file exclusively, refusing to replace an earlier run or overwrite public source files. Use a new output filename for each repeat and retain both records. The profile is fixed; the record contains the actual run time and runtime identity.

New runs have different wall/job timings and recording timestamps. Numerical repeatability applies within the stated runtime/profile; original lab files must match the recorded hashes. The worker denies network access and reads of repository data/notes, and cannot write outside its temporary directory. No inference, paid service or unisolated fallback is used.
