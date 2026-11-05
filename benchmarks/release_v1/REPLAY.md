# Classical replay harness

`python -m benchtop.replay` plans and runs cached classical episodes with immutable attempt records. It provides an engineering mode for bounded checks and a development mode gated by the six human reference reviews. It starts no model sessions, executes no lab workers, and creates no model development, candidate, or fresh-cohort lock.

## Policies and sanity controls

The three required acquisition policies use the [shared estimator and public interface](BASELINES.md). Three sanity controls are also implemented:

| Control | Rule |
|---|---|
| `paper_prior_only` | Predict `paper_better` without experiments. |
| `development_majority_label` | Predict the majority corrected claim label of the four selected development labs, without experiments. Ties use the protocol's primary-label order. |
| `random_experiment_design` | Randomly draw configurations with remaining visible capacity, within the same budget and call limits; reconstruct with `paired_context_v1`. |

The two fixed priors have no observation estimator; their records explicitly use `estimator_id = null`. Random design uses the same estimator as the main policies and can leave unmatched arms. Historical baseline commands and saved results retain their old behavior.

## Engineering mode

Create a bounded plan explicitly, for example:

```sh
.venv/bin/python -m benchtop.replay plan \
  --mode engineering \
  --env envs/2610-36935-gaussian-process-bandits-for \
  --repeats 2 \
  --output artifacts/classical-engineering-plan.json

.venv/bin/python -m benchtop.replay run \
  artifacts/classical-engineering-plan.json \
  --output-dir artifacts/classical-engineering-run \
  --max-attempts 5
```

This mode permits one or two frozen historical snapshots or controlled corrected fixtures, one or two policy seeds, and the three protocol budgets. It excludes the development-majority prior, which requires the reviewed development subset. It rejects fresh paper sources and three-part fresh seed partitions. A two-part fixture's declared metadata is an engineering contract, not authenticated evidence that it represents a real or synthetic paper.

The example creates 30 slots: one reference, five policies, three budgets, and two policy seeds. It is an engineering check, excluded from the development comparison. It does not select this reference into the development set or establish a performance result on admitted labs. Plan files are exclusive: use a new filename for a new plan rather than overwriting its provenance.

## Development gate

```sh
.venv/bin/python -m benchtop.replay gate \
  --reviews-dir artifacts/reference-human-reviews
```

The gate requires all six source-bound human records to be structurally complete and current. Assistant/unassigned drafts do not count. Accepted references also need a qualified corrected claim. It selects the first four eligible references in lexicographic ID order, following the fixed protocol, and derives the majority label from those four only. Recorded human `needs_review` or rejection decisions remain in the funnel. Reviewer identity and scientific judgments are operator-reported; the gate does not authenticate them.

The current owner records remain unassigned, so this gate is blocked. Complete the [owner review process](admission/REVIEW.md) before creating a development plan. If fewer than four references qualify, preserve the original protocol and document an amendment before model budget-selection starts; the harness does not silently replace them with easier labs.

After the gate is ready:

```sh
.venv/bin/python -m benchtop.replay plan \
  --mode development \
  --reviews-dir artifacts/reference-human-reviews \
  --output artifacts/classical-development-plan.json

.venv/bin/python -m benchtop.replay run \
  artifacts/classical-development-plan.json \
  --reviews-dir artifacts/reference-human-reviews \
  --output-dir artifacts/classical-development-run
```

This plans all four primary claim tasks, six policies, three budgets, and 200 policy seeds: 14,400 cached slots. Labels are read for the fixed majority control and after acquisition for scoring, never by the main acquisition policies. The scientific admission records bind the historical source snapshots. The plan separately binds the deterministic corrected task and qualification-price replay views, preserving both provenances. It is a classical comparison plan, not the protocol's later model development lock.

## Bindings and visible ordering

A plan records the exact protocol file checksum and canonical protocol hash, runtime identity, code checksums, source paths/files, execution/task/charging versions, policies, budgets, and policy seeds. Development plans also bind the six review-file hashes and gate result. Before resume, the full grid and all bindings are revalidated; changed code, caches, task labels, prices, or review records cannot silently resume an old plan.

The classical lab content hash is the canonical SHA-256 object containing the `lab.py` and `space.json` file hashes. The predefined protocol prefix plus that content hash chooses a lexicographic visible-seed permutation. All policies, budgets, and policy seeds within a lab share this order. It is set through a validated evaluator-only `visible_order` constructor argument; the agent description still exposes no seed IDs or ordering. Actual order and its `explicit` source are recorded in the evaluator output. Default episodes retain their original seed-driven ordering.

Binding, group, slot, and attempt identities use canonical UTF-8 JSON with sorted keys, compact separators, and no NaN. The binding is hashed before groups and slots are derived; the outer plan hash binds the final grid. Each slot has distinct attempt-zero and optional attempt-one identities. These are namespaced classical replay identities, not model rollout slots, and consume none of the model-session reserve.

## Resume and failure accounting

Repeat the same run command to resume. Completed outcomes, including reward zero, are immutable and skipped. Policy errors such as invalid requests are terminal zero-reward outcomes. A recorded run-callback exception, per-experiment error, explicitly recognized service/cache failure, or unexpected partial reply is a technical missing outcome. A provisional subset-data score can remain in raw diagnostics, but its primary reward is `null`.

Technical retries require an explicit flag:

```sh
.venv/bin/python -m benchtop.replay run \
  artifacts/classical-engineering-plan.json \
  --output-dir artifacts/classical-engineering-run \
  --retry-technical
```

At most one retry is allowed per slot. All earlier attempts remain recorded. A zero-data fallback that never contacted a failed service is a completed policy outcome, not an infrastructure failure. Unknown policy exceptions are not automatically reclassified as service failures to obtain a better rerun.

The output directory contains an immutable `plan.json` and atomic, exclusive start/finish JSON files under `attempts/`. A nonblocking lock prevents concurrent runners from owning the same directory. Each record has its own checksum. Raw requests, responses, errors, partial observations, evaluator seed lineage, actual visible order, and score diagnostics remain available. The runner refuses unrelated nonempty directories and protected repository output locations.

If a previous invocation has a start but no published finish, summary reports it as interrupted. Resume records it as technical missing before any optional retry. Its experimental spend is unknown, not silently zero. A temporary file left by an abrupt interruption is never interpreted as a completed outcome. Malformed, orphaned, or checksum-invalid published records fail visibly rather than being discarded. The atomic-file scheme handles ordinary process interruption; it is not a claim of authenticated measurements or a distributed execution service.

Export a full summary snapshot:

```sh
.venv/bin/python -m benchtop.replay summary \
  artifacts/classical-engineering-run \
  --output artifacts/classical-engineering-summary.json
```

Summaries retain every planned slot, completed zeros, policy failures, technical missing outcomes, unattempted slots, known experimental charges across attempts, and unknown-spend counts. Each group has a complete-case accuracy and conservative bounds treating missing/unattempted rewards as either zero or one. With no completed outcomes, complete-case accuracy is unavailable. Policy-seed repetitions reuse cached data; they are not independent new labs or independent model sessions.

## Validation and next work

The tests use temporary simulated human records to exercise a ready gate; those are not actual reference reviews or public admissions. Other controls cover bounded plans, full-grid checks, raw trace capture, completed-wrong-answer resume, technical failure/retry limits, interrupted starts, unknown costs, invalid-request zeros, record/input tampering, fixed priors, random design, explicit visible ordering, majority tie rules, output protection, and concurrent-runner exclusion.

The actual 200-seed development comparison remains unrun. Next: complete the human review gate, inspect the admitted development results and estimator sensitivity, then bind policy inclusion and code at the candidate lock before fresh generation. Model configuration, Qualia UI use, fresh three-part evaluation, packaging/demo, and outreach readiness remain separate work.
