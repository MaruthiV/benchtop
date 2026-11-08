# Frozen confirmation arms

The confirmation lock binds the verified [closed pilot collection](PILOT_CALIBRATION.md), each lab's calibrated budget or global fallback, both arm descriptors, and every planned slot before confirmation outcomes are collected or inspected. **Tests use synthetic labs and simulated provider/paper metadata. No actual confirmation lock, confirmation data or model result exists.**

Finish remaining study code and public documentation before actual model sessions. Later source changes invalidate bound preparations and cannot refund starts.

## Prepare the fixed arms

```sh
.venv/bin/python -m benchtop.cli study plan-confirmation \
  --output-dir artifacts/confirmation-preparation
```

For a nonempty cohort, preparation requires the verified canonical pilot closure. It inherits the cohort, model, prompt, lab/code/runtime, question, frozen prices and global control. Each lab's selected budget or explicit fallback comes from its saved selector output. There is no budget, model, membership or order override.

Each member has a `global_budget` group and a `calibrated_budget` group with eight independent replicate slots per arm. Identical budgets still have different arm/group/slot/attempt IDs. Initial planned work is 16 slots per member, at most 128 for eight labs. Technical-retry IDs do not create a new retry allowance; the shared study limits still apply.

Slots follow cohort order, alternating arms for each replicate. Odd-numbered lab indices reverse the starting arm. Both arms use the cohort's deterministic confirmation visible order, the next permutation after the pilot order. Model configuration, prompt, question, prices, grader source hashes, paper visibility and tool limits stay matched.

The group task hash anchors the frozen qualification question and answer schema. Confirmation truth requires a separate post-lock assessment from seeds 16–27. The plan explicitly binds that requirement and the complete/settled/agrees-with-qualification stability rule. Its private qualification cache does not establish independent confirmation truth. Ordinary environment/task loading retains its existing seed-partition restrictions.

Preparing a plan writes only an immutable preparation and receipt with `execution_ready: false`. It starts no models, collects no confirmation rows, probes no live authentication and creates no study lock.

## Publish the protocol lock

```sh
.venv/bin/python -m benchtop.cli study lock-confirmation \
  artifacts/confirmation-preparation
.venv/bin/python -m benchtop.cli study inspect-confirmation
```

Publication revalidates the preparation while holding the canonical journal lock. It requires idle primary accounting without a model/tool halt, no prior confirmation start history, and no preexisting canonical `confirmation_data/` directory. It checks directory existence without reading premature outcomes. The cohort's original source snapshots must still be unchanged.

The private `confirmation_plan.json` contains the arm groups and interleaved slots. `confirmation_lock.json` binds its plan/core hashes, the cohort, all retained pilot evidence and selector decisions, the exact closed pilot file, the original ledger identity and publication timestamp. Publication appends the lock hash to `benchmarks/release_v1/bindings.json`; earlier binding records are preserved.

Inspection verifies the plan, private lock, prerequisite evidence and exact public binding. It does not read post-lock confirmation outcomes and remains stable as later phase accounting grows. It can reuse a caller's held canonical journal context without a second acquisition of the same file lock.

Repeated publication refuses to replace the original arms. Partial publication stays visible and blocks replacement. If public acknowledgement was lost after a successful append, inspection can verify the saved lock; if the public binding is absent, inspection fails until that partial state is investigated. Do not erase records to restart the experiment.

## Execution boundary and empty cohorts

This lock does not register confirmation rollout slots for execution or authorize provider calls. The [independent collector and truth/stability assessment](CONFIRMATION_DATA.md) now verify this lock before writing a separate `confirmation_data/` snapshot, preserving qualification files and the planned full-cohort denominator. Separate [provider authorization](CONFIRMATION_EXECUTION.md) is now implemented: it verifies the independent assessment, registers every original slot and permits only stable tasks in the frozen dispatch order.

A zero-member cohort has no pilot grid or comparison. Its empty preparation/lock records `no_primary_comparison`, zero slots and no fabricated pilot choice. It does not authorize data collection or model sessions. Smaller cohorts keep their actual membership; inconvenient calibration or unstable confirmation labels cannot trigger replacements.

## What remains

Independent confirmation collection/truth/stability are implemented with synthetic fixtures. Provider execution with independent private grading is implemented. [Matched classical replay and full-cohort analysis](CONFIRMATION_RESULTS.md) are implemented with canonical evidence closure and explicit unavailable comparisons. [Shared Qualia invocation accounting](QUALIA_PRIMARY.md) is implemented. Actual observed UI compatibility, actual scientific evidence and bounded study runs, a working demonstration, reproducible public reporting and release still precede outreach. These tests establish planning and phase-boundary behavior, not measured model performance.
