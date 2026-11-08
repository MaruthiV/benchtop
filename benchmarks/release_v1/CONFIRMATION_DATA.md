# Independent confirmation truth

The confirmation collector runs the locked lab code on seeds 16–27 in a separate private cache, then applies the existing corrected grader and stability rule. It requires the verified [confirmation lock](CONFIRMATION.md) before accessing collection state. **Tests use synthetic labs, simulated provider/paper/reviewer metadata and real local sandbox jobs. No actual-study confirmation data, real model session or performance result exists.**

Finish remaining study code and public documentation before actual model sessions. Source changes invalidate earlier preparations and cannot refund starts.

## Collect after the arm lock

```sh
.venv/bin/python -m benchtop.cli study collect-confirmation-data
.venv/bin/python -m benchtop.cli study inspect-confirmation-data
```

Collection accepts no paper, member, seed, worker, timeout or margin override. It follows the frozen cohort order and writes beneath the canonical study's `confirmation_data/` directory. It starts no models and probes no live authentication. New CPU work requires idle primary accounting without a model/tool halt, and no confirmation rollout history preceding completed truth data.

An immutable collection intent binds the protocol, arm lock, plan, ledger, code/runtime, member order, exact confirmation seeds and frozen execution rules. Each lab gets byte-identical copies of its approved code and space, a private `confirmation_only` manifest and execution cache. Qualification code, space, manifest, cache, results, tasks and frozen prices remain unchanged.

The collector schedules the full paired configuration grid on only the twelve confirmation seeds, using the existing two-worker, five-second per-job sandbox rules. Workers execute a temporary code copy and cannot read the original study directory, including qualification or confirmation results. The cache has a separate private format that unchanged ordinary environment and sweep loaders reject. The assessment validates that format and every sealed row, constructs a table for the existing statistical grader, and explicitly supplies confirmation seeds. It writes no ordinary `tasks.json`.

Each lab has an execution intent before workers start, sealed result rows, retained worker logs and a completion proof after workers stop. The collector checks protected inputs during execution and revalidates the lock chain before saving completion evidence. Changed source/copy/plan/lock inputs preserve partial work and stop collection. Result rows are flushed to disk before completion publication.

## Truth and stability

The private assessment uses the same seed-block t interval, fixed effect margin and bootstrap diagnostic as qualification. It retains twelve independent seed blocks and the paired contexts rather than treating all cells as independent observations. Every frozen member remains in the assessment:

| Status | Meaning | Usable confirmation reward |
| --- | --- | --- |
| `stable` | Complete, settled, and agrees with qualification | Eligible for separately authorized grading |
| `unstable` | Complete and settled, with a different label | Unavailable |
| `ambiguous` | Complete data without a settled claim | Unavailable |
| `incomplete` | Missing or failed required cells, or invalid data | Unavailable |

The per-lab assessment binds the independent label, seed partition, margin, corrected task artifact, original question hash and execution proof. The aggregate assessment binds every member and all collected files. Unavailable members are not replaced or removed from the full-cohort denominator. No private truth label, metric table or grade appears in the CLI summary.

## Failure, resume and costs

Finished failures remain failed; repeated collection does not retry them to obtain a preferred label. An execution start without a durable completion proof blocks automatic reruns, even if a full set of rows exists. Inspection reports `pending_reconciliation` and preserves those rows for explicit investigation. There is no reset or implicit reconciliation command.

When completion evidence was saved but acknowledgement or assessment publication failed, collection verifies that evidence and finalizes the assessment without more jobs. A saved aggregate can be inspected or reused byte-for-byte without execution, including after later model accounting grows. Missing authority, changed data, duplicate cells, cross-partition seeds, edited truth records and altered logs fail verification before producing usable grading evidence.

Reports distinguish recorded job intents, observed results, failed cells, worker-reported elapsed seconds, measured collection wall time and unavailable process CPU time. Missing/invalid timing retains known partial sums and a null total. Timeout thresholds are reported work timing, not measured CPU. Model-session count is zero because this collector invokes no inference; evaluation and generation allowances remain separate.

A zero-member cohort returns `no_primary_comparison` without creating a data directory or starting workers.

## What remains

An assessment establishes private truth/stability, not provider execution permission. [Confirmation rollout authorization](CONFIRMATION_EXECUTION.md) and independent private grading are implemented. [Canonical results, matched classical replay and full-cohort analysis](CONFIRMATION_RESULTS.md) are implemented. Actual independent sessions under both frozen budgets and real study results remain pending. [Shared Qualia invocation accounting](QUALIA_PRIMARY.md) is implemented. Actual observed UI compatibility, actual scientific records and bounded study runs, a working demonstration, reproducible public reporting and release still precede outreach.
