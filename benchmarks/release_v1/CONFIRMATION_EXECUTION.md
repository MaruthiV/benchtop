# Confirmation provider execution

The confirmation runner uses the original subscription provider and canonical study journal. Separate authorization binds the [arm lock](CONFIRMATION.md) to the verified [independent truth assessment](CONFIRMATION_DATA.md). **Validation uses synthetic labs, simulated provider metadata and real local sandbox/MCP transport. No actual-study confirmation sessions or measured model result exists.**

Finish remaining study implementation and public documentation before actual model starts. Preparations bind the source and runtime; later changes cannot refund consumed intents.

## Authorize after independent assessment

```sh
.venv/bin/python -m benchtop.cli study authorize-confirmation artifacts/confirmation-preparation
.venv/bin/python -m benchtop.cli study inspect-confirmation-execution
.venv/bin/python -m benchtop.cli study run-provider PLAN_SHA256 --slot SLOT_ID
```

Use the exact preparation saved before confirmation data collection. Authorization revalidates the protocol lock, frozen pilot choices, qualification sources and independent data/assessment. It inherits the original pinned model, prompts, runtime, provider and extra-usage-disabled attestation. Inspection performs no live authentication probe or model call and prints hashes and counts without private labels or metrics.

Authorization requires idle shared evaluation/generation accounting without a model/tool halt or prior confirmation starts. It saves a separate private `confirmation_authorization.json` and registers the original complete plan in the existing ledger. It creates no new protocol lock or allowance. Repeating authorization verifies the same immutable record and completes missing registration if acknowledgement was lost; it cannot reset counts or change choices.

Every locked lab retains both arms and eight planned slots per arm. Only complete, settled confirmation claims agreeing with qualification are eligible for starts. Unstable, ambiguous and incomplete labs retain their membership and unattempted slots with unavailable rewards. No invalid task becomes an agent-failure zero. With no gradable labs, authorization records `no_gradable_labs`; an empty cohort records `no_primary_comparison` and starts no data workers or models.

## Dispatch and private grading

Initial starts follow the preparation's ordered slots: alternate global/calibrated arms within each replicate, reverse the first arm for alternating labs, and preserve cohort order. Unavailable members are skipped for dispatch while their slots stay registered. The journal checks the next eligible slot under the same lock that publishes the start intent. Two processes can overlap; completion order does not control dispatch order. An explicit technical retry keeps its predeclared attempt identity without changing the sequence of initial starts.

All original limits apply: at most two active or unresolved evaluation/generation intents together, 300 total evaluation starts, 272 initial primary starts, 128 initial confirmation starts and 20 technical retries across the study. Every launch consumes its intent before the process starts. Completed wrong answers and valid-task agent failures remain zero, pending attempts never expire automatically, and model/tool contract violations halt later starts.

Observations, question identity, visible-seed prices and charging remain bound to the original qualification environment. Both arms use its frozen confirmation visible order. The private confirmation cache cannot be served as an ordinary environment. Each episode receives its own private claim task from the independent assessment; shared environment tasks are untouched. Connection and sealed evaluator evidence bind the confirmation lock, aggregate/member assessment, independent task artifact, claim hash and original question hash.

The provider receives public description/budget and its MCP capability. Describe, experiment and submission responses contain no grade or private assessment. The first accepted answer and confidence remain final. Evidence verification checks the private context and reward against the independent claim. A committed answer survives a later provider error. Unknown provider failures remain unresolved until evidence supports an allowed classification; they do not automatically earn a retry.

Saved completion verification checks the original authorization, frozen independent assessment, every retained data-file hash and the per-attempt private context without requiring new execution permission or reacquiring the journal lock. It can acknowledge a saved outcome after source drift while new starts remain blocked. Altered truth files or grading bindings prevent replay; verification never launches another process, sweep or model call.

## Remaining work

[Full-cohort outcome collection and analysis](CONFIRMATION_RESULTS.md) now verify the raw evidence, close confirmation starts/outcome changes, replay matched classical policies and export paired effects with conservative missing-data bounds. [Shared Qualia invocation accounting](QUALIA_PRIMARY.md) is implemented. Actual observed UI compatibility, actual scientific records and bounded study runs, public results, a working demonstration and reproducible release still precede outreach. Synthetic execution controls establish engineering behavior, not company interest, scientific fidelity or measured model performance.
