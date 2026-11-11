# Smoke-prompt continuation, before primary model data

V1's first real Haiku episode correctly used experiments/submission and retained reward 1. Its prompt embedded the whole description and described `describe_lab` as redundant. The model did not call it, while the smoke validator requires that call in every episode. No primary development, pilot or confirmation model data had been collected.

This prospective continuation explicitly instructs the model to call `describe_lab` before experiments. It keeps the strict three-tool smoke checks. No old receipt is edited, no valid completion is reclassified as a technical failure, and the v1 smoke is excluded from primary outcomes. The original [v1 protocol](../release_v1/spec.json), its journal and the frozen source/input epoch remain intact.

## Accounting across versions

- V1 consumed one start and has zero active sessions.
- V2 permits at most 299 further evaluation/integration starts: 272 primary, six smoke, two UI and nineteen technical retries.
- Together these stay within the original 300-start ceiling. No attempt is refunded or relabeled as a retry.
- V2 authorizes only after verifying the exact v1 journal identity, complete private-file inventory, consumed counts, zero active work and unchanged scientific protocol. It retains the same pinned model configuration.
- An exclusive continuation record retires v1 from new starts in this implementation. Every v2 phase/start verifies that authority and the unchanged parent. Parent changes block further starts.
- Earlier adapter-reported usage remains separate, visible overhead; it is not pooled into primary performance or silently treated as free.

The scientific question, lab selection, seed partitions, grader, margins, budgets, policies, calibration rule, confirmation design and analysis are unchanged. The ordered reference list and historical diagnostics in [references.json](references.json) are a declared derivative of the original manifest; no original scientific evidence has been rewritten or claimed to have run under v2.

## Explicit protocol selection

```sh
BENCHTOP_PROTOCOL=release_v2 uv run python -m benchtop.cli study gate --model-config artifacts/resolved-model.json
```

Use this prefix for v2 planning, authorization, model runs, generation, calibration, UI operations and reproduction. Omitting it selects v1 for compatibility; retired v1 cannot start new work. Arbitrary protocol paths or names are refused. The original [execution order](../release_v1/EXECUTION.md) still applies with the prefix and new output directories.

The package's generated README records the correct reproduction command. Numerical public reproduction requires no private parent journal; live study authorization does require the retained parent and its immutable continuation record. Integrity hashes are not independent authentication of operator/provider identity.
