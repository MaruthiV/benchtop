# Run a budgeted experiment over MCP

This is a deterministic classical policy talking to a real local HTTP MCP server. It gives a reviewer a runnable example without model credentials. It is separate from the real-model study and the actual Qualia demonstration.

```sh
uv sync --locked
uv run python -m benchmarks.release_v1.walkthrough --output-dir artifacts/walkthrough
```

Open `artifacts/walkthrough/report.md`, then `report.json` for the full public transcript. Use a new output directory for each repeat; an existing report is never overwritten. The command needs permission to bind a loopback port. It starts no model or lab worker, needs no tunnel and stops its own server when finished.

## What happens

1. Load the preserved data-curation lab in read-only replay mode and derive its corrected task in memory. Original files stay unchanged.
2. Start a separate local cached server and discover exactly `describe_lab`, `run_experiments` and `submit_answer`.
3. Give the paired-uniform policy the public description and experiment callback. The fixed 12% budget and policy seed 0 are not chosen after seeing a successful answer.
4. Record public experiment observations and commit the policy's answer. Submission returns only `received: true`.
5. Probe repeated submission: it must return the same grade-free receipt. This is an explicit adversarial check after the policy's submission, not a second model conversation.
6. Seal the episode, revoke its capability, and only then read the evaluator's held-out grade. The old capability must return HTTP 401.

The report separates the agent transcript from the evaluator's post-submission result. Transport logs are suppressed so the terminal does not print capability URLs. No raw connection, token or URL is exported. A failure retains an `.incomplete` marker rather than producing a passing report.

## Interpreting a repeat

Episode identities, timestamps, controller timing and the report hash change between runs. With unchanged code, dependencies and data, public numerical observations, the chosen answer, held-out grade and frozen charge should repeat. The automated checks compare those quantities while allowing a fresh episode identity.

The budget is a fraction of the frozen full-grid charging schedule. Reported charges are not new CPU timings. The policy never receives seed identities or grading fields through its MCP responses.

## Limits

This uses historical cached data and the corrected `seed_block_t_v1` grader. It creates no human scientific admission, provider session, primary journal or fresh confirmation evidence. A successful scripted answer is not a measurement of model performance or a claim that the miniature faithfully reproduces its source paper. It cannot count as an observed Qualia client session.

The [admission evidence](admission/REPORT.md), [real-study runbook](EXECUTION.md) and [Qualia primary workflow](QUALIA_PRIMARY.md) cover those separate steps. Keep the six actual reference reviews unresolved until they have been performed.
