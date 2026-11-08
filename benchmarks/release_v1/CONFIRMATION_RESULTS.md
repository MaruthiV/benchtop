# Confirmation collection and analysis

The results path closes the original confirmation plan, verifies its private provider/MCP evidence, replays frozen classical policies, and exports paired analysis. **Validation uses synthetic labs and simulated provider/paper/reviewer metadata. No actual-study confirmation result or measured model comparison exists.** Finish remaining implementation and public documentation before real model sessions; changed sources invalidate preparations and do not refund starts.

## Collect every planned slot

```sh
.venv/bin/python -m benchtop.cli study collect-confirmation PLAN_SHA256 \
  --output artifacts/confirmation-open.json
.venv/bin/python -m benchtop.cli study collect-confirmation PLAN_SHA256 \
  --output artifacts/confirmation-closed.json --close
```

The collector requires the registered canonical confirmation plan and verified independent assessment/authorization. Every finished attempt must have replay-verifiable process, stdout/stderr, MCP and sealed grading evidence. Opaque operator outcomes cannot substitute for provider evidence, including an earlier failed attempt hidden behind a later retry. Start-only intents remain explicitly pending with unknown costs.

The private snapshot retains every cohort member, both arms, all eight slots per arm, every attempt, answers, failure states, tool-event traces and costs. Stable, unstable, ambiguous and incomplete members stay in the same denominator. Unavailable tasks stay unattempted with unavailable rewards. Wrong answers and valid-task agent failures retain zero rewards; technical and unresolved outcomes remain missing.

`--close` freezes collection under the same journal lock used for starts and completion publication. It requires stopped evaluation and generation work. A stopped but unresolved outcome may remain unavailable; it cannot silently become zero. The closed marker blocks subsequent confirmation starts, retries and outcome changes. Identical saved completion acknowledgements remain idempotent. Other phases retain their own authorization gates.

Closed exports revalidate the original raw proof and are byte-identical copies. Their shared accounting is reconstructed at the original snapshot time, so later work in another phase cannot change the report. A lost output acknowledgement can recover from the durable private closure without new inference. Output paths must be new files outside canonical study state. Preserve partial work rather than deleting records to regain an allowance.

## Matched classical replay

```sh
.venv/bin/python -m benchtop.cli study replay-confirmation
.venv/bin/python -m benchtop.cli study inspect-confirmation-replay
```

Replay requires closed confirmation evidence and the original candidate-bound policy list, shared estimator and development majority label. It runs the three required acquisition policies and three sanity controls using policy seeds 0–199. It reuses the global confirmation budget, public question, paper visibility, frozen prices, confirmation visible order and tool-call cap. It creates no additional model comparison arm.

Policies receive only the public description and experiment callback. Cached visible observations drive acquisition; independent confirmation truth is used privately after acquisition for grading. Qualification questions and prices remain unchanged. Unavailable labs retain all six reported policy groups but execute no trials. Each gradable lab/policy keeps all 200 outcomes, including failures. A policy mean requires all 200 usable rewards.

`confirmation_replay/` is a separate private namespace with an immutable plan and exclusive per-trial start/finish records. Durable finishes are reused without another policy execution. An interrupted start without a finish blocks automatic reruns and remains pending for investigation. No reset or automatic retry command is provided. Aggregate publication can recover after acknowledgement loss without repeating trials.

These are cached policy trials, not new lab executions or model sessions. Controller CPU and wall time are measured separately; charged experiment costs retain their original units. Two hundred policy seeds reuse one lab's visible data and do not create 200 independent labs.

## Paired analysis and report

```sh
.venv/bin/python -m benchtop.cli study analyse-confirmation \
  --output artifacts/confirmation-analysis.json \
  --report artifacts/confirmation-report.md
```

JSON retains sanitized raw slots, normalized attempts, failure states, stable task status, budget choices/fallbacks, classical outcomes and costs. Markdown provides a readable summary. Capabilities, connection details, private file paths, grading contexts and raw provider metadata are excluded. Export starts no models, lab workers or classical trials. Missing classical replay remains explicitly pending.

The primary endpoint is each lab's calibrated-arm mixed-group indicator minus its global-arm indicator. An indicator is available only for a stable task with all eight usable slot rewards after closure. The complete-case estimate averages these paired differences equally across labs.

The interval resamples whole paired labs 10,000 times with RNG seed 1729 and reports the 95% percentile interval. It does not resample slots or policy seeds as independent observations. One complete lab gets a raw estimate and no between-lab interval. With no complete pairs, the estimate and interval are unavailable.

Worst-case bounds retain all locked labs: each unavailable arm's mixed-group indicator can be zero or one. A partially observed arm is still unavailable even when both reward values have already appeared. Zero labs yield unavailable bounds, not a zero effect. The report separately counts locked labs, stable labs, complete pairs and unavailable comparisons.

Secondary reporting includes observed accuracy and variance, failure counts, mixed groups without agent failures, selected budgets, stability and fallback counts. For each classical policy, the matched comparison averages per-lab global-arm accuracy minus its 200-replay accuracy; it uses the existing global arm and no extra model batch. Missing classical outcomes make that lab/policy comparison unavailable.

Costs include pilot overhead, every confirmation attempt and retry, the historical shared evaluation snapshot, generation/repair usage, collected CPU-work diagnostics and cached replay controller time. Missing token/time counters retain known partial sums and null totals. Worker-reported elapsed seconds are not labeled process CPU time. A smaller selected budget or mixed rewards alone does not establish efficiency; no break-even claim is calculated without a stated future-use assumption.

## Before outreach

This is an exploratory pilot with at most eight labs. Complete-case results are conditional on stable labels and complete groups; secondary comparisons are descriptive. [Shared Qualia invocation accounting](QUALIA_PRIMARY.md) is implemented. Actual observed UI compatibility, recorded scientific reviews and actual bounded study runs, a working demonstration, public release packaging and factual outreach review still remain.
