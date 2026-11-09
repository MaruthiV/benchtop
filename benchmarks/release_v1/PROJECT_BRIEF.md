# Benchtop in one minute

Paper-inspired miniature labs for agents that choose experiments. Each episode exposes a question, a compute budget and three MCP tools; grading uses separate held-out runs. The engineering problem is making the reward inspectable before treating it as useful feedback.

## A positive score can hide a bad comparison

The CaLLM-inspired miniature has a favorable cached treatment contrast, but its routed predictors are logistic heads while its baseline is a sequentially trained global MLP. Architecture, output scope and training change along with routing. That reward does not isolate the effect of routing.

The [paper-to-code review](admission/REPORT.md#2610-36994-callm-continual-adaptation-of) shows the implementation, cached contrast, assumptions and unresolved questions together. This is a problem in the generated miniature, not a finding that the source paper is wrong. Its human admission remains pending.

## Inspect the evidence

| Question | Start here |
|---|---|
| Can a seemingly successful lab give misleading feedback? | [CaLLM review and source links](admission/REPORT.md#2610-36994-callm-continual-adaptation-of) |
| Are concrete failure cases tested? | [Controlled defect suite](admission/defects/README.md): six deliberate defects and ten controls; all 16 expected outcomes in the recorded run, including 24 sandbox jobs. These are covered cases, not a general detection rate. |
| Does grading preserve shared-seed dependence? | [Seed-block grading](GRADING.md): average paired differences within each seed, then estimate uncertainty across seeds; ambiguous claims stay unavailable. |
| Can the agent see the private grade? | [Agent/evaluator separation](ACCESS.md): episode-scoped MCP tools and separate local administration; submission returns a receipt. |
| Can someone reproduce the engineering evidence? | [Offline verification](REPRODUCTION.md) and [publication reproducer](PUBLICATION.md). |

## Run without a model

For a single budgeted episode with a public MCP transcript and a separate evaluator result, follow the [runnable walkthrough](WALKTHROUGH.md). It checks grade-free submission, repeated receipts and capability revocation without a model or lab worker.

In a disposable checkout, after installing the locked dependencies:

```sh
uv sync --locked
uv run python -m benchtop.cli verify --output artifacts/foundation_check.json
```

This reconstructs the archived 30-lab, 78-question historical report from public inputs and checks the cached adapter, without model inference. The [corrected grading review](GRADING.md) retains 75 questions; historical agent scores are not new evaluations under the corrected grader. Preserve any frozen evaluation workspace when trying commands.

## Preparation status, November 8, 2026

At this preparation checkpoint, the fresh budget-calibration study, actual Qualia UI observation and public demonstration are still pending. Calibration selection, independent confirmation, matched classical replay and sanitized release packaging are implemented and tested with synthetic controls. That does not establish a measured calibration benefit, scientific fidelity across arbitrary papers or successful Qualia usage. Final results belong in the separate release's report; this frozen brief records the preparation state.

The [execution runbook](EXECUTION.md) defines the remaining actual work. The intended final evidence is a reproducible result, including missing or negative outcomes, plus a short demonstration of the question, experiment and committed answer.
