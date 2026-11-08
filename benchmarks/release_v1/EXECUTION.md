# Execution through outreach

Synthetic checks establish engineering behavior. Real reviews, resolved provider metadata, CLI smoke, an observed Qualia conversation, the bounded study and a public demonstration remain separate gates. Start at the first incomplete step. A favorable calibration result is not required for an honest release.

## 1. Record reviews and freeze evaluation inputs

Transfer completed human notes into the six files under `artifacts/reference-human-reviews` using the [review packet](admission/REVIEW.md). Retain actual names, dates, citations, scientific judgments and effect-margin rationale. Do not mark an unreviewed template accepted. Keep exclusions and ambiguity visible.

Finish code, dependencies and public docs before preparing the study. Preserve a dedicated evaluation workspace through completion: its foundation binds README/docs as well as code/data. Publication later creates a separate directory. Editing the evaluation README after model starts invalidates current-evidence checks.

```sh
.venv/bin/python -m benchtop.cli verify --output artifacts/foundation_check.json
.venv/bin/python -m benchtop.cli study provider-preflight --output artifacts/provider-preflight.json
```

Preflight starts no model. Resolve the actual model configuration using [PROVIDER.md](PROVIDER.md) and [STUDY.md](STUDY.md), saving `artifacts/resolved-model.json`. Pin the exact reported ID and supported controls; unavailable controls stay unavailable. Use existing subscription capacity, disable extra usage and keep new spending at $0. Guessed IDs and synthetic metadata cannot substitute for this step.

## 2. Validate classical controls

```sh
.venv/bin/python -m benchtop.replay gate --reviews-dir artifacts/reference-human-reviews
.venv/bin/python -m benchtop.replay plan --mode development --repeats 200 --reviews-dir artifacts/reference-human-reviews --output artifacts/classical-development-plan.json
.venv/bin/python -m benchtop.replay run artifacts/classical-development-plan.json --reviews-dir artifacts/reference-human-reviews --output-dir artifacts/classical-development
.venv/bin/python -m benchtop.replay summary artifacts/classical-development --output artifacts/classical-development-summary.json
```

This uses cached data, without models. Complete the dated human correctness/sensitivity review in [CANDIDATES.md](CANDIDATES.md), referencing the full replay and shared estimator. Save the actual baseline freeze before candidate locking.

## 3. Initialize the original journal and verify real connectivity

```sh
.venv/bin/python -m benchtop.cli study plan --mode study_preparation --phase smoke --model-config artifacts/resolved-model.json --foundation artifacts/foundation_check.json --reviews-dir artifacts/reference-human-reviews --output-dir artifacts/smoke-preparation
.venv/bin/python -m benchtop.cli study plan --mode study_preparation --phase development --model-config artifacts/resolved-model.json --foundation artifacts/foundation_check.json --reviews-dir artifacts/reference-human-reviews --output-dir artifacts/development-preparation
.venv/bin/python -m benchtop.cli study authorize-smoke artifacts/smoke-preparation --preflight artifacts/provider-preflight.json --confirm-extra-usage-disabled
```

The next command starts **one real subscription model attempt**. Use the saved plan hash and one unused slot ID. Run the six fixed smoke slots and inspect every retained provider/MCP result.

```sh
.venv/bin/python -m benchtop.cli study run-provider PLAN_SHA256 --slot SLOT_ID
```

A wrong answer can be a valid zero-reward smoke result; simulated transport is not real smoke. After all six real outcomes pass the model/three-tool checks:

```sh
.venv/bin/python -m benchtop.cli study lock-development artifacts/development-preparation
.venv/bin/python -m benchtop.cli study authorize-qualia
.venv/bin/python -m benchtop.cli study inspect-qualia
```

Follow [QUALIA_PRIMARY.md](QUALIA_PRIMARY.md): prepare an inactive connection, configure the actual client privately, reserve before sending one prompt, stop the client and collect the observed capture. Use included/free usage only. Unknown model identity remains unavailable; connectivity alone does not supply a scientific reward. Neither metadata nor a surrogate client satisfies actual UI observation. Keep capabilities and hidden grades out of recordings.

## 4. Complete development and freeze candidates

Run the 48 fixed development slots with `study run-provider PLAN_SHA256 --slot SLOT_ID`. Reconcile all work, then:

```sh
.venv/bin/python -m benchtop.cli study calibrate-development DEVELOPMENT_PLAN_SHA256 --output artifacts/development-calibration.json --close
.venv/bin/python -m benchtop.cli study archive-candidates --exclusions artifacts/candidate-exclusions.json --output-dir artifacts/candidate-archive
.venv/bin/python -m benchtop.cli study lock-candidates artifacts/candidate-archive --baseline-review artifacts/baseline-freeze.json --generation-models artifacts/generation-models.json
```

Prepare the actual exclusions and stage-specific model configurations using [CANDIDATES.md](CANDIDATES.md). Keep the protocol's date window, archived versions and ordering. Do not select papers after confirmation outcomes.

## 5. Generate, review and qualify the fresh cohort

```sh
.venv/bin/python -m benchtop.cli study generate-next
.venv/bin/python -m benchtop.cli study generation-status
```

[Generation](GENERATION.md) stops for human review of source fidelity, implementation, metric, margin and evidence. Record that actual review before qualification:

```sh
.venv/bin/python -m benchtop.cli study review-generated PATH_TO_COMPLETED_HUMAN_REVIEW
.venv/bin/python -m benchtop.cli study qualify-generated
```

Qualification uses only four visible and twelve qualification seeds. Follow the bounded funnel to its stopping rule. Retain rejected, failed and skipped candidates; do not manufacture eight admissions. Then:

```sh
.venv/bin/python -m benchtop.cli study lock-cohort
.venv/bin/python -m benchtop.cli study plan-pilots --output-dir artifacts/pilot-preparation
.venv/bin/python -m benchtop.cli study authorize-pilots artifacts/pilot-preparation
```

## 6. Close pilots and lock confirmation before its data

Run the fixed twelve pilot slots per member. Reconcile active work before closing; missing pilots retain the global-budget fallback.

```sh
.venv/bin/python -m benchtop.cli study calibrate-pilots PILOT_PLAN_SHA256 --output artifacts/pilot-calibration.json --close
.venv/bin/python -m benchtop.cli study plan-confirmation --output-dir artifacts/confirmation-preparation
.venv/bin/python -m benchtop.cli study lock-confirmation artifacts/confirmation-preparation
.venv/bin/python -m benchtop.cli study collect-confirmation-data
.venv/bin/python -m benchtop.cli study authorize-confirmation artifacts/confirmation-preparation
.venv/bin/python -m benchtop.cli study inspect-confirmation-execution
```

Only now collect the independent twelve confirmation seeds. Instability, ambiguity and incomplete data stay in the full cohort. Run fixed interleaved confirmation slots only where independent truth allows grading. Retain unavailable members and slots in results.

## 7. Close results and reproduce a separate public package

```sh
.venv/bin/python -m benchtop.cli study collect-confirmation CONFIRMATION_PLAN_SHA256 --output artifacts/confirmation-collection.json --close
.venv/bin/python -m benchtop.cli study replay-confirmation
.venv/bin/python -m benchtop.cli study analyse-confirmation --output artifacts/confirmation-analysis.json --report artifacts/confirmation-report.md
.venv/bin/python -m benchtop.cli release status
.venv/bin/python -m benchtop.cli release build --output-dir artifacts/public-release
.venv/bin/python -m benchtop.cli release reproduce artifacts/public-release
```

The builder starts no trials/models; finish the fixed classical collection first. Reproduction reconstructs cached truth, budget choices, grading, completed classical acquisitions and analysis. Recorded failures remain failed/missing. [PUBLICATION.md](PUBLICATION.md) explains portability, privacy and the frozen README mapping. Inspect package files, figures and limitations; publish from that separate copy and preserve the evaluation workspace.

## 8. Demonstrate and draft

Record a real 60–90 second demonstration: question, budgeted experiment, committed answer and public result. Show public observations only; revoke the demonstrated capability before publication. Publish the reproducible repository/video and check both from a clean view. Draft a short personal email with the Summer 2027 SWE context, one measured finding or concrete engineering lesson, links and a small request for a conversation. Do not turn synthetic checks into usage claims or missing/negative results into improvement claims. Drafting/readiness checks do not authorize sending.

## Limits and recovery throughout

Maximum two active model sessions across evaluation, generation and UI. The shared journal permits 300 evaluation/integration start intents total: 272 initial primary slots, six smoke slots, two UI slots and up to twenty technical retries. Smaller cohorts reduce the primary ceiling. Generation has its separate frozen funnel limits and shares concurrency. UI has no retry allowance. Failures/lost acknowledgements retain consumption.

Never automatically retry a real invocation. Use `study resume-provider ATTEMPT_ID` for saved CLI evidence and the documented generation/UI resume commands for their retained evidence. Technical retries require classification, per-slot eligibility and reserve. Timeout does not prove a remote client stopped; pending work never silently expires. Known model drift halts starts. Never edit sources or recreate the journal to bypass a gate.
