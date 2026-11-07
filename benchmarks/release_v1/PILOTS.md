# Cohort-bound fresh pilots

Fresh pilot preparation and subscription-provider execution use the immutable [cohort lock](COHORT.md). There are three budgets and four independent replicate slots per lab, at most 96 initial pilot sessions for eight labs. Smaller cohorts reduce planned work. **Tests use synthetic labs, simulated provider metadata and real local MCP calls. No actual fresh model result or pilot authorization has been observed.**

Finish remaining study code and public documentation before actual model sessions. Source changes invalidate earlier preparations and cannot refund starts.

## Prepare without inference

```sh
.venv/bin/python -m benchtop.cli study plan-pilots \
  --output-dir artifacts/fresh-pilot-preparation
```

The planner reads the verified canonical cohort. It inherits all admitted labs in order, the original model configuration and prompt, global control budget, qualification tasks, frozen prices, four-seed visible order, code and runtime. The budgets are exactly 6%, 12% and 25%, with four independent slots each. Each group, slot and its initial/technical-retry attempt IDs use the protocol's hashes. It accepts no lab list, model, margin, label, budget or ordering override.

The saved plan and receipt are immutable preparation with `execution_ready: false`; preparing them starts no models, probes no live authentication, reserves no allowance and creates no study lock. A zero-member cohort produces an empty plan and cannot authorize pilots or a primary comparison. Generic historical/engineering preparation continues to reject fresh sources.

## Authorize the existing journal

```sh
.venv/bin/python -m benchtop.cli study authorize-pilots \
  artifacts/fresh-pilot-preparation
.venv/bin/python -m benchtop.cli study inspect-pilots
```

Authorization revalidates the complete cohort and plan, saved smoke authorization, subscription CLI/account binding and zero-extra-usage attestation. It requires an idle primary journal without a model/tool halt and no prior fresh model observations. The exact plan is registered in the original canonical journal; a separate private authorization record binds its ledger, cohort, plan and provider hashes. This is execution permission under the existing cohort lock, not an additional protocol lock. Public binding history is unchanged. Authorization itself starts zero models and cannot replace an earlier authorization or reset starts.

Inspection checks saved evidence without inference or live authentication. A copied journal cannot authorize new starts. Cohort/development checks reuse the journal context while a caller holds its lock; they do not acquire the same file lock recursively.

## Run one planned slot

Only after actual study prerequisites exist and the remaining code is frozen:

```sh
.venv/bin/python -m benchtop.cli study run-provider PLAN_SHA256 \
  --slot SLOT_ID
```

This uses the existing [provider runner](PROVIDER.md), its strict three-tool MCP connection and private grading. Each start verifies the current cohort, plan, authorization, source files, qualification task, prices, model and runtime. Pilot sources must contain only visible and qualification data; confirmation seeds 16–27 remain unevaluated. Qualification labels, hidden seeds, results files and grades do not enter the agent's tools or prompt. The controller reads cached visible observations without changing the frozen qualification snapshot.

Each invoked attempt consumes the original shared allowance. Two pending/unreconciled sessions fill the concurrency ceiling; start intents never expire or get refunded. Wrong completed answers and exits without an answer remain zero rewards. Model/capability changes halt usable collection. Technical missing outcomes retain the existing single explicit retry rule. Retained evidence can be resumed without another model invocation. A pilot-collection or confirmation-lock marker closes new pilot starts and outcome changes; identical recorded acknowledgements remain immutable.

## What remains

[Canonical pilot closure and per-lab calibration](PILOT_CALIBRATION.md) now verify provider/MCP evidence, retain every planned slot and apply the preregistered fallbacks. Confirmation locking, separate confirmation data/truth/stability, rollout collection, matched classical replay and full-denominator analysis remain. Actual scientific reviews and bounded runs, shared Qualia invocation accounting, a working demonstration, reproducible reporting and public release still precede outreach. Pilot tests establish controller behavior, not measured model difficulty.
