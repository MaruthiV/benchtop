# Canonical fresh pilot collection

`study calibrate-pilots` verifies the canonical [fresh pilot](PILOTS.md) journal and feeds each cohort member's full 12-slot grid to the existing [budget selector](CALIBRATION.md). It starts no models, probes no live authentication, runs no lab sweeps and does not regrade saved answers. **Tests use synthetic labs and simulated provider/paper responses with real local MCP calls. No actual pilot collection, selected fresh budget or model performance result exists.**

Finish the remaining study code and public documentation before actual model sessions. Source changes invalidate bound preparations and cannot refund starts.

## Inspect before closing

```sh
.venv/bin/python -m benchtop.cli study calibrate-pilots PILOT_PLAN_SHA256 \
  --output artifacts/pilot-progress.json
```

The command accepts only a registered, authorized fresh pilot plan from the original canonical primary journal. It verifies the current cohort, plan, source/runtime, saved authorization, each finished attempt's retained provider/MCP evidence and the journal outcome against that evidence. Operator-only finish records, altered raw streams, wrong phases, missing authorization and model/tool halts cannot become calibration results.

The export retains every cohort member in order and all three budgets with four independent replicate slots each. An unattempted slot stays empty; a pending intent stays unresolved with unknown costs. Completed wrong answers and agent-failure zeros count as observed outcomes. Technical missing or unresolved outcomes remain missing. An explicit permitted retry retains both attempts' costs but contributes at most one final slot outcome. An opaque earlier attempt cannot disappear behind a later one.

Selection is separate for each lab. Open exports always report `pending` with no chosen budget, including when all observations are complete. Smoke and development outcomes never enter a lab's success counts. The frozen global control comes from the candidate/cohort binding and has no override argument.

## Close and freeze choices

```sh
.venv/bin/python -m benchtop.cli study calibrate-pilots PILOT_PLAN_SHA256 \
  --output artifacts/pilot-closed.json --close
```

Closure requires all active evaluation intents to be reconciled and uses the same journal lock as new starts. It writes a private `pilot_collection.json` under the canonical study root. Existing pilot guards then refuse new pilot starts and outcome changes. An identical saved completion acknowledgement can still be read; closure cannot acquire extra observations.

Closing an incomplete collection is allowed and remains explicitly incomplete. Each lab uses the preregistered rule:

- Any budget lacking four completed pilot outcomes: `incomplete_pilots`, using the frozen global budget.
- A complete grid with no observed mixed budget: `no_observed_mixed_budget`, also using the global budget.
- Otherwise: choose the smallest budget within the fixed tolerance of the best Beta predictive mixed-group objective.

The result preserves success/missing counts, posterior intervals, objectives, diagnostics, planned slot/attempt identities, chosen budget and fallback reason for every lab. No candidate or lab is replaced because calibration is inconvenient. An empty cohort has no authorized pilot grid and no primary comparison; it does not acquire a budget through this bridge.

## Evidence and accounting

The closed snapshot binds the cohort, pilot authorization, shared ledger, protocol, plan/group identities, normalized per-lab inputs, selector output, and private raw provider/MCP files. A later read verifies those records and recomputes the same selector outputs from the frozen pilot observations. Verification can reuse a caller's held canonical journal context without acquiring the same file lock recursively.

Pilot costs include all recorded attempts and retries; unknown fields keep null totals and known partial sums. The snapshot also records global evaluation start/cost accounting at closure. That global snapshot stays fixed as later phases add work. Generation accounting remains separate. These are conservative recorded intents and adapter-reported costs, not authenticated provider usage or billing.

Exports are exclusive and cannot overwrite existing files. Repeated exports after closure verify the saved collection and return byte-identical copies, even without `--close`. A failed output acknowledgement leaves an already saved closure intact; exporting again to a new path recovers it without another model call. Raw evidence or frozen selection changes fail verification. Preserve partial/corrupt state for investigation rather than erasing starts or reopening collection.

## What remains

This snapshot is collection finalization, not the protocol's confirmation lock or execution authorization. Confirmation planning must bind all pilot records and calibrated choices or fallbacks before inspecting confirmation outcomes. Separate confirmation data/truth/stability, model collection, classical replay and full-cohort analysis remain. Actual scientific records and bounded runs, shared Qualia invocation accounting, the demonstration, reproducible reporting and public release still precede outreach.
