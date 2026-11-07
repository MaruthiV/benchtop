# Session accounting and reconciliation

`session_ledger_v1` records start intents, immutable outcomes, and explicit reconciliation in one shared directory. It supports engineering journals and a separately authorized canonical primary journal. The journal itself invokes no model or process. The [scripted runner](RUNNER.md) exercises engineering controls; the [provider path](PROVIDER.md) checks current authorization before primary starts.

## Create and inspect an engineering journal

Prepare a current [engineering study plan](STUDY.md), then initialize its ledger:

```sh
.venv/bin/python -m benchtop.cli study plan \
  --mode engineering --phase smoke \
  --env envs/2610-36935-gaussian-process-bandits-for \
  --model-config benchmarks/release_v1/study/engineering_model.json \
  --output-dir artifacts/session-engineering-plan

.venv/bin/python -m benchtop.cli study sessions init \
  artifacts/session-engineering-plan \
  --output-dir artifacts/session-engineering-ledger

.venv/bin/python -m benchtop.cli study sessions inspect \
  artifacts/session-engineering-ledger \
  --output artifacts/session-engineering-snapshot.json
```

`sessions register LEDGER PREPARATION` adds another compatible engineering plan. Model configuration, prompts, code, runtime, and protocol must match the ledger. Re-registering the same plan is idempotent. The general engineering initializer refuses `study_preparation`; those plans require `authorize-smoke` or `lock-development` through the primary path. The model IDs in the supplied engineering file are synthetic, not identities measured from a real invocation.

Directories and exports are exclusive, with repository destinations under ignored `artifacts/`. A ledger has an immutable sealed `ledger.json`, registered snapshots under `plans/`, and per-attempt JSON under `attempts/`. Each new instance has its own ID. Inspection snapshots bind their content with `snapshot_sha256`; export cannot overwrite an existing file or place an unrelated JSON inside the journal.

## One accounting scope

The [fresh generation journal](GENERATION.md) records its calls separately from evaluation episode allowances. Primary starts include pending generation intents when enforcing the shared two-active-session ceiling; generation checks the same evaluation journal before launching. Inspection reports evaluation, generation and combined active intents. Neither journal silently expires unknown processes.

Mutations hold a local filesystem lock. A begin transaction reads all registered histories and checks the shared allowances before publishing its intent. Registering another plan does not reset counters.

| Allowance | Maximum |
|---|---:|
| In-flight or unreconciled attempts | 2 |
| Total start intents | 300 |
| Initial primary starts | 272 |
| Initial development / fresh-pilot / confirmation starts | 48 / 96 / 128 |
| Initial smoke / integration starts | 6 / 2 |
| Technical retry starts across phases | 20 |
| Technical retries per slot | 1 |

The arithmetic includes the full frozen protocol. Current plans support smoke/development engineering controls and authorized primary attempts; fresh and primary integration execution are not enabled by these limits.

Engineering counters belong to one directory and do not coordinate unrelated commands. Primary starts must use the single canonical journal, preventing copied directories from resetting capacity. Existing legacy integration usage blocks authorization pending explicit reconciliation; the legacy begin path is blocked after authorization. Shared primary integration routing remains pending. These controls do not limit unrelated subscription processes.

## Start intents and interruption

The Python `begin(directory, plan_sha256, slot_id, retry_technical=False)` API writes a start record before a future adapter could invoke a model. It verifies current code/runtime/protocol and the complete registered plan. Begin does not invoke anything itself. The record carries the exact plan, slot, attempt, phase, bucket, and UTC time.

Each recorded intent conservatively consumes an allowance and is never refunded. A crash before an actual invocation can therefore overcount usage; it cannot create a hidden extra allowance. Reported intent counts must not be described as measured provider calls. The ledger's own model-start count stays zero.

A start without a finish is `in_flight_or_interrupted`. Time passing does not prove that a model stopped. It keeps its concurrency slot and cannot be automatically rerun. The adapter must inspect its process/server evidence and reconcile it. After uncertain work is verified stopped, it can be finished as `unresolved`, releasing concurrency while still forbidding a retry until submission status is known.

The core writes atomic exclusive files and preserves unfinished work. It handles ordinary process interruption; it is not a power-loss, distributed-consensus, or authenticated event-log guarantee. Temporary unpublished files do not count as completed records. Published orphaned, malformed, checksum-invalid, or allowance-inconsistent records fail visibly.

[Fresh pilot authorization](PILOTS.md) registers the cohort-bound three-budget/four-replicate grid in this same primary journal. Starts inherit all existing caps, retain the original ledger identity, revalidate frozen qualification sources and require separate saved authorization. [Canonical pilot closure](PILOT_CALIBRATION.md) verifies the provider/MCP evidence and freezes per-lab choices or fallbacks under this same lock. Pilot-collection or confirmation-lock markers block new pilot starts and outcome changes. Closed-development and closed-pilot verification accept the caller's canonical journal context to avoid reacquiring its held lock.

## Finish and reconcile

The future adapter calls `finish` with the normalized outcome and:

- `process_stopped = true`, supplied only after verifying the old process ended.
- `submission_status`: `committed`, `not_committed`, or `unknown`.
- `reported_model_id`, or null when that identity is unavailable.
- `evidence_sha256`, referring to retained adapter/server reconciliation evidence.

The journal validates structure and references; it does not authenticate a caller's process or server claims, read a receipt from the hash alone, or reconcile the server itself. That work remains the runner's responsibility.

Completed outcomes require a known committed submission and binary reward. Agent failures require an allowlisted cause and zero reward. Technical failures require an allowlisted infrastructure cause, null reward, a stopped process, and a known absence of a committed submission. Unknown submission status must remain unresolved. Incorrect answers and agent failures are final, including zero scores.

An explicit `retry_technical=True` can begin attempt one only after a definite technical failure. A second retry is refused. A technical-looking transport error with uncertain submission is not enough: the first answer may already be committed.

`reconcile` appends one separate resolution to an unresolved finish. A late committed answer becomes its usable outcome without another start. A proven technical failure with no submission can become retryable. The original finish remains byte-for-byte intact; a resolution references its checksum. Reconciliation cannot overwrite an already-final answer, replace an existing resolution, or turn a known reported cost into another value. Exact repeated completion calls return the existing record.

If a reported model differs from the bound identity or the provider reports a capability-contract violation, the claimed outcome remains diagnostic, its usable reward becomes unavailable, and the entire ledger halts new work. Other already-started attempts can still finish. An unavailable model identity also keeps a submitted outcome unresolved until verified, without silently treating it as a primary reward.

New plans and starts require current source/runtime/protocol bindings. Late completion/reconciliation evidence can still be retained after those inputs drift, using the recorded protocol and a checksum of the recorder implementation/runtime. It authorizes no new execution. The adapter must still verify that the actual server/model evidence belongs to the original attempt and versions. This keeps a code change from erasing an already-committed answer.

## Costs and denominators

The [canonical calibration bridge](CALIBRATION.md#canonical-development-collection) can close development collection under this same journal lock. Closed development refuses new starts and new finish/resolution records, including retries. Exact repeated completion acknowledgements remain idempotent. Other phases retain their separate authorization gates. Closure with a pending process is refused; incomplete stopped outcomes remain missing in the full planned denominator.

Inspection retains every planned slot, including unattempted work, pending starts, terminal zeros, unresolved outcomes, and technical failures. It reports all start/retry intents and current concurrency occupancy. A resolved attempt contributes one effective outcome, while its earlier raw records remain available.

Costs use the selector's normalized fields. Every attempt contributes, including failed attempts and retries. Resolutions can fill previously unknown costs and inherit previously known values; they do not count as another invocation. Missing cost fields remain unknown, with a known partial sum and a null total. Zero is used only for an actual reported zero or no recorded attempt. No tokens or wall time are inferred from a receipt or from intent counts.

## Validation and next work

Controls cover shared concurrency, immutable wrong answers, terminal agent failures, pending-process occupancy, explicit and bounded technical retry, late committed answers, unknown/model-mismatched outcomes, halted starts, all protocol allowance boundaries, the actual 20-retry limit, cross-plan smoke accounting, bad/corrupt/orphaned evidence, source drift, unknown costs, immutable reconciliation, exclusive snapshots, and copied-project CLI use. All synthetic start records are explicitly engineering data; no real model or paper experiment is run by these controls.

The scripted adapter now connects owned process/MCP evidence to these accounting primitives. The [provider adapter](PROVIDER.md) now checks canonical smoke/development authorization; actual provider smoke evidence, fresh locks, and shared Qualia invocation routing remain pending. Primary evaluation, fresh confirmation grading, Qualia UI verification, and outreach readiness remain separate work.
