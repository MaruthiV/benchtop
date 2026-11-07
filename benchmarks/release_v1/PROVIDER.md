# Authorized subscription provider

`claude_mcp_runner_v1` implements one primary smoke, development or [fresh pilot](PILOTS.md) attempt through the installed Claude CLI. This checkpoint exercises the controller with test fixtures and read-only authentication metadata. **No actual model smoke, development lock, or primary performance result has been observed.** Fresh generation, cohort locking, pilot preparation/authorization and [pilot closure/calibration](PILOT_CALIBRATION.md) are implemented; confirmation execution remains pending.

## Metadata before inference

```sh
.venv/bin/python -m benchtop.cli study provider-preflight \
  --output artifacts/provider-preflight.json
```

This runs only `--version`, `--help`, and `auth status --json`. It records the resolved executable/checksum, required flags, personal Pro/Max subscription authentication, and a private account fingerprint. Email, organization names, credentials, and tokens are excluded. The CLI prints only non-identifying metadata. Nothing here proves model availability or a successful rollout. If a restricted execution environment hides the OS login, run this metadata command in the normal authenticated terminal.

Every actual attempt rechecks the binary, version, account, flags, and authentication. API/provider environment overrides and detected managed-policy files block authorization. Temperature and model seed must be explicitly unavailable because this adapter cannot apply them. Resolve a model ID before preparing the study; the synthetic engineering model file is only a schema fixture.

The invocation uses `--tools ""`, one explicit strict MCP configuration, three allowed lab tools, noninteractive permissions, a replacement system prompt, no session persistence, restricted mode, no settings discovery, disabled slash commands, and disabled hooks. Initialization must report exactly those tools and one connected Benchtop server. Changed reported models or capabilities withhold usable reward and halt new starts. These checks are not an OS sandbox or authenticated proof of provider execution. Detected managed policies are refused because they can override local settings. See the [official CLI reference](https://code.claude.com/docs/en/cli-reference).

The child environment disables CLAUDE.md loading, auto memory, tool search, and CLI auto-update. The server config sets `alwaysLoad: true` so the tools must connect before the first prompt. See [environment controls](https://code.claude.com/docs/en/env-vars) and [upfront MCP loading](https://code.claude.com/docs/en/mcp#exempt-a-server-from-deferral). Safe mode is intentionally excluded: the installed CLI filters external HTTP servers in that mode. The documented `EndConversation` tool is excluded from noninteractive print sessions; see [tool availability](https://code.claude.com/docs/en/tools-reference#endconversation-tool-behavior). Actual initialization still has to satisfy the exact tool contract.

The adapter avoids `--bare`, whose authentication path requires API credentials, and rejects API keys that could override subscription authentication. See [programmatic execution](https://code.claude.com/docs/en/headless) and [authentication precedence](https://support.claude.com/en/articles/12304248-manage-api-key-environment-variables-in-claude-code).

## Authorize smoke, then lock development

Finish the remaining study implementation before spending the bound smoke allowance. Freeze the code, refresh the foundation proof, complete the six real source-bound reference reviews, and save smoke/development [preparations](STUDY.md) with the same resolved model, reviews, and foundation file. Do not refresh or edit those inputs during the run.

```sh
.venv/bin/python -m benchtop.cli study plan \
  --mode study_preparation --phase smoke \
  --model-config artifacts/resolved-model.json \
  --foundation artifacts/foundation_check.json \
  --reviews-dir artifacts/reference-human-reviews \
  --output-dir artifacts/primary-smoke-plan

.venv/bin/python -m benchtop.cli study plan \
  --mode study_preparation --phase development \
  --model-config artifacts/resolved-model.json \
  --foundation artifacts/foundation_check.json \
  --reviews-dir artifacts/reference-human-reviews \
  --output-dir artifacts/primary-development-plan

.venv/bin/python -m benchtop.cli study authorize-smoke \
  artifacts/primary-smoke-plan --preflight artifacts/provider-preflight.json \
  --confirm-extra-usage-disabled
```

The last flag attests that subscription extra usage is disabled, required by this project's zero-new-spend constraint. Authentication metadata does not verify that account setting. Authorization invokes no model and is not the development lock. It creates `.state/study/CANONICAL_PROTOCOL_SHA256/` with one primary `ledger/`. A chosen second path cannot reset its allowance.

Existing legacy integration starts block authorization until explicitly reconciled; they are never discarded or assumed stopped. Once authorization exists, old `integration begin` calls are blocked under their own file lock. Qualia invocation through the shared primary integration reserve still needs supported routing. Engineering journals cannot authorize primary execution.

Read `plan_sha256` and the six `slot_id` values from the smoke plan. This command **starts one real model session**:

```sh
.venv/bin/python -m benchtop.cli study run-provider PLAN_SHA256 --slot SLOT_ID
```

Attempt each smoke slot once: the first two reviewed development labs at all three budgets. At most two attempts may be active across the canonical journal. Each intent consumes its allowance before launch, including interrupted starts. Failed, pending, and wrong attempts cannot be silently rerun. A wrong committed answer can satisfy a transport smoke; a correct answer is not required.

After six committed, model-verified outcomes, stopped processes, intact raw evidence, and successful use of all three MCP tools:

```sh
.venv/bin/python -m benchtop.cli study lock-development \
  artifacts/primary-development-plan
```

The verifier binds model, prompt, runtime, code, reviews, lab/task/pricing identities, and original smoke evidence. It writes private immutable `development_lock.json`, appends a compact public `bindings.json` record, and registers 48 development slots in the same journal. Those slots require the lock before `run-provider` can invoke them. Smoke outcomes are excluded from development selection. Interrupted publication remains visible for investigation; never delete records to regain capacity.

## Evidence and failures

The [owned controller](RUNNER.md) retains raw stdout/stderr, private connection/configuration, process status, sealed evaluator receipt, and checksummed completion evidence. The child receives public lab information and its episode capability. Cached experiments start no lab workers. Grades remain in private sealed receipts; submission returns only an acknowledgement.

A verified committed answer survives a later error or timeout. A known timeout without an answer is an agent-failure zero. Unknown provider errors remain unresolved: synthetic quota/error mappings never establish a real provider failure. Only independently reconciled, allowlisted technical failures can use an explicit retry. Changed model/capability contracts permanently halt new work in that journal.

Costs include experiment requests/charges, zero new lab executions, process wall time, and nonnegative reported `modelUsage.inputTokens` / `outputTokens`. Cache token counters remain in raw metadata without being silently added. Missing counters stay unknown. CLI invocations and conservative start intents are separate from measured inference or subscription billing.

```sh
.venv/bin/python -m benchtop.cli study resume-provider ATTEMPT_ID
```

Resume verifies saved files and reparses stdout, recording the original outcome without another model call or grading. It can preserve a stopped attempt after source drift while new attempts stay blocked. Missing stop evidence or incomplete files leave work pending. Changed model/capability contracts cannot be reconciled into usable primary data.

The [canonical calibration bridge](CALIBRATION.md#canonical-development-collection) verifies saved development evidence and runs the offline selector. Its collection closure blocks new development starts and outcome changes. The [candidate lock](CANDIDATES.md) then binds that choice, human baseline freeze, generation rules and archived paper versions. The [fresh driver](GENERATION.md) uses a separate text-only provider command with no built-in tools or MCP servers, and shares active-session accounting with evaluation.

Temporary positive tests use labeled simulated admission/auth/provider records, real local MCP transport and sandboxed synthetic labs, with locks only under temporary roots. They do not certify real reviews, provider compatibility, or study outcomes. Cohort/confirmation locks, independent confirmation collection, fresh selector/analysis export, shared Qualia invocation accounting, and presentation remain. Complete those code paths before live smoke: implementation edits invalidate authorization, and spent starts cannot be erased.
