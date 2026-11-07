# Reconciled engineering runner

`scripted_mcp_runner_v1` connects immutable plans and session accounting to an owned process and an actual MCP endpoint. It uses a deterministic Python stand-in for an agent. **It invokes no language model and enables no primary-study phase.** A separate [provider adapter](PROVIDER.md) supports authorized smoke/development attempts in a canonical primary journal. Scripted evidence remains excluded from its smoke gate; no actual provider smoke or study result has been observed.

## Run one control

Prepare a current engineering plan and journal using [SESSIONS.md](SESSIONS.md). Read a plan's `plan_sha256` and one `slot_id` from its JSON, then run:

```sh
.venv/bin/python -m benchtop.cli study run-engineering \
  artifacts/session-engineering-ledger PLAN_SHA256 \
  --slot SLOT_ID --answer baseline_better
```

The scripted client supplies the requested label. It does not derive an answer from the evaluator's hidden truth, and its score is not evidence of model ability. All three claim labels are accepted control inputs. A single invocation attempts one slot; an already-final or pending slot is not automatically relaunched.

`--scenario` supports `submit`, `no_answer`, `quota`, `unknown_error`, `post_submit_error`, `timeout`, `post_submit_timeout`, and `model_change`. These are deliberately constructed controls. For example, `post_submit_error` submits through MCP before reporting a synthetic transport error. The committed answer still wins over that error. `--timeout` defaults to 20 seconds and cannot exceed the protocol's 1,200-second ceiling. An explicit `--retry-technical` is required for a permitted retry.

The implementation runs the fixed `_scripted_agent.py` helper with the current Python interpreter in isolated mode. There is no arbitrary-command or Claude-execution option. The helper uses standard-library HTTP against its owned loopback capability URL, initializes MCP, checks the three tools, describes the lab, requests a visible paired experiment, and optionally submits. It emits clearly synthetic process metadata plus the actual public tool replies. It sees no admin credential, qualification/confirmation data, evaluator grade, or actual visible-seed IDs in those replies.

This is a functional transport/process control, not an adversarial agent-filesystem sandbox or a tested real-provider configuration. The earlier Claude CLI adapter is not called or modified. Local CLI `--version`/`--help` inspection is not a model smoke or provider-compatibility pass.

## Owned episode and finalization

Each attempt gets a new episode on a temporary loopback agent-only server. Existing servers, profiles, tunnels, and Qualia sessions are untouched. The controller loads the plan's verified source, uses corrected read-only historical views where required, and pins its charging identity, budget, visible ordering, paper visibility, task/version, and tool-call cap. It starts no lab workers; experiments are cached reads.

The model/process side receives only its episode capability. Trusted preparation, summary, and sealing operate inside the controller. No administrative HTTP listener is exposed by this owned server.

`Hub.seal` and the authenticated admin `POST /episodes/{id}/seal` provide a finalization barrier. Sealing waits for the per-episode mutation lock, revokes the capability, and blocks queued/future runs and submissions from mutating the episode. It returns a private summary and trace without replacing a committed answer. Repeated seals retain the original result. This prevents a late in-flight submission from racing a no-answer classification.

The barrier covers episode mutations through the Hub. It does not prove that a separate client process stopped or that all shared background CPU jobs stopped. This runner uses cached execution only; other live-worker accounting remains separate. Trace bookkeeping can be incomplete at a finalization boundary, in which case unverified provenance remains unresolved rather than inventing a reward.

## Process and outcome evidence

The ledger publishes a conservative start intent before the helper launches. Each attempt has a private 0700 workspace, with 0600 connection, stdout/stderr, process, sealed-receipt, and evidence files. Capabilities remain in private connection data and are never printed by the CLI. Raw public tool replies and grader-free submission receipts are preserved; evaluator grades appear only in the separate private sealed receipt.

The helper owns a new process group. On timeout the runner kills that owned group and waits for the child. It confirms the group is gone before releasing the attempt. If process stop is uncertain, it keeps the ledger attempt pending. It never kills a PID recovered from an old record or assumes that age establishes process death. Startup/interruption failures without a complete durable receipt also remain pending for explicit investigation; no automatic retry fills that gap.

Stdout/stderr are retained on disk and hashed incrementally. Only a bounded prefix is parsed. Truncated process metadata is not enough to establish a usable outcome. A committed result additionally requires matching episode/version/order/budget/call-cap bindings, MCP submission attribution, and a grader-free receipt. Missing or conflicting provenance stays unresolved, with the raw receipt retained.

With valid evidence and a confirmed stopped process:

| Evidence | Outcome |
|---|---|
| Verified committed MCP answer | Preserve the evaluator's binary reward, even after a timeout or reported process error |
| Timeout with a sealed episode and no submission | Agent-failure zero |
| Successful scripted exit without submission | Agent-failure zero |
| Explicit synthetic allowlisted infrastructure cause and no submission | Technical missing outcome, eligible only for an explicit bounded retry |
| Unknown error, metadata, provenance, or binding | Unresolved missing outcome |
| Different reported model | Ledger retains diagnostics, withholds usable reward, and halts new work |

Those machine-error mappings apply to the controlled helper. These controls do not validate a real provider error taxonomy, usage fields, authentication, or reported model. The separate provider path validates metadata and retains unknown errors as unresolved; its actual model compatibility remains untested.

The evidence bundle is published before committing the ledger outcome. It hashes the connection, process report, sealed receipt, stdout, and stderr, and binds the plan/attempt/ledger. Ledger completion remains immutable and idempotent. A technical retry gets a distinct attempt and fresh episode while preserving the earlier records.

## Resume without a second process

If the runner stops after publishing a sealed bundle but before acknowledging the ledger finish:

```sh
.venv/bin/python -m benchtop.cli study resume-engineering \
  artifacts/session-engineering-ledger ATTEMPT_ID
```

Resume reads and verifies the original private files, their checksums, bindings, classification, and stopped-process evidence. It starts no server or subprocess and does not use current mutable lab data to regrade the answer. It records the original finish, or a separate definite resolution if an unresolved finish already exists. A missing, changed, incomplete, or unverified bundle cannot silently trigger another run.

If the server or parent dies before a durable sealed receipt exists, this recovery path cannot prove the old submission status. The journal remains conservative; a future primary adapter needs its own live-server/process reconciliation. These local checksums validate integrity and consistency, not a hostile operator's claims or provider authenticity.

## Costs and scope

The receipt records actual cached request counts and charged experimental budget, with zero newly executed lab jobs. Scripted process wall time is recorded separately as control-process timing. It is not relabeled as LLM inference time. Unreported model tokens, model wall time, and environment CPU time stay unknown. Synthetic process starts and ledger intents are not billed model sessions.

Controls cover real MCP calls, raw visible replies, receipt privacy, wrong-answer retention, timeout/process cleanup, post-submit errors, post-submit timeout, no-answer/unknown/quota cases, fresh-episode retry, model change, source/budget/receipt checks, process uncertainty, altered evidence, primary-mode refusal, authenticated sealing, queued mutation ordering, CLI redaction, and resume after an interrupted ledger acknowledgement. Full provider execution and the study's scientific conclusions remain untested here.
