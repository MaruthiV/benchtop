# Qualia in the shared study journal

The primary Qualia route uses the original canonical journal and its two integration slots. **Tests use synthetic authorizations and surrogate MCP clients. No actual Qualia UI conversation, provider compatibility pass or scientific result is established.** Generic MCP preflight and accounting inspection never satisfy the observed-client gate.

Finish remaining release implementation and public documentation before real model sessions. Preparations bind source/runtime; spent intents cannot be refunded after edits. Keep the existing server/tunnel unchanged until an actual run is deliberately prepared.

## Authorize without invoking a model

After the original smoke configuration is authorized:

```sh
.venv/bin/python -m benchtop.cli study authorize-qualia
.venv/bin/python -m benchtop.cli study inspect-qualia
```

Authorization registers two connectivity slots in the same journal as smoke, development, pilots and confirmation. It creates no new protocol lock or session allowance. The slots use the first two original smoke references, claim questions, frozen prices/ordering and a 25% experimental allowance. They remain excluded from smoke success, budget selection, fresh calibration and primary comparisons.

The shared study model/prompt identity remains frozen. A separate client-prompt binding records the copied UI instructions and public episode rendering. Qualia's system prompt and unsupported decoding/tool controls are not assumed identical to the CLI evaluation configuration. Client observations remain operator reports.

Existing legacy integration starts still prevent primary initialization until reconciled. Once primary authorization exists, the old `integration prepare/begin` path cannot create a separate allowance. Isolated engineering fixtures remain separate from actual primary accounting.

## Prepare an inactive connection

Use one slot ID printed by inspection and the authenticated loopback administrative server:

```sh
.venv/bin/python -m benchtop.cli study prepare-qualia SLOT_ID \
  --base http://127.0.0.1:8787
```

The server must run cached episodes. Preparation fixes the task, budget, call limit and visible ordering from the authorized plan; arbitrary overrides are refused. It performs MCP initialization, discovery and description, and checks that administrative/unscoped agent routes return 404. It writes a mode-0600 connection file under private canonical `qualia/connections/` state, containing the capability URL and exact prompt. The CLI prints its path without the URL or administrative credential.

Description/discovery can run while connecting the client. Experiment and submission tools remain inactive, including through administrative routes, until a canonical start is reserved and the server activates that exact episode. Multiple inactive preparations consume no model starts; only one prepared episode can win a given slot's reservation. A server restart invalidates its episode identity. Before any reservation, a new inactive preparation can use the same unused slot.

Configure the Qualia client with the private agent URL. Cloud connections require a public HTTPS origin; desktop connections may use loopback. The existing [connectivity guide](QUALIA.md) explains the agent/admin separation and tunnel lifetime. Do not send a model prompt during preparation. Keep capabilities and private grades out of recordings and commits.

## Reserve before sending one conversation

```sh
.venv/bin/python -m benchtop.cli study begin-qualia EPISODE_ID \
  --client qualia-desktop \
  --selected-model 'ACTUAL_UI_LABEL' \
  --reported-model not_reported \
  --restrictions 'ACTUAL_SUPPORTED_CONTROLS_OR_EXPLICIT_UNAVAILABILITY' \
  --confirm-no-new-spend
```

Use `qualia-cloud` for the cloud client. The billing flag attests that this Qualia invocation uses included/free usage without new billing; the original Claude subscription setting does not establish that fact. The helper cannot authenticate Qualia billing. Without the attestation it refuses the reservation.

Record the selected UI label separately from an exact reported model ID. Leave the reported ID unavailable if the client does not expose it. A known ID differing from the frozen model is refused before invocation. A later reported model change halts new work in the shared journal. Unsupported UI restrictions remain explicit, rather than being treated as enforced isolation.

`begin-qualia` records the intent before enabling tools, then activates the matching episode and server instance. It starts no model itself. Send the saved prompt in exactly one Qualia conversation. Every additional conversation needs a different unused slot. The server checks the active reservation before experiments/submission and enforces the budget, call limit and episode lifetime.

UI intents share the two-active-session ceiling with evaluation and generation. Each consumes one integration slot and the 300-start total, including failed or abandoned reservations. Integration does not consume the 272 initial primary-study slots. There are two initial UI slots and no authorized UI retry path; the scientific technical-retry reserve cannot manufacture more UI attempts.

The server gates its own tools. It cannot count conversations or provider inference launched outside this workflow, authenticate an operator's model observations, or stop the Qualia backend. The operator must record every intended conversation and stop it when finished. An expired tool lease does not silently release an active intent.

## Capture only after the conversation stopped

```sh
.venv/bin/python -m benchtop.cli study collect-qualia EPISODE_ID \
  --client-stopped --client-observed \
  --reported-model not_reported \
  --output artifacts/qualia-primary-connectivity.json
```

`--client-stopped` is an explicit operator attestation, not process inspection. Without it the helper refuses to release the reservation. `--client-observed` records that the activity was actually seen in Qualia; omit it when that observation is unavailable.

Capture seals the episode and revokes its capability. Evidence binds the original start, prepared connection, server instance, activation boundary, task/grader/prices and private server receipt. A connectivity pass needs successful calls to all three tools after activation, an intact trace, every experiment and final submission through MCP, a receipt-only submission response and the factual UI observation. Preparation, administrative calls, truncated traces and unobserved clients remain incomplete.

The private grade and normalized outcome are separate from connectivity. An unavailable model identity leaves the journal reward unavailable even when the operator observed successful UI transport. These integration outcomes never enter scientific comparisons. Known stopped outcomes retain reported experimental costs; unreported model tokens and wall time remain unknown. Export strips capability URLs and private server details.

## Recover without another invocation

If activation acknowledgement was lost after reservation:

```sh
.venv/bin/python -m benchtop.cli study activate-qualia EPISODE_ID
```

This acknowledges activation of the same intent. It is not permission to resend a conversation. If a private capture exists but completion acknowledgement was lost:

```sh
.venv/bin/python -m benchtop.cli study resume-qualia EPISODE_ID
```

Resume verifies retained evidence and completes the same journal entry without server, network or model calls. Saved outcomes can be acknowledged after source drift while new starts remain blocked. Changed connections, authority or captured grading evidence fail verification.

When server evidence is unavailable, first stop the actual client, then preserve an unavailable result explicitly:

```sh
.venv/bin/python -m benchtop.cli study abandon-qualia EPISODE_ID \
  --client-stopped --reason 'WHAT_ACTUALLY_HAPPENED'
```

Abandonment does not claim a UI pass, infer an answer or known cost, refund the slot, or authorize a retry. It releases shared concurrency only on the stated stopped-client attestation; later experiment/submission requests fail the reservation check. Preserve all connection, intent and capture files.

## Remaining release gates

An actual observed UI pass still needs to happen. Remaining release preparation, recorded scientific reviews and bounded study runs, the public report/demo/release and factual email review also remain. Surrogate tests and preflight are engineering evidence only.
