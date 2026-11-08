# Qualia connectivity check

These legacy commands apply before primary study authorization. Once primary accounting exists, use the [shared Qualia route](QUALIA_PRIMARY.md), which reserves slots in the canonical journal and gates tools until activation. No actual UI pass is implied by either preparation path.

The agent endpoint has passed generic MCP preflight through a temporary public HTTPS tunnel. A real Qualia conversation has not yet been observed. This is the remaining foundation gate; preflight alone does not satisfy it.

## Current client setup

Qualia's official documentation describes custom servers under a workspace's gear menu → Customize agent → MCP Servers → New custom server. Supply a name and the HTTP URL, create it, then refresh the tools if needed. Cloud needs a publicly reachable destination; local desktop connections can use loopback. These are documented steps, not a claim that this session controlled that UI. [Qualia MCP documentation](https://docs.quadrillion.ai/mcp-integrations).

Only the agent port is tunneled. Administration remains authenticated on loopback. For an ephemeral test, an installed `cloudflared` can supply the public origin:

```sh
cloudflared tunnel --url http://127.0.0.1:8788
```

In another terminal, advertise that exact origin:

```sh
.venv/bin/python -m benchtop.cli serve \
  --agent-base https://REPLACE_WITH_TUNNEL_HOST \
  --allowed-origin https://app.quadrillion.ai
```

Quick Tunnels need no account or domain, and end when their process stops. They do not support SSE; Benchtop's JSON Streamable HTTP path passed preflight, while actual Qualia transport compatibility remains to be observed. [Cloudflare Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).

## Prepare without starting a model

```sh
.venv/bin/python -m benchtop.cli integration prepare
```

The default is the preserved data-curation lab's `where:gap_deg` question, seed 777, at a 25% allowance. This is `historical_v0` integration evidence, excluded from the release study. The command creates a scoped episode and tests initialize, initialized notification, tool discovery, and description. It verifies that administrative and unscoped paths return 404 through the agent origin. It neither runs experiments nor submits nor reserves a model invocation.

The printed private connection-file path is under `.state/integration/episodes/`. Its file mode is 0600. Open it locally and use `mcp_url` in Qualia's custom-server URL field. Its `mcpServers` object is also available for compatible configuration import. Neither the admin credential nor private answers are in this file. The CLI does not print the capability URL. Keep the file out of commits, public recordings, and shared logs.

Refresh the Qualia server's tools. Confirm that exactly `describe_lab`, `run_experiments`, and `submit_answer` appear. Official tool permissions include Always, Require permission, and Never. Allow the three lab tools as appropriate and restrict unrelated tools where the client permits it. Record unavailable restrictions explicitly. [Qualia tool permissions](https://docs.quadrillion.ai/mcp-integrations#tool-permissions).

## Record one intended invocation

Once signed in, connected, and ready to send one prompt:

```sh
.venv/bin/python -m benchtop.cli integration begin EPISODE_ID \
  --client qualia-cloud \
  --requested-model 'LABEL_SELECTED_IN_QUALIA' \
  --reported-model not_reported \
  --restrictions 'DESCRIBE_ACTUAL_AVAILABLE_CONTROLS'
```

Use `qualia-desktop` for a local desktop connection. Record an exact reported model ID if the client exposes one; otherwise retain `not_reported`. The selected label and actual model identity are different facts.

`begin` registers the intended invocation before the UI prompt. It does not launch a model. The append-only, mode-0600 ledger uses a file lock, rejects duplicate episode starts, and permits at most the protocol's two integration registrations. Failed or abandoned registrations are retained; it does not automatically reclaim them. Each new conversation needs a newly prepared episode and registration. Operator discipline is still required: the server cannot count provider sessions started outside this workflow.

The private file contains the exact prompt and hash. Send that prompt once in Qualia. It requests the three lab tools, no parallel agents or unrelated tools, and a stop after the submission receipt. The client may not enforce all requested restrictions; record what was actually available. These are connectivity checks, not comparable scientific evaluation rollouts.

## Capture after the actual UI run

```sh
.venv/bin/python -m benchtop.cli integration collect EPISODE_ID \
  --client-observed \
  --output artifacts/qualia-connectivity.json
```

`--client-observed` records the operator's factual observation that the calls occurred in the actual Qualia UI. Omit it while that observation is unavailable. It is not authenticated by MCP.

The authenticated evaluator reads a separate MCP trace containing tool names, order, response keys, error flags, and whether submission returned only a receipt. It also records whether the first final submission came through MCP or administration, under the episode lock. The bounded trace omits URLs, credentials, arguments, and observations; the existing private episode log contains the experimental data. Preparation calls are excluded using the recorded start boundary. A pass requires successful calls to all three tools after that boundary, all experiment calls and the committed submission through MCP, a receipt-only response, a complete retained trace, matching episode versions, and the explicit client observation.

Admin-only runs, preparation alone, missing calls, truncated traces, or an unobserved client remain incomplete. The evidence distinguishes operator-reported client/model identity from server-observed tool activity. Its private grade is read separately after submission, and capability strings are redacted from the exported JSON. Output in the repository is restricted to `artifacts/`; original result files cannot be overwritten.

The helper's test suite uses controlled surrogate clients. Those tests do not count as actual Qualia runs. Likewise, the foundation reproduction check's synthetic fixture does not consume a model-session reserve.

## Lifetime and next gates

Keep the server and tunnel alive while connecting and running the client. Restarting the server invalidates episode capabilities. A new Quick Tunnel has a different origin. When finished, stop both processes and remove or disable the connector in Qualia. Any public demo must hide the raw capability or use an endpoint stopped before publication.

If the connection fails, record the actual status/error before changing configuration. 421 indicates a disallowed Host; 403 an Origin rejection; 401 an invalid capability. Use explicit host/origin entries and a fresh episode after restart. A successful generic MCP handshake remains distinct from a Qualia client test.

After a recorded Qualia pass, proceed to admission evidence and the later policy/study locks. This early historical connectivity run does not select development budgets, admit a lab, or satisfy the fresh-cohort protocol gates.
