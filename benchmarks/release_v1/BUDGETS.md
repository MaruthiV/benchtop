# Charging and episode operations

An episode has a frozen experiment allowance, separate from the physical work performed by the live backend. A cached observation and a newly executed observation have the same charge. Identical work shared across episodes executes once, while each episode pays for its own delivered observation.

## Public charging schedule

The first episode on a loaded environment freezes one positive, finite cost for every configuration. Subsequent episodes on that same loaded environment share the immutable schedule. Live timing updates cannot change its prices, full-grid cost, or remaining allowances.

For corrected tasks, each price is the median recorded runtime for that configuration on the task's qualification seeds. Visible runtimes and later confirmation runtimes do not contribute. An unmeasured configuration uses a one-second estimate. Historical environments retain median pricing over all their recorded seeds; all 1,059 configuration prices in the 30-lab historical catalog are unchanged by this checkpoint. Repeated rows for one seed use the latest successful timing.

`describe_lab` publishes `pricing.by_config`, keyed by canonical configuration JSON, with unrounded costs and a `schedule_id`. Agents and classical policies can use the same estimates. The ID hashes the costs and their seed source; the public pricing object does not contain held-out seed IDs or observations. Private episode summaries, new baseline outputs, and new rollout records include `charging_id`. Binding that schedule to versioned environment and execution inputs is the next checkpoint.

The full-grid cost is the sum of configuration prices multiplied by the number of visible seeds. An episode receives its specified fraction of that total. Budgets must be finite and between zero and one. Budget balances and per-result costs use the same unrounded values; a zero allowance cannot buy a very small positive-cost observation. Charges are estimates in compute-seconds, not a measurement of wall time consumed by a request. Their meaning stays the same for cached and live episodes.

## Serialized operations

Both administrative HTTP and agent MCP requests go through one `Hub` and one event loop. Each episode has its own lock. A run holds it while validating and reserving fresh replicates, preparing live work, and committing observations and charges. Submission, description, and private summary reads use the same lock.

- Overlapping runs cannot reserve the same fresh replicate or spend the same remaining allowance.
- A submission waits for a run that already holds the lock. A run admitted after final submission returns an error without preparing work.
- Simultaneous submissions commit the first valid final answer in lock order. Later submissions return receipts without changing the answer, confidence, score, or log.
- Separate episodes can run concurrently and share identical live executions without sharing their allowances.

The synchronous `Episode.run` path supports single-caller cached replay. Concurrent network callers must use `Hub`; direct `Episode` calls do not acquire an asynchronous lock.

## Failure, retry, and cancellation

After successful request validation, each run consumes one call slot before live work starts. Failed work is reported per replicate using a typed error, for example `{"code":"execution_failed","retryable":true}`. It spends no observation allowance and does not mark that replicate as delivered. A retry uses a new call slot and fills the earliest undelivered replicate, including a hole left by a partially successful batch. Successfully delivered replicates are charged once. A retryable error does not bypass the episode's call limit.

Malformed requests do not use a call slot. Finished episodes and episodes at their call limit do not prepare work. A budget-exhausted request can use a call slot, but prepares no unaffordable experiments.

Each live store owns a task for every missing configuration/seed pair. Waiters are shielded from cancelling that shared task. Cancelling a run while awaiting work releases its episode lock, retains its used call slot, and leaves its allowance and delivered-replicate state unchanged. Other episodes can still receive the result. A later retry can use the completed cache entry. Work with no remaining waiters may still finish and populate the cache; its physical cost is not charged as a delivered observation.

Live results must contain the primary metric, finite numeric metrics, and a finite nonnegative runtime. They are persisted before entering the observation table. Worker failure, invalid output, or failed persistence produces no delivered observation or charge. A failed pool startup is cleaned up so a later attempt can create a new pool. Store shutdown explicitly cancels owned work, waits for it, closes its workers, and rejects further requests. The two-port server invokes this cleanup during shutdown.

Cancellation before commit and losing a response after commit are different cases: once a run commits, its delivered replicates and charges remain committed even if the caller loses the response. Transport retries have no request-id deduplication in this checkpoint; they may request another fresh replicate. The private trace records the committed state for reconciliation.

## Evidence and limits

Run logs distinguish `attempted` cells from delivered `ran` cells, with `complete`, `partial`, `failed`, or `cancelled` status. New rollout records count both requested and delivered observations. Failed physical executions and coalescing are counted separately in live-store statistics. A cancelled run can have attempted cells and no delivered cells.

Regression tests cover frozen pricing and qualification timing sources, zero and invalid budgets, overlapping runs, run-versus-submit ordering across HTTP/MCP, simultaneous first submissions on both interfaces, call limits, partial failure and retry, shared-work cancellation, shutdown, failed startup, invalid worker results, and failed persistence. A real macOS sandbox test exercises 20 concurrent episodes sharing four live executions. These are deterministic engineering checks; no model episodes or fresh scientific evaluation were run.

Execution-cache input versioning, rejection of stale caches, unsupported-platform sandbox rejection, and real Qualia connectivity remain open foundation work. This checkpoint does not change the release protocol or admit any lab to the fresh cohort.
