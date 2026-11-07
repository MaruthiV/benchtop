# Bounded fresh generation

The fresh driver consumes the verified [candidate lock](CANDIDATES.md). It follows archived candidate order, logs generation calls separately from the evaluation episode cap, and shares the two-active-session limit with evaluation. It uses the locked models/prompts, empty built-in tools, no MCP servers, no memory/settings discovery, and fresh CLI sessions. **Tests use simulated provider/paper responses and sandboxed synthetic labs. No actual paper generation or model session has been observed.**

## Generate one candidate

Finish and freeze the remaining study code and public documentation before actual smoke, development, and generation. Bound-input changes invalidate earlier preparations and cannot refund spent starts. The real development choice, baseline review and candidate lock must exist before these commands can invoke a model.

```sh
.venv/bin/python -m benchtop.cli study generate-next
.venv/bin/python -m benchtop.cli study generation-status
```

`generate-next` processes only the next unresolved candidate in the locked list. It may invoke one triage, one synthesis and at most two code repairs, then pauses for human review. Triage rejection, malformed output and exhausted mechanical repairs remain recorded candidate outcomes. It never reranks or substitutes papers. After eight admitted qualified labs, the next call marks the remaining candidates `not_processed_after_target` without invoking models; with fewer admissions it processes at most the locked 16.

Paper content uses the exact archived version's HTML URL. Redirects, server errors and mismatched versions pause the same candidate. HTTP 404 alone permits the frozen abstract-only fallback. Raw retrieval responses and failed attempts remain private. A successful excerpt is checked against archived bytes on reuse.

Each call has an immutable intent, exact prompt/model/stage, raw stdout/stderr, CLI configuration, stopped-process report and completion evidence under the canonical `generation/` directory. Four stages per paper bound the maximum at 64 generation intents. They do not consume the evaluation's 272+28 starts. Recorded intents are conservative accounting rather than authenticated inference or billing. Usage stays in raw CLI reports; unavailable counters remain unknown.

Changed reported model/tool contracts halt further work. Unknown provider errors remain unresolved and never trigger a silent retry. A pending call occupies a shared concurrency slot until retained stopped-process evidence is reconciled. Generation is currently serial within its driver, which satisfies the two-session ceiling across model activity.

```sh
.venv/bin/python -m benchtop.cli study resume-generation CALL_ID
```

Resume verifies retained process evidence and reparses stdout without another model call. If a stopped process has no complete durable evidence, preserve it for investigation. Once a durable acknowledgement is recovered, `generate-next` reuses that result. It never kills a recovered PID or assumes that elapsed time proves a process stopped.

## Review before qualification

Generated code is checked statically and in the existing macOS sandbox using visible seeds 0 and 1. The driver creates a separate lab/space snapshot per synthesis/repair attempt and an unassigned review template at:

```text
.state/study/PROTOCOL_SHA256/generation/papers/ARXIV_VERSION/review_template.json
```

Copy that template to your own review file and fill all seven scientific checks, paper sections, miniature scope, an actual reviewer identity and timestamp, and the decision rationale. Its proposed margin comes from generation; the reviewer must choose and justify a finite positive `margin.value` using the metric's meaning before qualification. The source-bound subject pins the draft code, space, version and mechanical proof. No automated check proves mechanism fidelity or fair baselines.

```sh
.venv/bin/python -m benchtop.cli study review-generated artifacts/my-fresh-review.json
```

An accepted review freezes the chosen margin, lab, space and 4/12/12 seed manifest in a new private lab directory. A rejected review needs an evidenced failed check and is retained without qualification. Unassigned, stale, incomplete or wrong-version reviews cannot authorize execution. The driver preserves actual review bytes and never converts an assistant draft into human signoff.

## Qualify the fixed lab

```sh
.venv/bin/python -m benchtop.cli study qualify-generated
```

Qualification requires the accepted pre-qualification review and permit. It runs the visible and qualification partitions separately with two workers and a five-second per-job timeout. Acceptance requires complete valid cached rows, passing mechanical checks, at most 1,200 reported sweep CPU seconds, and a settled qualification claim. Confirmation seeds 16–27 remain unevaluated. The qualification screen is separate from the human pre-data review; it does not invent a new reviewer decision or timestamp.

Results are `admitted`, `qualification_ambiguous`, or `generation_failed`. Ambiguity and failures remain in candidate order and cannot trigger code repairs based on hidden qualification results. The next generation call moves to the next locked candidate. Original public labs, caches, tasks and catalog are untouched.

A qualification start without a durable finish blocks automatic re-execution. Preserve partial records for explicit reconciliation. If the complete qualification proof was saved before acknowledgement failed, rerunning `qualify-generated` verifies and finalizes it without more CPU work. Outcome snapshots bind lab data, reviews and generator transcripts. Later edits or a cohort-lock marker block further admission mutations.

## Freeze membership

Once the ordered funnel is complete, use the [cohort lock](COHORT.md) to verify the retained transcripts, human reviews and cached qualification evidence and freeze the actual membership. It starts no models or sweeps. A published cohort closes further generation, review and qualification mutations; smaller or empty cohorts remain valid outcomes.

## Email readiness

Generation-to-qualification and cohort locking are implemented. Still required are fresh pilot/confirmation execution and analysis, shared Qualia invocation routing, actual scientific reviews and bounded study runs, the reproducible public report, and a working demonstration. No actual selected cohort, fresh performance result or Qualia pass is claimed by these tests.
