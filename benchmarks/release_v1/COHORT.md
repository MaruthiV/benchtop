# Qualification-only cohort lock

The cohort lock freezes the complete ordered generation funnel and its first eight admitted labs, or the smaller actual cohort. It reads the [candidate lock](CANDIDATES.md), retained generation transcripts, pre-qualification human decisions, and cached qualification results. **Tests use simulated provider/paper responses, synthetic labs and temporary journals. No actual cohort or fresh model result exists.**

Finish the remaining study code and public documentation before actual model sessions. Earlier source-bound preparations become stale when these inputs change; spent starts cannot be refunded.

## Close the funnel

Complete [bounded generation and qualification](GENERATION.md) in the archived candidate order. Every candidate must have a terminal disposition. Human review, qualification awaiting reconciliation, active model calls, unknown generation errors, or model/tool halts prevent publication. After eight admissions, run `generate-next` once to record the remaining candidates as `not_processed_after_target` without invoking models. With fewer admissions, finish the locked list without extending it.

```sh
.venv/bin/python -m benchtop.cli study lock-cohort
.venv/bin/python -m benchtop.cli study inspect-cohort
```

These commands accept no replacement list, label override, margin override or budget choice. They start zero model sessions, make no paper requests, and run no lab sweeps. Inspection recomputes the release screen from the existing qualification cache; it never evaluates confirmation seeds. The CLI shows cohort size and status while retaining detailed labels and review evidence in private state.

## What is verified and frozen

The lock verifies that triage, synthesis and bounded repairs used the exact frozen prompts and recorded model responses, that the valid draft has its original mechanical evidence, and that its named, dated scientific review cites the correct paper version. An accepted human review must precede qualification and fix a justified positive effect margin. The final lab may differ from that reviewed draft only in the approved margin. Rejected reviews and failed generation remain in the funnel without hidden qualification work.

Qualification must have its original review permit, complete paired visible and qualification cells, the 4/12/12 seed partition, matching cached screen and corrected grader, and the recorded CPU acceptance decision. Any row from confirmation seeds 16–27 or an undeclared seed prevents locking before the cached grader is called. Ambiguous claims and failed qualification remain excluded without replacement.

The private `cohort_lock.json` binds:

- Every candidate's disposition, reason and outcome hash, in candidate order.
- Each admitted lab's code, space, cache, manifest, results and task artifact hashes; qualification label and fixed margin; and review, permit and qualification evidence.
- Execution identity, claim-task version/hash, charging schedule, seed partitions and the protocol's deterministic pilot/confirmation visible orders.
- The candidate lock, selected global budget, baseline/estimator binding, model configuration, prompt, code and runtime, plus the complete generation evidence inventory.
- Reduced episode ceilings of 12 pilots and 16 confirmation slots per admitted lab. These are ceilings, not execution authorization.

Publication appends the lock's hash to `benchmarks/release_v1/bindings.json` under the same canonical study identity. Concurrent generation is excluded during publication; waiting generation rechecks closure. Once published, generation calls, reviews and qualification are closed. Repeating publication refuses to replace the original lock. Inspection verifies both the private record and its exact public binding. A missing canonical study fails without creating state.

Zero eligible labs produce `no_gradable_labs` and zero planned fresh slots. Report no primary comparison in that case. Smaller cohorts stay smaller. Neither inconvenient model scores nor unstable confirmation labels can justify replacing a member.

## What remains

The cohort lock alone authorizes neither pilots nor confirmation. [Fresh pilot preparation and authorization](PILOTS.md) bind the exact cohort to the existing primary provider journal; [pilot closure/calibration](PILOT_CALIBRATION.md) freezes each member's choice or fallback. The [confirmation lock](CONFIRMATION.md) freezes both arms before [independent truth/stability collection](CONFIRMATION_DATA.md), whose private snapshot preserves qualification evidence frozen here. [Confirmation execution/analysis](CONFIRMATION_RESULTS.md) and [shared Qualia accounting](QUALIA_PRIMARY.md) are implemented. Actual UI observation, scientific reviews and study runs, reproducible reporting, the demonstration and public release still precede outreach.
