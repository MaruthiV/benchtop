# Candidate metadata and rule lock

The candidate lock binds the development-selected global budget, classical policies and estimator, generation/admission rules, and an ordered versioned paper list before fresh generation. Tests use local XML and simulated review/provider boundaries. **No actual candidate lock, fresh generation, or model budget result exists.** A verified new lock reports `generation_ready: true` for the implemented [bounded generation driver](GENERATION.md). The [cohort lock](COHORT.md) freezes its reviewed qualification outcomes. [Fresh pilot preparation/authorization](PILOTS.md), [pilot closure/calibration](PILOT_CALIBRATION.md), [confirmation locking/planning](CONFIRMATION.md), and [independent truth/stability](CONFIRMATION_DATA.md) are implemented; [confirmation execution](CONFIRMATION_EXECUTION.md) adds independent private grading and ordered starts. Actual model study results remain pending.

## Archive the date selection

Create an ignored exclusion declaration with `schema_version: 1`, `complete_prior_history: true`, and `additional_ids: []`. Each additional exclusion is `{"id":"VERSIONLESS_ID","reason":"..."}`. Completeness is an operator attestation about earlier manual reviews and generation. Automatic history includes arXiv lab manifests, the private generation funnel including failures, and identifiable synthesis drafts. It cannot discover undocumented human reading.

```sh
.venv/bin/python -m benchtop.cli study archive-candidates \
  --exclusions artifacts/prior-paper-exclusions.json \
  --output-dir artifacts/fresh-metadata
```

This fetches metadata only. Starting from yesterday in UTC, it moves backward only after a complete date contains no unseen paper whose primary category is `cs.LG` or `stat.ML`. It archives exact query parameters, UTC times, HTTP status, raw Atom bytes, and hashes. Paging must agree with OpenSearch totals and offsets. The arXiv API defines the feed fields and versioned identifiers in its [official manual](https://info.arxiv.org/help/api/user-manual.html).

Deduplicate by versionless ID, sort numerically ascending, and retain the first 16 with their fetched versions. Conflicting versions block selection. Secondary-category matches do not establish primary-category membership. No relevance ranking or model score affects order.

A failed request stops the search on that date. Rerunning the same directory preserves prior files and retries. Inconsistent pagination records a restart marker and refetches the whole date. All attempted response files, including failures, are hashed into the completed archive. Completed archives are idempotent. Safety bounds of 31 dates or 50,000 entries per date stop retrieval for investigation instead of selecting an unverified date.

The archive must still cover the latest completed UTC date when locking. Preserve stale archives as evidence; they do not authorize a later selection. After locking, verification uses the original timestamp so the fixed list remains valid on subsequent days. Later fresh generation does not change the recorded pre-generation exclusion snapshot.

## Review the baselines

Complete the real [review-gated development replay](REPLAY.md): four labs, six policies, three budgets and 200 seeds, or 14,400 slots. The lock validates that exact current plan and raw attempt history. Unattempted/interrupted slots block freezing; policy-failure zeros and technical missing outcomes remain in the summary.

Copy the [unassigned review template](study/candidate_review_template.json) into `artifacts/`. Supply its replay directory reference and the canonical hash of `replay.summarize(plan, directory)`, excluding command-specific fields such as `attempts_invoked_this_run`. Each correctness/sensitivity check needs an accepted judgment, rationale and local source/hash evidence. Record an actual human name and UTC review time after the replay finishes. The template and assistant notes do not satisfy that review.

The estimator remains `paired_context_v1`, with equal context weights and its explicit neutral prior for unobserved contexts. Explain the development sensitivity evidence supporting it. Record why adaptive allocation is deferred; inclusion is refused because no adaptive release policy is implemented. The required policies, sanity controls, and majority label derived from development labels are frozen together.

## Publish and inspect

First close the complete [canonical development calibration](CALIBRATION.md#canonical-development-collection). An incomplete/open collection or generic operator-input diagnostic cannot supply the global budget. There is no supplied-budget override.

Create a generation-model JSON with exactly `triage`, `synthesis`, and `repair`, each containing a resolved [model configuration](STUDY.md#bound-configuration). Each must match the saved subscription CLI preflight and declare unsupported decoding controls unavailable. These are declarations; actual model availability remains untested until the driver runs.

```sh
.venv/bin/python -m benchtop.cli study lock-candidates \
  artifacts/fresh-metadata \
  --baseline-review artifacts/baseline-freeze.json \
  --generation-models artifacts/generation-models.json

.venv/bin/python -m benchtop.cli study inspect-candidates
```

The lock binds the closed development snapshot/budget, baseline review and evidence, policies/estimator/majority label, all implementation/runtime hashes, metadata archive, and exact prompts. Generation rules include one initial attempt, at most two repairs, two concurrent sessions, 4/12/12 seeds, admission requirements and pre-qualification effect-margin review. Legacy `build`/`synthesize` are not authorized by this record; their defaults differ from the fresh protocol.

Publication stores immutable private `candidate_lock.json`, a byte-preserved baseline review, and only the bound metadata files in the canonical study directory. It appends a compact public binding without replacing the development record. A second publication cannot overwrite the lock or choose another list. Partial publication remains visible for investigation.

Inspection revalidates the original development decision, copied raw metadata and failed requests, review evidence, replay, prompts, models and public binding. It runs no policy acquisition, independent regrading, live authentication or inference. Checksums establish local consistency rather than authenticated provider, arXiv or human assertions.

[Full-cohort analysis and matched classical replay](CONFIRMATION_RESULTS.md) are implemented. [Shared Qualia routing](QUALIA_PRIMARY.md) is implemented. Finish remaining release code before spending real smoke sessions. Later Python edits invalidate bound preparations; spent starts cannot be erased. Real reviews, development results, fresh admission, the model study and demonstration still precede outreach.
