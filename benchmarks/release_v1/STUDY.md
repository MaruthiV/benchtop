# Immutable study preparation

`benchtop study` prepares and inspects bound smoke/development plans. It freezes rollout identities before a future driver can execute them. It does not start a model, reserve session quota, or write any of the protocol's four study locks. Every plan has `execution_ready = false`.

The primary provider adapter, primary session-ledger integration, successful-smoke verification, candidate/cohort/confirmation planners, and actual phase-lock publication remain to be implemented. The [engineering session ledger](SESSIONS.md) and [scripted MCP runner](RUNNER.md) now provide shared counters, resumable records, process lifecycle checks, and sealed episode evidence without enabling primary execution. Preparation reports the missing prerequisites explicitly.

## Engineering plans

Use one or two frozen historical references or controlled two-part fixtures for bounded planning checks:

```sh
.venv/bin/python -m benchtop.cli study plan \
  --mode engineering --phase smoke \
  --env envs/2610-36935-gaussian-process-bandits-for \
  --model-config benchmarks/release_v1/study/engineering_model.json \
  --output-dir artifacts/study-engineering-smoke

.venv/bin/python -m benchtop.cli study inspect \
  artifacts/study-engineering-smoke
```

The [engineering model file](study/engineering_model.json) contains explicitly synthetic IDs and unavailable-setting explanations. No real CLI or model was invoked to produce it. It supplies a schema control, not a resolved study model. Engineering plans bypass admission selection and are excluded from study evidence; their mode is part of the binding hash.

A smoke plan has one slot per budget per source, at most six slots. All three budgets are represented. Development engineering plans have four distinct replicate slots per budget, at most 24. These are planned identities, not invoked sessions. Repeated seeds or identical visible ordering never merge independent replicate slots.

The source loader rejects fresh paper metadata and explicit fresh seed partitions. Historical snapshots are verified and reconstructed through the corrected read-only task view; no original task, price, observation, or manifest is upgraded in place.

## Readiness diagnostics

```sh
.venv/bin/python -m benchtop.cli study gate \
  --phase development \
  --model-config PATH_TO_RESOLVED_MODEL_JSON \
  --foundation artifacts/foundation_check.json \
  --reviews-dir artifacts/reference-human-reviews
```

Readiness checks the existing six-record [development admission rule](REPLAY.md), current source-bound foundation evidence, and explicit model configuration. Foundation evidence must cover every current Python implementation file, dependency files, and protocol; all recorded input hashes and the verifier's runtime must still match. Unrelated, partial, malformed, or stale files cannot satisfy it. Saved measurements and reviewer/model identities are operator-reported; matching hashes do not authenticate those assertions.

`planning_status = ready` means a preparation can be written. It does not remove execution blockers. Development execution still needs verified successful smoke outcomes and the development lock. Fresh-pilot and confirmation gate requests always remain blocked because their planners and phase-lock verifiers are not implemented. The gate includes the specification's actual lock requirements for inspection.

`--mode study_preparation` derives exactly four development labs from the reviewed rule and refuses explicit `--env` membership. It requires readiness to pass. Its development plan has all 48 slots. Its smoke plan uses the first two selected labs at all three budgets, for six slots. The four selected development IDs remain bound in either plan. A successful preparation still produces no study lock or execution authorization.

## Bound configuration

Model files require exactly `backend`, `requested_model_id`, `reported_model_id`, `cli_version`, and `decoding`. Backend is `claude_cli`. Requested/reported resolved IDs must agree; empty IDs and bare family aliases such as `haiku` are rejected. This validates declarations, not provider identity. The future runner must verify the actual reported model for every invocation and stop on a change.

`decoding` records `temperature` and `model_seed`. Each is either `{"status":"fixed","value":...}` or `{"status":"unavailable","reason":"..."}`. Fixed values must be finite and nonnegative, with an integer model seed. Unsupported controls must remain explicitly unavailable. This planner neither discovers CLI capabilities nor applies settings; the driver must verify supported flags rather than assuming these declarations took effect.

The binding core contains:

- Canonical protocol hash and exact specification checksum.
- All Python source checksums, dependency files, and recorded analysis runtime.
- Exact shared system text, first-message renderer source, and description-renderer checksum.
- Declared model/CLI/decoding configuration.
- Source paths and lab/space/manifest/result/task/cache hashes, historical index checksum where relevant, corrected execution/task versions, and qualification charging identity.
- Frozen visible ordering and, for study preparation, source-bound admission/foundation readiness and review file hashes.

Plans and receipts are evaluator artifacts. They must not be supplied to the investigating agent. No administrative token, capability URL, extra MCP configuration, or arbitrary CLI flags are accepted in the model-file schema.

## Identity derivation

The core is hashed before adding groups or slots, avoiding a circular binding hash. Group descriptors use exactly the specification's 13 fields: protocol/binding hashes, phase/arm, lab/task/grader/model/prompt/charging hashes, budget, visible order, and paper visibility. The group ID hashes this descriptor.

Slot IDs hash `{group_id, replicate_index}`. Each slot predeclares separate initial and retry attempt IDs, hashing `{slot_id, attempt_index}`. These retry identities do not authorize a retry or consume the reserve. Identical source content that would collide across groups is rejected.

Visible ordering uses the existing canonical lab.py/space.json content hash and protocol prefix to choose a lexicographic permutation. Every budget and replicate within a lab shares that order. Phase-specific binding/descriptor hashes keep smoke and development slots distinct, so smoke outcomes cannot silently become development pilot outcomes.

## Publication and revalidation

Publication reserves a new directory exclusively, then writes `plan.json` and `receipt.json` as atomic exclusive files. The receipt records UTC publication time and both hashes, with `study_lock = false` and zero model starts. A second publisher cannot overwrite the directory. If interrupted between files, the partial directory remains visible and cannot be loaded as a complete preparation; preserve it and use a new destination. This is ordinary process-interruption handling, not a distributed or authenticated journal.

Repository destinations must be under ignored `artifacts/`; external destinations are allowed. The planner never writes `benchmarks/release_v1/bindings.json` or a result file. Inspection checks both checksums, receipt/plan agreement, and reconstructs the entire expected plan from current inputs. Changed code, runtime, model declaration, source, reviews, foundation proof, groups, slots, or execution flag cannot silently resume the old preparation. Retain old artifacts and create a new version.

## Validation and remaining work

Controls cover exact descriptor fields and canonical IDs, independent replicates, bounded smoke plans, frozen prompts/ordering, historical source preservation, malformed model/decoding metadata, simulated ready and real draft-blocked gates, stale/incomplete foundation evidence, future-phase refusal, code/runtime/source/review changes, tampered or promoted plans/receipts, fresh-source refusal, interrupted publication, concurrent ownership, exclusive output, and copied-project CLI use.

Positive admission/foundation tests use temporary simulated records only. They do not certify any actual reference or model. Engineering accounting and scripted process/MCP reconciliation are implemented in [SESSIONS.md](SESSIONS.md) and [RUNNER.md](RUNNER.md). Next connect the real provider adapter and successful-smoke evidence under verified primary phase locks. The fresh study, independent confirmation grading, primary analysis, Qualia demonstration, and email readiness remain separate work.
