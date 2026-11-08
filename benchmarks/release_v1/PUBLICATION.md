# Reproducible public package

`release build` creates a new directory from closed canonical confirmation evidence and a completed classical collection. It starts no models or policy trials. It verifies the journal, phase chain, source/runtime identity, foundation inputs and absence of active evaluation/generation. The destination must be new and outside canonical state. Failed builds retain an incomplete marker: inspect them, resolve the cause and use a new destination.

```sh
.venv/bin/python -m benchtop.cli release status
.venv/bin/python -m benchtop.cli release build --output-dir artifacts/public-release
.venv/bin/python -m benchtop.cli release reproduce artifacts/public-release
```

## Preserve evaluation inputs

The foundation hashes README, public docs, code, dependency lock and historical data. Keep the evaluation tree unchanged through real phases. The package copies those bytes, moves the original README to `evaluation/README.md` and writes a publication README at its root. `release-manifest.json` maps every original input to unchanged bytes and hashes all packaged files. The private inventory is checked before/after building and retained as a digest. Original evidence is never rebound to later sources.

Inspect the package before publishing. Raw `.state`, `artifacts`, account files, capabilities and provider transcripts remain private. Ignored operator notes/tests are unnecessary for reproduction. The package retains public code/dependencies/protocol, cached historical inputs, ordered candidates, membership decisions, environment versions, seed partitions, projected reviews, costs and missingness.

## New projections, original provenance

Public evidence is an explicitly new sanitized projection, with new hashes and original record hashes as provenance. Answers become allowed labels or unavailable, confidence finite numbers or unavailable, and traces tool names/response-field summaries. Raw text/connections are omitted. Failed lab rows retain failure with a generic error. Projected manifests omit private generation paths; numerical rows are resealed with original row hashes. Lab code, search spaces, task artifacts and recorded execution bindings remain unchanged.

Projections preserve scientific outputs, costs, failures, denominators and independent truth. They do not authenticate human/provider identities, preserve every private byte or authorize new execution. Review rationale/generated code must themselves be publishable; capability/credential/private-path detection refuses suspicious inputs. Automated checks supplement inspection of the final directory.

## Offline reproduction

Run from the package's own directory without the original workspace/journal:

```sh
uv sync --locked
uv run python -m benchtop.cli release reproduce .
```

Installing dependencies can need network access. Reproduction itself makes no network requests, starts no models and executes no lab code. It verifies exact file inventories, symlinks, incomplete builds, frozen inputs and source identity; rebuilds qualification/independent confirmation truth from cached numerical rows; checks disjoint seeds/frozen prices; reruns global/per-lab budget selection; regrades completed model answers; repeats completed cached classical acquisitions; and regenerates paired analysis/readable report.

Failed or missing trials remain failed or missing. Optional `--skip-classical-replay` skips completed acquisitions while retaining integrity, truth, budget, model grading and analysis checks; it is not full reproduction. Output reports checked selections, grades and policy trials, recorded-runtime match and bitwise analysis match. Numeric comparison permits relative/absolute tolerance `1e-12`; labels are exact. Reproducing cached numbers on a compatible different runtime does not authorize new lab execution.

## Results and email gates

`release-results/report.md` and `analysis.json` preserve complete-case effects, full-cohort bounds, interval availability, label instability, pilot/generation/classical overhead and unknown costs. PNG/SVG show all locked labs' mixed-group outcomes or explicit unavailability plus selected budgets. No admitted labs yields an unavailable comparison. Small, negative or missing results stay visible.

The builder always reports `email_ready: false`. Actual Qualia observation, a public demo, working public links and factual outreach review need separate evidence. Status inspects recorded gates; it cannot infer a human review, observed UI or permission to send. Follow [EXECUTION.md](EXECUTION.md).
