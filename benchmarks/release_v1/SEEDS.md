# Fresh seed partitions

Fresh lab manifests can declare three disjoint seed groups. The release protocol fixes their membership before observations are collected:

| Group | Seed IDs | Purpose |
|---|---|---|
| Visible | 0–3 | Experiments delivered through agent tools |
| Qualification | 4–15 | Admission eligibility, pilot labels, and frozen configuration prices |
| Confirmation | 16–27 | Later independent reward-stability check and final grading |

This checkpoint implements partition validation, qualification tasks, and staged visible/qualification sweeps. It does not implement the locked confirmation runner, select a fresh cohort, or authorize a model study. Synthetic controls establish engineering behavior, not scientific fidelity or a real paper result.

## Manifest contract

Keep `seeds = 28`, `visible_seeds = 4`, and a `seed_splits` object with exactly `visible`, `qualification`, and `confirmation`. Each value is a list of native integer IDs. Groups must be disjoint and cover every ID in `range(seeds)`. The visible list must equal the declared prefix `range(visible_seeds)`. Qualification and confirmation each require at least two seeds; the release protocol requires twelve each.

For a smaller synthetic engineering fixture, a valid manifest is:

```json
{
  "id": "synthetic-split-control",
  "seeds": 8,
  "visible_seeds": 2,
  "seed_splits": {
    "visible": [0, 1],
    "qualification": [2, 3, 4],
    "confirmation": [5, 6, 7]
  }
}
```

`Env.load` and `sweep` reject malformed partitions before accepting cached observations or writing sweep outputs. Existing cache identities already bind the entire partition definition. Changing membership or total seed count cannot reuse an existing execution cache, including in recorded replay mode. Do not extend seed groups in response to their outcomes.

## Staged sweeps

For an environment already prepared under the release's generation and lock rules:

```sh
.venv/bin/python -m benchtop.cli sweep PATH_TO_FRESH_ENV --workers 1 --split visible
.venv/bin/python -m benchtop.cli sweep PATH_TO_FRESH_ENV --workers 1 --split qualification
```

Without `--split`, a manifest with explicit groups sweeps visible plus qualification only. The manifest retains all 28 declared seeds, while confirmation observations remain absent. An ordinary sweep refuses `split="confirmation"`; the CLI offers only visible and qualification. The future confirmation runner must enforce the cohort and confirmation locks before collecting or inspecting those observations.

Sweeps resume missing or failed jobs in the selected groups. Complete cached jobs start no workers. A visible-only stage normally leaves an incomplete qualification artifact with no eligible tasks. Missing qualification pairs remain explicit, even if complete confirmation data happens to exist in an operator-supplied cache. Completing the qualification stage can make the task artifact eligible; neither stage adds confirmation seeds to fill a gap.

Fresh sweep summaries record `seed_stage`, `swept_seed_ids`, and `confirmation_evaluated = false`, alongside the existing whole-cache success/failure counts. The stage is an engineering record, not a cohort-lock attestation. These helpers do not enforce human admission or candidate membership themselves; that remains the study runner's responsibility.

## Labels, charging, and access

`Env.qualification_seeds` exposes the qualification list. The existing task builder reads `Env.heldout_seeds`, which is now a compatibility alias for this grading pool on fresh manifests. It continues to mean all nonvisible seeds on legacy two-part manifests. `Env.confirmation_seeds` exposes the separate final group to evaluator code and returns an empty list for legacy manifests.

Default `build_task_set(env)` therefore reconstructs fresh tasks using qualification only. Fresh `tasks.json` must be a versioned artifact bound to exactly that list, including every task's binding. A confirmation artifact or mixed qualification/confirmation artifact cannot be loaded as ordinary episode tasks. Explicit-seed diagnostic reconstruction remains an evaluator operation; it is not permission to inspect real confirmation outcomes before the required locks.

Fresh charging always uses qualification timings, including before task eligibility is known. Confirmation or visible timings cannot set configuration prices. Qualification must be complete before the later study freezes its charging schedule; the generic table's missing-timing fallback is not evidence of complete admission.

Agent tools retain the visible replicate cap and episode-local ordinals. They publish neither partition IDs nor held-out observations. The evaluator's private logs preserve actual visible seed lineage. Final confirmation grading and unavailable rewards for changed or ambiguous confirmation labels remain to be implemented in the locked evaluation path; a qualification reward must not be presented as a final study reward.

## Validation and scope

Controls cover invalid/overlapping/noninteger/missing groups, tiny held-out groups, exact cache partition bindings, staged CLI acquisition, execution-free resume, failed qualification retries, missing qualification pairs, private visible-only observations, and rejection of confirmation task artifacts. A counterfactual synthetic cache with reversed confirmation effects and inflated confirmation timings leaves qualification tasks, admission subjects, and charging unchanged.

Historical manifests without `seed_splits` keep their original sweep targets, task seeds, and pricing rules. No original environment, historical result, protocol, reference draft, or human review needs to be relabeled for this checkpoint. Public screening and bounded fixture reports are regenerated against the changed loader hashes, retaining the actual checks and provenance.
