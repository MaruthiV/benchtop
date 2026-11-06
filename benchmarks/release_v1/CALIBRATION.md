# Budget calibration selector

`benchtop calibrate INPUT --output NEW_FILE` evaluates saved normalized pilot outcomes. It starts no models, runs no experiments, and makes no study lock. `beta_mixed_group_v1` implements the rule in [spec.json](spec.json); the model driver, immutable phase bindings, and independent confirmation evaluation remain separate work.

## Rule

For `s` correct outcomes from `n` completed pilots, success probability has posterior `Beta(s + 1, n - s + 1)`. The selector reports its mean and equal-tailed 95% credible interval. This interval describes success probability under the assumed model, not uncertainty in the final study effect.

For a future group of eight, the objective is:

```text
1 - E[p^8] - E[(1-p)^8]
```

The two expectations use exact Beta rising-factor moments. Substituting the posterior mean into the powers gives a different answer. For example, the uniform prior's mixed-group probability is `7/9`, not `1 - 2*(1/2)^8`.

The candidate budgets are 6%, 12%, and 25%. Among objectives at least the maximum minus 0.05, choose the smallest budget. A `1e-14` absolute comparison tolerance handles floating-point error at that boundary. Costs are reported separately and do not change this objective or tie rule.

| Input phase | Selection |
|---|---|
| `development` | Require all 48 planned slots: four labs × three budgets × four pilots. Average the predictive objective equally over all four labs at each budget. Select the global control with the same near-best rule. If any group is incomplete, return `incomplete_development` with no chosen budget; never average only the available labs. |
| `fresh_pilot` | Require all 12 planned slots for one lab. If complete and at least one budget has observed both rewards, apply the near-best rule. If every budget is all correct or all wrong, use the supplied global control and mark `no_observed_mixed_budget`. If incomplete after collection closes, use that control and mark `incomplete_pilots`. |

An all-endpoint development dataset can still select a global control, with a diagnostic that no mixed budget was observed. It has no earlier global control to fall back to.

## Collection and missing outcomes

Inputs require an explicit boolean `closed`. While false, the result is `pending`, with no chosen budget or fallback, even when provisional pilot counts look complete. Set it true only when the upstream collector has finalized the planned slots under the study's retry/quota rules. The selector checks this declaration, but cannot verify that an external collector actually exhausted its allowances or respected the overall session cap.

Every planned slot remains in the input. Unattempted slots use `attempts: []`. Recorded technical or unresolved attempts have `reward: null`. They do not increase the completed count or become incorrect answers. Completed agent failures have reward zero and count in `n`, following the protocol.

The input retains at most two attempts per slot. Only an allowlisted technical failure can precede a retry. A completed answer, agent failure, or unresolved failure cannot be replaced by a later result. The last attempt supplies at most one usable outcome; both attempts contribute their reported starts and costs. These checks validate normalized history, not the authenticity of provider/server classifications or whether a submitted answer needs reconciliation.

## Input contract

The runnable [engineering input](calibration/engineering_pilots.json) is entirely synthetic. Its three budgets have 0/4, 2/4, and 4/4 correct outcomes. The expected diagnostic choice is 12%; this is not a result from a paper lab or a language model.

```sh
.venv/bin/python -m benchtop.cli calibrate \
  benchmarks/release_v1/calibration/engineering_pilots.json \
  --output artifacts/calibration-engineering.json
```

Use a new output filename for each preserved diagnostic. Publication is atomic and exclusive; existing output files are never overwritten. Repository outputs must be under ignored `artifacts/`; external paths are also allowed. This command deliberately does not publish a study result into `results/` or `benchmarks/`.

The version-1 input has these fields:

| Field | Meaning |
|---|---|
| `schema_version` | Integer `1` |
| `protocol_hash` | Canonical UTF-8 SHA-256 of the full frozen specification, sorted keys, compact separators, `ensure_ascii=false`, no NaN |
| `evidence_kind` | `synthetic_control` or `operator_supplied`; retained in output |
| `phase` | `development` or `fresh_pilot`; confirmation/smoke data are not accepted phases |
| `closed` | Explicit finalization declaration described above |
| `lab_ids` | Ordered distinct IDs: four for development, one for fresh pilots |
| `global_budget_fraction` | Required protocol candidate for fresh pilots; forbidden for development |
| `slots` | The entire planned lab/budget/replicate grid, including missing outcomes |

Each slot has `group_id`, `slot_id`, `lab_id`, `budget_fraction`, `replicate_index` (0–3), and `attempts`. Every lab/budget cell has its own shared group ID. `slot_id` is the canonical hash of `{group_id, replicate_index}`. Each attempt has `attempt_id`, `attempt_index` (0 or 1), `state`, and `reward`, with optional `failure_category` and `costs`. `attempt_id` hashes `{slot_id, attempt_index}`. IDs must match their relationships; duplicate slots and reused groups are rejected.

Normalized states are `completed`, `agent_failure`, `technical_failure`, and `unresolved`. Completed rewards are numeric 0 or 1, with no failure category. Agent failures need a protocol-allowlisted cause and zero reward. Technical failures need a protocol-allowlisted infrastructure cause and null reward. Unresolved attempts need an explicit cause and null reward. Unknown fields, fractional/boolean rewards, invalid costs, and incomplete grid descriptions fail visibly.

The group ID's relationship to model/prompt/lab/charging hashes is not authenticated here. Neither is the supplied global control checked against a development lock. The future study runner must supply and verify those bindings, admission, membership, visible ordering, phase, outcome provenance, finalization, and the session ledger before using these diagnostics for the primary study. Relabeling operator data as a phase does not establish valid study provenance.

## Output and cost accounting

The output retains every pilot slot and attempt ID, per-budget counts/states/posteriors, per-lab and aggregate objectives, chosen budget, explicit fallback, and upfront cost. Missing groups keep descriptive partial posteriors, but have no eligible aggregate objective. A selection checksum binds the full result; bindings include the input's canonical hash and exact source-file checksum, the protocol checksum/hash, selector source checksum, and analysis runtime. The report is deterministic for the same bound inputs and code.

Attempt costs can report requested runs, unique executed runs, charged experiment cost, actual environment CPU seconds, environment wall seconds, model wall seconds, input tokens, and output tokens. Counts must be nonnegative integers; times/charges must be finite nonnegative numbers. Omitted or null values remain unreported. Each field has a known partial sum, an unreported-attempt count, and a total that stays null while any invoked attempt lacks that field. A reported zero remains distinguishable from unknown. Unattempted slots add no invocation. Synthetic input counts describe its simulated records; the command itself starts zero sessions.

Four pilots provide limited evidence about model success probability. Mixed rewards can include guessing and agent failures. This selector does not establish improved experimental behavior or training value; the later independent confirmation comparison tests the chosen rule and records its costs.
