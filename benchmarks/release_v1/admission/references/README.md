# Six reference cases: paper-to-code review drafts

These cases cover the six engineering roles in [the protocol](../../spec.json). They are a deliberate development sample from the historical catalog, selected using source inspection and corrected cached diagnostics before any new model scores. Existing historical model results are not a selection criterion. This is an assistant proposal, with **zero human admissions** and no development lock.

[selection.json](selection.json) records membership, reasons, coverage, paper versions, source capture hashes, review hashes, and reproducible diagnostics. Each [draft](drafts/) binds the exact implementation, space, qualification artifact, and charging schedule through the admission review schema. The primary method sections and implementation were inspected; the drafts do not claim proof verification or new runtime measurements. Original labs and scores are preserved.

## Coverage and cached evidence

All numbers below use the twelve historical held-out seeds, 4–15, with `seed_block_t_v1`. Positive effects favor the designated paper arm. These are retrospective miniature results, not replications of paper results or fresh confirmation.

| Reference | Coverage role | Cached claim effect and label | Max/min charged cost | Seed-block SE / context-independent SE |
|---|---|---|---:|---:|
| [GP bandits](drafts/2610-36935-gaussian-process-bandits-for.json) | Average effect; unequal costs; shared seeds | +3.9743 regret units, paper better | 15.1 | 2.190 |
| [Monoculture](drafts/2610-36941-algorithmic-monoculture-in-machine.json) | Average effect; strong shared-seed dependence | +0.07587 social-loss units, paper better | 4.2 | 2.399 |
| [Data curation](drafts/2610-36983-advancement-or-acceleration-how.json) | Near-null average; interaction | −0.00028 accuracy, no meaningful difference | 2.4 | 1.110 |
| [CaLLM](drafts/2610-36994-callm-continual-adaptation-of.json) | Known fairness concern; unequal costs | +0.17612 macro accuracy, paper better | 43.6 | 1.587 |
| [Muon](drafts/2610-37225-sharpnessaware-muon-matrixaware-optimization.json) | Optimizer comparison; shared seeds; unequal costs | −0.01753 accuracy, baseline better | 2.4 | 1.655 |
| [Para-Gaussian](drafts/2610-37388-paragaussian-approximations-of-random.json) | Scientific-scope concern; unequal costs | −9.93722 signed bin discrepancy, baseline better | 42.2 | 1.255 |

The SE ratio compares the correct whole-seed uncertainty with a diagnostic that assumes independent contexts. It is not a correlation coefficient. Prices are frozen charges from historical timing data, not newly measured wall-clock ratios. Effect signs do not determine scientific admission.

Data curation illustrates why an average is insufficient: its marginal accuracy effects at teacher angles 0°, 30°, and 60° are −0.09394, −0.00352, and +0.09663. The pooled 95% interval is [−0.00962, +0.00906], inside the declared ±0.01 band. These marginal effects are descriptive, not an independently confirmed interaction test. Near-null coverage does not mean each context is null.

## Findings that require a decision

**GP bandits — unresolved fidelity and tuning scope.** The [source setup and algorithms](https://arxiv.org/html/2610.36935v1#S4) and the [implementation](../../../../envs/2610-36935-gaussian-process-bandits-for/lab.py) differ in landscape distributions and exploration/tuning policy. The simulator shares exogenous randomness, but reactive rewards depend on the selected action. Review the miniature as its own declared simulator; do not describe it as an exact source reproduction. Cold versus cached tuning behavior also needs a runtime probe.

**Monoculture — modeled objective and risk-cap scope.** The [source model](https://arxiv.org/html/2610.36941v1#S3) maps to the [synthetic loss and portfolio constraints](../../../../envs/2610-36941-algorithmic-monoculture-in-machine/lab.py). The optimizer knows structural parameters and constrains estimated risk; `cap_ok` checks true risk separately. A social-loss improvement does not establish constraint satisfaction or empirical agricultural welfare. Inspect cap violations and optimizer fallback before approving a constrained-benefit claim.

**Data curation — extra target supervision and finite scope.** The [source selection setting](https://arxiv.org/html/2610.36983v1#S3.SS1) motivates an [implementation](../../../../envs/2610-36983-advancement-or-acceleration-how/lab.py) that fits a target probe using 40 labeled examples. The baseline does not use those labels. Declare that information advantage or add an appropriate control in a new version. Two finite pool sizes cannot establish an asymptotic advancement claim. L-BFGS convergence is unchecked, and selection can fall back to the full pool.

**CaLLM — recommend rejecting the current primary comparison.** The [source method](https://arxiv.org/html/2610.36994v1#S3) and [digits adaptation](../../../../envs/2610-36994-callm-continual-adaptation-of/lab.py) share a prototype-routing idea. However, the lab compares independently fitted task-specific logistic heads with a sequential global MLP. Routing, architecture, head scope, capacity, and training all change. At ten tasks, the routed heads become constant-class predictors. Test chunks are grouped using true task membership. These code findings undermine a routing-only performance interpretation despite the favorable cached effect. This is an assistant rejection recommendation, **not a recorded human rejection**. A repair must preserve the original and use new versioned artifacts.

**Muon — plausible restricted mechanism; runtime evidence pending.** The [source update order](https://arxiv.org/html/2610.37225v1#S3.SS2) is recognizable in the [momentum perturbation branch](../../../../envs/2610-37225-sharpnessaware-muon-matrixaware-optimization/lab.py). The draft passes static checks for a fixed-step miniature, with bias perturbations and approximate orthogonalization declared. Probe zero-radius identity, initial zero momentum, and repeated executions. The unfavorable cached result stays in the evidence; it is not a reason to discard this case.

**Para-Gaussian — signedness is legitimate; regime and metric need review.** The [paper defines signed total variation](https://arxiv.org/html/2610.37388v1#S1.SS3) and discusses the [depth-to-width regime](https://arxiv.org/html/2610.37388v1#S1.SS6.SSS2). The [miniature](../../../../envs/2610-37388-paragaussian-approximations-of-random/lab.py) includes narrow/deep stress cases; one context accounts for about 94.68% of absolute mean effects. A signed discrepancy above one is not automatically a bug. Check the input/layer correspondence, zero-variance floor, binning, and Monte Carlo error before interpreting the average as a test of a rate theorem. The clipped predictor remains a separate control.

Every draft leaves practical margin justification and live execution behavior unresolved. Static passes are scoped code findings, not completed human attestations. No model difficulty, fresh-seed reward stability, or semantic-guard detection rate has been measured here.

## Reproduce and review

From the repository root:

```sh
.venv/bin/python -m benchmarks.release_v1.admission.check_references
```

This checks six unique catalog members, protocol coverage, bound drafts, pinned paper identities, code citation ranges, and recomputed cached diagnostics. It reads public repository files only, with no `.state/`, paper download, lab execution, or model call. It does not validate the meaning of a citation or a scientific recommendation. Source HTML/text hashes document the retrieved v1 snapshots; paper contents are not bundled, and the offline command does not authenticate remote bytes. Read the linked versions when reviewing. The original prototype text cache could be truncated; these draft sources were fetched separately as complete version-pinned HTML.

Validation at this checkpoint: the command passed in a disposable public-file copy without private state or paper snapshots. Seven altered-evidence controls were detected: a changed reviewer kind, stale subject, invalid code lines, unpinned citation, changed cached diagnostic, missing coverage role, and duplicate case. The existing 20 admission tests and Ruff passed. All 241 archived files verify, and all 200 original environment/result/figure files still match the archive. These checks establish package integrity for the covered cases; the planned semantic lab-defect fixtures are separate work.

For an owner review, copy a draft into a separate location so the assistant proposal remains intact:

```sh
mkdir -p artifacts/reference-human-reviews
cp benchmarks/release_v1/admission/references/drafts/2610-36994-callm-continual-adaptation-of.json \
  artifacts/reference-human-reviews/2610-36994-callm-continual-adaptation-of.json
```

Read the cited paper sections and code, resolve or retain each concern, and record your own decision and evidence. Set `reviewer.kind` to `human`, your own name and actual UTC review time **only after performing the review**. Reconsider every assistant status. Keep unresolved checks unresolved when evidence is missing. A rejection needs an evidenced failed check; acceptance needs all seven checks to pass, including runtime evidence and a meaningful margin. Remove or update `assistant_assessment` to distinguish your rationale from the original suggestion.

Then inspect the separate record:

```sh
.venv/bin/python -m benchtop.cli admission inspect \
  envs/2610-36994-callm-continual-adaptation-of \
  --review artifacts/reference-human-reviews/2610-36994-callm-continual-adaptation-of.json \
  --output artifacts/callm-human-admission.json
```

The human-review directory is ignored scratch space until records are deliberately versioned for release. The checker validates structure and bindings; it does not authenticate a human identity. These assistant drafts always remain `needs_review` under the admission engine, including the CaLLM rejection recommendation.

After human review, the protocol chooses the first four accepted, qualified references in lexicographic ID order. None is selected now. If fewer than four qualify, record an amendment before budget-selection model runs. Do not relabel an assistant draft or silently substitute an easier lab to reach four.

The companion [controlled defect suite](../defects/README.md) now provides six named changes, ten valid controls, and bounded Muon runtime evidence. The assistant drafts above retain their original scope and timestamps; the new measurements can inform a separate owner review. Consolidating the evidence and recording actual human decisions remain pending. The actual Qualia UI smoke also remains independently pending.
