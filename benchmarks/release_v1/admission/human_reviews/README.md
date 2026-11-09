# Recorded reference decisions

Maruthi Vemula reviewed the source-linked packet and explicitly approved these six scoped decisions on November 9, 2026. The JSON files retain the actual approval acknowledgement time, original source subject, evidence and limitations. They were prepared with assistant evidence and approved by the named reviewer; identity and scientific judgments are operator-reported, not independently authenticated or cryptographically signed.

| Reference | Decision | Scope and unchanged margin |
|---|---|---|
| [GP bandits](2610-36935-gaussian-process-bandits-for.json) | Accepted | Independently tuned algorithms in the declared 52-round simulator; no production replication or equal-wall-clock comparison. Margin: 0.3 cumulative regret, about 0.00577 normalized reward per round. |
| [Algorithmic monoculture](2610-36941-algorithmic-monoculture-in-machine.json) | Accepted | Model-consistent optimization in the synthetic known-parameter world; no agricultural/welfare or true-risk-cap guarantee. Margin: 0.01 normalized objective units. |
| [Data curation](2610-36983-advancement-or-acceleration-how.json) | Accepted | Complete target-assisted pipeline versus source-only training, disclosing the extra target labels, variable subset and fallback. No equal-information selector effect or asymptotic result. Margin: one percentage point of accuracy. |
| [CaLLM](2610-36994-callm-continual-adaptation-of.json) | Rejected current comparison | Architecture, training, output scope and capacity change with routing. A routing-only effect is not identified. The original 0.02 margin is not endorsed as a remedy. |
| [Sharpness-aware Muon](2610-37225-sharpnessaware-muon-matrixaware-optimization.json) | Accepted | Declared 150-step digits-MLP comparison with approximate updates and bounded input-sharing/perturbation controls. No LLM reproduction or universal optimizer guarantee. Margin: one percentage point of accuracy. |
| [Para-Gaussian](2610-37388-paragaussian-approximations-of-random.json) | Needs review; not admitted | Signed-distance, reference/bin sensitivity and stress-regime dominance remain incompletely validated. Signed negative mass alone is not a defect. |

The accepted margins are exploratory resolution conventions for these miniatures, not externally validated practical thresholds or significance criteria. They retain the original values and are approved before new model budget selection. All original paper-inspired assumptions and simplifications remain in the individual records.

[Bounded runtime measurements](../runtime_review/README.md) add fresh-worker and optimizer diagnostics on existing visible seeds. The [original report](../REPORT.md) is a preserved snapshot generated without supplied human records; its zero-admission count is not the count after supplying these records. Original assistant drafts were not rewritten. Proposal and earlier owner-record hashes remain as provenance; their preparation/acknowledgement materials are retained privately.

## Check the gate from public files

```sh
uv run python -m benchtop.replay gate --reviews-dir benchmarks/release_v1/admission/human_reviews
```

The current protocol selects the first four accepted, corrected-qualified references in ID order: GP bandits, monoculture, data curation and Muon. A recorded unresolved Para-Gaussian review satisfies record completeness while remaining excluded from admission. These are historical development references; none is an admitted fresh-cohort member or a new model performance result.
