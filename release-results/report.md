# Benchtop confirmation analysis

Protocol: `ba515089163fa749acff31fc991dd74264bcb02668d87ef07890051109750ac3`. Analysis: `72184c018c82f9834eb7c8f63d0572722f94822508f3c12dd545cdd812c1b7b6`.

Status: analysed; reporting: complete.

Locked labs: 1; stable: 0; complete pairs: 0; unavailable comparisons: 1.

Primary difference in mixed-group frequency, calibrated minus global:

Estimate: unavailable. Whole-lab bootstrap interval: unavailable. Full-cohort worst-case bounds: [-1, 1].

| Lab | Stability | Selected budget | Global mixed | Calibrated mixed | Global observations | Calibrated observations | Fallback |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2611.08959v1 | ambiguous | 0.06 | unavailable | unavailable | 0/8 | 0/8 | no_observed_mixed_budget |

Classical comparisons use the existing global model arm and 200 policy seeds at its frozen budget.

| Classical policy | Complete labs | Model minus classical accuracy | Whole-lab interval |
| --- | --- | --- | --- |
| paired_uniform | 0 | unavailable | unavailable |
| question_coverage | 0 | unavailable | unavailable |
| coverage_plus_cost | 0 | unavailable | unavailable |
| paper_prior_only | 0 | unavailable | unavailable |
| development_majority_label | 0 | unavailable | unavailable |
| random_experiment_design | 0 | unavailable | unavailable |

| Reported cost | Known sum | Unreported attempts | Total |
| --- | --- | --- | --- |
| pilot_costs/charged_experiment_cost | 0.1366 | 0 | 0.1366 |
| pilot_costs/model_wall_seconds | 204.4 | 0 | 204.4 |
| pilot_costs/input_tokens | 116 | 0 | 116 |
| pilot_costs/output_tokens | 39944 | 0 | 39944 |
| confirmation_costs/charged_experiment_cost | 0 | 0 | 0 |
| confirmation_costs/model_wall_seconds | 0 | 0 | 0 |
| confirmation_costs/input_tokens | 0 | 0 | 0 |
| confirmation_costs/output_tokens | 0 | 0 | 0 |

Earlier excluded protocol: 1 consumed compatibility start(s), retained outside primary outcomes. The global cap remains 300 starts across versions.

Earlier protocol charged_experiment_cost: known 0.03743; total 0.03743; unreported 0.
Earlier protocol model_wall_seconds: known 9.272; total 9.272; unreported 0.
Earlier protocol input_tokens: known 8; total 8; unreported 0.
Earlier protocol output_tokens: known 1713; total 1713; unreported 0.

Raw slot outcomes, retries, failure states, stability and generation/CPU costs are retained in the JSON export.

- Exploratory pilot with at most eight labs; no equivalence claim, universal generalization, or learning claim. Secondary results are descriptive.
- Complete-case effects are conditional on stable labels and complete groups. Full-cohort bounds include every unavailable arm.
- Bootstrap resamples whole paired labs, not slots or classical policy seeds. Secondary comparisons are descriptive.
- Reported costs are adapter measurements, not authenticated subscription billing. Unknown usage remains null.
