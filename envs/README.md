# Environment catalog

[catalog.json](catalog.json) fixes the default evaluation cohort and its recorded generation funnel. Evaluation and reporting read this file rather than ignored pipeline or triage logs. Its ordered `environments` list contains directory names relative to `envs/`; unrelated directories do not enter an evaluation automatically.

The current catalog preserves the historical prototype: 30 paper-derived labs with 78 questions under `historical_v0` grading. The hand-written label-smoothing example is available separately and is excluded from this cohort. Historical pipeline acceptance does not certify scientific fidelity or admission to the next release.

The membership comes from the [frozen historical baseline](../benchmarks/historical_v0/README.md). Funnel totals are copied from the preserved `results/summary.json`, with the original file checksum recorded as provenance. These totals describe that historical generation run; the report does not infer them from the number of directories or from later private logs.

Catalog schema version 1 requires a nonempty cohort ID, unique environment directory names, and nonnegative integer funnel counts for `screened`, `selected`, `built`, `lab_written`, and `accepted`. The accepted count must equal the cohort size. Each selected directory must contain `lab.py`, `space.json`, `manifest.json`, `tasks.json`, and `results.jsonl`; its manifest identity must match its catalog entry. Missing or inconsistent inputs raise an error instead of silently reducing the cohort.

The evaluation functions and report use the same membership. `all_envs(only_accepted=False)` is an explicit diagnostic mode that lists all directories containing tasks. It does not change the catalog or admit those directories to default evaluation.

Newly generated labs require an intentional cohort update after review. A reproducible experiment must retain its membership, grader version, and funnel evidence. The fresh release cohort remains pending the admission and lock gates in the [release protocol](../benchmarks/release_v1/README.md).
