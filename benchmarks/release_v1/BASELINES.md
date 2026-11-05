# Classical policies and shared reconstruction

The first policy checkpoint implements a versioned paired-uniform control and one common visible-data estimator. It does not select development labs, run the planned 200-seed policy comparison, freeze candidate policies, or evaluate new model sessions. Human admission reviews remain pending in the [review packet](admission/REVIEW.md).

## Separate historical behavior

The original `prior`, `random`, and `uniform` registry, answer reconstruction, and saved results retain their historical behavior. The new `RELEASE_BASELINES` registry initially contains `paired_uniform`; `play_release` requires corrected task artifacts and a fresh episode. Existing evaluation commands do not silently switch policy cohorts or overwrite historical results with these new records.

The paired-uniform acquisition schedule matches the original contrast sweep: randomly permute target contexts once, request treatment then baseline at each context, and repeat that order over visible replicates. All requests pass through the existing budget and call accounting. A budget cutoff can deliver only one arm of the final pair; that observation remains charged and recorded, but contributes no paired effect until its matching arm exists. A reported experiment failure stops this policy; it does not count missing work as a delivered pair.

The policy core accepts only a public description, a run callback, and its random generator. It receives no environment object, hidden result table, qualification answer, or grading response. `play_release` submits and reads the trusted score only after acquisition and reconstruction end. Tests match original acquisition traces and charges at 0%, 6%, 12%, 25%, and 100% budgets and verify that changed hidden outcomes cannot change the new policy's choices.

`describe_lab` now provides structured `question_spec`, `metric.min_effect`, and `replicate_limit`. These encode the question, tolerance, and replicate cap already stated in the public prose. They expose no hidden labels, qualified-level subset, seed IDs, or seed ordering. Prices and call/budget balances use the same public values available to an agent.

## Shared estimator: `paired_context_v1`

[estimator.py](../../benchtop/agents/estimator.py) accepts a space, a public question specification, and observations from one episode. It supports claim, where, and two-level component questions. Best-configuration questions retain the historical path and are outside this release estimator's scope.

For each target context, match the two arms on the **same delivered replicate identity**. Average their signed differences within that context, then assign every declared context weight `1 / C`, where `C` is the full target context count. Repeating a cheap context does not increase its weight. Metric direction is applied before averaging; positive effects favor the designated treatment.

For a context with no complete pair, this initial version uses a **fixed zero-effect prior**, keeping the full denominator. The output explicitly reports missing coverage and a `no_pairs_fallback` when no pairs exist. This assumption can bias sparse-data predictions toward `no_meaningful_difference`. It is not evidence of equivalence, an unbiased missing-data correction, or a reason to call an environment easy. Review its development behavior before the candidate lock; any later revision needs a distinct frozen code binding and a disclosed rationale.

The submitted label is the existing point-threshold heuristic, not a qualification significance decision. Hidden corrected qualification still uses its own complete held-out data and seed-block t interval. Every policy in the main classical comparison must use the same estimator version and missing-context rule.

The estimator also records:

- Paired identities, effect estimates, and contribution for every target context.
- Observed versus total contexts, pair counts, unmatched observations, and ignored exact duplicates.
- A descriptive seed-block t interval only when all contexts have the same two or more paired identities. Unequal or missing identity sets leave this diagnostic unavailable. It is not a validated sequential interval under adaptive sampling.

Repeated identical observations do not gain extra weight. Conflicting values for one configuration/identity, missing identities, incomplete configurations, nonfinite metrics, and out-of-partition identities are rejected. Unmatched treatment and baseline samples are never paired by position or averaged as if they shared a seed.

Replicate ordinals are scoped to one episode: ordinal 1 can refer to a different actual seed in another episode. Do not pool raw ordinals across episodes. The evaluation record preserves episode ID, policy seed, episode seed, grader/execution/task/charging IDs, code hashes, public observation rows with replicate ordinals, and the evaluator's actual `(config, seed)` lineage.

## Read-only corrected historical replay

`benchtop.evaluate.corrected_replay` reloads and validates a frozen historical snapshot, reconstructs its corrected task artifact from historical held-out data, and creates an in-memory view. It uses a separate table for qualification-based prices and a new task version. It rejects fresh seed partitions and cannot authorize live work. Original files, historical task labels, and the caller's frozen charging schedule remain unchanged.

For example, an engineering replay can explicitly opt in:

```python
from pathlib import Path
from benchtop.agents.baselines import play_release
from benchtop.env import Env
from benchtop.episode import Episode
from benchtop.evaluate import corrected_replay

source = Env.load(Path("envs/2610-36935-gaussian-process-bandits-for"), replay=True)
view = corrected_replay(source)
record = play_release(Episode(view, "claim", seed=7, budget_frac=0.12), seed=17)
```

This demonstrates a retrospective cached replay, not human admission or selection into the four-lab development set. Policy seed 17 orders contexts; episode seed 7 orders visible seeds. The record does not write to saved results automatically. A historical claim that becomes ambiguous under corrected qualification is absent from the view; it is not forced back into eligibility.

## Trace reconstruction

`analysis.runs_from_pairs` now retains actual seed identities. The explicit `shared_estimate_from_pairs` diagnostic requires complete visible `(config, seed)` lineage, rejects missing observations or hidden-seed IDs, and rounds cached metrics to the six-decimal precision originally delivered through the interface. It uses the same estimator with actual seeds as identity keys. Legacy `implied_answer` and historical report decomposition retain their original defaults.

New policy records also retain the delivered observations directly, so later reconstruction need not silently replace an old observation with a different cache value. Historical common-estimator diagnostics require a validated matching cache and sufficient lineage; they are not new agent evaluations or causal explanations of an error.

## Validation and remaining work

Run the focused checks from the repository root:

```sh
.venv/bin/python -m pytest -q tests/test_shared_estimator.py tests/test_paired_uniform.py tests/test_analysis.py
```

Controls cover deliberately mismatched replicates, repeated cheap contexts, missing contexts, shared and opposing seed noise, unequal identity sets, duplicate/conflicting observations, metric direction, where/component output, failure accounting, hidden-outcome independence, safe historical conversion, trace precision, and unchanged acquisition ordering.

Next implement question coverage and coverage-plus-cost acquisition using this same public interface and estimator. Keep cost preference separate from target weighting. The optional adaptive policy remains deferred. Run the planned 200 policy seeds at the three protocol budgets only after selecting the reviewed development subset under the existing rule. Freeze policies and estimator at the candidate lock before fresh evaluation; no such lock has been made here.
