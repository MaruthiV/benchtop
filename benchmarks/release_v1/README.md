# Release evaluation protocol

[spec.json](spec.json) defines the next evaluation before fresh model results exist. It is a protocol, not a completed experiment or a runnable CLI configuration. Corrected grading, endpoint separation, frozen charging, and execution-cache versioning are implemented; the fresh seed-split harness and protocol-specific rollout records remain pending.

The primary question is whether a small calibration batch can select a useful experiment budget for one agent better than choosing one global budget on development labs. The measured outcome is how often a group of eight rollouts contains both correct and incorrect answers. That can inform environment selection for training, but mixed rewards can also arise from guessing, malformed answers, or timeouts. Report accuracy, failures, classical controls, and costs alongside it.

The historical baseline remains separate in [historical_v0](../historical_v0/README.md).

## Fixed protocol

| Choice | Rule |
|---|---|
| Primary task | One `claim` question per lab, graded 0 or 1 |
| Development | Four qualified, manually reviewed historical claim labs |
| Fresh cohort | First eight eligible labs from at most 16 ordered unseen papers |
| Budgets | 6%, 12%, and 25% of a frozen visible-grid charging schedule |
| Seed splits | Four visible seeds, 12 for qualification, 12 for independent confirmation |
| Grading | Equal-context contrasts aggregated within each seed, then a 95% Student t interval across seed means |
| Calibration | Four pilots per candidate budget, Beta(1,1) prior, posterior-predictive probability of a mixed group of eight |
| Control | One global budget selected on development data using the same objective |
| Confirmation | Eight new rollouts per arm per fresh lab |
| Resource limit | 272 planned evaluation sessions, plus 28 reserved starts; no new paid services |
| Analysis | Paired lab-level comparison, raw outcomes, uncertainty, missing-data bounds, and separate pilot costs |

The candidate ordering uses metadata, not a relevance ranking or model performance. All candidate failures remain in the funnel. A smaller eligible cohort is reported as smaller. There are no replacements for poor model scores or unstable confirmation labels.

## Details locked as implementation becomes available

The scientific rules above are defined now. Exact artifacts are bound at four explicit gates in `spec.json`:

1. **Development lock:** reference reviews, four development lab IDs, exact model and CLI configuration, prompts, grader, charging schedule, and code hashes. Smoke runs use historical development labs only.
2. **Candidate lock:** global budget, classical policies and shared estimator, generation and admission rules, and the archived fresh paper list. These are fixed before generating fresh labs.
3. **Cohort lock:** human reviews, qualification labels, admitted lab IDs, margins, and all environment versions. These are fixed before fresh pilots or looking at confirmation outcomes.
4. **Confirmation lock:** recorded pilots, chosen budgets or fallbacks, and all planned confirmation slots. These are fixed before confirmation outcomes are inspected.

The exact model ID and fresh date are intentionally unresolved. `bindings.json` will record actual model/configuration IDs and immutable, dated lock records. It has not been created. The date follows a predefined metadata rule; it is not selected after viewing model results. Do not represent the experiment as execution-ready while these gates or required correctness fixes remain open.

If a protocol choice changes, preserve the earlier version and record what changed and which observations already existed. Choices changed after the affected outcomes are observed belong to exploratory analysis.

## Rollout identity and failure handling

Every group has a hash derived from the protocol, phase, arm, environment/task versions, model configuration, prompt template, grader, budget, charging schedule, and visible-seed ordering. Each independent replicate has its own planned `slot_id`. Its initial invocation and optional technical retry have distinct `attempt_id` values linked to that slot.

These identities keep independent rollouts separate even when they share an episode seed and prevent an incorrect completed answer from being replaced with a better rerun. Even when calibrated and global arms choose the same budget, their model sessions and slots remain distinct. Binding cores are hashed before group and slot IDs are computed; plan hashes live in separate lock records to avoid circular identities.

Incorrect answers, exhausted format correction, agent timeouts, and exits without an answer are agent outcomes with reward zero on a valid task. Confirmed service or execution failures remain missing, receive at most one technical retry per slot, and remain visible. An ambiguous or changed confirmation label makes the lab's reward comparison unavailable; it stays in cohort accounting and is not replaced. Missing observations are included in conservative bounds as well as the complete-case report.

The 28-start reserve covers at most six smoke episodes, two integration episodes, and 20 technical retries. Every invoked rollout session counts, including failures. Generation calls and cheap classical replay are logged separately.

## Validation of this checkpoint

The specification was checked for valid JSON, disjoint seed splits, resource arithmetic, consistent group sizes, required lock gates, unique reference identities, and agreement with the historical archive checksum. No fresh papers were fetched, model episodes run, or application behavior changed for this checkpoint.

Submission responses now return completion receipts without grading details, including repeated submissions and exhausted format corrections. Regression coverage checks MCP and HTTP and confirms that later answers cannot replace the committed result.

Evaluation and reporting now share the explicit [environment catalog](../../envs/README.md), including the recorded historical funnel. A copied project without private state, saved summaries, or figures discovers the same 30 labs and 78 questions and regenerates its report and figures. This preserves the historical cohort; it does not admit those labs to the fresh release.

The [corrected grader and versioned task artifacts](GRADING.md) are implemented. Their separate cached-data review retains 75 eligible questions and makes three previously settled contrasts ambiguous; no settled label flips. Historical files and scores remain under `historical_v0`, and reports reject mismatched grader versions.

[Agent and evaluator access](ACCESS.md) now use separate listeners, episode-scoped capabilities, and a private local admin credential. The integration tests exercise both ports and credential cleanup, but Qualia connectivity is still unverified.

[Charging and episode operations](BUDGETS.md) use a shared immutable price schedule and one lock per episode across both interfaces. Failed and cancelled work uses a call slot without charging for undelivered observations; retries preserve missing replicates, and waiter cancellation preserves shared live work.

[Cache provenance](CACHE.md) now binds execution inputs, result rows, qualification data, and task generators. Historical files remain verified snapshots; recorded replay cannot authorize live execution. Stale sweeps preserve old data, and unsupported live sandboxes fail closed.

[Foundation reproduction](REPRODUCTION.md) now provides a CLI check in a disposable copy without private state, compares all saved historical report fields, and optionally exercises a new synthetic sweep and worker-free resume. The in-process adapter supports wrapped artifacts, catalog membership, recorded replay, and one terminal trainer reward. The next checkpoint is an actual Qualia smoke. The fresh three-part split and evaluation harness remain pending.

[Qualia integration preparation](QUALIA.md) now performs a non-model MCP preflight, saves private connection configuration, records up to two intended UI invocations, and captures MCP trace plus evaluator evidence. A successful public-tunnel preflight has been observed; the actual Qualia UI run is still pending. Preparation and surrogate test clients do not establish a Qualia pass.

[Admission evidence](ADMISSION.md) now has a read-only cached screen and source-bound human-review records. All 30 historical labs pass the mechanical screen, 28 have corrected qualified claims, and none has been scientifically admitted yet. The screen retains ambiguous and invalid cases, prior model concerns, and context-dominance signals while keeping model difficulty unmeasured. The [six-case reference package](admission/references/README.md) adds pinned paper citations, assistant review drafts, and reproducible coverage diagnostics. The [controlled defect suite](admission/defects/README.md) adds six specific changes, ten valid controls, and bounded sandbox measurements of the Muon update path and input sharing. Human decisions, remaining scientific/runtime evidence, and the consolidated report remain pending, independently of the Qualia UI step.
