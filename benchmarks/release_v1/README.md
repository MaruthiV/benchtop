# Release evaluation protocol

The [candidate metadata/rule lock](CANDIDATES.md) binds a verified development budget, reviewed baseline/estimator freeze, generation rules and the first 16 numerically ordered unseen paper IDs with fetched versions. [Bounded fresh generation](GENERATION.md) consumes that lock, preserves provider/repair evidence, freezes human scientific reviews and margins before qualification, and reserves confirmation seeds. The [cohort lock](COHORT.md) verifies the full funnel and freezes the first eight eligible labs or the smaller actual cohort. Actual candidate selection and fresh model results remain pending.

[spec.json](spec.json) defines the next evaluation before fresh model results exist. It is a protocol, not a completed experiment or a runnable CLI configuration. Corrected grading, endpoint separation, frozen charging, and execution-cache versioning are implemented; the fresh seed-split harness and smoke/development rollout records are implemented; locked fresh collection remains pending.

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

[Versioned classical baselines](BASELINES.md) implement paired-uniform, question-coverage, and coverage-plus-cost acquisition with a shared visible-data estimator. All use the same public problem/cost/budget interface; reconstruction pairs identities, keeps equal target-context weights, and exposes its neutral missing-context prior. Coverage policies plan affordable complete pairs and spread factor-level coverage; cost preference only breaks coverage-priority ties. Read-only corrected historical views keep source labels and prices intact. The actual human-selected development comparison, estimator sensitivity review, and candidate lock remain pending; the lock writer and verifier are implemented.

The [classical replay harness](REPLAY.md) now implements the three sanity controls, bound plans and slots, raw immutable attempt records, fixed visible ordering, resumable execution, and missing-outcome summaries. The actual reviewed 200-seed development comparison remains gated by six completed human records and four eligible claim labs. No model development or candidate lock is created by a classical plan.

[Fresh seed partitions](SEEDS.md) now validate three separate groups and support staged visible/qualification sweeps. Qualification labels and prices exclude confirmation data, and ordinary sweeps reserve the confirmation group. The [locked confirmation runner](CONFIRMATION_EXECUTION.md) binds private independent grading; actual model evaluation remains pending.

The [budget-calibration selector](CALIBRATION.md) computes Beta predictive mixed-group probabilities, selects the development global control, and applies explicit fresh-pilot fallbacks. Its CLI reads saved normalized outcomes and preserves unknown costs and missing slots. The included example is synthetic; no model study or phase lock is implied by a selector diagnostic.

The [canonical development bridge](CALIBRATION.md#canonical-development-collection) verifies primary journal, provider, and MCP evidence before running that selector. It retains the full planned grid, excludes smoke from selection, and closes collection under the same lock as new starts. The [fresh pilot bridge](PILOT_CALIBRATION.md) applies the rule separately to each cohort member and freezes choices or explicit global-budget fallbacks. Closed results cannot acquire extra phase observations. No actual model budget has been selected.

The [confirmation lock](CONFIRMATION.md) binds that closed collection to two independent arms, eight replicate slots each, with fixed dispatch order and matched controls. It appends a protocol binding before [independent confirmation truth](CONFIRMATION_DATA.md) is collected into a separate private cache. [Separate execution authorization](CONFIRMATION_EXECUTION.md) now binds stable independent grading and ordered starts to the original journal. Neither the lock nor a truth assessment alone authorizes provider calls. [Canonical collection, paired analysis and matched classical replay](CONFIRMATION_RESULTS.md) are implemented; actual model results remain pending.

[Shared Qualia accounting](QUALIA_PRIMARY.md) uses two guarded UI slots in the original journal, with explicit operator observations and no promotion into scientific results. Actual UI compatibility remains unobserved.

[Immutable study preparation](STUDY.md) binds smoke/development configuration and exact group/slot/attempt identities, publishes exclusive plans and receipts, and checks admission/foundation/model prerequisites. [Fresh pilot preparation](PILOTS.md) binds the immutable cohort to the same canonical journal and subscription provider. Preparation alone remains non-executable. The [provider path](PROVIDER.md) adds primary accounting, six-smoke verification, development-lock publication and pilot authorization; confirmation execution uses the verified independent assessment; actual provider evidence and study results remain pending.

The [engineering session ledger](SESSIONS.md) adds shared start/concurrency allowances, immutable completion and reconciliation records, explicit technical retries, and unknown-cost accounting. It preserves late committed answers and stops new work after a reported model change. The journal itself launches no model. Primary authorization requires the canonical directory and refuses unreconciled legacy integration usage; shared primary Qualia invocation routing remains pending.

The [scripted MCP runner](RUNNER.md) now connects those records to an owned loopback endpoint and Python process, preserves raw visible tool replies and private sealed receipts, and resumes completed evidence without relaunching work. Episode sealing prevents queued or late tool mutations from changing a finalized result. The scripted controls call no language model. A separate provider command can start one authorized smoke/development model attempt; no actual smoke result or primary lock exists yet.

## Validation of this checkpoint

The specification was checked for valid JSON, disjoint seed splits, resource arithmetic, consistent group sizes, required lock gates, unique reference identities, and agreement with the historical archive checksum. No fresh papers were fetched, model episodes run, or application behavior changed for this checkpoint.

Submission responses now return completion receipts without grading details, including repeated submissions and exhausted format corrections. Regression coverage checks MCP and HTTP and confirms that later answers cannot replace the committed result.

Evaluation and reporting now share the explicit [environment catalog](../../envs/README.md), including the recorded historical funnel. A copied project without private state, saved summaries, or figures discovers the same 30 labs and 78 questions and regenerates its report and figures. This preserves the historical cohort; it does not admit those labs to the fresh release.

The [corrected grader and versioned task artifacts](GRADING.md) are implemented. Their separate cached-data review retains 75 eligible questions and makes three previously settled contrasts ambiguous; no settled label flips. Historical files and scores remain under `historical_v0`, and reports reject mismatched grader versions.

[Agent and evaluator access](ACCESS.md) now use separate listeners, episode-scoped capabilities, and a private local admin credential. The integration tests exercise both ports and credential cleanup, but Qualia connectivity is still unverified.

[Charging and episode operations](BUDGETS.md) use a shared immutable price schedule and one lock per episode across both interfaces. Failed and cancelled work uses a call slot without charging for undelivered observations; retries preserve missing replicates, and waiter cancellation preserves shared live work.

[Cache provenance](CACHE.md) now binds execution inputs, result rows, qualification data, and task generators. Historical files remain verified snapshots; recorded replay cannot authorize live execution. Stale sweeps preserve old data, and unsupported live sandboxes fail closed.

[Foundation reproduction](REPRODUCTION.md) now provides a CLI check in a disposable copy without private state, compares all saved historical report fields, and optionally exercises a new synthetic sweep and worker-free resume. The in-process adapter supports wrapped artifacts, catalog membership, recorded replay, and one terminal trainer reward. Actual Qualia use and the locked model-evaluation harness remain pending.

[Qualia integration preparation](QUALIA.md) now performs a non-model MCP preflight, saves private connection configuration, records up to two intended UI invocations, and captures MCP trace plus evaluator evidence. A successful public-tunnel preflight has been observed; the actual Qualia UI run is still pending. Preparation and surrogate test clients do not establish a Qualia pass.

[Admission evidence](ADMISSION.md) now has a read-only cached screen and source-bound human-review records. All 30 historical labs pass the mechanical screen, 28 have corrected qualified claims, and none has been scientifically admitted yet. The screen retains ambiguous and invalid cases, prior model concerns, and context-dominance signals while keeping model difficulty unmeasured. The [six-case reference package](admission/references/README.md) adds pinned paper citations, assistant review drafts, and reproducible coverage diagnostics. The [controlled defect suite](admission/defects/README.md) adds six specific changes, ten valid controls, and bounded sandbox measurements of the Muon update path and input sharing. The [consolidated report](admission/REPORT.md) and [owner guide](admission/REVIEW.md) now join the evidence and prepare separate unassigned records. Human decisions and remaining scientific/runtime evidence stay pending, independently of the Qualia UI step.
