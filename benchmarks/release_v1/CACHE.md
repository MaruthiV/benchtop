# Cache provenance and supported execution

New caches bind observations to their execution inputs. Historical observations remain verified snapshots without inventing runtime provenance for the old experiment. These checks establish artifact consistency; scientific fidelity still requires admission review.

## Execution identity

New labs have `cache.json`, schema version 1. Its ID hashes the exact bytes of `lab.py` and `space.json`; total/visible seeds and any `seed_splits` definition; Python version/implementation, OS/release, architecture, and installed distribution names/versions; and worker, sandbox, space parser, and cache source hashes.

Runtime matching is deliberately conservative: even an unrelated installed-package version change requires a new current-execution cache. Package versions are captured once per process; restart after editing application code or changing the installed environment. Modified dependency binaries under an unchanged version are outside this check. Catalog membership, paper metadata, and sweep summaries cannot make an old physical result current.

Each result row carries `execution_id` and `row_sha256`. Loading validates identities, normalized configurations, seed bounds, finite metrics, primary metric presence, and finite nonnegative runtimes. Mixed caches, edited rows, unknown schemas, and incomplete JSON raise errors. Duplicate cells use the latest successful row. These hashes detect accidental corruption and mixing; they do not authenticate an operator who rewrites data and matching hashes.

Loaded environments track file signatures. Runs and final submissions reject changed files before delivering observations or grading. Live work checks again before persistence and commit. Its own successful append updates the tracked result signature. External writers require a reload; multiple processes writing the same lab directory are unsupported.

## Task and score versions

Corrected task artifacts bind execution identity, qualification seed IDs and metric values, and generator sources (`tasks.py`, `stats.py`, `space.py`, and `cache.py`). Tasks and diagnostic artifacts have payload checksums. Loading rejects changed qualification values, stale generators, altered labels, and inconsistent seed sources. Visible observations are excluded from the qualification digest so live appends do not invalidate grading.

Private episode summaries and new baseline/rollout records carry `execution_id`, `task_version`, `grader_id`, and `charging_id`. Reports reject missing or mismatched execution, task, and charging versions for new caches, as well as grader mismatches. Original untagged historical scores retain their historical interpretation.

A task-generator change can reuse physical results when execution inputs still match. A sweep explicitly rebuilds the labels and diagnostic artifact from validated results. A fully cached sweep starts no workers. Saved scores remain bound to their original task version.

## Current execution and recorded replay

`Env.load(path)` checks current execution inputs, including runtime and engine sources. `Env.load(path, replay=True)` explicitly retains recorded runtime and engine provenance while checking local lab/space/seed inputs, result rows, and task bindings. Replay environments cannot authorize live execution and still require compatible task generators.

Cached server mode, report generation, and classical baseline replay use recorded provenance. Live mode requires a current binding and a supported sandbox even when requested cells are cached. Offline inspection on another host does not make that host the source of the recorded measurements.

The [charging schedule](BUDGETS.md) stays frozen per loaded environment and is retained separately from its execution and task identities.

## Historical snapshots and rebuilding

[envs/historical_cache.json](../../envs/historical_cache.json) records original lab, space, manifest, result, and task hashes for all 31 preserved lab directories: the 30-lab evaluation cohort plus the separate label-smoothing example. Hashes come from the unchanged historical archive manifest. Original files are not rewritten or assigned guessed package versions.

Unversioned results load only when they match that index. Changed historical files are rejected. Historical directories support cached replay and retrospective review; sweeps and live stores cannot append to or upgrade them. The index does not change cohort membership or admit any lab to the fresh release.

For new live work, create a separate directory with reviewed `lab.py`, `space.json`, and a manifest containing a new ID and explicit seed counts. Leave out old result, task, and cache files. On a supported host:

```sh
uv run benchtop sweep /path/to/new-lab
```

The sweep initializes provenance before collecting results and writes corrected tasks afterward. Without `--seeds`, the CLI preserves the manifest's total seed count; a new lab without a count defaults to 16. Later sweeps resume only under matching execution inputs. Stale caches raise errors without deleting old data or relabeling it. Preserve the old directory and rebuild under a new version. [Fresh three-part manifests](SEEDS.md) now validate group membership and default to sweeping visible plus qualification, reserving confirmation. Named visible/qualification stages share the same bound execution cache; changing stage does not change membership. The locked confirmation runner and release lock enforcement remain pending.

## Supported sandbox and evidence

Physical execution requires macOS with `sandbox-exec`. Automatic and explicit mode selection, live-server startup, live-store creation, sweeps with missing jobs, and worker startup reject unsupported execution. Linux, missing `sandbox-exec`, `none`, and unknown modes have no unisolated fallback. Cached replay does not require this backend.

Workers retain network/write restrictions. Their actual lab directory is added to read denials, including for custom `--envs` locations, so they cannot open its result files. A worker verifies its private lab copy against the pool's original code digest and executes the worker script from the selected checkout. Startup failures clean up partially created workers.

Regression evidence covers changed execution/generator inputs, corrupt and mixed rows, stale labels, non-destructive sweep rejection, active and in-flight file changes, read-only replay, historical verification, unsupported execution, changed worker copies, real live/cache equality, custom-directory read denial, and a fresh sweep with execution-free resume. No model sessions or fresh paper generation are needed for these checks.

The [foundation reproduction command](REPRODUCTION.md) now checks the documented CLI and in-process paths in a temporary public tree, with an optional new synthetic sweep. Real Qualia connectivity remains unverified.
