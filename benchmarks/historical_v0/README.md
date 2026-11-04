# Historical prototype baseline

This snapshot preserves the prototype before the release fixes change grading, environment discovery, or agent responses. It contains the existing public source, tests, examples, labs, results, figures, dependency lock, and README. Raw transcripts, private state, planning notes, credentials, virtual environments, and caches are excluded.

`snapshot.tar.gz` stores the original file bytes. `manifest.json` records a SHA-256 checksum and size for every archived file, the historical grading behavior, collection counts, and the installed runtime versions. `baseline_tests.json` records the test baseline and its execution conditions. This directory is an immutable historical reference; future experiments belong in a new version.

The archive contains 241 files, including 30 paper-derived labs with 78 questions, the label-smoothing example, and 833 recorded rollout rows. Its compressed size is about 1.4 MB.

Verify the archive using only the Python standard library:

```sh
python3 benchmarks/historical_v0/verify.py
```

To also compare the current project files with the snapshot:

```sh
python3 benchmarks/historical_v0/verify.py --working-tree
```

The archive check should continue to pass after development. The working-tree comparison should report differences once the preserved files change. Verification checks stored bytes and membership; it does not rerun model rollouts or certify scientific conclusions. The archived data can be inspected with standard tar tools without model credentials.

The baseline suite passed 15 tests outside the outer agent sandbox. In the outer sandbox, 13 passed and the two macOS worker tests failed during startup. Lab workers in the successful run still used macOS `sandbox-exec`; the restriction around the test runner was removed. Neither run makes additional tests for the known bugs unnecessary.

The archived grader is identified as `historical_v0`. It uses a normal multiplier of 1.96 and combines per-context variances as if contexts were independent, although contexts reuse seeds. Some labels can change under a corrected estimator. The archived server also has the known repeated-submission grading leak, and accepted-environment discovery depends on ignored pipeline logs. These are preserved historical behaviors, not fixed by this snapshot.

Treat these results as development evidence. Keep their version distinct from the corrected grader and fresh evaluation, and do not regenerate reports into this snapshot.
