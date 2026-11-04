# Foundation reproduction

Run these engineering checks before the Qualia smoke or any release evaluation. They rebuild the historical report and exercise current interfaces; they do not run the fresh calibration study or establish scientific fidelity.

## Install and verify

From the repository root, install the locked dependencies once:

```sh
uv sync --frozen
```

Then run the checker without fetching papers, installing packages, calling models, or starting workers:

```sh
.venv/bin/python -m benchtop.cli verify
```

The default proof is `artifacts/foundation_check.json`. The terminal prints a short result; the full JSON includes input hashes, runtime/dependency versions, the timestamp, and the scope of each check. Failures return a nonzero exit status. This command runs on a host with the installed dependencies and does not require the live sandbox.

To include a new synthetic sweep on macOS with `sandbox-exec`:

```sh
.venv/bin/python -m benchtop.cli verify --live --output artifacts/foundation_live.json
```

This executes 24 tiny runs of a stdlib-only fixture, then resumes it through the CLI with workers forbidden. The fixture has six total seeds and two visible seeds. Its wrapped corrected-task artifact is discovered through a separate catalog and played through the in-process adapter. It is an engineering fixture, not a paper-derived reference or an admitted release lab. Unsupported live execution fails closed.

Output paths inside the repository must be under `artifacts/`; paths outside the repository are also supported. The checker cannot overwrite a repository input with its proof file. Proof files are local generated artifacts and are ignored by Git.

## What is reproduced

The checker verifies the frozen 241-file archive and all 200 original lab/result/figure files against the historical manifest. It copies the declared public code/data inputs into a temporary directory, excluding `.state/`, saved figures, Git metadata, local docs, and the installed environment. Child processes explicitly import code from that selected copy, using the already installed dependencies.

In the copy it removes the saved summary, creates an empty figure directory, and invokes the real `report` CLI. It compares every saved historical summary field, including the scores and supporting analyses, with regenerated values. Numeric comparison permits only roundoff at `1e-12`; new metadata fields can be added. The current snapshot has 20 saved summary fields, 30 labs, and 78 questions. Five SVGs and five PNGs are regenerated. Rendering metadata and fonts can vary, so regenerated figure bytes are not compared with the original files.

The offline path also checks catalog agreement between evaluation and the in-process adapter, cached observations, a grade-free completion observation, one terminal trainer reward, and deterministic paired-uniform replay. The trusted checker supplies the known fixture/reference answer to test the interface; this is not an agent accuracy measurement. It asserts that the offline copy never needs or creates `.state/`.

With `--live`, new worker logs and fixture files exist only in the disposable copy. The sweep binds its current inputs and rows, cached resume preserves the manifest's seed count and execution/task versions, and the modern artifact passes the same adapter checks. All temporary files are removed afterward. The original source/data hashes are checked again before writing a successful proof.

## In-process contract

```python
from benchtop.gym import list_episodes, make

episodes = list_episodes()  # validated catalog membership and task artifacts
lab = make(episodes[0][0], episodes[0][1])
observation, info = lab.reset(seed=7)
```

`list_episodes(envs_dir=...)` and `make(..., envs_dir=...)` support explicit local roots. Cached adapters retain recorded execution provenance and cannot authorize live work. Each root has its own cached environment objects. `step` requires `reset`; after completion, later steps return a completion observation, zero reward, and no repeated private result. Reset starts a new episode.

Observations go to the agent. The separate reward and info fields are for the trusted trainer, so do not pass the full trainer tuple into an agent tool response. This adapter is single-caller in-process replay; concurrent network mutations use the server Hub and its episode locks.

For a new sweep, `--seeds N` sets the total seed count explicitly. Without it, the sweep uses the manifest count, or 16 when none is recorded. An explicit change to a bound cache is rejected without altering the old data. Workers execute the `_worker.py` belonging to the selected checkout, avoiding a different editable installation resolved through `python -m`.

## Regression suite and remaining gates

```sh
.venv/bin/python -m pytest -q
```

On macOS, sandbox and socket tests need the host permissions required to start local workers/listeners. Inside an outer agent sandbox, those permissions may be restricted even though the lab's own sandbox remains enabled. Cached reproduction works without that live backend. The suite separately exercises grading, authorization, concurrency, cancellation, cache validation, real socket separation, and worker restrictions.

The checker records zero model sessions and `qualia: not_tested`. A real Qualia connection is still the final foundation gate. Admission reviews, stronger baselines, the fresh-cohort study, the public release/demo, and outreach readiness remain later work. This proof does not bind an exact evaluation model or satisfy the protocol's later execution locks.
