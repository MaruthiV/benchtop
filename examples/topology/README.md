# A loop that never closes

The generated lab treats an essential loop as if it dies at the last filtration value. That value can come from an unrelated vertex.

The control is a four-edge cycle with vertex values 0.1, 0.4, 0.7 and 0.2. Add an isolated vertex with value 10. The cycle still has no filling triangle. Its reported lifetime changes from 0 to 9.3 under the baseline filtration. The other two filtration rules also change. The essential-loop count stays one.

This does not prove the paper is wrong, or measure an effect on classification accuracy. It shows why the generated features need an explicit convention for essential classes before the miniature can support its stated comparison. The draft was excluded before qualification.

The nine recorded controls cover a filled triangle, the cycle, and the cycle with the isolated vertex under all three rules. `lab.py` preserves the generated draft plus its original small observer. `observed.json` contains the recorded outputs. The structured source assessment is included in the review supplement.

To execute the controls again on macOS with the supported sandbox:

```sh
uv run python -m examples.topology.check
```

This starts one local sandbox worker and no models. The main release reproduction uses cached tables and does not execute this example.
