import contextlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback


def load_lab(path: str):
    spec = importlib.util.spec_from_file_location("lab", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run", None)):
        raise TypeError("lab.py must define run(config, seed)")
    return module


def seed_everything(seed: int) -> None:
    random.seed(seed)
    if "numpy" in sys.modules:
        sys.modules["numpy"].random.seed(seed)
    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(seed)


def clean_metrics(out) -> dict:
    if not isinstance(out, dict) or not out:
        raise TypeError(f"run() must return a non-empty dict of metrics, got {type(out).__name__}")
    metrics = {}
    for name, value in out.items():
        if isinstance(value, bool):
            value = float(value)
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise TypeError(f"metric {name!r} is not a number: {value!r}") from None
        if not math.isfinite(value):
            raise ValueError(f"metric {name!r} is not finite: {value}")
        metrics[str(name)] = value
    return metrics


def main() -> None:
    # keep the real stdout for the protocol, send lab prints to stderr
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    try:
        lab = load_lab(sys.argv[1])
        if "torch" in sys.modules:
            torch = sys.modules["torch"]
            torch.set_num_threads(1)
            with contextlib.suppress(RuntimeError):
                torch.set_num_interop_threads(1)
        proto.write(json.dumps({"ready": True}) + "\n")
    except Exception as e:
        proto.write(
            json.dumps(
                {"ready": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}
            )
            + "\n"
        )
        return

    for line in sys.stdin:
        job = json.loads(line)
        start = time.perf_counter()
        try:
            seed_everything(int(job["seed"]))
            metrics = clean_metrics(lab.run(dict(job["config"]), int(job["seed"])))
            reply = {"ok": True, "metrics": metrics}
        except Exception as e:
            reply = {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-3000:],
            }
        reply["secs"] = time.perf_counter() - start
        proto.write(json.dumps(reply) + "\n")


if __name__ == "__main__":
    main()
