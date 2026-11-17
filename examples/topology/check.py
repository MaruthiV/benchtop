import asyncio
import json
import math
from pathlib import Path

from benchtop.sandbox import run_jobs


async def main():
    folder = Path(__file__).resolve().parent
    expected = json.loads((folder / "observed.json").read_text())["jobs"]
    jobs = [(row["config"], row["seed"]) for row in expected]
    actual = await run_jobs(folder / "lab.py", jobs, workers=1, timeout=10)
    for observed, recorded in zip(actual, expected, strict=True):
        if not observed["ok"]:
            raise RuntimeError("control worker failed: " + str(observed.get("error")))
        for metric, value in recorded["metrics"].items():
            if not math.isclose(observed["metrics"][metric], value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("control differs: " + metric)
    print(json.dumps({"status": "reproduced", "controls": len(actual), "model_calls": 0,
                      "reported_worker_seconds": sum(row["secs"] for row in actual),
                      "scope": "Hand-constructed feature controls; no classifier-accuracy or paper-defect claim."}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
