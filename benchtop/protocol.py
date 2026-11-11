import os

from benchtop.env import ROOT

NAME = os.environ.get("BENCHTOP_PROTOCOL", "release_v1")
if NAME not in ("release_v1", "release_v2"):
    raise ValueError("BENCHTOP_PROTOCOL must be release_v1 or release_v2")

DIRECTORY = ROOT / "benchmarks" / NAME
SPEC = DIRECTORY / "spec.json"
REFERENCES = DIRECTORY / "references.json" if NAME == "release_v2" else DIRECTORY / "admission/references/selection.json"
BINDINGS = DIRECTORY / "bindings.json"
