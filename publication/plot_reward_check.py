import argparse
import os
import tempfile
from pathlib import Path

from publication.figures import INK, reward

parser = argparse.ArgumentParser(description="Render the recorded reward check without running experiments.")
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
with tempfile.TemporaryDirectory(prefix="benchtop-reward-plot-") as cache:
    os.environ.setdefault("MPLCONFIGDIR", cache)
    import matplotlib
    matplotlib.use("Agg")
    with matplotlib.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                                "text.color": INK, "axes.labelcolor": INK,
                                "svg.hashsalt": "benchtop-publication-figures-v3"}):
        reward(args.output_dir)
