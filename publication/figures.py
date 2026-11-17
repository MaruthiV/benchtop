import argparse
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INK = "#292724"
MUTED = "#77716A"
LIGHT = "#E3DFD9"
PALE = "#F5F3EF"
ACCENT = "#A34535"


def load(name):
    return json.loads((ROOT / name).read_text())


def canvas(height):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, height), facecolor="white")
    FigureCanvasAgg(fig)
    return fig


def heading(fig, section, title, subtitle):
    fig.text(0.035, 0.95, section.upper(), size=9, color=MUTED, va="top")
    fig.text(0.035, 0.875, title, size=18, weight="bold", va="top")
    fig.text(0.035, 0.785, subtitle, size=10.5, color=MUTED, va="top")


def save(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        kwargs = {"dpi": 180} if extension == "png" else {"metadata": {"Date": None}}
        fig.savefig(directory / f"{name}.{extension}", facecolor="white", **kwargs)


def overview(directory):
    fig = canvas(2.35)
    ax = fig.add_axes((0.025, 0.05, 0.95, 0.9))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    steps = [
        (0.015, "Question", "Paper-inspired lab\nFixed experiment budget"),
        (0.27, "Investigate", "Choose experiments\nRead the measurements"),
        (0.525, "Submit", "Commit one answer\nReceive a receipt"),
        (0.79, "Grade", "Held-out seeds\nPrivate result"),
    ]
    for i, (x, title, caption) in enumerate(steps):
        color = ACCENT if i == 3 else INK
        ax.text(x, 0.73, title, size=16, weight="bold", color=color, va="top")
        ax.text(x, 0.48, caption, size=11, color=MUTED, va="top", linespacing=1.7)
    for left, right in ((0.19, 0.25), (0.445, 0.505), (0.705, 0.77)):
        ax.annotate("", xy=(right, 0.66), xytext=(left, 0.66),
                    arrowprops={"arrowstyle": "->", "lw": 1.2, "color": MUTED})
    ax.plot((0.015, 0.705), (0.17, 0.17), color=LIGHT, lw=1)
    ax.plot((0.79, 0.985), (0.17, 0.17), color=ACCENT, lw=1)
    ax.plot((0.745, 0.745), (0.17, 0.83), color=LIGHT, lw=1, ls=(0, (3, 3)))
    ax.text(0.015, 0.07, "Agent · three MCP tools", size=9.5, color=MUTED)
    ax.text(0.79, 0.07, "Separate evaluator", size=9.5, color=ACCENT)
    save(fig, directory, "overview")


def reward(directory):
    data = load("demo/reward-check.json")
    q, c = data["qualification"], data["independent_confirmation"]
    series = [
        ("Qualification", "Primary t interval", q["delta"], q["interval"], INK, "-"),
        ("Independent seeds", "Primary t interval", c["delta"], c["interval"], ACCENT, "-"),
        ("Independent seeds", "Bootstrap diagnostic", c["delta"], c["bootstrap_interval"], MUTED, "--"),
    ]
    fig = canvas(4.4)
    heading(fig, "Independent reward check", "The independent interval crosses the margin",
            "Paired-seed contrasts · 12 seeds per phase · fixed margin 0.05")
    ax = fig.add_axes((0.27, 0.23, 0.58, 0.43))
    margin = data["min_effect"]
    ax.axvspan(0, margin, color=PALE, zorder=0)
    ax.axvline(margin, color=MUTED, lw=0.9, ls=(0, (3, 3)))
    for i, (name, method, estimate, interval, color, style) in enumerate(series):
        y = 2 - i
        ax.plot(interval, (y, y), color=color, lw=1.9, ls=style, zorder=3)
        ax.plot(interval, (y, y), color=color, marker="|", markersize=9, ls="none")
        ax.plot(estimate, y, "o", ms=6.5, color=color,
                markerfacecolor="white" if i == 2 else color, markeredgewidth=1.4)
        fig.text(0.035, 0.23 + 0.43 * ((y + 0.5) / 3), name,
                 size=11, color=color, va="bottom", weight="bold" if i == 1 else "normal")
        fig.text(0.035, 0.23 + 0.43 * ((y + 0.5) / 3) - 0.038, method,
                 size=9, color=MUTED, va="top")
        fig.text(0.965, 0.23 + 0.43 * ((y + 0.5) / 3),
                 f"{interval[0]:.3f}–{interval[1]:.3f}", size=9.5,
                 color=color, ha="right", va="center")
    ax.set(xlim=(0, 0.85), ylim=(-0.5, 2.5), yticks=[], xticks=[0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("Estimated improvement in oracle-normalized risk ratio", labelpad=10, size=10)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_color(LIGHT)
    ax.tick_params(axis="x", length=0, pad=8, colors=MUTED, labelsize=9)
    ax.text(margin + 0.012, 2.58, "0.05", size=9, color=MUTED)
    fig.text(0.035, 0.065, "The bootstrap diagnostic gives a different label. The primary grading rule stays fixed.",
             size=10, color=MUTED)
    save(fig, directory, "reward-check")


def development(directory):
    data = load("development/summary.json")["development"]
    names = list(dict.fromkeys(row["lab"] for row in data["cells"]))
    budgets = (0.06, 0.12, 0.25)
    fig = canvas(4.6)
    heading(fig, "Development · Haiku 5.5", "Two labs stayed perfect at every budget",
            "Correct answers out of four independent sessions in each cell")
    ax = fig.add_axes((0.035, 0.16, 0.93, 0.53))
    ax.set(xlim=(0, 10), ylim=(-0.5, 4.5))
    ax.axis("off")
    columns = (4.1, 6.35, 8.6)
    ax.axvspan(3.13, 5.08, color=PALE, zorder=0)
    for x, budget in zip(columns, budgets, strict=True):
        ax.text(x, 4.2, f"{budget:.0%} budget", ha="center", va="center", size=11,
                weight="bold", color=ACCENT if budget == 0.06 else INK)
    for i, name in enumerate(names):
        y = 3.3 - i
        ax.text(0, y, name, va="center", size=12)
        if i:
            ax.plot((0, 10), (y + 0.5, y + 0.5), color=LIGHT, lw=0.7, zorder=1)
        for x, budget in zip(columns, budgets, strict=True):
            row = next(r for r in data["cells"] if r["lab"] == name and r["budget_fraction"] == budget)
            ax.text(x - 0.2, y, f"{row['correct']}/{row['completed']}", va="center",
                    ha="right", size=17, weight="bold")
            for j in range(row["completed"]):
                ax.plot(x + 0.12 + j * 0.16, y, "s", ms=5,
                        markerfacecolor=INK if j < row["correct"] else "white",
                        markeredgecolor=INK if j < row["correct"] else LIGHT, markeredgewidth=0.8)
    fig.text(0.035, 0.065, "40/48 correct overall. The fixed selector chose 6%; these are historical development results.",
             size=10, color=MUTED)
    save(fig, directory, "development")


def classical(directory):
    data = load("development/summary.json")["classical"]
    policies = [
        ("question_coverage", "Question coverage", ACCENT),
        ("paired_uniform", "Paired uniform", INK),
        ("coverage_plus_cost", "Coverage + cost", MUTED),
        ("paper_prior_only", "Paper prior", MUTED),
        ("development_majority_label", "Development majority", MUTED),
        ("random_experiment_design", "Random design", MUTED),
    ]
    fig = canvas(5.2)
    heading(fig, "Classical controls · 14,400 cached replays", "Coverage helps most at the smallest budget",
            "Equal-lab mean accuracy · four historical labs · 200 policy seeds per lab and budget")
    axes = [fig.add_axes((0.275 + j * 0.238, 0.23, 0.182, 0.45)) for j in range(3)]
    for j, (ax, budget) in enumerate(zip(axes, (0.06, 0.12, 0.25), strict=True)):
        ax.set(xlim=(0, 1.22), ylim=(-0.7, 5.7), yticks=[], xticks=[0, 0.5, 1], xticklabels=["0", "50", "100%"])
        ax.set_title(f"{budget:.0%} budget", fontsize=11, weight="bold", loc="left", pad=14)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(LIGHT)
        ax.tick_params(axis="x", length=0, pad=8, colors=MUTED, labelsize=9)
        for x in (0, 0.5, 1):
            ax.axvline(x, color=LIGHT, lw=0.6, zorder=0)
        for i, (policy, name, color) in enumerate(policies):
            y = 5 - i
            row = next(r for r in data["equal_lab_means"] if r["policy"] == policy and r["budget_fraction"] == budget)
            value = row["equal_lab_mean_accuracy"]
            ax.plot((0, value), (y, y), color=LIGHT, lw=1, zorder=1)
            ax.plot(value, y, "o", color=color, ms=5.5, zorder=3)
            ax.text(value + 0.07, y, f"{value * 100:.1f}", color=color, size=9, va="center")
            if j == 0:
                fig.text(0.035, 0.23 + 0.45 * ((y + 0.7) / 6.4), name,
                         size=10.5, color=color, va="center",
                         weight="bold" if i == 0 else "normal")
    fig.text(0.035, 0.09, "Cost-aware acquisition did not consistently help. Identical prior/majority results are shown separately.",
             size=9.5, color=MUTED)
    fig.text(0.035, 0.045, "Replays share cached data; repeated policy seeds do not add independent labs.", size=9.5, color=MUTED)
    save(fig, directory, "classical")


def topology(directory):
    jobs = load("examples/topology/observed.json")["jobs"]
    values = [next(row["metrics"]["lifetime_sum"] for row in jobs
                   if row["config"] == {"probe": probe, "method": "LSF"})
              for probe in ("cycle", "cycle_with_isolated_vertex")]
    fig = canvas(3.7)
    heading(fig, "Source check · topology miniature", "One extra vertex changes the feature",
            "Hand-constructed control · baseline filtration · the loop remains unfilled")
    axes = [fig.add_axes((0.06, 0.13, 0.36, 0.52)), fig.add_axes((0.55, 0.13, 0.39, 0.52))]
    points = [(0.1, 0.1), (0.65, 0.1), (0.65, 0.65), (0.1, 0.65)]
    for j, ax in enumerate(axes):
        ax.set(xlim=(-0.2, 1.3), ylim=(-0.46, 0.95))
        ax.set_aspect("equal")
        ax.axis("off")
        for i, point in enumerate(points):
            nxt = points[(i + 1) % 4]
            ax.plot((point[0], nxt[0]), (point[1], nxt[1]), color=INK, lw=1.6)
            ax.plot(*point, "o", ms=5, color=INK)
        ax.text(0.375, 0.375, "unfilled", ha="center", va="center", size=11, color=MUTED)
        if j:
            ax.plot(1.1, 0.65, "o", color=ACCENT, ms=7)
            ax.text(1.1, 0.81, "value 10", ha="center", size=10, color=ACCENT)
            ax.text(1.1, 0.42, "no edges", ha="center", size=9, color=MUTED)
        ax.text(0.375, -0.20, f"Reported lifetime  {values[j]:g}", ha="center", size=13,
                weight="bold", color=ACCENT if j else INK)
    fig.text(0.485, 0.43, "→", ha="center", size=26, color=MUTED)
    fig.text(0.035, 0.055, "The feature changes from 0 to 9.3 while the existing loop stays essential. The draft was excluded.",
             size=9.5, color=MUTED)
    save(fig, directory, "topology")


def confirmation(directory):
    report = load("release-results/analysis.json")
    lab = report["labs"][0]
    fig = canvas(3.8)
    heading(fig, "Fresh confirmation · " + lab["env_id"], "The comparison stayed unavailable",
            "Independent label: ambiguous · original slots retained · no confirmation models started")
    ax = fig.add_axes((0.035, 0.18, 0.93, 0.48))
    ax.axis("off")
    ax.set(xlim=(0, 10), ylim=(-0.3, 2.7))
    for x, label in ((0, "Arm"), (3.8, "Budget"), (6, "Observed / planned"), (9.4, "Mixed group")):
        ax.text(x, 2.35, label, color=MUTED, size=10, ha="right" if x == 9.4 else "left")
    for i, (arm, label) in enumerate((("global_budget", "Global"), ("calibrated_budget", "Calibrated"))):
        y = 1.6 - i
        row = lab["arms"][arm]
        ax.plot((0, 10), (y + 0.47, y + 0.47), color=LIGHT, lw=0.7)
        ax.text(0, y, label, size=12, va="center")
        budget = report["efficiency"]["global_budget_fraction"] if i == 0 else lab["selected_budget"]
        ax.text(3.8, y, f"{budget:.0%}", size=12, va="center")
        ax.text(6, y, f"{row['observed']} / 8", size=12, va="center")
        ax.text(9.4, y, "Unavailable", color=ACCENT, size=11, va="center", ha="right")
    fig.text(0.035, 0.075, "No primary effect or uncertainty interval is available. Full-cohort bounds remain [−1, 1].", size=10, color=MUTED)
    save(fig, directory, "confirmation")


def render(output):
    with tempfile.TemporaryDirectory(prefix="benchtop-figure-cache-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        import matplotlib

        matplotlib.use("Agg")
        with matplotlib.rc_context({
            "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 11, "text.color": INK, "axes.labelcolor": INK,
            "svg.hashsalt": "benchtop-publication-figures-v3", "axes.unicode_minus": True,
        }):
            for plot in (overview, reward, development, classical, topology):
                plot(output / "demo")
            confirmation(output / "release-results/figures")


def main():
    parser = argparse.ArgumentParser(description="Render publication figures from the packaged, unchanged evidence.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render(args.output_dir)
    print(json.dumps({"figures": 6, "model_calls": 0, "lab_workers": 0, "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
