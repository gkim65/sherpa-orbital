"""
plot_reward_figure.py — a typeset figure of the science+safety POMDP reward.

Renders the reward R(s,a) with its four terms colour-coded to their meaning (survive /
fuel / science / safety), a small table of the constants, and an honest "weights are
placeholders" note. Matches the deck style (Computer Modern serif via LaTeX or the
mathtext-cm fallback, so it always renders — no external math preview needed).

Numbers mirror pomdp-julia/src/stationkeeping_pomdp.jl exactly.

Run:  python -m scripts.plot_reward_figure
"""
from __future__ import annotations
import shutil
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

C = {"ink": "#1a1a1a", "muted": "#5f6368",
     "survive": "#2e8b57", "fuel": "#c98a00", "science": "#3572A5", "safety": "#c0392b"}


def _use_serif():
    if shutil.which("latex"):
        matplotlib.rcParams.update({"text.usetex": True, "font.family": "serif",
                                    "font.serif": ["Computer Modern Roman"]})
    else:
        matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif",
                                    "mathtext.fontset": "cm"})
    matplotlib.rcParams.update({"font.size": 12})


def main():
    _use_serif()
    fig = plt.figure(figsize=(9.6, 5.2))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)

    ax.text(5, 5.65, "Reward for a non-terminal state $s=(\\mathrm{dev},\\,\\mathrm{cov})$ "
            "and action $a$", ha="center", fontsize=13.5, color=C["ink"])

    # The equation, term by term (colour-coded). Assembled as separate coloured pieces.
    y = 4.35
    ax.text(0.55, y, r"$R(s,a)\;=$", fontsize=15, color=C["ink"], va="center")
    ax.text(2.15, y, r"$+\,0.5$", fontsize=15, color=C["survive"], va="center")
    ax.text(3.35, y, r"$-\;\Delta V(a)$", fontsize=15, color=C["fuel"], va="center")
    ax.text(5.15, y, r"$+\;20\,\cdot\,\mathbf{1}[\mathrm{new\ band}]$",
            fontsize=15, color=C["science"], va="center")
    ax.text(7.95, y, r"$-\;200\,\cdot\,P_{\mathrm{lose}}(s,a)$",
            fontsize=15, color=C["safety"], va="center")

    # Under-braces / labels for each term.
    yl = 3.75
    ax.text(2.15, yl, "survive\n(per pass alive)", ha="left", fontsize=9, color=C["survive"], va="top")
    ax.text(3.35, yl, "fuel\n($\\Delta V$ of the action)", ha="left", fontsize=9, color=C["fuel"], va="top")
    ax.text(5.15, yl, "science\n(first visit to a band)", ha="left", fontsize=9, color=C["science"], va="top")
    ax.text(7.95, yl, "safety\n(exp. crash/escape)", ha="left", fontsize=9, color=C["safety"], va="top")

    ax.text(5, 2.75, r"and $R=0$ once terminal (CRASHED / LOST): the episode ends.",
            ha="center", fontsize=10.5, color=C["muted"])

    # Constants table.
    rows = [
        ("survive / step", r"$+0.5$", C["survive"]),
        ("fuel: OBSERVE / CORRECT", r"$-0 \; / \; -1.3$", C["fuel"]),
        ("fuel: EXCURSE LOW / MID / HIGH", r"$-2.1 \, / \, -3.1 \, / \, -9.9$", C["fuel"]),
        ("science: new band", r"$+20$", C["science"]),
        ("safety: crash or escape", r"$-200$", C["safety"]),
    ]
    y0 = 2.15; dy = 0.36
    ax.text(1.4, y0 + 0.32, "term", fontsize=9.5, color=C["muted"])
    ax.text(6.6, y0 + 0.32, "value (reward units $\\approx$ m/s equiv.)", fontsize=9.5, color=C["muted"])
    for i, (name, val, col) in enumerate(rows):
        yy = y0 - i * dy
        ax.text(1.4, yy, name, fontsize=10, color=C["ink"], va="center")
        ax.text(6.6, yy, val, fontsize=10.5, color=col, va="center")

    ax.text(5, 0.12,
            r"Weights are hand-set placeholders encoding a ranking "
            r"(mission loss $\gg$ science $\gg$ fuel); not yet tuned to a mission utility "
            r"or a real science model.",
            ha="center", fontsize=8.5, color=C["muted"], style="italic")

    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"reward_figure.{ext}", transparent=True,
                    bbox_inches="tight", dpi=200)
    print(f"wrote {FIG_DIR}/reward_figure.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
