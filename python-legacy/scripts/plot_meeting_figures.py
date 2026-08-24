"""
plot_meeting_figures.py — two figures for the 2026-07-06 status meeting.

  Fig 1  pipeline  : the Julia (solve) -> JSON policy -> Python (evaluate) architecture,
                     and where the crude toy model vs the real CR3BP truth model sit.
  Fig 2  results   : POMDP-vs-MPC survival time at each fidelity rung (the headline table).

Styling follows the user's global rules: Computer Modern serif via LaTeX (mathtext-cm
fallback), sentence-case labels, no overlaps, fits an 8.5x11 page, and a light/dark
theme toggle with transparent saves. Writes PNG + PDF + SVG to figures/.

Run:  python -m scripts.plot_meeting_figures            # light theme
      python -m scripts.plot_meeting_figures --dark      # dark theme
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

# ---- Headline numbers (reproduced 2026-07-06; see docs/session-log/2026-07-06.md §4).
RUNGS = ["CR3BP\n+ Enc J2", "CR3BP\n+ Saturn J2\n+ Enc J2"]
MPC_SURV_D = [30.00, 0.77]      # MPC Strategy 3 survival time (days); 30 = full horizon
POMDP_SURV_D = [1.17, 0.75]     # toy POMDP median survival time (days)
HORIZON_D = 30.0


def _use_serif() -> None:
    """Computer Modern via LaTeX if present, else mathtext-cm fallback."""
    if shutil.which("latex"):
        matplotlib.rcParams.update({"text.usetex": True,
                                    "font.family": "serif",
                                    "font.serif": ["Computer Modern Roman"]})
    else:
        matplotlib.rcParams.update({"text.usetex": False,
                                    "font.family": "serif",
                                    "mathtext.fontset": "cm"})


def _theme(dark: bool) -> dict:
    """Foreground/accent colors that stay legible on either background."""
    fg = "#f2f2f2" if dark else "#1a1a1a"
    return {
        "fg": fg,
        "muted": "#9aa0a6" if dark else "#5f6368",
        "julia": "#9558B2",      # Julia purple
        "python": "#3572A5",     # Python blue
        "good": "#2e8b57",       # holds / survives
        "bad": "#c0392b",        # fails
        "seam": "#e6a817",       # the JSON policy seam
        "grid": "#3a3a3a" if dark else "#d7d7d7",
    }


def _save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight",
                    transparent=True, dpi=200)


def _box(ax, xy, w, h, text, face, t):
    """A rounded node box with centered label."""
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.4, edgecolor=t["fg"], facecolor=face,
                                alpha=0.92, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color="white", fontsize=9.5, zorder=3, linespacing=1.3)


def _arrow(ax, p0, p1, t, label=None, color=None):
    color = color or t["muted"]
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.6, color=color, zorder=1,
                                 shrinkA=2, shrinkB=2))
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + 0.12, label, ha="center", va="bottom",
                color=t["fg"], fontsize=9.5, zorder=3)


def fig_pipeline(dark: bool) -> None:
    t = _theme(dark)
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Julia side (solve)
    _box(ax, (0.3, 4.1), 3.3, 1.2,
         "Julia\nPOMDPs.jl + NativeSARSOP\n(offline solve)", t["julia"], t)
    _box(ax, (0.3, 2.3), 3.3, 1.0,
         "Crude drift + burn\ntoy dynamics\n(builds T, O tables)", t["muted"], t)
    _arrow(ax, (1.95, 3.3), (1.95, 4.1), t)

    # The seam
    _box(ax, (4.6, 4.2), 2.4, 1.0, "policy JSON\n($\\alpha$-vectors)", t["seam"], t)
    _arrow(ax, (3.6, 4.7), (4.6, 4.7), t, label="export once")

    # Python side (evaluate)
    _box(ax, (8.1, 4.1), 3.5, 1.2,
         "Python\nrollout harness\n(evaluate policy)", t["python"], t)
    _arrow(ax, (7.0, 4.7), (8.1, 4.7), t, label="load")

    _box(ax, (8.1, 2.1), 3.5, 1.2,
         "Real CR3BP truth\nEnc J2 / Sat J2 / SPICE\n+ nav \\& thruster noise",
         t["good"], t)
    _arrow(ax, (9.55, 4.1), (9.55, 3.3), t)
    _arrow(ax, (9.55, 3.3), (9.55, 4.1), t)

    # Honest-test callout
    ax.text(6.0, 1.0,
            "The policy is solved on the crude model but scored on the real physics\n"
            "$\\rightarrow$ breaks the circularity, localizes where the gap really is.",
            ha="center", va="center", color=t["fg"], fontsize=9.8,
            style="italic", linespacing=1.4)

    ax.text(6.0, 5.75, "POMDP pipeline: solve in Julia, evaluate on real physics in Python",
            ha="center", va="center", color=t["fg"], fontsize=12.5, weight="bold")

    _save(fig, f"meeting_pipeline{'_dark' if dark else ''}")
    plt.close(fig)


def fig_results(dark: bool) -> None:
    t = _theme(dark)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    x = range(len(RUNGS))
    w = 0.36
    mpc_bars = ax.bar([i - w / 2 for i in x], MPC_SURV_D, w,
                      label="MPC (Strategy 3)", color=t["python"], edgecolor=t["fg"],
                      linewidth=0.8)
    pomdp_bars = ax.bar([i + w / 2 for i in x], POMDP_SURV_D, w,
                        label="Toy POMDP (SARSOP)", color=t["julia"], edgecolor=t["fg"],
                        linewidth=0.8)

    ax.axhline(HORIZON_D, ls="--", lw=1.2, color=t["muted"])
    ax.text(len(RUNGS) - 0.5, HORIZON_D - 1.4, "30-day horizon",
            ha="right", va="top", color=t["muted"], fontsize=9.5)

    # value labels
    for bars, vals in ((mpc_bars, MPC_SURV_D), (pomdp_bars, POMDP_SURV_D)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.4,
                    f"{v:.2f} d" if v < HORIZON_D else "held",
                    ha="center", va="bottom", color=t["fg"], fontsize=9.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(RUNGS, color=t["fg"], fontsize=10.5)
    ax.set_ylabel("Survival time (days)", color=t["fg"], fontsize=11.5)
    ax.set_ylim(0, HORIZON_D * 1.12)
    ax.set_title("Stationkeeping survival by fidelity rung",
                 color=t["fg"], fontsize=12.5, weight="bold", pad=12)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(t["fg"])
    ax.tick_params(colors=t["fg"])
    ax.yaxis.grid(True, color=t["grid"], lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    leg = ax.legend(loc="upper right", frameon=False, fontsize=10.5)
    for txt in leg.get_texts():
        txt.set_color(t["fg"])

    # readable takeaway placed in clear space above the short right-hand bars
    ax.text(0.55, 0.60,
            "Enc J2: orbit is holdable\n(MPC holds 30 d); toy\naction model cannot.\n\n"
            "Saturn J2: instability-\ndominated — both fail.",
            transform=ax.transAxes, ha="left", va="top", color=t["fg"],
            fontsize=9.5, style="italic", linespacing=1.4)

    _save(fig, f"meeting_results{'_dark' if dark else ''}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dark", action="store_true", help="dark-background theme")
    args = ap.parse_args()
    _use_serif()
    fig_pipeline(args.dark)
    fig_results(args.dark)
    theme = "dark" if args.dark else "light"
    print(f"wrote figures/meeting_pipeline* and figures/meeting_results* ({theme} theme)")


if __name__ == "__main__":
    main()
