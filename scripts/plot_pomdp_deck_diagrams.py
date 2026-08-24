"""
plot_pomdp_deck_diagrams.py — schematic diagrams for the science+safety POMDP deck.

Two figures (no data — schematic, drawn with matplotlib patches):
  1. pipeline  : the Julia (solve) -> policy JSON -> Python (rollout vs real truth) flow,
                 showing where the MEASURED physics enters and where solve_burn lives.
  2. statespace: the (dev x cov) state grid + the action set + reward, as a labelled map.

Styling: Computer Modern serif (LaTeX or mathtext-cm fallback), sentence-case labels,
no overlaps, fits an 8.5x11 slide. Saves PNG + PDF + SVG to figures/. (Light theme only;
black-background toggle skipped per request 2026-07-15.)

Run:  python -m scripts.plot_pomdp_deck_diagrams
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

# Palette (sequential/categorical, legible on white; brand-neutral).
C = {
    "julia": "#9558B2", "python": "#3572A5", "physics": "#2e8b57",
    "seam": "#e6a817", "ink": "#1a1a1a", "muted": "#5f6368",
    "ok": "#2e8b57", "drift": "#e6a817", "far": "#d98a2b", "term": "#c0392b",
    "box": "#eef1f5", "sci": "#3572A5",
}


def _use_serif() -> None:
    if shutil.which("latex"):
        matplotlib.rcParams.update({"text.usetex": True, "font.family": "serif",
                                    "font.serif": ["Computer Modern Roman"]})
    else:
        matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif",
                                    "mathtext.fontset": "cm"})
    matplotlib.rcParams.update({"font.size": 12})


def _box(ax, x, y, w, h, text, fc, ec=None, fs=11, tc="#1a1a1a"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                 facecolor=fc, edgecolor=ec or C["muted"], lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def _arrow(ax, x0, y0, x1, y1, color, lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                 mutation_scale=16, color=color, lw=lw))


# ── Figure 1: pipeline ───────────────────────────────────────────────────────────
def fig_pipeline() -> None:
    _use_serif()
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    ax.text(2.5, 5.7, "Offline — solve (Julia)", ha="center", fontsize=12.5, color=C["julia"])
    ax.text(7.5, 5.7, "Rollout — test vs real truth (Python)", ha="center", fontsize=12.5, color=C["python"])
    ax.plot([5, 5], [0.3, 5.4], color=C["muted"], lw=0.8, ls=":")

    # Julia side
    _box(ax, 0.4, 4.2, 4.2, 0.8,
         "Measured CR3BP+EncJ2 experiments\n(exps 01/03/06/07/11b/12)", C["box"], C["physics"], 10)
    _box(ax, 0.9, 3.0, 3.2, 0.8, "dynamics.jl\n(dev,cov) T / O tables", C["box"], C["julia"], 10)
    _box(ax, 0.9, 1.8, 3.2, 0.8, "NativeSARSOP solve\n(3.0 s)", C["box"], C["julia"], 10)
    _arrow(ax, 2.5, 4.2, 2.5, 3.8, C["physics"])
    _arrow(ax, 2.5, 3.0, 2.5, 2.6, C["julia"])

    # Seam
    _box(ax, 3.6, 0.5, 2.8, 0.8, "policy JSON\n(alpha vectors + tables)", "#fff6e0", C["seam"], 10)
    _arrow(ax, 2.5, 1.8, 4.2, 1.3, C["seam"])
    _arrow(ax, 6.4, 0.9, 7.5, 1.8, C["seam"])
    ax.text(5.0, 0.28, "crosses the language boundary once, as a file",
            ha="center", fontsize=8.5, color=C["muted"])

    # Python side
    _box(ax, 5.6, 4.2, 4.0, 0.8,
         "Real truth EOM\ncr3bp_j2 / +SaturnJ2 (integrator)", C["box"], C["physics"], 10)
    _box(ax, 5.9, 3.0, 3.4, 0.8, "mpc.solve_burn (actuator)\ndirection solved live", C["box"], C["python"], 10)
    _box(ax, 5.9, 1.8, 3.4, 0.8, "pomdp_rollout.py\nbelief + action per pass", C["box"], C["python"], 10)
    _arrow(ax, 7.6, 4.2, 7.6, 3.8, C["physics"])
    _arrow(ax, 7.6, 3.0, 7.6, 2.6, C["python"])

    ax.text(5.0, 5.15, "SARSOP never calls solve_burn — it only sees the discrete tables",
            ha="center", fontsize=8.5, color=C["muted"], style="italic")

    fig.suptitle("Pipeline: measured physics $\\rightarrow$ SARSOP policy $\\rightarrow$ rollout vs real truth",
                 fontsize=13.5, y=0.99, color=C["ink"])
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"deck_pipeline.{ext}", transparent=True, bbox_inches="tight", dpi=200)
    print(f"wrote {FIG_DIR}/deck_pipeline.{{png,pdf,svg}}")


# ── Figure 2: state + action space ────────────────────────────────────────────────
def fig_statespace() -> None:
    _use_serif()
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")

    # dev (rows, safety) x cov (cols, science). 3 non-terminal dev x 8 cov.
    devs = [("OK", C["ok"]), ("DRIFT", C["drift"]), ("FAR", C["far"])]
    x0, y0, cw, ch = 1.7, 1.9, 0.86, 0.86
    # Vertical axis label, clear to the LEFT of the per-row dev labels.
    ax.text(0.35, y0 + 1.5 * ch, "dev (safety)", ha="center", va="center",
            fontsize=11, color=C["muted"], rotation=90)
    ax.text(x0 + 4 * cw, y0 + 3 * ch + 0.35, "cov = which of 3 science bands sampled (0..7)",
            ha="center", fontsize=10.5, color=C["muted"])
    for r, (dname, dcol) in enumerate(devs):
        yy = y0 + (2 - r) * ch
        ax.text(x0 - 0.15, yy + ch / 2, dname, ha="right", va="center", fontsize=10.5, color=dcol)
        for c in range(8):
            ax.add_patch(Rectangle((x0 + c * cw, yy), cw - 0.06, ch - 0.06,
                         facecolor=C["box"], edgecolor=C["muted"], lw=0.8))
            ax.text(x0 + c * cw + (cw - 0.06) / 2, yy + (ch - 0.06) / 2, f"{c:03b}",
                    ha="center", va="center", fontsize=7.5, color=C["muted"])
    ax.text(x0 + 4 * cw, y0 - 0.5, "coverage mask  (000 = none  ..  111 = all 3 bands)",
            ha="center", fontsize=9, color=C["muted"])

    # terminal states
    _box(ax, x0 + 8 * cw + 0.2, y0 + 1.15 * ch, 1.5, 0.7, "LOST /\nCRASHED\n(terminal)",
         "#fbe9e7", C["term"], 9, C["term"])

    ax.text(x0 + 4 * cw, y0 + 3 * ch + 0.9,
            r"State $= (\mathrm{dev},\ \mathrm{cov})$ : 3 dev $\times$ 8 cov $+$ 2 terminal $= 26$",
            ha="center", fontsize=12, color=C["ink"])

    # Actions band
    ay = 0.55
    acts = [("OBSERVE\n0 m/s", C["muted"]), ("CORRECT\n1.3 m/s", C["ok"]),
            ("EXCURSE\\_LOW\n2.1 m/s", C["sci"]), ("EXCURSE\\_MID\n3.1 m/s", C["sci"]),
            ("EXCURSE\\_HIGH\n9.9 m/s", C["sci"])]
    aw = 1.7
    ax.text(0.6, ay + 0.28, "actions", ha="center", va="center", fontsize=11, color=C["muted"])
    for i, (label, col) in enumerate(acts):
        _box(ax, 1.4 + i * aw, ay, aw - 0.12, 0.62, label, C["box"], col, 8.5)

    fig.suptitle("State, actions, and cost (measured $\\Delta V$)", fontsize=13.5, y=0.99, color=C["ink"])
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"deck_statespace.{ext}", transparent=True, bbox_inches="tight", dpi=200)
    print(f"wrote {FIG_DIR}/deck_statespace.{{png,pdf,svg}}")


if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)
    fig_pipeline()
    fig_statespace()
