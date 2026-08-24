"""
plot_pomdp_findings.py — figures for the 2026-07-13 stationkeeping-POMDP findings,
for sharing with collaborators.

Four panels, each a de-risking result from scripts/pomdp_experiments/ (numbers are
reproduced here with provenance; re-run the experiments to regenerate):

  Fig 1 escape    : uncontrolled periapsis walk-out + escape (EncJ2) vs MPC holds.
                    → "the orbit is unstable but holdable" (exp 01).
  Fig 2 direction : 30-day survival under degraded burns — exact / dir-quantized hold,
                    prograde-only / fixed-magnitude fail. → "direction matters, not
                    magnitude precision; the burn is cross-track" (exp 03/04).
  Fig 3 reachable : commanded vs achieved excursion periapsis altitude, single-burn
                    (robust) vs two-burn (precise-but-fragile). → the GP action menu
                    (exp 07/08).
  Fig 4 excursion : periapsis altitude per pass with one-pass excursions every 4
                    passes — holds + recovers (exp 06).

Styling follows the user's global rules: Computer Modern serif via LaTeX (mathtext-cm
fallback), sentence-case labels, no overlaps, fits an 8.5x11 page, light/dark theme
toggle with transparent saves. Writes PNG + PDF + SVG to figures/.

Run:  python -m scripts.plot_pomdp_findings           # light theme
      python -m scripts.plot_pomdp_findings --dark     # dark theme
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

# ── Measured numbers (provenance: scripts/pomdp_experiments/, 2026-07-13) ──────
# Fig 1 — uncontrolled EncJ2 periapsis passes + escape (exp 01).
UNCTRL_DAYS = [0.250, 0.756, 1.285, 1.893]
UNCTRL_PERI = [31.2, 52.0, 113.5, 949.6]
ESCAPE_DAY = 2.274
MPC_HOLD_PERI = 31.0          # MPC-S3 holds periapsis flat at ~31 km for 30 d

# Fig 2 — 30-day survival under degraded burns (exp 03/04). held = full 30 d.
DIR_LABELS = ["Exact\nStrategy 3", "Direction +\nquant. mag.",
              "Prograde\nonly", "Fixed 1 m/s"]
DIR_SURV_D = [30.00, 30.00, 1.74, 0.73]
DIR_HELD = [True, True, False, False]

# Fig 3 — commanded vs achieved excursion periapsis altitude.
CMD = [15, 20, 25, 31, 40, 50, 60, 70, 90, 120]                 # exp 07 commanded
SINGLE_ACH = [32.8, 33.9, 35.3, 36.9, 39.4, 42.4, 45.5, 48.7, 54.5, 65.1]
# Two-burn, damped+capped (exp 08 for 60/90; exp 08b retry for 15/20/120). Target 40
# still escapes even damped (ill-conditioned apoapsis-burn geometry there).
TWO_CMD = [15, 20, 60, 90, 120]
TWO_ACH = [15.4, 21.5, 60.3, 90.3, 120.3]
TWO_FAIL_CMD = [40]                                             # only 40 km still escapes

# Fig 4 — multi-altitude touch-and-go timeline (exp 06b); loaded from npz at runtime.
_EXC_NPZ = FIG_DIR / "excursion_timeline_data.npz"


def _use_serif() -> None:
    """Computer Modern via LaTeX if present, else mathtext-cm fallback."""
    if shutil.which("latex"):
        matplotlib.rcParams.update({"text.usetex": True, "font.family": "serif",
                                    "font.serif": ["Computer Modern Roman"]})
    else:
        matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif",
                                    "mathtext.fontset": "cm"})
    matplotlib.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                                "axes.labelsize": 11, "legend.fontsize": 9.5})


def _theme(dark: bool) -> dict:
    """Foreground/accent colors legible on either background. Categorical hues in a
    FIXED order (dataviz rule); status colors (good/bad) reserved and never reused."""
    fg = "#f2f2f2" if dark else "#1a1a1a"
    return {
        "fg": fg,
        "muted": "#9aa0a6" if dark else "#5f6368",
        "grid": "#3a3a3a" if dark else "#d7d7d7",
        # categorical (identity), fixed order
        "c1": "#3572A5",     # blue   — primary series (uncontrolled / single-burn)
        "c2": "#9558B2",     # purple — second series (MPC / two-burn)
        # status (reserved)
        "good": "#2e8b57",   # holds / survives
        "bad": "#c0392b",    # fails / escapes
        "accent": "#e6a817", # excursion marker / escape flag
    }


def _style_axes(ax, th: dict) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(th["muted"])
    ax.tick_params(colors=th["fg"])
    ax.xaxis.label.set_color(th["fg"]); ax.yaxis.label.set_color(th["fg"])
    ax.title.set_color(th["fg"])
    ax.grid(True, color=th["grid"], lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def _fig1(ax, th):
    """Uncontrolled escape vs MPC hold."""
    ax.plot(UNCTRL_DAYS, UNCTRL_PERI, "-o", color=th["c1"], lw=2, ms=7,
            label="Uncontrolled (Enc J2)")
    ax.annotate("escape", xy=(ESCAPE_DAY, 949.6), xytext=(ESCAPE_DAY - 0.05, 1400),
                color=th["bad"], ha="right", fontsize=10,
                arrowprops=dict(arrowstyle="->", color=th["bad"], lw=1.5))
    ax.axhline(MPC_HOLD_PERI, color=th["good"], lw=2, ls="--",
               label="MPC Strategy 3 (holds 30 d)")
    ax.axhspan(0, 5, color=th["bad"], alpha=0.15)
    ax.text(0.05, 6.5, r"crash $<$ 5 km", color=th["bad"], fontsize=8.5)
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Periapsis altitude (km)")
    ax.set_title("(a) Orbit is unstable but holdable")
    ax.set_yscale("log"); ax.set_ylim(3, 2500); ax.set_xlim(0, 2.5)
    ax.legend(loc="lower right", frameon=False, labelcolor=th["fg"])


def _fig2(ax, th):
    """Direction matters — 30-day survival under degraded burns."""
    x = np.arange(len(DIR_LABELS))
    colors = [th["good"] if h else th["bad"] for h in DIR_HELD]
    bars = ax.bar(x, DIR_SURV_D, color=colors, width=0.62)
    ax.axhline(30, color=th["muted"], lw=1, ls=":")
    ax.text(len(x) - 0.5, 30.6, "30-day horizon", color=th["muted"],
            ha="right", fontsize=8.5)
    for xi, (v, h) in enumerate(zip(DIR_SURV_D, DIR_HELD)):
        txt = "held" if h else f"{v:.2f} d"
        ax.text(xi, v + 0.7, txt, ha="center", color=th["fg"], fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(DIR_LABELS)
    ax.set_ylabel("Survival time (days)")
    ax.set_title("(b) Direction matters, magnitude precision does not")
    ax.set_ylim(0, 33)


def _fig3(ax, th):
    """Reachable excursion menu: commanded vs achieved."""
    ax.plot([10, 125], [10, 125], color=th["muted"], lw=1, ls=":",
            label="Ideal (achieved = commanded)")
    ax.plot(CMD, SINGLE_ACH, "-o", color=th["c1"], lw=2, ms=6,
            label="Single burn (robust)")
    ax.plot(TWO_CMD, TWO_ACH, "s", color=th["c2"], ms=9,
            label="Two-burn (precise, damped)")
    ax.plot(TWO_FAIL_CMD, [11] * len(TWO_FAIL_CMD), "x", color=th["bad"], ms=9,
            mew=2, label="Two-burn escaped (40 km)")
    ax.set_xlabel("Commanded periapsis altitude (km)")
    ax.set_ylabel("Achieved periapsis altitude (km)")
    ax.set_title("(c) Reachable excursion menu")
    ax.set_xlim(10, 125); ax.set_ylim(10, 125)
    ax.legend(loc="upper left", frameon=False, labelcolor=th["fg"])


def _fig4(ax, th):
    """Multi-altitude touch-and-go timeline — visits a range of altitudes + recovers."""
    if not _EXC_NPZ.exists():
        raise SystemExit(f"missing {_EXC_NPZ}; run "
                         "scripts/pomdp_experiments/06b_multi_altitude_timeline.py")
    d = np.load(_EXC_NPZ)
    days, peri, is_exc = d["days"], d["peri"], d["is_exc"].astype(bool)
    ax.plot(days, peri, "-", color=th["muted"], lw=1.1, zorder=1)
    ax.plot(days[~is_exc], peri[~is_exc], "o", color=th["c1"], ms=6,
            label="Hold pass (nominal)", zorder=2)
    ax.plot(days[is_exc], peri[is_exc], "D", color=th["accent"], ms=9,
            label="Excursion pass", zorder=3)
    # Label the achieved altitude on each excursion so the RANGE is visible.
    for x, y in zip(days[is_exc], peri[is_exc]):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=th["fg"])
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Periapsis altitude (km)")
    ax.set_title("(d) Touch-and-go over a range of altitudes")
    ax.set_ylim(min(peri) - 6, max(peri) + 10)
    ax.legend(loc="lower right", frameon=False, labelcolor=th["fg"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dark", action="store_true", help="dark theme")
    args = ap.parse_args()

    _use_serif()
    th = _theme(args.dark)
    # Make ALL default text (tick labels, suptitle) legible on the target background.
    matplotlib.rcParams.update({
        "text.color": th["fg"], "axes.labelcolor": th["fg"],
        "xtick.color": th["fg"], "ytick.color": th["fg"],
        "axes.edgecolor": th["muted"],
    })
    FIG_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 9.2))
    for ax in axes.flat:
        _style_axes(ax, th)
    _fig1(axes[0, 0], th)
    _fig2(axes[0, 1], th)
    _fig3(axes[1, 0], th)
    _fig4(axes[1, 1], th)
    fig.suptitle("Stationkeeping-POMDP de-risking findings (Enc J2 truth)",
                 color=th["fg"], fontsize=13.5, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    stem = "pomdp_findings" + ("_dark" if args.dark else "")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", transparent=True,
                    bbox_inches="tight", dpi=200)
    print(f"wrote {FIG_DIR/stem}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
