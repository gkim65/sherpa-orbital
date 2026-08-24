"""
plot_burn_geometry.py — WHERE the stationkeeping burns happen and WHICH direction
the ΔV points, relative to the orbit around Enceladus.

Reads figures/burn_geometry_data.npz (written by
scripts/pomdp_experiments/09_burn_geometry.py) and draws:

  (a) The controlled orbit around Enceladus (X–Z, Enceladus-relative, to scale) with
      the 600-km control shell, the burn locations marked, and each ΔV drawn as an
      arrow (magnified) so the OUT-OF-PLANE / cross-track direction is visible.
  (b) Per-burn ΔV decomposition into prograde / in-plane cross-track / orbit-normal
      components, showing the prograde component is ~0 — the holding burn is NOT a
      speed change, it is an out-of-plane correction.

Styling: Computer Modern serif (LaTeX or mathtext-cm fallback), sentence-case labels,
light/dark toggle, transparent PDF+SVG+PNG saved to figures/.

Run:  python -m scripts.plot_burn_geometry            # light
      python -m scripts.plot_burn_geometry --dark      # dark
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
DATA = FIG_DIR / "burn_geometry_data.npz"


def _use_serif() -> None:
    if shutil.which("latex"):
        matplotlib.rcParams.update({"text.usetex": True, "font.family": "serif",
                                    "font.serif": ["Computer Modern Roman"]})
    else:
        matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif",
                                    "mathtext.fontset": "cm"})
    matplotlib.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                                "axes.labelsize": 11, "legend.fontsize": 9.5})


def _theme(dark: bool) -> dict:
    fg = "#f2f2f2" if dark else "#1a1a1a"
    return {"fg": fg, "muted": "#9aa0a6" if dark else "#5f6368",
            "grid": "#3a3a3a" if dark else "#d7d7d7",
            "orbit": "#3572A5", "body": "#8a8f98" if dark else "#c4c9d0",
            "shell": "#e6a817", "dv": "#c0392b",
            "prograde": "#3572A5", "cross": "#2e8b57", "normal": "#9558B2"}


def _style(ax, th):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(th["muted"])
    ax.tick_params(colors=th["fg"])
    ax.xaxis.label.set_color(th["fg"]); ax.yaxis.label.set_color(th["fg"])
    ax.title.set_color(th["fg"])
    ax.grid(True, color=th["grid"], lw=0.6, alpha=0.6); ax.set_axisbelow(True)


def _panel_orbit(ax, d, th):
    traj = d["traj"]; bp = d["burn_pos"]; bdv = d["burn_dv"]; Renc = float(d["r_enceladus"])
    # X-Z projection (Enceladus-relative km). x index 0, z index 2.
    ax.plot(traj[:, 0], traj[:, 2], color=th["orbit"], lw=1.1, alpha=0.9,
            label="Controlled orbit")
    ax.add_patch(Circle((0, 0), Renc, facecolor=th["body"], edgecolor=th["muted"],
                        lw=1.0, zorder=1, label=f"Enceladus ($R={Renc:.0f}$ km)"))
    ax.add_patch(Circle((0, 0), Renc + 600.0, facecolor="none", edgecolor=th["shell"],
                        lw=1.2, ls="--", zorder=1, label="600 km control shell"))
    # ΔV arrows (magnify km/s so they are visible against the ~1000 km orbit).
    scale = 6.0e5  # km per (km/s)
    for i, (p, dv) in enumerate(zip(bp, bdv)):
        ax.annotate("", xy=(p[0] + dv[0] * scale, p[2] + dv[2] * scale),
                    xytext=(p[0], p[2]),
                    arrowprops=dict(arrowstyle="-|>", color=th["dv"], lw=1.8))
    ax.scatter(bp[:, 0], bp[:, 2], s=32, color=th["dv"], zorder=5,
               label=r"Burn ($\Delta V$, magnified)")
    ax.set_aspect("equal")
    ax.set_xlabel("x, Enceladus-relative (km)")
    ax.set_ylabel("z, Enceladus-relative (km)")
    ax.set_title("(a) Where the burns fire")
    ax.legend(loc="upper left", frameon=False, labelcolor=th["fg"], fontsize=8.5)


def _panel_decomp(ax, d, th):
    prog = d["prograde"]; cross = d["cross"]; norm = d["normal"]
    n = len(prog); x = np.arange(n); w = 0.26
    ax.bar(x - w, prog, w, color=th["prograde"], label="Prograde (along $v$)")
    ax.bar(x, cross, w, color=th["cross"], label="In-plane cross-track")
    ax.bar(x + w, norm, w, color=th["normal"], label="Orbit-normal")
    ax.axhline(0, color=th["muted"], lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{i+1}" for i in range(n)])
    ax.set_xlabel("Burn number")
    ax.set_ylabel(r"$\Delta V$ component (m/s)")
    ax.set_title("(b) The burn is out-of-plane, not prograde")
    ax.legend(loc="upper right", frameon=False, labelcolor=th["fg"], fontsize=9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args()
    if not DATA.exists():
        raise SystemExit(f"missing {DATA}; run "
                         "scripts/pomdp_experiments/09_burn_geometry.py first")
    d = np.load(DATA)
    _use_serif(); th = _theme(args.dark)
    matplotlib.rcParams.update({"text.color": th["fg"], "axes.labelcolor": th["fg"],
                                "xtick.color": th["fg"], "ytick.color": th["fg"],
                                "axes.edgecolor": th["muted"]})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    for a in axes:
        _style(a, th)
    _panel_orbit(axes[0], d, th)
    _panel_decomp(axes[1], d, th)
    fig.suptitle("Stationkeeping burn geometry (MPC Strategy 3, Enc J2)",
                 color=th["fg"], fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    stem = "burn_geometry" + ("_dark" if args.dark else "")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", transparent=True,
                    bbox_inches="tight", dpi=200)
    print(f"wrote {FIG_DIR/stem}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
