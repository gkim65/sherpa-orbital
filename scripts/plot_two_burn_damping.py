"""
plot_two_burn_damping.py — the two-burn damping/cap story as a heatmap.

Reads figures/two_burn_damping_data.npz (from
scripts/pomdp_experiments/10_two_burn_damping_sweep.py) and shows, for each escaped-
in-exp-08 excursion target, the achieved periapsis altitude as a function of the
per-burn magnitude cap. Cells that ESCAPED/crashed are hatched and labeled. The point:
capping/damping the apoapsis burn recovers most targets, but the working cap DIFFERS
per target (no single setting), and 40 km stays problematic — two-burn excursions need
per-target tuning, they are not a free precise controller.

Sequential colormap = one hue light->dark for the achieved-altitude magnitude
(dataviz rule). Escapes are a reserved status treatment (hatch + label), never a hue.

Run:  python -m scripts.plot_two_burn_damping [--dark]
"""
from __future__ import annotations
import argparse, shutil
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
DATA = FIG_DIR / "two_burn_damping_data.npz"


def _use_serif():
    if shutil.which("latex"):
        matplotlib.rcParams.update({"text.usetex": True, "font.family": "serif",
                                    "font.serif": ["Computer Modern Roman"]})
    else:
        matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif",
                                    "mathtext.fontset": "cm"})
    matplotlib.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                                "axes.labelsize": 11})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dark", action="store_true")
    args = ap.parse_args()
    if not DATA.exists():
        raise SystemExit(f"missing {DATA}; run "
                         "scripts/pomdp_experiments/10_two_burn_damping_sweep.py")
    d = np.load(DATA, allow_pickle=True)
    targets = d["targets"]; caps = d["caps"]; ach = d["achieved"]; out = d["outcome"]
    fg = "#f2f2f2" if args.dark else "#1a1a1a"
    muted = "#9aa0a6" if args.dark else "#5f6368"
    bad = "#c0392b"
    _use_serif()
    matplotlib.rcParams.update({"text.color": fg, "axes.labelcolor": fg,
                                "xtick.color": fg, "ytick.color": fg,
                                "axes.edgecolor": muted})

    esc = np.isin(out, ["escape", "crash"]) | ~np.isfinite(ach)
    masked = np.ma.masked_where(esc, ach)
    cmap = matplotlib.colormaps["viridis"].copy()

    fig, axx = plt.subplots(figsize=(8.0, 4.6))
    im = axx.imshow(masked, aspect="auto", cmap=cmap, origin="lower")
    axx.set_xticks(range(len(caps)))
    cap_lbls = [("uncapped" if c > 1e6 else f"{c:.0f}") for c in caps]
    axx.set_xticklabels(cap_lbls)
    axx.set_yticks(range(len(targets)))
    axx.set_yticklabels([f"{t:.0f}" for t in targets])
    axx.set_xlabel("Per-burn magnitude cap (m/s)")
    axx.set_ylabel("Commanded periapsis altitude (km)")
    axx.set_title("(a) Two-burn excursion: achieved altitude vs burn cap")
    # cell annotations
    for i in range(len(targets)):
        for j in range(len(caps)):
            if esc[i, j]:
                axx.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True,
                              facecolor=bad, alpha=0.30, hatch="///",
                              edgecolor=bad, lw=0))
                axx.text(j, i, "escape", ha="center", va="center", color=bad,
                         fontsize=7.5)
            else:
                axx.text(j, i, f"{ach[i, j]:.0f}", ha="center", va="center",
                         color="white", fontsize=8.5)
    cb = fig.colorbar(im, ax=axx, fraction=0.046, pad=0.03)
    cb.set_label("Achieved periapsis altitude (km)", color=fg)
    cb.ax.yaxis.set_tick_params(color=fg)
    plt.setp(cb.ax.get_yticklabels(), color=fg)
    fig.suptitle("Damping/capping the apoapsis burn recovers most (not all) targets",
                 color=fg, fontsize=12.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    stem = "two_burn_damping" + ("_dark" if args.dark else "")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", transparent=True,
                    bbox_inches="tight", dpi=200)
    print(f"wrote {FIG_DIR/stem}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
