"""
plot_rollout_trace.py — visualize a science+safety POMDP rollout vs real CR3BP+EncJ2.

Reads figures/rollout_trace_data.npz (from scripts/pomdp_experiments/13_rollout_trace.py)
and makes two figures:

  A) trajectory : the actual orbit around Enceladus (x-z, Enceladus-relative, to scale)
     with the 600-km control shell, each MANEUVER location marked + coloured by action
     (CORRECT vs EXCURSE_*), the ΔV vector drawn (magnified), and periapsis passes shown.
     "Where did we position ourselves, where did we fire, what maneuver was computed."

  B) deviation : per-pass TRUE vs NOISY-OBSERVED apse deviation over time (how real the
     observation was), the dev bin edges, the ΔV magnitude per pass, and the action taken
     — the safety story + the maneuvers the SARSOP policy produced.

Styling: Computer Modern serif, sentence-case labels, no overlaps, fits a slide.
PNG + PDF + SVG to figures/. (Light theme; black toggle skipped per request.)

Run:  python -m scripts.plot_rollout_trace
"""
from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
DATA = FIG_DIR / "rollout_trace_data.npz"

C = {"ink": "#1a1a1a", "muted": "#5f6368", "grid": "#d7d7d7",
     "orbit": "#3572A5", "body": "#c4c9d0", "shell": "#e6a817",
     "correct": "#2e8b57", "excurse": "#9558B2", "peri": "#c0392b",
     "true": "#3572A5", "obs": "#e6a817"}


def _use_serif():
    if shutil.which("latex"):
        matplotlib.rcParams.update({"text.usetex": True, "font.family": "serif",
                                    "font.serif": ["Computer Modern Roman"]})
    else:
        matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif",
                                    "mathtext.fontset": "cm"})
    matplotlib.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                                "axes.labelsize": 11, "legend.fontsize": 9})


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["muted"])
    ax.grid(True, color=C["grid"], lw=0.6, alpha=0.6); ax.set_axisbelow(True)


def _is_excurse(a):
    return a.startswith("EXCURSE")


def fig_trajectory(d):
    _use_serif()
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    _style(ax)
    Renc = float(d["r_enceladus"]); ctrl = float(d["control_alt"])
    traj = d["traj"]                          # (N,3) enc-relative
    ax.plot(traj[:, 0], traj[:, 2], color=C["orbit"], lw=0.8, alpha=0.8, zorder=2,
            label="Trajectory (Enc J2 truth)")
    ax.add_patch(Circle((0, 0), Renc, facecolor=C["body"], edgecolor=C["muted"],
                        lw=1.0, zorder=1, label=f"Enceladus ($R={Renc:.0f}$ km)"))
    ax.add_patch(Circle((0, 0), Renc + ctrl, facecolor="none", edgecolor=C["shell"],
                        lw=1.1, ls="--", zorder=1, label=f"{ctrl:.0f} km control shell"))

    bp = d["burn_pos"]; bdv = d["burn_dv"]; bact = d["burn_action"]
    scale = 2.5e5
    # separate CORRECT vs EXCURSE for colour/legend
    seen = set()
    for p, dv, a in zip(bp, bdv, bact):
        col = C["excurse"] if _is_excurse(str(a)) else C["correct"]
        lab = ("Excurse burn" if _is_excurse(str(a)) else "Correct burn")
        ax.add_patch(FancyArrowPatch((p[0], p[2]), (p[0] + dv[0]*scale, p[2] + dv[2]*scale),
                     arrowstyle="-|>", mutation_scale=12, color=col, lw=1.6, zorder=5))
        ax.scatter([p[0]], [p[2]], s=34, color=col, zorder=6,
                   label=lab if lab not in seen else None)
        seen.add(lab)
    pp = d["peri_pos"]
    ax.scatter(pp[:, 0], pp[:, 2], s=16, marker="x", color=C["peri"], zorder=5,
               label="Periapsis pass")
    ax.set_aspect("equal")
    ax.set_xlabel("x, Enceladus-relative (km)")
    ax.set_ylabel("z, Enceladus-relative (km)")
    ax.set_title(r"(a) Rollout trajectory + maneuvers "
                 r"($\Delta V$ arrows $\times 2.5{\times}10^5$, x--z projection)")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.tight_layout()
    _save(fig, "rollout_trajectory")


def fig_deviation(d):
    _use_serif()
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    for a in axes:
        _style(a)
    t = d["peri_t"] / 86400.0
    dev_true = d["dev_true"]; dev_obs = d["dev_obs"]; edges = d["dev_edges"]

    # top: true vs observed deviation + bin edges
    ax = axes[0]
    ymax = max(45.0, float(max(dev_true.max(), dev_obs.max())) * 1.25)
    ax.plot(t, dev_true, "-o", color=C["true"], ms=5, lw=1.6, label="True deviation")
    ax.plot(t, dev_obs, "s", color=C["obs"], ms=6, label="Observed (nav-noisy)")
    for ti, (yt, yo) in enumerate(zip(dev_true, dev_obs)):
        ax.plot([t[ti], t[ti]], [yt, yo], color=C["muted"], lw=0.6, alpha=0.6)
    # Only annotate bin edges that fall INSIDE the visible y-range (no floating labels).
    for e, name in zip(edges, ["OK / DRIFT", "DRIFT / FAR", "FAR / LOST"]):
        if e <= ymax:
            ax.axhline(e, color=C["muted"], lw=0.7, ls=":")
            ax.text(t[0], e + 0.6, f"{name} ({e:.0f} km)", va="bottom", ha="left",
                    fontsize=7.5, color=C["muted"])
    ax.set_ylabel("Apse-position deviation (km)")
    ax.set_title("(b) True vs observed deviation, and the maneuvers taken")
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper right", frameon=False)

    # bottom: ΔV per pass coloured by action
    ax2 = axes[1]
    bt = d["burn_t"] / 86400.0; bms = d["burn_dv_ms"]; bact = d["burn_action"]
    cols = [C["excurse"] if _is_excurse(str(a)) else C["correct"] for a in bact]
    ax2.bar(bt, bms, width=0.12, color=cols)
    for ti, a in zip(bt, bact):
        short = str(a).replace("EXCURSE_", "EXC-").replace("CORRECT", "COR").replace("OBSERVE", "OBS")
        ax2.text(ti, 0.3, short, rotation=90, ha="center", va="bottom",
                 fontsize=6.5, color=C["ink"])
    ax2.set_ylabel(r"$\Delta V$ (m/s)")
    ax2.set_xlabel("Time (days)")
    # legend proxy
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(color=C["correct"], label="Correct (hold)"),
                        Patch(color=C["excurse"], label="Excurse (science)")],
               loc="upper right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    _save(fig, "rollout_deviation")


def _save(fig, stem):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", transparent=True, bbox_inches="tight", dpi=200)
    print(f"wrote {FIG_DIR}/{stem}.{{png,pdf,svg}}")


if __name__ == "__main__":
    if not DATA.exists():
        raise SystemExit(f"missing {DATA}; run scripts/pomdp_experiments/13_rollout_trace.py")
    d = np.load(DATA, allow_pickle=True)
    fig_trajectory(d)
    fig_deviation(d)
