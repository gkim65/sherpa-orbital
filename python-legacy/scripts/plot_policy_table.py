"""
plot_policy_table.py — the LEARNED SARSOP policy as a slide-ready heatmap.

Reads policy/sarsop_policy.json and renders the greedy action a*(dev, cov) over the
full state grid: rows = dev safety bin (OK/DRIFT/FAR), columns = science coverage mask
(which of LOW/MID/HIGH already sampled). Each cell is coloured by the action and
labelled. Shows the science-vs-safety tradeoff at a glance:
  - dev=OK + science left  -> EXCURSE (go sample an unsampled band)
  - dev=OK + all sampled   -> CORRECT (hold)
  - dev=DRIFT / FAR        -> CORRECT (safety first, always)

Styling matches the deck (Computer Modern serif, sentence-case). PNG+PDF+SVG.

Run:  python -m scripts.plot_policy_table
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
POLICY = FIG_DIR.parent / "policy" / "sarsop_policy.json"

# One reserved colour per action (categorical, fixed order).
ACT_COLOR = {
    "OBSERVE": "#9aa0a6",       # grey — never chosen here
    "CORRECT": "#2e8b57",       # green — hold / safety
    "EXCURSE_LOW": "#8fb8de",   # light blue
    "EXCURSE_MID": "#3572A5",   # mid blue
    "EXCURSE_HIGH": "#9558B2",  # purple
}
ACT_SHORT = {"OBSERVE": "OBS", "CORRECT": "CORRECT", "EXCURSE_LOW": "EXC\nLOW",
             "EXCURSE_MID": "EXC\nMID", "EXCURSE_HIGH": "EXC\nHIGH"}
DEVS = ["OK", "DRIFT", "FAR"]
BANDS = ["LOW", "MID", "HIGH"]


def _use_serif():
    if shutil.which("latex"):
        matplotlib.rcParams.update({"text.usetex": True, "font.family": "serif",
                                    "font.serif": ["Computer Modern Roman"]})
    else:
        matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif",
                                    "mathtext.fontset": "cm"})
    matplotlib.rcParams.update({"font.size": 12})


def _cov_label(mask: int) -> str:
    if mask == 0:
        return "none"
    return "+".join(b[0] for i, b in enumerate(BANDS) if mask & (1 << i))


def main():
    _use_serif()
    with open(POLICY) as f:
        d = json.load(f)
    states, actions = d["states"], d["actions"]
    alphas = np.array(d["alphas"]); aidx = np.array(d["alpha_actions"]) - 1
    sidx = {s: i for i, s in enumerate(states)}

    def act(dev, cov):
        i = sidx[f"{dev}|cov{cov}"]
        b = np.zeros(len(states)); b[i] = 1.0
        return actions[aidx[int(np.argmax(alphas @ b))]]

    grid = [[act(dev, c) for c in range(8)] for dev in DEVS]

    fig, ax = plt.subplots(figsize=(10.0, 3.7))
    ax.set_xlim(0, 8); ax.set_ylim(0, 3); ax.axis("off")
    for r, dev in enumerate(DEVS):
        yy = 2 - r
        for c in range(8):
            a = grid[r][c]
            ax.add_patch(plt.Rectangle((c + 0.03, yy + 0.03), 0.94, 0.94,
                         facecolor=ACT_COLOR[a], edgecolor="white", lw=1.5))
            txt = ACT_SHORT[a]
            # white text on the darker cells, dark on light
            light = a in ("OBSERVE", "EXCURSE_LOW")
            ax.text(c + 0.5, yy + 0.5, txt, ha="center", va="center",
                    fontsize=8.5, color=("#1a1a1a" if light else "white"))
    # row labels (dev)
    for r, dev in enumerate(DEVS):
        ax.text(-0.12, (2 - r) + 0.5, dev, ha="right", va="center", fontsize=11)
    ax.text(-0.95, 1.5, "dev (safety)", ha="center", va="center", fontsize=11,
            rotation=90, color="#5f6368")
    # column labels (cov)
    for c in range(8):
        ax.text(c + 0.5, 3.12, _cov_label(c), ha="center", va="bottom", fontsize=9)
    ax.text(4, 3.55, "science coverage — bands already sampled "
            "(L=low, M=mid, H=high)", ha="center", fontsize=10.5, color="#5f6368")

    # legend of actions that actually appear
    used = sorted({a for row in grid for a in row},
                  key=lambda a: list(ACT_COLOR).index(a))
    handles = [Patch(facecolor=ACT_COLOR[a], edgecolor="white",
                     label=a.replace("_", " ")) for a in used]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.22),
              ncol=len(handles), frameon=False, fontsize=9.5)

    fig.suptitle("Learned SARSOP policy: greedy action $a^*(\\mathrm{dev},\\,"
                 "\\mathrm{cov})$", fontsize=13.5, y=1.05)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.98])
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"policy_table.{ext}", transparent=True,
                    bbox_inches="tight", dpi=200)
    print(f"wrote {FIG_DIR}/policy_table.{{png,pdf,svg}}")
    # also print a plain-text table for the outline
    print("\n" + " " * 7 + "  ".join(f"{_cov_label(c):>6}" for c in range(8)))
    for r, dev in enumerate(DEVS):
        print(f"{dev:6s} " + "  ".join(f"{grid[r][c].replace('EXCURSE_','EXC-'):>6}"
              for c in range(8)))


if __name__ == "__main__":
    main()
