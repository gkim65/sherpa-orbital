"""
plot_rollout_3d.py — 3D view of a flown rollout around Enceladus, coloured by the maneuver
objective the POMDP chose on each pass.

    uv run python scripts/plot_rollout_3d.py                 # one panel per sigma, full orbit
    uv run python scripts/plot_rollout_3d.py --zoom          # cropped to the close approach
    uv run python scripts/plot_rollout_3d.py --sigma 0.0 0.1 # pick levels

Reads figures/arcs.npz, written by:

    julia --project=experiments experiments/export_arcs.jl 0.0,0.05,0.1

Julia computes and Python draws this one figure. CairoMakie has no depth buffer, so a
surface cannot occlude a line behind it whatever the geometry says, and any alpha below 1 on
a surface emits a PDF soft mask per quad — measured at 2204 masks and 2.3 MB for one moon.
mplot3d z-sorts polygons and emits a single graphics state, which is why the trajectory view
belongs here. Everything 2D stays in Julia.

Styling follows the project conventions: Computer Modern serif via LaTeX when available with
a mathtext-cm fallback, sentence-case labels, transparent PDF/SVG/PNG.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "figures"
DATA = FIG_DIR / "arcs.npz"

# One colour per maneuver objective, matching the Julia action-band figure so a reader maps
# a panel to a band row by colour alone.
ACTION_COLOR = {
    "INITIAL": "#9e9e9e",
    "CORRECT": "#9e9e9e",
    "EXCURSE_LOW": "#4477aa",
    "EXCURSE_MID": "#228833",
    "EXCURSE_HIGH": "#ee7733",
}
ACTION_LABEL = {
    "CORRECT": "Correct (37 km)",
    "EXCURSE_LOW": "Low (24 km)",
    "EXCURSE_MID": "Mid (30 km)",
    "EXCURSE_HIGH": "High (46 km)",
}
BODY = "#c4c9d0"


def use_serif() -> None:
    """Computer Modern serif, via LaTeX when it is on PATH and mathtext-cm otherwise."""
    if shutil.which("latex"):
        matplotlib.rcParams.update(
            {"text.usetex": True, "font.family": "serif",
             "font.serif": ["Computer Modern Roman"]}
        )
    else:
        matplotlib.rcParams.update(
            {"text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm"}
        )
    matplotlib.rcParams.update({"font.size": 11})


def sphere(ax, radius: float, color: str = BODY, alpha: float = 0.55) -> None:
    """Enceladus, as a shaded translucent sphere.

    mplot3d sorts the quads by depth, so the far side of the trajectory is occluded and the
    near side is not — which is the reading we want and the reason this figure is not drawn
    in Julia.
    """
    # Fine enough that the limb reads as smooth at a cropped zoom. mplot3d emits one
    # graphics state for the whole surface, so quad count costs little here — unlike
    # CairoMakie, which emits a soft mask per quad.
    u = np.linspace(0, 2 * np.pi, 120)
    v = np.linspace(0, np.pi, 70)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=True, zorder=1)


def plume(ax, radius: float, height: float = 150.0, n_jets: int = 7,
          seed: int = 3) -> None:
    """The south-polar plume, as several narrow jets fanning out from the pole.

    Cassini resolved the plume as a set of discrete jets along the tiger-stripe fractures
    rather than one axisymmetric cone, so a few tilted jets read closer to the real thing
    and look less like a traffic cone.

    Illustrative geometry only: the Cassini transits are approximately horizontal cuts at
    differing latitudes and speeds and do not resolve sample yield against altitude at these
    heights, which is why the plume gradient is a swept parameter rather than a fitted
    profile.
    """
    rng = np.random.default_rng(seed)
    h = np.linspace(0, height, 10)
    t = np.linspace(0, 2 * np.pi, 20)
    T, H = np.meshgrid(t, h)
    for _ in range(n_jets):
        # Each jet leans a little off the pole and flares with altitude.
        tilt = rng.uniform(-0.28, 0.28, size=2)
        flare = rng.uniform(0.10, 0.20)
        ax.plot_surface(
            flare * H * np.cos(T) + tilt[0] * H,
            flare * H * np.sin(T) + tilt[1] * H,
            -radius - H,
            color="#7fb3e0", alpha=0.13, linewidth=0, shade=False, zorder=0,
        )


def draw(ax, xyz, starts, codes, names, radius, zoom: bool) -> set[str]:
    """Draw the arcs, one line per pass, coloured by that pass's objective."""
    sphere(ax, radius)
    plume(ax, radius)

    used: set[str] = set()
    ends = list(starts[1:]) + [len(xyz)]
    for i0, i1, code in zip(starts, ends, codes):
        name = names[int(code)]
        seg = xyz[int(i0):int(i1)]
        if len(seg) < 2:
            continue
        used.add(name)
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                color=ACTION_COLOR.get(name, "#9e9e9e"),
                lw=1.4 if name != "CORRECT" else 0.8,
                alpha=0.95 if name != "CORRECT" else 0.55,
                zorder=3)

    if zoom:
        # Tight on the close approach: the bands differ by ~23 km at periapsis on a
        # ~1500 km loop, so the full view cannot show what distinguishes the objectives.
        lim = radius + 190.0
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim - 60, lim - 60)
    else:
        lim = float(np.abs(xyz).max()) * 1.05
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.grid(False)
    ax.set_xlabel("x (km)", labelpad=2)
    ax.set_ylabel("y (km)", labelpad=2)
    ax.set_zlabel("z (km)", labelpad=2)
    ax.tick_params(labelsize=8)
    return used


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--sigma", type=float, nargs="*", default=None,
                    help="navigation levels to draw; default every level in the file")
    ap.add_argument("--zoom", action="store_true",
                    help="crop to the close approach instead of the whole orbit")
    ap.add_argument("--elev", type=float, default=16.0)
    ap.add_argument("--azim", type=float, default=-72.0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"{args.data} not found — generate it with:\n"
            "  julia --project=experiments experiments/export_arcs.jl 0.0,0.05,0.1"
        )

    use_serif()
    d = np.load(args.data)
    names = bytes(d["action_names_utf8"]).decode().split(",")
    radius = float(d["r_enceladus"])
    sigmas = list(d["sigmas"]) if args.sigma is None else args.sigma

    n = len(sigmas)
    fig = plt.figure(figsize=(3.2 * n, 3.6))
    used_all: set[str] = set()
    for k, sig in enumerate(sigmas):
        key = str(float(sig)).replace(".", "p")
        if f"xyz_{key}" not in d:
            raise SystemExit(f"no arcs for sigma = {sig} in {args.data}")
        ax = fig.add_subplot(1, n, k + 1, projection="3d")
        used_all |= draw(ax, d[f"xyz_{key}"], d[f"starts_{key}"], d[f"actions_{key}"],
                         names, radius, args.zoom)
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_title(rf"$\sigma = {float(sig):g}$ km", pad=-2, fontsize=11)

    # One legend for the whole figure, listing only the objectives that actually appear —
    # the absence of "Low" at higher sigma is the result, so it must not be implied present.
    handles = [
        plt.Line2D([], [], color=ACTION_COLOR[a], lw=1.8, label=ACTION_LABEL[a])
        for a in ("EXCURSE_HIGH", "CORRECT", "EXCURSE_MID", "EXCURSE_LOW")
        if a in used_all
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()

    stem = args.out or ("rollout_3d_zoom" if args.zoom else "rollout_3d")
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        path = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(path, transparent=True, bbox_inches="tight", dpi=200)
    print(f"wrote {FIG_DIR}/{stem}.{{pdf,svg,png}}  "
          f"({(FIG_DIR / f'{stem}.pdf').stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()