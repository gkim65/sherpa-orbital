"""
plot_policy_panel.py — action bands beside the flown geometry, one row per navigation level.

    uv run python scripts/plot_policy_panel.py --sigma 0.0            # prototype, one row
    uv run python scripts/plot_policy_panel.py --sigma 0.0 0.05 0.1 0.2

Left of each row: which objective the policy chose at each pass, as a strip per objective
shaded by the fraction of rollouts that chose it. Right: one rollout's trajectory at that
level, coloured by the same objectives, cropped to the close approach.

Reads two files, both written from Julia:

    julia --project=experiments experiments/export_arcs.jl 0.0,0.05,0.1,0.2
    # and the band matrices, see the snippet in experiments/ (figures/bands.npz)

The bands are a 1000-rollout statistic and the trajectory is a single rollout, so the two
halves of a row answer different questions: what the policy does in general, and what one
episode looks like.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

# Inches trimmed off the right edge on save. mplot3d reserves a wide internal margin inside
# its axes box that no layout setting removes, so the dead space right of the inset rule is
# cut from the bounding box instead.
RIGHT_TRIM_IN = 0.46

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "figures"

ACTS = ["EXCURSE_HIGH", "CORRECT", "EXCURSE_MID", "EXCURSE_LOW"]
TICKS = ["High (46 km)", "Correct (37 km)", "Mid (30 km)", "Low (24 km)"]
ACTION_COLOR = {
    "INITIAL": "#9e9e9e",
    "CORRECT": "#9e9e9e",
    "EXCURSE_LOW": "#4477aa",
    "EXCURSE_MID": "#228833",
    "EXCURSE_HIGH": "#ee7733",
}
BODY = "#c4c9d0"


def use_serif() -> None:
    """Computer Modern serif, via LaTeX when it is on PATH and mathtext-cm otherwise."""
    if shutil.which("latex"):
        matplotlib.rcParams.update({"text.usetex": True, "font.family": "serif",
                                    "font.serif": ["Computer Modern Roman"]})
    else:
        matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif",
                                    "mathtext.fontset": "cm"})
    matplotlib.rcParams.update({"font.size": 11})


def band_row(ax, M, t_days, last: bool) -> None:
    """One level's action bands: a strip per objective, opacity carrying the fraction."""
    for i, act in enumerate(ACTS):
        # Transparent-to-solid in the objective's own colour, so the strip is identifiable
        # without reading its label and opacity reads as "how often".
        cmap = LinearSegmentedColormap.from_list(
            act, [(*to_rgb(ACTION_COLOR[act]), 0.0), (*to_rgb(ACTION_COLOR[act]), 1.0)]
        )
        # `imshow` with an explicit extent rather than pcolormesh: the strip is a regular
        # grid in time, and imshow takes the fractions as-is without needing edge arrays.
        ax.imshow(M[i][None, :], cmap=cmap, vmin=0, vmax=1, aspect="auto",
                  extent=(0.0, float(np.max(t_days)), i + 0.5, i - 0.5),
                  interpolation="nearest", rasterized=True)
    for y in (0.5, 1.5, 2.5):
        ax.axhline(y, color="0.85", lw=0.5)
    ax.set_yticks(range(len(ACTS)))
    ax.set_yticklabels(TICKS, fontsize=10.5)
    # ACTS runs top-to-bottom by altitude, so the axis is inverted to keep High on top.
    ax.set_ylim(len(ACTS) - 0.5, -0.5)
    ax.set_xlim(0, float(np.max(t_days)))
    ax.set_xlabel("Mission time (days)" if last else "")
    ax.tick_params(labelbottom=last, labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def orbit_panel(ax, xyz, starts, codes, names, radius, zoom_pad: float,
                elev: float, azim: float) -> None:
    """One rollout's trajectory, cropped to the close approach."""
    # Enceladus. mplot3d depth-sorts the quads, so the far side of the trajectory is
    # occluded and the near side is not.
    u = np.linspace(0, 2 * np.pi, 120)
    v = np.linspace(0, np.pi, 70)
    ax.plot_surface(radius * np.outer(np.cos(u), np.sin(v)),
                    radius * np.outer(np.sin(u), np.sin(v)),
                    radius * np.outer(np.ones_like(u), np.cos(v)),
                    color=BODY, alpha=0.55, linewidth=0, shade=True, zorder=1)

    # Plume as a FAN of jets, each rooted at a different point along the south-polar
    # fractures and leaning outward — Cassini resolved discrete jets along the tiger
    # stripes, not one axisymmetric cone, and a fan reads as a plume rather than a traffic
    # cone.
    #
    # `lean` is applied in Y because at this azimuth (~-78 deg) Y is the left-right
    # direction on the page; leaning in X would fan the jets toward and away from the
    # camera, where they collapse onto each other.
    h = np.linspace(0, 150.0, 10)
    t = np.linspace(0, 2 * np.pi, 16)
    T, H = np.meshgrid(t, h)
    # Thirteen jets rather than seven, spaced more tightly: at seven the gaps between them
    # read as a defect rather than as structure.
    roots = np.linspace(-95.0, 95.0, 13)
    for root in roots:
        lean = 0.0075 * root          # leaning outward, proportional to the root offset
        # Rooted on the surface at that offset, so the jets start apart rather than all
        # from the pole.
        z0 = -np.sqrt(max(radius ** 2 - root ** 2, 0.0))
        ax.plot_surface(0.05 * H * np.cos(T),
                        0.05 * H * np.sin(T) + root + lean * H,
                        z0 - H,
                        color="#7fb3e0", alpha=0.16, linewidth=0, shade=False, zorder=0)

    ends = list(starts[1:]) + [len(xyz)]
    for i0, i1, code in zip(starts, ends, codes):
        name = names[int(code)]
        seg = xyz[int(i0):int(i1)]
        if len(seg) < 2:
            continue
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                color=ACTION_COLOR.get(name, "#9e9e9e"),
                lw=1.6 if name != "CORRECT" else 0.9,
                alpha=0.95 if name != "CORRECT" else 0.5, zorder=3)

    # Cropped on the CLOSE APPROACH, not on the body centre. The altitude spread between
    # objectives is ~23 km on a 252 km body, so a box centred on the origin spends its whole
    # extent on the moon and compresses the spread to nothing. Centring on the lowest points
    # of the trajectory and padding from there is what magnifies it.
    r = np.sqrt((xyz ** 2).sum(axis=1))
    close = xyz[r < radius + 120.0]
    ctr = close.mean(axis=0) if len(close) else np.zeros(3)
    # Tight on the body: the altitude spread between objectives is ~23 km on a 252 km
    # radius, so every extra 100 km of frame compresses what the panel exists to show.
    half = radius * 1.05 + zoom_pad
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    # Z window shifted UP relative to the pass, which moves the drawn geometry DOWN in the
    # panel: centring on the close approach leaves the lower third empty, since the
    # trajectory only ever goes up from there.
    # Asymmetric about the pass: the trajectory only climbs from the close approach, so a
    # centred window leaves the lower third empty. Shifted only slightly, since going
    # further pushes the body past the panel edge.
    ax.set_zlim(ctr[2] - 0.62 * half, ctr[2] + 1.38 * half)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    # NO AXES AT ALL.
    #
    # Coordinates here would be Enceladus-CENTRED, so the z values near the pass run -157
    # to -297 km while the altitudes there are 23 to 120 km — printing "-400" beside
    # trajectories the figure labels "24 km" invites exactly the wrong reading. The moon
    # carries the scale and the band labels beside it carry the altitudes, so a frame adds
    # nothing and boxes in a panel that reads better open.
    ax.set_axis_off()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arcs", type=Path, default=FIG_DIR / "arcs.npz")
    ap.add_argument("--bands", type=Path, default=FIG_DIR / "bands.npz")
    ap.add_argument("--sigma", type=float, nargs="+", required=True)
    ap.add_argument("--zoom-pad", type=float, default=35.0,
                    help="km beyond the surface to keep in frame; smaller = deeper zoom")
    ap.add_argument("--elev", type=float, default=14.0)
    # ~20 deg further round than the near-edge-on -78, so the front-facing axis is visible
    # and the orbit does not read as a flat vertical loop.
    ap.add_argument("--azim", type=float, default=-58.0)
    ap.add_argument("--out", type=str, default="policy_panel")
    args = ap.parse_args()

    for p in (args.arcs, args.bands):
        if not p.exists():
            raise SystemExit(f"{p} not found — see the module docstring for how to write it")

    use_serif()
    A = np.load(args.arcs)
    B = np.load(args.bands)
    names = bytes(A["action_names_utf8"]).decode().split(",")
    radius = float(A["r_enceladus"])

    n = len(args.sigma)
    # 6.55 in rather than 7.2: mplot3d reserves a wide internal margin inside its own axes
    # box which no layout setting removes, so the canvas is narrowed instead and `left` is
    # raised in step to keep the rotated labels off the edge.
    fig = plt.figure(figsize=(6.55, 1.30 * n + 0.40))
    # Bands take most of the width; the orbit is an inset beside them.
    # `left` leaves room for the rotated level labels, which sit outside the axes.
    gs = fig.add_gridspec(n, 2, width_ratios=[1.0, 0.46], wspace=-0.075, hspace=0.34,
                          left=0.213, right=0.995, top=0.985, bottom=0.085)

    for k, sig in enumerate(args.sigma):
        key = str(float(sig)).replace(".", "p")
        for f, src in ((f"M_{key}", args.bands), (f"xyz_{key}", args.arcs)):
            if f not in (B if src is args.bands else A):
                raise SystemExit(f"no {f} in {src}")

        axb = fig.add_subplot(gs[k, 0])
        band_row(axb, B[f"M_{key}"], B[f"tdays_{key}"], last=(k == n - 1))
        # The level label sits OUTSIDE the row, to the left of the objective names and
        # rotated: stacked tightly, a title above a row reads as belonging to the row above
        # it, and a label inside the axes competes with the strips.
        # Shaded backing so each row reads as its own block: with four rows stacked, a bare
        # rotated label does not visually bind to the strips beside it. Moved in closer to
        # the row now that the box gives it its own footprint.
        axb.annotate(rf"$\sigma = {float(sig):g}$ km",
                     xy=(0, 0.5), xycoords="axes fraction",
                     xytext=(-88, 0), textcoords="offset points",
                     rotation=90, ha="center", va="center", fontsize=10.5,
                     bbox=dict(boxstyle="round,pad=0.42", facecolor="0.91",
                               edgecolor="0.72", linewidth=0.6))

        axo = fig.add_subplot(gs[k, 1], projection="3d")
        # mplot3d centres its content in the axes box, so the way to move the drawn
        # geometry DOWN inside the dotted rule is to move the AXES down. Re-cropping with
        # z limits only changes what is visible, not where it lands.
        p0 = axo.get_position()
        axo.set_position([p0.x0, p0.y0 - 0.018, p0.width, p0.height])
        orbit_panel(axo, A[f"xyz_{key}"], A[f"starts_{key}"], A[f"actions_{key}"],
                    names, radius, args.zoom_pad, args.elev, args.azim)
        # Dotted outline so the panel reads as an inset rather than as part of the strip
        # axes beside it. Drawn on the figure, not the 3D axes, which have no frame.
        # A dotted rule around the cell, from the ORIGINAL box: the axes was just shifted
        # down, so reading its position now would drag the rule with it and reopen the gap.
        bb = p0
        # Shifted LEFT and pulled in on the RIGHT: mplot3d leaves a wide asymmetric margin
        # inside its own bounding box, so a rule matching that box leaves much more white
        # space on the right of the drawn geometry than on the left.
        fig.add_artist(plt.Rectangle((bb.x0 - 0.008, bb.y0 + 0.006),
                                     bb.width - 0.010, bb.height - 0.012,
                                     transform=fig.transFigure, fill=False,
                                     edgecolor="0.62", linestyle=(0, (2, 2)),
                                     linewidth=0.7, zorder=10))

    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        # A small pad rather than the 0.1 default: `bbox_inches="tight"` crops to the
        # artists and LaTeX supplies its own spacing, but at 0.01 the rotated level labels
        # sit hard against the left edge and their boxes get shaved.
        # The left margin is reserved in the layout (`left=` on the gridspec) because the
        # rotated level labels are annotations offset outside the axes, and a tight bbox
        # measures only the axes and their decorations — it would shave the label boxes.
        #
        # The right margin is mplot3d's own internal padding inside its axes box, which it
        # keeps whatever the layout says, so the bounding box is trimmed there explicitly:
        # `Bbox` in inches, from the figure's own size, ending just past the dotted rule.
        bb_all = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        trim = Bbox.from_extents(bb_all.x0, bb_all.y0,
                                 bb_all.x1 - RIGHT_TRIM_IN, bb_all.y1)
        fig.savefig(FIG_DIR / f"{args.out}.{ext}", transparent=True, dpi=200,
                    bbox_inches=trim)
    kb = (FIG_DIR / f"{args.out}.pdf").stat().st_size / 1024
    print(f"wrote {FIG_DIR}/{args.out}.{{pdf,svg,png}}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()