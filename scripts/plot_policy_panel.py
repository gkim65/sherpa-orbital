"""
plot_policy_panel.py — action bands beside the flown geometry, one row per navigation level.

    uv run python scripts/plot_policy_panel.py --sigma 0.0            # prototype, one row
    uv run python scripts/plot_policy_panel.py --sigma 0.0 0.1 0.2

Three columns per row. Left: which objective the policy chose at each pass, as a strip per
objective shaded by the fraction of rollouts that chose it. Middle: one rollout's pass under
the moon, coloured by the same objectives, cropped close enough to separate the commanded
altitudes, with a dotted rule marking the crop. Right: the same rollout's whole revolution
at a tilted angle, carrying the orbit shape the crop cannot -- apoapsis is ~1076 km against
a 24-46 km periapsis, so no single frame resolves both.

Reads two files, both written from Julia:

    julia --project=experiments experiments/export_arcs.jl 0.0,0.1,0.2
    # and the band matrices, see the snippet in experiments/ (figures/bands.npz)

The bands are a 1000-rollout statistic and the geometry is a single rollout, so the columns
of a row answer different questions: what the policy does in general, and what one episode
looks like.
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
from mpl_toolkits.mplot3d import proj3d  # noqa: E402

# Figure-fraction drop for the context column, so its content lines up with the zoomed inset
# beside it rather than floating above it.
CONTEXT_DROP = 0.042

# Column geometry in figure fractions. The two 3D columns are PLACED rather than gridded:
# mplot3d pads inside its own axes box by a fixed fraction no layout setting exposes, so the
# axes are deliberately oversized (the *_PAD_* terms) while the nominal column spans below
# describe where the visible geometry lands.
BANDS_RIGHT = 0.560     # right edge of the bands column
ZOOM_LEFT, ZOOM_W = 0.575, 0.150      # zoomed periapsis inset
# Pulled well left of the zoom: mplot3d centres the moon inside a box much wider than it, so
# the axes has to start left of where the moon should land. Measured, the drawn geometry sits
# ~0.038 right of the axes edge.
CTX_LEFT, CTX_W = 0.632, 0.330        # tilted context view
ZOOM_PAD_W, ZOOM_PAD_H = 0.030, 0.006
CTX_PAD_W, CTX_PAD_H = 0.030, 0.034

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
# Base point size. A full-width figure is placed at close to its natural size, so type can be
# larger in absolute terms than in a single-column figure that gets shrunk to fit.
FONT_BASE = 12.5


def use_serif() -> None:
    """Computer Modern serif, via LaTeX when it is on PATH and mathtext-cm otherwise."""
    if shutil.which("latex"):
        matplotlib.rcParams.update({"text.usetex": True, "font.family": "serif",
                                    "font.serif": ["Computer Modern Roman"]})
    else:
        matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif",
                                    "mathtext.fontset": "cm"})
    matplotlib.rcParams.update({"font.size": FONT_BASE})


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
    ax.set_yticklabels(TICKS, fontsize=FONT_BASE)
    # ACTS runs top-to-bottom by altitude, so the axis is inverted to keep High on top.
    ax.set_ylim(len(ACTS) - 0.5, -0.5)
    ax.set_xlim(0, float(np.max(t_days)))
    ax.set_xlabel("Mission time (days)" if last else "", fontsize=FONT_BASE)
    ax.tick_params(labelbottom=last, labelsize=FONT_BASE - 1.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _ink_bbox(fig, ax) -> Bbox:
    """Figure-fraction bounding box of the pixels `ax` actually drew.

    Rasterises the figure once and scans the alpha channel inside the axes' own region, so
    the result is what LANDED on the canvas rather than what a projection predicts. Needed
    because matplotlib renders a surface quad or line segment whenever any part of it is
    inside the view window, which no per-point visibility test reproduces.

      - `fig` — the figure, already drawn
      - `ax` — the axes whose ink is measured

    Returns the ink extent in figure fractions, or the axes' own box if it drew nothing.
    """
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba())
    p = ax.get_position()
    # Pixel window of the axes. Rows run top-down in the buffer, y runs bottom-up in figure
    # fractions, hence the flip.
    px0, px1 = int(np.floor(p.x0 * w)), int(np.ceil(p.x1 * w))
    py0, py1 = int(np.floor((1.0 - p.y1) * h)), int(np.ceil((1.0 - p.y0) * h))
    px0, px1 = max(px0, 0), min(px1, w)
    py0, py1 = max(py0, 0), min(py1, h)
    if px1 <= px0 or py1 <= py0:
        return p
    alpha = buf[py0:py1, px0:px1, 3]
    rows, cols = np.where(alpha > 8)
    if not len(rows):
        return p
    x0 = (px0 + cols.min()) / w
    x1 = (px0 + cols.max() + 1) / w
    y1 = 1.0 - (py0 + rows.min()) / h
    y0 = 1.0 - (py0 + rows.max() + 1) / h
    return Bbox.from_extents(x0, y0, x1, y1)


def _resample(seg, factor: int = 8):
    """Interpolate a trajectory segment onto a finer grid, by arclength.

    The arcs are stored at 60 points of uniform TIME, and periapsis is where the spacecraft
    is fastest, so the sampling is sparsest exactly where the curve bends hardest: altitude
    steps ~1.5 km per point inbound against ~30 km outbound. Drawn raw that turn becomes a
    few straight chords with a visible kink, which reads as a wobble in the orbit rather than
    as the sampling artifact it is.

      - `seg` — (n, 3) positions (km)
      - `factor` — output points per input point

    Returns the resampled segment (km), or `seg` unchanged if it is too short to interpolate.
    """
    if len(seg) < 4:
        return seg
    # Cumulative chord length as the parameter, so points are spread evenly along the PATH
    # rather than along time; that is what closes the gap at periapsis.
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))])
    if d[-1] <= 0:
        return seg
    t = np.linspace(0.0, d[-1], len(seg) * factor)
    return np.column_stack([np.interp(t, d, seg[:, i]) for i in range(3)])


def _exaggerate(pts, radius: float, r_draw: float, gain: float):
    """Compress the body and stretch the altitudes above it, about the moon's centre.

    Maps radius `r` to `r_draw + (r - radius) * gain`, so the surface lands at `r_draw` while
    each km of altitude occupies `gain` km of drawn space. Direction is untouched, so the
    orbit keeps its shape and only the radial scale is distorted.

      - `pts` — (n, 3) Enceladus-centred positions (km)
      - `radius` — the body's true radius (km)
      - `r_draw` — radius to draw the body at (km)
      - `gain` — km of drawn space per km of true altitude

    Returns the remapped positions (km, drawn scale).
    """
    r = np.sqrt((pts ** 2).sum(axis=1, keepdims=True))
    r = np.where(r < 1e-9, 1e-9, r)
    return pts / r * (r_draw + (r - radius) * gain)


def orbit_panel(ax, xyz, starts, codes, names, radius, zoom_pad: float,
                elev: float, azim: float, span_km: float | None = None,
                r_draw: float | None = None, gain: float = 1.0,
                inset_box: float | None = None) -> Bbox:
    """One rollout's trajectory, cropped to the close approach.

    Returns the figure-fraction bounding box of the drawn geometry, for placing the rule.
    """
    # Enceladus. mplot3d depth-sorts the quads, so the far side of the trajectory is
    # occluded and the near side is not.
    u = np.linspace(0, 2 * np.pi, 220)
    v = np.linspace(0, np.pi, 130)
    r_body = radius if r_draw is None else r_draw
    ax.plot_surface(r_body * np.outer(np.cos(u), np.sin(v)),
                    r_body * np.outer(np.sin(u), np.sin(v)),
                    r_body * np.outer(np.ones_like(u), np.cos(v)),
                    color=BODY, alpha=0.55, linewidth=0, edgecolors=BODY,
                    # Flat when zoomed right in: the lit-sphere falloff darkens the limb, and
                    # a tight frame sits entirely inside that falloff, rendering near-black.
                    shade=(span_km is None or span_km > 260.0), zorder=1)

    # Plume as a FAN of jets, each rooted at a different point along the south-polar
    # fractures and leaning outward — Cassini resolved discrete jets along the tiger
    # stripes, not one axisymmetric cone, and a fan reads as a plume rather than a traffic
    # cone.
    #
    # `lean` is applied in Y because at this azimuth (~-78 deg) Y is the left-right
    # direction on the page; leaning in X would fan the jets toward and away from the
    # camera, where they collapse onto each other.
    h = np.linspace(0, 150.0, 10) * (r_body / radius)
    t = np.linspace(0, 2 * np.pi, 16)
    T, H = np.meshgrid(t, h)
    # Thirteen jets rather than seven, spaced more tightly: at seven the gaps between them
    # read as a defect rather than as structure.
    roots = np.linspace(-95.0, 95.0, 13) * (r_body / radius)
    for root in roots:
        lean = 0.0075 * root          # leaning outward, proportional to the root offset
        # Rooted on the surface at that offset, so the jets start apart rather than all
        # from the pole.
        z0 = -np.sqrt(max(r_body ** 2 - root ** 2, 0.0))
        ax.plot_surface(0.05 * H * np.cos(T),
                        0.05 * H * np.sin(T) + root + lean * H,
                        z0 - H,
                        color="#7fb3e0", alpha=0.16, linewidth=0, shade=False, zorder=0)

    draw_xyz = xyz if r_draw is None else _exaggerate(xyz, radius, r_draw, gain)
    ends = list(starts[1:]) + [len(xyz)]
    for i0, i1, code in zip(starts, ends, codes):
        name = names[int(code)]
        seg = draw_xyz[int(i0):int(i1)]
        if len(seg) < 2:
            continue
        seg = _resample(seg)
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                color=ACTION_COLOR.get(name, "#9e9e9e"),
                lw=1.6 if name != "CORRECT" else 0.9,
                alpha=0.95 if name != "CORRECT" else 0.5, zorder=3)

    # Cropped on the CLOSE APPROACH, not on the body centre. The altitude spread between
    # objectives is ~23 km on a 252 km body, so a box centred on the origin spends its whole
    # extent on the moon and compresses the spread to nothing. Centring on the lowest points
    # of the trajectory and padding from there is what magnifies it.
    half = (span_km / 2.0) if span_km is not None else (radius * 1.05 + zoom_pad)
    if r_draw is None:
        # Centred on the periapsis arc itself: the mean of the points actually near the body,
        # so the four commanded altitudes sit in the middle of the frame rather than at an
        # edge. `close` is the pass, not the whole revolution.
        r = np.sqrt((draw_xyz ** 2).sum(axis=1))
        alt = r - r_body
        # The lowest slice of the pass, so the centre sits on the commanded altitudes
        # themselves. A shell test wide enough to catch them also catches the climbing legs,
        # whose mean pulls the frame up and off the part being magnified.
        lo = np.quantile(alt, 0.04)
        close = draw_xyz[alt <= max(lo, alt.min() + 30.0)]
        ctr = close.mean(axis=0) if len(close) else np.zeros(3)
    else:
        # Body-centred: with the altitudes exaggerated the pass sits well off the surface, so
        # a window centred on the arc pushes the moon into a corner.
        ctr = np.zeros(3)
    # Framed on the PERIAPSIS ARC rather than on the whole body.
    #
    # The commanded bands sit at 24, 30, 37 and 46 km altitude — a 22 km spread on a 252 km
    # radius. A frame that fits the moon is ~540 km across, so the spread the panel exists to
    # show occupies ~4% of it and the four traces collapse onto one line. `half` is set from
    # the spread instead, which magnifies it about tenfold; the moon then fills the lower part
    # of the frame as a limb rather than appearing whole.
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    # Z window shifted UP relative to the pass, which moves the drawn geometry DOWN in the
    # panel: centring on the close approach leaves the lower third empty, since the
    # trajectory only ever goes up from there.
    # Asymmetric about the pass: the trajectory only climbs from the close approach, so a
    # centred window leaves the lower third empty. Shifted only slightly, since going
    # further pushes the body past the panel edge.
    ax.set_zlim(ctr[2] - 0.80 * half, ctr[2] + 1.20 * half)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    # Dotted box marking the region a companion panel magnifies. Drawn in DATA coordinates
    # and facing the camera, so it lands on the periapsis arc under this projection rather
    # than floating in screen space.
    if inset_box is not None:
        b = inset_box / 2.0
        a_rad, e_rad = np.radians(azim), np.radians(elev)
        # Screen right and screen up, in data space, for the current camera.
        right = np.array([-np.sin(a_rad), np.cos(a_rad), 0.0])
        up = np.array([-np.cos(a_rad) * np.sin(e_rad), -np.sin(a_rad) * np.sin(e_rad),
                       np.cos(e_rad)])
        # Centred on the lowest slice of the pass, the same points the zoom centres on.
        alt_b = np.sqrt((draw_xyz ** 2).sum(axis=1)) - r_body
        low = draw_xyz[alt_b <= max(np.quantile(alt_b, 0.04), alt_b.min() + 30.0)]
        c = low.mean(axis=0) if len(low) else np.zeros(3)
        # Pulled toward the camera so the sphere does not occlude it: this is an annotation
        # on the view, not an object in the scene.
        a_rad_v, e_rad_v = np.radians(azim), np.radians(elev)
        view = np.array([np.cos(e_rad_v) * np.cos(a_rad_v),
                         np.cos(e_rad_v) * np.sin(a_rad_v), np.sin(e_rad_v)])
        c = c + view * (2.2 * r_body)
        loop = [c - b * right - b * up, c + b * right - b * up,
                c + b * right + b * up, c - b * right + b * up]
        loop.append(loop[0])
        loop = np.array(loop)
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="0.30",
                linestyle=(0, (2.5, 2.5)), linewidth=0.9, zorder=20)

    ax.view_init(elev=elev, azim=azim)
    # NO AXES AT ALL.
    #
    # Coordinates here would be Enceladus-CENTRED, so the z values near the pass run -157
    # to -297 km while the altitudes there are 23 to 120 km — printing "-400" beside
    # trajectories the figure labels "24 km" invites exactly the wrong reading. The moon
    # carries the scale and the band labels beside it carry the altitudes, so a frame adds
    # nothing and boxes in a panel that reads better open.
    ax.set_axis_off()

    # FIGURE-fraction extent of what was actually drawn. mplot3d pads its axes box
    # asymmetrically by an amount no layout setting exposes, so the only reliable way to put
    # a rule around the geometry is to project the drawn points and measure them.
    fig = ax.get_figure()
    # Every drawn thing, sampled: the moon's surface, the flown path, and the plume jets.
    # mplot3d clips rendering to the axes limits, so a point is visible iff it lies inside
    # the view window -- which makes "what is drawn" and "what is in frame" the same test.
    (x0, x1), (y0, y1), (z0, z1) = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()

    sph = np.array([[r_body * np.cos(a) * np.sin(b), r_body * np.sin(a) * np.sin(b),
                     r_body * np.cos(b)]
                    for a in np.linspace(0, 2 * np.pi, 90)
                    for b in np.linspace(0, np.pi, 46)])
    # Jets along their whole length, not just the tips: the visible part is wherever the jet
    # crosses the window, and sampling only the endpoints misses it when the tip falls
    # outside.
    jet_len = 150.0 * (r_body / radius)
    jets = np.array([[0.0, root + 0.0075 * root * hh,
                      -np.sqrt(max(r_body ** 2 - root ** 2, 0.0)) - hh]
                     for root in np.linspace(-95.0, 95.0, 13) * (r_body / radius)
                     for hh in np.linspace(0.0, jet_len, 24)])
    content = np.vstack([sph, draw_xyz, jets])

    # Keep only what the axes will actually render, then measure THAT. Intersecting a content
    # box with the frustum box instead would drop the jets: they leave the window low and to
    # the side, so the intersection clips them out even though their upper parts are drawn.
    inside = ((content[:, 0] >= x0) & (content[:, 0] <= x1)
              & (content[:, 1] >= y0) & (content[:, 1] <= y1)
              & (content[:, 2] >= z0) & (content[:, 2] <= z1))
    vis = content[inside] if inside.any() else content

    px, py, _ = proj3d.proj_transform(vis[:, 0], vis[:, 1], vis[:, 2], ax.get_proj())
    inv = fig.transFigure.inverted().transform(
        ax.transData.transform(np.column_stack([px, py])))
    return Bbox.from_extents(inv[:, 0].min(), inv[:, 1].min(),
                             inv[:, 0].max(), inv[:, 1].max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arcs", type=Path, default=FIG_DIR / "arcs.npz")
    ap.add_argument("--bands", type=Path, default=FIG_DIR / "bands.npz")
    ap.add_argument("--sigma", type=float, nargs="+", required=True)
    ap.add_argument("--zoom-pad", type=float, default=35.0,
                    help="km beyond the surface to keep in frame; smaller = deeper zoom")
    ap.add_argument("--r-draw", type=float, default=0.0,
                    help="radius to draw Enceladus at (km); below the true 252 the body "
                         "shrinks and altitudes stretch, which separates the bands. 0 = true "
                         "scale")
    ap.add_argument("--gain", type=float, default=0.0,
                    help="km drawn per km of altitude; 0 picks the value that maps the band "
                         "spread onto the freed-up space")
    ap.add_argument("--span-km", type=float, default=130.0,
                    help="frame width in km centred on the close approach, which crops to "
                         "the 24-46 km band spread rather than to the whole body; pass 0 to "
                         "fall back to --zoom-pad and frame the moon instead")
    ap.add_argument("--elev", type=float, default=14.0)
    # ~20 deg further round than the near-edge-on -78, so the front-facing axis is visible
    # and the orbit does not read as a flat vertical loop.
    ap.add_argument("--azim", type=float, default=0.0)
    ap.add_argument("--width", type=float, default=10.5,
                    help="figure width in inches")
    ap.add_argument("--aspect", type=float, default=0.40,
                    help="height/width; 0.4 is the 2:5 two-column shape")
    ap.add_argument("--context-span", type=float, default=620.0,
                    help="frame width in km for the context inset: the tilted view with the "
                         "body low in frame and the passes sweeping past it")
    ap.add_argument("--context-azim", type=float, default=-58.0,
                    help="azimuth for the context inset; the original tilted view")
    ap.add_argument("--context-elev", type=float, default=14.0)
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

    # Default gain fills the space the shrunken body frees, so the bands spread across the
    # panel rather than needing a hand-tuned pair of numbers.
    r_draw = args.r_draw if args.r_draw > 0 else None
    gain = args.gain if args.gain > 0 else (
        1.0 if r_draw is None else (radius - r_draw) / 22.3 + 1.0)

    n = len(args.sigma)
    # 2:5 height:width, drawn large and placed across both columns. Width leads and the
    # height follows, so the aspect holds at any `--width`.
    fig = plt.figure(figsize=(args.width, args.width * args.aspect))
    # `left` leaves room for the rotated level labels, which sit outside the axes.
    gs = fig.add_gridspec(n, 1, wspace=0.0, hspace=0.34,
                          left=0.172, right=BANDS_RIGHT, top=0.952, bottom=0.155)

    right_edge = 0.0
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
        # Shaded backing so each row reads as its own block: stacked, a bare rotated label
        # does not visually bind to the strips beside it.
        # Positioned as a FIGURE x-coordinate rather than an offset in points: an offset
        # scales with nothing, so raising the font pushed the box off the reserved margin and
        # the tight bbox shaved it. This pins it inside `left` whatever the font size.
        axb.annotate(rf"$\sigma = {float(sig):g}$ km",
                     xy=(0.022, axb.get_position().y0 + axb.get_position().height / 2),
                     xycoords="figure fraction",
                     rotation=90, ha="center", va="center", fontsize=FONT_BASE,
                     bbox=dict(boxstyle="round,pad=0.42", facecolor="0.91",
                               edgecolor="0.72", linewidth=0.6))

        row = axb.get_position()
        axo = fig.add_axes([ZOOM_LEFT - ZOOM_PAD_W, row.y0 - ZOOM_PAD_H,
                            ZOOM_W + 2 * ZOOM_PAD_W, row.height + 2 * ZOOM_PAD_H],
                           projection="3d")
        # The axes is oversized past its nominal span by the *_PAD_* terms: mplot3d keeps a
        # fixed internal margin that no layout setting removes, so an axes sized to where the
        # geometry should land draws the moon smaller than that. The dotted rule is measured
        # off the render instead, below.
        drawn = orbit_panel(axo, A[f"xyz_{key}"], A[f"starts_{key}"], A[f"actions_{key}"],
                            names, radius, args.zoom_pad, args.elev, args.azim,
                            args.span_km if args.span_km > 0 else None,
                            r_draw, gain)

        # Box on the PANEL, not on the arcs: `drawn` measures the trajectory only, so a rule
        # fitted to it slices through the moon above and the jets below.
        # Box from the RENDERED extent of the inset, measured off the canvas after drawing.
        # Projecting sampled points cannot do this: matplotlib renders a quad or segment when
        # any part of it lies in the window, so a per-point test misses the edges.
        fig.canvas.draw()
        ink = _ink_bbox(fig, axo)
        bpad = 0.006
        fig.add_artist(plt.Rectangle((ink.x0 - bpad, ink.y0 - bpad),
                                     ink.width + 2 * bpad, ink.height + 2 * bpad,
                                     transform=fig.transFigure, fill=False,
                                     edgecolor="0.55", linestyle=(0, (2.5, 2.5)),
                                     linewidth=0.8, zorder=12))

        # Context view: the same rollout at the ORIGINAL tilted angle, framed wide enough to
        # take in the whole revolution. The zoom beside it resolves the bands but shows only
        # a sliver of the orbit, so this carries the shape the zoom cannot.
        axc = fig.add_axes([CTX_LEFT - CTX_PAD_W, row.y0 - CTX_PAD_H - CONTEXT_DROP,
                            CTX_W + 2 * CTX_PAD_W, row.height + 2 * CTX_PAD_H],
                           projection="3d")
        drawn_c = orbit_panel(axc, A[f"xyz_{key}"], A[f"starts_{key}"], A[f"actions_{key}"],
                              names, radius, args.zoom_pad, args.context_elev,
                              args.context_azim, args.context_span, None, 1.0,
                              inset_box=args.span_km)
        fig.canvas.draw()
        inkc = _ink_bbox(fig, axc)
        right_edge = max(right_edge, ink.x1, inkc.x1)

        if k == n - 1:
            # Same boxed treatment as the level labels, so the two kinds of annotation
            # read as one family. `dx` nudges a label off its ink centre: mplot3d leaves
            # asymmetric dead margin, so the drawn orbit does not sit centred in its
            # measured extent.
            for bb, text, dx in ((ink, "Periapsis (zoom)", 0.0),
                                 (inkc, "3D orbit", -0.012)):
                fig.text(0.5 * (bb.x0 + bb.x1) + dx, row.y0 - 0.055, text,
                         ha="center", va="top", fontsize=FONT_BASE - 1.0,
                         bbox=dict(boxstyle="round,pad=0.42", facecolor="0.91",
                                   edgecolor="0.72", linewidth=0.6))

    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        # A small pad rather than the 0.1 default: `bbox_inches="tight"` crops to the
        # artists and LaTeX supplies its own spacing, but at 0.01 the rotated level labels
        # sit hard against the left edge and their boxes get shaved.
        # The left margin is reserved in the layout (`left=` on the gridspec) because the
        # rotated level labels are annotations offset outside the axes, and a tight bbox
        # measures only the axes and their decorations — it would shave the label boxes.
        #
        # Trimmed to just past the inset rule. A tight bbox is wrong here (it measures only
        # the axes, so it shaves the rotated level labels) and the full canvas is wrong too
        # (mplot3d's internal right margin shows as dead space), so the bound is taken from
        # the measured rule and converted to the inches `bbox_inches` wants.
        w_in, h_in = fig.get_size_inches()
        trim = Bbox.from_extents(0.0, 0.0, (right_edge + 0.008) * w_in, h_in)
        fig.savefig(FIG_DIR / f"{args.out}.{ext}", transparent=True, dpi=200,
                    bbox_inches=trim)
    kb = (FIG_DIR / f"{args.out}.pdf").stat().st_size / 1024
    print(f"wrote {FIG_DIR}/{args.out}.{{pdf,svg,png}}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()