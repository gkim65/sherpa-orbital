"""
plot_rollout_3d.py — 3D tilted view + animated GIF of a science+safety POMDP rollout.

Reads figures/rollout_trace_data.npz (from scripts/pomdp_experiments/13_rollout_trace.py).

  --static : one tilted 3D view of the full trajectory around Enceladus with maneuvers
             marked (green = CORRECT / purple = EXCURSE) and their ΔV vectors.
  --gif    : an animation that draws the trajectory over time and pops each burn in as
             it fires, while slowly rotating the view. Writes figures/rollout_3d.gif.
  --spin   : a static-scene GIF that just rotates the finished trajectory (cheap, smooth).

Styling: Computer Modern serif, sentence-case labels. Light theme. Saves PNG/PDF/SVG for
the static view; GIF for animations.

Run:  python -m scripts.plot_rollout_3d --static
      python -m scripts.plot_rollout_3d --gif
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
DATA = FIG_DIR / "rollout_trace_data.npz"

C = {"ink": "#1a1a1a", "muted": "#5f6368", "orbit": "#3572A5", "body": "#c4c9d0",
     "correct": "#2e8b57", "excurse": "#9558B2", "peri": "#c0392b"}


def _use_serif():
    if shutil.which("latex"):
        matplotlib.rcParams.update({"text.usetex": True, "font.family": "serif",
                                    "font.serif": ["Computer Modern Roman"]})
    else:
        matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif",
                                    "mathtext.fontset": "cm"})
    matplotlib.rcParams.update({"font.size": 11})


def _sphere(ax, R, color):
    u = np.linspace(0, 2 * np.pi, 30); v = np.linspace(0, np.pi, 20)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=0.55, linewidth=0, shade=True, zorder=1)


def _is_exc(a):
    return str(a).startswith("EXCURSE")


def _setup_axes(ax, traj, Renc):
    lim = np.max(np.abs(traj)) * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel("x (km)", labelpad=6); ax.set_ylabel("y (km)", labelpad=6)
    ax.set_zlabel("z (km)", labelpad=6)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.grid(False)


def _draw_burns(ax, bp, bdv, bact, scale, upto=None):
    n = len(bp) if upto is None else upto
    for i in range(n):
        p, dv, a = bp[i], bdv[i], bact[i]
        col = C["excurse"] if _is_exc(a) else C["correct"]
        ax.quiver(p[0], p[1], p[2], dv[0]*scale, dv[1]*scale, dv[2]*scale,
                  color=col, linewidth=1.8, arrow_length_ratio=0.25, zorder=6)
        ax.scatter([p[0]], [p[1]], [p[2]], s=30, color=col, zorder=7)


def _legend(ax):
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color=C["orbit"], lw=1.5, label="Trajectory (Enc J2)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["correct"],
               markersize=8, label="Correct burn (hold)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["excurse"],
               markersize=8, label="Excurse burn (science)"),
    ], loc="upper left", frameon=False, fontsize=8.5)


def static_view(d):
    _use_serif()
    traj = d["traj"]; Renc = float(d["r_enceladus"])
    bp, bdv, bact = d["burn_pos"], d["burn_dv"], d["burn_action"]
    scale = 2.5e5
    fig = plt.figure(figsize=(8.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    _sphere(ax, Renc, C["body"])
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=C["orbit"], lw=0.7, alpha=0.85, zorder=3)
    _draw_burns(ax, bp, bdv, bact, scale)
    _setup_axes(ax, traj, Renc)
    ax.view_init(elev=22, azim=-60)
    ax.set_title(r"Rollout trajectory in 3D + maneuvers ($\Delta V$ magnified)", pad=0)
    _legend(ax)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"rollout_3d_static.{ext}", transparent=True,
                    bbox_inches="tight", dpi=200)
    print(f"wrote {FIG_DIR}/rollout_3d_static.{{png,pdf,svg}}")


def make_gif(d, spin_only=False, n_frames=120, rotate=False):
    _use_serif()
    traj = d["traj"]; Renc = float(d["r_enceladus"])
    bp, bdv, bact = d["burn_pos"], d["burn_dv"], d["burn_action"]
    scale = 2.5e5
    # Map each burn to the trajectory index nearest its position (for reveal timing).
    burn_idx = [int(np.argmin(np.linalg.norm(traj - bp[i], axis=1))) for i in range(len(bp))]
    order = np.argsort(burn_idx)
    burn_idx = np.array(burn_idx)[order]
    bp, bdv, bact = bp[order], bdv[order], bact[order]

    fig = plt.figure(figsize=(7.4, 6.8))
    ax = fig.add_subplot(111, projection="3d")

    def draw(frame):
        ax.clear()
        _sphere(ax, Renc, C["body"])
        _setup_axes(ax, traj, Renc)
        # Camera azimuth. `rotate` (default on) is a SLOW, FULL 360° turntable — a
        # purely COSMETIC depth aid, NOT physics: Enceladus is stationary in the
        # Saturn-Enceladus rotating frame, so nothing in the scene is actually spinning.
        # The spin RATE = 360°/n_frames per frame, so MORE frames = slower spin (use
        # --frames to slow it further); one full revolution over the animation.
        if rotate or spin_only:
            azim = -60 + 360.0 * frame / n_frames
        else:
            azim = -60
        ax.view_init(elev=22, azim=azim)
        if spin_only:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=C["orbit"], lw=0.7, alpha=0.85)
            _draw_burns(ax, bp, bdv, bact, scale)
            ax.set_title("Rollout trajectory in 3D (rotating)", pad=0)
        else:
            k = int((frame + 1) / n_frames * len(traj))
            k = max(2, k)
            ax.plot(traj[:k, 0], traj[:k, 1], traj[:k, 2], color=C["orbit"], lw=0.9, alpha=0.9)
            ax.scatter([traj[k-1, 0]], [traj[k-1, 1]], [traj[k-1, 2]], s=25,
                       color=C["ink"], zorder=8)   # current spacecraft
            nb = int(np.sum(burn_idx < k))
            _draw_burns(ax, bp, bdv, bact, scale, upto=nb)
            done = sum(1 for a in bact[:nb] if _is_exc(a))
            ax.set_title(f"Rollout — day {k/len(traj)*8:.1f},  science bands sampled: {done}/3",
                         pad=0, fontsize=11)
            # Honest label: the camera spin is a viewing aid, not a rotating body.
            fig.text(0.5, 0.055, "camera orbits for depth; Enceladus fixed in the "
                     "rotating frame", ha="center",
                     fontsize=7.5, color=C["muted"], style="italic")
        _legend(ax)
        return []

    anim = FuncAnimation(fig, draw, frames=n_frames, blit=False)
    out = FIG_DIR / ("rollout_3d_spin.gif" if spin_only else "rollout_3d.gif")
    anim.save(out, writer=PillowWriter(fps=20), dpi=90)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--spin", action="store_true")
    ap.add_argument("--no-rotate", dest="rotate", action="store_false",
                    help="disable the gentle cosmetic camera pan in --gif "
                         "(default: on, a slow labeled turntable)")
    ap.set_defaults(rotate=True)
    ap.add_argument("--frames", type=int, default=120)
    args = ap.parse_args()
    if not DATA.exists():
        raise SystemExit(f"missing {DATA}; run scripts/pomdp_experiments/13_rollout_trace.py")
    d = np.load(DATA, allow_pickle=True)
    if args.static or not (args.gif or args.spin):
        static_view(d)
    if args.gif:
        make_gif(d, spin_only=False, n_frames=args.frames, rotate=args.rotate)
    if args.spin:
        make_gif(d, spin_only=True, n_frames=args.frames)
