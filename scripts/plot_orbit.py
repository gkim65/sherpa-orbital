"""
Reproduce Exhibit B-21 from MacKenzie et al. 2020 §B.2.3.

Three-panel figure for the period-3 southern L1 halo orbit:

  Panel (a): X–Z trajectory (CR3BP rotating frame, Enceladus-centred).
             All 3 loops are geometrically identical in X-Z (they separate
             only in time), so the curve looks like a single loop — matching
             MacKenzie Exhibit B-21(a). Green/red markers = apoapsis/periapsis.

  Panel (b): X–Y groundtrack (top-down view, Enceladus-centred).
             Shows the orbit shape from above with Enceladus sphere to scale.
             The 3 loops are superimposed in X-Y (same path, different times)
             because our CR3BP IC is at the period-3 bifurcation. The
             MacKenzie figure shows 3 distinct periapsis locations because
             their higher-fidelity model breaks this degeneracy.
             We colour the approach (y>0), periapsis, and departure (y<0)
             arcs distinctly so the orbit's Y-symmetry is visible.

  Panel (c): Periapsis altitude vs elapsed days (24 revolutions).
             CR3BP (black, flat at 31 km) vs CR3BP+J2 truth (blue, escaping)
             without stationkeeping.

Run:
    python scripts/plot_orbit.py

Output:
    figures/exhibit_b21.png

References:
  MacKenzie et al. (2020). Enceladus Orbilander Mission Concept Study, §B.2.3.
  Howell (1984). Three-Dimensional, Periodic, Halo Orbits.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.constants import R_ENCELADUS, PERIAPSIS_ALT_MIN, PERIAPSIS_ALT_MAX
from src.dynamics.cr3bp import cr3bp_eom, X_ENCELADUS
from src.dynamics.cr3bp_j2 import cr3bp_j2_eom
from src.dynamics.integrator import (
    propagate, make_periapsis_event, make_crash_event,
    RTOL_TRUTH, ATOL_TRUTH, RTOL_ONBOARD, ATOL_ONBOARD,
)
from src.utils.halo_ic import PERIOD3_IC_ND, PERIOD3_PERIOD_S, _nd_to_phys

N_REVS: int   = 24
X_ENC_KM: float = X_ENCELADUS


# ── Propagation ───────────────────────────────────────────────────────────────

def propagate_one_orbit(ic_km: np.ndarray, period_s: float) -> object:
    return propagate(
        cr3bp_eom, ic_km, (0.0, period_s),
        rtol=RTOL_ONBOARD, atol=ATOL_ONBOARD,
        dense_output=True, max_step=period_s / 4000,
    )


def propagate_n_revs_periapses(ic_km, period_s, n_revs, eom, rtol, atol):
    peri_alts = []
    state = ic_km.copy()
    peri_ev  = make_periapsis_event(terminal=False)
    crash_ev = make_crash_event()
    for rev in range(n_revs):
        sol = propagate(eom, state, (0.0, period_s),
                        events=[peri_ev, crash_ev], rtol=rtol, atol=atol)
        for ps in sol.y_events[0]:
            r = np.linalg.norm(ps[:3] - np.array([X_ENC_KM, 0.0, 0.0]))
            peri_alts.append(r - R_ENCELADUS)
        state = sol.y[:, -1]
        if len(sol.t_events[1]) > 0:
            print(f"  Crash at revolution {rev+1}")
            break
        if np.linalg.norm(state[:3] - np.array([X_ENC_KM, 0.0, 0.0])) > 1e4:
            print(f"  Orbit escaped at revolution {rev+1}")
            break
    return peri_alts


def _orbit_arrays(sol):
    """Return (dx, y, z, alt) arrays relative to Enceladus centre."""
    dx  = sol.y[0] - X_ENC_KM
    y   = sol.y[1]
    z   = sol.y[2]
    r   = np.sqrt(dx**2 + y**2 + z**2)
    alt = r - R_ENCELADUS
    return dx, y, z, alt


# ── Panel (a): X–Z trajectory ─────────────────────────────────────────────────

def plot_xz(ax: plt.Axes, sol: object) -> None:
    """
    X–Z projection (Enceladus-centred), styled after MacKenzie Exhibit B-21(a).

    Trajectory coloured by elapsed time (blue→red) so orbit direction is clear.
    Apoapsis (green ▲) and periapsis (red ●) are marked.
    Enceladus sphere drawn to scale at origin.
    """
    dx, y, z, alt = _orbit_arrays(sol)
    t_norm = sol.t / sol.t[-1]   # 0→1 over the full period

    # Colour segments by time using a line collection
    from matplotlib.collections import LineCollection
    points  = np.array([dx, z]).T.reshape(-1, 1, 2)
    segs    = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap="plasma", linewidth=1.8, zorder=3)
    lc.set_array(t_norm[:-1])
    ax.add_collection(lc)

    # Apoapsis and periapsis markers
    apo_i  = int(np.argmax(alt))
    peri_i = int(np.argmin(alt))
    ax.scatter(dx[apo_i],  z[apo_i],  color="#2ca02c", s=90, zorder=8,
               marker="^", edgecolors="black", linewidths=0.5,
               label=f"Apoapsis  {alt[apo_i]:.0f} km")
    ax.scatter(dx[peri_i], z[peri_i], color="#d62728", s=90, zorder=8,
               marker="o", edgecolors="black", linewidths=0.5,
               label=f"Periapsis {alt[peri_i]:.0f} km")

    # Enceladus sphere
    ax.add_patch(Circle((0, 0), R_ENCELADUS,
                         facecolor="#aaddff", edgecolor="#3399cc",
                         linewidth=0.8, alpha=0.55, zorder=5))

    # Colourbar for time
    sm = ScalarMappable(cmap="plasma", norm=Normalize(0, 1))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Elapsed fraction of 36-hr period", fontsize=7)
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(["0 hr", "18 hr", "36 hr"])
    cb.ax.tick_params(labelsize=7)

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("X  (km, Enceladus-centred)", fontsize=9)
    ax.set_ylabel("Z  (km)", fontsize=9)
    ax.set_title(
        "(a)  X–Z trajectory  (CR3BP rotating frame, 1 period = 36 hr)\n"
        "      All 3 loops superimposed — colour = elapsed time",
        fontsize=8.5)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, loc="lower right")

    # R_Enc secondary axes
    def to_r(v): return v / R_ENCELADUS
    def to_km(v): return v * R_ENCELADUS
    try:
        ax.secondary_xaxis("top",   functions=(to_r, to_km)).set_xlabel("R$_{Enc}$", fontsize=7)
        ax.secondary_yaxis("right", functions=(to_r, to_km)).set_ylabel("R$_{Enc}$", fontsize=7)
    except Exception:
        pass


# ── Panel (b): X–Y groundtrack ────────────────────────────────────────────────

def plot_yz(ax: plt.Axes, sol: object) -> None:
    """
    Y–Z side view (looking along −X, i.e. from Saturn toward Enceladus).

    This projection correctly shows the spacecraft skimming above the pole:
    periapsis is at (y≈0, z≈283 km) — 31 km outside the Enceladus limb —
    which is clearly visible in Y-Z but invisible in X-Y (where the orbit
    appears to pass through the body because the spacecraft is nearly directly
    above the pole along X).

    The orbit traces a tall narrow figure-8 in Y-Z: the spacecraft swings
    from apoapsis at z≈−1163 km (south, deep below Enceladus), arcs up through
    y=±654 km, and converges to periapsis just above the north pole (z≈+283 km).
    Track coloured by altitude (dark purple = apoapsis, bright yellow = periapsis).
    """
    from matplotlib.collections import LineCollection

    dx, y, z, alt = _orbit_arrays(sol)

    # One representative loop (all 3 identical)
    T3   = sol.t[-1] / 3.0
    mask = sol.t <= T3 + 1.0
    y1   = y[mask]
    z1   = z[mask]
    alt1 = alt[mask]

    # Line coloured by altitude
    points = np.array([y1, z1]).T.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap="plasma_r", linewidth=2.2, zorder=4)
    lc.set_array(alt1[:-1])
    lc.set_clim(0, 1100)
    ax.add_collection(lc)

    # Periapsis marker (should sit visibly outside Enceladus limb)
    pi = int(np.argmin(alt1))
    ax.scatter(y1[pi], z1[pi], color="yellow", s=130, zorder=10,
               edgecolors="black", linewidths=0.8, marker="o",
               label=f"Periapsis  {alt1[pi]:.0f} km alt")

    # Apoapsis marker
    ai = int(np.argmax(alt1))
    ax.scatter(y1[ai], z1[ai], color="#2ca02c", s=90, zorder=10,
               edgecolors="black", linewidths=0.6, marker="^",
               label=f"Apoapsis {alt1[ai]:.0f} km alt")

    # Enceladus body — circle at origin, radius R_enc
    ax.add_patch(Circle((0, 0), R_ENCELADUS,
                         facecolor="#aaddff", edgecolor="#3399cc",
                         linewidth=1.0, alpha=0.6, zorder=5,
                         label=f"Enceladus  R={R_ENCELADUS:.0f} km"))

    # Pole labels
    ax.text(0,  R_ENCELADUS + 30, "N pole", fontsize=7, ha="center", va="bottom", color="#336699")
    ax.text(0, -R_ENCELADUS - 30, "S pole", fontsize=7, ha="center", va="top",    color="#336699")

    # Colourbar
    sm = ScalarMappable(cmap="plasma_r", norm=Normalize(0, 1100))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Altitude above Enceladus (km)", fontsize=7)
    cb.ax.tick_params(labelsize=7)

    zoom_km = 750.0
    ax.set_xlim(-zoom_km, zoom_km)
    ax.set_ylim(-zoom_km - 500, zoom_km)
    ax.set_xlabel("Y  (km)", fontsize=9)
    ax.set_ylabel("Z  (km)", fontsize=9)
    ax.set_title(
        "(b)  Y–Z view  (looking along −X, from Saturn toward Enceladus)\n"
        "      Periapsis skims 31 km above N pole — clearance visible here",
        fontsize=8.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, loc="upper right")

    try:
        def to_r(v): return v / R_ENCELADUS
        def to_km(v): return v * R_ENCELADUS
        ax.secondary_xaxis("top",   functions=(to_r, to_km)).set_xlabel("R$_{Enc}$", fontsize=7)
        ax.secondary_yaxis("right", functions=(to_r, to_km)).set_ylabel("R$_{Enc}$", fontsize=7)
    except Exception:
        pass


# ── Panel (c): periapsis altitude vs time ─────────────────────────────────────

def plot_periapsis_altitude(ax, peri_cr3bp, peri_j2, period_s):
    dt = period_s / (3 * 86400.0)
    t_c = np.arange(len(peri_cr3bp)) * dt
    t_j = np.arange(len(peri_j2))    * dt
    MAX = 1200.0

    ax.plot(t_c, np.minimum(peri_cr3bp, MAX), color="black", linewidth=1.4,
            marker=".", markersize=3, label="CR3BP (periodic, onboard model)")
    ax.plot(t_j, np.minimum(peri_j2,    MAX), color="royalblue", linewidth=1.4,
            marker=".", markersize=3, label="CR3BP + J2 (truth, uncontrolled)")

    if len(t_j) > 0 and (len(t_c) == 0 or t_j[-1] < t_c[-1]):
        ax.axvline(t_j[-1], color="royalblue", linewidth=0.9, linestyle=":", alpha=0.7)
        ax.text(t_j[-1] + 0.3, MAX * 0.48,
                "orbit escapes\n(no stationkeeping)",
                color="royalblue", fontsize=7, va="center")

    ax.axhspan(PERIAPSIS_ALT_MIN, PERIAPSIS_ALT_MAX, color="lightgreen",
               alpha=0.3,
               label=f"Mission band ({PERIAPSIS_ALT_MIN}–{PERIAPSIS_ALT_MAX} km)")
    ax.axhline(PERIAPSIS_ALT_MIN, color="green", linewidth=0.8, linestyle="--")
    ax.axhline(PERIAPSIS_ALT_MAX, color="green", linewidth=0.8, linestyle="--")

    n_days = len(peri_cr3bp) * dt
    ax.set_title(
        f"(c)  Periapsis altitude vs time  ({n_days:.0f} days, {N_REVS} revolutions)\n"
        f"      CR3BP stays flat; J2 truth model escapes without stationkeeping",
        fontsize=8.5)
    ax.set_xlabel("Elapsed time (days)", fontsize=9)
    ax.set_ylabel("Periapsis altitude (km)", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, MAX)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Exhibit B-21 — MacKenzie et al. 2020 §B.2.3 ===\n")

    ic_km    = _nd_to_phys(PERIOD3_IC_ND)
    period_s = PERIOD3_PERIOD_S

    ic_nd = PERIOD3_IC_ND
    print(f"IC (nd):  x={ic_nd[0]:.10f}  z={ic_nd[2]:.6e}  vy={ic_nd[4]:.6e}")
    print(f"IC (km):  x={ic_km[0]:.1f}  z={ic_km[2]:.2f}  vy={ic_km[4]:.5f} km/s")
    print(f"Period:   {period_s/3600:.3f} hr\n")

    print("Propagating 1 full orbit (36 hr) …")
    sol = propagate_one_orbit(ic_km, period_s)
    print(f"  {len(sol.t)} steps.")

    dx, y, z, alt = _orbit_arrays(sol)
    T3 = period_s / 3
    for k in range(3):
        t0, t1 = k*T3, (k+1)*T3
        mask = (sol.t >= t0) & (sol.t <= t1)
        la = alt[mask]
        pi = int(np.argmin(la))
        print(f"  Loop {k+1}: peri {la[pi]:.1f} km  "
              f"(dx={dx[mask][pi]:.1f}, y={y[mask][pi]:.1f}, z={z[mask][pi]:.1f} km)")

    print(f"\nPropagating {N_REVS} revolutions for panel (c):")
    print("  CR3BP …")
    peri_c = propagate_n_revs_periapses(
        ic_km, period_s, N_REVS, cr3bp_eom, RTOL_ONBOARD, ATOL_ONBOARD)
    print(f"    {len(peri_c)} passes  min={min(peri_c):.1f}  max={max(peri_c):.1f} km")
    print("  CR3BP+J2 …")
    peri_j = propagate_n_revs_periapses(
        ic_km, period_s, N_REVS, cr3bp_j2_eom, RTOL_TRUTH, ATOL_TRUTH)
    print(f"    {len(peri_j)} passes  min={min(peri_j):.1f}  max={max(peri_j):.1f} km")

    print("\nBuilding figure …")
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))
    fig.suptitle(
        "Enceladus Orbilander — Period-3 L1 Halo Science Orbit  "
        "(MacKenzie et al. 2020, Exhibit B-21)",
        fontsize=11, fontweight="bold")

    plot_xz(axes[0], sol)
    plot_yz(axes[1], sol)
    plot_periapsis_altitude(axes[2], peri_c, peri_j, period_s)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])

    out_dir  = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exhibit_b21.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
