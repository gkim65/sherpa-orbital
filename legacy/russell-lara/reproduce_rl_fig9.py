"""
Reproduce Russell & Lara (2009) Fig. 9 — the south-pole-grazing L1 halo —
in their Hill + J2/J3 model. Hill-only; see the FROZEN note below.

Russell, R. P., & Lara, M. (2009). "On the design of an Enceladus science
orbit." Acta Astronautica 65, 27-39.

Their Fig. 9 shows ONE period of the example stable halo: trajectory (left) and
osculating orbital elements (right). We reproduce:

  Panel (a): X–Z trajectory, Enceladus-centred rotating frame (their left plot).
  Panel (b): X–Y trajectory (top-down), Enceladus to scale.
  Panel (c): osculating a, e, i vs time over one period (their right plot).
             Elements are computed from the ROTATING-frame velocity — the
             convention under which our values match their Table 2 exactly
             (a=1195.7 km, e=0.756). See constants_russell_lara.py.
  Panel (d): Hill-model altitude history over 3 periods.

FROZEN (Session 5). Panel (d) ORIGINALLY held a model-gap comparison against our
barycentric CR3BP+EncJ2 truth model. That panel was removed when python-legacy/
was deleted, since it imported the CR3BP modules. Its last recorded result is
quoted in README.md; this script is now Hill-only and needs numpy + matplotlib +
scipy alone.

Run (from this directory):
    python reproduce_rl_fig9.py
Output:
    rl_fig9.png (written alongside this script)

NOTE ON FIDELITY: we propagate with the dominant, unambiguous J2+J3 zonal field.
Russell-Lara's C22 phase convention is underspecified in the paper and could not
be matched to sub-km one-period closure; J2+J3 reproduces the orbit's character
(period, altitude band, shape) which is what Fig. 9 conveys. One-period closure
is ~17 km. See hill_nonspherical.py for the full discussion.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# FROZEN (Session 5): this directory is self-contained and imports only its own two
# modules, which sit alongside this file. The former `src.*` imports (src.constants,
# src.dynamics.cr3bp, src.dynamics.cr3bp_j2) pointed into python-legacy/, now deleted;
# the CR3BP comparison panel that needed them has been removed. See README.md.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constants_russell_lara import (
    IC_HALO_RL, N_RL, GM_ENCELADUS_RL, R_ENCELADUS_RL,
    A0_HALO_RL, E0_HALO_RL, rl_ic_to_rotating,
)
from hill_nonspherical import hill_nonspherical_eom

PERIOD_S = 12.0 * 3600.0   # ~one period (their halo closes near 12 hr)


def osc_elements_rotframe(state: np.ndarray, n: float, GM: float) -> tuple:
    """
    Osculating (a, e, i) from a ROTATING-frame state — Russell-Lara Table 2
    convention (velocity taken as-is in the rotating frame).

    state : [x,y,z,vx,vy,vz] Enceladus-centred (km, km/s).
    Returns (a [km], e, i [deg]).
    """
    r = state[:3]
    v = state[3:]
    rn = np.linalg.norm(r)
    vn = np.linalg.norm(v)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    a = -GM / (2.0 * (vn**2 / 2.0 - GM / rn))
    e = np.linalg.norm(np.cross(v, h) / GM - r / rn)
    i = np.degrees(np.arccos(np.clip(h[2] / hn, -1.0, 1.0)))
    return a, e, i



def main() -> None:
    # ── Their Hill model ─────────────────────────────────────────────────────
    ic_hill = rl_ic_to_rotating(IC_HALO_RL)
    t_eval = np.linspace(0.0, PERIOD_S, 2000)
    sol = solve_ivp(hill_nonspherical_eom, [0.0, PERIOD_S], ic_hill,
                    t_eval=t_eval, rtol=1e-11, atol=1e-13, max_step=200.0)
    X, Y, Z = sol.y[0], sol.y[1], sol.y[2]
    alt_hill = np.linalg.norm(sol.y[:3], axis=0) - R_ENCELADUS_RL

    a_t, e_t, i_t = [], [], []
    for k in range(sol.y.shape[1]):
        a, e, i = osc_elements_rotframe(sol.y[:, k], N_RL, GM_ENCELADUS_RL)
        a_t.append(a); e_t.append(e); i_t.append(i)
    a_t, e_t, i_t = np.array(a_t), np.array(e_t), np.array(i_t)

    # ── Hill model out to 3 periods, for the altitude-history panel ───────────
    # (The barycentric CR3BP+EncJ2 comparison that used to share this panel was
    #  removed with python-legacy/. Its last recorded result is in README.md.)
    sol_h3 = solve_ivp(hill_nonspherical_eom, [0.0, 3 * PERIOD_S], ic_hill,
                       t_eval=np.linspace(0, 3 * PERIOD_S, 3000),
                       rtol=1e-11, atol=1e-13, max_step=200.0)
    alt_hill3 = np.linalg.norm(sol_h3.y[:3], axis=0) - R_ENCELADUS_RL

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 9))

    ax = fig.add_subplot(2, 2, 1)
    ax.plot(X, Z, lw=1.2, color="navy")
    _enceladus_circle(ax, R_ENCELADUS_RL)
    ax.scatter([X[0]], [Z[0]], c="k", s=25, zorder=5, label="IC")
    ax.set_xlabel("x (km)"); ax.set_ylabel("z (km)")
    ax.set_title("(a) Halo trajectory — x–z (rotating frame)")
    ax.axis("equal"); ax.legend(); ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(X, Y, lw=1.2, color="darkgreen")
    _enceladus_circle(ax, R_ENCELADUS_RL)
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_title("(b) Halo trajectory — x–y (top-down)")
    ax.axis("equal"); ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 2, 3)
    hrs = sol.t / 3600.0
    ax.plot(hrs, a_t, label=f"a (km), a₀={A0_HALO_RL:.0f}", color="C0")
    ax.plot(hrs, i_t, label="i (deg)", color="C1")
    ax2 = ax.twinx()
    ax2.plot(hrs, e_t, label=f"e, e₀={E0_HALO_RL:.3f}", color="C2", ls="--")
    ax2.set_ylabel("eccentricity")
    ax.set_xlabel("time (hr)"); ax.set_ylabel("a (km) / i (deg)")
    ax.set_title("(c) Osculating elements (rotating-frame convention)")
    lines, labels = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lab2, fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(sol_h3.t / 3600.0, alt_hill3, color="navy",
            label="Russell-Lara Hill + J2/J3")
    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("time (hr)"); ax.set_ylabel("altitude above Enceladus (km)")
    ax.set_title("(d) Altitude history over 3 periods (Hill)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Russell & Lara (2009) Fig. 9 reproduction — Enceladus L1 halo",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl_fig9.png")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")

    # Console summary. The two [CR3BP] lines that used to appear here needed the deleted
    # python-legacy/ pipeline; their last recorded values are quoted in README.md.
    print(f"[Hill]   1-period closure altitude band: "
          f"{alt_hill.min():.1f}..{alt_hill.max():.1f} km")
    print(f"[Hill]   osc a {a_t.min():.0f}..{a_t.max():.0f} km, "
          f"e {e_t.min():.3f}..{e_t.max():.3f}, i {i_t.min():.1f}..{i_t.max():.1f} deg")
    print(f"[Hill]   altitude over 3 periods: "
          f"{alt_hill3.min():.1f}..{alt_hill3.max():.1f} km")


def _enceladus_circle(ax, R: float) -> None:
    th = np.linspace(0, 2 * np.pi, 200)
    ax.fill(R * np.cos(th), R * np.sin(th), color="lightgray",
            alpha=0.6, zorder=0)


if __name__ == "__main__":
    main()