"""
Divergence study (Phase 1.5 Step A): does Saturn J2 matter at Enceladus altitude?

Propagates the period-3 southern L1 halo IC under two TRUTH models:
  (1) CR3BP + Enceladus J2                 — the current truth model
  (2) CR3BP + Saturn J2 + Enceladus J2     — the high-fidelity Step A truth model

and plots the position-error magnitude ‖Δr(t)‖ between them over ~1-2 Titan
orbits around Saturn (Titan period ≈ 15.95 days; reviewers say 16-32 days sweeps
most moon geometries — here it bounds the timescale even though the moons aren't
modeled yet).

This quantifies the reviewers' open question: how much does Saturn's oblateness
perturb the science orbit relative to Enceladus' own oblateness? A divergence
that grows past the ~31 km periapsis clearance within a few days is itself a
result — it means the cheaper EncJ2-only truth is not a good stand-in, and that
an uncontrolled onboard CR3BP model has no hope (which is the POMDP's job).

Run:  python scripts/divergence_saturn_j2.py
Saves: figures/divergence_saturn_j2.png   (not committed with code)

Reference: external review note docs/external-review-2026-06-22.md.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.constants import R_ENCELADUS
from src.dynamics.cr3bp import X_ENCELADUS
from src.dynamics.cr3bp_j2 import cr3bp_j2_eom
from src.dynamics.cr3bp_saturn_j2 import cr3bp_saturn_enc_j2_eom
from src.dynamics.integrator import propagate, RTOL_TRUTH, ATOL_TRUTH
from src.utils.halo_ic import PERIOD3_IC_ND, _nd_to_phys

TITAN_PERIOD_S: float = 15.945 * 86400.0  # Titan sidereal period (days → s)


def main(n_titan_orbits: float = 2.0) -> None:
    ic = _nd_to_phys(PERIOD3_IC_ND)
    t_end = n_titan_orbits * TITAN_PERIOD_S

    # Shared evaluation grid so the two solutions are compared at identical times.
    n_pts = 4000
    t_eval = np.linspace(0.0, t_end, n_pts)

    print(f"Propagating period-3 IC for {n_titan_orbits:.1f} Titan orbits "
          f"({t_end/86400.0:.1f} days)...")

    sol_enc = propagate(cr3bp_j2_eom, ic, (0.0, t_end),
                        rtol=RTOL_TRUTH, atol=ATOL_TRUTH, dense_output=True)
    sol_sat = propagate(cr3bp_saturn_enc_j2_eom, ic, (0.0, t_end),
                        rtol=RTOL_TRUTH, atol=ATOL_TRUTH, dense_output=True)

    y_enc = sol_enc.sol(t_eval)
    y_sat = sol_sat.sol(t_eval)

    dr = np.linalg.norm(y_sat[:3] - y_enc[:3], axis=0)  # km
    dv = np.linalg.norm(y_sat[3:] - y_enc[3:], axis=0)  # km/s
    days = t_eval / 86400.0

    # Console summary.
    peri_clear = 31.0  # km, the period-3 orbit's periapsis clearance
    print(f"\nDivergence summary (CR3BP+EncJ2  vs  CR3BP+SatJ2+EncJ2):")
    print(f"  ‖Δr‖ at  1 day : {np.interp(1.0, days, dr):8.3f} km")
    print(f"  ‖Δr‖ at  5 days: {np.interp(5.0, days, dr):8.3f} km")
    print(f"  ‖Δr‖ at end    : {dr[-1]:8.3f} km  ({days[-1]:.1f} days)")
    cross = days[dr > peri_clear]
    if cross.size:
        print(f"  Δr exceeds the {peri_clear:.0f} km periapsis clearance "
              f"after {cross[0]:.2f} days.")
    else:
        print(f"  Δr never exceeds the {peri_clear:.0f} km periapsis clearance.")

    # Plot.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(days, dr, lw=1.2, color="C3")
    ax1.axhline(peri_clear, ls="--", lw=1.0, color="gray",
                label=f"periapsis clearance ({peri_clear:.0f} km)")
    ax1.set_ylabel("‖Δr‖  (km)")
    ax1.set_title("Saturn-J2 divergence of the period-3 halo orbit\n"
                  "CR3BP+EncJ2  vs  CR3BP+SaturnJ2+EncJ2")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.plot(days, dv * 1e3, lw=1.2, color="C0")  # m/s
    ax2.set_ylabel("‖Δv‖  (m/s)")
    ax2.set_xlabel("time (days)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, "divergence_saturn_j2.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()