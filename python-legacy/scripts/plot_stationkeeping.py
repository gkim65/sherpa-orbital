"""
Stationkeeping performance plots for the MPC controller (baselines/mpc.py).

Compares MacKenzie Strategy 3 (apse position-vector targeting) against the
weaker apse-altitude targeting, on two orbits:
  - the MacKenzie period-3 L1 halo (the design orbit), and
  - the recreated Russell-Lara halo (a harder, higher-eccentricity orbit).

Panels per run:
  (a) ΔV per burn vs time            — are maneuver magnitudes bounded? (the
                                        MacKenzie success criterion for Strategy 3)
  (b) cumulative ΔV vs time          — total propellant cost / trend
  (c) burn residual vs time          — how well each burn hits its target (km)

Run:
    python scripts/plot_stationkeeping.py
Output:
    figures/stationkeeping_performance.png

References:
  MacKenzie et al. (2020), Appendix B.2.3, Exhibit B-22/B-23 (Strategy 3).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.halo_ic import PERIOD3_IC_ND, PERIOD3_PERIOD_S
from src.utils.orbital_elements import nondim_to_cr3bp
from src.constants_russell_lara import IC_HALO_RL
from src.dynamics.cr3bp_j2 import cr3bp_j2_eom
from baselines.mpc import run_mpc
from scripts.reproduce_rl_fig9 import rl_to_barycentric_cr3bp

DAY = 86400.0


def _summarize(name: str, res: dict) -> None:
    """Print a one-line performance summary for a run."""
    print(f"{name:42s}: {res['outcome']:9s}  "
          f"survived {res['survival_time_s']/DAY:5.2f} d  "
          f"{res['n_burns']:3d} burns  "
          f"{res['total_dv_ms']:7.1f} m/s  "
          f"min periapsis {res['min_peri_alt_km']:.0f} km")


def main() -> None:
    mac_ic = nondim_to_cr3bp(PERIOD3_IC_ND)
    mac_P = PERIOD3_PERIOD_S / 3.0
    rl_ic = rl_to_barycentric_cr3bp(IC_HALO_RL)
    rl_P = 12.0 * 3600.0

    # (name, ic, period, horizon_days, run-kwargs)
    runs = [
        ("MacKenzie halo — altitude target",
         mac_ic, mac_P, 30, dict(mode="altitude")),
        ("MacKenzie halo — Strategy 3 (position)",
         mac_ic, mac_P, 30, dict(mode="position")),
        ("Russell-Lara halo — altitude target",
         rl_ic, rl_P, 10, dict(mode="altitude",
                               peri_target_km=29.9, apo_target_km=1060.9)),
        ("Russell-Lara halo — Strategy 3 (position)",
         rl_ic, rl_P, 10, dict(mode="position")),
    ]

    results = []
    for name, ic, P, hdays, kw in runs:
        res = run_mpc(ic, cr3bp_j2_eom, period_s=P,
                      horizon_s=hdays * DAY, n_revs=3, **kw)
        _summarize(name, res)
        results.append((name, hdays, res))

    colors = {"altitude": "crimson", "position": "navy"}
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))

    # Column 0 = MacKenzie halo, column 1 = RL halo
    groups = [("MacKenzie halo", 0), ("Russell-Lara halo", 1)]
    for title, col in groups:
        runs_here = [(n, h, r) for (n, h, r) in results if title in n]
        ax_dv, ax_cum, ax_res = axes[0][col], axes[1][col], axes[2][col]
        for name, hdays, res in runs_here:
            mode = "position" if "Strategy 3" in name else "altitude"
            c = colors[mode]
            burns = res["burns"]
            if not burns:
                continue
            t = np.array([b["t_s"] for b in burns]) / DAY
            dv = np.array([b["dv_ms"] for b in burns])
            resid = np.array([b["residual_km"] for b in burns])
            lab = f"{mode} ({res['outcome']}, {res['total_dv_ms']:.0f} m/s)"
            ax_dv.plot(t, dv, "o-", ms=3, color=c, label=lab)
            ax_cum.plot(t, np.cumsum(dv), "-", color=c, label=lab)
            ax_res.semilogy(t, np.clip(resid, 1e-3, None), "o-", ms=3, color=c,
                            label=lab)

        ax_dv.set_title(f"{title}: ΔV per burn")
        ax_dv.set_ylabel("ΔV (m/s)"); ax_dv.grid(alpha=0.3); ax_dv.legend(fontsize=8)
        ax_cum.set_title(f"{title}: cumulative ΔV")
        ax_cum.set_ylabel("Σ ΔV (m/s)"); ax_cum.grid(alpha=0.3); ax_cum.legend(fontsize=8)
        ax_res.set_title(f"{title}: burn targeting residual")
        ax_res.set_ylabel("‖residual‖ (km)"); ax_res.set_xlabel("time (days)")
        ax_res.axhline(1.0, color="gray", ls=":", lw=0.8, label="1 km target")
        ax_res.grid(alpha=0.3, which="both"); ax_res.legend(fontsize=8)

    fig.suptitle("MPC stationkeeping performance — Strategy 3 (position) vs "
                 "apse-altitude targeting", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figures", "stationkeeping_performance.png")
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()